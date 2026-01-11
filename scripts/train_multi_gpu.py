#!/usr/bin/env python3
"""
Multi-GPU Training Script for BDH NLI Pipeline.

Uses PyTorch DistributedDataParallel (DDP) for efficient multi-GPU training.
Optimized for 2x H100 GPUs.

Usage:
    # Single command to use both GPUs:
    torchrun --nproc_per_node=2 scripts/train_multi_gpu.py --config config/h100.yaml
    
    # Or with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_multi_gpu.py
"""

import os
import sys
from pathlib import Path
from contextlib import nullcontext

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm

from src.data.dataset import create_dataloaders, get_class_weights, DataConfig, collate_fn
from src.models.pipeline import BDHNLIPipeline, PipelineConfig
from src.training.loss import compute_loss


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_device_and_dtype(local_rank: int, dtype_config: str = "auto"):
    device = torch.device(f"cuda:{local_rank}")
    
    if dtype_config != "auto":
        dtype = dtype_config
    else:
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == "float16"))
    
    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    if is_main_process():
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        print(f"GPU {local_rank}: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"Using dtype: {dtype}")
    
    return device, dtype, ptdtype, ctx, scaler


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-GPU Training")
    parser.add_argument("--config", type=str, default="config/h100.yaml")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint dir")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, scaler, ctx, device, pos_weight, grad_accumulation, epoch):
    model.train()
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    loss_acc, loss_steps, correct, total = 0.0, 0, 0, 0
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Training", disable=not is_main_process())
    
    for step, batch in enumerate(pbar):
        backstory_tokens = batch['backstory_tokens'].to(device)
        chunk_tokens = batch['chunk_tokens'].to(device)
        labels = batch['label'].to(device)
        
        if backstory_tokens.dim() == 3:
            backstory_tokens = backstory_tokens.squeeze(0)
        if chunk_tokens.dim() == 3:
            chunk_tokens = chunk_tokens.squeeze(0)
        
        with ctx:
            result = model(chunk_tokens=chunk_tokens, backstory_tokens=backstory_tokens)
            loss = compute_loss(result['prediction'], labels.squeeze().float(), pos_weight=pos_weight)
        
        loss_acc += loss.item()
        loss_steps += 1
        
        loss = loss / grad_accumulation
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        pred = (result['prediction'] > 0.5).long()
        correct += (pred == labels.squeeze()).sum().item()
        total += 1
        
        if is_main_process() and step % 5 == 0:
            pbar.set_postfix({'loss': f"{loss_acc/loss_steps:.4f}", 'acc': f"{correct/total:.3f}"})
    
    if (step + 1) % grad_accumulation != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    metrics = torch.tensor([loss_acc, float(loss_steps), float(correct), float(total)], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item() / metrics[1].item(), metrics[2].item() / metrics[3].item()


@torch.no_grad()
def validate(model, val_loader, device, ctx):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    
    for batch in tqdm(val_loader, desc="Validating", disable=not is_main_process()):
        backstory_tokens = batch['backstory_tokens'].to(device)
        chunk_tokens = batch['chunk_tokens'].to(device)
        labels = batch['label'].to(device)
        
        if backstory_tokens.dim() == 3:
            backstory_tokens = backstory_tokens.squeeze(0)
        if chunk_tokens.dim() == 3:
            chunk_tokens = chunk_tokens.squeeze(0)
        
        with ctx:
            result = model(chunk_tokens=chunk_tokens, backstory_tokens=backstory_tokens)
        
        loss = compute_loss(result['prediction'], labels.squeeze().float())
        total_loss += loss.item()
        pred = (result['prediction'] > 0.5).long()
        correct += (pred == labels.squeeze()).sum().item()
        total += 1
    
    metrics = torch.tensor([total_loss, float(correct), float(total)], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[1].item() / metrics[2].item(), metrics[0].item() / metrics[2].item()


def save_checkpoint(model, optimizer, scaler, epoch, best_val_acc, config, path):
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'config': config
        }, path)


def main():
    args = parse_args()
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if is_main_process():
        print("=" * 60)
        print(f"BDH NLI Pipeline - Multi-GPU Training ({world_size} GPUs)")
        print("=" * 60)
    
    config = load_config(args.config)
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.checkpoint_dir: config['paths']['checkpoint_dir'] = args.checkpoint_dir
    if args.compile: config['training']['compile'] = True
    
    model_cfg = config['model']
    paths_cfg = config['paths']
    training_cfg = config['training']
    data_cfg = config.get('data', {})
    
    seed = config.get('seed', 1337)
    torch.manual_seed(seed + local_rank)
    torch.cuda.manual_seed_all(seed + local_rank)
    
    dtype_config = config.get('dtype', 'auto')
    device, dtype, ptdtype, ctx, scaler = setup_device_and_dtype(local_rank, dtype_config)
    
    if is_main_process():
        print("\nLoading data...")
    
    data_config = DataConfig(
        train_csv=paths_cfg.get('train_csv', 'train - train.csv'),
        test_csv=paths_cfg.get('test_csv', 'test - test.csv'),
        novel_dir=paths_cfg.get('data_dir', './data'),
        max_backstory_tokens=data_cfg.get('max_backstory_tokens', 512),
        max_chunk_tokens=data_cfg.get('max_chunk_tokens', 512),
        validation_split=data_cfg.get('validation_split', 0.2)
    )
    if 'novel_files' in paths_cfg:
        data_config.novel_files = paths_cfg['novel_files']
    
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=paths_cfg.get('data_dir', './data'),
        config=data_config, batch_size=1, seed=seed
    )
    
    train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=1, sampler=train_sampler, num_workers=0, collate_fn=collate_fn
    )
    
    if is_main_process():
        pos_weight = get_class_weights(train_loader)
    else:
        pos_weight = 1.0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    dist.broadcast(pos_weight_tensor, src=0)
    pos_weight = pos_weight_tensor.item()
    
    if is_main_process():
        print("\nCreating model...")
    
    pipeline_config = PipelineConfig(
        n_layer=model_cfg.get('n_layer', 6), n_embd=model_cfg.get('n_embd', 256),
        n_head=model_cfg.get('n_head', 4), mlp_multiplier=model_cfg.get('mlp_multiplier', 128),
        dropout=model_cfg.get('dropout', 0.1),
        top_k_retrieval=config.get('retrieval', {}).get('top_k', 20),
        aggregation=config.get('aggregation', {}).get('strategy', 'max')
    )
    
    model = BDHNLIPipeline(pipeline_config).to(device)
    # find_unused_parameters=True: Some params may not be used in every forward pass
    # (e.g., retriever selects subset of chunks, some encoder params may be skipped)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if training_cfg.get('compile', False) and hasattr(torch, 'compile'):
        if is_main_process():
            print("Compiling model...")
        model = torch.compile(model)
    
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_cfg.get('learning_rate', 1e-3),
        weight_decay=training_cfg.get('weight_decay', 0.1)
    )
    
    checkpoint_dir = paths_cfg.get('checkpoint_dir', './checkpoints')
    if is_main_process():
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    start_epoch, best_val_acc = 0, 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    epochs = training_cfg.get('epochs', 20)
    grad_accumulation = training_cfg.get('gradient_accumulation', 1)
    save_every = training_cfg.get('save_every', 10)
    
    if is_main_process():
        print(f"\nStarting training: {world_size} GPUs, {len(train_loader)} batches/GPU")
    
    for epoch in range(start_epoch, epochs):
        if is_main_process():
            print(f"\n{'='*60}\nEpoch {epoch + 1}/{epochs}\n{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, ctx, device, pos_weight, grad_accumulation, epoch)
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        if val_loader:
            val_acc, val_loss = validate(model, val_loader, device, ctx)
            if is_main_process():
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"New best! Saving...")
                    save_checkpoint(model, optimizer, scaler, epoch, best_val_acc, config, os.path.join(checkpoint_dir, "best_model.pt"))
        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch, best_val_acc, config, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        dist.barrier()
    
    save_checkpoint(model, optimizer, scaler, epochs - 1, best_val_acc, config, os.path.join(checkpoint_dir, "final_model.pt"))
    
    if is_main_process():
        print(f"\n{'='*60}\nTraining complete! Best accuracy: {best_val_acc:.4f}\n{'='*60}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
