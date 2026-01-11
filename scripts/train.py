#!/usr/bin/env python3
"""
Training Script for BDH NLI Pipeline.

Based on official Pathway BDH training patterns.

Usage:
    python train.py                         # Use config/default.yaml
    python train.py --config custom.yaml    # Use custom config
    python train.py --epochs 30             # Override config values
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
import numpy as np
from tqdm import tqdm

from src.data.dataset import create_dataloaders, get_class_weights, DataConfig
from src.models.pipeline import BDHNLIPipeline, PipelineConfig
from src.training.loss import compute_loss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device_and_dtype(device_config: str = "auto", dtype_config: str = "auto"):
    """Setup device and dtype following official BDH patterns with H100 optimizations."""
    # Determine device
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    # Determine dtype
    if dtype_config != "auto":
        dtype = dtype_config
    else:
        # Auto-detect: prefer bfloat16 for H100/A100
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    
    # Autocast context (from official BDH)
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if "cuda" in device.type
        else nullcontext()
    )
    
    # GradScaler - disabled for bfloat16 (not needed on H100/A100)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
    
    # H100/A100 optimizations
    if torch.cuda.is_available():
        # TF32 for Tensor Cores (from official BDH)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Set high precision for matmul (H100 specific)
        torch.set_float32_matmul_precision('high')
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print(f"Using device: {device} with dtype {dtype}")
    
    return device, dtype, ptdtype, ctx, scaler



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train BDH NLI Pipeline")
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML file"
    )
    
    # Override options (these override config file values)
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu/mps)")
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint directory")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    return parser.parse_args()


def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    scaler, 
    ctx, 
    device, 
    pos_weight,
    grad_accumulation,
    log_freq
):
    """Train for one epoch following official BDH patterns."""
    model.train()
    
    loss_acc = 0.0
    loss_steps = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(pbar):
        # Move data to device
        backstory_tokens = batch['backstory_tokens'].to(device)
        chunk_tokens = batch['chunk_tokens'].to(device)
        labels = batch['label'].to(device)
        
        # Handle batch size 1
        if backstory_tokens.dim() == 3:
            backstory_tokens = backstory_tokens.squeeze(0)
        if chunk_tokens.dim() == 3:
            chunk_tokens = chunk_tokens.squeeze(0)
        
        # Forward pass with autocast (official BDH pattern)
        with ctx:
            result = model(
                chunk_tokens=chunk_tokens,
                backstory_tokens=backstory_tokens
            )
            
            loss = compute_loss(
                result['prediction'],
                labels.squeeze().float(),
                pos_weight=pos_weight
            )
        
        # Accumulate loss for logging
        loss_acc += loss.item()
        loss_steps += 1
        
        # Backward with gradient accumulation
        loss = loss / grad_accumulation
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accuracy tracking (apply sigmoid since model outputs logits)
        pred = (torch.sigmoid(result['prediction']) > 0.5).long()
        correct += (pred == labels.squeeze()).sum().item()
        total += 1
        
        # Logging (official BDH pattern)
        if step % log_freq == 0 and loss_steps > 0:
            avg_loss = loss_acc / loss_steps
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{correct/total:.3f}" if total > 0 else "N/A"
            })
    
    # Final step if remaining gradients
    if (step + 1) % grad_accumulation != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return loss_acc / max(loss_steps, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, val_loader, device, ctx):
    """Validate on validation set."""
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    for batch in tqdm(val_loader, desc="Validating"):
        backstory_tokens = batch['backstory_tokens'].to(device)
        chunk_tokens = batch['chunk_tokens'].to(device)
        labels = batch['label'].to(device)
        
        if backstory_tokens.dim() == 3:
            backstory_tokens = backstory_tokens.squeeze(0)
        if chunk_tokens.dim() == 3:
            chunk_tokens = chunk_tokens.squeeze(0)
        
        with ctx:
            result = model(
                chunk_tokens=chunk_tokens,
                backstory_tokens=backstory_tokens
            )
        
        loss = compute_loss(result['prediction'], labels.squeeze().float())
        total_loss += loss.item()
        
        pred = (torch.sigmoid(result['prediction']) > 0.5).long()
        correct += (pred == labels.squeeze()).sum().item()
        total += 1
    
    return correct / max(total, 1), total_loss / max(total, 1)


def save_checkpoint(model, optimizer, scaler, epoch, best_val_acc, config, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_acc': best_val_acc,
        'config': config
    }, path)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BDH NLI Pipeline - Training")
    print("(Based on official Pathway BDH training patterns)")
    print("=" * 60)
    
    # Load config from YAML
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['device'] = args.device
    if args.checkpoint_dir:
        config['paths']['checkpoint_dir'] = args.checkpoint_dir
    if args.compile:
        config['training']['compile'] = True
    
    # Extract config sections
    model_cfg = config['model']
    paths_cfg = config['paths']
    training_cfg = config['training']
    data_cfg = config.get('data', {})
    
    # Set seed (official BDH uses 1337)
    seed = config.get('seed', 1337)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device and dtype (official BDH pattern + H100 optimizations)
    device_config = config.get('device', 'auto')
    dtype_config = config.get('dtype', 'auto')
    device, dtype, ptdtype, ctx, scaler = setup_device_and_dtype(device_config, dtype_config)
    
    # Data config
    data_config = DataConfig(
        train_csv=paths_cfg.get('train_csv', 'train - train.csv'),
        test_csv=paths_cfg.get('test_csv', 'test - test.csv'),
        novel_dir=paths_cfg.get('data_dir', './data'),
        max_backstory_tokens=data_cfg.get('max_backstory_tokens', 512),
        max_chunk_tokens=data_cfg.get('max_chunk_tokens', 512),
        validation_split=data_cfg.get('validation_split', 0.2)
    )
    
    # Override novel files if specified
    if 'novel_files' in paths_cfg:
        data_config.novel_files = paths_cfg['novel_files']
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=paths_cfg.get('data_dir', './data'),
        config=data_config,
        batch_size=1,
        seed=seed
    )
    
    # Get class weights
    pos_weight = get_class_weights(train_loader)
    
    # Pipeline config
    pipeline_config = PipelineConfig(
        n_layer=model_cfg.get('n_layer', 6),
        n_embd=model_cfg.get('n_embd', 256),
        n_head=model_cfg.get('n_head', 4),
        mlp_multiplier=model_cfg.get('mlp_multiplier', 128),
        dropout=model_cfg.get('dropout', 0.1),
        top_k_retrieval=config.get('retrieval', {}).get('top_k', 20),
        aggregation=config.get('aggregation', {}).get('strategy', 'max')
    )
    
    # Create model
    print("\nCreating model...")
    model = BDHNLIPipeline(pipeline_config).to(device)
    
    # Compile model (official BDH pattern)
    if training_cfg.get('compile', False) and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Optimizer (official BDH pattern)
    base_lr = training_cfg.get('learning_rate', 1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=training_cfg.get('weight_decay', 0.1)
    )
    
    # Learning rate scheduler with warmup
    epochs = training_cfg.get('epochs', 20)
    warmup_epochs = training_cfg.get('warmup_epochs', 2)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create checkpoint dir
    checkpoint_dir = paths_cfg.get('checkpoint_dir', './checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Resume if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    epochs = training_cfg.get('epochs', 20)
    grad_accumulation = training_cfg.get('gradient_accumulation', 4)
    log_freq = training_cfg.get('log_freq', 10)
    save_every = training_cfg.get('save_every', 5)
    early_stopping_patience = training_cfg.get('early_stopping_patience', 5)
    
    # Early stopping tracking
    epochs_without_improvement = 0
    best_val_loss = float('inf')
    
    print(f"\nStarting training on {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            ctx=ctx,
            device=device,
            pos_weight=pos_weight,
            grad_accumulation=grad_accumulation,
            log_freq=log_freq
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {current_lr:.6f}")
        
        # Validate
        if val_loader:
            val_acc, val_loss = validate(model, val_loader, device, ctx)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model (by accuracy)
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                print(f"New best model! Accuracy: {val_acc:.4f}")
                save_checkpoint(
                    model, optimizer, scaler, epoch, best_val_acc,
                    config,
                    os.path.join(checkpoint_dir, "best_model.pt")
                )
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING: No improvement for {early_stopping_patience} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                print(f"{'='*60}")
                break
        
        # Regular checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch, best_val_acc,
                config,
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scaler, epochs - 1, best_val_acc,
        config,
        os.path.join(checkpoint_dir, "final_model.pt")
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
