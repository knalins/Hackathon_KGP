#!/usr/bin/env python3
"""
Fast Training Script with Cached Embeddings.

Uses precomputed embeddings for 10x speedup.

Usage:
    1. First precompute embeddings:
       python scripts/precompute_embeddings.py --config config/fast.yaml
    
    2. Then train:
       python scripts/train_fast.py --config config/fast.yaml
"""

import os
import sys
from pathlib import Path
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from src.data.preprocessing import TextNormalizer, format_backstory_input, encode_label
from src.data.tokenizer import ByteTokenizer
from src.models.bdh_encoder import BDHEncoder, BDHConfig
from src.models.retriever import SparseNeuronRetriever
from src.models.cross_encoder import LightweightVerifier
from src.models.aggregator import MILAggregator


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FastNLIModel(nn.Module):
    """Simplified model using cached embeddings."""
    
    def __init__(self, n_embd: int, neuron_dim: int, top_k: int = 10, dropout: float = 0.2):
        super().__init__()
        self.top_k = top_k
        
        # Retriever
        self.retriever = SparseNeuronRetriever(top_k=top_k, similarity_type='cosine')
        
        # Verifier
        self.verifier = LightweightVerifier(
            embedding_dim=n_embd,
            neuron_dim=neuron_dim,
            dropout=dropout
        )
        
        # Simple aggregation
        self.aggregator = MILAggregator(aggregation='max')
    
    def forward(
        self,
        backstory_embedding: torch.Tensor,
        backstory_trace: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        chunk_traces: torch.Tensor
    ) -> dict:
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            backstory_embedding: (D,)
            backstory_trace: (H*N,)
            chunk_embeddings: (num_chunks, D)
            chunk_traces: (num_chunks, H*N)
        """
        # Retrieve top-K
        retrieved_indices, retrieval_scores = self.retriever(
            backstory_trace, chunk_traces
        )
        
        # Score each retrieved chunk
        chunk_scores = []
        for idx in retrieved_indices:
            score = self.verifier(
                chunk_embeddings[idx],
                backstory_embedding,
                chunk_traces[idx],
                backstory_trace
            )
            chunk_scores.append(score)
        
        chunk_scores = torch.stack(chunk_scores).squeeze(-1)
        
        # Aggregate
        agg_result = self.aggregator(
            chunk_scores,
            chunk_traces[retrieved_indices],
            backstory_trace
        )
        
        return {
            'prediction': agg_result['final_score'],
            'chunk_scores': chunk_scores,
            'retrieved_indices': retrieved_indices,
            'best_chunk_idx': retrieved_indices[agg_result['best_chunk_idx']]
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/fast.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fast Training with Cached Embeddings")
    print("=" * 60)
    
    config = load_config(args.config)
    paths_cfg = config['paths']
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = config.get('data', {})
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(config.get('seed', 42))
    
    # Load cached embeddings
    cache_dir = paths_cfg.get('embeddings_cache', './cache')
    print(f"\nLoading cached embeddings from: {cache_dir}")
    
    chunk_cache = {}
    for book_name in paths_cfg.get('novel_files', {}).keys():
        cache_file = os.path.join(cache_dir, f"{book_name.replace(' ', '_')}.pt")
        if not os.path.exists(cache_file):
            print(f"ERROR: Cache not found for {book_name}")
            print(f"Run: python scripts/precompute_embeddings.py --config {args.config}")
            sys.exit(1)
        
        data = torch.load(cache_file, weights_only=False)
        chunk_cache[book_name] = {
            'embeddings': data['embeddings'].to(device),
            'traces': data['neuron_traces'].to(device),
            'texts': data['chunk_texts']
        }
        print(f"  Loaded {book_name}: {data['embeddings'].shape[0]} chunks")
    
    # Load training data
    data_dir = paths_cfg.get('data_dir', './data')
    train_csv = os.path.join(data_dir, paths_cfg.get('train_csv', 'train - train.csv'))
    train_df = pd.read_csv(train_csv)
    
    print(f"\nTrain samples: {len(train_df)}")
    
    # Split
    val_split = data_cfg.get('validation_split', 0.15)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(train_df) * val_split)
    val_df = train_df[:n_val]
    train_df = train_df[n_val:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Setup tokenizer and encoder for backstory
    tokenizer = ByteTokenizer()
    normalizer = TextNormalizer()
    max_backstory = data_cfg.get('max_backstory_tokens', 256)
    
    # Backstory encoder (small)
    bdh_config = BDHConfig(
        n_layer=model_cfg.get('n_layer', 4),
        n_embd=model_cfg.get('n_embd', 256),
        n_head=model_cfg.get('n_head', 4),
        mlp_internal_dim_multiplier=model_cfg.get('mlp_multiplier', 64),
        vocab_size=256,
        dropout=model_cfg.get('dropout', 0.2),
        use_causal_mask=False
    )
    
    backstory_encoder = BDHEncoder(bdh_config).to(device)
    
    # Model
    neuron_dim = bdh_config.total_neurons
    model = FastNLIModel(
        n_embd=bdh_config.n_embd,
        neuron_dim=neuron_dim,
        top_k=config.get('retrieval', {}).get('top_k', 10),
        dropout=model_cfg.get('dropout', 0.2)
    ).to(device)
    
    total_params = sum(p.numel() for p in backstory_encoder.parameters()) + \
                   sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable params: {total_params:,}")
    
    # Optimizer
    all_params = list(backstory_encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=training_cfg.get('learning_rate', 0.001),
        weight_decay=training_cfg.get('weight_decay', 0.01)
    )
    
    # Loss
    pos_count = sum(1 for _, row in train_df.iterrows() if row['label'] == 'consistent')
    neg_count = len(train_df) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    print(f"Class balance - Pos: {pos_count}, Neg: {neg_count}, pos_weight: {pos_weight:.2f}")
    
    # Checkpoint dir
    checkpoint_dir = paths_cfg.get('checkpoint_dir', './checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    epochs = training_cfg.get('epochs', 15)
    validate_every = training_cfg.get('validate_every', 3)
    save_every = training_cfg.get('save_every', 5)
    
    best_val_acc = 0.0
    
    # AMP
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
    
    print(f"\nStarting training: {epochs} epochs")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*60)
        
        # Train
        backstory_encoder.train()
        model.train()
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_df.iterrows(), total=len(train_df), desc="Training")
        
        for idx, row in pbar:
            book_name = row['book_name']
            content = row['content']
            label = 1.0 if row['label'] == 'consistent' else 0.0
            
            # Format backstory
            backstory_text = format_backstory_input(
                book_name=book_name,
                character=row.get('char', ''),
                caption=row.get('caption', ''),
                content=content
            )
            backstory_text = normalizer.normalize(backstory_text)
            
            # Tokenize backstory
            backstory_tokens = tokenizer.encode(backstory_text, max_length=max_backstory, padding=True, return_tensors='pt')
            backstory_tokens = backstory_tokens.unsqueeze(0).to(device)
            
            # Get cached chunks
            cache = chunk_cache[book_name]
            
            optimizer.zero_grad()
            
            with ctx:
                # Encode backstory
                back_result = backstory_encoder(
                    backstory_tokens,
                    return_neuron_trace=True,
                    pool_output=True,
                    use_causal=False
                )
                back_emb = back_result['embedding'].squeeze(0)
                back_trace = back_result['neuron_trace'].squeeze(0)
                
                # Forward
                result = model(
                    back_emb, back_trace,
                    cache['embeddings'], cache['traces']
                )
            
            # Loss (outside autocast for BCE safety)
            pred = result['prediction'].float()
            target = torch.tensor([label], device=device, dtype=torch.float32)
            
            weight = pos_weight if label == 1.0 else 1.0
            loss = F.binary_cross_entropy(pred.view(-1), target, reduction='none') * weight
            loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred_label = 1 if pred.item() > 0.5 else 0
            correct += (pred_label == int(label))
            total += 1
            
            if total % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{train_loss/total:.4f}",
                    'acc': f"{correct/total:.3f}"
                })
        
        train_acc = correct / total
        print(f"Train Loss: {train_loss/total:.4f}, Acc: {train_acc:.4f}")
        
        # Validate
        if (epoch + 1) % validate_every == 0:
            backstory_encoder.eval()
            model.eval()
            
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validating"):
                    book_name = row['book_name']
                    content = row['content']
                    label = 1.0 if row['label'] == 'consistent' else 0.0
                    
                    backstory_text = format_backstory_input(
                        book_name=book_name,
                        character=row.get('char', ''),
                        caption=row.get('caption', ''),
                        content=content
                    )
                    backstory_text = normalizer.normalize(backstory_text)
                    backstory_tokens = tokenizer.encode(backstory_text, max_length=max_backstory, padding=True, return_tensors='pt')
                    backstory_tokens = backstory_tokens.unsqueeze(0).to(device)
                    
                    cache = chunk_cache[book_name]
                    
                    with ctx:
                        back_result = backstory_encoder(
                            backstory_tokens,
                            return_neuron_trace=True,
                            pool_output=True,
                            use_causal=False
                        )
                        back_emb = back_result['embedding'].squeeze(0)
                        back_trace = back_result['neuron_trace'].squeeze(0)
                        
                        result = model(back_emb, back_trace, cache['embeddings'], cache['traces'])
                    
                    pred_label = 1 if result['prediction'].item() > 0.5 else 0
                    val_correct += (pred_label == int(label))
                    val_total += 1
            
            val_acc = val_correct / val_total
            print(f"Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("New best! Saving...")
                torch.save({
                    'epoch': epoch,
                    'backstory_encoder': backstory_encoder.state_dict(),
                    'model': model.state_dict(),
                    'config': config,
                    'best_val_acc': best_val_acc
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'backstory_encoder': backstory_encoder.state_dict(),
                'model': model.state_dict(),
                'config': config
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Final save
    torch.save({
        'epoch': epochs - 1,
        'backstory_encoder': backstory_encoder.state_dict(),
        'model': model.state_dict(),
        'config': config,
        'best_val_acc': best_val_acc
    }, os.path.join(checkpoint_dir, 'final_model.pt'))
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
