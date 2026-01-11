#!/usr/bin/env python3
"""
Simple Baseline Training with Pretrained Sentence Encoder.

Uses sentence-transformers for meaningful embeddings.
Fixes identified bugs: no double sigmoid, end-to-end training.

Usage:
    pip install sentence-transformers
    python scripts/train_simple.py --epochs 50
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class SimpleNLIClassifier(nn.Module):
    """Simple classifier using pretrained sentence embeddings."""
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        # Compare chunk embedding with backstory embedding
        # Input: [chunk_emb, backstory_emb, chunk_emb * backstory_emb, |chunk_emb - backstory_emb|]
        input_dim = embedding_dim * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output LOGITS, not probabilities
        )
    
    def forward(self, chunk_emb: torch.Tensor, backstory_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_emb: (D,) or (B, D)
            backstory_emb: (D,) or (B, D)
        
        Returns:
            Logits (not sigmoid) for BCEWithLogitsLoss
        """
        if chunk_emb.dim() == 1:
            chunk_emb = chunk_emb.unsqueeze(0)
            backstory_emb = backstory_emb.unsqueeze(0)
        
        # Combine features
        combined = torch.cat([
            chunk_emb,
            backstory_emb,
            chunk_emb * backstory_emb,  # Element-wise product
            torch.abs(chunk_emb - backstory_emb)  # Absolute difference
        ], dim=-1)
        
        return self.classifier(combined)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list:
    """Simple character-based chunking."""
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunks.append(text[pos:end])
        pos = end - overlap if end < len(text) else end
    return chunks


def load_novel(filepath: str) -> str:
    """Load and clean novel text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Remove Project Gutenberg header/footer
    if '*** START' in text:
        text = text.split('*** START')[1]
    if '*** END' in text:
        text = text.split('*** END')[0]
    return text.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Simple Baseline Training")
    print("=" * 60)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load pretrained sentence encoder
    print("\nLoading sentence-transformers model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
    embedding_dim = encoder.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embedding_dim}")
    
    # Load data
    data_dir = "./data"
    
    print("\nLoading novels...")
    novels = {
        "In Search of the Castaways": load_novel(os.path.join(data_dir, "In search of the castaways.txt")),
        "The Count of Monte Cristo": load_novel(os.path.join(data_dir, "The Count of Monte Cristo.txt"))
    }
    
    # Chunk novels
    print("\nChunking novels...")
    novel_chunks = {}
    for name, text in novels.items():
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        novel_chunks[name] = chunks
        print(f"  {name}: {len(chunks)} chunks")
    
    # Encode all chunks (this is expensive but only once)
    print("\nEncoding chunks (this may take a few minutes)...")
    chunk_embeddings = {}
    for name, chunks in novel_chunks.items():
        embeddings = encoder.encode(chunks, show_progress_bar=True, convert_to_tensor=True)
        chunk_embeddings[name] = embeddings.to(device)
        print(f"  {name}: {embeddings.shape}")
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv(os.path.join(data_dir, "train - train.csv"))
    print(f"Total samples: {len(train_df)}")
    
    # Split
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(train_df) * 0.15)
    val_df = train_df[:n_val]
    train_df = train_df[n_val:]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Class balance
    pos_count = sum(1 for _, row in train_df.iterrows() if row['label'] == 'consistent')
    neg_count = len(train_df) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    print(f"Class balance - Pos: {pos_count}, Neg: {neg_count}, pos_weight: {pos_weight.item():.2f}")
    
    # Create model
    model = SimpleNLIClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        dropout=0.3
    ).to(device)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # IMPORTANT: logits, not sigmoid!
    
    # Checkpoint
    checkpoint_dir = "./checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    
    print(f"\nStarting training: {args.epochs} epochs")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for idx, row in pbar:
            book_name = row['book_name']
            backstory = row['content']
            label = 1.0 if row['label'] == 'consistent' else 0.0
            
            # Encode backstory
            with torch.no_grad():
                backstory_emb = encoder.encode(backstory, convert_to_tensor=True).to(device)
            
            # Get chunk embeddings for this book
            chunk_embs = chunk_embeddings[book_name]
            
            # Find top-K similar chunks
            with torch.no_grad():
                similarities = F.cosine_similarity(backstory_emb.unsqueeze(0), chunk_embs)
                top_k_indices = torch.topk(similarities, min(args.top_k, len(chunk_embs))).indices
            
            # Score each top-K chunk and aggregate
            chunk_logits = []
            for chunk_idx in top_k_indices:
                logit = model(chunk_embs[chunk_idx], backstory_emb)
                chunk_logits.append(logit)
            
            chunk_logits = torch.cat(chunk_logits)
            
            # MIL aggregation: max pooling on logits
            final_logit = chunk_logits.max()
            
            # Loss
            target = torch.tensor([label], device=device)
            loss = criterion(final_logit.view(-1), target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = 1 if torch.sigmoid(final_logit).item() > 0.5 else 0
            correct += (pred == int(label))
            total += 1
            
            pbar.set_postfix({
                'loss': f"{train_loss/total:.4f}",
                'acc': f"{correct/total:.3f}"
            })
        
        train_acc = correct / total
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for _, row in val_df.iterrows():
                    book_name = row['book_name']
                    backstory = row['content']
                    label = 1.0 if row['label'] == 'consistent' else 0.0
                    
                    backstory_emb = encoder.encode(backstory, convert_to_tensor=True).to(device)
                    chunk_embs = chunk_embeddings[book_name]
                    
                    similarities = F.cosine_similarity(backstory_emb.unsqueeze(0), chunk_embs)
                    top_k_indices = torch.topk(similarities, min(args.top_k, len(chunk_embs))).indices
                    
                    chunk_logits = []
                    for chunk_idx in top_k_indices:
                        logit = model(chunk_embs[chunk_idx], backstory_emb)
                        chunk_logits.append(logit)
                    
                    chunk_logits = torch.cat(chunk_logits)
                    final_logit = chunk_logits.max()
                    
                    pred = 1 if torch.sigmoid(final_logit).item() > 0.5 else 0
                    val_correct += (pred == int(label))
                    val_total += 1
            
            val_acc = val_correct / val_total
            print(f"\n  Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("  New best! Saving...")
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'args': vars(args)
                }, os.path.join(checkpoint_dir, 'best_model_simple.pt'))
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, os.path.join(checkpoint_dir, f'checkpoint_simple_epoch_{epoch+1}.pt'))
    
    # Final save
    torch.save({
        'model': model.state_dict(),
        'epoch': args.epochs - 1,
        'val_acc': best_val_acc,
        'args': vars(args)
    }, os.path.join(checkpoint_dir, 'final_model_simple.pt'))
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best val acc: {best_val_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
