#!/usr/bin/env python3
"""
Simple Baseline Prediction Script.

Usage:
    python scripts/predict_simple.py --checkpoint ./checkpoints/best_model_simple.pt --output results.csv
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class SimpleNLIClassifier(nn.Module):
    """Simple classifier using pretrained sentence embeddings."""
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        input_dim = embedding_dim * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, chunk_emb: torch.Tensor, backstory_emb: torch.Tensor) -> torch.Tensor:
        if chunk_emb.dim() == 1:
            chunk_emb = chunk_emb.unsqueeze(0)
            backstory_emb = backstory_emb.unsqueeze(0)
        
        combined = torch.cat([
            chunk_emb,
            backstory_emb,
            chunk_emb * backstory_emb,
            torch.abs(chunk_emb - backstory_emb)
        ], dim=-1)
        
        return self.classifier(combined)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list:
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunks.append(text[pos:end])
        pos = end - overlap if end < len(text) else end
    return chunks


def load_novel(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    if '*** START' in text:
        text = text.split('*** START')[1]
    if '*** END' in text:
        text = text.split('*** END')[0]
    return text.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model_simple.pt")
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Simple Baseline Prediction")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load sentence encoder
    print("\nLoading sentence-transformers model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
    embedding_dim = encoder.get_sentence_embedding_dimension()
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Create and load model
    model = SimpleNLIClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        dropout=0.0  # No dropout for inference
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Load data
    data_dir = "./data"
    
    print("\nLoading novels...")
    novels = {
        "In Search of the Castaways": load_novel(os.path.join(data_dir, "In search of the castaways.txt")),
        "The Count of Monte Cristo": load_novel(os.path.join(data_dir, "The Count of Monte Cristo.txt"))
    }
    
    print("\nChunking and encoding novels...")
    chunk_embeddings = {}
    for name, text in novels.items():
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        embeddings = encoder.encode(chunks, show_progress_bar=True, convert_to_tensor=True)
        chunk_embeddings[name] = embeddings.to(device)
        print(f"  {name}: {len(chunks)} chunks")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(os.path.join(data_dir, "test - test.csv"))
    print(f"Test samples: {len(test_df)}")
    
    # Predict
    results = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            book_name = row['book_name']
            backstory = row['content']
            sample_id = row['id']
            
            # Encode backstory
            backstory_emb = encoder.encode(backstory, convert_to_tensor=True).to(device)
            
            # Get chunk embeddings
            chunk_embs = chunk_embeddings[book_name]
            
            # Find top-K similar chunks
            similarities = F.cosine_similarity(backstory_emb.unsqueeze(0), chunk_embs)
            top_k_indices = torch.topk(similarities, min(args.top_k, len(chunk_embs))).indices
            
            # Score each chunk
            chunk_logits = []
            for chunk_idx in top_k_indices:
                logit = model(chunk_embs[chunk_idx], backstory_emb)
                chunk_logits.append(logit)
            
            chunk_logits = torch.cat(chunk_logits)
            final_logit = chunk_logits.max()
            
            prob = torch.sigmoid(final_logit).item()
            label = 'consistent' if prob >= args.threshold else 'contradict'
            
            results.append({
                'id': sample_id,
                'label': label,
                'confidence': prob
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    
    # Summary
    consistent = sum(1 for r in results if r['label'] == 'consistent')
    contradict = len(results) - consistent
    print(f"Consistent: {consistent} ({consistent/len(results)*100:.1f}%)")
    print(f"Contradict: {contradict} ({contradict/len(results)*100:.1f}%)")
    print('='*60)


if __name__ == "__main__":
    main()
