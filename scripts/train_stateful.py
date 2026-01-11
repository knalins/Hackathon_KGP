#!/usr/bin/env python3
"""
Stateful BDH Training Script.

Uses StatefulBDHProcessor to process entire novel without chunking,
exploiting BDH's key advantage of incremental state accumulation.

Usage:
    python scripts/train_stateful.py --config config/h100_stable.yaml --epochs 50
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
from src.data.chunking import NovelChunker, ChunkConfig
from src.models.stateful_processor import StatefulBDHProcessor, StatefulState
from src.models.bdh_encoder import BDHConfig
from src.models.neuron_interpreter import NeuronInterpreter
from src.models.cross_encoder import LightweightVerifier
from src.models.aggregator import MILAggregator


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class StatefulNLIModel(nn.Module):
    """
    Full NLI model using stateful BDH processing.
    
    Key advantages:
    1. Processes entire novel statelessly - no information loss
    2. Backstory queries full novel context
    3. Interpretable through NeuronInterpreter
    """
    
    def __init__(
        self,
        config: BDHConfig,
        top_k: int = 10,
        use_interpreter: bool = True
    ):
        super().__init__()
        
        self.stateful_processor = StatefulBDHProcessor(config)
        
        neuron_dim = config.total_neurons
        emb_dim = config.n_embd
        
        # Project state neurons to embedding space
        self.state_proj = nn.Linear(neuron_dim, emb_dim)
        
        # Verifier head: compare backstory embedding with projected state
        self.verifier = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim // 2, 1)  # Output logits
        )
        
        # Optional neuron interpreter
        self.use_interpreter = use_interpreter
        if use_interpreter:
            self.interpreter = NeuronInterpreter(neuron_dim, num_concepts=32)
    
    def forward(
        self,
        novel_chunks: list,
        backstory_ids: torch.Tensor
    ) -> dict:
        """
        Forward pass with stateful novel processing.
        
        Args:
            novel_chunks: List of token tensors for novel chunks
            backstory_ids: Backstory tokens (B, T)
            
        Returns:
            Dict with prediction and explanations
        """
        # Process novel and query with backstory
        result = self.stateful_processor(novel_chunks, backstory_ids)
        
        # Get backstory embedding and projected state
        backstory_emb = result['embedding']  # (B, D)
        state_neurons = result['state_neurons']  # (B, H*N)
        
        # Project state neurons to embedding space
        state_emb = self.state_proj(state_neurons)  # (B, D)
        
        # Combine backstory with state for verification
        combined = torch.cat([backstory_emb, state_emb], dim=-1)  # (B, 2*D)
        logit = self.verifier(combined)
        
        output = {
            'logit': logit,
            'prediction': torch.sigmoid(logit),
            'backstory_trace': result['neuron_trace'],
            'state_neurons': state_neurons,
            'num_tokens': result['num_tokens_processed']
        }
        
        # Add interpretations
        if self.use_interpreter:
            concepts = self.interpreter(result['neuron_trace'])
            output['concepts'] = concepts
        
        return output
    
    def explain(
        self,
        novel_chunks: list,
        backstory_ids: torch.Tensor
    ) -> dict:
        """Generate human-readable explanation."""
        with torch.no_grad():
            result = self(novel_chunks, backstory_ids)
            
            if self.use_interpreter:
                backstory_trace = result['backstory_trace'].squeeze(0)
                concepts = self.interpreter.get_top_concepts(backstory_trace, top_k=5)
                
                pred = result['prediction'].item()
                label = "CONSISTENT" if pred > 0.5 else "CONTRADICTORY"
                
                return {
                    'label': label,
                    'confidence': f"{max(pred, 1-pred):.1%}",
                    'activated_concepts': concepts,
                    'tokens_processed': result['num_tokens']
                }
        
        return {'label': 'unknown', 'confidence': '0%'}


def load_novel_as_chunks(
    novel_path: str,
    tokenizer: ByteTokenizer,
    max_chunk_tokens: int = 512
) -> list:
    """Load novel and split into token chunks."""
    with open(novel_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove Gutenberg headers
    if '*** START' in text:
        text = text.split('*** START')[1]
    if '*** END' in text:
        text = text.split('*** END')[0]
    text = text.strip()
    
    # Simple character chunking for tokenization
    char_chunk_size = max_chunk_tokens * 4  # ~4 chars per token
    chunks = []
    
    for i in range(0, len(text), char_chunk_size):
        chunk_text = text[i:i + char_chunk_size]
        tokens = tokenizer.encode(
            chunk_text, 
            max_length=max_chunk_tokens,
            padding=True,
            return_tensors='pt'
        )
        chunks.append(tokens)
    
    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/h100_stable.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stateful BDH Training")
    print("=" * 60)
    
    config = load_config(args.config)
    paths_cfg = config['paths']
    model_cfg = config['model']
    training_cfg = config.get('training', {})
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Seed
    torch.manual_seed(config.get('seed', 42))
    
    # Setup
    tokenizer = ByteTokenizer()
    normalizer = TextNormalizer()
    max_tokens = config.get('data', {}).get('max_chunk_tokens', 512)
    max_backstory = config.get('data', {}).get('max_backstory_tokens', 512)
    
    # Create BDH config
    bdh_config = BDHConfig(
        n_layer=model_cfg.get('n_layer', 6),
        n_embd=model_cfg.get('n_embd', 256),
        n_head=model_cfg.get('n_head', 4),
        mlp_internal_dim_multiplier=model_cfg.get('mlp_multiplier', 128),
        vocab_size=256,
        dropout=model_cfg.get('dropout', 0.1)
    )
    
    # Load novels as chunks (FULL NOVEL - no_grad processing enables this)
    print("\nLoading and chunking novels...")
    data_dir = paths_cfg.get('data_dir', './data')
    novel_files = paths_cfg.get('novel_files', {})
    
    novel_chunks = {}
    for book_name, filename in novel_files.items():
        novel_path = os.path.join(data_dir, filename)
        chunks = load_novel_as_chunks(novel_path, tokenizer, max_tokens)
        novel_chunks[book_name] = chunks
        print(f"  {book_name}: {len(chunks)} chunks (FULL)")
    
    # Create model
    model = StatefulNLIModel(
        config=bdh_config,
        top_k=config.get('retrieval', {}).get('top_k', 10),
        use_interpreter=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total_params:,}")
    
    # Load training data
    train_csv = os.path.join(data_dir, paths_cfg.get('train_csv', 'train - train.csv'))
    train_df = pd.read_csv(train_csv)
    print(f"Train samples: {len(train_df)}")
    
    # Split
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(train_df) * 0.15)
    val_df = train_df[:n_val]
    train_df = train_df[n_val:]
    
    # Class balance
    pos_count = sum(1 for _, row in train_df.iterrows() if row['label'] == 'consistent')
    neg_count = len(train_df) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    print(f"Class balance - Pos: {pos_count}, Neg: {neg_count}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 0.001),
        weight_decay=training_cfg.get('weight_decay', 0.01)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Checkpoints
    checkpoint_dir = paths_cfg.get('checkpoint_dir', './checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    epochs = args.epochs or training_cfg.get('epochs', 50)
    early_stopping_patience = training_cfg.get('early_stopping_patience', 3)
    epochs_without_improvement = 0
    
    print(f"\nTraining for {epochs} epochs (early stopping patience: {early_stopping_patience})...")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}")
        
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
            
            # Tokenize
            backstory_tokens = tokenizer.encode(
                backstory_text, 
                max_length=max_backstory, 
                padding=True,
                return_tensors='pt'
            ).unsqueeze(0).to(device)
            
            # Get novel chunks
            chunks = novel_chunks[book_name]
            
            optimizer.zero_grad()
            
            # Forward
            result = model(chunks, backstory_tokens)
            
            # Loss
            target = torch.tensor([[label]], device=device)
            loss = criterion(result['logit'], target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = 1 if result['prediction'].item() > 0.5 else 0
            correct += (pred == int(label))
            total += 1
            
            pbar.set_postfix({
                'loss': f"{train_loss/total:.4f}",
                'acc': f"{correct/total:.3f}"
            })
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_correct = 0
            
            with torch.no_grad():
                for _, row in val_df.iterrows():
                    book_name = row['book_name']
                    backstory_text = format_backstory_input(
                        book_name=book_name,
                        character=row.get('char', ''),
                        caption=row.get('caption', ''),
                        content=row['content']
                    )
                    backstory_text = normalizer.normalize(backstory_text)
                    backstory_tokens = tokenizer.encode(
                        backstory_text, max_length=max_backstory, 
                        padding=True, return_tensors='pt'
                    ).unsqueeze(0).to(device)
                    
                    result = model(novel_chunks[book_name], backstory_tokens)
                    pred = 1 if result['prediction'].item() > 0.5 else 0
                    label = 1 if row['label'] == 'consistent' else 0
                    val_correct += (pred == label)
            
            val_acc = val_correct / len(val_df)
            print(f"  Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save({
                    'model': model.state_dict(),
                    'config': config,
                    'bdh_config': bdh_config.__dict__,
                    'val_acc': val_acc
                }, os.path.join(checkpoint_dir, 'best_stateful_model.pt'))
                print("  Saved best model!")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} validation(s)")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEARLY STOPPING: No improvement for {early_stopping_patience} validations")
                break
    
    print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
