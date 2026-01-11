#!/usr/bin/env python3
"""
Precompute chunk embeddings ONCE.
Saves to cache for fast training.

Usage:
    python scripts/precompute_embeddings.py --config config/fast.yaml
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from tqdm import tqdm

from src.data.preprocessing import TextNormalizer
from src.data.chunking import NovelChunker, ChunkConfig
from src.data.tokenizer import ByteTokenizer
from src.models.bdh_encoder import BDHEncoder, BDHConfig


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/fast.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Precomputing Chunk Embeddings")
    print("=" * 60)
    
    config = load_config(args.config)
    paths_cfg = config['paths']
    model_cfg = config['model']
    data_cfg = config.get('data', {})
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create cache directory
    cache_dir = paths_cfg.get('embeddings_cache', './cache')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Create BDH encoder (small version)
    bdh_config = BDHConfig(
        n_layer=model_cfg.get('n_layer', 4),
        n_embd=model_cfg.get('n_embd', 256),
        n_head=model_cfg.get('n_head', 4),
        mlp_internal_dim_multiplier=model_cfg.get('mlp_multiplier', 64),
        vocab_size=256,
        dropout=0.0,  # No dropout for inference
        use_causal_mask=False  # Bidirectional
    )
    
    encoder = BDHEncoder(bdh_config).to(device)
    encoder.eval()
    
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Setup tokenizer and chunker
    tokenizer = ByteTokenizer()
    normalizer = TextNormalizer()
    
    chunk_config = ChunkConfig(
        target_chars=data_cfg.get('chunk_size', 1024),
        overlap_chars=data_cfg.get('chunk_overlap', 256)
    )
    chunker = NovelChunker(chunk_config)
    
    max_tokens = data_cfg.get('max_chunk_tokens', 256)
    
    # Process each novel
    novel_files = paths_cfg.get('novel_files', {})
    data_dir = paths_cfg.get('data_dir', './data')
    
    for book_name, filename in novel_files.items():
        print(f"\nProcessing: {book_name}")
        
        # Load novel
        novel_path = os.path.join(data_dir, filename)
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Normalize
        text = normalizer.normalize(text)
        
        # Chunk
        chunks = chunker.chunk_text(text)
        print(f"  Chunks: {len(chunks)}")
        
        # Tokenize
        chunk_texts = [c['text'] for c in chunks]
        chunk_tokens = tokenizer.batch_encode(
            chunk_texts, 
            max_length=max_tokens, 
            padding=True,
            return_tensors='pt'
        ).to(device)
        print(f"  Token shape: {chunk_tokens.shape}")
        
        # Encode in batches
        batch_size = 64
        all_embeddings = []
        all_traces = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(chunk_tokens), batch_size), desc="  Encoding"):
                batch = chunk_tokens[i:i + batch_size]
                
                result = encoder(
                    batch,
                    return_neuron_trace=True,
                    pool_output=True,
                    use_causal=False
                )
                
                all_embeddings.append(result['embedding'].cpu())
                all_traces.append(result['neuron_trace'].cpu())
        
        embeddings = torch.cat(all_embeddings, dim=0)
        traces = torch.cat(all_traces, dim=0)
        
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Traces: {traces.shape}")
        
        # Save to cache
        cache_file = os.path.join(cache_dir, f"{book_name.replace(' ', '_')}.pt")
        torch.save({
            'book_name': book_name,
            'chunk_texts': chunk_texts,
            'chunk_tokens': chunk_tokens.cpu(),
            'embeddings': embeddings,
            'neuron_traces': traces,
            'config': {
                'n_layer': bdh_config.n_layer,
                'n_embd': bdh_config.n_embd,
                'max_tokens': max_tokens
            }
        }, cache_file)
        
        print(f"  Saved: {cache_file}")
    
    print("\n" + "=" * 60)
    print("Precomputation complete!")
    print(f"Cache saved to: {cache_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
