#!/usr/bin/env python3
"""
Fast Inference Script.

Uses cached embeddings for fast prediction.

Usage:
    python scripts/predict_fast.py --checkpoint ./checkpoints/best_model.pt --output results.csv
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import pandas as pd
from tqdm import tqdm

from src.data.preprocessing import TextNormalizer, format_backstory_input
from src.data.tokenizer import ByteTokenizer
from src.models.bdh_encoder import BDHEncoder, BDHConfig
from src.models.retriever import SparseNeuronRetriever
from src.models.cross_encoder import LightweightVerifier
from src.models.aggregator import MILAggregator


class FastNLIModel(torch.nn.Module):
    """Same model as train_fast.py."""
    
    def __init__(self, n_embd: int, neuron_dim: int, top_k: int = 10, dropout: float = 0.0):
        super().__init__()
        self.top_k = top_k
        self.retriever = SparseNeuronRetriever(top_k=top_k, similarity_type='cosine')
        self.verifier = LightweightVerifier(
            embedding_dim=n_embd,
            neuron_dim=neuron_dim,
            dropout=dropout
        )
        self.aggregator = MILAggregator(aggregation='max')
    
    def forward(self, backstory_embedding, backstory_trace, chunk_embeddings, chunk_traces):
        retrieved_indices, _ = self.retriever(backstory_trace, chunk_traces)
        
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
        agg_result = self.aggregator(chunk_scores, chunk_traces[retrieved_indices], backstory_trace)
        
        return {
            'prediction': agg_result['final_score'],
            'best_chunk_idx': retrieved_indices[agg_result['best_chunk_idx']]
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fast Inference")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    paths_cfg = config['paths']
    model_cfg = config['model']
    data_cfg = config.get('data', {})
    
    # Load cached embeddings
    cache_dir = paths_cfg.get('embeddings_cache', './cache')
    print(f"Loading cached embeddings from: {cache_dir}")
    
    chunk_cache = {}
    for book_name in paths_cfg.get('novel_files', {}).keys():
        cache_file = os.path.join(cache_dir, f"{book_name.replace(' ', '_')}.pt")
        data = torch.load(cache_file, weights_only=False)
        chunk_cache[book_name] = {
            'embeddings': data['embeddings'].to(device),
            'traces': data['neuron_traces'].to(device),
            'texts': data['chunk_texts']
        }
        print(f"  Loaded {book_name}: {data['embeddings'].shape[0]} chunks")
    
    # Create models
    bdh_config = BDHConfig(
        n_layer=model_cfg.get('n_layer', 4),
        n_embd=model_cfg.get('n_embd', 256),
        n_head=model_cfg.get('n_head', 4),
        mlp_internal_dim_multiplier=model_cfg.get('mlp_multiplier', 64),
        vocab_size=256,
        dropout=0.0,
        use_causal_mask=False
    )
    
    backstory_encoder = BDHEncoder(bdh_config).to(device)
    backstory_encoder.load_state_dict(checkpoint['backstory_encoder'])
    backstory_encoder.eval()
    
    neuron_dim = bdh_config.total_neurons
    model = FastNLIModel(
        n_embd=bdh_config.n_embd,
        neuron_dim=neuron_dim,
        top_k=config.get('retrieval', {}).get('top_k', 10),
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"\nModels loaded successfully")
    
    # Load test data
    data_dir = paths_cfg.get('data_dir', './data')
    test_csv = os.path.join(data_dir, paths_cfg.get('test_csv', 'test - test.csv'))
    test_df = pd.read_csv(test_csv)
    
    print(f"Test samples: {len(test_df)}")
    
    # Setup
    tokenizer = ByteTokenizer()
    normalizer = TextNormalizer()
    max_backstory = data_cfg.get('max_backstory_tokens', 256)
    
    # Predict
    results = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            book_name = row['book_name']
            content = row['content']
            sample_id = row['id']
            
            # Format backstory
            backstory_text = format_backstory_input(
                book_name=book_name,
                character=row.get('char', ''),
                caption=row.get('caption', ''),
                content=content
            )
            backstory_text = normalizer.normalize(backstory_text)
            
            # Tokenize
            backstory_tokens = tokenizer.encode(backstory_text, max_length=max_backstory, padding=True, return_tensors='pt')
            backstory_tokens = backstory_tokens.unsqueeze(0).to(device)
            
            # Get cache
            cache = chunk_cache[book_name]
            
            # Encode backstory
            back_result = backstory_encoder(
                backstory_tokens,
                return_neuron_trace=True,
                pool_output=True,
                use_causal=False
            )
            back_emb = back_result['embedding'].squeeze(0)
            back_trace = back_result['neuron_trace'].squeeze(0)
            
            # Predict
            result = model(back_emb, back_trace, cache['embeddings'], cache['traces'])
            
            prob = result['prediction'].item()
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
