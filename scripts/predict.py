#!/usr/bin/env python3
"""
Inference Script for BDH NLI Pipeline.

Usage:
    python predict.py --checkpoint checkpoints/best_model.pt
    python predict.py --checkpoint checkpoints/best_model.pt --config config/default.yaml
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch

from src.data.dataset import create_dataloaders, DataConfig
from src.inference.predictor import Predictor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with BDH NLI Pipeline")
    
    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML file"
    )
    
    # Overrides
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV path")
    parser.add_argument("--include_explanation", action="store_true", help="Include explanations")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BDH NLI Pipeline - Inference")
    print("=" * 60)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    paths_cfg = config['paths']
    data_cfg = config.get('data', {})
    
    if args.data_dir:
        paths_cfg['data_dir'] = args.data_dir
    
    # Determine device
    if args.device:
        device = args.device
    elif config.get('device', 'auto') == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['device']
    
    # Load predictor
    print(f"\nLoading model from {args.checkpoint}...")
    predictor = Predictor.from_checkpoint(args.checkpoint, device)
    predictor.threshold = args.threshold
    
    # Create data config
    data_config = DataConfig(
        train_csv=paths_cfg.get('train_csv', 'train - train.csv'),
        test_csv=paths_cfg.get('test_csv', 'test - test.csv'),
        novel_dir=paths_cfg.get('data_dir', './data'),
        max_backstory_tokens=data_cfg.get('max_backstory_tokens', 512),
        max_chunk_tokens=data_cfg.get('max_chunk_tokens', 512),
        validation_split=0  # No validation for inference
    )
    
    # Override novel files if specified
    if 'novel_files' in paths_cfg:
        data_config.novel_files = paths_cfg['novel_files']
    
    # Create dataloader for test data
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=paths_cfg.get('data_dir', './data'),
        config=data_config,
        batch_size=1
    )
    
    # Get chunk texts for explanations
    chunk_texts_by_book = {}
    if args.include_explanation:
        for batch in test_loader:
            book = batch['book_name'][0]
            if book not in chunk_texts_by_book:
                chunk_texts_by_book[book] = batch['chunk_texts']
    
    # Run inference
    print(f"\nRunning inference on {len(test_loader)} samples...")
    results = predictor.run_inference(
        test_loader=test_loader,
        output_path=args.output,
        chunk_texts_by_book=chunk_texts_by_book if args.include_explanation else None
    )
    
    print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
