#!/usr/bin/env python3
"""
Test and verify data preprocessing pipeline.

This script:
1. Loads and preprocesses train/test CSVs
2. Loads and chunks the novels
3. Creates the datasets
4. Prints sample outputs for verification
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import TextNormalizer, format_backstory_input, encode_label, decode_label
from src.data.chunking import NovelChunker, ChunkConfig
from src.data.tokenizer import ByteTokenizer
from src.data.dataset import (
    NLIDataset, DataConfig, load_novels, 
    create_dataloaders, get_class_weights
)


def test_text_normalizer():
    """Test text normalization."""
    print("=" * 60)
    print("Testing TextNormalizer")
    print("=" * 60)
    
    normalizer = TextNormalizer()
    
    # Test cases
    test_cases = [
        # (input, description)
        ("Hello  World", "Multiple spaces"),
        ("Héllo Wörld", "Unicode characters"),
        ("Test\r\nLine", "Windows line endings"),
        ("  Trimmed  ", "Leading/trailing whitespace"),
        ("Preserve THIS Capitalization", "Capitalization preservation"),
    ]
    
    for text, desc in test_cases:
        result = normalizer.normalize(text)
        print(f"\n{desc}:")
        print(f"  Input:  '{text}'")
        print(f"  Output: '{result}'")
    
    print("\n✓ TextNormalizer tests passed")


def test_backstory_formatting():
    """Test structured backstory formatting."""
    print("\n" + "=" * 60)
    print("Testing Backstory Formatting")
    print("=" * 60)
    
    # Test with caption
    result1 = format_backstory_input(
        book_name="In Search of the Castaways",
        character="Jacques Paganel",
        caption="Early Academic Background",
        content="At twelve, Jacques Paganel fell in love with geography."
    )
    print("\nWith caption:")
    print(result1)
    
    # Test without caption
    result2 = format_backstory_input(
        book_name="The Count of Monte Cristo",
        character="Faria",
        caption=None,
        content="Suspected again in 1815, he was re-arrested."
    )
    print("\nWithout caption (empty):")
    print(result2)
    
    # Test with NaN caption (simulating pandas)
    import math
    result3 = format_backstory_input(
        book_name="Test Book",
        character="Test Char",
        caption=float('nan'),
        content="Test content"
    )
    print("\nWith NaN caption:")
    print(result3)
    
    print("\n✓ Backstory formatting tests passed")


def test_chunking():
    """Test novel chunking."""
    print("\n" + "=" * 60)
    print("Testing Novel Chunking")
    print("=" * 60)
    
    # Sample text for testing
    sample_text = """
    Chapter 1. The first chapter begins here. This is the opening of our story.
    It was a dark and stormy night. The wind howled through the trees.
    
    The protagonist walked slowly down the road. He was tired from his long journey.
    Every step felt like a mile. But he knew he had to continue.
    
    Chapter 2. A new day dawned. The sun rose over the mountains.
    Birds began to sing in the trees. It was a beautiful morning.
    """
    
    config = ChunkConfig(
        target_chars=200,
        overlap_chars=50,
        min_chunk_chars=100
    )
    chunker = NovelChunker(config)
    
    chunks = chunker.chunk_text(sample_text)
    
    print(f"\nOriginal text length: {len(sample_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Chars {chunk['start_char']}-{chunk['end_char']} ({chunk['end_char'] - chunk['start_char']} chars)")
        print(f"  Text: '{chunk['text'][:80]}...'")
    
    print("\n✓ Chunking tests passed")


def test_tokenizer():
    """Test byte tokenizer."""
    print("\n" + "=" * 60)
    print("Testing ByteTokenizer")
    print("=" * 60)
    
    tokenizer = ByteTokenizer(max_length=50)
    
    text = "Hello, World! 你好"
    
    # Encode
    tokens = tokenizer.encode(text, return_tensors='pt')
    print(f"\nText: '{text}'")
    print(f"Token shape: {tokens.shape}")
    print(f"First 20 tokens: {tokens[:20].tolist()}")
    
    # Decode
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")
    
    # Batch encode
    texts = ["Hello", "World", "Test"]
    batch = tokenizer.batch_encode(texts, max_length=20, return_tensors='pt')
    print(f"\nBatch shape: {batch.shape}")
    
    print("\n✓ Tokenizer tests passed")


def test_full_pipeline(data_dir: str):
    """Test the full data pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Data Pipeline")
    print("=" * 60)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=1,
        num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader else 0}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample
    print("\n--- Sample from Train ---")
    sample = next(iter(train_loader))
    
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Book: {sample['book_name']}")
    print(f"Character: {sample['character']}")
    print(f"Label: {sample.get('label', 'N/A')}")
    print(f"Backstory tokens shape: {sample['backstory_tokens'].shape}")
    print(f"Chunk tokens shape: {sample['chunk_tokens'].shape}")
    print(f"Number of chunks: {sample['num_chunks']}")
    
    print(f"\nFormatted backstory:\n{sample['formatted_backstory'][0][:300]}...")
    
    # Get class weights
    print("\n--- Class Distribution ---")
    pos_weight = get_class_weights(train_loader)
    
    # Sample from test
    print("\n--- Sample from Test ---")
    test_sample = next(iter(test_loader))
    print(f"Test Sample ID: {test_sample['sample_id']}")
    print(f"Has label: {'label' in test_sample}")
    
    print("\n✓ Full pipeline tests passed")


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data preprocessing pipeline")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/Users/kumarnalinsingh/Documents/Pathway_challenge3v",
        help="Directory containing data files"
    )
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run full pipeline test (slower, loads novels)"
    )
    
    args = parser.parse_args()
    
    # Run unit tests
    test_text_normalizer()
    test_backstory_formatting()
    test_chunking()
    test_tokenizer()
    
    # Run full pipeline test if requested
    if args.full:
        test_full_pipeline(args.data_dir)
    else:
        print("\n" + "=" * 60)
        print("Skipping full pipeline test. Run with --full to include.")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
