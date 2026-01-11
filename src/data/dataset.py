"""
PyTorch Dataset and DataLoader for BDH NLI Pipeline

Handles:
- Loading train/test CSVs
- Loading and chunking novels
- Creating properly formatted samples
- Collating batches
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from .preprocessing import TextNormalizer, format_backstory_input, encode_label
from .chunking import NovelChunker, ChunkConfig, chunk_novels
from .tokenizer import ByteTokenizer


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Paths
    train_csv: str = "train - train.csv"
    test_csv: str = "test - test.csv"
    novel_dir: str = "./"
    
    # Novel file names
    novel_files: Dict[str, str] = None
    
    # Tokenization
    max_backstory_tokens: int = 512
    max_chunk_tokens: int = 512
    
    # Training
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.novel_files is None:
            self.novel_files = {
                "In Search of the Castaways": "In search of the castaways.txt",
                "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
            }


class NLIDataset(Dataset):
    """
    PyTorch Dataset for document-level NLI task.
    
    Each sample contains:
    - formatted_backstory: Structured backstory text
    - backstory_tokens: Tokenized backstory
    - novel_chunks: List of tokenized novel chunks
    - chunk_texts: List of chunk texts (for explainability)
    - label: 0 (contradict) or 1 (consistent), or None for test
    - sample_id: Original ID from CSV
    - book_name: Name of the novel
    - character: Character name
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        chunked_novels: Dict[str, List[dict]],
        tokenizer: ByteTokenizer,
        config: DataConfig,
        is_train: bool = True
    ):
        """
        Args:
            df: DataFrame with columns: id, book_name, char, caption, content, [label]
            chunked_novels: Dict mapping book_name -> list of chunk dicts
            tokenizer: ByteTokenizer instance
            config: DataConfig instance
            is_train: Whether this is training data (has labels)
        """
        self.df = df.reset_index(drop=True)
        self.chunked_novels = chunked_novels
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        
        self.normalizer = TextNormalizer()
        
        # Pre-process all samples
        self.samples = self._preprocess_all()
        
        # Pre-tokenize all novel chunks for efficiency
        self.tokenized_chunks = self._tokenize_all_chunks()
    
    def _preprocess_all(self) -> List[dict]:
        """Pre-process all samples from DataFrame."""
        samples = []
        
        for idx, row in self.df.iterrows():
            sample = self._preprocess_row(row)
            if sample is not None:
                samples.append(sample)
        
        return samples
    
    def _preprocess_row(self, row: pd.Series) -> Optional[dict]:
        """Process a single row from the DataFrame."""
        try:
            # Extract fields
            sample_id = row['id']
            book_name = str(row['book_name']).strip()
            character = str(row['char']).strip()
            content = str(row['content']).strip()
            
            # Handle caption (may be NaN)
            caption = row.get('caption', None)
            if pd.isna(caption):
                caption = None
            elif isinstance(caption, str):
                caption = caption.strip() if caption.strip() else None
            
            # Format backstory
            formatted_backstory = format_backstory_input(
                book_name=book_name,
                character=character,
                caption=caption,
                content=content,
                normalizer=self.normalizer
            )
            
            # Tokenize backstory
            backstory_tokens = self.tokenizer.encode(
                formatted_backstory,
                max_length=self.config.max_backstory_tokens,
                padding=True,
                return_tensors='pt'
            )
            
            # Get label if training
            label = None
            if self.is_train and 'label' in row:
                label = encode_label(row['label'])
            
            # Find matching novel
            # Handle case variations in book names
            matched_book = None
            for known_book in self.chunked_novels.keys():
                if book_name.lower() in known_book.lower() or known_book.lower() in book_name.lower():
                    matched_book = known_book
                    break
            
            if matched_book is None:
                print(f"Warning: Could not find novel for '{book_name}'")
                return None
            
            return {
                'sample_id': sample_id,
                'book_name': matched_book,
                'character': character,
                'caption': caption,
                'content': content,
                'formatted_backstory': formatted_backstory,
                'backstory_tokens': backstory_tokens,
                'label': label
            }
            
        except Exception as e:
            print(f"Error processing row: {e}")
            return None
    
    def _tokenize_all_chunks(self) -> Dict[str, torch.Tensor]:
        """Pre-tokenize all novel chunks."""
        tokenized = {}
        
        for book_name, chunks in self.chunked_novels.items():
            chunk_texts = [c['text'] for c in chunks]
            
            # Tokenize all chunks for this book
            tokenized_chunks = self.tokenizer.batch_encode(
                chunk_texts,
                max_length=self.config.max_chunk_tokens,
                padding=True,
                return_tensors='pt'
            )
            
            tokenized[book_name] = {
                'tokens': tokenized_chunks,
                'texts': chunk_texts,
                'metadata': chunks
            }
            
            print(f"Tokenized {len(chunks)} chunks for '{book_name}'")
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Get the novel chunks for this sample
        book_name = sample['book_name']
        chunk_data = self.tokenized_chunks[book_name]
        
        result = {
            'sample_id': sample['sample_id'],
            'book_name': sample['book_name'],
            'character': sample['character'],
            'formatted_backstory': sample['formatted_backstory'],
            'backstory_tokens': sample['backstory_tokens'],
            'chunk_tokens': chunk_data['tokens'],
            'chunk_texts': chunk_data['texts'],
            'num_chunks': len(chunk_data['texts'])
        }
        
        if sample['label'] is not None:
            result['label'] = torch.tensor(sample['label'], dtype=torch.long)
        
        return result


def load_novels(config: DataConfig, normalizer: TextNormalizer) -> Dict[str, str]:
    """Load and normalize novel texts."""
    novels = {}
    
    for book_name, filename in config.novel_files.items():
        filepath = os.path.join(config.novel_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Novel file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Normalize
        text = normalizer.normalize_novel(text)
        novels[book_name] = text
        
        print(f"Loaded '{book_name}': {len(text):,} characters")
    
    return novels


def create_dataloaders(
    data_dir: str,
    config: Optional[DataConfig] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing CSV files and novels
        config: DataConfig instance
        batch_size: Batch size (note: 1 is recommended due to variable chunk counts)
        num_workers: Number of data loading workers
        seed: Random seed for train/val split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = DataConfig(novel_dir=data_dir)
    else:
        config.novel_dir = data_dir
    
    # Initialize components
    normalizer = TextNormalizer()
    tokenizer = ByteTokenizer(max_length=config.max_backstory_tokens)
    chunker = NovelChunker(ChunkConfig())
    
    # Load novels
    print("Loading novels...")
    novels = load_novels(config, normalizer)
    
    # Chunk novels
    print("\nChunking novels...")
    chunked_novels = chunk_novels(novels, chunker)
    
    # Load CSVs
    print("\nLoading CSVs...")
    train_df = pd.read_csv(os.path.join(data_dir, config.train_csv))
    test_df = pd.read_csv(os.path.join(data_dir, config.test_csv))
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Split train into train/val
    val_loader = None
    if config.validation_split > 0:
        torch.manual_seed(seed)
        n_val = int(len(train_df) * config.validation_split)
        indices = torch.randperm(len(train_df)).tolist()
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        val_df = train_df.iloc[val_indices].reset_index(drop=True)
        train_df = train_df.iloc[train_indices].reset_index(drop=True)
        
        print(f"After split - Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Create validation dataset
        val_dataset = NLIDataset(
            df=val_df,
            chunked_novels=chunked_novels,
            tokenizer=tokenizer,
            config=config,
            is_train=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    # Create train dataset
    train_dataset = NLIDataset(
        df=train_df,
        chunked_novels=chunked_novels,
        tokenizer=tokenizer,
        config=config,
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Create test dataset
    test_dataset = NLIDataset(
        df=test_df,
        chunked_novels=chunked_novels,
        tokenizer=tokenizer,
        config=config,
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for variable-length chunk lists.
    
    Note: With batch_size=1, this mostly just unpacks the single sample.
    For batch_size>1, chunk handling becomes complex due to different novels.
    """
    if len(batch) == 1:
        # Single sample - just return it with proper tensor handling
        sample = batch[0]
        result = {
            'sample_id': [sample['sample_id']],
            'book_name': [sample['book_name']],
            'character': [sample['character']],
            'formatted_backstory': [sample['formatted_backstory']],
            'backstory_tokens': sample['backstory_tokens'].unsqueeze(0) if sample['backstory_tokens'].dim() == 1 else sample['backstory_tokens'],
            'chunk_tokens': sample['chunk_tokens'],
            'chunk_texts': sample['chunk_texts'],
            'num_chunks': [sample['num_chunks']]
        }
        
        if 'label' in sample:
            result['label'] = sample['label'].unsqueeze(0)
        
        return result
    
    # Multiple samples - group by book to share chunks
    result = {
        'sample_id': [s['sample_id'] for s in batch],
        'book_name': [s['book_name'] for s in batch],
        'character': [s['character'] for s in batch],
        'formatted_backstory': [s['formatted_backstory'] for s in batch],
        'backstory_tokens': torch.stack([s['backstory_tokens'].squeeze(0) for s in batch]),
        'chunk_tokens': {},  # Group by book
        'chunk_texts': {},
        'num_chunks': [s['num_chunks'] for s in batch]
    }
    
    if 'label' in batch[0]:
        result['label'] = torch.stack([s['label'] for s in batch])
    
    # Group chunks by book (avoid duplication)
    for sample in batch:
        book = sample['book_name']
        if book not in result['chunk_tokens']:
            result['chunk_tokens'][book] = sample['chunk_tokens']
            result['chunk_texts'][book] = sample['chunk_texts']
    
    return result


def get_class_weights(train_loader: DataLoader) -> float:
    """
    Calculate class weights for imbalanced data.
    
    Returns pos_weight for BCE loss (weight for positive class).
    """
    n_positive = 0
    n_negative = 0
    
    for batch in train_loader:
        if 'label' in batch:
            labels = batch['label']
            n_positive += (labels == 1).sum().item()
            n_negative += (labels == 0).sum().item()
    
    if n_positive == 0 or n_negative == 0:
        return 1.0
    
    # pos_weight = n_negative / n_positive
    pos_weight = n_negative / n_positive
    
    print(f"Class distribution - Positive: {n_positive}, Negative: {n_negative}")
    print(f"Calculated pos_weight: {pos_weight:.2f}")
    
    return pos_weight
