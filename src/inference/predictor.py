"""
Predictor for BDH NLI Pipeline.

Handles inference on test data and generates results.csv.
"""

import os
import csv
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.pipeline import BDHNLIPipeline, BDHNLIWithPrecompute, PipelineConfig
from .explainability import ExplanationExtractor


class Predictor:
    """
    Predictor for making inference on test data.
    
    Handles:
    - Loading trained model
    - Making predictions
    - Extracting explanations
    - Generating results.csv
    """
    
    def __init__(
        self,
        model: BDHNLIPipeline,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        threshold: float = 0.5
    ):
        """
        Args:
            model: Trained BDH NLI pipeline
            device: Device for inference
            threshold: Decision threshold for classification
        """
        self.model = model
        self.device = torch.device(device)
        self.threshold = threshold
        
        self.model.to(self.device)
        self.model.eval()
        
        self.explainer = ExplanationExtractor()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'Predictor':
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device for inference
            
        Returns:
            Predictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct config
        pipeline_config = PipelineConfig(**checkpoint['pipeline_config'])
        
        # Create model
        model = BDHNLIPipeline(pipeline_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device)
    
    @torch.no_grad()
    def predict_single(
        self,
        chunk_tokens: torch.Tensor,
        backstory_tokens: torch.Tensor,
        chunk_texts: Optional[List[str]] = None
    ) -> Dict:
        """
        Make prediction for a single sample.
        
        Args:
            chunk_tokens: Novel chunks (num_chunks, T)
            backstory_tokens: Backstory (1, T) or (T,)
            chunk_texts: Optional chunk texts for explanation
            
        Returns:
            Dict with prediction, confidence, and explanation
        """
        chunk_tokens = chunk_tokens.to(self.device)
        backstory_tokens = backstory_tokens.to(self.device)
        
        result = self.model(
            chunk_tokens=chunk_tokens,
            backstory_tokens=backstory_tokens,
            return_evidence=True
        )
        
        prob = result['prediction'].item()
        label = 'consistent' if prob >= self.threshold else 'contradict'
        
        output = {
            'prediction': label,
            'confidence': prob,
            'best_chunk_idx': result['best_chunk_idx'].item(),
            'retrieval_scores': result['retrieval_scores'].cpu().tolist(),
            'retrieved_indices': result['retrieved_indices'].cpu().tolist()
        }
        
        # Add explanation if chunk texts provided
        if chunk_texts is not None:
            explanation = self.explainer.extract(
                result,
                chunk_texts
            )
            output['explanation'] = explanation
        
        return output
    
    @torch.no_grad()
    def predict_batch(
        self,
        test_loader: DataLoader,
        chunk_texts_by_book: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict]:
        """
        Make predictions for a batch of samples.
        
        Args:
            test_loader: DataLoader for test data
            chunk_texts_by_book: Optional dict mapping book names to chunk texts
            
        Returns:
            List of prediction dicts
        """
        results = []
        
        for batch in tqdm(test_loader, desc="Predicting"):
            sample_ids = batch['sample_id']
            book_names = batch['book_name']
            
            backstory_tokens = batch['backstory_tokens'].to(self.device)
            chunk_tokens = batch['chunk_tokens'].to(self.device)
            
            if backstory_tokens.dim() == 3:
                backstory_tokens = backstory_tokens.squeeze(0)
            if chunk_tokens.dim() == 3:
                chunk_tokens = chunk_tokens.squeeze(0)
            
            # Get chunk texts if available
            chunk_texts = None
            if chunk_texts_by_book is not None:
                book = book_names[0] if isinstance(book_names, list) else book_names
                chunk_texts = chunk_texts_by_book.get(book)
            
            # Also get from batch if available
            if 'chunk_texts' in batch:
                chunk_texts = batch['chunk_texts']
            
            result = self.predict_single(
                chunk_tokens,
                backstory_tokens,
                chunk_texts
            )
            
            # Add sample metadata
            result['id'] = sample_ids[0] if isinstance(sample_ids, list) else sample_ids
            result['book_name'] = book_names[0] if isinstance(book_names, list) else book_names
            
            results.append(result)
        
        return results
    
    def generate_results_csv(
        self,
        results: List[Dict],
        output_path: str,
        include_explanation: bool = True
    ):
        """
        Generate results.csv from predictions.
        
        Args:
            results: List of prediction dicts
            output_path: Path to output CSV
            include_explanation: Whether to include explanation column
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine columns
        columns = ['id', 'prediction', 'confidence']
        if include_explanation:
            columns.extend(['evidence_chunk', 'reasoning'])
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in results:
                row = {
                    'id': result['id'],
                    'prediction': result['prediction'],
                    'confidence': f"{result['confidence']:.4f}"
                }
                
                if include_explanation and 'explanation' in result:
                    exp = result['explanation']
                    row['evidence_chunk'] = result['best_chunk_idx']
                    row['reasoning'] = exp.get('evidence_text', '')[:500]  # Truncate
                
                writer.writerow(row)
        
        print(f"Results saved to {output_path}")
    
    def run_inference(
        self,
        test_loader: DataLoader,
        output_path: str = 'results.csv',
        chunk_texts_by_book: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict]:
        """
        Run full inference pipeline.
        
        Args:
            test_loader: DataLoader for test data
            output_path: Path to output results CSV
            chunk_texts_by_book: Optional chunk texts for explanations
            
        Returns:
            List of prediction results
        """
        print(f"Running inference on {len(test_loader)} samples...")
        
        results = self.predict_batch(test_loader, chunk_texts_by_book)
        
        self.generate_results_csv(results, output_path)
        
        # Print summary
        predictions = [r['prediction'] for r in results]
        consistent_count = sum(1 for p in predictions if p == 'consistent')
        contradict_count = len(predictions) - consistent_count
        
        print(f"\nPrediction Summary:")
        print(f"  Consistent: {consistent_count} ({consistent_count/len(predictions)*100:.1f}%)")
        print(f"  Contradict: {contradict_count} ({contradict_count/len(predictions)*100:.1f}%)")
        
        return results
