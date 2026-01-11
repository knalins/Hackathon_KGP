"""
Explainability Module for BDH NLI Pipeline.

Provides faithful explanations (not generative):
- Returns the evidence chunk text
- Shows top activated neurons
- Highlights keyword overlaps
"""

import torch
from typing import Dict, List, Optional, Tuple


class ExplanationExtractor:
    """
    Extracts faithful explanations from model predictions.
    
    Key principle: Never generate explanations.
    Only return actual evidence from the model's decision.
    """
    
    def __init__(self, top_k_neurons: int = 10):
        """
        Args:
            top_k_neurons: Number of top neurons to include
        """
        self.top_k_neurons = top_k_neurons
    
    def extract(
        self,
        result: Dict[str, torch.Tensor],
        chunk_texts: List[str]
    ) -> Dict:
        """
        Extract explanation from prediction result.
        
        Args:
            result: Model output dict containing:
                - best_chunk_idx
                - evidence_neurons (optional)
                - chunk_scores
                - retrieved_indices
            chunk_texts: List of chunk text strings
            
        Returns:
            Explanation dict with:
                - evidence_text: Text of best evidence chunk
                - top_neurons: IDs of most active neurons
                - confidence_breakdown: Score components
        """
        best_idx = result['best_chunk_idx']
        if isinstance(best_idx, torch.Tensor):
            best_idx = best_idx.item()
        
        # Get evidence text
        evidence_text = chunk_texts[best_idx] if best_idx < len(chunk_texts) else ""
        
        explanation = {
            'evidence_text': evidence_text,
            'evidence_chunk_idx': best_idx,
        }
        
        # Top activated neurons
        if 'evidence_neurons' in result and result['evidence_neurons'] is not None:
            neurons = result['evidence_neurons']
            if isinstance(neurons, torch.Tensor):
                neurons = neurons.cpu()
                top_values, top_indices = neurons.topk(
                    min(self.top_k_neurons, neurons.numel())
                )
                explanation['top_neurons'] = top_indices.tolist()
                explanation['neuron_activations'] = top_values.tolist()
        
        # Retrieval scores
        if 'retrieval_scores' in result:
            scores = result['retrieval_scores']
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
            explanation['retrieval_scores'] = scores
        
        # Chunk scores for retrieved chunks
        if 'chunk_scores' in result:
            scores = result['chunk_scores']
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
            explanation['chunk_scores'] = scores
        
        return explanation
    
    def format_for_display(
        self,
        explanation: Dict,
        max_text_length: int = 300
    ) -> str:
        """
        Format explanation for human-readable display.
        
        Args:
            explanation: Explanation dict from extract()
            max_text_length: Maximum characters for evidence text
            
        Returns:
            Formatted string
        """
        lines = []
        
        # Evidence text
        text = explanation.get('evidence_text', '')
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        lines.append(f"Evidence (chunk {explanation.get('evidence_chunk_idx', '?')}):")
        lines.append(f"  \"{text}\"")
        
        # Top neurons
        if 'top_neurons' in explanation:
            neurons = explanation['top_neurons'][:5]
            activations = explanation.get('neuron_activations', [])[:5]
            neuron_str = ", ".join(
                f"N{n}({a:.2f})" if i < len(activations) else f"N{n}"
                for i, (n, a) in enumerate(zip(neurons, activations + [None] * len(neurons)))
            )
            lines.append(f"Top neurons: {neuron_str}")
        
        return "\n".join(lines)
    
    def compare_neurons(
        self,
        backstory_neurons: torch.Tensor,
        chunk_neurons: torch.Tensor,
        top_k: int = 10
    ) -> Dict:
        """
        Compare neuron activations between backstory and chunk.
        
        Args:
            backstory_neurons: (H*N,)
            chunk_neurons: (H*N,)
            top_k: Number of top overlapping neurons
            
        Returns:
            Dict with overlapping neuron information
        """
        # Find neurons active in both
        backstory_active = backstory_neurons > 0.1
        chunk_active = chunk_neurons > 0.1
        both_active = backstory_active & chunk_active
        
        # Get indices of co-active neurons
        overlap_indices = both_active.nonzero().squeeze(-1)
        
        if overlap_indices.numel() == 0:
            return {'overlapping_neurons': [], 'overlap_score': 0.0}
        
        # Get activation values for overlapping neurons
        backstory_vals = backstory_neurons[overlap_indices]
        chunk_vals = chunk_neurons[overlap_indices]
        
        # Score by product of activations
        overlap_scores = backstory_vals * chunk_vals
        
        # Get top-K
        k = min(top_k, overlap_indices.numel())
        top_scores, top_local_idx = overlap_scores.topk(k)
        top_global_idx = overlap_indices[top_local_idx]
        
        return {
            'overlapping_neurons': top_global_idx.tolist(),
            'overlap_activations': top_scores.tolist(),
            'overlap_score': both_active.float().mean().item(),
            'num_overlapping': both_active.sum().item()
        }


def extract_keywords(
    text: str,
    min_length: int = 4,
    max_keywords: int = 10
) -> List[str]:
    """
    Extract potential keywords from text.
    
    Simple extraction based on capitalization and length.
    
    Args:
        text: Input text
        min_length: Minimum word length
        max_keywords: Maximum keywords to return
        
    Returns:
        List of keyword strings
    """
    import re
    
    # Find capitalized words (likely proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    
    # Find long words (likely meaningful)
    words = re.findall(r'\b\w{6,}\b', text.lower())
    
    # Combine and deduplicate
    keywords = list(dict.fromkeys(proper_nouns[:max_keywords//2] + words[:max_keywords//2]))
    
    return keywords[:max_keywords]


def highlight_overlap(
    chunk_text: str,
    backstory_text: str,
    max_highlights: int = 5
) -> List[str]:
    """
    Find overlapping phrases between chunk and backstory.
    
    Args:
        chunk_text: Text from evidence chunk
        backstory_text: Text from backstory
        max_highlights: Maximum overlaps to return
        
    Returns:
        List of overlapping phrases
    """
    chunk_keywords = set(extract_keywords(chunk_text, max_keywords=20))
    backstory_keywords = set(extract_keywords(backstory_text, max_keywords=20))
    
    overlaps = list(chunk_keywords & backstory_keywords)
    
    return overlaps[:max_highlights]
