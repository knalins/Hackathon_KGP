"""
Multi-Instance Learning Aggregator for BDH NLI Pipeline.

Aggregates chunk-level scores to document-level predictions.
Handles the weak supervision problem where we only have global labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal


class MILAggregator(nn.Module):
    """
    Multi-Instance Learning aggregator.
    
    Aggregates multiple chunk scores into a single prediction.
    Supports various aggregation strategies for different assumptions.
    """
    
    def __init__(
        self,
        aggregation: Literal['max', 'mean', 'attention', 'noisy_or', 'top_k'] = 'max',
        top_k: int = 3,
        temperature: float = 1.0
    ):
        """
        Args:
            aggregation: Aggregation strategy
                - 'max': Take maximum score (assumes one chunk is sufficient)
                - 'mean': Average all scores
                - 'attention': Learned attention-weighted average
                - 'noisy_or': P(at least one) = 1 - Π(1 - p_i)
                - 'top_k': Average top-K scores
            top_k: K for top_k aggregation
            temperature: Temperature for attention weights
        """
        super().__init__()
        self.aggregation = aggregation
        self.top_k = top_k
        self.temperature = temperature
    
    def forward(
        self,
        chunk_scores: torch.Tensor,
        chunk_neurons: Optional[torch.Tensor] = None,
        backstory_neurons: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate chunk scores.
        
        Args:
            chunk_scores: Scores for each chunk (num_chunks,) or (B, num_chunks)
            chunk_neurons: Optional neuron traces (num_chunks, H*N)
            backstory_neurons: Optional backstory trace (H*N,)
            mask: Optional mask for valid chunks (num_chunks,)
            
        Returns:
            Dict with:
            - 'final_score': Aggregated score
            - 'best_chunk_idx': Index of highest-scoring chunk
            - 'weights': Aggregation weights if applicable
            - 'evidence_neurons': Neurons from best chunk if provided
        """
        # Handle batch dimension
        batched = chunk_scores.dim() > 1
        if not batched:
            chunk_scores = chunk_scores.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        B, num_chunks = chunk_scores.size()
        
        # Apply mask
        if mask is not None:
            # Set masked positions to very low score
            chunk_scores = chunk_scores.masked_fill(~mask.bool(), -1e9)
        
        if self.aggregation == 'max':
            final_score, best_idx = chunk_scores.max(dim=-1)
            weights = F.one_hot(best_idx, num_chunks).float()
            
        elif self.aggregation == 'mean':
            if mask is not None:
                lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                chunk_scores_masked = chunk_scores.masked_fill(~mask.bool(), 0)
                final_score = chunk_scores_masked.sum(dim=-1) / lengths.squeeze(-1)
            else:
                final_score = chunk_scores.mean(dim=-1)
            best_idx = chunk_scores.argmax(dim=-1)
            weights = torch.ones_like(chunk_scores) / num_chunks
            
        elif self.aggregation == 'attention':
            # Attention weights based on scores
            attn_logits = chunk_scores / self.temperature
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask.bool(), -1e9)
            weights = F.softmax(attn_logits, dim=-1)
            final_score = (weights * chunk_scores).sum(dim=-1)
            best_idx = chunk_scores.argmax(dim=-1)
            
        elif self.aggregation == 'noisy_or':
            # Noisy-OR: P(at least one) = 1 - Π(1 - p_i)
            # Clamp to [eps, 1-eps] for numerical stability
            scores_clamped = chunk_scores.clamp(1e-6, 1 - 1e-6)
            if mask is not None:
                # Masked positions contribute 0 to the product (1 - 0 = 1)
                scores_clamped = scores_clamped.masked_fill(~mask.bool(), 0)
            log_complement = torch.log(1 - scores_clamped)
            final_score = 1 - torch.exp(log_complement.sum(dim=-1))
            best_idx = chunk_scores.argmax(dim=-1)
            weights = scores_clamped / scores_clamped.sum(dim=-1, keepdim=True)
            
        elif self.aggregation == 'top_k':
            k = min(self.top_k, num_chunks)
            top_scores, top_indices = chunk_scores.topk(k, dim=-1)
            final_score = top_scores.mean(dim=-1)
            best_idx = top_indices[:, 0]  # Best is first of top-K
            weights = torch.zeros_like(chunk_scores)
            weights.scatter_(-1, top_indices, 1.0 / k)
            
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        result = {
            'final_score': final_score.squeeze(0) if not batched else final_score,
            'best_chunk_idx': best_idx.squeeze(0) if not batched else best_idx,
            'weights': weights.squeeze(0) if not batched else weights
        }
        
        # Extract evidence neurons
        if chunk_neurons is not None:
            if not batched:
                result['evidence_neurons'] = chunk_neurons[best_idx.item()]
            else:
                # Batch: gather neurons for each sample's best idx
                result['evidence_neurons'] = torch.gather(
                    chunk_neurons, 
                    0, 
                    best_idx.unsqueeze(-1).expand(-1, chunk_neurons.size(-1))
                )
        
        return result


class LearnedMILAggregator(nn.Module):
    """
    MIL aggregator with learned attention mechanism.
    
    Uses neuron traces to learn which chunks are most relevant.
    """
    
    def __init__(
        self,
        neuron_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            neuron_dim: Dimension of neuron traces (H*N)
            hidden_dim: Hidden dimension for attention
            dropout: Dropout rate
        """
        super().__init__()
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(neuron_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        chunk_scores: torch.Tensor,
        chunk_neurons: torch.Tensor,
        backstory_neurons: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with learned attention.
        
        Args:
            chunk_scores: (num_chunks,)
            chunk_neurons: (num_chunks, H*N)
            backstory_neurons: (H*N,)
            mask: Optional (num_chunks,)
            
        Returns:
            Dict with final_score, best_chunk_idx, attention_weights
        """
        num_chunks = chunk_scores.size(0)
        
        # Compute attention based on chunk-backstory neuron interaction
        backstory_expanded = backstory_neurons.unsqueeze(0).expand(num_chunks, -1)
        combined = torch.cat([chunk_neurons, backstory_expanded], dim=-1)
        
        attn_logits = self.attention(combined).squeeze(-1)  # (num_chunks,)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.bool(), -1e9)
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Weighted sum of scores
        final_score = (attn_weights * chunk_scores).sum()
        
        # Best chunk
        weighted_scores = attn_weights * chunk_scores
        best_idx = weighted_scores.argmax()
        
        return {
            'final_score': final_score,
            'best_chunk_idx': best_idx,
            'attention_weights': attn_weights,
            'evidence_neurons': chunk_neurons[best_idx]
        }
