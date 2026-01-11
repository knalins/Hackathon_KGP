"""
Sparse Neuron Retriever for BDH NLI Pipeline.

Uses BDH's sparse positive neuron activations for retrieval.
Key insight: If the same neurons fire for backstory and chunk,
they likely encode similar concepts (monosemanticity property).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SparseNeuronRetriever(nn.Module):
    """
    Retriever based on sparse neuron overlap.
    
    Exploits BDH's monosemantic property: individual neurons
    activate for specific concepts, so overlap indicates relevance.
    """
    
    def __init__(self, top_k: int = 20, similarity_type: str = 'cosine'):
        """
        Args:
            top_k: Number of chunks to retrieve
            similarity_type: 'cosine', 'dot', or 'overlap'
        """
        super().__init__()
        self.top_k = top_k
        self.similarity_type = similarity_type
    
    def compute_similarity(
        self,
        query_trace: torch.Tensor,
        chunk_traces: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between query and chunk traces.
        
        Args:
            query_trace: Backstory neuron trace (H*N,) or (B, H*N)
            chunk_traces: Chunk neuron traces (num_chunks, H*N)
            
        Returns:
            Similarity scores (num_chunks,) or (B, num_chunks)
        """
        # Handle batch dimension
        if query_trace.dim() == 1:
            query_trace = query_trace.unsqueeze(0)
        
        if self.similarity_type == 'cosine':
            # Cosine similarity - works well for sparse positive vectors
            query_norm = F.normalize(query_trace, p=2, dim=-1)
            chunk_norms = F.normalize(chunk_traces, p=2, dim=-1)
            similarities = query_norm @ chunk_norms.T
            
        elif self.similarity_type == 'dot':
            # Dot product - emphasizes magnitude
            similarities = query_trace @ chunk_traces.T
            
        elif self.similarity_type == 'overlap':
            # Count of co-active neurons (thresholded)
            query_active = (query_trace > 0.1).float()
            chunk_active = (chunk_traces > 0.1).float()
            overlaps = query_active @ chunk_active.T
            # Normalize by query activity
            query_count = query_active.sum(dim=-1, keepdim=True).clamp(min=1)
            similarities = overlaps / query_count
            
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        return similarities.squeeze(0)
    
    def retrieve(
        self,
        query_trace: torch.Tensor,
        chunk_traces: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-K most similar chunks.
        
        Args:
            query_trace: Backstory neuron trace (H*N,)
            chunk_traces: All chunk traces (num_chunks, H*N)
            top_k: Override default top_k
            
        Returns:
            Tuple of (indices, scores) for top-K chunks
        """
        k = top_k or self.top_k
        k = min(k, chunk_traces.size(0))  # Don't exceed available chunks
        
        similarities = self.compute_similarity(query_trace, chunk_traces)
        
        # Get top-K
        scores, indices = similarities.topk(k)
        
        return indices, scores
    
    def forward(
        self,
        query_trace: torch.Tensor,
        chunk_traces: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (same as retrieve)."""
        return self.retrieve(query_trace, chunk_traces, top_k)


class HybridRetriever(nn.Module):
    """
    Hybrid retriever combining sparse neurons with embedding similarity.
    
    Provides fallback when neuron traces are too sparse.
    """
    
    def __init__(
        self,
        top_k: int = 20,
        neuron_weight: float = 0.7,
        embedding_weight: float = 0.3
    ):
        super().__init__()
        self.top_k = top_k
        self.neuron_weight = neuron_weight
        self.embedding_weight = embedding_weight
        
        self.neuron_retriever = SparseNeuronRetriever(
            top_k=top_k * 2,  # Get more candidates
            similarity_type='cosine'
        )
    
    def forward(
        self,
        query_trace: torch.Tensor,
        query_embedding: torch.Tensor,
        chunk_traces: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve using both neuron traces and embeddings.
        
        Args:
            query_trace: Backstory neuron trace (H*N,)
            query_embedding: Backstory embedding (D,)
            chunk_traces: Chunk neuron traces (num_chunks, H*N)
            chunk_embeddings: Chunk embeddings (num_chunks, D)
            top_k: Override default
            
        Returns:
            Tuple of (indices, combined_scores)
        """
        k = top_k or self.top_k
        
        # Neuron similarity
        neuron_sims = self.neuron_retriever.compute_similarity(
            query_trace, chunk_traces
        )
        
        # Embedding similarity
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=-1)
        chunk_norms = F.normalize(chunk_embeddings, p=2, dim=-1)
        embed_sims = (query_norm @ chunk_norms.T).squeeze(0)
        
        # Combine
        combined = (
            self.neuron_weight * neuron_sims + 
            self.embedding_weight * embed_sims
        )
        
        # Get top-K
        k = min(k, combined.size(0))
        scores, indices = combined.topk(k)
        
        return indices, scores
