"""
Full BDH NLI Pipeline.

Integrates all components:
- BDH Encoder
- Sparse Neuron Retriever
- Verification Cross-Encoder
- MIL Aggregator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .bdh_encoder import BDHEncoder, BDHConfig, StatefulBDHProcessor
from .retriever import SparseNeuronRetriever
from .cross_encoder import VerificationCrossEncoder, LightweightVerifier
from .aggregator import MILAggregator


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    
    # BDH config
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_multiplier: int = 128
    vocab_size: int = 256
    dropout: float = 0.1
    
    # Retrieval
    top_k_retrieval: int = 20
    similarity_type: str = 'cosine'
    
    # Aggregation
    aggregation: str = 'max'
    
    # Processing
    chunk_batch_size: int = 32
    
    def to_bdh_config(self) -> BDHConfig:
        return BDHConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            mlp_internal_dim_multiplier=self.mlp_multiplier,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            use_causal_mask=False  # Bidirectional for encoding
        )


class BDHNLIPipeline(nn.Module):
    """
    Complete pipeline for document-level NLI.
    
    Flow:
    1. Encode all novel chunks, extract neuron traces
    2. Encode backstory, extract neuron trace
    3. Retrieve top-K relevant chunks using sparse neuron overlap
    4. Score each (chunk, backstory) pair with cross-encoder
    5. Aggregate scores with MIL
    6. Return final prediction and evidence
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        
        # Initialize BDH encoder
        bdh_config = config.to_bdh_config()
        self.encoder = BDHEncoder(bdh_config)
        
        # Retriever
        self.retriever = SparseNeuronRetriever(
            top_k=config.top_k_retrieval,
            similarity_type=config.similarity_type
        )
        
        # Verifier (lightweight, uses pre-computed embeddings)
        self.verifier = LightweightVerifier(
            embedding_dim=config.n_embd,
            neuron_dim=bdh_config.total_neurons,
            dropout=config.dropout
        )
        
        # Aggregator
        self.aggregator = MILAggregator(
            aggregation=config.aggregation
        )
        
        # Stateful processor for chunks
        self.chunk_processor = StatefulBDHProcessor(self.encoder)
    
    def encode_chunks(
        self,
        chunk_tokens: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all chunks of a novel.
        
        Args:
            chunk_tokens: (num_chunks, T)
            batch_size: Batch size for processing
            
        Returns:
            Dict with embeddings and neuron traces
        """
        bs = batch_size or self.config.chunk_batch_size
        return self.chunk_processor.process_all_chunks(chunk_tokens, bs)
    
    def encode_backstory(
        self,
        backstory_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a backstory.
        
        Args:
            backstory_tokens: (1, T) or (T,)
            
        Returns:
            Dict with embedding and neuron trace
        """
        if backstory_tokens.dim() == 1:
            backstory_tokens = backstory_tokens.unsqueeze(0)
        
        return self.encoder(
            backstory_tokens,
            return_neuron_trace=True,
            pool_output=True
        )
    
    def forward(
        self,
        chunk_tokens: torch.Tensor,
        backstory_tokens: torch.Tensor,
        chunk_embeddings: Optional[torch.Tensor] = None,
        chunk_traces: Optional[torch.Tensor] = None,
        return_evidence: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for one sample.
        
        Args:
            chunk_tokens: Novel chunks (num_chunks, T)
            backstory_tokens: Backstory (1, T) or (T,)
            chunk_embeddings: Pre-computed chunk embeddings (optional)
            chunk_traces: Pre-computed chunk neuron traces (optional)
            return_evidence: Whether to return evidence information
            
        Returns:
            Dict with:
            - 'prediction': Consistency probability (scalar)
            - 'chunk_scores': Scores for retrieved chunks
            - 'retrieved_indices': Indices of retrieved chunks
            - 'best_chunk_idx': Index of best evidence chunk
            - 'evidence_neurons': Neuron trace of best chunk
        """
        device = backstory_tokens.device
        
        # Encode chunks if not pre-computed
        if chunk_embeddings is None or chunk_traces is None:
            with torch.no_grad():
                chunk_result = self.encode_chunks(chunk_tokens)
                chunk_embeddings = chunk_result['embeddings']
                chunk_traces = chunk_result['neuron_traces']
        
        # Encode backstory
        if backstory_tokens.dim() == 1:
            backstory_tokens = backstory_tokens.unsqueeze(0)
        
        back_result = self.encoder(
            backstory_tokens,
            return_neuron_trace=True,
            pool_output=True
        )
        back_embedding = back_result['embedding'].squeeze(0)  # (D,)
        back_trace = back_result['neuron_trace'].squeeze(0)   # (H*N,)
        
        # Retrieve top-K chunks
        retrieved_indices, retrieval_scores = self.retriever(
            back_trace,
            chunk_traces
        )
        
        # Score retrieved chunks
        chunk_scores = []
        for idx in retrieved_indices:
            score = self.verifier(
                chunk_embeddings[idx],
                back_embedding,
                chunk_traces[idx],
                back_trace
            )
            chunk_scores.append(score)
        
        chunk_scores = torch.stack(chunk_scores).squeeze(-1)  # (K,)
        
        # Aggregate
        agg_result = self.aggregator(
            chunk_scores,
            chunk_traces[retrieved_indices],
            back_trace
        )
        
        result = {
            'prediction': agg_result['final_score'],
            'chunk_scores': chunk_scores,
            'retrieved_indices': retrieved_indices,
            'retrieval_scores': retrieval_scores
        }
        
        if return_evidence:
            # Map best chunk back to original index
            best_local_idx = agg_result['best_chunk_idx']
            best_global_idx = retrieved_indices[best_local_idx]
            result['best_chunk_idx'] = best_global_idx
            result['evidence_neurons'] = agg_result.get('evidence_neurons')
            result['aggregation_weights'] = agg_result['weights']
        
        return result
    
    def predict(
        self,
        chunk_tokens: torch.Tensor,
        backstory_tokens: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[str, float, int]:
        """
        Make a prediction with label.
        
        Args:
            chunk_tokens: Novel chunks (num_chunks, T)
            backstory_tokens: Backstory (1, T) or (T,)
            threshold: Decision threshold
            
        Returns:
            Tuple of (label, confidence, best_chunk_idx)
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(chunk_tokens, backstory_tokens)
        
        prob = result['prediction'].item()
        label = 'consistent' if prob >= threshold else 'contradict'
        
        return label, prob, result['best_chunk_idx'].item()


class BDHNLIWithPrecompute(nn.Module):
    """
    Pipeline variant that precomputes chunk encodings.
    
    More efficient when processing multiple samples from same novel.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.pipeline = BDHNLIPipeline(config)
        
        # Cache for pre-computed chunk encodings
        self._chunk_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def precompute_chunks(
        self,
        book_name: str,
        chunk_tokens: torch.Tensor
    ):
        """
        Precompute and cache chunk encodings.
        
        Args:
            book_name: Name of the book (cache key)
            chunk_tokens: (num_chunks, T)
        """
        with torch.no_grad():
            result = self.pipeline.encode_chunks(chunk_tokens)
            self._chunk_cache[book_name] = {
                'tokens': chunk_tokens,
                'embeddings': result['embeddings'],
                'traces': result['neuron_traces']
            }
    
    def forward(
        self,
        book_name: str,
        backstory_tokens: torch.Tensor,
        return_evidence: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using cached chunks.
        
        Args:
            book_name: Name of the book (to lookup cache)
            backstory_tokens: (1, T) or (T,)
            return_evidence: Whether to return evidence
            
        Returns:
            Prediction result dict
        """
        if book_name not in self._chunk_cache:
            raise ValueError(f"Book '{book_name}' not precomputed. Call precompute_chunks first.")
        
        cache = self._chunk_cache[book_name]
        
        return self.pipeline(
            chunk_tokens=cache['tokens'],
            backstory_tokens=backstory_tokens,
            chunk_embeddings=cache['embeddings'],
            chunk_traces=cache['traces'],
            return_evidence=return_evidence
        )
    
    def clear_cache(self):
        """Clear the chunk cache."""
        self._chunk_cache.clear()
