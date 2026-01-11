"""
Cross-Encoder for Verification in BDH NLI Pipeline.

Scores (chunk, backstory) pairs for consistency/contradiction.
Uses BDH encoder with concatenated inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .bdh_encoder import BDHEncoder, BDHConfig


class VerificationCrossEncoder(nn.Module):
    """
    Cross-encoder that scores chunk-backstory pairs.
    
    Takes a chunk and backstory, encodes both with shared BDH encoder,
    and predicts consistency probability.
    """
    
    def __init__(
        self,
        encoder: BDHEncoder,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            encoder: Pre-initialized BDH encoder (shared)
            hidden_dim: Hidden dimension for classifier (default: n_embd)
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = encoder
        
        D = encoder.config.n_embd
        hidden = hidden_dim or D
        
        # Classifier head
        # Input: concatenation of chunk and backstory embeddings
        self.classifier = nn.Sequential(
            nn.Linear(D * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        # Optional: learned interaction between neuron traces
        self.neuron_interaction = nn.Sequential(
            nn.Linear(encoder.config.total_neurons * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        # Combination weight
        self.alpha = nn.Parameter(torch.tensor(0.8))
    
    def forward(
        self,
        chunk_ids: torch.Tensor,
        backstory_ids: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        backstory_mask: Optional[torch.Tensor] = None,
        use_neuron_interaction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Score a chunk-backstory pair.
        
        Args:
            chunk_ids: Chunk token IDs (B, T_chunk)
            backstory_ids: Backstory token IDs (B, T_back)
            chunk_mask: Optional attention mask for chunk
            backstory_mask: Optional attention mask for backstory
            use_neuron_interaction: Whether to use neuron trace interaction
            
        Returns:
            Dict with:
            - 'score': Consistency probability (B, 1)
            - 'chunk_embedding': Chunk embedding (B, D)
            - 'backstory_embedding': Backstory embedding (B, D)
            - 'chunk_neurons': Chunk neuron trace (B, H*N)
            - 'backstory_neurons': Backstory neuron trace (B, H*N)
        """
        # Encode chunk
        chunk_out = self.encoder(
            chunk_ids,
            attention_mask=chunk_mask,
            return_neuron_trace=True,
            pool_output=True
        )
        
        # Encode backstory
        back_out = self.encoder(
            backstory_ids,
            attention_mask=backstory_mask,
            return_neuron_trace=True,
            pool_output=True
        )
        
        chunk_emb = chunk_out['embedding']
        back_emb = back_out['embedding']
        chunk_neurons = chunk_out['neuron_trace']
        back_neurons = back_out['neuron_trace']
        
        # Concatenate embeddings and classify
        combined = torch.cat([chunk_emb, back_emb], dim=-1)
        emb_score = self.classifier(combined)  # (B, 1)
        
        if use_neuron_interaction:
            # Also use neuron trace interaction
            neuron_combined = torch.cat([chunk_neurons, back_neurons], dim=-1)
            neuron_score = self.neuron_interaction(neuron_combined)  # (B, 1)
            
            # Weighted combination
            alpha = torch.sigmoid(self.alpha)
            score = alpha * emb_score + (1 - alpha) * neuron_score
        else:
            score = emb_score
        
        # Sigmoid for probability
        score = torch.sigmoid(score)
        
        return {
            'score': score,
            'chunk_embedding': chunk_emb,
            'backstory_embedding': back_emb,
            'chunk_neurons': chunk_neurons,
            'backstory_neurons': back_neurons
        }


class LightweightVerifier(nn.Module):
    """
    Lightweight verifier using pre-computed embeddings.
    
    Faster than cross-encoder as it doesn't re-encode chunks.
    """
    
    def __init__(self, embedding_dim: int, neuron_dim: int, dropout: float = 0.1):
        """
        Args:
            embedding_dim: Dimension of embeddings (D)
            neuron_dim: Dimension of neuron traces (H*N)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Embedding-based scorer
        self.emb_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        # Neuron-based scorer
        self.neuron_scorer = nn.Sequential(
            nn.Linear(neuron_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        # Combination
        self.combiner = nn.Linear(2, 1)
    
    def forward(
        self,
        chunk_embedding: torch.Tensor,
        backstory_embedding: torch.Tensor,
        chunk_neurons: torch.Tensor,
        backstory_neurons: torch.Tensor
    ) -> torch.Tensor:
        """
        Score pre-computed embeddings and neuron traces.
        
        Args:
            chunk_embedding: (B, D) or (D,)
            backstory_embedding: (B, D) or (D,)
            chunk_neurons: (B, H*N) or (H*N,)
            backstory_neurons: (B, H*N) or (H*N,)
            
        Returns:
            Consistency score (B, 1) or (1,)
        """
        # Handle single sample
        squeeze_output = False
        if chunk_embedding.dim() == 1:
            chunk_embedding = chunk_embedding.unsqueeze(0)
            backstory_embedding = backstory_embedding.unsqueeze(0)
            chunk_neurons = chunk_neurons.unsqueeze(0)
            backstory_neurons = backstory_neurons.unsqueeze(0)
            squeeze_output = True
        
        # Embedding score
        emb_combined = torch.cat([chunk_embedding, backstory_embedding], dim=-1)
        emb_logit = self.emb_scorer(emb_combined)
        
        # Neuron score
        neuron_combined = torch.cat([chunk_neurons, backstory_neurons], dim=-1)
        neuron_logit = self.neuron_scorer(neuron_combined)
        
        # Combine - output LOGITS (no sigmoid!) for BCEWithLogitsLoss
        combined = torch.cat([emb_logit, neuron_logit], dim=-1)
        logit = self.combiner(combined)  # Raw logit output
        
        if squeeze_output:
            logit = logit.squeeze(0)
        
        return logit
