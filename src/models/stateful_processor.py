"""
Stateful BDH Processor for Long Document Processing.

Implements BDH's key advantage: stateful processing that allows
processing very long sequences incrementally without chunking.

Key insight from BDH paper:
- Linear attention allows cumulative state: State_new = State_old + new_KV
- This means we can process entire novel in chunks while maintaining full context
- Query (backstory) can attend to accumulated state representing entire novel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .bdh_encoder import BDHEncoder, BDHConfig, get_freqs, Attention


@dataclass
class StatefulState:
    """Accumulated state from processing document chunks."""
    # Cumulative key-value products (B, H, N, D) - for linear attention
    kv_sum: torch.Tensor
    # Cumulative key sums (B, H, N) - for normalization
    k_sum: torch.Tensor
    # Accumulated neuron activations (B, H*N) - for retrieval
    neuron_accumulator: torch.Tensor
    # Number of tokens processed
    num_tokens: int
    
    def to(self, device):
        """Move state to device."""
        return StatefulState(
            kv_sum=self.kv_sum.to(device),
            k_sum=self.k_sum.to(device),
            neuron_accumulator=self.neuron_accumulator.to(device),
            num_tokens=self.num_tokens
        )


class StatefulBDHProcessor(nn.Module):
    """
    Stateful processor that accumulates information across chunks.
    
    This is BDH's KEY ADVANTAGE over transformers:
    - Process novel in 512-token chunks
    - Accumulate state incrementally
    - Query accumulated state with backstory
    - No information loss from chunking
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, D)
        
        # BDH core parameters
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        
        # RoPE frequencies
        self.freqs = nn.Buffer(
            get_freqs(N, theta=config.theta, dtype=torch.float32).view(1, 1, 1, N)
        )
        
        # Normalization
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        
        # Cross-attention for query
        self.query_proj = nn.Linear(D, D)
        
    def init_state(self, batch_size: int, device: torch.device) -> StatefulState:
        """Initialize empty state for accumulation."""
        nh = self.config.n_head
        D = self.config.n_embd
        N = self.config.mlp_internal_dim_multiplier * D // nh
        
        return StatefulState(
            kv_sum=torch.zeros(batch_size, nh, N, D, device=device),
            k_sum=torch.zeros(batch_size, nh, N, device=device),
            neuron_accumulator=torch.zeros(batch_size, nh * N, device=device),
            num_tokens=0
        )
    
    def process_chunk(
        self,
        input_ids: torch.Tensor,
        state: StatefulState,
        position_offset: int = 0
    ) -> Tuple[StatefulState, torch.Tensor]:
        """
        Process a chunk and update accumulated state.
        
        This is the KEY STATEFUL OPERATION:
        - Process chunk with BDH
        - Accumulate key-value products into state
        - Return updated state
        
        Args:
            input_ids: Token IDs for chunk (B, T)
            state: Current accumulated state
            position_offset: Global position for RoPE
            
        Returns:
            Updated state and chunk embeddings
        """
        C = self.config
        B = input_ids.size(0)
        T = input_ids.size(1)
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        # Embed
        x = self.embed(input_ids)  # (B, T, D)
        x = x.unsqueeze(1)  # (B, 1, T, D)
        
        chunk_neurons = []
        
        for level in range(C.n_layer):
            # Project to neuron space
            x_latent = x @ self.encoder  # (B, nh, T, N)
            
            # Sparse activation
            x_sparse = F.relu(x_latent)  # (B, nh, T, N)
            
            # Compute positions with offset
            positions = torch.arange(
                position_offset, position_offset + T,
                device=x.device, dtype=torch.float32
            ).view(1, 1, T, 1)
            r_phases = positions * self.freqs
            
            # Apply RoPE
            phases = (r_phases % 1) * (2 * 3.14159)
            x_sparse_rot = x_sparse * torch.cos(phases) + \
                          torch.stack((-x_sparse[..., 1::2], x_sparse[..., ::2]), dim=-1).view_as(x_sparse) * torch.sin(phases)
            
            # Accumulate into state: KV products
            # key = x_sparse_rot: (B, nh, T, N)
            # value = x: (B, 1, T, D)
            
            # Sum over time dimension for state accumulation
            k_sum_new = x_sparse_rot.sum(dim=2)  # (B, nh, N)
            kv_sum_new = torch.einsum('bhtn,bktd->bhnd', x_sparse_rot, x)  # (B, nh, N, D)
            
            # Update state
            state.k_sum = state.k_sum + k_sum_new
            state.kv_sum = state.kv_sum + kv_sum_new
            
            # Compute attention output using full state
            # Denominator for normalization
            denom = state.k_sum.unsqueeze(-1).clamp(min=1e-6)  # (B, nh, N, 1)
            # Attention output: weighted average of values
            attn_out = state.kv_sum / denom  # (B, nh, N, D)
            
            # For current chunk, compute local attention contribution
            local_attn = x_sparse_rot @ attn_out  # (B, nh, T, D)
            yKV = local_attn.mean(dim=1, keepdim=True)  # (B, 1, T, D)
            yKV = self.ln(yKV)
            
            # Value projection and gating
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            
            chunk_neurons.append(xy_sparse.mean(dim=2))  # Avg over time
            
            # Dropout and decode
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            
            # Residual
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        # Update neuron accumulator (running average)
        chunk_neuron_trace = torch.cat([n.reshape(B, -1) for n in chunk_neurons], dim=-1)
        chunk_neuron_trace = chunk_neuron_trace[:, :state.neuron_accumulator.size(-1)]  # Match size
        
        alpha = T / (state.num_tokens + T)  # Weight by token count
        state.neuron_accumulator = (1 - alpha) * state.neuron_accumulator + alpha * chunk_neuron_trace.mean(dim=-1, keepdim=True).expand_as(state.neuron_accumulator)
        state.num_tokens += T
        
        # Pool chunk embedding
        chunk_embedding = x.squeeze(1).mean(dim=1)  # (B, D)
        
        return state, chunk_embedding
    
    def query_state(
        self,
        query_ids: torch.Tensor,
        state: StatefulState
    ) -> Dict[str, torch.Tensor]:
        """
        Query accumulated state with backstory.
        
        The backstory attends to the accumulated novel representation.
        
        Args:
            query_ids: Backstory token IDs (B, T)
            state: Accumulated state from novel processing
            
        Returns:
            Dict with embedding and attention info
        """
        C = self.config
        B = query_ids.size(0)
        T = query_ids.size(1)
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        # Embed query
        q = self.embed(query_ids)  # (B, T, D)
        q = self.query_proj(q)
        
        # Cross-attend to accumulated state
        # state.kv_sum: (B, nh, N, D) - represents entire novel
        
        # Project query to neuron space
        q_expanded = q.unsqueeze(1)  # (B, 1, T, D)
        q_latent = q_expanded @ self.encoder  # (B, nh, T, N)
        q_sparse = F.relu(q_latent)
        
        # Attend to accumulated state
        denom = state.k_sum.unsqueeze(-1).clamp(min=1e-6)
        state_values = state.kv_sum / denom  # (B, nh, N, D)
        
        # Query-state attention
        attn_weights = q_sparse  # (B, nh, T, N)
        attn_out = torch.einsum('bhtn,bhnd->bhtd', attn_weights, state_values)
        
        # Average over heads and time
        output = attn_out.mean(dim=1).mean(dim=1)  # (B, D)
        
        # Get query neuron trace
        query_trace = q_sparse.mean(dim=2).reshape(B, -1)  # (B, nh*N)
        
        return {
            'embedding': output,
            'neuron_trace': query_trace,
            'state_neurons': state.neuron_accumulator
        }

    def forward(
        self,
        novel_chunks: List[torch.Tensor],
        backstory_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: process all novel chunks, then query with backstory.
        
        NOTE: Chunk processing is done WITHOUT gradients to enable full novel
        processing without OOM. Only the query_state (backstory attention) 
        and subsequent verifier layers compute gradients.
        
        Args:
            novel_chunks: List of token tensors for novel chunks
            backstory_ids: Backstory token IDs (B, T)
            
        Returns:
            Dict with embeddings and traces
        """
        B = backstory_ids.size(0)
        device = backstory_ids.device
        
        # Initialize state
        state = self.init_state(B, device)
        
        # Process chunks WITHOUT gradients (memory efficient for full novel)
        # This allows processing 3000+ chunks without OOM
        position = 0
        
        with torch.no_grad():
            for chunk in novel_chunks:
                chunk = chunk.to(device)
                if chunk.size(0) != B:
                    chunk = chunk.expand(B, -1)
                
                state, _ = self.process_chunk(chunk, state, position)
                position += chunk.size(1)
        
        # Detach state for gradient computation in query
        state.kv_sum = state.kv_sum.detach().requires_grad_(True)
        state.k_sum = state.k_sum.detach().requires_grad_(True)
        state.neuron_accumulator = state.neuron_accumulator.detach().requires_grad_(True)
        
        # Query with backstory - THIS part computes gradients
        result = self.query_state(backstory_ids, state)
        result['num_tokens_processed'] = state.num_tokens
        
        return result
