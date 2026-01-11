"""
BDH Encoder adapted for document-level NLI.

Based on official Pathway BDH implementation.
Adapted for:
1. Embedding extraction (not just next-token prediction)
2. Sparse neuron trace extraction as first-class output
3. Optional bidirectional attention for encoding tasks
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BDHConfig:
    """Configuration for BDH Encoder."""
    
    # Architecture (matching official BDH)
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128  # N = D * multiplier / n_head
    vocab_size: int = 256  # Byte-level encoding
    
    # Regularization
    dropout: float = 0.1
    
    # Attention
    theta: float = 2**16  # RoPE theta (from official)
    use_causal_mask: bool = True  # Official BDH uses causal
    
    @property
    def n_neurons(self) -> int:
        """Total neurons per head."""
        return self.n_embd * self.mlp_internal_dim_multiplier // self.n_head
    
    @property
    def total_neurons(self) -> int:
        """Total neurons across all heads."""
        return self.n_neurons * self.n_head


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """
    Compute RoPE frequency bands.
    Matches official BDH implementation exactly.
    """
    def quantize(t, q=2):
        return (t / q).floor() * q
    
    return (
        1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    """
    Attention module matching official Pathway BDH.
    
    Uses RoPE for positional encoding and linear attention (no softmax).
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        self.freqs = nn.Buffer(
            get_freqs(N, theta=config.theta, dtype=torch.float32).view(1, 1, 1, N)
        )
    
    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for RoPE."""
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin
    
    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
    
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        use_causal: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Compute linear attention.
        
        Args:
            Q: Query tensor (B, H, T, N)
            K: Key tensor (B, H, T, N) - same as Q for BDH
            V: Value tensor (B, 1, T, D)
            use_causal: Override config.use_causal_mask
            
        Returns:
            Attention output (B, 1, T, D)
        """
        assert self.freqs.dtype == torch.float32
        # In official BDH, K is Q (same reference)
        assert K is Q, "BDH requires K == Q for self-attention"
        
        _, _, T, _ = Q.size()
        
        # Compute position phases
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE
        QR = self.rope(r_phases, Q)
        KR = QR  # Same as official: KR = QR when K is Q
        
        # Compute attention scores
        scores = QR @ KR.mT  # (B, H, T, T)
        
        # Apply causal mask (official BDH uses .tril(diagonal=-1))
        causal = use_causal if use_causal is not None else self.config.use_causal_mask
        if causal:
            scores = scores.tril(diagonal=-1)
        
        # Linear attention: no softmax, direct weighting
        return scores @ V


class BDHEncoder(nn.Module):
    """
    BDH encoder adapted for embedding extraction.
    
    Based on official Pathway BDH, with additions for:
    - Returns sequence embeddings (not logits)
    - Extracts sparse neuron traces
    - Supports optional bidirectional attention
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Embedding layer
        self.embed = nn.Embedding(config.vocab_size, D)
        
        # BDH core parameters (matching official shapes)
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        
        # Attention
        self.attn = Attention(config)
        
        # Normalization and regularization (matching official)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        
        # LM head for optional language modeling
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward_layers(
        self,
        x: torch.Tensor,
        return_neuron_trace: bool = True,
        use_causal: Optional[bool] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Process through BDH layers.
        
        Args:
            x: Input tensor (B, 1, T, D)
            return_neuron_trace: Whether to collect sparse activations
            use_causal: Override causal masking
            
        Returns:
            Tuple of (output tensor, list of sparse activations)
        """
        C = self.config
        B = x.size(0)
        T = x.size(2)
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        all_sparse = []
        
        for level in range(C.n_layer):
            # Project to neuron space
            x_latent = x @ self.encoder  # (B, H, T, N)
            
            # Sparse positive activation (key BDH property)
            x_sparse = F.relu(x_latent)  # (B, nh, T, N)
            
            # Linear attention (Q=K=x_sparse, V=x)
            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,  # Same as Q
                V=x,
                use_causal=use_causal
            )
            yKV = self.ln(yKV)
            
            # Value projection
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            
            # Gated combination (multiplicative) - key BDH property
            xy_sparse = x_sparse * y_sparse  # (B, nh, T, N)
            
            if return_neuron_trace:
                all_sparse.append(xy_sparse)
            
            # Dropout
            xy_sparse = self.drop(xy_sparse)
            
            # Decode back to embedding space
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # (B, 1, T, D)
            
            # Residual connection with layer norm
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        return x, all_sparse
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_neuron_trace: bool = True,
        pool_output: bool = True,
        use_causal: Optional[bool] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for encoding.
        
        Args:
            input_ids: Token IDs (B, T)
            attention_mask: Optional mask (B, T), 1 for real tokens
            return_neuron_trace: Whether to return sparse neuron activations
            pool_output: Whether to pool sequence to single embedding
            use_causal: Override causal masking (None = use config)
            targets: Optional targets for language modeling loss
            
        Returns:
            Dict with:
            - 'embedding': Pooled embedding (B, D) or sequence (B, T, D)
            - 'neuron_trace': Sparse activations (B, H*N) if requested
            - 'logits': Language model logits if needed
            - 'loss': LM loss if targets provided
        """
        C = self.config
        B, T = input_ids.size()
        D = C.n_embd
        H = C.n_head
        N = C.n_neurons
        
        # Embed tokens
        x = self.embed(input_ids).unsqueeze(1)  # (B, 1, T, D)
        
        # Layer norm (helps with training - from official)
        x = self.ln(x)
        
        # Process through layers
        x, all_sparse = self.forward_layers(
            x, 
            return_neuron_trace=return_neuron_trace,
            use_causal=use_causal
        )
        
        # Remove head dimension
        x = x.squeeze(1)  # (B, T, D)
        
        result = {}
        
        # Compute logits and loss if needed
        logits = x @ self.lm_head  # (B, T, vocab_size)
        result['logits'] = logits
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            result['loss'] = loss
        
        # Apply attention mask for pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded
        else:
            x_masked = x
        
        # Pool to sequence embedding
        if pool_output:
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                embedding = x_masked.sum(dim=1) / lengths
            else:
                embedding = x.mean(dim=1)
            result['embedding'] = embedding
        else:
            result['embedding'] = x
        
        # Extract neuron trace
        if return_neuron_trace and all_sparse:
            final_sparse = all_sparse[-1]  # (B, H, T, N)
            
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(1).unsqueeze(-1).float()
                final_sparse = final_sparse * mask_exp
                lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                neuron_trace = final_sparse.sum(dim=2) / lengths.unsqueeze(-1)
            else:
                neuron_trace = final_sparse.mean(dim=2)
            
            neuron_trace = neuron_trace.view(B, -1)
            result['neuron_trace'] = neuron_trace
            result['all_sparse'] = all_sparse
        
        return result
    
    def encode(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal: bool = False  # Typically false for encoding
    ) -> torch.Tensor:
        """
        Simple interface for getting embeddings.
        
        Args:
            input_ids: Token IDs (B, T)
            attention_mask: Optional mask (B, T)
            use_causal: Whether to use causal attention
            
        Returns:
            Embedding tensor (B, D)
        """
        result = self.forward(
            input_ids, 
            attention_mask,
            return_neuron_trace=False,
            pool_output=True,
            use_causal=use_causal
        )
        return result['embedding']
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens (matching official BDH).
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx
            result = self.forward(idx_cond, use_causal=True)
            logits = result['logits'][:, -1, :] / temperature
            
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


class StatefulBDHProcessor:
    """
    Stateful processor for sequential chunk processing.
    
    Maintains BDH state across chunks to capture long-range dependencies.
    """
    
    def __init__(self, encoder: BDHEncoder):
        self.encoder = encoder
        self.accumulated_trace = None
        self.chunk_count = 0
    
    def reset(self):
        """Reset state for new document."""
        self.accumulated_trace = None
        self.chunk_count = 0
    
    def process_chunk(
        self, 
        chunk_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single chunk while accumulating state.
        """
        result = self.encoder.forward(
            chunk_ids,
            attention_mask,
            return_neuron_trace=True,
            pool_output=True,
            use_causal=False  # Bidirectional for encoding
        )
        
        if self.accumulated_trace is None:
            self.accumulated_trace = result['neuron_trace']
        else:
            self.accumulated_trace = (
                self.accumulated_trace * self.chunk_count + result['neuron_trace']
            ) / (self.chunk_count + 1)
        
        self.chunk_count += 1
        
        return {
            'embedding': result['embedding'],
            'neuron_trace': result['neuron_trace'],
            'accumulated_trace': self.accumulated_trace
        }
    
    def process_all_chunks(
        self,
        chunk_tokens: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[str, torch.Tensor]:
        """
        Process all chunks of a document.
        """
        self.reset()
        
        num_chunks = chunk_tokens.size(0)
        all_embeddings = []
        all_traces = []
        
        for i in range(0, num_chunks, batch_size):
            batch = chunk_tokens[i:i + batch_size]
            
            with torch.no_grad():
                result = self.encoder.forward(
                    batch,
                    return_neuron_trace=True,
                    pool_output=True,
                    use_causal=False  # Bidirectional for encoding
                )
            
            all_embeddings.append(result['embedding'])
            all_traces.append(result['neuron_trace'])
        
        embeddings = torch.cat(all_embeddings, dim=0)
        traces = torch.cat(all_traces, dim=0)
        
        return {
            'embeddings': embeddings,
            'neuron_traces': traces
        }
