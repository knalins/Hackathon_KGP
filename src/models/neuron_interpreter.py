"""
Neuron Interpreter for Concept-Level Explanations.

Exploits BDH's monosemanticity: individual neurons encode specific concepts.
This module:
1. Learns to map neurons to semantic concepts
2. Provides interpretable explanations for predictions
3. Shows which concepts were activated for the decision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class NeuronConcept:
    """A learned concept associated with neurons."""
    concept_id: int
    name: str
    description: str
    neuron_indices: List[int]  # Which neurons activate for this concept
    activation_threshold: float = 0.5
    

class NeuronInterpreter(nn.Module):
    """
    Interprets BDH neuron activations as semantic concepts.
    
    Key insight: BDH neurons are monosemantic - each encodes a specific concept.
    This module learns to cluster and name these concepts.
    """
    
    def __init__(
        self,
        neuron_dim: int,
        num_concepts: int = 64,
        hidden_dim: int = 256
    ):
        """
        Args:
            neuron_dim: Total neurons (H * N)
            num_concepts: Number of semantic concepts to learn
            hidden_dim: Hidden dimension for concept embeddings
        """
        super().__init__()
        self.neuron_dim = neuron_dim
        self.num_concepts = num_concepts
        
        # Concept embeddings - each concept is a pattern of neurons
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, neuron_dim) * 0.02
        )
        
        # Concept importance predictor
        self.importance_scorer = nn.Sequential(
            nn.Linear(neuron_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        # Predefined concept names (can be learned or predefined)
        self.concept_names = [
            # Character concepts
            "Character Identity", "Character Relationship", "Character Emotion",
            "Character Action", "Character Dialogue", "Character Backstory",
            # Location concepts
            "Location Description", "Location Change", "Time Period",
            # Plot concepts
            "Plot Event", "Conflict", "Resolution", "Mystery", "Revelation",
            # Theme concepts
            "Theme Loyalty", "Theme Betrayal", "Theme Love", "Theme Death",
            # Style concepts  
            "Narrative Voice", "Description Style", "Dialogue Style",
            # Consistency concepts
            "Factual Detail", "Timeline Reference", "Character Trait",
            "Relationship Status", "Location Reference", "Object Reference",
            # Contradiction indicators
            "Contradiction Signal", "Inconsistency", "Timeline Conflict",
            "Character Conflict", "Location Mismatch",
            # Neutral
            "Neutral", "Unknown", "Ambiguous"
        ] + [f"Concept_{i}" for i in range(64 - 32)]  # Pad to 64
        
        self.concept_names = self.concept_names[:num_concepts]
    
    def compute_concept_activations(
        self,
        neuron_trace: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute which concepts are activated by a neuron trace.
        
        Args:
            neuron_trace: Neuron activations (B, H*N) or (H*N,)
            
        Returns:
            Tuple of (concept_scores, importance_weights)
        """
        if neuron_trace.dim() == 1:
            neuron_trace = neuron_trace.unsqueeze(0)
        
        # Compute similarity to concept embeddings
        # concept_embeddings: (num_concepts, neuron_dim)
        # neuron_trace: (B, neuron_dim)
        
        concept_sims = F.cosine_similarity(
            neuron_trace.unsqueeze(1),  # (B, 1, N)
            self.concept_embeddings.unsqueeze(0),  # (1, C, N)
            dim=-1
        )  # (B, C)
        
        # Compute importance
        importance = self.importance_scorer(neuron_trace)  # (B, C)
        importance = F.softmax(importance, dim=-1)
        
        return concept_sims, importance
    
    def get_top_concepts(
        self,
        neuron_trace: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get top-K activated concepts for explanation.
        
        Args:
            neuron_trace: (H*N,) single sample
            top_k: Number of concepts to return
            
        Returns:
            List of concept dicts with name, score, importance
        """
        concept_sims, importance = self.compute_concept_activations(neuron_trace)
        concept_sims = concept_sims.squeeze(0)
        importance = importance.squeeze(0)
        
        # Combined score
        combined = concept_sims * importance
        
        # Get top-K
        top_vals, top_idx = combined.topk(top_k)
        
        results = []
        for i, idx in enumerate(top_idx.tolist()):
            results.append({
                'concept': self.concept_names[idx],
                'activation': concept_sims[idx].item(),
                'importance': importance[idx].item(),
                'combined_score': top_vals[i].item()
            })
        
        return results
    
    def explain_prediction(
        self,
        backstory_trace: torch.Tensor,
        chunk_traces: torch.Tensor,
        prediction: float,
        retrieved_indices: torch.Tensor
    ) -> Dict:
        """
        Generate human-readable explanation for a prediction.
        
        Args:
            backstory_trace: Backstory neurons (H*N,)
            chunk_traces: Retrieved chunk neurons (K, H*N)
            prediction: Model prediction (0-1)
            retrieved_indices: Which chunks were retrieved
            
        Returns:
            Explanation dict
        """
        # Get backstory concepts
        backstory_concepts = self.get_top_concepts(backstory_trace, top_k=5)
        
        # Get shared concepts across chunks
        all_chunk_concepts = []
        for idx in range(chunk_traces.size(0)):
            chunk_concepts = self.get_top_concepts(chunk_traces[idx], top_k=3)
            all_chunk_concepts.append({
                'chunk_idx': retrieved_indices[idx].item() if retrieved_indices is not None else idx,
                'concepts': chunk_concepts
            })
        
        # Find overlap between backstory and chunk concepts
        backstory_concept_names = {c['concept'] for c in backstory_concepts}
        shared_concepts = []
        conflicting_concepts = []
        
        for chunk_info in all_chunk_concepts:
            for concept in chunk_info['concepts']:
                if concept['concept'] in backstory_concept_names:
                    shared_concepts.append({
                        'concept': concept['concept'],
                        'chunk_idx': chunk_info['chunk_idx'],
                        'activation': concept['activation']
                    })
                elif 'Conflict' in concept['concept'] or 'Mismatch' in concept['concept']:
                    conflicting_concepts.append({
                        'concept': concept['concept'],
                        'chunk_idx': chunk_info['chunk_idx'],
                        'activation': concept['activation']
                    })
        
        # Generate explanation text
        label = "CONSISTENT" if prediction > 0.5 else "CONTRADICTORY"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        explanation = {
            'label': label,
            'confidence': f"{confidence:.1%}",
            'backstory_concepts': backstory_concepts,
            'shared_concepts': shared_concepts[:5],
            'conflicting_concepts': conflicting_concepts[:3],
            'evidence_chunks': all_chunk_concepts[:3],
            'reasoning': self._generate_reasoning(
                label, backstory_concepts, shared_concepts, conflicting_concepts
            )
        }
        
        return explanation
    
    def _generate_reasoning(
        self,
        label: str,
        backstory_concepts: List[Dict],
        shared_concepts: List[Dict],
        conflicting_concepts: List[Dict]
    ) -> str:
        """Generate human-readable reasoning text."""
        parts = []
        
        # Key concepts in backstory
        top_concept_names = [c['concept'] for c in backstory_concepts[:3]]
        parts.append(f"The backstory mentions: {', '.join(top_concept_names)}.")
        
        if label == "CONSISTENT":
            if shared_concepts:
                shared_names = list(set(c['concept'] for c in shared_concepts))[:3]
                parts.append(
                    f"The novel confirms these through: {', '.join(shared_names)}."
                )
            parts.append("No significant contradictions were detected.")
        else:
            if conflicting_concepts:
                conflict_names = [c['concept'] for c in conflicting_concepts[:2]]
                parts.append(
                    f"Contradictions detected in: {', '.join(conflict_names)}."
                )
            else:
                parts.append("Key backstory elements were not found in the novel.")
        
        return " ".join(parts)
    
    def forward(
        self,
        neuron_trace: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returns concept activations.
        
        Args:
            neuron_trace: (B, H*N)
            
        Returns:
            Dict with concept scores and importance
        """
        concept_sims, importance = self.compute_concept_activations(neuron_trace)
        
        return {
            'concept_scores': concept_sims,
            'importance': importance,
            'combined': concept_sims * importance
        }


class LearnedConceptMapper(nn.Module):
    """
    Learns to map neurons to interpretable concepts during training.
    
    Uses contrastive learning to cluster neurons that fire together.
    """
    
    def __init__(
        self,
        neuron_dim: int,
        num_clusters: int = 32,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Cluster assignment network
        self.cluster_net = nn.Sequential(
            nn.Linear(neuron_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters)
        )
        
        # Cluster centroids (learned)
        self.centroids = nn.Parameter(
            torch.randn(num_clusters, neuron_dim) * 0.1
        )
        
    def forward(self, neuron_trace: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute cluster assignments."""
        # Soft assignment
        logits = self.cluster_net(neuron_trace)
        assignments = F.softmax(logits, dim=-1)
        
        # Hard assignment
        hard_assignment = assignments.argmax(dim=-1)
        
        return {
            'soft_assignment': assignments,
            'hard_assignment': hard_assignment,
            'cluster_logits': logits
        }
    
    def clustering_loss(
        self,
        neuron_trace: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Contrastive loss to encourage clean clustering.
        
        Neurons that fire together should cluster together.
        """
        # Get assignments
        result = self(neuron_trace)
        assignments = result['soft_assignment']  # (B, K)
        
        # Pull neurons toward their centroid
        # Push centroids apart
        
        # Compute distance to centroids
        dists = torch.cdist(neuron_trace, self.centroids)  # (B, K)
        
        # Weighted distance (weight by assignment)
        pull_loss = (assignments * dists).sum(dim=-1).mean()
        
        # Push centroids apart
        centroid_dists = torch.cdist(self.centroids, self.centroids)
        push_loss = -centroid_dists.mean() * 0.1
        
        return pull_loss + push_loss
