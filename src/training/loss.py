"""
Loss Functions for BDH NLI Pipeline.

Implements BCE with class weighting and MIL-specific losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[float] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Binary Cross Entropy loss with optional class weighting.
    
    Args:
        predictions: Predicted probabilities (B,) or (B, 1)
        targets: Ground truth labels (B,), 0 or 1
        pos_weight: Weight for positive class (for imbalanced data)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Loss tensor
    """
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1).float()
    
    # Clamp predictions for numerical stability
    predictions = predictions.clamp(1e-7, 1 - 1e-7)
    
    # BCE: -[y*log(p) + (1-y)*log(1-p)]
    loss = -targets * torch.log(predictions) - (1 - targets) * torch.log(1 - predictions)
    
    # Apply class weighting
    if pos_weight is not None:
        weights = torch.where(targets == 1, pos_weight, 1.0)
        loss = loss * weights
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class MILBCELoss(nn.Module):
    """
    MIL-aware BCE Loss.
    
    Includes optional margin loss to encourage separation.
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        margin_weight: float = 0.1,
        margin: float = 0.3
    ):
        """
        Args:
            pos_weight: Weight for positive class
            margin_weight: Weight for margin loss term
            margin: Margin for MIL margin loss
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.margin_weight = margin_weight
        self.margin = margin
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        chunk_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            prediction: Aggregated prediction (B,) or scalar
            target: Ground truth (B,) or scalar
            chunk_scores: Optional chunk-level scores for margin loss
            
        Returns:
            Total loss
        """
        # BCE loss
        bce_loss = compute_loss(
            prediction,
            target,
            pos_weight=self.pos_weight
        )
        
        # MIL margin loss (if chunk scores provided)
        if chunk_scores is not None and self.margin_weight > 0:
            margin_loss = self._margin_loss(chunk_scores, target)
            return bce_loss + self.margin_weight * margin_loss
        
        return bce_loss
    
    def _margin_loss(
        self,
        chunk_scores: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Margin loss to encourage separation.
        
        For consistent: best chunk should score > 0.5 + margin
        For contradict: best chunk should score < 0.5 - margin
        """
        if chunk_scores.dim() == 1:
            best_score = chunk_scores.max()
            is_positive = target.item() == 1 if target.dim() == 0 else target[0].item() == 1
        else:
            best_score = chunk_scores.max(dim=-1)[0]
            is_positive = target[0].item() == 1
        
        threshold = 0.5
        if is_positive:
            # Should be > threshold + margin
            loss = F.relu(threshold + self.margin - best_score)
        else:
            # Should be < threshold - margin
            loss = F.relu(best_score - (threshold - self.margin))
        
        return loss.mean() if loss.dim() > 0 else loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces loss for well-classified examples, focuses on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: (B,)
            targets: (B,)
            
        Returns:
            Loss tensor
        """
        predictions = predictions.view(-1).clamp(1e-7, 1 - 1e-7)
        targets = targets.view(-1).float()
        
        # BCE component
        bce = -targets * torch.log(predictions) - (1 - targets) * torch.log(1 - predictions)
        
        # Focal component
        pt = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
