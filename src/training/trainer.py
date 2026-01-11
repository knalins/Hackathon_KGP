"""
Trainer for BDH NLI Pipeline.

Handles training loop, validation, checkpointing, and logging.
"""

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models.pipeline import BDHNLIPipeline, PipelineConfig, BDHNLIWithPrecompute
from .loss import MILBCELoss, compute_loss


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    epochs: int = 20
    warmup_epochs: int = 2
    
    # Batching
    batch_size: int = 1  # 1 due to variable chunk counts
    gradient_accumulation: int = 4
    
    # Loss
    pos_weight: float = 1.0
    margin_weight: float = 0.1
    
    # Regularization
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 5
    
    # Logging
    log_every: int = 10
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp: bool = True
    
    def save(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON."""
        with open(path) as f:
            return cls(**json.load(f))


class Trainer:
    """
    Trainer for BDH NLI Pipeline.
    
    Handles the full training loop with:
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Validation
    - Logging
    """
    
    def __init__(
        self,
        model: BDHNLIPipeline,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        pipeline_config: PipelineConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.pipeline_config = pipeline_config
        
        # Device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=config.learning_rate / 10
        )
        
        # Loss
        self.loss_fn = MILBCELoss(
            pos_weight=config.pos_weight,
            margin_weight=config.margin_weight
        )
        
        # AMP
        self.scaler = torch.amp.GradScaler(
            device=config.device,
            enabled=config.use_amp
        )
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Create checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Set seed
        self._set_seed(config.seed)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            backstory_tokens = batch['backstory_tokens'].to(self.device)
            chunk_tokens = batch['chunk_tokens'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Handle batch size 1
            if backstory_tokens.dim() == 3:
                backstory_tokens = backstory_tokens.squeeze(0)
            if chunk_tokens.dim() == 3:
                chunk_tokens = chunk_tokens.squeeze(0)
            
            # Forward pass with AMP
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.use_amp
            ):
                result = self.model(
                    chunk_tokens=chunk_tokens,
                    backstory_tokens=backstory_tokens
                )
                
                loss = self.loss_fn(
                    result['prediction'],
                    labels.squeeze(),
                    result.get('chunk_scores')
                )
            
            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation
            
            # Update progress bar
            if batch_idx % self.config.log_every == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.config.gradient_accumulation:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            Tuple of (accuracy, average_loss)
        """
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            backstory_tokens = batch['backstory_tokens'].to(self.device)
            chunk_tokens = batch['chunk_tokens'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if backstory_tokens.dim() == 3:
                backstory_tokens = backstory_tokens.squeeze(0)
            if chunk_tokens.dim() == 3:
                chunk_tokens = chunk_tokens.squeeze(0)
            
            result = self.model(
                chunk_tokens=chunk_tokens,
                backstory_tokens=backstory_tokens
            )
            
            loss = compute_loss(result['prediction'], labels.squeeze())
            total_loss += loss.item()
            
            # Prediction
            pred = (result['prediction'] > 0.5).long()
            correct += (pred == labels.squeeze()).sum().item()
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.val_loader)
        
        self.val_accuracies.append(accuracy)
        
        return accuracy, avg_loss
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_epoch_{self.epoch + 1}.pt"
            )
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'config': asdict(self.config),
            'pipeline_config': asdict(self.pipeline_config)
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
    
    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns:
            Dict with training history
        """
        print(f"Starting training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_acc, val_loss = self.validate()
            
            # LR scheduling
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.val_loader:
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Accuracy: {val_acc:.4f}")
            
            # Checkpointing
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"  New best model! Accuracy: {val_acc:.4f}")
            
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed / 60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(
            os.path.join(self.config.checkpoint_dir, "final_model.pt")
        )
        
        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
