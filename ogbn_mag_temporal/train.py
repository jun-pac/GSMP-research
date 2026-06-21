"""
Training loop for temporal message passing models.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from ogb.nodeproppred import Evaluator
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for temporal message passing models."""
    
    def __init__(
        self,
        model: nn.Module,
        data: HeteroData,
        split_idx: Dict,
        device: torch.device,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        num_epochs: int = 200,
        patience: int = 50,
        eval_steps: int = 1,
        run_label: str = ""
    ):
        """
        Args:
            model: PyTorch model
            data: HeteroData with features and labels
            split_idx: Dict with 'train', 'valid', 'test' node indices for paper nodes
            device: torch device
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            eval_steps: Evaluate every N epochs
        """
        self.model = model
        self.data = data
        self.split_idx = split_idx
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        self.num_epochs = num_epochs
        self.patience = patience
        self.eval_steps = eval_steps
        self.run_label = run_label
        
        self.best_valid_acc = 0.0
        self.best_test_acc = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # For OGB evaluation
        self.evaluator = Evaluator(name="ogbn-mag")
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  LR: {lr}, Weight decay: {weight_decay}")
        logger.info(f"  Max epochs: {num_epochs}, Patience: {patience}")
    
    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        
        # Forward pass
        logits = self.model(self.data)
        
        # Get training node indices
        train_idx = self.split_idx['train'].to(self.device)
        
        # Compute loss on training nodes only
        train_logits = logits[train_idx]
        train_labels = self.data['paper'].y[train_idx]
        
        loss = self.loss_fn(train_logits, train_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on valid and test sets."""
        self.model.eval()
        
        # Forward pass
        logits = self.model(self.data)
        
        # Get predictions
        pred = logits.argmax(dim=1).cpu().numpy()
        
        # Get ground truth labels
        y = self.data['paper'].y.cpu().numpy()
        
        # Evaluate using OGB evaluator
        valid_idx = self.split_idx['valid'].cpu().numpy()
        test_idx = self.split_idx['test'].cpu().numpy()
        
        valid_acc = self.evaluator.eval({
            'y_true': y[valid_idx].reshape(-1, 1),
            'y_pred': pred[valid_idx].reshape(-1, 1)
        })['acc']
        
        test_acc = self.evaluator.eval({
            'y_true': y[test_idx].reshape(-1, 1),
            'y_pred': pred[test_idx].reshape(-1, 1)
        })['acc']
        
        return valid_acc, test_acc
    
    def train(self) -> Tuple[float, float, int]:
        """
        Run training loop with early stopping.
        
        Returns:
            (best_valid_acc, best_test_acc, best_epoch)
        """
        label = f"{self.run_label} | " if self.run_label else ""
        logger.info(f"\nStarting training for {self.run_label or 'run'}...")
        
        for epoch in range(self.num_epochs):
            loss = self.train_epoch()
            
            # Evaluate periodically
            if epoch % self.eval_steps == 0 or epoch == self.num_epochs - 1:
                valid_acc, test_acc = self.evaluate()
                
                if valid_acc > self.best_valid_acc:
                    self.best_valid_acc = valid_acc
                    self.best_test_acc = test_acc
                    self.best_epoch = epoch
                    self.early_stop_counter = 0
                    status = " [BEST]"
                else:
                    self.early_stop_counter += 1
                    status = ""

                logger.info(f"{label}Epoch {epoch:05d} | Loss: {loss:.4f} | "
                            f"Valid Acc: {valid_acc:.4f} | Test Acc: {test_acc:.4f}{status}")
                
                # Early stopping
                if self.early_stop_counter >= self.patience:
                    logger.info(f"\n{label}Early stopping at epoch {epoch} "
                                f"(best epoch: {self.best_epoch})")
                    break
        
        logger.info(f"\n{label}Training complete!")
        logger.info(f"{label}Best Valid Acc: {self.best_valid_acc:.4f}")
        logger.info(f"{label}Best Test Acc:  {self.best_test_acc:.4f}")
        logger.info(f"{label}Best Epoch: {self.best_epoch}")
        
        return self.best_valid_acc, self.best_test_acc, self.best_epoch


def train_model(
    model: nn.Module,
    data: HeteroData,
    split_idx: Dict,
    device: torch.device,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    num_epochs: int = 200,
    patience: int = 50,
    eval_steps: int = 1,
    run_label: str = "",
) -> Tuple[float, float, int]:
    """
    Convenience function to train a model.
    
    Returns:
        (best_valid_acc, best_test_acc, best_epoch)
    """
    trainer = Trainer(
        model=model,
        data=data,
        split_idx=split_idx,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        eval_steps=eval_steps,
        run_label=run_label
    )
    
    return trainer.train()
