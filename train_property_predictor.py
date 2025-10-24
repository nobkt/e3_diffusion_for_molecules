"""
Training Script for Property Predictor

Trains a property prediction model on crystal structures with known properties.
This model can then be used to validate generated crystals.

Usage:
    python train_property_predictor.py \\
        --crystal_db_path data/crystals_with_props.db \\
        --property_names bandgap melting_point \\
        --exp_name property_predictor \\
        --n_epochs 100 \\
        --batch_size 32

This implements Component P4-1 (Property Validation System) from Phase 4.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from crystal.data.crystal_dataset_with_properties import CrystalDatasetWithProperties
from crystal.evaluation.property_predictor import PropertyPredictor


class PropertyPredictorTrainer:
    """
    Trainer for property prediction model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: PropertyPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        property_names: List[str],
        exp_name: str,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.property_names = property_names
        self.exp_name = exp_name
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        # Create output directory
        self.output_dir = Path('outputs') / exp_name / 'property_predictor'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        property_losses = {name: 0.0 for name in self.property_names}
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')
        for batch in pbar:
            # Move batch to device
            positions = batch['crystal_positions'].to(self.device)
            cell = batch['crystal_cell'].to(self.device)
            atomic_numbers = batch['crystal_atomic_numbers'].to(self.device)
            properties_true = batch['properties'].to(self.device)
            
            # Forward pass
            predictions = self.model(
                positions, cell, atomic_numbers,
                return_normalized=True  # Train on normalized values
            )
            
            # Compute loss for each property
            batch_loss = 0.0
            for i, prop_name in enumerate(self.property_names):
                pred = predictions[prop_name]
                true = properties_true[:, i:i+1]
                
                # Normalize true values
                mean = self.model.property_mean[i]
                std = self.model.property_std[i]
                true_normalized = (true - mean) / (std + self.model.eps)
                
                loss = self.criterion(pred, true_normalized)
                batch_loss += loss
                property_losses[prop_name] += loss.item()
            
            # Average loss over properties
            batch_loss = batch_loss / len(self.property_names)
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': batch_loss.item()})
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / n_batches,
        }
        for prop_name in self.property_names:
            metrics[f'train_{prop_name}_loss'] = property_losses[prop_name] / n_batches
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        property_losses = {name: 0.0 for name in self.property_names}
        property_maes = {name: 0.0 for name in self.property_names}
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]')
            for batch in pbar:
                # Move batch to device
                positions = batch['crystal_positions'].to(self.device)
                cell = batch['crystal_cell'].to(self.device)
                atomic_numbers = batch['crystal_atomic_numbers'].to(self.device)
                properties_true = batch['properties'].to(self.device)
                
                # Forward pass
                predictions = self.model(
                    positions, cell, atomic_numbers,
                    return_normalized=False  # Get denormalized predictions
                )
                
                # Compute metrics for each property
                batch_loss = 0.0
                for i, prop_name in enumerate(self.property_names):
                    pred = predictions[prop_name]
                    true = properties_true[:, i:i+1]
                    
                    # MSE loss
                    loss = self.criterion(pred, true)
                    batch_loss += loss
                    property_losses[prop_name] += loss.item()
                    
                    # MAE
                    mae = torch.abs(pred - true).mean()
                    property_maes[prop_name] += mae.item()
                
                # Average loss over properties
                batch_loss = batch_loss / len(self.property_names)
                total_loss += batch_loss.item()
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': batch_loss.item()})
        
        # Compute epoch metrics
        metrics = {
            'val_loss': total_loss / n_batches,
        }
        for prop_name in self.property_names:
            metrics[f'val_{prop_name}_loss'] = property_losses[prop_name] / n_batches
            metrics[f'val_{prop_name}_mae'] = property_maes[prop_name] / n_batches
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'property_names': self.property_names,
            'property_mean': self.model.property_mean,
            'property_std': self.model.property_std,
            'model_config': {
                'hidden_dim': self.model.hidden_dim,
                'n_layers': self.model.n_layers,
                'max_neighbors': self.model.max_neighbors,
                'cutoff_radius': self.model.cutoff_radius,
            }
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f'Saved best checkpoint to {best_path}')
    
    def train(self, n_epochs: int):
        """
        Train for multiple epochs.
        
        Args:
            n_epochs: Number of epochs to train
        """
        print(f'Training property predictor for {n_epochs} epochs')
        print(f'Device: {self.device}')
        print(f'Properties: {self.property_names}')
        print(f'Output directory: {self.output_dir}')
        
        for epoch in range(1, n_epochs + 1):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Print metrics
            print(f'\nEpoch {epoch}/{n_epochs}')
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            for prop_name in self.property_names:
                print(f"  Val {prop_name} MAE: {val_metrics[f'val_{prop_name}_mae']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(is_best=is_best)
        
        print(f'\nTraining complete!')
        print(f'Best validation loss: {self.best_val_loss:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='Train property predictor for crystal validation'
    )
    
    # Data arguments
    parser.add_argument(
        '--crystal_db_path',
        type=str,
        required=True,
        help='Path to crystal database with properties'
    )
    parser.add_argument(
        '--property_names',
        type=str,
        nargs='+',
        required=True,
        help='Names of properties to predict (e.g., bandgap melting_point)'
    )
    
    # Model arguments
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for model (default: 256)'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of EGNN layers (default: 4)'
    )
    parser.add_argument(
        '--max_neighbors',
        type=int,
        default=32,
        help='Maximum number of neighbors (default: 32)'
    )
    parser.add_argument(
        '--cutoff_radius',
        type=float,
        default=8.0,
        help='Cutoff radius for neighbor search in Angstroms (default: 8.0)'
    )
    
    # Training arguments
    parser.add_argument(
        '--exp_name',
        type=str,
        default='property_predictor',
        help='Experiment name (default: property_predictor)'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.crystal_db_path):
        raise ValueError(f'Crystal database not found: {args.crystal_db_path}')
    
    if not args.property_names:
        raise ValueError('At least one property name must be specified')
    
    print('Loading datasets...')
    
    # Create datasets
    train_dataset = CrystalDatasetWithProperties(
        db_path=args.crystal_db_path,
        property_names=args.property_names,
        split='train'
    )
    
    val_dataset = CrystalDatasetWithProperties(
        db_path=args.crystal_db_path,
        property_names=args.property_names,
        split='val'
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get property statistics from dataset
    property_mean = train_dataset.get_property_statistics()['mean']
    property_std = train_dataset.get_property_statistics()['std']
    
    print(f'Property statistics:')
    for i, prop_name in enumerate(args.property_names):
        print(f'  {prop_name}: mean={property_mean[i]:.4f}, std={property_std[i]:.4f}')
    
    # Create model
    print('Creating model...')
    model = PropertyPredictor(
        property_names=args.property_names,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        max_neighbors=args.max_neighbors,
        cutoff_radius=args.cutoff_radius,
    )
    
    # Set normalization parameters
    model.set_normalization_params(
        torch.tensor(property_mean),
        torch.tensor(property_std)
    )
    
    print(f'Model: {model}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create trainer
    trainer = PropertyPredictorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        property_names=args.property_names,
        exp_name=args.exp_name,
        learning_rate=args.learning_rate,
    )
    
    # Train
    trainer.train(n_epochs=args.n_epochs)


if __name__ == '__main__':
    main()
