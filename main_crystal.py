"""
Main training script for molecular crystal generation.

This script trains the crystal dynamics model on ASE database of crystal structures.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from crystal.data.crystal_loader import CrystalDataset, collate_crystal_batch
from crystal.models import CrystalDynamics
from crystal.data.periodic_utils import wrap_to_unit_cell


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train crystal generation model')
    
    # Data arguments
    parser.add_argument('--ase_db_path', type=str, required=True,
                        help='Path to ASE database file')
    parser.add_argument('--max_atoms', type=int, default=100,
                        help='Maximum number of atoms per structure')
    parser.add_argument('--use_fractional', action='store_true', default=True,
                        help='Use fractional coordinates')
    
    # Model arguments
    parser.add_argument('--in_node_nf', type=int, default=1,
                        help='Input node feature dimension')
    parser.add_argument('--hidden_nf', type=int, default=128,
                        help='Hidden feature dimension')
    parser.add_argument('--out_node_nf', type=int, default=1,
                        help='Output node feature dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of EGNN layers')
    parser.add_argument('--attention', action='store_true',
                        help='Use attention in EGNN')
    parser.add_argument('--learn_lattice', action='store_true', default=True,
                        help='Learn lattice parameters')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    # Output arguments
    parser.add_argument('--exp_name', type=str, default='crystal_exp',
                        help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        positions = batch['positions'].to(device)  # (batch_size, max_atoms, 3)
        atom_types = batch['atom_types'].to(device)  # (batch_size, max_atoms)
        cell_vectors = batch['cell_vectors'].to(device)  # (batch_size, 3, 3)
        atom_mask = batch['atom_mask'].to(device)  # (batch_size, max_atoms)
        
        batch_size = positions.shape[0]
        max_atoms = positions.shape[1]
        
        # Flatten batch dimension
        positions_flat = positions.reshape(-1, 3)  # (batch_size * max_atoms, 3)
        atom_types_flat = atom_types.reshape(-1)  # (batch_size * max_atoms)
        atom_mask_flat = atom_mask.reshape(-1, 1)  # (batch_size * max_atoms, 1)
        
        # Create one-hot encoding for atom types (simplified)
        # In practice, you'd want proper embeddings
        h = torch.zeros(batch_size * max_atoms, 1, device=device)
        h[atom_mask_flat.squeeze() > 0] = 1.0
        
        # Forward pass (simplified - no diffusion yet)
        # This just tests the model
        try:
            # Use first cell for now (proper batching requires more work)
            h_out, x_out, cell_out = model(
                h, positions_flat, cell_vectors[0],
                node_mask=atom_mask_flat
            )
            
            # Simple reconstruction loss
            loss = nn.functional.mse_loss(
                x_out * atom_mask_flat,
                positions_flat * atom_mask_flat
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.save_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {args.exp_name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    # Create dataset
    print("\nLoading dataset...")
    try:
        dataset = CrystalDataset(
            db_path=args.ase_db_path,
            use_fractional_coords=args.use_fractional,
            max_atoms=args.max_atoms
        )
        print(f"Loaded {len(dataset)} crystal structures")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please provide a valid ASE database file with --ase_db_path")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_crystal_batch
    )
    
    # Create model
    print("\nCreating model...")
    model = CrystalDynamics(
        in_node_nf=args.in_node_nf,
        hidden_nf=args.hidden_nf,
        out_node_nf=args.out_node_nf,
        n_layers=args.n_layers,
        attention=args.attention,
        learn_lattice=args.learn_lattice
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, args.device, epoch)
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    print("\nTraining complete!")
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model to {final_model_path}')


if __name__ == '__main__':
    main()
