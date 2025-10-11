"""
Main training script for molecule-conditioned crystal generation.

This script trains the ConditionalCrystalDynamics model on paired
molecule-crystal datasets using molecular conditioning.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import json
import os

from crystal.data.molecular_crystal_loader import (
    MolecularCrystalDataset,
    collate_molecular_crystal_batch,
    load_paired_datasets,
)
from crystal.models import MoleculeEncoder, ConditionalCrystalDynamics
from crystal.data.periodic_utils import wrap_to_unit_cell

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train molecule-conditioned crystal generation model'
    )
    
    # Data arguments
    parser.add_argument('--molecule_db_path', type=str, required=True,
                        help='Path to ASE database file containing molecules')
    parser.add_argument('--crystal_db_path', type=str, required=True,
                        help='Path to ASE database file containing crystals')
    parser.add_argument('--max_atoms', type=int, default=100,
                        help='Maximum number of atoms per structure')
    parser.add_argument('--use_fractional', action='store_true', default=True,
                        help='Use fractional coordinates')
    parser.add_argument('--remove_h', action='store_true',
                        help='Remove hydrogen atoms')
    parser.add_argument('--split_ratios', nargs=3, type=float,
                        default=[0.8, 0.1, 0.1],
                        help='Train/valid/test split ratios')
    
    # Model arguments - Molecule Encoder
    parser.add_argument('--molecule_encoder_layers', type=int, default=6,
                        help='Number of EGNN layers in molecule encoder')
    parser.add_argument('--molecule_encoder_hidden', type=int, default=256,
                        help='Hidden dimension for molecule encoder')
    parser.add_argument('--molecule_encoding_dim', type=int, default=128,
                        help='Dimension of molecular encoding')
    parser.add_argument('--encoder_aggregation', type=str, default='mean_max',
                        choices=['mean', 'max', 'mean_max', 'attention'],
                        help='Aggregation method for molecule encoder')
    
    # Model arguments - Crystal Dynamics
    parser.add_argument('--hidden_nf', type=int, default=256,
                        help='Hidden feature dimension for crystal dynamics')
    parser.add_argument('--n_layers', type=int, default=9,
                        help='Number of EGNN layers in crystal dynamics')
    parser.add_argument('--attention', action='store_true',
                        help='Use attention in EGNN')
    parser.add_argument('--learn_lattice', action='store_true', default=True,
                        help='Learn lattice parameters')
    parser.add_argument('--conditioning_method', type=str, default='film',
                        choices=['film', 'add'],
                        help='Conditioning method (film or add)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # Conditioning arguments
    parser.add_argument('--guidance_scale', type=float, default=0.0,
                        help='Guidance scale for classifier-free guidance (0=no CFG)')
    parser.add_argument('--cfg_dropout', type=float, default=0.1,
                        help='Dropout probability for classifier-free guidance')
    
    # Output arguments
    parser.add_argument('--exp_name', type=str, default='crystal_conditional',
                        help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N batches')
    
    return parser.parse_args()


def train_epoch(
    molecule_encoder,
    crystal_dynamics,
    dataloader,
    optimizer,
    device,
    epoch,
    args,
):
    """Train for one epoch."""
    molecule_encoder.train()
    crystal_dynamics.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # Get molecule data
            mol_positions = batch['molecule']['positions'].to(device)  # [batch, n_mol_atoms, 3]
            mol_one_hot = batch['molecule']['one_hot'].to(device)  # [batch, n_mol_atoms, n_types]
            mol_mask = batch['molecule']['node_mask'].to(device)  # [batch, n_mol_atoms]
            
            # Get crystal data
            cryst_positions = batch['crystal']['positions'].to(device)  # [batch, n_cryst_atoms, 3]
            cryst_one_hot = batch['crystal']['one_hot'].to(device)  # [batch, n_cryst_atoms, n_types]
            cryst_mask = batch['crystal']['node_mask'].to(device)  # [batch, n_cryst_atoms]
            cryst_cell = batch['crystal']['cell'].to(device)  # [batch, 3, 3]
            
            batch_size = mol_positions.shape[0]
            
            # Encode molecules to context vectors
            # Note: MoleculeEncoder expects batched input
            # We need to process each molecule separately due to variable sizes
            contexts = []
            for i in range(batch_size):
                n_mol = batch['molecule']['num_atoms'][i].item()
                c_mol = molecule_encoder(
                    h=mol_one_hot[i:i+1, :n_mol],
                    x=mol_positions[i:i+1, :n_mol],
                    node_mask=mol_mask[i:i+1, :n_mol],
                )
                contexts.append(c_mol)
            
            context = torch.cat(contexts, dim=0)  # [batch, encoding_dim]
            
            # Classifier-free guidance: randomly drop context
            if args.cfg_dropout > 0 and torch.rand(1).item() < args.cfg_dropout:
                context = torch.zeros_like(context)
            
            # Process crystal (flatten batch for PeriodicEGNN)
            # For simplicity, process one at a time
            losses = []
            for i in range(batch_size):
                n_cryst = batch['crystal']['num_atoms'][i].item()
                
                h_cryst = cryst_one_hot[i, :n_cryst]  # [n_atoms, n_types]
                x_cryst = cryst_positions[i, :n_cryst]  # [n_atoms, 3]
                mask_cryst = cryst_mask[i, :n_cryst].unsqueeze(-1)  # [n_atoms, 1]
                cell_cryst = cryst_cell[i]  # [3, 3]
                ctx_cryst = context[i:i+1]  # [1, encoding_dim]
                
                # Forward pass
                h_out, x_out, cell_out = crystal_dynamics(
                    h=h_cryst,
                    x=x_cryst,
                    cell_vectors=cell_cryst,
                    context=ctx_cryst,
                    node_mask=mask_cryst,
                )
                
                # Simple reconstruction loss (this would be replaced with diffusion loss)
                coord_loss = nn.functional.mse_loss(
                    x_out * mask_cryst,
                    x_cryst * mask_cryst,
                )
                
                feature_loss = nn.functional.mse_loss(
                    h_out * mask_cryst,
                    h_cryst * mask_cryst,
                )
                
                loss = coord_loss + 0.1 * feature_loss
                losses.append(loss)
            
            # Average loss over batch
            loss = torch.stack(losses).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(molecule_encoder.parameters()) + list(crystal_dynamics.parameters()),
                args.gradient_clip
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % args.log_every == 0:
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                    f'Loss: {loss.item():.6f}'
                )
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.save_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")
    
    # Load datasets
    logger.info("\nLoading paired datasets...")
    try:
        datasets, dataset_info = load_paired_datasets(
            molecule_db_path=args.molecule_db_path,
            crystal_db_path=args.crystal_db_path,
            split_ratios=tuple(args.split_ratios),
            remove_h=args.remove_h,
            use_fractional_coords=args.use_fractional,
            max_atoms=args.max_atoms,
        )
        
        logger.info(f"Train: {dataset_info['n_train']} samples")
        logger.info(f"Valid: {dataset_info['n_valid']} samples")
        logger.info(f"Test: {dataset_info['n_test']} samples")
        logger.info(f"Number of atom types: {dataset_info['num_atom_types']}")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        logger.error("Please provide valid ASE database files")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_molecular_crystal_batch,
        num_workers=args.num_workers,
    )
    
    valid_loader = DataLoader(
        datasets['valid'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_molecular_crystal_batch,
        num_workers=args.num_workers,
    )
    
    # Create models
    logger.info("\nCreating models...")
    
    molecule_encoder = MoleculeEncoder(
        in_node_nf=dataset_info['num_atom_types'],
        hidden_nf=args.molecule_encoder_hidden,
        out_nf=args.molecule_encoding_dim,
        n_layers=args.molecule_encoder_layers,
        aggregation_method=args.encoder_aggregation,
    ).to(args.device)
    
    crystal_dynamics = ConditionalCrystalDynamics(
        in_node_nf=dataset_info['num_atom_types'],
        context_node_nf=args.molecule_encoding_dim,
        hidden_nf=args.hidden_nf,
        n_layers=args.n_layers,
        attention=args.attention,
        learn_lattice=args.learn_lattice,
        conditioning_method=args.conditioning_method,
    ).to(args.device)
    
    n_params_encoder = sum(p.numel() for p in molecule_encoder.parameters())
    n_params_dynamics = sum(p.numel() for p in crystal_dynamics.parameters())
    logger.info(f"Molecule encoder parameters: {n_params_encoder:,}")
    logger.info(f"Crystal dynamics parameters: {n_params_dynamics:,}")
    logger.info(f"Total parameters: {n_params_encoder + n_params_dynamics:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(molecule_encoder.parameters()) + list(crystal_dynamics.parameters()),
        lr=args.lr,
    )
    
    # Training loop
    logger.info("\nStarting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            molecule_encoder,
            crystal_dynamics,
            train_loader,
            optimizer,
            args.device,
            epoch,
            args,
        )
        
        logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'molecule_encoder_state_dict': molecule_encoder.state_dict(),
                'crystal_dynamics_state_dict': crystal_dynamics.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'args': vars(args),
                'dataset_info': dataset_info,
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
    
    logger.info("\nTraining complete!")
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'molecule_encoder_state_dict': molecule_encoder.state_dict(),
        'crystal_dynamics_state_dict': crystal_dynamics.state_dict(),
        'args': vars(args),
        'dataset_info': dataset_info,
    }, final_model_path)
    logger.info(f'Saved final model to {final_model_path}')


if __name__ == '__main__':
    main()
