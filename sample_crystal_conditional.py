"""
Sampling script for molecule-conditioned crystal generation.

This script loads a trained model and generates crystal structures
conditioned on input molecules from an ASE database.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from ase import Atoms
from ase.db import connect

from crystal.models import MoleculeEncoder, ConditionalCrystalDynamics
from crystal.data.periodic_utils import fractional_to_cartesian

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sample crystals conditioned on molecules'
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--molecule_db_path', type=str, required=True,
                        help='Path to ASE database with input molecules')
    parser.add_argument('--molecule_ids', type=str, default=None,
                        help='Comma-separated list of molecule IDs (default: all)')
    
    # Sampling arguments
    parser.add_argument('--n_samples_per_molecule', type=int, default=10,
                        help='Number of crystal samples per molecule')
    parser.add_argument('--guidance_scale', type=float, default=0.0,
                        help='Guidance scale (0=no guidance)')
    parser.add_argument('--Z', type=int, default=4,
                        help='Number of molecules per unit cell')
    parser.add_argument('--initial_cell_size', type=float, default=10.0,
                        help='Initial cubic cell size in Angstroms')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/generated',
                        help='Directory to save generated crystals')
    parser.add_argument('--output_format', type=str, default='cif',
                        choices=['cif', 'xyz', 'ase_db'],
                        help='Output format for crystals')
    
    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model arguments
    args = checkpoint['args']
    dataset_info = checkpoint['dataset_info']
    
    # Create models
    molecule_encoder = MoleculeEncoder(
        in_node_nf=dataset_info['num_atom_types'],
        hidden_nf=args['molecule_encoder_hidden'],
        out_nf=args['molecule_encoding_dim'],
        n_layers=args['molecule_encoder_layers'],
        aggregation_method=args['encoder_aggregation'],
    ).to(device)
    
    crystal_dynamics = ConditionalCrystalDynamics(
        in_node_nf=dataset_info['num_atom_types'],
        context_node_nf=args['molecule_encoding_dim'],
        hidden_nf=args['hidden_nf'],
        n_layers=args['n_layers'],
        attention=args['attention'],
        learn_lattice=args['learn_lattice'],
        conditioning_method=args['conditioning_method'],
    ).to(device)
    
    # Load state dicts
    molecule_encoder.load_state_dict(checkpoint['molecule_encoder_state_dict'])
    crystal_dynamics.load_state_dict(checkpoint['crystal_dynamics_state_dict'])
    
    molecule_encoder.eval()
    crystal_dynamics.eval()
    
    logger.info("Model loaded successfully")
    
    return molecule_encoder, crystal_dynamics, dataset_info


def load_molecule(db_path, mol_id, dataset_info, remove_h=False):
    """Load a molecule from ASE database."""
    db = connect(db_path)
    row = db.get(mol_id)
    atoms = row.toatoms()
    
    if remove_h:
        mask = atoms.numbers != 1
        atoms = atoms[mask]
    
    # Convert to tensors
    atomic_numbers = atoms.numbers
    positions = torch.tensor(atoms.positions, dtype=torch.float32)
    
    # Center at origin
    positions = positions - positions.mean(dim=0, keepdim=True)
    
    # Create one-hot encoding
    atom_encoder = dataset_info['atom_encoder']
    atom_types = torch.tensor(
        [atom_encoder[num] for num in atomic_numbers],
        dtype=torch.long
    )
    
    n_atoms = len(atoms)
    num_atom_types = dataset_info['num_atom_types']
    one_hot = torch.zeros(n_atoms, num_atom_types, dtype=torch.float32)
    one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
    
    node_mask = torch.ones(n_atoms, dtype=torch.bool)
    
    return {
        'positions': positions,
        'one_hot': one_hot,
        'node_mask': node_mask,
        'atom_types': atom_types,
        'num_atoms': n_atoms,
        'atomic_numbers': atomic_numbers,
    }


def sample_crystal(
    molecule_encoder,
    crystal_dynamics,
    molecule,
    device,
    Z=4,
    initial_cell_size=10.0,
):
    """
    Sample a crystal structure conditioned on a molecule.
    
    This is a simplified version - in practice, would use proper diffusion sampling.
    """
    with torch.no_grad():
        # Encode molecule
        h_mol = molecule['one_hot'].unsqueeze(0).to(device)
        x_mol = molecule['positions'].unsqueeze(0).to(device)
        mask_mol = molecule['node_mask'].unsqueeze(0).to(device)
        
        context = molecule_encoder(h_mol, x_mol, mask_mol)
        
        # Initialize crystal (replicate molecule Z times with random rotations)
        n_mol_atoms = molecule['num_atoms']
        n_cryst_atoms = n_mol_atoms * Z
        
        # Random initialization (simplified)
        h_cryst = torch.randn(n_cryst_atoms, h_mol.shape[-1], device=device)
        x_cryst = torch.rand(n_cryst_atoms, 3, device=device)  # Fractional coords
        cell_cryst = torch.eye(3, device=device) * initial_cell_size
        mask_cryst = torch.ones(n_cryst_atoms, 1, device=device)
        
        # Forward pass (in practice, would be part of diffusion loop)
        h_out, x_out, cell_out = crystal_dynamics(
            h=h_cryst,
            x=x_cryst,
            cell_vectors=cell_cryst,
            context=context,
            node_mask=mask_cryst,
        )
        
        # Convert to Cartesian coordinates
        x_cart = fractional_to_cartesian(x_out.unsqueeze(0), cell_out.unsqueeze(0)).squeeze(0)
        
        # Get atom types
        atom_types = torch.argmax(h_out, dim=-1)
        
        return {
            'positions': x_cart.cpu().numpy(),
            'atom_types': atom_types.cpu().numpy(),
            'cell': cell_out.cpu().numpy(),
            'fractional_coords': x_out.cpu().numpy(),
        }


def save_crystal(crystal, dataset_info, output_path, format='cif'):
    """Save crystal structure to file."""
    # Convert atom types back to atomic numbers
    atom_decoder = dataset_info['atom_decoder']
    atomic_numbers = [atom_decoder[i] for i in crystal['atom_types']]
    
    # Create ASE Atoms object
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=crystal['positions'],
        cell=crystal['cell'],
        pbc=[True, True, True],
    )
    
    # Save in requested format
    if format == 'cif':
        atoms.write(output_path + '.cif')
    elif format == 'xyz':
        atoms.write(output_path + '.xyz')
    elif format == 'ase_db':
        db = connect(output_path + '.db')
        db.write(atoms)
    
    logger.info(f"Saved crystal to {output_path}.{format}")


def main():
    """Main sampling function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    molecule_encoder, crystal_dynamics, dataset_info = load_model(
        args.model_path,
        args.device,
    )
    
    # Get molecule IDs
    db = connect(args.molecule_db_path)
    if args.molecule_ids is not None:
        molecule_ids = [int(x.strip()) for x in args.molecule_ids.split(',')]
    else:
        molecule_ids = list(range(1, len(db) + 1))
    
    logger.info(f"Sampling for {len(molecule_ids)} molecules")
    
    # Sample crystals for each molecule
    for mol_id in molecule_ids:
        logger.info(f"\nProcessing molecule {mol_id}")
        
        try:
            # Load molecule
            molecule = load_molecule(
                args.molecule_db_path,
                mol_id,
                dataset_info,
            )
            
            logger.info(f"Molecule has {molecule['num_atoms']} atoms")
            
            # Generate multiple samples
            for sample_idx in range(args.n_samples_per_molecule):
                crystal = sample_crystal(
                    molecule_encoder,
                    crystal_dynamics,
                    molecule,
                    args.device,
                    Z=args.Z,
                    initial_cell_size=args.initial_cell_size,
                )
                
                # Save crystal
                output_name = f"mol_{mol_id}_sample_{sample_idx}"
                output_path = output_dir / output_name
                
                save_crystal(
                    crystal,
                    dataset_info,
                    str(output_path),
                    format=args.output_format,
                )
            
            logger.info(
                f"Generated {args.n_samples_per_molecule} crystals "
                f"for molecule {mol_id}"
            )
            
        except Exception as e:
            logger.error(f"Error processing molecule {mol_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\nSampling complete!")


if __name__ == '__main__':
    main()
