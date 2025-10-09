"""
Sample script for generating crystal structures.

This script loads a trained crystal dynamics model and generates new crystal structures.
"""

import argparse
import torch
from pathlib import Path
from ase import Atoms
from ase.io import write

from crystal.models import CrystalDynamics
from crystal.data.periodic_utils import fractional_to_cartesian, wrap_to_unit_cell


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate crystal structures')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--in_node_nf', type=int, default=1,
                        help='Input node feature dimension')
    parser.add_argument('--hidden_nf', type=int, default=128,
                        help='Hidden feature dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of EGNN layers')
    
    # Sampling arguments
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of structures to generate')
    parser.add_argument('--n_atoms', type=int, default=20,
                        help='Number of atoms per structure')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='generated_crystals',
                        help='Directory to save generated structures')
    parser.add_argument('--output_format', type=str, default='cif',
                        choices=['cif', 'xyz', 'both'],
                        help='Output file format')
    
    return parser.parse_args()


def generate_crystals(model, n_samples, n_atoms, device, output_dir, output_format='cif'):
    """Generate crystal structures and save them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    print(f"Generating {n_samples} crystal structures with {n_atoms} atoms each...")
    
    with torch.no_grad():
        for i in range(n_samples):
            # Sample a crystal structure
            h, x_frac, cell_vectors = model.sample(
                n_nodes=n_atoms,
                in_node_nf=1,
                device=device
            )
            
            # Wrap fractional coordinates to unit cell
            x_frac_wrapped = wrap_to_unit_cell(x_frac)
            
            # Convert to Cartesian coordinates
            x_cart = fractional_to_cartesian(x_frac_wrapped, cell_vectors)
            
            # Create ASE Atoms object
            # For simplicity, use atomic number 1 (H) for all atoms
            # In practice, you'd decode the atom types from h
            atoms = Atoms(
                numbers=[1] * n_atoms,  # Hydrogen atoms
                positions=x_cart.cpu().numpy(),
                cell=cell_vectors.cpu().numpy(),
                pbc=[True, True, True]
            )
            
            # Save structure
            if output_format in ['cif', 'both']:
                cif_path = output_dir / f'crystal_{i:03d}.cif'
                write(cif_path, atoms)
                print(f'Saved {cif_path}')
            
            if output_format in ['xyz', 'both']:
                xyz_path = output_dir / f'crystal_{i:03d}.xyz'
                write(xyz_path, atoms)
                print(f'Saved {xyz_path}')
    
    print(f"\nGenerated {n_samples} structures in {output_dir}")


def main():
    """Main sampling function."""
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    print(f"Device: {args.device}")
    
    # Create model
    model = CrystalDynamics(
        in_node_nf=args.in_node_nf,
        hidden_nf=args.hidden_nf,
        out_node_nf=1,
        n_layers=args.n_layers,
        learn_lattice=True
    ).to(args.device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate crystals
    generate_crystals(
        model=model,
        n_samples=args.n_samples,
        n_atoms=args.n_atoms,
        device=args.device,
        output_dir=args.output_dir,
        output_format=args.output_format
    )


if __name__ == '__main__':
    main()
