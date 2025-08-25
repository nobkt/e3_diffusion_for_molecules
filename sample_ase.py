#!/usr/bin/env python3
"""
Sample generation script for EDM models trained on ASE databases.
This script provides the same functionality as eval_sample.py but for ASE datasets.
"""

import argparse
import torch
import pickle
import os
import time
import numpy as np
from os.path import join

# Import E3 Diffusion components
from qm9.models import get_model
from qm9.sampling import sample
from qm9.ase_dataset import get_ase_dataset_info
from qm9 import visualizer as vis
from equivariant_diffusion import utils as flow_utils


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Generate molecules from trained ASE model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--model_name', type=str, default='generative_model_ema.npy',
                        help='Name of model file to load')
    
    # Sampling parameters
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for generation')
    parser.add_argument('--n_tries', type=int, default=1,
                        help='Number of tries for each sample')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated molecules (default: model_path/generated)')
    parser.add_argument('--save_xyz', action='store_true',
                        help='Save molecules as XYZ files')
    parser.add_argument('--save_tensors', action='store_true', default=True,
                        help='Save molecules as PyTorch tensors')
    
    # Device parameters
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Visualization parameters
    parser.add_argument('--visualize_examples', type=int, default=0,
                        help='Number of example molecules to visualize (0 to disable)')
    
    return parser.parse_args()


def setup_device(args):
    """Setup device."""
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def load_model_and_args(args, device):
    """Load trained model and its arguments."""
    print(f"Loading model from: {args.model_path}")
    
    # Load model arguments
    args_path = join(args.model_path, 'args.pickle')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Model arguments not found at {args_path}")
    
    with open(args_path, 'rb') as f:
        model_args = pickle.load(f)
    
    print(f"Model configuration:")
    print(f"  Model type: {model_args.model}")
    print(f"  Features: {model_args.nf}")
    print(f"  Layers: {model_args.n_layers}")
    print(f"  Database: {model_args.ase_db_path}")
    
    # Load dataset info from the ASE database
    if not os.path.exists(model_args.ase_db_path):
        raise FileNotFoundError(f"ASE database not found at {model_args.ase_db_path}")
    
    dataset_info = get_ase_dataset_info(
        model_args.ase_db_path, 
        max_atoms=getattr(model_args, 'max_atoms', 100),
        name='ase_sampling'
    )
    
    # Initialize model
    model, _ = get_model(model_args, device, dataset_info, None)
    
    # Load model weights
    model_path = join(args.model_path, args.model_name)
    if not os.path.exists(model_path):
        # Try alternative names
        alternatives = ['generative_model.npy', 'generative_model_ema.npy']
        for alt in alternatives:
            alt_path = join(args.model_path, alt)
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative model file: {alt}")
                break
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"Model loaded successfully from: {model_path}")
    
    return model, model_args, dataset_info


def setup_sampling(model_args, dataset_info):
    """Setup sampling components."""
    # Initialize node distribution
    nodes_dist = flow_utils.DistributionNodes(dataset_info['n_nodes'])
    
    # Property distribution (not used for ASE datasets currently)
    prop_dist = None
    
    return nodes_dist, prop_dist


def setup_output_dir(args):
    """Setup output directory."""
    if args.output_dir is None:
        output_dir = join(args.model_path, 'generated')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    return output_dir


def generate_molecules(args, model, model_args, device, dataset_info, nodes_dist, prop_dist):
    """Generate molecules using the trained model."""
    print(f"Generating {args.n_samples} molecules...")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize storage
    generated_molecules = {
        'one_hot': [],
        'x': [],
        'node_mask': [],
        'charges': []
    }
    
    # Calculate number of batches
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    # Generation loop
    start_time = time.time()
    total_generated = 0
    
    for batch_idx in range(n_batches):
        current_batch_size = min(args.batch_size, args.n_samples - batch_idx * args.batch_size)
        
        # Sample node counts
        nodesxsample = nodes_dist.sample(current_batch_size)
        
        print(f"Generating batch {batch_idx + 1}/{n_batches} ({current_batch_size} molecules)...")
        
        # Generate molecules
        with torch.no_grad():
            one_hot, charges, x, node_mask = sample(
                model_args, device, model, dataset_info,
                prop_dist=prop_dist, nodesxsample=nodesxsample
            )
        
        # Store results
        generated_molecules['one_hot'].append(one_hot.detach().cpu())
        generated_molecules['x'].append(x.detach().cpu())
        generated_molecules['node_mask'].append(node_mask.detach().cpu())
        generated_molecules['charges'].append(charges.detach().cpu())
        
        total_generated += current_batch_size
        
        # Print progress
        elapsed_time = time.time() - start_time
        molecules_per_sec = total_generated / elapsed_time
        print(f"  Progress: {total_generated}/{args.n_samples} ({molecules_per_sec:.1f} mol/s)")
    
    # Concatenate all batches
    for key in generated_molecules:
        generated_molecules[key] = torch.cat(generated_molecules[key], dim=0)
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.1f}s ({args.n_samples/generation_time:.1f} mol/s)")
    
    return generated_molecules


def save_molecules(args, generated_molecules, dataset_info, output_dir):
    """Save generated molecules in various formats."""
    print("Saving generated molecules...")
    
    n_molecules = generated_molecules['one_hot'].shape[0]
    
    # Save as PyTorch tensors
    if args.save_tensors:
        tensor_path = join(output_dir, 'generated_molecules.pt')
        torch.save(generated_molecules, tensor_path)
        print(f"✓ Saved PyTorch tensors to: {tensor_path}")
    
    # Save as XYZ files
    if args.save_xyz:
        print("Saving XYZ files...")
        vis.save_xyz_file(
            output_dir,
            generated_molecules['one_hot'],
            generated_molecules['charges'],
            generated_molecules['x'],
            dataset_info,
            id_from=0,
            name='generated'
        )
        print(f"✓ Saved XYZ files to: {output_dir}/generated_*.xyz")
    
    # Save summary statistics
    summary_path = join(output_dir, 'generation_summary.txt')
    
    # Calculate molecule sizes
    molecule_sizes = []
    atom_counts = {element: 0 for element in dataset_info['atom_decoder']}
    
    for i in range(n_molecules):
        mask = generated_molecules['node_mask'][i]
        one_hot = generated_molecules['one_hot'][i]
        
        n_atoms = int(mask.sum())
        molecule_sizes.append(n_atoms)
        
        if n_atoms > 0:
            atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
            for atom_type in atom_types:
                if atom_type < len(dataset_info['atom_decoder']):
                    atom_counts[dataset_info['atom_decoder'][atom_type]] += 1
    
    # Write summary
    with open(summary_path, 'w') as f:
        f.write(f"Generation Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Generated molecules: {n_molecules}\n")
        f.write(f"Average size: {np.mean(molecule_sizes):.1f} atoms\n")
        f.write(f"Size range: {min(molecule_sizes)} - {max(molecule_sizes)} atoms\n")
        f.write(f"Dataset atom types: {dataset_info['atom_decoder']}\n\n")
        f.write(f"Atom type distribution:\n")
        for element, count in atom_counts.items():
            f.write(f"  {element}: {count} ({count/sum(atom_counts.values())*100:.1f}%)\n")
    
    print(f"✓ Saved summary to: {summary_path}")


def visualize_examples(args, generated_molecules, dataset_info, output_dir):
    """Visualize example molecules."""
    if args.visualize_examples <= 0:
        return
    
    print(f"Creating visualizations for {args.visualize_examples} example molecules...")
    
    n_molecules = generated_molecules['one_hot'].shape[0]
    n_examples = min(args.visualize_examples, n_molecules)
    
    # Create visualization directory
    viz_dir = join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select random examples
    example_indices = np.random.choice(n_molecules, n_examples, replace=False)
    
    for i, mol_idx in enumerate(example_indices):
        one_hot = generated_molecules['one_hot'][mol_idx]
        positions = generated_molecules['x'][mol_idx]
        mask = generated_molecules['node_mask'][mol_idx]
        
        # Get valid atoms
        n_atoms = int(mask.sum())
        
        if n_atoms > 0:
            valid_positions = positions[:n_atoms]
            atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
            
            # Create molecule visualization
            try:
                vis.plot_molecule(
                    positions=valid_positions.numpy(),
                    atom_types=atom_types.numpy(),
                    dataset_info=dataset_info,
                    save_path=join(viz_dir, f'molecule_{mol_idx:04d}.png'),
                    title=f"Generated Molecule {mol_idx} ({n_atoms} atoms)"
                )
            except Exception as e:
                print(f"Warning: Could not visualize molecule {mol_idx}: {e}")
    
    print(f"✓ Visualizations saved to: {viz_dir}")


def print_generation_stats(generated_molecules, dataset_info):
    """Print generation statistics."""
    n_molecules = generated_molecules['one_hot'].shape[0]
    
    # Calculate molecule sizes
    molecule_sizes = []
    for i in range(n_molecules):
        mask = generated_molecules['node_mask'][i]
        n_atoms = int(mask.sum())
        molecule_sizes.append(n_atoms)
    
    # Calculate atom type distribution
    atom_counts = {element: 0 for element in dataset_info['atom_decoder']}
    for i in range(n_molecules):
        mask = generated_molecules['node_mask'][i]
        one_hot = generated_molecules['one_hot'][i]
        n_atoms = int(mask.sum())
        
        if n_atoms > 0:
            atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
            for atom_type in atom_types:
                if atom_type < len(dataset_info['atom_decoder']):
                    atom_counts[dataset_info['atom_decoder'][atom_type]] += 1
    
    print(f"\nGeneration Statistics:")
    print(f"======================")
    print(f"Total molecules: {n_molecules}")
    print(f"Average size: {np.mean(molecule_sizes):.1f} atoms")
    print(f"Size range: {min(molecule_sizes)} - {max(molecule_sizes)} atoms")
    print(f"Size distribution:")
    
    # Size distribution
    from collections import Counter
    size_counts = Counter(molecule_sizes)
    for size in sorted(size_counts.keys())[:10]:  # Show top 10
        count = size_counts[size]
        print(f"  {size} atoms: {count} molecules ({count/n_molecules*100:.1f}%)")
    
    print(f"\nAtom type distribution:")
    total_atoms = sum(atom_counts.values())
    for element, count in atom_counts.items():
        if count > 0:
            print(f"  {element}: {count} atoms ({count/total_atoms*100:.1f}%)")


def main():
    """Main function."""
    # Setup
    args = setup_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = setup_device(args)
    
    # Load model and dataset info
    model, model_args, dataset_info = load_model_and_args(args, device)
    
    # Setup sampling components
    nodes_dist, prop_dist = setup_sampling(model_args, dataset_info)
    
    # Setup output directory
    output_dir = setup_output_dir(args)
    
    # Generate molecules
    generated_molecules = generate_molecules(
        args, model, model_args, device, dataset_info, nodes_dist, prop_dist
    )
    
    # Print statistics
    print_generation_stats(generated_molecules, dataset_info)
    
    # Save molecules
    save_molecules(args, generated_molecules, dataset_info, output_dir)
    
    # Create visualizations
    visualize_examples(args, generated_molecules, dataset_info, output_dir)
    
    print(f"\nGeneration completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated {args.n_samples} molecules")


if __name__ == "__main__":
    main()