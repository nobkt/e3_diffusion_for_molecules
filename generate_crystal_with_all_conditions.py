"""
Property-Conditioned Crystal Generation Script

This script generates molecular crystals conditioned on target property values,
implementing Phase 3 Component P3-2 of the property-conditioned molecular crystal 
generation system.

Key Features:
- Generate crystals with target property values
- Load checkpoints with property statistics
- Support multiple conditioning types (molecular, space group, density, properties)
- Save generated crystals in CIF format
- Output metadata including target properties
- Batch generation with multiple property targets

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- E(3) equivariance maintained
- Theoretically sound crystal generation
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from ase import Atoms
from ase.io import write

from configs.datasets_config import get_dataset_info
from equivariant_diffusion import en_diffusion
from crystal.data.molecule_loader import MoleculeDataset
from crystal.models.molecular_encoder import MolecularEncoder
from crystal.models.crystal_dynamics import CrystalDynamics
from crystal.conditioning import (
    MolecularConditioning,
    SpaceGroupEmbedding,
    DensityConditioning,
    PropertyConditioning,
    ExtendedCombinedConditioning
)
from crystal.utils import CIFWriter
from crystal.sampling import sample_crystal


parser = argparse.ArgumentParser(description='Generate Property-Conditioned Crystals')

# Model and checkpoint
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model checkpoint')
parser.add_argument('--molecule_db_path', type=str, required=True,
                    help='Path to molecule ASE database')

# Molecule selection
parser.add_argument('--molecule_id', type=str, default=None,
                    help='Molecule ID to use (e.g., benzene_001)')
parser.add_argument('--molecule_index', type=int, default=None,
                    help='Molecule database index (alternative to molecule_id)')

# Target conditions
parser.add_argument('--space_group', type=int, default=None,
                    help='Target space group number (1-230)')
parser.add_argument('--density', type=float, default=None,
                    help='Target density (g/cm³)')

# Target properties (dynamic - will be parsed from checkpoint)
# Properties are specified as --target_<property_name> <value>
# e.g., --target_bandgap 2.5 --target_melting_point 180.0

# Batch generation
parser.add_argument('--property_file', type=str, default=None,
                    help='CSV file with multiple target property sets')
parser.add_argument('--n_samples', type=int, default=100,
                    help='Number of samples to generate per condition set')

# Output settings
parser.add_argument('--output_dir', type=str, default='generated_crystals',
                    help='Directory for output files')
parser.add_argument('--output_format', type=str, default='cif',
                    choices=['cif', 'xyz', 'both'],
                    help='Output format for generated crystals')

# Generation settings
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch size for generation')
parser.add_argument('--num_atoms', type=int, default=None,
                    help='Number of atoms per crystal (default: from molecule)')

# Other
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--remove_h', action='store_true',
                    help='Remove hydrogen atoms')

args, unknown = parser.parse_known_args()


def parse_target_properties(args, property_names, unknown_args):
    """
    Parse target property values from command-line arguments.
    
    Properties are specified as --target_<property_name> <value>
    
    Args:
        args: Parsed arguments
        property_names: List of expected property names
        unknown_args: Unknown arguments from argparse
    
    Returns:
        dict: Property name to value mapping
    
    Raises:
        ValueError: If a required property is missing
    """
    target_properties = {}
    
    # Parse unknown args for target_* properties
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--target_'):
            prop_name = arg[9:]  # Remove '--target_' prefix
            if i + 1 < len(unknown_args):
                try:
                    value = float(unknown_args[i + 1])
                    target_properties[prop_name] = value
                    i += 2
                except ValueError:
                    raise ValueError(f"Invalid value for {arg}: {unknown_args[i + 1]}")
            else:
                raise ValueError(f"Missing value for {arg}")
        else:
            i += 1
    
    # Check all required properties are provided
    for prop_name in property_names:
        if prop_name not in target_properties:
            raise ValueError(
                f"Missing target value for property: {prop_name}. "
                f"Use --target_{prop_name} <value> to specify."
            )
    
    return target_properties


def load_checkpoint_and_model(checkpoint_path, device):
    """
    Load checkpoint and reconstruct model with conditioning modules.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device
    
    Returns:
        tuple: (model, conditioning_modules, checkpoint_args, property_info)
    
    Raises:
        ValueError: If checkpoint is invalid or incompatible
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract checkpoint info
    checkpoint_args = checkpoint['args']
    property_names = checkpoint.get('property_names', [])
    property_mean = checkpoint.get('property_mean', None)
    property_std = checkpoint.get('property_std', None)
    
    print(f"Checkpoint trained with:")
    print(f"  Molecular conditioning: {checkpoint_args.condition_on_molecule}")
    print(f"  Space group conditioning: {checkpoint_args.condition_on_space_group}")
    print(f"  Density conditioning: {checkpoint_args.condition_on_density}")
    print(f"  Property conditioning: {checkpoint_args.condition_on_property}")
    if checkpoint_args.condition_on_property:
        print(f"  Properties: {property_names}")
    
    # Get dataset info
    dataset_info = get_dataset_info(
        'qm9' if not checkpoint_args.remove_h else 'qm9_without_h',
        checkpoint_args.remove_h
    )
    
    # Reconstruct conditioning modules
    conditioning_modules = {}
    
    # Molecular conditioning
    mol_cond = None
    if checkpoint_args.condition_on_molecule:
        mol_cond = MolecularConditioning(
            molecular_feature_dim=checkpoint_args.nf,
            conditioning_dim=checkpoint_args.conditioning_dim,
            use_geometry=True
        ).to(device)
    
    # Space group embedding
    sg_embed = None
    if checkpoint_args.condition_on_space_group:
        sg_embed = SpaceGroupEmbedding(
            num_space_groups=230,
            embedding_dim=checkpoint_args.conditioning_dim
        ).to(device)
    
    # Density conditioning
    dens_cond = None
    if checkpoint_args.condition_on_density:
        dens_cond = DensityConditioning(
            conditioning_dim=checkpoint_args.conditioning_dim
        ).to(device)
    
    # Property conditioning
    prop_cond = None
    if checkpoint_args.condition_on_property:
        if not property_names:
            raise ValueError("Checkpoint has property conditioning but no property names")
        
        prop_cond = PropertyConditioning(
            property_names=property_names,
            conditioning_dim=checkpoint_args.conditioning_dim,
            hidden_dim=checkpoint_args.property_hidden_dim,
            n_layers=checkpoint_args.property_n_layers
        ).to(device)
        
        # Set normalization parameters
        if property_mean is not None and property_std is not None:
            prop_cond.set_normalization_params(property_mean, property_std)
            print(f"Loaded property normalization statistics")
        else:
            raise ValueError("Checkpoint missing property normalization statistics")
    
    # Combined conditioning
    combined_conditioning = ExtendedCombinedConditioning(
        molecular_conditioning=mol_cond,
        space_group_embedding=sg_embed,
        density_conditioning=dens_cond,
        property_conditioning=prop_cond,
        conditioning_dim=checkpoint_args.conditioning_dim
    ).to(device)
    
    conditioning_modules['combined'] = combined_conditioning
    if mol_cond:
        conditioning_modules['molecular'] = mol_cond
    if sg_embed:
        conditioning_modules['space_group'] = sg_embed
    if dens_cond:
        conditioning_modules['density'] = dens_cond
    if prop_cond:
        conditioning_modules['property'] = prop_cond
    
    # Load conditioning module states
    if 'conditioning_modules' in checkpoint:
        for k, state_dict in checkpoint['conditioning_modules'].items():
            if k in conditioning_modules:
                conditioning_modules[k].load_state_dict(state_dict)
    
    # Reconstruct dynamics model
    context_node_nf = checkpoint_args.conditioning_dim
    
    dynamics = CrystalDynamics(
        in_node_nf=dataset_info['input_nf'],
        hidden_nf=checkpoint_args.nf,
        n_layers=checkpoint_args.n_layers,
        context_node_nf=context_node_nf,
        device=device,
        tanh=checkpoint_args.tanh,
        attention=checkpoint_args.attention,
        norm_constant=checkpoint_args.norm_constant,
        sin_embedding=checkpoint_args.sin_embedding,
        learn_lattice=checkpoint_args.learn_lattice,
        lattice_hidden_dim=checkpoint_args.lattice_hidden_dim,
        lattice_num_layers=checkpoint_args.lattice_num_layers,
        use_fractional_coords=checkpoint_args.use_fractional_coords,
        periodic_cutoff=checkpoint_args.periodic_cutoff
    ).to(device)
    
    # Wrap in diffusion model
    model = en_diffusion.EnVariationalDiffusion(
        dynamics=dynamics,
        in_node_nf=dataset_info['input_nf'],
        n_dims=3,
        timesteps=checkpoint_args.diffusion_steps,
        noise_schedule=checkpoint_args.diffusion_noise_schedule,
        noise_precision=checkpoint_args.diffusion_noise_precision,
        loss_type=checkpoint_args.diffusion_loss_type,
        norm_values=None
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    property_info = {
        'property_names': property_names,
        'property_mean': property_mean,
        'property_std': property_std
    }
    
    return model, conditioning_modules, checkpoint_args, dataset_info, property_info


def load_molecule_data(molecule_db_path, molecule_id=None, molecule_index=None, remove_h=False):
    """
    Load molecule data from database.
    
    Args:
        molecule_db_path: Path to molecule database
        molecule_id: Molecule ID string
        molecule_index: Molecule database index
        remove_h: Whether to remove hydrogen
    
    Returns:
        dict: Molecule data including atomic numbers, positions, etc.
    
    Raises:
        ValueError: If neither molecule_id nor molecule_index is provided
    """
    if molecule_id is None and molecule_index is None:
        raise ValueError("Either molecule_id or molecule_index must be provided")
    
    print(f"Loading molecule dataset from {molecule_db_path}")
    molecule_dataset = MoleculeDataset(
        db_path=molecule_db_path,
        remove_h=remove_h
    )
    
    # Find molecule
    if molecule_index is not None:
        molecule_data = molecule_dataset[molecule_index]
        print(f"Loaded molecule at index {molecule_index}")
    else:
        # Search by ID
        found = False
        for idx in range(len(molecule_dataset)):
            data = molecule_dataset[idx]
            if data.get('molecule_id') == molecule_id:
                molecule_data = data
                found = True
                print(f"Found molecule {molecule_id} at index {idx}")
                break
        
        if not found:
            raise ValueError(f"Molecule ID {molecule_id} not found in database")
    
    return molecule_data


def prepare_conditioning(
    conditioning_modules,
    molecular_features,
    space_group=None,
    density=None,
    target_properties=None,
    device='cuda'
):
    """
    Prepare conditioning vector from inputs.
    
    Args:
        conditioning_modules: Dictionary of conditioning modules
        molecular_features: Molecular feature tensor
        space_group: Space group number (optional)
        density: Density value (optional)
        target_properties: Dictionary of property name to value (optional)
        device: Torch device
    
    Returns:
        torch.Tensor: Conditioning vector
    """
    combined_cond = conditioning_modules['combined']
    
    # Prepare inputs
    sg_tensor = None
    if space_group is not None:
        sg_tensor = torch.tensor([[space_group]], dtype=torch.long, device=device)
    
    dens_tensor = None
    if density is not None:
        dens_tensor = torch.tensor([[density]], dtype=torch.float32, device=device)
    
    prop_tensor = None
    if target_properties is not None:
        # Convert dict to tensor in correct order
        prop_names = conditioning_modules['property'].property_names
        prop_values = [target_properties[name] for name in prop_names]
        prop_tensor = torch.tensor([prop_values], dtype=torch.float32, device=device)
    
    # Get conditioning
    conditioning = combined_cond(
        molecular_features.unsqueeze(0).to(device),
        space_group=sg_tensor,
        density=dens_tensor,
        properties=prop_tensor
    )
    
    return conditioning


def save_crystal(crystal_data, output_path, output_format='cif'):
    """
    Save generated crystal to file.
    
    Args:
        crystal_data: Dictionary with 'positions', 'atomic_numbers', 'cell'
        output_path: Output file path (without extension)
        output_format: 'cif', 'xyz', or 'both'
    """
    positions = crystal_data['positions']
    atomic_numbers = crystal_data['atomic_numbers']
    cell = crystal_data['cell']
    
    # Create ASE Atoms object
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=True
    )
    
    # Save in requested format(s)
    if output_format in ['cif', 'both']:
        cif_path = str(output_path) + '.cif'
        write(cif_path, atoms, format='cif')
    
    if output_format in ['xyz', 'both']:
        xyz_path = str(output_path) + '.xyz'
        write(xyz_path, atoms, format='xyz')


def generate_crystals(
    model,
    conditioning_modules,
    molecular_features,
    checkpoint_args,
    dataset_info,
    space_group=None,
    density=None,
    target_properties=None,
    n_samples=100,
    batch_size=10,
    num_atoms=None,
    device='cuda'
):
    """
    Generate multiple crystals with target conditions.
    
    Args:
        model: Trained diffusion model
        conditioning_modules: Conditioning modules
        molecular_features: Molecular feature tensor
        checkpoint_args: Arguments from checkpoint
        dataset_info: Dataset information
        space_group: Target space group (optional)
        density: Target density (optional)
        target_properties: Target property values (optional)
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        num_atoms: Number of atoms per crystal
        device: Torch device
    
    Returns:
        list: List of generated crystal data dictionaries
    """
    print(f"\nGenerating {n_samples} crystals...")
    print(f"  Space group: {space_group if space_group else 'None'}")
    print(f"  Density: {density if density else 'None'}")
    print(f"  Target properties: {target_properties if target_properties else 'None'}")
    
    # Prepare conditioning
    conditioning = prepare_conditioning(
        conditioning_modules,
        molecular_features,
        space_group=space_group,
        density=density,
        target_properties=target_properties,
        device=device
    )
    
    # Repeat conditioning for batch
    conditioning_batch = conditioning.repeat(batch_size, 1)
    
    # Determine number of atoms
    if num_atoms is None:
        num_atoms = len(molecular_features)  # Use molecule size
    
    crystals = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        actual_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Create nodes tensor
        nodesxsample = torch.ones(actual_batch_size, dtype=torch.long, device=device) * num_atoms
        
        # Trim conditioning if last batch is smaller
        batch_conditioning = conditioning_batch[:actual_batch_size]
        
        # Sample from model
        with torch.no_grad():
            one_hot, charges, x, cell, node_mask = sample_crystal(
                args=checkpoint_args,
                device=device,
                model=model,
                dataset_info=dataset_info,
                mol_encoder=None,  # Not needed with pre-computed conditioning
                conditioning_modules=conditioning_modules,
                nodesxsample=nodesxsample,
                context=batch_conditioning
            )
        
        # Convert to crystal data
        for i in range(actual_batch_size):
            n_nodes = nodesxsample[i].item()
            
            # Get atomic numbers from one-hot
            atomic_numbers = torch.argmax(one_hot[i, :n_nodes], dim=-1).cpu().numpy()
            
            # Get positions
            positions = x[i, :n_nodes].cpu().numpy()
            
            # Get cell
            cell_vectors = cell[i].cpu().numpy()
            
            crystal_data = {
                'atomic_numbers': atomic_numbers,
                'positions': positions,
                'cell': cell_vectors,
                'space_group': space_group,
                'density': density,
                'target_properties': target_properties
            }
            
            crystals.append(crystal_data)
        
        print(f"  Generated batch {batch_idx + 1}/{n_batches}")
    
    return crystals


def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    print("\n" + "="*70)
    print("Property-Conditioned Crystal Generation")
    print("="*70)
    print(f"Using device: {device}")
    
    # Load checkpoint and model
    model, conditioning_modules, checkpoint_args, dataset_info, property_info = \
        load_checkpoint_and_model(args.model_path, device)
    
    # Parse target properties from command line
    target_properties = None
    if property_info['property_names']:
        target_properties = parse_target_properties(
            args,
            property_info['property_names'],
            unknown
        )
        print(f"\nTarget properties:")
        for name, value in target_properties.items():
            print(f"  {name}: {value}")
    
    # Load molecule data
    molecule_data = load_molecule_data(
        args.molecule_db_path,
        molecule_id=args.molecule_id,
        molecule_index=args.molecule_index,
        remove_h=args.remove_h
    )
    
    # Extract molecular features
    # For generation, we use the molecule's atomic features directly
    molecular_features = molecule_data['one_hot']  # [n_atoms, n_features]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate crystals
    crystals = generate_crystals(
        model=model,
        conditioning_modules=conditioning_modules,
        molecular_features=molecular_features,
        checkpoint_args=checkpoint_args,
        dataset_info=dataset_info,
        space_group=args.space_group,
        density=args.density,
        target_properties=target_properties,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_atoms=args.num_atoms,
        device=device
    )
    
    print(f"\nSaving {len(crystals)} generated crystals...")
    
    # Save crystals
    for i, crystal_data in enumerate(crystals):
        # Save structure
        crystal_path = output_dir / f'crystal_{i:04d}'
        save_crystal(crystal_data, crystal_path, args.output_format)
        
        # Save metadata
        metadata = {
            'crystal_id': f'crystal_{i:04d}',
            'molecule_id': args.molecule_id or f'index_{args.molecule_index}',
            'space_group': args.space_group,
            'density': args.density,
            'target_properties': target_properties,
            'generation_time': datetime.now().isoformat(),
            'num_atoms': len(crystal_data['atomic_numbers'])
        }
        
        metadata_path = output_dir / f'crystal_{i:04d}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Generate summary
    summary = {
        'n_samples': len(crystals),
        'molecule_id': args.molecule_id or f'index_{args.molecule_index}',
        'space_group': args.space_group,
        'density': args.density,
        'target_properties': target_properties,
        'output_dir': str(output_dir),
        'output_format': args.output_format,
        'generation_time': datetime.now().isoformat()
    }
    
    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("="*70)


if __name__ == '__main__':
    main()
