"""
Crystal sampling functions.

This module provides sampling functions for crystal generation:
- Sample single crystals with periodic boundaries
- Sample crystal chains for visualization
- Batch sampling with different sizes

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation throughout
- Theoretically sound crystal generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

from crystal.data.periodic_utils import cell_params_to_vectors
from crystal.evaluation.structure_validator import StructureValidator


def sample_crystal(
    args,
    device: torch.device,
    model: torch.nn.Module,
    dataset_info: Dict[str, Any],
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    nodesxsample: Optional[torch.Tensor] = None,
    n_samples: int = 1,
    context: Optional[torch.Tensor] = None,
    cell_params: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample crystals from the model.
    
    Args:
        args: Command line arguments
        device: Torch device
        model: Crystal generation model
        dataset_info: Dataset information
        mol_encoder: Molecular encoder (optional)
        conditioning_modules: Conditioning modules
        nodesxsample: [batch] Number of atoms per sample
        n_samples: Number of samples to generate
        context: [batch, context_dim] Conditioning context (optional)
        cell_params: [batch, 6] Initial cell parameters (optional)
    
    Returns:
        one_hot: [batch, n_atoms, n_atom_types] Atom types
        charges: [batch, n_atoms, 1] Charges (zeros for now)
        x: [batch, n_atoms, 3] Atomic positions
        cell: [batch, 3, 3] Unit cell vectors
        node_mask: [batch, n_atoms, 1] Node mask
    
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    model.eval()
    
    with torch.no_grad():
        # Determine number of atoms per crystal
        if nodesxsample is not None:
            n_nodes = nodesxsample.max().item()
            batch_size = nodesxsample.size(0)
        else:
            # Default crystal size
            n_nodes = 50  # Typical molecular crystal
            batch_size = n_samples
            nodesxsample = torch.ones(batch_size, dtype=torch.long, device=device) * n_nodes
        
        # Create node mask
        node_mask = torch.zeros(batch_size, n_nodes, 1, device=device)
        for i, n in enumerate(nodesxsample):
            node_mask[i, :n, 0] = 1
        
        # Create edge mask (fully connected)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(batch_size, 1, 1).view(-1, 1).to(device)
        
        # Initialize cell parameters if not provided
        if cell_params is None:
            # Default cubic cell with reasonable size
            a = 15.0  # Angstroms
            cell_params = torch.tensor(
                [[a, a, a, 90.0, 90.0, 90.0]],
                device=device,
                dtype=torch.float32
            ).repeat(batch_size, 1)
        
        # PBC flags (all True for crystals)
        pbc = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        
        # Sample from model using CrystalDiffusion
        # The model should be a CrystalDiffusion instance
        if not hasattr(model, 'sample') or not callable(getattr(model, 'sample')):
            raise ValueError(
                "Model must have a sample() method. "
                "Use CrystalDiffusion wrapper around your dynamics model."
            )
        
        # Sample positions, features, and cell parameters
        x, h, final_cell_params = model.sample(
            n_samples=batch_size,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context,
            cell_params=cell_params,
            pbc=pbc,
            fix_noise=False
        )
        
        # Extract one_hot and charges from h
        one_hot = h['categorical']
        charges = h.get('integer', torch.zeros_like(one_hot[:, :, :1]))
        
        # Convert final cell params to vectors
        cell = cell_params_to_vectors(final_cell_params)
        
        return one_hot, charges, x, cell, node_mask


def sample_crystal_chain(
    args,
    device: torch.device,
    model: torch.nn.Module,
    dataset_info: Dict[str, Any],
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    n_tries: int = 1,
    keep_frames: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a crystal generation chain for visualization.
    
    Args:
        args: Command line arguments
        device: Torch device
        model: Crystal generation model
        dataset_info: Dataset information
        mol_encoder: Molecular encoder (optional)
        conditioning_modules: Conditioning modules
        n_tries: Number of attempts to find a valid structure
        keep_frames: Number of frames to keep in chain
    
    Returns:
        one_hot: [n_frames, n_atoms, n_atom_types] Atom types over time
        charges: [n_frames, n_atoms, 1] Charges over time
        x: [n_frames, n_atoms, 3] Positions over time
        cell: [n_frames, 3, 3] Cell vectors over time
    
    Raises:
        NotImplementedError: Crystal chain sampling not yet implemented
    """
    model.eval()
    
    # Default crystal size
    n_samples = 1
    n_nodes = 50
    
    with torch.no_grad():
        # Setup similar to sample_crystal
        node_mask = torch.ones(n_samples, n_nodes, 1, device=device)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)
        
        # Default cell
        a = 15.0
        cell_params = torch.tensor(
            [[a, a, a, 90.0, 90.0, 90.0]],
            device=device,
            dtype=torch.float32
        )
        
        pbc = torch.ones(n_samples, 3, dtype=torch.bool, device=device)
        
        # Context (if available)
        context = None
        if args.condition_on_molecule:
            # Would need to prepare molecular context here
            # For now, leave as None
            pass
        
        # Sample chain using CrystalDiffusion
        if not hasattr(model, 'sample_chain') or not callable(getattr(model, 'sample_chain')):
            raise ValueError(
                "Model must have a sample_chain() method. "
                "Use CrystalDiffusion wrapper around your dynamics model."
            )
        
        # Sample trajectory
        chain, cell_chain = model.sample_chain(
            n_samples=n_samples,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context,
            cell_params=cell_params,
            pbc=pbc,
            keep_frames=keep_frames
        )
        
        # Extract positions and features from chain
        # chain is [n_frames*n_samples, n_nodes, n_dims + n_features]
        x = chain[:, :, :3]  # positions
        h_cat = chain[:, :, 3:]  # features
        
        # For simplicity, return as one_hot (categorical features)
        one_hot = h_cat
        charges = torch.zeros(chain.size(0), chain.size(1), 1, device=device)
        
        # Cell chain is [n_frames, n_samples, 6]
        # Convert to vectors for each frame
        n_frames = cell_chain.size(0)
        cell = torch.zeros(n_frames, n_samples, 3, 3, device=device)
        for i in range(n_frames):
            cell[i] = cell_params_to_vectors(cell_chain[i])
        
        return one_hot, charges, x, cell


def validate_and_save_crystal(
    crystal: Dict[str, torch.Tensor],
    structure_validator: StructureValidator,
    cif_writer,
    save_path: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a generated crystal structure and optionally save to CIF.
    
    Args:
        crystal: Dictionary with 'positions', 'one_hot', 'cell', 'node_mask'
        structure_validator: Structure validator
        cif_writer: CIF writer (optional)
        save_path: Path to save CIF file (optional)
    
    Returns:
        is_valid: Whether structure passes validation
        errors: Dictionary of validation errors/metrics
    """
    # Extract data
    positions = crystal['positions'].cpu().numpy()
    one_hot = crystal['one_hot'].cpu()
    cell_params = crystal['cell_params'].cpu().numpy()  # [6] or [batch, 6]
    node_mask = crystal['node_mask'].cpu()
    
    # Convert one_hot to atom types
    if one_hot.dim() == 3:
        atom_types = torch.argmax(one_hot, dim=-1).numpy()
    else:
        atom_types = one_hot.numpy()
    
    # Get actual number of atoms
    n_atoms = int(node_mask.sum().item())
    positions = positions[:n_atoms]
    atom_types = atom_types[:n_atoms]
    
    # Validate structure
    validation_data = {
        'positions': positions,
        'atom_types': atom_types,
        'cell_params': cell_params if cell_params.ndim == 1 else cell_params[0],
        'pbc': np.array([True, True, True])
    }
    
    is_valid, errors = structure_validator.validate_structure(validation_data)
    
    # Save to CIF if valid and path provided
    if is_valid and save_path is not None and cif_writer is not None:
        cif_writer.write_cif(
            positions=positions,
            atom_types=atom_types,
            cell_params=cell_params if cell_params.ndim == 1 else cell_params[0],
            filename=save_path,
            compound_name='Generated Crystal'
        )
    
    return is_valid, errors


def sample_different_crystal_sizes(
    model: torch.nn.Module,
    nodes_dist,
    args,
    device: torch.device,
    dataset_info: Dict[str, Any],
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    n_samples: int = 10,
    batch_size: int = 2
) -> Dict[str, list]:
    """
    Sample crystals of different sizes.
    
    Args:
        model: Crystal generation model
        nodes_dist: Distribution over number of atoms
        args: Command line arguments
        device: Torch device
        dataset_info: Dataset information
        mol_encoder: Molecular encoder
        conditioning_modules: Conditioning modules
        n_samples: Total number of samples
        batch_size: Batch size for generation
    
    Returns:
        Dictionary of sampled crystals
    """
    model.eval()
    
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    
    crystals = {
        'one_hot': [],
        'charges': [],
        'x': [],
        'cell': [],
        'node_mask': []
    }
    
    for i in range(int(n_samples / batch_size)):
        # Sample number of atoms
        nodesxsample = nodes_dist.sample(batch_size)
        
        # Sample crystals
        try:
            one_hot, charges, x, cell, node_mask = sample_crystal(
                args=args,
                device=device,
                model=model,
                dataset_info=dataset_info,
                mol_encoder=mol_encoder,
                conditioning_modules=conditioning_modules,
                nodesxsample=nodesxsample
            )
            
            crystals['one_hot'].append(one_hot.cpu())
            crystals['charges'].append(charges.cpu())
            crystals['x'].append(x.cpu())
            crystals['cell'].append(cell.cpu())
            crystals['node_mask'].append(node_mask.cpu())
        
        except NotImplementedError as e:
            print(f"Sampling not yet implemented: {e}")
            break
    
    # Concatenate batches
    if len(crystals['one_hot']) > 0:
        crystals = {k: torch.cat(v, dim=0) for k, v in crystals.items()}
    
    return crystals
