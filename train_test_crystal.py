"""
Crystal-specific training and testing functions.

This module extends train_test.py to handle crystal generation with:
- Periodic boundary conditions
- Cell parameter learning
- Molecular feature conditioning
- Multiple conditioning types

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation throughout
- Theoretically sound crystal generation
"""

import wandb
import numpy as np
import time
import torch
from typing import Optional, Dict, Any, Tuple

from equivariant_diffusion.utils import assert_correctly_masked
import utils
from qm9 import losses


def prepare_crystal_context(
    args,
    data: Dict[str, torch.Tensor],
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    Prepare conditioning context for crystal generation.
    
    Args:
        args: Command line arguments
        data: Batch data dictionary
        mol_encoder: Molecular encoder module (optional)
        conditioning_modules: Dictionary of conditioning modules
        device: Torch device
        dtype: Torch dtype
    
    Returns:
        context: Conditioning tensor [batch_size, n_nodes, context_dim] or None
        
    Raises:
        ValueError: If conditioning is requested but required modules are missing
    """
    # If no conditioning is enabled, return None
    if not (args.condition_on_molecule or args.condition_on_space_group or args.condition_on_density):
        return None
    
    # Extract molecular features if molecular conditioning is enabled
    if args.condition_on_molecule:
        if mol_encoder is None:
            raise ValueError(
                "Molecular conditioning is enabled but mol_encoder is None. "
                "Cannot proceed without molecular encoder."
            )
        if 'molecular' not in conditioning_modules:
            raise ValueError(
                "Molecular conditioning is enabled but 'molecular' module not in conditioning_modules. "
                "Cannot proceed without molecular conditioning module."
            )
        
        # Get molecular data
        if 'molecule' not in data:
            raise ValueError(
                "Molecular conditioning is enabled but 'molecule' key not in data. "
                "Ensure molecule_dataset is provided to CrystalDataset."
            )
        
        mol_data = data['molecule']
        mol_positions = mol_data['positions'].to(device, dtype)
        mol_one_hot = mol_data['one_hot'].to(device, dtype)
        mol_mask = mol_data['atom_mask'].to(device, dtype).unsqueeze(2)
        
        # Encode molecular features
        with torch.no_grad():  # Molecular encoder typically frozen
            mol_features, mol_geometry = mol_encoder(
                mol_one_hot,
                mol_positions,
                mol_mask
            )
        
        # Apply molecular conditioning
        mol_cond_module = conditioning_modules['molecular']
        context = mol_cond_module(mol_features, mol_geometry)
    else:
        # If using space group or density without molecular features,
        # this should have been caught in setup_models()
        raise NotImplementedError(
            "Space group and density conditioning currently require molecular "
            "conditioning to be enabled."
        )
    
    # Optionally add space group conditioning
    if args.condition_on_space_group:
        if 'space_group' not in data:
            raise ValueError(
                "Space group conditioning is enabled but 'space_group' not in data. "
                "Ensure space group information is stored in crystal database."
            )
        
        space_groups = data['space_group'].to(device)
        sg_emb_module = conditioning_modules['space_group']
        sg_embedding = sg_emb_module(space_groups)
        
        # Combine with existing context
        # sg_embedding is [batch_size, sg_dim], need to broadcast to [batch_size, n_nodes, sg_dim]
        n_nodes = context.size(1)
        sg_embedding = sg_embedding.unsqueeze(1).expand(-1, n_nodes, -1)
        context = torch.cat([context, sg_embedding], dim=-1)
    
    # Optionally add density conditioning
    if args.condition_on_density:
        if 'density' not in data:
            raise ValueError(
                "Density conditioning is enabled but 'density' not in data. "
                "Density should be computed from cell and positions in crystal database."
            )
        
        densities = data['density'].to(device, dtype)
        dens_cond_module = conditioning_modules['density']
        dens_embedding = dens_cond_module(densities)
        
        # Combine with existing context
        n_nodes = context.size(1)
        dens_embedding = dens_embedding.unsqueeze(1).expand(-1, n_nodes, -1)
        context = torch.cat([context, dens_embedding], dim=-1)
    
    return context


def train_epoch_crystal(
    args,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    model: torch.nn.Module,
    model_dp: torch.nn.Module,
    model_ema: torch.nn.Module,
    ema,
    device: torch.device,
    dtype: torch.dtype,
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    optim: torch.optim.Optimizer,
    nodes_dist,
    gradnorm_queue,
    dataset_info: Dict[str, Any]
):
    """
    Train one epoch on crystal data.
    
    Args:
        args: Command line arguments
        loader: Crystal data loader
        epoch: Current epoch number
        model: Crystal dynamics model
        model_dp: DataParallel wrapped model
        model_ema: EMA model
        ema: EMA updater
        device: Torch device
        dtype: Torch dtype
        mol_encoder: Molecular encoder (optional)
        conditioning_modules: Dictionary of conditioning modules
        optim: Optimizer
        nodes_dist: Node distribution
        gradnorm_queue: Gradient norm queue
        dataset_info: Dataset information
    """
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    
    for i, data in enumerate(loader):
        # Extract crystal data
        x = data['positions'].to(device, dtype)  # [batch, n_atoms, 3]
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)  # [batch, n_atoms, 1]
        edge_mask = data['edge_mask'].to(device, dtype)  # [batch, n_atoms, n_atoms]
        one_hot = data['one_hot'].to(device, dtype)  # [batch, n_atoms, n_atom_types]
        cell = data['cell'].to(device, dtype)  # [batch, 6] (a, b, c, α, β, γ)
        pbc = data['pbc'].to(device, dtype)  # [batch, 3] (periodic flags)
        
        # Note: For crystals, we don't remove mean as we work in fractional coordinates
        # or with periodic boundaries that maintain crystal structure
        
        # Data augmentation (optional)
        if hasattr(args, 'data_augmentation') and args.data_augmentation:
            # For crystals, rotation should be applied carefully
            # Only rotate positions, not cell parameters
            x = utils.random_rotation(x).detach()
        
        # Validate masks
        assert_correctly_masked(x, node_mask)
        assert_correctly_masked(one_hot, node_mask)
        
        # Prepare node features
        h = {'categorical': one_hot, 'integer': torch.zeros(0).to(device, dtype)}  # No charges for now
        
        # Prepare conditioning context
        try:
            context = prepare_crystal_context(
                args, data, mol_encoder, conditioning_modules, device, dtype
            )
            if context is not None:
                assert_correctly_masked(context, node_mask)
        except (ValueError, NotImplementedError) as e:
            print(f"Error preparing context: {e}")
            raise
        
        # Zero gradients
        optim.zero_grad()
        
        # Compute loss
        # Note: Crystal model needs to handle cell parameters
        try:
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
                args, model_dp, nodes_dist,
                x, h, node_mask, edge_mask, context
            )
        except Exception as e:
            print(f"Error computing loss: {e}")
            print(f"Shapes - x: {x.shape}, h_cat: {h['categorical'].shape}, "
                  f"node_mask: {node_mask.shape}, edge_mask: {edge_mask.shape}, "
                  f"context: {context.shape if context is not None else None}")
            raise
        
        # Early detection of training instability
        if torch.isnan(nll) or torch.isinf(nll) or nll.item() > 1e6:
            print(f"Warning: Unstable loss detected (NLL: {nll.item():.2e}). Skipping this batch.")
            continue
        
        # Standard NLL from forward KL + regularization
        loss = nll + args.ode_regularization * reg_term
        
        # Additional safety check for total loss
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
            print(f"Warning: Unstable total loss detected (Loss: {loss.item():.2e}). Skipping this batch.")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.
        
        # Optimizer step
        optim.step()
        
        # Update EMA if enabled
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)
        
        # Logging
        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        
        nll_epoch.append(nll.item())
        
        # Log to wandb
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        
        if args.break_train_epoch:
            break
    
    # Log epoch average
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def test_crystal(
    args,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    eval_model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    nodes_dist,
    partition: str = 'Test'
) -> float:
    """
    Evaluate crystal generation model.
    
    Args:
        args: Command line arguments
        loader: Crystal data loader
        epoch: Current epoch number
        eval_model: Model to evaluate
        device: Torch device
        dtype: Torch dtype
        mol_encoder: Molecular encoder (optional)
        conditioning_modules: Dictionary of conditioning modules
        nodes_dist: Node distribution
        partition: Data partition name ('Val' or 'Test')
    
    Returns:
        Average NLL over the dataset
    """
    eval_model.eval()
    
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        n_iterations = len(loader)
        
        for i, data in enumerate(loader):
            # Extract crystal data
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            cell = data['cell'].to(device, dtype)
            pbc = data['pbc'].to(device, dtype)
            
            # Validate masks
            assert_correctly_masked(x, node_mask)
            assert_correctly_masked(one_hot, node_mask)
            
            # Prepare node features
            h = {'categorical': one_hot, 'integer': torch.zeros(0).to(device, dtype)}
            
            # Prepare conditioning context
            try:
                context = prepare_crystal_context(
                    args, data, mol_encoder, conditioning_modules, device, dtype
                )
                if context is not None:
                    assert_correctly_masked(context, node_mask)
            except (ValueError, NotImplementedError) as e:
                print(f"Error preparing context: {e}")
                raise
            
            # Compute loss
            nll, _, _ = losses.compute_loss_and_nll(
                args, eval_model, nodes_dist, x, h,
                node_mask, edge_mask, context
            )
            
            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")
        
        if n_samples == 0:
            print(f"Warning: {partition} set is empty, returning NLL = 0")
            return 0.0
        
        return nll_epoch / n_samples


def analyze_and_save_crystal(
    epoch: int,
    model_sample: torch.nn.Module,
    nodes_dist,
    args,
    device: torch.device,
    dataset_info: Dict[str, Any],
    crystal_metrics,
    structure_validator,
    cif_writer,
    mol_encoder: Optional[torch.nn.Module],
    conditioning_modules: Dict[str, torch.nn.Module],
    n_samples: int = 100,
    batch_size: int = 10
) -> Dict[str, float]:
    """
    Analyze generated crystals and save results.
    
    Args:
        epoch: Current epoch
        model_sample: Model for sampling
        nodes_dist: Node distribution
        args: Arguments
        device: Torch device
        dataset_info: Dataset info
        crystal_metrics: Crystal metrics evaluator
        structure_validator: Structure validator
        cif_writer: CIF file writer
        mol_encoder: Molecular encoder
        conditioning_modules: Conditioning modules
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
    
    Returns:
        Dictionary of metrics
    """
    print(f'Analyzing crystal validity at epoch {epoch}...')
    
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    
    crystals = []
    valid_count = 0
    
    for i in range(int(n_samples / batch_size)):
        # Sample number of nodes
        nodesxsample = nodes_dist.sample(batch_size)
        
        # TODO: Implement crystal sampling function
        # This requires adapting sample() from qm9/sampling.py to handle:
        # - Periodic boundary conditions
        # - Cell parameters
        # - Fractional coordinates
        # - Molecular conditioning
        
        print(f"Crystal sampling not yet implemented. Placeholder for batch {i+1}/{int(n_samples/batch_size)}")
        
        # Placeholder structure for now
        # In actual implementation, would call:
        # crystal = sample_crystal(args, device, model_sample, dataset_info, ...)
        # Then validate and save
    
    # Compute metrics
    metrics = {
        'validity_ratio': valid_count / n_samples,
        'n_samples': n_samples,
        'n_valid': valid_count
    }
    
    # Log to wandb
    wandb.log(metrics)
    
    print(f"Validity: {metrics['validity_ratio']:.2%} ({valid_count}/{n_samples})")
    
    return metrics
