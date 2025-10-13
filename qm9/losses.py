import torch
import torch.nn.functional as F


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z


def compute_loss_and_nll_crystal(
    args,
    generative_model,
    nodes_dist,
    x,
    h,
    node_mask,
    edge_mask,
    context,
    cell_params=None,
    pbc=None
):
    """
    Compute loss for crystal generation including cell parameters.
    
    This extends compute_loss_and_nll to handle cell parameter diffusion
    for crystal structures.
    
    Args:
        args: Command line arguments
        generative_model: Diffusion model (should be CrystalDiffusion)
        nodes_dist: Distribution over number of atoms
        x: [batch, n_nodes, 3] Atomic positions
        h: Dictionary with 'categorical' and 'integer' keys
        node_mask: [batch, n_nodes, 1] Node mask
        edge_mask: [batch, n_nodes, n_nodes, 1] or [batch, n_nodes*n_nodes, 1] Edge mask
        context: [batch, n_nodes, context_dim] Conditioning context (optional)
        cell_params: [batch, 6] Cell parameters (optional)
        pbc: [batch, 3] Periodic boundary flags (optional)
        
    Returns:
        nll: Negative log likelihood
        reg_term: Regularization term
        mean_abs_z: Mean absolute value of z (for monitoring)
    """
    bs, n_nodes, n_dims = x.size()
    
    if args.probabilistic_model != 'diffusion':
        raise ValueError(f"Unsupported probabilistic model: {args.probabilistic_model}")
    
    # Reshape edge mask if needed
    if edge_mask.dim() == 4:
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes, -1)
    if edge_mask.dim() == 3:
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    
    assert_correctly_masked(x, node_mask)
    
    # Check if model is crystal-aware
    is_crystal_model = (
        cell_params is not None and
        hasattr(generative_model, 'learn_lattice') and
        generative_model.learn_lattice
    )
    
    if is_crystal_model:
        # For crystal models, we might need to pass cell and pbc to forward
        # However, the standard forward() signature doesn't include these
        # So we use the standard call for now
        # The model should handle cell internally during training
        nll = generative_model(x, h, node_mask, edge_mask, context)
        
        # TODO: Add cell parameter loss term
        # This would require modifying the forward pass or adding a separate term
        # For now, cell parameters are learned implicitly through the dynamics
    else:
        # Standard molecular diffusion
        nll = generative_model(x, h, node_mask, edge_mask, context)
    
    # Node distribution term
    N = node_mask.squeeze(2).sum(1).long()
    log_pN = nodes_dist.log_prob(N)
    
    assert nll.size() == log_pN.size()
    nll = nll - log_pN
    
    # Average over batch
    nll = nll.mean(0)
    
    # Regularization term
    reg_term = torch.tensor([0.]).to(nll.device)
    
    # Add cell parameter regularization if applicable
    if is_crystal_model and cell_params is not None:
        # Add physical constraints as regularization
        # 1. Cell lengths should be positive
        cell_lengths = cell_params[:, :3]
        length_penalty = F.relu(-cell_lengths + 0.1).mean()  # Penalize lengths < 0.1 Angstrom
        
        # 2. Cell angles should be in valid range (slightly less than 0-180 degrees)
        cell_angles = cell_params[:, 3:]
        angle_penalty = (
            F.relu(-cell_angles + 10.0).mean() +  # Penalize angles < 10 degrees
            F.relu(cell_angles - 170.0).mean()     # Penalize angles > 170 degrees
        )
        
        # Combine penalties with small weight
        cell_reg = 0.01 * (length_penalty + angle_penalty)
        reg_term = reg_term + cell_reg
    
    mean_abs_z = 0.
    
    return nll, reg_term, mean_abs_z

