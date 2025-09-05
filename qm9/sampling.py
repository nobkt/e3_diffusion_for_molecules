import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    elif 'ase_db' in args.dataset:
        # For ASE database datasets, use a reasonable default molecule size
        n_nodes = 19  # Same default as QM9 datasets
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are: qm9, qm9_second_half, qm9_first_half, geom, ase_db")

    # Handle context creation with proper dimensions
    if args.context_node_nf > 0:
        # Create context tensor with correct dimensions for all conditioning features
        # This ensures compatibility with both scalar and multi-dimensional features
        context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
        
        # If we have a property distribution for scalar features, use it to fill the context
        if prop_dist is not None:
            scalar_context = prop_dist.sample(n_nodes).unsqueeze(0)
            # Fill the beginning of context with scalar properties
            # Non-scalar features (like atom_types_encoding) remain as zeros during sampling
            scalar_dims = scalar_context.size(1)
            context[:, :, :scalar_dims] = scalar_context.unsqueeze(1).repeat(1, n_nodes, 1)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # Handle context creation with proper dimensions
    if args.context_node_nf > 0:
        if context is None:
            # Create context tensor with correct dimensions for all conditioning features
            # This ensures compatibility with both scalar and multi-dimensional features
            context = torch.zeros(batch_size, max_n_nodes, args.context_node_nf).to(device)
            
            # If we have a property distribution for scalar features, use it to fill the context
            if prop_dist is not None:
                scalar_context = prop_dist.sample_batch(nodesxsample)
                # Fill the beginning of context with scalar properties
                # Non-scalar features (like atom_types_encoding) remain as zeros during sampling
                scalar_dims = scalar_context.size(1)
                context[:, :, :scalar_dims] = scalar_context.unsqueeze(1).repeat(1, max_n_nodes, 1)
            
            # Apply node mask to context
            context = context * node_mask
        else:
            # Use provided context, ensure it has the right shape
            if context.dim() == 2:
                context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device)
            context = context.to(device) * node_mask
    else:
        context = None

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask


def sample_exact_conditional(args, device, generative_model, dataset_info, prop_dist, exact_context, n_nodes=19, n_frames=100):
    """
    Sample molecules with exact conditional values instead of sweeps.
    
    Args:
        args: Model arguments
        device: Device to run on
        generative_model: Trained generative model
        dataset_info: Dataset information
        prop_dist: Property distribution (can be None for exact conditioning)
        exact_context: Pre-computed context tensor with exact conditions
        n_nodes: Number of nodes per molecule
        n_frames: Number of molecules to generate
        
    Returns:
        tuple: (one_hot, charges, x, node_mask) - Generated molecules
    """
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=exact_context, fix_noise=True)
    return one_hot, charges, x, node_mask


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    
    # Handle all conditioning features to match training context shape
    for key in args.conditioning:
        if prop_dist is not None and key in prop_dist.distributions:
            # Scalar property with distribution - create sweep
            min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
            mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
            min_val = (min_val - mean) / (mad)
            max_val = (max_val - mean) / (mad)
            context_row = torch.from_numpy(np.linspace(float(min_val), float(max_val), n_frames).astype(np.float32)).unsqueeze(1)
            context.append(context_row)
        else:
            # Multi-dimensional or non-scalar property - use mean/default values
            if prop_dist is not None and hasattr(prop_dist, 'normalizer') and key in prop_dist.normalizer:
                # Use normalized mean (zero after normalization)
                mean = prop_dist.normalizer[key]['mean']
                if hasattr(mean, 'dim') and mean.dim() == 0:
                    # Scalar mean - create single feature column
                    context_row = torch.zeros(n_frames, 1)
                elif hasattr(mean, 'shape') and len(mean.shape) > 0:
                    # Multi-dimensional mean - create multiple feature columns  
                    n_features = mean.shape[0] if len(mean.shape) == 1 else mean.numel()
                    context_row = torch.zeros(n_frames, n_features)
                else:
                    # Fallback for scalar-like means
                    context_row = torch.zeros(n_frames, 1)
                context.append(context_row)
            else:
                # No normalization info available - assume single feature with zero
                context_row = torch.zeros(n_frames, 1)
                context.append(context_row)
    
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask