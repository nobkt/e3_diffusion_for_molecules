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
        # CRITICAL FIX: For ASE database datasets, use dataset-specific max nodes but cap for stability
        # The problem statement shows max 152 atoms per molecule, but we need to be practical
        max_nodes_from_dataset = dataset_info.get('max_n_nodes', 152)
        # Use a reasonable size for generation that's larger than QM9 but not too large
        # This balances generation quality with computational stability
        if max_nodes_from_dataset > 100:
            n_nodes = 50  # Use 50 for large databases (better than 19, manageable size)
        else:
            n_nodes = min(max_nodes_from_dataset, 30)  # Cap at 30 for smaller databases
        print(f"Using n_nodes={n_nodes} for ASE database (dataset max: {max_nodes_from_dataset})")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are: qm9, qm9_second_half, qm9_first_half, geom, ase_db")

    # Handle context creation with proper dimensions
    if args.context_node_nf > 0:
        # Create context tensor with correct dimensions for all conditioning features
        # This ensures compatibility with both scalar and multi-dimensional features
        context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
        
        # CRITICAL FIX: Initialize context with realistic distributions to prevent generation bias
        # This is crucial for proper molecule generation instead of biased atom types
        feature_start_idx = 0
        
        # Handle each conditioning feature properly
        for feat in args.conditioning:
            if feat == 'molecular_weight':
                # Use a reasonable molecular weight range (log-normalized)
                if prop_dist is not None and hasattr(prop_dist, 'sample'):
                    mw_sample = prop_dist.sample(n_samples)
                    context[:, :, feature_start_idx] = mw_sample.unsqueeze(1).repeat(1, n_nodes)
                else:
                    # Default to normalized mean (around 0)
                    context[:, :, feature_start_idx] = 0.0
                feature_start_idx += 1
                
            elif feat == 'pi_conjugation_ratio':
                # Use reasonable pi conjugation ratio
                if prop_dist is not None and hasattr(prop_dist, 'sample'):
                    pi_sample = prop_dist.sample(n_samples)
                    context[:, :, feature_start_idx] = pi_sample.unsqueeze(1).repeat(1, n_nodes)
                else:
                    # Default to small positive value for pi conjugation
                    context[:, :, feature_start_idx] = 0.1
                feature_start_idx += 1
                
            elif feat == 'atom_types_encoding':
                # CRITICAL FIX: Instead of zeros, use realistic atom type distribution
                # This prevents bias toward specific atoms (P, S, Br)
                n_atom_types = 11  # H, C, N, O, F, Si, P, S, Cl, Br, I
                
                # Create a realistic atom type distribution (normalized for training)
                # Based on typical organic molecule distributions: H (~45%), C (~40%), others
                realistic_dist = torch.tensor([
                    0.45, 0.40, 0.08, 0.05, 0.01,  # H, C, N, O, F
                    0.001, 0.001, 0.001, 0.001, 0.001, 0.001  # Si, P, S, Cl, Br, I
                ])
                
                # Apply normalization (mean should be close to 0 after training normalization)
                # Use small values around 0 to represent the normalized distribution
                normalized_dist = (realistic_dist - realistic_dist.mean()) * 0.1
                
                # Fill the context with this distribution for each node
                for i in range(n_atom_types):
                    if feature_start_idx + i < args.context_node_nf:
                        context[:, :, feature_start_idx + i] = normalized_dist[i]
                feature_start_idx += n_atom_types
                
            elif feat == 'functional_groups_encoding':
                # Use balanced functional group distribution
                n_functional_groups = 10  # Estimated number of functional groups
                
                # Small balanced values to prevent bias
                for i in range(n_functional_groups):
                    if feature_start_idx + i < args.context_node_nf:
                        context[:, :, feature_start_idx + i] = 0.05  # Small balanced value
                feature_start_idx += n_functional_groups
                
            else:
                # Handle other features with default values
                context[:, :, feature_start_idx] = 0.0
                feature_start_idx += 1
                    
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
                scalar_dims = scalar_context.size(1)
                if scalar_dims <= args.context_node_nf:
                    context[:, :, :scalar_dims] = scalar_context.unsqueeze(1).repeat(1, max_n_nodes, 1)
                
            # CRITICAL FIX: Initialize context with realistic distributions to prevent generation bias
            # This is crucial for proper molecule generation instead of biased atom types
            feature_start_idx = 0
            
            # Handle each conditioning feature properly to prevent atom type bias
            if hasattr(args, 'conditioning'):
                for feat in args.conditioning:
                    if feat == 'molecular_weight':
                        # Use proper molecular weight distribution
                        if prop_dist is not None and hasattr(prop_dist, 'sample_batch'):
                            mw_sample = prop_dist.sample_batch(nodesxsample)
                            if mw_sample.size(1) > feature_start_idx and feature_start_idx < args.context_node_nf:
                                context[:, :, feature_start_idx] = mw_sample[:, feature_start_idx].unsqueeze(1).repeat(1, max_n_nodes)
                        else:
                            # Default to normalized mean (around 0)
                            context[:, :, feature_start_idx] = 0.0
                        feature_start_idx += 1
                        
                    elif feat == 'pi_conjugation_ratio':
                        # Use reasonable pi conjugation ratio
                        if prop_dist is not None and hasattr(prop_dist, 'sample_batch'):
                            pi_sample = prop_dist.sample_batch(nodesxsample)
                            if pi_sample.size(1) > feature_start_idx and feature_start_idx < args.context_node_nf:
                                context[:, :, feature_start_idx] = pi_sample[:, feature_start_idx].unsqueeze(1).repeat(1, max_n_nodes)
                        else:
                            # Default to small positive value for pi conjugation
                            context[:, :, feature_start_idx] = 0.1
                        feature_start_idx += 1
                        
                    elif feat == 'atom_types_encoding':
                        # CRITICAL FIX: Instead of zeros, use realistic atom type distribution
                        # This prevents bias toward specific atoms (P, S, Br)
                        n_atom_types = 11  # H, C, N, O, F, Si, P, S, Cl, Br, I
                        
                        # Create a realistic atom type distribution (normalized for training)
                        # Based on typical organic molecule distributions: H (~45%), C (~40%), others
                        realistic_dist = torch.tensor([
                            0.45, 0.40, 0.08, 0.05, 0.01,  # H, C, N, O, F
                            0.001, 0.001, 0.001, 0.001, 0.001, 0.001  # Si, P, S, Cl, Br, I
                        ]).to(device)
                        
                        # Apply normalization (mean should be close to 0 after training normalization)
                        # Use small values around 0 to represent the normalized distribution
                        normalized_dist = (realistic_dist - realistic_dist.mean()) * 0.1
                        
                        # Fill the context with this distribution for each node
                        for i in range(n_atom_types):
                            if feature_start_idx + i < args.context_node_nf:
                                context[:, :, feature_start_idx + i] = normalized_dist[i]
                        feature_start_idx += n_atom_types
                        
                    elif feat == 'functional_groups_encoding':
                        # Use balanced functional group distribution
                        n_functional_groups = 10  # Estimated number of functional groups
                        
                        # Small balanced values to prevent bias
                        for i in range(n_functional_groups):
                            if feature_start_idx + i < args.context_node_nf:
                                context[:, :, feature_start_idx + i] = 0.05  # Small balanced value
                        feature_start_idx += n_functional_groups
                        
                    else:
                        # Handle other features with default values
                        if feature_start_idx < args.context_node_nf:
                            context[:, :, feature_start_idx] = 0.0
                        feature_start_idx += 1
            
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

        # CRITICAL FIX: Ensure unused nodes have no atoms (prevent Br at (0,0,0))
        # This addresses the issue where unused positions get filled with default atoms
        one_hot = one_hot * node_mask  # Zero out one_hot for unused nodes
        x = x * node_mask  # Zero out coordinates for unused nodes  
        if args.include_charges:
            charges = charges * node_mask.squeeze(-1).long()  # Zero out charges for unused nodes

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
    
    # CRITICAL FIX: Instead of filtering out binary features, handle them properly to prevent bias
    # The original approach was creating context mismatches during training vs sampling
    
    print(f"Creating context for conditional generation with {len(args.conditioning)} features")
    
    # Handle each conditioning feature in the exact same order as training
    for i, key in enumerate(args.conditioning):
        if key in ['atom_types_encoding', 'functional_groups_encoding']:
            # For problematic binary features, use statistical distribution instead of zeros
            if prop_dist is not None and hasattr(prop_dist, 'normalizer') and key in prop_dist.normalizer:
                mean = prop_dist.normalizer[key]['mean']
                if hasattr(mean, 'shape') and len(mean.shape) > 0:
                    n_features = mean.shape[0] if len(mean.shape) == 1 else mean.numel()
                    # CRITICAL FIX: Use realistic statistical mean distribution
                    # Instead of pure zeros or uniform bias, use the actual data distribution
                    context_row = torch.zeros(n_frames, n_features)
                    # Add slight variation around the mean to create realistic sampling
                    context_row += torch.mean(mean).item() * torch.ones(n_frames, n_features)
                    # Add small random variation to prevent identical contexts
                    context_row += torch.randn(n_frames, n_features) * 0.01
                else:
                    # Single feature case
                    context_row = torch.zeros(n_frames, 1)
                    context_row += 0.1  # Small positive baseline instead of zero
                context.append(context_row)
                print(f"  {key}: using statistical distribution (shape: {context_row.shape})")
            else:
                # Fallback: avoid pure zeros which can cause halogen bias
                context_row = torch.ones(n_frames, 1) * 0.1  # Small positive value instead of zero
                context.append(context_row)
                print(f"  {key}: using fallback non-zero values (shape: {context_row.shape})")
        elif prop_dist is not None and key in prop_dist.distributions:
            # Scalar property with distribution - create sweep
            min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
            mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
            min_val = (min_val - mean) / (mad)
            max_val = (max_val - mean) / (mad)
            context_row = torch.from_numpy(np.linspace(float(min_val), float(max_val), n_frames).astype(np.float32)).unsqueeze(1)
            context.append(context_row)
            print(f"  {key}: using sweep from {min_val:.3f} to {max_val:.3f} (shape: {context_row.shape})")
        else:
            # Multi-dimensional or non-scalar property - use mean/default values
            if prop_dist is not None and hasattr(prop_dist, 'normalizer') and key in prop_dist.normalizer:
                # Use normalized mean (zero after normalization)
                mean = prop_dist.normalizer[key]['mean']
                # CRITICAL FIX: Handle both tensor and scalar mean values properly
                if isinstance(mean, (int, float)) or (hasattr(mean, 'dim') and mean.dim() == 0):
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
                print(f"  {key}: using normalized mean (shape: {context_row.shape})")
            else:
                # No normalization info available - assume single feature with zero
                context_row = torch.zeros(n_frames, 1)
                context.append(context_row)
                print(f"  {key}: using zero default (shape: {context_row.shape})")
    
    context = torch.cat(context, dim=1).float().to(device)
    print(f"Final context tensor shape: {context.shape}")

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask