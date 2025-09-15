import torch


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == 'qm9':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
        return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    elif dataset_name == 'ase_db':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        
        # Handle multi-dimensional features by computing norms per feature dimension
        if values.dim() > 1:
            # For multi-dimensional features, compute mean and mad across the batch dimension (dim=0)
            mean = torch.mean(values, dim=0)
            ma = torch.abs(values - mean.unsqueeze(0))
            mad = torch.mean(ma, dim=0)
            
            # Improved minimum MAD computation for numerical stability
            # Check if this appears to be a binary feature (values are mostly 0 and 1)
            is_binary = torch.all((values >= -0.01) & (values <= 1.01))  # Allow small numerical errors
            unique_vals = torch.unique(values.flatten())
            is_sparse_binary = is_binary and len(unique_vals) <= 3  # Binary + maybe some noise
            
            if is_sparse_binary:
                # For sparse binary features, use a more conservative minimum MAD
                # based on sparsity to prevent extreme normalization
                sparsity = torch.mean(values, dim=0)  # Proportion of 1s (or non-zero values)
                # Use larger minimum for sparser features, smaller for denser features
                min_mad = torch.clamp(0.3 + 0.2 * (1.0 - sparsity), min=0.2, max=0.8)
                print(f"Debug: Binary feature '{property_key}' detected, using adaptive MAD: {torch.mean(min_mad):.3f}")
            else:
                # For continuous features, use more conservative minimum
                min_mad = torch.clamp(torch.abs(mean) * 0.05, min=0.3)
            
            mad = torch.clamp(mad, min=min_mad)
        else:
            # For scalar features, use the original logic with improvements
            mean = torch.mean(values)
            ma = torch.abs(values - mean)
            mad = torch.mean(ma)
            
            # Check if this looks like a binary feature
            unique_vals = torch.unique(values)
            is_binary = len(unique_vals) <= 3 and torch.all((values >= -0.01) & (values <= 1.01))
            
            if is_binary:
                # For binary scalar features, use more conservative minimum
                min_mad = 0.5
                print(f"Debug: Binary scalar feature '{property_key}' detected, using MAD: {min_mad}")
            else:
                # For continuous features, use more conservative minimum
                min_mad = max(abs(float(mean)) * 0.05, 0.3)
            
            mad = torch.max(mad, torch.tensor(min_mad))
        
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)


    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    batch_size, n_nodes, _ = minibatch['positions'].size()
    node_mask = minibatch['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        
        # Handle normalization for both scalar and multi-dimensional features
        mean = property_norms[key]['mean']
        mad = property_norms[key]['mad']
        
        # Apply normalization with proper broadcasting
        if mean.dim() == 0:  # Scalar mean/mad
            properties = (properties - mean) / mad
        else:  # Multi-dimensional mean/mad
            # Ensure proper broadcasting
            if properties.dim() == 2 and mean.dim() == 1:
                # properties: (batch_size, n_features), mean/mad: (n_features,)
                properties = (properties - mean.unsqueeze(0)) / mad.unsqueeze(0)
            else:
                properties = (properties - mean) / mad
        
        # Add numerical stability checks
        if torch.any(torch.isnan(properties)) or torch.any(torch.isinf(properties)):
            print(f"Warning: NaN or Inf detected in property '{key}' after normalization")
            print(f"  Original range: [{torch.min(minibatch[key]):.3f}, {torch.max(minibatch[key]):.3f}]")
            print(f"  Mean: {mean}, MAD: {mad}")
            # Replace NaN/Inf with zeros
            properties = torch.nan_to_num(properties, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # More conservative clamping to prevent numerical instability
        # Use smaller range for better gradient stability
        properties = torch.clamp(properties, min=-5.0, max=5.0)
        
        # Final check and warning for large values that might cause instability
        max_abs_val = torch.max(torch.abs(properties))
        if max_abs_val > 3.0:
            print(f"Warning: Large normalized values in '{key}': max_abs = {max_abs_val:.2f}")
            print(f"  This might cause training instability. Consider feature engineering.")
        
        if len(properties.size()) == 1:
            # Global feature.
            assert properties.size() == (batch_size,)
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2:
            # Could be node feature or global feature
            if properties.size(1) == n_nodes:
                # Node feature with shape (batch_size, n_nodes)
                context_key = properties.unsqueeze(2)
                context_list.append(context_key)
                context_node_nf += 1
            else:
                # Global feature with shape (batch_size, n_features) - broadcast to all nodes
                n_features = properties.size(1)
                reshaped = properties.view(batch_size, 1, n_features).repeat(1, n_nodes, 1)
                context_list.append(reshaped)
                context_node_nf += n_features
        elif len(properties.size()) == 3:
            # Node feature with shape (batch_size, n_nodes, n_features)
            assert properties.size()[:2] == (batch_size, n_nodes)
            context_list.append(properties)
            context_node_nf += properties.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    return context

