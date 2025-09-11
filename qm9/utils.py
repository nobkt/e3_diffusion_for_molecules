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
            # Use a more reasonable minimum MAD to prevent extreme normalization
            # Use 1% of the mean absolute value or a minimum of 0.1 for numerical stability
            min_mad = torch.clamp(torch.abs(mean) * 0.01, min=0.1)
            mad = torch.clamp(mad, min=min_mad)
        else:
            # For scalar features, use the original logic
            mean = torch.mean(values)
            ma = torch.abs(values - mean)
            mad = torch.mean(ma)
            # Use a more reasonable minimum MAD to prevent extreme normalization
            # Use 1% of the mean absolute value or a minimum of 0.1 for numerical stability
            min_mad = max(abs(float(mean)) * 0.01, 0.1)
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
            properties = torch.nan_to_num(properties, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clamp extreme values to prevent numerical instability
        properties = torch.clamp(properties, min=-50.0, max=50.0)
        
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

