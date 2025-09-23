import torch


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props):
        return props
    elif props.dim() == 0:
        return props
    elif props.dim() == 1:
        # 1D tensors (global properties) don't need atom masking
        return props
    elif props.dim() >= 2 and props.size(1) != to_keep.size(0):
        # Tensors that don't match the number of atoms (e.g., global features, empty one_hot) 
        # don't need atom masking
        return props
    else:
        # Apply masking to position/charge-like tensors
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        # Separate metadata keys (starting with '_') from data keys
        data_keys = [key for key in batch[0].keys() if not key.startswith('_')]
        metadata_keys = [key for key in batch[0].keys() if key.startswith('_')]
        
        # Collate only data keys
        collated_batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in data_keys}
        
        # Add metadata keys without collating (they should be the same across all samples)
        for key in metadata_keys:
            collated_batch[key] = batch[0][key]  # Just take from first sample

        # Handle the case where charges might be empty (when include_charges=False)
        if collated_batch['charges'].numel() > 0:
            to_keep = (collated_batch['charges'].sum(0) > 0)
        else:
            # When charges is empty, determine to_keep based on actual molecule sizes
            batch_size = collated_batch['positions'].size(0)
            max_atoms = collated_batch['positions'].size(1)
            num_atoms_batch = collated_batch['num_atoms']
            
            # Create a mask for atoms that exist in any molecule in the batch
            atom_exists = torch.zeros(max_atoms, dtype=torch.bool)
            for n_atoms in num_atoms_batch:
                if n_atoms > 0:
                    atom_exists[:n_atoms] = True
            to_keep = atom_exists

        # Apply drop_zeros only to data keys, not metadata
        for key in data_keys:
            collated_batch[key] = drop_zeros(collated_batch[key], to_keep)

        if collated_batch['charges'].numel() > 0:
            atom_mask = collated_batch['charges'] > 0
        else:
            # When charges is empty, create atom mask based on positions and num_atoms
            batch_size = collated_batch['positions'].size(0)
            n_nodes = collated_batch['positions'].size(1)
            atom_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool)
            for i, n_atoms in enumerate(collated_batch['num_atoms']):
                atom_mask[i, :n_atoms] = True
        
        collated_batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        collated_batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            collated_batch['charges'] = collated_batch['charges'].unsqueeze(2)
        else:
            collated_batch['charges'] = torch.zeros(0)
        return collated_batch
