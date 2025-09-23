import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf

import logging

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        # Determine number of data points - use charges if available, otherwise use num_atoms or positions
        if 'charges' in data and len(data['charges']) > 0:
            self.data_length = len(data['charges'])
        elif 'num_atoms' in data:
            self.data_length = len(data['num_atoms'])
        elif 'positions' in data:
            self.data_length = len(data['positions'])
        else:
            self.data_length = 0

        if num_pts < 0:
            self.num_pts = self.data_length
        else:
            if num_pts > self.data_length:
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, self.data_length))
                self.num_pts = self.data_length
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if len(included_species) > 0 and included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        # Handle empty datasets gracefully
        if len(included_species) > 0:
            # Use atomic_numbers for one_hot encoding if available, otherwise use charges
            if 'atomic_numbers' in self.data:
                source_tensor = self.data['atomic_numbers']
            else:
                source_tensor = self.data['charges']
            self.data['one_hot'] = source_tensor.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
            self.max_charge = max(included_species)
        else:
            # For empty datasets, create an empty one_hot tensor with correct shape
            # The shape should be [num_molecules, max_atoms, 0] to match positions dimension
            if 'positions' in self.data:
                num_molecules = self.data['positions'].size(0)
                max_atoms = self.data['positions'].size(1)
                self.data['one_hot'] = torch.empty(num_molecules, max_atoms, 0, dtype=torch.bool)
            else:
                self.data['one_hot'] = torch.empty(self.num_pts, 0, dtype=torch.bool)
            self.max_charge = 0

        self.num_species = len(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle and self.num_pts > 0:
            self.perm = torch.randperm(self.data_length)[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        result = {}
        for key, val in self.data.items():
            if key.startswith('_'):
                # Metadata keys - don't index, return as-is
                result[key] = val
            else:
                # Regular data - index with idx
                # Handle empty tensors (like charges when include_charges=False)
                if isinstance(val, torch.Tensor) and val.numel() == 0 and val.dim() <= 1:
                    # Only skip indexing for 0D or 1D empty tensors (global properties)
                    result[key] = val  # Keep empty tensor as-is
                else:
                    # Index all other tensors, including empty multi-dimensional ones like one_hot
                    result[key] = val[idx]
        return result
