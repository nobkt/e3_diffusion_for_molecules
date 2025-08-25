"""
ASE database dataset support for e3_diffusion_for_molecules.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging

try:
    from ase.db import connect
    from ase import Atoms
    ase_available = True
except ImportError:
    ase_available = False


class ASEDBDataset(Dataset):
    """
    Dataset class for ASE database files.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
    dataset_info : dict
        Dataset configuration dictionary containing atom encoders/decoders
    split : str
        Dataset split ('train', 'valid', 'test')
    split_ratio : tuple
        Train/valid/test split ratios (default: (0.8, 0.1, 0.1))
    remove_h : bool
        Whether to remove hydrogen atoms (default: False)
    max_atoms : int
        Maximum number of atoms to include (default: None for no limit)
    """
    
    def __init__(self, db_path, dataset_info, split='train', split_ratio=(0.8, 0.1, 0.1), 
                 remove_h=False, max_atoms=None, random_seed=42):
        
        if not ase_available:
            raise ImportError("ASE is required for ASE database support")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ASE database file not found: {db_path}")
        
        self.db_path = db_path
        self.dataset_info = dataset_info
        self.split = split
        self.remove_h = remove_h
        self.max_atoms = max_atoms
        
        # Load and process data
        self.data = self._load_data()
        
        # Split data
        self._split_data(split_ratio, random_seed)
        
        # Create one-hot encoding
        self._create_one_hot()
        
    def _load_data(self):
        """Load data from ASE database."""
        db = connect(self.db_path)
        
        data = {
            'positions': [],
            'charges': [],
            'num_atoms': [],
        }
        
        # Check if properties exist in database
        sample_row = next(db.select())
        available_properties = set(sample_row.key_value_pairs.keys()) if sample_row.key_value_pairs else set()
        
        # Add available properties to data dict
        property_keys = ['energy', 'forces', 'dipole', 'gap', 'homo', 'lumo', 'alpha', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        for prop in property_keys:
            if prop in available_properties:
                data[prop] = []
        
        valid_molecules = []
        
        for i, row in enumerate(db.select()):
            atoms = row.toatoms()
            
            # Skip if too many atoms
            if self.max_atoms and len(atoms) > self.max_atoms:
                continue
                
            # Get positions and atomic numbers
            positions = atoms.positions
            atomic_numbers = atoms.get_atomic_numbers()
            
            # Remove hydrogens if requested
            if self.remove_h:
                mask = atomic_numbers != 1  # hydrogen has atomic number 1
                positions = positions[mask]
                atomic_numbers = atomic_numbers[mask]
            
            # Skip empty molecules
            if len(atomic_numbers) == 0:
                continue
            
            # Convert atomic numbers to charges and check if all atoms are supported
            charges = []
            valid_molecule = True
            
            for atomic_num in atomic_numbers:
                # Convert atomic number to symbol
                symbol = atoms.get_chemical_symbols()[list(atoms.get_atomic_numbers()).index(atomic_num)]
                
                if symbol in self.dataset_info['atom_encoder']:
                    charges.append(atomic_num)
                else:
                    valid_molecule = False
                    break
            
            if not valid_molecule:
                continue
                
            # Store molecular data
            data['positions'].append(torch.tensor(positions, dtype=torch.float32))
            data['charges'].append(torch.tensor(charges, dtype=torch.long))
            data['num_atoms'].append(len(charges))
            
            # Store additional properties if available
            for prop in property_keys:
                if prop in available_properties:
                    prop_value = row.key_value_pairs.get(prop, 0.0)
                    if isinstance(prop_value, (list, np.ndarray)):
                        prop_value = torch.tensor(prop_value, dtype=torch.float32)
                    else:
                        prop_value = torch.tensor([float(prop_value)], dtype=torch.float32)
                    data[prop].append(prop_value)
            
            valid_molecules.append(i)
            
        logging.info(f"Loaded {len(valid_molecules)} valid molecules from ASE database")
        
        # Convert lists to padded tensors
        if len(data['positions']) > 0:
            max_atoms_actual = max(len(pos) for pos in data['positions'])
            
            # Pad positions and charges
            padded_positions = []
            padded_charges = []
            
            for pos, charge in zip(data['positions'], data['charges']):
                n_atoms = len(pos)
                # Pad positions
                padded_pos = torch.zeros(max_atoms_actual, 3)
                padded_pos[:n_atoms] = pos
                padded_positions.append(padded_pos)
                
                # Pad charges
                padded_charge = torch.zeros(max_atoms_actual, dtype=torch.long)
                padded_charge[:n_atoms] = charge
                padded_charges.append(padded_charge)
            
            data['positions'] = torch.stack(padded_positions)
            data['charges'] = torch.stack(padded_charges)
            data['num_atoms'] = torch.tensor(data['num_atoms'], dtype=torch.long)
            
            # Stack other properties
            for prop in property_keys:
                if prop in data and len(data[prop]) > 0:
                    data[prop] = torch.stack(data[prop])
        
        return data
    
    def _split_data(self, split_ratio, random_seed):
        """Split data into train/valid/test sets."""
        n_total = len(self.data['num_atoms'])
        
        # Generate random permutation
        np.random.seed(random_seed)
        indices = np.random.permutation(n_total)
        
        # Calculate split sizes
        n_train = int(split_ratio[0] * n_total)
        n_valid = int(split_ratio[1] * n_total)
        n_test = n_total - n_train - n_valid
        
        # Split indices
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'valid':
            selected_indices = indices[n_train:n_train + n_valid]
        elif self.split == 'test':
            selected_indices = indices[n_train + n_valid:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Filter data for this split
        for key in self.data:
            if isinstance(self.data[key], torch.Tensor):
                self.data[key] = self.data[key][selected_indices]
        
        self.num_pts = len(selected_indices)
        logging.info(f"Split '{self.split}': {self.num_pts} molecules")
    
    def _create_one_hot(self):
        """Create one-hot encoding for atom types."""
        # Get included species
        included_species = torch.tensor(list(self.dataset_info['atom_encoder'].values()), dtype=torch.long)
        self.included_species = included_species
        
        # Create one-hot encoding
        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        
        # Create atom mask (True where atoms exist)
        self.data['atom_mask'] = self.data['charges'] > 0
        
        self.num_species = len(included_species)
        self.max_charge = max(included_species) if len(included_species) > 0 else 0
    
    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}
    
    def convert_units(self, units_dict):
        """Convert units for specified properties."""
        for key in self.data.keys():
            if key in units_dict and key in self.data:
                self.data[key] *= units_dict[key]


def load_ase_db_datasets(db_path, dataset_info, split_ratio=(0.8, 0.1, 0.1), 
                        remove_h=False, max_atoms=None, random_seed=42):
    """
    Load train/valid/test datasets from ASE database.
    
    Returns:
        Dictionary with 'train', 'valid', 'test' datasets
    """
    datasets = {}
    
    for split in ['train', 'valid', 'test']:
        datasets[split] = ASEDBDataset(
            db_path=db_path,
            dataset_info=dataset_info,
            split=split,
            split_ratio=split_ratio,
            remove_h=remove_h,
            max_atoms=max_atoms,
            random_seed=random_seed
        )
    
    return datasets


def convert_qm9_to_ase_db(qm9_data_dir, output_db_path, include_properties=True):
    """
    Convert QM9 dataset to ASE database format.
    
    Args:
        qm9_data_dir: Directory containing QM9 .npz files
        output_db_path: Output path for ASE database
        include_properties: Whether to include molecular properties
    """
    if not ase_available:
        raise ImportError("ASE is required for this function")
    
    from ase.db import connect
    from ase import Atoms
    
    # Create database
    db = connect(output_db_path)
    
    property_keys = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    
    total_molecules = 0
    
    for split in ['train', 'valid', 'test']:
        npz_file = os.path.join(qm9_data_dir, f'{split}.npz')
        
        if not os.path.exists(npz_file):
            logging.warning(f"QM9 file not found: {npz_file}")
            continue
            
        # Load QM9 data
        with np.load(npz_file) as data:
            positions = data['positions']
            charges = data['charges'] 
            num_atoms = data['num_atoms']
            
            properties = {}
            if include_properties:
                for prop in property_keys:
                    if prop in data:
                        properties[prop] = data[prop]
            
            n_molecules = len(num_atoms)
            logging.info(f"Converting {n_molecules} molecules from {split} split")
            
            for i in range(n_molecules):
                n_atoms = int(num_atoms[i])
                
                # Get atomic positions and numbers
                mol_positions = positions[i, :n_atoms]
                mol_charges = charges[i, :n_atoms]
                
                # Skip if no atoms
                if n_atoms == 0:
                    continue
                
                # Convert charges to symbols
                charge_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
                symbols = []
                for charge in mol_charges:
                    charge_int = int(charge)
                    if charge_int in charge_to_symbol:
                        symbols.append(charge_to_symbol[charge_int])
                    else:
                        logging.warning(f"Unknown atomic number: {charge_int}")
                        symbols.append('X')  # Unknown element
                
                # Create ASE Atoms object
                atoms = Atoms(symbols=symbols, positions=mol_positions)
                
                # Prepare properties
                mol_properties = {'split': split}
                if include_properties:
                    for prop in property_keys:
                        if prop in properties:
                            mol_properties[prop] = float(properties[prop][i])
                
                # Add to database
                db.write(atoms, key_value_pairs=mol_properties)
                total_molecules += 1
                
                if (i + 1) % 1000 == 0:
                    logging.info(f"  Processed {i + 1}/{n_molecules} molecules from {split}")
    
    logging.info(f"Successfully converted {total_molecules} molecules to ASE database: {output_db_path}")
    return output_db_path