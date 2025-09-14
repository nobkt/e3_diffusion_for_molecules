from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
from qm9.data.dataset_class import ProcessedDataset
import os
import torch
import numpy as np


def load_ase_database(db_path, split_ratios=(0.8, 0.1, 0.1), seed=42, include_charges=True, remove_h=False):
    """
    Load dataset from ASE database format.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
    split_ratios : tuple
        Train, validation, test split ratios (should sum to 1.0)
    seed : int
        Random seed for reproducible splits
    include_charges : bool
        Whether to include atomic charges in the dataset
    remove_h : bool
        Whether to remove hydrogen atoms
        
    Returns
    -------
    datasets : dict
        Dictionary with 'train', 'valid', 'test' keys containing processed datasets
    num_species : int
        Number of unique atomic species
    charge_scale : float
        Scale factor for charges (compatibility with QM9)
    """
    try:
        from ase.db import connect
    except ImportError:
        raise ImportError("ASE package is required for database loading. Install with: pip install ase")
    
    # Connect to database
    db = connect(db_path)
    
    # Load all atoms and properties
    all_atoms = []
    all_properties = []
    
    for row in db.select():
        atoms = row.toatoms()
        properties = {}
        
        # Get molecular properties if available
        if hasattr(row, 'data') and row.data:
            properties.update(row.data)
        if hasattr(row, 'key_value_pairs') and row.key_value_pairs:
            properties.update(row.key_value_pairs)
            
        all_atoms.append(atoms)
        all_properties.append(properties)
    
    if len(all_atoms) == 0:
        raise ValueError(f"No molecules found in database {db_path}")
    
    print(f"Loaded {len(all_atoms)} molecules from ASE database")
    
    # Convert ASE atoms to the required format
    dataset_data = convert_ase_to_dataset_format(all_atoms, all_properties, include_charges, remove_h)
    
    # Update dataset configuration based on actual data
    update_ase_dataset_config(all_atoms, remove_h)
    
    # Split the data
    np.random.seed(seed)
    n_total = len(all_atoms)
    indices = np.random.permutation(n_total)
    
    n_train = int(split_ratios[0] * n_total)
    n_valid = int(split_ratios[1] * n_total)
    n_test = n_total - n_train - n_valid
    
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]
    
    print(f"Split: {n_train} train, {n_valid} valid, {n_test} test")
    
    # Create split datasets
    datasets = {}
    for split_name, split_indices in [('train', train_indices), ('valid', valid_indices), ('test', test_indices)]:
        split_data = {}
        for key, values in dataset_data.items():
            if key.startswith('_'):
                # Metadata keys - copy as-is to all splits
                split_data[key] = values
            elif len(split_indices) > 0:
                split_data[key] = values[split_indices]
            else:
                # Handle empty splits - create empty tensor with correct shape
                if isinstance(values, torch.Tensor):
                    if len(values.shape) == 1:
                        split_data[key] = torch.empty(0, dtype=values.dtype)
                    else:
                        split_data[key] = torch.empty(0, *values.shape[1:], dtype=values.dtype)
                else:
                    split_data[key] = values  # For non-tensor metadata
        datasets[split_name] = split_data
    
    # Get species information
    all_species = torch.unique(dataset_data['charges'], sorted=True)
    if all_species[0] == 0:
        all_species = all_species[1:]
    
    # Create ProcessedDataset objects
    processed_datasets = {}
    for split, data in datasets.items():
        processed_datasets[split] = ProcessedDataset(
            data, 
            included_species=all_species, 
            num_pts=-1, 
            subtract_thermo=False  # ASE data typically doesn't have thermo corrections
        )
    
    num_species = len(all_species)
    charge_scale = torch.max(all_species).item()  # For compatibility
    
    return processed_datasets, num_species, charge_scale


def update_ase_dataset_config(atoms_list, remove_h=False):
    """
    Update the ASE dataset configuration based on actual data.
    
    Parameters
    ----------
    atoms_list : list of ase.Atoms
        List of ASE Atoms objects
    remove_h : bool
        Whether hydrogen atoms are removed
    """
    from configs.datasets_config import ase_db_with_h, ase_db_without_h
    
    # Get all unique atomic numbers and collect statistics
    all_atomic_numbers = set()
    max_atoms = 0
    n_nodes_count = {}
    element_counts = {}  # Track actual element frequencies
    
    for atoms in atoms_list:
        atomic_numbers = atoms.numbers
        if remove_h:
            atomic_numbers = atomic_numbers[atomic_numbers != 1]
        
        all_atomic_numbers.update(atomic_numbers)
        n_atoms = len(atomic_numbers)
        max_atoms = max(max_atoms, n_atoms)
        n_nodes_count[n_atoms] = n_nodes_count.get(n_atoms, 0) + 1
        
        # Count occurrences of each atomic number
        for atomic_num in atomic_numbers:
            element_counts[atomic_num] = element_counts.get(atomic_num, 0) + 1
    
    # Create atom mappings using comprehensive element data
    all_atomic_numbers = sorted(list(all_atomic_numbers))
    
    # Use ASE's complete element mapping
    try:
        from ase.data import chemical_symbols
        atomic_num_to_symbol = {}
        for i, symbol in enumerate(chemical_symbols):
            if i > 0 and symbol != 'X':  # Skip index 0 and placeholder
                atomic_num_to_symbol[i] = symbol
    except ImportError:
        # Fallback to manual mapping if ASE data not available
        atomic_num_to_symbol = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
            14: 'Si', 13: 'Al', 32: 'Ge', 33: 'As', 34: 'Se', 5: 'B', 4: 'Be', 3: 'Li', 11: 'Na', 12: 'Mg',
            19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
            29: 'Cu', 30: 'Zn', 31: 'Ga', 50: 'Sn', 51: 'Sb', 52: 'Te', 82: 'Pb', 83: 'Bi',
            # Add more elements as needed
            2: 'He', 10: 'Ne', 18: 'Ar', 36: 'Kr', 54: 'Xe', 86: 'Rn',  # Noble gases
            37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd',
            47: 'Ag', 48: 'Cd', 49: 'In', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
            79: 'Au', 80: 'Hg', 81: 'Tl', 84: 'Po', 85: 'At', 87: 'Fr', 88: 'Ra'
        }
    
    atom_decoder = []
    atom_encoder = {}
    atom_types = {}  # Map encoder index to count
    
    for i, atomic_num in enumerate(all_atomic_numbers):
        symbol = atomic_num_to_symbol.get(atomic_num, f'X{atomic_num}')
        atom_decoder.append(symbol)
        atom_encoder[symbol] = i
        # Map encoder index to actual count
        atom_types[i] = element_counts.get(atomic_num, 0)
    
    # Calculate optimal normalization factor
    n_elements = len(atom_decoder)
    if n_elements <= 5:
        optimal_categorical_norm = 4.0
    elif n_elements <= 10:
        optimal_categorical_norm = 2.0
    else:
        optimal_categorical_norm = 1.0
    
    # Update the appropriate configuration
    config = ase_db_without_h if remove_h else ase_db_with_h
    config['atom_encoder'] = atom_encoder
    config['atom_decoder'] = atom_decoder
    config['max_n_nodes'] = max_atoms
    config['n_nodes'] = n_nodes_count
    config['atom_types'] = atom_types  # Now properly populated
    config['recommended_categorical_norm'] = optimal_categorical_norm
    config['n_elements'] = n_elements
    config['is_fallback_config'] = False  # Mark as dynamically updated
    
    # Update colors and radius for visualization (extend if needed)
    
    # Use a more diverse color palette
    base_colors = [
        '#FFFFFF99',  # White for H
        'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#FFB347', '#87CEEB', '#F0E68C', '#FFE4E1',
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471'
    ]
    
    # More sophisticated radius assignment
    default_radii = {
        'H': 0.46, 'C': 0.77, 'N': 0.77, 'O': 0.77, 'F': 0.77,
        'P': 0.77, 'S': 0.77, 'Cl': 0.77, 'Br': 0.77, 'I': 0.77,
        'Si': 0.77, 'Al': 0.77, 'B': 0.77, 'Li': 0.77, 'Na': 0.77,
        'K': 0.77, 'Ca': 0.77, 'Mg': 0.77, 'Fe': 0.77, 'Zn': 0.77,
        'Cu': 0.77, 'Mn': 0.77, 'Co': 0.77, 'Ni': 0.77, 'Cr': 0.77
    }
    
    if not remove_h:
        # Keep hydrogen special color if present
        config['colors_dic'] = []
        config['radius_dic'] = []
        for i, symbol in enumerate(atom_decoder):
            if symbol == 'H':
                config['colors_dic'].append('#FFFFFF99')
                config['radius_dic'].append(0.46)
            else:
                color_idx = (i - (1 if 'H' in atom_decoder else 0)) % len(base_colors[1:])
                config['colors_dic'].append(base_colors[1:][color_idx])
                config['radius_dic'].append(default_radii.get(symbol, 0.77))
    else:
        config['colors_dic'] = []
        config['radius_dic'] = []
        for i, symbol in enumerate(atom_decoder):
            color_idx = i % len(base_colors[1:])  # Skip hydrogen color
            config['colors_dic'].append(base_colors[1:][color_idx])
            config['radius_dic'].append(default_radii.get(symbol, 0.77))
    
    print(f"Updated ASE dataset config: {len(atom_decoder)} atom types, max {max_atoms} atoms per molecule")
    print(f"Elements found: {', '.join(atom_decoder)}")
    print(f"Element frequencies: {dict((atom_decoder[i], count) for i, count in atom_types.items())}")
    print(f"Recommended categorical normalization factor: {optimal_categorical_norm}")
    
    # Report element distribution for debugging
    total_atoms = sum(atom_types.values())
    print("Element distribution:")
    for i, symbol in enumerate(atom_decoder):
        count = atom_types.get(i, 0)
        percentage = (count / total_atoms * 100) if total_atoms > 0 else 0
        print(f"  {symbol}: {count:,} atoms ({percentage:.1f}%)")


def convert_ase_to_dataset_format(atoms_list, properties_list, include_charges=True, remove_h=False):
    """
    Convert list of ASE Atoms objects to the dataset format expected by ProcessedDataset.
    
    Parameters
    ----------
    atoms_list : list of ase.Atoms
        List of ASE Atoms objects
    properties_list : list of dict
        List of property dictionaries for each molecule
    include_charges : bool
        Whether to include atomic charges
    remove_h : bool
        Whether to remove hydrogen atoms
        
    Returns
    -------
    dataset_data : dict
        Dictionary with tensors in the required format
    """
    from qm9.openbabel_functions import extract_molecular_descriptors_ase_openbabel
    
    max_atoms = max(len(atoms) for atoms in atoms_list)
    n_molecules = len(atoms_list)
    
    # Initialize arrays
    positions = torch.zeros(n_molecules, max_atoms, 3, dtype=torch.float32)
    charges = torch.zeros(n_molecules, max_atoms, dtype=torch.long)
    num_atoms = torch.zeros(n_molecules, dtype=torch.long)
    
    # Property arrays - we'll collect available properties
    all_properties = set()
    for props in properties_list:
        all_properties.update(props.keys())
    
    property_tensors = {}
    
    # Common QM9-style properties to look for
    qm9_properties = ['energy', 'homo', 'lumo', 'gap', 'mu', 'alpha', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    
    for prop in qm9_properties:
        if any(prop in props for props in properties_list):
            property_tensors[prop] = torch.zeros(n_molecules, dtype=torch.float32)
    
    # Add new molecular descriptor properties
    property_tensors['molecular_weight'] = torch.zeros(n_molecules, dtype=torch.float32)
    property_tensors['pi_conjugation_ratio'] = torch.zeros(n_molecules, dtype=torch.float32)
    
    # For categorical features, we'll store them as lists first, then encode
    all_atom_types = set()
    all_functional_groups = set()
    atom_types_list = []
    functional_groups_list = []

    for i, (atoms, props) in enumerate(zip(atoms_list, properties_list)):
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        
        if remove_h:
            # Remove hydrogen atoms (atomic number 1)
            mask = atomic_numbers != 1
            pos = pos[mask]
            atomic_numbers = atomic_numbers[mask]
        
        n_atoms = len(atomic_numbers)
        num_atoms[i] = n_atoms
        
        if n_atoms > 0:
            # Center the molecule
            pos = pos - pos.mean(dim=0)
            
            positions[i, :n_atoms] = pos
            charges[i, :n_atoms] = atomic_numbers
        
        # Store existing properties
        for prop_name, tensor in property_tensors.items():
            if prop_name in props:
                try:
                    tensor[i] = float(props[prop_name])
                except (ValueError, TypeError):
                    # Handle cases where property can't be converted to float
                    tensor[i] = 0.0
        
        # Extract molecular descriptors using ASE and OpenBabel
        try:
            descriptors = extract_molecular_descriptors_ase_openbabel(atoms)
            
            # Store molecular weight and π conjugation ratio
            property_tensors['molecular_weight'][i] = descriptors['molecular_weight']
            property_tensors['pi_conjugation_ratio'][i] = descriptors['pi_conjugation_ratio']
            
            # Collect atom types and functional groups for later encoding
            atom_types_list.append(descriptors['atom_types'])
            functional_groups_list.append(descriptors['functional_groups'])
            
            # Update sets for encoding
            all_atom_types.update(descriptors['atom_types'])
            all_functional_groups.update(descriptors['functional_groups'])
            
        except Exception as e:
            print(f"Warning: Failed to extract descriptors for molecule {i}: {e}")
            # Fill with default values
            property_tensors['molecular_weight'][i] = 0.0
            property_tensors['pi_conjugation_ratio'][i] = 0.0
            atom_types_list.append([])
            functional_groups_list.append([])
    
    # Create encodings for categorical features
    atom_types_sorted = sorted(list(all_atom_types))
    functional_groups_sorted = sorted(list(all_functional_groups))
    
    # Create individual scalar features for each atom type and functional group
    # This is more compatible with the existing conditioning framework
    
    # For atom types - create one feature per atom type
    if atom_types_sorted:
        for atom_type in atom_types_sorted:
            feature_name = f'has_{atom_type}'
            property_tensors[feature_name] = torch.zeros(n_molecules, dtype=torch.float32)
            for i, mol_atom_types in enumerate(atom_types_list):
                if atom_type in mol_atom_types:
                    property_tensors[feature_name][i] = 1.0
    
    # For functional groups - create one feature per functional group  
    if functional_groups_sorted:
        for fg in functional_groups_sorted:
            feature_name = f'has_{fg}'
            property_tensors[feature_name] = torch.zeros(n_molecules, dtype=torch.float32)
            for i, mol_functional_groups in enumerate(functional_groups_list):
                if fg in mol_functional_groups:
                    property_tensors[feature_name][i] = 1.0
    
    # Also keep the original multi-dimensional encodings for backward compatibility
    # but mark them as special by adding them to dataset_data directly
    atom_types_encoding = torch.zeros(n_molecules, max(len(atom_types_sorted), 1), dtype=torch.float32)
    if atom_types_sorted:
        for i, mol_atom_types in enumerate(atom_types_list):
            for atom_type in mol_atom_types:
                if atom_type in atom_types_sorted:
                    idx = atom_types_sorted.index(atom_type)
                    atom_types_encoding[i, idx] = 1.0
    
    functional_groups_encoding = torch.zeros(n_molecules, max(len(functional_groups_sorted), 1), dtype=torch.float32)
    if functional_groups_sorted:
        for i, mol_functional_groups in enumerate(functional_groups_list):
            for fg in mol_functional_groups:
                if fg in functional_groups_sorted:
                    idx = functional_groups_sorted.index(fg)
                    functional_groups_encoding[i, idx] = 1.0

    dataset_data = {
        'positions': positions,
        'charges': charges,
        'num_atoms': num_atoms,
        'atom_types_encoding': atom_types_encoding,
        'functional_groups_encoding': functional_groups_encoding
    }
    
    # Add property tensors
    dataset_data.update(property_tensors)
    
    # Store the atom types and functional groups mappings for later use
    dataset_data['_atom_types_mapping'] = atom_types_sorted
    dataset_data['_functional_groups_mapping'] = functional_groups_sorted
    
    return dataset_data


def retrieve_dataloaders(cfg):
    if 'ase_db' in cfg.dataset:
        # ASE database loading
        if not hasattr(cfg, 'ase_db_path') or cfg.ase_db_path is None:
            raise ValueError("ASE database path must be specified with --ase_db_path argument")
        
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        
        # Load from ASE database
        datasets, num_species, charge_scale = load_ase_database(
            cfg.ase_db_path,
            split_ratios=getattr(cfg, 'split_ratios', (0.8, 0.1, 0.1)),
            seed=getattr(cfg, 'seed', 42),
            include_charges=cfg.include_charges,
            remove_h=cfg.remove_h
        )
        
        # Convert units if needed (ASE typically uses eV, Angstrom)
        # You might need to adjust these conversion factors based on your data
        ase_to_eV = getattr(cfg, 'ase_to_eV', {})
        if ase_to_eV:
            for dataset in datasets.values():
                dataset.convert_units(ase_to_eV)
        
        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)
        
        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'),
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                       for split, dataset in datasets.items()}
        
    elif 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets