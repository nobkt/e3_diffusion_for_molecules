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
    coord_norm_factor : float
        Coordinate normalization factor for proper scaling
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
    
    # Compute coordinate normalization factor
    coord_norm_factor = compute_ase_coordinate_normalization(dataset_data)
    
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
    
    return processed_datasets, num_species, charge_scale, coord_norm_factor


def compute_ase_coordinate_normalization(dataset_data):
    """
    Compute appropriate coordinate normalization factors for ASE data.
    
    This function computes a normalization factor that scales molecular coordinates
    to an appropriate range for diffusion model training. It uses the typical 
    molecular size scale rather than just standard deviation to ensure better
    training dynamics.
    
    Parameters
    ----------
    dataset_data : dict
        Dataset data containing positions
        
    Returns
    -------
    float
        Coordinate normalization factor (molecular size scale)
    """
    positions = dataset_data['positions']  # Shape: [n_molecules, max_atoms, 3]
    atom_mask = torch.ones_like(positions[:, :, 0])  # Default mask if not available
    
    # Create proper atom mask if num_atoms is available
    if 'num_atoms' in dataset_data:
        num_atoms = dataset_data['num_atoms']
        atom_mask = torch.zeros_like(positions[:, :, 0])
        for i, n in enumerate(num_atoms):
            atom_mask[i, :n] = 1.0
    
    # Extract valid coordinates (masked)
    valid_coords = positions[atom_mask.unsqueeze(-1).expand_as(positions) == 1]
    
    if len(valid_coords) == 0:
        print("Warning: No valid coordinates found, using default normalization factor 3.0")
        return 3.0
    
    # Compute molecular size statistics
    coord_std = torch.std(valid_coords).item()
    coord_max = torch.max(torch.abs(valid_coords)).item()
    coord_mean_abs = torch.mean(torch.abs(valid_coords)).item()
    
    # Use a scale factor based on typical molecular dimensions
    # This approach is more robust than just standard deviation
    # Target: scale coordinates to ~1-2 range for optimal diffusion dynamics
    
    # Method 1: Use 90th percentile of absolute coordinates (robust to outliers)
    coord_90th = torch.quantile(torch.abs(valid_coords), 0.9).item()
    
    # Method 2: Use mean absolute deviation + margin for molecular size
    coord_mad = torch.mean(torch.abs(valid_coords - torch.mean(valid_coords))).item()
    
    # Choose normalization factor: use the larger of 90th percentile or 2*MAD
    # This ensures we capture the typical molecular size while being robust to outliers
    norm_factor = max(coord_90th, 2 * coord_mad)
    
    # Ensure reasonable bounds: molecular coordinates should be in 1-10 Angstrom range
    # So normalization factor should be 1.5-8.0 for good diffusion dynamics
    norm_factor = max(norm_factor, 1.5)  # Minimum normalization for small molecules
    norm_factor = min(norm_factor, 8.0)  # Maximum normalization for large molecules
    
    print(f"=== ASE Coordinate Normalization Analysis ===")
    print(f"Coordinate statistics:")
    print(f"  Mean absolute value: {coord_mean_abs:.4f}")
    print(f"  Standard deviation: {coord_std:.4f}")
    print(f"  90th percentile: {coord_90th:.4f}")
    print(f"  Mean absolute deviation: {coord_mad:.4f}")
    print(f"  Maximum absolute value: {coord_max:.4f}")
    print(f"Computed coordinate normalization factor: {norm_factor:.4f}")
    print(f"This will scale coordinates to ~{coord_90th/norm_factor:.2f} typical range")
    
    return norm_factor


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
    
    # Get all unique atomic numbers and statistics
    all_atomic_numbers = set()
    max_atoms = 0
    n_nodes_count = {}
    atomic_number_count = {}
    
    for atoms in atoms_list:
        atomic_numbers = atoms.numbers
        if remove_h:
            atomic_numbers = atomic_numbers[atomic_numbers != 1]
        
        all_atomic_numbers.update(atomic_numbers)
        n_atoms = len(atomic_numbers)
        max_atoms = max(max_atoms, n_atoms)
        n_nodes_count[n_atoms] = n_nodes_count.get(n_atoms, 0) + 1
        
        # Count atomic numbers for statistics
        for atomic_num in atomic_numbers:
            atomic_number_count[atomic_num] = atomic_number_count.get(atomic_num, 0) + 1
    
    # Create atom mappings
    all_atomic_numbers = sorted(list(all_atomic_numbers))
    
    # Complete element symbols mapping (periodic table)
    atomic_num_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
        91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm'
    }
    
    atom_decoder = []
    atom_encoder = {}
    atom_types_stats = {}
    
    for i, atomic_num in enumerate(all_atomic_numbers):
        symbol = atomic_num_to_symbol.get(atomic_num, f'X{atomic_num}')
        atom_decoder.append(symbol)
        atom_encoder[symbol] = i
        # Map atomic number counts to config indices
        atom_types_stats[i] = atomic_number_count.get(atomic_num, 0)
    
    # Update the appropriate configuration
    config = ase_db_without_h if remove_h else ase_db_with_h
    config['atom_encoder'] = atom_encoder
    config['atom_decoder'] = atom_decoder
    config['max_n_nodes'] = max_atoms
    config['n_nodes'] = n_nodes_count
    config['atom_types'] = atom_types_stats
    
    # Update colors and radius for visualization (ensure adequate coverage)
    n_types = len(atom_decoder)
    
    # Generate sufficient colors and radii
    base_colors = ['#FFFFFF99', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 
                   'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive',
                   'cyan', 'magenta', 'yellow', 'lime', 'navy', 'maroon', 'teal', 'silver']
    
    base_radii = [0.46, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,
                  0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,
                  0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]
    
    # Extend colors and radii as needed
    needed_colors = max(n_types, 1)
    colors_multiplier = (needed_colors // len(base_colors)) + 1
    radii_multiplier = (needed_colors // len(base_radii)) + 1
    
    extended_colors = (base_colors * colors_multiplier)[:needed_colors]
    extended_radii = (base_radii * radii_multiplier)[:needed_colors]
    
    if not remove_h and 'H' in atom_encoder:
        # For hydrogen-inclusive: keep white for H if it exists
        h_index = atom_encoder['H']
        if h_index < len(extended_colors):
            extended_colors[h_index] = '#FFFFFF99'
            extended_radii[h_index] = 0.46
    
    config['colors_dic'] = extended_colors
    config['radius_dic'] = extended_radii
    
    print(f"Updated ASE dataset config: {len(atom_decoder)} atom types, max {max_atoms} atoms per molecule")
    print(f"  Element distribution: {dict(sorted([(atomic_num_to_symbol.get(k, f'X{k}'), v) for k, v in atomic_number_count.items()]))}")


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
        datasets, num_species, charge_scale, coord_norm_factor = load_ase_database(
            cfg.ase_db_path,
            split_ratios=getattr(cfg, 'split_ratios', (0.8, 0.1, 0.1)),
            seed=getattr(cfg, 'seed', 42),
            include_charges=cfg.include_charges,
            remove_h=cfg.remove_h
        )
        
        # Store coordinate normalization factor for use in model
        cfg.ase_coord_norm_factor = coord_norm_factor
        
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