from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
from qm9.data.dataset_class import ProcessedDataset
import os
import torch
import numpy as np


def load_ase_database(db_path, split_ratios=(0.8, 0.1, 0.1), seed=42, include_charges=False, remove_h=False, 
                      remove_duplicates=True, duplicate_tolerance=1e-6, debug_csv_path=None, debug_xyz_path=None):
    """
    Load dataset from ASE database format.
    
    Note: For ASE databases, atomic charges (nuclear charges/atomic numbers) are not 
    included in the database by default, so include_charges defaults to False.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
    split_ratios : tuple
        Train, validation, test split ratios (should sum to 1.0)
    seed : int
        Random seed for reproducible splits
    include_charges : bool
        Whether to include atomic charges (atomic numbers) in the dataset.
        For ASE databases, this defaults to False since atomic charges are
        not typically included in the database.
    remove_h : bool
        Whether to remove hydrogen atoms
    remove_duplicates : bool
        Whether to remove duplicate molecules based on geometry comparison
    duplicate_tolerance : float
        Tolerance for considering two molecules as duplicates based on position differences
    debug_csv_path : str, optional
        Path to save CSV file with dataset composition analysis
    debug_xyz_path : str, optional
        Directory path to save XYZ files for all molecules
        
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
    
    # Remove duplicates if requested
    if remove_duplicates and len(all_atoms) > 1:
        print("Detecting and removing duplicate molecules...")
        print(f"Using duplicate tolerance: {duplicate_tolerance} Angstrom")
        all_atoms, all_properties = _remove_duplicates_optimized(
            all_atoms, all_properties, remove_h, duplicate_tolerance
        )
    
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
                # Handle empty tensors (like charges when include_charges=False)
                if isinstance(values, torch.Tensor) and values.numel() == 0:
                    split_data[key] = values  # Keep empty tensor as-is
                else:
                    split_data[key] = values[split_indices]
            else:
                # Handle empty splits - create empty tensor with correct shape
                if isinstance(values, torch.Tensor):
                    if values.numel() == 0:
                        split_data[key] = values  # Keep empty tensor as-is
                    elif len(values.shape) == 1:
                        split_data[key] = torch.empty(0, dtype=values.dtype)
                    else:
                        split_data[key] = torch.empty(0, *values.shape[1:], dtype=values.dtype)
                else:
                    split_data[key] = values  # For non-tensor metadata
        datasets[split_name] = split_data
    
    # Get species information
    if include_charges and 'charges' in dataset_data and dataset_data['charges'].numel() > 0:
        all_species = torch.unique(dataset_data['charges'], sorted=True)
        if all_species[0] == 0:
            all_species = all_species[1:]
    else:
        # Use atomic_numbers for species information (always available for ASE databases)
        if 'atomic_numbers' in dataset_data and dataset_data['atomic_numbers'].numel() > 0:
            all_species = torch.unique(dataset_data['atomic_numbers'], sorted=True)
            if all_species[0] == 0:
                all_species = all_species[1:]
        else:
            # Fallback: determine species from the actual atoms in the database
            all_atomic_numbers = set()
            for atoms in all_atoms:
                atomic_numbers = atoms.numbers
                if remove_h:
                    atomic_numbers = atomic_numbers[atomic_numbers != 1]
                all_atomic_numbers.update(atomic_numbers)
            all_species = torch.tensor(sorted(list(all_atomic_numbers)), dtype=torch.long)
    
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
    charge_scale = torch.max(all_species).item() if len(all_species) > 0 else 1  # For compatibility
    
    # Generate debug outputs if requested
    if debug_csv_path or debug_xyz_path:
        _generate_debug_outputs(all_atoms, all_properties, all_species, debug_csv_path, debug_xyz_path, remove_h)
    
    return processed_datasets, num_species, charge_scale


def _remove_duplicates_optimized(all_atoms, all_properties, remove_h, duplicate_tolerance):
    """
    Optimized duplicate removal using hashing for O(n log n) performance instead of O(n²).
    
    This function uses molecular fingerprints based on sorted atomic numbers and position
    hashes to quickly identify potential duplicates, then performs detailed comparison
    only on candidates.
    
    Parameters
    ----------
    all_atoms : list of ase.Atoms
        List of ASE Atoms objects
    all_properties : list of dict  
        List of property dictionaries
    remove_h : bool
        Whether to remove hydrogen atoms for comparison
    duplicate_tolerance : float
        Tolerance for position comparison
        
    Returns
    -------
    unique_atoms : list
        List of unique ASE Atoms objects
    unique_properties : list
        List of corresponding property dictionaries
    """
    import hashlib
    from collections import defaultdict
    import time
    
    n_total = len(all_atoms)
    print(f"Processing {n_total} molecules for duplicate detection...")
    
    # Step 1: Create molecular fingerprints for fast pre-filtering
    print("Step 1/3: Creating molecular fingerprints...")
    fingerprint_to_indices = defaultdict(list)
    processed_molecules = []
    
    start_time = time.time()
    
    for i, (atoms, properties) in enumerate(zip(all_atoms, all_properties)):
        # Show progress every 1000 molecules
        if i > 0 and i % 1000 == 0:
            elapsed = time.time() - start_time
            progress = i / n_total
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Progress: {i}/{n_total} ({progress*100:.1f}%) - ETA: {eta:.0f}s")
        
        # Process molecule for comparison
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_nums = torch.tensor(atoms.numbers, dtype=torch.long)
        
        if remove_h:
            mask = atomic_nums != 1
            pos = pos[mask]
            atomic_nums = atomic_nums[mask]
        
        # Center the molecule
        if len(pos) > 0:
            pos = pos - pos.mean(dim=0)
        
        # Create a molecular fingerprint for fast comparison
        # Fingerprint includes: number of atoms, sorted atomic numbers, and rough position hash
        atomic_nums_sorted = torch.sort(atomic_nums)[0]  # Sort atomic numbers
        
        # Create coarse position bins for hashing (larger than tolerance to catch near-duplicates)
        position_bins = torch.round(pos / (duplicate_tolerance * 10)).long() if len(pos) > 0 else torch.tensor([], dtype=torch.long)
        
        # Create fingerprint string
        fingerprint_data = f"{len(atomic_nums)}_{atomic_nums_sorted.tolist()}_{position_bins.flatten().tolist()}"
        fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]  # Use first 16 chars for efficiency
        
        # Store processed data
        processed_molecules.append({
            'index': i,
            'atoms': atoms,
            'properties': properties,
            'pos': pos,
            'atomic_nums': atomic_nums,
            'fingerprint': fingerprint
        })
        
        fingerprint_to_indices[fingerprint].append(i)
    
    print(f"Step 1 completed in {time.time() - start_time:.1f}s")
    print(f"Found {len(fingerprint_to_indices)} unique fingerprints")
    
    # Step 2: Detailed comparison only within fingerprint groups
    print("Step 2/3: Detailed duplicate detection within fingerprint groups...")
    
    unique_indices = set()
    duplicate_count = 0
    duplicate_examples = []
    groups_processed = 0
    
    start_time = time.time()
    
    for fingerprint, indices in fingerprint_to_indices.items():
        groups_processed += 1
        if groups_processed % 100 == 0:
            elapsed = time.time() - start_time
            progress = groups_processed / len(fingerprint_to_indices)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Progress: {groups_processed}/{len(fingerprint_to_indices)} groups ({progress*100:.1f}%) - ETA: {eta:.0f}s")
        
        if len(indices) == 1:
            # No duplicates possible in this group
            unique_indices.add(indices[0])
        else:
            # Detailed comparison needed within this group
            group_unique_indices = []
            
            for i, idx in enumerate(indices):
                mol_i = processed_molecules[idx]
                is_duplicate = False
                
                # Compare with already selected unique molecules in this group
                for unique_idx in group_unique_indices:
                    mol_j = processed_molecules[unique_idx]
                    
                    # Detailed comparison
                    if (_molecules_are_identical(mol_i['pos'], mol_i['atomic_nums'], 
                                               mol_j['pos'], mol_j['atomic_nums'], 
                                               duplicate_tolerance)):
                        is_duplicate = True
                        duplicate_count += 1
                        
                        # Store example for debugging (first few)
                        if len(duplicate_examples) < 3:
                            pos_diff = torch.max(torch.abs(mol_i['pos'] - mol_j['pos'])).item() if len(mol_i['pos']) > 0 else 0.0
                            duplicate_examples.append({
                                'molecule_id': idx,
                                'duplicate_of': unique_idx,
                                'atomic_nums': mol_i['atomic_nums'].tolist(),
                                'pos_diff_max': pos_diff
                            })
                        break
                
                if not is_duplicate:
                    group_unique_indices.append(idx)
                    unique_indices.add(idx)
    
    print(f"Step 2 completed in {time.time() - start_time:.1f}s")
    
    # Step 3: Build result lists
    print("Step 3/3: Building results...")
    unique_indices_sorted = sorted(list(unique_indices))
    unique_atoms = [all_atoms[i] for i in unique_indices_sorted]
    unique_properties = [all_properties[i] for i in unique_indices_sorted]
    
    print(f"Removed {duplicate_count} duplicate molecules")
    print(f"Keeping {len(unique_atoms)} unique molecules")
    
    # Show duplicate examples for debugging
    if duplicate_examples:
        print("\nDuplicate detection examples (first few):")
        for example in duplicate_examples:
            print(f"  Molecule {example['molecule_id']} is duplicate of molecule {example['duplicate_of']} "
                  f"(max position difference: {example['pos_diff_max']:.6f} Angstrom)")
    
    if len(unique_atoms) < 10:
        print(f"WARNING: Only {len(unique_atoms)} unique molecules found. This may cause training instability.")
        print("Consider using a more diverse dataset or setting remove_duplicates=False.")
        print("You can also try increasing --duplicate_tolerance if molecules are similar but not identical.")
    
    return unique_atoms, unique_properties


def _molecules_are_identical(pos1, atomic_nums1, pos2, atomic_nums2, tolerance):
    """
    Compare two molecules for exact identity within tolerance.
    
    Parameters
    ----------
    pos1, pos2 : torch.Tensor
        Centered positions of the two molecules
    atomic_nums1, atomic_nums2 : torch.Tensor
        Atomic numbers of the two molecules  
    tolerance : float
        Position tolerance for comparison
        
    Returns
    -------
    bool
        True if molecules are identical within tolerance
    """
    # Quick checks first
    if len(pos1) != len(pos2):
        return False
        
    if len(atomic_nums1) != len(atomic_nums2):
        return False
        
    if len(pos1) == 0:
        return True  # Both empty
        
    # Check if atomic numbers match exactly
    if not torch.allclose(atomic_nums1, atomic_nums2):
        return False
    
    # Check positions within tolerance
    if not torch.allclose(pos1, pos2, atol=tolerance):
        return False
        
    return True


def _generate_debug_outputs(atoms_list, properties_list, all_species, csv_path=None, xyz_path=None, remove_h=False):
    """
    Generate debugging outputs for dataset analysis.
    
    Parameters
    ----------
    atoms_list : list of ase.Atoms
        List of ASE Atoms objects
    properties_list : list of dict
        List of molecular properties dictionaries
    all_species : torch.Tensor
        Tensor of unique atomic species
    csv_path : str, optional
        Path to save CSV file with composition analysis
    xyz_path : str, optional
        Directory to save XYZ files
    remove_h : bool
        Whether hydrogen atoms are removed
    """
    import csv
    import os
    
    if csv_path:
        print(f"Generating dataset composition CSV at: {csv_path}")
        
        # Create CSV with molecule composition analysis
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['molecule_id', 'num_atoms', 'molecular_formula', 'atomic_numbers', 'positions_summary'] + list(properties_list[0].keys() if properties_list else [])
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (atoms, properties) in enumerate(zip(atoms_list, properties_list)):
                atomic_numbers = atoms.numbers
                positions = atoms.positions
                
                if remove_h:
                    mask = atomic_numbers != 1
                    atomic_numbers = atomic_numbers[mask]
                    positions = positions[mask]
                
                # Generate molecular formula
                from collections import Counter
                element_count = Counter(atomic_numbers)
                molecular_formula = ""
                for atomic_num in sorted(element_count.keys()):
                    count = element_count[atomic_num]
                    # Convert atomic number to symbol (simplified)
                    symbol_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
                    symbol = symbol_map.get(atomic_num, f'Z{atomic_num}')
                    molecular_formula += f"{symbol}{count if count > 1 else ''}"
                
                row = {
                    'molecule_id': i,
                    'num_atoms': len(atomic_numbers),
                    'molecular_formula': molecular_formula,
                    'atomic_numbers': ','.join(map(str, atomic_numbers)),
                    'positions_summary': f"min:{positions.min():.3f},max:{positions.max():.3f},center:({positions.mean(axis=0)[0]:.3f},{positions.mean(axis=0)[1]:.3f},{positions.mean(axis=0)[2]:.3f})"
                }
                row.update(properties)
                writer.writerow(row)
        
        print(f"CSV file saved with {len(atoms_list)} molecules")
    
    if xyz_path:
        print(f"Generating XYZ files in directory: {xyz_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(xyz_path, exist_ok=True)
        
        # Create XYZ files for each molecule
        for i, atoms in enumerate(atoms_list):
            atomic_numbers = atoms.numbers
            positions = atoms.positions
            
            if remove_h:
                mask = atomic_numbers != 1
                atomic_numbers = atomic_numbers[mask]
                positions = positions[mask]
            
            xyz_filename = os.path.join(xyz_path, f"molecule_{i:04d}.xyz")
            with open(xyz_filename, 'w') as f:
                f.write(f"{len(atomic_numbers)}\n")
                f.write(f"Molecule {i} from ASE database\n")
                
                # Convert atomic numbers to symbols
                symbol_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
                for atomic_num, pos in zip(atomic_numbers, positions):
                    symbol = symbol_map.get(atomic_num, f'Z{atomic_num}')
                    f.write(f"{symbol} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}\n")
        
        print(f"Generated {len(atoms_list)} XYZ files")


def update_ase_dataset_config(atoms_list, remove_h=False):
    """
    Update the ASE dataset configuration based on actual data analysis.
    
    This function comprehensively analyzes the entire ASE database to determine:
    - All constituent elements without any heuristic processing
    - Maximum number of atoms per molecule
    - Distribution of molecule sizes (n_nodes)
    - Atom type frequency distribution
    
    Parameters
    ----------
    atoms_list : list of ase.Atoms
        List of ASE Atoms objects representing the entire database
    remove_h : bool
        Whether hydrogen atoms are removed
    """
    from configs.datasets_config import ase_db_with_h, ase_db_without_h
    
    # Comprehensive analysis of the entire database
    all_atomic_numbers = set()
    max_atoms = 0
    n_nodes_count = {}
    atom_type_count = {}
    
    print(f"Analyzing {len(atoms_list)} molecules from ASE database...")
    
    # Analyze each molecule in the database
    for i, atoms in enumerate(atoms_list):
        atomic_numbers = atoms.numbers
        if remove_h:
            atomic_numbers = atomic_numbers[atomic_numbers != 1]
        
        # Ensure atomic_numbers is iterable (handle single atom case)
        if hasattr(atomic_numbers, '__iter__'):
            atomic_numbers_list = atomic_numbers
        else:
            atomic_numbers_list = [atomic_numbers]
        
        # Track all unique atomic numbers (constituent elements)
        all_atomic_numbers.update(atomic_numbers_list)
        
        # Track molecule size distribution
        n_atoms = len(atomic_numbers_list)
        max_atoms = max(max_atoms, n_atoms)
        n_nodes_count[n_atoms] = n_nodes_count.get(n_atoms, 0) + 1
        
        # Track atom type frequency distribution
        for atomic_num in atomic_numbers_list:
            atom_type_count[atomic_num] = atom_type_count.get(atomic_num, 0) + 1
    
    # Sort atomic numbers for consistent ordering
    all_atomic_numbers = sorted(list(all_atomic_numbers))
    
    # Create comprehensive element mapping without heuristics
    # Use standard atomic number to symbol mapping
    atomic_num_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
        19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
        29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd',
        47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
        55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl',
        82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn'
    }
    
    # Build atom decoder and encoder based on actual data analysis
    atom_decoder = []
    atom_encoder = {}
    atom_type_distribution = {}
    
    for i, atomic_num in enumerate(all_atomic_numbers):
        symbol = atomic_num_to_symbol.get(atomic_num, f'X{atomic_num}')
        atom_decoder.append(symbol)
        atom_encoder[symbol] = i
        atom_type_distribution[i] = atom_type_count.get(atomic_num, 0)
    
    # Update the appropriate configuration based on actual database analysis
    config = ase_db_without_h if remove_h else ase_db_with_h
    config['atom_encoder'] = atom_encoder
    config['atom_decoder'] = atom_decoder
    config['max_n_nodes'] = max_atoms
    config['n_nodes'] = n_nodes_count
    config['atom_types'] = atom_type_distribution
    
    # Update colors and radius for visualization (extend if needed)
    n_types = len(atom_decoder)
    default_colors = ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1', 'C2', 'C4', 'C5', 'C6', 'C8', 'C9']
    default_radius = [0.46, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]
    
    if not remove_h:
        config['colors_dic'] = (default_colors * ((n_types // len(default_colors)) + 1))[:n_types]
        config['radius_dic'] = (default_radius * ((n_types // len(default_radius)) + 1))[:n_types]
    else:
        # Remove hydrogen colors/radius
        config['colors_dic'] = (default_colors[1:] * ((n_types // len(default_colors[1:])) + 1))[:n_types]
        config['radius_dic'] = (default_radius[1:] * ((n_types // len(default_radius[1:])) + 1))[:n_types]
    
    print(f"Database analysis complete:")
    print(f"  - Found {len(atom_decoder)} unique element types: {atom_decoder}")
    print(f"  - Maximum atoms per molecule: {max_atoms}")
    print(f"  - Total molecules analyzed: {len(atoms_list)}")
    print(f"  - Atom type distribution: {atom_type_distribution}")
    print(f"Updated ASE dataset configuration without heuristic processing")


def convert_ase_to_dataset_format(atoms_list, properties_list, include_charges=False, remove_h=False):
    """
    Convert list of ASE Atoms objects to the dataset format expected by ProcessedDataset.
    
    Note: For ASE databases, atomic charges (atomic numbers) are not included by default
    since they are not typically stored in ASE databases.
    
    Parameters
    ----------
    atoms_list : list of ase.Atoms
        List of ASE Atoms objects
    properties_list : list of dict
        List of property dictionaries for each molecule
    include_charges : bool
        Whether to include atomic charges (atomic numbers). Defaults to False
        for ASE databases since atomic charges are not included in the database.
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
    num_atoms = torch.zeros(n_molecules, dtype=torch.long)
    
    # Always create atomic_numbers for one_hot encoding (these are element types, not charges)
    atomic_numbers = torch.zeros(n_molecules, max_atoms, dtype=torch.long)
    
    # Initialize charges only if include_charges is True
    if include_charges:
        charges = torch.zeros(n_molecules, max_atoms, dtype=torch.long)
    else:
        # For ASE databases, atomic charges are not included in the database
        charges = torch.zeros(0, dtype=torch.long)  # Empty tensor for compatibility
    
    # Property arrays - comprehensive collection of ALL available properties
    all_properties = set()
    for props in properties_list:
        all_properties.update(props.keys())
    
    print(f"Found {len(all_properties)} unique properties in ASE database: {sorted(all_properties)}")
    
    property_tensors = {}
    
    # Load ALL available numeric properties from the database
    for prop_name in sorted(all_properties):
        # Check if this property contains numeric data that can be converted to tensors
        sample_values = []
        for props in properties_list:
            if prop_name in props:
                try:
                    val = float(props[prop_name])
                    sample_values.append(val)
                except (ValueError, TypeError):
                    # Skip non-numeric properties
                    break
        
        # If we found numeric values, create a tensor for this property
        if len(sample_values) > 0:
            property_tensors[prop_name] = torch.zeros(n_molecules, dtype=torch.float32)
            print(f"  Loading property: {prop_name}")
    
    # Add molecular descriptor properties (computed from structure)
    property_tensors['molecular_weight'] = torch.zeros(n_molecules, dtype=torch.float32)
    property_tensors['pi_conjugation_ratio'] = torch.zeros(n_molecules, dtype=torch.float32)
    
    # For categorical features, we'll store them as lists first, then encode
    all_atom_types = set()
    all_functional_groups = set()
    atom_types_list = []
    functional_groups_list = []

    for i, (atoms, props) in enumerate(zip(atoms_list, properties_list)):
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_nums = torch.tensor(atoms.numbers, dtype=torch.long)
        
        if remove_h:
            # Remove hydrogen atoms (atomic number 1)
            mask = atomic_nums != 1
            pos = pos[mask]
            atomic_nums = atomic_nums[mask]
        
        n_atoms = len(atomic_nums)
        num_atoms[i] = n_atoms
        
        if n_atoms > 0:
            # Center the molecule
            pos = pos - pos.mean(dim=0)
            
            positions[i, :n_atoms] = pos
            
            # Always store atomic numbers for one_hot encoding
            atomic_numbers[i, :n_atoms] = atomic_nums
            
            # Only assign charges if include_charges is True
            if include_charges:
                charges[i, :n_atoms] = atomic_nums
        
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
        'atomic_numbers': atomic_numbers,  # Always include for one_hot encoding
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
            remove_h=cfg.remove_h,
            remove_duplicates=getattr(cfg, 'remove_duplicates', True),
            duplicate_tolerance=getattr(cfg, 'duplicate_tolerance', 1e-6),
            debug_csv_path=getattr(cfg, 'debug_dataset_csv', None),
            debug_xyz_path=getattr(cfg, 'debug_dataset_xyz', None)
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
        dataloaders = {}
        for split, dataset in datasets.items():
            # Handle empty datasets - don't shuffle empty datasets to avoid RandomSampler error
            should_shuffle = (split == 'train') and (len(dataset) > 0)
            dataloaders[split] = DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=should_shuffle,
                                           num_workers=num_workers,
                                           collate_fn=preprocess.collate_fn)
        
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
        dataloaders = {}
        for split, dataset in datasets.items():
            # Handle empty datasets - don't shuffle empty datasets to avoid RandomSampler error
            should_shuffle = (args.shuffle if (split == 'train') else False) and (len(dataset) > 0)
            dataloaders[split] = DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=should_shuffle,
                                           num_workers=num_workers,
                                           collate_fn=preprocess.collate_fn)
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


def export_generation_conditions_to_csv(datasets, output_dir='.', dataset_info=None):
    """
    Export generation conditions (molecular_weight, pi_conjugation_ratio, 
    atom_types_encoding, functional_groups_encoding) to CSV files.
    
    Creates one CSV file per condition type containing data for all molecules.
    
    Parameters
    ----------
    datasets : dict
        Dictionary with 'train', 'valid', 'test' keys containing dataset splits
    output_dir : str
        Directory to save the CSV files
    dataset_info : dict, optional
        Dataset information containing atom_decoder for composition strings
    """
    import csv
    import os
    from collections import Counter
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Exporting generation conditions to CSV files...")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Combine all splits to export all molecules
    all_data = {}
    split_info = []
    
    for split_name, dataset in datasets.items():
        n_molecules = len(dataset)
        split_info.append((split_name, n_molecules))
        
        for key, values in dataset.data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(values)
    
    # Concatenate tensors from all splits
    combined_data = {}
    for key, values_list in all_data.items():
        if len(values_list) > 0:
            if isinstance(values_list[0], torch.Tensor):
                # Only concatenate non-empty tensors
                non_empty = [v for v in values_list if v.numel() > 0]
                if non_empty:
                    combined_data[key] = torch.cat(non_empty, dim=0)
                else:
                    combined_data[key] = values_list[0]  # Keep empty tensor
            elif isinstance(values_list[0], (list, tuple)):
                # For list/tuple metadata like mappings
                combined_data[key] = values_list[0]
            else:
                # For other metadata, just keep the first value
                combined_data[key] = values_list[0]
    
    n_total = len(combined_data['num_atoms'])
    print(f"Total molecules to export: {n_total}")
    for split_name, n_molecules in split_info:
        print(f"  {split_name}: {n_molecules}")
    print()
    
    # Get atom decoder for composition strings
    if dataset_info and 'atom_decoder' in dataset_info:
        atom_decoder = dataset_info['atom_decoder']
    else:
        # Default QM9 decoder
        atom_decoder = ['H', 'C', 'N', 'O', 'F']
    
    # Helper function to get molecular composition string
    def get_composition_string(charges_or_atomic_nums, num_atoms_val):
        """Generate molecular formula from atomic charges or atomic numbers."""
        if charges_or_atomic_nums is None:
            return "Unknown"
        
        atoms_array = charges_or_atomic_nums[:num_atoms_val].cpu().numpy()
        atom_counts = Counter()
        
        for atom_val in atoms_array:
            atom_val = int(atom_val)
            if atom_val > 0:
                # Try to get symbol from decoder
                if atom_val < len(atom_decoder):
                    atom_symbol = atom_decoder[atom_val]
                else:
                    # Fallback to atomic number notation
                    atom_symbol = f"Z{atom_val}"
                atom_counts[atom_symbol] += 1
        
        # Create formula string (e.g., C6H12O6)
        formula = ""
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            formula += f"{atom}{count if count > 1 else ''}"
        
        return formula if formula else "Unknown"
    
    # Get references to charges or atomic_numbers for composition
    charges_or_atomic = combined_data.get('charges')
    if charges_or_atomic is None or charges_or_atomic.numel() == 0:
        charges_or_atomic = combined_data.get('atomic_numbers')
    
    # Get atom types and functional groups mappings if available
    atom_types_mapping = combined_data.get('_atom_types_mapping', [])
    functional_groups_mapping = combined_data.get('_functional_groups_mapping', [])
    
    num_atoms = combined_data['num_atoms']
    
    # 1. Export molecular_weight.csv
    print("Generating molecular_weight.csv...")
    csv_path = os.path.join(output_dir, 'molecular_weight.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', '分子の組成', '分子量'])
        
        if 'molecular_weight' in combined_data:
            molecular_weights = combined_data['molecular_weight']
            
            for i in range(n_total):
                mol_id = i
                composition = get_composition_string(charges_or_atomic[i] if charges_or_atomic is not None else None, 
                                                    num_atoms[i].item())
                mol_weight = molecular_weights[i].item()
                writer.writerow([mol_id, composition, f"{mol_weight:.4f}"])
            
            print(f"  ✓ Exported {n_total} molecules to {csv_path}")
        else:
            print(f"  ✗ molecular_weight not found in dataset")
            print(f"     Available keys: {list(combined_data.keys())[:10]}...")
    
    # 2. Export pi_conjugation_ratio.csv
    print("Generating pi_conjugation_ratio.csv...")
    csv_path = os.path.join(output_dir, 'pi_conjugation_ratio.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', '分子の組成', 'π共役比率'])
        
        if 'pi_conjugation_ratio' in combined_data:
            pi_conjugation_ratios = combined_data['pi_conjugation_ratio']
            
            for i in range(n_total):
                mol_id = i
                composition = get_composition_string(charges_or_atomic[i] if charges_or_atomic is not None else None,
                                                    num_atoms[i].item())
                pi_ratio = pi_conjugation_ratios[i].item()
                writer.writerow([mol_id, composition, f"{pi_ratio:.4f}"])
            
            print(f"  ✓ Exported {n_total} molecules to {csv_path}")
        else:
            print(f"  ✗ pi_conjugation_ratio not found in dataset")
    
    # 3. Export atom_types_encoding.csv
    print("Generating atom_types_encoding.csv...")
    csv_path = os.path.join(output_dir, 'atom_types_encoding.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if 'atom_types_encoding' in combined_data:
            atom_types_encoding = combined_data['atom_types_encoding']
            
            # Create header with atom type names
            header = ['ID', '分子の組成']
            if len(atom_types_mapping) > 0:
                header.extend(atom_types_mapping)
            else:
                # Fallback: use indices
                n_atom_types = atom_types_encoding.shape[1] if len(atom_types_encoding.shape) > 1 else 1
                header.extend([f'AtomType_{j}' for j in range(n_atom_types)])
            writer.writerow(header)
            
            for i in range(n_total):
                mol_id = i
                composition = get_composition_string(charges_or_atomic[i] if charges_or_atomic is not None else None,
                                                    num_atoms[i].item())
                
                # Get encoding values
                if len(atom_types_encoding.shape) > 1:
                    encoding_values = [f"{val:.0f}" for val in atom_types_encoding[i].tolist()]
                else:
                    encoding_values = [f"{atom_types_encoding[i].item():.0f}"]
                
                writer.writerow([mol_id, composition] + encoding_values)
            
            print(f"  ✓ Exported {n_total} molecules to {csv_path}")
        else:
            print(f"  ✗ atom_types_encoding not found in dataset")
    
    # 4. Export functional_groups_encoding.csv
    print("Generating functional_groups_encoding.csv...")
    csv_path = os.path.join(output_dir, 'functional_groups_encoding.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if 'functional_groups_encoding' in combined_data:
            functional_groups_encoding = combined_data['functional_groups_encoding']
            
            # Create header with functional group names
            header = ['ID', '分子の組成']
            if len(functional_groups_mapping) > 0:
                header.extend(functional_groups_mapping)
            else:
                # Fallback: use indices
                n_functional_groups = functional_groups_encoding.shape[1] if len(functional_groups_encoding.shape) > 1 else 1
                header.extend([f'FunctionalGroup_{j}' for j in range(n_functional_groups)])
            writer.writerow(header)
            
            for i in range(n_total):
                mol_id = i
                composition = get_composition_string(charges_or_atomic[i] if charges_or_atomic is not None else None,
                                                    num_atoms[i].item())
                
                # Get encoding values
                if len(functional_groups_encoding.shape) > 1:
                    encoding_values = [f"{val:.0f}" for val in functional_groups_encoding[i].tolist()]
                else:
                    encoding_values = [f"{functional_groups_encoding[i].item():.0f}"]
                
                writer.writerow([mol_id, composition] + encoding_values)
            
            print(f"  ✓ Exported {n_total} molecules to {csv_path}")
        else:
            print(f"  ✗ functional_groups_encoding not found in dataset")
    
    print(f"\n{'='*60}")
    print("CSV export completed successfully!")
    print(f"Files saved in: {output_dir}")
    print(f"{'='*60}\n")
    return datasets