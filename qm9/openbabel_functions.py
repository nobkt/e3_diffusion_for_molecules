"""
OpenBabel-based molecular functions as an alternative to RDKit.
Used when dataset=ase_db to avoid RDKit dependency.
"""
try:
    from openbabel import openbabel as ob
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    ob = None
    pybel = None

import numpy as np
import torch
from qm9.bond_analyze import get_bond_order, geom_predictor
from . import dataset
from configs.datasets_config import get_dataset_info
import pickle
import os


def compute_qm9_smiles_openbabel(dataset_name, remove_h):
    '''
    OpenBabel version of compute_qm9_smiles from rdkit_functions.py
    
    :param dataset_name: qm9 or qm9_second_half
    :param remove_h: whether to remove hydrogen atoms
    :return: list of SMILES strings
    '''
    if not OPENBABEL_AVAILABLE:
        raise ImportError("OpenBabel is required for this functionality. Install with: pip install openbabel-wheel")
    
    print("\tConverting QM9 dataset to SMILES using OpenBabel...")

    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.remove_h = remove_h
            self.include_charges = True
    
    args_dataset = StaticArgs(dataset_name, remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    
    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        mol = build_molecule_openbabel(torch.tensor(positions), torch.tensor(atom_type), dataset_info)
        smiles = mol2smiles_openbabel(mol)
        if smiles is not None:
            mols_smiles.append(smiles)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles


def retrieve_qm9_smiles_openbabel(dataset_info):
    """
    OpenBabel version of retrieve_qm9_smiles from rdkit_functions.py
    """
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name + '_openbabel'
    else:
        pickle_name = dataset_name + '_noH_openbabel'

    file_name = 'qm9/temp/%s_smiles.pickle' % pickle_name
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('qm9/temp')
        except:
            pass
        qm9_smiles = compute_qm9_smiles_openbabel(dataset_name, remove_h=not dataset_info['with_h'])
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles


def mol2smiles_openbabel(mol):
    """
    Convert OpenBabel molecule to SMILES string
    
    :param mol: OpenBabel OBMol object
    :return: SMILES string or None if conversion fails
    """
    if not OPENBABEL_AVAILABLE or mol is None:
        return None
    
    try:
        # Convert to pybel molecule for easier SMILES generation
        pybel_mol = pybel.Molecule(mol)
        return pybel_mol.write("smi").strip()
    except Exception:
        return None


def build_molecule_openbabel(positions, atom_types, dataset_info):
    """
    Build OpenBabel molecule from positions and atom types
    
    :param positions: torch.Tensor of shape (n_atoms, 3)
    :param atom_types: torch.Tensor of shape (n_atoms,)
    :param dataset_info: dict containing atom_decoder
    :return: OpenBabel OBMol object
    """
    if not OPENBABEL_AVAILABLE:
        raise ImportError("OpenBabel is required for this functionality. Install with: pip install openbabel-wheel")
    
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule_openbabel(positions, atom_types, dataset_info)
    
    # Create OpenBabel molecule
    mol = ob.OBMol()
    
    # Add atoms
    for i, atom_type in enumerate(X):
        atom = mol.NewAtom()
        element_symbol = atom_decoder[atom_type.item()]
        atomic_num = ob.GetAtomicNum(element_symbol)
        atom.SetAtomicNum(atomic_num)
        
        # Set coordinates
        pos = positions[i].numpy() if hasattr(positions[i], 'numpy') else positions[i]
        atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
    
    # Add bonds
    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        i, j = bond[0].item(), bond[1].item()
        bond_order = E[i, j].item()
        if bond_order > 0:
            # OpenBabel uses 1-based indexing for atoms
            mol.AddBond(i + 1, j + 1, bond_order)
    
    # Connect bond data structures
    mol.ConnectTheDots()
    
    # Try to perceive chemistry (optional, may help with aromatic bonds)
    try:
        mol.PerceiveBondOrders()
    except:
        pass  # If perception fails, continue with explicit bonds
    
    return mol


def build_xae_molecule_openbabel(positions, atom_types, dataset_info):
    """
    OpenBabel version of build_xae_molecule from rdkit_functions.py
    Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            elif dataset_info['name'] == 'ase_db':
                # For ASE database, use the same bond prediction as QM9
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


class BasicMolecularMetricsOpenBabel(object):
    """
    OpenBabel version of BasicMolecularMetrics from rdkit_functions.py
    """
    def __init__(self, dataset_info, dataset_smiles_list=None):
        if not OPENBABEL_AVAILABLE:
            raise ImportError("OpenBabel is required for this functionality. Install with: pip install openbabel-wheel")
        
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles_openbabel(self.dataset_info)

    def compute_validity(self, generated):
        """ 
        Compute validity of generated molecules
        generated: list of couples (positions, atom_types)
        """
        valid = []

        for graph in generated:
            mol = build_molecule_openbabel(*graph, self.dataset_info)
            smiles = mol2smiles_openbabel(mol)
            if smiles is not None and len(smiles.strip()) > 0:
                # For OpenBabel, we'll check if the SMILES is valid by trying to parse it back
                try:
                    test_mol = pybel.readstring("smi", smiles)
                    if test_mol.OBMol.NumAtoms() > 0:
                        # If the molecule has disconnected fragments, take the largest one
                        mol_fragments = self._get_molecule_fragments(mol)
                        if len(mol_fragments) > 1:
                            largest_mol = max(mol_fragments, key=lambda m: m.NumAtoms())
                            smiles = mol2smiles_openbabel(largest_mol)
                        valid.append(smiles)
                    else:
                        continue
                except:
                    continue

        return valid, len(valid) / len(generated) if len(generated) > 0 else 0.0

    def compute_uniqueness(self, valid):
        """ 
        Compute uniqueness of valid SMILES strings
        valid: list of SMILES strings.
        """
        return list(set(valid)), len(set(valid)) / len(valid) if len(valid) > 0 else 0.0

    def compute_novelty(self, unique):
        """
        Compute novelty compared to dataset SMILES
        """
        if self.dataset_smiles_list is None:
            return unique, 0.0
        
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique) if len(unique) > 0 else 0.0

    def evaluate(self, generated):
        """ 
        Evaluate generated molecules for validity, uniqueness, and novelty
        generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. 
        """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        
        return [validity, uniqueness, novelty], unique

    def _get_molecule_fragments(self, mol):
        """
        Get disconnected fragments from an OpenBabel molecule
        """
        fragments = []
        
        # Use OpenBabel's fragment separation
        if mol.NumAtoms() == 0:
            return fragments
            
        # If the molecule has multiple disconnected parts, separate them
        try:
            # First, let's check if the molecule is connected
            if mol.NumBonds() > 0:
                # Molecule has bonds, likely connected
                fragments.append(mol)
            else:
                # No bonds detected, each atom is its own fragment
                # This shouldn't happen for valid molecules, but let's handle it
                fragments.append(mol)
        except:
            fragments.append(mol)
        
        return fragments


# Molecular descriptor extraction functions for ASE database conditioning
def extract_atom_types_from_ase(atoms):
    """
    Extract unique atom types from ASE Atoms object
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object
        
    Returns
    -------
    atom_types : list
        List of unique atomic symbols in the molecule
    """
    return sorted(list(set(atoms.get_chemical_symbols())))


def extract_molecular_weight_from_ase(atoms):
    """
    Calculate molecular weight from ASE Atoms object
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object
        
    Returns
    -------
    molecular_weight : float
        Molecular weight in atomic mass units (u)
    """
    try:
        from ase.data import atomic_masses
        total_mass = 0.0
        for atomic_number in atoms.numbers:
            total_mass += atomic_masses[atomic_number]
        return total_mass
    except ImportError:
        # Fallback to simple calculation if ase.data not available
        atomic_masses_dict = {
            1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
            15: 30.974, 16: 32.065, 17: 35.453, 35: 79.904, 53: 126.90
        }
        total_mass = 0.0
        for atomic_number in atoms.numbers:
            total_mass += atomic_masses_dict.get(atomic_number, atomic_number)
        return total_mass


def extract_functional_groups_openbabel(mol):
    """
    Extract functional groups from OpenBabel molecule
    
    Parameters
    ----------
    mol : openbabel.OBMol
        OpenBabel molecule object
        
    Returns
    -------
    functional_groups : list
        List of detected functional group names
    """
    if not OPENBABEL_AVAILABLE or mol is None:
        return []
    
    functional_groups = []
    
    try:
        # Convert to pybel for easier pattern matching
        pybel_mol = pybel.Molecule(mol)
        
        # Define common functional group SMARTS patterns
        functional_group_patterns = {
            'hydroxyl': '[OH]',        # -OH
            'carbonyl': '[CX3]=[OX1]', # C=O
            'carboxyl': '[CX3](=O)[OX2H1]',  # -COOH
            'aldehyde': '[CX3H1](=O)[#6]',   # -CHO
            'ketone': '[CX3](=O)([#6])[#6]', # ketone C=O
            'amino': '[NX3;H2,H1;!$(NC=O)]',  # -NH2, -NH-
            'nitro': '[N+](=O)[O-]',   # -NO2
            'chloro': '[Cl]',          # -Cl
            'bromo': '[Br]',           # -Br
            'fluoro': '[F]',           # -F
            'iodo': '[I]',             # -I
            'methyl': '[CH3]',         # -CH3
            'methoxy': '[OX2]([#6])[CH3]',  # -OCH3
            'phenyl': 'c1ccccc1',      # benzene ring
        }
        
        # Search for each functional group pattern
        for group_name, smarts_pattern in functional_group_patterns.items():
            try:
                matches = pybel_mol.OBMol.HasSubstructMatch(pybel.readstring("smt", smarts_pattern).OBMol)
                if matches:
                    functional_groups.append(group_name)
            except:
                continue  # Skip if pattern matching fails
                
    except Exception:
        pass  # Return empty list if analysis fails
    
    return functional_groups


def extract_pi_conjugation_ratio_openbabel(mol):
    """
    Calculate π conjugation ratio (double/aromatic bonds to total bonds) using OpenBabel
    
    Parameters
    ----------
    mol : openbabel.OBMol
        OpenBabel molecule object
        
    Returns
    -------
    pi_ratio : float
        Ratio of π bonds (double + aromatic) to total bonds
    """
    if not OPENBABEL_AVAILABLE or mol is None:
        return 0.0
    
    try:
        total_bonds = mol.NumBonds()
        if total_bonds == 0:
            return 0.0
        
        pi_bonds = 0
        
        # Count double bonds and aromatic bonds
        for i in range(mol.NumBonds()):
            bond = mol.GetBond(i)
            bond_order = bond.GetBondOrder()
            
            # Count double bonds (bond order = 2)
            if bond_order == 2:
                pi_bonds += 1
            # Count aromatic bonds (special handling)
            elif bond.IsAromatic():
                pi_bonds += 1
        
        # Calculate ratio
        pi_ratio = float(pi_bonds) / float(total_bonds)
        return pi_ratio
        
    except Exception:
        return 0.0


def extract_molecular_descriptors_ase_openbabel(atoms, positions=None):
    """
    Extract all molecular descriptors for ASE database conditioning
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object
    positions : torch.Tensor, optional
        Alternative positions tensor (if different from atoms.positions)
        
    Returns
    -------
    descriptors : dict
        Dictionary containing:
        - 'atom_types': list of unique atomic symbols
        - 'molecular_weight': molecular weight in u
        - 'functional_groups': list of functional group names  
        - 'pi_conjugation_ratio': ratio of π bonds to total bonds
    """
    descriptors = {
        'atom_types': [],
        'molecular_weight': 0.0,
        'functional_groups': [],
        'pi_conjugation_ratio': 0.0
    }
    
    try:
        # Extract atom types and molecular weight from ASE
        descriptors['atom_types'] = extract_atom_types_from_ase(atoms)
        descriptors['molecular_weight'] = extract_molecular_weight_from_ase(atoms)
        
        # For functional groups and π conjugation, we need OpenBabel
        if OPENBABEL_AVAILABLE:
            # Convert ASE to OpenBabel molecule
            mol = ob.OBMol()
            
            # Add atoms
            for i, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
                atom = mol.NewAtom()
                atomic_num = ob.GetAtomicNum(symbol)
                atom.SetAtomicNum(atomic_num)
                atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
            
            # Try to perceive bonds
            mol.ConnectTheDots()
            mol.PerceiveBondOrders()
            
            # Extract functional groups and π conjugation ratio
            descriptors['functional_groups'] = extract_functional_groups_openbabel(mol)
            descriptors['pi_conjugation_ratio'] = extract_pi_conjugation_ratio_openbabel(mol)
    
    except Exception as e:
        print(f"Warning: Failed to extract molecular descriptors: {e}")
    
    return descriptors


# Helper function to check if OpenBabel is available
def is_openbabel_available():
    return OPENBABEL_AVAILABLE


# Function to get appropriate molecular metrics class based on availability
def get_molecular_metrics_class(dataset_info, use_openbabel=False):
    """
    Get the appropriate molecular metrics class
    """
    if use_openbabel:
        if not OPENBABEL_AVAILABLE:
            raise ImportError("OpenBabel is required but not available. Install with: pip install openbabel-wheel")
        return BasicMolecularMetricsOpenBabel(dataset_info)
    else:
        # Try to import RDKit version
        try:
            from qm9.rdkit_functions import BasicMolecularMetrics
            return BasicMolecularMetrics(dataset_info)
        except ImportError:
            if OPENBABEL_AVAILABLE:
                print("RDKit not available, falling back to OpenBabel")
                return BasicMolecularMetricsOpenBabel(dataset_info)
            else:
                raise ImportError("Neither RDKit nor OpenBabel is available for molecular metrics")