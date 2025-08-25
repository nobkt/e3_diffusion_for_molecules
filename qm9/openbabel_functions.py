"""
OpenBabel-based replacements for RDKit functionality.
This module provides molecular analysis functions using OpenBabel instead of RDKit.
"""

import numpy as np
import torch
from openbabel import pybel, openbabel
from typing import List, Tuple, Optional, Union
import logging
import os

# Try to import bond analysis functions, provide fallback if not available
try:
    from qm9.bond_analyze import get_bond_order, geom_predictor
    BOND_ANALYSIS_AVAILABLE = True
except ImportError:
    logging.warning("Bond analysis module not available, using distance-based bond prediction")
    BOND_ANALYSIS_AVAILABLE = False
    get_bond_order = None
    geom_predictor = None


class OpenBabelMolecularMetrics:
    """OpenBabel-based molecular metrics calculator (replaces BasicMolecularMetrics)."""
    
    def __init__(self, dataset_info, dataset_smiles_list=None):
        """
        Initialize OpenBabel molecular metrics.
        
        Parameters
        ----------
        dataset_info : dict
            Dataset information containing atom decoder
        dataset_smiles_list : list, optional
            List of dataset SMILES for novelty calculation
        """
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info
        
        # If dataset SMILES list is not provided and it's QM9, try to compute it
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            try:
                self.dataset_smiles_list = self._compute_qm9_smiles_openbabel(dataset_info)
            except Exception as e:
                logging.warning(f"Could not compute QM9 SMILES with OpenBabel: {e}")
                self.dataset_smiles_list = []
    
    def _compute_qm9_smiles_openbabel(self, dataset_info):
        """Compute QM9 SMILES using OpenBabel instead of RDKit."""
        # This would need access to the actual QM9 dataset
        # For now, return empty list - this can be populated when needed
        logging.info("Computing QM9 SMILES with OpenBabel - placeholder implementation")
        return []
    
    def compute_validity(self, generated):
        """
        Compute validity of generated molecules.
        
        Parameters
        ----------
        generated : list
            List of tuples (positions, atom_types)
            
        Returns
        -------
        valid : list
            List of valid SMILES strings
        validity : float
            Fraction of valid molecules
        """
        valid = []
        for positions, atom_types in generated:
            try:
                mol = build_molecule_openbabel(positions, atom_types, self.dataset_info)
                smiles = mol2smiles_openbabel(mol)
                if smiles is not None:
                    valid.append(smiles)
            except Exception as e:
                logging.debug(f"Failed to validate molecule: {e}")
                continue
        
        validity = len(valid) / len(generated) if generated else 0.0
        return valid, validity
    
    def compute_uniqueness(self, valid):
        """
        Compute uniqueness of valid molecules.
        
        Parameters
        ----------
        valid : list
            List of valid SMILES strings
            
        Returns
        -------
        unique : list
            List of unique SMILES strings
        uniqueness : float
            Fraction of unique molecules among valid ones
        """
        unique = list(set(valid))
        uniqueness = len(unique) / len(valid) if valid else 0.0
        return unique, uniqueness
    
    def compute_novelty(self, unique):
        """
        Compute novelty of unique molecules.
        
        Parameters
        ----------
        unique : list
            List of unique SMILES strings
            
        Returns
        -------
        novel : list
            List of novel SMILES strings
        novelty : float
            Fraction of novel molecules among unique ones
        """
        if not self.dataset_smiles_list:
            logging.warning("No dataset SMILES available for novelty calculation")
            return unique, 1.0
        
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
        
        novelty = len(novel) / len(unique) if unique else 0.0
        return novel, novelty
    
    def evaluate(self, generated):
        """
        Evaluate generated molecules for validity, uniqueness, and novelty.
        
        Parameters
        ----------
        generated : list
            List of tuples (positions, atom_types)
            
        Returns
        -------
        results : dict
            Dictionary containing evaluation metrics
        """
        valid, validity = self.compute_validity(generated)
        unique, uniqueness = self.compute_uniqueness(valid)
        novel, novelty = self.compute_novelty(unique)
        
        results = {
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'valid_smiles': valid,
            'unique_smiles': unique,
            'novel_smiles': novel
        }
        
        logging.info(f"Evaluation results: Validity={validity:.3f}, "
                    f"Uniqueness={uniqueness:.3f}, Novelty={novelty:.3f}")
        
        return results


def mol2smiles_openbabel(mol) -> Optional[str]:
    """
    Convert OpenBabel molecule to SMILES string.
    
    Parameters
    ----------
    mol : pybel.Molecule
        OpenBabel molecule object
        
    Returns
    -------
    smiles : str or None
        SMILES string representation of the molecule
    """
    try:
        if mol is None:
            return None
        
        # Convert to SMILES
        smiles = mol.write("smi").strip()
        
        # Basic validation - check if SMILES is non-empty and valid
        if smiles and len(smiles) > 0:
            # Try to parse it back to validate
            test_mol = pybel.readstring("smi", smiles)
            if test_mol:
                return smiles
        
        return None
    except Exception as e:
        logging.debug(f"Failed to convert molecule to SMILES: {e}")
        return None


def build_molecule_openbabel(positions, atom_types, dataset_info):
    """
    Build OpenBabel molecule from positions and atom types.
    
    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions (N x 3)
    atom_types : torch.Tensor  
        Atomic types (N,)
    dataset_info : dict
        Dataset information containing atom decoder
        
    Returns
    -------
    mol : pybel.Molecule
        OpenBabel molecule object
    """
    try:
        atom_decoder = dataset_info["atom_decoder"]
        
        # Convert tensors to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.detach().cpu().numpy()
        
        # Create OpenBabel molecule
        mol = openbabel.OBMol()
        mol.BeginModify()
        
        # Add atoms
        atom_indices = {}
        valid_atoms = 0
        for i, atom_type in enumerate(atom_types):
            if atom_type == 0:  # Skip padding
                continue
                
            if atom_type not in atom_decoder:
                logging.warning(f"Unknown atom type: {atom_type}")
                continue
                
            symbol = atom_decoder[atom_type]
            
            # Create atom
            atom = mol.NewAtom()
            atom_num = openbabel.GetAtomicNum(symbol)
            if atom_num == 0:
                logging.warning(f"Unknown element symbol: {symbol}")
                continue
                
            atom.SetAtomicNum(atom_num)
            atom.SetVector(float(positions[i][0]), float(positions[i][1]), float(positions[i][2]))
            
            atom_indices[i] = atom.GetIdx()
            valid_atoms += 1
        
        if valid_atoms == 0:
            logging.warning("No valid atoms found")
            return None
        
        mol.EndModify()
        
        # Add bonds based on distance and bond order prediction
        try:
            X, A, E = build_xae_molecule_openbabel(positions, atom_types, dataset_info)
            
            mol.BeginModify()
            bonds_added = 0
            for i in range(len(atom_types)):
                for j in range(i + 1, len(atom_types)):
                    if i in atom_indices and j in atom_indices and A[i, j]:
                        bond_order = int(E[i, j])
                        if bond_order > 0:
                            # OpenBabel uses 1-based indexing
                            mol.AddBond(atom_indices[i], atom_indices[j], bond_order)
                            bonds_added += 1
            
            mol.EndModify()
            logging.debug(f"Added {bonds_added} bonds to molecule")
            
        except Exception as e:
            logging.warning(f"Failed to add bonds: {e}")
            # Continue without bonds
        
        # Convert to pybel molecule for easier manipulation
        pybel_mol = pybel.Molecule(mol)
        
        # Basic validation - check if molecule has atoms
        if pybel_mol.OBMol.NumAtoms() == 0:
            logging.warning("Created molecule has no atoms")
            return None
        
        return pybel_mol
        
    except Exception as e:
        logging.warning(f"Failed to build molecule: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None


def build_xae_molecule_openbabel(positions, atom_types, dataset_info):
    """
    Build molecule adjacency and edge information using OpenBabel-compatible approach.
    
    Parameters
    ----------
    positions : torch.Tensor or np.ndarray
        Atomic positions (N x 3)
    atom_types : torch.Tensor or np.ndarray
        Atomic types (N,)
    dataset_info : dict
        Dataset information containing atom decoder
        
    Returns
    -------
    X : torch.Tensor
        Atom types (N,)
    A : torch.Tensor
        Adjacency matrix (N x N, bool)
    E : torch.Tensor
        Edge types/bond orders (N x N, int)
    """
    # Convert to tensors if needed
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).float()
    if isinstance(atom_types, np.ndarray):
        atom_types = torch.from_numpy(atom_types).long()
    
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)
    
    # Calculate pairwise distances
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    
    # Determine bonds based on distances and atom types
    for i in range(n):
        if atom_types[i] == 0:  # Skip padding
            continue
        for j in range(i + 1, n):
            if atom_types[j] == 0:  # Skip padding
                continue
                
            # Get bond order based on atom types and distance
            atom_pair = sorted([atom_types[i].item(), atom_types[j].item()])
            
            if BOND_ANALYSIS_AVAILABLE:
                if dataset_info['name'] in ['qm9', 'qm9_second_half', 'qm9_first_half']:
                    order = get_bond_order(atom_decoder[atom_pair[0]], atom_decoder[atom_pair[1]], dists[i, j].item())
                elif dataset_info['name'] == 'geom':
                    order = geom_predictor((atom_decoder[atom_pair[0]], atom_decoder[atom_pair[1]]), dists[i, j].item(), limit_bonds_to_one=True)
                else:
                    # Default bond order prediction based on distance
                    order = _predict_bond_order_distance(atom_decoder[atom_pair[0]], atom_decoder[atom_pair[1]], dists[i, j].item())
            else:
                # Use distance-based bond order prediction as fallback
                order = _predict_bond_order_distance(atom_decoder[atom_pair[0]], atom_decoder[atom_pair[1]], dists[i, j].item())
            
            if order > 0:
                # Set symmetric adjacency and edge information
                A[i, j] = A[j, i] = True
                E[i, j] = E[j, i] = order
    
    return X, A, E


def _predict_bond_order_distance(atom1: str, atom2: str, distance: float) -> int:
    """
    Predict bond order based on atomic symbols and distance.
    
    Parameters
    ----------
    atom1, atom2 : str
        Atomic symbols
    distance : float
        Distance between atoms in Angstroms
        
    Returns
    -------
    bond_order : int
        Predicted bond order (0 = no bond, 1 = single, 2 = double, 3 = triple)
    """
    # Approximate covalent radii (in Angstroms)
    covalent_radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57}
    
    # Get covalent radii
    r1 = covalent_radii.get(atom1, 0.8)
    r2 = covalent_radii.get(atom2, 0.8)
    
    # Expected single bond distance
    single_bond_dist = r1 + r2
    
    # Tolerance for bond detection
    tolerance = 0.4
    
    if distance <= single_bond_dist + tolerance:
        # Determine bond order based on distance
        if distance <= single_bond_dist * 0.85:
            return 2  # Double bond (shorter)
        elif distance <= single_bond_dist * 0.75:
            return 3  # Triple bond (very short)
        else:
            return 1  # Single bond
    
    return 0  # No bond


def smiles_to_molecule_openbabel(smiles: str):
    """
    Convert SMILES string to OpenBabel molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string
        
    Returns
    -------
    mol : pybel.Molecule or None
        OpenBabel molecule object
    """
    try:
        mol = pybel.readstring("smi", smiles)
        return mol
    except Exception as e:
        logging.debug(f"Failed to parse SMILES {smiles}: {e}")
        return None


def compute_qm9_smiles_openbabel(dataset_name: str, remove_h: bool = False):
    """
    Compute QM9 SMILES using OpenBabel (replacement for RDKit version).
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    remove_h : bool
        Whether to remove hydrogen atoms
        
    Returns
    -------
    mols_smiles : list
        List of SMILES strings
    """
    print("\tConverting QM9 dataset to SMILES using OpenBabel...")
    
    # This function would need to interface with the dataset loading
    # For now, return empty list as placeholder
    logging.warning("compute_qm9_smiles_openbabel is a placeholder - needs dataset integration")
    return []


def retrieve_qm9_smiles_openbabel(dataset_info):
    """
    Retrieve QM9 SMILES using OpenBabel (replacement for RDKit version).
    
    Parameters
    ----------
    dataset_info : dict
        Dataset information
        
    Returns
    -------
    qm9_smiles : list
        List of QM9 SMILES strings
    """
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + '_noH'
    
    file_name = f'qm9/temp/{pickle_name}_smiles_openbabel.pickle'
    
    try:
        import pickle
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('qm9/temp', exist_ok=True)
        except:
            pass
        
        qm9_smiles = compute_qm9_smiles_openbabel(dataset_name, remove_h=not dataset_info['with_h'])
        
        try:
            import pickle
            with open(file_name, 'wb') as f:
                pickle.dump(qm9_smiles, f)
        except Exception as e:
            logging.warning(f"Could not save SMILES to {file_name}: {e}")
        
        return qm9_smiles


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test OpenBabel molecule building
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])  # Simple diatomic
    atom_types = torch.tensor([6, 6])  # Two carbon atoms
    dataset_info = {'atom_decoder': {6: 'C', 1: 'H'}, 'name': 'test'}
    
    mol = build_molecule_openbabel(positions, atom_types, dataset_info)
    if mol:
        smiles = mol2smiles_openbabel(mol)
        print(f"Generated SMILES: {smiles}")
    else:
        print("Failed to build molecule")