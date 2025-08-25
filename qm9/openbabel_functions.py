"""
OpenBabel-based molecular functions to replace RDKit functionality.
This module provides molecular operations that work with general molecules
that may not be representable as SMILES.
"""

import numpy as np
import torch
from openbabel import openbabel
import tempfile
import os
from qm9.bond_analyze import get_bond_order, geom_predictor
from typing import List, Tuple, Optional, Any, Dict
import hashlib
import json


class BasicMolecularMetricsOB(object):
    """
    OpenBabel-based molecular metrics that don't rely on SMILES.
    Provides validity, uniqueness, and novelty calculations using molecular fingerprints.
    """
    
    def __init__(self, dataset_info: Dict[str, Any], dataset_fingerprints: Optional[List[str]] = None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_fingerprints = dataset_fingerprints
        self.dataset_info = dataset_info
        
        # Initialize OpenBabel objects
        self.ob_conversion = openbabel.OBConversion()
        self.ob_conversion.SetInAndOutFormats("xyz", "sdf")
        
        # For fingerprint-based comparisons
        self.fptype = openbabel.OBFingerprint.FindFingerprint("FP2")
        
        # Retrieve dataset fingerprints if not provided
        if dataset_fingerprints is None and 'qm9' in dataset_info['name']:
            # For QM9, we could pre-compute fingerprints, but for now we'll skip
            # this to maintain compatibility
            self.dataset_fingerprints = None

    def mol_to_fingerprint(self, mol: openbabel.OBMol) -> Optional[str]:
        """Convert an OpenBabel molecule to a fingerprint string."""
        if mol is None or mol.NumAtoms() == 0:
            return None
        
        try:
            # Create fingerprint
            fingerprint = openbabel.vectorUnsignedInt()
            self.fptype.GetFingerprint(mol, fingerprint)
            
            # Convert to hex string for comparison
            fp_hex = ""
            for i in range(fingerprint.size()):
                fp_hex += f"{fingerprint[i]:08x}"
            
            return fp_hex
        except Exception:
            return None

    def mol_to_hash(self, mol: openbabel.OBMol) -> Optional[str]:
        """Convert molecule to a hash for uniqueness comparison."""
        if mol is None or mol.NumAtoms() == 0:
            return None
        
        try:
            # Use InChI as a canonical representation
            conv = openbabel.OBConversion()
            conv.SetOutFormat("inchi")
            inchi = conv.WriteString(mol).strip()
            
            if inchi and inchi != "":
                # Create hash from InChI
                return hashlib.md5(inchi.encode()).hexdigest()
            else:
                # Fallback to molecular formula hash
                formula = mol.GetFormula()
                return hashlib.md5(formula.encode()).hexdigest()
        except Exception:
            # Last resort: hash the atomic numbers and positions
            try:
                atoms_data = []
                for i in range(mol.NumAtoms()):
                    atom = mol.GetAtom(i + 1)  # OpenBabel uses 1-based indexing
                    atoms_data.append((atom.GetAtomicNum(), 
                                     round(atom.GetX(), 2), 
                                     round(atom.GetY(), 2), 
                                     round(atom.GetZ(), 2)))
                atoms_data.sort()  # Sort for canonical representation
                data_str = str(atoms_data)
                return hashlib.md5(data_str.encode()).hexdigest()
            except Exception:
                return None

    def compute_validity(self, generated: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[str], float]:
        """
        Compute validity of generated molecules.
        
        Args:
            generated: List of (positions, atom_types) tuples
            
        Returns:
            Tuple of (valid_hashes, validity_fraction)
        """
        valid = []
        
        for graph in generated:
            mol = build_molecule_ob(*graph, self.dataset_info)
            if mol is not None and mol.NumAtoms() > 0:
                # Check if molecule is valid by attempting to process it
                mol_hash = self.mol_to_hash(mol)
                if mol_hash is not None:
                    valid.append(mol_hash)
        
        validity = len(valid) / len(generated) if len(generated) > 0 else 0.0
        return valid, validity

    def compute_uniqueness(self, valid: List[str]) -> Tuple[List[str], float]:
        """
        Compute uniqueness among valid molecules.
        
        Args:
            valid: List of molecule hashes
            
        Returns:
            Tuple of (unique_hashes, uniqueness_fraction)
        """
        unique = list(set(valid))
        uniqueness = len(unique) / len(valid) if len(valid) > 0 else 0.0
        return unique, uniqueness

    def compute_novelty(self, unique: List[str]) -> Tuple[List[str], float]:
        """
        Compute novelty compared to dataset.
        
        Args:
            unique: List of unique molecule hashes
            
        Returns:
            Tuple of (novel_hashes, novelty_fraction)
        """
        if self.dataset_fingerprints is None:
            # If no dataset fingerprints available, assume all are novel
            return unique, 1.0
        
        novel = []
        for mol_hash in unique:
            if mol_hash not in self.dataset_fingerprints:
                novel.append(mol_hash)
        
        novelty = len(novel) / len(unique) if len(unique) > 0 else 0.0
        return novel, novelty

    def evaluate(self, generated: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[float], Optional[List[str]]]:
        """
        Evaluate generated molecules for validity, uniqueness, and novelty.
        
        Args:
            generated: List of (positions, atom_types) tuples
            
        Returns:
            Tuple of ([validity, uniqueness, novelty], unique_hashes)
        """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100:.2f}%")
        
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100:.2f}%")
            
            if self.dataset_fingerprints is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100:.2f}%")
            else:
                novelty = 0.0
                print("Dataset fingerprints not available, novelty set to 0.0")
        else:
            uniqueness = 0.0
            novelty = 0.0
            unique = None
            
        return [validity, uniqueness, novelty], unique


def build_molecule_ob(positions: torch.Tensor, atom_types: torch.Tensor, dataset_info: Dict[str, Any]) -> Optional[openbabel.OBMol]:
    """
    Build OpenBabel molecule from positions and atom types.
    
    Args:
        positions: Tensor of shape (n_atoms, 3) with atomic positions
        atom_types: Tensor of shape (n_atoms,) with atom type indices
        dataset_info: Dataset configuration dictionary
        
    Returns:
        OpenBabel OBMol object or None if failed
    """
    atom_decoder = dataset_info["atom_decoder"]
    
    try:
        # Convert tensors to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.detach().cpu().numpy()
        
        # Create OpenBabel molecule
        mol = openbabel.OBMol()
        
        # Add atoms
        for i, atom_type_idx in enumerate(atom_types):
            if atom_type_idx < len(atom_decoder):
                element = atom_decoder[atom_type_idx]
                atom = mol.NewAtom()
                
                # Set atomic number based on element symbol
                atomic_num = openbabel.GetAtomicNum(element)
                atom.SetAtomicNum(atomic_num)
                
                # Set position
                pos = positions[i]
                atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
        
        # Build bonds using same logic as original
        X, A, E = build_xae_molecule_ob(positions, atom_types, dataset_info)
        
        # Add bonds to molecule
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                if A[i, j]:
                    bond_order = int(E[i, j])
                    if bond_order > 0:
                        mol.AddBond(i + 1, j + 1, bond_order)  # OpenBabel uses 1-based indexing
        
        # Finalize molecule
        mol.PerceiveBondOrders()
        mol.ConnectTheDots()
        
        return mol
        
    except Exception as e:
        print(f"Error building molecule: {e}")
        return None


def build_xae_molecule_ob(positions: np.ndarray, atom_types: np.ndarray, dataset_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build adjacency and edge matrices for molecule using OpenBabel-compatible approach.
    
    Args:
        positions: Array of shape (n_atoms, 3) with atomic positions
        atom_types: Array of shape (n_atoms,) with atom type indices
        dataset_info: Dataset configuration dictionary
        
    Returns:
        Tuple of (X, A, E) where:
        - X: atom types (n_atoms,)
        - A: adjacency matrix (n_atoms, n_atoms) - boolean
        - E: edge types matrix (n_atoms, n_atoms) - int
    """
    atom_decoder = dataset_info['atom_decoder']
    n = len(positions)
    
    X = atom_types
    A = np.zeros((n, n), dtype=bool)
    E = np.zeros((n, n), dtype=int)
    
    # Calculate distances
    for i in range(n):
        for j in range(i + 1, n):
            pos_i = positions[i]
            pos_j = positions[j]
            dist = np.linalg.norm(pos_i - pos_j)
            
            # Get atom types
            atom_i = atom_decoder[atom_types[i]]
            atom_j = atom_decoder[atom_types[j]]
            
            # Determine bond order based on dataset type
            if dataset_info['name'] in ['qm9', 'qm9_second_half', 'qm9_first_half']:
                order = get_bond_order(atom_i, atom_j, dist)
            elif dataset_info['name'] == 'geom':
                pair = (atom_i, atom_j)
                order = geom_predictor(pair, dist, limit_bonds_to_one=True)
            else:
                # For general molecules, use a simple distance-based approach
                order = predict_bond_order_general(atom_i, atom_j, dist)
            
            if order > 0:
                A[i, j] = True
                A[j, i] = True  # Make symmetric
                E[i, j] = order
                E[j, i] = order
                
    return X, A, E


def predict_bond_order_general(atom1: str, atom2: str, distance: float) -> int:
    """
    Predict bond order for general molecules based on atomic radii and distance.
    This is a simple heuristic that can be improved with more sophisticated methods.
    
    Args:
        atom1: Element symbol of first atom
        atom2: Element symbol of second atom  
        distance: Distance between atoms in Angstroms
        
    Returns:
        Bond order (0=no bond, 1=single, 2=double, 3=triple)
    """
    # Covalent radii in Angstroms (simplified set)
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39,
        'B': 0.84, 'Si': 1.11, 'Al': 1.21, 'As': 1.19, 'Hg': 1.32, 'Bi': 1.48
    }
    
    # Get covalent radii, use default if not found
    r1 = covalent_radii.get(atom1, 1.0)
    r2 = covalent_radii.get(atom2, 1.0)
    
    # Expected single bond distance
    single_bond_dist = r1 + r2
    
    # Tolerance factors for different bond orders
    tolerance = 0.4  # Angstroms
    
    if distance <= single_bond_dist + tolerance:
        # Determine bond order based on distance
        if distance <= single_bond_dist * 0.8:
            return 3  # Triple bond
        elif distance <= single_bond_dist * 0.9:
            return 2  # Double bond
        else:
            return 1  # Single bond
    else:
        return 0  # No bond


def mol_to_xyz_string(mol: openbabel.OBMol) -> Optional[str]:
    """
    Convert OpenBabel molecule to XYZ format string.
    
    Args:
        mol: OpenBabel molecule
        
    Returns:
        XYZ format string or None if failed
    """
    if mol is None or mol.NumAtoms() == 0:
        return None
        
    try:
        conv = openbabel.OBConversion()
        conv.SetOutFormat("xyz")
        return conv.WriteString(mol)
    except Exception:
        return None


def mol_to_sdf_string(mol: openbabel.OBMol) -> Optional[str]:
    """
    Convert OpenBabel molecule to SDF format string.
    
    Args:
        mol: OpenBabel molecule
        
    Returns:
        SDF format string or None if failed  
    """
    if mol is None or mol.NumAtoms() == 0:
        return None
        
    try:
        conv = openbabel.OBConversion()
        conv.SetOutFormat("sdf")
        return conv.WriteString(mol)
    except Exception:
        return None


# Backward compatibility function names
def mol2smiles(mol):
    """
    Backward compatibility function that returns None for OpenBabel molecules.
    Since we're moving away from SMILES, this returns None.
    """
    return None


def build_molecule(positions, atom_types, dataset_info):
    """
    Backward compatibility wrapper for build_molecule_ob.
    """
    return build_molecule_ob(positions, atom_types, dataset_info)


def build_xae_molecule(positions, atom_types, dataset_info):
    """
    Backward compatibility wrapper for build_xae_molecule_ob.
    """
    # Convert torch tensors to numpy if needed
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.detach().cpu().numpy()
        
    X, A, E = build_xae_molecule_ob(positions, atom_types, dataset_info)
    
    # Convert back to torch tensors for compatibility
    return torch.tensor(X), torch.tensor(A), torch.tensor(E)