"""
OpenBabel-based molecular processing functions equivalent to RDKit functions.
Used for ASE database datasets.
"""

try:
    import openbabel as ob
    from openbabel import pybel
    openbabel_available = True
except ImportError:
    openbabel_available = False

import numpy as np
import torch
from qm9.bond_analyze import get_bond_order, geom_predictor
import pickle
import os


def mol2smiles_ob(mol):
    """Convert OpenBabel molecule to SMILES string."""
    if not openbabel_available:
        raise ImportError("OpenBabel is required for this function")
    
    try:
        if hasattr(mol, 'write'):
            # pybel.Molecule object
            return mol.write("smi").strip()
        else:
            # OBMol object  
            conv = ob.OBConversion()
            conv.SetOutFormat("smi")
            return conv.WriteString(mol).strip()
    except:
        return None


def build_molecule_ob(positions, atom_types, dataset_info):
    """Build OpenBabel molecule from positions and atom types."""
    if not openbabel_available:
        raise ImportError("OpenBabel is required for this function")
    
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule_ob(positions, atom_types, dataset_info)
    
    mol = ob.OBMol()
    
    # Add atoms
    for i, atom in enumerate(X):
        atom_symbol = atom_decoder[atom.item()]
        atom_num = ob.GetAtomicNum(atom_symbol)
        ob_atom = mol.NewAtom()
        ob_atom.SetAtomicNum(atom_num)
        # Set atom coordinates
        pos = positions[i]
        ob_atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
    
    # Add bonds
    all_bonds = torch.nonzero(A)
    bond_order_map = {1: 1, 2: 2, 3: 3, 4: 5}  # 4 maps to aromatic
    
    for bond in all_bonds:
        i, j = bond[0].item(), bond[1].item()
        bond_order = E[i, j].item()
        if bond_order > 0:
            # OpenBabel uses 1-based indexing for bonds
            mol.AddBond(i+1, j+1, bond_order_map.get(bond_order, 1))
    
    return mol


def build_xae_molecule_ob(positions, atom_types, dataset_info):
    """
    Build molecular representation (X, A, E) using OpenBabel bond perception.
    
    Returns:
        X: atom types (N,)
        A: adjacency matrix (N, N) 
        E: edge types/bond orders (N, N)
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
            else:
                # For ASE DB datasets, use same logic as QM9
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            
            if order > 0:
                A[i, j] = 1
                E[i, j] = order
    
    return X, A, E


class BasicMolecularMetricsOB(object):
    """OpenBabel-based molecular metrics equivalent to RDKit version."""
    
    def __init__(self, dataset_info, dataset_smiles_list=None):
        if not openbabel_available:
            raise ImportError("OpenBabel is required for this class")
            
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # For ASE DB datasets, we don't have pre-computed SMILES
        # Could be computed on-demand if needed
        if dataset_smiles_list is None and 'ase_db' in dataset_info.get('name', ''):
            self.dataset_smiles_list = None

    def compute_validity(self, generated):
        """Compute validity of generated molecules."""
        valid = []

        for graph in generated:
            mol = build_molecule_ob(*graph, self.dataset_info)
            smiles = mol2smiles_ob(mol)
            if smiles is not None and smiles != '':
                # For OpenBabel, check if molecule is valid
                pybel_mol = pybel.Molecule(mol)
                try:
                    # Try to compute some basic properties to check validity
                    _ = pybel_mol.exactmass
                    valid.append(smiles)
                except:
                    pass

        return valid, len(valid) / len(generated) if len(generated) > 0 else 0.0

    def compute_uniqueness(self, valid):
        """Compute uniqueness of valid molecules."""
        unique_smiles = list(set(valid))
        uniqueness = len(unique_smiles) / len(valid) if len(valid) > 0 else 0.0
        return unique_smiles, uniqueness

    def compute_novelty(self, unique):
        """Compute novelty compared to dataset."""
        if self.dataset_smiles_list is None:
            return unique, 1.0  # All considered novel if no reference set
        
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        
        novelty = num_novel / len(unique) if len(unique) > 0 else 0.0
        return novel, novelty

    def evaluate(self, generated):
        """Evaluate generated molecules for validity, uniqueness, and novelty."""
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100:.2f}%")
        
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100:.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100:.2f}%")
            else:
                novelty = 1.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
            
        return [validity, uniqueness, novelty], unique


def compute_ase_db_smiles(dataset_path, dataset_info):
    """
    Compute SMILES for molecules in ASE database.
    
    Args:
        dataset_path: Path to ASE database file
        dataset_info: Dataset configuration dictionary
    
    Returns:
        List of SMILES strings
    """
    if not openbabel_available:
        raise ImportError("OpenBabel is required for this function")
    
    try:
        from ase.db import connect
    except ImportError:
        raise ImportError("ASE is required for this function")
    
    db = connect(dataset_path)
    mols_smiles = []
    
    for i, row in enumerate(db.select()):
        atoms = row.toatoms()
        positions = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # Convert atomic numbers to atom types for dataset
        atomic_numbers = atoms.get_atomic_numbers()
        atom_types = []
        for atomic_num in atomic_numbers:
            atom_symbol = atoms.get_chemical_symbols()[list(atomic_numbers).index(atomic_num)]
            if atom_symbol in dataset_info['atom_encoder']:
                atom_types.append(dataset_info['atom_encoder'][atom_symbol])
            else:
                # Skip molecules with unsupported atoms
                break
        else:
            # Only process if all atoms are supported
            atom_types = torch.tensor(atom_types, dtype=torch.long)
            
            try:
                mol = build_molecule_ob(positions, atom_types, dataset_info)
                smiles = mol2smiles_ob(mol)
                if smiles is not None and smiles != '':
                    mols_smiles.append(smiles)
            except:
                pass
                
        if i % 1000 == 0:
            print(f"\tConverting ASE DB to SMILES {i+1} molecules processed")
    
    return mols_smiles