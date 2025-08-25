"""
OpenBabel-based molecular processing functions.
Provides alternatives to RDKit functions using OpenBabel for broader molecular support.
"""

import numpy as np
import torch
from openbabel import openbabel as ob
from openbabel import pybel
from qm9.bond_analyze import get_bond_order, geom_predictor
import logging


class OpenBabelMolecularMetrics:
    """OpenBabel-based molecular metrics calculation."""
    
    def __init__(self, dataset_info, dataset_smiles_list=None):
        """
        Initialize OpenBabel molecular metrics.
        
        Args:
            dataset_info: Dataset configuration dictionary
            dataset_smiles_list: Optional list of reference SMILES for comparison
        """
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_info = dataset_info
        self.dataset_smiles_list = dataset_smiles_list
        
    def mol_to_smiles(self, mol):
        """Convert OpenBabel molecule to SMILES string."""
        try:
            return mol.write("smi").strip()
        except Exception as e:
            logging.warning(f"Failed to convert molecule to SMILES: {e}")
            return None
            
    def smiles_to_mol(self, smiles):
        """Convert SMILES string to OpenBabel molecule."""
        try:
            mol = pybel.readstring("smi", smiles)
            return mol
        except Exception as e:
            logging.warning(f"Failed to parse SMILES {smiles}: {e}")
            return None
            
    def build_molecule_from_positions(self, positions, atom_types, dataset_info):
        """
        Build OpenBabel molecule from atomic positions and types.
        
        Args:
            positions: Atomic positions (N, 3)
            atom_types: Atomic type indices (N,)
            dataset_info: Dataset configuration
            
        Returns:
            OpenBabel molecule object
        """
        atom_decoder = dataset_info["atom_decoder"]
        
        # Create OpenBabel molecule
        mol = ob.OBMol()
        
        # Add atoms
        for i, atom_type_idx in enumerate(atom_types):
            if atom_type_idx.item() < len(atom_decoder):
                symbol = atom_decoder[atom_type_idx.item()]
                atom = mol.NewAtom()
                atom.SetAtomicNum(ob.GetAtomicNum(symbol))
                pos = positions[i]
                atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
                
        # Determine bonds using distance-based approach
        self._add_bonds_from_distances(mol, positions, atom_types, dataset_info)
        
        # Convert to pybel for easier manipulation
        pybel_mol = pybel.Molecule(mol)
        return pybel_mol
        
    def _add_bonds_from_distances(self, mol, positions, atom_types, dataset_info):
        """Add bonds to molecule based on atomic distances."""
        atom_decoder = dataset_info["atom_decoder"]
        n_atoms = len(atom_types)
        
        # Calculate distance matrix
        pos_np = positions.detach().cpu().numpy() if hasattr(positions, 'detach') else positions
        dists = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=2)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if atom_types[i].item() >= len(atom_decoder) or atom_types[j].item() >= len(atom_decoder):
                    continue
                    
                # Determine bond order based on distance and atom types
                atom1_symbol = atom_decoder[atom_types[i].item()]
                atom2_symbol = atom_decoder[atom_types[j].item()]
                distance = dists[i, j]
                
                # Use existing bond analysis functions
                if dataset_info['name'] in ['qm9', 'qm9_second_half', 'qm9_first_half']:
                    bond_order = get_bond_order(atom1_symbol, atom2_symbol, distance)
                elif dataset_info['name'] == 'geom':
                    bond_order = geom_predictor((atom1_symbol, atom2_symbol), distance, limit_bonds_to_one=True)
                else:
                    # Generic distance-based bonding
                    bond_order = self._generic_bond_order(atom1_symbol, atom2_symbol, distance)
                    
                if bond_order > 0:
                    # Add bond to molecule
                    mol.AddBond(i + 1, j + 1, int(bond_order))  # OpenBabel uses 1-based indexing
                    
    def _generic_bond_order(self, atom1, atom2, distance):
        """Generic bond order determination for unknown datasets."""
        # Simple distance-based bonding rules
        covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39
        }
        
        r1 = covalent_radii.get(atom1, 0.77)
        r2 = covalent_radii.get(atom2, 0.77)
        bond_threshold = (r1 + r2) * 1.3  # Allow some flexibility
        
        if distance < bond_threshold:
            # Estimate bond order based on distance
            if distance < (r1 + r2) * 0.9:
                return 2  # Likely double/triple bond
            else:
                return 1  # Single bond
        return 0
        
    def build_xae_molecule(self, positions, atom_types, dataset_info):
        """
        Build molecule representation for analysis (X, A, E format).
        
        Args:
            positions: Atomic positions
            atom_types: Atomic type indices
            dataset_info: Dataset configuration
            
        Returns:
            X: Atom types array
            A: Adjacency matrix  
            E: Edge/bond type matrix
        """
        atom_decoder = dataset_info["atom_decoder"]
        n_atoms = len(atom_types)
        
        X = atom_types.clone() if hasattr(atom_types, 'clone') else torch.tensor(atom_types)
        A = torch.zeros(n_atoms, n_atoms)
        E = torch.zeros(n_atoms, n_atoms)
        
        # Calculate distances
        if hasattr(positions, 'detach'):
            pos_np = positions.detach().cpu().numpy()
        else:
            pos_np = np.array(positions)
            
        dists = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=2)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if atom_types[i].item() >= len(atom_decoder) or atom_types[j].item() >= len(atom_decoder):
                    continue
                    
                atom1_symbol = atom_decoder[atom_types[i].item()]
                atom2_symbol = atom_decoder[atom_types[j].item()]
                distance = dists[i, j]
                
                # Determine bond order
                if dataset_info['name'] in ['qm9', 'qm9_second_half', 'qm9_first_half']:
                    order = get_bond_order(atom1_symbol, atom2_symbol, distance)
                elif dataset_info['name'] == 'geom':
                    order = geom_predictor((atom1_symbol, atom2_symbol), distance, limit_bonds_to_one=True)
                else:
                    order = self._generic_bond_order(atom1_symbol, atom2_symbol, distance)
                    
                if order > 0:
                    A[i, j] = 1
                    A[j, i] = 1  # Symmetric
                    E[i, j] = order
                    E[j, i] = order
                    
        return X, A, E
        
    def evaluate(self, molecule_list, return_dict=False):
        """
        Evaluate molecular validity, uniqueness, and novelty.
        
        Args:
            molecule_list: List of molecules (positions, atom_types tuples)
            return_dict: Whether to return detailed dictionary
            
        Returns:
            Tuple of (validity, uniqueness, novelty) or detailed dict
        """
        valid_molecules = []
        all_smiles = []
        
        for positions, atom_types in molecule_list:
            try:
                mol = self.build_molecule_from_positions(positions, atom_types, self.dataset_info)
                smiles = self.mol_to_smiles(mol)
                
                if smiles and len(smiles) > 0:
                    valid_molecules.append(mol)
                    all_smiles.append(smiles)
                    
            except Exception as e:
                logging.debug(f"Failed to process molecule: {e}")
                continue
                
        # Calculate metrics
        n_total = len(molecule_list)
        n_valid = len(valid_molecules)
        validity = n_valid / n_total if n_total > 0 else 0.0
        
        # Uniqueness
        unique_smiles = set(all_smiles)
        uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0.0
        
        # Novelty (if reference dataset available)
        novelty = 0.0
        if self.dataset_smiles_list:
            reference_set = set(self.dataset_smiles_list)
            novel_smiles = unique_smiles - reference_set
            novelty = len(novel_smiles) / len(unique_smiles) if len(unique_smiles) > 0 else 0.0
            
        if return_dict:
            return {
                'validity': validity,
                'uniqueness': uniqueness, 
                'novelty': novelty,
                'valid_molecules': valid_molecules,
                'all_smiles': all_smiles,
                'unique_smiles': unique_smiles
            }
        else:
            return (validity, uniqueness, novelty)
            
    def compute_molecular_properties(self, mol):
        """Compute molecular properties using OpenBabel."""
        properties = {}
        
        try:
            # Basic properties
            properties['molecular_weight'] = mol.molwt
            properties['num_atoms'] = len(mol.atoms)
            properties['num_bonds'] = len(mol.bonds)
            
            # Calculate descriptors if available
            if hasattr(mol, 'calcdesc'):
                descriptors = mol.calcdesc()
                properties.update(descriptors)
                
        except Exception as e:
            logging.warning(f"Failed to compute properties: {e}")
            
        return properties


def xyz_to_smiles_openbabel(positions, atom_types, atom_decoder):
    """
    Convert atomic positions and types to SMILES using OpenBabel.
    
    Args:
        positions: Atomic positions (N, 3)
        atom_types: Atomic type indices (N,)
        atom_decoder: List mapping indices to atom symbols
        
    Returns:
        SMILES string or None if conversion fails
    """
    try:
        mol = ob.OBMol()
        
        # Add atoms
        for i, atom_type_idx in enumerate(atom_types):
            if atom_type_idx < len(atom_decoder):
                symbol = atom_decoder[atom_type_idx]
                atom = mol.NewAtom()
                atom.SetAtomicNum(ob.GetAtomicNum(symbol))
                pos = positions[i]
                atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))
                
        # Add bonds based on distance
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        
        # Convert to SMILES
        conv = ob.OBConversion()
        conv.SetOutFormat("smi")
        smiles = conv.WriteString(mol).strip()
        
        return smiles if smiles else None
        
    except Exception as e:
        logging.warning(f"Failed to convert XYZ to SMILES: {e}")
        return None


def smiles_to_xyz_openbabel(smiles):
    """
    Convert SMILES to 3D coordinates using OpenBabel.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Tuple of (positions, atom_symbols) or None if conversion fails
    """
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()  # Generate 3D coordinates
        
        positions = []
        symbols = []
        
        for atom in mol.atoms:
            positions.append(atom.coords)
            symbols.append(atom.type)
            
        return np.array(positions), symbols
        
    except Exception as e:
        logging.warning(f"Failed to convert SMILES to XYZ: {e}")
        return None, None