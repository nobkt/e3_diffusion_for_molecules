"""
Molecule-crystal mapping manager

Manages the correspondence between molecules and their crystal structures.
Supports polymorphs (multiple crystal forms of the same molecule).
"""

import json
from typing import Dict, List, Optional
from pathlib import Path


class MoleculeCrystalMapper:
    """
    Manages molecule-crystal correspondence for homocrystal generation
    
    Maintains mapping between molecule_id and crystal_ids, supporting
    polymorphs (1 molecule : N crystals relationship).
    
    Args:
        map_file_path: Path to JSON mapping file (optional)
    """
    
    def __init__(self, map_file_path: Optional[str] = None):
        self.map_file_path = map_file_path
        self.mol_to_crystals: Dict[str, List[str]] = {}
        self.crystal_to_mol: Dict[str, str] = {}
        
        if map_file_path and Path(map_file_path).exists():
            self._load_mapping(map_file_path)
    
    def _load_mapping(self, map_file_path: str):
        """Load mapping from JSON file"""
        with open(map_file_path, 'r') as f:
            data = json.load(f)
        
        for mol_id, mol_data in data.items():
            crystal_ids = mol_data.get('crystal_ids', [])
            self.mol_to_crystals[mol_id] = crystal_ids
            
            for crys_id in crystal_ids:
                self.crystal_to_mol[crys_id] = mol_id
    
    def build_from_databases(
        self,
        molecule_db_path: str,
        crystal_db_path: str
    ):
        """
        Build mapping dynamically from databases
        
        Scans crystal database and extracts molecule_id from each entry.
        NO FALLBACK - all crystals must have molecule_id.
        
        Args:
            molecule_db_path: Path to molecules database
            crystal_db_path: Path to crystals database
        """
        from ase.db import connect
        
        crystal_db = connect(crystal_db_path)
        
        for row in crystal_db.select():
            # Extract crystal_id
            crystal_id = self._get_crystal_id(row)
            
            # Extract molecule_id (REQUIRED - no fallback!)
            molecule_id = self._get_molecule_id(row, crystal_id)
            
            # Add to mapping
            if molecule_id not in self.mol_to_crystals:
                self.mol_to_crystals[molecule_id] = []
            self.mol_to_crystals[molecule_id].append(crystal_id)
            self.crystal_to_mol[crystal_id] = molecule_id
    
    def _get_crystal_id(self, row) -> str:
        """Extract crystal_id from row"""
        if hasattr(row, 'crystal_id'):
            return str(row.crystal_id)
        if hasattr(row, 'data') and 'crystal_id' in row.data:
            return str(row.data['crystal_id'])
        if hasattr(row, 'key_value_pairs') and 'crystal_id' in row.key_value_pairs:
            return str(row.key_value_pairs['crystal_id'])
        # Use database id as fallback for crystal_id only
        return str(row.id)
    
    def _get_molecule_id(self, row, crystal_id: str) -> str:
        """
        Extract molecule_id from crystal row
        
        NO FALLBACK! All crystals must have molecule_id.
        """
        if hasattr(row, 'molecule_id'):
            return str(row.molecule_id)
        
        if hasattr(row, 'data') and 'molecule_id' in row.data:
            return str(row.data['molecule_id'])
        
        if hasattr(row, 'key_value_pairs') and 'molecule_id' in row.key_value_pairs:
            return str(row.key_value_pairs['molecule_id'])
        
        # NO FALLBACK - raise clear error
        raise ValueError(
            f"molecule_id not found in crystal entry {crystal_id}. "
            f"All crystals must have an explicit molecule_id field to link to molecules. "
            f"This is REQUIRED for homocrystal generation."
        )
    
    def get_crystals_for_molecule(self, molecule_id: str) -> List[str]:
        """Get list of crystal IDs for a given molecule"""
        return self.mol_to_crystals.get(molecule_id, [])
    
    def get_molecule_for_crystal(self, crystal_id: str) -> Optional[str]:
        """Get molecule ID for a given crystal"""
        return self.crystal_to_mol.get(crystal_id)
    
    def get_num_polymorphs(self, molecule_id: str) -> int:
        """Get number of polymorphs for a molecule"""
        return len(self.get_crystals_for_molecule(molecule_id))
    
    def has_polymorphs(self, molecule_id: str) -> bool:
        """Check if molecule has multiple crystal forms"""
        return self.get_num_polymorphs(molecule_id) > 1
    
    def save_mapping(self, output_path: str):
        """Save mapping to JSON file"""
        data = {}
        for mol_id, crys_ids in self.mol_to_crystals.items():
            data[mol_id] = {
                'molecule_id': mol_id,
                'crystal_ids': crys_ids,
                'num_polymorphs': len(crys_ids)
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the mapping"""
        num_molecules = len(self.mol_to_crystals)
        num_crystals = len(self.crystal_to_mol)
        num_with_polymorphs = sum(1 for mol_id in self.mol_to_crystals 
                                   if self.has_polymorphs(mol_id))
        max_polymorphs = max((self.get_num_polymorphs(mol_id) 
                             for mol_id in self.mol_to_crystals), default=0)
        
        return {
            'num_molecules': num_molecules,
            'num_crystals': num_crystals,
            'num_with_polymorphs': num_with_polymorphs,
            'max_polymorphs': max_polymorphs,
            'avg_crystals_per_molecule': num_crystals / num_molecules if num_molecules > 0 else 0,
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"MoleculeCrystalMapper("
            f"molecules={stats['num_molecules']}, "
            f"crystals={stats['num_crystals']}, "
            f"with_polymorphs={stats['num_with_polymorphs']})"
        )
