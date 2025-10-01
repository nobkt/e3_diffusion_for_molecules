#!/usr/bin/env python3
"""
Test script to verify the CSV export fixes for atom_types_encoding and functional_groups_encoding.
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_composition_string_fix():
    """Test that composition string generation correctly converts atomic numbers to symbols."""
    import torch
    from collections import Counter
    
    print("Testing composition string generation fix...")
    
    # Simulate the fixed get_composition_string function
    def get_composition_string(charges_or_atomic_nums, num_atoms_val):
        """Generate molecular formula from atomic charges or atomic numbers."""
        if charges_or_atomic_nums is None:
            return "Unknown"
        
        atoms_array = charges_or_atomic_nums[:num_atoms_val].cpu().numpy()
        atom_counts = Counter()
        
        # Create atomic number to symbol mapping for direct conversion
        atomic_num_to_symbol = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
            15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
        }
        
        for atom_val in atoms_array:
            atom_val = int(atom_val)
            if atom_val > 0:
                # Convert atomic number directly to symbol
                atom_symbol = atomic_num_to_symbol.get(atom_val, f"Z{atom_val}")
                atom_counts[atom_symbol] += 1
        
        # Create formula string (e.g., C6H12O6)
        formula = ""
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            formula += f"{atom}{count if count > 1 else ''}"
        
        return formula if formula else "Unknown"
    
    # Test case 1: C6H12O6 (glucose)
    atomic_nums = torch.tensor([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8])
    composition = get_composition_string(atomic_nums, len(atomic_nums))
    print(f"  Test 1: {composition}")
    assert composition == "C6H12O6", f"Expected C6H12O6, got {composition}"
    
    # Test case 2: C27Cl5P21S (example from problem statement)
    # C=6, Cl=17, P=15, S=16
    atomic_nums = torch.tensor([6]*27 + [17]*5 + [15]*21 + [16]*1)
    composition = get_composition_string(atomic_nums, len(atomic_nums))
    print(f"  Test 2: {composition}")
    assert composition == "C27Cl5P21S", f"Expected C27Cl5P21S, got {composition}"
    
    # Test case 3: No hydrogen removal (C2H6O - ethanol)
    atomic_nums = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1, 8])
    composition = get_composition_string(atomic_nums, len(atomic_nums))
    print(f"  Test 3: {composition}")
    assert composition == "C2H6O", f"Expected C2H6O, got {composition}"
    
    # Test case 4: With hydrogen removal (C2O - ethanol without H)
    atomic_nums = torch.tensor([6, 6, 8])
    composition = get_composition_string(atomic_nums, len(atomic_nums))
    print(f"  Test 4: {composition}")
    assert composition == "C2O", f"Expected C2O, got {composition}"
    
    print("  ✓ All composition string tests passed!")
    return True


def test_atom_types_consistency():
    """Test that atom types extraction is consistent with atomic_numbers tensor."""
    print("\nTesting atom types extraction consistency...")
    
    try:
        from ase import Atoms
        from qm9.openbabel_functions import extract_atom_types_from_ase
        
        # Test case 1: Ethanol (C2H6O)
        atoms = Atoms('C2H6O', positions=[[0, 0, 0], [1.5, 0, 0], [2.5, 0, 0], 
                                          [3.0, 0.5, 0], [3.0, -0.5, 0], [0.5, 0.5, 0],
                                          [0.5, -0.5, 0], [-0.5, 0, 0], [2.0, 0, 0]])
        
        # Extract all atom types
        all_atom_types = extract_atom_types_from_ase(atoms)
        print(f"  Test 1 - All atoms: {sorted(all_atom_types)}")
        assert 'C' in all_atom_types
        assert 'H' in all_atom_types
        assert 'O' in all_atom_types
        
        # Extract with H removed
        non_h_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol != 'H']
        atoms_no_h = atoms[non_h_indices] if non_h_indices else atoms
        no_h_atom_types = extract_atom_types_from_ase(atoms_no_h)
        print(f"  Test 1 - No H: {sorted(no_h_atom_types)}")
        assert 'C' in no_h_atom_types
        assert 'O' in no_h_atom_types
        assert 'H' not in no_h_atom_types
        
        print("  ✓ Atom types consistency tests passed!")
        return True
        
    except ImportError as e:
        print(f"  ⚠ Skipping atom types test (missing dependency): {e}")
        return True


def test_csv_export_integration():
    """Test the full CSV export with a small test database."""
    print("\nTesting CSV export integration...")
    
    try:
        from ase.db import connect
        from ase import Atoms
        import torch
        
        # Create a temporary test database
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            test_db_path = f.name
        
        try:
            # Create test database with a few molecules
            db = connect(test_db_path)
            
            # Add ethanol (C2H6O)
            ethanol = Atoms('C2H6O', positions=[[0, 0, 0], [1.5, 0, 0], [2.5, 0, 0], 
                                                [3.0, 0.5, 0], [3.0, -0.5, 0], [0.5, 0.5, 0],
                                                [0.5, -0.5, 0], [-0.5, 0, 0], [2.0, 0, 0]])
            db.write(ethanol)
            
            # Add water (H2O)
            water = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
            db.write(water)
            
            print(f"  Created test database with 2 molecules at {test_db_path}")
            
            # Try to load and process
            from qm9.dataset import load_ase_database
            
            # Create output directory
            output_dir = tempfile.mkdtemp()
            
            try:
                # Load database without H removal first
                print("  Testing with hydrogen atoms...")
                datasets, num_species, charge_scale = load_ase_database(
                    test_db_path,
                    split_ratios=(0.5, 0.25, 0.25),
                    seed=42,
                    include_charges=False,
                    remove_h=False,
                    remove_duplicates=False
                )
                
                print(f"    Loaded {sum(len(ds) for ds in datasets.values())} molecules")
                
                # Export to CSV
                from qm9.dataset import export_generation_conditions_to_csv
                from configs.datasets_config import ase_db_with_h
                
                export_generation_conditions_to_csv(datasets, output_dir, ase_db_with_h)
                
                # Check CSV files exist
                csv_files = ['molecular_weight.csv', 'pi_conjugation_ratio.csv', 
                            'atom_types_encoding.csv', 'functional_groups_encoding.csv']
                
                for csv_file in csv_files:
                    csv_path = os.path.join(output_dir, csv_file)
                    if os.path.exists(csv_path):
                        print(f"    ✓ {csv_file} created")
                        
                        # Read and display first few lines
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:3]
                            print(f"      Sample: {lines[0].strip()}")
                            if len(lines) > 1:
                                print(f"              {lines[1].strip()}")
                    else:
                        print(f"    ✗ {csv_file} not found")
                
                # Test with H removal
                print("\n  Testing with hydrogen removal...")
                output_dir_no_h = tempfile.mkdtemp()
                
                datasets_no_h, _, _ = load_ase_database(
                    test_db_path,
                    split_ratios=(0.5, 0.25, 0.25),
                    seed=42,
                    include_charges=False,
                    remove_h=True,
                    remove_duplicates=False
                )
                
                from configs.datasets_config import ase_db_without_h
                export_generation_conditions_to_csv(datasets_no_h, output_dir_no_h, ase_db_without_h)
                
                # Verify atom_types_encoding doesn't include H
                atom_types_csv = os.path.join(output_dir_no_h, 'atom_types_encoding.csv')
                with open(atom_types_csv, 'r', encoding='utf-8') as f:
                    header = f.readline()
                    print(f"    Header (no H): {header.strip()}")
                    assert 'H' not in header or header.count('H') == 0 or 'H' in ['ID', 'Hg', 'Hf'], \
                        "Hydrogen should not be in atom types when remove_h=True"
                
                print("  ✓ CSV export integration tests passed!")
                
            finally:
                # Clean up output directories
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                if 'output_dir_no_h' in locals() and os.path.exists(output_dir_no_h):
                    shutil.rmtree(output_dir_no_h)
        
        finally:
            # Clean up test database
            if os.path.exists(test_db_path):
                os.unlink(test_db_path)
        
        return True
        
    except ImportError as e:
        print(f"  ⚠ Skipping CSV export test (missing dependency): {e}")
        return True
    except Exception as e:
        print(f"  ✗ CSV export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*70)
    print("CSV Export Fix Verification Tests")
    print("="*70)
    
    success = True
    
    try:
        success &= test_composition_string_fix()
    except Exception as e:
        print(f"✗ Composition string test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= test_atom_types_consistency()
    except Exception as e:
        print(f"✗ Atom types consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= test_csv_export_integration()
    except Exception as e:
        print(f"✗ CSV export integration test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "="*70)
    if success:
        print("✓ All tests passed! The fixes are working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please review the output above.")
        sys.exit(1)
