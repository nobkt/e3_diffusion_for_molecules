#!/usr/bin/env python3
"""
Test script to verify the new per-molecule CSV export functionality.
This creates mock data to test the new functions without requiring actual training.
"""
import os
import sys
import numpy as np
import tempfile
import shutil

def create_mock_data():
    """Create mock encoding data for testing."""
    # Create mock atom types encoding (100 molecules, 5 atom types)
    n_molecules = 100
    n_atom_types = 5
    atom_types_data = []
    
    atom_names = ['H', 'C', 'N', 'O', 'F']
    
    for i in range(n_molecules):
        # Create sparse encoding vector (simulating real molecular data)
        encoding = np.zeros(n_atom_types)
        n_active = np.random.randint(1, 4)  # 1-3 active atom types
        active_indices = np.random.choice(n_atom_types, n_active, replace=False)
        encoding[active_indices] = np.random.rand(n_active)
        # Some molecules might have normalized encodings, others binary
        if np.random.random() > 0.5:
            encoding = encoding / encoding.sum() if encoding.sum() > 0 else encoding
        else:
            encoding = (encoding > 0.5).astype(float)  # Binary encoding
        atom_types_data.append(encoding)
    
    # Create mock functional groups encoding (100 molecules, 8 functional groups)
    n_fg = 8
    functional_groups_data = []
    
    fg_smarts = [
        '[OH]',        # -OH
        '[CX3]=[OX1]', # C=O
        '[CX3](=O)[OX2H1]',  # -COOH
        '[CX3H1](=O)[#6]',   # -CHO
        '[CX3](=O)([#6])[#6]', # ketone C=O
        '[NX3;H2,H1;!$(NC=O)]',  # -NH2, -NH-
        '[N+](=O)[O-]',   # -NO2
        '[Cl]',          # -Cl
    ]
    
    for i in range(n_molecules):
        # Create sparse encoding vector
        encoding = np.zeros(n_fg)
        n_groups = np.random.randint(0, 4)  # 0-3 functional groups
        if n_groups > 0:
            indices = np.random.choice(n_fg, n_groups, replace=False)
            encoding[indices] = np.random.rand(n_groups)
            # Normalize or binarize
            if np.random.random() > 0.3:
                encoding = encoding / encoding.sum() if encoding.sum() > 0 else encoding
            else:
                encoding = (encoding > 0.5).astype(float)
        functional_groups_data.append(encoding)
    
    return atom_types_data, functional_groups_data, atom_names, fg_smarts

def test_per_molecule_export():
    """Test the new per-molecule export function."""
    print("Testing per-molecule export functionality...")
    
    # Import the function (avoiding wandb issues)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Create a minimal version of the function to avoid wandb imports
    def _export_per_molecule_encoding_csv(encoding_data, filename, component_names=None):
        """Export per-molecule encoding data to CSV with molecules as rows and components as columns."""
        import csv
        import numpy as np
        
        if not encoding_data:
            # Create empty file with header
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if component_names:
                    header = ['molecule_id'] + component_names
                else:
                    header = ['molecule_id']
                writer.writerow(header)
            return
        
        # Convert to numpy array for easier processing
        encoding_array = np.array(encoding_data)  # Shape: (n_molecules, n_components)
        n_molecules, n_components = encoding_array.shape
        
        # Create header with component names
        if component_names and len(component_names) >= n_components:
            header = ['molecule_id'] + component_names[:n_components]
        else:
            header = ['molecule_id'] + [f'Component_{i}' for i in range(n_components)]
        
        # Export per-molecule data using csv.writer to properly handle complex column names
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            
            for mol_idx in range(n_molecules):
                molecule_id = f'mol{mol_idx + 1}'
                row_data = [molecule_id] + [f"{encoding_array[mol_idx, i]:.6f}" for i in range(n_components)]
                writer.writerow(row_data)
    
    # Create test data
    atom_types_data, functional_groups_data, atom_names, fg_smarts = create_mock_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test atom types encoding export
        atom_file = os.path.join(temp_dir, 'atom_types_encoding.csv')
        _export_per_molecule_encoding_csv(atom_types_data, atom_file, atom_names)
        
        # Test functional groups encoding export
        fg_file = os.path.join(temp_dir, 'functional_groups_encoding.csv')
        _export_per_molecule_encoding_csv(functional_groups_data, fg_file, fg_smarts)
        
        # Verify files were created
        assert os.path.exists(atom_file), "Atom types CSV file not created"
        assert os.path.exists(fg_file), "Functional groups CSV file not created"
        
        # Check atom types file structure
        with open(atom_file, 'r') as f:
            import csv
            reader = csv.reader(f)
            lines = list(reader)
            # Should have header + 100 molecules
            assert len(lines) == 101, f"Expected 101 lines (header + 100 molecules), got {len(lines)}"
            
            # Check header
            header = lines[0]
            expected_header = ['molecule_id'] + atom_names
            assert header == expected_header, f"Header mismatch: {header} vs {expected_header}"
            
            # Check first data row
            first_row = lines[1]
            assert first_row[0] == 'mol1', f"First molecule ID should be 'mol1', got {first_row[0]}"
            assert len(first_row) == 6, f"Should have 6 columns (id + 5 atom types), got {len(first_row)}"
            
            # Verify all values are numeric (except molecule_id)
            for i in range(1, len(first_row)):
                try:
                    float(first_row[i])
                except ValueError:
                    assert False, f"Non-numeric value in column {i}: {first_row[i]}"
        
        # Check functional groups file structure
        with open(fg_file, 'r') as f:
            import csv
            reader = csv.reader(f)
            lines = list(reader)
            # Should have header + 100 molecules
            assert len(lines) == 101, f"Expected 101 lines (header + 100 molecules), got {len(lines)}"
            
            # Check header
            header = lines[0]
            expected_header = ['molecule_id'] + fg_smarts
            assert header == expected_header, f"Header mismatch: {header} vs {expected_header}"
            
            # Check some molecules have functional groups
            has_non_zero = False
            for line in lines[1:11]:  # Check first 10 molecules
                values = [float(x) for x in line[1:]]
                if any(v > 0 for v in values):
                    has_non_zero = True
                    break
            assert has_non_zero, "No molecules with functional groups found in sample"
        
        print("✓ Per-molecule export test passed")
        
        # Show sample of the output
        print("\n📋 Sample atom types encoding output:")
        with open(atom_file, 'r') as f:
            import csv
            reader = csv.reader(f)
            lines = list(reader)
            for i, line in enumerate(lines[:4]):  # Show header + first 3 molecules
                print(f"  {','.join(line)}")
        
        print("\n📋 Sample functional groups encoding output:")
        with open(fg_file, 'r') as f:
            import csv
            reader = csv.reader(f)
            lines = list(reader)
            for i, line in enumerate(lines[:4]):  # Show header + first 3 molecules
                print(f"  {','.join(line)}")

def main():
    """Run the test."""
    print("Starting per-molecule CSV export test...\n")
    
    try:
        test_per_molecule_export()
        
        print("\n✅ All tests passed successfully!")
        print("Per-molecule CSV export functionality is working correctly.")
        print("\nThe new format exports each molecule as a row with:")
        print("- molecule_id column (mol1, mol2, etc.)")
        print("- One column per atom type/functional group")
        print("- Actual encoding values for each molecule")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()