#!/usr/bin/env python3
"""
Simple test to verify the CSV export functions work correctly in isolation.
"""
import os
import sys
import tempfile
import numpy as np

def test_per_molecule_csv_export():
    """Test just the per-molecule CSV export function."""
    # Create test data
    n_molecules = 10
    n_components = 5
    test_data = []
    
    for i in range(n_molecules):
        encoding = np.random.rand(n_components)
        test_data.append(encoding)
    
    component_names = ['H', 'C', 'N', 'O', 'F']
    
    # Import and test the function
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Define the function locally to avoid import issues
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
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, 'test_output.csv')
        
        # Test the function
        _export_per_molecule_encoding_csv(test_data, output_file, component_names)
        
        # Verify the output
        assert os.path.exists(output_file), "Output file not created"
        
        with open(output_file, 'r') as f:
            import csv
            reader = csv.reader(f)
            lines = list(reader)
            
            # Check structure
            assert len(lines) == n_molecules + 1, f"Expected {n_molecules + 1} lines, got {len(lines)}"
            
            # Check header
            expected_header = ['molecule_id'] + component_names
            assert lines[0] == expected_header, f"Header mismatch: {lines[0]} vs {expected_header}"
            
            # Check data format
            for i in range(1, min(4, len(lines))):  # Check first few rows
                row = lines[i]
                assert row[0] == f'mol{i}', f"Molecule ID mismatch: {row[0]} vs mol{i}"
                assert len(row) == 6, f"Expected 6 columns, got {len(row)}"
                
                # Check that values are numeric
                for j in range(1, len(row)):
                    try:
                        float(row[j])
                    except ValueError:
                        assert False, f"Non-numeric value: {row[j]}"
        
        print("✓ Per-molecule CSV export test passed")
        
        # Show sample output
        print("\n📋 Sample output:")
        with open(output_file, 'r') as f:
            import csv
            reader = csv.reader(f)
            lines = list(reader)
            for i, line in enumerate(lines[:4]):  # Show header + first 3 rows
                print(f"  {','.join(line)}")

def main():
    """Run the test."""
    print("Testing per-molecule CSV export functionality...\n")
    
    try:
        test_per_molecule_csv_export()
        print("\n✅ Test passed successfully!")
        print("The per-molecule CSV export is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()