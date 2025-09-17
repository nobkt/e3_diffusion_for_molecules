#!/usr/bin/env python3
"""
Test script to verify CSV export functionality works properly.
This creates mock data to test the export functions without requiring actual training.
"""
import os
import sys
import torch
import numpy as np
import tempfile
import shutil

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_qm9 import _export_scalar_histogram_csv, _export_encoding_statistics_csv, export_training_statistics

def create_mock_dataloader():
    """Create a mock dataloader with sample data for testing."""
    
    class MockDataset:
        def __init__(self, n_samples=100):
            self.n_samples = n_samples
            
        def __len__(self):
            return 1  # One batch
            
        def __iter__(self):
            # Create mock data with varying molecular properties
            batch_size = self.n_samples
            
            # Mock molecular weights (realistic range: 50-500 u)
            molecular_weights = torch.rand(batch_size) * 450 + 50
            
            # Mock pi conjugation ratios (range: 0-1)
            pi_ratios = torch.rand(batch_size)
            
            # Mock atom types encoding (5 atom types)
            n_atom_types = 5
            atom_types_encoding = torch.zeros(batch_size, n_atom_types)
            for i in range(batch_size):
                # Randomly assign 1-3 atom types per molecule
                n_types = np.random.randint(1, 4)
                indices = np.random.choice(n_atom_types, n_types, replace=False)
                atom_types_encoding[i, indices] = torch.rand(n_types)
                # Normalize to sum to 1 (probability distribution)
                atom_types_encoding[i] = atom_types_encoding[i] / atom_types_encoding[i].sum()
            
            # Mock functional groups encoding (8 functional groups)
            n_fg = 8
            fg_encoding = torch.zeros(batch_size, n_fg)
            for i in range(batch_size):
                # Randomly assign 0-3 functional groups per molecule
                n_groups = np.random.randint(0, 4)
                if n_groups > 0:
                    indices = np.random.choice(n_fg, n_groups, replace=False)
                    fg_encoding[i, indices] = torch.rand(n_groups)
                    fg_encoding[i] = fg_encoding[i] / fg_encoding[i].sum()
                else:
                    # Uniform small probability for molecules with no functional groups
                    fg_encoding[i] = torch.full((n_fg,), 0.01 / n_fg)
            
            # Mock positions for batch size compatibility
            positions = torch.rand(batch_size, 10, 3)  # Max 10 atoms per molecule
            
            batch_data = {
                'molecular_weight': molecular_weights,
                'pi_conjugation_ratio': pi_ratios,
                'atom_types_encoding': atom_types_encoding,
                'functional_groups_encoding': fg_encoding,
                'positions': positions
            }
            
            yield batch_data
    
    return MockDataset()

def test_scalar_histogram_export():
    """Test the scalar histogram export function."""
    print("Testing scalar histogram export...")
    
    # Create test data
    test_values = np.random.normal(100, 20, 1000).tolist()  # Normal distribution around 100
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'test_histogram.csv')
        
        # Test the export function
        _export_scalar_histogram_csv(test_values, test_file, 'Test Value', 'Count')
        
        # Verify files were created
        assert os.path.exists(test_file), "Histogram CSV file not created"
        
        summary_file = test_file.replace('.csv', '_summary.csv')
        assert os.path.exists(summary_file), "Summary CSV file not created"
        
        # Check file contents
        with open(test_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1, "Histogram file should have header + data"
            assert 'Test Value,Count' in lines[0], "Header not correct"
        
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1, "Summary file should have header + data"
            assert 'Statistic,Value' in lines[0], "Summary header not correct"
        
        print("✓ Scalar histogram export test passed")

def test_encoding_statistics_export():
    """Test the encoding statistics export function."""
    print("Testing encoding statistics export...")
    
    # Create test encoding data (100 samples, 5 components each)
    n_samples = 100
    n_components = 5
    test_data = []
    
    for i in range(n_samples):
        # Create sparse encoding vector
        encoding = np.zeros(n_components)
        n_active = np.random.randint(1, 3)  # 1-2 active components
        active_indices = np.random.choice(n_components, n_active, replace=False)
        encoding[active_indices] = np.random.rand(n_active)
        encoding = encoding / encoding.sum() if encoding.sum() > 0 else encoding
        test_data.append(encoding)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'test_encoding.csv')
        
        # Test the export function
        _export_encoding_statistics_csv(test_data, test_file, 'Component', 'Stats')
        
        # Verify main stats file was created
        assert os.path.exists(test_file), "Encoding stats CSV file not created"
        
        # Check file contents
        with open(test_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == n_components + 1, f"Should have {n_components + 1} lines (header + {n_components} components)"
            assert 'Component,Mean,Std,Min,Max' in lines[0], "Header not correct"
        
        # Check that component histogram files were created for non-zero components
        component_files = [f for f in os.listdir(temp_dir) if 'component_' in f and 'histogram.csv' in f]
        assert len(component_files) > 0, "No component histogram files created"
        
        print("✓ Encoding statistics export test passed")

def test_full_export_functionality():
    """Test the complete export functionality with mock dataloaders."""
    print("Testing full export functionality...")
    
    # Create mock dataloaders
    mock_train_loader = create_mock_dataloader()
    dataloaders = {'train': mock_train_loader}
    
    # Create mock args
    class MockArgs:
        pass
    
    args = MockArgs()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test the full export function
        export_training_statistics(dataloaders, args, temp_dir)
        
        # Check that expected files were created
        expected_files = [
            'molecular_weight_histogram.csv',
            'molecular_weight_histogram_summary.csv',
            'pi_conjugation_ratio_histogram.csv', 
            'pi_conjugation_ratio_histogram_summary.csv',
            'atom_types_encoding_stats.csv',
            'functional_groups_encoding_stats.csv',
            'export_summary.csv'
        ]
        
        for expected_file in expected_files:
            file_path = os.path.join(temp_dir, expected_file)
            assert os.path.exists(file_path), f"Expected file {expected_file} was not created"
        
        # Check that component histogram files were created
        all_files = os.listdir(temp_dir)
        component_files = [f for f in all_files if 'component_' in f and 'histogram' in f]
        assert len(component_files) > 0, "No component histogram files created"
        
        print("✓ Full export functionality test passed")
        print(f"✓ Created {len(all_files)} files total in test directory")

def main():
    """Run all tests."""
    print("Starting CSV export functionality tests...\n")
    
    try:
        test_scalar_histogram_export()
        test_encoding_statistics_export()
        test_full_export_functionality()
        
        print("\n✅ All tests passed successfully!")
        print("CSV export functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()