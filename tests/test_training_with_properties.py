"""
Unit tests for property-conditioned training script components.

Tests the key functions and workflows in main_crystal_with_properties.py
"""

import unittest
import torch
import tempfile
import shutil
from pathlib import Path
from ase.db import connect
from ase import Atoms
import numpy as np

# Import functions from the training script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPropertyTrainingComponents(unittest.TestCase):
    """Test suite for property-conditioned training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create a small test database with properties
        self.db_path = self.temp_path / 'test_crystals.db'
        self._create_test_database()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_database(self):
        """Create a small test crystal database with properties."""
        db = connect(str(self.db_path))
        
        # Create 10 simple test crystals
        for i in range(10):
            atoms = Atoms(
                'CH4',
                positions=[[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                cell=[10, 10, 10],
                pbc=True
            )
            
            data = {
                'crystal_id': f'crystal_{i:03d}',
                'bandgap': 2.0 + i * 0.1,
                'melting_point': 150.0 + i * 10.0,
                'dielectric_constant': 2.5 + i * 0.05
            }
            
            db.write(atoms, data=data)
        
        db._close()
    
    def test_database_creation(self):
        """Test that test database is created correctly."""
        db = connect(str(self.db_path))
        n_rows = len([row for row in db.select()])
        self.assertEqual(n_rows, 10)
        
        # Check properties exist
        row = next(db.select())
        self.assertIn('bandgap', row.data)
        self.assertIn('melting_point', row.data)
        self.assertIn('dielectric_constant', row.data)
        
        db._close()
    
    def test_checkpoint_save_load_structure(self):
        """Test checkpoint save/load includes property statistics."""
        from main_crystal_with_properties import save_checkpoint_with_properties
        
        # Create mock objects
        class MockArgs:
            condition_on_property = True
            property_names = ['bandgap', 'melting_point']
        
        class MockModel:
            def state_dict(self):
                return {'weight': torch.randn(10, 10)}
        
        class MockOptim:
            def state_dict(self):
                return {'lr': 0.001}
        
        class MockDataset:
            property_mean = torch.tensor([2.5, 180.0])
            property_std = torch.tensor([1.2, 50.0])
        
        args = MockArgs()
        model = MockModel()
        optim = MockOptim()
        dataset = MockDataset()
        
        checkpoint_path = self.temp_path / 'test_checkpoint.pt'
        
        # Save checkpoint
        save_checkpoint_with_properties(
            args, epoch=10, model=model, optim=optim,
            mol_encoder=None, conditioning_modules={},
            crystal_dataset=dataset, best_val_loss=0.5,
            checkpoint_path=checkpoint_path
        )
        
        # Verify checkpoint exists
        self.assertTrue(checkpoint_path.exists())
        
        # Load and verify contents
        checkpoint = torch.load(checkpoint_path)
        
        self.assertIn('property_names', checkpoint)
        self.assertIn('property_mean', checkpoint)
        self.assertIn('property_std', checkpoint)
        self.assertEqual(checkpoint['property_names'], ['bandgap', 'melting_point'])
        self.assertEqual(checkpoint['epoch'], 10)
        self.assertEqual(checkpoint['best_val_loss'], 0.5)


class TestArgumentValidation(unittest.TestCase):
    """Test argument validation for training script."""
    
    def test_property_conditioning_requires_property_names(self):
        """Test that property conditioning requires property names."""
        # This would be tested by importing and calling the validation
        # For now, we document the expected behavior
        
        # When condition_on_property is True, property_names must be provided
        # Otherwise a ValueError should be raised
        pass


class TestDataLoaderWithProperties(unittest.TestCase):
    """Test data loading with properties."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test database
        self.db_path = self.temp_path / 'test.db'
        db = connect(str(self.db_path))
        
        for i in range(20):
            atoms = Atoms(
                'H2O',
                positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                cell=[10, 10, 10],
                pbc=True
            )
            
            data = {
                'crystal_id': f'crystal_{i:03d}',
                'bandgap': 2.0 + i * 0.1,
                'melting_point': 150.0 + i * 10.0
            }
            
            db.write(atoms, data=data)
        
        db._close()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_loads_properties(self):
        """Test that CrystalDatasetWithProperties loads correctly."""
        from crystal.data.crystal_loader import CrystalDatasetWithProperties
        from ase.db import connect
        
        db = connect(str(self.db_path))
        all_indices = [row.id for row in db.select()]
        db._close()
        
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=all_indices,
            property_names=['bandgap', 'melting_point']
        )
        
        # Check dataset properties
        self.assertEqual(len(dataset), 20)
        self.assertEqual(len(dataset.property_names), 2)
        self.assertEqual(len(dataset.property_mean), 2)
        self.assertEqual(len(dataset.property_std), 2)
        
        # Check statistics are reasonable
        self.assertGreater(dataset.property_mean[0], 0)  # bandgap mean > 0
        self.assertGreater(dataset.property_std[0], 0)   # bandgap std > 0
        
        # Get a sample
        sample = dataset[0]
        self.assertIn('properties', sample)
        self.assertEqual(len(sample['properties']), 2)


if __name__ == '__main__':
    unittest.main()
