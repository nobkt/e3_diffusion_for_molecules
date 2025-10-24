"""
Unit tests for CrystalDatasetWithProperties
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
from ase import Atoms
from ase.db import connect
from crystal.data.crystal_loader import CrystalDatasetWithProperties


class TestCrystalDatasetWithProperties(unittest.TestCase):
    """Test suite for CrystalDatasetWithProperties."""
    
    def setUp(self):
        """Set up test database with properties."""
        # Create temporary directory
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / 'test_crystals.db'
        
        # Create test database
        db = connect(str(self.db_path))
        
        # Add test crystals with properties
        for i in range(10):
            # Create simple crystal structure
            atoms = Atoms(
                'C4',
                positions=[
                    [0, 0, 0],
                    [1.5, 0, 0],
                    [0, 1.5, 0],
                    [1.5, 1.5, 0]
                ],
                cell=[5, 5, 5],
                pbc=True
            )
            
            # Add properties to data dict
            data = {
                'crystal_id': f'crystal_{i:03d}',
                'molecule_id': f'mol_{i%3:03d}',
                'bandgap': 2.0 + i * 0.1,  # Varying bandgap
                'melting_point': 150.0 + i * 10.0,  # Varying melting point
                'density': 1.0 + i * 0.05,
                'space_group': 1
            }
            
            db.write(atoms, data=data)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_initialization(self):
        """Test dataset initialization with properties."""
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=['bandgap', 'melting_point']
        )
        
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.property_names, ['bandgap', 'melting_point'])
        self.assertIsNotNone(dataset.property_mean)
        self.assertIsNotNone(dataset.property_std)
    
    def test_initialization_no_properties(self):
        """Test initialization without properties."""
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=[]
        )
        
        self.assertEqual(len(dataset), 10)
        self.assertIsNone(dataset.property_mean)
        self.assertIsNone(dataset.property_std)
    
    def test_property_statistics(self):
        """Test that statistics are computed correctly."""
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=['bandgap', 'melting_point']
        )
        
        # Check that mean and std have correct shape
        self.assertEqual(dataset.property_mean.shape, (2,))
        self.assertEqual(dataset.property_std.shape, (2,))
        
        # Check approximate values (bandgap: 2.0 to 2.9, melting_point: 150 to 240)
        self.assertAlmostEqual(dataset.property_mean[0].item(), 2.45, places=1)
        self.assertAlmostEqual(dataset.property_mean[1].item(), 195.0, places=0)
        
        # Std should be > 0
        self.assertGreater(dataset.property_std[0].item(), 0.0)
        self.assertGreater(dataset.property_std[1].item(), 0.0)
    
    def test_getitem_with_properties(self):
        """Test __getitem__ returns properties."""
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=['bandgap', 'melting_point']
        )
        
        data = dataset[0]
        
        # Check that properties are present
        self.assertIn('properties', data)
        self.assertEqual(data['properties'].shape, (2,))
        
        # Check approximate values for first crystal
        self.assertAlmostEqual(data['properties'][0].item(), 2.0, places=1)
        self.assertAlmostEqual(data['properties'][1].item(), 150.0, places=0)
    
    def test_getitem_without_properties(self):
        """Test __getitem__ without properties."""
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=[]
        )
        
        data = dataset[0]
        
        # Properties should not be present
        self.assertNotIn('properties', data)
    
    def test_missing_property_error(self):
        """Test error when property is missing."""
        # Create database with missing property
        db_path2 = Path(self.tmpdir) / 'test_crystals2.db'
        db = connect(str(db_path2))
        
        # Add crystals, some missing bandgap
        for i in range(5):
            atoms = Atoms('C2', positions=[[0, 0, 0], [1, 0, 0]], cell=[3, 3, 3], pbc=True)
            data = {
                'crystal_id': f'crystal_{i:03d}',
                'molecule_id': f'mol_{i:03d}',
            }
            # Only add bandgap to first 3
            if i < 3:
                data['bandgap'] = 2.0 + i * 0.1
            db.write(atoms, data=data)
        
        # Should raise error during initialization
        with self.assertRaises(ValueError) as cm:
            CrystalDatasetWithProperties(
                db_path=str(db_path2),
                indices=list(range(5)),
                property_names=['bandgap']
            )
        
        self.assertIn('missing', str(cm.exception).lower())
    
    def test_zero_std_error(self):
        """Test error when property has zero standard deviation."""
        # Create database with constant property
        db_path3 = Path(self.tmpdir) / 'test_crystals3.db'
        db = connect(str(db_path3))
        
        for i in range(5):
            atoms = Atoms('C2', positions=[[0, 0, 0], [1, 0, 0]], cell=[3, 3, 3], pbc=True)
            data = {
                'crystal_id': f'crystal_{i:03d}',
                'molecule_id': f'mol_{i:03d}',
                'constant_prop': 2.5  # Same value for all
            }
            db.write(atoms, data=data)
        
        # Should raise error during initialization
        with self.assertRaises(ValueError) as cm:
            CrystalDatasetWithProperties(
                db_path=str(db_path3),
                indices=list(range(5)),
                property_names=['constant_prop']
            )
        
        self.assertIn('zero standard deviation', str(cm.exception).lower())
    
    def test_batch_collation(self):
        """Test that batching works with properties."""
        from torch.utils.data import DataLoader
        from crystal.data.crystal_loader import collate_crystal_batch
        
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=['bandgap', 'melting_point']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_crystal_batch
        )
        
        batch = next(iter(dataloader))
        
        # Check that properties are in batch
        self.assertIn('properties', batch)
        self.assertEqual(batch['properties'].shape, (4, 2))
    
    def test_inheritance(self):
        """Test that CrystalDatasetWithProperties inherits from CrystalDataset."""
        from crystal.data.crystal_loader import CrystalDataset
        
        dataset = CrystalDatasetWithProperties(
            db_path=str(self.db_path),
            indices=list(range(10)),
            property_names=['bandgap']
        )
        
        self.assertIsInstance(dataset, CrystalDataset)
        
        # Should have all CrystalDataset attributes
        self.assertTrue(hasattr(dataset, 'db'))
        self.assertTrue(hasattr(dataset, 'atom_encoder'))
        self.assertTrue(hasattr(dataset, 'num_atom_types'))


if __name__ == '__main__':
    unittest.main()
