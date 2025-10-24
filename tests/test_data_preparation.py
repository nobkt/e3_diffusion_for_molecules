"""
Unit tests for data preparation script.

Tests the key functions in scripts/prepare_property_dataset.py
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from ase.db import connect
from ase import Atoms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from prepare_property_dataset import (
    validate_properties,
    merge_properties,
    compute_and_display_statistics,
    create_splits
)


class TestPropertyValidation(unittest.TestCase):
    """Test property validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test database
        self.db_path = self.temp_path / 'test.db'
        db = connect(str(self.db_path))
        
        for i in range(10):
            atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                         cell=[10, 10, 10], pbc=True)
            data = {'crystal_id': f'crystal_{i:03d}'}
            db.write(atoms, data=data)
        
        db._close()
        
        # Create test CSV with properties
        self.csv_path = self.temp_path / 'properties.csv'
        self.create_valid_csv()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_valid_csv(self):
        """Create a valid property CSV."""
        data = {
            'crystal_id': [f'crystal_{i:03d}' for i in range(10)],
            'bandgap': [2.0 + i * 0.1 for i in range(10)],
            'melting_point': [150.0 + i * 10.0 for i in range(10)]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        return df
    
    def test_validate_valid_data(self):
        """Test validation passes with valid data."""
        db = connect(str(self.db_path))
        df = pd.read_csv(self.csv_path)
        
        # Should not raise any errors
        validate_properties(db, df, ['bandgap', 'melting_point'], 'crystal_id')
        
        db._close()
    
    def test_validate_missing_crystal_id_column(self):
        """Test validation fails when crystal_id column is missing."""
        db = connect(str(self.db_path))
        
        # Create CSV without crystal_id
        data = {
            'id': [f'crystal_{i:03d}' for i in range(10)],  # Wrong column name
            'bandgap': [2.0 + i * 0.1 for i in range(10)]
        }
        df = pd.DataFrame(data)
        
        with self.assertRaises(ValueError) as cm:
            validate_properties(db, df, ['bandgap'], 'crystal_id')
        
        self.assertIn('crystal_id', str(cm.exception).lower())
        
        db._close()
    
    def test_validate_missing_property_column(self):
        """Test validation fails when property column is missing."""
        db = connect(str(self.db_path))
        df = pd.read_csv(self.csv_path)
        
        with self.assertRaises(ValueError) as cm:
            validate_properties(db, df, ['bandgap', 'nonexistent_property'], 'crystal_id')
        
        self.assertIn('nonexistent_property', str(cm.exception))
        
        db._close()
    
    def test_validate_missing_properties_for_crystals(self):
        """Test validation fails when some crystals lack properties."""
        db = connect(str(self.db_path))
        
        # Create CSV with only half the crystals
        data = {
            'crystal_id': [f'crystal_{i:03d}' for i in range(5)],  # Only 5 out of 10
            'bandgap': [2.0 + i * 0.1 for i in range(5)]
        }
        df = pd.DataFrame(data)
        
        with self.assertRaises(ValueError) as cm:
            validate_properties(db, df, ['bandgap'], 'crystal_id')
        
        self.assertIn('missing', str(cm.exception).lower())
        
        db._close()
    
    def test_validate_zero_variance_property(self):
        """Test validation fails for zero-variance properties."""
        db = connect(str(self.db_path))
        
        # Create CSV with constant property
        data = {
            'crystal_id': [f'crystal_{i:03d}' for i in range(10)],
            'bandgap': [2.5] * 10,  # All same value
            'melting_point': [150.0 + i * 10.0 for i in range(10)]
        }
        df = pd.DataFrame(data)
        
        with self.assertRaises(ValueError) as cm:
            validate_properties(db, df, ['bandgap', 'melting_point'], 'crystal_id')
        
        self.assertIn('zero variance', str(cm.exception).lower())
        self.assertIn('bandgap', str(cm.exception))
        
        db._close()
    
    def test_validate_nan_values(self):
        """Test validation fails when properties contain NaN."""
        db = connect(str(self.db_path))
        
        # Create CSV with NaN
        data = {
            'crystal_id': [f'crystal_{i:03d}' for i in range(10)],
            'bandgap': [2.0 + i * 0.1 if i < 9 else np.nan for i in range(10)]
        }
        df = pd.DataFrame(data)
        
        with self.assertRaises(ValueError) as cm:
            validate_properties(db, df, ['bandgap'], 'crystal_id')
        
        self.assertIn('nan', str(cm.exception).lower())
        
        db._close()


class TestPropertyMerging(unittest.TestCase):
    """Test property merging into database."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create input database
        self.input_db = self.temp_path / 'input.db'
        db = connect(str(self.input_db))
        
        for i in range(5):
            atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]],
                         cell=[10, 10, 10], pbc=True)
            data = {'crystal_id': f'crystal_{i:03d}'}
            db.write(atoms, data=data)
        
        db._close()
        
        # Create properties CSV
        data = {
            'crystal_id': [f'crystal_{i:03d}' for i in range(5)],
            'bandgap': [2.0 + i * 0.1 for i in range(5)],
            'melting_point': [150.0 + i * 10.0 for i in range(5)]
        }
        self.df = pd.DataFrame(data)
        
        self.output_db = self.temp_path / 'output.db'
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_merge_properties(self):
        """Test merging properties into database."""
        merge_properties(
            str(self.input_db),
            self.df,
            str(self.output_db),
            ['bandgap', 'melting_point'],
            'crystal_id'
        )
        
        # Verify output database
        db = connect(str(self.output_db))
        
        # Check all rows have properties
        for row in db.select():
            self.assertIn('bandgap', row.data)
            self.assertIn('melting_point', row.data)
            self.assertIn('crystal_id', row.data)
        
        # Check number of rows
        n_rows = len([row for row in db.select()])
        self.assertEqual(n_rows, 5)
        
        db._close()
    
    def test_merge_preserves_structure(self):
        """Test that merging preserves crystal structure."""
        merge_properties(
            str(self.input_db),
            self.df,
            str(self.output_db),
            ['bandgap', 'melting_point'],
            'crystal_id'
        )
        
        # Compare structures
        input_db = connect(str(self.input_db))
        output_db = connect(str(self.output_db))
        
        input_atoms = next(input_db.select()).toatoms()
        output_atoms = next(output_db.select()).toatoms()
        
        # Check atomic numbers match
        self.assertTrue(np.array_equal(
            input_atoms.get_atomic_numbers(),
            output_atoms.get_atomic_numbers()
        ))
        
        input_db._close()
        output_db._close()


class TestStatisticsComputation(unittest.TestCase):
    """Test statistics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create database with properties
        self.db_path = self.temp_path / 'test.db'
        db = connect(str(self.db_path))
        
        for i in range(10):
            atoms = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            data = {
                'bandgap': 2.0 + i * 0.1,
                'melting_point': 150.0 + i * 10.0
            }
            db.write(atoms, data=data)
        
        db._close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_compute_statistics(self):
        """Test computing statistics."""
        stats = compute_and_display_statistics(
            str(self.db_path),
            ['bandgap', 'melting_point']
        )
        
        # Check structure
        self.assertIn('bandgap', stats)
        self.assertIn('melting_point', stats)
        
        # Check bandgap stats
        self.assertIn('mean', stats['bandgap'])
        self.assertIn('std', stats['bandgap'])
        self.assertIn('min', stats['bandgap'])
        self.assertIn('max', stats['bandgap'])
        self.assertIn('n_samples', stats['bandgap'])
        
        # Check values are reasonable
        self.assertEqual(stats['bandgap']['n_samples'], 10)
        self.assertGreater(stats['bandgap']['mean'], 0)
        self.assertGreater(stats['bandgap']['std'], 0)


class TestDatasetSplits(unittest.TestCase):
    """Test dataset splitting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create database
        self.db_path = self.temp_path / 'test.db'
        db = connect(str(self.db_path))
        
        for i in range(100):
            atoms = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
            db.write(atoms, data={})
        
        db._close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_splits(self):
        """Test creating train/val/test splits."""
        output_dir = self.temp_path / 'splits'
        
        create_splits(
            str(self.db_path),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            output_dir=output_dir
        )
        
        # Check splits file exists
        splits_path = output_dir / 'dataset_splits.json'
        self.assertTrue(splits_path.exists())
        
        # Load and verify splits
        with open(splits_path) as f:
            splits = json.load(f)
        
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        # Check sizes
        self.assertEqual(len(splits['train']), 80)
        self.assertEqual(len(splits['val']), 10)
        self.assertEqual(len(splits['test']), 10)
        
        # Check no overlap
        train_set = set(splits['train'])
        val_set = set(splits['val'])
        test_set = set(splits['test'])
        
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)
    
    def test_invalid_split_ratios(self):
        """Test that invalid split ratios raise an error."""
        output_dir = self.temp_path / 'splits'
        
        with self.assertRaises(ValueError):
            create_splits(
                str(self.db_path),
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1.0
                output_dir=output_dir
            )


if __name__ == '__main__':
    import json
    unittest.main()
