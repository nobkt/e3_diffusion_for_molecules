"""
Integration tests for property-conditioned crystal generation workflow.

Tests the end-to-end workflow from data preparation to training to generation.
"""

import unittest
import tempfile
import shutil
import subprocess
import pandas as pd
import torch
from pathlib import Path
from ase.db import connect
from ase import Atoms
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEndToEndPropertyWorkflow(unittest.TestCase):
    """Integration test for complete property-conditioned workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create small test datasets
        self._create_test_molecule_db()
        self._create_test_crystal_db()
        self._create_test_property_csv()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_molecule_db(self):
        """Create a small molecule database."""
        self.mol_db_path = self.temp_path / 'molecules.db'
        db = connect(str(self.mol_db_path))
        
        # Create 5 simple molecules
        for i in range(5):
            atoms = Atoms(
                'CH4',
                positions=[[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
            )
            data = {'molecule_id': f'mol_{i:03d}'}
            db.write(atoms, data=data)
        
        db._close()
    
    def _create_test_crystal_db(self):
        """Create a small crystal database."""
        self.crystal_db_path = self.temp_path / 'crystals.db'
        db = connect(str(self.crystal_db_path))
        
        # Create 20 simple crystals
        for i in range(20):
            atoms = Atoms(
                'CH4',
                positions=[[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                cell=[10 + i * 0.5, 10 + i * 0.5, 10 + i * 0.5],
                pbc=True
            )
            data = {
                'crystal_id': f'crystal_{i:03d}',
                'molecule_id': f'mol_{i % 5:03d}'
            }
            db.write(atoms, data=data)
        
        db._close()
    
    def _create_test_property_csv(self):
        """Create a test property CSV."""
        self.property_csv_path = self.temp_path / 'properties.csv'
        
        data = {
            'crystal_id': [f'crystal_{i:03d}' for i in range(20)],
            'bandgap': [2.0 + i * 0.1 for i in range(20)],
            'melting_point': [150.0 + i * 10.0 for i in range(20)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.property_csv_path, index=False)
    
    def test_data_preparation(self):
        """Test data preparation script."""
        from scripts.prepare_property_dataset import (
            merge_properties,
            validate_properties,
            compute_and_display_statistics
        )
        
        output_db_path = self.temp_path / 'crystals_with_props.db'
        
        # Load CSV
        df = pd.read_csv(self.property_csv_path)
        
        # Validate
        db = connect(str(self.crystal_db_path))
        validate_properties(db, df, ['bandgap', 'melting_point'], 'crystal_id')
        db._close()
        
        # Merge
        merge_properties(
            str(self.crystal_db_path),
            df,
            str(output_db_path),
            ['bandgap', 'melting_point'],
            'crystal_id'
        )
        
        # Verify output
        db = connect(str(output_db_path))
        n_rows = len([row for row in db.select()])
        self.assertEqual(n_rows, 20)
        
        # Check properties exist
        row = next(db.select())
        self.assertIn('bandgap', row.data)
        self.assertIn('melting_point', row.data)
        
        db._close()
        
        # Compute statistics
        stats = compute_and_display_statistics(
            str(output_db_path),
            ['bandgap', 'melting_point']
        )
        
        self.assertIn('bandgap', stats)
        self.assertIn('melting_point', stats)
    
    def test_dataset_loading_with_properties(self):
        """Test loading dataset with properties."""
        from crystal.data.crystal_loader import CrystalDatasetWithProperties
        
        # First prepare database
        output_db_path = self.temp_path / 'crystals_with_props.db'
        df = pd.read_csv(self.property_csv_path)
        
        from scripts.prepare_property_dataset import merge_properties
        merge_properties(
            str(self.crystal_db_path),
            df,
            str(output_db_path),
            ['bandgap', 'melting_point'],
            'crystal_id'
        )
        
        # Load dataset
        db = connect(str(output_db_path))
        all_indices = [row.id for row in db.select()]
        db._close()
        
        dataset = CrystalDatasetWithProperties(
            db_path=str(output_db_path),
            indices=all_indices,
            property_names=['bandgap', 'melting_point']
        )
        
        # Verify dataset
        self.assertEqual(len(dataset), 20)
        self.assertEqual(len(dataset.property_names), 2)
        
        # Get a sample
        sample = dataset[0]
        self.assertIn('properties', sample)
        self.assertEqual(len(sample['properties']), 2)
        
        # Check statistics
        self.assertEqual(len(dataset.property_mean), 2)
        self.assertEqual(len(dataset.property_std), 2)
        self.assertGreater(dataset.property_mean[0], 0)
        self.assertGreater(dataset.property_std[0], 0)
    
    def test_property_conditioning_initialization(self):
        """Test property conditioning module initialization."""
        from crystal.conditioning import PropertyConditioning
        
        # Prepare dataset to get statistics
        output_db_path = self.temp_path / 'crystals_with_props.db'
        df = pd.read_csv(self.property_csv_path)
        
        from scripts.prepare_property_dataset import merge_properties
        merge_properties(
            str(self.crystal_db_path),
            df,
            str(output_db_path),
            ['bandgap', 'melting_point'],
            'crystal_id'
        )
        
        from crystal.data.crystal_loader import CrystalDatasetWithProperties
        db = connect(str(output_db_path))
        all_indices = [row.id for row in db.select()]
        db._close()
        
        dataset = CrystalDatasetWithProperties(
            db_path=str(output_db_path),
            indices=all_indices,
            property_names=['bandgap', 'melting_point']
        )
        
        # Create property conditioning
        prop_cond = PropertyConditioning(
            property_names=['bandgap', 'melting_point'],
            conditioning_dim=256
        )
        
        # Set normalization params
        prop_cond.set_normalization_params(
            dataset.property_mean,
            dataset.property_std
        )
        
        # Test forward pass
        properties = torch.tensor([[2.5, 180.0]])
        conditioning = prop_cond(properties)
        
        self.assertEqual(conditioning.shape, (1, 256))
    
    def test_extended_combined_conditioning(self):
        """Test ExtendedCombinedConditioning with properties."""
        from crystal.conditioning import (
            PropertyConditioning,
            MolecularConditioning,
            ExtendedCombinedConditioning
        )
        
        # Create property conditioning
        prop_cond = PropertyConditioning(
            property_names=['bandgap', 'melting_point'],
            conditioning_dim=256
        )
        prop_cond.set_normalization_params(
            torch.tensor([2.5, 180.0]),
            torch.tensor([1.2, 50.0])
        )
        
        # Create molecular conditioning
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256
        )
        
        # Create combined conditioning
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=mol_cond,
            property_conditioning=prop_cond,
            conditioning_dim=256
        )
        
        # Test forward pass
        mol_features = torch.randn(1, 128)
        properties = torch.tensor([[2.5, 180.0]])
        
        conditioning = combined(mol_features, properties=properties)
        
        self.assertEqual(conditioning.shape, (1, 256))
    
    def test_checkpoint_with_properties(self):
        """Test saving and loading checkpoint with property statistics."""
        from main_crystal_with_properties import (
            save_checkpoint_with_properties,
            load_checkpoint_with_properties
        )
        
        # Create mock objects
        class MockArgs:
            condition_on_property = True
            property_names = ['bandgap', 'melting_point']
        
        class MockModel:
            def state_dict(self):
                return {'weight': torch.randn(10, 10)}
            def load_state_dict(self, state_dict):
                pass
        
        class MockOptim:
            def state_dict(self):
                return {'lr': 0.001}
            def load_state_dict(self, state_dict):
                pass
        
        class MockDataset:
            property_mean = torch.tensor([2.5, 180.0])
            property_std = torch.tensor([1.2, 50.0])
        
        class MockCondModule:
            def state_dict(self):
                return {}
            def load_state_dict(self, state_dict):
                pass
            def set_normalization_params(self, mean, std):
                pass
        
        args = MockArgs()
        model = MockModel()
        optim = MockOptim()
        dataset = MockDataset()
        
        checkpoint_path = self.temp_path / 'checkpoint.pt'
        
        # Save checkpoint
        save_checkpoint_with_properties(
            args, epoch=5, model=model, optim=optim,
            mol_encoder=None,
            conditioning_modules={'property': MockCondModule()},
            crystal_dataset=dataset,
            best_val_loss=0.5,
            checkpoint_path=checkpoint_path
        )
        
        # Verify checkpoint exists
        self.assertTrue(checkpoint_path.exists())
        
        # Load checkpoint
        start_epoch, best_val_loss = load_checkpoint_with_properties(
            args, checkpoint_path, model, optim, None,
            {'property': MockCondModule()}, 'cpu'
        )
        
        self.assertEqual(start_epoch, 6)  # epoch + 1
        self.assertEqual(best_val_loss, 0.5)


class TestWorkflowIntegration(unittest.TestCase):
    """Test workflow integration without actual model training."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_workflow_components_exist(self):
        """Test that all workflow components are importable."""
        # Training script
        import main_crystal_with_properties
        
        # Generation script
        import generate_crystal_with_all_conditions
        
        # Data preparation
        from scripts import prepare_property_dataset
        
        # Core modules
        from crystal.conditioning import (
            PropertyConditioning,
            ExtendedCombinedConditioning
        )
        from crystal.data.crystal_loader import CrystalDatasetWithProperties
        
        # All imports successful
        self.assertTrue(True)
    
    def test_property_normalizer_persistence(self):
        """Test PropertyNormalizer JSON persistence."""
        from crystal.data.property_normalizer import PropertyNormalizer
        
        # Create normalizer
        normalizer = PropertyNormalizer(
            property_names=['bandgap', 'melting_point']
        )
        
        # Fit with sample data
        properties = torch.tensor([
            [2.0, 150.0],
            [2.5, 180.0],
            [3.0, 200.0]
        ])
        normalizer.fit(properties)
        
        # Save to file
        save_path = self.temp_path / 'normalizer.json'
        normalizer.save_to_json(str(save_path))
        
        # Load from file
        loaded_normalizer = PropertyNormalizer.load_from_json(str(save_path))
        
        # Verify loaded correctly
        self.assertEqual(loaded_normalizer.property_names, normalizer.property_names)
        self.assertTrue(torch.allclose(
            loaded_normalizer.mean,
            normalizer.mean
        ))
        self.assertTrue(torch.allclose(
            loaded_normalizer.std,
            normalizer.std
        ))


if __name__ == '__main__':
    unittest.main()
