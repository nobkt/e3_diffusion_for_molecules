"""
Unit tests for property-conditioned generation script components.

Tests the key functions and workflows in generate_crystal_with_all_conditions.py
"""

import unittest
import torch
import tempfile
import shutil
import json
from pathlib import Path
from ase.db import connect
from ase import Atoms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_crystal_with_all_conditions import (
    parse_target_properties,
    save_crystal
)


class TestTargetPropertyParsing(unittest.TestCase):
    """Test parsing of target property values from command line."""
    
    def test_parse_valid_properties(self):
        """Test parsing valid property arguments."""
        class MockArgs:
            pass
        
        args = MockArgs()
        property_names = ['bandgap', 'melting_point']
        unknown_args = ['--target_bandgap', '2.5', '--target_melting_point', '180.0']
        
        target_props = parse_target_properties(args, property_names, unknown_args)
        
        self.assertEqual(len(target_props), 2)
        self.assertAlmostEqual(target_props['bandgap'], 2.5)
        self.assertAlmostEqual(target_props['melting_point'], 180.0)
    
    def test_parse_missing_property(self):
        """Test that missing properties raise an error."""
        class MockArgs:
            pass
        
        args = MockArgs()
        property_names = ['bandgap', 'melting_point']
        unknown_args = ['--target_bandgap', '2.5']  # Missing melting_point
        
        with self.assertRaises(ValueError) as cm:
            parse_target_properties(args, property_names, unknown_args)
        
        self.assertIn('melting_point', str(cm.exception))
    
    def test_parse_invalid_value(self):
        """Test that non-numeric values raise an error."""
        class MockArgs:
            pass
        
        args = MockArgs()
        property_names = ['bandgap']
        unknown_args = ['--target_bandgap', 'invalid']
        
        with self.assertRaises(ValueError) as cm:
            parse_target_properties(args, property_names, unknown_args)
        
        self.assertIn('Invalid value', str(cm.exception))
    
    def test_parse_extra_properties(self):
        """Test that extra properties are ignored."""
        class MockArgs:
            pass
        
        args = MockArgs()
        property_names = ['bandgap']
        unknown_args = [
            '--target_bandgap', '2.5',
            '--target_extra', '100.0'  # Extra, should be ignored
        ]
        
        target_props = parse_target_properties(args, property_names, unknown_args)
        
        self.assertEqual(len(target_props), 2)  # Will include extra
        self.assertAlmostEqual(target_props['bandgap'], 2.5)


class TestCrystalSaving(unittest.TestCase):
    """Test crystal structure saving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_crystal_cif(self):
        """Test saving crystal in CIF format."""
        crystal_data = {
            'positions': [[0, 0, 0], [1, 1, 1]],
            'atomic_numbers': [6, 8],  # C and O
            'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        }
        
        output_path = self.temp_path / 'test_crystal'
        
        save_crystal(crystal_data, output_path, output_format='cif')
        
        # Verify CIF file was created
        cif_path = Path(str(output_path) + '.cif')
        self.assertTrue(cif_path.exists())
        
        # Check file is not empty
        self.assertGreater(cif_path.stat().st_size, 0)
    
    def test_save_crystal_xyz(self):
        """Test saving crystal in XYZ format."""
        crystal_data = {
            'positions': [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            'atomic_numbers': [6, 8, 7],  # C, O, N
            'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        }
        
        output_path = self.temp_path / 'test_crystal'
        
        save_crystal(crystal_data, output_path, output_format='xyz')
        
        # Verify XYZ file was created
        xyz_path = Path(str(output_path) + '.xyz')
        self.assertTrue(xyz_path.exists())
        
        # Check file is not empty
        self.assertGreater(xyz_path.stat().st_size, 0)
    
    def test_save_crystal_both_formats(self):
        """Test saving crystal in both CIF and XYZ formats."""
        crystal_data = {
            'positions': [[0, 0, 0], [1, 1, 1]],
            'atomic_numbers': [6, 6],
            'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        }
        
        output_path = self.temp_path / 'test_crystal'
        
        save_crystal(crystal_data, output_path, output_format='both')
        
        # Verify both files were created
        cif_path = Path(str(output_path) + '.cif')
        xyz_path = Path(str(output_path) + '.xyz')
        
        self.assertTrue(cif_path.exists())
        self.assertTrue(xyz_path.exists())


class TestCheckpointLoading(unittest.TestCase):
    """Test checkpoint loading with property statistics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_structure(self):
        """Test that checkpoint has required fields."""
        # Create a mock checkpoint
        checkpoint = {
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'conditioning_modules': {},
            'args': type('Args', (), {
                'condition_on_molecule': True,
                'condition_on_space_group': False,
                'condition_on_density': False,
                'condition_on_property': True,
                'nf': 128,
                'conditioning_dim': 256,
                'property_hidden_dim': 512,
                'property_n_layers': 3,
                'n_layers': 6,
                'tanh': True,
                'attention': True,
                'norm_constant': 1.0,
                'sin_embedding': False,
                'learn_lattice': True,
                'lattice_hidden_dim': 128,
                'lattice_num_layers': 3,
                'use_fractional_coords': True,
                'periodic_cutoff': 10.0,
                'diffusion_steps': 500,
                'diffusion_noise_schedule': 'polynomial_2',
                'diffusion_noise_precision': 1e-5,
                'diffusion_loss_type': 'l2',
                'remove_h': False
            })(),
            'property_names': ['bandgap', 'melting_point'],
            'property_mean': torch.tensor([2.5, 180.0]),
            'property_std': torch.tensor([1.2, 50.0])
        }
        
        # Save checkpoint
        checkpoint_path = self.temp_path / 'test_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Load and verify
        loaded = torch.load(checkpoint_path)
        
        self.assertIn('property_names', loaded)
        self.assertIn('property_mean', loaded)
        self.assertIn('property_std', loaded)
        self.assertEqual(loaded['property_names'], ['bandgap', 'melting_point'])


class TestMoleculeDataLoading(unittest.TestCase):
    """Test molecule data loading for generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create a small molecule database
        self.mol_db_path = self.temp_path / 'molecules.db'
        db = connect(str(self.mol_db_path))
        
        for i in range(5):
            atoms = Atoms(
                'H2O',
                positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            )
            
            data = {'molecule_id': f'mol_{i:03d}'}
            db.write(atoms, data=data)
        
        db._close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_molecule_loading_by_index(self):
        """Test loading molecule by index."""
        from generate_crystal_with_all_conditions import load_molecule_data
        
        molecule_data = load_molecule_data(
            str(self.mol_db_path),
            molecule_index=0,
            remove_h=False
        )
        
        self.assertIsNotNone(molecule_data)
        self.assertIn('one_hot', molecule_data)
    
    def test_molecule_loading_requires_identifier(self):
        """Test that loading requires either ID or index."""
        from generate_crystal_with_all_conditions import load_molecule_data
        
        with self.assertRaises(ValueError) as cm:
            load_molecule_data(str(self.mol_db_path), remove_h=False)
        
        self.assertIn('molecule_id', str(cm.exception).lower())


if __name__ == '__main__':
    unittest.main()
