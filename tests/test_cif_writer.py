"""
Tests for CIF Writer

Test coverage:
- CIF file writing for single and multiple crystals
- Cell parameter and position handling
- Validation and error cases
- Space group information
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from crystal.utils.cif_writer import CIFWriter


class TestCIFWriter(unittest.TestCase):
    """Test CIF writer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_info = {
            'atom_decoder': ['H', 'C', 'N', 'O', 'F']
        }
        
        self.writer = CIFWriter(self.dataset_info)
        
        # Create temporary directory for output
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test CIFWriter initialization."""
        writer = CIFWriter(self.dataset_info, precision=4)
        self.assertEqual(writer.precision, 4)
        self.assertTrue(writer.validate)
        
        # Test without validation
        writer_no_val = CIFWriter(self.dataset_info, validate=False)
        self.assertFalse(writer_no_val.validate)
    
    def test_initialization_errors(self):
        """Test initialization error cases."""
        # None dataset_info
        with self.assertRaises(ValueError) as ctx:
            CIFWriter(None)
        self.assertIn("dataset_info cannot be None", str(ctx.exception))
        
        # Wrong type
        with self.assertRaises(ValueError) as ctx:
            CIFWriter("not a dict")
        self.assertIn("must be a dict", str(ctx.exception))
        
        # Missing atom_decoder
        with self.assertRaises(ValueError) as ctx:
            CIFWriter({})
        self.assertIn("atom_decoder", str(ctx.exception))
    
    def test_write_simple_crystal(self):
        """Test writing a simple crystal structure."""
        crystal = {
            'positions': np.array([
                [0.0, 0.0, 0.0],
                [2.5, 2.5, 2.5]
            ]),
            'atom_types': np.array([1, 1]),  # Both carbon
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "test.cif"
        self.writer.write_cif(crystal, output_path)
        
        self.assertTrue(output_path.exists())
        
        # Read and check content
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("data_crystal", content)
        self.assertIn("_cell_length_a 5.000000", content)
        self.assertIn("_cell_angle_alpha 90.000000", content)
        self.assertIn("C1 C", content)  # Carbon atom
    
    def test_write_with_fractional_coordinates(self):
        """Test writing crystal with fractional coordinates."""
        crystal = {
            'fractional_positions': np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5]
            ]),
            'atom_types': np.array([1, 3]),  # C and O
            'cell_params': np.array([4.0, 4.0, 6.0, 90.0, 90.0, 120.0])
        }
        
        output_path = Path(self.test_dir) / "frac.cif"
        self.writer.write_cif(crystal, output_path)
        
        self.assertTrue(output_path.exists())
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("0.000000 0.000000 0.000000", content)
        self.assertIn("0.500000 0.500000 0.500000", content)
    
    def test_write_with_space_group(self):
        """Test writing crystal with space group information."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([1]),
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "sg.cif"
        self.writer.write_cif(
            crystal,
            output_path,
            space_group_number=225,
            space_group_symbol='Fm-3m'
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("_space_group_IT_number 225", content)
        self.assertIn("_space_group_name_H-M_alt 'Fm-3m'", content)
    
    def test_write_multiple_crystals(self):
        """Test writing multiple crystals."""
        crystals = [
            {
                'positions': np.array([[0.0, 0.0, 0.0]]),
                'atom_types': np.array([1]),
                'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
            },
            {
                'positions': np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]),
                'atom_types': np.array([1, 3]),
                'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
            }
        ]
        
        output_dir = Path(self.test_dir) / "multi"
        paths = self.writer.write_multiple_cifs(crystals, output_dir, prefix="struct")
        
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[0].exists())
        self.assertTrue(paths[1].exists())
        self.assertEqual(paths[0].name, "struct_0000.cif")
        self.assertEqual(paths[1].name, "struct_0001.cif")
    
    def test_write_with_cell_vectors(self):
        """Test writing crystal with cell vectors instead of params."""
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([1]),
            'cell_vectors': cell_vectors
        }
        
        output_path = Path(self.test_dir) / "vectors.cif"
        self.writer.write_cif(crystal, output_path)
        
        self.assertTrue(output_path.exists())
    
    def test_validation_missing_required_field(self):
        """Test validation catches missing required fields."""
        # Missing atom_types
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("atom_types", str(ctx.exception))
    
    def test_validation_missing_positions(self):
        """Test validation catches missing positions."""
        crystal = {
            'atom_types': np.array([1]),
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("positions", str(ctx.exception))
    
    def test_validation_missing_cell(self):
        """Test validation catches missing cell info."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([1])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("cell", str(ctx.exception))
    
    def test_validation_nan_positions(self):
        """Test validation catches NaN in positions."""
        crystal = {
            'positions': np.array([[0.0, np.nan, 0.0]]),
            'atom_types': np.array([1]),
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("NaN", str(ctx.exception))
    
    def test_validation_invalid_cell_lengths(self):
        """Test validation catches invalid cell lengths."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([1]),
            'cell_params': np.array([-5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("positive", str(ctx.exception))
    
    def test_validation_invalid_cell_angles(self):
        """Test validation catches invalid cell angles."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([1]),
            'cell_params': np.array([5.0, 5.0, 5.0, 200.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("angles", str(ctx.exception).lower())
    
    def test_validation_invalid_atom_type(self):
        """Test validation catches invalid atom type indices."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([10]),  # Out of range
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path)
        self.assertIn("Invalid atom type", str(ctx.exception))
    
    def test_validation_invalid_space_group(self):
        """Test validation catches invalid space group numbers."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'atom_types': np.array([1]),
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "invalid.cif"
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_cif(crystal, output_path, space_group_number=300)
        self.assertIn("space_group_number", str(ctx.exception))
    
    def test_fractional_coordinate_wrapping(self):
        """Test that fractional coordinates are wrapped to [0, 1)."""
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0], [7.5, 7.5, 7.5]]),
            'atom_types': np.array([1, 1]),
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "wrapped.cif"
        self.writer.write_cif(crystal, output_path)
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        # Both atoms should be wrapped to [0, 1)
        # Position [7.5, 7.5, 7.5] / [5.0, 5.0, 5.0] = [1.5, 1.5, 1.5] -> [0.5, 0.5, 0.5]
        self.assertIn("0.500000 0.500000 0.500000", content)
    
    def test_one_hot_atom_types(self):
        """Test handling of one-hot encoded atom types."""
        # One-hot encoding: [0, 1, 0, 0, 0] = Carbon (index 1)
        atom_types_one_hot = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0]  # Oxygen (index 3)
        ])
        
        crystal = {
            'positions': np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]),
            'atom_types': atom_types_one_hot,
            'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        }
        
        output_path = Path(self.test_dir) / "onehot.cif"
        self.writer.write_cif(crystal, output_path)
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("C1 C", content)
        self.assertIn("O2 O", content)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensor_support(self):
        """Test that torch tensors are properly handled."""
        crystal = {
            'positions': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            'atom_types': torch.tensor([1], dtype=torch.long),
            'cell_params': torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0], dtype=torch.float32)
        }
        
        output_path = Path(self.test_dir) / "torch.cif"
        self.writer.write_cif(crystal, output_path)
        
        self.assertTrue(output_path.exists())
    
    def test_empty_crystals_list(self):
        """Test error handling for empty crystals list."""
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_multiple_cifs([], Path(self.test_dir))
        self.assertIn("empty", str(ctx.exception))
    
    def test_mismatched_space_group_list(self):
        """Test error handling for mismatched space group list length."""
        crystals = [
            {
                'positions': np.array([[0.0, 0.0, 0.0]]),
                'atom_types': np.array([1]),
                'cell_params': np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
            }
        ]
        
        with self.assertRaises(ValueError) as ctx:
            self.writer.write_multiple_cifs(
                crystals,
                Path(self.test_dir),
                space_group_numbers=[1, 2]  # Wrong length
            )
        self.assertIn("Length", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
