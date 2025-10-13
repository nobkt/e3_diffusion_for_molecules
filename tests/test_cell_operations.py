"""
Tests for Cell Operations

Test coverage:
- Cell parameter to vector conversion and vice versa
- Volume calculations
- Cell standardization and reduction
- Cell transformations
- Cell validation
- Reciprocal lattice
"""

import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from crystal.utils.cell_operations import CellOperations


class TestCellOperations(unittest.TestCase):
    """Test cell operations functionality."""
    
    def test_cell_params_to_vectors_cubic(self):
        """Test conversion for cubic cell."""
        cell_params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        cell_vectors = CellOperations.cell_params_to_vectors(cell_params)
        
        expected = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        np.testing.assert_array_almost_equal(cell_vectors, expected)
    
    def test_cell_params_to_vectors_tetragonal(self):
        """Test conversion for tetragonal cell."""
        cell_params = np.array([5.0, 5.0, 7.0, 90.0, 90.0, 90.0])
        cell_vectors = CellOperations.cell_params_to_vectors(cell_params)
        
        # a and b should be orthogonal, c along z
        np.testing.assert_almost_equal(cell_vectors[0, 1], 0.0)
        np.testing.assert_almost_equal(cell_vectors[1, 0], 0.0)
        np.testing.assert_almost_equal(cell_vectors[2, 2], 7.0)
    
    def test_cell_params_to_vectors_hexagonal(self):
        """Test conversion for hexagonal cell."""
        cell_params = np.array([5.0, 5.0, 7.0, 90.0, 90.0, 120.0])
        cell_vectors = CellOperations.cell_params_to_vectors(cell_params)
        
        # Check gamma angle (120 degrees)
        a_vec = cell_vectors[0]
        b_vec = cell_vectors[1]
        cos_gamma = np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
        angle_gamma = np.rad2deg(np.arccos(cos_gamma))
        
        np.testing.assert_almost_equal(angle_gamma, 120.0, decimal=5)
    
    def test_cell_vectors_to_params_cubic(self):
        """Test conversion back to params for cubic cell."""
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        cell_params = CellOperations.cell_vectors_to_params(cell_vectors)
        
        expected = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        np.testing.assert_array_almost_equal(cell_params, expected)
    
    def test_params_vectors_roundtrip(self):
        """Test roundtrip conversion params -> vectors -> params."""
        original_params = np.array([4.5, 5.0, 6.5, 85.0, 95.0, 110.0])
        
        vectors = CellOperations.cell_params_to_vectors(original_params)
        recovered_params = CellOperations.cell_vectors_to_params(vectors)
        
        np.testing.assert_array_almost_equal(original_params, recovered_params, decimal=5)
    
    def test_vectors_params_roundtrip(self):
        """Test roundtrip conversion vectors -> params -> vectors."""
        original_vectors = np.array([
            [5.0, 0.0, 0.0],
            [1.0, 4.5, 0.0],
            [0.5, 1.0, 6.0]
        ])
        
        params = CellOperations.cell_vectors_to_params(original_vectors)
        recovered_vectors = CellOperations.cell_params_to_vectors(params)
        
        np.testing.assert_array_almost_equal(original_vectors, recovered_vectors, decimal=5)
    
    def test_batched_params_to_vectors(self):
        """Test batched conversion of cell parameters."""
        batch_params = np.array([
            [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
            [4.0, 4.0, 6.0, 90.0, 90.0, 120.0]
        ])
        
        batch_vectors = CellOperations.cell_params_to_vectors(batch_params)
        
        self.assertEqual(batch_vectors.shape, (2, 3, 3))
        
        # Check first batch element (cubic)
        expected_cubic = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        np.testing.assert_array_almost_equal(batch_vectors[0], expected_cubic)
    
    def test_compute_volume_from_params(self):
        """Test volume computation from cell parameters."""
        # Cubic cell: V = a^3
        cell_params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        volume = CellOperations.compute_volume(cell_params=cell_params)
        
        expected_volume = 125.0  # 5^3
        np.testing.assert_almost_equal(volume, expected_volume)
    
    def test_compute_volume_from_vectors(self):
        """Test volume computation from cell vectors."""
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        volume = CellOperations.compute_volume(cell_vectors=cell_vectors)
        
        expected_volume = 125.0
        np.testing.assert_almost_equal(volume, expected_volume)
    
    def test_compute_volume_error_both_inputs(self):
        """Test error when both cell_params and cell_vectors provided."""
        cell_params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        cell_vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.compute_volume(cell_params=cell_params, cell_vectors=cell_vectors)
        self.assertIn("both", str(ctx.exception).lower())
    
    def test_compute_volume_error_neither_input(self):
        """Test error when neither input provided."""
        with self.assertRaises(ValueError) as ctx:
            CellOperations.compute_volume()
        self.assertIn("must provide", str(ctx.exception).lower())
    
    def test_standardize_cell(self):
        """Test cell standardization."""
        # Start with a tilted cell
        cell_vectors = np.array([
            [4.0, 1.0, 0.5],
            [0.0, 5.0, 0.3],
            [0.0, 0.0, 6.0]
        ])
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        
        std_cell, std_pos = CellOperations.standardize_cell(cell_vectors, positions)
        
        # Standardized cell should have specific form
        # a along x, b in xy plane, etc.
        self.assertAlmostEqual(std_cell[0, 1], 0.0)
        self.assertAlmostEqual(std_cell[0, 2], 0.0)
        self.assertAlmostEqual(std_cell[1, 2], 0.0)
        
        # Check volume is preserved
        orig_vol = np.abs(np.linalg.det(cell_vectors))
        std_vol = np.abs(np.linalg.det(std_cell))
        np.testing.assert_almost_equal(orig_vol, std_vol)
    
    def test_transform_cell(self):
        """Test cell transformation."""
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        
        # Scale by 2 in all directions
        transform = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        new_cell, new_pos = CellOperations.transform_cell(cell_vectors, positions, transform)
        
        # Cell should be doubled
        expected_cell = cell_vectors * 2.0
        np.testing.assert_array_almost_equal(new_cell, expected_cell)
        
        # Positions in fractional coords should be unchanged
        np.testing.assert_array_almost_equal(new_pos, positions)
    
    def test_transform_cell_singular_matrix_error(self):
        """Test error for singular transformation matrix."""
        cell_vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        positions = np.array([[0.0, 0.0, 0.0]])
        
        # Singular matrix (det = 0)
        singular_transform = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.transform_cell(cell_vectors, positions, singular_transform)
        self.assertIn("singular", str(ctx.exception).lower())
    
    def test_validate_cell_valid(self):
        """Test validation of valid cell."""
        cell_params = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        is_valid = CellOperations.validate_cell(cell_params=cell_params)
        self.assertTrue(is_valid)
    
    def test_validate_cell_length_too_small(self):
        """Test validation rejects too small lengths."""
        cell_params = np.array([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        is_valid = CellOperations.validate_cell(
            cell_params=cell_params,
            length_bounds=(1.0, 100.0)
        )
        self.assertFalse(is_valid)
    
    def test_validate_cell_length_too_large(self):
        """Test validation rejects too large lengths."""
        cell_params = np.array([150.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        is_valid = CellOperations.validate_cell(
            cell_params=cell_params,
            length_bounds=(1.0, 100.0)
        )
        self.assertFalse(is_valid)
    
    def test_validate_cell_angle_too_small(self):
        """Test validation rejects too small angles."""
        cell_params = np.array([5.0, 5.0, 5.0, 25.0, 90.0, 90.0])
        
        is_valid = CellOperations.validate_cell(
            cell_params=cell_params,
            angle_bounds=(30.0, 150.0)
        )
        self.assertFalse(is_valid)
    
    def test_validate_cell_angle_too_large(self):
        """Test validation rejects too large angles."""
        cell_params = np.array([5.0, 5.0, 5.0, 90.0, 160.0, 90.0])
        
        is_valid = CellOperations.validate_cell(
            cell_params=cell_params,
            angle_bounds=(30.0, 150.0)
        )
        self.assertFalse(is_valid)
    
    def test_validate_cell_volume_too_small(self):
        """Test validation rejects too small volume."""
        # Very thin cell
        cell_params = np.array([10.0, 10.0, 0.05, 90.0, 90.0, 90.0])
        
        is_valid = CellOperations.validate_cell(
            cell_params=cell_params,
            volume_min=1.0
        )
        self.assertFalse(is_valid)
    
    def test_get_reciprocal_cell(self):
        """Test reciprocal lattice computation."""
        # Simple cubic cell
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        reciprocal = CellOperations.get_reciprocal_cell(cell_vectors)
        
        # For cubic cell, reciprocal should be scaled by 2π/a
        expected_scale = 2 * np.pi / 5.0
        expected = np.eye(3) * expected_scale
        
        np.testing.assert_array_almost_equal(reciprocal, expected, decimal=5)
    
    def test_reciprocal_orthogonality(self):
        """Test orthogonality relation for reciprocal lattice."""
        # Arbitrary cell
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [1.0, 4.0, 0.0],
            [0.5, 0.5, 6.0]
        ])
        
        reciprocal = CellOperations.get_reciprocal_cell(cell_vectors)
        
        # Check: a_i · b*_j = 2π δ_ij
        for i in range(3):
            for j in range(3):
                dot_product = np.dot(cell_vectors[i], reciprocal[j])
                if i == j:
                    expected = 2 * np.pi
                else:
                    expected = 0.0
                np.testing.assert_almost_equal(dot_product, expected, decimal=5)
    
    def test_error_negative_cell_length(self):
        """Test error for negative cell length."""
        cell_params = np.array([-5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.cell_params_to_vectors(cell_params)
        self.assertIn("positive", str(ctx.exception))
    
    def test_error_invalid_angle(self):
        """Test error for invalid angle (>180 or <0)."""
        cell_params = np.array([5.0, 5.0, 5.0, 200.0, 90.0, 90.0])
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.cell_params_to_vectors(cell_params)
        self.assertIn("180", str(ctx.exception))
    
    def test_error_negative_volume(self):
        """Test error for parameters that would give negative volume."""
        # Impossible geometry - would require cz^2 < 0
        cell_params = np.array([5.0, 5.0, 5.0, 10.0, 170.0, 170.0])
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.cell_params_to_vectors(cell_params)
        self.assertIn("volume", str(ctx.exception).lower())
    
    def test_error_wrong_shape_params(self):
        """Test error for wrong shape of cell_params."""
        cell_params = np.array([5.0, 5.0, 5.0])  # Only 3 elements
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.cell_params_to_vectors(cell_params)
        self.assertIn("shape", str(ctx.exception).lower())
    
    def test_error_wrong_shape_vectors(self):
        """Test error for wrong shape of cell_vectors."""
        cell_vectors = np.array([[5.0, 0.0], [0.0, 5.0]])  # Wrong shape
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.cell_vectors_to_params(cell_vectors)
        self.assertIn("shape", str(ctx.exception).lower())
    
    def test_error_zero_length_vector(self):
        """Test error for zero-length cell vector."""
        cell_vectors = np.array([
            [0.0, 0.0, 0.0],  # Zero vector
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        with self.assertRaises(ValueError) as ctx:
            CellOperations.cell_vectors_to_params(cell_vectors)
        self.assertIn("non-zero", str(ctx.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensor_support_params_to_vectors(self):
        """Test torch tensor support for params to vectors."""
        cell_params = torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        cell_vectors = CellOperations.cell_params_to_vectors(cell_params)
        
        self.assertIsInstance(cell_vectors, torch.Tensor)
        self.assertEqual(cell_vectors.shape, (3, 3))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensor_support_vectors_to_params(self):
        """Test torch tensor support for vectors to params."""
        cell_vectors = torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        cell_params = CellOperations.cell_vectors_to_params(cell_vectors)
        
        self.assertIsInstance(cell_params, torch.Tensor)
        self.assertEqual(cell_params.shape, (6,))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensor_volume(self):
        """Test torch tensor support for volume computation."""
        cell_vectors = torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        volume = CellOperations.compute_volume(cell_vectors=cell_vectors)
        
        self.assertIsInstance(volume, torch.Tensor)
        self.assertAlmostEqual(volume.item(), 125.0)


if __name__ == '__main__':
    unittest.main()
