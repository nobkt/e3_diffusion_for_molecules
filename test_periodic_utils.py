"""
Unit tests for periodic boundary condition utilities.
"""

import torch
import numpy as np

from crystal.data.periodic_utils import (
    minimum_image_distance,
    cartesian_to_fractional,
    fractional_to_cartesian,
    cell_params_to_vectors,
    cell_vectors_to_params,
    wrap_to_unit_cell,
    build_periodic_neighbor_list,
)


def test_coordinate_conversion():
    """Test conversion between Cartesian and fractional coordinates."""
    # Create a simple cubic cell
    cell_vectors = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ], dtype=torch.float32)
    
    # Test positions
    cartesian = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.5, 2.5, 2.5]
    ], dtype=torch.float32)
    
    # Convert to fractional and back
    fractional = cartesian_to_fractional(cartesian, cell_vectors)
    cartesian_back = fractional_to_cartesian(fractional, cell_vectors)
    
    # Check round-trip conversion
    assert torch.allclose(cartesian, cartesian_back, atol=1e-5)
    
    # Check fractional values
    expected_frac = cartesian / 5.0
    assert torch.allclose(fractional, expected_frac, atol=1e-5)


def test_cell_params_conversion():
    """Test conversion between cell parameters and vectors."""
    # Test with a cubic cell
    lengths = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
    angles = torch.tensor([90.0, 90.0, 90.0], dtype=torch.float32)
    
    # Convert to vectors and back
    cell_vectors = cell_params_to_vectors(lengths, angles)
    lengths_back, angles_back = cell_vectors_to_params(cell_vectors)
    
    # Check round-trip conversion
    assert torch.allclose(lengths, lengths_back, atol=1e-4)
    assert torch.allclose(angles, angles_back, atol=1e-3)
    
    # Check cell vectors for cubic cell
    expected_vectors = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ], dtype=torch.float32)
    assert torch.allclose(cell_vectors, expected_vectors, atol=1e-4)


def test_minimum_image_distance():
    """Test minimum image distance calculation."""
    # Create a cubic cell
    cell_vectors = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ], dtype=torch.float32)
    
    # Test positions at opposite ends of cell
    pos1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    pos2 = torch.tensor([[9.0, 0.0, 0.0]], dtype=torch.float32)
    
    # Compute distance with PBC
    distances, vectors = minimum_image_distance(pos1, pos2, cell_vectors)
    
    # Distance should be 2.0 (wrapping around)
    assert torch.allclose(distances, torch.tensor([[2.0]]), atol=1e-4)


def test_wrap_to_unit_cell():
    """Test wrapping fractional coordinates to unit cell."""
    fractional = torch.tensor([
        [0.5, 0.5, 0.5],
        [1.5, -0.3, 2.1]
    ], dtype=torch.float32)
    
    wrapped = wrap_to_unit_cell(fractional)
    
    # Check that all coordinates are in [0, 1)
    assert torch.all(wrapped >= 0.0)
    assert torch.all(wrapped < 1.0)
    
    # Check specific values
    expected = torch.tensor([
        [0.5, 0.5, 0.5],
        [0.5, 0.7, 0.1]
    ], dtype=torch.float32)
    assert torch.allclose(wrapped, expected, atol=1e-4)


def test_build_periodic_neighbor_list():
    """Test building periodic neighbor list."""
    # Create a simple cubic cell
    cell_vectors = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ], dtype=torch.float32)
    
    # Place atoms
    positions = torch.tensor([
        [0.5, 0.5, 0.5],
        [4.5, 0.5, 0.5],  # Close to first atom through PBC
        [2.5, 2.5, 2.5]   # Far from others
    ], dtype=torch.float32)
    
    # Build neighbor list with cutoff 2.0
    edge_index, edge_vectors, edge_distances = build_periodic_neighbor_list(
        positions, cell_vectors, cutoff=2.0
    )
    
    # Should find neighbors between atoms 0 and 1 (distance ~1.0 with PBC)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] >= 2  # At least the pair (0, 1) and (1, 0)


if __name__ == '__main__':
    print("Running periodic utilities tests...")
    test_coordinate_conversion()
    print("✓ Coordinate conversion test passed")
    
    test_cell_params_conversion()
    print("✓ Cell parameter conversion test passed")
    
    test_minimum_image_distance()
    print("✓ Minimum image distance test passed")
    
    test_wrap_to_unit_cell()
    print("✓ Wrap to unit cell test passed")
    
    test_build_periodic_neighbor_list()
    print("✓ Build neighbor list test passed")
    
    print("\nAll tests passed! ✓")
