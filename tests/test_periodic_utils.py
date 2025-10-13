"""
Unit tests for periodic_utils module

Tests the correctness of periodic boundary condition implementations,
coordinate transformations, and cell parameter manipulations.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crystal.data.periodic_utils import (
    minimum_image_distance,
    cartesian_to_fractional,
    fractional_to_cartesian,
    wrap_positions,
    compute_cell_volume,
    cell_params_to_vectors,
    cell_vectors_to_params,
)


def test_coordinate_conversion():
    """Test round-trip conversion between Cartesian and fractional coordinates"""
    # Create a cubic cell
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ]], dtype=torch.float32)
    
    # Test positions
    positions_cart = torch.tensor([[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]], dtype=torch.float32)
    
    # Cartesian -> Fractional -> Cartesian
    positions_frac = cartesian_to_fractional(positions_cart, cell_vectors)
    positions_cart_recovered = fractional_to_cartesian(positions_frac, cell_vectors)
    
    assert torch.allclose(positions_cart, positions_cart_recovered, atol=1e-5), \
        "Round-trip coordinate conversion failed"
    
    # Check fractional coordinates are correct
    expected_frac = positions_cart / 10.0
    assert torch.allclose(positions_frac, expected_frac, atol=1e-5), \
        "Fractional coordinates incorrect"


def test_minimum_image_distance_cubic():
    """Test minimum image distance in cubic cell"""
    # Cubic cell 10x10x10
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ]], dtype=torch.float32)
    
    # Two atoms across periodic boundary
    pos1 = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    pos2 = torch.tensor([[[9.0, 0.0, 0.0]]], dtype=torch.float32)
    
    pbc = torch.tensor([[True, True, True]])
    
    distances, vectors = minimum_image_distance(
        pos1, pos2, cell_vectors, pbc, use_fractional=False
    )
    
    # Minimum distance should be 2.0 (9.0 - 1.0 = 8.0, but wraps to -2.0)
    expected_distance = 2.0
    assert torch.allclose(distances, torch.tensor([[[expected_distance]]]), atol=1e-4), \
        f"Expected distance {expected_distance}, got {distances.item()}"


def test_minimum_image_distance_no_pbc():
    """Test minimum image distance without PBC"""
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ]], dtype=torch.float32)
    
    pos1 = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    pos2 = torch.tensor([[[9.0, 0.0, 0.0]]], dtype=torch.float32)
    
    # No PBC
    pbc = torch.tensor([[False, False, False]])
    
    distances, vectors = minimum_image_distance(
        pos1, pos2, cell_vectors, pbc, use_fractional=False
    )
    
    # Without PBC, distance should be direct: 8.0
    expected_distance = 8.0
    assert torch.allclose(distances, torch.tensor([[[expected_distance]]]), atol=1e-4), \
        f"Expected distance {expected_distance}, got {distances.item()}"


def test_cell_param_conversion():
    """Test round-trip conversion between cell parameters and vectors"""
    # Test case 1: Cubic cell
    cell_params = torch.tensor([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]], dtype=torch.float32)
    cell_vectors = cell_params_to_vectors(cell_params)
    cell_params_recovered = cell_vectors_to_params(cell_vectors)
    
    assert torch.allclose(cell_params, cell_params_recovered, atol=1e-3), \
        "Round-trip cell parameter conversion failed for cubic cell"
    
    # Test case 2: Orthorhombic cell
    cell_params = torch.tensor([[10.0, 12.0, 15.0, 90.0, 90.0, 90.0]], dtype=torch.float32)
    cell_vectors = cell_params_to_vectors(cell_params)
    cell_params_recovered = cell_vectors_to_params(cell_vectors)
    
    assert torch.allclose(cell_params, cell_params_recovered, atol=1e-3), \
        "Round-trip cell parameter conversion failed for orthorhombic cell"
    
    # Test case 3: Monoclinic cell
    cell_params = torch.tensor([[10.0, 12.0, 15.0, 90.0, 105.0, 90.0]], dtype=torch.float32)
    cell_vectors = cell_params_to_vectors(cell_params)
    cell_params_recovered = cell_vectors_to_params(cell_vectors)
    
    assert torch.allclose(cell_params, cell_params_recovered, atol=1e-2), \
        "Round-trip cell parameter conversion failed for monoclinic cell"


def test_wrap_positions():
    """Test wrapping of fractional coordinates"""
    # Fractional coordinates outside unit cell
    positions_frac = torch.tensor([[
        [0.5, 0.5, 0.5],    # Inside
        [1.2, -0.3, 2.7],   # Outside
        [-0.1, 1.5, 0.8],   # Outside
    ]], dtype=torch.float32)
    
    pbc = torch.tensor([[True, True, True]])
    
    wrapped = wrap_positions(positions_frac, pbc)
    
    # All coordinates should be in [0, 1)
    assert torch.all((wrapped >= 0.0) & (wrapped < 1.0)), \
        "Wrapped positions not in [0, 1) range"
    
    # Test with selective PBC
    pbc_partial = torch.tensor([[True, False, True]])
    wrapped_partial = wrap_positions(positions_frac, pbc_partial)
    
    # Only x and z should be wrapped
    assert torch.all((wrapped_partial[..., 0] >= 0.0) & (wrapped_partial[..., 0] < 1.0)), \
        "X coordinate not properly wrapped"
    assert torch.all((wrapped_partial[..., 2] >= 0.0) & (wrapped_partial[..., 2] < 1.0)), \
        "Z coordinate not properly wrapped"
    # Y should be unchanged
    assert torch.allclose(wrapped_partial[..., 1], positions_frac[..., 1]), \
        "Y coordinate should not be wrapped"


def test_compute_cell_volume():
    """Test cell volume computation"""
    # Test case 1: Cubic cell (10x10x10)
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ]], dtype=torch.float32)
    
    volume = compute_cell_volume(cell_vectors)
    expected_volume = 1000.0
    
    assert torch.allclose(volume, torch.tensor([expected_volume]), atol=1e-3), \
        f"Expected volume {expected_volume}, got {volume.item()}"
    
    # Test case 2: Orthorhombic cell (10x12x15)
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 12.0, 0.0],
        [0.0, 0.0, 15.0],
    ]], dtype=torch.float32)
    
    volume = compute_cell_volume(cell_vectors)
    expected_volume = 1800.0
    
    assert torch.allclose(volume, torch.tensor([expected_volume]), atol=1e-3), \
        f"Expected volume {expected_volume}, got {volume.item()}"


def test_batch_processing():
    """Test that functions correctly handle batched inputs"""
    batch_size = 3
    n_atoms = 5
    
    # Create batched data
    positions = torch.randn(batch_size, n_atoms, 3)
    cell_vectors = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1) * 10.0
    pbc = torch.ones(batch_size, 3, dtype=torch.bool)
    
    # Test coordinate conversion
    frac = cartesian_to_fractional(positions, cell_vectors)
    cart = fractional_to_cartesian(frac, cell_vectors)
    
    assert frac.shape == (batch_size, n_atoms, 3), "Fractional shape incorrect"
    assert cart.shape == (batch_size, n_atoms, 3), "Cartesian shape incorrect"
    assert torch.allclose(positions, cart, atol=1e-5), "Batch conversion failed"
    
    # Test minimum image distance
    distances, vectors = minimum_image_distance(
        positions, positions, cell_vectors, pbc, use_fractional=False
    )
    
    assert distances.shape == (batch_size, n_atoms, n_atoms), "Distance shape incorrect"
    assert vectors.shape == (batch_size, n_atoms, n_atoms, 3), "Vector shape incorrect"


if __name__ == '__main__':
    # Run tests
    print("Running periodic_utils tests...")
    
    test_coordinate_conversion()
    print("✓ Coordinate conversion test passed")
    
    test_minimum_image_distance_cubic()
    print("✓ Minimum image distance (cubic) test passed")
    
    test_minimum_image_distance_no_pbc()
    print("✓ Minimum image distance (no PBC) test passed")
    
    test_cell_param_conversion()
    print("✓ Cell parameter conversion test passed")
    
    test_wrap_positions()
    print("✓ Wrap positions test passed")
    
    test_compute_cell_volume()
    print("✓ Cell volume computation test passed")
    
    test_batch_processing()
    print("✓ Batch processing test passed")
    
    print("\nAll tests passed!")
