"""
Tests for conditioning and evaluation modules.
"""

import torch
import numpy as np
from crystal.conditioning import (
    SpaceGroupEmbedding,
    CrystalSystemEmbedding,
    CombinedSymmetryEmbedding,
    DensityConditioning,
    VolumeConditioning,
    CombinedPropertyConditioning
)
from crystal.evaluation import CrystalMetrics


def test_space_group_embedding():
    """Test space group embedding."""
    print("Testing Space Group Embedding...")
    
    # Create model
    model = SpaceGroupEmbedding(embedding_dim=32)
    
    # Test input
    space_groups = torch.tensor([1, 50, 100, 150, 230])
    
    # Forward pass
    embeddings = model(space_groups)
    
    # Check shape
    assert embeddings.shape == (5, 32)
    
    print("✓ Space Group Embedding test passed")


def test_crystal_system_embedding():
    """Test crystal system embedding."""
    print("Testing Crystal System Embedding...")
    
    # Create model
    model = CrystalSystemEmbedding(embedding_dim=16)
    
    # Test input (various space groups)
    space_groups = torch.tensor([1, 15, 74, 142, 167, 194, 230])
    
    # Forward pass
    embeddings = model(space_groups)
    
    # Check shape
    assert embeddings.shape == (7, 16)
    
    print("✓ Crystal System Embedding test passed")


def test_combined_symmetry_embedding():
    """Test combined symmetry embedding."""
    print("Testing Combined Symmetry Embedding...")
    
    # Create model
    model = CombinedSymmetryEmbedding(sg_embedding_dim=32, cs_embedding_dim=16)
    
    # Test input
    space_groups = torch.tensor([1, 100, 230])
    
    # Forward pass
    embeddings = model(space_groups)
    
    # Check shape (should be sg_dim + cs_dim = 48)
    assert embeddings.shape == (3, 48)
    
    print("✓ Combined Symmetry Embedding test passed")


def test_density_conditioning():
    """Test density conditioning."""
    print("Testing Density Conditioning...")
    
    # Create model
    model = DensityConditioning(embedding_dim=32)
    
    # Test input (densities in g/cm³)
    densities = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # Forward pass
    embeddings = model(densities)
    
    # Check shape
    assert embeddings.shape == (4, 32)
    
    # Test normalization/denormalization
    normalized = model.normalize_density(densities)
    denormalized = model.denormalize_density(normalized)
    
    # Should be close to original
    assert torch.allclose(densities, denormalized, rtol=1e-3, atol=1e-3)
    
    print("✓ Density Conditioning test passed")


def test_volume_conditioning():
    """Test volume conditioning."""
    print("Testing Volume Conditioning...")
    
    # Create model
    model = VolumeConditioning(embedding_dim=32)
    
    # Test input (volumes in Ų)
    volumes = torch.tensor([500.0, 1000.0, 5000.0])
    
    # Forward pass
    embeddings = model(volumes)
    
    # Check shape
    assert embeddings.shape == (3, 32)
    
    print("✓ Volume Conditioning test passed")


def test_combined_property_conditioning():
    """Test combined property conditioning."""
    print("Testing Combined Property Conditioning...")
    
    # Create model
    model = CombinedPropertyConditioning(
        density_dim=32,
        volume_dim=32,
        use_density=True,
        use_volume=True
    )
    
    # Test input
    densities = torch.tensor([1.0, 2.0, 3.0])
    volumes = torch.tensor([500.0, 1000.0, 1500.0])
    
    # Forward pass
    embeddings = model(density=densities, volume=volumes)
    
    # Check shape (should be 64)
    assert embeddings.shape == (3, 64)
    
    print("✓ Combined Property Conditioning test passed")


def test_crystal_metrics():
    """Test crystal metrics."""
    print("Testing Crystal Metrics...")
    
    # Create fake data
    gen_params = np.random.rand(10, 6) * 10 + 5  # Lattice params
    ref_params = np.random.rand(20, 6) * 10 + 5
    
    # Test lattice parameter MAE
    mae_metrics = CrystalMetrics.lattice_parameter_mae(gen_params, ref_params)
    assert 'mae_a' in mae_metrics
    assert 'mae_lengths' in mae_metrics
    
    # Test density statistics
    gen_densities = np.random.rand(10) * 2 + 1
    ref_densities = np.random.rand(20) * 2 + 1
    density_metrics = CrystalMetrics.density_statistics(gen_densities, ref_densities)
    assert 'density_mae' in density_metrics
    assert 'density_emd' in density_metrics
    
    # Test volume statistics
    gen_volumes = np.random.rand(10) * 1000 + 500
    ref_volumes = np.random.rand(20) * 1000 + 500
    volume_metrics = CrystalMetrics.volume_statistics(gen_volumes, ref_volumes)
    assert 'volume_mae' in volume_metrics
    
    # Test validity check
    positions_list = [np.random.rand(10, 3) * 10 for _ in range(5)]
    validity_metrics = CrystalMetrics.validity_check(positions_list, min_distance=0.5)
    assert 'validity_rate' in validity_metrics
    assert 0.0 <= validity_metrics['validity_rate'] <= 1.0
    
    # Test diversity score
    diversity = CrystalMetrics.diversity_score(gen_params)
    assert isinstance(diversity, (float, np.floating))
    
    print("✓ Crystal Metrics test passed")


if __name__ == '__main__':
    print("Running conditioning and evaluation tests...\n")
    
    test_space_group_embedding()
    test_crystal_system_embedding()
    test_combined_symmetry_embedding()
    test_density_conditioning()
    test_volume_conditioning()
    test_combined_property_conditioning()
    test_crystal_metrics()
    
    print("\n✅ All conditioning and evaluation tests passed!")
