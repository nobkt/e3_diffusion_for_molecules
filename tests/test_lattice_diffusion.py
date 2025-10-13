"""
Unit tests for Lattice Diffusion Model.

Tests the lattice_diffusion module for diffusing lattice parameters
with proper normalization and physical constraints.
"""

import torch
import pytest
from crystal.models.lattice_diffusion import LatticeDiffusion


def test_lattice_diffusion_initialization():
    """Test that LatticeDiffusion initializes correctly."""
    model = LatticeDiffusion(hidden_dim=128, num_layers=3, condition_dim=64)
    
    assert model.hidden_dim == 128
    assert model.condition_dim == 64
    assert model.mlp is not None


def test_lattice_diffusion_forward():
    """Test forward pass produces correct output shapes."""
    batch_size = 4
    model = LatticeDiffusion(hidden_dim=64, num_layers=2, condition_dim=32)
    
    # Input: lattice parameters (a, b, c, α, β, γ)
    lattice_params = torch.tensor([
        [10.0, 12.0, 15.0, 90.0, 90.0, 90.0],
        [8.0, 8.0, 12.0, 90.0, 90.0, 120.0],
        [15.0, 15.0, 10.0, 90.0, 90.0, 90.0],
        [12.0, 10.0, 8.0, 85.0, 95.0, 100.0],
    ])
    
    t = torch.rand(batch_size, 1)
    condition = torch.randn(batch_size, 32)
    
    velocity = model(lattice_params, t, condition)
    
    # Check output shape
    assert velocity.shape == (batch_size, 6)


def test_lattice_diffusion_unconditional():
    """Test forward pass without conditioning."""
    batch_size = 3
    model = LatticeDiffusion(hidden_dim=64, num_layers=2, condition_dim=0)
    
    lattice_params = torch.tensor([
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        [12.0, 12.0, 15.0, 90.0, 90.0, 90.0],
        [8.0, 8.0, 10.0, 90.0, 90.0, 120.0],
    ])
    
    t = torch.rand(batch_size, 1)
    
    velocity = model(lattice_params, t)
    
    assert velocity.shape == (batch_size, 6)


def test_normalize_denormalize_roundtrip():
    """Test that normalization and denormalization are inverse operations."""
    lattice_params = torch.tensor([
        [10.0, 12.0, 15.0, 90.0, 90.0, 90.0],
        [8.0, 8.0, 12.0, 90.0, 90.0, 120.0],
        [15.0, 15.0, 10.0, 85.0, 95.0, 100.0],
    ])
    
    # Normalize
    normalized, stats = LatticeDiffusion.normalize_lattice_params(lattice_params)
    
    # Denormalize
    recovered = LatticeDiffusion.denormalize_lattice_params(normalized, stats)
    
    # Check round-trip (with tolerance for constraints)
    assert torch.allclose(lattice_params, recovered, atol=0.1)


def test_physical_constraints():
    """Test that physical constraints are properly applied."""
    # Create parameters that violate constraints
    lattice_params = torch.tensor([
        [0.5, 150.0, 10.0, 20.0, 170.0, 90.0],  # Too small/large lengths, bad angles
        [-5.0, 10.0, 10.0, 90.0, 90.0, 90.0],   # Negative length
        [10.0, 10.0, 10.0, 10.0, 170.0, 200.0], # Bad angles
    ])
    
    constrained = LatticeDiffusion.apply_physical_constraints(lattice_params)
    
    # Check all lengths are in valid range [1, 100]
    lengths = constrained[:, :3]
    assert (lengths >= 1.0).all()
    assert (lengths <= 100.0).all()
    
    # Check all angles are in valid range [30, 150]
    angles = constrained[:, 3:]
    assert (angles >= 30.0).all()
    assert (angles <= 150.0).all()


def test_compute_volume():
    """Test unit cell volume computation."""
    # Cubic cell: a=b=c, α=β=γ=90°
    cubic = torch.tensor([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]])
    volume_cubic = LatticeDiffusion.compute_volume(cubic)
    assert torch.abs(volume_cubic - 1000.0) < 1e-3  # 10^3 = 1000
    
    # Orthorhombic: a≠b≠c, α=β=γ=90°
    ortho = torch.tensor([[10.0, 12.0, 15.0, 90.0, 90.0, 90.0]])
    volume_ortho = LatticeDiffusion.compute_volume(ortho)
    assert torch.abs(volume_ortho - 1800.0) < 1e-3  # 10*12*15 = 1800
    
    # Hexagonal: a=b≠c, α=β=90°, γ=120°
    hex_cell = torch.tensor([[10.0, 10.0, 15.0, 90.0, 90.0, 120.0]])
    volume_hex = LatticeDiffusion.compute_volume(hex_cell)
    expected_hex = 10 * 10 * 15 * torch.sqrt(torch.tensor(3.0)) / 2
    assert torch.abs(volume_hex - expected_hex) < 1e-2


def test_check_validity():
    """Test validity checking for lattice parameters."""
    # Valid parameters
    valid = torch.tensor([
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        [8.0, 12.0, 15.0, 85.0, 95.0, 100.0],
    ])
    
    validity = LatticeDiffusion.check_validity(valid)
    assert validity.all()
    
    # Invalid parameters
    invalid = torch.tensor([
        [0.0, 10.0, 10.0, 90.0, 90.0, 90.0],    # Zero length
        [10.0, 10.0, 10.0, 0.0, 90.0, 90.0],    # Zero angle
        [10.0, 10.0, 10.0, 90.0, 90.0, 180.0],  # 180° angle
    ])
    
    validity = LatticeDiffusion.check_validity(invalid)
    assert not validity.any()


def test_lattice_diffusion_gradient_flow():
    """Test that gradients flow through the model."""
    model = LatticeDiffusion(hidden_dim=64, num_layers=2, condition_dim=32)
    
    lattice_params = torch.tensor([
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
    ], requires_grad=True)
    
    t = torch.tensor([[0.5]], requires_grad=True)
    condition = torch.randn(1, 32, requires_grad=True)
    
    velocity = model(lattice_params, t, condition)
    
    loss = velocity.sum()
    loss.backward()
    
    # Check gradients exist
    assert lattice_params.grad is not None
    assert t.grad is not None
    assert condition.grad is not None


def test_lattice_diffusion_time_dependence():
    """Test that output varies with time."""
    model = LatticeDiffusion(hidden_dim=64, num_layers=2, condition_dim=0)
    
    lattice_params = torch.tensor([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]])
    
    t1 = torch.tensor([[0.1]])
    t2 = torch.tensor([[0.9]])
    
    velocity1 = model(lattice_params, t1)
    velocity2 = model(lattice_params, t2)
    
    # Outputs should differ for different times
    assert not torch.allclose(velocity1, velocity2, atol=1e-3)


def test_lattice_diffusion_batch_processing():
    """Test batch processing consistency."""
    model = LatticeDiffusion(hidden_dim=64, num_layers=2, condition_dim=0)
    
    # Process individually
    params1 = torch.tensor([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]])
    params2 = torch.tensor([[12.0, 12.0, 15.0, 90.0, 90.0, 120.0]])
    
    t = torch.tensor([[0.5]])
    
    vel1 = model(params1, t)
    vel2 = model(params2, t)
    
    # Process together
    params_batch = torch.cat([params1, params2], dim=0)
    t_batch = t.repeat(2, 1)
    vel_batch = model(params_batch, t_batch)
    
    # Results should match
    assert torch.allclose(vel1, vel_batch[0:1], atol=1e-5)
    assert torch.allclose(vel2, vel_batch[1:2], atol=1e-5)


def test_normalization_stability():
    """Test that normalization handles edge cases."""
    # Very small and very large values
    lattice_params = torch.tensor([
        [1.0, 1.0, 1.0, 45.0, 45.0, 45.0],
        [50.0, 50.0, 50.0, 135.0, 135.0, 135.0],
        [5.0, 10.0, 20.0, 70.0, 100.0, 110.0],
    ])
    
    # Should not crash
    normalized, stats = LatticeDiffusion.normalize_lattice_params(lattice_params)
    recovered = LatticeDiffusion.denormalize_lattice_params(normalized, stats)
    
    # Check shapes are preserved
    assert normalized.shape == lattice_params.shape
    assert recovered.shape == lattice_params.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
