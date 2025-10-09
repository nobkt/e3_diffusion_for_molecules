"""
Tests for crystal models.
"""

import torch
from crystal.models import PeriodicEGNN, LatticeDiffusion, CrystalDynamics
from crystal.data.periodic_utils import cell_params_to_vectors


def test_periodic_egnn():
    """Test Periodic EGNN forward pass."""
    print("Testing Periodic EGNN...")
    
    # Setup
    n_nodes = 10
    in_node_nf = 5
    hidden_nf = 32
    out_node_nf = 1
    
    # Create model
    model = PeriodicEGNN(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        out_node_nf=out_node_nf,
        n_layers=2
    )
    
    # Create inputs
    h = torch.randn(n_nodes, in_node_nf)
    x = torch.rand(n_nodes, 3)  # Fractional coordinates
    cell_vectors = torch.eye(3) * 10.0  # 10 Angstrom cubic cell
    
    # Forward pass
    h_out, x_out = model(h, x, cell_vectors)
    
    # Check shapes
    assert h_out.shape == (n_nodes, out_node_nf)
    assert x_out.shape == (n_nodes, 3)
    
    print("✓ Periodic EGNN test passed")


def test_lattice_diffusion():
    """Test Lattice Diffusion model."""
    print("Testing Lattice Diffusion...")
    
    # Setup
    n_nodes = 10
    hidden_nf = 32
    
    # Create model
    model = LatticeDiffusion(hidden_nf=hidden_nf)
    
    # Create inputs
    lengths = torch.tensor([[5.0, 5.0, 5.0]])
    angles = torch.tensor([[90.0, 90.0, 90.0]])
    lattice_params = LatticeDiffusion.normalize_lattice_params(lengths, angles)
    
    h = torch.randn(n_nodes, hidden_nf)
    node_mask = torch.ones(n_nodes, 1)
    
    # Forward pass
    lattice_update = model(lattice_params, h, node_mask)
    
    # Check shape
    assert lattice_update.shape == (1, 6)
    
    # Test denormalization
    lengths_new, angles_new = LatticeDiffusion.denormalize_lattice_params(
        lattice_params + lattice_update
    )
    assert lengths_new.shape == (1, 3)
    assert angles_new.shape == (1, 3)
    
    print("✓ Lattice Diffusion test passed")


def test_crystal_dynamics():
    """Test integrated Crystal Dynamics model."""
    print("Testing Crystal Dynamics...")
    
    # Setup
    n_nodes = 10
    in_node_nf = 5
    hidden_nf = 32
    out_node_nf = 1
    
    # Create model
    model = CrystalDynamics(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        out_node_nf=out_node_nf,
        n_layers=2,
        learn_lattice=True
    )
    
    # Create inputs
    h = torch.randn(n_nodes, in_node_nf)
    x = torch.rand(n_nodes, 3)  # Fractional coordinates
    cell_vectors = torch.eye(3) * 10.0
    node_mask = torch.ones(n_nodes, 1)
    
    # Forward pass
    h_out, x_out, cell_out = model(h, x, cell_vectors, node_mask=node_mask)
    
    # Check shapes
    assert h_out.shape == (n_nodes, out_node_nf)
    assert x_out.shape == (n_nodes, 3)
    assert cell_out.shape == (3, 3)
    
    print("✓ Crystal Dynamics test passed")


def test_crystal_sampling():
    """Test crystal structure sampling."""
    print("Testing crystal sampling...")
    
    # Setup
    n_nodes = 10
    in_node_nf = 5
    hidden_nf = 32
    device = torch.device('cpu')
    
    # Create model
    model = CrystalDynamics(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        n_layers=2,
        learn_lattice=True
    )
    
    # Sample
    h, x, cell = model.sample(n_nodes, in_node_nf, device)
    
    # Check shapes
    assert h.shape == (n_nodes, 1)
    assert x.shape == (n_nodes, 3)
    assert cell.shape == (3, 3)
    
    # Check fractional coordinates are in valid range
    assert torch.all(x >= 0.0) and torch.all(x <= 1.0)
    
    print("✓ Crystal sampling test passed")


if __name__ == '__main__':
    print("Running crystal model tests...\n")
    
    test_periodic_egnn()
    test_lattice_diffusion()
    test_crystal_dynamics()
    test_crystal_sampling()
    
    print("\n✅ All crystal model tests passed!")
