"""
Integration test for Phase 2 Model Core components.

Tests that all model components can work together in a simple forward pass.
"""

import torch
import pytest
from crystal.models import (
    MolecularEncoder,
    PeriodicEGNN,
    LatticeDiffusion,
    CrystalDynamics,
    create_fully_connected_edges
)
from crystal.data.periodic_utils import (
    cell_params_to_vectors,
    cell_vectors_to_params
)


def test_molecular_encoder_integration():
    """Test molecular encoder produces usable features."""
    batch_size = 2
    n_mol_atoms = 5
    in_node_nf = 4
    
    encoder = MolecularEncoder(in_node_nf=in_node_nf, hidden_nf=32, n_layers=2)
    
    h = torch.randn(batch_size, n_mol_atoms, in_node_nf)
    x = torch.randn(batch_size, n_mol_atoms, 3)
    edge_index = create_fully_connected_edges(n_mol_atoms, batch_size, h.device)
    
    features = encoder(h, x, edge_index)
    
    assert 'global_features' in features
    assert features['global_features'].shape[0] == batch_size
    print(f"✓ Molecular encoder output shape: {features['global_features'].shape}")


def test_periodic_egnn_with_cell():
    """Test periodic EGNN with cell parameters."""
    batch_size = 2
    n_atoms = 6
    in_node_nf = 4
    
    model = PeriodicEGNN(in_node_nf=in_node_nf, hidden_nf=32, out_node_nf=3, n_layers=2)
    
    h = torch.randn(batch_size, n_atoms, in_node_nf)
    x = torch.randn(batch_size, n_atoms, 3)
    cell = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1) * 10.0
    pbc = torch.ones(batch_size, 3, dtype=torch.bool)
    node_mask = torch.ones(batch_size, n_atoms, 1)
    
    h_out, x_out = model(h, x, cell, pbc, node_mask=node_mask)
    
    assert h_out.shape == (batch_size, n_atoms, 3)
    assert x_out.shape == (batch_size, n_atoms, 3)
    print(f"✓ Periodic EGNN output shapes: h={h_out.shape}, x={x_out.shape}")


def test_lattice_diffusion_with_params():
    """Test lattice diffusion with cell parameters."""
    batch_size = 2
    
    model = LatticeDiffusion(hidden_dim=64, num_layers=2, condition_dim=32)
    
    lattice_params = torch.tensor([
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        [12.0, 8.0, 15.0, 90.0, 90.0, 120.0],
    ])
    
    t = torch.rand(batch_size, 1)
    condition = torch.randn(batch_size, 32)
    
    velocity = model(lattice_params, t, condition)
    
    assert velocity.shape == (batch_size, 6)
    print(f"✓ Lattice diffusion velocity shape: {velocity.shape}")


def test_crystal_dynamics_full_pipeline():
    """Test full crystal dynamics forward pass."""
    batch_size = 2
    n_atoms = 8
    in_node_nf = 4
    context_dim = 32
    
    model = CrystalDynamics(
        in_node_nf=in_node_nf,
        hidden_nf=32,
        n_layers=2,
        context_node_nf=context_dim,
        learn_lattice=True
    )
    
    # Crystal data
    x = torch.randn(batch_size, n_atoms, 3)
    h = torch.randn(batch_size, n_atoms, in_node_nf)
    cell = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1) * 10.0
    pbc = torch.ones(batch_size, 3, dtype=torch.bool)
    node_mask = torch.ones(batch_size, n_atoms, 1)
    
    # Time and context (from molecular encoder)
    t = torch.rand(batch_size)
    context = torch.randn(batch_size, context_dim)
    
    # Forward pass
    velocity_x, velocity_h, velocity_cell = model(
        t=t,
        xh=(x, h),
        cell=cell,
        pbc=pbc,
        node_mask=node_mask,
        context=context
    )
    
    # Note: velocity_h has dimension 3 (out_node_nf) not in_node_nf
    # This is the coordinate velocity output from Periodic EGNN
    assert velocity_x.shape == (batch_size, n_atoms, 3)
    assert velocity_h.shape == (batch_size, n_atoms, 3)  # out_node_nf = 3
    assert velocity_cell.shape == (batch_size, 3, 3)
    
    print(f"✓ Crystal dynamics output shapes:")
    print(f"  velocity_x: {velocity_x.shape}")
    print(f"  velocity_h: {velocity_h.shape}")
    print(f"  velocity_cell: {velocity_cell.shape}")


def test_end_to_end_molecular_to_crystal():
    """Test end-to-end: molecular features → crystal generation."""
    batch_size = 2
    n_mol_atoms = 4
    n_crystal_atoms = 10
    in_node_nf = 3
    
    # 1. Extract molecular features
    mol_encoder = MolecularEncoder(in_node_nf=in_node_nf, hidden_nf=32, n_layers=2)
    
    h_mol = torch.randn(batch_size, n_mol_atoms, in_node_nf)
    x_mol = torch.randn(batch_size, n_mol_atoms, 3)
    edge_index_mol = create_fully_connected_edges(n_mol_atoms, batch_size, h_mol.device)
    
    mol_features = mol_encoder(h_mol, x_mol, edge_index_mol)
    context = mol_features['global_features']  # [batch, global_dim]
    
    # 2. Use molecular features to condition crystal generation
    crystal_model = CrystalDynamics(
        in_node_nf=in_node_nf,
        hidden_nf=32,
        n_layers=2,
        context_node_nf=context.shape[1],
        learn_lattice=True
    )
    
    # Crystal data
    x_crystal = torch.randn(batch_size, n_crystal_atoms, 3)
    h_crystal = torch.randn(batch_size, n_crystal_atoms, in_node_nf)
    cell = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1) * 10.0
    pbc = torch.ones(batch_size, 3, dtype=torch.bool)
    node_mask = torch.ones(batch_size, n_crystal_atoms, 1)
    t = torch.rand(batch_size)
    
    # Generate crystal with molecular conditioning
    velocity_x, velocity_h, velocity_cell = crystal_model(
        t=t,
        xh=(x_crystal, h_crystal),
        cell=cell,
        pbc=pbc,
        node_mask=node_mask,
        context=context  # Molecular features as conditioning
    )
    
    # Verify shapes
    # Note: velocity_h has dimension 3 (out_node_nf) not in_node_nf
    assert velocity_x.shape == (batch_size, n_crystal_atoms, 3)
    assert velocity_h.shape == (batch_size, n_crystal_atoms, 3)  # out_node_nf = 3
    assert velocity_cell.shape == (batch_size, 3, 3)
    
    print("✓ End-to-end molecular → crystal pipeline successful!")
    print(f"  Molecular features shape: {context.shape}")
    print(f"  Crystal velocities computed with molecular conditioning")


def test_cell_parameter_conversions():
    """Test cell parameter conversion utilities."""
    batch_size = 3
    
    # Create cell parameters
    cell_params = torch.tensor([
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        [12.0, 8.0, 15.0, 90.0, 90.0, 120.0],
        [8.0, 8.0, 12.0, 90.0, 90.0, 90.0],
    ])
    
    # Convert to vectors
    cell_vectors = cell_params_to_vectors(cell_params)
    assert cell_vectors.shape == (batch_size, 3, 3)
    
    # Convert back
    cell_params_recovered = cell_vectors_to_params(cell_vectors)
    assert cell_params_recovered.shape == (batch_size, 6)
    
    # Check round-trip (with some tolerance)
    assert torch.allclose(cell_params, cell_params_recovered, atol=1e-3)
    
    print("✓ Cell parameter conversions work correctly")


def test_gradient_flow_through_pipeline():
    """Test that gradients flow through the entire pipeline."""
    batch_size = 1
    n_mol_atoms = 3
    n_crystal_atoms = 5
    in_node_nf = 2
    
    # Create models
    mol_encoder = MolecularEncoder(in_node_nf=in_node_nf, hidden_nf=16, n_layers=1)
    crystal_model = CrystalDynamics(
        in_node_nf=in_node_nf,
        hidden_nf=16,
        n_layers=1,
        context_node_nf=mol_encoder.global_feature_dim,
        learn_lattice=True
    )
    
    # Create data with gradients
    h_mol = torch.randn(batch_size, n_mol_atoms, in_node_nf, requires_grad=True)
    x_mol = torch.randn(batch_size, n_mol_atoms, 3, requires_grad=True)
    edge_index_mol = create_fully_connected_edges(n_mol_atoms, batch_size, h_mol.device)
    
    x_crystal = torch.randn(batch_size, n_crystal_atoms, 3, requires_grad=True)
    h_crystal = torch.randn(batch_size, n_crystal_atoms, in_node_nf, requires_grad=True)
    cell = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1) * 10.0
    cell.requires_grad = True
    pbc = torch.ones(batch_size, 3, dtype=torch.bool)
    node_mask = torch.ones(batch_size, n_crystal_atoms, 1)
    t = torch.tensor([0.5])
    
    # Forward pass
    mol_features = mol_encoder(h_mol, x_mol, edge_index_mol)
    velocity_x, velocity_h, velocity_cell = crystal_model(
        t=t,
        xh=(x_crystal, h_crystal),
        cell=cell,
        pbc=pbc,
        node_mask=node_mask,
        context=mol_features['global_features']
    )
    
    # Compute loss and backprop
    loss = velocity_x.sum() + velocity_h.sum() + velocity_cell.sum()
    loss.backward()
    
    # Check gradients exist
    assert h_mol.grad is not None
    assert x_mol.grad is not None
    assert x_crystal.grad is not None
    assert h_crystal.grad is not None
    assert cell.grad is not None
    
    print("✓ Gradients flow correctly through entire pipeline")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Phase 2 Model Core Integration Tests")
    print("="*60 + "\n")
    
    test_molecular_encoder_integration()
    test_periodic_egnn_with_cell()
    test_lattice_diffusion_with_params()
    test_crystal_dynamics_full_pipeline()
    test_end_to_end_molecular_to_crystal()
    test_cell_parameter_conversions()
    test_gradient_flow_through_pipeline()
    
    print("\n" + "="*60)
    print("✅ All integration tests passed!")
    print("="*60 + "\n")
