"""
Unit tests for Molecular EGNN Feature Encoder.

Tests the molecular_encoder module for extracting EGNN features from
single molecules.
"""

import torch
import pytest
from crystal.models.molecular_encoder import MolecularEncoder, create_fully_connected_edges


def test_molecular_encoder_initialization():
    """Test that MolecularEncoder initializes correctly."""
    encoder = MolecularEncoder(
        in_node_nf=10,
        hidden_nf=64,
        n_layers=3,
        global_feature_dim=128,
    )
    
    assert encoder.in_node_nf == 10
    assert encoder.hidden_nf == 64
    assert encoder.global_feature_dim == 128
    assert encoder.molecular_egnn is not None
    assert encoder.global_mlp is not None
    assert encoder.geometry_head is not None


def test_molecular_encoder_forward():
    """Test forward pass produces correct output shapes."""
    batch_size = 4
    n_atoms = 10
    in_node_nf = 5
    hidden_nf = 32
    global_feature_dim = 64
    
    encoder = MolecularEncoder(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        n_layers=2,
        global_feature_dim=global_feature_dim,
    )
    
    # Create input
    h = torch.randn(batch_size, n_atoms, in_node_nf)
    x = torch.randn(batch_size, n_atoms, 3)
    edge_index = create_fully_connected_edges(n_atoms, batch_size, h.device)
    
    # Forward pass
    features = encoder(h, x, edge_index)
    
    # Check outputs
    assert 'node_features' in features
    assert 'global_features' in features
    assert 'mol_size' in features
    assert 'mol_volume' in features
    assert 'principal_axes' in features
    
    # Check shapes
    assert features['node_features'].shape == (batch_size, n_atoms, hidden_nf)
    assert features['global_features'].shape == (batch_size, global_feature_dim)
    assert features['mol_size'].shape == (batch_size, 3)
    assert features['mol_volume'].shape == (batch_size, 1)
    assert features['principal_axes'].shape == (batch_size, 3, 3)


def test_molecular_encoder_with_mask():
    """Test forward pass with node masking."""
    batch_size = 2
    n_atoms = 8
    in_node_nf = 5
    
    encoder = MolecularEncoder(in_node_nf=in_node_nf, hidden_nf=32, n_layers=2)
    
    h = torch.randn(batch_size, n_atoms, in_node_nf)
    x = torch.randn(batch_size, n_atoms, 3)
    edge_index = create_fully_connected_edges(n_atoms, batch_size, h.device)
    
    # Create mask (mask out last 2 atoms)
    node_mask = torch.ones(batch_size, n_atoms, 1)
    node_mask[:, -2:, :] = 0
    
    features = encoder(h, x, edge_index, node_mask=node_mask)
    
    # Masked atoms should have less contribution
    assert features['global_features'].shape == (batch_size, encoder.global_feature_dim)


def test_molecular_encoder_without_geometry():
    """Test forward pass without geometry extraction."""
    batch_size = 2
    n_atoms = 6
    in_node_nf = 4
    
    encoder = MolecularEncoder(in_node_nf=in_node_nf, hidden_nf=32, n_layers=2)
    
    h = torch.randn(batch_size, n_atoms, in_node_nf)
    x = torch.randn(batch_size, n_atoms, 3)
    edge_index = create_fully_connected_edges(n_atoms, batch_size, h.device)
    
    features = encoder(h, x, edge_index, extract_geometry=False)
    
    # Should not have geometry features
    assert 'mol_size' not in features
    assert 'mol_volume' not in features
    assert 'principal_axes' not in features
    
    # Should have basic features
    assert 'node_features' in features
    assert 'global_features' in features


def test_orthogonalize():
    """Test Gram-Schmidt orthogonalization."""
    batch_size = 3
    
    encoder = MolecularEncoder(in_node_nf=5, hidden_nf=32, n_layers=2)
    
    # Create random matrices
    matrices = torch.randn(batch_size, 3, 3)
    
    # Orthogonalize
    orthogonal = encoder._orthogonalize(matrices)
    
    # Check orthogonality
    for b in range(batch_size):
        u1 = orthogonal[b, 0, :]
        u2 = orthogonal[b, 1, :]
        u3 = orthogonal[b, 2, :]
        
        # Check unit vectors
        assert torch.abs(torch.norm(u1) - 1.0) < 1e-5
        assert torch.abs(torch.norm(u2) - 1.0) < 1e-5
        assert torch.abs(torch.norm(u3) - 1.0) < 1e-5
        
        # Check orthogonality
        assert torch.abs(torch.dot(u1, u2)) < 1e-5
        assert torch.abs(torch.dot(u1, u3)) < 1e-5
        assert torch.abs(torch.dot(u2, u3)) < 1e-5


def test_create_fully_connected_edges():
    """Test edge creation for fully connected graphs."""
    n_atoms = 5
    batch_size = 2
    device = torch.device('cpu')
    
    edge_index = create_fully_connected_edges(n_atoms, batch_size, device)
    
    # Should have shape [2, batch * (n_atoms * (n_atoms - 1))]
    expected_edges_per_batch = n_atoms * (n_atoms - 1)
    expected_total_edges = batch_size * expected_edges_per_batch
    
    assert edge_index.shape == (2, expected_total_edges)
    
    # Check no self-loops
    assert (edge_index[0] != edge_index[1]).all()


def test_molecular_encoder_gradient_flow():
    """Test that gradients flow through the encoder."""
    encoder = MolecularEncoder(in_node_nf=5, hidden_nf=32, n_layers=2)
    
    h = torch.randn(2, 6, 5, requires_grad=True)
    x = torch.randn(2, 6, 3, requires_grad=True)
    edge_index = create_fully_connected_edges(6, 2, h.device)
    
    features = encoder(h, x, edge_index)
    
    # Compute loss on global features
    loss = features['global_features'].sum()
    loss.backward()
    
    # Check gradients exist
    assert h.grad is not None
    assert x.grad is not None
    assert torch.any(h.grad != 0)


def test_molecular_encoder_batch_independence():
    """Test that batch samples are processed independently."""
    encoder = MolecularEncoder(in_node_nf=5, hidden_nf=32, n_layers=2)
    
    n_atoms = 6
    
    # Process separately
    h1 = torch.randn(1, n_atoms, 5)
    x1 = torch.randn(1, n_atoms, 3)
    edge_index1 = create_fully_connected_edges(n_atoms, 1, h1.device)
    features1 = encoder(h1, x1, edge_index1)
    
    h2 = torch.randn(1, n_atoms, 5)
    x2 = torch.randn(1, n_atoms, 3)
    edge_index2 = create_fully_connected_edges(n_atoms, 1, h2.device)
    features2 = encoder(h2, x2, edge_index2)
    
    # Process together
    h_batch = torch.cat([h1, h2], dim=0)
    x_batch = torch.cat([x1, x2], dim=0)
    edge_index_batch = create_fully_connected_edges(n_atoms, 2, h_batch.device)
    features_batch = encoder(h_batch, x_batch, edge_index_batch)
    
    # Results should match (within numerical precision)
    assert torch.allclose(features1['global_features'], features_batch['global_features'][0:1], atol=1e-5)
    assert torch.allclose(features2['global_features'], features_batch['global_features'][1:2], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
