"""
Unit tests for conditioning modules.

Tests space group embedding, density conditioning, and molecular conditioning.
"""

import torch
import pytest
from crystal.conditioning import (
    SpaceGroupEmbedding,
    DensityConditioning,
    MolecularConditioning,
    CombinedConditioning,
)


class TestSpaceGroupEmbedding:
    """Tests for SpaceGroupEmbedding module."""
    
    def test_initialization(self):
        """Test that SpaceGroupEmbedding initializes correctly."""
        embedding = SpaceGroupEmbedding(embedding_dim=64, num_space_groups=230)
        
        assert embedding.embedding_dim == 64
        assert embedding.num_space_groups == 230
        assert embedding.embedding is not None
        assert embedding.projection is not None
        
        # Check embedding table size: 231 (0-230, where 0 is padding)
        assert embedding.embedding.num_embeddings == 231
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            SpaceGroupEmbedding(embedding_dim=-1)
        
        with pytest.raises(ValueError):
            SpaceGroupEmbedding(num_space_groups=0)
    
    def test_forward_pass_1d(self):
        """Test forward pass with 1D input."""
        batch_size = 4
        embedding_dim = 32
        embedding = SpaceGroupEmbedding(embedding_dim=embedding_dim)
        
        # Create valid space group numbers
        space_group = torch.randint(1, 231, (batch_size,))
        
        output = embedding(space_group)
        
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.any(torch.isnan(output))
    
    def test_forward_pass_2d(self):
        """Test forward pass with 2D input."""
        batch_size = 4
        embedding_dim = 32
        embedding = SpaceGroupEmbedding(embedding_dim=embedding_dim)
        
        # Create valid space group numbers [batch, 1]
        space_group = torch.randint(1, 231, (batch_size, 1))
        
        output = embedding(space_group)
        
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.any(torch.isnan(output))
    
    def test_padding_index(self):
        """Test that padding index (0) returns zero vector."""
        embedding_dim = 32
        embedding = SpaceGroupEmbedding(embedding_dim=embedding_dim)
        
        # Space group 0 should return zeros
        space_group = torch.tensor([0])
        output = embedding(space_group)
        
        # After projection, may not be exactly zero, but embedding layer should pad
        # Check that the embedding itself is zero
        raw_emb = embedding.embedding(space_group)
        assert torch.allclose(raw_emb, torch.zeros_like(raw_emb))
    
    def test_invalid_space_group_negative(self):
        """Test that negative space groups raise error."""
        embedding = SpaceGroupEmbedding()
        
        space_group = torch.tensor([-1, 1, 2])
        
        with pytest.raises(ValueError, match="negative"):
            embedding(space_group)
    
    def test_invalid_space_group_too_large(self):
        """Test that space groups > 230 raise error."""
        embedding = SpaceGroupEmbedding(num_space_groups=230)
        
        space_group = torch.tensor([1, 2, 231])
        
        with pytest.raises(ValueError, match="230"):
            embedding(space_group)
    
    def test_invalid_shape(self):
        """Test that invalid tensor shapes raise error."""
        embedding = SpaceGroupEmbedding()
        
        # 3D tensor not allowed
        space_group = torch.randint(1, 231, (4, 2, 3))
        
        with pytest.raises(ValueError, match="shape"):
            embedding(space_group)
    
    def test_batch_independence(self):
        """Test that samples in batch are processed independently."""
        embedding = SpaceGroupEmbedding(embedding_dim=32)
        
        # Process individually
        sg1 = torch.tensor([1])
        sg2 = torch.tensor([2])
        
        out1 = embedding(sg1)
        out2 = embedding(sg2)
        
        # Process as batch
        sg_batch = torch.tensor([1, 2])
        out_batch = embedding(sg_batch)
        
        assert torch.allclose(out_batch[0], out1[0], atol=1e-6)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-6)
    
    def test_get_embedding_table(self):
        """Test getting raw embedding table."""
        embedding_dim = 32
        num_space_groups = 230
        embedding = SpaceGroupEmbedding(
            embedding_dim=embedding_dim,
            num_space_groups=num_space_groups
        )
        
        table = embedding.get_embedding_table()
        
        assert table.shape == (num_space_groups + 1, embedding_dim)
        # Index 0 should be zeros (padding)
        assert torch.allclose(table[0], torch.zeros(embedding_dim))


class TestDensityConditioning:
    """Tests for DensityConditioning module."""
    
    def test_initialization(self):
        """Test that DensityConditioning initializes correctly."""
        conditioning = DensityConditioning(
            embedding_dim=64,
            density_min=0.5,
            density_max=5.0
        )
        
        assert conditioning.embedding_dim == 64
        assert conditioning.density_min == 0.5
        assert conditioning.density_max == 5.0
        assert conditioning.mlp is not None
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            DensityConditioning(embedding_dim=-1)
        
        with pytest.raises(ValueError):
            DensityConditioning(density_min=-1.0)
        
        with pytest.raises(ValueError):
            DensityConditioning(density_min=5.0, density_max=1.0)
    
    def test_forward_pass_1d(self):
        """Test forward pass with 1D input."""
        batch_size = 4
        embedding_dim = 32
        conditioning = DensityConditioning(embedding_dim=embedding_dim)
        
        # Create valid density values
        density = torch.rand(batch_size) * 4.0 + 0.5  # Range [0.5, 4.5]
        
        output = conditioning(density)
        
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.any(torch.isnan(output))
    
    def test_forward_pass_2d(self):
        """Test forward pass with 2D input."""
        batch_size = 4
        embedding_dim = 32
        conditioning = DensityConditioning(embedding_dim=embedding_dim)
        
        # Create valid density values [batch, 1]
        density = torch.rand(batch_size, 1) * 4.0 + 0.5
        
        output = conditioning(density)
        
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.any(torch.isnan(output))
    
    def test_density_normalization(self):
        """Test that densities are properly normalized."""
        conditioning = DensityConditioning(
            embedding_dim=32,
            density_min=1.0,
            density_max=3.0
        )
        
        # Densities at boundaries
        density = torch.tensor([1.0, 2.0, 3.0])
        
        output = conditioning(density)
        
        # Should not raise errors and produce valid outputs
        assert output.shape == (3, 32)
        assert not torch.any(torch.isnan(output))
    
    def test_out_of_range_clamping(self):
        """Test that out-of-range densities are clamped."""
        conditioning = DensityConditioning(
            embedding_dim=32,
            density_min=1.0,
            density_max=3.0
        )
        
        # Slightly out of range (should be clamped, not error)
        density = torch.tensor([0.5, 2.0, 5.0])
        
        output = conditioning(density)
        
        # Should produce valid output (clamped internally)
        assert output.shape == (3, 32)
        assert not torch.any(torch.isnan(output))
    
    def test_invalid_density_nan(self):
        """Test that NaN densities raise error."""
        conditioning = DensityConditioning()
        
        density = torch.tensor([1.0, float('nan'), 2.0])
        
        with pytest.raises(ValueError, match="NaN"):
            conditioning(density)
    
    def test_invalid_density_inf(self):
        """Test that infinite densities raise error."""
        conditioning = DensityConditioning()
        
        density = torch.tensor([1.0, float('inf'), 2.0])
        
        with pytest.raises(ValueError, match="infinite"):
            conditioning(density)
    
    def test_invalid_density_negative(self):
        """Test that negative/zero densities raise error."""
        conditioning = DensityConditioning()
        
        density = torch.tensor([1.0, -1.0, 2.0])
        
        with pytest.raises(ValueError, match="positive"):
            conditioning(density)
        
        density = torch.tensor([0.0, 1.0, 2.0])
        
        with pytest.raises(ValueError, match="positive"):
            conditioning(density)
    
    def test_invalid_shape(self):
        """Test that invalid tensor shapes raise error."""
        conditioning = DensityConditioning()
        
        # 3D tensor not allowed
        density = torch.rand(4, 2, 3)
        
        with pytest.raises(ValueError, match="shape"):
            conditioning(density)
    
    def test_denormalize_density(self):
        """Test density denormalization."""
        conditioning = DensityConditioning(
            density_min=1.0,
            density_max=3.0
        )
        
        normalized = torch.tensor([[0.0], [0.5], [1.0]])
        
        density = conditioning.denormalize_density(normalized)
        
        expected = torch.tensor([[1.0], [2.0], [3.0]])
        assert torch.allclose(density, expected, atol=1e-6)


class TestMolecularConditioning:
    """Tests for MolecularConditioning module."""
    
    def test_initialization_without_geometry(self):
        """Test initialization without geometric features."""
        conditioning = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        assert conditioning.molecular_feature_dim == 128
        assert conditioning.conditioning_dim == 256
        assert not conditioning.use_geometry
    
    def test_initialization_with_geometry(self):
        """Test initialization with geometric features."""
        conditioning = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=True
        )
        
        assert conditioning.use_geometry
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            MolecularConditioning(molecular_feature_dim=-1, conditioning_dim=256)
        
        with pytest.raises(ValueError):
            MolecularConditioning(molecular_feature_dim=128, conditioning_dim=-1)
    
    def test_forward_without_geometry(self):
        """Test forward pass without geometric features."""
        batch_size = 4
        feature_dim = 128
        conditioning_dim = 256
        
        conditioning = MolecularConditioning(
            molecular_feature_dim=feature_dim,
            conditioning_dim=conditioning_dim,
            use_geometry=False
        )
        
        # Create molecular features
        features = {
            'global_features': torch.randn(batch_size, feature_dim)
        }
        
        output = conditioning(features)
        
        assert output.shape == (batch_size, conditioning_dim)
        assert not torch.any(torch.isnan(output))
    
    def test_forward_with_geometry(self):
        """Test forward pass with geometric features."""
        batch_size = 4
        feature_dim = 128
        conditioning_dim = 256
        
        conditioning = MolecularConditioning(
            molecular_feature_dim=feature_dim,
            conditioning_dim=conditioning_dim,
            use_geometry=True
        )
        
        # Create molecular features with geometry
        features = {
            'global_features': torch.randn(batch_size, feature_dim),
            'mol_size': torch.rand(batch_size, 3),
            'mol_volume': torch.rand(batch_size, 1),
            'principal_axes': torch.randn(batch_size, 3, 3)
        }
        
        output = conditioning(features)
        
        assert output.shape == (batch_size, conditioning_dim)
        assert not torch.any(torch.isnan(output))
    
    def test_missing_global_features(self):
        """Test that missing global_features raises error."""
        conditioning = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        features = {}  # Missing global_features
        
        with pytest.raises(ValueError, match="global_features"):
            conditioning(features)
    
    def test_missing_geometry_features(self):
        """Test that missing geometry features raise error when use_geometry=True."""
        batch_size = 4
        conditioning = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=True
        )
        
        # Missing geometric features
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        
        with pytest.raises(ValueError, match="missing features"):
            conditioning(features)
    
    def test_wrong_feature_dimensions(self):
        """Test that wrong feature dimensions raise error."""
        batch_size = 4
        conditioning = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        # Wrong dimension
        features = {
            'global_features': torch.randn(batch_size, 64)  # Should be 128
        }
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            conditioning(features)
    
    def test_wrong_geometry_shapes(self):
        """Test that wrong geometry shapes raise error."""
        batch_size = 4
        conditioning = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=True
        )
        
        # Wrong mol_size shape
        features = {
            'global_features': torch.randn(batch_size, 128),
            'mol_size': torch.rand(batch_size, 2),  # Should be [batch, 3]
            'mol_volume': torch.rand(batch_size, 1),
            'principal_axes': torch.randn(batch_size, 3, 3)
        }
        
        with pytest.raises(ValueError, match="mol_size"):
            conditioning(features)


class TestCombinedConditioning:
    """Tests for CombinedConditioning module."""
    
    def test_initialization_molecular_only(self):
        """Test initialization with only molecular conditioning."""
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256
        )
        
        assert combined.molecular_conditioning is not None
        assert combined.space_group_embedding is None
        assert combined.density_conditioning is None
    
    def test_initialization_all_conditions(self):
        """Test initialization with all conditioning types."""
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        sg_emb = SpaceGroupEmbedding(embedding_dim=64)
        dens_cond = DensityConditioning(embedding_dim=64)
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256,
            space_group_embedding=sg_emb,
            density_conditioning=dens_cond
        )
        
        assert combined.molecular_conditioning is not None
        assert combined.space_group_embedding is not None
        assert combined.density_conditioning is not None
        # Should have 3 weights (molecular + sg + density)
        assert combined.condition_weights.shape == (3,)
    
    def test_missing_molecular_conditioning(self):
        """Test that missing molecular conditioning raises error."""
        with pytest.raises(ValueError, match="molecular_conditioning"):
            CombinedConditioning(
                molecular_conditioning=None,
                conditioning_dim=256
            )
    
    def test_forward_molecular_only(self):
        """Test forward with only molecular features."""
        batch_size = 4
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256
        )
        
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        
        output = combined(features)
        
        assert output.shape == (batch_size, 256)
        assert not torch.any(torch.isnan(output))
    
    def test_forward_with_space_group(self):
        """Test forward with molecular and space group conditioning."""
        batch_size = 4
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        sg_emb = SpaceGroupEmbedding(embedding_dim=64)
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256,
            space_group_embedding=sg_emb
        )
        
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        space_group = torch.randint(1, 231, (batch_size,))
        
        output = combined(features, space_group=space_group)
        
        assert output.shape == (batch_size, 256)
        assert not torch.any(torch.isnan(output))
    
    def test_forward_with_density(self):
        """Test forward with molecular and density conditioning."""
        batch_size = 4
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        dens_cond = DensityConditioning(embedding_dim=64)
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256,
            density_conditioning=dens_cond
        )
        
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        density = torch.rand(batch_size) * 4.0 + 0.5
        
        output = combined(features, density=density)
        
        assert output.shape == (batch_size, 256)
        assert not torch.any(torch.isnan(output))
    
    def test_forward_all_conditions(self):
        """Test forward with all conditioning types."""
        batch_size = 4
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        sg_emb = SpaceGroupEmbedding(embedding_dim=64)
        dens_cond = DensityConditioning(embedding_dim=64)
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256,
            space_group_embedding=sg_emb,
            density_conditioning=dens_cond
        )
        
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        space_group = torch.randint(1, 231, (batch_size,))
        density = torch.rand(batch_size) * 4.0 + 0.5
        
        output = combined(features, space_group=space_group, density=density)
        
        assert output.shape == (batch_size, 256)
        assert not torch.any(torch.isnan(output))
    
    def test_space_group_without_embedding(self):
        """Test that providing space_group without embedding raises error."""
        batch_size = 4
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256
        )
        
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        space_group = torch.randint(1, 231, (batch_size,))
        
        with pytest.raises(ValueError, match="space_group_embedding not initialized"):
            combined(features, space_group=space_group)
    
    def test_density_without_conditioning(self):
        """Test that providing density without conditioning raises error."""
        batch_size = 4
        mol_cond = MolecularConditioning(
            molecular_feature_dim=128,
            conditioning_dim=256,
            use_geometry=False
        )
        
        combined = CombinedConditioning(
            molecular_conditioning=mol_cond,
            conditioning_dim=256
        )
        
        features = {
            'global_features': torch.randn(batch_size, 128)
        }
        density = torch.rand(batch_size) * 4.0 + 0.5
        
        with pytest.raises(ValueError, match="density_conditioning not initialized"):
            combined(features, density=density)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
