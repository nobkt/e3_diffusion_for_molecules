"""
Unit tests for ExtendedCombinedConditioning module
"""

import unittest
import torch
import torch.nn as nn
from crystal.conditioning import (
    MolecularConditioning,
    PropertyConditioning,
    ExtendedCombinedConditioning,
    SpaceGroupEmbedding,
    DensityConditioning,
)


class TestExtendedCombinedConditioning(unittest.TestCase):
    """Test suite for ExtendedCombinedConditioning module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.molecular_feature_dim = 128
        self.conditioning_dim = 256
        
        # Create molecular conditioning (required)
        self.mol_cond = MolecularConditioning(
            molecular_feature_dim=self.molecular_feature_dim,
            conditioning_dim=self.conditioning_dim,
            use_geometry=False  # Simplified for testing
        )
        
        # Create property conditioning
        self.prop_cond = PropertyConditioning(
            property_names=['bandgap', 'melting_point'],
            conditioning_dim=self.conditioning_dim
        )
        self.prop_cond.set_normalization_params(
            torch.tensor([2.5, 180.0]),
            torch.tensor([1.2, 50.0])
        )
        
        # Create space group embedding
        self.sg_embed = SpaceGroupEmbedding(
            embedding_dim=self.conditioning_dim
        )
        
        # Create density conditioning
        self.dens_cond = DensityConditioning(
            embedding_dim=self.conditioning_dim
        )
    
    def test_initialization_molecular_only(self):
        """Test initialization with only molecular conditioning."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        self.assertEqual(combined.num_conditionings, 1)
        self.assertIsNotNone(combined.molecular_conditioning)
        self.assertIsNone(combined.property_conditioning)
    
    def test_initialization_with_property(self):
        """Test initialization with property conditioning."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        self.assertEqual(combined.num_conditionings, 2)
        self.assertIsNotNone(combined.property_conditioning)
    
    def test_initialization_all_conditions(self):
        """Test initialization with all conditioning types."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            space_group_embedding=self.sg_embed,
            density_conditioning=self.dens_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        self.assertEqual(combined.num_conditionings, 4)
    
    def test_initialization_no_molecular(self):
        """Test that initialization fails without molecular conditioning."""
        with self.assertRaises(ValueError):
            ExtendedCombinedConditioning(
                molecular_conditioning=None,
                property_conditioning=self.prop_cond
            )
    
    def test_forward_molecular_only(self):
        """Test forward pass with only molecular features."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        mol_features = {
            'global_features': torch.randn(self.batch_size, self.molecular_feature_dim)
        }
        
        output = combined(mol_features)
        
        self.assertEqual(output.shape, (self.batch_size, self.conditioning_dim))
    
    def test_forward_with_properties(self):
        """Test forward pass with molecular and property conditioning."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        mol_features = {
            'global_features': torch.randn(self.batch_size, self.molecular_feature_dim)
        }
        properties = torch.randn(self.batch_size, 2)
        
        output = combined(mol_features, properties=properties)
        
        self.assertEqual(output.shape, (self.batch_size, self.conditioning_dim))
    
    def test_forward_all_conditions(self):
        """Test forward pass with all conditioning types."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            space_group_embedding=self.sg_embed,
            density_conditioning=self.dens_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        mol_features = {
            'global_features': torch.randn(self.batch_size, self.molecular_feature_dim)
        }
        space_group = torch.randint(1, 231, (self.batch_size,))
        density = torch.rand(self.batch_size, 1) * 2.0
        properties = torch.randn(self.batch_size, 2)
        
        output = combined(
            mol_features,
            space_group=space_group,
            density=density,
            properties=properties
        )
        
        self.assertEqual(output.shape, (self.batch_size, self.conditioning_dim))
    
    def test_forward_missing_module_error(self):
        """Test that providing condition without module raises error."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            conditioning_dim=self.conditioning_dim
            # No property_conditioning
        )
        
        mol_features = {
            'global_features': torch.randn(self.batch_size, self.molecular_feature_dim)
        }
        properties = torch.randn(self.batch_size, 2)
        
        # Should raise error
        with self.assertRaises(ValueError):
            combined(mol_features, properties=properties)
    
    def test_gradient_flow(self):
        """Test that gradients flow through all pathways."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        mol_features = {
            'global_features': torch.randn(
                self.batch_size, 
                self.molecular_feature_dim,
                requires_grad=True
            )
        }
        properties = torch.randn(self.batch_size, 2, requires_grad=True)
        
        output = combined(mol_features, properties=properties)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(mol_features['global_features'].grad)
        self.assertIsNotNone(properties.grad)
    
    def test_device_consistency(self):
        """Test module works on CUDA if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        ).cuda()
        
        mol_features = {
            'global_features': torch.randn(
                self.batch_size,
                self.molecular_feature_dim
            ).cuda()
        }
        properties = torch.randn(self.batch_size, 2).cuda()
        
        output = combined(mol_features, properties=properties)
        
        self.assertTrue(output.is_cuda)
        self.assertEqual(output.shape, (self.batch_size, self.conditioning_dim))
    
    def test_optional_conditions(self):
        """Test that optional conditions can be omitted."""
        combined = ExtendedCombinedConditioning(
            molecular_conditioning=self.mol_cond,
            space_group_embedding=self.sg_embed,
            density_conditioning=self.dens_cond,
            property_conditioning=self.prop_cond,
            conditioning_dim=self.conditioning_dim
        )
        
        mol_features = {
            'global_features': torch.randn(self.batch_size, self.molecular_feature_dim)
        }
        
        # Only molecular - should work
        output1 = combined(mol_features)
        self.assertEqual(output1.shape, (self.batch_size, self.conditioning_dim))
        
        # Molecular + properties - should work
        properties = torch.randn(self.batch_size, 2)
        output2 = combined(mol_features, properties=properties)
        self.assertEqual(output2.shape, (self.batch_size, self.conditioning_dim))
        
        # Different outputs when different conditions provided
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
