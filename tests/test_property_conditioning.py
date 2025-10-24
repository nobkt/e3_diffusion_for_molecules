"""
Unit tests for PropertyConditioning module
"""

import unittest
import torch
from crystal.conditioning import PropertyConditioning


class TestPropertyConditioning(unittest.TestCase):
    """Test suite for PropertyConditioning module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.property_names = ['bandgap', 'melting_point']
        self.property_dim = len(self.property_names)
        self.conditioning_dim = 256
        self.batch_size = 8
        
        # Create module
        self.prop_cond = PropertyConditioning(
            property_names=self.property_names,
            conditioning_dim=self.conditioning_dim
        )
        
        # Set normalization parameters
        self.mean = torch.tensor([2.5, 180.0])
        self.std = torch.tensor([1.2, 50.0])
        self.prop_cond.set_normalization_params(self.mean, self.std)
    
    def test_initialization(self):
        """Test module initialization."""
        self.assertEqual(self.prop_cond.property_dim, self.property_dim)
        self.assertEqual(self.prop_cond.conditioning_dim, self.conditioning_dim)
        self.assertEqual(len(self.prop_cond.property_names), self.property_dim)
    
    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Empty property names
        with self.assertRaises(ValueError):
            PropertyConditioning(property_names=[])
        
        # Invalid n_layers
        with self.assertRaises(ValueError):
            PropertyConditioning(
                property_names=['prop1'],
                n_layers=1
            )
        
        # Negative dimensions
        with self.assertRaises(ValueError):
            PropertyConditioning(
                property_names=['prop1'],
                conditioning_dim=-1
            )
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        properties = torch.randn(self.batch_size, self.property_dim)
        output = self.prop_cond(properties)
        
        self.assertEqual(output.shape, (self.batch_size, self.conditioning_dim))
    
    def test_forward_invalid_shape(self):
        """Test forward pass with invalid input shape."""
        # Wrong number of properties
        properties_wrong = torch.randn(self.batch_size, 3)
        with self.assertRaises(ValueError):
            self.prop_cond(properties_wrong)
        
        # 1D tensor
        properties_1d = torch.randn(self.property_dim)
        with self.assertRaises(ValueError):
            self.prop_cond(properties_1d)
    
    def test_normalization(self):
        """Test property normalization."""
        # Properties at mean should normalize to ~0
        properties_mean = self.mean.unsqueeze(0).repeat(self.batch_size, 1)
        normalized = self.prop_cond._normalize(properties_mean)
        
        # Should be close to zero
        self.assertTrue(torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6))
    
    def test_set_normalization_params(self):
        """Test setting normalization parameters."""
        new_mean = torch.tensor([3.0, 200.0])
        new_std = torch.tensor([1.5, 60.0])
        
        self.prop_cond.set_normalization_params(new_mean, new_std)
        
        # Check if parameters were updated
        self.assertTrue(torch.allclose(self.prop_cond.property_mean, new_mean))
        self.assertTrue(torch.allclose(self.prop_cond.property_std, new_std))
    
    def test_set_normalization_params_invalid(self):
        """Test setting normalization parameters with invalid shapes."""
        # Wrong shape for mean
        with self.assertRaises(ValueError):
            self.prop_cond.set_normalization_params(
                torch.tensor([1.0]),
                torch.tensor([1.0, 1.0])
            )
        
        # Zero std
        with self.assertRaises(ValueError):
            self.prop_cond.set_normalization_params(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.0, 1.0])
            )
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        properties = torch.randn(self.batch_size, self.property_dim, requires_grad=True)
        output = self.prop_cond(properties)
        
        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(properties.grad)
        self.assertFalse(torch.all(properties.grad == 0))
    
    def test_device_consistency(self):
        """Test that module works on different devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Move to CUDA
        self.prop_cond.cuda()
        properties = torch.randn(self.batch_size, self.property_dim).cuda()
        
        output = self.prop_cond(properties)
        
        self.assertTrue(output.is_cuda)
        self.assertEqual(output.shape, (self.batch_size, self.conditioning_dim))
    
    def test_batch_independence(self):
        """Test that different batch samples are processed independently."""
        # Process batch
        properties_batch = torch.randn(self.batch_size, self.property_dim)
        output_batch = self.prop_cond(properties_batch)
        
        # Process samples individually
        for i in range(self.batch_size):
            properties_single = properties_batch[i:i+1]
            output_single = self.prop_cond(properties_single)
            
            # Should match batch output
            self.assertTrue(torch.allclose(output_batch[i], output_single[0], atol=1e-5))
    
    def test_get_normalization_params(self):
        """Test retrieving normalization parameters."""
        mean, std = self.prop_cond.get_normalization_params()
        
        self.assertTrue(torch.allclose(mean, self.mean))
        self.assertTrue(torch.allclose(std, self.std))


if __name__ == '__main__':
    unittest.main()
