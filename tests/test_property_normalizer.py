"""
Unit tests for PropertyNormalizer utility
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
from crystal.data.property_normalizer import PropertyNormalizer


class TestPropertyNormalizer(unittest.TestCase):
    """Test suite for PropertyNormalizer utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mean = torch.tensor([2.5, 180.0, 3.0])
        self.std = torch.tensor([1.2, 50.0, 0.8])
        self.property_names = ['bandgap', 'melting_point', 'dielectric_constant']
        
        self.normalizer = PropertyNormalizer(
            self.mean,
            self.std,
            self.property_names
        )
    
    def test_initialization(self):
        """Test normalizer initialization."""
        self.assertTrue(torch.allclose(self.normalizer.mean, self.mean))
        self.assertTrue(torch.allclose(self.normalizer.std, self.std))
        self.assertEqual(self.normalizer.property_names, self.property_names)
    
    def test_initialization_invalid(self):
        """Test initialization with invalid parameters."""
        # Mismatched shapes
        with self.assertRaises(ValueError):
            PropertyNormalizer(
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0])
            )
        
        # Negative std
        with self.assertRaises(ValueError):
            PropertyNormalizer(
                torch.tensor([1.0, 2.0]),
                torch.tensor([-1.0, 1.0])
            )
        
        # Zero std
        with self.assertRaises(ValueError):
            PropertyNormalizer(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.0, 1.0])
            )
        
        # Property names length mismatch
        with self.assertRaises(ValueError):
            PropertyNormalizer(
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 1.0]),
                ['prop1']  # Should be 2 names
            )
    
    def test_normalize(self):
        """Test property normalization."""
        properties = torch.tensor([
            [2.5, 180.0, 3.0],  # Mean values
            [3.7, 230.0, 3.8],  # Other values
        ])
        
        normalized = self.normalizer.normalize(properties)
        
        # First sample (at mean) should be ~0
        self.assertTrue(torch.allclose(normalized[0], torch.zeros(3), atol=1e-6))
        
        # Check second sample
        expected = (properties[1] - self.mean) / (self.std + 1e-8)
        self.assertTrue(torch.allclose(normalized[1], expected, atol=1e-5))
    
    def test_denormalize(self):
        """Test property denormalization."""
        # Normalize then denormalize should recover original
        properties = torch.tensor([
            [2.5, 180.0, 3.0],
            [3.7, 230.0, 3.8],
            [1.3, 130.0, 2.2],
        ])
        
        normalized = self.normalizer.normalize(properties)
        recovered = self.normalizer.denormalize(normalized)
        
        self.assertTrue(torch.allclose(properties, recovered, atol=1e-5))
    
    def test_normalize_invalid_shape(self):
        """Test normalization with invalid shape."""
        # Wrong property dimension
        properties_wrong = torch.randn(8, 2)  # Should be 3
        with self.assertRaises(ValueError):
            self.normalizer.normalize(properties_wrong)
    
    def test_denormalize_invalid_shape(self):
        """Test denormalization with invalid shape."""
        # Wrong property dimension
        properties_wrong = torch.randn(8, 5)  # Should be 3
        with self.assertRaises(ValueError):
            self.normalizer.denormalize(properties_wrong)
    
    def test_save_load(self):
        """Test saving and loading normalization parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'normalization.json'
            
            # Save
            self.normalizer.save(path)
            self.assertTrue(path.exists())
            
            # Load
            loaded_normalizer = PropertyNormalizer.load(path)
            
            # Check loaded values match original
            self.assertTrue(torch.allclose(loaded_normalizer.mean, self.mean))
            self.assertTrue(torch.allclose(loaded_normalizer.std, self.std))
            self.assertEqual(loaded_normalizer.property_names, self.property_names)
    
    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'subdir' / 'normalization.json'
            
            # Directory doesn't exist yet
            self.assertFalse(path.parent.exists())
            
            # Save should create it
            self.normalizer.save(path)
            self.assertTrue(path.exists())
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            PropertyNormalizer.load('nonexistent.json')
    
    def test_device_transfer(self):
        """Test that normalizer works with tensors on different devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Properties on CUDA
        properties = torch.randn(4, 3).cuda()
        
        # Normalize (normalizer parameters on CPU)
        normalized = self.normalizer.normalize(properties)
        
        # Output should be on same device as input
        self.assertTrue(normalized.is_cuda)
        
        # Denormalize
        recovered = self.normalizer.denormalize(normalized)
        self.assertTrue(recovered.is_cuda)
        
        # Should recover original values
        self.assertTrue(torch.allclose(properties, recovered, atol=1e-5))
    
    def test_multiple_batch_dimensions(self):
        """Test with multiple batch dimensions."""
        # Shape: [2, 4, 3] - two batches of 4 samples each
        properties = torch.randn(2, 4, 3)
        
        normalized = self.normalizer.normalize(properties)
        recovered = self.normalizer.denormalize(normalized)
        
        self.assertEqual(normalized.shape, properties.shape)
        self.assertTrue(torch.allclose(properties, recovered, atol=1e-5))
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.normalizer)
        
        self.assertIn('PropertyNormalizer', repr_str)
        self.assertIn('property_dim=3', repr_str)
        self.assertIn('bandgap', repr_str)


if __name__ == '__main__':
    unittest.main()
