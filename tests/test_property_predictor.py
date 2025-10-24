"""
Unit Tests for Property Predictor

Tests the PropertyPredictor model and related functionality.
"""

import unittest
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crystal.evaluation.property_predictor import PropertyPredictor, SimpleEGNNLayer


class TestPropertyPredictor(unittest.TestCase):
    """Test PropertyPredictor model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.property_names = ['bandgap', 'melting_point']
        self.batch_size = 4
        self.n_atoms = 20
        self.hidden_dim = 64  # Smaller for faster tests
        
    def test_initialization(self):
        """Test model initialization."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=3,
        )
        
        self.assertEqual(predictor.property_names, self.property_names)
        self.assertEqual(predictor.hidden_dim, self.hidden_dim)
        self.assertEqual(predictor.n_layers, 3)
        
        # Check property heads
        self.assertEqual(len(predictor.property_heads), len(self.property_names))
        for prop_name in self.property_names:
            self.assertIn(prop_name, predictor.property_heads)
    
    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Empty property names
        with self.assertRaises(ValueError):
            PropertyPredictor(property_names=[])
        
        # Invalid n_layers
        with self.assertRaises(ValueError):
            PropertyPredictor(property_names=['bandgap'], n_layers=1)
        
        with self.assertRaises(ValueError):
            PropertyPredictor(property_names=['bandgap'], n_layers=10)
        
        # Invalid hidden_dim
        with self.assertRaises(ValueError):
            PropertyPredictor(property_names=['bandgap'], hidden_dim=-1)
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=2,
        )
        
        # Create dummy data
        positions = torch.randn(self.batch_size, self.n_atoms, 3)
        cell = torch.eye(3).unsqueeze(0).expand(self.batch_size, -1, -1)
        atomic_numbers = torch.randint(1, 20, (self.batch_size, self.n_atoms))
        
        # Forward pass
        predictions = predictor(positions, cell, atomic_numbers)
        
        # Check output
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), len(self.property_names))
        
        for prop_name in self.property_names:
            self.assertIn(prop_name, predictions)
            self.assertEqual(predictions[prop_name].shape, (self.batch_size, 1))
    
    def test_forward_pass_normalized(self):
        """Test forward pass with normalized output."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=2,
        )
        
        # Set normalization parameters
        mean = torch.tensor([2.5, 180.0])
        std = torch.tensor([1.2, 50.0])
        predictor.set_normalization_params(mean, std)
        
        # Create dummy data
        positions = torch.randn(self.batch_size, self.n_atoms, 3)
        cell = torch.eye(3).unsqueeze(0).expand(self.batch_size, -1, -1)
        atomic_numbers = torch.randint(1, 20, (self.batch_size, self.n_atoms))
        
        # Forward pass with normalized output
        predictions_norm = predictor(
            positions, cell, atomic_numbers,
            return_normalized=True
        )
        
        # Forward pass with denormalized output
        predictions_denorm = predictor(
            positions, cell, atomic_numbers,
            return_normalized=False
        )
        
        # Check that outputs differ
        for prop_name in self.property_names:
            self.assertFalse(
                torch.allclose(predictions_norm[prop_name], predictions_denorm[prop_name])
            )
    
    def test_input_validation(self):
        """Test input validation."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=2,
        )
        
        # Wrong positions shape
        with self.assertRaises(ValueError):
            positions = torch.randn(self.batch_size, self.n_atoms)  # Missing dimension
            cell = torch.eye(3).unsqueeze(0).expand(self.batch_size, -1, -1)
            atomic_numbers = torch.randint(1, 20, (self.batch_size, self.n_atoms))
            predictor(positions, cell, atomic_numbers)
        
        # Wrong cell shape
        with self.assertRaises(ValueError):
            positions = torch.randn(self.batch_size, self.n_atoms, 3)
            cell = torch.eye(3)  # Missing batch dimension
            atomic_numbers = torch.randint(1, 20, (self.batch_size, self.n_atoms))
            predictor(positions, cell, atomic_numbers)
        
        # Wrong atomic_numbers shape
        with self.assertRaises(ValueError):
            positions = torch.randn(self.batch_size, self.n_atoms, 3)
            cell = torch.eye(3).unsqueeze(0).expand(self.batch_size, -1, -1)
            atomic_numbers = torch.randint(1, 20, (self.batch_size, self.n_atoms, 1))
            predictor(positions, cell, atomic_numbers)
    
    def test_normalization_params(self):
        """Test setting and getting normalization parameters."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=2,
        )
        
        # Set normalization parameters
        mean = torch.tensor([2.5, 180.0])
        std = torch.tensor([1.2, 50.0])
        predictor.set_normalization_params(mean, std)
        
        # Get normalization parameters
        mean_out, std_out = predictor.get_normalization_params()
        
        self.assertTrue(torch.allclose(mean, mean_out))
        self.assertTrue(torch.allclose(std, std_out))
    
    def test_normalization_params_validation(self):
        """Test normalization parameter validation."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=2,
        )
        
        # Wrong shape for mean
        with self.assertRaises(ValueError):
            mean = torch.tensor([2.5])  # Should be 2 elements
            std = torch.tensor([1.2, 50.0])
            predictor.set_normalization_params(mean, std)
        
        # Wrong shape for std
        with self.assertRaises(ValueError):
            mean = torch.tensor([2.5, 180.0])
            std = torch.tensor([1.2])  # Should be 2 elements
            predictor.set_normalization_params(mean, std)
        
        # Zero std
        with self.assertRaises(ValueError):
            mean = torch.tensor([2.5, 180.0])
            std = torch.tensor([1.2, 0.0])  # Zero std
            predictor.set_normalization_params(mean, std)
        
        # Negative std
        with self.assertRaises(ValueError):
            mean = torch.tensor([2.5, 180.0])
            std = torch.tensor([1.2, -1.0])  # Negative std
            predictor.set_normalization_params(mean, std)
    
    def test_save_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        predictor = PropertyPredictor(
            property_names=self.property_names,
            hidden_dim=self.hidden_dim,
            n_layers=2,
        )
        
        # Set normalization parameters
        mean = torch.tensor([2.5, 180.0])
        std = torch.tensor([1.2, 50.0])
        predictor.set_normalization_params(mean, std)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            
            checkpoint = {
                'model_state_dict': predictor.state_dict(),
                'property_names': self.property_names,
                'property_mean': predictor.property_mean,
                'property_std': predictor.property_std,
                'model_config': {
                    'hidden_dim': self.hidden_dim,
                    'n_layers': 2,
                    'max_neighbors': 32,
                    'cutoff_radius': 8.0,
                }
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path)
            
            # Create new model and load state
            new_predictor = PropertyPredictor(
                property_names=loaded_checkpoint['property_names'],
                **loaded_checkpoint['model_config']
            )
            new_predictor.load_state_dict(loaded_checkpoint['model_state_dict'])
            new_predictor.set_normalization_params(
                loaded_checkpoint['property_mean'],
                loaded_checkpoint['property_std']
            )
            
            # Test that loaded model produces same output
            positions = torch.randn(self.batch_size, self.n_atoms, 3)
            cell = torch.eye(3).unsqueeze(0).expand(self.batch_size, -1, -1)
            atomic_numbers = torch.randint(1, 20, (self.batch_size, self.n_atoms))
            
            predictor.eval()
            new_predictor.eval()
            
            with torch.no_grad():
                pred1 = predictor(positions, cell, atomic_numbers)
                pred2 = new_predictor(positions, cell, atomic_numbers)
            
            for prop_name in self.property_names:
                self.assertTrue(
                    torch.allclose(pred1[prop_name], pred2[prop_name], atol=1e-5)
                )


class TestSimpleEGNNLayer(unittest.TestCase):
    """Test SimpleEGNNLayer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = SimpleEGNNLayer(hidden_dim=64, edge_dim=1)
        self.assertEqual(layer.hidden_dim, 64)
        self.assertEqual(layer.edge_dim, 1)
    
    def test_forward_pass(self):
        """Test forward pass through layer."""
        layer = SimpleEGNNLayer(hidden_dim=64, edge_dim=1)
        
        n_atoms = 20
        n_edges = 100
        
        h = torch.randn(n_atoms, 64)
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.randint(0, n_atoms, (2, n_edges))
        edge_attr = torch.randn(n_edges, 1)
        
        h_out = layer(h, positions, edge_index, edge_attr)
        
        # Check output shape
        self.assertEqual(h_out.shape, (n_atoms, 64))
    
    def test_forward_pass_no_edges(self):
        """Test forward pass with no edges."""
        layer = SimpleEGNNLayer(hidden_dim=64, edge_dim=1)
        
        n_atoms = 20
        
        h = torch.randn(n_atoms, 64)
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)
        
        h_out = layer(h, positions, edge_index, edge_attr)
        
        # Should return input unchanged
        self.assertTrue(torch.allclose(h, h_out))


if __name__ == '__main__':
    unittest.main()
