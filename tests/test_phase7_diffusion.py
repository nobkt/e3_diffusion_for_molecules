"""
Tests for Phase 7: Crystal Diffusion Sampling and Loss Computation.

Tests the crystal-specific diffusion model, node distribution, and loss computation.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crystal.models.crystal_diffusion import CrystalDiffusion
from crystal.models.crystal_dynamics import CrystalDynamics
from equivariant_diffusion.crystal_distributions import CrystalNodesDistribution, UniformNodesDistribution
from qm9.losses import compute_loss_and_nll_crystal
from crystal.data.periodic_utils import cell_params_to_vectors


class DummyArgs:
    """Dummy arguments for testing."""
    def __init__(self):
        self.probabilistic_model = 'diffusion'
        self.ode_regularization = 0.01


class TestCrystalDiffusion:
    """Test CrystalDiffusion class."""
    
    def test_init(self):
        """Test CrystalDiffusion initialization."""
        # Create dummy dynamics
        dynamics = CrystalDynamics(
            in_node_nf=5,
            hidden_nf=32,
            n_layers=2,
            learn_lattice=True
        )
        
        # Create CrystalDiffusion
        crystal_diff = CrystalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=10,  # Small for testing
            learn_lattice=True
        )
        
        assert crystal_diff.learn_lattice == True
        assert crystal_diff.T == 10
        print("✓ CrystalDiffusion initialization successful")
    
    def test_init_without_lattice_support(self):
        """Test that CrystalDiffusion raises error if dynamics doesn't support lattice."""
        # Create dynamics without lattice support
        class DummyDynamics(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_node_nf = 5
                self.n_dims = 3
            
            def _forward(self, t, x, h, node_mask, edge_mask, context):
                return torch.zeros_like(torch.cat([x, h['categorical'], h.get('integer', torch.zeros_like(h['categorical'][:, :, :1]))], dim=2))
        
        dynamics = DummyDynamics()
        
        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            crystal_diff = CrystalDiffusion(
                dynamics=dynamics,
                in_node_nf=5,
                n_dims=3,
                learn_lattice=True
            )
        
        assert "lattice_diffusion" in str(excinfo.value)
        print("✓ CrystalDiffusion correctly rejects dynamics without lattice support")
    
    def test_sample_cell_noise(self):
        """Test cell noise sampling."""
        dynamics = CrystalDynamics(
            in_node_nf=5,
            hidden_nf=32,
            n_layers=2,
            learn_lattice=True
        )
        
        crystal_diff = CrystalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=10,
            learn_lattice=True
        )
        
        # Sample noise
        batch_size = 4
        cell_params = torch.tensor([[10., 10., 10., 90., 90., 90.]] * batch_size)
        sigma = torch.ones(batch_size, 1) * 0.5
        
        noise = crystal_diff.sample_cell_noise(cell_params, sigma)
        
        assert noise.shape == (batch_size, 6)
        print("✓ Cell noise sampling works correctly")
    
    def test_sample(self):
        """Test crystal sampling."""
        dynamics = CrystalDynamics(
            in_node_nf=5,
            hidden_nf=32,
            n_layers=2,
            learn_lattice=True
        )
        
        crystal_diff = CrystalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=5,  # Very small for fast testing
            learn_lattice=True
        )
        
        # Setup inputs
        n_samples = 2
        n_nodes = 10
        device = torch.device('cpu')
        
        node_mask = torch.ones(n_samples, n_nodes, 1)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0).repeat(n_samples, 1, 1).view(-1, 1)
        context = None
        cell_params = torch.tensor([[15., 15., 15., 90., 90., 90.]] * n_samples)
        pbc = torch.ones(n_samples, 3, dtype=torch.bool)
        
        # Sample
        x, h, final_cell_params = crystal_diff.sample(
            n_samples=n_samples,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context,
            cell_params=cell_params,
            pbc=pbc,
            fix_noise=False
        )
        
        # Check outputs
        assert x.shape == (n_samples, n_nodes, 3)
        assert 'categorical' in h
        assert h['categorical'].shape[0] == n_samples
        assert final_cell_params.shape == (n_samples, 6)
        
        print("✓ Crystal sampling works correctly")
    
    def test_sample_chain(self):
        """Test crystal chain sampling."""
        dynamics = CrystalDynamics(
            in_node_nf=5,
            hidden_nf=32,
            n_layers=2,
            learn_lattice=True
        )
        
        crystal_diff = CrystalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=10,
            learn_lattice=True
        )
        
        # Setup inputs
        n_samples = 1
        n_nodes = 8
        keep_frames = 5
        
        node_mask = torch.ones(n_samples, n_nodes, 1)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0).repeat(n_samples, 1, 1).view(-1, 1)
        context = None
        cell_params = torch.tensor([[15., 15., 15., 90., 90., 90.]])
        pbc = torch.ones(n_samples, 3, dtype=torch.bool)
        
        # Sample chain
        chain, cell_chain = crystal_diff.sample_chain(
            n_samples=n_samples,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context,
            cell_params=cell_params,
            pbc=pbc,
            keep_frames=keep_frames
        )
        
        # Check outputs
        assert chain.shape == (keep_frames * n_samples, n_nodes, 3 + 5)  # positions + features
        assert cell_chain.shape == (keep_frames, n_samples, 6)
        
        print("✓ Crystal chain sampling works correctly")


class TestCrystalNodesDistribution:
    """Test CrystalNodesDistribution class."""
    
    def test_init_uniform(self):
        """Test initialization with uniform distribution."""
        dist = CrystalNodesDistribution(min_nodes=10, max_nodes=50)
        
        assert dist.min_nodes == 10
        assert dist.max_nodes == 50
        assert dist.n_bins == 41
        
        print("✓ CrystalNodesDistribution initialization works")
    
    def test_init_with_histogram(self):
        """Test initialization with provided histogram."""
        # Create simple histogram
        histogram = torch.ones(41)  # Uniform
        histogram[20] = 10.0  # Peak at middle
        
        dist = CrystalNodesDistribution(
            histogram=histogram,
            min_nodes=10,
            max_nodes=50
        )
        
        # Sample should work
        samples = dist.sample(100)
        assert samples.min() >= 10
        assert samples.max() <= 50
        
        print("✓ CrystalNodesDistribution with histogram works")
    
    def test_log_prob(self):
        """Test log probability computation."""
        dist = CrystalNodesDistribution(min_nodes=10, max_nodes=20)
        
        # Test valid values
        x = torch.tensor([10, 15, 20])
        log_probs = dist.log_prob(x)
        
        assert log_probs.shape == (3,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()
        
        print("✓ CrystalNodesDistribution log_prob works")
    
    def test_log_prob_out_of_range(self):
        """Test that out-of-range values raise error."""
        dist = CrystalNodesDistribution(min_nodes=10, max_nodes=20)
        
        # Test out of range
        x = torch.tensor([5, 25])  # Both out of range
        
        with pytest.raises(ValueError) as excinfo:
            dist.log_prob(x)
        
        assert "must be in range" in str(excinfo.value)
        print("✓ CrystalNodesDistribution correctly rejects out-of-range values")
    
    def test_sample(self):
        """Test sampling from distribution."""
        dist = CrystalNodesDistribution(min_nodes=10, max_nodes=50)
        
        samples = dist.sample(1000)
        
        assert samples.shape == (1000,)
        assert samples.min() >= 10
        assert samples.max() <= 50
        assert samples.dtype == torch.long or samples.dtype == torch.int64
        
        print("✓ CrystalNodesDistribution sampling works")
    
    def test_sample_batch(self):
        """Test batch sampling."""
        dist = CrystalNodesDistribution(min_nodes=10, max_nodes=50)
        
        sizes = torch.tensor([10, 10, 10, 10])
        samples = dist.sample_batch(sizes)
        
        assert samples.shape == (4,)
        assert samples.min() >= 10
        assert samples.max() <= 50
        
        print("✓ CrystalNodesDistribution batch sampling works")
    
    def test_from_dataset(self):
        """Test creating distribution from dataset."""
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.data = [
                    {'atom_mask': torch.ones(15, 1)},
                    {'atom_mask': torch.ones(20, 1)},
                    {'atom_mask': torch.ones(18, 1)},
                    {'atom_mask': torch.ones(25, 1)},
                    {'atom_mask': torch.ones(20, 1)},
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = MockDataset()
        
        dist = CrystalNodesDistribution.from_dataset(
            dataset,
            min_nodes=10,
            max_nodes=30
        )
        
        assert dist.min_nodes == 10
        assert dist.max_nodes == 30
        
        # Check that distribution learned from data
        samples = dist.sample(100)
        assert samples.min() >= 10
        assert samples.max() <= 30
        
        print("✓ CrystalNodesDistribution.from_dataset works")


class TestUniformNodesDistribution:
    """Test UniformNodesDistribution class."""
    
    def test_init(self):
        """Test initialization."""
        dist = UniformNodesDistribution(min_nodes=10, max_nodes=50)
        
        assert dist.min_nodes == 10
        assert dist.max_nodes == 50
        assert dist.n_values == 41
        
        print("✓ UniformNodesDistribution initialization works")
    
    def test_log_prob(self):
        """Test log probability (constant for uniform)."""
        dist = UniformNodesDistribution(min_nodes=10, max_nodes=50)
        
        x = torch.tensor([10, 25, 50])
        log_probs = dist.log_prob(x)
        
        # Should be constant
        assert torch.allclose(log_probs, log_probs[0].expand(3))
        
        print("✓ UniformNodesDistribution log_prob works")
    
    def test_sample(self):
        """Test uniform sampling."""
        dist = UniformNodesDistribution(min_nodes=10, max_nodes=50)
        
        samples = dist.sample(1000)
        
        assert samples.shape == (1000,)
        assert samples.min() >= 10
        assert samples.max() <= 50
        
        # Check approximate uniformity
        hist = torch.histc(samples.float(), bins=41, min=10, max=50)
        # Each bin should have roughly 1000/41 ≈ 24 samples
        # Allow significant variance for small sample size
        assert hist.min() > 5
        assert hist.max() < 50
        
        print("✓ UniformNodesDistribution sampling is approximately uniform")


class TestCrystalLossComputation:
    """Test crystal-specific loss computation."""
    
    def test_compute_loss_basic(self):
        """Test basic loss computation."""
        args = DummyArgs()
        
        # Create model
        dynamics = CrystalDynamics(
            in_node_nf=5,
            hidden_nf=32,
            n_layers=2,
            learn_lattice=True
        )
        
        crystal_diff = CrystalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=10,
            learn_lattice=True
        )
        
        # Create nodes distribution
        nodes_dist = UniformNodesDistribution(min_nodes=5, max_nodes=15)
        
        # Create inputs
        batch_size = 4
        n_nodes = 10
        
        x = torch.randn(batch_size, n_nodes, 3)
        h = {
            'categorical': torch.randn(batch_size, n_nodes, 4),
            'integer': torch.randn(batch_size, n_nodes, 1)
        }
        node_mask = torch.ones(batch_size, n_nodes, 1)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1)
        context = None
        cell_params = torch.tensor([[15., 15., 15., 90., 90., 90.]] * batch_size)
        pbc = torch.ones(batch_size, 3, dtype=torch.bool)
        
        # Zero out masked positions
        x = x * node_mask
        
        # Compute loss
        nll, reg_term, mean_abs_z = compute_loss_and_nll_crystal(
            args, crystal_diff, nodes_dist,
            x, h, node_mask, edge_mask, context,
            cell_params=cell_params, pbc=pbc
        )
        
        # Check outputs
        assert nll.dim() == 0  # Scalar
        assert reg_term.dim() == 0  # Scalar
        assert not torch.isnan(nll)
        assert not torch.isinf(nll)
        
        print("✓ Crystal loss computation works")
    
    def test_compute_loss_with_regularization(self):
        """Test loss computation includes cell regularization."""
        args = DummyArgs()
        
        dynamics = CrystalDynamics(
            in_node_nf=5,
            hidden_nf=32,
            n_layers=2,
            learn_lattice=True
        )
        
        crystal_diff = CrystalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=10,
            learn_lattice=True
        )
        
        nodes_dist = UniformNodesDistribution(min_nodes=5, max_nodes=15)
        
        # Create inputs with bad cell parameters
        batch_size = 4
        n_nodes = 10
        
        x = torch.randn(batch_size, n_nodes, 3) * torch.ones(batch_size, n_nodes, 1)
        h = {
            'categorical': torch.randn(batch_size, n_nodes, 4),
            'integer': torch.randn(batch_size, n_nodes, 1)
        }
        node_mask = torch.ones(batch_size, n_nodes, 1)
        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1)
        context = None
        
        # Bad cell params: very small length, extreme angle
        bad_cell_params = torch.tensor([[0.05, 15., 15., 5., 90., 90.]] * batch_size)
        pbc = torch.ones(batch_size, 3, dtype=torch.bool)
        
        # Compute loss
        nll, reg_term, mean_abs_z = compute_loss_and_nll_crystal(
            args, crystal_diff, nodes_dist,
            x, h, node_mask, edge_mask, context,
            cell_params=bad_cell_params, pbc=pbc
        )
        
        # Should have non-zero regularization
        assert reg_term > 0
        
        print("✓ Crystal loss computation includes regularization")


def run_all_tests():
    """Run all Phase 7 tests."""
    print("\n" + "="*60)
    print("Running Phase 7: Crystal Diffusion Tests")
    print("="*60 + "\n")
    
    # Test CrystalDiffusion
    print("Testing CrystalDiffusion...")
    test_class = TestCrystalDiffusion()
    test_class.test_init()
    test_class.test_init_without_lattice_support()
    test_class.test_sample_cell_noise()
    test_class.test_sample()
    test_class.test_sample_chain()
    print()
    
    # Test CrystalNodesDistribution
    print("Testing CrystalNodesDistribution...")
    test_class = TestCrystalNodesDistribution()
    test_class.test_init_uniform()
    test_class.test_init_with_histogram()
    test_class.test_log_prob()
    test_class.test_log_prob_out_of_range()
    test_class.test_sample()
    test_class.test_sample_batch()
    test_class.test_from_dataset()
    print()
    
    # Test UniformNodesDistribution
    print("Testing UniformNodesDistribution...")
    test_class = TestUniformNodesDistribution()
    test_class.test_init()
    test_class.test_log_prob()
    test_class.test_sample()
    print()
    
    # Test Loss Computation
    print("Testing Crystal Loss Computation...")
    test_class = TestCrystalLossComputation()
    test_class.test_compute_loss_basic()
    test_class.test_compute_loss_with_regularization()
    print()
    
    print("="*60)
    print("✅ All Phase 7 tests passed!")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
