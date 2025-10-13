"""
Unit tests for crystal evaluation metrics

Tests CrystalMetrics class following the "no fallback heuristics" principle.
All tests verify strict validation and correct metric computation.
"""

import pytest
import torch
import numpy as np
from crystal.evaluation import CrystalMetrics


@pytest.fixture
def dataset_info():
    """Create dataset info for testing"""
    return {
        'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
        'name': 'test_dataset',
    }


@pytest.fixture
def valid_crystal():
    """Create a valid crystal structure for testing"""
    n_atoms = 10
    
    # Simple cubic-like structure
    cell = torch.eye(3) * 5.0  # 5 Å cube
    cell_params = torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
    cell_volume = torch.tensor(125.0)  # 5^3
    
    # Random positions within cell
    torch.manual_seed(42)
    positions_cart = torch.rand(n_atoms, 3) * 5.0
    
    pbc = torch.ones(3, dtype=torch.bool)
    
    return {
        'positions_cart': positions_cart,
        'cell': cell,
        'cell_params': cell_params,
        'cell_volume': cell_volume,
        'pbc': pbc,
        'density': torch.tensor(1.2),  # g/cm³
    }


@pytest.fixture
def crystal_list(valid_crystal):
    """Create list of crystals for batch testing"""
    crystals = []
    for i in range(5):
        crystal = valid_crystal.copy()
        # Vary cell parameters slightly
        crystal['cell_params'] = valid_crystal['cell_params'].clone() + torch.randn(6) * 0.1
        crystal['cell_volume'] = valid_crystal['cell_volume'] + torch.randn(1) * 5.0
        crystal['density'] = valid_crystal['density'] + torch.randn(1) * 0.1
        crystals.append(crystal)
    return crystals


class TestCrystalMetricsInitialization:
    """Test CrystalMetrics initialization"""
    
    def test_valid_initialization(self, dataset_info):
        """Test initialization with valid dataset_info"""
        metrics = CrystalMetrics(dataset_info)
        assert metrics.dataset_info == dataset_info
        assert metrics.atom_decoder == ['H', 'C', 'N', 'O', 'F']
    
    def test_initialization_without_atom_decoder(self):
        """Test initialization without atom_decoder"""
        dataset_info = {'name': 'test'}
        metrics = CrystalMetrics(dataset_info)
        assert metrics.atom_decoder == []
    
    def test_initialization_with_none_raises_error(self):
        """Test that None dataset_info raises ValueError"""
        with pytest.raises(ValueError, match="dataset_info cannot be None"):
            CrystalMetrics(None)
    
    def test_initialization_with_invalid_type_raises_error(self):
        """Test that invalid dataset_info type raises ValueError"""
        with pytest.raises(ValueError, match="dataset_info must be a dictionary"):
            CrystalMetrics("invalid")


class TestStructuralMetrics:
    """Test structural metrics computation"""
    
    def test_compute_structural_metrics_basic(self, dataset_info, crystal_list):
        """Test basic structural metrics computation"""
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_structural_metrics(crystal_list)
        
        # Check all expected metrics are present
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for param in param_names:
            assert f'{param}_mean' in metrics
            assert f'{param}_std' in metrics
            assert f'{param}_min' in metrics
            assert f'{param}_max' in metrics
        
        assert 'volume_mean' in metrics
        assert 'volume_std' in metrics
        assert 'density_mean' in metrics
        assert 'density_std' in metrics
    
    def test_structural_metrics_values_are_finite(self, dataset_info, crystal_list):
        """Test that all metric values are finite"""
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_structural_metrics(crystal_list)
        
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"
    
    def test_empty_crystal_list_raises_error(self, dataset_info):
        """Test that empty crystal list raises ValueError"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="crystals cannot be empty"):
            metrics_obj.compute_structural_metrics([])
    
    def test_missing_cell_params_raises_error(self, dataset_info, valid_crystal):
        """Test that missing cell_params raises ValueError"""
        crystal = valid_crystal.copy()
        del crystal['cell_params']
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="missing 'cell_params' field"):
            metrics_obj.compute_structural_metrics([crystal])
    
    def test_missing_cell_volume_raises_error(self, dataset_info, valid_crystal):
        """Test that missing cell_volume raises ValueError"""
        crystal = valid_crystal.copy()
        del crystal['cell_volume']
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="missing 'cell_volume' field"):
            metrics_obj.compute_structural_metrics([crystal])
    
    def test_invalid_cell_params_shape_raises_error(self, dataset_info, valid_crystal):
        """Test that invalid cell_params shape raises ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell_params'] = torch.randn(5)  # Wrong shape
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="cell_params must have shape"):
            metrics_obj.compute_structural_metrics([crystal])
    
    def test_negative_volume_raises_error(self, dataset_info, valid_crystal):
        """Test that negative volume raises ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell_volume'] = torch.tensor(-10.0)
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="Invalid cell volume"):
            metrics_obj.compute_structural_metrics([crystal])
    
    def test_nan_volume_raises_error(self, dataset_info, valid_crystal):
        """Test that NaN volume raises ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell_volume'] = torch.tensor(float('nan'))
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="Invalid cell volume"):
            metrics_obj.compute_structural_metrics([crystal])


class TestValidityMetrics:
    """Test validity metrics computation"""
    
    def test_compute_validity_metrics_basic(self, dataset_info, crystal_list):
        """Test basic validity metrics computation"""
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_validity_metrics(crystal_list)
        
        # Check all expected metrics are present
        assert 'validity_ratio' in metrics
        assert 'min_distance_violation_ratio' in metrics
        assert 'cell_param_violation_ratio' in metrics
        assert 'n_valid' in metrics
        assert 'n_total' in metrics
        
        # Check values are in valid ranges
        assert 0 <= metrics['validity_ratio'] <= 1
        assert 0 <= metrics['min_distance_violation_ratio'] <= 1
        assert 0 <= metrics['cell_param_violation_ratio'] <= 1
        assert metrics['n_total'] == len(crystal_list)
    
    def test_empty_crystal_list_raises_error(self, dataset_info):
        """Test that empty crystal list raises ValueError"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="crystals cannot be empty"):
            metrics_obj.compute_validity_metrics([])
    
    def test_negative_threshold_raises_error(self, dataset_info, crystal_list):
        """Test that negative threshold raises ValueError"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="min_distance_threshold must be positive"):
            metrics_obj.compute_validity_metrics(crystal_list, min_distance_threshold=-1.0)
    
    def test_check_crystal_validity_valid_crystal(self, dataset_info, valid_crystal):
        """Test validity check on valid crystal"""
        metrics_obj = CrystalMetrics(dataset_info)
        is_valid, has_min_dist_viol, has_cell_viol = \
            metrics_obj.check_crystal_validity(valid_crystal)
        
        # With random positions, might or might not be valid
        # But check return types
        assert isinstance(is_valid, bool)
        assert isinstance(has_min_dist_viol, bool)
        assert isinstance(has_cell_viol, bool)
    
    def test_check_crystal_validity_missing_field_raises_error(self, dataset_info):
        """Test that missing required field raises ValueError"""
        crystal = {'positions_cart': torch.randn(5, 3)}
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="missing required field"):
            metrics_obj.check_crystal_validity(crystal)
    
    def test_check_crystal_validity_invalid_cell_lengths(self, dataset_info, valid_crystal):
        """Test detection of invalid cell lengths"""
        crystal = valid_crystal.copy()
        crystal['cell_params'] = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])  # Too small
        
        metrics_obj = CrystalMetrics(dataset_info)
        is_valid, has_min_dist_viol, has_cell_viol = \
            metrics_obj.check_crystal_validity(crystal)
        
        assert has_cell_viol is True
        assert is_valid is False
    
    def test_check_crystal_validity_invalid_cell_angles(self, dataset_info, valid_crystal):
        """Test detection of invalid cell angles"""
        crystal = valid_crystal.copy()
        crystal['cell_params'] = torch.tensor([5.0, 5.0, 5.0, 20.0, 90.0, 90.0])  # Too small
        
        metrics_obj = CrystalMetrics(dataset_info)
        is_valid, has_min_dist_viol, has_cell_viol = \
            metrics_obj.check_crystal_validity(crystal)
        
        assert has_cell_viol is True
        assert is_valid is False
    
    def test_check_crystal_validity_single_atom_no_distance_check(self, dataset_info):
        """Test that single atom structure doesn't check distances"""
        crystal = {
            'positions_cart': torch.tensor([[2.5, 2.5, 2.5]]),
            'cell': torch.eye(3) * 5.0,
            'cell_params': torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
            'pbc': torch.ones(3, dtype=torch.bool),
        }
        
        metrics_obj = CrystalMetrics(dataset_info)
        is_valid, has_min_dist_viol, has_cell_viol = \
            metrics_obj.check_crystal_validity(crystal)
        
        # Should be valid (no distance violations possible with 1 atom)
        assert has_min_dist_viol is False


class TestDistributionMetrics:
    """Test distribution comparison metrics"""
    
    def test_compute_distribution_metrics_basic(self, dataset_info, crystal_list):
        """Test basic distribution metrics computation"""
        # Split into generated and reference
        gen_crystals = crystal_list[:3]
        ref_crystals = crystal_list[3:]
        
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_distribution_metrics(gen_crystals, ref_crystals)
        
        # Check Wasserstein distances are computed
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for param in param_names:
            assert f'{param}_wasserstein' in metrics
            assert metrics[f'{param}_wasserstein'] >= 0  # Wasserstein distance is non-negative
        
        assert 'volume_wasserstein' in metrics
        assert 'density_wasserstein' in metrics
    
    def test_empty_generated_raises_error(self, dataset_info, crystal_list):
        """Test that empty generated list raises ValueError"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="generated_crystals cannot be empty"):
            metrics_obj.compute_distribution_metrics([], crystal_list)
    
    def test_empty_reference_raises_error(self, dataset_info, crystal_list):
        """Test that empty reference list raises ValueError"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="reference_crystals cannot be empty"):
            metrics_obj.compute_distribution_metrics(crystal_list, [])
    
    def test_identical_distributions_have_zero_distance(self, dataset_info, crystal_list):
        """Test that identical distributions have zero Wasserstein distance"""
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_distribution_metrics(crystal_list, crystal_list)
        
        # All Wasserstein distances should be zero (or very small)
        for key, value in metrics.items():
            if 'wasserstein' in key:
                assert value < 1e-6, f"{key} should be ~0 for identical distributions"


class TestComputeAllMetrics:
    """Test compute_all_metrics integration"""
    
    def test_compute_all_metrics_without_reference(self, dataset_info, crystal_list):
        """Test computing all metrics without reference crystals"""
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_all_metrics(crystal_list)
        
        # Should have structural and validity metrics
        assert 'a_mean' in metrics
        assert 'validity_ratio' in metrics
        
        # Should NOT have distribution metrics
        assert 'a_wasserstein' not in metrics
    
    def test_compute_all_metrics_with_reference(self, dataset_info, crystal_list):
        """Test computing all metrics with reference crystals"""
        gen_crystals = crystal_list[:3]
        ref_crystals = crystal_list[3:]
        
        metrics_obj = CrystalMetrics(dataset_info)
        metrics = metrics_obj.compute_all_metrics(gen_crystals, ref_crystals)
        
        # Should have all three types of metrics
        assert 'a_mean' in metrics  # Structural
        assert 'validity_ratio' in metrics  # Validity
        assert 'a_wasserstein' in metrics  # Distribution
    
    def test_compute_all_metrics_empty_list_raises_error(self, dataset_info):
        """Test that empty crystal list raises ValueError"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="generated_crystals cannot be empty"):
            metrics_obj.compute_all_metrics([])
    
    def test_validate_crystal_list_invalid_crystal_raises_error(self, dataset_info):
        """Test that invalid crystal in list raises ValueError"""
        crystals = [{'positions_cart': torch.randn(5, 3)}]  # Missing required fields
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="missing required field"):
            metrics_obj.compute_all_metrics(crystals)


class TestValidation:
    """Test internal validation methods"""
    
    def test_validate_crystal_list_valid(self, dataset_info, crystal_list):
        """Test validation of valid crystal list"""
        metrics_obj = CrystalMetrics(dataset_info)
        
        # Should not raise
        metrics_obj._validate_crystal_list(crystal_list)
    
    def test_validate_crystal_list_non_dict_raises_error(self, dataset_info):
        """Test that non-dict crystal raises ValueError"""
        crystals = ["not a dict"]
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="is not a dictionary"):
            metrics_obj._validate_crystal_list(crystals)
    
    def test_validate_crystal_list_missing_field_raises_error(self, dataset_info):
        """Test that missing required field raises ValueError"""
        crystals = [{'positions_cart': torch.randn(5, 3)}]
        
        metrics_obj = CrystalMetrics(dataset_info)
        
        with pytest.raises(ValueError, match="missing required field"):
            metrics_obj._validate_crystal_list(crystals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
