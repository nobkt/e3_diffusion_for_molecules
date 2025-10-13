"""
Unit tests for structure validator

Tests StructureValidator class following the "no fallback heuristics" principle.
All tests verify strict validation and proper error handling.
"""

import pytest
import torch
import numpy as np
from crystal.evaluation import StructureValidator


@pytest.fixture
def valid_crystal():
    """Create a valid crystal structure for testing"""
    n_atoms = 8
    
    # Simple cubic structure
    cell = torch.eye(3) * 5.0
    cell_params = torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
    cell_volume = torch.tensor(125.0)
    
    # Positions with reasonable spacing
    positions_cart = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [2.5, 2.5, 0.0],
        [0.0, 0.0, 2.5],
        [2.5, 0.0, 2.5],
        [0.0, 2.5, 2.5],
        [2.5, 2.5, 2.5],
    ])
    
    # Corresponding fractional coordinates
    positions_frac = positions_cart / 5.0
    
    pbc = torch.ones(3, dtype=torch.bool)
    
    return {
        'positions_cart': positions_cart,
        'positions_frac': positions_frac,
        'cell': cell,
        'cell_params': cell_params,
        'cell_volume': cell_volume,
        'pbc': pbc,
    }


class TestStructureValidatorInitialization:
    """Test StructureValidator initialization"""
    
    def test_valid_initialization_default(self):
        """Test initialization with default parameters"""
        validator = StructureValidator()
        assert validator.min_distance == StructureValidator.MIN_INTERATOMIC_DISTANCE
        assert validator.length_min == StructureValidator.LENGTH_MIN
        assert validator.length_max == StructureValidator.LENGTH_MAX
        assert validator.angle_min == StructureValidator.ANGLE_MIN
        assert validator.angle_max == StructureValidator.ANGLE_MAX
        assert validator.strict_mode is True
    
    def test_valid_initialization_custom(self):
        """Test initialization with custom parameters"""
        validator = StructureValidator(
            min_distance=1.0,
            length_bounds=(2.0, 50.0),
            angle_bounds=(45.0, 135.0),
            strict_mode=False
        )
        assert validator.min_distance == 1.0
        assert validator.length_min == 2.0
        assert validator.length_max == 50.0
        assert validator.angle_min == 45.0
        assert validator.angle_max == 135.0
        assert validator.strict_mode is False
    
    def test_negative_min_distance_raises_error(self):
        """Test that negative min_distance raises ValueError"""
        with pytest.raises(ValueError, match="min_distance must be positive"):
            StructureValidator(min_distance=-1.0)
    
    def test_invalid_length_bounds_raises_error(self):
        """Test that invalid length bounds raise ValueError"""
        with pytest.raises(ValueError, match="Invalid length bounds"):
            StructureValidator(length_bounds=(10.0, 5.0))  # max < min
        
        with pytest.raises(ValueError, match="Invalid length bounds"):
            StructureValidator(length_bounds=(-1.0, 10.0))  # negative
    
    def test_invalid_angle_bounds_raises_error(self):
        """Test that invalid angle bounds raise ValueError"""
        with pytest.raises(ValueError, match="Invalid angle bounds"):
            StructureValidator(angle_bounds=(90.0, 45.0))  # max < min


class TestValidateDataFormat:
    """Test _validate_data_format method"""
    
    def test_valid_data_format(self, valid_crystal):
        """Test validation of valid crystal data format"""
        validator = StructureValidator()
        # Should not raise
        validator._validate_data_format(valid_crystal)
    
    def test_missing_field_raises_error(self, valid_crystal):
        """Test that missing required field raises ValueError"""
        crystal = valid_crystal.copy()
        del crystal['cell']
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="Missing required field: cell"):
            validator._validate_data_format(crystal)
    
    def test_wrong_type_raises_error(self, valid_crystal):
        """Test that wrong field type raises ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # List instead of tensor
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="has wrong type"):
            validator._validate_data_format(crystal)
    
    def test_wrong_shape_raises_error(self, valid_crystal):
        """Test that wrong field shape raises ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell_params'] = torch.randn(5)  # Wrong shape
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="has wrong shape"):
            validator._validate_data_format(crystal)
    
    def test_nan_values_raise_error(self, valid_crystal):
        """Test that NaN values raise ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell'][0, 0] = float('nan')
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            validator._validate_data_format(crystal)
    
    def test_inf_values_raise_error(self, valid_crystal):
        """Test that Inf values raise ValueError"""
        crystal = valid_crystal.copy()
        crystal['cell_volume'] = torch.tensor(float('inf'))
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            validator._validate_data_format(crystal)
    
    def test_empty_positions_raise_error(self, valid_crystal):
        """Test that empty positions raise ValueError"""
        crystal = valid_crystal.copy()
        crystal['positions_cart'] = torch.empty(0, 3)
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="must contain at least one atom"):
            validator._validate_data_format(crystal)
    
    def test_wrong_position_dimensions_raise_error(self, valid_crystal):
        """Test that wrong position dimensions raise ValueError"""
        crystal = valid_crystal.copy()
        crystal['positions_cart'] = torch.randn(5, 2)  # Wrong last dimension
        
        validator = StructureValidator()
        
        with pytest.raises(ValueError, match="must have shape"):
            validator._validate_data_format(crystal)


class TestValidateCellParameters:
    """Test _validate_cell_parameters method"""
    
    def test_valid_cell_parameters(self):
        """Test validation of valid cell parameters"""
        validator = StructureValidator()
        cell_params = torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        # Should not raise
        validator._validate_cell_parameters(cell_params)
    
    def test_length_too_small_raises_error(self):
        """Test that too small length raises ValueError"""
        validator = StructureValidator()
        cell_params = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        with pytest.raises(ValueError, match="out of bounds"):
            validator._validate_cell_parameters(cell_params)
    
    def test_length_too_large_raises_error(self):
        """Test that too large length raises ValueError"""
        validator = StructureValidator()
        cell_params = torch.tensor([150.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        with pytest.raises(ValueError, match="out of bounds"):
            validator._validate_cell_parameters(cell_params)
    
    def test_angle_too_small_raises_error(self):
        """Test that too small angle raises ValueError"""
        validator = StructureValidator()
        cell_params = torch.tensor([5.0, 5.0, 5.0, 20.0, 90.0, 90.0])
        
        with pytest.raises(ValueError, match="out of bounds"):
            validator._validate_cell_parameters(cell_params)
    
    def test_angle_too_large_raises_error(self):
        """Test that too large angle raises ValueError"""
        validator = StructureValidator()
        cell_params = torch.tensor([5.0, 5.0, 5.0, 90.0, 160.0, 90.0])
        
        with pytest.raises(ValueError, match="out of bounds"):
            validator._validate_cell_parameters(cell_params)


class TestValidateCellVolume:
    """Test _validate_cell_volume method"""
    
    def test_valid_cell_volume(self):
        """Test validation of valid cell volume"""
        validator = StructureValidator()
        cell = torch.eye(3) * 5.0
        cell_volume = torch.tensor(125.0)
        
        # Should not raise
        validator._validate_cell_volume(cell, cell_volume)
    
    def test_negative_volume_raises_error(self):
        """Test that negative volume raises ValueError"""
        validator = StructureValidator()
        cell = torch.eye(3) * 5.0
        cell_volume = torch.tensor(-10.0)
        
        with pytest.raises(ValueError, match="too small"):
            validator._validate_cell_volume(cell, cell_volume)
    
    def test_inconsistent_volume_raises_error(self):
        """Test that inconsistent volume raises ValueError"""
        validator = StructureValidator()
        cell = torch.eye(3) * 5.0
        cell_volume = torch.tensor(200.0)  # Wrong value
        
        with pytest.raises(ValueError, match="inconsistent"):
            validator._validate_cell_volume(cell, cell_volume)


class TestValidateCoordinateConsistency:
    """Test _validate_coordinate_consistency method"""
    
    def test_consistent_coordinates(self, valid_crystal):
        """Test validation of consistent coordinates"""
        validator = StructureValidator()
        
        # Should not raise
        validator._validate_coordinate_consistency(
            valid_crystal['positions_cart'],
            valid_crystal['positions_frac'],
            valid_crystal['cell']
        )
    
    def test_inconsistent_coordinates_raise_error(self):
        """Test that inconsistent coordinates raise ValueError"""
        validator = StructureValidator()
        cell = torch.eye(3) * 5.0
        positions_cart = torch.randn(5, 3)
        positions_frac = torch.randn(5, 3)  # Inconsistent
        
        with pytest.raises(ValueError, match="inconsistent"):
            validator._validate_coordinate_consistency(
                positions_cart, positions_frac, cell
            )
    
    def test_missing_coordinates_no_check(self):
        """Test that missing coordinates skip consistency check"""
        validator = StructureValidator()
        cell = torch.eye(3) * 5.0
        
        # Should not raise when one type is None
        validator._validate_coordinate_consistency(
            None, torch.randn(5, 3), cell
        )
        validator._validate_coordinate_consistency(
            torch.randn(5, 3), None, cell
        )
    
    def test_extreme_fractional_coordinates_raise_error(self):
        """Test that extreme fractional coordinates raise ValueError"""
        validator = StructureValidator()
        cell = torch.eye(3) * 5.0
        positions_frac = torch.tensor([[3.0, 0.0, 0.0]])  # Out of reasonable range
        
        from crystal.data.periodic_utils import fractional_to_cartesian
        positions_cart = fractional_to_cartesian(
            positions_frac.unsqueeze(0), cell.unsqueeze(0)
        ).squeeze(0)
        
        with pytest.raises(ValueError, match="out of reasonable range"):
            validator._validate_coordinate_consistency(
                positions_cart, positions_frac, cell
            )


class TestValidateMinimumDistances:
    """Test _validate_minimum_distances method"""
    
    def test_valid_distances(self, valid_crystal):
        """Test validation of valid distances"""
        validator = StructureValidator(min_distance=1.0)
        
        # Should not raise (positions are well-spaced)
        validator._validate_minimum_distances(
            valid_crystal['positions_cart'],
            valid_crystal['cell'],
            valid_crystal['pbc']
        )
    
    def test_too_close_atoms_raise_error(self):
        """Test that too close atoms raise ValueError"""
        validator = StructureValidator(min_distance=1.0)
        
        # Two atoms very close
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],  # Only 0.1 Å apart
        ])
        cell = torch.eye(3) * 5.0
        pbc = torch.ones(3, dtype=torch.bool)
        
        with pytest.raises(ValueError, match="below threshold"):
            validator._validate_minimum_distances(positions, cell, pbc)
    
    def test_single_atom_no_check(self):
        """Test that single atom structure passes (no pairs to check)"""
        validator = StructureValidator()
        
        positions = torch.tensor([[2.5, 2.5, 2.5]])
        cell = torch.eye(3) * 5.0
        pbc = torch.ones(3, dtype=torch.bool)
        
        # Should not raise
        validator._validate_minimum_distances(positions, cell, pbc)


class TestValidateStructure:
    """Test validate_structure integration method"""
    
    def test_valid_structure(self, valid_crystal):
        """Test validation of valid structure"""
        validator = StructureValidator()
        is_valid, errors = validator.validate_structure(valid_crystal)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_invalid_structure_strict_mode_raises(self, valid_crystal):
        """Test that invalid structure raises in strict mode"""
        crystal = valid_crystal.copy()
        crystal['cell_params'] = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])  # Invalid
        
        validator = StructureValidator(strict_mode=True)
        
        with pytest.raises(ValueError):
            validator.validate_structure(crystal)
    
    def test_invalid_structure_non_strict_mode_returns_errors(self, valid_crystal):
        """Test that invalid structure returns errors in non-strict mode"""
        crystal = valid_crystal.copy()
        crystal['cell_params'] = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])  # Invalid
        
        validator = StructureValidator(strict_mode=False)
        is_valid, errors = validator.validate_structure(crystal)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any('Cell parameters' in err for err in errors)
    
    def test_skip_distance_check(self, valid_crystal):
        """Test skipping distance check"""
        crystal = valid_crystal.copy()
        # Add very close atoms
        crystal['positions_cart'] = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ])
        
        validator = StructureValidator(strict_mode=False)
        
        # Should pass without distance check
        is_valid, errors = validator.validate_structure(
            crystal, check_distances=False
        )
        # Other checks might fail, but not distance check
    
    def test_skip_coordinate_check(self, valid_crystal):
        """Test skipping coordinate consistency check"""
        crystal = valid_crystal.copy()
        # Make coordinates inconsistent
        crystal['positions_frac'] = torch.randn(8, 3)
        
        validator = StructureValidator(strict_mode=False)
        
        # Should pass without coordinate check
        is_valid, errors = validator.validate_structure(
            crystal, check_coordinates=False
        )
        # Might fail other checks, but not coordinate check


class TestValidateBatch:
    """Test validate_batch method"""
    
    def test_valid_batch(self, valid_crystal):
        """Test validation of valid batch"""
        crystals = [valid_crystal.copy() for _ in range(3)]
        
        validator = StructureValidator()
        all_valid, details = validator.validate_batch(crystals)
        
        assert all_valid is True
        assert details is None
    
    def test_batch_with_invalid_structure(self, valid_crystal):
        """Test batch with one invalid structure"""
        crystals = [valid_crystal.copy() for _ in range(3)]
        crystals[1]['cell_params'] = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])  # Invalid
        
        validator = StructureValidator(strict_mode=False)
        all_valid, details = validator.validate_batch(crystals)
        
        assert all_valid is False
    
    def test_batch_with_details(self, valid_crystal):
        """Test batch validation with detailed errors"""
        crystals = [valid_crystal.copy() for _ in range(3)]
        crystals[1]['cell_params'] = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])  # Invalid
        
        validator = StructureValidator(strict_mode=False)
        all_valid, details = validator.validate_batch(crystals, return_details=True)
        
        assert all_valid is False
        assert details is not None
        assert len(details) > 0
        assert details[0][0] == 1  # Index of invalid crystal
        assert len(details[0][1]) > 0  # Has error messages
    
    def test_batch_strict_mode_raises_on_first_invalid(self, valid_crystal):
        """Test that strict mode raises on first invalid structure"""
        crystals = [valid_crystal.copy() for _ in range(3)]
        crystals[0]['cell_params'] = torch.tensor([0.5, 5.0, 5.0, 90.0, 90.0, 90.0])  # Invalid
        
        validator = StructureValidator(strict_mode=True)
        
        with pytest.raises(ValueError):
            validator.validate_batch(crystals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
