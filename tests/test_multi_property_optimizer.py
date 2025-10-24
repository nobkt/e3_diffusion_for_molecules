"""
Tests for Multi-Property Optimization Module

Tests all components of the multi-property optimization system without
using any heuristic processing or fallback mechanisms.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crystal.evaluation.multi_property_optimizer import (
    PropertyConstraint,
    ConstrainedCrystalGenerator,
    ParetoFrontierSearcher,
)


class MockPropertyPredictor(nn.Module):
    """Mock property predictor for testing."""
    
    def __init__(self, property_names: List[str]):
        super().__init__()
        self.property_names = property_names
    
    def forward(self, positions, cell, atomic_numbers):
        # Return dummy predictions
        batch_size = positions.shape[0] if len(positions.shape) > 2 else 1
        predictions = {}
        for prop in self.property_names:
            predictions[prop] = torch.randn(batch_size, 1)
        return predictions


class TestPropertyConstraint:
    """Test PropertyConstraint class."""
    
    def test_init_valid_operators(self):
        """Test initialization with valid operators."""
        operators = ['>', '<', '>=', '<=', '==']
        for op in operators:
            constraint = PropertyConstraint('bandgap', op, 2.0)
            assert constraint.operator == op
            assert constraint.property_name == 'bandgap'
            assert constraint.value == 2.0
    
    def test_init_range_constraint(self):
        """Test initialization with range constraint."""
        constraint = PropertyConstraint('melting_point', 'in', (150.0, 200.0))
        assert constraint.operator == 'in'
        assert constraint.value == (150.0, 200.0)
    
    def test_init_invalid_operator(self):
        """Test initialization with invalid operator."""
        with pytest.raises(ValueError, match="Invalid operator"):
            PropertyConstraint('bandgap', 'invalid', 2.0)
    
    def test_init_invalid_range_value(self):
        """Test initialization with invalid range value."""
        with pytest.raises(ValueError, match="requires tuple"):
            PropertyConstraint('bandgap', 'in', 2.0)
        
        with pytest.raises(ValueError, match="Invalid range"):
            PropertyConstraint('bandgap', 'in', (200.0, 150.0))
    
    def test_check_greater_than(self):
        """Test check with > operator."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        assert constraint.check(2.5) is True
        assert constraint.check(2.0) is False
        assert constraint.check(1.5) is False
    
    def test_check_less_than(self):
        """Test check with < operator."""
        constraint = PropertyConstraint('bandgap', '<', 2.0)
        assert constraint.check(1.5) is True
        assert constraint.check(2.0) is False
        assert constraint.check(2.5) is False
    
    def test_check_greater_equal(self):
        """Test check with >= operator."""
        constraint = PropertyConstraint('bandgap', '>=', 2.0)
        assert constraint.check(2.5) is True
        assert constraint.check(2.0) is True
        assert constraint.check(1.5) is False
    
    def test_check_less_equal(self):
        """Test check with <= operator."""
        constraint = PropertyConstraint('bandgap', '<=', 2.0)
        assert constraint.check(1.5) is True
        assert constraint.check(2.0) is True
        assert constraint.check(2.5) is False
    
    def test_check_equal(self):
        """Test check with == operator."""
        constraint = PropertyConstraint('bandgap', '==', 2.0)
        assert constraint.check(2.0) is True
        assert constraint.check(2.0 + 1e-7) is True  # Within tolerance
        assert constraint.check(2.1) is False
    
    def test_check_in_range(self):
        """Test check with 'in' operator."""
        constraint = PropertyConstraint('melting_point', 'in', (150.0, 200.0))
        assert constraint.check(175.0) is True
        assert constraint.check(150.0) is True
        assert constraint.check(200.0) is True
        assert constraint.check(149.9) is False
        assert constraint.check(200.1) is False
    
    def test_check_with_tensor(self):
        """Test check with torch tensor input."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        value = torch.tensor([2.5])
        assert constraint.check(value) is True
    
    def test_check_with_numpy(self):
        """Test check with numpy input."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        value = np.array(2.5)
        assert constraint.check(value) is True
    
    def test_check_invalid_value(self):
        """Test check with invalid value."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        with pytest.raises(ValueError, match="must be numeric"):
            constraint.check("invalid")
    
    def test_check_nan_value(self):
        """Test check with NaN value."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        with pytest.raises(ValueError, match="must be finite"):
            constraint.check(float('nan'))
    
    def test_sample_target_value_range(self):
        """Test sampling from range constraint."""
        constraint = PropertyConstraint('melting_point', 'in', (150.0, 200.0))
        rng = np.random.default_rng(42)
        
        # Sample multiple times and check all are in range
        for _ in range(10):
            value = constraint.sample_target_value(rng)
            assert 150.0 <= value <= 200.0
    
    def test_sample_target_value_greater_than(self):
        """Test sampling from > constraint."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            value = constraint.sample_target_value(rng)
            assert value > 2.0
            assert value <= 4.0  # Should be in [2.0 * 1.01, 2.0 * 2.0]
    
    def test_sample_target_value_less_than(self):
        """Test sampling from < constraint."""
        constraint = PropertyConstraint('bandgap', '<', 2.0)
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            value = constraint.sample_target_value(rng)
            assert value < 2.0
    
    def test_sample_target_value_equal(self):
        """Test sampling from == constraint."""
        constraint = PropertyConstraint('bandgap', '==', 2.0)
        value = constraint.sample_target_value()
        assert value == 2.0
    
    def test_sample_target_value_invalid_constraint(self):
        """Test sampling from invalid constraint."""
        constraint = PropertyConstraint('bandgap', '>', -1.0)
        with pytest.raises(ValueError, match="Cannot sample"):
            constraint.sample_target_value()
    
    def test_repr(self):
        """Test string representation."""
        constraint = PropertyConstraint('bandgap', '>', 2.0)
        repr_str = repr(constraint)
        assert 'bandgap' in repr_str
        assert '>' in repr_str


class TestConstrainedCrystalGenerator:
    """Test ConstrainedCrystalGenerator class."""
    
    def test_init_valid(self):
        """Test initialization with valid inputs."""
        predictor = MockPropertyPredictor(['bandgap', 'melting_point'])
        constraints = [
            PropertyConstraint('bandgap', '>', 2.0),
            PropertyConstraint('melting_point', '<', 200.0),
        ]
        
        generator = ConstrainedCrystalGenerator(predictor, constraints)
        assert generator.property_predictor is predictor
        assert generator.constraints == constraints
    
    def test_init_empty_constraints(self):
        """Test initialization with empty constraints."""
        predictor = MockPropertyPredictor(['bandgap'])
        
        with pytest.raises(ValueError, match="must be non-empty"):
            ConstrainedCrystalGenerator(predictor, [])
    
    def test_init_none_predictor(self):
        """Test initialization with None predictor."""
        constraints = [PropertyConstraint('bandgap', '>', 2.0)]
        
        with pytest.raises(ValueError, match="cannot be None"):
            ConstrainedCrystalGenerator(None, constraints)
    
    def test_init_missing_property(self):
        """Test initialization with constraint on missing property."""
        predictor = MockPropertyPredictor(['bandgap'])
        constraints = [PropertyConstraint('missing_prop', '>', 2.0)]
        
        with pytest.raises(ValueError, match="does not support properties"):
            ConstrainedCrystalGenerator(predictor, constraints)
    
    def test_check_constraints_satisfied(self):
        """Test constraint checking with satisfied constraints."""
        predictor = MockPropertyPredictor(['bandgap', 'melting_point'])
        constraints = [
            PropertyConstraint('bandgap', '>', 2.0),
            PropertyConstraint('melting_point', '<', 200.0),
        ]
        
        generator = ConstrainedCrystalGenerator(predictor, constraints)
        
        predictions = {
            'bandgap': torch.tensor([2.5]),
            'melting_point': torch.tensor([175.0]),
        }
        
        assert generator.check_constraints(predictions) is True
    
    def test_check_constraints_unsatisfied(self):
        """Test constraint checking with unsatisfied constraints."""
        predictor = MockPropertyPredictor(['bandgap', 'melting_point'])
        constraints = [
            PropertyConstraint('bandgap', '>', 2.0),
            PropertyConstraint('melting_point', '<', 200.0),
        ]
        
        generator = ConstrainedCrystalGenerator(predictor, constraints)
        
        predictions = {
            'bandgap': torch.tensor([1.5]),  # Fails constraint
            'melting_point': torch.tensor([175.0]),
        }
        
        assert generator.check_constraints(predictions) is False
    
    def test_check_constraints_missing_property(self):
        """Test constraint checking with missing property."""
        predictor = MockPropertyPredictor(['bandgap', 'melting_point'])
        constraints = [PropertyConstraint('bandgap', '>', 2.0)]
        
        generator = ConstrainedCrystalGenerator(predictor, constraints)
        
        predictions = {
            'melting_point': torch.tensor([175.0]),
        }
        
        with pytest.raises(ValueError, match="Missing property"):
            generator.check_constraints(predictions)
    
    def test_sample_target_properties(self):
        """Test sampling target properties."""
        predictor = MockPropertyPredictor(['bandgap', 'melting_point'])
        constraints = [
            PropertyConstraint('bandgap', 'in', (2.0, 3.0)),
            PropertyConstraint('melting_point', 'in', (150.0, 200.0)),
        ]
        
        generator = ConstrainedCrystalGenerator(predictor, constraints)
        rng = np.random.default_rng(42)
        
        targets = generator.sample_target_properties(rng)
        
        assert 'bandgap' in targets
        assert 'melting_point' in targets
        assert 2.0 <= targets['bandgap'] <= 3.0
        assert 150.0 <= targets['melting_point'] <= 200.0
    
    def test_repr(self):
        """Test string representation."""
        predictor = MockPropertyPredictor(['bandgap'])
        constraints = [PropertyConstraint('bandgap', '>', 2.0)]
        generator = ConstrainedCrystalGenerator(predictor, constraints)
        
        repr_str = repr(generator)
        assert 'ConstrainedCrystalGenerator' in repr_str


class TestParetoFrontierSearcher:
    """Test ParetoFrontierSearcher class."""
    
    def test_init_valid(self):
        """Test initialization with valid inputs."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        objectives = ['bandgap', 'formation_energy']
        minimize = [False, True]
        
        searcher = ParetoFrontierSearcher(predictor, objectives, minimize)
        assert searcher.objectives == objectives
        assert searcher.minimize == minimize
    
    def test_init_default_minimize(self):
        """Test initialization with default minimize."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        objectives = ['bandgap', 'formation_energy']
        
        searcher = ParetoFrontierSearcher(predictor, objectives)
        assert searcher.minimize == [True, True]
    
    def test_init_empty_objectives(self):
        """Test initialization with empty objectives."""
        predictor = MockPropertyPredictor(['bandgap'])
        
        with pytest.raises(ValueError, match="must be non-empty"):
            ParetoFrontierSearcher(predictor, [])
    
    def test_init_duplicate_objectives(self):
        """Test initialization with duplicate objectives."""
        predictor = MockPropertyPredictor(['bandgap'])
        
        with pytest.raises(ValueError, match="contains duplicates"):
            ParetoFrontierSearcher(predictor, ['bandgap', 'bandgap'])
    
    def test_init_none_predictor(self):
        """Test initialization with None predictor."""
        with pytest.raises(ValueError, match="cannot be None"):
            ParetoFrontierSearcher(None, ['bandgap'])
    
    def test_init_missing_objective(self):
        """Test initialization with missing objective."""
        predictor = MockPropertyPredictor(['bandgap'])
        
        with pytest.raises(ValueError, match="does not support objectives"):
            ParetoFrontierSearcher(predictor, ['missing_prop'])
    
    def test_init_mismatched_minimize_length(self):
        """Test initialization with mismatched minimize length."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        
        with pytest.raises(ValueError, match="same length"):
            ParetoFrontierSearcher(
                predictor,
                ['bandgap', 'formation_energy'],
                minimize=[True]  # Wrong length
            )
    
    def test_find_pareto_frontier_simple(self):
        """Test finding Pareto frontier with simple case."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        searcher = ParetoFrontierSearcher(
            predictor,
            ['bandgap', 'formation_energy'],
            minimize=[False, True]  # Maximize bandgap, minimize energy
        )
        
        # Create test data
        # Point 0: (bandgap=3.0, energy=1.0) - Pareto optimal
        # Point 1: (bandgap=2.0, energy=2.0) - Dominated by 0
        # Point 2: (bandgap=2.5, energy=0.5) - Pareto optimal
        # Point 3: (bandgap=1.0, energy=0.8) - Dominated by 2
        
        predictions_list = [
            {'bandgap': torch.tensor([3.0]), 'formation_energy': torch.tensor([1.0])},
            {'bandgap': torch.tensor([2.0]), 'formation_energy': torch.tensor([2.0])},
            {'bandgap': torch.tensor([2.5]), 'formation_energy': torch.tensor([0.5])},
            {'bandgap': torch.tensor([1.0]), 'formation_energy': torch.tensor([0.8])},
        ]
        
        pareto_indices = searcher.find_pareto_frontier(predictions_list)
        
        # Should find points 0 and 2 as Pareto optimal
        assert set(pareto_indices) == {0, 2}
    
    def test_find_pareto_frontier_all_optimal(self):
        """Test finding Pareto frontier when all points are optimal."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        searcher = ParetoFrontierSearcher(
            predictor,
            ['bandgap', 'formation_energy'],
            minimize=[True, True]
        )
        
        # All points are non-dominated
        predictions_list = [
            {'bandgap': torch.tensor([1.0]), 'formation_energy': torch.tensor([2.0])},
            {'bandgap': torch.tensor([2.0]), 'formation_energy': torch.tensor([1.0])},
            {'bandgap': torch.tensor([1.5]), 'formation_energy': torch.tensor([1.5])},
        ]
        
        pareto_indices = searcher.find_pareto_frontier(predictions_list)
        
        # All three points should be on the frontier
        assert set(pareto_indices) == {0, 1, 2}
    
    def test_find_pareto_frontier_single_optimal(self):
        """Test finding Pareto frontier with single optimal point."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        searcher = ParetoFrontierSearcher(
            predictor,
            ['bandgap', 'formation_energy'],
            minimize=[True, True]
        )
        
        # One point dominates all others
        predictions_list = [
            {'bandgap': torch.tensor([1.0]), 'formation_energy': torch.tensor([1.0])},
            {'bandgap': torch.tensor([2.0]), 'formation_energy': torch.tensor([2.0])},
            {'bandgap': torch.tensor([3.0]), 'formation_energy': torch.tensor([3.0])},
        ]
        
        pareto_indices = searcher.find_pareto_frontier(predictions_list)
        
        # Only point 0 is optimal
        assert pareto_indices == [0]
    
    def test_find_pareto_frontier_empty_list(self):
        """Test finding Pareto frontier with empty list."""
        predictor = MockPropertyPredictor(['bandgap'])
        searcher = ParetoFrontierSearcher(predictor, ['bandgap'])
        
        with pytest.raises(ValueError, match="must be non-empty"):
            searcher.find_pareto_frontier([])
    
    def test_find_pareto_frontier_missing_objective(self):
        """Test finding Pareto frontier with missing objective."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        searcher = ParetoFrontierSearcher(
            predictor,
            ['bandgap', 'formation_energy']
        )
        
        predictions_list = [
            {'bandgap': torch.tensor([1.0])},  # Missing formation_energy
        ]
        
        with pytest.raises(ValueError, match="Missing objective"):
            searcher.find_pareto_frontier(predictions_list)
    
    def test_repr(self):
        """Test string representation."""
        predictor = MockPropertyPredictor(['bandgap', 'formation_energy'])
        searcher = ParetoFrontierSearcher(
            predictor,
            ['bandgap', 'formation_energy'],
            minimize=[False, True]
        )
        
        repr_str = repr(searcher)
        assert 'ParetoFrontierSearcher' in repr_str
        assert 'bandgap' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
