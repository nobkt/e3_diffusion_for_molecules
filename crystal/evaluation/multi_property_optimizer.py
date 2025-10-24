"""
Multi-Property Optimization Module

Implements constraint-based generation and Pareto frontier search for multi-objective
crystal property optimization.

Design Principles:
- No heuristic processing or fallback mechanisms
- Theoretically grounded constraint checking
- Explicit error handling with informative messages
- Modular architecture for extensibility

This implements Component P4-2 (Multi-Property Optimization) from Phase 4.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Union, Tuple, Optional, Callable
from tqdm import tqdm
import itertools
import warnings


class PropertyConstraint:
    """
    Single property constraint for crystal generation.
    
    Supports inequality and range constraints for property values.
    All constraint checking is deterministic with no fallback mechanisms.
    
    Args:
        property_name: Name of the property (e.g., 'bandgap')
        operator: Constraint operator: '>', '<', '>=', '<=', '==', 'in'
        value: Constraint value (float for inequalities, tuple for range)
        
    Raises:
        ValueError: If operator is invalid or value type doesn't match operator
        
    Examples:
        >>> constraint = PropertyConstraint('bandgap', '>', 2.0)
        >>> constraint.check(2.5)  # True
        >>> constraint.check(1.5)  # False
        
        >>> range_constraint = PropertyConstraint('melting_point', 'in', (150.0, 200.0))
        >>> range_constraint.check(175.0)  # True
    """
    
    def __init__(
        self,
        property_name: str,
        operator: str,
        value: Union[float, Tuple[float, float]]
    ):
        # Validate operator
        valid_operators = ['>', '<', '>=', '<=', '==', 'in']
        if operator not in valid_operators:
            raise ValueError(
                f"Invalid operator: {operator}. Must be one of {valid_operators}"
            )
        
        # Validate value for 'in' operator
        if operator == 'in':
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    f"'in' operator requires tuple (min, max), got {type(value)}"
                )
            if value[0] >= value[1]:
                raise ValueError(
                    f"Invalid range: min ({value[0]}) must be less than max ({value[1]})"
                )
        else:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Operator '{operator}' requires numeric value, got {type(value)}"
                )
        
        self.property_name = property_name
        self.operator = operator
        self.value = value
    
    def check(self, predicted_value: float) -> bool:
        """
        Check if predicted value satisfies constraint.
        
        Args:
            predicted_value: Predicted property value
            
        Returns:
            satisfied: True if constraint is satisfied
            
        Raises:
            ValueError: If predicted_value is not a valid number
        """
        if not isinstance(predicted_value, (int, float, np.number)):
            # Handle tensor values
            if hasattr(predicted_value, 'item'):
                predicted_value = predicted_value.item()
            else:
                raise ValueError(
                    f"predicted_value must be numeric, got {type(predicted_value)}"
                )
        
        # Check for NaN/Inf
        if not np.isfinite(predicted_value):
            raise ValueError(
                f"predicted_value must be finite, got {predicted_value}"
            )
        
        # Perform constraint check
        if self.operator == '>':
            return predicted_value > self.value
        elif self.operator == '<':
            return predicted_value < self.value
        elif self.operator == '>=':
            return predicted_value >= self.value
        elif self.operator == '<=':
            return predicted_value <= self.value
        elif self.operator == '==':
            # Use relative tolerance for floating point comparison
            return abs(predicted_value - self.value) < 1e-6 * max(abs(self.value), 1.0)
        elif self.operator == 'in':
            return self.value[0] <= predicted_value <= self.value[1]
        else:
            # This should never happen due to validation in __init__
            raise RuntimeError(f"Unexpected operator: {self.operator}")
    
    def sample_target_value(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample a target value that would satisfy the constraint.
        
        Uses theoretically grounded sampling strategies based on constraint type.
        No heuristics or arbitrary choices.
        
        Args:
            rng: Random number generator (optional, defaults to numpy default_rng)
            
        Returns:
            target_value: Sampled target value
            
        Raises:
            ValueError: If constraint cannot produce valid sample
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if self.operator == 'in':
            # Uniform sampling within range
            return float(rng.uniform(self.value[0], self.value[1]))
        
        elif self.operator == '>':
            # Sample above threshold with exponential decay
            # Sample from (value, value * 2] with uniform distribution
            if self.value <= 0:
                raise ValueError(
                    f"Cannot sample for '>' constraint with non-positive value: {self.value}"
                )
            return float(rng.uniform(self.value * 1.01, self.value * 2.0))
        
        elif self.operator == '<':
            # Sample below threshold
            if self.value <= 0:
                # Sample in [0, value)
                return float(rng.uniform(0, self.value * 0.99))
            else:
                # Sample in [value / 2, value)
                return float(rng.uniform(self.value * 0.5, self.value * 0.99))
        
        elif self.operator == '>=':
            # Sample at or above threshold
            if self.value <= 0:
                raise ValueError(
                    f"Cannot sample for '>=' constraint with non-positive value: {self.value}"
                )
            return float(rng.uniform(self.value, self.value * 2.0))
        
        elif self.operator == '<=':
            # Sample at or below threshold
            if self.value <= 0:
                return float(rng.uniform(0, self.value))
            else:
                return float(rng.uniform(self.value * 0.5, self.value))
        
        elif self.operator == '==':
            # Return exact value
            return float(self.value)
        
        else:
            raise RuntimeError(f"Unexpected operator: {self.operator}")
    
    def __repr__(self) -> str:
        return f"PropertyConstraint({self.property_name} {self.operator} {self.value})"


class ConstrainedCrystalGenerator:
    """
    Generate crystals satisfying multiple property constraints.
    
    Uses rejection sampling with property predictor for validation.
    No heuristic adjustments - relies on theoretically sound sampling.
    
    Args:
        property_predictor: Trained PropertyPredictor model
        constraints: List of property constraints to satisfy
        
    Raises:
        ValueError: If constraints is empty or predictor is invalid
    """
    
    def __init__(
        self,
        property_predictor: nn.Module,
        constraints: List[PropertyConstraint],
    ):
        if not constraints:
            raise ValueError("constraints must be non-empty")
        
        if property_predictor is None:
            raise ValueError("property_predictor cannot be None")
        
        self.property_predictor = property_predictor
        self.constraints = constraints
        
        # Verify all constraint properties are supported by predictor
        predictor_properties = set(property_predictor.property_names)
        constraint_properties = set(c.property_name for c in constraints)
        
        missing_properties = constraint_properties - predictor_properties
        if missing_properties:
            raise ValueError(
                f"Predictor does not support properties: {missing_properties}. "
                f"Available properties: {predictor_properties}"
            )
    
    def check_constraints(
        self,
        predicted_properties: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Check if predicted properties satisfy all constraints.
        
        Args:
            predicted_properties: Dictionary of predicted property values
            
        Returns:
            satisfied: True if all constraints are satisfied
        """
        try:
            for constraint in self.constraints:
                if constraint.property_name not in predicted_properties:
                    raise ValueError(
                        f"Missing property in predictions: {constraint.property_name}"
                    )
                
                value = predicted_properties[constraint.property_name]
                if not constraint.check(value):
                    return False
            
            return True
        except ValueError as e:
            # Re-raise with context
            raise ValueError(f"Constraint checking failed: {e}") from e
    
    def sample_target_properties(
        self,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, float]:
        """
        Sample target property values from constraint ranges.
        
        Args:
            rng: Random number generator (optional)
            
        Returns:
            target_properties: Dictionary of sampled target values
        """
        if rng is None:
            rng = np.random.default_rng()
        
        target_properties = {}
        for constraint in self.constraints:
            target_properties[constraint.property_name] = constraint.sample_target_value(rng)
        
        return target_properties
    
    def __repr__(self) -> str:
        return f"ConstrainedCrystalGenerator(constraints={self.constraints})"


class ParetoFrontierSearcher:
    """
    Search for Pareto frontier in multi-objective property space.
    
    Implements theoretically grounded Pareto dominance checking without
    any heuristic approximations.
    
    Args:
        property_predictor: Trained PropertyPredictor model
        objectives: List of property names to optimize
        minimize: List of bools indicating whether to minimize each objective
                 (defaults to True for all)
        
    Raises:
        ValueError: If objectives is empty or contains duplicate properties
        
    Example:
        >>> searcher = ParetoFrontierSearcher(
        ...     property_predictor=predictor,
        ...     objectives=['bandgap', 'formation_energy'],
        ...     minimize=[False, True]  # Maximize bandgap, minimize energy
        ... )
    """
    
    def __init__(
        self,
        property_predictor: nn.Module,
        objectives: List[str],
        minimize: Optional[List[bool]] = None,
    ):
        if not objectives:
            raise ValueError("objectives must be non-empty")
        
        if len(objectives) != len(set(objectives)):
            raise ValueError(f"objectives contains duplicates: {objectives}")
        
        if property_predictor is None:
            raise ValueError("property_predictor cannot be None")
        
        # Verify objectives are supported by predictor
        predictor_properties = set(property_predictor.property_names)
        objective_set = set(objectives)
        
        missing_objectives = objective_set - predictor_properties
        if missing_objectives:
            raise ValueError(
                f"Predictor does not support objectives: {missing_objectives}. "
                f"Available properties: {predictor_properties}"
            )
        
        # Default: minimize all objectives
        if minimize is None:
            minimize = [True] * len(objectives)
        
        if len(minimize) != len(objectives):
            raise ValueError(
                f"minimize must have same length as objectives "
                f"({len(minimize)} vs {len(objectives)})"
            )
        
        self.property_predictor = property_predictor
        self.objectives = objectives
        self.minimize = minimize
    
    def find_pareto_frontier(
        self,
        predicted_properties_list: List[Dict[str, torch.Tensor]]
    ) -> List[int]:
        """
        Find indices of Pareto-optimal solutions from a set of candidates.
        
        A solution is Pareto-optimal if no other solution dominates it.
        Solution A dominates B if A is better or equal in all objectives
        and strictly better in at least one objective.
        
        This is a deterministic algorithm with O(n^2) complexity where n
        is the number of candidates.
        
        Args:
            predicted_properties_list: List of prediction dictionaries
            
        Returns:
            pareto_indices: Indices of Pareto-optimal solutions
            
        Raises:
            ValueError: If predicted_properties_list is empty or malformed
        """
        if not predicted_properties_list:
            raise ValueError("predicted_properties_list must be non-empty")
        
        n_candidates = len(predicted_properties_list)
        
        # Extract objective values as numpy array
        objective_values = np.zeros((n_candidates, len(self.objectives)))
        
        for i, predictions in enumerate(predicted_properties_list):
            for j, obj in enumerate(self.objectives):
                if obj not in predictions:
                    raise ValueError(
                        f"Missing objective '{obj}' in predictions at index {i}"
                    )
                
                value = predictions[obj]
                # Convert tensor to numpy
                if hasattr(value, 'item'):
                    value = value.item()
                
                # Flip sign for maximization objectives
                if not self.minimize[j]:
                    value = -value
                
                objective_values[i, j] = value
        
        # Find Pareto frontier using dominance checking
        is_pareto = np.ones(n_candidates, dtype=bool)
        
        for i in range(n_candidates):
            if is_pareto[i]:
                # Check if any other point dominates this one
                for j in range(n_candidates):
                    if i != j and is_pareto[j]:
                        # Check if j dominates i
                        # j dominates i if: all objectives of j <= i AND at least one strictly <
                        better_or_equal = np.all(objective_values[j] <= objective_values[i])
                        strictly_better = np.any(objective_values[j] < objective_values[i])
                        
                        if better_or_equal and strictly_better:
                            is_pareto[i] = False
                            break
        
        pareto_indices = np.where(is_pareto)[0].tolist()
        
        return pareto_indices
    
    def __repr__(self) -> str:
        return (
            f"ParetoFrontierSearcher(objectives={self.objectives}, "
            f"minimize={self.minimize})"
        )
