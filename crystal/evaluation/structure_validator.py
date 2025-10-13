"""
Structure validation utilities for crystal structures

This module provides comprehensive validation of crystal structures, checking:
- Coordinate validity and consistency
- Cell parameter physical constraints
- Geometric constraints (minimum distances, overlaps)
- Data format and completeness

Following the "no fallback heuristics" principle (ごまかしのためのfallbackは絶対にしない).
All validations raise explicit errors for invalid data.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from crystal.data.periodic_utils import (
    minimum_image_distance,
    compute_cell_volume,
    cartesian_to_fractional,
    fractional_to_cartesian,
)


class StructureValidator:
    """
    Comprehensive validation of crystal structures
    
    Validates:
    1. Data format and completeness
    2. Physical constraints on lattice parameters
    3. Coordinate validity (within cell boundaries)
    4. Geometric constraints (minimum distances, overlaps)
    5. Consistency between Cartesian and fractional coordinates
    """
    
    # Physical bounds for lattice parameters
    LENGTH_MIN = 1.0      # Angstrom
    LENGTH_MAX = 100.0    # Angstrom
    ANGLE_MIN = 30.0      # degrees
    ANGLE_MAX = 150.0     # degrees
    VOLUME_MIN = 1.0      # Angstrom³
    
    # Geometric constraints
    MIN_INTERATOMIC_DISTANCE = 0.5  # Angstrom
    
    def __init__(
        self,
        min_distance: float = MIN_INTERATOMIC_DISTANCE,
        length_bounds: Tuple[float, float] = (LENGTH_MIN, LENGTH_MAX),
        angle_bounds: Tuple[float, float] = (ANGLE_MIN, ANGLE_MAX),
        strict_mode: bool = True,
    ):
        """
        Initialize structure validator
        
        Args:
            min_distance: Minimum allowed interatomic distance in Angstroms
            length_bounds: (min, max) bounds for cell lengths in Angstroms
            angle_bounds: (min, max) bounds for cell angles in degrees
            strict_mode: If True, raises errors on validation failures.
                        If False, returns validation status without raising.
        
        Raises:
            ValueError: If any bounds are invalid
        """
        if min_distance <= 0:
            raise ValueError("min_distance must be positive")
        if length_bounds[0] <= 0 or length_bounds[1] <= length_bounds[0]:
            raise ValueError("Invalid length bounds")
        if angle_bounds[0] <= 0 or angle_bounds[1] <= angle_bounds[0]:
            raise ValueError("Invalid angle bounds")
        
        self.min_distance = min_distance
        self.length_min, self.length_max = length_bounds
        self.angle_min, self.angle_max = angle_bounds
        self.strict_mode = strict_mode
    
    def validate_structure(
        self,
        crystal: Dict,
        check_distances: bool = True,
        check_coordinates: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of a crystal structure
        
        Args:
            crystal: Dictionary containing crystal data
            check_distances: Whether to check interatomic distances (expensive for large systems)
            check_coordinates: Whether to validate coordinate consistency
        
        Returns:
            Tuple of (is_valid, error_messages)
            - is_valid: True if all checks pass
            - error_messages: List of error messages (empty if valid)
        
        Raises:
            ValueError: In strict mode, raises on first validation failure
        """
        errors = []
        
        # 1. Check data format
        try:
            self._validate_data_format(crystal)
        except ValueError as e:
            errors.append(str(e))
            if self.strict_mode:
                raise
        
        # If data format is invalid, cannot continue
        if errors:
            return False, errors
        
        # 2. Check cell parameters
        try:
            self._validate_cell_parameters(crystal['cell_params'])
        except ValueError as e:
            errors.append(f"Cell parameters: {e}")
            if self.strict_mode:
                raise ValueError(f"Cell parameters: {e}")
        
        # 3. Check cell volume
        try:
            self._validate_cell_volume(crystal['cell'], crystal['cell_volume'])
        except ValueError as e:
            errors.append(f"Cell volume: {e}")
            if self.strict_mode:
                raise ValueError(f"Cell volume: {e}")
        
        # 4. Check coordinate consistency
        if check_coordinates:
            try:
                self._validate_coordinate_consistency(
                    crystal.get('positions_cart'),
                    crystal.get('positions_frac'),
                    crystal['cell']
                )
            except ValueError as e:
                errors.append(f"Coordinates: {e}")
                if self.strict_mode:
                    raise ValueError(f"Coordinates: {e}")
        
        # 5. Check interatomic distances
        if check_distances:
            try:
                self._validate_minimum_distances(
                    crystal['positions_cart'],
                    crystal['cell'],
                    crystal['pbc']
                )
            except ValueError as e:
                errors.append(f"Interatomic distances: {e}")
                if self.strict_mode:
                    raise ValueError(f"Interatomic distances: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_data_format(self, crystal: Dict) -> None:
        """
        Validate that crystal dictionary contains all required fields with correct types
        
        Args:
            crystal: Crystal dictionary
        
        Raises:
            ValueError: If any required field is missing or has wrong type/shape
        """
        required_fields = {
            'positions_cart': (torch.Tensor, None),  # [n_atoms, 3]
            'cell': (torch.Tensor, (3, 3)),
            'cell_params': (torch.Tensor, (6,)),
            'cell_volume': (torch.Tensor, ()),  # scalar
            'pbc': (torch.Tensor, (3,)),
        }
        
        for field, (expected_type, expected_shape) in required_fields.items():
            # Check field exists
            if field not in crystal:
                raise ValueError(f"Missing required field: {field}")
            
            value = crystal[field]
            
            # Check type
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Field {field} has wrong type: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            
            # Check shape (if specified)
            if expected_shape is not None:
                if value.shape != expected_shape:
                    raise ValueError(
                        f"Field {field} has wrong shape: expected {expected_shape}, "
                        f"got {value.shape}"
                    )
            
            # Check for NaN or Inf in numeric fields
            if field != 'pbc' and torch.is_tensor(value):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Field {field} contains NaN or Inf values")
        
        # Additional shape checks for positions
        positions = crystal['positions_cart']
        if positions.dim() != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions_cart must have shape [n_atoms, 3], got {positions.shape}"
            )
        
        if positions.shape[0] == 0:
            raise ValueError("Crystal must contain at least one atom")
    
    def _validate_cell_parameters(self, cell_params: torch.Tensor) -> None:
        """
        Validate that cell parameters satisfy physical constraints
        
        Args:
            cell_params: Cell parameters [6] (a, b, c, alpha, beta, gamma)
        
        Raises:
            ValueError: If any parameter is out of physical bounds
        """
        lengths = cell_params[:3]
        angles = cell_params[3:]
        
        # Check lengths
        for i, (length, name) in enumerate(zip(lengths, ['a', 'b', 'c'])):
            length_val = length.item()
            if length_val < self.length_min or length_val > self.length_max:
                raise ValueError(
                    f"Cell length {name}={length_val:.3f} Å is out of bounds "
                    f"[{self.length_min}, {self.length_max}]"
                )
        
        # Check angles
        for i, (angle, name) in enumerate(zip(angles, ['alpha', 'beta', 'gamma'])):
            angle_val = angle.item()
            if angle_val < self.angle_min or angle_val > self.angle_max:
                raise ValueError(
                    f"Cell angle {name}={angle_val:.3f}° is out of bounds "
                    f"[{self.angle_min}, {self.angle_max}]"
                )
    
    def _validate_cell_volume(
        self,
        cell: torch.Tensor,
        cell_volume: torch.Tensor
    ) -> None:
        """
        Validate cell volume consistency and physical bounds
        
        Args:
            cell: Cell vectors [3, 3]
            cell_volume: Declared cell volume (scalar)
        
        Raises:
            ValueError: If volume is invalid or inconsistent
        """
        volume_val = cell_volume.item()
        
        # Check volume is positive
        if volume_val <= self.VOLUME_MIN:
            raise ValueError(f"Cell volume {volume_val:.3f} Å³ is too small")
        
        # Compute volume from cell vectors and check consistency
        computed_volume = compute_cell_volume(cell.unsqueeze(0)).squeeze(0)
        computed_val = computed_volume.item()
        
        # Allow 1% relative error for numerical precision
        relative_error = abs(computed_val - volume_val) / max(computed_val, volume_val)
        if relative_error > 0.01:
            raise ValueError(
                f"Cell volume inconsistent: declared={volume_val:.3f} Å³, "
                f"computed={computed_val:.3f} Å³ (error={relative_error*100:.2f}%)"
            )
    
    def _validate_coordinate_consistency(
        self,
        positions_cart: Optional[torch.Tensor],
        positions_frac: Optional[torch.Tensor],
        cell: torch.Tensor,
    ) -> None:
        """
        Validate consistency between Cartesian and fractional coordinates
        
        Args:
            positions_cart: Cartesian positions [n_atoms, 3] or None
            positions_frac: Fractional positions [n_atoms, 3] or None
            cell: Cell vectors [3, 3]
        
        Raises:
            ValueError: If coordinates are inconsistent
        """
        # If only one type of coordinates provided, cannot check consistency
        if positions_cart is None or positions_frac is None:
            return
        
        # Convert fractional to Cartesian
        positions_cart_from_frac = fractional_to_cartesian(
            positions_frac.unsqueeze(0),
            cell.unsqueeze(0)
        ).squeeze(0)
        
        # Check consistency (allow small numerical error)
        max_diff = torch.max(torch.abs(positions_cart - positions_cart_from_frac)).item()
        if max_diff > 1e-3:  # 0.001 Angstrom tolerance
            raise ValueError(
                f"Cartesian and fractional coordinates inconsistent: "
                f"max difference = {max_diff:.6f} Å"
            )
        
        # Check fractional coordinates are reasonably bounded
        # (should typically be in [0, 1] but can be outside for wrapped systems)
        frac_min = torch.min(positions_frac).item()
        frac_max = torch.max(positions_frac).item()
        
        # Allow [-1, 2] range (reasonable for wrapped coordinates)
        if frac_min < -1.0 or frac_max > 2.0:
            raise ValueError(
                f"Fractional coordinates out of reasonable range: "
                f"[{frac_min:.3f}, {frac_max:.3f}]"
            )
    
    def _validate_minimum_distances(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ) -> None:
        """
        Validate that all interatomic distances respect minimum threshold
        
        Args:
            positions: Atomic positions [n_atoms, 3]
            cell: Cell vectors [3, 3]
            pbc: Periodic boundary conditions [3]
        
        Raises:
            ValueError: If any pair of atoms is closer than min_distance
        """
        n_atoms = positions.shape[0]
        
        # Need at least 2 atoms to check distances
        if n_atoms < 2:
            return
        
        # Compute all pairwise distances with PBC
        distances, _ = minimum_image_distance(
            positions.unsqueeze(0),  # [1, n_atoms, 3]
            positions.unsqueeze(0),
            cell.unsqueeze(0),       # [1, 3, 3]
            pbc.unsqueeze(0),        # [1, 3]
            use_fractional=False,
        )
        
        distances = distances.squeeze(0)  # [n_atoms, n_atoms]
        
        # Exclude self-distances (diagonal)
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=distances.device)
        off_diagonal_distances = distances[mask]
        
        # Find minimum distance
        min_dist = torch.min(off_diagonal_distances).item()
        
        if min_dist < self.min_distance:
            raise ValueError(
                f"Minimum interatomic distance {min_dist:.3f} Å is below "
                f"threshold {self.min_distance:.3f} Å"
            )
    
    def validate_batch(
        self,
        crystals: List[Dict],
        check_distances: bool = True,
        check_coordinates: bool = True,
        return_details: bool = False,
    ) -> Tuple[bool, Optional[List[Tuple[int, List[str]]]]]:
        """
        Validate a batch of crystal structures
        
        Args:
            crystals: List of crystal dictionaries
            check_distances: Whether to check interatomic distances
            check_coordinates: Whether to check coordinate consistency
            return_details: Whether to return detailed error messages per structure
        
        Returns:
            If return_details=False: (all_valid, None)
            If return_details=True: (all_valid, [(idx, errors), ...])
        
        Raises:
            ValueError: In strict mode, raises on first validation failure
        """
        all_valid = True
        details = [] if return_details else None
        
        for idx, crystal in enumerate(crystals):
            is_valid, errors = self.validate_structure(
                crystal,
                check_distances=check_distances,
                check_coordinates=check_coordinates,
            )
            
            if not is_valid:
                all_valid = False
                if return_details:
                    details.append((idx, errors))
                elif self.strict_mode:
                    raise ValueError(f"Crystal {idx} validation failed: {errors}")
        
        return all_valid, details
