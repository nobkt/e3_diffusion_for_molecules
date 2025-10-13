"""
Symmetry analysis tools for crystal structures

This module provides symmetry analysis capabilities for crystal structures:
- Space group detection from structure
- Symmetry operation identification
- Structure fingerprinting for comparison
- Wyckoff position analysis

Following the "no fallback heuristics" principle (ごまかしのためのfallbackは絶対にしない).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class SymmetryAnalyzer:
    """
    Symmetry analysis for crystal structures
    
    Provides tools for:
    1. Space group detection (requires spglib)
    2. Symmetry operation identification
    3. Structure fingerprinting
    4. Wyckoff position analysis
    
    Note: Advanced symmetry detection requires spglib library.
    Basic fingerprinting is available without external dependencies.
    """
    
    def __init__(
        self,
        symprec: float = 1e-3,
        angle_tolerance: float = 5.0,
    ):
        """
        Initialize symmetry analyzer
        
        Args:
            symprec: Symmetry detection precision in Angstroms
            angle_tolerance: Angle tolerance in degrees for symmetry operations
        
        Raises:
            ValueError: If precision parameters are invalid
        """
        if symprec <= 0:
            raise ValueError("symprec must be positive")
        if angle_tolerance <= 0:
            raise ValueError("angle_tolerance must be positive")
        
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        
        # Check if spglib is available for advanced symmetry analysis
        try:
            import spglib
            self.spglib = spglib
            self.has_spglib = True
        except ImportError:
            self.spglib = None
            self.has_spglib = False
            warnings.warn(
                "spglib not available. Advanced symmetry detection will be disabled. "
                "Install with: pip install spglib"
            )
    
    def detect_space_group(
        self,
        crystal: Dict,
    ) -> Optional[Dict[str, any]]:
        """
        Detect space group from crystal structure
        
        Requires spglib to be installed.
        
        Args:
            crystal: Crystal dictionary containing:
                - 'positions_cart' or 'positions_frac': atomic positions
                - 'cell': cell vectors [3, 3]
                - 'atom_types': atomic numbers or types [n_atoms]
        
        Returns:
            Dictionary containing:
                - 'space_group_number': International space group number (1-230)
                - 'space_group_symbol': Hermann-Mauguin symbol
                - 'point_group': Point group symbol
                - 'crystal_system': Crystal system name
                - 'hall_number': Hall space group number
            Returns None if spglib is not available or detection fails
        
        Raises:
            ValueError: If crystal data is invalid or missing required fields
        """
        if not self.has_spglib:
            warnings.warn("spglib not available, cannot detect space group")
            return None
        
        # Validate required fields
        if 'cell' not in crystal:
            raise ValueError("Crystal missing 'cell' field")
        if 'atom_types' not in crystal:
            raise ValueError("Crystal missing 'atom_types' field")
        
        # Get positions (prefer fractional, convert if needed)
        if 'positions_frac' in crystal:
            positions = crystal['positions_frac']
        elif 'positions_cart' in crystal:
            from crystal.data.periodic_utils import cartesian_to_fractional
            cell = crystal['cell']
            if not isinstance(cell, torch.Tensor):
                cell = torch.tensor(cell, dtype=torch.float32)
            positions_cart = crystal['positions_cart']
            if not isinstance(positions_cart, torch.Tensor):
                positions_cart = torch.tensor(positions_cart, dtype=torch.float32)
            
            positions = cartesian_to_fractional(
                positions_cart.unsqueeze(0),
                cell.unsqueeze(0)
            ).squeeze(0)
        else:
            raise ValueError("Crystal missing position data")
        
        # Convert to numpy for spglib
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        
        cell = crystal['cell']
        if isinstance(cell, torch.Tensor):
            cell = cell.cpu().numpy()
        
        atom_types = crystal['atom_types']
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.cpu().numpy()
        
        # Create cell tuple for spglib: (lattice, positions, numbers)
        # spglib expects lattice as row vectors
        lattice = cell.T  # Transpose to get row vectors
        
        spglib_cell = (lattice, positions, atom_types)
        
        # Detect space group
        try:
            dataset = self.spglib.get_symmetry_dataset(
                spglib_cell,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance
            )
            
            if dataset is None:
                warnings.warn("Space group detection failed")
                return None
            
            result = {
                'space_group_number': int(dataset['number']),
                'space_group_symbol': dataset['international'],
                'point_group': dataset['pointgroup'],
                'crystal_system': self._get_crystal_system(dataset['number']),
                'hall_number': int(dataset['hall_number']),
            }
            
            return result
            
        except Exception as e:
            warnings.warn(f"Space group detection error: {e}")
            return None
    
    def compute_structure_fingerprint(
        self,
        crystal: Dict,
        n_bins: int = 100,
        r_max: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute structure fingerprint using radial distribution function
        
        Creates a fingerprint based on pairwise distance distribution,
        useful for comparing crystal structures without symmetry detection.
        
        Args:
            crystal: Crystal dictionary
            n_bins: Number of bins for distance histogram
            r_max: Maximum distance in Angstroms
        
        Returns:
            Fingerprint tensor [n_bins] normalized to sum to 1
        
        Raises:
            ValueError: If crystal data is invalid
        """
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if r_max <= 0:
            raise ValueError("r_max must be positive")
        
        # Validate required fields
        required_fields = ['positions_cart', 'cell', 'pbc']
        for field in required_fields:
            if field not in crystal:
                raise ValueError(f"Crystal missing required field: {field}")
        
        positions = crystal['positions_cart']
        cell = crystal['cell']
        pbc = crystal['pbc']
        
        # Convert to tensors if needed
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        if not isinstance(cell, torch.Tensor):
            cell = torch.tensor(cell, dtype=torch.float32)
        if not isinstance(pbc, torch.Tensor):
            pbc = torch.tensor(pbc, dtype=torch.bool)
        
        # Compute all pairwise distances with PBC
        from crystal.data.periodic_utils import minimum_image_distance
        
        distances, _ = minimum_image_distance(
            positions.unsqueeze(0),
            positions.unsqueeze(0),
            cell.unsqueeze(0),
            pbc.unsqueeze(0),
            use_fractional=False,
        )
        
        distances = distances.squeeze(0)  # [n_atoms, n_atoms]
        
        # Exclude self-distances (diagonal)
        n_atoms = distances.shape[0]
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=distances.device)
        distances = distances[mask].flatten()
        
        # Filter distances within r_max
        distances = distances[distances <= r_max]
        
        if distances.numel() == 0:
            warnings.warn("No distances found within r_max, returning zero fingerprint")
            return torch.zeros(n_bins)
        
        # Create histogram
        hist = torch.histc(distances, bins=n_bins, min=0, max=r_max)
        
        # Normalize to create probability distribution
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def compare_fingerprints(
        self,
        fingerprint1: torch.Tensor,
        fingerprint2: torch.Tensor,
        metric: str = 'l2',
    ) -> float:
        """
        Compare two structure fingerprints
        
        Args:
            fingerprint1: First fingerprint tensor
            fingerprint2: Second fingerprint tensor
            metric: Comparison metric ('l2', 'l1', or 'cosine')
        
        Returns:
            Distance/similarity measure (lower is more similar for l1/l2,
            higher is more similar for cosine)
        
        Raises:
            ValueError: If fingerprints have different shapes or metric is invalid
        """
        if fingerprint1.shape != fingerprint2.shape:
            raise ValueError(
                f"Fingerprints must have same shape: "
                f"{fingerprint1.shape} vs {fingerprint2.shape}"
            )
        
        if metric == 'l2':
            # Euclidean distance
            return torch.norm(fingerprint1 - fingerprint2, p=2).item()
        elif metric == 'l1':
            # Manhattan distance
            return torch.norm(fingerprint1 - fingerprint2, p=1).item()
        elif metric == 'cosine':
            # Cosine similarity
            dot = torch.dot(fingerprint1, fingerprint2)
            norm1 = torch.norm(fingerprint1)
            norm2 = torch.norm(fingerprint2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return (dot / (norm1 * norm2)).item()
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'l1', 'l2', or 'cosine'")
    
    def analyze_lattice_symmetry(
        self,
        cell_params: torch.Tensor,
        tolerance: float = 1e-2,
    ) -> Dict[str, any]:
        """
        Analyze lattice symmetry from cell parameters
        
        Identifies lattice type based on relationships between cell parameters:
        - Cubic: a = b = c, α = β = γ = 90°
        - Tetragonal: a = b ≠ c, α = β = γ = 90°
        - Orthorhombic: a ≠ b ≠ c, α = β = γ = 90°
        - Hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
        - Rhombohedral: a = b = c, α = β = γ ≠ 90°
        - Monoclinic: a ≠ b ≠ c, α = γ = 90° ≠ β
        - Triclinic: a ≠ b ≠ c, α ≠ β ≠ γ
        
        Args:
            cell_params: Cell parameters [6] (a, b, c, alpha, beta, gamma)
            tolerance: Relative tolerance for comparing values
        
        Returns:
            Dictionary with:
                - 'lattice_type': One of the seven lattice types
                - 'is_orthogonal': Whether all angles are 90°
                - 'length_ratios': Ratios of cell lengths
        
        Raises:
            ValueError: If cell_params has wrong shape
        """
        if cell_params.shape != (6,):
            raise ValueError(f"cell_params must have shape (6,), got {cell_params.shape}")
        
        if isinstance(cell_params, torch.Tensor):
            cell_params = cell_params.cpu().numpy()
        
        a, b, c = cell_params[:3]
        alpha, beta, gamma = cell_params[3:]
        
        # Helper function for approximate equality
        def approx_equal(x, y, tol=tolerance):
            return abs(x - y) / max(abs(x), abs(y)) < tol
        
        # Check angle relationships
        all_90 = (approx_equal(alpha, 90.0, tolerance) and
                  approx_equal(beta, 90.0, tolerance) and
                  approx_equal(gamma, 90.0, tolerance))
        
        two_90 = sum([approx_equal(alpha, 90.0, tolerance),
                      approx_equal(beta, 90.0, tolerance),
                      approx_equal(gamma, 90.0, tolerance)]) == 2
        
        gamma_120 = approx_equal(gamma, 120.0, tolerance)
        all_equal_angles = (approx_equal(alpha, beta, tolerance) and
                           approx_equal(beta, gamma, tolerance))
        
        # Check length relationships
        a_eq_b = approx_equal(a, b, tolerance)
        b_eq_c = approx_equal(b, c, tolerance)
        a_eq_c = approx_equal(a, c, tolerance)
        all_equal_lengths = a_eq_b and b_eq_c
        
        # Determine lattice type
        if all_equal_lengths and all_90:
            lattice_type = 'cubic'
        elif a_eq_b and not b_eq_c and all_90:
            lattice_type = 'tetragonal'
        elif not a_eq_b and not b_eq_c and not a_eq_c and all_90:
            lattice_type = 'orthorhombic'
        elif a_eq_b and not b_eq_c and two_90 and gamma_120:
            lattice_type = 'hexagonal'
        elif all_equal_lengths and all_equal_angles and not all_90:
            lattice_type = 'rhombohedral'
        elif two_90 and not all_90:
            lattice_type = 'monoclinic'
        else:
            lattice_type = 'triclinic'
        
        return {
            'lattice_type': lattice_type,
            'is_orthogonal': all_90,
            'length_ratios': {
                'b_over_a': float(b / a),
                'c_over_a': float(c / a),
                'c_over_b': float(c / b),
            },
            'angles': {
                'alpha': float(alpha),
                'beta': float(beta),
                'gamma': float(gamma),
            },
        }
    
    def _get_crystal_system(self, space_group_number: int) -> str:
        """
        Get crystal system from space group number
        
        Args:
            space_group_number: International space group number (1-230)
        
        Returns:
            Crystal system name
        
        Raises:
            ValueError: If space_group_number is out of range
        """
        if space_group_number < 1 or space_group_number > 230:
            raise ValueError(
                f"Space group number must be between 1 and 230, got {space_group_number}"
            )
        
        # Crystal system ranges (based on International Tables)
        if 1 <= space_group_number <= 2:
            return 'triclinic'
        elif 3 <= space_group_number <= 15:
            return 'monoclinic'
        elif 16 <= space_group_number <= 74:
            return 'orthorhombic'
        elif 75 <= space_group_number <= 142:
            return 'tetragonal'
        elif 143 <= space_group_number <= 167:
            return 'trigonal'
        elif 168 <= space_group_number <= 194:
            return 'hexagonal'
        elif 195 <= space_group_number <= 230:
            return 'cubic'
        else:
            # Should never reach here due to initial check
            raise ValueError(f"Invalid space group number: {space_group_number}")
