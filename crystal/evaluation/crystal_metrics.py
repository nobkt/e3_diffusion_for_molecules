"""
Crystal structure evaluation metrics

This module implements comprehensive evaluation metrics for generated molecular crystals,
following the "no fallback heuristics" principle (ごまかしのためのfallbackは絶対にしない).

All metrics are theoretically sound and physically meaningful.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from crystal.data.periodic_utils import minimum_image_distance, compute_cell_volume


class CrystalMetrics:
    """
    Comprehensive evaluation metrics for generated crystal structures
    
    Provides three categories of metrics:
    1. Structural metrics: Statistics of lattice parameters, volume, density
    2. Validity metrics: Physical constraint violations, minimum distance checks
    3. Distribution metrics: Comparison with reference distributions (Wasserstein distance)
    
    All metrics strictly validate inputs and raise errors for invalid data.
    """
    
    def __init__(self, dataset_info: dict):
        """
        Initialize crystal metrics evaluator
        
        Args:
            dataset_info: Dictionary containing dataset information including:
                - 'atom_decoder': List mapping atom type indices to elements (optional)
                - Other dataset-specific metadata
        
        Raises:
            ValueError: If dataset_info is None or invalid
        """
        if dataset_info is None:
            raise ValueError("dataset_info cannot be None")
        if not isinstance(dataset_info, dict):
            raise ValueError("dataset_info must be a dictionary")
            
        self.dataset_info = dataset_info
        self.atom_decoder = dataset_info.get('atom_decoder', [])
    
    def compute_all_metrics(
        self,
        generated_crystals: List[Dict],
        reference_crystals: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics for generated crystals
        
        Args:
            generated_crystals: List of generated crystal dictionaries, each containing:
                - 'positions_cart': Cartesian positions [n_atoms, 3]
                - 'cell': Cell vectors [3, 3]
                - 'cell_params': Cell parameters [6] (a, b, c, alpha, beta, gamma)
                - 'cell_volume': Cell volume (scalar)
                - 'pbc': Periodic boundary conditions [3] (bool)
                - 'density': Density in g/cm³ (optional)
            reference_crystals: Optional list of reference crystal dictionaries
                (same format as generated_crystals)
        
        Returns:
            Dictionary containing all computed metrics
        
        Raises:
            ValueError: If generated_crystals is empty or contains invalid data
        """
        if not generated_crystals:
            raise ValueError("generated_crystals cannot be empty")
        
        # Validate crystal data format
        self._validate_crystal_list(generated_crystals)
        if reference_crystals is not None:
            self._validate_crystal_list(reference_crystals)
        
        metrics = {}
        
        # Structural metrics
        metrics.update(self.compute_structural_metrics(generated_crystals))
        
        # Validity metrics
        metrics.update(self.compute_validity_metrics(generated_crystals))
        
        # Distribution comparison metrics (if reference data available)
        if reference_crystals is not None:
            metrics.update(self.compute_distribution_metrics(
                generated_crystals, reference_crystals
            ))
        
        return metrics
    
    def compute_structural_metrics(
        self,
        crystals: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute structural statistics for crystal structures
        
        Metrics include:
        - Mean and standard deviation of lattice parameters (a, b, c, alpha, beta, gamma)
        - Mean and standard deviation of cell volume
        - Mean and standard deviation of density (if available)
        
        Args:
            crystals: List of crystal dictionaries
        
        Returns:
            Dictionary of structural metrics
        
        Raises:
            ValueError: If crystals is empty or data is invalid
        """
        if not crystals:
            raise ValueError("crystals cannot be empty")
        
        metrics = {}
        
        # Lattice parameters statistics
        cell_params_list = []
        for crystal in crystals:
            if 'cell_params' not in crystal:
                raise ValueError("Crystal missing 'cell_params' field")
            params = crystal['cell_params']
            if isinstance(params, torch.Tensor):
                params = params.cpu().numpy()
            if params.shape != (6,):
                raise ValueError(f"cell_params must have shape (6,), got {params.shape}")
            cell_params_list.append(params)
        
        cell_params_array = np.array(cell_params_list)
        
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for i, param_name in enumerate(param_names):
            values = cell_params_array[:, i]
            metrics[f'{param_name}_mean'] = float(np.mean(values))
            metrics[f'{param_name}_std'] = float(np.std(values))
            metrics[f'{param_name}_min'] = float(np.min(values))
            metrics[f'{param_name}_max'] = float(np.max(values))
        
        # Volume statistics
        volumes = []
        for crystal in crystals:
            if 'cell_volume' not in crystal:
                raise ValueError("Crystal missing 'cell_volume' field")
            volume = crystal['cell_volume']
            if isinstance(volume, torch.Tensor):
                volume = volume.item()
            if not np.isfinite(volume) or volume <= 0:
                raise ValueError(f"Invalid cell volume: {volume}")
            volumes.append(volume)
        
        volumes = np.array(volumes)
        metrics['volume_mean'] = float(np.mean(volumes))
        metrics['volume_std'] = float(np.std(volumes))
        metrics['volume_min'] = float(np.min(volumes))
        metrics['volume_max'] = float(np.max(volumes))
        
        # Density statistics (if available)
        if 'density' in crystals[0]:
            densities = []
            for crystal in crystals:
                density = crystal['density']
                if isinstance(density, torch.Tensor):
                    density = density.item()
                if not np.isfinite(density) or density <= 0:
                    raise ValueError(f"Invalid density: {density}")
                densities.append(density)
            
            densities = np.array(densities)
            metrics['density_mean'] = float(np.mean(densities))
            metrics['density_std'] = float(np.std(densities))
            metrics['density_min'] = float(np.min(densities))
            metrics['density_max'] = float(np.max(densities))
        
        return metrics
    
    def compute_validity_metrics(
        self,
        crystals: List[Dict],
        min_distance_threshold: float = 0.5,  # Angstrom
    ) -> Dict[str, float]:
        """
        Compute validity metrics for crystal structures
        
        Checks:
        - Physical validity of lattice parameters
        - Minimum interatomic distance violations (through PBC)
        - Overall structure validity
        
        Args:
            crystals: List of crystal dictionaries
            min_distance_threshold: Minimum allowed interatomic distance in Angstroms
        
        Returns:
            Dictionary of validity metrics
        
        Raises:
            ValueError: If crystals is empty or min_distance_threshold is invalid
        """
        if not crystals:
            raise ValueError("crystals cannot be empty")
        if min_distance_threshold <= 0:
            raise ValueError("min_distance_threshold must be positive")
        
        metrics = {}
        
        valid_count = 0
        min_distance_violations = 0
        cell_param_violations = 0
        
        for crystal in crystals:
            is_valid, has_min_dist_violation, has_cell_violation = \
                self.check_crystal_validity(crystal, min_distance_threshold)
            
            if is_valid:
                valid_count += 1
            if has_min_dist_violation:
                min_distance_violations += 1
            if has_cell_violation:
                cell_param_violations += 1
        
        n_crystals = len(crystals)
        metrics['validity_ratio'] = valid_count / n_crystals
        metrics['min_distance_violation_ratio'] = min_distance_violations / n_crystals
        metrics['cell_param_violation_ratio'] = cell_param_violations / n_crystals
        metrics['n_valid'] = valid_count
        metrics['n_total'] = n_crystals
        
        return metrics
    
    def check_crystal_validity(
        self,
        crystal: Dict,
        min_distance_threshold: float = 0.5  # Angstrom
    ) -> Tuple[bool, bool, bool]:
        """
        Check validity of a single crystal structure
        
        Performs three checks:
        1. Lattice parameter validity (lengths: 1-100 Å, angles: 30-150°)
        2. Minimum interatomic distance (with PBC)
        3. Overall validity (passes all checks)
        
        Args:
            crystal: Crystal dictionary
            min_distance_threshold: Minimum allowed distance in Angstroms
        
        Returns:
            Tuple of (is_valid, has_min_distance_violation, has_cell_param_violation)
        
        Raises:
            ValueError: If crystal data is missing or invalid
        """
        # Validate required fields
        required_fields = ['positions_cart', 'cell', 'cell_params', 'pbc']
        for field in required_fields:
            if field not in crystal:
                raise ValueError(f"Crystal missing required field: {field}")
        
        positions = crystal['positions_cart']
        cell = crystal['cell']
        cell_params = crystal['cell_params']
        pbc = crystal['pbc']
        
        # Convert to tensors if needed
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        if not isinstance(cell, torch.Tensor):
            cell = torch.tensor(cell, dtype=torch.float32)
        if not isinstance(cell_params, torch.Tensor):
            cell_params = torch.tensor(cell_params, dtype=torch.float32)
        if not isinstance(pbc, torch.Tensor):
            pbc = torch.tensor(pbc, dtype=torch.bool)
        
        # Check lattice parameter validity
        lengths = cell_params[:3]
        angles = cell_params[3:]
        
        valid_lengths = torch.all(lengths >= 1.0) and torch.all(lengths <= 100.0)
        valid_angles = torch.all(angles >= 30.0) and torch.all(angles <= 150.0)
        has_cell_param_violation = not (valid_lengths and valid_angles)
        
        # Check minimum interatomic distances with PBC
        has_min_distance_violation = False
        
        if positions.shape[0] > 1:  # Need at least 2 atoms
            try:
                # Compute pairwise distances with PBC
                distances, _ = minimum_image_distance(
                    positions.unsqueeze(0),  # [1, n_atoms, 3]
                    positions.unsqueeze(0),
                    cell.unsqueeze(0),  # [1, 3, 3]
                    pbc.unsqueeze(0),  # [1, 3]
                    use_fractional=False,
                )
                
                distances = distances.squeeze(0)  # [n_atoms, n_atoms]
                
                # Exclude self-distances (diagonal)
                n = distances.shape[0]
                mask = ~torch.eye(n, dtype=torch.bool, device=distances.device)
                off_diagonal_distances = distances[mask]
                
                if off_diagonal_distances.numel() > 0:
                    min_distance = torch.min(off_diagonal_distances).item()
                    has_min_distance_violation = min_distance < min_distance_threshold
            except Exception as e:
                # If distance calculation fails, consider it a violation
                has_min_distance_violation = True
        
        # Overall validity: passes all checks
        is_valid = (not has_cell_param_violation) and (not has_min_distance_violation)
        
        return is_valid, has_min_distance_violation, has_cell_param_violation
    
    def compute_distribution_metrics(
        self,
        generated_crystals: List[Dict],
        reference_crystals: List[Dict]
    ) -> Dict[str, float]:
        """
        Compare distributions of generated and reference crystals
        
        Uses Wasserstein distance (Earth Mover's Distance) to compare:
        - Lattice parameter distributions
        - Volume distribution
        - Density distribution (if available)
        
        Args:
            generated_crystals: List of generated crystal dictionaries
            reference_crystals: List of reference crystal dictionaries
        
        Returns:
            Dictionary of distribution comparison metrics
        
        Raises:
            ValueError: If either crystal list is empty
            ImportError: If scipy is not available
        """
        if not generated_crystals:
            raise ValueError("generated_crystals cannot be empty")
        if not reference_crystals:
            raise ValueError("reference_crystals cannot be empty")
        
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            raise ImportError("scipy is required for distribution metrics")
        
        metrics = {}
        
        # Extract lattice parameters
        gen_params = []
        for crystal in generated_crystals:
            params = crystal['cell_params']
            if isinstance(params, torch.Tensor):
                params = params.cpu().numpy()
            gen_params.append(params)
        gen_params = np.array(gen_params)
        
        ref_params = []
        for crystal in reference_crystals:
            params = crystal['cell_params']
            if isinstance(params, torch.Tensor):
                params = params.cpu().numpy()
            ref_params.append(params)
        ref_params = np.array(ref_params)
        
        # Wasserstein distance for each lattice parameter
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for i, param_name in enumerate(param_names):
            wd = wasserstein_distance(gen_params[:, i], ref_params[:, i])
            metrics[f'{param_name}_wasserstein'] = float(wd)
        
        # Volume distribution comparison
        gen_volumes = np.array([
            crystal['cell_volume'].item() if isinstance(crystal['cell_volume'], torch.Tensor)
            else crystal['cell_volume']
            for crystal in generated_crystals
        ])
        ref_volumes = np.array([
            crystal['cell_volume'].item() if isinstance(crystal['cell_volume'], torch.Tensor)
            else crystal['cell_volume']
            for crystal in reference_crystals
        ])
        metrics['volume_wasserstein'] = float(wasserstein_distance(gen_volumes, ref_volumes))
        
        # Density distribution comparison (if available)
        if 'density' in generated_crystals[0] and 'density' in reference_crystals[0]:
            gen_densities = np.array([
                crystal['density'].item() if isinstance(crystal['density'], torch.Tensor)
                else crystal['density']
                for crystal in generated_crystals
            ])
            ref_densities = np.array([
                crystal['density'].item() if isinstance(crystal['density'], torch.Tensor)
                else crystal['density']
                for crystal in reference_crystals
            ])
            metrics['density_wasserstein'] = float(wasserstein_distance(gen_densities, ref_densities))
        
        return metrics
    
    def _validate_crystal_list(self, crystals: List[Dict]) -> None:
        """
        Validate that all crystals in list have required fields
        
        Args:
            crystals: List of crystal dictionaries
        
        Raises:
            ValueError: If any crystal is missing required fields
        """
        required_fields = ['positions_cart', 'cell', 'cell_params', 'cell_volume', 'pbc']
        
        for i, crystal in enumerate(crystals):
            if not isinstance(crystal, dict):
                raise ValueError(f"Crystal {i} is not a dictionary")
            for field in required_fields:
                if field not in crystal:
                    raise ValueError(f"Crystal {i} missing required field: {field}")
