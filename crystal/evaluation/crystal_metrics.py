"""
Evaluation metrics for crystal structures.

This module provides various metrics for evaluating the quality of generated crystal structures.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


class CrystalMetrics:
    """
    Collection of metrics for evaluating crystal structures.
    """
    
    @staticmethod
    def lattice_parameter_mae(
        generated_params: np.ndarray,
        reference_params: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Mean Absolute Error for lattice parameters.
        
        Args:
            generated_params: (n_gen, 6) array of [a, b, c, alpha, beta, gamma]
            reference_params: (n_ref, 6) array of [a, b, c, alpha, beta, gamma]
            
        Returns:
            Dictionary with MAE for each parameter
        """
        # Compute MAE for lengths (a, b, c)
        mae_a = np.abs(generated_params[:, 0] - reference_params[:, 0].mean()).mean()
        mae_b = np.abs(generated_params[:, 1] - reference_params[:, 1].mean()).mean()
        mae_c = np.abs(generated_params[:, 2] - reference_params[:, 2].mean()).mean()
        
        # Compute MAE for angles (alpha, beta, gamma)
        mae_alpha = np.abs(generated_params[:, 3] - reference_params[:, 3].mean()).mean()
        mae_beta = np.abs(generated_params[:, 4] - reference_params[:, 4].mean()).mean()
        mae_gamma = np.abs(generated_params[:, 5] - reference_params[:, 5].mean()).mean()
        
        return {
            'mae_a': mae_a,
            'mae_b': mae_b,
            'mae_c': mae_c,
            'mae_alpha': mae_alpha,
            'mae_beta': mae_beta,
            'mae_gamma': mae_gamma,
            'mae_lengths': (mae_a + mae_b + mae_c) / 3,
            'mae_angles': (mae_alpha + mae_beta + mae_gamma) / 3
        }
    
    @staticmethod
    def density_statistics(
        generated_densities: np.ndarray,
        reference_densities: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute density statistics.
        
        Args:
            generated_densities: (n_gen,) array of densities
            reference_densities: (n_ref,) array of densities
            
        Returns:
            Dictionary with density statistics
        """
        mae = np.abs(generated_densities.mean() - reference_densities.mean())
        mse = (generated_densities.mean() - reference_densities.mean()) ** 2
        
        # Wasserstein distance (Earth Mover's Distance)
        emd = wasserstein_distance(generated_densities, reference_densities)
        
        return {
            'density_mae': mae,
            'density_mse': mse,
            'density_emd': emd,
            'gen_density_mean': generated_densities.mean(),
            'gen_density_std': generated_densities.std(),
            'ref_density_mean': reference_densities.mean(),
            'ref_density_std': reference_densities.std()
        }
    
    @staticmethod
    def volume_statistics(
        generated_volumes: np.ndarray,
        reference_volumes: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute volume statistics.
        
        Args:
            generated_volumes: (n_gen,) array of volumes
            reference_volumes: (n_ref,) array of volumes
            
        Returns:
            Dictionary with volume statistics
        """
        mae = np.abs(generated_volumes.mean() - reference_volumes.mean())
        relative_error = mae / reference_volumes.mean()
        
        # Wasserstein distance
        emd = wasserstein_distance(generated_volumes, reference_volumes)
        
        return {
            'volume_mae': mae,
            'volume_relative_error': relative_error,
            'volume_emd': emd,
            'gen_volume_mean': generated_volumes.mean(),
            'gen_volume_std': generated_volumes.std(),
            'ref_volume_mean': reference_volumes.mean(),
            'ref_volume_std': reference_volumes.std()
        }
    
    @staticmethod
    def coordination_statistics(
        positions_list: List[np.ndarray],
        cell_list: List[np.ndarray],
        cutoff: float = 3.0
    ) -> Dict[str, float]:
        """
        Compute coordination number statistics.
        
        Args:
            positions_list: List of (n_atoms, 3) position arrays
            cell_list: List of (3, 3) cell arrays
            cutoff: Cutoff distance for coordination
            
        Returns:
            Dictionary with coordination statistics
        """
        all_coords = []
        
        for positions, cell in zip(positions_list, cell_list):
            n_atoms = positions.shape[0]
            coords = np.zeros(n_atoms)
            
            # Compute pairwise distances
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j:
                        # Simple distance (not considering PBC properly)
                        diff = positions[i] - positions[j]
                        dist = np.linalg.norm(diff)
                        if dist < cutoff:
                            coords[i] += 1
            
            all_coords.extend(coords)
        
        coords_array = np.array(all_coords)
        
        return {
            'coord_mean': coords_array.mean(),
            'coord_std': coords_array.std(),
            'coord_min': coords_array.min(),
            'coord_max': coords_array.max()
        }
    
    @staticmethod
    def validity_check(
        positions_list: List[np.ndarray],
        min_distance: float = 0.5
    ) -> Dict[str, float]:
        """
        Check validity of structures (no atoms too close).
        
        Args:
            positions_list: List of (n_atoms, 3) position arrays
            min_distance: Minimum allowed distance between atoms
            
        Returns:
            Dictionary with validity statistics
        """
        n_valid = 0
        n_total = len(positions_list)
        min_distances = []
        
        for positions in positions_list:
            # Compute pairwise distances
            dists = cdist(positions, positions)
            # Set diagonal to infinity
            np.fill_diagonal(dists, np.inf)
            # Get minimum distance
            min_dist = dists.min()
            min_distances.append(min_dist)
            
            # Check if valid
            if min_dist >= min_distance:
                n_valid += 1
        
        validity_rate = n_valid / n_total if n_total > 0 else 0.0
        
        return {
            'validity_rate': validity_rate,
            'n_valid': n_valid,
            'n_total': n_total,
            'min_distance_mean': np.mean(min_distances),
            'min_distance_min': np.min(min_distances)
        }
    
    @staticmethod
    def diversity_score(
        generated_params: np.ndarray
    ) -> float:
        """
        Compute diversity score based on lattice parameter variance.
        
        Args:
            generated_params: (n_gen, 6) array of lattice parameters
            
        Returns:
            Diversity score (higher = more diverse)
        """
        # Normalize each dimension
        normalized = (generated_params - generated_params.mean(axis=0)) / (generated_params.std(axis=0) + 1e-8)
        
        # Compute average pairwise distance
        dists = cdist(normalized, normalized)
        # Exclude diagonal
        mask = ~np.eye(dists.shape[0], dtype=bool)
        avg_dist = dists[mask].mean()
        
        return avg_dist
    
    @staticmethod
    def compute_all_metrics(
        generated_structures: List[Dict],
        reference_structures: List[Dict],
        cutoff: float = 3.0,
        min_distance: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            generated_structures: List of dicts with 'params', 'positions', 'cell', 'density', 'volume'
            reference_structures: List of dicts with same format
            cutoff: Cutoff for coordination
            min_distance: Minimum distance for validity
            
        Returns:
            Dictionary with all metrics
        """
        # Extract data
        gen_params = np.array([s['params'] for s in generated_structures])
        ref_params = np.array([s['params'] for s in reference_structures])
        
        gen_densities = np.array([s['density'] for s in generated_structures])
        ref_densities = np.array([s['density'] for s in reference_structures])
        
        gen_volumes = np.array([s['volume'] for s in generated_structures])
        ref_volumes = np.array([s['volume'] for s in reference_structures])
        
        gen_positions = [s['positions'] for s in generated_structures]
        gen_cells = [s['cell'] for s in generated_structures]
        
        # Compute metrics
        metrics = {}
        
        # Lattice parameters
        metrics.update(CrystalMetrics.lattice_parameter_mae(gen_params, ref_params))
        
        # Density
        metrics.update(CrystalMetrics.density_statistics(gen_densities, ref_densities))
        
        # Volume
        metrics.update(CrystalMetrics.volume_statistics(gen_volumes, ref_volumes))
        
        # Coordination
        metrics.update(CrystalMetrics.coordination_statistics(gen_positions, gen_cells, cutoff))
        
        # Validity
        metrics.update(CrystalMetrics.validity_check(gen_positions, min_distance))
        
        # Diversity
        metrics['diversity_score'] = CrystalMetrics.diversity_score(gen_params)
        
        return metrics
