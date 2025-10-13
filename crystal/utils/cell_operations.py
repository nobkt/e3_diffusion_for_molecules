"""
Cell Operations for Crystal Structures

This module provides operations for manipulating unit cells, including
standardization, reduction, and transformation.

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation of all inputs
- Explicit error messages for invalid data
- Theoretically sound crystallographic operations
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


class CellOperations:
    """
    Operations for crystal unit cells.
    
    Provides methods for cell transformations, standardization, reduction,
    and geometric calculations.
    """
    
    @staticmethod
    def cell_params_to_vectors(
        cell_params: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert cell parameters to cell vectors.
        
        Args:
            cell_params: [..., 6] array (a, b, c, alpha, beta, gamma)
                        Lengths in Angstroms, angles in degrees
                        
        Returns:
            cell_vectors: [..., 3, 3] array of cell vectors
            
        Raises:
            ValueError: If cell_params has invalid shape or values
        """
        # Type handling
        is_torch = isinstance(cell_params, torch.Tensor)
        if is_torch:
            device = cell_params.device
            dtype = cell_params.dtype
            cell_params_np = cell_params.detach().cpu().numpy()
        else:
            cell_params_np = np.asarray(cell_params)
        
        # Validate shape
        if cell_params_np.shape[-1] != 6:
            raise ValueError(
                f"cell_params must have last dimension 6, got shape {cell_params_np.shape}"
            )
        
        # Extract parameters
        original_shape = cell_params_np.shape[:-1]
        cell_params_flat = cell_params_np.reshape(-1, 6)
        
        a = cell_params_flat[:, 0]
        b = cell_params_flat[:, 1]
        c = cell_params_flat[:, 2]
        alpha = np.deg2rad(cell_params_flat[:, 3])
        beta = np.deg2rad(cell_params_flat[:, 4])
        gamma = np.deg2rad(cell_params_flat[:, 5])
        
        # Validate values
        if np.any(a <= 0) or np.any(b <= 0) or np.any(c <= 0):
            raise ValueError("Cell lengths must be positive")
        
        if np.any(alpha <= 0) or np.any(alpha >= np.pi):
            raise ValueError("alpha must be in (0, 180) degrees")
        if np.any(beta <= 0) or np.any(beta >= np.pi):
            raise ValueError("beta must be in (0, 180) degrees")
        if np.any(gamma <= 0) or np.any(gamma >= np.pi):
            raise ValueError("gamma must be in (0, 180) degrees")
        
        # Compute cell vectors
        # a vector along x-axis
        ax = a
        ay = np.zeros_like(a)
        az = np.zeros_like(a)
        
        # b vector in xy-plane
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = np.zeros_like(b)
        
        # c vector
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz_sq = c**2 - cx**2 - cy**2
        
        # Check for valid cell (positive volume)
        if np.any(cz_sq < 0):
            raise ValueError("Invalid cell parameters: would result in negative volume")
        
        cz = np.sqrt(cz_sq)
        
        # Stack into cell vectors
        cell_vectors_flat = np.stack([
            np.stack([ax, ay, az], axis=-1),
            np.stack([bx, by, bz], axis=-1),
            np.stack([cx, cy, cz], axis=-1)
        ], axis=-2)
        
        # Reshape to original batch shape
        cell_vectors = cell_vectors_flat.reshape(*original_shape, 3, 3)
        
        if is_torch:
            cell_vectors = torch.from_numpy(cell_vectors).to(device=device, dtype=dtype)
        
        return cell_vectors
    
    @staticmethod
    def cell_vectors_to_params(
        cell_vectors: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert cell vectors to cell parameters.
        
        Args:
            cell_vectors: [..., 3, 3] array of cell vectors
            
        Returns:
            cell_params: [..., 6] array (a, b, c, alpha, beta, gamma)
                        Lengths in Angstroms, angles in degrees
                        
        Raises:
            ValueError: If cell_vectors has invalid shape
        """
        # Type handling
        is_torch = isinstance(cell_vectors, torch.Tensor)
        if is_torch:
            device = cell_vectors.device
            dtype = cell_vectors.dtype
            cell_vectors_np = cell_vectors.detach().cpu().numpy()
        else:
            cell_vectors_np = np.asarray(cell_vectors)
        
        # Validate shape
        if cell_vectors_np.shape[-2:] != (3, 3):
            raise ValueError(
                f"cell_vectors must have last two dimensions (3, 3), "
                f"got shape {cell_vectors_np.shape}"
            )
        
        # Extract vectors
        original_shape = cell_vectors_np.shape[:-2]
        cell_vectors_flat = cell_vectors_np.reshape(-1, 3, 3)
        
        a_vec = cell_vectors_flat[:, 0, :]
        b_vec = cell_vectors_flat[:, 1, :]
        c_vec = cell_vectors_flat[:, 2, :]
        
        # Compute lengths
        a = np.linalg.norm(a_vec, axis=-1)
        b = np.linalg.norm(b_vec, axis=-1)
        c = np.linalg.norm(c_vec, axis=-1)
        
        # Check for zero-length vectors
        if np.any(a < 1e-8) or np.any(b < 1e-8) or np.any(c < 1e-8):
            raise ValueError("Cell vectors must have non-zero length")
        
        # Compute angles
        cos_alpha = np.einsum('bi,bi->b', b_vec, c_vec) / (b * c)
        cos_beta = np.einsum('bi,bi->b', a_vec, c_vec) / (a * c)
        cos_gamma = np.einsum('bi,bi->b', a_vec, b_vec) / (a * b)
        
        # Clip to valid range to avoid numerical issues
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        
        alpha = np.rad2deg(np.arccos(cos_alpha))
        beta = np.rad2deg(np.arccos(cos_beta))
        gamma = np.rad2deg(np.arccos(cos_gamma))
        
        # Stack into cell params
        cell_params_flat = np.stack([a, b, c, alpha, beta, gamma], axis=-1)
        
        # Reshape to original batch shape
        cell_params = cell_params_flat.reshape(*original_shape, 6)
        
        if is_torch:
            cell_params = torch.from_numpy(cell_params).to(device=device, dtype=dtype)
        
        return cell_params
    
    @staticmethod
    def compute_volume(
        cell_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
        cell_vectors: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute unit cell volume.
        
        Args:
            cell_params: [..., 6] cell parameters (a, b, c, alpha, beta, gamma)
            cell_vectors: [..., 3, 3] cell vectors (alternative to cell_params)
            
        Returns:
            volume: [...] cell volume in Angstrom^3
            
        Raises:
            ValueError: If neither or both inputs are provided
        """
        if cell_params is None and cell_vectors is None:
            raise ValueError("Must provide either cell_params or cell_vectors")
        if cell_params is not None and cell_vectors is not None:
            raise ValueError("Cannot provide both cell_params and cell_vectors")
        
        # Convert to vectors if needed
        if cell_params is not None:
            cell_vectors = CellOperations.cell_params_to_vectors(cell_params)
        
        # Type handling
        is_torch = isinstance(cell_vectors, torch.Tensor)
        if is_torch:
            # Volume = |det(cell_vectors)|
            # For batched input, compute determinant for each matrix
            volume = torch.abs(torch.det(cell_vectors))
        else:
            cell_vectors_np = np.asarray(cell_vectors)
            volume = np.abs(np.linalg.det(cell_vectors_np))
        
        return volume
    
    @staticmethod
    def standardize_cell(
        cell_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        tol: float = 1e-5
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Standardize unit cell to conventional form.
        
        Applies transformation to put cell in standard orientation:
        - a along x-axis
        - b in xy-plane with positive y component
        - c with positive z component
        
        Args:
            cell_vectors: [3, 3] cell vectors
            positions: [n_atoms, 3] atomic positions (fractional coordinates)
            tol: Tolerance for numerical operations
            
        Returns:
            Tuple of (standardized_cell_vectors, transformed_positions)
            
        Raises:
            ValueError: If inputs have invalid shapes or dimensions
        """
        # Type handling
        is_torch = isinstance(cell_vectors, torch.Tensor)
        if is_torch:
            device = cell_vectors.device
            dtype = cell_vectors.dtype
            cell_np = cell_vectors.detach().cpu().numpy()
            pos_np = positions.detach().cpu().numpy()
        else:
            cell_np = np.asarray(cell_vectors)
            pos_np = np.asarray(positions)
        
        # Validate shapes
        if cell_np.shape != (3, 3):
            raise ValueError(f"cell_vectors must be [3, 3], got {cell_np.shape}")
        if pos_np.shape[-1] != 3:
            raise ValueError(f"positions must have last dim 3, got {pos_np.shape}")
        
        # Convert to cell parameters and back to get standard form
        cell_params = CellOperations.cell_vectors_to_params(cell_np)
        cell_standard = CellOperations.cell_params_to_vectors(cell_params)
        
        # The positions remain the same in fractional coordinates
        # (standardization only changes the cell vectors)
        
        if is_torch:
            cell_standard = torch.from_numpy(cell_standard).to(device=device, dtype=dtype)
            pos_standard = positions  # Positions unchanged in fractional coords
        else:
            pos_standard = pos_np
        
        return cell_standard, pos_standard
    
    @staticmethod
    def reduce_cell(
        cell_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        method: str = 'niggli'
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Reduce cell to a canonical form (e.g., Niggli reduced cell).
        
        Note: Full Niggli reduction requires iterative algorithm.
        This is a simplified version for demonstration.
        
        Args:
            cell_vectors: [3, 3] cell vectors
            positions: [n_atoms, 3] atomic positions (fractional coordinates)
            method: Reduction method ('niggli' or 'standard')
            
        Returns:
            Tuple of (reduced_cell_vectors, transformed_positions)
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in ['niggli', 'standard']:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        if method == 'standard':
            # Use standardization as simplified reduction
            return CellOperations.standardize_cell(cell_vectors, positions)
        else:
            # Niggli reduction requires external library (spglib)
            # For now, fall back to standardization
            # Note: This is NOT a fallback heuristic - it's an explicit limitation
            raise NotImplementedError(
                "Full Niggli reduction requires spglib. "
                "Use method='standard' for basic standardization."
            )
    
    @staticmethod
    def transform_cell(
        cell_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        transformation_matrix: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Apply linear transformation to cell and positions.
        
        Args:
            cell_vectors: [3, 3] cell vectors
            positions: [n_atoms, 3] atomic positions (fractional coordinates)
            transformation_matrix: [3, 3] transformation matrix
            
        Returns:
            Tuple of (transformed_cell_vectors, transformed_positions)
            
        Raises:
            ValueError: If transformation matrix is singular
        """
        # Type handling
        is_torch = isinstance(cell_vectors, torch.Tensor)
        if is_torch:
            device = cell_vectors.device
            dtype = cell_vectors.dtype
        
        # Validate transformation matrix
        if is_torch:
            det = torch.det(transformation_matrix)
            if torch.abs(det) < 1e-8:
                raise ValueError("Transformation matrix is singular")
        else:
            transformation_matrix = np.asarray(transformation_matrix)
            det = np.linalg.det(transformation_matrix)
            if abs(det) < 1e-8:
                raise ValueError("Transformation matrix is singular")
        
        # Transform cell vectors: new_cell = transformation @ cell
        if is_torch:
            new_cell = torch.matmul(transformation_matrix, cell_vectors)
            # Transform positions: new_pos = pos @ transformation^(-1)
            transform_inv = torch.inverse(transformation_matrix)
            new_positions = torch.matmul(positions, transform_inv)
        else:
            new_cell = np.matmul(transformation_matrix, cell_vectors)
            transform_inv = np.linalg.inv(transformation_matrix)
            new_positions = np.matmul(positions, transform_inv)
        
        return new_cell, new_positions
    
    @staticmethod
    def validate_cell(
        cell_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
        cell_vectors: Optional[Union[torch.Tensor, np.ndarray]] = None,
        length_bounds: Tuple[float, float] = (1.0, 100.0),
        angle_bounds: Tuple[float, float] = (30.0, 150.0),
        volume_min: float = 1.0
    ) -> bool:
        """
        Validate that cell parameters are physically reasonable.
        
        Args:
            cell_params: [..., 6] cell parameters
            cell_vectors: [..., 3, 3] cell vectors (alternative to cell_params)
            length_bounds: (min, max) bounds for cell lengths in Angstroms
            angle_bounds: (min, max) bounds for cell angles in degrees
            volume_min: Minimum cell volume in Angstrom^3
            
        Returns:
            True if cell is valid, False otherwise
            
        Raises:
            ValueError: If neither or both inputs are provided
        """
        if cell_params is None and cell_vectors is None:
            raise ValueError("Must provide either cell_params or cell_vectors")
        if cell_params is not None and cell_vectors is not None:
            raise ValueError("Cannot provide both cell_params and cell_vectors")
        
        # Convert to params if needed
        if cell_vectors is not None:
            cell_params = CellOperations.cell_vectors_to_params(cell_vectors)
        
        # Type handling
        is_torch = isinstance(cell_params, torch.Tensor)
        if is_torch:
            cell_params = cell_params.detach().cpu().numpy()
        else:
            cell_params = np.asarray(cell_params)
        
        # Extract parameters
        a, b, c = cell_params[..., 0], cell_params[..., 1], cell_params[..., 2]
        alpha, beta, gamma = cell_params[..., 3], cell_params[..., 4], cell_params[..., 5]
        
        # Check length bounds
        if np.any(a < length_bounds[0]) or np.any(a > length_bounds[1]):
            return False
        if np.any(b < length_bounds[0]) or np.any(b > length_bounds[1]):
            return False
        if np.any(c < length_bounds[0]) or np.any(c > length_bounds[1]):
            return False
        
        # Check angle bounds
        if np.any(alpha < angle_bounds[0]) or np.any(alpha > angle_bounds[1]):
            return False
        if np.any(beta < angle_bounds[0]) or np.any(beta > angle_bounds[1]):
            return False
        if np.any(gamma < angle_bounds[0]) or np.any(gamma > angle_bounds[1]):
            return False
        
        # Check volume
        volume = CellOperations.compute_volume(cell_params=cell_params)
        if is_torch:
            volume = volume.detach().cpu().numpy()
        
        if np.any(volume < volume_min):
            return False
        
        return True
    
    @staticmethod
    def get_reciprocal_cell(
        cell_vectors: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute reciprocal lattice vectors.
        
        Args:
            cell_vectors: [..., 3, 3] real space cell vectors
            
        Returns:
            reciprocal_vectors: [..., 3, 3] reciprocal space cell vectors
            
        Note:
            Reciprocal vectors satisfy: a_i · b*_j = 2π δ_ij
        """
        # Type handling
        is_torch = isinstance(cell_vectors, torch.Tensor)
        
        # Compute reciprocal lattice
        # b* = 2π (b × c) / V, where V = a · (b × c)
        if is_torch:
            # For batched computation
            volume = CellOperations.compute_volume(cell_vectors=cell_vectors)
            
            # Add batch dimensions if needed
            if cell_vectors.dim() == 2:
                cell_vectors = cell_vectors.unsqueeze(0)
                volume = volume.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            a = cell_vectors[..., 0, :]
            b = cell_vectors[..., 1, :]
            c = cell_vectors[..., 2, :]
            
            # Reciprocal vectors
            b_star = 2 * np.pi * torch.cross(b, c, dim=-1) / volume.unsqueeze(-1)
            c_star = 2 * np.pi * torch.cross(c, a, dim=-1) / volume.unsqueeze(-1)
            a_star = 2 * np.pi * torch.cross(a, b, dim=-1) / volume.unsqueeze(-1)
            
            reciprocal = torch.stack([a_star, b_star, c_star], dim=-2)
            
            if squeeze_output:
                reciprocal = reciprocal.squeeze(0)
        else:
            cell_vectors = np.asarray(cell_vectors)
            volume = CellOperations.compute_volume(cell_vectors=cell_vectors)
            
            if cell_vectors.ndim == 2:
                cell_vectors = cell_vectors[np.newaxis]
                volume = np.array([volume])
                squeeze_output = True
            else:
                squeeze_output = False
            
            a = cell_vectors[..., 0, :]
            b = cell_vectors[..., 1, :]
            c = cell_vectors[..., 2, :]
            
            # Reciprocal vectors
            b_star = 2 * np.pi * np.cross(b, c, axis=-1) / volume[..., np.newaxis]
            c_star = 2 * np.pi * np.cross(c, a, axis=-1) / volume[..., np.newaxis]
            a_star = 2 * np.pi * np.cross(a, b, axis=-1) / volume[..., np.newaxis]
            
            reciprocal = np.stack([a_star, b_star, c_star], axis=-2)
            
            if squeeze_output:
                reciprocal = reciprocal[0]
        
        return reciprocal
