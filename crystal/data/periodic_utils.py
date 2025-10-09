"""
Periodic boundary condition utilities for crystal structure generation.

This module provides functions for handling periodic boundary conditions,
coordinate transformations, and distance calculations in crystal systems.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def minimum_image_distance(
    positions1: torch.Tensor,
    positions2: torch.Tensor,
    cell_vectors: torch.Tensor,
    pbc: torch.Tensor = None,
    use_fractional: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minimum image distance between atoms considering periodic boundary conditions.
    
    Args:
        positions1: Shape (..., N, 3) - First set of positions
        positions2: Shape (..., M, 3) - Second set of positions
        cell_vectors: Shape (..., 3, 3) - Unit cell vectors as rows
        pbc: Shape (3,) - Periodic boundary conditions for each dimension
        use_fractional: If True, positions are in fractional coordinates
        
    Returns:
        distances: Shape (..., N, M) - Minimum image distances
        vectors: Shape (..., N, M, 3) - Minimum image displacement vectors
    """
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=positions1.device)
    
    # Convert to fractional coordinates if needed
    if not use_fractional:
        frac1 = cartesian_to_fractional(positions1, cell_vectors)
        frac2 = cartesian_to_fractional(positions2, cell_vectors)
    else:
        frac1 = positions1
        frac2 = positions2
    
    # Compute fractional displacement
    # Shape: (..., N, 1, 3) - (..., 1, M, 3) = (..., N, M, 3)
    frac_disp = frac1.unsqueeze(-2) - frac2.unsqueeze(-3)
    
    # Apply minimum image convention in fractional coordinates
    for i in range(3):
        if pbc[i]:
            frac_disp[..., i] = frac_disp[..., i] - torch.round(frac_disp[..., i])
    
    # Convert back to Cartesian coordinates
    vectors = fractional_to_cartesian(frac_disp, cell_vectors)
    
    # Compute distances
    distances = torch.norm(vectors, dim=-1)
    
    return distances, vectors


def cartesian_to_fractional(
    positions: torch.Tensor,
    cell_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert Cartesian coordinates to fractional coordinates.
    
    Args:
        positions: Shape (..., N, 3) - Cartesian coordinates
        cell_vectors: Shape (..., 3, 3) - Unit cell vectors as rows
        
    Returns:
        fractional: Shape (..., N, 3) - Fractional coordinates
    """
    # cell_vectors: (3, 3) or (..., 3, 3)
    # positions: (..., N, 3)
    
    # Compute inverse of cell matrix
    cell_inv = torch.linalg.inv(cell_vectors)  # (..., 3, 3)
    
    # Apply transformation: frac = pos @ cell_inv.T
    fractional = torch.matmul(positions, cell_inv.transpose(-2, -1))
    
    return fractional


def fractional_to_cartesian(
    fractional: torch.Tensor,
    cell_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert fractional coordinates to Cartesian coordinates.
    
    Args:
        fractional: Shape (..., N, 3) - Fractional coordinates
        cell_vectors: Shape (..., 3, 3) - Unit cell vectors as rows
        
    Returns:
        positions: Shape (..., N, 3) - Cartesian coordinates
    """
    # Apply transformation: pos = frac @ cell
    positions = torch.matmul(fractional, cell_vectors)
    
    return positions


def cell_params_to_vectors(
    lengths: torch.Tensor,
    angles: torch.Tensor
) -> torch.Tensor:
    """
    Convert cell parameters (a, b, c, alpha, beta, gamma) to cell vectors.
    
    Args:
        lengths: Shape (..., 3) - Cell lengths [a, b, c] in Angstroms
        angles: Shape (..., 3) - Cell angles [alpha, beta, gamma] in degrees
        
    Returns:
        cell_vectors: Shape (..., 3, 3) - Cell vectors as rows
    """
    # Convert angles to radians
    angles_rad = angles * torch.pi / 180.0
    
    alpha = angles_rad[..., 0]
    beta = angles_rad[..., 1]
    gamma = angles_rad[..., 2]
    
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    
    # Compute cell vectors following standard crystallographic convention
    # a vector along x-axis
    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)
    
    # b vector in xy-plane
    bx = b * torch.cos(gamma)
    by = b * torch.sin(gamma)
    bz = torch.zeros_like(b)
    
    # c vector
    cx = c * torch.cos(beta)
    cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
    cz = torch.sqrt(c**2 - cx**2 - cy**2)
    
    # Stack into matrix
    cell_vectors = torch.stack([
        torch.stack([ax, ay, az], dim=-1),
        torch.stack([bx, by, bz], dim=-1),
        torch.stack([cx, cy, cz], dim=-1)
    ], dim=-2)
    
    return cell_vectors


def cell_vectors_to_params(
    cell_vectors: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert cell vectors to cell parameters (a, b, c, alpha, beta, gamma).
    
    Args:
        cell_vectors: Shape (..., 3, 3) - Cell vectors as rows
        
    Returns:
        lengths: Shape (..., 3) - Cell lengths [a, b, c] in Angstroms
        angles: Shape (..., 3) - Cell angles [alpha, beta, gamma] in degrees
    """
    # Extract vectors
    a_vec = cell_vectors[..., 0, :]
    b_vec = cell_vectors[..., 1, :]
    c_vec = cell_vectors[..., 2, :]
    
    # Compute lengths
    a = torch.norm(a_vec, dim=-1)
    b = torch.norm(b_vec, dim=-1)
    c = torch.norm(c_vec, dim=-1)
    lengths = torch.stack([a, b, c], dim=-1)
    
    # Compute angles
    cos_alpha = torch.sum(b_vec * c_vec, dim=-1) / (b * c)
    cos_beta = torch.sum(a_vec * c_vec, dim=-1) / (a * c)
    cos_gamma = torch.sum(a_vec * b_vec, dim=-1) / (a * b)
    
    # Clamp to avoid numerical issues
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
    cos_gamma = torch.clamp(cos_gamma, -1.0, 1.0)
    
    alpha = torch.acos(cos_alpha) * 180.0 / torch.pi
    beta = torch.acos(cos_beta) * 180.0 / torch.pi
    gamma = torch.acos(cos_gamma) * 180.0 / torch.pi
    angles = torch.stack([alpha, beta, gamma], dim=-1)
    
    return lengths, angles


def wrap_to_unit_cell(
    fractional: torch.Tensor,
    pbc: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Wrap fractional coordinates to [0, 1) range.
    
    Args:
        fractional: Shape (..., N, 3) - Fractional coordinates
        pbc: Shape (3,) - Periodic boundary conditions for each dimension
        
    Returns:
        wrapped: Shape (..., N, 3) - Wrapped fractional coordinates
    """
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=fractional.device)
    
    wrapped = fractional.clone()
    for i in range(3):
        if pbc[i]:
            wrapped[..., i] = wrapped[..., i] - torch.floor(wrapped[..., i])
    
    return wrapped


def build_periodic_neighbor_list(
    positions: torch.Tensor,
    cell_vectors: torch.Tensor,
    cutoff: float,
    pbc: Optional[torch.Tensor] = None,
    use_fractional: bool = False,
    self_interaction: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build neighbor list considering periodic boundary conditions.
    
    Args:
        positions: Shape (N, 3) - Atomic positions
        cell_vectors: Shape (3, 3) - Unit cell vectors as rows
        cutoff: Cutoff radius for neighbors
        pbc: Shape (3,) - Periodic boundary conditions
        use_fractional: If True, positions are in fractional coordinates
        self_interaction: If True, include self-interaction (i==j)
        
    Returns:
        edge_index: Shape (2, E) - Edge indices [source, target]
        edge_vectors: Shape (E, 3) - Edge displacement vectors
        edge_distances: Shape (E,) - Edge distances
    """
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=positions.device)
    
    N = positions.shape[0]
    
    # Compute all pairwise distances
    distances, vectors = minimum_image_distance(
        positions, positions, cell_vectors, pbc, use_fractional
    )
    
    # Create mask for neighbors within cutoff
    mask = distances < cutoff
    
    if not self_interaction:
        # Remove self-interactions
        eye_mask = ~torch.eye(N, dtype=torch.bool, device=positions.device)
        mask = mask & eye_mask
    
    # Get edge indices
    edge_index = torch.nonzero(mask, as_tuple=False).t()  # (2, E)
    
    # Get corresponding vectors and distances
    edge_vectors = vectors[mask]  # (E, 3)
    edge_distances = distances[mask]  # (E,)
    
    return edge_index, edge_vectors, edge_distances
