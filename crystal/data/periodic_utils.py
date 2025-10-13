"""
Periodic boundary condition utilities for crystal structures

This module provides functions for handling periodic boundary conditions,
coordinate transformations, and cell parameter manipulations.

Functions implement the minimum image convention for distance calculations
in periodic systems.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def minimum_image_distance(
    positions1: torch.Tensor,
    positions2: torch.Tensor,
    cell_vectors: torch.Tensor,
    pbc: torch.Tensor,
    use_fractional: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minimum image distance under periodic boundary conditions
    
    Args:
        positions1: [batch, n1, 3] or [n1, 3] - First set of positions
        positions2: [batch, n2, 3] or [n2, 3] - Second set of positions
        cell_vectors: [batch, 3, 3] or [3, 3] - Unit cell vectors
        pbc: [batch, 3] or [3] - Periodic boundary condition flags
        use_fractional: Whether positions are in fractional coordinates
        
    Returns:
        distances: [batch, n1, n2] or [n1, n2] - Pairwise distances
        vectors: [batch, n1, n2, 3] or [n1, n2, 3] - Displacement vectors
    """
    # Convert to fractional coordinates if needed
    if not use_fractional:
        frac1 = cartesian_to_fractional(positions1, cell_vectors)
        frac2 = cartesian_to_fractional(positions2, cell_vectors)
    else:
        frac1 = positions1
        frac2 = positions2
    
    # Handle batching
    if frac1.dim() == 2:
        frac1 = frac1.unsqueeze(0)
        frac2 = frac2.unsqueeze(0)
        cell_vectors = cell_vectors.unsqueeze(0)
        pbc = pbc.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    batch_size, n1, _ = frac1.shape
    n2 = frac2.shape[1]
    
    # Compute difference vectors [batch, n1, n2, 3]
    frac_diff = frac2.unsqueeze(1) - frac1.unsqueeze(2)
    
    # Apply periodic boundary conditions (-0.5 < diff <= 0.5)
    pbc_expanded = pbc.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, 3]
    
    # Only apply PBC where flag is True
    frac_diff_periodic = torch.where(
        pbc_expanded,
        frac_diff - torch.round(frac_diff),
        frac_diff
    )
    
    # Convert back to Cartesian coordinates
    # vectors = cell_vectors @ frac_diff_periodic
    vectors = torch.einsum('bij,bmnj->bmni', cell_vectors, frac_diff_periodic)
    
    # Compute distances
    distances = torch.norm(vectors, dim=-1)
    
    if squeeze_batch:
        distances = distances.squeeze(0)
        vectors = vectors.squeeze(0)
    
    return distances, vectors


def cartesian_to_fractional(
    positions_cart: torch.Tensor,
    cell_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert Cartesian coordinates to fractional coordinates
    
    Args:
        positions_cart: [..., n_atoms, 3] or [..., 3] - Cartesian coordinates
        cell_vectors: [..., 3, 3] - Unit cell vectors
        
    Returns:
        positions_frac: [..., n_atoms, 3] or [..., 3] - Fractional coordinates
    """
    # cell^(-1) @ positions_cart^T = positions_frac^T
    # So: positions_frac = positions_cart @ cell^(-T)
    cell_inv = torch.linalg.inv(cell_vectors)
    
    # positions_cart [..., n_atoms, 3] @ cell_inv [..., 3, 3]^T
    # Result: [..., n_atoms, 3]
    positions_frac = torch.matmul(positions_cart, cell_inv.transpose(-2, -1))
    
    return positions_frac


def fractional_to_cartesian(
    positions_frac: torch.Tensor,
    cell_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert fractional coordinates to Cartesian coordinates
    
    Args:
        positions_frac: [..., n_atoms, 3] or [..., 3] - Fractional coordinates
        cell_vectors: [..., 3, 3] - Unit cell vectors
        
    Returns:
        positions_cart: [..., n_atoms, 3] or [..., 3] - Cartesian coordinates
    """
    # cell @ positions_frac^T = positions_cart^T
    # So: positions_cart = positions_frac @ cell^T
    
    # positions_frac [..., n_atoms, 3] @ cell_vectors [..., 3, 3]^T
    # Result: [..., n_atoms, 3]
    positions_cart = torch.matmul(positions_frac, cell_vectors.transpose(-2, -1))
    
    return positions_cart


def wrap_positions(
    positions_frac: torch.Tensor,
    pbc: torch.Tensor
) -> torch.Tensor:
    """
    Wrap fractional coordinates to unit cell (0-1)
    
    Args:
        positions_frac: [..., 3] - Fractional coordinates
        pbc: [..., 3] - Periodic boundary condition flags
        
    Returns:
        wrapped_positions: [..., 3] - Wrapped fractional coordinates
    """
    # Only wrap where PBC is True
    if pbc.dim() < positions_frac.dim():
        pbc = pbc.unsqueeze(-2)
    
    wrapped = torch.where(
        pbc,
        positions_frac % 1.0,  # Wrap to 0-1 range
        positions_frac
    )
    
    return wrapped


def compute_cell_volume(cell_vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute unit cell volume
    
    Args:
        cell_vectors: [..., 3, 3] - Unit cell vectors
        
    Returns:
        volume: [...] - Cell volume
    """
    # Volume = |a · (b × c)|
    a = cell_vectors[..., 0, :]
    b = cell_vectors[..., 1, :]
    c = cell_vectors[..., 2, :]
    
    cross_bc = torch.cross(b, c, dim=-1)
    volume = torch.abs(torch.sum(a * cross_bc, dim=-1))
    
    return volume


def cell_params_to_vectors(cell_params: torch.Tensor) -> torch.Tensor:
    """
    Convert cell parameters (a, b, c, α, β, γ) to cell vectors
    
    Args:
        cell_params: [..., 6] - (a, b, c, α, β, γ) with angles in degrees
        
    Returns:
        cell_vectors: [..., 3, 3] - Unit cell vectors
    """
    a, b, c = cell_params[..., 0], cell_params[..., 1], cell_params[..., 2]
    alpha, beta, gamma = cell_params[..., 3], cell_params[..., 4], cell_params[..., 5]
    
    # Convert degrees to radians
    alpha_rad = torch.deg2rad(alpha)
    beta_rad = torch.deg2rad(beta)
    gamma_rad = torch.deg2rad(gamma)
    
    # Compute trigonometric values
    cos_alpha = torch.cos(alpha_rad)
    cos_beta = torch.cos(beta_rad)
    cos_gamma = torch.cos(gamma_rad)
    sin_gamma = torch.sin(gamma_rad)
    
    # a vector along x-axis
    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)
    
    # b vector in xy-plane
    bx = b * cos_gamma
    by = b * sin_gamma
    bz = torch.zeros_like(b)
    
    # c vector
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt(c**2 - cx**2 - cy**2 + 1e-10)  # Add small value for numerical stability
    
    # Stack into cell_vectors tensor
    shape = list(cell_params.shape[:-1]) + [3, 3]
    cell_vectors = torch.zeros(shape, device=cell_params.device, dtype=cell_params.dtype)
    
    cell_vectors[..., 0, 0] = ax
    cell_vectors[..., 0, 1] = ay
    cell_vectors[..., 0, 2] = az
    
    cell_vectors[..., 1, 0] = bx
    cell_vectors[..., 1, 1] = by
    cell_vectors[..., 1, 2] = bz
    
    cell_vectors[..., 2, 0] = cx
    cell_vectors[..., 2, 1] = cy
    cell_vectors[..., 2, 2] = cz
    
    return cell_vectors


def cell_vectors_to_params(cell_vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert cell vectors to cell parameters (a, b, c, α, β, γ)
    
    Args:
        cell_vectors: [..., 3, 3] - Unit cell vectors
        
    Returns:
        cell_params: [..., 6] - (a, b, c, α, β, γ) with angles in degrees
    """
    a_vec = cell_vectors[..., 0, :]
    b_vec = cell_vectors[..., 1, :]
    c_vec = cell_vectors[..., 2, :]
    
    # Compute lengths
    a = torch.norm(a_vec, dim=-1)
    b = torch.norm(b_vec, dim=-1)
    c = torch.norm(c_vec, dim=-1)
    
    # Compute angles (in radians)
    cos_alpha = torch.sum(b_vec * c_vec, dim=-1) / (b * c + 1e-10)
    cos_beta = torch.sum(a_vec * c_vec, dim=-1) / (a * c + 1e-10)
    cos_gamma = torch.sum(a_vec * b_vec, dim=-1) / (a * b + 1e-10)
    
    # Clamp to avoid numerical errors in acos
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
    cos_gamma = torch.clamp(cos_gamma, -1.0, 1.0)
    
    # Convert to degrees
    alpha = torch.rad2deg(torch.acos(cos_alpha))
    beta = torch.rad2deg(torch.acos(cos_beta))
    gamma = torch.rad2deg(torch.acos(cos_gamma))
    
    # Stack into cell_params tensor
    cell_params = torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
    
    return cell_params


def build_neighbor_list(
    positions: torch.Tensor,
    cell_vectors: torch.Tensor,
    pbc: torch.Tensor,
    cutoff_radius: float,
    use_fractional: bool = True,
    max_neighbors: Optional[int] = None
) -> Tuple[list, list, list]:
    """
    Build neighbor list under periodic boundary conditions
    
    Args:
        positions: [batch, n_atoms, 3] - Atomic positions
        cell_vectors: [batch, 3, 3] - Unit cell vectors
        pbc: [batch, 3] - Periodic boundary condition flags
        cutoff_radius: Cutoff radius in Angstroms
        use_fractional: Whether positions are in fractional coordinates
        max_neighbors: Maximum neighbors per atom (for memory efficiency)
        
    Returns:
        edge_index_list: List of [2, n_edges] tensors for each batch
        edge_attr_list: List of [n_edges, 3] edge vectors for each batch
        edge_dist_list: List of [n_edges] edge distances for each batch
    """
    batch_size, n_atoms, _ = positions.shape
    
    # Compute all pairwise distances
    distances, vectors = minimum_image_distance(
        positions, positions, cell_vectors, pbc, use_fractional
    )
    
    # Exclude self-loops
    mask_diag = torch.eye(n_atoms, device=distances.device, dtype=torch.bool)
    distances = distances.masked_fill(mask_diag.unsqueeze(0), float('inf'))
    
    # Select edges within cutoff
    edge_mask = distances < cutoff_radius  # [batch, n_atoms, n_atoms]
    
    # Process each batch separately
    edge_index_list = []
    edge_attr_list = []
    edge_dist_list = []
    
    for b in range(batch_size):
        # Get edge indices
        src, dst = torch.where(edge_mask[b])
        
        # Apply max_neighbors constraint if specified
        if max_neighbors is not None and len(src) > n_atoms * max_neighbors:
            # Sort by distance and keep closest neighbors
            edge_distances = distances[b, src, dst]
            sorted_indices = torch.argsort(edge_distances)
            
            # Keep max_neighbors closest for each atom
            keep_mask = torch.zeros(len(src), dtype=torch.bool, device=src.device)
            for atom_idx in range(n_atoms):
                atom_edges = src == atom_idx
                if atom_edges.sum() > max_neighbors:
                    atom_edge_indices = torch.where(atom_edges)[0]
                    sorted_atom_indices = sorted_indices[
                        torch.isin(sorted_indices, atom_edge_indices)
                    ][:max_neighbors]
                    keep_mask[sorted_atom_indices] = True
                else:
                    keep_mask[atom_edges] = True
            
            src = src[keep_mask]
            dst = dst[keep_mask]
        
        edge_index = torch.stack([src, dst], dim=0)  # [2, n_edges]
        edge_attr = vectors[b, src, dst]  # [n_edges, 3]
        edge_dist = distances[b, src, dst]  # [n_edges]
        
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        edge_dist_list.append(edge_dist)
    
    return edge_index_list, edge_attr_list, edge_dist_list
