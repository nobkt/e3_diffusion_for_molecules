"""
Neighbor List Construction for Periodic Systems

This module provides efficient neighbor list construction considering
periodic boundary conditions.

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation of all inputs
- Explicit error messages for invalid data
- Efficient computation using periodic images
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


class NeighborList:
    """
    Efficient neighbor list construction for periodic systems.
    
    Builds lists of atom pairs within a cutoff distance, accounting for
    periodic boundary conditions using the minimum image convention.
    """
    
    def __init__(
        self,
        cutoff: float,
        self_interaction: bool = False,
        strict_cutoff: bool = True
    ):
        """
        Initialize neighbor list builder.
        
        Args:
            cutoff: Cutoff distance in Angstroms
            self_interaction: Whether to include i-i pairs
            strict_cutoff: If True, enforce strict cutoff (d <= cutoff)
                          If False, use d < cutoff
                          
        Raises:
            ValueError: If cutoff is non-positive
        """
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        
        self.cutoff = cutoff
        self.self_interaction = self_interaction
        self.strict_cutoff = strict_cutoff
    
    def build(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        cell_vectors: Union[torch.Tensor, np.ndarray],
        pbc: Union[torch.Tensor, np.ndarray, bool] = True,
        batch_size: Optional[int] = None
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Build neighbor list for a crystal structure.
        
        Args:
            positions: [n_atoms, 3] or [batch, n_atoms, 3] atomic positions (Cartesian)
            cell_vectors: [3, 3] or [batch, 3, 3] unit cell vectors
            pbc: [3] or bool indicating which dimensions have periodic boundaries
            batch_size: If provided, treat first dimension as batch
            
        Returns:
            edge_index: [2, n_edges] indices of atom pairs (i, j)
            edge_shift: [n_edges, 3] cell shift vectors for each edge
            
        Raises:
            ValueError: If inputs have invalid shapes or cutoff is too large
        """
        # Type handling
        is_torch = isinstance(positions, torch.Tensor)
        if is_torch:
            device = positions.device
            dtype = positions.dtype
            positions_np = positions.detach().cpu().numpy()
            cell_np = cell_vectors.detach().cpu().numpy()
            if isinstance(pbc, torch.Tensor):
                pbc_np = pbc.detach().cpu().numpy()
            else:
                pbc_np = pbc
        else:
            positions_np = np.asarray(positions)
            cell_np = np.asarray(cell_vectors)
            pbc_np = pbc
        
        # Handle pbc
        if isinstance(pbc_np, bool):
            pbc_np = np.array([pbc_np, pbc_np, pbc_np])
        else:
            pbc_np = np.asarray(pbc_np, dtype=bool)
            if pbc_np.shape != (3,):
                raise ValueError(f"pbc must be shape [3], got {pbc_np.shape}")
        
        # Validate inputs
        if positions_np.ndim == 2:
            n_atoms = positions_np.shape[0]
            if positions_np.shape[1] != 3:
                raise ValueError(f"positions must have shape [n_atoms, 3], got {positions_np.shape}")
            if cell_np.shape != (3, 3):
                raise ValueError(f"cell_vectors must have shape [3, 3], got {cell_np.shape}")
            
            # Build neighbor list for single structure
            edge_index_np, edge_shift_np = self._build_single(
                positions_np, cell_np, pbc_np
            )
        elif positions_np.ndim == 3:
            # Batched input
            batch_size = positions_np.shape[0]
            n_atoms = positions_np.shape[1]
            
            if positions_np.shape[2] != 3:
                raise ValueError(
                    f"positions must have shape [batch, n_atoms, 3], got {positions_np.shape}"
                )
            if cell_np.shape != (batch_size, 3, 3):
                raise ValueError(
                    f"cell_vectors must have shape [batch, 3, 3], got {cell_np.shape}"
                )
            
            # Build for each structure in batch
            all_edge_indices = []
            all_edge_shifts = []
            
            for i in range(batch_size):
                edge_idx, edge_shf = self._build_single(
                    positions_np[i], cell_np[i], pbc_np
                )
                # Offset indices for batch
                edge_idx = edge_idx + i * n_atoms
                all_edge_indices.append(edge_idx)
                all_edge_shifts.append(edge_shf)
            
            edge_index_np = np.concatenate(all_edge_indices, axis=1)
            edge_shift_np = np.concatenate(all_edge_shifts, axis=0)
        else:
            raise ValueError(
                f"positions must have 2 or 3 dimensions, got {positions_np.ndim}"
            )
        
        # Convert back to torch if needed
        if is_torch:
            edge_index = torch.from_numpy(edge_index_np).to(device)
            edge_shift = torch.from_numpy(edge_shift_np).to(device=device, dtype=dtype)
        else:
            edge_index = edge_index_np
            edge_shift = edge_shift_np
        
        return edge_index, edge_shift
    
    def _build_single(
        self,
        positions: np.ndarray,
        cell_vectors: np.ndarray,
        pbc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build neighbor list for a single structure.
        
        Args:
            positions: [n_atoms, 3] atomic positions
            cell_vectors: [3, 3] unit cell vectors
            pbc: [3] periodic boundary conditions
            
        Returns:
            edge_index: [2, n_edges] edge indices
            edge_shift: [n_edges, 3] cell shift vectors
        """
        n_atoms = positions.shape[0]
        
        # Determine number of periodic images to check
        # Based on cutoff and cell dimensions
        cell_lengths = np.linalg.norm(cell_vectors, axis=1)
        n_images = np.ceil(self.cutoff / cell_lengths).astype(int)
        n_images = np.where(pbc, n_images, 0)  # Only for periodic dimensions
        
        # Check if cutoff is reasonable
        max_images = 10  # Arbitrary limit to prevent excessive computation
        if np.any(n_images > max_images):
            raise ValueError(
                f"Cutoff {self.cutoff} is too large relative to cell dimensions. "
                f"Would require checking {n_images} images in each direction. "
                f"Consider reducing cutoff or increasing cell size."
            )
        
        # Generate all periodic image shifts
        shifts_list = []
        for dim in range(3):
            if pbc[dim]:
                dim_shifts = np.arange(-n_images[dim], n_images[dim] + 1)
            else:
                dim_shifts = np.array([0])
            shifts_list.append(dim_shifts)
        
        # Create meshgrid of all shift combinations
        shift_x, shift_y, shift_z = np.meshgrid(*shifts_list, indexing='ij')
        all_shifts = np.stack([shift_x.ravel(), shift_y.ravel(), shift_z.ravel()], axis=1)
        
        # Build edge list
        edge_indices = []
        edge_shifts = []
        
        for shift in all_shifts:
            # Compute shifted positions
            shift_vector = shift @ cell_vectors
            positions_shifted = positions + shift_vector
            
            # Compute pairwise distances
            for i in range(n_atoms):
                for j in range(n_atoms):
                    # Skip self-interaction for central cell
                    if not self.self_interaction and np.all(shift == 0) and i == j:
                        continue
                    
                    # Compute distance
                    diff = positions_shifted[j] - positions[i]
                    distance = np.linalg.norm(diff)
                    
                    # Check cutoff
                    if self.strict_cutoff:
                        in_cutoff = distance <= self.cutoff
                    else:
                        in_cutoff = distance < self.cutoff
                    
                    if in_cutoff:
                        edge_indices.append([i, j])
                        edge_shifts.append(shift)
        
        if len(edge_indices) == 0:
            # No neighbors found
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_shift = np.zeros((0, 3), dtype=np.float32)
        else:
            edge_index = np.array(edge_indices, dtype=np.int64).T
            edge_shift = np.array(edge_shifts, dtype=np.float32)
        
        return edge_index, edge_shift
    
    def compute_distances(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        cell_vectors: Union[torch.Tensor, np.ndarray],
        edge_index: Union[torch.Tensor, np.ndarray],
        edge_shift: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute distances for edges in neighbor list.
        
        Args:
            positions: [n_atoms, 3] atomic positions
            cell_vectors: [3, 3] unit cell vectors
            edge_index: [2, n_edges] edge indices
            edge_shift: [n_edges, 3] cell shift vectors
            
        Returns:
            distances: [n_edges] distances between atom pairs
        """
        # Type handling
        is_torch = isinstance(positions, torch.Tensor)
        
        if is_torch:
            # Get source and target positions
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            
            src_pos = positions[src_idx]  # [n_edges, 3]
            dst_pos = positions[dst_idx]  # [n_edges, 3]
            
            # Apply periodic shift
            shift_vectors = torch.matmul(edge_shift, cell_vectors)  # [n_edges, 3]
            dst_pos_shifted = dst_pos + shift_vectors
            
            # Compute distances
            diff = dst_pos_shifted - src_pos
            distances = torch.norm(diff, dim=-1)
        else:
            positions = np.asarray(positions)
            cell_vectors = np.asarray(cell_vectors)
            edge_index = np.asarray(edge_index)
            edge_shift = np.asarray(edge_shift)
            
            # Get source and target positions
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            
            src_pos = positions[src_idx]
            dst_pos = positions[dst_idx]
            
            # Apply periodic shift
            shift_vectors = edge_shift @ cell_vectors
            dst_pos_shifted = dst_pos + shift_vectors
            
            # Compute distances
            diff = dst_pos_shifted - src_pos
            distances = np.linalg.norm(diff, axis=-1)
        
        return distances
    
    def compute_vectors(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        cell_vectors: Union[torch.Tensor, np.ndarray],
        edge_index: Union[torch.Tensor, np.ndarray],
        edge_shift: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute displacement vectors for edges in neighbor list.
        
        Args:
            positions: [n_atoms, 3] atomic positions
            cell_vectors: [3, 3] unit cell vectors
            edge_index: [2, n_edges] edge indices
            edge_shift: [n_edges, 3] cell shift vectors
            
        Returns:
            vectors: [n_edges, 3] displacement vectors from i to j
        """
        # Type handling
        is_torch = isinstance(positions, torch.Tensor)
        
        if is_torch:
            # Get source and target positions
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            
            src_pos = positions[src_idx]
            dst_pos = positions[dst_idx]
            
            # Apply periodic shift
            shift_vectors = torch.matmul(edge_shift, cell_vectors)
            dst_pos_shifted = dst_pos + shift_vectors
            
            # Compute vectors
            vectors = dst_pos_shifted - src_pos
        else:
            positions = np.asarray(positions)
            cell_vectors = np.asarray(cell_vectors)
            edge_index = np.asarray(edge_index)
            edge_shift = np.asarray(edge_shift)
            
            # Get source and target positions
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            
            src_pos = positions[src_idx]
            dst_pos = positions[dst_idx]
            
            # Apply periodic shift
            shift_vectors = edge_shift @ cell_vectors
            dst_pos_shifted = dst_pos + shift_vectors
            
            # Compute vectors
            vectors = dst_pos_shifted - src_pos
        
        return vectors
    
    def update_cutoff(self, new_cutoff: float) -> None:
        """
        Update cutoff distance.
        
        Args:
            new_cutoff: New cutoff distance in Angstroms
            
        Raises:
            ValueError: If new_cutoff is non-positive
        """
        if new_cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {new_cutoff}")
        
        self.cutoff = new_cutoff


def build_fully_connected_edges(
    n_atoms: int,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    self_interaction: bool = False
) -> torch.Tensor:
    """
    Build fully connected edge index (no periodic boundaries).
    
    Useful for small molecules or when not using neighbor lists.
    
    Args:
        n_atoms: Number of atoms
        batch_size: Optional batch size
        device: Target device for tensor
        self_interaction: Whether to include i-i edges
        
    Returns:
        edge_index: [2, n_edges] edge indices
        
    Raises:
        ValueError: If n_atoms is non-positive
    """
    if n_atoms <= 0:
        raise ValueError(f"n_atoms must be positive, got {n_atoms}")
    
    if batch_size is None:
        # Single structure
        rows = []
        cols = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if not self_interaction and i == j:
                    continue
                rows.append(i)
                cols.append(j)
        
        edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    else:
        # Batched structures
        all_rows = []
        all_cols = []
        for b in range(batch_size):
            offset = b * n_atoms
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if not self_interaction and i == j:
                        continue
                    all_rows.append(offset + i)
                    all_cols.append(offset + j)
        
        edge_index = torch.tensor([all_rows, all_cols], dtype=torch.long, device=device)
    
    return edge_index
