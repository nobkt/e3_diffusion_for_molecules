"""
Crystal dynamics model integrating atomic positions and lattice parameters.

This module combines the periodic EGNN for atomic coordinates with
lattice parameter diffusion for complete crystal structure generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .periodic_egnn import PeriodicEGNN
from .lattice_diffusion import LatticeDiffusion
from ..data.periodic_utils import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    cell_params_to_vectors,
    cell_vectors_to_params,
)


class CrystalDynamics(nn.Module):
    """
    Integrated model for crystal structure dynamics.
    
    This model handles both:
    1. Atomic positions (via Periodic EGNN)
    2. Lattice parameters (via Lattice Diffusion)
    
    Args:
        in_node_nf: Input node feature dimension (e.g., atom types)
        hidden_nf: Hidden feature dimension
        out_node_nf: Output node feature dimension
        n_layers: Number of EGNN layers
        attention: Whether to use attention in EGNN
        normalize: Whether to normalize features
        tanh: Whether to use tanh in coordinate updates
        coords_range: Range for coordinate updates
        learn_lattice: Whether to learn lattice parameters
        context_nf: Context feature dimension
    """
    
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int = 128,
        out_node_nf: int = 1,
        in_edge_nf: int = 0,
        n_layers: int = 4,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        coords_range: float = 15.0,
        learn_lattice: bool = True,
        context_nf: int = 0,
        aggregation_method: str = 'sum'
    ):
        super(CrystalDynamics, self).__init__()
        
        self.learn_lattice = learn_lattice
        self.hidden_nf = hidden_nf
        
        # Periodic EGNN for atomic positions
        self.egnn = PeriodicEGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            in_edge_nf=in_edge_nf,
            n_layers=n_layers,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
            coords_range=coords_range,
            aggregation_method=aggregation_method
        )
        
        # Lattice diffusion model
        if learn_lattice:
            self.lattice_model = LatticeDiffusion(
                hidden_nf=hidden_nf,
                context_nf=context_nf
            )
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell_vectors: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for crystal dynamics.
        
        Args:
            h: Node features (n_nodes, in_node_nf)
            x: Atomic positions in fractional coordinates (n_nodes, 3)
            cell_vectors: Unit cell vectors (3, 3) or (batch_size, 3, 3)
            edge_index: Edge connectivity (2, n_edges)
            node_mask: Node mask (n_nodes, 1)
            edge_mask: Edge mask (n_edges, 1)
            context: Context features for conditioning (batch_size, context_nf)
            
        Returns:
            h_out: Updated node features (n_nodes, out_node_nf)
            x_out: Updated fractional coordinates (n_nodes, 3)
            cell_vectors_out: Updated cell vectors (3, 3) or (batch_size, 3, 3)
        """
        # Update atomic positions and features with EGNN
        h_out, x_out = self.egnn(
            h, x, cell_vectors,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask
        )
        
        # Update lattice parameters if learning
        if self.learn_lattice:
            # Convert cell vectors to parameters
            if cell_vectors.dim() == 2:
                lengths, angles = cell_vectors_to_params(cell_vectors.unsqueeze(0))
                lengths = lengths.squeeze(0)
                angles = angles.squeeze(0)
                batch_size = 1
            else:
                lengths, angles = cell_vectors_to_params(cell_vectors)
                batch_size = cell_vectors.shape[0]
            
            # Normalize lattice parameters
            lattice_params_norm = LatticeDiffusion.normalize_lattice_params(
                lengths.unsqueeze(0) if lengths.dim() == 1 else lengths,
                angles.unsqueeze(0) if angles.dim() == 1 else angles
            )
            
            # Predict lattice update
            # Get EGNN hidden features (use h_out from EGNN)
            # We need to extract the hidden features, but h_out is the decoded output
            # For simplicity, we'll use the egnn.embedding to get features
            h_hidden = self.egnn.embedding(h)
            
            lattice_update = self.lattice_model(
                lattice_params_norm,
                h_hidden,
                node_mask if node_mask is not None else torch.ones(h.shape[0], 1, device=h.device),
                context=context
            )
            
            # Apply update
            lattice_params_new = lattice_params_norm + lattice_update
            
            # Denormalize
            lengths_new, angles_new = LatticeDiffusion.denormalize_lattice_params(
                lattice_params_new
            )
            
            # Convert back to cell vectors
            cell_vectors_out = cell_params_to_vectors(lengths_new, angles_new)
            
            # Remove batch dimension if it was added
            if batch_size == 1 and cell_vectors.dim() == 2:
                cell_vectors_out = cell_vectors_out.squeeze(0)
        else:
            cell_vectors_out = cell_vectors
        
        return h_out, x_out, cell_vectors_out
    
    def sample(
        self,
        n_nodes: int,
        in_node_nf: int,
        device: torch.device,
        cell_init: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a crystal structure from random noise.
        
        Args:
            n_nodes: Number of atoms
            in_node_nf: Node feature dimension
            device: Device to use
            cell_init: Initial cell vectors (optional)
            context: Context features for conditioning
            
        Returns:
            h: Node features
            x: Fractional coordinates
            cell_vectors: Cell vectors
        """
        # Initialize random features and coordinates
        h = torch.randn(n_nodes, in_node_nf, device=device)
        x = torch.rand(n_nodes, 3, device=device)  # Fractional coords in [0, 1]
        
        # Initialize cell vectors
        if cell_init is None:
            # Start with cubic cell
            cell_vectors = torch.eye(3, device=device) * 10.0
        else:
            cell_vectors = cell_init
        
        # Single forward pass (in practice, this would be part of a diffusion loop)
        node_mask = torch.ones(n_nodes, 1, device=device)
        h_out, x_out, cell_out = self.forward(
            h, x, cell_vectors,
            node_mask=node_mask,
            context=context
        )
        
        return h_out, x_out, cell_out
