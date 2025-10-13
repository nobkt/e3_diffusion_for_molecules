"""
Periodic E(3) Equivariant Graph Neural Network.

This module implements E(3) equivariant message passing with periodic boundary
conditions for molecular crystal generation. Follows specifications in
MOLECULAR_CRYSTAL_DESIGN.md with no fallback heuristics.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from crystal.data.periodic_utils import minimum_image_distance


class PeriodicEGNN(nn.Module):
    """
    E(3) Equivariant GNN with periodic boundary conditions.
    
    Key features:
    - Minimum image convention for distance calculation
    - Periodic neighbor lists
    - E(3) equivariant coordinate updates
    - Compatible with fractional or Cartesian coordinates
    """
    
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        in_edge_nf: int = 0,
        n_layers: int = 4,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
        use_fractional_coords: bool = True,
    ):
        """
        Initialize Periodic EGNN.
        
        Args:
            in_node_nf: Input node feature dimension
            hidden_nf: Hidden layer dimension
            out_node_nf: Output node feature dimension
            in_edge_nf: Input edge feature dimension
            n_layers: Number of EGNN layers
            attention: Use attention mechanism
            normalize: Normalize coordinates
            tanh: Use tanh in coordinate MLP
            use_fractional_coords: Work with fractional coordinates
        """
        super().__init__()
        
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.use_fractional_coords = use_fractional_coords
        
        # Node embedding
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            PeriodicEGNNLayer(
                hidden_nf=hidden_nf,
                edge_nf=in_edge_nf,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            )
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_nf, out_node_nf)
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with periodic boundary conditions.
        
        Args:
            h: [batch, n_atoms, in_node_nf] Node features
            x: [batch, n_atoms, 3] Coordinates (fractional or Cartesian)
            cell: [batch, 3, 3] Unit cell vectors
            pbc: [batch, 3] Periodic boundary flags
            edge_index: [batch, 2, n_edges] Edge indices (optional)
            edge_attr: [batch, n_edges, edge_nf] Edge features (optional)
            node_mask: [batch, n_atoms, 1] Node mask (optional)
            
        Returns:
            h_out: [batch, n_atoms, out_node_nf] Output node features
            x_out: [batch, n_atoms, 3] Output coordinates
        """
        # Embed nodes
        h = self.embedding(h)
        
        # Apply each layer
        for layer in self.layers:
            h, x = layer(
                h=h,
                x=x,
                cell=cell,
                pbc=pbc,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
            )
        
        # Output projection
        h_out = self.output_layer(h)
        x_out = x
        
        # Apply mask if provided
        if node_mask is not None:
            h_out = h_out * node_mask
        
        return h_out, x_out


class PeriodicEGNNLayer(nn.Module):
    """
    Single layer of Periodic EGNN.
    
    Implements message passing, coordinate updates, and node updates
    with periodic boundary conditions.
    """
    
    def __init__(
        self,
        hidden_nf: int,
        edge_nf: int = 0,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
    ):
        """
        Initialize a single Periodic EGNN layer.
        
        Args:
            hidden_nf: Hidden feature dimension
            edge_nf: Edge feature dimension
            attention: Use attention mechanism
            normalize: Normalize coordinates
            tanh: Use tanh in coordinate MLP
        """
        super().__init__()
        
        self.hidden_nf = hidden_nf
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        
        # Edge model: h_i, h_j, edge_attr, distance -> edge_features
        edge_input_nf = hidden_nf * 2 + edge_nf + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
        )
        
        # Coordinate update model
        coord_input_nf = hidden_nf
        coord_layers = [
            nn.Linear(coord_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1, bias=False),
        ]
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)
        
        # Node update model
        node_input_nf = hidden_nf * 2
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )
        
        # Attention model
        if attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single layer.
        
        Args:
            h: [batch, n_atoms, hidden_nf] Node features
            x: [batch, n_atoms, 3] Coordinates
            cell: [batch, 3, 3] Unit cell vectors
            pbc: [batch, 3] Periodic boundary flags
            edge_index: [batch, 2, n_edges] Edge indices
            edge_attr: [batch, n_edges, edge_nf] Edge features
            node_mask: [batch, n_atoms, 1] Node mask
            
        Returns:
            h_new: [batch, n_atoms, hidden_nf] Updated node features
            x_new: [batch, n_atoms, 3] Updated coordinates
        """
        batch_size, n_atoms, _ = h.shape
        
        # Build edge index if not provided (fully connected)
        if edge_index is None:
            edge_index = self._build_fully_connected_edges(n_atoms, batch_size, h.device)
        
        # Compute periodic distances and vectors
        # Note: minimum_image_distance expects [batch, n_atoms, 3] input
        distances, rel_coords = minimum_image_distance(
            x, x, cell, pbc, use_fractional=False
        )  # [batch, n_atoms, n_atoms], [batch, n_atoms, n_atoms, 3]
        
        # Extract edge-wise distances and vectors
        src_idx = edge_index[:, 0, :]  # [batch, n_edges]
        dst_idx = edge_index[:, 1, :]  # [batch, n_edges]
        
        # Gather distances and vectors for edges
        edge_distances = torch.gather(
            distances.view(batch_size, -1).unsqueeze(1),
            2,
            (src_idx * n_atoms + dst_idx).unsqueeze(1)
        ).squeeze(1).unsqueeze(-1)  # [batch, n_edges, 1]
        
        # Gather relative coordinate vectors
        batch_indices = torch.arange(batch_size, device=h.device).view(-1, 1, 1)
        edge_vectors = rel_coords[
            batch_indices,
            src_idx.unsqueeze(-1),
            dst_idx.unsqueeze(-1),
            :
        ].squeeze(2)  # [batch, n_edges, 3]
        
        # Gather node features for edges
        h_src = torch.gather(
            h, 1,
            src_idx.unsqueeze(-1).expand(-1, -1, self.hidden_nf)
        )  # [batch, n_edges, hidden_nf]
        h_dst = torch.gather(
            h, 1,
            dst_idx.unsqueeze(-1).expand(-1, -1, self.hidden_nf)
        )  # [batch, n_edges, hidden_nf]
        
        # Edge model: compute edge features
        edge_input = torch.cat([h_src, h_dst, edge_distances], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        
        edge_features = self.edge_mlp(edge_input)  # [batch, n_edges, hidden_nf]
        
        # Apply attention if enabled
        if self.attention:
            att = self.attention_mlp(edge_features)  # [batch, n_edges, 1]
            edge_features = edge_features * att
        
        # Coordinate update
        coord_weights = self.coord_mlp(edge_features)  # [batch, n_edges, 1]
        coord_update = edge_vectors * coord_weights  # [batch, n_edges, 3]
        
        # Aggregate coordinate updates (scatter add)
        x_new = x.clone()
        for b in range(batch_size):
            src_b = src_idx[b]  # [n_edges]
            coord_update_b = coord_update[b]  # [n_edges, 3]
            x_new[b] = x_new[b].index_add(0, src_b, coord_update_b)
        
        # Aggregate edge features for node update (scatter add)
        h_agg = torch.zeros_like(h)
        for b in range(batch_size):
            src_b = src_idx[b]  # [n_edges]
            edge_features_b = edge_features[b]  # [n_edges, hidden_nf]
            h_agg[b] = h_agg[b].index_add(0, src_b, edge_features_b)
        
        # Node update
        node_input = torch.cat([h, h_agg], dim=-1)
        h_new = h + self.node_mlp(node_input)
        
        # Apply mask if provided
        if node_mask is not None:
            h_new = h_new * node_mask
            x_new = x_new * node_mask
        
        return h_new, x_new
    
    def _build_fully_connected_edges(
        self,
        n_atoms: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Build fully connected edge indices.
        
        Args:
            n_atoms: Number of atoms
            batch_size: Batch size
            device: Device
            
        Returns:
            edge_index: [batch, 2, n_edges] Edge indices
        """
        # Create edges for single graph
        src = torch.arange(n_atoms, device=device).repeat(n_atoms)
        dst = torch.arange(n_atoms, device=device).repeat_interleave(n_atoms)
        
        # Remove self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        
        # Replicate for batch
        edge_index = torch.stack([src, dst], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        
        return edge_index
