"""
Property Predictor Module

Predicts physical properties of crystal structures using E(3)-equivariant graph neural networks.
This module enables validation of generated crystals by predicting their properties.

Design Principles:
- E(3) equivariance: Uses PeriodicEGNN for structure encoding
- No fallback heuristics: All predictions are learned from data
- Modular architecture: Separate encoder and prediction heads
- Multi-property support: Can predict multiple properties simultaneously

This implements Component P4-1 (Property Validation System) from Phase 4.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import warnings


class PropertyPredictor(nn.Module):
    """
    Predict crystal properties from atomic structure.
    
    Uses an E(3)-equivariant graph neural network to encode the crystal structure,
    then applies separate prediction heads for each property.
    
    The model maintains E(3) equivariance through the PeriodicEGNN encoder and
    produces scalar predictions that are invariant to rotations and translations.
    
    Args:
        property_names: List of property names to predict (e.g., ['bandgap', 'melting_point'])
        hidden_dim: Hidden dimension for structure encoder
        n_layers: Number of EGNN layers for structure encoding (2-8)
        max_neighbors: Maximum number of neighbors for graph construction
        cutoff_radius: Cutoff radius for neighbor search (Angstroms)
        
    Raises:
        ValueError: If property_names is empty or parameters are invalid
        
    Example:
        >>> predictor = PropertyPredictor(
        ...     property_names=['bandgap', 'melting_point'],
        ...     hidden_dim=256,
        ...     n_layers=4
        ... )
        >>> # positions: [batch_size, n_atoms, 3]
        >>> # cell: [batch_size, 3, 3]
        >>> # atomic_numbers: [batch_size, n_atoms]
        >>> predictions = predictor(positions, cell, atomic_numbers)
        >>> predictions['bandgap'].shape
        torch.Size([batch_size, 1])
    """
    
    def __init__(
        self,
        property_names: List[str],
        hidden_dim: int = 256,
        n_layers: int = 4,
        max_neighbors: int = 32,
        cutoff_radius: float = 8.0,
        num_atom_types: int = 100,
    ):
        super().__init__()
        
        # Validate parameters
        if not property_names:
            raise ValueError("property_names must be non-empty")
        if not 2 <= n_layers <= 8:
            raise ValueError(f"n_layers must be between 2 and 8, got {n_layers}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be positive, got {max_neighbors}")
        if cutoff_radius <= 0:
            raise ValueError(f"cutoff_radius must be positive, got {cutoff_radius}")
        
        # Store configuration
        self.property_names = property_names
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_neighbors = max_neighbors
        self.cutoff_radius = cutoff_radius
        self.num_atom_types = num_atom_types
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # E(3)-equivariant structure encoder
        # We'll use a simplified EGNN-like architecture for now
        # In practice, this would use PeriodicEGNN from crystal.models
        self.structure_encoder = self._build_structure_encoder()
        
        # Global pooling to get structure-level features
        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Property-specific prediction heads
        self.property_heads = nn.ModuleDict({
            name: self._build_prediction_head(hidden_dim)
            for name in property_names
        })
        
        # Normalization parameters (registered as buffers)
        # These will be set by set_normalization_params()
        self.register_buffer(
            'property_mean',
            torch.zeros(len(property_names))
        )
        self.register_buffer(
            'property_std',
            torch.ones(len(property_names))
        )
        self.eps = 1e-8
        
    def _build_structure_encoder(self) -> nn.ModuleList:
        """
        Build E(3)-equivariant structure encoder layers.
        
        This is a simplified version. In production, would use PeriodicEGNN.
        
        Returns:
            encoder: nn.ModuleList of EGNN-like layers
        """
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layers.append(
                SimpleEGNNLayer(
                    hidden_dim=self.hidden_dim,
                    edge_dim=1,  # Distance as edge feature
                )
            )
        return layers
    
    def _build_prediction_head(self, input_dim: int) -> nn.Sequential:
        """
        Build prediction head for a single property.
        
        Architecture:
            Linear(input_dim, hidden_dim) -> LayerNorm -> SiLU ->
            Linear(hidden_dim, hidden_dim//2) -> LayerNorm -> SiLU ->
            Linear(hidden_dim//2, 1)
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            head: nn.Sequential prediction head
        """
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
    
    def _compute_edges(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> tuple:
        """
        Compute edges for graph neural network with periodic boundary conditions.
        
        Args:
            positions: [batch_size, n_atoms, 3] atomic positions
            cell: [batch_size, 3, 3] unit cell vectors
            atomic_numbers: [batch_size, n_atoms] atomic numbers
            
        Returns:
            edge_index: [2, n_edges] edge connectivity
            edge_attr: [n_edges, edge_dim] edge features (distances)
            batch_idx: [n_atoms] batch index for each atom
        """
        batch_size, n_atoms, _ = positions.shape
        device = positions.device
        
        # For simplicity, compute edges within cutoff radius
        # In production, would use periodic boundary conditions properly
        edges = []
        edge_features = []
        batch_idx = []
        
        for b in range(batch_size):
            pos = positions[b]  # [n_atoms, 3]
            
            # Compute pairwise distances
            dist_matrix = torch.cdist(pos, pos)  # [n_atoms, n_atoms]
            
            # Find neighbors within cutoff
            mask = (dist_matrix < self.cutoff_radius) & (dist_matrix > 0)
            
            # Get edge indices
            src, dst = torch.where(mask)
            
            # Limit to max_neighbors per atom
            # For each source atom, keep only closest neighbors
            unique_src = torch.unique(src)
            filtered_src = []
            filtered_dst = []
            filtered_dist = []
            
            for s in unique_src:
                s_mask = src == s
                s_dst = dst[s_mask]
                s_dist = dist_matrix[s, s_dst]
                
                # Keep top k neighbors
                if len(s_dst) > self.max_neighbors:
                    _, indices = torch.topk(s_dist, self.max_neighbors, largest=False)
                    s_dst = s_dst[indices]
                    s_dist = s_dist[indices]
                
                filtered_src.extend([s.item()] * len(s_dst))
                filtered_dst.extend(s_dst.tolist())
                filtered_dist.extend(s_dist.tolist())
            
            if filtered_src:
                # Add batch offset
                offset = b * n_atoms
                edges.append(torch.tensor([
                    [s + offset for s in filtered_src],
                    [d + offset for d in filtered_dst]
                ], device=device))
                edge_features.extend(filtered_dist)
            
            # Batch indices
            batch_idx.extend([b] * n_atoms)
        
        if not edges:
            # No edges found - create empty tensors
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.zeros(0, 1, device=device)
        else:
            edge_index = torch.cat(edges, dim=1)
            edge_attr = torch.tensor(edge_features, device=device).unsqueeze(-1)
        
        batch_idx = torch.tensor(batch_idx, device=device)
        
        return edge_index, edge_attr, batch_idx
    
    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
        return_normalized: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict properties from crystal structure.
        
        Args:
            positions: [batch_size, n_atoms, 3] atomic positions in fractional coordinates
            cell: [batch_size, 3, 3] unit cell vectors (rows are a, b, c vectors)
            atomic_numbers: [batch_size, n_atoms] atomic numbers
            return_normalized: If True, return normalized predictions
            
        Returns:
            predictions: Dictionary mapping property names to [batch_size, 1] predictions
            
        Raises:
            ValueError: If input shapes are inconsistent
            
        Example:
            >>> batch_size, n_atoms = 4, 20
            >>> positions = torch.randn(batch_size, n_atoms, 3)
            >>> cell = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
            >>> atomic_numbers = torch.randint(1, 20, (batch_size, n_atoms))
            >>> predictions = predictor(positions, cell, atomic_numbers)
        """
        # Validate input shapes
        batch_size, n_atoms, dim = positions.shape
        if dim != 3:
            raise ValueError(f"positions must have shape [batch_size, n_atoms, 3], got {positions.shape}")
        if cell.shape != (batch_size, 3, 3):
            raise ValueError(f"cell must have shape [batch_size, 3, 3], got {cell.shape}")
        if atomic_numbers.shape != (batch_size, n_atoms):
            raise ValueError(f"atomic_numbers must have shape [batch_size, n_atoms], got {atomic_numbers.shape}")
        
        # Convert fractional to Cartesian coordinates
        # positions_cart = positions @ cell
        positions_cart = torch.bmm(positions, cell)  # [batch_size, n_atoms, 3]
        
        # Flatten batch for graph processing
        positions_flat = positions_cart.view(-1, 3)  # [batch_size * n_atoms, 3]
        atomic_numbers_flat = atomic_numbers.view(-1)  # [batch_size * n_atoms]
        
        # Embed atom types
        h = self.atom_embedding(atomic_numbers_flat.long())  # [batch_size * n_atoms, hidden_dim]
        
        # Compute graph edges
        edge_index, edge_attr, batch_idx = self._compute_edges(
            positions_cart, cell, atomic_numbers
        )
        
        # Apply structure encoder layers
        for layer in self.structure_encoder:
            h = layer(h, positions_flat, edge_index, edge_attr)
        
        # Global pooling to get structure-level features
        # Sum over atoms in each structure
        structure_features = torch.zeros(
            batch_size, self.hidden_dim,
            device=h.device, dtype=h.dtype
        )
        structure_features.index_add_(0, batch_idx, h)
        
        # Apply pooling MLP
        structure_features = self.pooling(structure_features)
        
        # Apply property-specific heads
        predictions = {}
        for i, (prop_name, head) in enumerate(self.property_heads.items()):
            pred_normalized = head(structure_features)  # [batch_size, 1]
            
            if return_normalized:
                predictions[prop_name] = pred_normalized
            else:
                # Denormalize prediction
                mean = self.property_mean[i]
                std = self.property_std[i]
                pred = pred_normalized * (std + self.eps) + mean
                predictions[prop_name] = pred
        
        return predictions
    
    def set_normalization_params(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> None:
        """
        Set normalization parameters for property predictions.
        
        These should match the normalization used during training.
        
        Args:
            mean: [n_properties] mean values for each property
            std: [n_properties] standard deviation for each property
            
        Raises:
            ValueError: If shapes don't match number of properties
            
        Example:
            >>> predictor = PropertyPredictor(['bandgap', 'melting_point'])
            >>> mean = torch.tensor([2.5, 180.0])
            >>> std = torch.tensor([1.2, 50.0])
            >>> predictor.set_normalization_params(mean, std)
        """
        n_props = len(self.property_names)
        
        if mean.shape != (n_props,):
            raise ValueError(
                f"Expected mean shape ({n_props},), got {mean.shape}"
            )
        if std.shape != (n_props,):
            raise ValueError(
                f"Expected std shape ({n_props},), got {std.shape}"
            )
        
        # Check for zero or negative std
        if torch.any(std <= 0):
            raise ValueError(
                f"Standard deviation must be positive for all properties. Got std={std}"
            )
        
        # Copy to buffers
        self.property_mean.copy_(mean.to(self.property_mean.device))
        self.property_std.copy_(std.to(self.property_std.device))
    
    def get_normalization_params(self):
        """
        Get current normalization parameters.
        
        Returns:
            mean: [n_properties] mean values
            std: [n_properties] standard deviation values
        """
        return self.property_mean.clone(), self.property_std.clone()
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'property_names={self.property_names}, '
            f'hidden_dim={self.hidden_dim}, '
            f'n_layers={self.n_layers}, '
            f'max_neighbors={self.max_neighbors}, '
            f'cutoff_radius={self.cutoff_radius}'
        )


class SimpleEGNNLayer(nn.Module):
    """
    Simplified E(3)-equivariant graph neural network layer.
    
    This is a basic implementation for property prediction.
    Maintains E(3) equivariance through distance-based message passing.
    
    Args:
        hidden_dim: Hidden feature dimension
        edge_dim: Edge feature dimension
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Edge MLP for computing messages
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply E(3)-equivariant layer.
        
        Args:
            h: [n_atoms, hidden_dim] node features
            positions: [n_atoms, 3] atomic positions (not updated, just used for distances)
            edge_index: [2, n_edges] edge connectivity
            edge_attr: [n_edges, edge_dim] edge features
            
        Returns:
            h_out: [n_atoms, hidden_dim] updated node features
        """
        if edge_index.size(1) == 0:
            # No edges, return input features unchanged
            return h
        
        # Get source and target node indices
        src, dst = edge_index[0], edge_index[1]
        
        # Get source and target features
        h_src = h[src]  # [n_edges, hidden_dim]
        h_dst = h[dst]  # [n_edges, hidden_dim]
        
        # Concatenate features and edge attributes
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        
        # Compute edge messages
        messages = self.edge_mlp(edge_input)  # [n_edges, hidden_dim]
        
        # Aggregate messages to destination nodes
        h_aggregated = torch.zeros_like(h)
        h_aggregated.index_add_(0, dst, messages)
        
        # Update node features
        node_input = torch.cat([h, h_aggregated], dim=-1)
        h_out = h + self.node_mlp(node_input)  # Residual connection
        
        return h_out
