"""
Molecular EGNN Feature Encoder for extracting features from single molecules.

This module extracts EGNN features from single molecules to be used as conditioning
for homocrystal generation. It follows the specification in MOLECULAR_CRYSTAL_DESIGN.md
with no fallback heuristics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from egnn.egnn_new import EGNN


class MolecularEncoder(nn.Module):
    """
    Extract EGNN features from single molecules.
    
    Uses the existing EGNN architecture from egnn_new.py to extract molecular-level
    features that will condition crystal generation.
    
    Key features:
    - Node-level features from EGNN layers
    - Global molecular features via pooling
    - Geometric properties (size, volume, principal axes)
    - Compatible with pre-trained models
    """
    
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int = 128,
        n_layers: int = 4,
        global_feature_dim: int = 128,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize the molecular encoder.
        
        Args:
            in_node_nf: Input node feature dimension (atom one-hot encoding)
            hidden_nf: Hidden layer dimension
            n_layers: Number of EGNN layers
            global_feature_dim: Output global feature dimension
            attention: Use attention in EGNN
            normalize: Normalize coordinates
            tanh: Use tanh in coordinate MLP
            pretrained_path: Path to pre-trained model (optional)
        """
        super().__init__()
        
        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.global_feature_dim = global_feature_dim
        
        # Molecular EGNN (reuse existing implementation)
        self.molecular_egnn = EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=0,  # No edge features for molecules
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_diff=normalize,
        )
        
        # Global pooling transformation
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, global_feature_dim),
        )
        
        # Geometry prediction head
        # Output: size(3) + volume(1) + principal_axes(9) + padding(3) = 16
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, 16),
        )
        
        # Load pre-trained model if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        extract_geometry: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract molecular EGNN features.
        
        Args:
            h: [batch, n_atoms, in_node_nf] Node features (atom one-hot)
            x: [batch, n_atoms, 3] Atomic coordinates
            edge_index: [2, n_edges] Edge indices
            node_mask: [batch, n_atoms, 1] Node mask for padding
            extract_geometry: Whether to extract geometric properties
            
        Returns:
            Dictionary containing:
                - node_features: [batch, n_atoms, hidden_nf] Per-atom features
                - global_features: [batch, global_feature_dim] Molecular features
                - mol_size: [batch, 3] Molecular size (optional)
                - mol_volume: [batch, 1] Molecular volume (optional)
                - principal_axes: [batch, 3, 3] Principal axes (optional)
        """
        batch_size, n_atoms, _ = h.shape
        
        # Flatten batch dimension for EGNN
        h_flat = h.view(-1, h.shape[-1])  # [batch * n_atoms, in_node_nf]
        x_flat = x.view(-1, 3)  # [batch * n_atoms, 3]
        
        # Create node mask if not provided
        if node_mask is None:
            node_mask = torch.ones(batch_size, n_atoms, 1, device=h.device)
        
        node_mask_flat = node_mask.view(-1, 1)  # [batch * n_atoms, 1]
        
        # Pass through EGNN
        node_features_flat, _ = self.molecular_egnn(
            h=h_flat,
            x=x_flat,
            edge_index=edge_index,
            node_mask=node_mask_flat,
        )
        
        # Reshape back to batch
        node_features = node_features_flat.view(batch_size, n_atoms, self.hidden_nf)
        
        # Global pooling (masked average)
        masked_features = node_features * node_mask
        feature_sum = masked_features.sum(dim=1)  # [batch, hidden_nf]
        mask_sum = node_mask.sum(dim=1)  # [batch, 1]
        global_features_raw = feature_sum / (mask_sum + 1e-8)
        
        # Transform global features
        global_features = self.global_mlp(global_features_raw)
        
        # Prepare output
        features = {
            'node_features': node_features,
            'global_features': global_features,
        }
        
        # Extract geometry if requested
        if extract_geometry:
            geometry = self.geometry_head(global_features_raw)
            
            # Parse geometry output
            mol_size = torch.abs(geometry[:, :3])  # [batch, 3]
            mol_volume = torch.abs(geometry[:, 3:4])  # [batch, 1]
            principal_axes_flat = geometry[:, 4:13]  # [batch, 9]
            principal_axes = principal_axes_flat.view(batch_size, 3, 3)
            
            # Orthogonalize principal axes
            principal_axes = self._orthogonalize(principal_axes)
            
            features.update({
                'mol_size': mol_size,
                'mol_volume': mol_volume,
                'principal_axes': principal_axes,
            })
        
        return features
    
    def _orthogonalize(self, matrices: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize matrices using Gram-Schmidt process.
        
        Args:
            matrices: [batch, 3, 3] Input matrices
            
        Returns:
            orthogonal_matrices: [batch, 3, 3] Orthogonalized matrices
        """
        v1 = matrices[:, 0, :]  # [batch, 3]
        v2 = matrices[:, 1, :]
        v3 = matrices[:, 2, :]
        
        # Normalize v1
        u1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        
        # Orthogonalize v2 against u1
        u2 = v2 - (torch.sum(v2 * u1, dim=-1, keepdim=True) * u1)
        u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-8)
        
        # Orthogonalize v3 against u1 and u2
        u3 = v3 - (torch.sum(v3 * u1, dim=-1, keepdim=True) * u1)
        u3 = u3 - (torch.sum(u3 * u2, dim=-1, keepdim=True) * u2)
        u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + 1e-8)
        
        # Stack into matrix
        orthogonal = torch.stack([u1, u2, u3], dim=1)  # [batch, 3, 3]
        
        return orthogonal
    
    def load_pretrained(self, pretrained_path: str):
        """
        Load pre-trained EGNN weights.
        
        Args:
            pretrained_path: Path to checkpoint file
        """
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Extract EGNN weights
        egnn_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('molecular_egnn.'):
                egnn_state_dict[key.replace('molecular_egnn.', '')] = value
        
        # Load with strict=False to allow partial loading
        self.molecular_egnn.load_state_dict(egnn_state_dict, strict=False)
        print(f"Loaded pre-trained molecular EGNN from {pretrained_path}")


def create_fully_connected_edges(n_atoms: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Create fully connected edge indices for molecular graphs.
    
    Args:
        n_atoms: Number of atoms per molecule
        batch_size: Batch size
        device: Device to create tensors on
        
    Returns:
        edge_index: [2, n_edges] Edge indices
    """
    # Create edges for single graph
    src = torch.arange(n_atoms, device=device).repeat(n_atoms)
    dst = torch.arange(n_atoms, device=device).repeat_interleave(n_atoms)
    
    # Remove self-loops
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    # Replicate for batch
    edge_list = []
    for b in range(batch_size):
        offset = b * n_atoms
        edge_list.append(torch.stack([src + offset, dst + offset], dim=0))
    
    edge_index = torch.cat(edge_list, dim=1)  # [2, batch * n_edges]
    
    return edge_index
