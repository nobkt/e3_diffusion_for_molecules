"""
Molecule Encoder for Crystal Conditioning

This module provides an E(3) equivariant encoder that transforms a single molecule
into a fixed-length context vector for conditional crystal generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MoleculeEncoder(nn.Module):
    """
    E(3) Equivariant Molecule Encoder.
    
    Encodes a single molecule into a fixed-length context vector that is
    invariant to rotation and translation. Uses EGNN layers followed by
    graph pooling.
    """
    
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int = 256,
        out_nf: int = 128,
        n_layers: int = 6,
        attention: bool = True,
        normalization_factor: float = 100.0,
        aggregation_method: str = 'mean_max',
        tanh: bool = False,
    ):
        """
        Args:
            in_node_nf: Input node feature dimension (usually num_atom_types)
            hidden_nf: Hidden layer dimension
            out_nf: Output embedding dimension
            n_layers: Number of EGNN layers
            attention: Whether to use attention in EGNN
            normalization_factor: Normalization factor for coordinates
            aggregation_method: 'mean', 'max', 'mean_max', or 'attention'
            tanh: Whether to use tanh in coord_mlp
        """
        super().__init__()
        
        self.hidden_nf = hidden_nf
        self.out_nf = out_nf
        self.n_layers = n_layers
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        
        # Initial embedding
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_nf=hidden_nf,
                edge_nf=0,
                attention=attention,
                normalize=False,
                tanh=tanh,
            )
            for _ in range(n_layers)
        ])
        
        # Aggregation
        if aggregation_method == 'mean':
            agg_input_dim = hidden_nf
        elif aggregation_method == 'max':
            agg_input_dim = hidden_nf
        elif aggregation_method == 'mean_max':
            agg_input_dim = hidden_nf * 2
        elif aggregation_method == 'attention':
            self.attention_query = nn.Parameter(torch.randn(1, hidden_nf))
            agg_input_dim = hidden_nf
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Output projection
        self.output_mlp = nn.Sequential(
            nn.Linear(agg_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_nf),
        )
        
        logger.info(f"MoleculeEncoder initialized: {n_layers} layers, "
                   f"hidden_dim={hidden_nf}, out_dim={out_nf}, "
                   f"aggregation={aggregation_method}")
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a batch of molecules.
        
        Args:
            h: Node features [batch, n_atoms, in_node_nf]
            x: Atomic coordinates [batch, n_atoms, 3]
            node_mask: Node mask [batch, n_atoms]
            edge_mask: Edge mask [batch, n_edges] (optional)
            
        Returns:
            c_mol: Molecular embedding [batch, out_nf]
        """
        batch_size, n_atoms, _ = h.shape
        
        # Normalize coordinates
        x = x / self.normalization_factor
        
        # Initial embedding
        h = self.embedding(h)
        
        # Pass through EGNN layers
        for layer in self.egnn_layers:
            h, x = layer(h, x, node_mask, edge_mask)
        
        # Aggregate node features
        c_mol = self._aggregate(h, node_mask)
        
        # Output projection
        c_mol = self.output_mlp(c_mol)
        
        return c_mol
    
    def _aggregate(
        self,
        h: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate node features into a global representation.
        
        Args:
            h: Node features [batch, n_atoms, hidden_nf]
            node_mask: Node mask [batch, n_atoms]
            
        Returns:
            aggregated: Global features [batch, agg_dim]
        """
        # Mask features
        h_masked = h * node_mask.unsqueeze(-1)
        
        if self.aggregation_method == 'mean':
            # Mean pooling
            num_atoms = node_mask.sum(dim=1, keepdim=True)
            aggregated = h_masked.sum(dim=1) / (num_atoms + 1e-8)
            
        elif self.aggregation_method == 'max':
            # Max pooling (mask with large negative values)
            h_masked = h_masked + (1 - node_mask.unsqueeze(-1)) * (-1e9)
            aggregated = h_masked.max(dim=1)[0]
            
        elif self.aggregation_method == 'mean_max':
            # Combination of mean and max pooling
            num_atoms = node_mask.sum(dim=1, keepdim=True)
            mean_pool = h_masked.sum(dim=1) / (num_atoms + 1e-8)
            
            h_masked_max = h_masked + (1 - node_mask.unsqueeze(-1)) * (-1e9)
            max_pool = h_masked_max.max(dim=1)[0]
            
            aggregated = torch.cat([mean_pool, max_pool], dim=-1)
            
        elif self.aggregation_method == 'attention':
            # Attention-based pooling
            # Compute attention scores
            scores = torch.matmul(h, self.attention_query.T)  # [batch, n_atoms, 1]
            scores = scores.squeeze(-1)  # [batch, n_atoms]
            
            # Mask and normalize
            scores = scores.masked_fill(~node_mask, -1e9)
            attn_weights = torch.softmax(scores, dim=1)  # [batch, n_atoms]
            
            # Weighted sum
            aggregated = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_nf]
        
        return aggregated


class EGNNLayer(nn.Module):
    """
    E(3) Equivariant Graph Neural Network Layer.
    
    Based on Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021).
    """
    
    def __init__(
        self,
        hidden_nf: int,
        edge_nf: int = 0,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        
        self.hidden_nf = hidden_nf
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        
        # Edge model
        edge_input_nf = hidden_nf * 2 + edge_nf + 1  # h_i, h_j, edge_attr, distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
        )
        
        # Coordinate update
        coord_input_nf = hidden_nf
        coord_layers = [
            nn.Linear(coord_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1, bias=False),
        ]
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)
        
        # Node update
        node_input_nf = hidden_nf * 2
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )
        
        # Attention
        if attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            h: Node features [batch, n_atoms, hidden_nf]
            x: Coordinates [batch, n_atoms, 3]
            node_mask: Node mask [batch, n_atoms]
            edge_mask: Edge mask [batch, n_edges] (optional)
            
        Returns:
            h_out: Updated node features [batch, n_atoms, hidden_nf]
            x_out: Updated coordinates [batch, n_atoms, 3]
        """
        batch_size, n_atoms, _ = h.shape
        
        # Compute pairwise distances and relative vectors
        # [batch, n_atoms, n_atoms, 3]
        rel_vectors = x.unsqueeze(2) - x.unsqueeze(1)
        # [batch, n_atoms, n_atoms]
        distances = torch.norm(rel_vectors, dim=-1, keepdim=True)
        
        # Create edge features
        # [batch, n_atoms, n_atoms, hidden_nf]
        h_i = h.unsqueeze(2).expand(-1, -1, n_atoms, -1)
        h_j = h.unsqueeze(1).expand(-1, n_atoms, -1, -1)
        
        # [batch, n_atoms, n_atoms, hidden_nf * 2 + 1]
        edge_input = torch.cat([h_i, h_j, distances], dim=-1)
        
        # Edge model
        # [batch, n_atoms, n_atoms, hidden_nf]
        edge_feat = self.edge_mlp(edge_input)
        
        # Apply edge mask (exclude self-connections and padding)
        # [batch, n_atoms, n_atoms]
        edge_mask_full = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        # Exclude self-connections
        eye = torch.eye(n_atoms, device=h.device, dtype=torch.bool).unsqueeze(0)
        edge_mask_full = edge_mask_full * (~eye)
        
        # Apply mask to edge features
        edge_feat = edge_feat * edge_mask_full.unsqueeze(-1)
        
        # Attention
        if self.attention:
            att = self.attention_mlp(edge_feat)  # [batch, n_atoms, n_atoms, 1]
            edge_feat = edge_feat * att
        
        # Coordinate update
        coord_weight = self.coord_mlp(edge_feat)  # [batch, n_atoms, n_atoms, 1]
        
        if self.normalize:
            norm = distances + 1e-8
            coord_diff = rel_vectors / norm
        else:
            coord_diff = rel_vectors
        
        # [batch, n_atoms, 3]
        coord_update = torch.sum(coord_weight * coord_diff, dim=2)
        x_out = x + coord_update
        
        # Node update
        # Aggregate edge features
        # [batch, n_atoms, hidden_nf]
        agg_feat = torch.sum(edge_feat, dim=2)
        
        # [batch, n_atoms, hidden_nf * 2]
        node_input = torch.cat([h, agg_feat], dim=-1)
        
        # [batch, n_atoms, hidden_nf]
        h_update = self.node_mlp(node_input)
        
        # Residual connection
        h_out = h + h_update
        
        # Apply node mask
        h_out = h_out * node_mask.unsqueeze(-1)
        x_out = x_out * node_mask.unsqueeze(-1)
        
        return h_out, x_out


# Test function
def test_molecule_encoder():
    """Test the molecule encoder."""
    print("Testing MoleculeEncoder...")
    
    # Parameters
    batch_size = 4
    n_atoms = 10
    num_atom_types = 5
    hidden_nf = 128
    out_nf = 64
    
    # Create encoder
    encoder = MoleculeEncoder(
        in_node_nf=num_atom_types,
        hidden_nf=hidden_nf,
        out_nf=out_nf,
        n_layers=3,
        aggregation_method='mean_max',
    )
    
    # Create random input
    h = torch.randn(batch_size, n_atoms, num_atom_types)
    x = torch.randn(batch_size, n_atoms, 3)
    node_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    # Mask some atoms
    node_mask[0, 8:] = False
    node_mask[1, 9:] = False
    
    # Forward pass
    c_mol = encoder(h, x, node_mask)
    
    print(f"Input shape: h={h.shape}, x={x.shape}")
    print(f"Output shape: c_mol={c_mol.shape}")
    print(f"Expected output shape: [{batch_size}, {out_nf}]")
    
    assert c_mol.shape == (batch_size, out_nf), "Output shape mismatch"
    
    # Test E(3) equivariance
    print("\nTesting E(3) equivariance...")
    
    # Apply rotation and translation
    R = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    t = torch.tensor([1.0, 2.0, 3.0])
    
    x_transformed = (x @ R.T) + t
    
    c_mol_transformed = encoder(h, x_transformed, node_mask)
    
    # Embeddings should be approximately equal (invariant)
    diff = torch.abs(c_mol - c_mol_transformed).max().item()
    print(f"Max difference after transformation: {diff}")
    
    if diff < 1e-5:
        print("✓ Encoder is E(3) equivariant (invariant)")
    else:
        print(f"✗ Encoder may not be perfectly equivariant (diff={diff})")
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    test_molecule_encoder()
