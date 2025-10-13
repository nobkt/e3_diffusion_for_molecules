"""
Space Group Embedding Module

Converts space group numbers (1-230) to learnable embedding vectors.
Follows crystallographic principles with no fallback heuristics.
"""

import torch
import torch.nn as nn


class SpaceGroupEmbedding(nn.Module):
    """
    Space group number to embedding vector converter.
    
    Supports 230 space groups with learned embeddings.
    Space group 0 is reserved for unknown/missing space groups (padding).
    
    Design Principles:
    - No fallback heuristics: Invalid space groups are explicitly handled
    - Learned representations: Each space group has a unique embedding
    - Optional hierarchical structure: MLP projection for refinement
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_space_groups: int = 230,
    ):
        """
        Initialize Space Group Embedding.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            num_space_groups: Total number of space groups (default: 230)
        """
        super().__init__()
        
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_space_groups <= 0:
            raise ValueError(f"num_space_groups must be positive, got {num_space_groups}")
        
        self.embedding_dim = embedding_dim
        self.num_space_groups = num_space_groups
        
        # Space group embedding table
        # Space group numbers are 1-230, so we need num_space_groups + 1 for padding
        # Index 0: padding/unknown (will be zeroed)
        # Index 1-230: actual space groups
        self.embedding = nn.Embedding(
            num_embeddings=num_space_groups + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,  # Index 0 returns zeros
        )
        
        # Optional: MLP for embedding projection
        # This allows the model to learn hierarchical relationships between space groups
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, space_group: torch.Tensor) -> torch.Tensor:
        """
        Convert space group numbers to embedding vectors.
        
        Args:
            space_group: [batch] or [batch, 1] Space group numbers (0-230)
                        0 = unknown/padding (returns zero vector)
                        1-230 = valid space groups
        
        Returns:
            embedding: [batch, embedding_dim] Embedding vectors
            
        Raises:
            ValueError: If space_group contains values outside [0, num_space_groups]
        """
        # Handle input shape
        if space_group.dim() == 2:
            if space_group.size(1) != 1:
                raise ValueError(
                    f"space_group with dim=2 must have shape [batch, 1], "
                    f"got {space_group.shape}"
                )
            space_group = space_group.squeeze(1)
        elif space_group.dim() != 1:
            raise ValueError(
                f"space_group must be 1D or 2D tensor, got shape {space_group.shape}"
            )
        
        # Validate space group numbers (no fallback heuristics!)
        if torch.any(space_group < 0):
            raise ValueError(
                f"space_group contains negative values: {space_group[space_group < 0].tolist()}"
            )
        if torch.any(space_group > self.num_space_groups):
            invalid = space_group[space_group > self.num_space_groups]
            raise ValueError(
                f"space_group contains values > {self.num_space_groups}: {invalid.tolist()}"
            )
        
        # Get embeddings
        emb = self.embedding(space_group)
        
        # Project through MLP
        emb = self.projection(emb)
        
        return emb
    
    def get_embedding_table(self) -> torch.Tensor:
        """
        Get the raw embedding table for inspection or initialization.
        
        Returns:
            embeddings: [num_space_groups + 1, embedding_dim]
                       Index 0 is padding (zeros)
                       Indices 1-230 are space group embeddings
        """
        return self.embedding.weight.data
