"""
Space group embedding for conditional crystal generation.

This module provides embeddings for the 230 crystallographic space groups
to enable space-group-conditioned crystal generation.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpaceGroupEmbedding(nn.Module):
    """
    Embedding layer for crystallographic space groups.
    
    Maps space group numbers (1-230) to continuous embedding vectors.
    """
    
    def __init__(self, embedding_dim: int = 32):
        """
        Initialize space group embedding.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        super(SpaceGroupEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding table for 230 space groups (+ 1 for padding/unknown)
        self.embedding = nn.Embedding(231, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, space_group_numbers: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for space groups.
        
        Args:
            space_group_numbers: Tensor of shape (batch_size,) with space group numbers (1-230)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Clamp to valid range
        sg_clamped = torch.clamp(space_group_numbers, 1, 230)
        
        # Get embeddings
        embeddings = self.embedding(sg_clamped)
        
        return embeddings


class CrystalSystemEmbedding(nn.Module):
    """
    Embedding for crystal systems (7 systems).
    
    Crystal systems:
    1. Triclinic
    2. Monoclinic
    3. Orthorhombic
    4. Tetragonal
    5. Trigonal
    6. Hexagonal
    7. Cubic
    """
    
    # Mapping from space group to crystal system
    SPACE_GROUP_TO_SYSTEM = {
        # Triclinic (1-2)
        **{i: 1 for i in range(1, 3)},
        # Monoclinic (3-15)
        **{i: 2 for i in range(3, 16)},
        # Orthorhombic (16-74)
        **{i: 3 for i in range(16, 75)},
        # Tetragonal (75-142)
        **{i: 4 for i in range(75, 143)},
        # Trigonal (143-167)
        **{i: 5 for i in range(143, 168)},
        # Hexagonal (168-194)
        **{i: 6 for i in range(168, 195)},
        # Cubic (195-230)
        **{i: 7 for i in range(195, 231)},
    }
    
    def __init__(self, embedding_dim: int = 16):
        """
        Initialize crystal system embedding.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        super(CrystalSystemEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding table for 7 crystal systems (+ 1 for padding/unknown)
        self.embedding = nn.Embedding(8, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
    
    @staticmethod
    def space_group_to_crystal_system(space_group: int) -> int:
        """Convert space group number to crystal system number."""
        return CrystalSystemEmbedding.SPACE_GROUP_TO_SYSTEM.get(space_group, 0)
    
    def forward(self, space_group_numbers: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for crystal systems based on space groups.
        
        Args:
            space_group_numbers: Tensor of shape (batch_size,) with space group numbers
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Convert space groups to crystal systems
        crystal_systems = torch.tensor(
            [self.space_group_to_crystal_system(sg.item()) for sg in space_group_numbers],
            device=space_group_numbers.device,
            dtype=torch.long
        )
        
        # Get embeddings
        embeddings = self.embedding(crystal_systems)
        
        return embeddings


class CombinedSymmetryEmbedding(nn.Module):
    """
    Combined embedding using both space group and crystal system information.
    """
    
    def __init__(self, sg_embedding_dim: int = 32, cs_embedding_dim: int = 16):
        """
        Initialize combined symmetry embedding.
        
        Args:
            sg_embedding_dim: Space group embedding dimension
            cs_embedding_dim: Crystal system embedding dimension
        """
        super(CombinedSymmetryEmbedding, self).__init__()
        
        self.space_group_embedding = SpaceGroupEmbedding(sg_embedding_dim)
        self.crystal_system_embedding = CrystalSystemEmbedding(cs_embedding_dim)
        
        self.output_dim = sg_embedding_dim + cs_embedding_dim
        
        # Optional MLP to combine embeddings
        self.combine_mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
    
    def forward(self, space_group_numbers: torch.Tensor) -> torch.Tensor:
        """
        Compute combined symmetry embeddings.
        
        Args:
            space_group_numbers: Tensor of shape (batch_size,) with space group numbers
            
        Returns:
            embeddings: Tensor of shape (batch_size, output_dim)
        """
        # Get individual embeddings
        sg_emb = self.space_group_embedding(space_group_numbers)
        cs_emb = self.crystal_system_embedding(space_group_numbers)
        
        # Concatenate
        combined = torch.cat([sg_emb, cs_emb], dim=-1)
        
        # Apply MLP
        output = self.combine_mlp(combined)
        
        return output
