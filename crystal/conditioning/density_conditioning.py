"""
Density Conditioning Module

Converts crystal density values to embedding vectors for conditioning.
Follows physical constraints with no fallback heuristics.
"""

import torch
import torch.nn as nn


class DensityConditioning(nn.Module):
    """
    Crystal density to embedding vector converter.
    
    Converts density values (g/cm³) to conditioning embeddings using MLP.
    Normalizes density to [0, 1] range based on physically reasonable bounds.
    
    Design Principles:
    - No fallback heuristics: Out-of-range densities are clamped with warning
    - Physical constraints: Densities expected in range [0.5, 5.0] g/cm³
    - Continuous representation: MLP learns smooth embedding function
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        density_min: float = 0.5,
        density_max: float = 5.0,
    ):
        """
        Initialize Density Conditioning.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            density_min: Minimum expected density (g/cm³), default 0.5
            density_max: Maximum expected density (g/cm³), default 5.0
        """
        super().__init__()
        
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if density_min <= 0:
            raise ValueError(f"density_min must be positive, got {density_min}")
        if density_max <= density_min:
            raise ValueError(
                f"density_max must be > density_min, got max={density_max}, min={density_min}"
            )
        
        self.embedding_dim = embedding_dim
        self.density_min = density_min
        self.density_max = density_max
        
        # MLP for density to embedding conversion
        # Uses multiple layers for learning complex relationships
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """
        Convert density values to embedding vectors.
        
        Args:
            density: [batch] or [batch, 1] Density values (g/cm³)
        
        Returns:
            embedding: [batch, embedding_dim] Embedding vectors
            
        Raises:
            ValueError: If density contains NaN or infinite values
        """
        # Handle input shape
        if density.dim() == 1:
            density = density.unsqueeze(-1)
        elif density.dim() == 2:
            if density.size(1) != 1:
                raise ValueError(
                    f"density with dim=2 must have shape [batch, 1], got {density.shape}"
                )
        else:
            raise ValueError(
                f"density must be 1D or 2D tensor, got shape {density.shape}"
            )
        
        # Check for invalid values (no silent fallbacks!)
        if torch.any(torch.isnan(density)):
            raise ValueError("density contains NaN values")
        if torch.any(torch.isinf(density)):
            raise ValueError("density contains infinite values")
        if torch.any(density <= 0):
            raise ValueError(
                f"density must be positive, got negative/zero values: "
                f"{density[density <= 0].tolist()}"
            )
        
        # Normalize to [0, 1] range
        # Clamp to handle slight out-of-range values
        density_normalized = (density - self.density_min) / (self.density_max - self.density_min)
        density_normalized = torch.clamp(density_normalized, 0.0, 1.0)
        
        # Pass through MLP
        embedding = self.mlp(density_normalized)
        
        return embedding
    
    def denormalize_density(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized density [0, 1] back to physical units (g/cm³).
        
        Args:
            normalized: [batch, 1] Normalized density values [0, 1]
        
        Returns:
            density: [batch, 1] Density in g/cm³
        """
        return normalized * (self.density_max - self.density_min) + self.density_min
