"""
Density conditioning for crystal generation.

This module provides conditioning based on crystal density (g/cm³).
"""

import torch
import torch.nn as nn
from typing import Optional


class DensityConditioning(nn.Module):
    """
    Density conditioning module.
    
    Converts density values to embedding vectors for conditional generation.
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        min_density: float = 0.5,
        max_density: float = 5.0,
        use_log: bool = True
    ):
        """
        Initialize density conditioning.
        
        Args:
            embedding_dim: Dimension of output embeddings
            min_density: Minimum expected density (g/cm³)
            max_density: Maximum expected density (g/cm³)
            use_log: Whether to use log-scale for density
        """
        super(DensityConditioning, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.min_density = min_density
        self.max_density = max_density
        self.use_log = use_log
        
        # MLP to encode density
        self.density_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def normalize_density(self, density: torch.Tensor) -> torch.Tensor:
        """
        Normalize density to [0, 1] range.
        
        Args:
            density: Tensor of shape (...,) with density values
            
        Returns:
            normalized: Tensor of shape (...,) with normalized values
        """
        if self.use_log:
            # Log-scale normalization
            log_density = torch.log(torch.clamp(density, min=1e-6))
            log_min = torch.log(torch.tensor(self.min_density))
            log_max = torch.log(torch.tensor(self.max_density))
            normalized = (log_density - log_min) / (log_max - log_min)
        else:
            # Linear normalization
            normalized = (density - self.min_density) / (self.max_density - self.min_density)
        
        # Clamp to [0, 1]
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        return normalized
    
    def denormalize_density(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize density from [0, 1] range.
        
        Args:
            normalized: Tensor of shape (...,) with normalized values
            
        Returns:
            density: Tensor of shape (...,) with density values
        """
        if self.use_log:
            # Log-scale denormalization
            log_min = torch.log(torch.tensor(self.min_density))
            log_max = torch.log(torch.tensor(self.max_density))
            log_density = normalized * (log_max - log_min) + log_min
            density = torch.exp(log_density)
        else:
            # Linear denormalization
            density = normalized * (self.max_density - self.min_density) + self.min_density
        
        return density
    
    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """
        Compute density embeddings.
        
        Args:
            density: Tensor of shape (batch_size,) with density values (g/cm³)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Normalize density
        density_norm = self.normalize_density(density)
        
        # Expand dimensions for MLP
        density_input = density_norm.unsqueeze(-1)  # (batch_size, 1)
        
        # Encode
        embeddings = self.density_encoder(density_input)
        
        return embeddings


class VolumeConditioning(nn.Module):
    """
    Volume conditioning module.
    
    Similar to density conditioning but for cell volume (Ų).
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        min_volume: float = 100.0,
        max_volume: float = 10000.0,
        use_log: bool = True
    ):
        """
        Initialize volume conditioning.
        
        Args:
            embedding_dim: Dimension of output embeddings
            min_volume: Minimum expected volume (Ų)
            max_volume: Maximum expected volume (Ų)
            use_log: Whether to use log-scale for volume
        """
        super(VolumeConditioning, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.use_log = use_log
        
        # MLP to encode volume
        self.volume_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Normalize volume to [0, 1] range."""
        if self.use_log:
            log_volume = torch.log(torch.clamp(volume, min=1e-6))
            log_min = torch.log(torch.tensor(self.min_volume))
            log_max = torch.log(torch.tensor(self.max_volume))
            normalized = (log_volume - log_min) / (log_max - log_min)
        else:
            normalized = (volume - self.min_volume) / (self.max_volume - self.min_volume)
        
        return torch.clamp(normalized, 0.0, 1.0)
    
    def denormalize_volume(self, normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize volume from [0, 1] range."""
        if self.use_log:
            log_min = torch.log(torch.tensor(self.min_volume))
            log_max = torch.log(torch.tensor(self.max_volume))
            log_volume = normalized * (log_max - log_min) + log_min
            volume = torch.exp(log_volume)
        else:
            volume = normalized * (self.max_volume - self.min_volume) + self.min_volume
        
        return volume
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Compute volume embeddings.
        
        Args:
            volume: Tensor of shape (batch_size,) with volume values (Ų)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Normalize volume
        volume_norm = self.normalize_volume(volume)
        
        # Expand dimensions for MLP
        volume_input = volume_norm.unsqueeze(-1)  # (batch_size, 1)
        
        # Encode
        embeddings = self.volume_encoder(volume_input)
        
        return embeddings


class CombinedPropertyConditioning(nn.Module):
    """
    Combined conditioning on multiple properties (density, volume, etc.).
    """
    
    def __init__(
        self,
        density_dim: int = 32,
        volume_dim: int = 32,
        use_density: bool = True,
        use_volume: bool = False
    ):
        """
        Initialize combined property conditioning.
        
        Args:
            density_dim: Density embedding dimension
            volume_dim: Volume embedding dimension
            use_density: Whether to use density conditioning
            use_volume: Whether to use volume conditioning
        """
        super(CombinedPropertyConditioning, self).__init__()
        
        self.use_density = use_density
        self.use_volume = use_volume
        
        if use_density:
            self.density_cond = DensityConditioning(density_dim)
        if use_volume:
            self.volume_cond = VolumeConditioning(volume_dim)
        
        # Compute output dimension
        self.output_dim = 0
        if use_density:
            self.output_dim += density_dim
        if use_volume:
            self.output_dim += volume_dim
        
        # Combine embeddings
        if self.output_dim > 0:
            self.combine_mlp = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                nn.SiLU(),
                nn.Linear(self.output_dim, self.output_dim)
            )
    
    def forward(
        self,
        density: Optional[torch.Tensor] = None,
        volume: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined property embeddings.
        
        Args:
            density: Tensor of shape (batch_size,) with density values
            volume: Tensor of shape (batch_size,) with volume values
            
        Returns:
            embeddings: Tensor of shape (batch_size, output_dim)
        """
        embeddings_list = []
        
        if self.use_density and density is not None:
            density_emb = self.density_cond(density)
            embeddings_list.append(density_emb)
        
        if self.use_volume and volume is not None:
            volume_emb = self.volume_cond(volume)
            embeddings_list.append(volume_emb)
        
        if len(embeddings_list) == 0:
            raise ValueError("No properties provided for conditioning")
        
        # Concatenate embeddings
        combined = torch.cat(embeddings_list, dim=-1)
        
        # Apply MLP
        output = self.combine_mlp(combined)
        
        return output
