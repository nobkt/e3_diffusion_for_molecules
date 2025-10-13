"""
Molecular Conditioning Module

Primary conditioning for crystal generation using molecular features.
Integrates EGNN features with geometric properties for homocrystal generation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MolecularConditioning(nn.Module):
    """
    Molecular feature to conditioning vector converter (PRIMARY CONDITIONING).
    
    Transforms molecular features extracted by MolecularEncoder into
    conditioning vectors for crystal generation.
    
    Combines:
    - Global molecular features (from EGNN)
    - Geometric properties (size, volume, principal axes)
    
    Design Principles:
    - No fallback heuristics: All features must be explicitly provided
    - Modular design: Geometric features are optional
    - Batch processing: Efficient tensor operations
    """
    
    def __init__(
        self,
        molecular_feature_dim: int,
        conditioning_dim: int = 256,
        use_geometry: bool = True,
    ):
        """
        Initialize Molecular Conditioning.
        
        Args:
            molecular_feature_dim: Dimension of molecular features from encoder
            conditioning_dim: Dimension of output conditioning vectors
            use_geometry: Whether to include geometric properties (size, volume, axes)
        """
        super().__init__()
        
        if molecular_feature_dim <= 0:
            raise ValueError(f"molecular_feature_dim must be positive, got {molecular_feature_dim}")
        if conditioning_dim <= 0:
            raise ValueError(f"conditioning_dim must be positive, got {conditioning_dim}")
        
        self.molecular_feature_dim = molecular_feature_dim
        self.conditioning_dim = conditioning_dim
        self.use_geometry = use_geometry
        
        # Calculate total feature dimension
        # Global features + (optional) geometry: size(3) + volume(1) + principal_axes(9)
        total_feature_dim = molecular_feature_dim
        if use_geometry:
            total_feature_dim += 13  # 3 (size) + 1 (volume) + 9 (axes)
        
        # MLP to transform features to conditioning vectors
        self.feature_mlp = nn.Sequential(
            nn.Linear(total_feature_dim, conditioning_dim * 2),
            nn.LayerNorm(conditioning_dim * 2),
            nn.SiLU(),
            nn.Linear(conditioning_dim * 2, conditioning_dim),
            nn.LayerNorm(conditioning_dim),
            nn.SiLU(),
        )
        
        # Final projection
        self.projection = nn.Linear(conditioning_dim, conditioning_dim)
    
    def forward(self, molecular_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert molecular features to conditioning vectors.
        
        Args:
            molecular_features: Dictionary containing:
                - 'global_features': [batch, molecular_feature_dim] (REQUIRED)
                - 'mol_size': [batch, 3] Molecular bounding box size (optional)
                - 'mol_volume': [batch, 1] Molecular volume (optional)
                - 'principal_axes': [batch, 3, 3] Principal axes (optional)
        
        Returns:
            conditioning: [batch, conditioning_dim] Conditioning vectors
            
        Raises:
            ValueError: If required features are missing or have incorrect shapes
        """
        # Validate required features
        if 'global_features' not in molecular_features:
            raise ValueError("molecular_features must contain 'global_features'")
        
        global_feat = molecular_features['global_features']
        
        # Validate shape
        if global_feat.dim() != 2:
            raise ValueError(
                f"global_features must be 2D tensor [batch, dim], got shape {global_feat.shape}"
            )
        if global_feat.size(1) != self.molecular_feature_dim:
            raise ValueError(
                f"global_features dimension mismatch: expected {self.molecular_feature_dim}, "
                f"got {global_feat.size(1)}"
            )
        
        batch_size = global_feat.size(0)
        features_to_concat = [global_feat]
        
        # Add geometric properties if enabled
        if self.use_geometry:
            # Check for required geometric features
            required_geom = ['mol_size', 'mol_volume', 'principal_axes']
            missing = [k for k in required_geom if k not in molecular_features]
            if missing:
                raise ValueError(
                    f"use_geometry=True but missing features: {missing}. "
                    f"Provide all of: {required_geom}"
                )
            
            # Extract and validate geometric features
            mol_size = molecular_features['mol_size']
            mol_volume = molecular_features['mol_volume']
            principal_axes = molecular_features['principal_axes']
            
            # Validate shapes
            if mol_size.shape != (batch_size, 3):
                raise ValueError(
                    f"mol_size must have shape [{batch_size}, 3], got {mol_size.shape}"
                )
            if mol_volume.shape != (batch_size, 1):
                raise ValueError(
                    f"mol_volume must have shape [{batch_size}, 1], got {mol_volume.shape}"
                )
            if principal_axes.shape != (batch_size, 3, 3):
                raise ValueError(
                    f"principal_axes must have shape [{batch_size}, 3, 3], "
                    f"got {principal_axes.shape}"
                )
            
            # Flatten principal axes: [batch, 3, 3] -> [batch, 9]
            principal_axes_flat = principal_axes.reshape(batch_size, 9)
            
            # Concatenate geometric features
            features_to_concat.extend([
                mol_size,
                mol_volume,
                principal_axes_flat,
            ])
        
        # Concatenate all features
        combined_features = torch.cat(features_to_concat, dim=-1)
        
        # Pass through MLP
        conditioning_vector = self.feature_mlp(combined_features)
        
        # Final projection
        conditioning_vector = self.projection(conditioning_vector)
        
        return conditioning_vector


class CombinedConditioning(nn.Module):
    """
    Combines multiple conditioning modules with learned weights.
    
    Integrates molecular conditioning (PRIMARY) with optional space group
    and density conditioning.
    
    Design Principles:
    - No fallback heuristics: Missing optional conditions simply not included
    - Learned fusion: Learnable weights balance different condition types
    - Flexible: Supports any combination of conditions
    """
    
    def __init__(
        self,
        molecular_conditioning: MolecularConditioning,
        conditioning_dim: int = 256,
        space_group_embedding: Optional[nn.Module] = None,
        density_conditioning: Optional[nn.Module] = None,
    ):
        """
        Initialize Combined Conditioning.
        
        Args:
            molecular_conditioning: Molecular conditioning module (REQUIRED)
            conditioning_dim: Dimension of conditioning vectors
            space_group_embedding: Optional space group embedding module
            density_conditioning: Optional density conditioning module
        """
        super().__init__()
        
        if molecular_conditioning is None:
            raise ValueError("molecular_conditioning is required (no fallback!)")
        
        self.molecular_conditioning = molecular_conditioning
        self.space_group_embedding = space_group_embedding
        self.density_conditioning = density_conditioning
        self.conditioning_dim = conditioning_dim
        
        # Count active conditioning types
        num_conditions = 1  # Molecular is always present
        if space_group_embedding is not None:
            num_conditions += 1
        if density_conditioning is not None:
            num_conditions += 1
        
        # Learnable weights for each condition type
        self.condition_weights = nn.Parameter(
            torch.ones(num_conditions) / num_conditions
        )
        
        # Fusion MLP to combine conditions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.LayerNorm(conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
    
    def forward(
        self,
        molecular_features: Dict[str, torch.Tensor],
        space_group: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine multiple conditions into unified conditioning vector.
        
        Args:
            molecular_features: Molecular features (REQUIRED)
            space_group: Optional space group numbers [batch]
            density: Optional density values [batch] or [batch, 1]
        
        Returns:
            combined_conditioning: [batch, conditioning_dim]
            
        Raises:
            ValueError: If space_group provided but no space_group_embedding
            ValueError: If density provided but no density_conditioning
        """
        # Get molecular conditioning (always present)
        mol_cond = self.molecular_conditioning(molecular_features)
        
        conditions = [mol_cond]
        
        # Add space group conditioning if provided
        if space_group is not None:
            if self.space_group_embedding is None:
                raise ValueError(
                    "space_group provided but space_group_embedding not initialized"
                )
            sg_emb = self.space_group_embedding(space_group)
            
            # Match dimensions if needed
            if sg_emb.size(-1) != self.conditioning_dim:
                # Pad or project to match conditioning_dim
                if sg_emb.size(-1) < self.conditioning_dim:
                    # Pad with zeros
                    padding = self.conditioning_dim - sg_emb.size(-1)
                    sg_emb = torch.nn.functional.pad(sg_emb, (0, padding))
                else:
                    # Project down
                    if not hasattr(self, 'sg_projection'):
                        self.sg_projection = nn.Linear(
                            sg_emb.size(-1), self.conditioning_dim
                        ).to(sg_emb.device)
                    sg_emb = self.sg_projection(sg_emb)
            
            conditions.append(sg_emb)
        
        # Add density conditioning if provided
        if density is not None:
            if self.density_conditioning is None:
                raise ValueError(
                    "density provided but density_conditioning not initialized"
                )
            dens_emb = self.density_conditioning(density)
            
            # Match dimensions if needed
            if dens_emb.size(-1) != self.conditioning_dim:
                if dens_emb.size(-1) < self.conditioning_dim:
                    # Pad with zeros
                    padding = self.conditioning_dim - dens_emb.size(-1)
                    dens_emb = torch.nn.functional.pad(dens_emb, (0, padding))
                else:
                    # Project down
                    if not hasattr(self, 'dens_projection'):
                        self.dens_projection = nn.Linear(
                            dens_emb.size(-1), self.conditioning_dim
                        ).to(dens_emb.device)
                    dens_emb = self.dens_projection(dens_emb)
            
            conditions.append(dens_emb)
        
        # Weighted combination using softmax for normalization
        weights = torch.softmax(self.condition_weights[:len(conditions)], dim=0)
        
        # Combine with learned weights
        combined = torch.zeros_like(conditions[0])
        for i, cond in enumerate(conditions):
            combined = combined + weights[i] * cond
        
        # Pass through fusion MLP
        combined_conditioning = self.fusion_mlp(combined)
        
        return combined_conditioning
