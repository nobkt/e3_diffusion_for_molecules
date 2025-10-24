"""
Extended Combined Conditioning Module

Integrates multiple conditioning types including property values.
Extends the original CombinedConditioning to support property-conditioned generation.

This module implements the ExtendedCombinedConditioning component as specified in
doc/gen_molecular_crystal/design.md

Design Principles:
- No fallback heuristics: Missing optional conditions are simply not included
- Modular design: Each conditioning type is independent
- Learned fusion: MLP combines all conditions into unified representation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class ExtendedCombinedConditioning(nn.Module):
    """
    Extended combined conditioning module with property support.
    
    Combines multiple conditioning types:
    1. Molecular conditioning (REQUIRED) - from molecular structure
    2. Space group conditioning (OPTIONAL) - crystallographic space group
    3. Density conditioning (OPTIONAL) - crystal density
    4. Property conditioning (OPTIONAL) - physical properties (NEW)
    
    All conditioning vectors are concatenated and processed through a fusion MLP
    to produce a unified conditioning vector.
    
    Args:
        molecular_conditioning: MolecularConditioning module (required)
        space_group_embedding: SpaceGroupEmbedding module (optional)
        density_conditioning: DensityConditioning module (optional)
        property_conditioning: PropertyConditioning module (optional)
        conditioning_dim: Dimension of final conditioning vector
    
    Raises:
        ValueError: If molecular_conditioning is None (required)
    
    Example:
        >>> from crystal.conditioning import (
        ...     MolecularConditioning,
        ...     PropertyConditioning,
        ...     ExtendedCombinedConditioning
        ... )
        >>> mol_cond = MolecularConditioning(molecular_feature_dim=128)
        >>> prop_cond = PropertyConditioning(['bandgap', 'melting_point'])
        >>> combined = ExtendedCombinedConditioning(
        ...     molecular_conditioning=mol_cond,
        ...     property_conditioning=prop_cond
        ... )
    """
    
    def __init__(
        self,
        molecular_conditioning: nn.Module,
        space_group_embedding: Optional[nn.Module] = None,
        density_conditioning: Optional[nn.Module] = None,
        property_conditioning: Optional[nn.Module] = None,
        conditioning_dim: int = 256
    ):
        super().__init__()
        
        # Validate required arguments
        if molecular_conditioning is None:
            raise ValueError(
                "molecular_conditioning is required (no fallback mechanism)"
            )
        
        # Store conditioning modules
        self.molecular_conditioning = molecular_conditioning
        self.space_group_embedding = space_group_embedding
        self.density_conditioning = density_conditioning
        self.property_conditioning = property_conditioning
        self.conditioning_dim = conditioning_dim
        
        # Count active conditioning types
        self.num_conditionings = self._count_conditionings()
        
        # Build fusion MLP to combine all conditioning vectors
        # Input: concatenated conditioning vectors
        # Output: unified conditioning vector
        self.combine_mlp = self._build_combine_mlp()
    
    def _count_conditionings(self) -> int:
        """
        Count number of active conditioning types.
        
        Returns:
            count: Number of conditioning modules that are not None
        """
        count = 1  # Molecular conditioning is always present
        if self.space_group_embedding is not None:
            count += 1
        if self.density_conditioning is not None:
            count += 1
        if self.property_conditioning is not None:
            count += 1
        return count
    
    def _build_combine_mlp(self) -> nn.ModuleDict:
        """
        Build MLPs for combining different numbers of conditioning vectors.
        
        We create separate MLPs for each possible number of conditionings
        (1 to num_conditionings), so we can handle cases where optional
        conditions are not provided.
        
        Returns:
            mlp_dict: nn.ModuleDict with keys '1', '2', '3', '4' for different numbers of conditions
        """
        mlp_dict = nn.ModuleDict()
        
        for n_cond in range(1, self.num_conditionings + 1):
            input_dim = self.conditioning_dim * n_cond
            hidden_dim = self.conditioning_dim * 2
            output_dim = self.conditioning_dim
            
            mlp_dict[str(n_cond)] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        return mlp_dict
    
    def forward(
        self,
        molecular_features: Dict[str, torch.Tensor],
        space_group: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine multiple conditions into unified conditioning vector.
        
        Args:
            molecular_features: Dictionary with molecular features (REQUIRED)
                Expected keys: 'global_features', optionally 'mol_size', 
                'mol_volume', 'principal_axes'
            space_group: [batch_size] space group numbers (optional)
            density: [batch_size] or [batch_size, 1] density values (optional)
            properties: [batch_size, property_dim] property values (optional)
        
        Returns:
            combined_conditioning: [batch_size, conditioning_dim] unified conditioning
        
        Raises:
            ValueError: If provided condition doesn't have corresponding module
            ValueError: If molecular_features is missing or invalid
        
        Example:
            >>> batch_size = 8
            >>> mol_features = {'global_features': torch.randn(batch_size, 128)}
            >>> properties = torch.randn(batch_size, 2)
            >>> conditioning = combined(mol_features, properties=properties)
            >>> conditioning.shape
            torch.Size([8, 256])
        """
        conditionings = []
        
        # 1. Molecular conditioning (required)
        mol_cond = self.molecular_conditioning(molecular_features)
        if mol_cond.size(-1) != self.conditioning_dim:
            raise ValueError(
                f"Molecular conditioning output dimension ({mol_cond.size(-1)}) "
                f"doesn't match conditioning_dim ({self.conditioning_dim})"
            )
        conditionings.append(mol_cond)
        
        # 2. Space group conditioning (optional)
        if space_group is not None:
            if self.space_group_embedding is None:
                raise ValueError(
                    "space_group provided but space_group_embedding is None. "
                    "Initialize ExtendedCombinedConditioning with space_group_embedding."
                )
            sg_cond = self.space_group_embedding(space_group)
            
            # Ensure dimension matches
            if sg_cond.size(-1) != self.conditioning_dim:
                # Project to correct dimension if needed
                if not hasattr(self, '_sg_projection'):
                    self._sg_projection = nn.Linear(
                        sg_cond.size(-1), 
                        self.conditioning_dim
                    ).to(sg_cond.device)
                sg_cond = self._sg_projection(sg_cond)
            
            conditionings.append(sg_cond)
        
        # 3. Density conditioning (optional)
        if density is not None:
            if self.density_conditioning is None:
                raise ValueError(
                    "density provided but density_conditioning is None. "
                    "Initialize ExtendedCombinedConditioning with density_conditioning."
                )
            
            # Ensure density has correct shape [batch_size, 1]
            if density.dim() == 1:
                density = density.unsqueeze(-1)
            
            dens_cond = self.density_conditioning(density)
            
            # Ensure dimension matches
            if dens_cond.size(-1) != self.conditioning_dim:
                if not hasattr(self, '_dens_projection'):
                    self._dens_projection = nn.Linear(
                        dens_cond.size(-1),
                        self.conditioning_dim
                    ).to(dens_cond.device)
                dens_cond = self._dens_projection(dens_cond)
            
            conditionings.append(dens_cond)
        
        # 4. Property conditioning (optional, NEW)
        if properties is not None:
            if self.property_conditioning is None:
                raise ValueError(
                    "properties provided but property_conditioning is None. "
                    "Initialize ExtendedCombinedConditioning with property_conditioning."
                )
            
            prop_cond = self.property_conditioning(properties)
            
            # Ensure dimension matches
            if prop_cond.size(-1) != self.conditioning_dim:
                if not hasattr(self, '_prop_projection'):
                    self._prop_projection = nn.Linear(
                        prop_cond.size(-1),
                        self.conditioning_dim
                    ).to(prop_cond.device)
                prop_cond = self._prop_projection(prop_cond)
            
            conditionings.append(prop_cond)
        
        # Concatenate all conditioning vectors
        combined = torch.cat(conditionings, dim=-1)
        
        # Select appropriate MLP based on number of conditionings
        n_cond = len(conditionings)
        combine_mlp = self.combine_mlp[str(n_cond)]
        
        # Pass through fusion MLP
        combined_conditioning = combine_mlp(combined)
        
        return combined_conditioning
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        active = []
        active.append('molecular')
        if self.space_group_embedding is not None:
            active.append('space_group')
        if self.density_conditioning is not None:
            active.append('density')
        if self.property_conditioning is not None:
            active.append('property')
        
        return (
            f'conditioning_dim={self.conditioning_dim}, '
            f'num_conditionings={self.num_conditionings}, '
            f'active=[{", ".join(active)}]'
        )
