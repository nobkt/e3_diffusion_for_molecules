"""
Property Conditioning Module

Transforms physical property values (e.g., bandgap, melting point) into conditioning vectors
for property-conditioned crystal generation.

This module implements the PropertyConditioning component as specified in
doc/gen_molecular_crystal/design.md

Design Principles:
- No fallback heuristics: All transformations are theoretically grounded
- Proper normalization: Uses training data statistics
- MLP-based transformation: Learnable property-to-condition mapping
"""

import torch
import torch.nn as nn
from typing import List, Optional
import warnings


class PropertyConditioning(nn.Module):
    """
    Physical property value to conditioning vector converter.
    
    Transforms normalized property values into conditioning vectors through
    a multi-layer perceptron (MLP). Property normalization is essential for
    stable training across different property scales.
    
    Properties are normalized using:
        properties_norm = (properties - mean) / (std + eps)
    
    where mean and std are computed from training data.
    
    Args:
        property_names: List of property names (e.g., ['bandgap', 'melting_point'])
        conditioning_dim: Output conditioning vector dimension
        hidden_dim: Hidden layer dimension in MLP
        n_layers: Number of MLP layers (2-5)
    
    Raises:
        ValueError: If property_names is empty or n_layers is out of range
    """
    
    def __init__(
        self,
        property_names: List[str],
        conditioning_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        
        # Validate parameters
        if not property_names:
            raise ValueError("property_names must be non-empty")
        if not 2 <= n_layers <= 5:
            raise ValueError(f"n_layers must be between 2 and 5, got {n_layers}")
        if conditioning_dim <= 0:
            raise ValueError(f"conditioning_dim must be positive, got {conditioning_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        # Store configuration
        self.property_names = property_names
        self.property_dim = len(property_names)
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Normalization parameters (registered as buffers for save/load)
        # These will be set by set_normalization_params() before training
        self.register_buffer(
            'property_mean',
            torch.zeros(self.property_dim)
        )
        self.register_buffer(
            'property_std',
            torch.ones(self.property_dim)
        )
        
        # Small constant for numerical stability
        self.eps = 1e-8
        
        # Build MLP for property transformation
        self.property_mlp = self._build_mlp()
    
    def _build_mlp(self) -> nn.Sequential:
        """
        Build multi-layer perceptron for property transformation.
        
        Architecture:
            [Linear(property_dim, hidden_dim), SiLU()] × (n_layers - 1)
            [Linear(hidden_dim, conditioning_dim)]
        
        Returns:
            mlp: nn.Sequential module
        """
        layers = []
        in_dim = self.property_dim
        
        for i in range(self.n_layers):
            out_dim = (
                self.hidden_dim if i < self.n_layers - 1
                else self.conditioning_dim
            )
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Add activation for all layers except the last
            if i < self.n_layers - 1:
                layers.append(nn.SiLU())
            
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _normalize(self, properties: torch.Tensor) -> torch.Tensor:
        """
        Normalize property values using training statistics.
        
        Args:
            properties: [batch_size, property_dim] raw property values
        
        Returns:
            properties_norm: [batch_size, property_dim] normalized values
        """
        return (properties - self.property_mean) / (self.property_std + self.eps)
    
    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        """
        Transform property values to conditioning vectors.
        
        Args:
            properties: [batch_size, property_dim] property values
        
        Returns:
            conditioning: [batch_size, conditioning_dim] conditioning vectors
        
        Raises:
            ValueError: If input shape doesn't match property_dim
        
        Example:
            >>> prop_cond = PropertyConditioning(['bandgap', 'melting_point'])
            >>> prop_cond.set_normalization_params(
            ...     torch.tensor([2.5, 180.0]),
            ...     torch.tensor([1.2, 50.0])
            ... )
            >>> properties = torch.tensor([[2.5, 180.0], [3.0, 200.0]])
            >>> conditioning = prop_cond(properties)
            >>> conditioning.shape
            torch.Size([2, 256])
        """
        # Validate input shape
        if properties.dim() != 2:
            raise ValueError(
                f"properties must be 2D tensor [batch_size, property_dim], "
                f"got shape {properties.shape}"
            )
        if properties.shape[-1] != self.property_dim:
            raise ValueError(
                f"Expected property_dim={self.property_dim}, "
                f"got {properties.shape[-1]}"
            )
        
        # Check if normalization parameters have been set
        # (default buffers are all zeros for mean, all ones for std)
        if torch.all(self.property_mean == 0) and torch.all(self.property_std == 1):
            warnings.warn(
                "Normalization parameters not set. "
                "Call set_normalization_params() before training. "
                "Using unnormalized properties may lead to poor performance.",
                UserWarning
            )
        
        # Normalize properties
        properties_norm = self._normalize(properties)
        
        # Transform through MLP
        conditioning = self.property_mlp(properties_norm)
        
        return conditioning
    
    def set_normalization_params(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> None:
        """
        Set normalization parameters from training data statistics.
        
        This should be called before training with statistics computed from
        the training dataset.
        
        Args:
            mean: [property_dim] mean values for each property
            std: [property_dim] standard deviation for each property
        
        Raises:
            ValueError: If shapes don't match property_dim
        
        Example:
            >>> prop_cond = PropertyConditioning(['bandgap', 'melting_point'])
            >>> mean = torch.tensor([2.5, 180.0])
            >>> std = torch.tensor([1.2, 50.0])
            >>> prop_cond.set_normalization_params(mean, std)
        """
        # Validate shapes
        if mean.shape != (self.property_dim,):
            raise ValueError(
                f"Expected mean shape ({self.property_dim},), "
                f"got {mean.shape}"
            )
        if std.shape != (self.property_dim,):
            raise ValueError(
                f"Expected std shape ({self.property_dim},), "
                f"got {std.shape}"
            )
        
        # Check for zero or negative std
        if torch.any(std <= 0):
            raise ValueError(
                "Standard deviation must be positive for all properties. "
                f"Got std={std}"
            )
        
        # Copy to buffers (convert to same device as module)
        self.property_mean.copy_(mean.to(self.property_mean.device))
        self.property_std.copy_(std.to(self.property_std.device))
    
    def get_normalization_params(self):
        """
        Get current normalization parameters.
        
        Returns:
            mean: [property_dim] mean values
            std: [property_dim] standard deviation values
        """
        return self.property_mean.clone(), self.property_std.clone()
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'property_names={self.property_names}, '
            f'property_dim={self.property_dim}, '
            f'conditioning_dim={self.conditioning_dim}, '
            f'hidden_dim={self.hidden_dim}, '
            f'n_layers={self.n_layers}'
        )
