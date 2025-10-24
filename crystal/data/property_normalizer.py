"""
Property Normalizer Utility

Handles normalization and denormalization of property values.
Provides save/load functionality for persistence.

This module implements the PropertyNormalizer component as specified in
doc/gen_molecular_crystal/design.md
"""

import torch
import json
from pathlib import Path
from typing import Union, List, Optional


class PropertyNormalizer:
    """
    Property value normalizer with save/load functionality.
    
    Normalizes property values to have zero mean and unit variance:
        properties_norm = (properties - mean) / (std + eps)
    
    And denormalizes back to original scale:
        properties = properties_norm * std + mean
    
    Args:
        mean: [property_dim] mean values for each property
        std: [property_dim] standard deviations for each property
        property_names: Names of properties (optional, for documentation)
    
    Example:
        >>> mean = torch.tensor([2.5, 180.0])
        >>> std = torch.tensor([1.2, 50.0])
        >>> normalizer = PropertyNormalizer(mean, std, ['bandgap', 'melting_point'])
        >>> properties = torch.tensor([[2.5, 180.0], [3.7, 230.0]])
        >>> normalized = normalizer.normalize(properties)
        >>> recovered = normalizer.denormalize(normalized)
        >>> torch.allclose(properties, recovered)
        True
    """
    
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        property_names: Optional[List[str]] = None
    ):
        """
        Initialize PropertyNormalizer.
        
        Args:
            mean: [property_dim] mean values
            std: [property_dim] standard deviations
            property_names: Property names (optional)
        
        Raises:
            ValueError: If shapes don't match or std has non-positive values
        """
        if mean.shape != std.shape:
            raise ValueError(
                f"mean and std must have same shape, "
                f"got mean.shape={mean.shape}, std.shape={std.shape}"
            )
        
        if torch.any(std <= 0):
            raise ValueError(
                f"std must be positive, got std={std}"
            )
        
        self.mean = mean
        self.std = std
        self.property_names = property_names or []
        self.eps = 1e-8  # For numerical stability
        
        # Validate property_names length if provided
        if self.property_names and len(self.property_names) != len(mean):
            raise ValueError(
                f"property_names length ({len(self.property_names)}) must match "
                f"property dimension ({len(mean)})"
            )
    
    def normalize(self, properties: torch.Tensor) -> torch.Tensor:
        """
        Normalize property values.
        
        Args:
            properties: [*, property_dim] raw property values
        
        Returns:
            properties_norm: [*, property_dim] normalized values
        
        Raises:
            ValueError: If last dimension doesn't match property_dim
        """
        if properties.shape[-1] != len(self.mean):
            raise ValueError(
                f"Expected property_dim={len(self.mean)}, "
                f"got {properties.shape[-1]}"
            )
        
        # Move mean and std to same device as properties
        mean = self.mean.to(properties.device)
        std = self.std.to(properties.device)
        
        return (properties - mean) / (std + self.eps)
    
    def denormalize(self, properties_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize property values back to original scale.
        
        Args:
            properties_norm: [*, property_dim] normalized values
        
        Returns:
            properties: [*, property_dim] original scale values
        
        Raises:
            ValueError: If last dimension doesn't match property_dim
        """
        if properties_norm.shape[-1] != len(self.mean):
            raise ValueError(
                f"Expected property_dim={len(self.mean)}, "
                f"got {properties_norm.shape[-1]}"
            )
        
        # Move mean and std to same device as properties_norm
        mean = self.mean.to(properties_norm.device)
        std = self.std.to(properties_norm.device)
        
        return properties_norm * std + mean
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save normalization parameters to JSON file.
        
        Args:
            path: Path to save file
        
        Example:
            >>> normalizer.save('normalization_params.json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'property_names': self.property_names,
            'property_dim': len(self.mean)
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PropertyNormalizer':
        """
        Load normalization parameters from JSON file.
        
        Args:
            path: Path to load from
        
        Returns:
            normalizer: PropertyNormalizer instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        
        Example:
            >>> normalizer = PropertyNormalizer.load('normalization_params.json')
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Normalization file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ['mean', 'std']
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Invalid normalization file, missing fields: {missing}"
            )
        
        mean = torch.tensor(data['mean'], dtype=torch.float32)
        std = torch.tensor(data['std'], dtype=torch.float32)
        property_names = data.get('property_names', [])
        
        return cls(mean, std, property_names)
    
    def __repr__(self) -> str:
        """String representation."""
        names_str = ', '.join(self.property_names) if self.property_names else 'unnamed'
        return (
            f'PropertyNormalizer('
            f'property_dim={len(self.mean)}, '
            f'properties=[{names_str}])'
        )
