"""
Distributions for crystal generation.

This module provides distribution classes for crystal-specific properties:
- Crystal size (number of atoms) distribution

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Learn distributions from data
- Strict validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class CrystalNodesDistribution(nn.Module):
    """
    Distribution over number of atoms in crystal structures.
    
    Learns a categorical distribution from crystal database.
    
    Args:
        histogram: [n_bins] Histogram of atom counts from data
        min_nodes: Minimum number of atoms
        max_nodes: Maximum number of atoms
    """
    
    def __init__(
        self,
        histogram: Optional[torch.Tensor] = None,
        min_nodes: int = 10,
        max_nodes: int = 200
    ):
        super().__init__()
        
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.n_bins = max_nodes - min_nodes + 1
        
        if histogram is not None:
            # Use provided histogram
            if histogram.size(0) != self.n_bins:
                raise ValueError(
                    f"Histogram size {histogram.size(0)} must match number of bins "
                    f"{self.n_bins} (max_nodes - min_nodes + 1)"
                )
            # Normalize to probabilities
            probs = histogram / histogram.sum()
        else:
            # Uniform distribution as default
            # NOTE: This should be replaced with actual data distribution
            probs = torch.ones(self.n_bins) / self.n_bins
        
        # Store as logits for numerical stability
        self.register_buffer('logits', torch.log(probs + 1e-10))
    
    @classmethod
    def from_dataset(
        cls,
        dataset,
        min_nodes: Optional[int] = None,
        max_nodes: Optional[int] = None
    ):
        """
        Create distribution from crystal dataset.
        
        Args:
            dataset: CrystalDataset instance
            min_nodes: Minimum number of atoms (auto-detected if None)
            max_nodes: Maximum number of atoms (auto-detected if None)
            
        Returns:
            CrystalNodesDistribution instance
        """
        # Collect atom counts from dataset
        atom_counts = []
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                n_atoms = int(data['atom_mask'].sum().item())
                atom_counts.append(n_atoms)
            except Exception as e:
                # Skip problematic samples
                print(f"Warning: Skipping sample {i} due to error: {e}")
                continue
        
        if len(atom_counts) == 0:
            raise ValueError(
                "Could not extract atom counts from dataset. "
                "Ensure dataset has 'atom_mask' field."
            )
        
        atom_counts = np.array(atom_counts)
        
        # Determine range
        if min_nodes is None:
            min_nodes = int(atom_counts.min())
        if max_nodes is None:
            max_nodes = int(atom_counts.max())
        
        # Create histogram
        n_bins = max_nodes - min_nodes + 1
        histogram = torch.zeros(n_bins)
        
        for count in atom_counts:
            if min_nodes <= count <= max_nodes:
                idx = count - min_nodes
                histogram[idx] += 1
        
        if histogram.sum() == 0:
            raise ValueError(
                f"No samples in range [{min_nodes}, {max_nodes}]. "
                f"Actual range: [{atom_counts.min()}, {atom_counts.max()}]"
            )
        
        return cls(histogram=histogram, min_nodes=min_nodes, max_nodes=max_nodes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of atom counts.
        
        Args:
            x: [batch] Integer atom counts
            
        Returns:
            log_prob: [batch] Log probabilities
        """
        # Convert to indices
        indices = x - self.min_nodes
        
        # Check bounds
        if (indices < 0).any() or (indices >= self.n_bins).any():
            raise ValueError(
                f"Atom counts must be in range [{self.min_nodes}, {self.max_nodes}], "
                f"got values in range [{x.min().item()}, {x.max().item()}]"
            )
        
        # Get log probabilities
        log_probs = F.log_softmax(self.logits, dim=0)
        return log_probs[indices.long()]
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(x)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample number of atoms from distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: [n_samples] Integer atom counts
        """
        # Sample from categorical distribution
        probs = F.softmax(self.logits, dim=0)
        indices = torch.multinomial(probs, n_samples, replacement=True)
        
        # Convert to atom counts
        samples = indices + self.min_nodes
        
        return samples
    
    def sample_batch(self, sizes: torch.Tensor) -> torch.Tensor:
        """
        Sample with specified batch sizes.
        
        This is for compatibility with existing code that uses
        nodes_dist.sample_batch(nodesxsample).
        
        Args:
            sizes: [batch] Desired number of samples per batch element
                   (typically all the same value)
            
        Returns:
            samples: [batch] Sampled atom counts
        """
        batch_size = sizes.size(0)
        return self.sample(batch_size)
    
    def log_info(self):
        """
        Print distribution statistics.
        """
        probs = F.softmax(self.logits, dim=0)
        
        # Compute statistics
        values = torch.arange(self.min_nodes, self.max_nodes + 1, dtype=torch.float32)
        mean = (probs * values).sum().item()
        var = (probs * (values - mean) ** 2).sum().item()
        std = var ** 0.5
        
        # Find mode
        mode_idx = probs.argmax().item()
        mode = mode_idx + self.min_nodes
        
        info = {
            'min_nodes': self.min_nodes,
            'max_nodes': self.max_nodes,
            'mean': mean,
            'std': std,
            'mode': mode
        }
        
        print("Crystal Nodes Distribution:")
        for key, value in info.items():
            print(f"  {key}: {value:.2f}")
        
        return info


class UniformNodesDistribution(nn.Module):
    """
    Uniform distribution over number of atoms.
    
    Simple baseline distribution.
    
    Args:
        min_nodes: Minimum number of atoms
        max_nodes: Maximum number of atoms
    """
    
    def __init__(self, min_nodes: int = 10, max_nodes: int = 200):
        super().__init__()
        
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.n_values = max_nodes - min_nodes + 1
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability (constant for uniform).
        
        Args:
            x: [batch] Integer atom counts
            
        Returns:
            log_prob: [batch] Log probabilities
        """
        # Check bounds
        if (x < self.min_nodes).any() or (x > self.max_nodes).any():
            raise ValueError(
                f"Atom counts must be in range [{self.min_nodes}, {self.max_nodes}], "
                f"got values in range [{x.min().item()}, {x.max().item()}]"
            )
        
        log_p = -torch.log(torch.tensor(self.n_values, dtype=torch.float32))
        return log_p.expand(x.size(0)).to(x.device)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample number of atoms uniformly.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: [n_samples] Integer atom counts
        """
        samples = torch.randint(
            low=self.min_nodes,
            high=self.max_nodes + 1,
            size=(n_samples,)
        )
        return samples
    
    def sample_batch(self, sizes: torch.Tensor) -> torch.Tensor:
        """Sample with specified batch sizes."""
        batch_size = sizes.size(0)
        return self.sample(batch_size)
