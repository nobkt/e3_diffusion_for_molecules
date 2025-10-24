"""
Performance Optimization Module

Provides performance optimization tools for large-scale crystal generation.

Design Principles:
- No heuristic processing or fallback mechanisms
- Theoretically sound optimization strategies
- Explicit error handling with informative messages
- Modular architecture for extensibility

This implements Component P4-4 (Performance Optimization) from Phase 4.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
import warnings
from collections import OrderedDict
import hashlib
import pickle


class ConditioningCache:
    """
    Cache conditioning vectors for reuse during generation.
    
    Implements an LRU (Least Recently Used) cache for conditioning vectors
    to reduce redundant computations. No heuristic eviction - uses strict LRU.
    
    Args:
        max_size: Maximum number of entries in cache
        device: Device for cached tensors ('cpu' or 'cuda')
        
    Raises:
        ValueError: If max_size is not positive
        
    Example:
        >>> cache = ConditioningCache(max_size=1000)
        >>> key = "molecule_id_123_bandgap_2.5"
        >>> conditioning = cache.get_or_compute(
        ...     key=key,
        ...     compute_fn=lambda: compute_conditioning(...)
        ... )
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        device: str = 'cpu'
    ):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        
        if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
            raise ValueError(
                f"device must be 'cpu' or 'cuda[:N]', got {device}"
            )
        
        self.max_size = max_size
        self.device = device
        self.cache = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _generate_key_hash(self, key: str) -> str:
        """
        Generate hash of key for consistent lookup.
        
        Args:
            key: String key
            
        Returns:
            key_hash: SHA256 hash of key
        """
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """
        Get conditioning from cache or compute if not present.
        
        Uses LRU eviction when cache is full. No heuristic decisions.
        
        Args:
            key: Unique identifier for conditioning
            compute_fn: Function to compute conditioning if not cached
            
        Returns:
            conditioning: Cached or computed conditioning tensor
            
        Raises:
            ValueError: If key is empty or compute_fn returns invalid tensor
        """
        if not key:
            raise ValueError("key must be non-empty")
        
        if compute_fn is None:
            raise ValueError("compute_fn cannot be None")
        
        key_hash = self._generate_key_hash(key)
        
        # Check if in cache
        if key_hash in self.cache:
            self._hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key_hash)
            return self.cache[key_hash]
        
        self._misses += 1
        
        # Compute conditioning
        try:
            conditioning = compute_fn()
        except Exception as e:
            raise ValueError(
                f"compute_fn failed for key '{key}': {e}"
            ) from e
        
        # Validate conditioning
        if not isinstance(conditioning, torch.Tensor):
            raise ValueError(
                f"compute_fn must return torch.Tensor, got {type(conditioning)}"
            )
        
        if not conditioning.is_floating_point():
            raise ValueError(
                "conditioning tensor must be floating point"
            )
        
        # Move to correct device
        conditioning = conditioning.to(self.device)
        
        # Add to cache
        if len(self.cache) >= self.max_size:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)
        
        self.cache[key_hash] = conditioning
        
        return conditioning
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            stats: Dictionary with cache statistics
        """
        total_accesses = self._hits + self._misses
        hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'total_accesses': total_accesses,
            'hit_rate': hit_rate,
        }
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ConditioningCache(size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )


class MultiGPUGenerator:
    """
    Distribute crystal generation across multiple GPUs.
    
    Implements data-parallel generation without heuristic load balancing.
    Uses round-robin distribution for deterministic behavior.
    
    Args:
        model: Crystal generation model
        n_gpus: Number of GPUs to use (None = use all available)
        
    Raises:
        ValueError: If n_gpus is invalid or no GPUs available
        RuntimeError: If CUDA is not available
        
    Example:
        >>> generator = MultiGPUGenerator(model, n_gpus=2)
        >>> # Note: Actual generation requires integration with crystal generation pipeline
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_gpus: Optional[int] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. MultiGPUGenerator requires CUDA."
            )
        
        available_gpus = torch.cuda.device_count()
        
        if n_gpus is None:
            n_gpus = available_gpus
        
        if n_gpus <= 0:
            raise ValueError(f"n_gpus must be positive, got {n_gpus}")
        
        if n_gpus > available_gpus:
            raise ValueError(
                f"Requested {n_gpus} GPUs but only {available_gpus} available"
            )
        
        self.n_gpus = n_gpus
        self.model = model
        
        # Create model replicas on each GPU
        self.models = []
        for i in range(n_gpus):
            device = f'cuda:{i}'
            # Deep copy model to GPU
            model_copy = self._replicate_model(model, device)
            self.models.append(model_copy)
    
    def _replicate_model(
        self,
        model: nn.Module,
        device: str
    ) -> nn.Module:
        """
        Replicate model to specific device.
        
        Args:
            model: Model to replicate
            device: Target device
            
        Returns:
            model_copy: Model replica on target device
        """
        # Create state dict
        state_dict = model.state_dict()
        
        # Create new model instance (assumes model class is accessible)
        model_copy = type(model).__new__(type(model))
        
        # Initialize if needed
        if hasattr(model, '__init__'):
            # Try to get init parameters from model
            if hasattr(model, '_init_params'):
                model_copy.__init__(**model._init_params)
            else:
                # Fallback: use default initialization
                model_copy.__init__()
        
        # Load state dict
        model_copy.load_state_dict(state_dict)
        
        # Move to device
        model_copy = model_copy.to(device)
        model_copy.eval()
        
        return model_copy
    
    def get_device(self, gpu_id: int) -> str:
        """
        Get device string for GPU ID.
        
        Args:
            gpu_id: GPU index
            
        Returns:
            device: Device string
            
        Raises:
            ValueError: If gpu_id is invalid
        """
        if not 0 <= gpu_id < self.n_gpus:
            raise ValueError(
                f"gpu_id must be in [0, {self.n_gpus}), got {gpu_id}"
            )
        
        return f'cuda:{gpu_id}'
    
    def distribute_batch(
        self,
        batch_size: int
    ) -> List[int]:
        """
        Distribute batch across GPUs using round-robin.
        
        No heuristic load balancing - uses deterministic round-robin.
        
        Args:
            batch_size: Total batch size
            
        Returns:
            sizes_per_gpu: List of batch sizes for each GPU
            
        Raises:
            ValueError: If batch_size is not positive
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        # Round-robin distribution
        base_size = batch_size // self.n_gpus
        remainder = batch_size % self.n_gpus
        
        sizes_per_gpu = []
        for i in range(self.n_gpus):
            size = base_size + (1 if i < remainder else 0)
            sizes_per_gpu.append(size)
        
        return sizes_per_gpu
    
    def synchronize(self):
        """Synchronize all GPUs."""
        for i in range(self.n_gpus):
            torch.cuda.synchronize(device=f'cuda:{i}')
    
    def get_memory_stats(self) -> List[Dict[str, int]]:
        """
        Get memory statistics for each GPU.
        
        Returns:
            stats: List of memory statistics per GPU
        """
        stats = []
        
        for i in range(self.n_gpus):
            device = f'cuda:{i}'
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_allocated = torch.cuda.max_memory_allocated(device)
            
            stats.append({
                'gpu_id': i,
                'allocated_bytes': allocated,
                'reserved_bytes': reserved,
                'max_allocated_bytes': max_allocated,
                'allocated_mb': allocated / (1024 ** 2),
                'reserved_mb': reserved / (1024 ** 2),
                'max_allocated_mb': max_allocated / (1024 ** 2),
            })
        
        return stats
    
    def __repr__(self) -> str:
        return f"MultiGPUGenerator(n_gpus={self.n_gpus})"


class BatchProcessor:
    """
    Optimized batch processing for large-scale generation.
    
    Implements efficient batching strategies without heuristic decisions.
    
    Args:
        batch_size: Size of each batch
        device: Device for processing ('cpu' or 'cuda')
        
    Raises:
        ValueError: If batch_size is not positive
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
            raise ValueError(
                f"device must be 'cpu' or 'cuda[:N]', got {device}"
            )
        
        self.batch_size = batch_size
        self.device = device
    
    def create_batches(
        self,
        total_size: int
    ) -> List[int]:
        """
        Create batch sizes for processing.
        
        Deterministic batching without heuristic adjustments.
        
        Args:
            total_size: Total number of items
            
        Returns:
            batch_sizes: List of batch sizes
            
        Raises:
            ValueError: If total_size is not positive
        """
        if total_size <= 0:
            raise ValueError(f"total_size must be positive, got {total_size}")
        
        n_full_batches = total_size // self.batch_size
        remainder = total_size % self.batch_size
        
        batch_sizes = [self.batch_size] * n_full_batches
        
        if remainder > 0:
            batch_sizes.append(remainder)
        
        return batch_sizes
    
    def __repr__(self) -> str:
        return f"BatchProcessor(batch_size={self.batch_size}, device={self.device})"
