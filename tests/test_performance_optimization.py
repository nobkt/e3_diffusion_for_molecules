"""
Tests for Performance Optimization Module

Tests all performance optimization components without using any heuristic processing.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crystal.evaluation.performance_optimization import (
    ConditioningCache,
    MultiGPUGenerator,
    BatchProcessor,
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self._init_params = {'hidden_dim': hidden_dim}
    
    def forward(self, x):
        return self.linear(x)


class TestConditioningCache:
    """Test ConditioningCache class."""
    
    def test_init_valid(self):
        """Test initialization with valid inputs."""
        cache = ConditioningCache(max_size=100, device='cpu')
        assert cache.max_size == 100
        assert cache.device == 'cpu'
        assert len(cache) == 0
    
    def test_init_invalid_max_size(self):
        """Test initialization with invalid max_size."""
        with pytest.raises(ValueError, match="must be positive"):
            ConditioningCache(max_size=0)
        
        with pytest.raises(ValueError, match="must be positive"):
            ConditioningCache(max_size=-1)
    
    def test_init_invalid_device(self):
        """Test initialization with invalid device."""
        with pytest.raises(ValueError, match="must be 'cpu' or 'cuda"):
            ConditioningCache(device='invalid')
    
    def test_get_or_compute_miss(self):
        """Test cache miss scenario."""
        cache = ConditioningCache(max_size=10)
        
        compute_count = [0]
        
        def compute_fn():
            compute_count[0] += 1
            return torch.randn(1, 32)
        
        key = "test_key_1"
        result1 = cache.get_or_compute(key, compute_fn)
        
        assert compute_count[0] == 1
        assert isinstance(result1, torch.Tensor)
        assert len(cache) == 1
        
        stats = cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0
    
    def test_get_or_compute_hit(self):
        """Test cache hit scenario."""
        cache = ConditioningCache(max_size=10)
        
        compute_count = [0]
        
        def compute_fn():
            compute_count[0] += 1
            return torch.randn(1, 32)
        
        key = "test_key_1"
        result1 = cache.get_or_compute(key, compute_fn)
        result2 = cache.get_or_compute(key, compute_fn)
        
        assert compute_count[0] == 1  # Only computed once
        assert torch.equal(result1, result2)
        assert len(cache) == 1
        
        stats = cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ConditioningCache(max_size=3)
        
        # Add 3 items
        for i in range(3):
            cache.get_or_compute(
                f"key_{i}",
                lambda i=i: torch.tensor([float(i)])
            )
        
        assert len(cache) == 3
        
        # Add 4th item - should evict least recently used (key_0)
        cache.get_or_compute("key_3", lambda: torch.tensor([3.0]))
        
        assert len(cache) == 3
        
        # Access key_0 should miss
        stats_before = cache.get_stats()
        cache.get_or_compute("key_0", lambda: torch.tensor([0.0]))
        stats_after = cache.get_stats()
        
        assert stats_after['misses'] == stats_before['misses'] + 1
    
    def test_lru_ordering(self):
        """Test that accessing item moves it to end (most recent)."""
        cache = ConditioningCache(max_size=3)
        
        # Add 3 items
        cache.get_or_compute("key_0", lambda: torch.tensor([0.0]))
        cache.get_or_compute("key_1", lambda: torch.tensor([1.0]))
        cache.get_or_compute("key_2", lambda: torch.tensor([2.0]))
        
        # Access key_0 to make it most recent
        cache.get_or_compute("key_0", lambda: torch.tensor([0.0]))
        
        # Add new item - should evict key_1 (now least recent)
        cache.get_or_compute("key_3", lambda: torch.tensor([3.0]))
        
        # key_0 should still be in cache (hit)
        stats_before = cache.get_stats()
        cache.get_or_compute("key_0", lambda: torch.tensor([0.0]))
        stats_after = cache.get_stats()
        
        assert stats_after['hits'] == stats_before['hits'] + 1
    
    def test_empty_key(self):
        """Test with empty key."""
        cache = ConditioningCache()
        
        with pytest.raises(ValueError, match="must be non-empty"):
            cache.get_or_compute("", lambda: torch.tensor([1.0]))
    
    def test_none_compute_fn(self):
        """Test with None compute function."""
        cache = ConditioningCache()
        
        with pytest.raises(ValueError, match="cannot be None"):
            cache.get_or_compute("key", None)
    
    def test_compute_fn_failure(self):
        """Test compute function that raises exception."""
        cache = ConditioningCache()
        
        def failing_fn():
            raise RuntimeError("Compute failed")
        
        with pytest.raises(ValueError, match="compute_fn failed"):
            cache.get_or_compute("key", failing_fn)
    
    def test_compute_fn_invalid_return(self):
        """Test compute function that returns non-tensor."""
        cache = ConditioningCache()
        
        with pytest.raises(ValueError, match="must return torch.Tensor"):
            cache.get_or_compute("key", lambda: [1, 2, 3])
    
    def test_compute_fn_non_float_tensor(self):
        """Test compute function that returns non-floating tensor."""
        cache = ConditioningCache()
        
        with pytest.raises(ValueError, match="must be floating point"):
            cache.get_or_compute("key", lambda: torch.tensor([1, 2, 3], dtype=torch.int64))
    
    def test_clear(self):
        """Test clearing cache."""
        cache = ConditioningCache(max_size=10)
        
        cache.get_or_compute("key_1", lambda: torch.randn(1, 32))
        cache.get_or_compute("key_2", lambda: torch.randn(1, 32))
        
        assert len(cache) == 2
        
        cache.clear()
        
        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = ConditioningCache(max_size=10)
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['max_size'] == 10
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['total_accesses'] == 0
        assert stats['hit_rate'] == 0.0
        
        cache.get_or_compute("key_1", lambda: torch.randn(1, 32))
        cache.get_or_compute("key_1", lambda: torch.randn(1, 32))
        
        stats = cache.get_stats()
        assert stats['size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_repr(self):
        """Test string representation."""
        cache = ConditioningCache(max_size=10)
        repr_str = repr(cache)
        assert 'ConditioningCache' in repr_str
        assert '0/10' in repr_str


class TestMultiGPUGenerator:
    """Test MultiGPUGenerator class."""
    
    def test_init_no_cuda(self):
        """Test initialization when CUDA is not available."""
        if not torch.cuda.is_available():
            model = MockModel()
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                MultiGPUGenerator(model)
        else:
            pytest.skip("CUDA is available, skipping this test")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_valid(self):
        """Test initialization with valid inputs."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=1)
        assert generator.n_gpus == 1
        assert len(generator.models) == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_auto_detect_gpus(self):
        """Test automatic GPU detection."""
        model = MockModel()
        generator = MultiGPUGenerator(model)
        assert generator.n_gpus == torch.cuda.device_count()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_invalid_n_gpus(self):
        """Test initialization with invalid n_gpus."""
        model = MockModel()
        
        with pytest.raises(ValueError, match="must be positive"):
            MultiGPUGenerator(model, n_gpus=0)
        
        with pytest.raises(ValueError, match="must be positive"):
            MultiGPUGenerator(model, n_gpus=-1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_too_many_gpus(self):
        """Test initialization with more GPUs than available."""
        model = MockModel()
        available_gpus = torch.cuda.device_count()
        
        with pytest.raises(ValueError, match="only .* available"):
            MultiGPUGenerator(model, n_gpus=available_gpus + 10)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device(self):
        """Test getting device string."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=1)
        
        device = generator.get_device(0)
        assert device == 'cuda:0'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_invalid_id(self):
        """Test getting device with invalid ID."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=1)
        
        with pytest.raises(ValueError, match="must be in"):
            generator.get_device(1)
        
        with pytest.raises(ValueError, match="must be in"):
            generator.get_device(-1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distribute_batch(self):
        """Test batch distribution across GPUs."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=2)
        
        # Even distribution
        sizes = generator.distribute_batch(10)
        assert sizes == [5, 5]
        
        # Uneven distribution
        sizes = generator.distribute_batch(11)
        assert sizes == [6, 5]
        assert sum(sizes) == 11
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distribute_batch_invalid(self):
        """Test batch distribution with invalid batch size."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=1)
        
        with pytest.raises(ValueError, match="must be positive"):
            generator.distribute_batch(0)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=1)
        
        stats = generator.get_memory_stats()
        
        assert len(stats) == 1
        assert 'gpu_id' in stats[0]
        assert 'allocated_bytes' in stats[0]
        assert 'reserved_bytes' in stats[0]
        assert 'allocated_mb' in stats[0]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_repr(self):
        """Test string representation."""
        model = MockModel()
        generator = MultiGPUGenerator(model, n_gpus=1)
        repr_str = repr(generator)
        assert 'MultiGPUGenerator' in repr_str
        assert 'n_gpus=1' in repr_str


class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_init_valid(self):
        """Test initialization with valid inputs."""
        processor = BatchProcessor(batch_size=32, device='cpu')
        assert processor.batch_size == 32
        assert processor.device == 'cpu'
    
    def test_init_invalid_batch_size(self):
        """Test initialization with invalid batch size."""
        with pytest.raises(ValueError, match="must be positive"):
            BatchProcessor(batch_size=0)
        
        with pytest.raises(ValueError, match="must be positive"):
            BatchProcessor(batch_size=-1)
    
    def test_init_invalid_device(self):
        """Test initialization with invalid device."""
        with pytest.raises(ValueError, match="must be 'cpu' or 'cuda"):
            BatchProcessor(device='invalid')
    
    def test_create_batches_even(self):
        """Test creating batches with even division."""
        processor = BatchProcessor(batch_size=10)
        
        batch_sizes = processor.create_batches(30)
        assert batch_sizes == [10, 10, 10]
        assert sum(batch_sizes) == 30
    
    def test_create_batches_uneven(self):
        """Test creating batches with uneven division."""
        processor = BatchProcessor(batch_size=10)
        
        batch_sizes = processor.create_batches(35)
        assert batch_sizes == [10, 10, 10, 5]
        assert sum(batch_sizes) == 35
    
    def test_create_batches_small(self):
        """Test creating batches smaller than batch size."""
        processor = BatchProcessor(batch_size=10)
        
        batch_sizes = processor.create_batches(5)
        assert batch_sizes == [5]
        assert sum(batch_sizes) == 5
    
    def test_create_batches_invalid(self):
        """Test creating batches with invalid total size."""
        processor = BatchProcessor(batch_size=10)
        
        with pytest.raises(ValueError, match="must be positive"):
            processor.create_batches(0)
        
        with pytest.raises(ValueError, match="must be positive"):
            processor.create_batches(-1)
    
    def test_repr(self):
        """Test string representation."""
        processor = BatchProcessor(batch_size=32, device='cpu')
        repr_str = repr(processor)
        assert 'BatchProcessor' in repr_str
        assert 'batch_size=32' in repr_str
        assert 'device=cpu' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
