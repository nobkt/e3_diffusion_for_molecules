"""
Tests for Neighbor List Construction

Test coverage:
- Neighbor list building for periodic systems
- Distance and vector computation
- Batch processing
- Fully connected edges
- Error handling
"""

import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from crystal.utils.neighbor_list import NeighborList, build_fully_connected_edges


class TestNeighborList(unittest.TestCase):
    """Test neighbor list functionality."""
    
    def test_initialization(self):
        """Test NeighborList initialization."""
        nl = NeighborList(cutoff=5.0)
        self.assertEqual(nl.cutoff, 5.0)
        self.assertFalse(nl.self_interaction)
        self.assertTrue(nl.strict_cutoff)
        
        nl_with_self = NeighborList(cutoff=3.0, self_interaction=True)
        self.assertTrue(nl_with_self.self_interaction)
    
    def test_initialization_error_negative_cutoff(self):
        """Test error for negative cutoff."""
        with self.assertRaises(ValueError) as ctx:
            NeighborList(cutoff=-1.0)
        self.assertIn("positive", str(ctx.exception))
    
    def test_initialization_error_zero_cutoff(self):
        """Test error for zero cutoff."""
        with self.assertRaises(ValueError) as ctx:
            NeighborList(cutoff=0.0)
        self.assertIn("positive", str(ctx.exception))
    
    def test_build_simple_cubic_no_pbc(self):
        """Test neighbor list for simple cubic without PBC."""
        # Two atoms at distance 3.0
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        
        nl = NeighborList(cutoff=4.0)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=False)
        
        # Should have edges 0->1 and 1->0 (no PBC, no self)
        self.assertEqual(edge_index.shape[1], 2)
        
        # Check edge directions
        edges_set = set((edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1]))
        self.assertIn((0, 1), edges_set)
        self.assertIn((1, 0), edges_set)
    
    def test_build_with_pbc(self):
        """Test neighbor list with periodic boundaries."""
        # Atom at corner, should connect to itself via PBC
        positions = np.array([[0.5, 0.5, 0.5]])
        
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        # With cutoff of 3.0, should see periodic images
        nl = NeighborList(cutoff=3.0, self_interaction=False)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=True)
        
        # Should have edges to periodic images
        # (not to self in central cell)
        self.assertGreater(edge_index.shape[1], 0)
        
        # All shifts should be non-zero (no self-interaction in central cell)
        for i in range(edge_shift.shape[0]):
            shift_sum = np.sum(np.abs(edge_shift[i]))
            self.assertGreater(shift_sum, 0.0)
    
    def test_build_two_atoms_pbc(self):
        """Test neighbor list for two atoms with PBC."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        nl = NeighborList(cutoff=3.0)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=True)
        
        # Should have connections in both directions for central cell
        self.assertGreater(edge_index.shape[1], 0)
    
    def test_build_cutoff_too_small(self):
        """Test neighbor list with cutoff too small to find neighbors."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ])
        
        nl = NeighborList(cutoff=1.0)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=False)
        
        # No neighbors within cutoff
        self.assertEqual(edge_index.shape[1], 0)
    
    def test_build_with_self_interaction(self):
        """Test neighbor list with self-interaction enabled."""
        positions = np.array([[0.0, 0.0, 0.0]])
        
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        nl = NeighborList(cutoff=1.0, self_interaction=True)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=False)
        
        # Should have self-edge 0->0
        self.assertEqual(edge_index.shape[1], 1)
        self.assertEqual(edge_index[0, 0], 0)
        self.assertEqual(edge_index[1, 0], 0)
    
    def test_build_anisotropic_pbc(self):
        """Test neighbor list with anisotropic PBC."""
        positions = np.array([[2.5, 2.5, 0.0]])
        
        cell_vectors = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        # Only PBC in x and y, not z
        pbc = np.array([True, True, False])
        
        nl = NeighborList(cutoff=3.0, self_interaction=False)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=pbc)
        
        # Should find periodic images in x,y but not z
        # All shifts should have shift[2] = 0
        for i in range(edge_shift.shape[0]):
            self.assertEqual(edge_shift[i, 2], 0.0)
    
    def test_compute_distances(self):
        """Test distance computation for neighbor list."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        
        nl = NeighborList(cutoff=4.0)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=False)
        
        distances = nl.compute_distances(positions, cell_vectors, edge_index, edge_shift)
        
        # All distances should be 3.0
        np.testing.assert_array_almost_equal(distances, np.full(distances.shape, 3.0))
    
    def test_compute_vectors(self):
        """Test vector computation for neighbor list."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        
        nl = NeighborList(cutoff=4.0)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=False)
        
        vectors = nl.compute_vectors(positions, cell_vectors, edge_index, edge_shift)
        
        # Check shapes
        self.assertEqual(vectors.shape[0], edge_index.shape[1])
        self.assertEqual(vectors.shape[1], 3)
        
        # Find 0->1 edge
        for i in range(edge_index.shape[1]):
            if edge_index[0, i] == 0 and edge_index[1, i] == 1:
                expected_vec = np.array([3.0, 0.0, 0.0])
                np.testing.assert_array_almost_equal(vectors[i], expected_vec)
    
    def test_update_cutoff(self):
        """Test updating cutoff distance."""
        nl = NeighborList(cutoff=5.0)
        nl.update_cutoff(7.5)
        
        self.assertEqual(nl.cutoff, 7.5)
    
    def test_update_cutoff_error(self):
        """Test error for invalid new cutoff."""
        nl = NeighborList(cutoff=5.0)
        
        with self.assertRaises(ValueError):
            nl.update_cutoff(-1.0)
    
    def test_error_cutoff_too_large(self):
        """Test error for cutoff too large relative to cell."""
        positions = np.array([[0.0, 0.0, 0.0]])
        
        # Very small cell
        cell_vectors = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        # Huge cutoff would require many periodic images
        nl = NeighborList(cutoff=50.0)
        
        with self.assertRaises(ValueError) as ctx:
            nl.build(positions, cell_vectors, pbc=True)
        self.assertIn("too large", str(ctx.exception))
    
    def test_strict_cutoff_mode(self):
        """Test strict cutoff mode (d <= cutoff)."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        
        # Cutoff exactly at distance
        nl = NeighborList(cutoff=3.0, strict_cutoff=True)
        edge_index, _ = nl.build(positions, cell_vectors, pbc=False)
        
        # Should include edge at exactly cutoff distance
        self.assertGreater(edge_index.shape[1], 0)
    
    def test_non_strict_cutoff_mode(self):
        """Test non-strict cutoff mode (d < cutoff)."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        
        cell_vectors = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        
        # Cutoff exactly at distance
        nl = NeighborList(cutoff=3.0, strict_cutoff=False)
        edge_index, _ = nl.build(positions, cell_vectors, pbc=False)
        
        # Should NOT include edge at exactly cutoff distance
        self.assertEqual(edge_index.shape[1], 0)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensor_support(self):
        """Test torch tensor support."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], dtype=torch.float32)
        
        cell_vectors = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ], dtype=torch.float32)
        
        nl = NeighborList(cutoff=4.0)
        edge_index, edge_shift = nl.build(positions, cell_vectors, pbc=False)
        
        self.assertIsInstance(edge_index, torch.Tensor)
        self.assertIsInstance(edge_shift, torch.Tensor)


class TestBuildFullyConnectedEdges(unittest.TestCase):
    """Test fully connected edge building."""
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_fully_connected_no_self(self):
        """Test fully connected edges without self-interaction."""
        edge_index = build_fully_connected_edges(3, self_interaction=False)
        
        # 3 atoms, 3*2 = 6 directed edges (no self)
        self.assertEqual(edge_index.shape, (2, 6))
        
        # Check no self-edges
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            self.assertNotEqual(src, dst)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_fully_connected_with_self(self):
        """Test fully connected edges with self-interaction."""
        edge_index = build_fully_connected_edges(3, self_interaction=True)
        
        # 3 atoms, 3*3 = 9 directed edges (including self)
        self.assertEqual(edge_index.shape, (2, 9))
        
        # Check self-edges exist
        has_self_edge = False
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src == dst:
                has_self_edge = True
                break
        self.assertTrue(has_self_edge)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_fully_connected_batch(self):
        """Test fully connected edges for batched structures."""
        edge_index = build_fully_connected_edges(3, batch_size=2, self_interaction=False)
        
        # 2 batches, 3 atoms each, 6 edges per batch = 12 total
        self.assertEqual(edge_index.shape, (2, 12))
        
        # Check indices are offset correctly
        max_idx = edge_index.max().item()
        self.assertEqual(max_idx, 5)  # 2 batches * 3 atoms - 1
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_fully_connected_single_atom(self):
        """Test fully connected for single atom."""
        edge_index = build_fully_connected_edges(1, self_interaction=False)
        
        # No edges (can't connect to self when self_interaction=False)
        self.assertEqual(edge_index.shape, (2, 0))
        
        edge_index_with_self = build_fully_connected_edges(1, self_interaction=True)
        
        # One self-edge
        self.assertEqual(edge_index_with_self.shape, (2, 1))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_fully_connected_device(self):
        """Test device placement."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        device = torch.device('cuda')
        edge_index = build_fully_connected_edges(3, device=device)
        
        self.assertEqual(edge_index.device.type, 'cuda')
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_error_non_positive_n_atoms(self):
        """Test error for non-positive n_atoms."""
        with self.assertRaises(ValueError) as ctx:
            build_fully_connected_edges(0)
        self.assertIn("positive", str(ctx.exception))
        
        with self.assertRaises(ValueError):
            build_fully_connected_edges(-1)


if __name__ == '__main__':
    unittest.main()
