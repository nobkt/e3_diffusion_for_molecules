"""
Model layer for crystal structure generation

Modules:
- molecular_encoder: Single-molecule EGNN feature extractor
- periodic_egnn: Periodic E(3) equivariant graph neural network
- lattice_diffusion: Lattice parameter diffusion model
- crystal_dynamics: Integrated crystal dynamics model
"""

from .molecular_encoder import MolecularEncoder, create_fully_connected_edges
from .periodic_egnn import PeriodicEGNN, PeriodicEGNNLayer
from .lattice_diffusion import LatticeDiffusion
from .crystal_dynamics import CrystalDynamics

__all__ = [
    'MolecularEncoder',
    'create_fully_connected_edges',
    'PeriodicEGNN',
    'PeriodicEGNNLayer',
    'LatticeDiffusion',
    'CrystalDynamics',
]
