"""
Lattice parameter diffusion for crystal structure generation.

This module handles the diffusion process for lattice parameters (a, b, c, α, β, γ),
ensuring physical constraints are maintained.
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..data.periodic_utils import cell_params_to_vectors, cell_vectors_to_params


class LatticeDiffusion(nn.Module):
    """
    Diffusion model for lattice parameters.
    
    Handles the generation of lattice parameters (a, b, c, α, β, γ) with
    proper normalization and physical constraints.
    """
    
    def __init__(self, hidden_nf, context_nf=0, act_fn=nn.SiLU()):
        super(LatticeDiffusion, self).__init__()
        
        self.hidden_nf = hidden_nf
        
        # Network to predict lattice parameter updates
        # Input: 6 (current lattice params) + hidden_nf (aggregated node info) + context_nf
        input_dim = 6 + hidden_nf + context_nf
        
        self.lattice_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 6)  # Output: 6 lattice parameters
        )
    
    @staticmethod
    def normalize_lattice_params(lengths, angles):
        """
        Normalize lattice parameters for neural network input.
        
        Args:
            lengths: (batch_size, 3) - Cell lengths [a, b, c] in Angstroms
            angles: (batch_size, 3) - Cell angles [alpha, beta, gamma] in degrees
            
        Returns:
            normalized: (batch_size, 6) - Normalized parameters
        """
        # Log-transform lengths (to handle scale)
        log_lengths = torch.log(lengths + 1e-8)
        
        # Convert angles to radians and use sin/cos representation
        angles_rad = angles * torch.pi / 180.0
        angle_features = torch.cat([
            torch.sin(angles_rad),
            torch.cos(angles_rad)
        ], dim=-1)  # (batch_size, 6)
        
        # For now, use a simpler normalization
        # Just normalize lengths by mean and use angles in radians
        normalized_lengths = log_lengths
        normalized_angles = angles_rad
        
        normalized = torch.cat([normalized_lengths, normalized_angles], dim=-1)
        return normalized
    
    @staticmethod
    def denormalize_lattice_params(normalized):
        """
        Denormalize lattice parameters from neural network output.
        
        Args:
            normalized: (batch_size, 6) - Normalized parameters
            
        Returns:
            lengths: (batch_size, 3) - Cell lengths in Angstroms
            angles: (batch_size, 3) - Cell angles in degrees
        """
        # Split into lengths and angles
        log_lengths = normalized[..., :3]
        angles_rad = normalized[..., 3:]
        
        # Denormalize lengths (exp transform)
        lengths = torch.exp(log_lengths)
        
        # Clip lengths to reasonable range (1-100 Angstroms)
        lengths = torch.clamp(lengths, 1.0, 100.0)
        
        # Denormalize angles (rad to degrees)
        angles = angles_rad * 180.0 / torch.pi
        
        # Clip angles to valid range (20-160 degrees for stability)
        angles = torch.clamp(angles, 20.0, 160.0)
        
        return lengths, angles
    
    def aggregate_node_features(self, h, node_mask):
        """
        Aggregate node features to get graph-level representation.
        
        Args:
            h: Node features (batch_size * n_nodes, hidden_nf)
            node_mask: Node mask (batch_size * n_nodes, 1)
            
        Returns:
            graph_features: (batch_size, hidden_nf)
        """
        # Mask features
        h_masked = h * node_mask
        
        # Sum over nodes (this is simplified - proper batching would group by batch index)
        # For now, assume single structure
        graph_features = h_masked.sum(dim=0, keepdim=True)  # (1, hidden_nf)
        
        return graph_features
    
    def forward(self, lattice_params, h, node_mask, context=None):
        """
        Predict lattice parameter update.
        
        Args:
            lattice_params: (batch_size, 6) - Current lattice parameters (normalized)
            h: Node features (batch_size * n_nodes, hidden_nf)
            node_mask: Node mask (batch_size * n_nodes, 1)
            context: Optional context features (batch_size, context_nf)
            
        Returns:
            lattice_update: (batch_size, 6) - Predicted lattice parameter update
        """
        # Aggregate node features to graph level
        graph_features = self.aggregate_node_features(h, node_mask)
        
        # Concatenate inputs
        if context is not None:
            mlp_input = torch.cat([lattice_params, graph_features, context], dim=-1)
        else:
            mlp_input = torch.cat([lattice_params, graph_features], dim=-1)
        
        # Predict update
        lattice_update = self.lattice_mlp(mlp_input)
        
        return lattice_update


class LatticeNoise(nn.Module):
    """
    Noise model for lattice parameters during diffusion.
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=1.0):
        super(LatticeNoise, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def add_noise(self, lattice_params, t):
        """
        Add noise to lattice parameters at timestep t.
        
        Args:
            lattice_params: (batch_size, 6) - Clean lattice parameters
            t: (batch_size,) - Timestep (0 to 1)
            
        Returns:
            noisy_params: (batch_size, 6) - Noisy lattice parameters
            noise: (batch_size, 6) - Added noise
        """
        # Compute noise level based on timestep
        sigma = self.sigma_min + t.unsqueeze(-1) * (self.sigma_max - self.sigma_min)
        
        # Sample noise
        noise = torch.randn_like(lattice_params) * sigma
        
        # Add noise
        noisy_params = lattice_params + noise
        
        return noisy_params, noise
    
    def remove_noise(self, noisy_params, predicted_noise, t):
        """
        Remove predicted noise from noisy parameters.
        
        Args:
            noisy_params: (batch_size, 6) - Noisy parameters
            predicted_noise: (batch_size, 6) - Predicted noise
            t: (batch_size,) - Timestep
            
        Returns:
            denoised_params: (batch_size, 6) - Denoised parameters
        """
        sigma = self.sigma_min + t.unsqueeze(-1) * (self.sigma_max - self.sigma_min)
        denoised_params = noisy_params - predicted_noise
        return denoised_params
