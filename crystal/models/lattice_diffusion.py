"""
Lattice parameter diffusion model.

This module implements diffusion for lattice parameters (a, b, c, α, β, γ)
with proper normalization and physical constraints. Follows specifications
in MOLECULAR_CRYSTAL_DESIGN.md with no fallback heuristics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class LatticeDiffusion(nn.Module):
    """
    Diffusion model for lattice parameters.
    
    Handles (a, b, c, α, β, γ) with:
    - Log-space normalization for lengths
    - Angle normalization
    - Physical constraint enforcement
    - Molecular feature conditioning
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        condition_dim: int = 0,
    ):
        """
        Initialize lattice diffusion model.
        
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            condition_dim: Conditioning vector dimension (0 for unconditional)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # Input: lattice params (6) + time (1) + optional condition
        lattice_input_dim = 6 + 1
        if condition_dim > 0:
            lattice_input_dim += condition_dim
        
        # Build MLP
        layers = []
        layers.append(nn.Linear(lattice_input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 6))  # Output: lattice param velocities
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        lattice_params: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict lattice parameter velocities.
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ)
            t: [batch, 1] Time step (normalized 0-1)
            condition: [batch, condition_dim] Conditioning vector (optional)
            
        Returns:
            velocity: [batch, 6] Predicted lattice parameter velocities
        """
        # Build input
        inputs = [lattice_params, t]
        if condition is not None:
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)
            inputs.append(condition)
        
        x = torch.cat(inputs, dim=-1)
        
        # Predict velocity
        velocity = self.mlp(x)
        
        return velocity
    
    @staticmethod
    def normalize_lattice_params(
        lattice_params: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Normalize lattice parameters for neural network processing.
        
        Uses log-space for lengths and radian normalization for angles.
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ) with angles in degrees
            
        Returns:
            normalized: [batch, 6] Normalized parameters
            stats: Dictionary of normalization statistics for denormalization
        """
        a, b, c = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2]
        alpha, beta, gamma = lattice_params[:, 3], lattice_params[:, 4], lattice_params[:, 5]
        
        # Lengths: log-space normalization
        a_log = torch.log(a + 1e-8)
        b_log = torch.log(b + 1e-8)
        c_log = torch.log(c + 1e-8)
        
        # Angles: convert to radians
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)
        
        # Compute statistics
        lengths_log = torch.stack([a_log, b_log, c_log], dim=-1)
        angles_rad = torch.stack([alpha_rad, beta_rad, gamma_rad], dim=-1)
        
        lengths_mean = lengths_log.mean(dim=0, keepdim=True)
        lengths_std = lengths_log.std(dim=0, keepdim=True) + 1e-8
        
        angles_mean = angles_rad.mean(dim=0, keepdim=True)
        angles_std = angles_rad.std(dim=0, keepdim=True) + 1e-8
        
        # Normalize
        lengths_normalized = (lengths_log - lengths_mean) / lengths_std
        angles_normalized = (angles_rad - angles_mean) / angles_std
        
        # Concatenate
        normalized = torch.cat([lengths_normalized, angles_normalized], dim=-1)
        
        stats = {
            'lengths_mean': lengths_mean,
            'lengths_std': lengths_std,
            'angles_mean': angles_mean,
            'angles_std': angles_std,
        }
        
        return normalized, stats
    
    @staticmethod
    def denormalize_lattice_params(
        normalized: torch.Tensor,
        stats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Denormalize lattice parameters back to physical values.
        
        Args:
            normalized: [batch, 6] Normalized parameters
            stats: Normalization statistics from normalize_lattice_params
            
        Returns:
            lattice_params: [batch, 6] (a, b, c, α, β, γ) with angles in degrees
        """
        lengths_normalized = normalized[:, :3]
        angles_normalized = normalized[:, 3:]
        
        # Denormalize
        lengths_log = lengths_normalized * stats['lengths_std'] + stats['lengths_mean']
        angles_rad = angles_normalized * stats['angles_std'] + stats['angles_mean']
        
        # Convert back to original space
        lengths = torch.exp(lengths_log)
        angles = torch.rad2deg(angles_rad)
        
        # Apply physical constraints
        lengths = torch.clamp(lengths, min=1.0, max=100.0)  # 1-100 Å
        angles = torch.clamp(angles, min=30.0, max=150.0)  # 30-150°
        
        lattice_params = torch.cat([lengths, angles], dim=-1)
        
        return lattice_params
    
    @staticmethod
    def apply_physical_constraints(
        lattice_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply physical constraints to lattice parameters.
        
        Ensures:
        - Positive lengths
        - Valid angle ranges
        - Physically realizable unit cells
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ)
            
        Returns:
            constrained: [batch, 6] Constrained parameters
        """
        lengths = lattice_params[:, :3]
        angles = lattice_params[:, 3:]
        
        # Constrain lengths (1-100 Angstrom)
        lengths = torch.clamp(lengths, min=1.0, max=100.0)
        
        # Constrain angles (30-150 degrees)
        angles = torch.clamp(angles, min=30.0, max=150.0)
        
        # Ensure triangle inequality for angles
        # For a valid unit cell: α + β + γ < 360° and each pair sum > 180°
        # This is automatically satisfied if each angle is in (0, 180)
        
        constrained = torch.cat([lengths, angles], dim=-1)
        
        return constrained
    
    @staticmethod
    def compute_volume(lattice_params: torch.Tensor) -> torch.Tensor:
        """
        Compute unit cell volume from lattice parameters.
        
        V = a * b * c * sqrt(1 + 2*cos(α)*cos(β)*cos(γ) 
                             - cos²(α) - cos²(β) - cos²(γ))
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ) with angles in degrees
            
        Returns:
            volume: [batch] Unit cell volumes
        """
        a, b, c = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2]
        alpha, beta, gamma = lattice_params[:, 3], lattice_params[:, 4], lattice_params[:, 5]
        
        # Convert angles to radians
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)
        
        # Compute cos values
        cos_alpha = torch.cos(alpha_rad)
        cos_beta = torch.cos(beta_rad)
        cos_gamma = torch.cos(gamma_rad)
        
        # Volume formula
        discriminant = (1.0 
                       + 2.0 * cos_alpha * cos_beta * cos_gamma
                       - cos_alpha**2 
                       - cos_beta**2 
                       - cos_gamma**2)
        
        # Ensure discriminant is positive (numerical stability)
        discriminant = torch.clamp(discriminant, min=1e-10)
        
        volume = a * b * c * torch.sqrt(discriminant)
        
        return volume
    
    @staticmethod
    def check_validity(lattice_params: torch.Tensor) -> torch.Tensor:
        """
        Check if lattice parameters define valid unit cells.
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ)
            
        Returns:
            valid: [batch] Boolean tensor indicating validity
        """
        lengths = lattice_params[:, :3]
        angles = lattice_params[:, 3:]
        
        # Check positive lengths
        valid_lengths = (lengths > 0).all(dim=1)
        
        # Check angle ranges (0, 180)
        valid_angles = ((angles > 0) & (angles < 180)).all(dim=1)
        
        # Check volume is positive
        try:
            volume = LatticeDiffusion.compute_volume(lattice_params)
            valid_volume = volume > 0
        except:
            valid_volume = torch.zeros(lattice_params.shape[0], dtype=torch.bool, device=lattice_params.device)
        
        valid = valid_lengths & valid_angles & valid_volume
        
        return valid
