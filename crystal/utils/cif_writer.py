"""
CIF File Writer for Crystal Structures

This module provides functionality to export generated crystal structures
to CIF (Crystallographic Information File) format.

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation of all inputs
- Explicit error messages for invalid data
- CIF format compliant with IUCr standards
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path


class CIFWriter:
    """
    Writer for CIF (Crystallographic Information File) format.
    
    Exports crystal structures with proper cell parameters, atomic positions,
    space group information, and symmetry data.
    """
    
    def __init__(
        self,
        dataset_info: Dict,
        precision: int = 6,
        validate: bool = True
    ):
        """
        Initialize CIF writer.
        
        Args:
            dataset_info: Dictionary containing atom_decoder and other dataset info
            precision: Number of decimal places for coordinates and cell parameters
            validate: Whether to validate crystal data before writing
            
        Raises:
            ValueError: If dataset_info is None or invalid
        """
        if dataset_info is None:
            raise ValueError("dataset_info cannot be None")
        if not isinstance(dataset_info, dict):
            raise ValueError(f"dataset_info must be a dict, got {type(dataset_info)}")
        if 'atom_decoder' not in dataset_info:
            raise ValueError("dataset_info must contain 'atom_decoder' key")
        
        self.dataset_info = dataset_info
        self.atom_decoder = dataset_info['atom_decoder']
        self.precision = precision
        self.validate = validate
    
    def write_cif(
        self,
        crystal: Dict,
        output_path: Union[str, Path],
        compound_name: str = "Generated Crystal",
        space_group_number: Optional[int] = None,
        space_group_symbol: Optional[str] = None
    ) -> None:
        """
        Write a single crystal structure to CIF file.
        
        Args:
            crystal: Dictionary containing crystal structure data with keys:
                - 'positions': [n_atoms, 3] atomic positions (Cartesian or fractional)
                - 'atom_types': [n_atoms] or [n_atoms, n_types] atom types
                - 'cell_params': [6] cell parameters (a, b, c, alpha, beta, gamma)
                - 'cell_vectors': [3, 3] cell vectors (alternative to cell_params)
                - 'fractional_positions': [n_atoms, 3] fractional coordinates (optional)
            output_path: Path to output CIF file
            compound_name: Name of the compound
            space_group_number: Space group number (1-230)
            space_group_symbol: Space group symbol (e.g., 'P 21/c')
            
        Raises:
            ValueError: If crystal data is invalid or missing required fields
            IOError: If file cannot be written
        """
        # Validate crystal data
        if self.validate:
            self._validate_crystal(crystal)
        
        # Extract and prepare data
        positions_cart, positions_frac = self._prepare_positions(crystal)
        atom_types = self._prepare_atom_types(crystal)
        cell_params = self._prepare_cell_params(crystal)
        
        # Open file and write CIF data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                self._write_header(f, compound_name)
                self._write_cell_parameters(f, cell_params)
                self._write_space_group(f, space_group_number, space_group_symbol)
                self._write_atomic_positions(f, positions_frac, atom_types)
        except IOError as e:
            raise IOError(f"Failed to write CIF file to {output_path}: {str(e)}")
    
    def write_multiple_cifs(
        self,
        crystals: List[Dict],
        output_dir: Union[str, Path],
        prefix: str = "crystal",
        space_group_numbers: Optional[List[int]] = None
    ) -> List[Path]:
        """
        Write multiple crystal structures to separate CIF files.
        
        Args:
            crystals: List of crystal structure dictionaries
            output_dir: Directory for output CIF files
            prefix: Prefix for output filenames
            space_group_numbers: Optional list of space group numbers
            
        Returns:
            List of paths to written CIF files
            
        Raises:
            ValueError: If crystals list is empty or contains invalid data
        """
        if not crystals:
            raise ValueError("crystals list cannot be empty")
        
        if space_group_numbers is not None:
            if len(space_group_numbers) != len(crystals):
                raise ValueError(
                    f"Length of space_group_numbers ({len(space_group_numbers)}) "
                    f"must match length of crystals ({len(crystals)})"
                )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        for i, crystal in enumerate(crystals):
            sg_number = space_group_numbers[i] if space_group_numbers else None
            output_path = output_dir / f"{prefix}_{i:04d}.cif"
            
            self.write_cif(
                crystal,
                output_path,
                compound_name=f"{prefix}_{i}",
                space_group_number=sg_number
            )
            output_paths.append(output_path)
        
        return output_paths
    
    def _validate_crystal(self, crystal: Dict) -> None:
        """
        Validate crystal structure data.
        
        Args:
            crystal: Crystal structure dictionary
            
        Raises:
            ValueError: If required fields are missing or data is invalid
        """
        if not isinstance(crystal, dict):
            raise ValueError(f"crystal must be a dict, got {type(crystal)}")
        
        # Check for required fields
        required_fields = ['atom_types']
        for field in required_fields:
            if field not in crystal:
                raise ValueError(f"crystal must contain '{field}' key")
        
        # Must have either positions or fractional_positions
        has_positions = 'positions' in crystal or 'fractional_positions' in crystal
        if not has_positions:
            raise ValueError(
                "crystal must contain either 'positions' or 'fractional_positions'"
            )
        
        # Must have either cell_params or cell_vectors
        has_cell = 'cell_params' in crystal or 'cell_vectors' in crystal
        if not has_cell:
            raise ValueError(
                "crystal must contain either 'cell_params' or 'cell_vectors'"
            )
        
        # Validate atom_types
        atom_types = crystal['atom_types']
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.detach().cpu()
        atom_types = np.asarray(atom_types)
        
        if len(atom_types.shape) not in [1, 2]:
            raise ValueError(
                f"atom_types must be 1D or 2D, got shape {atom_types.shape}"
            )
        
        # Check for NaN or Inf
        if 'positions' in crystal:
            positions = crystal['positions']
            if isinstance(positions, torch.Tensor):
                positions = positions.detach().cpu().numpy()
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                raise ValueError("positions contains NaN or Inf values")
        
        if 'cell_params' in crystal:
            cell_params = crystal['cell_params']
            if isinstance(cell_params, torch.Tensor):
                cell_params = cell_params.detach().cpu().numpy()
            if np.any(np.isnan(cell_params)) or np.any(np.isinf(cell_params)):
                raise ValueError("cell_params contains NaN or Inf values")
            
            # Check cell parameter bounds
            a, b, c = cell_params[0], cell_params[1], cell_params[2]
            alpha, beta, gamma = cell_params[3], cell_params[4], cell_params[5]
            
            if a <= 0 or b <= 0 or c <= 0:
                raise ValueError(
                    f"Cell lengths must be positive, got a={a}, b={b}, c={c}"
                )
            
            if alpha <= 0 or alpha >= 180 or beta <= 0 or beta >= 180 or gamma <= 0 or gamma >= 180:
                raise ValueError(
                    f"Cell angles must be in (0, 180) degrees, "
                    f"got alpha={alpha}, beta={beta}, gamma={gamma}"
                )
    
    def _prepare_positions(self, crystal: Dict) -> tuple:
        """
        Prepare Cartesian and fractional positions from crystal data.
        
        Args:
            crystal: Crystal structure dictionary
            
        Returns:
            Tuple of (positions_cart, positions_frac) as numpy arrays
        """
        # Get positions
        if 'fractional_positions' in crystal:
            positions_frac = crystal['fractional_positions']
            if isinstance(positions_frac, torch.Tensor):
                positions_frac = positions_frac.detach().cpu().numpy()
            positions_frac = np.asarray(positions_frac)
            
            # Convert to Cartesian if needed
            if 'positions' in crystal:
                positions_cart = crystal['positions']
                if isinstance(positions_cart, torch.Tensor):
                    positions_cart = positions_cart.detach().cpu().numpy()
            else:
                # Convert fractional to Cartesian
                cell_vectors = self._get_cell_vectors(crystal)
                positions_cart = positions_frac @ cell_vectors.T
        else:
            positions_cart = crystal['positions']
            if isinstance(positions_cart, torch.Tensor):
                positions_cart = positions_cart.detach().cpu().numpy()
            positions_cart = np.asarray(positions_cart)
            
            # Convert to fractional
            cell_vectors = self._get_cell_vectors(crystal)
            cell_inv = np.linalg.inv(cell_vectors.T)
            positions_frac = positions_cart @ cell_inv
        
        # Wrap fractional coordinates to [0, 1)
        positions_frac = positions_frac % 1.0
        
        return positions_cart, positions_frac
    
    def _prepare_atom_types(self, crystal: Dict) -> np.ndarray:
        """
        Prepare atom type indices from crystal data.
        
        Args:
            crystal: Crystal structure dictionary
            
        Returns:
            1D numpy array of atom type indices
        """
        atom_types = crystal['atom_types']
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.detach().cpu().numpy()
        atom_types = np.asarray(atom_types)
        
        # If one-hot encoded, convert to indices
        if len(atom_types.shape) == 2:
            atom_types = np.argmax(atom_types, axis=-1)
        
        return atom_types
    
    def _prepare_cell_params(self, crystal: Dict) -> np.ndarray:
        """
        Prepare cell parameters (a, b, c, alpha, beta, gamma).
        
        Args:
            crystal: Crystal structure dictionary
            
        Returns:
            Array of shape [6] with cell parameters
        """
        if 'cell_params' in crystal:
            cell_params = crystal['cell_params']
            if isinstance(cell_params, torch.Tensor):
                cell_params = cell_params.detach().cpu().numpy()
            return np.asarray(cell_params)
        else:
            # Compute from cell_vectors
            cell_vectors = crystal['cell_vectors']
            if isinstance(cell_vectors, torch.Tensor):
                cell_vectors = cell_vectors.detach().cpu().numpy()
            cell_vectors = np.asarray(cell_vectors)
            
            return self._cell_vectors_to_params(cell_vectors)
    
    def _get_cell_vectors(self, crystal: Dict) -> np.ndarray:
        """
        Get cell vectors from crystal data.
        
        Args:
            crystal: Crystal structure dictionary
            
        Returns:
            Array of shape [3, 3] with cell vectors
        """
        if 'cell_vectors' in crystal:
            cell_vectors = crystal['cell_vectors']
            if isinstance(cell_vectors, torch.Tensor):
                cell_vectors = cell_vectors.detach().cpu().numpy()
            return np.asarray(cell_vectors)
        else:
            # Compute from cell_params
            cell_params = crystal['cell_params']
            if isinstance(cell_params, torch.Tensor):
                cell_params = cell_params.detach().cpu().numpy()
            cell_params = np.asarray(cell_params)
            
            return self._cell_params_to_vectors(cell_params)
    
    def _cell_params_to_vectors(self, cell_params: np.ndarray) -> np.ndarray:
        """
        Convert cell parameters to cell vectors.
        
        Args:
            cell_params: [6] array (a, b, c, alpha, beta, gamma) in Angstroms and degrees
            
        Returns:
            [3, 3] array of cell vectors
        """
        a, b, c = cell_params[0], cell_params[1], cell_params[2]
        alpha, beta, gamma = np.deg2rad(cell_params[3:6])
        
        # Convert to Cartesian representation
        # a vector along x-axis
        ax = a
        ay = 0.0
        az = 0.0
        
        # b vector in xy-plane
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = 0.0
        
        # c vector
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        
        cell_vectors = np.array([
            [ax, ay, az],
            [bx, by, bz],
            [cx, cy, cz]
        ])
        
        return cell_vectors
    
    def _cell_vectors_to_params(self, cell_vectors: np.ndarray) -> np.ndarray:
        """
        Convert cell vectors to cell parameters.
        
        Args:
            cell_vectors: [3, 3] array of cell vectors
            
        Returns:
            [6] array (a, b, c, alpha, beta, gamma) in Angstroms and degrees
        """
        a_vec = cell_vectors[0]
        b_vec = cell_vectors[1]
        c_vec = cell_vectors[2]
        
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)
        
        alpha = np.rad2deg(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
        beta = np.rad2deg(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
        gamma = np.rad2deg(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
        
        return np.array([a, b, c, alpha, beta, gamma])
    
    def _write_header(self, f, compound_name: str) -> None:
        """Write CIF file header."""
        f.write("data_crystal\n")
        f.write(f"_chemical_name_common '{compound_name}'\n")
        f.write("_audit_creation_method 'Generated by E3 Diffusion Model'\n")
        f.write("\n")
    
    def _write_cell_parameters(self, f, cell_params: np.ndarray) -> None:
        """Write cell parameters to CIF file."""
        a, b, c = cell_params[0], cell_params[1], cell_params[2]
        alpha, beta, gamma = cell_params[3], cell_params[4], cell_params[5]
        
        f.write(f"_cell_length_a {a:.{self.precision}f}\n")
        f.write(f"_cell_length_b {b:.{self.precision}f}\n")
        f.write(f"_cell_length_c {c:.{self.precision}f}\n")
        f.write(f"_cell_angle_alpha {alpha:.{self.precision}f}\n")
        f.write(f"_cell_angle_beta {beta:.{self.precision}f}\n")
        f.write(f"_cell_angle_gamma {gamma:.{self.precision}f}\n")
        f.write("\n")
    
    def _write_space_group(
        self,
        f,
        space_group_number: Optional[int],
        space_group_symbol: Optional[str]
    ) -> None:
        """Write space group information to CIF file."""
        if space_group_number is not None:
            if space_group_number < 1 or space_group_number > 230:
                raise ValueError(
                    f"space_group_number must be between 1 and 230, "
                    f"got {space_group_number}"
                )
            f.write(f"_space_group_IT_number {space_group_number}\n")
        
        if space_group_symbol is not None:
            f.write(f"_space_group_name_H-M_alt '{space_group_symbol}'\n")
        elif space_group_number == 1:
            f.write("_space_group_name_H-M_alt 'P 1'\n")
        else:
            # Don't assume - leave empty if not provided
            pass
        
        # Default symmetry if not specified
        if space_group_number is None:
            f.write("_space_group_IT_number 1\n")
            f.write("_space_group_name_H-M_alt 'P 1'\n")
        
        f.write("_symmetry_cell_setting triclinic\n")
        f.write("\n")
    
    def _write_atomic_positions(
        self,
        f,
        positions_frac: np.ndarray,
        atom_types: np.ndarray
    ) -> None:
        """Write atomic positions to CIF file."""
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_site_occupancy\n")
        
        for i, (pos, atom_idx) in enumerate(zip(positions_frac, atom_types)):
            if atom_idx < 0 or atom_idx >= len(self.atom_decoder):
                raise ValueError(
                    f"Invalid atom type index {atom_idx} at position {i}. "
                    f"Must be in range [0, {len(self.atom_decoder)})"
                )
            
            atom_symbol = self.atom_decoder[atom_idx]
            label = f"{atom_symbol}{i+1}"
            x, y, z = pos[0], pos[1], pos[2]
            
            f.write(
                f"{label} {atom_symbol} "
                f"{x:.{self.precision}f} {y:.{self.precision}f} {z:.{self.precision}f} "
                f"1.00\n"
            )
