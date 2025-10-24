"""
Advanced Visualization Module

Provides comprehensive visualization tools for crystal properties and structure quality.

Design Principles:
- No heuristic processing or fallback mechanisms
- High-quality publication-ready figures
- Interactive HTML reports with Plotly
- Modular architecture for extensibility

This implements Component P4-3 (Advanced Visualization) from Phase 4.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Tuple
import warnings
import json
from pathlib import Path


class PropertyDistributionAnalyzer:
    """
    Analyze and visualize property distributions of generated crystals.
    
    Creates publication-quality plots comparing generated, training, and target
    property distributions.
    
    Args:
        property_names: List of property names to analyze
        
    Raises:
        ValueError: If property_names is empty
        
    Example:
        >>> analyzer = PropertyDistributionAnalyzer(['bandgap', 'melting_point'])
        >>> analyzer.plot_distributions(
        ...     generated_properties={'bandgap': [2.1, 2.5, 2.3]},
        ...     target_properties={'bandgap': 2.5},
        ...     output_path='distributions.png'
        ... )
    """
    
    def __init__(self, property_names: List[str]):
        if not property_names:
            raise ValueError("property_names must be non-empty")
        
        self.property_names = property_names
    
    def plot_distributions(
        self,
        generated_properties: Dict[str, List[float]],
        training_properties: Optional[Dict[str, List[float]]] = None,
        target_properties: Optional[Dict[str, float]] = None,
        output_path: str = 'property_distributions.png',
        figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 300,
    ) -> Figure:
        """
        Plot property distributions with comparisons.
        
        Args:
            generated_properties: Properties of generated crystals
            training_properties: Properties from training data (optional)
            target_properties: Target property values (optional)
            output_path: Path to save plot
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
            
        Returns:
            fig: Matplotlib figure object
            
        Raises:
            ValueError: If generated_properties is missing required properties
        """
        # Validate generated properties
        for prop_name in self.property_names:
            if prop_name not in generated_properties:
                raise ValueError(
                    f"Missing property '{prop_name}' in generated_properties"
                )
            if not generated_properties[prop_name]:
                raise ValueError(
                    f"Empty values for property '{prop_name}' in generated_properties"
                )
        
        # Determine figure size
        n_props = len(self.property_names)
        if figsize is None:
            figsize = (6 * n_props, 5)
        
        # Create figure
        fig, axes = plt.subplots(1, n_props, figsize=figsize)
        
        if n_props == 1:
            axes = [axes]
        
        for ax, prop_name in zip(axes, self.property_names):
            # Generated distribution
            gen_vals = np.array(generated_properties[prop_name])
            
            # Validate values
            if not np.all(np.isfinite(gen_vals)):
                raise ValueError(
                    f"Non-finite values in generated_properties['{prop_name}']"
                )
            
            ax.hist(
                gen_vals,
                bins=30,
                alpha=0.7,
                label='Generated',
                color='blue',
                density=True,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Training distribution
            if training_properties and prop_name in training_properties:
                train_vals = np.array(training_properties[prop_name])
                
                if not np.all(np.isfinite(train_vals)):
                    raise ValueError(
                        f"Non-finite values in training_properties['{prop_name}']"
                    )
                
                ax.hist(
                    train_vals,
                    bins=30,
                    alpha=0.5,
                    label='Training',
                    color='green',
                    density=True,
                    edgecolor='black',
                    linewidth=0.5
                )
            
            # Target value
            if target_properties and prop_name in target_properties:
                target_val = target_properties[prop_name]
                
                if not np.isfinite(target_val):
                    raise ValueError(
                        f"Non-finite target value for property '{prop_name}'"
                    )
                
                ax.axvline(
                    target_val,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label='Target'
                )
            
            ax.set_xlabel(prop_name, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{prop_name} Distribution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
        
        return fig
    
    def compute_statistics(
        self,
        properties: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive statistics for each property.
        
        Args:
            properties: Dictionary of property values
            
        Returns:
            stats: Dictionary of statistics for each property
            
        Raises:
            ValueError: If properties are missing or invalid
        """
        stats = {}
        
        for prop_name in self.property_names:
            if prop_name not in properties:
                raise ValueError(f"Missing property '{prop_name}' in properties")
            
            vals = np.array(properties[prop_name])
            
            if len(vals) == 0:
                raise ValueError(f"Empty values for property '{prop_name}'")
            
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Non-finite values in property '{prop_name}'")
            
            stats[prop_name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75)),
                'count': int(len(vals)),
            }
        
        return stats
    
    def save_statistics_json(
        self,
        properties: Dict[str, List[float]],
        output_path: str = 'property_statistics.json'
    ):
        """
        Compute and save statistics to JSON file.
        
        Args:
            properties: Dictionary of property values
            output_path: Path to save JSON file
        """
        stats = self.compute_statistics(properties)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {output_path}")


class StructureQualityAnalyzer:
    """
    Analyze structural quality of generated crystals.
    
    Provides tools for analyzing bond lengths, angles, and cell parameters
    to assess the physical realism of generated structures.
    
    No heuristic thresholds - all analysis is descriptive and data-driven.
    """
    
    def __init__(self):
        pass
    
    def analyze_bond_lengths(
        self,
        crystals: List[Dict],
        cutoff: float = 3.0,
        output_path: str = 'bond_lengths.png',
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300,
    ) -> Figure:
        """
        Analyze and plot bond length distributions.
        
        Args:
            crystals: List of crystal dictionaries with 'positions', 'cell', 'atomic_numbers'
            cutoff: Cutoff radius for neighbor search (Angstroms)
            output_path: Path to save plot
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
            
        Returns:
            fig: Matplotlib figure object
            
        Raises:
            ValueError: If crystals is empty or malformed
            ImportError: If ASE is not available
        """
        try:
            from ase import Atoms
            from ase.neighborlist import neighbor_list
        except ImportError as e:
            raise ImportError(
                "ASE is required for bond length analysis. "
                "Install with: pip install ase"
            ) from e
        
        if not crystals:
            raise ValueError("crystals must be non-empty")
        
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        
        all_bond_lengths = []
        
        for i, crystal in enumerate(crystals):
            # Validate crystal structure
            required_keys = ['positions', 'cell', 'atomic_numbers']
            for key in required_keys:
                if key not in crystal:
                    raise ValueError(
                        f"Crystal at index {i} missing required key: {key}"
                    )
            
            # Convert to ASE Atoms
            try:
                atoms = Atoms(
                    numbers=crystal['atomic_numbers'],
                    positions=crystal['positions'],
                    cell=crystal['cell'],
                    pbc=True
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create ASE Atoms from crystal at index {i}: {e}"
                ) from e
            
            # Compute neighbor list
            try:
                i_idx, j_idx, distances = neighbor_list('ijd', atoms, cutoff=cutoff)
            except Exception as e:
                raise ValueError(
                    f"Failed to compute neighbor list for crystal at index {i}: {e}"
                ) from e
            
            # Filter out self-interactions
            mask = i_idx != j_idx
            bond_lengths = distances[mask]
            
            all_bond_lengths.extend(bond_lengths)
        
        if not all_bond_lengths:
            raise ValueError("No bonds found in any crystal")
        
        all_bond_lengths = np.array(all_bond_lengths)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(
            all_bond_lengths,
            bins=50,
            alpha=0.7,
            color='blue',
            edgecolor='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Bond Length (Å)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Bond Length Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
        
        # Add statistics text
        stats_text = (
            f"Mean: {np.mean(all_bond_lengths):.3f} Å\n"
            f"Std: {np.std(all_bond_lengths):.3f} Å\n"
            f"Min: {np.min(all_bond_lengths):.3f} Å\n"
            f"Max: {np.max(all_bond_lengths):.3f} Å"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Bond length plot saved to {output_path}")
        
        return fig
    
    def analyze_cell_parameters(
        self,
        crystals: List[Dict],
        output_path: str = 'cell_parameters.png',
        figsize: Tuple[int, int] = (15, 5),
        dpi: int = 300,
    ) -> Figure:
        """
        Analyze and plot cell parameter distributions.
        
        Args:
            crystals: List of crystal dictionaries with 'cell'
            output_path: Path to save plot
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
            
        Returns:
            fig: Matplotlib figure object
            
        Raises:
            ValueError: If crystals is empty or malformed
        """
        if not crystals:
            raise ValueError("crystals must be non-empty")
        
        # Extract cell parameters
        a_lengths = []
        b_lengths = []
        c_lengths = []
        volumes = []
        
        for i, crystal in enumerate(crystals):
            if 'cell' not in crystal:
                raise ValueError(f"Crystal at index {i} missing 'cell' key")
            
            cell = np.array(crystal['cell'])
            
            if cell.shape != (3, 3):
                raise ValueError(
                    f"Crystal at index {i} has invalid cell shape: {cell.shape}. "
                    f"Expected (3, 3)"
                )
            
            # Compute cell lengths
            a = np.linalg.norm(cell[0])
            b = np.linalg.norm(cell[1])
            c = np.linalg.norm(cell[2])
            
            # Compute volume
            volume = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
            
            a_lengths.append(a)
            b_lengths.append(b)
            c_lengths.append(c)
            volumes.append(volume)
        
        # Create plot
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        datasets = [
            (a_lengths, 'a (Å)', 'a-axis Length'),
            (b_lengths, 'b (Å)', 'b-axis Length'),
            (c_lengths, 'c (Å)', 'c-axis Length'),
            (volumes, 'Volume (Å³)', 'Cell Volume'),
        ]
        
        for ax, (data, xlabel, title) in zip(axes, datasets):
            data_array = np.array(data)
            
            ax.hist(
                data_array,
                bins=30,
                alpha=0.7,
                color='blue',
                edgecolor='black',
                linewidth=0.5
            )
            
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=10)
            
            # Add statistics
            stats_text = (
                f"Mean: {np.mean(data_array):.2f}\n"
                f"Std: {np.std(data_array):.2f}"
            )
            ax.text(
                0.95, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Cell parameter plot saved to {output_path}")
        
        return fig


class HTMLReportGenerator:
    """
    Generate comprehensive HTML reports with interactive visualizations.
    
    Creates publication-quality reports using Plotly for interactive plots.
    
    Raises:
        ImportError: If plotly is not available
    """
    
    def __init__(self):
        try:
            import plotly
        except ImportError as e:
            raise ImportError(
                "Plotly is required for HTML report generation. "
                "Install with: pip install plotly"
            ) from e
    
    def generate_report(
        self,
        generated_properties: Dict[str, List[float]],
        training_properties: Optional[Dict[str, List[float]]] = None,
        target_properties: Optional[Dict[str, float]] = None,
        output_path: str = 'report.html',
        title: str = 'Crystal Generation Report',
    ):
        """
        Generate comprehensive HTML report with interactive plots.
        
        Args:
            generated_properties: Properties of generated crystals
            training_properties: Properties from training data (optional)
            target_properties: Target property values (optional)
            output_path: Path to save HTML report
            title: Report title
            
        Raises:
            ValueError: If generated_properties is empty
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if not generated_properties:
            raise ValueError("generated_properties must be non-empty")
        
        property_names = list(generated_properties.keys())
        n_props = len(property_names)
        
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=n_props,
            subplot_titles=[f'{prop} Distribution' for prop in property_names]
        )
        
        for i, prop_name in enumerate(property_names, start=1):
            gen_vals = generated_properties[prop_name]
            
            # Generated histogram
            fig.add_trace(
                go.Histogram(
                    x=gen_vals,
                    name='Generated',
                    opacity=0.7,
                    marker_color='blue',
                    nbinsx=30,
                    histnorm='probability density',
                ),
                row=1, col=i
            )
            
            # Training histogram
            if training_properties and prop_name in training_properties:
                train_vals = training_properties[prop_name]
                fig.add_trace(
                    go.Histogram(
                        x=train_vals,
                        name='Training',
                        opacity=0.5,
                        marker_color='green',
                        nbinsx=30,
                        histnorm='probability density',
                    ),
                    row=1, col=i
                )
            
            # Target line
            if target_properties and prop_name in target_properties:
                target_val = target_properties[prop_name]
                fig.add_vline(
                    x=target_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Target",
                    row=1, col=i
                )
            
            # Update axes
            fig.update_xaxes(title_text=prop_name, row=1, col=i)
            fig.update_yaxes(title_text='Density', row=1, col=i)
        
        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=500,
            width=400 * n_props,
        )
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .plot {{
            margin: 20px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .stat-label {{
            font-weight: bold;
            color: #555;
        }}
        .stat-value {{
            font-size: 1.2em;
            color: #007bff;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>Generated {len(next(iter(generated_properties.values())))} crystals</p>
        
        <h2>Property Statistics</h2>
        <div class="stats">
"""
        
        # Add statistics
        for prop_name in property_names:
            vals = np.array(generated_properties[prop_name])
            html_content += f"""
            <div class="stat-card">
                <div class="stat-label">{prop_name}</div>
                <div class="stat-value">
                    Mean: {np.mean(vals):.3f}<br>
                    Std: {np.std(vals):.3f}<br>
                    Range: [{np.min(vals):.3f}, {np.max(vals):.3f}]
                </div>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>Property Distributions</h2>
        <div id="plotly-div" class="plot"></div>
    </div>
    
    <script>
        var plotlyData = """ + fig.to_json() + """;
        Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout);
    </script>
</body>
</html>
"""
        
        # Save HTML
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
