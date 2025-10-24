"""
Tests for Advanced Visualization Module

Tests all visualization components without using any heuristic processing.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crystal.evaluation.advanced_visualization import (
    PropertyDistributionAnalyzer,
    StructureQualityAnalyzer,
    HTMLReportGenerator,
)


class TestPropertyDistributionAnalyzer:
    """Test PropertyDistributionAnalyzer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup temporary files."""
        shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_init_valid(self):
        """Test initialization with valid inputs."""
        analyzer = PropertyDistributionAnalyzer(['bandgap', 'melting_point'])
        assert analyzer.property_names == ['bandgap', 'melting_point']
    
    def test_init_empty_properties(self):
        """Test initialization with empty property list."""
        with pytest.raises(ValueError, match="must be non-empty"):
            PropertyDistributionAnalyzer([])
    
    def test_plot_distributions_basic(self):
        """Test basic distribution plotting."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        generated_props = {
            'bandgap': [2.1, 2.3, 2.5, 2.4, 2.2]
        }
        
        output_path = Path(self.temp_dir) / 'test_dist.png'
        fig = analyzer.plot_distributions(
            generated_properties=generated_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert fig is not None
    
    def test_plot_distributions_with_training(self):
        """Test distribution plotting with training data."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        generated_props = {'bandgap': [2.1, 2.3, 2.5]}
        training_props = {'bandgap': [2.0, 2.2, 2.4, 2.6]}
        
        output_path = Path(self.temp_dir) / 'test_dist_train.png'
        fig = analyzer.plot_distributions(
            generated_properties=generated_props,
            training_properties=training_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_plot_distributions_with_target(self):
        """Test distribution plotting with target values."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        generated_props = {'bandgap': [2.1, 2.3, 2.5]}
        target_props = {'bandgap': 2.5}
        
        output_path = Path(self.temp_dir) / 'test_dist_target.png'
        fig = analyzer.plot_distributions(
            generated_properties=generated_props,
            target_properties=target_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_plot_distributions_multiple_properties(self):
        """Test plotting with multiple properties."""
        analyzer = PropertyDistributionAnalyzer(['bandgap', 'melting_point'])
        
        generated_props = {
            'bandgap': [2.1, 2.3, 2.5],
            'melting_point': [150.0, 175.0, 200.0]
        }
        
        output_path = Path(self.temp_dir) / 'test_dist_multi.png'
        fig = analyzer.plot_distributions(
            generated_properties=generated_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_plot_distributions_missing_property(self):
        """Test plotting with missing property."""
        analyzer = PropertyDistributionAnalyzer(['bandgap', 'melting_point'])
        
        generated_props = {'bandgap': [2.1, 2.3, 2.5]}
        
        output_path = Path(self.temp_dir) / 'test_dist_missing.png'
        with pytest.raises(ValueError, match="Missing property"):
            analyzer.plot_distributions(
                generated_properties=generated_props,
                output_path=str(output_path)
            )
    
    def test_plot_distributions_empty_values(self):
        """Test plotting with empty values."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        generated_props = {'bandgap': []}
        
        output_path = Path(self.temp_dir) / 'test_dist_empty.png'
        with pytest.raises(ValueError, match="Empty values"):
            analyzer.plot_distributions(
                generated_properties=generated_props,
                output_path=str(output_path)
            )
    
    def test_plot_distributions_non_finite(self):
        """Test plotting with non-finite values."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        generated_props = {'bandgap': [2.1, float('nan'), 2.3]}
        
        output_path = Path(self.temp_dir) / 'test_dist_nan.png'
        with pytest.raises(ValueError, match="Non-finite"):
            analyzer.plot_distributions(
                generated_properties=generated_props,
                output_path=str(output_path)
            )
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        properties = {'bandgap': [2.0, 2.5, 3.0, 2.5, 2.0]}
        
        stats = analyzer.compute_statistics(properties)
        
        assert 'bandgap' in stats
        assert 'mean' in stats['bandgap']
        assert 'std' in stats['bandgap']
        assert 'min' in stats['bandgap']
        assert 'max' in stats['bandgap']
        assert 'median' in stats['bandgap']
        assert 'q25' in stats['bandgap']
        assert 'q75' in stats['bandgap']
        assert 'count' in stats['bandgap']
        
        assert stats['bandgap']['mean'] == pytest.approx(2.4)
        assert stats['bandgap']['min'] == 2.0
        assert stats['bandgap']['max'] == 3.0
        assert stats['bandgap']['count'] == 5
    
    def test_compute_statistics_missing_property(self):
        """Test statistics computation with missing property."""
        analyzer = PropertyDistributionAnalyzer(['bandgap', 'melting_point'])
        
        properties = {'bandgap': [2.0, 2.5, 3.0]}
        
        with pytest.raises(ValueError, match="Missing property"):
            analyzer.compute_statistics(properties)
    
    def test_compute_statistics_empty_values(self):
        """Test statistics computation with empty values."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        properties = {'bandgap': []}
        
        with pytest.raises(ValueError, match="Empty values"):
            analyzer.compute_statistics(properties)
    
    def test_save_statistics_json(self):
        """Test saving statistics to JSON."""
        analyzer = PropertyDistributionAnalyzer(['bandgap'])
        
        properties = {'bandgap': [2.0, 2.5, 3.0]}
        
        output_path = Path(self.temp_dir) / 'stats.json'
        analyzer.save_statistics_json(properties, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            saved_stats = json.load(f)
        
        assert 'bandgap' in saved_stats
        assert saved_stats['bandgap']['mean'] == pytest.approx(2.5)


class TestStructureQualityAnalyzer:
    """Test StructureQualityAnalyzer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup temporary files."""
        shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_init(self):
        """Test initialization."""
        analyzer = StructureQualityAnalyzer()
        assert analyzer is not None
    
    def test_analyze_bond_lengths(self):
        """Test bond length analysis."""
        pytest.importorskip("ase")
        
        analyzer = StructureQualityAnalyzer()
        
        # Create simple crystal
        crystals = [{
            'positions': np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            'cell': np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
            'atomic_numbers': np.array([6, 6]),
        }]
        
        output_path = Path(self.temp_dir) / 'bond_lengths.png'
        fig = analyzer.analyze_bond_lengths(
            crystals=crystals,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert fig is not None
    
    def test_analyze_bond_lengths_empty_crystals(self):
        """Test bond length analysis with empty crystals."""
        analyzer = StructureQualityAnalyzer()
        
        output_path = Path(self.temp_dir) / 'bond_lengths.png'
        with pytest.raises(ValueError, match="must be non-empty"):
            analyzer.analyze_bond_lengths(
                crystals=[],
                output_path=str(output_path)
            )
    
    def test_analyze_bond_lengths_missing_key(self):
        """Test bond length analysis with missing key."""
        pytest.importorskip("ase")
        
        analyzer = StructureQualityAnalyzer()
        
        crystals = [{
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'cell': np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
            # Missing atomic_numbers
        }]
        
        output_path = Path(self.temp_dir) / 'bond_lengths.png'
        with pytest.raises(ValueError, match="missing required key"):
            analyzer.analyze_bond_lengths(
                crystals=crystals,
                output_path=str(output_path)
            )
    
    def test_analyze_cell_parameters(self):
        """Test cell parameter analysis."""
        analyzer = StructureQualityAnalyzer()
        
        crystals = [
            {'cell': np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])},
            {'cell': np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])},
        ]
        
        output_path = Path(self.temp_dir) / 'cell_params.png'
        fig = analyzer.analyze_cell_parameters(
            crystals=crystals,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert fig is not None
    
    def test_analyze_cell_parameters_empty_crystals(self):
        """Test cell parameter analysis with empty crystals."""
        analyzer = StructureQualityAnalyzer()
        
        output_path = Path(self.temp_dir) / 'cell_params.png'
        with pytest.raises(ValueError, match="must be non-empty"):
            analyzer.analyze_cell_parameters(
                crystals=[],
                output_path=str(output_path)
            )
    
    def test_analyze_cell_parameters_missing_cell(self):
        """Test cell parameter analysis with missing cell."""
        analyzer = StructureQualityAnalyzer()
        
        crystals = [{'positions': np.array([[0.0, 0.0, 0.0]])}]
        
        output_path = Path(self.temp_dir) / 'cell_params.png'
        with pytest.raises(ValueError, match="missing 'cell' key"):
            analyzer.analyze_cell_parameters(
                crystals=crystals,
                output_path=str(output_path)
            )
    
    def test_analyze_cell_parameters_invalid_shape(self):
        """Test cell parameter analysis with invalid cell shape."""
        analyzer = StructureQualityAnalyzer()
        
        crystals = [{'cell': np.array([[5.0, 0.0], [0.0, 5.0]])}]
        
        output_path = Path(self.temp_dir) / 'cell_params.png'
        with pytest.raises(ValueError, match="invalid cell shape"):
            analyzer.analyze_cell_parameters(
                crystals=crystals,
                output_path=str(output_path)
            )


class TestHTMLReportGenerator:
    """Test HTMLReportGenerator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        pytest.importorskip("plotly")
        generator = HTMLReportGenerator()
        assert generator is not None
    
    def test_generate_report_basic(self):
        """Test basic report generation."""
        pytest.importorskip("plotly")
        
        generator = HTMLReportGenerator()
        
        generated_props = {
            'bandgap': [2.1, 2.3, 2.5, 2.4, 2.2]
        }
        
        output_path = Path(self.temp_dir) / 'report.html'
        generator.generate_report(
            generated_properties=generated_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        # Check HTML content
        with open(output_path) as f:
            content = f.read()
        
        assert 'Crystal Generation Report' in content
        assert 'bandgap' in content
    
    def test_generate_report_with_training(self):
        """Test report generation with training data."""
        pytest.importorskip("plotly")
        
        generator = HTMLReportGenerator()
        
        generated_props = {'bandgap': [2.1, 2.3, 2.5]}
        training_props = {'bandgap': [2.0, 2.2, 2.4]}
        
        output_path = Path(self.temp_dir) / 'report_train.html'
        generator.generate_report(
            generated_properties=generated_props,
            training_properties=training_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_generate_report_with_target(self):
        """Test report generation with target values."""
        pytest.importorskip("plotly")
        
        generator = HTMLReportGenerator()
        
        generated_props = {'bandgap': [2.1, 2.3, 2.5]}
        target_props = {'bandgap': 2.5}
        
        output_path = Path(self.temp_dir) / 'report_target.html'
        generator.generate_report(
            generated_properties=generated_props,
            target_properties=target_props,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_generate_report_custom_title(self):
        """Test report generation with custom title."""
        pytest.importorskip("plotly")
        
        generator = HTMLReportGenerator()
        
        generated_props = {'bandgap': [2.1, 2.3, 2.5]}
        
        output_path = Path(self.temp_dir) / 'report_custom.html'
        generator.generate_report(
            generated_properties=generated_props,
            output_path=str(output_path),
            title='Custom Report Title'
        )
        
        assert output_path.exists()
        
        with open(output_path) as f:
            content = f.read()
        
        assert 'Custom Report Title' in content
    
    def test_generate_report_empty_properties(self):
        """Test report generation with empty properties."""
        pytest.importorskip("plotly")
        
        generator = HTMLReportGenerator()
        
        output_path = Path(self.temp_dir) / 'report_empty.html'
        with pytest.raises(ValueError, match="must be non-empty"):
            generator.generate_report(
                generated_properties={},
                output_path=str(output_path)
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
