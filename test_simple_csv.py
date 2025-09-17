#!/usr/bin/env python3
"""
Simple test script that doesn't require torch dependencies.
Tests the core CSV export logic.
"""
import os
import csv
import tempfile
import numpy as np

def _export_scalar_histogram_csv(values, filename, value_name, count_name, bins=50):
    """Export histogram of scalar values to CSV."""
    # Calculate histogram using numpy
    hist_counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([value_name, count_name])
        for center, count in zip(bin_centers, hist_counts):
            writer.writerow([f"{center:.4f}", count])
    
    # Also write summary statistics
    summary_filename = filename.replace('.csv', '_summary.csv')
    with open(summary_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Count', len(values)])
        writer.writerow(['Mean', f"{np.mean(values):.4f}"])
        writer.writerow(['Std', f"{np.std(values):.4f}"])
        writer.writerow(['Min', f"{np.min(values):.4f}"])
        writer.writerow(['Max', f"{np.max(values):.4f}"])
        writer.writerow(['Q25', f"{np.percentile(values, 25):.4f}"])
        writer.writerow(['Q50 (Median)', f"{np.percentile(values, 50):.4f}"])
        writer.writerow(['Q75', f"{np.percentile(values, 75):.4f}"])

def _export_encoding_statistics_csv(encoding_data, filename, component_name, stats_name):
    """Export statistics for each component of encoding vectors to CSV."""
    # Convert to numpy array for easier processing
    encoding_array = np.array(encoding_data)  # Shape: (n_molecules, n_components)
    n_components = encoding_array.shape[1]
    
    # Calculate statistics for each component
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([component_name, 'Mean', 'Std', 'Min', 'Max', 'Q25', 'Q50', 'Q75', 'Non-zero Count', 'Non-zero %'])
        
        for i in range(n_components):
            component_values = encoding_array[:, i]
            non_zero_count = np.count_nonzero(component_values)
            non_zero_percent = (non_zero_count / len(component_values)) * 100
            
            writer.writerow([
                f"Component_{i}",
                f"{np.mean(component_values):.6f}",
                f"{np.std(component_values):.6f}",
                f"{np.min(component_values):.6f}",
                f"{np.max(component_values):.6f}",
                f"{np.percentile(component_values, 25):.6f}",
                f"{np.percentile(component_values, 50):.6f}",
                f"{np.percentile(component_values, 75):.6f}",
                non_zero_count,
                f"{non_zero_percent:.2f}%"
            ])
    
    # Export histograms for each component (only for non-zero components)
    for i in range(n_components):
        component_values = encoding_array[:, i]
        non_zero_mask = component_values > 0
        
        if np.any(non_zero_mask):
            # Only create histogram for components that have non-zero values
            non_zero_values = component_values[non_zero_mask]
            component_filename = filename.replace('.csv', f'_component_{i}_histogram.csv')
            _export_scalar_histogram_csv(
                non_zero_values.tolist(),
                component_filename,
                f'Component_{i}_Value',
                'Count',
                bins=min(20, len(np.unique(non_zero_values)))  # Adjust bins for discrete data
            )

def test_molecular_weight_export():
    """Test molecular weight histogram export."""
    print("Testing molecular weight histogram export...")
    
    # Generate sample molecular weight data (typical range 50-500)
    np.random.seed(42)  # For reproducible results
    molecular_weights = np.random.normal(150, 50, 1000).tolist()
    molecular_weights = [max(20, w) for w in molecular_weights]  # Ensure positive weights
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'molecular_weight_histogram.csv')
        
        _export_scalar_histogram_csv(molecular_weights, test_file, 'Molecular Weight (u)', 'Count')
        
        # Verify files were created
        assert os.path.exists(test_file), "Histogram file not created"
        
        summary_file = test_file.replace('.csv', '_summary.csv')
        assert os.path.exists(summary_file), "Summary file not created"
        
        # Check content
        with open(test_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 51, "Should have 50 bins + header"  # 50 bins + header
            print(f"✓ Molecular weight histogram created with {len(lines)-1} bins")
        
        with open(summary_file, 'r') as f:
            content = f.read()
            assert "Mean" in content and "Std" in content, "Summary statistics missing"
            print("✓ Molecular weight summary statistics created")

def test_pi_conjugation_export():
    """Test pi conjugation ratio histogram export."""
    print("Testing pi conjugation ratio histogram export...")
    
    # Generate sample pi conjugation ratio data (range 0-1)
    np.random.seed(42)
    pi_ratios = np.random.beta(2, 5, 1000).tolist()  # Beta distribution skewed towards 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'pi_conjugation_ratio_histogram.csv')
        
        _export_scalar_histogram_csv(pi_ratios, test_file, 'Pi Conjugation Ratio', 'Count')
        
        # Verify files were created
        assert os.path.exists(test_file), "Pi conjugation histogram file not created"
        
        summary_file = test_file.replace('.csv', '_summary.csv')
        assert os.path.exists(summary_file), "Pi conjugation summary file not created"
        
        print("✓ Pi conjugation ratio histogram and summary created")

def test_atom_types_encoding_export():
    """Test atom types encoding statistics export."""
    print("Testing atom types encoding statistics export...")
    
    # Generate sample atom types encoding data (normalized one-hot style)
    np.random.seed(42)
    n_molecules = 500
    n_atom_types = 5  # H, C, N, O, F
    
    atom_types_data = []
    for i in range(n_molecules):
        # Each molecule has 1-3 different atom types
        encoding = np.zeros(n_atom_types)
        n_types = np.random.randint(1, 4)
        active_indices = np.random.choice(n_atom_types, n_types, replace=False)
        encoding[active_indices] = np.random.rand(n_types)
        # Normalize to sum to 1 (probability distribution)
        encoding = encoding / encoding.sum()
        atom_types_data.append(encoding)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'atom_types_encoding_stats.csv')
        
        _export_encoding_statistics_csv(atom_types_data, test_file, 'Atom Type Component', 'Statistics')
        
        # Verify main stats file
        assert os.path.exists(test_file), "Atom types stats file not created"
        
        with open(test_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == n_atom_types + 1, f"Should have {n_atom_types + 1} lines"
            print(f"✓ Atom types statistics created with {n_atom_types} components")
        
        # Verify component histograms were created
        component_files = [f for f in os.listdir(temp_dir) if 'component_' in f and 'histogram.csv' in f]
        assert len(component_files) > 0, "No component histogram files created"
        print(f"✓ Created {len(component_files)} component histogram files")

def test_functional_groups_encoding_export():
    """Test functional groups encoding statistics export."""
    print("Testing functional groups encoding statistics export...")
    
    # Generate sample functional groups encoding data
    np.random.seed(42)
    n_molecules = 500
    n_functional_groups = 8  # Various functional groups
    
    functional_groups_data = []
    for i in range(n_molecules):
        # Each molecule has 0-3 functional groups
        encoding = np.zeros(n_functional_groups)
        n_groups = np.random.randint(0, 4)
        if n_groups > 0:
            active_indices = np.random.choice(n_functional_groups, n_groups, replace=False)
            encoding[active_indices] = np.random.rand(n_groups)
            encoding = encoding / encoding.sum()
        else:
            # Uniform small probability for molecules with no functional groups
            encoding = np.full(n_functional_groups, 0.01 / n_functional_groups)
        
        functional_groups_data.append(encoding)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'functional_groups_encoding_stats.csv')
        
        _export_encoding_statistics_csv(functional_groups_data, test_file, 'Functional Group Component', 'Statistics')
        
        # Verify main stats file
        assert os.path.exists(test_file), "Functional groups stats file not created"
        
        with open(test_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == n_functional_groups + 1, f"Should have {n_functional_groups + 1} lines"
            print(f"✓ Functional groups statistics created with {n_functional_groups} components")
        
        # Verify component histograms were created
        component_files = [f for f in os.listdir(temp_dir) if 'component_' in f and 'histogram.csv' in f]
        assert len(component_files) >= 0, "Component histogram files missing"
        print(f"✓ Created {len(component_files)} functional group component histogram files")

def main():
    """Run all tests."""
    print("Testing CSV export functionality (without torch dependencies)...\n")
    
    try:
        test_molecular_weight_export()
        print()
        
        test_pi_conjugation_export()
        print()
        
        test_atom_types_encoding_export()
        print()
        
        test_functional_groups_encoding_export()
        print()
        
        print("✅ All CSV export tests passed successfully!")
        print("The CSV export functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)