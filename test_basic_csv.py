#!/usr/bin/env python3
"""
Basic test script that doesn't require numpy.
Tests the core CSV export logic with basic Python.
"""
import os
import csv
import tempfile
import math
import random

def basic_histogram(values, bins=50):
    """Create a histogram using only basic Python."""
    if not values:
        return [], []
    
    min_val = min(values)
    max_val = max(values)
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return [len(values)], [min_val]
    
    bin_width = (max_val - min_val) / bins
    
    # Create bin edges
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
    
    # Count values in each bin
    counts = [0] * bins
    for value in values:
        # Find which bin this value belongs to
        bin_index = int((value - min_val) / bin_width)
        bin_index = min(bin_index, bins - 1)  # Handle edge case
        counts[bin_index] += 1
    
    # Calculate bin centers
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
    
    return counts, bin_centers

def basic_stats(values):
    """Calculate basic statistics using only Python."""
    if not values:
        return {}
    
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    
    # Calculate percentiles
    sorted_vals = sorted(values)
    q25_idx = int(0.25 * n)
    q50_idx = int(0.50 * n)
    q75_idx = int(0.75 * n)
    
    q25 = sorted_vals[q25_idx]
    q50 = sorted_vals[q50_idx]
    q75 = sorted_vals[q75_idx]
    
    return {
        'count': n,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'q25': q25,
        'q50': q50,
        'q75': q75
    }

def _export_scalar_histogram_csv(values, filename, value_name, count_name, bins=50):
    """Export histogram of scalar values to CSV."""
    # Calculate histogram
    hist_counts, bin_centers = basic_histogram(values, bins)
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([value_name, count_name])
        for center, count in zip(bin_centers, hist_counts):
            writer.writerow([f"{center:.4f}", count])
    
    # Also write summary statistics
    summary_filename = filename.replace('.csv', '_summary.csv')
    stats = basic_stats(values)
    
    with open(summary_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Count', stats['count']])
        writer.writerow(['Mean', f"{stats['mean']:.4f}"])
        writer.writerow(['Std', f"{stats['std']:.4f}"])
        writer.writerow(['Min', f"{stats['min']:.4f}"])
        writer.writerow(['Max', f"{stats['max']:.4f}"])
        writer.writerow(['Q25', f"{stats['q25']:.4f}"])
        writer.writerow(['Q50 (Median)', f"{stats['q50']:.4f}"])
        writer.writerow(['Q75', f"{stats['q75']:.4f}"])

def _export_encoding_statistics_csv(encoding_data, filename, component_name, stats_name):
    """Export statistics for each component of encoding vectors to CSV."""
    if not encoding_data:
        return
    
    n_components = len(encoding_data[0])
    
    # Calculate statistics for each component
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([component_name, 'Mean', 'Std', 'Min', 'Max', 'Q25', 'Q50', 'Q75', 'Non-zero Count', 'Non-zero %'])
        
        for i in range(n_components):
            component_values = [mol_encoding[i] for mol_encoding in encoding_data]
            non_zero_values = [v for v in component_values if v > 0]
            non_zero_count = len(non_zero_values)
            non_zero_percent = (non_zero_count / len(component_values)) * 100
            
            stats = basic_stats(component_values)
            
            writer.writerow([
                f"Component_{i}",
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                f"{stats['q25']:.6f}",
                f"{stats['q50']:.6f}",
                f"{stats['q75']:.6f}",
                non_zero_count,
                f"{non_zero_percent:.2f}%"
            ])
            
            # Export histograms for each component (only for non-zero components)
            if non_zero_values:
                component_filename = filename.replace('.csv', f'_component_{i}_histogram.csv')
                unique_values = len(set(non_zero_values))
                bins_to_use = min(20, unique_values)
                _export_scalar_histogram_csv(
                    non_zero_values,
                    component_filename,
                    f'Component_{i}_Value',
                    'Count',
                    bins=bins_to_use
                )

def test_molecular_weight_export():
    """Test molecular weight histogram export."""
    print("Testing molecular weight histogram export...")
    
    # Generate sample molecular weight data (typical range 50-500)
    random.seed(42)
    molecular_weights = []
    for _ in range(1000):
        # Approximate normal distribution using central limit theorem
        weight = sum(random.random() for _ in range(12)) - 6  # ~N(0,1)
        weight = weight * 50 + 150  # Scale to mean=150, std≈50
        weight = max(20, weight)  # Ensure positive weights
        molecular_weights.append(weight)
    
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
            assert len(lines) == 51, "Should have 50 bins + header"
            print(f"✓ Molecular weight histogram created with {len(lines)-1} bins")
        
        with open(summary_file, 'r') as f:
            content = f.read()
            assert "Mean" in content and "Std" in content, "Summary statistics missing"
            print("✓ Molecular weight summary statistics created")

def test_atom_types_encoding_export():
    """Test atom types encoding statistics export."""
    print("Testing atom types encoding statistics export...")
    
    # Generate sample atom types encoding data (normalized one-hot style)
    random.seed(42)
    n_molecules = 500
    n_atom_types = 5  # H, C, N, O, F
    
    atom_types_data = []
    for i in range(n_molecules):
        # Each molecule has 1-3 different atom types
        encoding = [0.0] * n_atom_types
        n_types = random.randint(1, 3)
        
        # Select random active indices
        active_indices = random.sample(range(n_atom_types), n_types)
        
        # Assign random values
        total = 0
        for idx in active_indices:
            value = random.random()
            encoding[idx] = value
            total += value
        
        # Normalize to sum to 1
        if total > 0:
            encoding = [v / total for v in encoding]
        
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

def test_functionality_with_sample_data():
    """Test with realistic sample data to show what the output looks like."""
    print("Testing with sample data to show output format...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating sample files in: {temp_dir}")
        
        # Test molecular weight
        molecular_weights = [124.5, 156.2, 98.1, 203.4, 87.6, 145.3, 176.8, 112.9, 189.2, 134.7]
        mw_file = os.path.join(temp_dir, 'sample_molecular_weight.csv')
        _export_scalar_histogram_csv(molecular_weights, mw_file, 'Molecular Weight (u)', 'Count', bins=5)
        
        print("\nMolecular Weight Histogram:")
        with open(mw_file, 'r') as f:
            print(f.read())
        
        print("Molecular Weight Summary:")
        with open(mw_file.replace('.csv', '_summary.csv'), 'r') as f:
            print(f.read())
        
        # Test atom types encoding
        atom_encodings = [
            [0.6, 0.4, 0.0, 0.0, 0.0],  # H and C
            [0.0, 0.7, 0.3, 0.0, 0.0],  # C and N
            [0.0, 0.5, 0.0, 0.5, 0.0],  # C and O
            [0.0, 0.8, 0.0, 0.0, 0.2],  # C and F
        ]
        
        atom_file = os.path.join(temp_dir, 'sample_atom_types.csv')
        _export_encoding_statistics_csv(atom_encodings, atom_file, 'Atom Type', 'Stats')
        
        print("Atom Types Encoding Statistics:")
        with open(atom_file, 'r') as f:
            print(f.read())
        
        print("Sample output files created successfully!")

def main():
    """Run all tests."""
    print("Testing CSV export functionality (basic Python only)...\n")
    
    try:
        test_molecular_weight_export()
        print()
        
        test_atom_types_encoding_export()
        print()
        
        test_functionality_with_sample_data()
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