#!/usr/bin/env python3
"""
Demo script showing what the CSV export feature produces.
This creates sample output files to demonstrate the functionality.
"""
import os
import csv
import tempfile

def create_demo_files():
    """Create demo CSV files showing the output format."""
    
    # Create a demo directory
    demo_dir = "demo_csv_output"
    os.makedirs(demo_dir, exist_ok=True)
    
    print(f"Creating demo CSV files in: {demo_dir}/")
    
    # 1. Molecular Weight Histogram
    mw_hist_file = os.path.join(demo_dir, "molecular_weight_histogram.csv")
    with open(mw_hist_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Molecular Weight (u)", "Count"])
        # Sample histogram data
        data = [
            (78.1, 12), (92.4, 18), (106.7, 25), (121.0, 34), (135.3, 28),
            (149.6, 22), (163.9, 15), (178.2, 8), (192.5, 5), (206.8, 2)
        ]
        for weight, count in data:
            writer.writerow([f"{weight:.1f}", count])
    
    # 2. Molecular Weight Summary
    mw_summary_file = os.path.join(demo_dir, "molecular_weight_histogram_summary.csv")
    with open(mw_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Statistic", "Value"])
        writer.writerow(["Count", 169])
        writer.writerow(["Mean", "135.4"])
        writer.writerow(["Std", "38.2"])
        writer.writerow(["Min", "78.1"])
        writer.writerow(["Max", "206.8"])
        writer.writerow(["Q25", "108.5"])
        writer.writerow(["Q50 (Median)", "133.2"])
        writer.writerow(["Q75", "158.7"])
    
    # 3. Pi Conjugation Ratio Histogram
    pi_hist_file = os.path.join(demo_dir, "pi_conjugation_ratio_histogram.csv")
    with open(pi_hist_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Pi Conjugation Ratio", "Count"])
        # Sample histogram data (biased towards low values)
        data = [
            (0.05, 45), (0.15, 38), (0.25, 32), (0.35, 25), (0.45, 18),
            (0.55, 12), (0.65, 8), (0.75, 5), (0.85, 3), (0.95, 1)
        ]
        for ratio, count in data:
            writer.writerow([f"{ratio:.2f}", count])
    
    # 4. Pi Conjugation Ratio Summary
    pi_summary_file = os.path.join(demo_dir, "pi_conjugation_ratio_histogram_summary.csv")
    with open(pi_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Statistic", "Value"])
        writer.writerow(["Count", 187])
        writer.writerow(["Mean", "0.32"])
        writer.writerow(["Std", "0.28"])
        writer.writerow(["Min", "0.00"])
        writer.writerow(["Max", "0.95"])
        writer.writerow(["Q25", "0.12"])
        writer.writerow(["Q50 (Median)", "0.24"])
        writer.writerow(["Q75", "0.48"])
    
    # 5. Atom Types Encoding Statistics
    atom_stats_file = os.path.join(demo_dir, "atom_types_encoding_stats.csv")
    with open(atom_stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Atom Type Component", "Mean", "Std", "Min", "Max", "Q25", "Q50", "Q75", "Non-zero Count", "Non-zero %"])
        # Typical for H, C, N, O, F
        data = [
            ("Component_0", "0.125000", "0.235681", "0.000000", "0.800000", "0.000000", "0.000000", "0.200000", 95, "57.23%"),
            ("Component_1", "0.650000", "0.184521", "0.200000", "1.000000", "0.500000", "0.700000", "0.800000", 166, "100.00%"),
            ("Component_2", "0.089000", "0.156743", "0.000000", "0.600000", "0.000000", "0.000000", "0.150000", 78, "46.99%"),
            ("Component_3", "0.076000", "0.142856", "0.000000", "0.500000", "0.000000", "0.000000", "0.120000", 65, "39.16%"),
            ("Component_4", "0.012000", "0.045231", "0.000000", "0.300000", "0.000000", "0.000000", "0.000000", 8, "4.82%"),
        ]
        for row in data:
            writer.writerow(row)
    
    # 6. Functional Groups Encoding Statistics
    fg_stats_file = os.path.join(demo_dir, "functional_groups_encoding_stats.csv")
    with open(fg_stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Functional Group Component", "Mean", "Std", "Min", "Max", "Q25", "Q50", "Q75", "Non-zero Count", "Non-zero %"])
        # Typical functional groups
        data = [
            ("Component_0", "0.045000", "0.134521", "0.000000", "0.800000", "0.000000", "0.000000", "0.000000", 23, "13.86%"),
            ("Component_1", "0.067000", "0.156743", "0.000000", "0.600000", "0.000000", "0.000000", "0.000000", 34, "20.48%"),
            ("Component_2", "0.032000", "0.098765", "0.000000", "0.500000", "0.000000", "0.000000", "0.000000", 18, "10.84%"),
            ("Component_3", "0.089000", "0.198432", "0.000000", "0.700000", "0.000000", "0.000000", "0.100000", 42, "25.30%"),
            ("Component_4", "0.021000", "0.076543", "0.000000", "0.400000", "0.000000", "0.000000", "0.000000", 12, "7.23%"),
            ("Component_5", "0.156000", "0.234567", "0.000000", "0.900000", "0.000000", "0.000000", "0.300000", 67, "40.36%"),
            ("Component_6", "0.078000", "0.165432", "0.000000", "0.600000", "0.000000", "0.000000", "0.000000", 35, "21.08%"),
            ("Component_7", "0.012000", "0.054321", "0.000000", "0.300000", "0.000000", "0.000000", "0.000000", 6, "3.61%"),
        ]
        for row in data:
            writer.writerow(row)
    
    # 7. Export Summary
    summary_file = os.path.join(demo_dir, "export_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Property", "Samples Found", "Files Generated"])
        writer.writerow(["molecular_weight", 169, 2])  # histogram + summary
        writer.writerow(["pi_conjugation_ratio", 187, 2])  # histogram + summary
        writer.writerow(["atom_types_encoding", 166, 6])  # stats + 5 component histograms
        writer.writerow(["functional_groups_encoding", 166, 9])  # stats + 8 component histograms
    
    # 8. Sample component histogram
    comp_hist_file = os.path.join(demo_dir, "atom_types_encoding_stats_component_1_histogram.csv")
    with open(comp_hist_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Component_1_Value", "Count"])
        # Carbon is present in most molecules
        data = [
            (0.25, 5), (0.35, 8), (0.45, 12), (0.55, 18), (0.65, 25),
            (0.75, 32), (0.85, 28), (0.95, 22), (1.05, 16)
        ]
        for value, count in data:
            writer.writerow([f"{value:.2f}", count])
    
    print("\n📊 Demo files created:")
    for filename in sorted(os.listdir(demo_dir)):
        filepath = os.path.join(demo_dir, filename)
        size = os.path.getsize(filepath)
        print(f"  • {filename} ({size} bytes)")
    
    print(f"\n✅ Demo CSV files created in '{demo_dir}/' directory")
    print("These files show the format and structure of the exported statistics.")
    
    return demo_dir

def show_sample_content(demo_dir):
    """Show sample content from some of the generated files."""
    
    print("\n📋 Sample file contents:\n")
    
    # Show molecular weight summary
    print("1. Molecular Weight Summary:")
    print("=" * 40)
    with open(os.path.join(demo_dir, "molecular_weight_histogram_summary.csv"), 'r') as f:
        print(f.read())
    
    # Show atom types encoding stats (first few lines)
    print("2. Atom Types Encoding Statistics (first few lines):")
    print("=" * 55)
    with open(os.path.join(demo_dir, "atom_types_encoding_stats.csv"), 'r') as f:
        lines = f.readlines()
        for line in lines[:4]:  # Header + first 3 components
            print(line.strip())
        if len(lines) > 4:
            print("...")
    
    print("\n3. Export Summary:")
    print("=" * 20)
    with open(os.path.join(demo_dir, "export_summary.csv"), 'r') as f:
        print(f.read())

def main():
    """Create demo files and show their content."""
    print("🚀 Creating demo CSV export files...")
    print("This shows what the --export_training_stats feature produces.\n")
    
    demo_dir = create_demo_files()
    show_sample_content(demo_dir)
    
    print(f"💡 To use this feature in training:")
    print(f"   python main_qm9.py --dataset ase_db --ase_db_path your_db.db --export_training_stats")

if __name__ == "__main__":
    main()