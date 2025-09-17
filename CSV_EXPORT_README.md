# CSV Statistics Export Feature

This feature allows exporting molecular statistics to CSV files when training with ASE databases in main_qm9.py.

## Usage

To enable CSV statistics export when training with an ASE database:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path /path/to/your/database.db \
    --export_training_stats \
    --stats_output_dir my_statistics
```

## Command Line Arguments

- `--export_training_stats`: Enable CSV statistics export (flag, no value needed)
- `--stats_output_dir`: Directory where CSV files will be saved (default: 'training_stats')

## Generated Files

The feature analyzes the training dataset and generates the following CSV files:

### 1. Molecular Weight Statistics
- `molecular_weight_histogram.csv`: Histogram of molecular weights
- `molecular_weight_histogram_summary.csv`: Summary statistics (mean, std, min, max, quartiles)

### 2. Pi Conjugation Ratio Statistics  
- `pi_conjugation_ratio_histogram.csv`: Histogram of pi conjugation ratios
- `pi_conjugation_ratio_histogram_summary.csv`: Summary statistics

### 3. Atom Types Encoding Statistics
- `atom_types_encoding_stats.csv`: Statistics for each atom type component
- `atom_types_encoding_stats_component_X_histogram.csv`: Histogram for each non-zero component

### 4. Functional Groups Encoding Statistics
- `functional_groups_encoding_stats.csv`: Statistics for each functional group component  
- `functional_groups_encoding_stats_component_X_histogram.csv`: Histogram for each non-zero component

### 5. Export Summary
- `export_summary.csv`: Overview of how many samples were found for each property

## Example Output

### Molecular Weight Summary
```csv
Statistic,Value
Count,1000
Mean,142.8700
Std,36.7515
Min,87.6000
Max,203.4000
Q25,112.9000
Q50 (Median),145.3000
Q75,176.8000
```

### Atom Types Encoding Statistics
```csv
Atom Type,Mean,Std,Min,Max,Q25,Q50,Q75,Non-zero Count,Non-zero %
Component_0,0.150000,0.259808,0.000000,0.600000,0.000000,0.000000,0.600000,250,25.00%
Component_1,0.600000,0.158114,0.400000,0.800000,0.500000,0.700000,0.800000,1000,100.00%
...
```

## Notes

- This feature only works with ASE database datasets (`--dataset ase_db`)
- The export runs once at the beginning of training, analyzing the entire training dataset
- If a molecular property is not available in the dataset, its export will be skipped
- Statistics are calculated only for the training split of the data
- Component histograms are only generated for encoding components that have non-zero values

## Use Cases

This feature is useful for:
- Understanding the distribution of molecular properties in your dataset
- Quality control and data validation before training
- Identifying potential biases or outliers in molecular descriptors
- Generating reports for publications or documentation
- Comparing different datasets or preprocessing approaches