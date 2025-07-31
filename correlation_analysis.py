import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Read the CSV file
df = pd.read_csv('postResults/processed/processed_sensitive_median.csv')

# Extract the columns we want to analyze
f_pattern = df['\\mathcal{F}_{\\text{pattern}}']
dp_avg = df['dp']
eo_avg = df['eo']
md_avg = df['md']
acc_avg = df['acc']
auc_avg = df['arc']

# Calculate Pearson correlation coefficients
metrics = ['dp', 'eo', 'md', 'acc', 'arc']
correlations = {}

print("Correlation Analysis between F_pattern and other metrics")
print("=" * 60)

for metric in metrics:
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(f_pattern, df[metric])

    # Spearman correlation (rank-based)
    spearman_corr, spearman_p = spearmanr(f_pattern, df[metric])

    correlations[metric] = {
        'pearson': pearson_corr,
        'pearson_p': pearson_p,
        'spearman': spearman_corr,
        'spearman_p': spearman_p
    }

    print(f"\n{metric}:")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

# Create a summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Metric':<20} {'Pearson':<12} {'p-value':<10} {'Spearman':<12} {'p-value':<10}")
print("-" * 60)

for metric in metrics:
    corr_data = correlations[metric]
    print(f"{metric:<20} {corr_data['pearson']:<12.4f} {corr_data['pearson_p']:<10.4f} "
          f"{corr_data['spearman']:<12.4f} {corr_data['spearman_p']:<10.4f}")

# Additional analysis: Check for significant correlations
print("\n" + "=" * 60)
print("SIGNIFICANT CORRELATIONS (p < 0.05)")
print("=" * 60)

significant_correlations = []
for metric in metrics:
    corr_data = correlations[metric]
    if corr_data['pearson_p'] < 0.05:
        significant_correlations.append((metric, 'Pearson', corr_data['pearson'], corr_data['pearson_p']))
    if corr_data['spearman_p'] < 0.05:
        significant_correlations.append((metric, 'Spearman', corr_data['spearman'], corr_data['spearman_p']))

if significant_correlations:
    for metric, corr_type, corr_val, p_val in significant_correlations:
        print(f"{metric} ({corr_type}): {corr_val:.4f} (p = {p_val:.4f})")
else:
    print("No significant correlations found (p < 0.05)")

# Dataset-wise analysis
print("\n" + "=" * 60)
print("DATASET-WISE ANALYSIS")
print("=" * 60)

datasets = df['dataset'].unique()
for dataset in datasets:
    dataset_data = df[df['dataset'] == dataset]
    print(f"\n{dataset} (n={len(dataset_data)}):")

    f_pattern_ds = dataset_data['\\mathcal{F}_{\\text{pattern}}']

    for metric in metrics:
        metric_data = dataset_data[metric]
        pearson_corr, pearson_p = pearsonr(f_pattern_ds, metric_data)
        print(f"  {metric}: r={pearson_corr:.4f} (p={pearson_p:.4f})")