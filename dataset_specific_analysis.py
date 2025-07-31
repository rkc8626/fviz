import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Read the CSV file
df = pd.read_csv('postResults/processed/processed_sensitive_median.csv')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Metrics to analyze - using the actual column names from the CSV
metrics = ['dp', 'eo', 'md', 'acc', 'arc']
metric_names = ['DP Average', 'EO Average', 'MD Average', 'Accuracy Average', 'AUC Average']

# Colors for different datasets
datasets = df['dataset'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create dataset-specific analysis for each metric
for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dataset-Specific Correlations: F_pattern vs {metric_name}', fontsize=16, fontweight='bold')

    for i, dataset in enumerate(datasets):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        dataset_data = df[df['dataset'] == dataset]

        # Plot data points
        ax.scatter(dataset_data['\\mathcal{F}_{\\text{pattern}}'],
                  dataset_data[metric],
                  c=colors[i],
                  alpha=0.7,
                  s=80)

        # Add trend line
        if len(dataset_data) > 2:
            z = np.polyfit(dataset_data['\\mathcal{F}_{\\text{pattern}}'],
                          dataset_data[metric], 1)
            p = np.poly1d(z)
            ax.plot(dataset_data['\\mathcal{F}_{\\text{pattern}}'],
                    p(dataset_data['\\mathcal{F}_{\\text{pattern}}']),
                    "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        corr, p_val = pearsonr(dataset_data['\\mathcal{F}_{\\text{pattern}}'],
                              dataset_data[metric])

        ax.set_xlabel('F_pattern')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{dataset}\nCorrelation: {corr:.3f} (p={p_val:.3f})')
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    axes[1, 2].remove()

    plt.tight_layout()
    plt.savefig(f'dataset_specific_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create a comprehensive summary heatmap
plt.figure(figsize=(12, 8))

# Create correlation matrix for each dataset
correlation_data = []
for dataset in datasets:
    dataset_data = df[df['dataset'] == dataset]
    correlations = []
    for metric in metrics:
        corr, _ = pearsonr(dataset_data['\\mathcal{F}_{\\text{pattern}}'], dataset_data[metric])
        correlations.append(corr)
    correlation_data.append(correlations)

# Create heatmap
correlation_df = pd.DataFrame(correlation_data,
                            index=datasets,
                            columns=metric_names)

sns.heatmap(correlation_df,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Dataset-Specific Correlations: F_pattern vs All Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dataset_specific_summary_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed summary table
print("=" * 80)
print("DATASET-SPECIFIC CORRELATION SUMMARY")
print("=" * 80)

summary_data = []
for dataset in datasets:
    dataset_data = df[df['dataset'] == dataset]
    row = [dataset]

    for metric in metrics:
        corr, p_val = pearsonr(dataset_data['\\mathcal{F}_{\\text{pattern}}'], dataset_data[metric])
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        row.append(f"{corr:.3f}{significance}")

    summary_data.append(row)

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data,
                         columns=['Dataset'] + metric_names)

print(summary_df.to_string(index=False))
print("\n*** p<0.001, ** p<0.01, * p<0.05")

# Additional analysis: Find strongest correlations per dataset
print("\n" + "=" * 80)
print("STRONGEST CORRELATIONS PER DATASET")
print("=" * 80)

for dataset in datasets:
    dataset_data = df[df['dataset'] == dataset]
    correlations = {}

    for metric in metrics:
        corr, p_val = pearsonr(dataset_data['\\mathcal{F}_{\\text{pattern}}'], dataset_data[metric])
        correlations[metric] = (abs(corr), corr, p_val)

    # Find strongest correlation
    strongest_metric = max(correlations.keys(), key=lambda x: correlations[x][0])
    strongest_corr, strongest_corr_val, strongest_p = correlations[strongest_metric]

    print(f"{dataset}:")
    print(f"  Strongest correlation: {strongest_metric} = {strongest_corr_val:.3f} (p={strongest_p:.3f})")

    # Show all correlations for this dataset
    for metric in metrics:
        abs_corr, corr_val, p_val = correlations[metric]
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"    {metric}: {corr_val:.3f}{significance} (p={p_val:.3f})")
    print()

print("Visualization completed! Check the generated PNG files.")