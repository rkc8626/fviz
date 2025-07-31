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

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Correlation Analysis: F_pattern vs Other Metrics', fontsize=16, fontweight='bold')

# Metrics to plot
metrics = ['dp', 'eo', 'md', 'acc', 'arc']
metric_names = ['DP Average', 'EO Average', 'MD Average', 'Accuracy Average', 'AUC Average']

# Colors for different datasets
datasets = df['dataset'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot each metric
for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    # Plot data points colored by dataset
    for j, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]
        ax.scatter(dataset_data['\\mathcal{F}_{\\text{pattern}}'],
                  dataset_data[metric],
                  c=colors[j],
                  label=dataset,
                  alpha=0.7,
                  s=60)

    # Add trend line
    z = np.polyfit(df['\\mathcal{F}_{\\text{pattern}}'], df[metric], 1)
    p = np.poly1d(z)
    ax.plot(df['\\mathcal{F}_{\\text{pattern}}'],
            p(df['\\mathcal{F}_{\\text{pattern}}']),
            "r--", alpha=0.8, linewidth=2)

    # Calculate and display correlation
    corr, p_val = pearsonr(df['\\mathcal{F}_{\\text{pattern}}'], df[metric])
    ax.set_xlabel('F_pattern')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}\nCorrelation: {corr:.3f} (p={p_val:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Remove the empty subplot
axes[1, 2].remove()

# Adjust layout
plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
df_renamed = df.rename(columns={'\\mathcal{F}_{\\text{pattern}}': 'F_pattern'})
correlation_matrix = df_renamed[['F_pattern'] + metrics].corr()
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix: F_pattern vs Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


print("Visualization completed! Check the generated PNG files.")