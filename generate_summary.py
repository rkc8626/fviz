#!/usr/bin/env python3
"""
Generate summary report from existing batch results.
"""

import os
import json
import pandas as pd
import numpy as np

# Set the output directory
OUTPUT_DIR = 'knee_BDD_batch_output'
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

def create_summary_report():
    """Create summary report from existing results."""

    # Load all results
    all_results = {}
    for filename in os.listdir(METRICS_DIR):
        if filename.endswith('_results.json'):
            filepath = os.path.join(METRICS_DIR, filename)
            with open(filepath, 'r') as f:
                result = json.load(f)
                algo = result['algorithm']
                step = result['step']
                all_results[(algo, step)] = result

    print(f"Loaded {len(all_results)} result files")

    # Create summary DataFrames
    summary_data = []
    median_distance_data = []
    knee_point_data = []

    # Define all possible columns
    all_columns = [
        'Algorithm', 'Step', 'Total_Records', 'Mean_Difference', 'Std_Difference',
        'Median_Difference', 'Min_Difference', 'Max_Difference', 'Positive_Count',
        'Negative_Count', 'Zero_Count', 'Positive_Percentage', 'Negative_Percentage',
        'Zero_Percentage',
        # Knee point data
        'Knee_Point_Overall_Index', 'Knee_Point_Overall_Value', 'Knee_Point_Overall_Percentile',
        'Knee_Point_Attr0_Index', 'Knee_Point_Attr0_Value', 'Knee_Point_Attr0_Percentile',
        'Knee_Point_Attr1_Index', 'Knee_Point_Attr1_Value', 'Knee_Point_Attr1_Percentile',
        # ACC metrics
        'ACC_Env0_In', 'ACC_Env0_Out', 'ACC_Env1_In', 'ACC_Env1_Out', 'ACC_Average',
        # MD metrics
        'MD_Env0_In', 'MD_Env0_Out', 'MD_Env1_In', 'MD_Env1_Out', 'MD_Average',
        # DP metrics
        'DP_Env0_In', 'DP_Env0_Out', 'DP_Env1_In', 'DP_Env1_Out', 'DP_Average',
        # EO metrics
        'EO_Env0_In', 'EO_Env0_Out', 'EO_Env1_In', 'EO_Env1_Out', 'EO_Average',
        # AUC metrics
        'AUC_Env0_In', 'AUC_Env0_Out', 'AUC_Env1_In', 'AUC_Env1_Out', 'AUC_Average'
    ]

    for (algo, step), result in all_results.items():
        stats = result.get('basic_statistics', {})

        # Create summary row
        summary_row = {
            'Algorithm': algo,
            'Step': step,
            'Total_Records': stats.get('total_records', 0),
            'Mean_Difference': stats.get('mean_difference', 0),
            'Std_Difference': stats.get('std_difference', 0),
            'Median_Difference': stats.get('median_difference', 0),
            'Min_Difference': stats.get('min_difference', 0),
            'Max_Difference': stats.get('max_difference', 0),
            'Positive_Count': stats.get('positive_count', 0),
            'Negative_Count': stats.get('negative_count', 0),
            'Zero_Count': stats.get('zero_count', 0),
            'Positive_Percentage': stats.get('positive_percentage', 0),
            'Negative_Percentage': stats.get('negative_percentage', 0),
            'Zero_Percentage': stats.get('zero_percentage', 0)
        }

        # Add TXT metrics if available
        step_metrics = result.get('step_metrics', {})
        for metric in ['acc', 'md', 'dp', 'eo', 'auc']:
            if metric in step_metrics:
                metric_data = step_metrics[metric]
                summary_row.update({
                    f'{metric.upper()}_Env0_In': metric_data.get('env0_in', 0),
                    f'{metric.upper()}_Env0_Out': metric_data.get('env0_out', 0),
                    f'{metric.upper()}_Env1_In': metric_data.get('env1_in', 0),
                    f'{metric.upper()}_Env1_Out': metric_data.get('env1_out', 0),
                    f'{metric.upper()}_Average': np.mean([
                        metric_data.get('env0_in', 0),
                        metric_data.get('env0_out', 0),
                        metric_data.get('env1_in', 0),
                        metric_data.get('env1_out', 0)
                    ])
                })

        # Add median distances
        median_distances = result.get('median_distances', {})
        for line_name, distance in median_distances.items():
            median_distance_data.append({
                'Algorithm': algo,
                'Step': step,
                'Median_Line': line_name,
                'Distance_from_Overall_Median': distance
            })

        # Add knee point data
        knee_points = result.get('knee_points', {})
        for group, points in knee_points.items():
            if 'knee' in points:
                x, y = points['knee']
                if group == 'overall':
                    knee_point_data.append({
                        'Algorithm': algo,
                        'Step': step,
                        'Group': 'Overall',
                        'Index': x,
                        'Value': y,
                        'Percentile': (int(x) / stats['total_records']) * 100
                    })
                else:
                    attr_value = group.split('_')[-1]
                    subset_size = stats.get(f'sensitive_attribute_{attr_value}_count', 0)
                    if subset_size > 0:
                        knee_point_data.append({
                            'Algorithm': algo,
                            'Step': step,
                            'Group': f'Sensitive_Attribute_{attr_value}',
                            'Index': x,
                            'Value': y,
                            'Percentile': (int(x) / subset_size) * 100
                        })

        # Add knee point data to summary row
        if 'overall' in knee_points and 'knee' in knee_points['overall']:
            x, y = knee_points['overall']['knee']
            summary_row.update({
                'Knee_Point_Overall_Index': x,
                'Knee_Point_Overall_Value': y,
                'Knee_Point_Overall_Percentile': (int(x) / stats['total_records']) * 100
            })

        if 'sensitive_attribute_0' in knee_points and 'knee' in knee_points['sensitive_attribute_0']:
            x, y = knee_points['sensitive_attribute_0']['knee']
            subset_size = stats.get('sensitive_attribute_0_count', 0)
            if subset_size > 0:
                summary_row.update({
                    'Knee_Point_Attr0_Index': x,
                    'Knee_Point_Attr0_Value': y,
                    'Knee_Point_Attr0_Percentile': (int(x) / subset_size) * 100
                })

        if 'sensitive_attribute_1' in knee_points and 'knee' in knee_points['sensitive_attribute_1']:
            x, y = knee_points['sensitive_attribute_1']['knee']
            subset_size = stats.get('sensitive_attribute_1_count', 0)
            if subset_size > 0:
                summary_row.update({
                    'Knee_Point_Attr1_Index': x,
                    'Knee_Point_Attr1_Value': y,
                    'Knee_Point_Attr1_Percentile': (int(x) / subset_size) * 100
                })

        summary_data.append(summary_row)

    # Create DataFrames
    summary_df = pd.DataFrame(summary_data, columns=all_columns)
    median_distance_df = pd.DataFrame(median_distance_data)
    knee_point_df = pd.DataFrame(knee_point_data)

    # Save summary reports
    summary_file = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    summary_df.to_csv(summary_file, index=False)

    median_distance_file = os.path.join(OUTPUT_DIR, 'median_distances.csv')
    median_distance_df.to_csv(median_distance_file, index=False)

    knee_point_file = os.path.join(OUTPUT_DIR, 'knee_point_analysis.csv')
    knee_point_df.to_csv(knee_point_file, index=False)

    # Create detailed summary report
    report_file = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(report_file, 'w') as f:
        f.write("BATCH VISUALIZATION ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Total combinations processed: {len(all_results)}\n\n")

        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("MEDIAN DISTANCES:\n")
        f.write("-" * 18 + "\n")
        f.write(median_distance_df.to_string(index=False))
        f.write("\n\n")

        f.write("KNEE POINT ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        f.write(knee_point_df.to_string(index=False))
        f.write("\n\n")

        f.write("FILES GENERATED:\n")
        f.write("-" * 18 + "\n")
        f.write(f"• Summary statistics: {summary_file}\n")
        f.write(f"• Median distances: {median_distance_file}\n")
        f.write(f"• Knee point analysis: {knee_point_file}\n")
        f.write(f"• Individual results: {METRICS_DIR}/\n")
        f.write(f"• Plots: {PLOTS_DIR}/\n")

        # Calculate overall statistics
        f.write("\n\nOVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total combinations processed: {len(all_results)}\n")
        f.write(f"Average mean difference across all: {summary_df['Mean_Difference'].mean():.4f}\n")
        f.write(f"Average std difference across all: {summary_df['Std_Difference'].mean():.4f}\n")
        f.write(f"Total median line distances calculated: {len(median_distance_df)}\n")
        f.write(f"Total knee points detected: {len(knee_point_df)}\n")

        # Best and worst performers
        best_mean = summary_df.loc[summary_df['Mean_Difference'].idxmin()]
        worst_mean = summary_df.loc[summary_df['Mean_Difference'].idxmax()]

        f.write(f"\nBest mean difference: {best_mean['Algorithm']} Step {best_mean['Step']} ({best_mean['Mean_Difference']:.4f})\n")
        f.write(f"Worst mean difference: {worst_mean['Algorithm']} Step {worst_mean['Step']} ({worst_mean['Mean_Difference']:.4f})\n")

        # TXT metrics summary
        if 'ACC_Average' in summary_df.columns:
            f.write("\n\nTXT METRICS SUMMARY:\n")
            f.write("-" * 20 + "\n")

            # Best performers for each metric
            for metric in ['ACC', 'MD', 'DP', 'EO', 'AUC']:
                avg_col = f'{metric}_Average'
                if avg_col in summary_df.columns and summary_df[avg_col].notna().any():
                    best_metric = summary_df.loc[summary_df[avg_col].idxmax()]
                    worst_metric = summary_df.loc[summary_df[avg_col].idxmin()]
                    f.write(f"\n{metric} - Best: {best_metric['Algorithm']} Step {best_metric['Step']} ({best_metric[avg_col]:.4f})\n")
                    f.write(f"{metric} - Worst: {worst_metric['Algorithm']} Step {worst_metric['Step']} ({worst_metric[avg_col]:.4f})\n")

    print(f"Summary report created: {report_file}")
    print(f"Summary statistics saved: {summary_file}")
    print(f"Median distances saved: {median_distance_file}")
    print(f"Knee point analysis saved: {knee_point_file}")

    # Print knee point summary
    print(f"\nKNEE POINT SUMMARY:")
    print(f"Total knee points detected: {len(knee_point_df)}")
    if len(knee_point_df) > 0:
        print("Knee points by group:")
        for group in knee_point_df['Group'].unique():
            group_data = knee_point_df[knee_point_df['Group'] == group]
            print(f"  {group}: {len(group_data)} points")
            for _, row in group_data.iterrows():
                print(f"    {row['Algorithm']} Step {row['Step']}: Index {row['Index']}, Value {row['Value']:.4f}, Percentile {row['Percentile']:.1f}%")

if __name__ == "__main__":
    create_summary_report()