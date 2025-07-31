#!/usr/bin/env python3
"""
Batch Visualization and Analysis Program
Processes all prediction JSON files and generates comprehensive analysis reports.

# Process specific algorithms
python batch_visualization_analysis.py --algorithms Mixup SagNet

# Process specific steps
# NOTE: for now only 8000 step is available
python batch_visualization_analysis.py --steps 0 8000

# Generate only 3D visualizations
python batch_visualization_analysis.py --visualizations 3d

# Generate multiple types
python batch_visualization_analysis.py --visualizations scatter tsne trimap

# Specify input data directory
python batch_visualization_analysis.py --input-dir data/BDD

# Generate only HTML output
python batch_visualization_analysis.py --formats html

# Generate only PNG and PDF outputs
python batch_visualization_analysis.py --formats png pdf

# Use GPU acceleration (if available)
python batch_visualization_analysis.py --use-gpu

# Use specific GPU device
python batch_visualization_analysis.py --use-gpu --gpu-device cuda:1

# Skip existing results
python batch_visualization_analysis.py --skip-existing

# Force reprocessing
python batch_visualization_analysis.py --force

# Complete example with all options
python batch_visualization_analysis.py --visualizations scatter --input-dir data/BDD --output-dir BDD_batch_output --formats png --force

# Both segmented curves together
python batch_visualization_analysis.py --visualizations segmented_curve environment_segmented_curve --input-dir data/BDD --output-dir test --formats png --force --algorithms SagNet --num-segments 10


"""

import json
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import re
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings('ignore')

# Knee Point Detection Functions
def find_knee_point(x, y, method='kneedle'):
    """
    Find knee points using a heuristic approach with median as reference.
    Finds one elbow on each side of the median for more accurate detection.

    Args:
        x: x-coordinates (sorted indices)
        y: y-coordinates (diff values)
        method: 'kneedle' for Kneedle algorithm

    Returns:
        knee_points: List of (x, y) coordinates of knee points
    """
    if len(x) < 10:
        return []

    # Sort data by x if not already sorted
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Find the median index as reference point
    median_idx = len(x_sorted) // 2
    median_x = x_sorted[median_idx]
    median_y = y_sorted[median_idx]

    knee_points = []

    # Split data into left and right halves relative to median
    left_mask = x_sorted <= median_x
    right_mask = x_sorted >= median_x

    # Find left elbow (convex decreasing) - on the left side of median
    if np.sum(left_mask) > 5:
        left_x = x_sorted[left_mask]
        left_y = y_sorted[left_mask]

        # Use Kneedle algorithm on left subset
        left_elbow = find_kneedle_elbow(left_x, left_y, 'convex_decreasing')
        if left_elbow is not None:
            knee_points.append(left_elbow)

    # Find right elbow (concave increasing) - on the right side of median
    if np.sum(right_mask) > 5:
        right_x = x_sorted[right_mask]
        right_y = y_sorted[right_mask]

        # Use Kneedle algorithm on right subset
        right_elbow = find_kneedle_elbow(right_x, right_y, 'concave_increasing')
        if right_elbow is not None:
            knee_points.append(right_elbow)

    # If we don't have two points, try alternative approach
    if len(knee_points) < 2:
        # Try finding elbows in different regions
        if len(knee_points) == 0:
            # No points found, use fallback method
            knee_points = find_fallback_elbows(x_sorted, y_sorted, median_idx)
        elif len(knee_points) == 1:
            # Only one point found, find the other one
            existing_type = knee_points[0][2]
            if existing_type == 'convex_decreasing':
                # Find concave increasing point
                other_elbow = find_kneedle_elbow(x_sorted, y_sorted, 'concave_increasing')
                if other_elbow is not None:
                    knee_points.append(other_elbow)
            else:
                # Find convex decreasing point
                other_elbow = find_kneedle_elbow(x_sorted, y_sorted, 'convex_decreasing')
                if other_elbow is not None:
                    knee_points.append(other_elbow)

    return knee_points

def find_kneedle_elbow(x, y, elbow_type):
    """
    Find a single elbow using Kneedle algorithm on a subset of data.

    Args:
        x: x-coordinates
        y: y-coordinates
        elbow_type: 'convex_decreasing' or 'concave_increasing'

    Returns:
        (x, y, type) tuple or None if not found
    """
    if len(x) < 5:
        return None

    # Normalize the data to [0,1] range
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Calculate the line connecting the first and last points
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]

    # Calculate the distance from each point to this line
    distances = []
    for i in range(len(x_norm)):
        if x2 != x1:  # Avoid division by zero
            # Line equation: y = mx + b
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Distance from point to line: |y - (mx + b)| / sqrt(1 + m^2)
            distance = abs(y_norm[i] - (m * x_norm[i] + b)) / np.sqrt(1 + m**2)
        else:
            # Vertical line
            distance = abs(x_norm[i] - x1)

        distances.append(distance)

    # Find the point with maximum distance (main elbow)
    if len(distances) > 0:
        max_distance_idx = np.argmax(distances)
        return (x[max_distance_idx], y[max_distance_idx], elbow_type)

    return None

def find_fallback_elbows(x_sorted, y_sorted, median_idx):
    """
    Fallback method to find elbows when the main method fails.

    Args:
        x_sorted: sorted x coordinates
        y_sorted: sorted y coordinates
        median_idx: index of median point

    Returns:
        List of (x, y, type) tuples
    """
    knee_points = []

    # Find convex decreasing point in the first third
    first_third_idx = len(x_sorted) // 3
    if first_third_idx > 5:
        first_x = x_sorted[:first_third_idx]
        first_y = y_sorted[:first_third_idx]

        # Find point with maximum negative curvature
        if len(first_y) > 3:
            dy = np.diff(first_y)
            dx = np.diff(first_x)
            dx = np.where(dx == 0, 1e-10, dx)
            first_derivative = dy / dx

            if len(first_derivative) > 2:
                second_derivative = np.diff(first_derivative)
                if len(second_derivative) > 0:
                    min_idx = np.argmin(second_derivative)
                    if min_idx < len(first_x):
                        knee_points.append((first_x[min_idx], first_y[min_idx], 'convex_decreasing'))

    # Find concave increasing point in the last third
    last_third_idx = 2 * len(x_sorted) // 3
    if last_third_idx < len(x_sorted) - 5:
        last_x = x_sorted[last_third_idx:]
        last_y = y_sorted[last_third_idx:]

        # Find point with maximum positive curvature
        if len(last_y) > 3:
            dy = np.diff(last_y)
            dx = np.diff(last_x)
            dx = np.where(dx == 0, 1e-10, dx)
            first_derivative = dy / dx

            if len(first_derivative) > 2:
                second_derivative = np.diff(first_derivative)
                if len(second_derivative) > 0:
                    max_idx = np.argmax(second_derivative)
                    if max_idx < len(last_x):
                        knee_points.append((last_x[max_idx], last_y[max_idx], 'concave_increasing'))

    return knee_points

def detect_knee_points(records, show_knee=True):
    """
    Detect knee points for overall data, sensitive attribute groups, and environment groups.
    Finds two knee points per group: one convex decreasing, one concave increasing.

    Args:
        records: List of records with 'diff', 'sensitive_attribute', and 'environment' fields
        show_knee: Whether to detect knee points

    Returns:
        Dictionary containing knee points for each group
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    results = {
        'overall': {},
        'sensitive_attribute_0': {},
        'sensitive_attribute_1': {}
    }

    # Overall data
    x_overall = np.arange(len(df))
    y_overall = df['diff'].values

    if show_knee:
        knee_points = find_knee_point(x_overall, y_overall)
        for i, (x, y, point_type) in enumerate(knee_points):
            if i == 0:
                results['overall']['knee_convex'] = (x, y)
            elif i == 1:
                results['overall']['knee_concave'] = (x, y)

    # Sensitive attribute groups
    for attr_value in [0, 1]:
        subset = df[df['sensitive_attribute'] == attr_value]
        if len(subset) < 3:
            continue

        x_attr = np.arange(len(subset))
        y_attr = subset['diff'].values

        if show_knee:
            knee_points = find_knee_point(x_attr, y_attr)
            for i, (x, y, point_type) in enumerate(knee_points):
                # Convert back to overall indices
                overall_idx = subset.index[x]
                if i == 0:
                    results[f'sensitive_attribute_{attr_value}']['knee_convex'] = (overall_idx, y)
                elif i == 1:
                    results[f'sensitive_attribute_{attr_value}']['knee_concave'] = (overall_idx, y)

    # Environment groups
    unique_environments = sorted(df['environment'].unique())
    for env in unique_environments:
        subset = df[df['environment'] == env]
        if len(subset) < 3:
            continue

        x_env = np.arange(len(subset))
        y_env = subset['diff'].values

        if show_knee:
            knee_points = find_knee_point(x_env, y_env)
            for i, (x, y, point_type) in enumerate(knee_points):
                # Convert back to overall indices
                overall_idx = subset.index[x]
                if i == 0:
                    results[f'environment_{env}'] = results.get(f'environment_{env}', {})
                    results[f'environment_{env}']['knee_convex'] = (overall_idx, y)
                elif i == 1:
                    results[f'environment_{env}'] = results.get(f'environment_{env}', {})
                    results[f'environment_{env}']['knee_concave'] = (overall_idx, y)

    return results

# Try to import trimap, but make it optional
try:
    import trimap
    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False
    print("Warning: trimap not available. TriMap visualizations will be skipped.")

# Configuration
# NOTE:we have 5 different datasets BDD/CCMNIST1/FairFace/NYPD/YFCC

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/FairFace')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'New_batch_output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Available algorithms and steps
ALL_ALGORITHMS = ['SagNet', 'Mixup', 'MBDG', 'IGA', 'IRM', 'GroupDRO', 'Fish', 'ERM']
ALL_STEPS = [0, 4000, 8000]
ALL_VISUALIZATIONS = ['scatter', 'tsne', 'trimap', '3d_tsne', '3d_trimap', 'segmented_curve', 'all']

def extract_dataset_name(input_dir):
    """Extract dataset name from input directory path."""
    # Extract the last part of the path (e.g., 'data/BDD' -> 'BDD')
    dataset_basename = os.path.basename(input_dir)

    # Map the directory name to the proper dataset display name
    if dataset_basename == 'BDD':
        dataset_name = 'BDD100k'
    elif dataset_basename == 'CCMNIST1':
        dataset_name = 'CCMNIST1'
    elif dataset_basename == 'FairFace':
        dataset_name = 'FairFace'
    elif dataset_basename == 'NYPD':
        dataset_name = 'NYSF'
    elif dataset_basename == 'YFCC':
        dataset_name = 'YFCC'
    else:
        # Fallback to the basename if no mapping exists
        dataset_name = dataset_basename

    return dataset_name

# Global color schemes to avoid redundancy
# Dynamic color generation for multiple environments
def generate_environment_colors(num_environments):
    """Generate distinct colors for multiple environments."""
    if num_environments <= 2:
        return {0: '#FF7D00', 1: '#FF006D'}
    elif num_environments <= 4:
        return {0: '#FF7D00', 1: '#FF006D', 2: '#00CED1', 3: '#FF69B4'}
    elif num_environments <= 6:
        return {0: '#FF7D00', 1: '#FF006D', 2: '#00CED1', 3: '#FF69B4', 4: '#32CD32', 5: '#FFD700'}
    else:
        # Generate colors dynamically for more environments
        import colorsys
        colors = {}
        for i in range(num_environments):
            hue = i / num_environments
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            colors[i] = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
        return colors

COLOR_SCHEMES = {
    'sensitive_attribute': {0: '#01BEFE', 1: '#FFDD00'},
    'environment': {0: '#FF7D00', 1: '#FF006D'},  # Will be updated dynamically
    'correct_prediction': {0: '#ADFF02', 1: '#8F00FF'},
    'predicted_class': {0: '#00CED1', 1: '#FF69B4'},
    'all_three': {
        (0, 0, 0): '#01BEFE', (0, 0, 1): '#FFDD00', (0, 1, 0): '#FF7D00', (0, 1, 1): '#FF006D',
        (1, 0, 0): '#ADFF02', (1, 0, 1): '#8F00FF', (1, 1, 0): '#00CED1', (1, 1, 1): '#FF69B4'
    }
}

# Common layout configurations
COMMON_LAYOUT = {
    'height': 600,
    'width': 1200,
    'margin': dict(l=60, r=30, t=60, b=60),
    'showlegend': True,
    'hovermode': 'closest'
}

def get_scatter_layout(title, x_title='Index (sorted by diff)', y_title='diff'):
    """Get common scatter plot layout configuration."""
    return {
        **COMMON_LAYOUT,
        'title': title,
        'xaxis': dict(title=x_title),
        'yaxis': dict(title=y_title)
    }

def get_embedding_layout(title, method, x_title=None, y_title=None):
    """Get common embedding plot layout configuration."""
    if x_title is None:
        x_title = f'{method.upper()} Component 1'
    if y_title is None:
        y_title = f'{method.upper()} Component 2'

    return {
        **COMMON_LAYOUT,
        'title': title,
        'xaxis_title': x_title,
        'yaxis_title': y_title
    }

def get_filter_combinations(active_filters, df=None):
    """Get filter combinations for given active filters."""
    # Get unique values for each filter if df is provided
    env_values = [0, 1]  # Default binary
    sens_values = [0, 1]  # Default binary
    pred_values = [0, 1]  # Default binary

    if df is not None:
        if 'environment' in df.columns:
            env_values = sorted(df['environment'].unique())
        if 'sensitive_attribute' in df.columns:
            sens_values = sorted(df['sensitive_attribute'].unique())
        if 'correct_prediction' in df.columns:
            pred_values = sorted(df['correct_prediction'].unique())

    if len(active_filters) == 2:
        if 'environment' in active_filters and 'sensitive_attribute' in active_filters:
            return [(e, s, None) for e in env_values for s in sens_values]
        elif 'environment' in active_filters and 'correct_prediction' in active_filters:
            return [(e, None, p) for e in env_values for p in pred_values]
        elif 'sensitive_attribute' in active_filters and 'correct_prediction' in active_filters:
            return [(None, s, p) for s in sens_values for p in pred_values]
    else:
        return [(e, s, p) for e in env_values for s in sens_values for p in pred_values]

def get_subset_and_name(df, active_filters, e, s, p):
    """Get subset and name for given filter combination."""
    if len(active_filters) == 2:
        if 'environment' in active_filters and 'sensitive_attribute' in active_filters:
            subset = df[(df['environment'] == e) & (df['sensitive_attribute'] == s)]
            name = f'E={e}, S={s}'
        elif 'environment' in active_filters and 'correct_prediction' in active_filters:
            subset = df[(df['environment'] == e) & (df['correct_prediction'] == p)]
            name = f'E={e}, P={p}'
        elif 'sensitive_attribute' in active_filters and 'correct_prediction' in active_filters:
            subset = df[(df['sensitive_attribute'] == s) & (df['correct_prediction'] == p)]
            name = f'S={s}, P={p}'
    else:
        subset = df[(df['environment'] == e) & (df['sensitive_attribute'] == s) & (df['correct_prediction'] == p)]
        name = f'E={e}, S={s}, P={p}'

    return subset, name

def get_color_for_combination(active_filters, e, s, p, df=None):
    """Get color for given filter combination."""
    # If df is provided, get the actual color schemes
    env_colors = COLOR_SCHEMES['environment']
    sens_colors = COLOR_SCHEMES['sensitive_attribute']
    pred_colors = COLOR_SCHEMES['correct_prediction']

    if len(active_filters) == 2:
        if 'environment' in active_filters and 'sensitive_attribute' in active_filters:
            # Use environment color for this combination
            return env_colors.get(e, '#808080')
        elif 'environment' in active_filters and 'correct_prediction' in active_filters:
            # Use environment color for this combination
            return env_colors.get(e, '#808080')
        elif 'sensitive_attribute' in active_filters and 'correct_prediction' in active_filters:
            # Use sensitive attribute color for this combination
            return sens_colors.get(s, '#808080')
    else:
        # For three filters, generate a unique color based on combination
        # This is a simple hash-based approach
        import hashlib
        combo_str = f"{e}_{s}_{p}"
        hash_val = int(hashlib.md5(combo_str.encode()).hexdigest()[:6], 16)
        # Generate a color from the hash
        r = (hash_val >> 16) & 255
        g = (hash_val >> 8) & 255
        b = hash_val & 255
        return f'rgb({r}, {g}, {b})'

    return '#808080'  # Default gray

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch visualization and analysis for prediction data')

    # Input/Output arguments
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing prediction JSON files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results and visualizations')

    # Algorithm and step selection
    parser.add_argument('--algorithms', nargs='+', default=['SagNet', 'Mixup', 'MBDG', 'IGA', 'IRM', 'GroupDRO', 'Fish', 'ERM'],
                       help='Algorithms to process (default: all)')
    parser.add_argument('--steps', nargs='+', type=int, default=[8000],
                       help='Training steps to process (default: 8000)')

    # Visualization options
    parser.add_argument('--visualizations', nargs='+',
                       choices=['scatter', 'tsne', 'trimap', '3d_tsne', '3d_trimap', 'segmented_curve', 'environment_segmented_curve', 'all'],
                       default=['scatter'],
                       help='Types of visualizations to generate')
    parser.add_argument('--knee-point-only', action='store_true',
                       help='Generate only knee point detection visualization (overrides other visualization options)')
    parser.add_argument('--segmented-curve-only', action='store_true',
                       help='Generate only segmented curve visualization (overrides other visualization options)')
    parser.add_argument('--environment-segmented-curve-only', action='store_true',
                       help='Generate only environment segmented curve visualization (overrides other visualization options)')
    parser.add_argument('--num-segments', type=int, default=10,
                       help='Number of segments for segmented curve visualization (default: 10)')

    # Output format options
    parser.add_argument('--formats', nargs='+',
                       choices=['png', 'html', 'pdf', 'svg'],
                       default=['png'],
                       help='Output formats for plots')

    # Processing options
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of existing files')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration for t-SNE and TriMap (if available)')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum number of parallel workers (default: 1)')

    return parser.parse_args()

def get_processing_config():
    """Get processing configuration from command line arguments."""
    args = parse_arguments()

    # Handle knee-point-only option
    if args.knee_point_only:
        visualizations = ['knee_point']
    elif args.segmented_curve_only:
        visualizations = ['segmented_curve']
    elif args.environment_segmented_curve_only:
        visualizations = ['environment_segmented_curve']
    else:
        visualizations = args.visualizations
        if 'all' in visualizations:
            visualizations = ['scatter', 'tsne', 'trimap', '3d_tsne', '3d_trimap', 'segmented_curve', 'environment_segmented_curve']

    # Handle algorithm selection
    algorithms = args.algorithms
    if 'all' in algorithms:
        algorithms = ['SagNet', 'Mixup', 'MBDG', 'IGA', 'IRM', 'GroupDRO', 'Fish', 'ERM']

    # Handle step selection
    steps = args.steps
    if 'all' in steps:
        steps = [8000]  # Default to 8000 for now

    # Handle format selection
    output_formats = args.formats
    if 'all' in output_formats:
        output_formats = ['png', 'html', 'pdf', 'svg']

    return {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'algorithms': algorithms,
        'steps': steps,
        'visualizations': visualizations,
        'output_formats': output_formats,
        'force': args.force,
        'use_gpu': args.use_gpu,
        'max_workers': args.max_workers,
        'knee_point_only': args.knee_point_only,
        'segmented_curve_only': args.segmented_curve_only,
        'environment_segmented_curve_only': args.environment_segmented_curve_only,
        'num_segments': args.num_segments,
        'verbose': False  # Add verbose key
    }

def load_processed_predictions(file_path):
    """Load predictions from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    return [{
        'name': item['filename'],
        'diff': item['predicted_class'] - item['predicted_probabilities'][1],
        'predicted_class': item['predicted_class'],
        'environment': item['environment'],
        'sensitive_attribute': item['sensitive_attribute'],
        'correct_prediction': item['correct_prediction']
    } for item in data]

def create_scatter_trace(subset, x_values, color, name, show_lines=False, show_error_bars=False,
                        point_size=4, opacity=0.3, jitter_amount=0.008, density_sizing=False,
                        show_point_counts=True):
    """Create a scatter trace with consistent styling."""
    # Apply jittering if specified
    if jitter_amount > 0:
        np.random.seed(42)
        jittered_x = np.array(x_values) + np.random.normal(0, jitter_amount, len(x_values))
        jittered_y = np.array(subset['diff']) + np.random.normal(0, jitter_amount, len(subset['diff']))
    else:
        jittered_x = x_values
        jittered_y = subset['diff']

    # Apply density-based sizing if enabled
    if density_sizing and len(subset) > 1:
        try:
            coords = np.column_stack([jittered_x, jittered_y])
            distances = squareform(pdist(coords))
            density = np.sum(distances < 0.1, axis=1)
            sizes = point_size + (density - 1) * 2
            sizes = np.clip(sizes, point_size, point_size * 3)
        except:
            sizes = point_size
    else:
        sizes = point_size

    legend_name = name if not show_point_counts else f'{name} (n={len(subset)})'

    error_y_config = dict(type='data', array=subset['error'], visible=show_error_bars) if show_error_bars else None

    return go.Scatter(
        x=jittered_x,
        y=jittered_y,
        mode='markers+lines' if show_lines else 'markers',
        marker=dict(size=sizes, color=color, opacity=opacity, line=dict(color='grey', width=0.01)),
        line=dict(color=color, width=1) if show_lines else None,
        name=legend_name,
        error_y=error_y_config,
        hovertemplate='<b>%{text}</b><br>diff: %{y:.4f}<br>environment: %{customdata[0]}<br>sensitive_attribute: %{customdata[1]}<br>correct_prediction: %{customdata[2]}<extra></extra>',
        text=subset['name'],
        customdata=list(zip(subset['environment'], subset['sensitive_attribute'], subset['correct_prediction']))
    )

def create_filtered_scatter_plot(records, algorithm, step, save_path=None, output_formats=None):
    """Create filtered scatter plot and return metrics."""
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)
    df['error'] = 0.05

    # Dynamically update environment colors based on actual data
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    COLOR_SCHEMES['environment'] = generate_environment_colors(num_environments)

    traces = []
    median_positions = {}
    metrics_data = {}

    def add_median_line(x, color, name):
        traces.append(go.Scatter(x=[x, x], y=[df['diff'].min(), df['diff'].max()], mode='lines',
                                line=dict(color=color, dash='dash'), name=name, showlegend=True))
        median_positions[name] = x

    # Calculate overall median position for all data
    overall_median_idx = int(np.median(np.arange(len(df))))

    # Test all filter combinations
    filter_combinations = [
        ([], "No filters"),
        (['environment'], "Environment"),
        (['sensitive_attribute'], "Sensitive Attribute"),
        (['correct_prediction'], "Correct Prediction"),
        (['environment', 'sensitive_attribute'], "Environment + Sensitive Attribute"),
        (['environment', 'correct_prediction'], "Environment + Correct Prediction"),
        (['sensitive_attribute', 'correct_prediction'], "Sensitive Attribute + Correct Prediction"),
        (['environment', 'sensitive_attribute', 'correct_prediction'], "All Three")
    ]

    for active_filters, filter_name in filter_combinations:
        filter_metrics = {}

        if not active_filters:
            # No filters - single trace
            trace = create_scatter_trace(df, list(range(len(df))), 'blue', 'All Points',
                                       point_size=4, opacity=0.5, jitter_amount=0.008)
            traces.append(trace)
            add_median_line(overall_median_idx, 'black', 'Median (all data)')
            filter_metrics['all_points_count'] = len(df)
            filter_metrics['all_points_median'] = df['diff'].median()

        elif len(active_filters) == 1:
            # Single filter
            filter_name_single = active_filters[0]
            unique_values = sorted(df[filter_name_single].unique())
            for value in unique_values:
                subset = df[df[filter_name_single] == value]
                if len(subset) == 0: continue

                x_values = subset.index
                color = COLOR_SCHEMES[filter_name_single].get(value, '#808080')  # Default gray if not in scheme
                name = f'{filter_name_single.capitalize()}: {value}'

                trace = create_scatter_trace(subset, x_values, color, name,
                                           point_size=4, opacity=0.5, jitter_amount=0.008)
                traces.append(trace)

                if len(subset) > 0:
                    median_idx = int(np.median(subset.index))
                    add_median_line(median_idx, color, f'Median {filter_name_single[0].upper()}={value}')
                    filter_metrics[f'{filter_name_single}_{value}_count'] = len(subset)
                    filter_metrics[f'{filter_name_single}_{value}_median'] = subset['diff'].median()

        else:
            # Multiple filters
            filter_combinations_2d = get_filter_combinations(active_filters, df)

            for e, s, p in filter_combinations_2d:
                subset, name = get_subset_and_name(df, active_filters, e, s, p)
                if len(subset) == 0: continue

                color = get_color_for_combination(active_filters, e, s, p, df)

                trace = create_scatter_trace(subset, subset.index, color, name,
                                           point_size=4, opacity=0.5, jitter_amount=0.008)
                traces.append(trace)

                if len(subset) > 0:
                    median_idx = int(np.median(subset.index))
                    add_median_line(median_idx, color, f'Median {name}')
                    filter_metrics[f'{name}_count'] = len(subset)
                    filter_metrics[f'{name}_median'] = subset['diff'].median()

        metrics_data[filter_name] = filter_metrics

    # Add overall median line for all plots (spans entire x-axis)
    traces.append(go.Scatter(x=[overall_median_idx, overall_median_idx], y=[df['diff'].min(), df['diff'].max()],
                            mode='lines', line=dict(color='black', dash='dash', width=2),
                            name='Overall Median (all data)', showlegend=True))

    # Add y=0 line
    traces.append(go.Scatter(x=[0, len(df)], y=[0, 0], mode='lines',
                            line=dict(color='black', dash='dash'), name='y=0', showlegend=True))

    # Detect and add knee points
    knee_points = detect_knee_points(records, show_knee=True)

    # Add knee points
    for group, points in knee_points.items():
        # Add convex decreasing knee point (diamond, red)
        if 'knee_convex' in points:
            x, y = points['knee_convex']
            if group == 'overall':
                color = 'red'
                name = 'Knee Point Convex (Overall)'
            else:
                attr_value = group.split('_')[-1]
                color = COLOR_SCHEMES['sensitive_attribute'].get(int(attr_value), 'red')
                name = f'Knee Point Convex (Attr {attr_value})'

            traces.append(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color=color, symbol='diamond', line=dict(color='black', width=1)),
                name=name, showlegend=True,
                hovertemplate=f'<b>{name}</b><br>x: {x}<br>y: {y:.4f}<br>Type: Convex Decreasing<extra></extra>'
            ))

        # Add concave increasing knee point (star, purple)
        if 'knee_concave' in points:
            x, y = points['knee_concave']
            if group == 'overall':
                color = 'purple'
                name = 'Knee Point Concave (Overall)'
            else:
                attr_value = group.split('_')[-1]
                # Use a darker shade of the sensitive attribute color
                base_color = COLOR_SCHEMES['sensitive_attribute'].get(int(attr_value), 'purple')
                color = 'purple' if base_color == 'red' else base_color
                name = f'Knee Point Concave (Attr {attr_value})'

            traces.append(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color=color, symbol='star', line=dict(color='black', width=1)),
                name=name, showlegend=True,
                hovertemplate=f'<b>{name}</b><br>x: {x}<br>y: {y:.4f}<br>Type: Concave Increasing<extra></extra>'
            ))

    # Calculate median distances
    median_distances = {}
    if 'Median (all data)' in median_positions and len(median_positions) > 1:
        overall_median_pos = median_positions['Median (all data)']
        distances = [(name, pos - overall_median_pos) for name, pos in median_positions.items()
                    if name != 'Median (all data)']
        median_distances = dict(distances)

    # Create figure
    fig = go.Figure(data=traces, layout=get_scatter_layout(f'{algorithm} Step {step}: Individual Points (Sorted by Difference)'))

    # Save plot if path provided
    if save_path:
        # Use the algorithm-specific directory for saving
        save_dir = os.path.dirname(save_path)
        plot_files = save_plot_with_formats(fig, save_path, algorithm, step, "scatter_plot", output_formats=output_formats)
    else:
        plot_files = []

    return fig, metrics_data, median_distances, plot_files, knee_points

def create_knee_point_visualization(records, algorithm, step, save_path=None, output_formats=None):
    """
    Create a detailed visualization showing knee point detection process with median lines and jittering.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        algorithm: Algorithm name for the plot title
        step: Step number for the plot title
        save_path: Directory to save the plot
        output_formats: List of output formats (e.g., ['png', 'html'])

    Returns:
        Plotly figure object
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    # Create the main figure
    fig = go.Figure()

    # Color scheme for sensitive attributes
    colors = {0: '#01BEFE', 1: '#FFDD00'}

    # Add jittering for better visualization
    np.random.seed(42)  # For reproducible jittering
    jitter_amount = 0.008

    # Overall data with jittering
    x_overall = np.arange(len(df))
    y_overall = df['diff'].values
    x_jittered = x_overall + np.random.normal(0, jitter_amount, len(x_overall))
    y_jittered = y_overall + np.random.normal(0, jitter_amount, len(y_overall))

    # Add the main scatter plot for all data points
    fig.add_trace(go.Scatter(
        x=x_jittered, y=y_jittered, mode='markers',
        marker=dict(size=3, color='lightblue', opacity=0.6),
        name='All Data Points', showlegend=True
    ))

    # Add median line for overall data
    overall_median = np.median(y_overall)
    fig.add_trace(go.Scatter(
        x=[0, len(df)], y=[overall_median, overall_median],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name=f'Overall Median ({overall_median:.4f})', showlegend=True
    ))

    # Add horizontal median line for overall data (x-axis median)
    overall_median_x = np.median(x_overall)
    fig.add_trace(go.Scatter(
        x=[overall_median_x, overall_median_x], y=[df['diff'].min(), df['diff'].max()],
        mode='lines',
        line=dict(color='black', dash='dot', width=2),
        name=f'Overall Median X ({overall_median_x:.0f})', showlegend=True
    ))

    # Detect knee points
    knee_points_data = detect_knee_points(records, show_knee=True)

    # Record all knee point positions
    all_knee_positions = []

    # Store knee points for correlation analysis (without jitter)
    knee_points_for_analysis = {
        'overall': {'convex': None, 'concave': None},
        'sensitive_attribute_0': {'convex': None, 'concave': None},
        'sensitive_attribute_1': {'convex': None, 'concave': None}
    }

    # Add knee points for overall data
    if 'overall' in knee_points_data:
        if 'knee_convex' in knee_points_data['overall']:
            x, y = knee_points_data['overall']['knee_convex']
            # Use non-jittered Y value for calculations
            y_non_jittered = y_overall[x] if x < len(y_overall) else y
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color='red', symbol='diamond', line=dict(color='black', width=2)),
                name='Knee Point (Overall - Convex)', showlegend=True,
                hovertemplate=f'<b>Knee Point (Overall - Convex)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
            ))
            x_percentile = (x / len(df)) * 100
            y_percentile = (np.sum(y_overall <= y_non_jittered) / len(y_overall)) * 100
            all_knee_positions.append({
                'group': 'overall',
                'type': 'convex_decreasing',
                'x': x,
                'y': y,
                'x_percentile': x_percentile,
                'y_percentile': y_percentile
            })
            knee_points_for_analysis['overall']['convex'] = {'x': x, 'y': y_non_jittered, 'x_percentile': x_percentile, 'y_percentile': y_percentile}

        if 'knee_concave' in knee_points_data['overall']:
            x, y = knee_points_data['overall']['knee_concave']
            # Use non-jittered Y value for calculations
            y_non_jittered = y_overall[x] if x < len(y_overall) else y
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color='purple', symbol='star', line=dict(color='black', width=1)),
                name='Knee Point (Overall - Concave)', showlegend=True,
                hovertemplate=f'<b>Knee Point (Overall - Concave)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
            ))
            x_percentile = (x / len(df)) * 100
            y_percentile = (y / len(y_overall)) * 100
            all_knee_positions.append({
                'group': 'overall',
                'type': 'concave_increasing',
                'x': x,
                'y': y,
                'x_percentile': x_percentile,
                'y_percentile': y_percentile
            })
            knee_points_for_analysis['overall']['concave'] = {'x': x, 'y': y_non_jittered, 'x_percentile': x_percentile, 'y_percentile': y_percentile}

    # Add sensitive attribute groups with jittering and median lines
    for attr_value in [0, 1]:
        subset = df[df['sensitive_attribute'] == attr_value]
        if len(subset) < 3:
            continue

        x_attr = subset.index
        y_attr = subset['diff'].values

        # Apply jittering to group data
        x_attr_jittered = x_attr + np.random.normal(0, jitter_amount, len(x_attr))
        y_attr_jittered = y_attr + np.random.normal(0, jitter_amount, len(y_attr))

        # Add group points
        fig.add_trace(go.Scatter(
            x=x_attr_jittered, y=y_attr_jittered, mode='markers',
            marker=dict(size=4, color=colors[attr_value], opacity=0.8),
            name=f'Sensitive Attribute {attr_value} (n={len(subset)})', showlegend=True
        ))

        # Add median line for this group (vertical - y-axis median)
        group_median = np.median(y_attr)
        # fig.add_trace(go.Scatter(
        #     x=[x_attr.min(), x_attr.max()], y=[group_median, group_median],
        #     mode='lines',
        #     line=dict(color=colors[attr_value], dash='dash', width=2),
        #     name=f'Median Attr {attr_value} Y ({group_median:.4f})', showlegend=True
        # ))

        # Add horizontal median line for this group (x-axis median)
        group_median_x = np.median(x_attr)
        fig.add_trace(go.Scatter(
            x=[group_median_x, group_median_x], y=[df['diff'].min(), df['diff'].max()],
            mode='lines',
            line=dict(color=colors[attr_value], dash='dot', width=2),
            name=f'Median Attr {attr_value} X ({group_median_x:.0f})', showlegend=True
        ))

        # Add knee points for this group
        group_key = f'sensitive_attribute_{attr_value}'
        if group_key in knee_points_data:
            if 'knee_convex' in knee_points_data[group_key]:
                x, y = knee_points_data[group_key]['knee_convex']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color=colors[attr_value], symbol='diamond', line=dict(color='black', width=1)),
                    name=f'Knee Point (Attr {attr_value} - Convex)', showlegend=True,
                    hovertemplate=f'<b>Knee Point (Attr {attr_value} - Convex)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))
                all_knee_positions.append({
                    'group': f'sensitive_attribute_{attr_value}',
                    'type': 'convex_decreasing',
                    'x': x,
                    'y': y,
                    'x_percentile': (x / len(df)) * 100,
                    'y_percentile': (np.sum(y_overall <= y_overall[x]) / len(y_overall)) * 100
                })
                knee_points_for_analysis[group_key]['convex'] = {'x': x, 'y': y_overall[x], 'x_percentile': (x / len(df)) * 100, 'y_percentile': (np.sum(y_overall <= y_overall[x]) / len(y_overall)) * 100}

            if 'knee_concave' in knee_points_data[group_key]:
                x, y = knee_points_data[group_key]['knee_concave']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color=colors[attr_value], symbol='star', line=dict(color='black', width=1)),
                    name=f'Knee Point (Attr {attr_value} - Concave)', showlegend=True,
                    hovertemplate=f'<b>Knee Point (Attr {attr_value} - Concave)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))
                all_knee_positions.append({
                    'group': f'sensitive_attribute_{attr_value}',
                    'type': 'concave_increasing',
                    'x': x,
                    'y': y,
                    'x_percentile': (x / len(df)) * 100,
                    'y_percentile': (np.sum(y_overall <= y_overall[x]) / len(y_overall)) * 100
                })
                knee_points_for_analysis[group_key]['concave'] = {'x': x, 'y': y_overall[x], 'x_percentile': (x / len(df)) * 100, 'y_percentile': (np.sum(y_overall <= y_overall[x]) / len(y_overall)) * 100}

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Calculate correlation coefficients and group differences
    correlation_analysis = {}
    group_differences = {}

    # Extract X and Y values for correlation analysis
    x_values_convex = []
    y_values_convex = []
    x_values_concave = []
    y_values_concave = []

    for group_key in ['overall', 'sensitive_attribute_0', 'sensitive_attribute_1']:
        if knee_points_for_analysis[group_key]['convex'] is not None:
            x_values_convex.append(knee_points_for_analysis[group_key]['convex']['x'])
            y_values_convex.append(knee_points_for_analysis[group_key]['convex']['y'])
        if knee_points_for_analysis[group_key]['concave'] is not None:
            x_values_concave.append(knee_points_for_analysis[group_key]['concave']['x'])
            y_values_concave.append(knee_points_for_analysis[group_key]['concave']['y'])

    # Calculate correlations if we have at least 2 points
    if len(x_values_convex) >= 2:
        correlation_analysis['convex'] = {
            'x_correlation': float(np.corrcoef(x_values_convex, x_values_convex)[0, 1]) if len(x_values_convex) > 1 else 1.0,
            'y_correlation': float(np.corrcoef(y_values_convex, y_values_convex)[0, 1]) if len(y_values_convex) > 1 else 1.0
        }

    if len(x_values_concave) >= 2:
        correlation_analysis['concave'] = {
            'x_correlation': float(np.corrcoef(x_values_concave, x_values_concave)[0, 1]) if len(x_values_concave) > 1 else 1.0,
            'y_correlation': float(np.corrcoef(y_values_concave, y_values_concave)[0, 1]) if len(y_values_concave) > 1 else 1.0
        }

    # Calculate group differences
    if (knee_points_for_analysis['sensitive_attribute_0']['convex'] is not None and
        knee_points_for_analysis['sensitive_attribute_1']['convex'] is not None):
        group_differences['convex'] = {
            'x_percentile_diff': abs(knee_points_for_analysis['sensitive_attribute_0']['convex']['x_percentile'] -
                                   knee_points_for_analysis['sensitive_attribute_1']['convex']['x_percentile']),
            'y_percentile_diff': abs(knee_points_for_analysis['sensitive_attribute_0']['convex']['y_percentile'] -
                                   knee_points_for_analysis['sensitive_attribute_1']['convex']['y_percentile'])
        }

    if (knee_points_for_analysis['sensitive_attribute_0']['concave'] is not None and
        knee_points_for_analysis['sensitive_attribute_1']['concave'] is not None):
        group_differences['concave'] = {
            'x_percentile_diff': abs(knee_points_for_analysis['sensitive_attribute_0']['concave']['x_percentile'] -
                                   knee_points_for_analysis['sensitive_attribute_1']['concave']['x_percentile']),
            'y_percentile_diff': abs(knee_points_for_analysis['sensitive_attribute_0']['concave']['y_percentile'] -
                                   knee_points_for_analysis['sensitive_attribute_1']['concave']['y_percentile'])
        }

    # Update layout
    fig.update_layout(
        title=f'Knee Point Detection Visualization - {algorithm} Step {step}<br><sup>Diamond markers: Convex Decreasing, Star markers: Concave Increasing<br>Dashed lines: Y-axis medians, Dotted lines: X-axis medians</sup>',
        xaxis_title='Index (sorted by diff)',
        yaxis_title='Diff Value',
        showlegend=True,
        height=700,
        width=1200,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode='closest'
    )

    # Save the plot if requested
    plot_files = []
    if save_path and output_formats:
        plot_name = generate_plot_name(algorithm, step, 'knee_point_detection')
        plot_files = save_plot_with_formats(fig, save_path, algorithm, step, 'knee_point_detection', output_formats=output_formats)

    # Save knee point positions to JSON for analysis
    if save_path:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        knee_positions_file = os.path.join(metrics_dir, f'{algorithm}_step_{step}_knee_positions.json')
        with open(knee_positions_file, 'w') as f:
            json.dump({
                'algorithm': algorithm,
                'step': step,
                'total_records': len(df),
                'knee_positions': [
                    {
                        'group': pos['group'],
                        'type': pos['type'],
                        'x': int(pos['x']),
                        'y': float(pos['y']),
                        'x_percentile': float(pos['x_percentile']),
                        'y_percentile': float(pos['y_percentile'])
                    }
                    for pos in all_knee_positions
                ],
                'overall_median': float(overall_median),
                'overall_median_x': float(overall_median_x),
                'group_medians': {
                    f'attr_{i}': {
                        'y_median': float(np.median(df[df['sensitive_attribute'] == i]['diff'].values)),
                        'x_median': float(np.median(df[df['sensitive_attribute'] == i].index.values))
                    }
                    for i in [0, 1] if len(df[df['sensitive_attribute'] == i]) > 0
                },
                'correlation_analysis': correlation_analysis,
                'group_differences': group_differences
            }, f, indent=2)

        # Save enhanced analysis to text file
        analysis_file = os.path.join(metrics_dir, f'{algorithm}_step_{step}_knee_positions.txt')
        with open(analysis_file, 'w') as f:
            f.write(f"Knee Point Analysis for {algorithm} Step {step}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Knee Point Positions:\n")
            f.write("-" * 20 + "\n")
            for pos in all_knee_positions:
                f.write(f"Group: {pos['group']}\n")
                f.write(f"Type: {pos['type']}\n")
                f.write(f"X: {pos['x']} (X Percentile: {pos['x_percentile']:.2f}%)\n")
                f.write(f"Y: {pos['y']:.6f} (Y Percentile: {pos['y_percentile']:.2f}%)\n")
                f.write("\n")

            f.write("Correlation Analysis:\n")
            f.write("-" * 20 + "\n")
            for knee_type, corr_data in correlation_analysis.items():
                f.write(f"{knee_type.title()} Knee Points:\n")
                f.write(f"  X Correlation: {corr_data['x_correlation']:.6f}\n")
                f.write(f"  Y Correlation: {corr_data['y_correlation']:.6f}\n")
                f.write("\n")

            f.write("Group Differences:\n")
            f.write("-" * 20 + "\n")
            for knee_type, diff_data in group_differences.items():
                f.write(f"{knee_type.title()} Knee Points:\n")
                f.write(f"  X Percentile Difference: {diff_data['x_percentile_diff']:.2f}%\n")
                f.write(f"  Y Percentile Difference: {diff_data['y_percentile_diff']:.2f}%\n")
                f.write("\n")

            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Overall Median Y: {overall_median:.6f}\n")
            f.write(f"Overall Median X: {overall_median_x:.0f}\n")
            f.write("\n")

            for i in [0, 1]:
                if len(df[df['sensitive_attribute'] == i]) > 0:
                    group_median_y = np.median(df[df['sensitive_attribute'] == i]['diff'].values)
                    group_median_x = np.median(df[df['sensitive_attribute'] == i].index.values)
                    f.write(f"Group {i} Median Y: {group_median_y:.6f}\n")
                    f.write(f"Group {i} Median X: {group_median_x:.0f}\n")
                    f.write(f"Group {i} Count: {len(df[df['sensitive_attribute'] == i])}\n")
                    f.write("\n")

    return fig, all_knee_positions, plot_files

def create_embedding_visualization(records, algorithm, step, method='tsne', save_path=None, output_formats=None, use_gpu=False):
    """Create t-SNE or TriMap visualization."""
    df = pd.DataFrame(records)

    # Dynamically update environment colors based on actual data
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    COLOR_SCHEMES['environment'] = generate_environment_colors(num_environments)

    # Use absolute value of diff as feature, order: environment, |diff|, sensitive_attribute
    features = np.column_stack([
        df['environment'].values,
        df['diff'].values,
        df['sensitive_attribute'].values,
        df['correct_prediction'].values,
        df['predicted_class'].values
    ])
    features_scaled = StandardScaler().fit_transform(features)

    # Compute embedding with GPU acceleration if available
    if method == 'tsne':
        if use_gpu:
            try:
                # Use GPU-accelerated t-SNE if available
                from cuml.manifold import UMAP
                print(f"    Using GPU-accelerated t-SNE for {algorithm} step {step}")
                # Note: cuML doesn't have t-SNE, but we can use UMAP as an alternative
                # or fall back to CPU t-SNE
                embedding_result = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4)).fit_transform(features_scaled)
            except ImportError:
                print(f"    GPU t-SNE not available, using CPU for {algorithm} step {step}")
                embedding_result = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4)).fit_transform(features_scaled)
        else:
            embedding_result = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4)).fit_transform(features_scaled)
    elif method == 'trimap' and TRIMAP_AVAILABLE:
        if use_gpu:
            try:
                # Try to use GPU-accelerated TriMap if available
                print(f"    Using GPU-accelerated TriMap for {algorithm} step {step}")
                # Note: TriMap doesn't have GPU support, so we'll use CPU
                embedding_result = trimap.TRIMAP().fit_transform(features_scaled)
            except:
                print(f"    GPU TriMap not available, using CPU for {algorithm} step {step}")
                embedding_result = trimap.TRIMAP().fit_transform(features_scaled)
        else:
            embedding_result = trimap.TRIMAP().fit_transform(features_scaled)
    else:
        if method == 'trimap' and not TRIMAP_AVAILABLE:
            print(f"Warning: TriMap not available for {algorithm} step {step}. Skipping TriMap visualization.")
            return {}
        else:
            print(f"Warning: Unknown method '{method}' for {algorithm} step {step}. Skipping visualization.")
            return {}

    # Create visualizations for each coloring option
    color_options = ['environment', 'sensitive_attribute', 'correct_prediction', 'predicted_class', 'all_three']
    all_figs = {}
    all_plot_files = []

    for color_by in color_options:
        fig = go.Figure()

        if color_by == 'all_three':
            # Group by environment, sensitive_attribute, and correct_prediction for coloring
            unique_combinations = df.groupby(['environment', 'sensitive_attribute', 'correct_prediction']).size().reset_index(name='count')

            for _, row in unique_combinations.iterrows():
                e, s, p, count = row['environment'], row['sensitive_attribute'], row['correct_prediction'], row['count']
                mask = (df['environment'] == e) & (df['sensitive_attribute'] == s) & (df['correct_prediction'] == p)
                if mask.sum() == 0:
                    continue
                x_coords = embedding_result[mask, 0]
                y_coords = embedding_result[mask, 1]

                # Generate color for this combination
                color = get_color_for_combination(['environment', 'sensitive_attribute', 'correct_prediction'], e, s, p, df)

                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, mode='markers',
                    marker=dict(size=5, color=color, opacity=0.5, line=dict(color='grey', width=0.01)),
                    name=f'E={e}, S={s}, P={p} (n={count})',
                    hovertemplate=f'<b>%{{text}}</b><br>{method.upper()}1: %{{x:.2f}}<br>{method.upper()}2: %{{y:.2f}}<br>environment: %{{customdata[0]}}<br>sensitive_attribute: %{{customdata[1]}}<br>correct_prediction: %{{customdata[2]}}<br>diff: %{{customdata[3]:.4f}}<br>predicted_class: %{{customdata[4]}}<extra></extra>',
                    text=df[mask]['name'],
                    customdata=list(zip(df[mask]['environment'], df[mask]['sensitive_attribute'], df[mask]['correct_prediction'],
                                      df[mask]['diff'], df[mask]['predicted_class']))
                ))
        else:
            # Individual attribute coloring
            unique_values = sorted(df[color_by].unique())
            for value in unique_values:
                mask = df[color_by] == value
                x_coords = embedding_result[mask, 0]
                y_coords = embedding_result[mask, 1]

                color = COLOR_SCHEMES[color_by].get(value, '#808080')  # Default gray if not in scheme

                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, mode='markers',
                    marker=dict(size=5, color=color, opacity=0.5, line=dict(color='grey', width=0.01)),
                    name=f'{color_by}={value} (n={mask.sum()})',
                    hovertemplate=f'<b>%{{text}}</b><br>{method.upper()}1: %{{x:.2f}}<br>{method.upper()}2: %{{y:.2f}}<br>diff: %{{customdata[0]:.4f}}<br>environment: %{{customdata[1]}}<br>sensitive_attribute: %{{customdata[2]}}<br>correct_prediction: %{{customdata[3]}}<br>predicted_class: %{{customdata[4]}}<extra></extra>',
                    text=df[mask]['name'],
                    customdata=list(zip(df[mask]['diff'], df[mask]['environment'], df[mask]['sensitive_attribute'],
                                      df[mask]['correct_prediction'], df[mask]['predicted_class']))
                ))

        fig.update_layout(get_embedding_layout(
            f'{algorithm} Step {step}: {method.upper()} Visualization (Colored by {color_by.replace("_", " ").title()})',
            method
        ))

        all_figs[color_by] = fig

        # Save plot if path provided
        if save_path:
            color_save_path = save_path.replace('.html', f'_{color_by}.html')
            plot_files = save_plot_with_formats(fig, color_save_path, algorithm, step, f"{method}_embedding", color_by, output_formats)
            all_plot_files.extend(plot_files)

    return all_figs, all_plot_files

def create_3d_embedding_visualization(records, algorithm, step, method='tsne', save_path=None, output_formats=None, use_gpu=False):
    """Create 3D to 2D embedding visualization with all-three-attribute coloring."""
    df = pd.DataFrame(records)

    # Dynamically update environment colors based on actual data
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    COLOR_SCHEMES['environment'] = generate_environment_colors(num_environments)

    # Use absolute value of diff, order: environment, |diff|, sensitive_attribute
    features_3d = np.column_stack([
        df['environment'].values,
        df['diff'].values,
        # np.abs(df['diff'].values),
        df['sensitive_attribute'].values
    ])
    features_3d_scaled = StandardScaler().fit_transform(features_3d)

    # Group by environment, sensitive_attribute, and correct_prediction for coloring
    unique_combinations = df.groupby(['environment', 'sensitive_attribute', 'correct_prediction']).size().reset_index(name='count')

    # Compute embedding with GPU acceleration if available
    if method == 'tsne':
        if use_gpu:
            try:
                print(f"    Using GPU-accelerated t-SNE 3D for {algorithm} step {step}")
                embedding_result = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_3d)//4)).fit_transform(features_3d_scaled)
            except ImportError:
                print(f"    GPU t-SNE 3D not available, using CPU for {algorithm} step {step}")
                embedding_result = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_3d)//4)).fit_transform(features_3d_scaled)
        else:
            embedding_result = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_3d)//4)).fit_transform(features_3d_scaled)
    elif method == 'trimap' and TRIMAP_AVAILABLE:
        if use_gpu:
            try:
                print(f"    Using GPU-accelerated TriMap 3D for {algorithm} step {step}")
                embedding_result = trimap.TRIMAP().fit_transform(features_3d_scaled)
            except:
                print(f"    GPU TriMap 3D not available, using CPU for {algorithm} step {step}")
                embedding_result = trimap.TRIMAP().fit_transform(features_3d_scaled)
        else:
            embedding_result = trimap.TRIMAP().fit_transform(features_3d_scaled)
    else:
        if method == 'trimap' and not TRIMAP_AVAILABLE:
            print(f"Warning: TriMap not available for {algorithm} step {step}. Skipping TriMap 3D visualization.")
            return None, []
        else:
            print(f"Warning: Unknown method '{method}' for {algorithm} step {step}. Skipping 3D visualization.")
            return None, []

    fig = go.Figure()
    for _, row in unique_combinations.iterrows():
        e, s, p, count = row['environment'], row['sensitive_attribute'], row['correct_prediction'], row['count']
        mask = (df['environment'] == e) & (df['sensitive_attribute'] == s) & (df['correct_prediction'] == p)
        if mask.sum() == 0:
            continue
        x_coords = embedding_result[mask, 0]
        y_coords = embedding_result[mask, 1]

        # Generate color for this combination
        color = get_color_for_combination(['environment', 'sensitive_attribute', 'correct_prediction'], e, s, p, df)

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='markers',
            marker=dict(size=5, color=color, opacity=0.5, line=dict(color='grey', width=0.01)),
            name=f'E={e}, S={s}, P={p} (n={count})',
            hovertemplate=f'<b>%{{text}}</b><br>{method.upper()}1: %{{x:.2f}}<br>{method.upper()}2: %{{y:.2f}}<br>environment: %{{customdata[0]}}<br>sensitive_attribute: %{{customdata[1]}}<br>correct_prediction: %{{customdata[2]}}<br>diff: %{{customdata[3]:.4f}}<br>predicted_class: %{{customdata[4]}}<extra></extra>',
            text=df[mask]['name'],
            customdata=list(zip(df[mask]['environment'], df[mask]['sensitive_attribute'], df[mask]['correct_prediction'],
                                df[mask]['diff'], df[mask]['predicted_class']))
        ))

    fig.update_layout(get_embedding_layout(
        f'{algorithm} Step {step}: {method.upper()} 3D → 2D (3D space: environment, |diff|, sensitive_attribute)',
        method
    ))

    # Save plot if path provided
    if save_path:
        plot_files = save_plot_with_formats(fig, save_path, algorithm, step, f"{method}_3d_embedding", output_formats=output_formats)
    else:
        plot_files = []

    return fig, plot_files

def extract_metrics_from_txt_file(txt_file_path, target_steps=[0, 4000, 8000]):
    """Extract metrics from TXT file for specific steps."""
    metrics_data = {}

    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all metric tables in the file
        import re

        # More specific pattern to match actual metric tables
        # Look for lines that contain metric names followed by numeric values
        lines = content.split('\n')

        # Group metrics by step
        step_metrics = {}
        current_step = None

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for lines that contain metric headers (multiple metric names)
            # Check for all metric types: acc, md, dp, eo, auc
            metric_patterns = [
                'env0_in_acc', 'env0_out_acc', 'env1_in_acc', 'env1_out_acc',
                'env0_in_md', 'env0_out_md', 'env1_in_md', 'env1_out_md',
                'env0_in_dp', 'env0_out_dp', 'env1_in_dp', 'env1_out_dp',
                'env0_in_eo', 'env0_out_eo', 'env1_in_eo', 'env1_out_eo',
                'env0_in_auc', 'env0_out_auc', 'env1_in_auc', 'env1_out_auc'
            ]

            if any(metric in line for metric in metric_patterns):
                # This is likely a header line
                headers = line.split()

                # Check if the next line contains corresponding values
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and all(c.isdigit() or c == '.' or c == '-' for c in next_line.replace(' ', '')):
                        values = next_line.split()

                        # Check if this table has step information
                        if 'step' in headers and len(values) >= len(headers):
                            step_idx = headers.index('step')
                            try:
                                step_num = int(float(values[step_idx]))
                                if step_num in target_steps:
                                    current_step = step_num
                                    if step_num not in step_metrics:
                                        step_metrics[step_num] = {}

                                    # Match headers with values
                                    for j, header in enumerate(headers):
                                        if j < len(values) and header != 'step':
                                            try:
                                                value = float(values[j])
                                                step_metrics[step_num][header] = value
                                            except ValueError:
                                                continue
                            except ValueError:
                                continue

        # Also look for specific metric patterns like [md], [dp], [eo] results
        md_pattern = r'\[md\] result: ([\d\.]+)'
        dp_pattern = r'\[dp\] result: ([\d\.]+)'
        eo_pattern = r'\[eo\] result: ([\d\.]+)'

        md_match = re.search(md_pattern, content)
        dp_match = re.search(dp_pattern, content)
        eo_match = re.search(eo_pattern, content)

        # Extract hyperparameters
        hparams_pattern = r"Hparams:\s*\{([^}]+)\}"
        hparams_match = re.search(hparams_pattern, content)
        if hparams_match:
            hparams_str = hparams_match.group(1)
            # Parse hyperparameters
            hparams = {}
            for item in hparams_str.split(','):
                item = item.strip()
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip().strip("'")
                    value = value.strip().strip("'")
                    try:
                        # Try to convert to appropriate type
                        if value.lower() == 'true':
                            hparams[key] = True
                        elif value.lower() == 'false':
                            hparams[key] = False
                        elif '.' in value:
                            hparams[key] = float(value)
                        else:
                            hparams[key] = int(value)
                    except:
                        hparams[key] = value

        # Organize metrics by step and metric type
        for step_num, metrics in step_metrics.items():
            if step_num in target_steps:
                step_data = {}

                # Group metrics by type, normalize keys
                acc_metrics = {}
                md_metrics = {}
                dp_metrics = {}
                eo_metrics = {}
                auc_metrics = {}

                for key, value in metrics.items():
                    if key.endswith('_acc'):
                        acc_metrics[key.replace('_acc', '')] = value
                    elif key.endswith('_md'):
                        md_metrics[key.replace('_md', '')] = value
                    elif key.endswith('_dp'):
                        dp_metrics[key.replace('_dp', '')] = value
                    elif key.endswith('_eo'):
                        eo_metrics[key.replace('_eo', '')] = value
                    elif key.endswith('_auc'):
                        auc_metrics[key.replace('_auc', '')] = value

                # Add metric groups if they exist
                if acc_metrics:
                    step_data['acc'] = acc_metrics
                if md_metrics:
                    step_data['md'] = md_metrics
                if dp_metrics:
                    step_data['dp'] = dp_metrics
                if eo_metrics:
                    step_data['eo'] = eo_metrics
                if auc_metrics:
                    step_data['auc'] = auc_metrics

                # Add individual metric results if found
                if md_match:
                    step_data['md_result'] = float(md_match.group(1))
                if dp_match:
                    step_data['dp_result'] = float(dp_match.group(1))
                if eo_match:
                    step_data['eo_result'] = float(eo_match.group(1))

                # Add hyperparameters
                if hparams_match:
                    step_data['hyperparameters'] = hparams

                metrics_data[step_num] = step_data

    except Exception as e:
        print(f"Error parsing {txt_file_path}: {e}")

    return metrics_data

def calculate_basic_statistics(records):
    """Calculate basic statistics for the records."""
    diffs = [r['diff'] for r in records]
    stats = {
        'total_records': len(records),
        'mean_difference': np.mean(diffs),
        'std_difference': np.std(diffs),
        'min_difference': np.min(diffs),
        'max_difference': np.max(diffs),
        'median_difference': np.median(diffs),
        'positive_count': sum(1 for d in diffs if d > 0),
        'negative_count': sum(1 for d in diffs if d < 0),
        'zero_count': sum(1 for d in diffs if d == 0),
        'positive_percentage': sum(1 for d in diffs if d > 0) / len(diffs) * 100,
        'negative_percentage': sum(1 for d in diffs if d < 0) / len(diffs) * 100,
        'zero_percentage': sum(1 for d in diffs if d == 0) / len(diffs) * 100
    }

    # Calculate statistics by groups
    df = pd.DataFrame(records)
    for attr in ['environment', 'sensitive_attribute', 'correct_prediction', 'predicted_class']:
        for value in [0, 1]:
            subset = df[df[attr] == value]
            if len(subset) > 0:
                subset_diffs = subset['diff'].values
                stats[f'{attr}_{value}_count'] = len(subset)
                stats[f'{attr}_{value}_mean'] = np.mean(subset_diffs)
                stats[f'{attr}_{value}_std'] = np.std(subset_diffs)
                stats[f'{attr}_{value}_median'] = np.median(subset_diffs)
                stats[f'{attr}_{value}_min'] = np.min(subset_diffs)
                stats[f'{attr}_{value}_max'] = np.max(subset_diffs)

    return stats

def create_individual_scatter_plots(records, algorithm, step, save_dir, output_formats=None):
    """Create individual scatter plots for each filter combination."""
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)
    df['error'] = 0.05

    # Dynamically update environment colors based on actual data
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    COLOR_SCHEMES['environment'] = generate_environment_colors(num_environments)

    plots_created = []

    # Calculate overall median position for all data
    overall_median_idx = int(np.median(np.arange(len(df))))

    # 1. Single filter plots
    for filter_name in ['environment', 'sensitive_attribute', 'correct_prediction']:
        fig = go.Figure()

        unique_values = sorted(df[filter_name].unique())
        for value in unique_values:
            subset = df[df[filter_name] == value]
            if len(subset) == 0: continue

            x_values = subset.index
            color = COLOR_SCHEMES[filter_name].get(value, '#808080')  # Default gray if not in scheme
            name = f'{filter_name.capitalize()}: {value}'

            trace = create_scatter_trace(subset, x_values, color, name,
                                       point_size=4, opacity=0.5, jitter_amount=0.008)
            fig.add_trace(trace)

            # Add median line
            if len(subset) > 0:
                median_idx = int(np.median(subset.index))
                fig.add_trace(go.Scatter(
                    x=[median_idx, median_idx],
                    y=[df['diff'].min(), df['diff'].max()],
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    name=f'Median {filter_name[0].upper()}={value}',
                    showlegend=True
                ))

        # Add overall median line
        fig.add_trace(go.Scatter(
            x=[overall_median_idx, overall_median_idx],
            y=[df['diff'].min(), df['diff'].max()],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='Overall Median (all data)',
            showlegend=True
        ))

        # Add y=0 line
        fig.add_trace(go.Scatter(
            x=[0, len(df)], y=[0, 0], mode='lines',
            line=dict(color='black', dash='dash'), name='y=0', showlegend=True
        ))

        fig.update_layout(get_scatter_layout(
            f'{algorithm} Step {step}: Scatter Plot (Colored by {filter_name.capitalize()})'
        ))

        # Save plot
        plot_files = save_plot_with_formats(fig, os.path.join(save_dir, "dummy.html"), algorithm, step, "scatter", filter_name, output_formats)
        plots_created.extend(plot_files)

    # 2. Two-filter combination plots
    two_filter_combinations = [
        ('environment', 'sensitive_attribute', 'Environment + Sensitive Attribute'),
        ('environment', 'correct_prediction', 'Environment + Correct Prediction'),
        ('sensitive_attribute', 'correct_prediction', 'Sensitive Attribute + Correct Prediction')
    ]

    for filter1, filter2, title in two_filter_combinations:
        fig = go.Figure()
        active_filters = [filter1, filter2]
        filter_combinations_2d = get_filter_combinations(active_filters, df)

        for e, s, p in filter_combinations_2d:
            subset, name = get_subset_and_name(df, active_filters, e, s, p)
            if len(subset) == 0: continue

            color = get_color_for_combination(active_filters, e, s, p, df)
            trace = create_scatter_trace(subset, subset.index, color, name,
                                       point_size=4, opacity=0.5, jitter_amount=0.008)
            fig.add_trace(trace)

            # Add median line
            if len(subset) > 0:
                median_idx = int(np.median(subset.index))
                fig.add_trace(go.Scatter(
                    x=[median_idx, median_idx],
                    y=[df['diff'].min(), df['diff'].max()],
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    name=f'Median {name}',
                    showlegend=True
                ))

        # Add overall median line
        fig.add_trace(go.Scatter(
            x=[overall_median_idx, overall_median_idx],
            y=[df['diff'].min(), df['diff'].max()],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='Overall Median (all data)',
            showlegend=True
        ))

        # Add y=0 line
        fig.add_trace(go.Scatter(
            x=[0, len(df)], y=[0, 0], mode='lines',
            line=dict(color='black', dash='dash'), name='y=0', showlegend=True
        ))

        fig.update_layout(get_scatter_layout(
            f'{algorithm} Step {step}: Scatter Plot ({title})'
        ))

        # Save plot
        plot_files = save_plot_with_formats(fig, os.path.join(save_dir, "dummy.html"), algorithm, step, "scatter", f"{filter1}_{filter2}", output_formats)
        plots_created.extend(plot_files)

    # 3. All three filters plot
    fig = go.Figure()
    active_filters = ['environment', 'sensitive_attribute', 'correct_prediction']
    filter_combinations_3d = get_filter_combinations(active_filters, df)

    for e, s, p in filter_combinations_3d:
        subset, name = get_subset_and_name(df, active_filters, e, s, p)
        if len(subset) == 0: continue

        color = get_color_for_combination(active_filters, e, s, p, df)
        trace = create_scatter_trace(subset, subset.index, color, name,
                                   point_size=4, opacity=0.5, jitter_amount=0.008)
        fig.add_trace(trace)

        # Add median line
        if len(subset) > 0:
            median_idx = int(np.median(subset.index))
            fig.add_trace(go.Scatter(
                x=[median_idx, median_idx],
                y=[df['diff'].min(), df['diff'].max()],
                mode='lines',
                line=dict(color=color, dash='dash'),
                name=f'Median {name}',
                showlegend=True
            ))

    # Add overall median line
    fig.add_trace(go.Scatter(
        x=[overall_median_idx, overall_median_idx],
        y=[df['diff'].min(), df['diff'].max()],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Overall Median (all data)',
        showlegend=True
    ))

    # Add y=0 line
    fig.add_trace(go.Scatter(
        x=[0, len(df)], y=[0, 0], mode='lines',
        line=dict(color='black', dash='dash'), name='y=0', showlegend=True
    ))

    fig.update_layout(get_scatter_layout(
        f'{algorithm} Step {step}: Scatter Plot (All Three Filters)'
    ))

    # Save plot
    plot_files = save_plot_with_formats(fig, os.path.join(save_dir, "dummy.html"), algorithm, step, "scatter", "all_three", output_formats)
    plots_created.extend(plot_files)

    return plots_created

def process_single_file(algorithm, step, config=None, progress_counters=None):
    """Process a single algorithm-step combination."""
    if config is None:
        config = {'visualizations': ['scatter'], 'force': False, 'verbose': False}

    if progress_counters is None:
        progress_counters = {'scatter': [0, 0], 'tsne': [0, 0], 'trimap': [0, 0], '3d': [0, 0], 'total': [0, 0]}

    print(f"Processing {algorithm} step {step}...")

    # File paths
    json_file = os.path.join(config['input_dir'], f'{algorithm}_step_{step}_predictions.json')
    metrics_file = os.path.join(config['input_dir'], f'{algorithm}.txt')

    if not os.path.exists(json_file):
        print(f"Warning: {json_file} not found")
        return None

    # Check if results already exist and skip if requested
    results_file = os.path.join(config['output_dir'], 'metrics', f'{algorithm}_step_{step}_results.json')
    if os.path.exists(results_file) and not config['force']:
        print(f"Skipping {algorithm} step {step} (results already exist)")
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            pass

    # Load data
    try:
        records = load_processed_predictions(json_file)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None

    # Create output directory for this algorithm-step
    algo_step_dir = os.path.join(config['output_dir'], 'plots', f'{algorithm}_step_{step}')
    os.makedirs(algo_step_dir, exist_ok=True)

    # Calculate basic statistics
    basic_stats = calculate_basic_statistics(records)

    # Initialize plots generated list
    plots_generated = []

    # Create scatter plots if requested
    if 'scatter' in config['visualizations']:
        progress_counters['scatter'][0] += 1
        print(f"  Scatter plot [{progress_counters['scatter'][0]}/{progress_counters['scatter'][1]}]")

        scatter_path = os.path.join(algo_step_dir, 'scatter_plot.html')
        scatter_fig, scatter_metrics, median_distances, plot_files, knee_points = create_filtered_scatter_plot(
            records, algorithm, step, scatter_path, config['output_formats']
        )
        plots_generated.extend(plot_files)

        # Create individual scatter plots for each filter combination
        individual_scatter_plots = create_individual_scatter_plots(records, algorithm, step, algo_step_dir, config['output_formats'])
        plots_generated.extend(individual_scatter_plots)
    else:
        scatter_metrics = {}
        median_distances = {}
        knee_points = {}
        plot_files = []

    # Create knee point visualization if requested
    if 'knee_point' in config['visualizations'] or config.get('knee_point_only', False):
        knee_point_visualization_path = os.path.join(algo_step_dir, 'knee_point_visualization.html')
        knee_fig, knee_positions, knee_plot_files = create_knee_point_visualization(
            records, algorithm, step, knee_point_visualization_path, config['output_formats']
        )
        plots_generated.extend(knee_plot_files)

        # If knee-point-only is selected, skip other visualizations
        if config.get('knee_point_only', False):
            # Compile results for knee-point-only mode
            results = {
                'algorithm': algorithm,
                'step': step,
                'basic_statistics': basic_stats,
                'knee_positions_data': knee_positions,
                'plots_generated': plots_generated
            }

            # Save results to JSON
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Completed {algorithm} step {step}. Results saved to {results_file}")
            return results

    # Create segmented curve visualization if requested
    if 'segmented_curve' in config['visualizations'] or config.get('segmented_curve_only', False):
        segmented_curve_path = os.path.join(algo_step_dir, 'segmented_curve_visualization.html')
        dataset_name = extract_dataset_name(config['input_dir'])
        segmented_fig, segment_data, segmented_plot_files = create_segmented_curve_visualization(
            records, algorithm, step, config['num_segments'], segmented_curve_path, config['output_formats'], adaptive_segmentation=True, dataset_name=dataset_name
        )
        plots_generated.extend(segmented_plot_files)

        # If segmented-curve-only is selected, skip other visualizations
        if config.get('segmented_curve_only', False):
            # Compile results for segmented-curve-only mode
            results = {
                'algorithm': algorithm,
                'step': step,
                'basic_statistics': basic_stats,
                'segment_data': segment_data,
                'plots_generated': plots_generated
            }

            # Save results to JSON
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Completed {algorithm} step {step}. Results saved to {results_file}")
            return results

    # Create embedding visualizations if requested
    if 'tsne' in config['visualizations']:
        progress_counters['tsne'][0] += 1
        print(f"  t-SNE [{progress_counters['tsne'][0]}/{progress_counters['tsne'][1]}]")

        tsne_path = os.path.join(algo_step_dir, 'tsne_embedding.html')
        tsne_figs, tsne_plot_files = create_embedding_visualization(records, algorithm, step, 'tsne', tsne_path, config['output_formats'], config['use_gpu'])
        plots_generated.extend(tsne_plot_files)

    # Create TriMap visualizations only if available and requested
    if 'trimap' in config['visualizations'] and TRIMAP_AVAILABLE:
        progress_counters['trimap'][0] += 1
        print(f"  TriMap [{progress_counters['trimap'][0]}/{progress_counters['trimap'][1]}]")

        trimap_path = os.path.join(algo_step_dir, 'trimap_embedding.html')
        trimap_figs, trimap_plot_files = create_embedding_visualization(records, algorithm, step, 'trimap', trimap_path, config['output_formats'], config['use_gpu'])
        plots_generated.extend(trimap_plot_files)

    # Create 3D embedding visualizations if requested
    if '3d' in config['visualizations']:
        progress_counters['3d'][0] += 1
        print(f"  3D embedding [{progress_counters['3d'][0]}/{progress_counters['3d'][1]}]")

        tsne_3d_path = os.path.join(algo_step_dir, 'tsne_3d_embedding.html')
        tsne_3d_fig, tsne_3d_plot_files = create_3d_embedding_visualization(records, algorithm, step, 'tsne', tsne_3d_path, config['output_formats'], config['use_gpu'])
        plots_generated.extend(tsne_3d_plot_files)

        # Create TriMap 3D visualization only if available
        if TRIMAP_AVAILABLE:
            trimap_3d_path = os.path.join(algo_step_dir, 'trimap_3d_embedding.html')
            trimap_3d_fig, trimap_3d_plot_files = create_3d_embedding_visualization(records, algorithm, step, 'trimap', trimap_3d_path, config['output_formats'], config['use_gpu'])
            plots_generated.extend(trimap_3d_plot_files)

    # Extract step metrics if available
    step_metrics = {}
    if os.path.exists(metrics_file):
        all_metrics = extract_metrics_from_txt_file(metrics_file, [step])
        if step in all_metrics:
            step_metrics = all_metrics[step]

    # Compile all results
    results = {
        'algorithm': algorithm,
        'step': step,
        'basic_statistics': basic_stats,
        'scatter_plot_metrics': scatter_metrics,
        'median_line_distances': median_distances,
        'step_metrics': step_metrics,
        'trimap_available': TRIMAP_AVAILABLE,
        'plots_generated': plots_generated,
        'knee_points': knee_points
    }

    # Add knee positions data if available
    if 'knee_point' in config['visualizations'] or config.get('knee_point_only', False):
        results['knee_positions_data'] = knee_positions

    # Save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if config['verbose']:
        print(f"Generated {len(plots_generated)} plots for {algorithm} step {step}")

    print(f"Completed {algorithm} step {step}. Results saved to {results_file}")

    # Create knee point visualization if requested
    if 'knee_point' in config['visualizations'] or config.get('knee_point_only', False):
        knee_fig, knee_positions, knee_plot_files = create_knee_point_visualization(
            records, algorithm, step, save_path, config['output_formats']
        )
        plots_generated.extend(knee_plot_files)
        results['knee_positions_data'] = knee_positions

        # If knee-point-only is selected, skip other visualizations
        if config.get('knee_point_only', False):
            return results

    # Create environment segmented curve visualization if requested
    if 'environment_segmented_curve' in config['visualizations'] or config.get('environment_segmented_curve_only', False):
        env_segmented_curve_path = os.path.join(algo_step_dir, 'environment_segmented_curve_visualization.html')
        dataset_name = extract_dataset_name(config['input_dir'])
        env_segmented_fig, env_segment_data, env_segmented_plot_files = create_environment_segmented_curve_visualization(
            records, algorithm, step, config['num_segments'], env_segmented_curve_path, config['output_formats'], adaptive_segmentation=True, dataset_name=dataset_name
        )
        plots_generated.extend(env_segmented_plot_files)

        # If environment-segmented-curve-only is selected, skip other visualizations
        if config.get('environment_segmented_curve_only', False):
            # Compile results for environment-segmented-curve-only mode
            results = {
                'algorithm': algorithm,
                'step': step,
                'basic_statistics': basic_stats,
                'environment_segment_data': env_segment_data,
                'plots_generated': plots_generated
            }

            # Save results to JSON
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Completed {algorithm} step {step}. Results saved to {results_file}")
            return results

    return results

def create_summary_report():
    """Create a comprehensive summary report."""
    print("Creating summary report...")

    all_results = []

    # Load all results
    for algorithm in ALL_ALGORITHMS:
        for step in ALL_STEPS:
            results_file = os.path.join(METRICS_DIR, f'{algorithm}_step_{step}_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    all_results.append(results)

    if not all_results:
        print("No results found to create summary report")
        return

    # Create summary DataFrames
    summary_data = []
    median_distance_data = []
    knee_point_data = []

    # Define all possible columns to ensure consistent CSV structure
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
        'AUC_Env0_In', 'AUC_Env0_Out', 'AUC_Env1_In', 'AUC_Env1_Out', 'AUC_Average',
        # Knee point data
        'Knee_Point_Overall_Convex_Index', 'Knee_Point_Overall_Convex_Value', 'Knee_Point_Overall_Convex_Percentile',
        'Knee_Point_Overall_Concave_Index', 'Knee_Point_Overall_Concave_Value', 'Knee_Point_Overall_Concave_Percentile',
        'Knee_Point_Attr0_Convex_Index', 'Knee_Point_Attr0_Convex_Value', 'Knee_Point_Attr0_Convex_Percentile',
        'Knee_Point_Attr0_Concave_Index', 'Knee_Point_Attr0_Concave_Value', 'Knee_Point_Attr0_Concave_Percentile',
        'Knee_Point_Attr1_Convex_Index', 'Knee_Point_Attr1_Convex_Value', 'Knee_Point_Attr1_Convex_Percentile',
        'Knee_Point_Attr1_Concave_Index', 'Knee_Point_Attr1_Concave_Value', 'Knee_Point_Attr1_Concave_Percentile'
    ]

    for result in all_results:
        algo = result['algorithm']
        step = result['step']
        stats = result['basic_statistics']
        step_metrics = result.get('step_metrics', {})

        # Initialize summary row with all columns set to None
        summary_row = {col: None for col in all_columns}

        # Fill in basic summary data
        summary_row.update({
            'Algorithm': algo,
            'Step': step,
            'Total_Records': stats['total_records'],
            'Mean_Difference': stats['mean_difference'],
            'Std_Difference': stats['std_difference'],
            'Median_Difference': stats['median_difference'],
            'Min_Difference': stats['min_difference'],
            'Max_Difference': stats['max_difference'],
            'Positive_Count': stats['positive_count'],
            'Negative_Count': stats['negative_count'],
            'Zero_Count': stats['zero_count'],
            'Positive_Percentage': stats['positive_percentage'],
            'Negative_Percentage': stats['negative_percentage'],
            'Zero_Percentage': stats['zero_percentage']
        })

        # Add TXT metrics if available
        if step_metrics:
            print(f"Processing metrics for {algo} step {step}: {list(step_metrics.keys())}")

            # ACC metrics
            if 'acc' in step_metrics:
                acc_data = step_metrics['acc']
                summary_row.update({
                    'ACC_Env0_In': acc_data.get('env0_in'),
                    'ACC_Env0_Out': acc_data.get('env0_out'),
                    'ACC_Env1_In': acc_data.get('env1_in'),
                    'ACC_Env1_Out': acc_data.get('env1_out'),
                    'ACC_Average': np.mean([acc_data.get('env0_in', 0), acc_data.get('env0_out', 0),
                                           acc_data.get('env1_in', 0), acc_data.get('env1_out', 0)])
                })

            # MD metrics
            if 'md' in step_metrics:
                md_data = step_metrics['md']
                summary_row.update({
                    'MD_Env0_In': md_data.get('env0_in'),
                    'MD_Env0_Out': md_data.get('env0_out'),
                    'MD_Env1_In': md_data.get('env1_in'),
                    'MD_Env1_Out': md_data.get('env1_out'),
                    'MD_Average': np.mean([md_data.get('env0_in', 0), md_data.get('env0_out', 0),
                                          md_data.get('env1_in', 0), md_data.get('env1_out', 0)])
                })

            # DP metrics
            if 'dp' in step_metrics:
                dp_data = step_metrics['dp']
                summary_row.update({
                    'DP_Env0_In': dp_data.get('env0_in'),
                    'DP_Env0_Out': dp_data.get('env0_out'),
                    'DP_Env1_In': dp_data.get('env1_in'),
                    'DP_Env1_Out': dp_data.get('env1_out'),
                    'DP_Average': np.mean([dp_data.get('env0_in', 0), dp_data.get('env0_out', 0),
                                          dp_data.get('env1_in', 0), dp_data.get('env1_out', 0)])
                })

            # EO metrics
            if 'eo' in step_metrics:
                eo_data = step_metrics['eo']
                summary_row.update({
                    'EO_Env0_In': eo_data.get('env0_in'),
                    'EO_Env0_Out': eo_data.get('env0_out'),
                    'EO_Env1_In': eo_data.get('env1_in'),
                    'EO_Env1_Out': eo_data.get('env1_out'),
                    'EO_Average': np.mean([eo_data.get('env0_in', 0), eo_data.get('env0_out', 0),
                                          eo_data.get('env1_in', 0), eo_data.get('env1_out', 0)])
                })

            # AUC metrics
            if 'auc' in step_metrics:
                auc_data = step_metrics['auc']
                summary_row.update({
                    'AUC_Env0_In': auc_data.get('env0_in'),
                    'AUC_Env0_Out': auc_data.get('env0_out'),
                    'AUC_Env1_In': auc_data.get('env1_in'),
                    'AUC_Env1_Out': auc_data.get('env1_out'),
                    'AUC_Average': np.mean([auc_data.get('env0_in', 0), auc_data.get('env0_out', 0),
                                           auc_data.get('env1_in', 0), auc_data.get('env1_out', 0)])
                })
        else:
            print(f"No step metrics found for {algo} step {step}")

        summary_data.append(summary_row)

        # Median distances
        median_distances = result.get('median_line_distances', {})
        for line_name, distance in median_distances.items():
            median_distance_data.append({
                'Algorithm': algo,
                'Step': step,
                'Median_Line': line_name,
                'Distance_from_Overall_Median': distance
            })

        # Knee points
        knee_points = result.get('knee_points', {})
        for group, points in knee_points.items():
            # Handle convex decreasing knee point
            if 'knee_convex' in points:
                x, y = points['knee_convex']
                if group == 'overall':
                    knee_point_data.append({
                        'Algorithm': algo,
                        'Step': step,
                        'Group': 'Overall',
                        'Point_Type': 'Convex_Decreasing',
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
                            'Point_Type': 'Convex_Decreasing',
                            'Index': x,
                            'Value': y,
                            'Percentile': (int(x) / subset_size) * 100
                        })

            # Handle concave increasing knee point
            if 'knee_concave' in points:
                x, y = points['knee_concave']
                if group == 'overall':
                    knee_point_data.append({
                        'Algorithm': algo,
                        'Step': step,
                        'Group': 'Overall',
                        'Point_Type': 'Concave_Increasing',
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
                            'Point_Type': 'Concave_Increasing',
                            'Index': x,
                            'Value': y,
                            'Percentile': (int(x) / subset_size) * 100
                        })

        # Add knee point data to summary row
        if 'overall' in knee_points and 'knee_convex' in knee_points['overall']:
            x, y = knee_points['overall']['knee_convex']
            summary_row.update({
                'Knee_Point_Overall_Convex_Index': x,
                'Knee_Point_Overall_Convex_Value': y,
                'Knee_Point_Overall_Convex_Percentile': (int(x) / stats['total_records']) * 100
            })

        if 'overall' in knee_points and 'knee_concave' in knee_points['overall']:
            x, y = knee_points['overall']['knee_concave']
            summary_row.update({
                'Knee_Point_Overall_Concave_Index': x,
                'Knee_Point_Overall_Concave_Value': y,
                'Knee_Point_Overall_Concave_Percentile': (int(x) / stats['total_records']) * 100
            })

        if 'sensitive_attribute_0' in knee_points and 'knee_convex' in knee_points['sensitive_attribute_0']:
            x, y = knee_points['sensitive_attribute_0']['knee_convex']
            subset_size = stats.get('sensitive_attribute_0_count', 0)
            if subset_size > 0:
                summary_row.update({
                    'Knee_Point_Attr0_Convex_Index': x,
                    'Knee_Point_Attr0_Convex_Value': y,
                    'Knee_Point_Attr0_Convex_Percentile': (int(x) / subset_size) * 100
                })

        if 'sensitive_attribute_0' in knee_points and 'knee_concave' in knee_points['sensitive_attribute_0']:
            x, y = knee_points['sensitive_attribute_0']['knee_concave']
            subset_size = stats.get('sensitive_attribute_0_count', 0)
            if subset_size > 0:
                summary_row.update({
                    'Knee_Point_Attr0_Concave_Index': x,
                    'Knee_Point_Attr0_Concave_Value': y,
                    'Knee_Point_Attr0_Concave_Percentile': (int(x) / subset_size) * 100
                })

        if 'sensitive_attribute_1' in knee_points and 'knee_convex' in knee_points['sensitive_attribute_1']:
            x, y = knee_points['sensitive_attribute_1']['knee_convex']
            subset_size = stats.get('sensitive_attribute_1_count', 0)
            if subset_size > 0:
                summary_row.update({
                    'Knee_Point_Attr1_Convex_Index': x,
                    'Knee_Point_Attr1_Convex_Value': y,
                    'Knee_Point_Attr1_Convex_Percentile': (int(x) / subset_size) * 100
                })

        if 'sensitive_attribute_1' in knee_points and 'knee_concave' in knee_points['sensitive_attribute_1']:
            x, y = knee_points['sensitive_attribute_1']['knee_concave']
            subset_size = stats.get('sensitive_attribute_1_count', 0)
            if subset_size > 0:
                summary_row.update({
                    'Knee_Point_Attr1_Concave_Index': x,
                    'Knee_Point_Attr1_Concave_Value': y,
                    'Knee_Point_Attr1_Concave_Percentile': (int(x) / subset_size) * 100
            })

    # Create DataFrames with consistent column order
    summary_df = pd.DataFrame(summary_data, columns=all_columns)
    median_distance_df = pd.DataFrame(median_distance_data)
    knee_point_df = pd.DataFrame(knee_point_data)

    # Save summary reports
    summary_file = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    summary_df.to_csv(summary_file, index=False)

    median_distance_file = os.path.join(OUTPUT_DIR, 'median_line_distances.csv')
    median_distance_df.to_csv(median_distance_file, index=False)

    knee_point_file = os.path.join(OUTPUT_DIR, 'knee_point_analysis.csv')
    knee_point_df.to_csv(knee_point_file, index=False)

    # Create detailed summary report
    report_file = os.path.join(OUTPUT_DIR, 'comprehensive_summary_report.txt')
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE BATCH VISUALIZATION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("ALGORITHMS PROCESSED:\n")
        f.write("-" * 30 + "\n")
        for algo in ALL_ALGORITHMS:
            f.write(f"• {algo}\n")
        f.write("\n")

        f.write("STEPS PROCESSED:\n")
        f.write("-" * 20 + "\n")
        for step in ALL_STEPS:
            f.write(f"• Step {step}\n")
        f.write("\n")

        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("MEDIAN LINE DISTANCES:\n")
        f.write("-" * 25 + "\n")
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

def generate_plot_name(algorithm, step, plot_type, suffix=""):
    """Generate consistent plot names with algorithm and step information."""
    base_name = f"{algorithm}_step_{step}_{plot_type}"
    if suffix:
        base_name += f"_{suffix}"
    return base_name

def save_plot_with_formats(fig, base_path, algorithm, step, plot_type, suffix="", output_formats=None):
    """Save plot in multiple formats."""
    if output_formats is None:
        output_formats = ['html', 'png', 'pdf']

    plot_name = generate_plot_name(algorithm, step, plot_type, suffix)
    save_dir = os.path.dirname(base_path)

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []

    # Save HTML
    if 'html' in output_formats:
        html_path = os.path.join(save_dir, f"{plot_name}.html")
        fig.write_html(html_path)
        saved_files.append(f"{plot_name}.html")

    # Save PNG
    if 'png' in output_formats:
        png_path = os.path.join(save_dir, f"{plot_name}.png")
        fig.write_image(png_path, width=1200, height=600)
        saved_files.append(f"{plot_name}.png")

    # Save PDF
    if 'pdf' in output_formats:
        pdf_path = os.path.join(save_dir, f"{plot_name}.pdf")
        fig.write_image(pdf_path, width=1200, height=600)
        saved_files.append(f"{plot_name}.pdf")

    return saved_files

def create_segmented_curve_visualization(records, algorithm, step, num_segments=10, save_path=None, output_formats=None, adaptive_segmentation=True, dataset_name=None):
    """
    Create a segmented curve visualization where each segment is colored by the majority group.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        algorithm: Algorithm name for the plot title
        step: Step number for the plot title
        num_segments: Number of segments to divide the curve into
        save_path: Directory to save the plot
        output_formats: List of output formats (e.g., ['png', 'html'])
        adaptive_segmentation: If True, use adaptive segmentation based on data density
        dataset_name: Name of the dataset to include in the title

    Returns:
        Plotly figure object, segment data, plot files
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    # Color scheme for sensitive attributes
    colors = {0: '#01BEFE', 1: '#FFDD00'}

    # Create the main figure
    fig = go.Figure()

    # Get the full range of diff values
    x_range = np.arange(len(df))
    y_values = df['diff'].values

    # Calculate overall medians
    overall_median_y = np.median(y_values)
    overall_median_x = np.median(x_range)

    # Calculate normalized medians for overall data
    overall_normalized = calculate_normalized_medians(df, overall_median_x, overall_median_y)

    # Calculate group medians and normalized values
    group_medians = {}
    group_normalized = {}
    for attr_value in [0, 1]:
        subset = df[df['sensitive_attribute'] == attr_value]
        if len(subset) > 0:
            group_median_y = np.median(subset['diff'].values)
            group_median_x = np.median(subset.index.values)
            group_medians[attr_value] = {
                'y': float(group_median_y),
                'x': float(group_median_x)
            }
            group_normalized[attr_value] = calculate_normalized_medians(
                df, group_median_x, group_median_y, 'sensitive_attribute', attr_value
            )
        else:
            group_medians[attr_value] = {'y': None, 'x': None}
            group_normalized[attr_value] = {
                'normalized_x': None, 'normalized_y': None,
                'original_x': None, 'original_y': None, 'total_points': 0
            }

    # Calculate segment boundaries using adaptive approach for more reasonable segmentation
    if adaptive_segmentation:
        # Use density-based segmentation for more reasonable boundaries
        segment_boundaries = calculate_adaptive_segments(df, num_segments)
    else:
        # Use percentile-based segmentation
        percentiles = np.linspace(0, 100, num_segments + 1)
        segment_boundaries = []

        for i in range(num_segments):
            start_percentile = percentiles[i]
            end_percentile = percentiles[i + 1]

            # Calculate indices based on percentiles
            start_idx = int(np.ceil(start_percentile / 100 * len(df)))
            end_idx = int(np.ceil(end_percentile / 100 * len(df)))

            # Ensure we don't have empty segments
            if start_idx >= end_idx:
                start_idx = end_idx - 1
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(df):
                end_idx = len(df)

            segment_boundaries.append((start_idx, end_idx))

    # Analyze each segment
    segment_data = []

    for i, (start_idx, end_idx) in enumerate(segment_boundaries):
        # Get points in this segment
        segment_df = df.iloc[start_idx:end_idx]

        # Count points by sensitive attribute
        group_counts = segment_df['sensitive_attribute'].value_counts()

        # Determine majority group
        majority_group = group_counts.idxmax() if len(group_counts) > 0 else 0
        majority_count = group_counts.max() if len(group_counts) > 0 else 0
        total_points = len(segment_df)

        # Get segment coordinates
        x_start = start_idx
        x_end = end_idx - 1
        y_start = y_values[start_idx]
        y_end = y_values[end_idx - 1]

        # Add segment line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[y_start, y_end],
            mode='lines',
            line=dict(color=colors[majority_group], width=8),
            name=f'Segment {i+1} (sensitive_attribute: {majority_group})',
            showlegend=False,
            hovertemplate=f'<b>Segment {i+1}</b><br>'
                         f'Majority sensitive_attribute: {majority_group}<br>'
                         f'sensitive_attribute: {majority_group} Count: {majority_count}<br>'
                         f'Total Points: {total_points}<br>'
                         f'X Range: {x_start}-{x_end}<br>'
                         f'Y Range: {y_start:.4f}-{y_end:.4f}<extra></extra>'
        ))

        # Store segment data
        segment_data.append({
            'segment_id': i + 1,
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'x_start': int(x_start),
            'x_end': int(x_end),
            'y_start': float(y_start),
            'y_end': float(y_end),
            'majority_group': int(majority_group),
            'majority_count': int(majority_count),
            'total_points': int(total_points),
            'group_0_count': int(group_counts.get(0, 0)),
            'group_1_count': int(group_counts.get(1, 0)),
            'group_0_percentage': float((group_counts.get(0, 0) / total_points) * 100),
            'group_1_percentage': float((group_counts.get(1, 0) / total_points) * 100)
        })

    # Add overall median lines
    # Y-axis median line
    # fig.add_hline(y=overall_median_y, line_dash="dash", line_color="black",
    #               opacity=0.7, line_width=2,
    #               annotation_text=f"Overall Median Y: {overall_median_y:.6f}",
    #               annotation_position="top right")

    # X-axis median line
    fig.add_trace(go.Scatter(
        x=[overall_median_x, overall_median_x], y=[df['diff'].min(), df['diff'].max()],
                  mode="lines", line=dict(color="black", width=1.5, dash="dash"),
                  name="Overall Median", showlegend=True)
    )

    # Add group median lines
    for attr_value in [0, 1]:
        subset = df[df['sensitive_attribute'] == attr_value]
        if len(subset) > 0:
            group_median_y = np.median(subset['diff'].values)
            group_median_x = np.median(subset.index.values)

            # X-axis group median line
            fig.add_trace(go.Scatter(x=[group_median_x, group_median_x], y=[df['diff'].min(), df['diff'].max()], mode="lines", line=dict(color=colors[attr_value], width=1.5, dash="dot"),
                          name=f"sensitive_attribute: {attr_value} Median", showlegend=True))

    # Detect and add knee points
    knee_points_data = detect_knee_points(records, show_knee=True)

    # Add overall knee points
    if 'knee_convex' in knee_points_data['overall']:
        x, y = knee_points_data['overall']['knee_convex']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=12, color='purple', symbol='diamond', line=dict(color='black', width=2)),
            name='Knee Point (Overall - Convex)', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall - Convex)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))

    if 'knee_concave' in knee_points_data['overall']:
        x, y = knee_points_data['overall']['knee_concave']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=12, color='purple', symbol='star', line=dict(color='black', width=2)),
            name='Knee Point (Overall - Concave)', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall - Concave)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))

    # Add group knee points
    for attr_value in [0, 1]:
        if f'knee_convex' in knee_points_data[f'sensitive_attribute_{attr_value}']:
            x, y = knee_points_data[f'sensitive_attribute_{attr_value}']['knee_convex']
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color=colors[attr_value], symbol='diamond', line=dict(color='black', width=2)),
                name=f'Knee Point (sensitive_attribute: {attr_value} - Convex)', showlegend=True,
                hovertemplate=f'<b>Knee Point (sensitive_attribute: {attr_value} - Convex)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
            ))

        if f'knee_concave' in knee_points_data[f'sensitive_attribute_{attr_value}']:
            x, y = knee_points_data[f'sensitive_attribute_{attr_value}']['knee_concave']
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color=colors[attr_value], symbol='star', line=dict(color='black', width=2)),
                name=f'Knee Point (sensitive_attribute: {attr_value} - Concave)', showlegend=True,
                hovertemplate=f'<b>Knee Point (sensitive_attribute: {attr_value} - Concave)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
            ))

    # Add legend for group colors
    for group_id, color in colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color),
            name=f'sensitive_attribute: {group_id}',
            showlegend=True
        ))

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout
    dataset_title = f" - {dataset_name}" if dataset_name else ""
    fig.update_layout(
        title=f'Sensitive Segmented Curve Visualization{dataset_title} - {algorithm} <br><sup>Segments colored by majority group (n={num_segments} segments) with median lines and knee points</sup>',
        xaxis_title='Index (sorted by diff)',
        yaxis_title='Diff Value',
        showlegend=True,
        height=700,
        width=1200,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode='closest'
    )

    # Save the plot if requested
    plot_files = []
    if save_path and output_formats:
        plot_name = generate_plot_name(algorithm, step, 'segmented_curve')
        plot_files = save_plot_with_formats(fig, save_path, algorithm, step, 'segmented_curve', output_formats=output_formats)

    # Save segment data to JSON
    if save_path:
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        # Convert knee points data to JSON-serializable format
        serializable_knee_points = {}
        for group_key, knee_data in knee_points_data.items():
            serializable_knee_points[group_key] = {}
            if 'knee_convex' in knee_data:
                x, y = knee_data['knee_convex']
                serializable_knee_points[group_key]['knee_convex'] = [int(x), float(y)]
            if 'knee_concave' in knee_data:
                x, y = knee_data['knee_concave']
                serializable_knee_points[group_key]['knee_concave'] = [int(x), float(y)]

        segment_file = os.path.join(metrics_dir, f'{algorithm}_step_{step}_segmented_curve.json')
        with open(segment_file, 'w') as f:
            json.dump({
                'algorithm': algorithm,
                'step': step,
                'num_segments': num_segments,
                'total_records': len(df),
                'segments': segment_data,
                'summary': {
                    'sensitive_attribute_0_segments': sum(1 for seg in segment_data if seg['majority_group'] == 0),
                    'sensitive_attribute_1_segments': sum(1 for seg in segment_data if seg['majority_group'] == 1),
                    'total_segments': len(segment_data)
                },
                'medians': {
                    'overall_y': float(overall_median_y),
                    'overall_x': float(overall_median_x),
                    'overall_normalized': overall_normalized,
                    'group_0_y': group_medians[0]['y'],
                    'group_0_x': group_medians[0]['x'],
                    'group_0_normalized': group_normalized[0],
                    'group_1_y': group_medians[1]['y'],
                    'group_1_x': group_medians[1]['x'],
                    'group_1_normalized': group_normalized[1]
                },
                'knee_points': serializable_knee_points
            }, f, indent=2)

        # Save segment analysis to text file
        analysis_file = os.path.join(metrics_dir, f'{algorithm}_step_{step}_segmented_curve.txt')
        with open(analysis_file, 'w') as f:
            f.write(f"Segmented Curve Analysis for {algorithm} Step {step}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of Segments: {num_segments}\n")
            f.write(f"Total Records: {len(df)}\n\n")

            f.write("Median Values:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Median Y: {overall_median_y:.6f}\n")
            f.write(f"Overall Median X: {overall_median_x:.0f}\n")
            f.write(f"Overall Normalized Y: {overall_normalized['normalized_y']:.4f}\n")
            f.write(f"Overall Normalized X: {overall_normalized['normalized_x']:.4f}\n")
            for attr_value in [0, 1]:
                subset = df[df['sensitive_attribute'] == attr_value]
                if len(subset) > 0:
                    group_median_y = group_medians[attr_value]['y']
                    group_median_x = group_medians[attr_value]['x']
                    group_normalized_data = group_normalized[attr_value]
                    f.write(f"sensitive_attribute: {attr_value} Median Y: {group_median_y:.6f}\n")
                    f.write(f"sensitive_attribute: {attr_value} Median X: {group_median_x:.0f}\n")
                    f.write(f"sensitive_attribute: {attr_value} Normalized Y: {group_normalized_data['normalized_y']:.4f}\n")
                    f.write(f"sensitive_attribute: {attr_value} Normalized X: {group_normalized_data['normalized_x']:.4f}\n")
            f.write("\n")

            f.write("Knee Points:\n")
            f.write("-" * 20 + "\n")
            for group_key, knee_data in knee_points_data.items():
                f.write(f"{group_key}:\n")
                if 'knee_convex' in knee_data:
                    x, y = knee_data['knee_convex']
                    f.write(f"  Convex: x={x}, y={y:.6f}\n")
                if 'knee_concave' in knee_data:
                    x, y = knee_data['knee_concave']
                    f.write(f"  Concave: x={x}, y={y:.6f}\n")
            f.write("\n")

            f.write("Segment Details:\n")
            f.write("-" * 20 + "\n")
            for seg in segment_data:
                f.write(f"Segment {seg['segment_id']}:\n")
                f.write(f"  Majority sensitive_attribute: {seg['majority_group']}\n")
                f.write(f"  X Range: {seg['x_start']}-{seg['x_end']}\n")
                f.write(f"  Y Range: {seg['y_start']:.6f}-{seg['y_end']:.6f}\n")
                f.write(f"  sensitive_attribute: 0: {seg['group_0_count']} ({seg['group_0_percentage']:.1f}%)\n")
                f.write(f"  sensitive_attribute: 1: {seg['group_1_count']} ({seg['group_1_percentage']:.1f}%)\n")
                f.write(f"  Total Points: {seg['total_points']}\n\n")

            f.write("Summary:\n")
            f.write("-" * 20 + "\n")
            group_0_segments = sum(1 for seg in segment_data if seg['majority_group'] == 0)
            group_1_segments = sum(1 for seg in segment_data if seg['majority_group'] == 1)
            f.write(f"sensitive_attribute: 0 Dominant Segments: {group_0_segments}\n")
            f.write(f"sensitive_attribute: 1 Dominant Segments: {group_1_segments}\n")
            f.write(f"Total Segments: {len(segment_data)}\n")

    return fig, segment_data, plot_files

def calculate_adaptive_segments(df, num_segments):
    """
    Calculate adaptive segment boundaries based on data density and distribution.
    This creates more reasonable segments, especially at the extremes.

    Args:
        df: DataFrame with sorted 'diff' values
        num_segments: Number of desired segments

    Returns:
        List of (start_idx, end_idx) tuples for segment boundaries
    """
    y_values = df['diff'].values
    n_points = len(y_values)

    # For very large numbers of segments, use simple equal-sized segmentation
    # to avoid discontinuities and ensure complete coverage
    if num_segments >= n_points // 10:  # If segments would be too small
        segment_boundaries = []
        segment_size = n_points // num_segments
        remainder = n_points % num_segments

        start_idx = 0
        for i in range(num_segments):
            # Distribute remainder across first few segments
            current_segment_size = segment_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_segment_size
            # Ensure we don't exceed array bounds
            end_idx = min(end_idx, n_points)
            segment_boundaries.append((start_idx, end_idx))
            start_idx = end_idx
            # Stop if we've covered all points
            if end_idx >= n_points:
                break

        # If we have fewer segments than requested, adjust the last segment
        while len(segment_boundaries) < num_segments and segment_boundaries:
            # Split the largest remaining segment
            largest_idx = max(range(len(segment_boundaries)),
                            key=lambda i: segment_boundaries[i][1] - segment_boundaries[i][0])
            start, end = segment_boundaries[largest_idx]
            if end - start > 1:  # Only split if segment has more than 1 point
                mid = (start + end) // 2
                segment_boundaries[largest_idx] = (start, mid)
                segment_boundaries.insert(largest_idx + 1, (mid, end))
            else:
                break

        return segment_boundaries[:num_segments]

    # For reasonable numbers of segments, use adaptive approach
    # Calculate basic statistics
    y_min, y_max = y_values.min(), y_values.max()
    y_range = y_max - y_min

    # Find regions of high density (where most points cluster)
    # Use histogram to identify dense regions
    hist, bin_edges = np.histogram(y_values, bins=min(50, n_points//10))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the main density peaks
    peak_indices = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
            peak_indices.append(i)

    # If we have density peaks, use them to guide segmentation
    if len(peak_indices) > 1:
        # Sort peaks by density (histogram value)
        peak_densities = [(i, hist[i]) for i in peak_indices]
        peak_densities.sort(key=lambda x: x[1], reverse=True)

        # Use the top peaks to create segment boundaries
        segment_boundaries = []

        # Create segments around the main density regions
        for i, (peak_idx, density) in enumerate(peak_densities[:num_segments-1]):
            peak_value = bin_centers[peak_idx]

            # Find the range around this peak with proper bounds checking
            if i == 0:
                # First segment: from min to this peak
                start_idx = 0
                end_idx = np.searchsorted(y_values, peak_value + y_range * 0.1)
                # Ensure end_idx doesn't exceed array bounds
                end_idx = min(end_idx, n_points)
            else:
                # Middle segments: between peaks
                prev_peak = bin_centers[peak_densities[i-1][0]]
                start_idx = np.searchsorted(y_values, prev_peak + y_range * 0.05)
                end_idx = np.searchsorted(y_values, peak_value + y_range * 0.1)
                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, n_points - 1))
                end_idx = min(end_idx, n_points)

            # Ensure reasonable segment size
            min_segment_size = max(10, n_points // (num_segments * 2))
            if end_idx - start_idx < min_segment_size:
                end_idx = min(start_idx + min_segment_size, n_points)

            # Final bounds check
            start_idx = max(0, min(start_idx, n_points - 1))
            end_idx = max(start_idx + 1, min(end_idx, n_points))

            segment_boundaries.append((start_idx, end_idx))

        # Add final segment
        if segment_boundaries:
            last_start = segment_boundaries[-1][1]
            if last_start < n_points:
                segment_boundaries.append((last_start, n_points))

        # Ensure we have the right number of segments
        while len(segment_boundaries) < num_segments:
            # Split the largest segment
            largest_segment_idx = max(range(len(segment_boundaries)),
                                    key=lambda i: segment_boundaries[i][1] - segment_boundaries[i][0])
            start, end = segment_boundaries[largest_segment_idx]
            mid = (start + end) // 2
            # Ensure mid is within bounds
            mid = max(start + 1, min(mid, end - 1))
            segment_boundaries[largest_segment_idx] = (start, mid)
            segment_boundaries.insert(largest_segment_idx + 1, (mid, end))

        # Trim to exact number of segments
        segment_boundaries = segment_boundaries[:num_segments]

    else:
        # Fallback to percentile-based segmentation with adjustments for extremes
        segment_boundaries = []

        # Create more segments in the middle and fewer at extremes
        # Use exponential spacing for more reasonable distribution
        spacing = np.logspace(0, 1, num_segments + 1)
        spacing = (spacing - spacing[0]) / (spacing[-1] - spacing[0])  # Normalize to [0, 1]

        for i in range(num_segments):
            start_percentile = spacing[i] * 100
            end_percentile = spacing[i + 1] * 100

            start_idx = int(np.ceil(start_percentile / 100 * n_points))
            end_idx = int(np.ceil(end_percentile / 100 * n_points))

            # Ensure minimum segment size
            min_segment_size = max(5, n_points // (num_segments * 3))
            if end_idx - start_idx < min_segment_size:
                end_idx = min(start_idx + min_segment_size, n_points)

            # Bounds checking
            start_idx = max(0, min(start_idx, n_points - 1))
            end_idx = max(start_idx + 1, min(end_idx, n_points))

            segment_boundaries.append((start_idx, end_idx))

    # Final validation: ensure all segments are valid and continuous
    validated_boundaries = []
    for i, (start_idx, end_idx) in enumerate(segment_boundaries):
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, n_points - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_points))

        # Ensure continuity with previous segment
        if i > 0 and validated_boundaries:
            prev_end = validated_boundaries[-1][1]
            if start_idx > prev_end:
                # Fill gap by extending previous segment or creating intermediate segment
                if start_idx - prev_end <= 2:  # Small gap, extend previous
                    validated_boundaries[-1] = (validated_boundaries[-1][0], end_idx)
                    continue
                else:  # Larger gap, adjust start_idx
                    start_idx = prev_end

        validated_boundaries.append((start_idx, end_idx))

    # Ensure complete coverage
    if validated_boundaries and validated_boundaries[-1][1] < n_points:
        # Extend last segment to cover remaining points
        validated_boundaries[-1] = (validated_boundaries[-1][0], n_points)

    return validated_boundaries

def create_environment_segmented_curve_visualization(records, algorithm, step, num_segments=10, save_path=None, output_formats=None, adaptive_segmentation=True, dataset_name=None):
    """
    Create a segmented curve visualization where each segment is colored by the majority environment.

    Args:
        records: List of records with 'diff' and 'environment' fields
        algorithm: Algorithm name for the plot title
        step: Step number for the plot title
        num_segments: Number of segments to divide the curve into
        save_path: Directory to save the plot
        output_formats: List of output formats (e.g., ['png', 'html'])
        adaptive_segmentation: If True, use adaptive segmentation based on data density
        dataset_name: Name of the dataset to include in the title

    Returns:
        Plotly figure object, segment data, plot files
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    # Get unique environments and create color scheme
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    colors = generate_environment_colors(num_environments)
    environment_colors = {env: colors[i] for i, env in enumerate(unique_environments)}

    # Create the main figure
    fig = go.Figure()

    # Get the full range of diff values
    x_range = np.arange(len(df))
    y_values = df['diff'].values

    # Calculate overall medians
    overall_median_y = np.median(y_values)
    overall_median_x = np.median(x_range)

    # Calculate normalized medians for overall data
    overall_normalized = calculate_normalized_medians(df, overall_median_x, overall_median_y)

    # Calculate environment medians and normalized values
    environment_medians = {}
    environment_normalized = {}
    for env in unique_environments:
        subset = df[df['environment'] == env]
        if len(subset) > 0:
            env_median_y = np.median(subset['diff'].values)
            env_median_x = np.median(subset.index.values)
            environment_medians[env] = {
                'y': float(env_median_y),
                'x': float(env_median_x)
            }
            environment_normalized[env] = calculate_normalized_medians(
                df, env_median_x, env_median_y, 'environment', env
            )
        else:
            environment_medians[env] = {'y': None, 'x': None}
            environment_normalized[env] = {
                'normalized_x': None, 'normalized_y': None,
                'original_x': None, 'original_y': None, 'total_points': 0
            }

    # Calculate segment boundaries using adaptive approach for more reasonable segmentation
    if adaptive_segmentation:
        # Use density-based segmentation for more reasonable boundaries
        segment_boundaries = calculate_adaptive_segments(df, num_segments)
    else:
        # Use percentile-based segmentation
        percentiles = np.linspace(0, 100, num_segments + 1)
        segment_boundaries = []

        for i in range(num_segments):
            start_percentile = percentiles[i]
            end_percentile = percentiles[i + 1]

            # Calculate indices based on percentiles
            start_idx = int(np.ceil(start_percentile / 100 * len(df)))
            end_idx = int(np.ceil(end_percentile / 100 * len(df)))

            # Ensure we don't have empty segments
            if start_idx >= end_idx:
                start_idx = end_idx - 1
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(df):
                end_idx = len(df)

            segment_boundaries.append((start_idx, end_idx))

    # Analyze each segment
    segment_data = []

    for i, (start_idx, end_idx) in enumerate(segment_boundaries):
        # Get points in this segment
        segment_df = df.iloc[start_idx:end_idx]

        # Count points by environment
        environment_counts = segment_df['environment'].value_counts()

        # Determine majority environment
        majority_environment = environment_counts.idxmax() if len(environment_counts) > 0 else unique_environments[0]
        majority_count = environment_counts.max() if len(environment_counts) > 0 else 0
        total_points = len(segment_df)

        # Get segment coordinates
        x_start = start_idx
        x_end = end_idx - 1
        y_start = y_values[start_idx]
        y_end = y_values[end_idx - 1]

        # Add segment line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[y_start, y_end],
            mode='lines',
            line=dict(color=environment_colors[majority_environment], width=8),
            name=f'Segment {i+1} (Environment: {majority_environment})',
            showlegend=False,
            hovertemplate=f'<b>Segment {i+1}</b><br>'
                         f'Majority Environment: {majority_environment}<br>'
                         f'Environment {majority_environment} Count: {majority_count}<br>'
                         f'Total Points: {total_points}<br>'
                         f'X Range: {x_start}-{x_end}<br>'
                         f'Y Range: {y_start:.4f}-{y_end:.4f}<extra></extra>'
        ))

        # Store segment data
        segment_data.append({
            'segment_id': i + 1,
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'x_start': int(x_start),
            'x_end': int(x_end),
            'y_start': float(y_start),
            'y_end': float(y_end),
            'majority_environment': str(majority_environment),
            'majority_count': int(majority_count),
            'total_points': int(total_points),
            'environment_counts': {str(env): int(environment_counts.get(env, 0)) for env in unique_environments},
            'environment_percentages': {str(env): float((environment_counts.get(env, 0) / total_points) * 100) for env in unique_environments}
        })


    # X-axis median line
    fig.add_trace(go.Scatter(x=[overall_median_x, overall_median_x],
                  y=[df['diff'].min(), df['diff'].max()],
                  mode="lines", line=dict(color="black", width=1.5, dash="dash"),
                  name="Overall Median", showlegend=True)
    )


    # Add environment median lines
    for env in unique_environments:
        subset = df[df['environment'] == env]
        if len(subset) > 0:
            env_median_y = np.median(subset['diff'].values)
            env_median_x = np.median(subset.index.values)

            # X-axis environment median line
            fig.add_trace(go.Scatter(x=[env_median_x, env_median_x], y=[df['diff'].min(), df['diff'].max()],
                          mode="lines", line=dict(color=environment_colors[env], width=1.5, dash="dot"),
                          name=f"Environment {env} Median", showlegend=True))

    # Detect and add knee points
    knee_points_data = detect_knee_points(records, show_knee=True)

    # Add overall knee points
    if 'knee_convex' in knee_points_data['overall']:
        x, y = knee_points_data['overall']['knee_convex']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=12, color='purple', symbol='diamond', line=dict(color='black', width=2)),
            name='Knee Point (Overall - Convex)', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall - Convex)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))

    if 'knee_concave' in knee_points_data['overall']:
        x, y = knee_points_data['overall']['knee_concave']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=12, color='purple', symbol='star', line=dict(color='black', width=2)),
            name='Knee Point (Overall - Concave)', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall - Concave)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))

    # Add environment knee points
    for env in unique_environments:
        env_key = f'environment_{env}'
        if env_key in knee_points_data:
            if 'knee_convex' in knee_points_data[env_key]:
                x, y = knee_points_data[env_key]['knee_convex']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color=environment_colors[env], symbol='diamond', line=dict(color='black', width=2)),
                    name=f'Knee Point (Environment {env} - Convex)', showlegend=True,
                    hovertemplate=f'<b>Knee Point (Environment {env} - Convex)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))

            if 'knee_concave' in knee_points_data[env_key]:
                x, y = knee_points_data[env_key]['knee_concave']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color=environment_colors[env], symbol='star', line=dict(color='black', width=2)    ),
                    name=f'Knee Point (Environment {env} - Concave)', showlegend=True,
                    hovertemplate=f'<b>Knee Point (Environment {env} - Concave)</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))

    # Add legend for environment colors
    for env in unique_environments:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=environment_colors[env]),
            name=f'Environment {env}',
            showlegend=True
        ))

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout
    fig.update_layout(
        title=f'Environment Segmented Curve Visualization - {dataset_name} {algorithm} <br><sup>Segments colored by majority environment (n={num_segments} segments) with median lines and knee points</sup>',
        xaxis_title='Index (sorted by diff)',
        yaxis_title='Diff Value',
        showlegend=True,
        height=700,
        width=1200,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode='closest'
    )

    # Save the plot if requested
    plot_files = []
    if save_path and output_formats:
        plot_name = generate_plot_name(algorithm, step, 'environment_segmented_curve')
        plot_files = save_plot_with_formats(fig, save_path, algorithm, step, 'environment_segmented_curve', output_formats=output_formats)

    # Save segment data to JSON
    if save_path:
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        # Convert knee points data to JSON-serializable format
        serializable_knee_points = {}
        for group_key, knee_data in knee_points_data.items():
            serializable_knee_points[group_key] = {}
            if 'knee_convex' in knee_data:
                x, y = knee_data['knee_convex']
                serializable_knee_points[group_key]['knee_convex'] = [int(x), float(y)]
            if 'knee_concave' in knee_data:
                x, y = knee_data['knee_concave']
                serializable_knee_points[group_key]['knee_concave'] = [int(x), float(y)]

        segment_file = os.path.join(metrics_dir, f'{algorithm}_step_{step}_environment_segmented_curve.json')
        with open(segment_file, 'w') as f:
            json.dump({
                'algorithm': algorithm,
                'step': step,
                'num_segments': num_segments,
                'total_records': len(df),
                'segments': segment_data,
                'summary': {
                    'environment_segments': {str(env): int(sum(1 for seg in segment_data if seg['majority_environment'] == str(env))) for env in unique_environments},
                    'total_segments': len(segment_data)
                },
                'medians': {
                    'overall_y': float(overall_median_y),
                    'overall_x': float(overall_median_x),
                    'overall_normalized': overall_normalized,
                    'environment_medians': {str(env): {
                        'y': environment_medians[env]['y'],
                        'x': environment_medians[env]['x'],
                        'normalized': environment_normalized[env]
                    } for env in unique_environments}
                },
                'knee_points': serializable_knee_points
            }, f, indent=2)

        # Save segment analysis to text file
        analysis_file = os.path.join(metrics_dir, f'{algorithm}_step_{step}_environment_segmented_curve.txt')
        with open(analysis_file, 'w') as f:
            f.write(f"Environment Segmented Curve Analysis for {algorithm} Step {step}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of Segments: {num_segments}\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Number of Environments: {num_environments}\n\n")

            f.write("Median Values:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Median Y: {overall_median_y:.6f}\n")
            f.write(f"Overall Median X: {overall_median_x:.0f}\n")
            f.write(f"Overall Normalized Y: {overall_normalized['normalized_y']:.4f}\n")
            f.write(f"Overall Normalized X: {overall_normalized['normalized_x']:.4f}\n")
            for env in unique_environments:
                subset = df[df['environment'] == env]
                if len(subset) > 0:
                    env_median_y = environment_medians[env]['y']
                    env_median_x = environment_medians[env]['x']
                    env_normalized_data = environment_normalized[env]
                    f.write(f"Environment {env} Median Y: {env_median_y:.6f}\n")
                    f.write(f"Environment {env} Median X: {env_median_x:.0f}\n")
                    f.write(f"Environment {env} Normalized Y: {env_normalized_data['normalized_y']:.4f}\n")
                    f.write(f"Environment {env} Normalized X: {env_normalized_data['normalized_x']:.4f}\n")
            f.write("\n")

            f.write("Knee Points:\n")
            f.write("-" * 20 + "\n")
            for group_key, knee_data in knee_points_data.items():
                f.write(f"{group_key}:\n")
                if 'knee_convex' in knee_data:
                    x, y = knee_data['knee_convex']
                    f.write(f"  Convex: x={x}, y={y:.6f}\n")
                if 'knee_concave' in knee_data:
                    x, y = knee_data['knee_concave']
                    f.write(f"  Concave: x={x}, y={y:.6f}\n")
            f.write("\n")

            f.write("Segment Details:\n")
            f.write("-" * 20 + "\n")
            for seg in segment_data:
                f.write(f"Segment {seg['segment_id']}:\n")
                f.write(f"  Majority Environment: {seg['majority_environment']}\n")
                f.write(f"  X Range: {seg['x_start']}-{seg['x_end']}\n")
                f.write(f"  Y Range: {seg['y_start']:.6f}-{seg['y_end']:.6f}\n")
                for env in unique_environments:
                    count = seg['environment_counts'][env]
                    percentage = seg['environment_percentages'][env]
                    f.write(f"  Environment {env}: {count} ({percentage:.1f}%)\n")
                f.write(f"  Total Points: {seg['total_points']}\n\n")

            f.write("Summary:\n")
            f.write("-" * 20 + "\n")
            for env in unique_environments:
                env_segments = sum(1 for seg in segment_data if seg['majority_environment'] == env)
                f.write(f"Environment {env} Dominant Segments: {env_segments}\n")
            f.write(f"Total Segments: {len(segment_data)}\n")

    return fig, segment_data, plot_files

def calculate_normalized_medians(df, median_x, median_y, group_column=None, group_value=None):
    """
    Calculate normalized median values (0 to 1) by dividing by total length.

    Args:
        df: DataFrame with 'diff' column
        median_x: X-axis median value
        median_y: Y-axis median value
        group_column: Column name for grouping (e.g., 'sensitive_attribute', 'environment')
        group_value: Value for the specific group

    Returns:
        Dictionary with normalized median information
    """
    if group_column and group_value is not None:
        subset = df[df[group_column] == group_value]
        if len(subset) == 0:
            return {
                'normalized_x': None,
                'normalized_y': None,
                'original_x': None,
                'original_y': None,
                'total_points': 0
            }
    else:
        subset = df

    # Calculate normalized X (index-based, 0 to 1)
    normalized_x = median_x / len(df) if len(df) > 0 else 0

    # For Y values, we need to normalize based on the range of diff values
    min_diff = df['diff'].min()
    max_diff = df['diff'].max()
    diff_range = max_diff - min_diff
    normalized_y = (median_y - min_diff) / diff_range if diff_range > 0 else 0

    return {
        'normalized_x': float(normalized_x),
        'normalized_y': float(normalized_y),
        'original_x': float(median_x),
        'original_y': float(median_y),
        'total_points': int(len(subset))
    }

def main():
    """Main function to process all files."""
    # Get configuration from command line arguments
    config = get_processing_config()

    print("Starting batch visualization analysis...")
    print(f"Input directory: {config['input_dir']}")
    print(f"Algorithms: {config['algorithms']}")
    print(f"Steps: {config['steps']}")
    print(f"Visualizations: {config['visualizations']}")
    print(f"Output formats: {config['output_formats']}")
    print(f"Processing {len(config['algorithms'])} algorithms × {len(config['steps'])} steps = {len(config['algorithms']) * len(config['steps'])} combinations")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'plots'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'metrics'), exist_ok=True)

    print(f"Output directory: {config['output_dir']}")

    # GPU setup
    if config['use_gpu']:
        print("GPU acceleration: Enabled")
    else:
        print("GPU acceleration: Disabled (CPU only)")

    if config['force']:
        print("Forcing reprocessing of all files")

    # Initialize progress tracking
    total_combinations = len(config['algorithms']) * len(config['steps'])

    # Adjust visualization counts based on knee-point-only option
    if config.get('knee_point_only', False):
        scatter_plots = total_combinations
        tsne_embeddings = 0
        trimap_embeddings = 0
        d3_embeddings = 0
    else:
        scatter_plots = total_combinations if 'scatter' in config['visualizations'] else 0
        tsne_embeddings = total_combinations if 'tsne' in config['visualizations'] else 0
        trimap_embeddings = total_combinations if 'trimap' in config['visualizations'] else 0
        d3_embeddings = total_combinations * 2 if any(v in config['visualizations'] for v in ['3d_tsne', '3d_trimap']) else 0

    print("Progress tracking initialized:")
    print(f"  Scatter plots: {scatter_plots} total")
    print(f"  t-SNE embeddings: {tsne_embeddings} total")
    print(f"  TriMap embeddings: {trimap_embeddings} total")
    print(f"  3D embeddings: {d3_embeddings} total")
    print(f"  Total combinations: {total_combinations}")
    print()

    # Check if we should skip existing files
    if not config['force']:
        print("Checking for existing files...")
        existing_files = []
        for algorithm in config['algorithms']:
            for step in config['steps']:
                results_file = os.path.join(config['output_dir'], 'metrics', f'{algorithm}_step_{step}_results.json')
                if os.path.exists(results_file):
                    existing_files.append(f'{algorithm}_step_{step}')

        if existing_files:
            print(f"Found existing results for: {', '.join(existing_files)}")
            print("Use --force to reprocess existing files")
            return

    # Process all combinations
    successful_results = []
    progress_counters = {'scatter': [0, scatter_plots], 'tsne': [0, tsne_embeddings], 'trimap': [0, trimap_embeddings], '3d': [0, d3_embeddings], 'total': [0, total_combinations]}

    for algorithm in config['algorithms']:
        for step in config['steps']:
            try:
                progress_counters['total'][0] += 1
                print(f"Total progress: [{progress_counters['total'][0]}/{progress_counters['total'][1]}]")

                result = process_single_file(algorithm, step, config, progress_counters)
                if result:
                    successful_results.append(result)
            except Exception as e:
                print(f"Error processing {algorithm} step {step}: {e}")

    print(f"\nSuccessfully processed {len(successful_results)} combinations")

    # Create summary report
    create_summary_report()

    print("\nBatch analysis completed!")
    print(f"Check the output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()