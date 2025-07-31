'''
To run:
cd vizls3/binary_vis
streamlit run interactive_viz.py
'''
import streamlit as st
import json
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import trimap
import re
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy import stats

# Knee/Elbow Point Detection Functions
def find_knee_point(x, y, method='kneedle'):
    """
    Find knee points using a heuristic approach with median as reference.
    Finds one elbow on each side of the median for more accurate detection.

    Args:
        x: x-coordinates (sorted indices)
        y: y-coordinates (diff values)
        method: 'kneedle' for Kneedle algorithm

    Returns:
        knee_points: List of (x, y, type) tuples of knee points
    """
    if len(x) < 3:
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

    # For small datasets (less than 10 points), use a simpler approach
    if len(x_sorted) < 10:
        # Use the elbow method for small datasets
        elbow_point = find_elbow_point(x_sorted, y_sorted)
        if elbow_point:
            knee_points.append((elbow_point[0], elbow_point[1], 'convex_decreasing'))
        return knee_points

    # Split data into left and right halves relative to median
    left_mask = x_sorted <= median_x
    right_mask = x_sorted >= median_x

    # Find left elbow (convex decreasing) - on the left side of median
    if np.sum(left_mask) > 3:  # Reduced minimum from 5 to 3
        left_x = x_sorted[left_mask]
        left_y = y_sorted[left_mask]

        # Use Kneedle algorithm on left subset
        left_elbow = find_kneedle_elbow(left_x, left_y, 'convex_decreasing')
        if left_elbow is not None:
            knee_points.append(left_elbow)

    # Find right elbow (concave increasing) - on the right side of median
    if np.sum(right_mask) > 3:  # Reduced minimum from 5 to 3
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
    if len(x) < 3:  # Reduced from 5 to 3 to match find_knee_point
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
    if first_third_idx > 3:  # Reduced from 5 to 3
        first_x = x_sorted[:first_third_idx]
        first_y = y_sorted[:first_third_idx]

        # Find point with maximum negative curvature
        if len(first_y) > 2:  # Reduced from 3 to 2
            dy = np.diff(first_y)
            dx = np.diff(first_x)
            dx = np.where(dx == 0, 1e-10, dx)
            first_derivative = dy / dx

            if len(first_derivative) > 1:  # Reduced from 2 to 1
                second_derivative = np.diff(first_derivative)
                if len(second_derivative) > 0:
                    min_idx = np.argmin(second_derivative)
                    if min_idx < len(first_x):
                        knee_points.append((first_x[min_idx], first_y[min_idx], 'convex_decreasing'))

    # Find concave increasing point in the last third
    last_third_idx = 2 * len(x_sorted) // 3
    if last_third_idx < len(x_sorted) - 3:  # Reduced from 5 to 3
        last_x = x_sorted[last_third_idx:]
        last_y = y_sorted[last_third_idx:]

        # Find point with maximum positive curvature
        if len(last_y) > 2:  # Reduced from 3 to 2
            dy = np.diff(last_y)
            dx = np.diff(last_x)
            dx = np.where(dx == 0, 1e-10, dx)
            first_derivative = dy / dx

            if len(first_derivative) > 1:  # Reduced from 2 to 1
                second_derivative = np.diff(first_derivative)
                if len(second_derivative) > 0:
                    max_idx = np.argmax(second_derivative)
                    if max_idx < len(last_x):
                        knee_points.append((last_x[max_idx], last_y[max_idx], 'concave_increasing'))

    return knee_points

def find_elbow_point(x, y, method='elbow'):
    """
    Find elbow point using the Elbow method.

    Args:
        x: x-coordinates (sorted indices)
        y: y-coordinates (diff values)
        method: 'elbow' for Elbow method

    Returns:
        elbow_point: (x, y) coordinates of the elbow point
    """
    if len(x) < 3:
        return None

    # For Elbow method, we find the point that maximizes the distance from the line
    # connecting the first and last points

    # Line connecting first and last points
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # Calculate the distance from each point to this line
    distances = []
    for i in range(len(x)):
        # Distance from point (x[i], y[i]) to line through (x1, y1) and (x2, y2)
        if x2 != x1:  # Avoid division by zero
            # Line equation: y = mx + b
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Distance from point to line: |y - (mx + b)| / sqrt(1 + m^2)
            distance = abs(y[i] - (m * x[i] + b)) / np.sqrt(1 + m**2)
        else:
            # Vertical line
            distance = abs(x[i] - x1)

        distances.append(distance)

    # Find the point with maximum distance
    # Use a window to smooth out noise and find the most significant elbow
    window_size = min(5, len(distances) // 4)
    if window_size > 0:
        # Apply moving average to smooth the distances
        smoothed_distances = np.convolve(distances,
                                       np.ones(window_size)/window_size,
                                       mode='valid')
        max_distance_idx = np.argmax(smoothed_distances)
        # Adjust index for the convolution
        elbow_idx = max_distance_idx + window_size // 2
    else:
        elbow_idx = np.argmax(distances)

    if elbow_idx < len(x):
        return (x[elbow_idx], y[elbow_idx])

    return None

def detect_knee_elbow_points(records, show_knee=True, show_elbow=True):
    """
    Detect knee and elbow points for overall data and sensitive attribute groups.
    Now supports two knee points per group: convex and concave.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        show_knee: Whether to detect knee points
        show_elbow: Whether to detect elbow points

    Returns:
        Dictionary containing knee and elbow points for each group
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    # Initialize results with overall and sensitive attribute groups
    results = {
        'overall': {},
        'sensitive_attribute_0': {},
        'sensitive_attribute_1': {},
        'correct_prediction_0': {},
        'correct_prediction_1': {}
    }

    # Dynamically add environment groups based on actual data
    unique_environments = sorted(df['environment'].unique())
    for env in unique_environments:
        results[f'environment_{env}'] = {}

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

    if show_elbow:
        elbow_point = find_elbow_point(x_overall, y_overall)
        if elbow_point:
            results['overall']['elbow'] = elbow_point

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

        if show_elbow:
            elbow_point = find_elbow_point(x_attr, y_attr)
            if elbow_point:
                # Convert back to overall indices
                overall_idx = subset.index[elbow_point[0]]
                results[f'sensitive_attribute_{attr_value}']['elbow'] = (overall_idx, elbow_point[1])

    # Environment groups
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
                    results[f'environment_{env}']['knee_convex'] = (overall_idx, y)
                elif i == 1:
                    results[f'environment_{env}']['knee_concave'] = (overall_idx, y)

        if show_elbow:
            elbow_point = find_elbow_point(x_env, y_env)
            if elbow_point:
                # Convert back to overall indices
                overall_idx = subset.index[elbow_point[0]]
                results[f'environment_{env}']['elbow'] = (overall_idx, elbow_point[1])

    # Correct prediction groups
    for pred_value in [0, 1]:
        subset = df[df['correct_prediction'] == pred_value]
        if len(subset) < 3:
            continue

        x_pred = np.arange(len(subset))
        y_pred = subset['diff'].values

        if show_knee:
            knee_points = find_knee_point(x_pred, y_pred)
            for i, (x, y, point_type) in enumerate(knee_points):
                # Convert back to overall indices
                overall_idx = subset.index[x]
                if i == 0:
                    results[f'correct_prediction_{pred_value}']['knee_convex'] = (overall_idx, y)
                elif i == 1:
                    results[f'correct_prediction_{pred_value}']['knee_concave'] = (overall_idx, y)

        if show_elbow:
            elbow_point = find_elbow_point(x_pred, y_pred)
            if elbow_point:
                # Convert back to overall indices
                overall_idx = subset.index[elbow_point[0]]
                results[f'correct_prediction_{pred_value}']['elbow'] = (overall_idx, elbow_point[1])

    return results

def create_knee_elbow_visualization(records, show_knee=True):
    """
    Create a detailed visualization showing knee point detection process with median lines and jittering.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        show_knee: Whether to show knee point detection

    Returns:
        Plotly figure object
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    # Create subplots
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

    # Add median line for overall data (vertical - y-axis median)
    overall_median = np.median(y_overall)
    fig.add_trace(go.Scatter(
        x=[0, len(df)], y=[overall_median, overall_median],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name=f'Overall Median Y ({overall_median:.4f})', showlegend=True
    ))

    # Add horizontal median line for overall data (x-axis median)
    overall_median_x = np.median(x_overall)
    fig.add_trace(go.Scatter(
        x=[overall_median_x, overall_median_x], y=[df['diff'].min(), df['diff'].max()],
        mode='lines',
        line=dict(color='black', dash='dot', width=2),
        name=f'Overall Median X ({overall_median_x:.0f})', showlegend=True
    ))

    # Detect points
    knee_elbow_points = detect_knee_elbow_points(records, show_knee=True, show_elbow=False)

    # Record all knee point positions
    all_knee_positions = []

    # Add knee points for overall data
    if show_knee and 'overall' in knee_elbow_points and 'knee_convex' in knee_elbow_points['overall']:
        x, y = knee_elbow_points['overall']['knee_convex']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=15, color='red', symbol='diamond', line=dict(color='black', width=2)),
            name='Knee Point (Overall) - Convex', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))
        all_knee_positions.append({
            'group': 'overall',
            'type': 'knee_convex',
            'x': x,
            'y': y,
            'percentile': (x / len(df)) * 100
        })

    if show_knee and 'overall' in knee_elbow_points and 'knee_concave' in knee_elbow_points['overall']:
        x, y = knee_elbow_points['overall']['knee_concave']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=15, color='blue', symbol='diamond', line=dict(color='black', width=2)),
            name='Knee Point (Overall) - Concave', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))
        all_knee_positions.append({
            'group': 'overall',
            'type': 'knee_concave',
            'x': x,
            'y': y,
            'percentile': (x / len(df)) * 100
        })

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
        fig.add_trace(go.Scatter(
            x=[x_attr.min(), x_attr.max()], y=[group_median, group_median],
            mode='lines',
            line=dict(color=colors[attr_value], dash='dash', width=2),
            name=f'Median Attr {attr_value} Y ({group_median:.4f})', showlegend=True
        ))

        # Add horizontal median line for this group (x-axis median)
        group_median_x = np.median(x_attr)
        fig.add_trace(go.Scatter(
            x=[group_median_x, group_median_x], y=[df['diff'].min(), df['diff'].max()],
            mode='lines',
            line=dict(color=colors[attr_value], dash='dot', width=2),
            name=f'Median Attr {attr_value} X ({group_median_x:.0f})', showlegend=True
        ))

        # Add knee points for this group
        if show_knee and f'sensitive_attribute_{attr_value}' in knee_elbow_points and 'knee_convex' in knee_elbow_points[f'sensitive_attribute_{attr_value}']:
            x, y = knee_elbow_points[f'sensitive_attribute_{attr_value}']['knee_convex']
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color=colors[attr_value], symbol='diamond', line=dict(color='black', width=2)),
                name=f'Knee Point (Attr {attr_value}) - Convex', showlegend=True,
                hovertemplate=f'<b>Knee Point (Attr {attr_value}) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
            ))
            all_knee_positions.append({
                'group': f'sensitive_attribute_{attr_value}',
                'type': 'knee_convex',
                'x': x,
                'y': y,
                'percentile': (x / len(df)) * 100
            })

        if show_knee and f'sensitive_attribute_{attr_value}' in knee_elbow_points and 'knee_concave' in knee_elbow_points[f'sensitive_attribute_{attr_value}']:
            x, y = knee_elbow_points[f'sensitive_attribute_{attr_value}']['knee_concave']
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=12, color=colors[attr_value], symbol='diamond', line=dict(color='black', width=2)),
                name=f'Knee Point (Attr {attr_value}) - Concave', showlegend=True,
                hovertemplate=f'<b>Knee Point (Attr {attr_value}) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
            ))
            all_knee_positions.append({
                'group': f'sensitive_attribute_{attr_value}',
                'type': 'knee_concave',
                'x': x,
                'y': y,
                'percentile': (x / len(df)) * 100
            })

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout
    fig.update_layout(
        title='Knee Point Detection Visualization<br><sup>Diamond markers show knee points, Dashed lines: Y-axis medians, Dotted lines: X-axis medians</sup>',
        xaxis_title='Index (sorted by diff)',
        yaxis_title='Diff Value',
        showlegend=True,
        height=700,
        width=1200,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode='closest'
    )

    # Store knee positions in session state for analysis
    if 'knee_positions' not in st.session_state:
        st.session_state.knee_positions = []
    st.session_state.knee_positions = all_knee_positions

    return fig

def display_knee_elbow_analysis(records, show_knee=True):
    """
    Display detailed analysis of knee points.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        show_knee: Whether to analyze knee points
    """
    if not show_knee:
        return

    knee_elbow_points = detect_knee_elbow_points(records, show_knee=True, show_elbow=False)
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    st.subheader("Knee Point Analysis")

    # Create a summary table
    analysis_data = []

    for group, points in knee_elbow_points.items():
        if group == 'overall':
            group_name = 'Overall Data'
            subset = df
        elif group.startswith('sensitive_attribute_'):
            attr_value = group.split('_')[-1]
            group_name = f'Sensitive Attribute {attr_value}'
            subset = df[df['sensitive_attribute'] == int(attr_value)]
        elif group.startswith('environment_'):
            env_value = group.split('_')[-1]
            group_name = f'Environment {env_value}'
            subset = df[df['environment'] == int(env_value)]
        elif group.startswith('correct_prediction_'):
            pred_value = group.split('_')[-1]
            group_name = f'Correct Prediction {pred_value}'
            subset = df[df['correct_prediction'] == int(pred_value)]
        else:
            continue

        if 'knee_convex' in points and show_knee:
            x, y = points['knee_convex']
            analysis_data.append({
                'Group': group_name,
                'Point Type': 'Knee Point - Convex',
                'Index': x,
                'Diff Value': f"{y:.4f}",
                'Sample Size': len(subset),
                'Percentile': f"{((x / len(df)) * 100):.1f}%"
            })

        if 'knee_concave' in points and show_knee:
            x, y = points['knee_concave']
            analysis_data.append({
                'Group': group_name,
                'Point Type': 'Knee Point - Concave',
                'Index': x,
                'Diff Value': f"{y:.4f}",
                'Sample Size': len(subset),
                'Percentile': f"{((x / len(df)) * 100):.1f}%"
            })

    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        st.dataframe(analysis_df, use_container_width=True)

        # Add interpretation
        st.subheader("Interpretation")
        st.write("""
        **Knee Points (Diamond/Star markers):** Points of maximum curvature where the rate of change in the diff values
        changes most dramatically. These often indicate transition points in the data distribution.

        **Convex Knee Points (Diamond):** Found on the left side of the median, typically indicate early transition points.
        **Concave Knee Points (Star):** Found on the right side of the median, typically indicate late transition points.

        **Overall vs. Filter Groups:** Comparing knee points across different groups can reveal
        differences in how prediction errors are distributed for different demographic groups, environments, or prediction outcomes.
        """)
    else:
        st.warning("No knee points detected. This may indicate insufficient data or very smooth distributions.")

# /Users/zc/Dev/vizls3/binary_vis/data/BDD/
# or we have 5 different datasets BDD/CCMNIST1/FairFace/NYPD/YFCC
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/CCMNIST1/')

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

# SagNet, Mixup, MBDG, IGA, IRM, GroupDRO, Fish, ERM
# 0, 4000, 8000(only available)
PREDICTIONS_JSON = os.path.join(DATA_DIR, 'MBDG_step_8000_predictions.json')
# METRICS_FILE will be set dynamically based on the predictions JSON
METRICS_FILE = None
# METRICS_FILE = os.path.join(DATA_DIR, 'SagNet.txt')

def load_processed_predictions(data):
    """Load predictions from either file path or uploaded file content."""
    if isinstance(data, str):
        with open(data, 'r') as f:
            data = json.load(f)
    elif isinstance(data, bytes):
        data = json.loads(data.decode('utf-8'))

    return [{
        'name': item['filename'],
        'diff': item['predicted_class'] - item['predicted_probabilities'][1],
        'predicted_class': item['predicted_class'],
        'environment': item['environment'],
        'sensitive_attribute': item['sensitive_attribute'],
        'correct_prediction': item['correct_prediction']
    } for item in data]

def create_scatter_trace(subset, x_values, color, name, show_lines, show_error_bars, overall_sort, point_size=4, opacity=0.4, jitter_amount=0.008, density_sizing=False, show_point_counts=True):
    """Create a scatter trace with consistent styling."""
    # Apply jittering if specified
    if jitter_amount > 0:
        # Use a fixed seed for reproducible jittering
        np.random.seed(42)
        jittered_x = np.array(x_values) + np.random.normal(0, jitter_amount, len(x_values))
        jittered_y = np.array(subset['diff']) + np.random.normal(0, jitter_amount, len(subset['diff']))
    else:
        jittered_x = x_values
        jittered_y = subset['diff']

    # Apply density-based sizing if enabled
    if density_sizing and len(subset) > 1:
        try:
            # Calculate point sizes based on local density
            coords = np.column_stack([jittered_x, jittered_y])
            distances = squareform(pdist(coords))
            # Count points within a small radius for each point
            density = np.sum(distances < 0.1, axis=1)
            # Scale sizes based on density (more density = larger points)
            sizes = point_size + (density - 1) * 2
            sizes = np.clip(sizes, point_size, point_size * 3)  # Cap the maximum size
        except ImportError:
            # Fallback if scipy is not available
            sizes = point_size
    else:
        sizes = point_size

    # Create legend name with or without point count
    legend_name = name if not show_point_counts else f'{name} (n={len(subset)})'

    error_y_config = dict(type='data', array=subset['error'], visible=show_error_bars) if show_error_bars else None
    return go.Scatter(
        x=jittered_x,
        y=jittered_y,
        mode='markers+lines' if show_lines else 'markers',
        marker=dict(size=sizes, color=color, opacity=opacity),
        line=dict(color=color, width=1) if show_lines else None,
        name=legend_name,
        error_y=error_y_config,
        hovertemplate='<b>%{text}</b><br>diff: %{y:.4f}<br>environment: %{customdata[0]}<br>sensitive_attribute: %{customdata[1]}<br>correct_prediction: %{customdata[2]}<extra></extra>',
        text=subset['name'],
        customdata=list(zip(subset['environment'], subset['sensitive_attribute'], subset['correct_prediction']))
    )

def create_filtered_scatter_plot(records, show_environment=False, show_sensitive_attribute=True, show_correct_prediction=False,
                                show_error_bars=False, overall_sort=True, show_lines=False,
                                point_size=4, opacity=0.8, enable_jitter=True, jitter_amount=0.008,
                                custom_jitter=0.008, use_custom_jitter=False, density_sizing=False,
                                show_point_counts=True, show_knee_points=True,
                                show_overall_knee=True, show_sensitive_knee=True, show_environment_knee=True, show_correct_prediction_knee=True,
                                show_convex_knee=True, show_concave_knee=True):
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)
    df['error'] = 0.05

    # Dynamically update environment colors based on actual data
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    environment_colors = generate_environment_colors(num_environments)

    # Color schemes
    colors = {
        'sensitive_attribute': {0: '#01BEFE', 1: '#FFDD00'},
        'environment': environment_colors,
        'correct_prediction': {0: '#ADFF02', 1: '#8F00FF'},
        'all_three': {
            (0, 0, 0): '#01BEFE', (0, 0, 1): '#FFDD00', (0, 1, 0): '#FF7D00', (0, 1, 1): '#FF006D',
            (1, 0, 0): '#ADFF02', (1, 0, 1): '#8F00FF', (1, 1, 0): '#01BEFE', (1, 1, 1): '#FFDD00'
        }
    }

    # Use custom jitter if specified
    if use_custom_jitter:
        jitter_amount = custom_jitter

    # Only apply jittering if enabled
    if not enable_jitter:
        jitter_amount = 0.0

    active_filters = [f for f, show in [('environment', show_environment), ('sensitive_attribute', show_sensitive_attribute), ('correct_prediction', show_correct_prediction)] if show]

    traces = []
    median_positions = {}

    def add_median_line(x, color, name):
        traces.append(go.Scatter(x=[x, x], y=[df['diff'].min(), df['diff'].max()], mode='lines',
                                line=dict(color=color, dash='dash'), name=name, showlegend=True))
        median_positions[name] = x

    # Create traces based on active filters
    if not active_filters:
        # No filters - single trace
        trace = create_scatter_trace(df, list(range(len(df))), 'blue', 'All Points', show_lines, show_error_bars, overall_sort, point_size, opacity, jitter_amount, density_sizing, show_point_counts)
        traces.append(trace)
        add_median_line(int(np.median(np.arange(len(df)))), 'black', 'Median (all data)')

    elif len(active_filters) == 1:
        # Single filter
        filter_name = active_filters[0]
        unique_values = sorted(df[filter_name].unique())
        for value in unique_values:
            subset = df[df[filter_name] == value]
            if len(subset) == 0: continue

            x_values = subset.index if overall_sort else list(range(len(subset)))
            color = colors[filter_name].get(value, '#808080')  # Default gray if not in scheme
            name = f'{filter_name.capitalize()}: {value}'

            trace = create_scatter_trace(subset, x_values, color, name, show_lines, show_error_bars, overall_sort, point_size, opacity, jitter_amount, density_sizing, show_point_counts)
            traces.append(trace)

            if len(subset) > 0:
                median_idx = int(np.median(subset.index if overall_sort else np.arange(len(subset))))
                add_median_line(median_idx, color, f'Median {filter_name[0].upper()}={value}')

    else:
        # Multiple filters
        filter_combinations = []
        if len(active_filters) == 2:
            if 'environment' in active_filters and 'sensitive_attribute' in active_filters:
                # Only group by environment and sensitive_attribute, ignore correct_prediction
                filter_combinations = [(e, s, None) for e in [0, 1] for s in [0, 1]]
            elif 'environment' in active_filters and 'correct_prediction' in active_filters:
                # Only group by environment and correct_prediction, ignore sensitive_attribute
                filter_combinations = [(e, None, p) for e in [0, 1] for p in [0, 1]]
            elif 'sensitive_attribute' in active_filters and 'correct_prediction' in active_filters:
                # Only group by sensitive_attribute and correct_prediction, ignore environment
                filter_combinations = [(None, s, p) for s in [0, 1] for p in [0, 1]]
        else:
            # All three filters
            filter_combinations = [(e, s, p) for e in [0, 1] for s in [0, 1] for p in [0, 1]]

        for e, s, p in filter_combinations:
            # Build filter condition based on which attributes are selected
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
                # All three filters
                subset = df[(df['environment'] == e) & (df['sensitive_attribute'] == s) & (df['correct_prediction'] == p)]
                name = f'E={e}, S={s}, P={p}'

            if len(subset) == 0: continue

            x_values = subset.index if overall_sort else list(range(len(subset)))
            # Use appropriate color scheme based on number of active filters
            if len(active_filters) == 2:
                if 'environment' in active_filters and 'sensitive_attribute' in active_filters:
                    color = colors['all_three'].get((e, s, 0), 'gray')  # Use first correct_prediction color
                elif 'environment' in active_filters and 'correct_prediction' in active_filters:
                    color = colors['all_three'].get((e, 0, p), 'gray')  # Use first sensitive_attribute color
                elif 'sensitive_attribute' in active_filters and 'correct_prediction' in active_filters:
                    color = colors['all_three'].get((0, s, p), 'gray')  # Use first environment color
            else:
                color = colors['all_three'].get((e, s, p), 'gray')

            trace = create_scatter_trace(subset, x_values, color, name, show_lines, show_error_bars, overall_sort, point_size, opacity, jitter_amount, density_sizing, show_point_counts)
            traces.append(trace)

            if len(subset) > 0:
                median_idx = int(np.median(subset.index if overall_sort else np.arange(len(subset))))
                add_median_line(median_idx, color, f'Median {name}')

    # Add overall median and y=0 line
    if len(df) > 0:
        add_median_line(int(np.median(np.arange(len(df)))), 'black', 'Median (all data)')

    traces.append(go.Scatter(x=[0, len(df)], y=[0, 0], mode='lines', line=dict(color='black', dash='dash'), name='y=0', showlegend=True))

    # Detect and add knee points based on filter options
    if show_knee_points:
        knee_elbow_points = detect_knee_elbow_points(records, show_knee=True, show_elbow=False)

        # Add overall knee points
        if show_overall_knee and 'overall' in knee_elbow_points:
            if show_convex_knee and 'knee_convex' in knee_elbow_points['overall']:
                x, y = knee_elbow_points['overall']['knee_convex']
                traces.append(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color='purple', symbol='diamond', line=dict(color='black', width=2)),
                    name='Knee Point (Overall) - Convex', showlegend=True,
                    hovertemplate=f'<b>Knee Point (Overall) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))

            if show_concave_knee and 'knee_concave' in knee_elbow_points['overall']:
                x, y = knee_elbow_points['overall']['knee_concave']
                traces.append(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color='purple', symbol='star', line=dict(color='black', width=2)),
                    name='Knee Point (Overall) - Concave', showlegend=True,
                    hovertemplate=f'<b>Knee Point (Overall) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))

        # Add sensitive attribute knee points
        if show_sensitive_knee:
            for attr_value in [0, 1]:
                group_key = f'sensitive_attribute_{attr_value}'
                if group_key in knee_elbow_points:
                    color = colors['sensitive_attribute'].get(attr_value, 'red')
                    if show_convex_knee and 'knee_convex' in knee_elbow_points[group_key]:
                        x, y = knee_elbow_points[group_key]['knee_convex']
                        traces.append(go.Scatter(
                            x=[x], y=[y], mode='markers',
                            marker=dict(size=12, color=color, symbol='diamond', line=dict(color='black', width=2)),
                            name=f'Knee Point (Attr {attr_value}) - Convex', showlegend=True,
                            hovertemplate=f'<b>Knee Point (Attr {attr_value}) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                        ))

                    if show_concave_knee and 'knee_concave' in knee_elbow_points[group_key]:
                        x, y = knee_elbow_points[group_key]['knee_concave']
                        traces.append(go.Scatter(
                            x=[x], y=[y], mode='markers',
                            marker=dict(size=12, color=color, symbol='star', line=dict(color='black', width=2)),
                            name=f'Knee Point (Attr {attr_value}) - Concave', showlegend=True,
                            hovertemplate=f'<b>Knee Point (Attr {attr_value}) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                        ))

        # Add environment knee points
        if show_environment_knee:
            for env in unique_environments:
                group_key = f'environment_{env}'
                if group_key in knee_elbow_points:
                    color = colors['environment'].get(env, 'green')
                    if show_convex_knee and 'knee_convex' in knee_elbow_points[group_key]:
                        x, y = knee_elbow_points[group_key]['knee_convex']
                        traces.append(go.Scatter(
                            x=[x], y=[y], mode='markers',
                            marker=dict(size=12, color=color, symbol='diamond', line=dict(color='black', width=2)),
                            name=f'Knee Point (Env {env}) - Convex', showlegend=True,
                            hovertemplate=f'<b>Knee Point (Env {env}) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                        ))

                    if show_concave_knee and 'knee_concave' in knee_elbow_points[group_key]:
                        x, y = knee_elbow_points[group_key]['knee_concave']
                        traces.append(go.Scatter(
                            x=[x], y=[y], mode='markers',
                            marker=dict(size=12, color=color, symbol='star', line=dict(color='black', width=2)),
                            name=f'Knee Point (Env {env}) - Concave', showlegend=True,
                            hovertemplate=f'<b>Knee Point (Env {env}) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                        ))

        # Add correct prediction knee points
        if show_correct_prediction_knee:
            for pred_value in [0, 1]:
                group_key = f'correct_prediction_{pred_value}'
                if group_key in knee_elbow_points:
                    color = colors['correct_prediction'].get(pred_value, 'purple')
                    if show_convex_knee and 'knee_convex' in knee_elbow_points[group_key]:
                        x, y = knee_elbow_points[group_key]['knee_convex']
                        traces.append(go.Scatter(
                            x=[x], y=[y], mode='markers',
                            marker=dict(size=12, color=color, symbol='diamond', line=dict(color='black', width=2)),
                            name=f'Knee Point (Pred {pred_value}) - Convex', showlegend=True,
                            hovertemplate=f'<b>Knee Point (Pred {pred_value}) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                        ))

                    if show_concave_knee and 'knee_concave' in knee_elbow_points[group_key]:
                        x, y = knee_elbow_points[group_key]['knee_concave']
                        traces.append(go.Scatter(
                            x=[x], y=[y], mode='markers',
                            marker=dict(size=12, color=color, symbol='star', line=dict(color='black', width=2)),
                            name=f'Knee Point (Pred {pred_value}) - Concave', showlegend=True,
                            hovertemplate=f'<b>Knee Point (Pred {pred_value}) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                        ))

    # Display median distances
    if 'Median (all data)' in median_positions and len(median_positions) > 1:
        st.subheader("Median Line Distances")
        overall_median_pos = median_positions['Median (all data)']
        distances = [(name, pos - overall_median_pos) for name, pos in median_positions.items() if name != 'Median (all data)']

        if distances:
            distance_df = pd.DataFrame(distances, columns=['Median Line', 'Distance from Overall Median'])
            distance_df['Distance from Overall Median'] = distance_df['Distance from Overall Median'].round(2)
            st.dataframe(distance_df, use_container_width=True)

            distances_values = [d[1] for d in distances]
            st.write(f"**Summary:** Average: {np.mean(distances_values):.2f}, Max: {np.max(distances_values):.2f}, Min: {np.min(distances_values):.2f}, Std: {np.std(distances_values):.2f}")

    return go.Figure(data=traces, layout=go.Layout(
        title='Individual Points: diff (Sorted by Difference)<br><sup>Filter by attributes using sidebar controls</sup>',
        xaxis=dict(title='Index (sorted by diff)'), yaxis=dict(title='diff'),
        showlegend=True, hovermode='closest', height=500, width=1200, margin=dict(l=60, r=30, t=60, b=60)
    ))

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

def create_segmented_curve_visualization(records, num_segments=10, adaptive_segmentation=True,
                                       color_by='sensitive_attribute', show_environment=False,
                                       show_sensitive_attribute=True, show_correct_prediction=False,
                                       show_knee_points=True, show_median_lines=True, segment_opacity=0.8,
                                       show_segment_boundaries=True):
    """
    Create a segmented curve visualization where each segment is colored by the majority group.

    Args:
        records: List of records with 'diff' and filter fields
        num_segments: Number of segments to divide the curve into
        adaptive_segmentation: If True, use adaptive segmentation based on data density
        color_by: Which attribute to use for coloring segments ('sensitive_attribute', 'environment', 'correct_prediction')
        show_environment: Whether to consider environment in filtering
        show_sensitive_attribute: Whether to consider sensitive attribute in filtering
        show_correct_prediction: Whether to consider correct prediction in filtering

    Returns:
        Plotly figure object and segment data
    """
    df = pd.DataFrame(records).sort_values('diff').reset_index(drop=True)

    # Determine which attribute to use for coloring based on filter options
    if show_environment and not show_sensitive_attribute and not show_correct_prediction:
        color_by = 'environment'
    elif show_sensitive_attribute and not show_environment and not show_correct_prediction:
        color_by = 'sensitive_attribute'
    elif show_correct_prediction and not show_environment and not show_sensitive_attribute:
        color_by = 'correct_prediction'
    else:
        # Default to sensitive_attribute if multiple filters or no specific preference
        color_by = 'sensitive_attribute'

    # Color schemes for different attributes
    colors = {
        'sensitive_attribute': {0: '#01BEFE', 1: '#FFDD00'},
        'environment': generate_environment_colors(len(df['environment'].unique())),
        'correct_prediction': {0: '#ADFF02', 1: '#8F00FF'}
    }

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

    # Get unique values for the coloring attribute
    unique_values = sorted(df[color_by].unique())

    for value in unique_values:
        subset = df[df[color_by] == value]
        if len(subset) > 0:
            group_median_y = np.median(subset['diff'].values)
            group_median_x = np.median(subset.index.values)
            group_medians[value] = {
                'y': float(group_median_y),
                'x': float(group_median_x)
            }
            group_normalized[value] = calculate_normalized_medians(
                df, group_median_x, group_median_y, color_by, value
            )
        else:
            group_medians[value] = {'y': None, 'x': None}
            group_normalized[value] = {
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

        # Count points by the coloring attribute
        group_counts = segment_df[color_by].value_counts()

        # Determine majority group
        majority_group = group_counts.idxmax() if len(group_counts) > 0 else unique_values[0]
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
            line=dict(color=colors[color_by][majority_group], width=8),
            name=f'Segment {i+1} ({color_by}: {majority_group})',
            showlegend=False,
            hovertemplate=f'<b>Segment {i+1}</b><br>'
                         f'Majority {color_by}: {majority_group}<br>'
                         f'{color_by}: {majority_group} Count: {majority_count}<br>'
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
            'coloring_attribute': color_by,
            'group_counts': group_counts.to_dict(),
            'group_percentages': {k: float((v / total_points) * 100) for k, v in group_counts.items()}
        })

    # Add overall median lines
    # X-axis median line
    fig.add_trace(go.Scatter(
        x=[overall_median_x, overall_median_x], y=[df['diff'].min(), df['diff'].max()],
                  mode="lines", line=dict(color="black", width=1.5, dash="dash"),
                  name="Overall Median", showlegend=True)
    )

    # Add group median lines
    for value in unique_values:
        subset = df[df[color_by] == value]
        if len(subset) > 0:
            group_median_y = np.median(subset['diff'].values)
            group_median_x = np.median(subset.index.values)

            # X-axis group median line
            fig.add_trace(go.Scatter(x=[group_median_x, group_median_x], y=[df['diff'].min(), df['diff'].max()],
                          mode="lines", line=dict(color=colors[color_by][value], width=1.5, dash="dot"),
                          name=f"{color_by}: {value} Median", showlegend=True))

    # Detect and add knee points
    knee_elbow_points = detect_knee_elbow_points(records, show_knee=True, show_elbow=False)

    # Add overall knee points
    if 'overall' in knee_elbow_points and 'knee_convex' in knee_elbow_points['overall']:
        x, y = knee_elbow_points['overall']['knee_convex']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=12, color='purple', symbol='diamond', line=dict(color='black', width=2)),
            name='Knee Point (Overall) - Convex', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))

    if 'overall' in knee_elbow_points and 'knee_concave' in knee_elbow_points['overall']:
        x, y = knee_elbow_points['overall']['knee_concave']
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=12, color='blue', symbol='star', line=dict(color='black', width=2)),
            name='Knee Point (Overall) - Concave', showlegend=True,
            hovertemplate=f'<b>Knee Point (Overall) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
        ))

    # Add knee points for the coloring attribute groups
    for value in unique_values:
        group_key = f'{color_by}_{value}'
        if group_key in knee_elbow_points:
            if 'knee_convex' in knee_elbow_points[group_key]:
                x, y = knee_elbow_points[group_key]['knee_convex']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color=colors[color_by][value], symbol='diamond', line=dict(color='black', width=2)),
                    name=f'Knee Point ({color_by}: {value}) - Convex', showlegend=True,
                    hovertemplate=f'<b>Knee Point ({color_by}: {value}) - Convex</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))

            if 'knee_concave' in knee_elbow_points[group_key]:
                x, y = knee_elbow_points[group_key]['knee_concave']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=12, color=colors[color_by][value], symbol='star', line=dict(color='black', width=2)),
                    name=f'Knee Point ({color_by}: {value}) - Concave', showlegend=True,
                    hovertemplate=f'<b>Knee Point ({color_by}: {value}) - Concave</b><br>x: {x}<br>y: {y:.4f}<extra></extra>'
                ))

    # Add legend for group colors
    for value, color in colors[color_by].items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color),
            name=f'{color_by}: {value}',
            showlegend=True
        ))

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout
    fig.update_layout(
        title=f'Segmented Curve Visualization - Colored by {color_by.replace("_", " ").title()}<br><sup>Segments colored by majority group (n={num_segments} segments) with median lines and knee points</sup>',
        xaxis_title='Index (sorted by diff)',
        yaxis_title='Diff Value',
        showlegend=True,
        height=700,
        width=1200,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode='closest'
    )

    # Add segment boundaries if requested
    if show_segment_boundaries:
        for start_idx, end_idx in segment_boundaries:
            fig.add_vline(x=start_idx, line_dash="dash", line_color="gray", opacity=0.5)

    return fig, segment_data

def create_scatter_plot_with_sidebar(records):
    """Create scatter plot with controls in the sidebar."""
    with st.sidebar:
        st.header("Difference Distribution Controls")

        # Visualization type selection
        st.subheader("Visualization Type")
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Scatter Plot", "Segmented Curve"],
            help="Scatter Plot: Individual points with filtering. Segmented Curve: Segments colored by majority group."
        )

        # Filter options
        st.subheader("Filter Options")
        show_environment = st.checkbox("Color by Environment", value=False)
        show_sensitive_attribute = st.checkbox("Color by Sensitive Attribute", value=True)
        show_correct_prediction = st.checkbox("Color by Correct Prediction", value=False)
        show_error_bars = st.checkbox("Show Error Bars", value=False)

        overall_sort = st.checkbox("Sort All Points by Y-Axis Values", value=True)
        show_lines = st.checkbox("Show Lines Between Points", value=False)

        # Point styling controls
        st.subheader("Point Styling Controls")
        point_size = st.slider("Point Size", min_value=1, max_value=20, value=4, step=1)
        opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

        enable_jitter = st.checkbox("Enable Jittering", value=True, help="Adds random noise to prevent overlapping points")
        jitter_amount = st.slider("Jitter Amount", min_value=0.0, max_value=0.5, value=0.008, step=0.001,
                                 disabled=not enable_jitter, help="Amount of random noise to add")

        # Direct jitter input for precise control
        if enable_jitter:
            custom_jitter = st.number_input("Custom Jitter Amount",
                                          min_value=0.0, max_value=1.0, value=0.008, step=0.001,
                                          format="%.3f", help="Enter exact jitter amount")
            use_custom_jitter = st.checkbox("Use Custom Jitter", value=False,
                                          help="Use the custom value instead of slider")
        else:
            custom_jitter = 0.008
            use_custom_jitter = False

        # Additional controls
        density_sizing = st.checkbox("Density-based Sizing", value=False,
                                   help="Larger points for areas with more overlapping data")
        show_point_counts = st.checkbox("Show Point Counts", value=True,
                                      help="Display count of points in each group")

        # Knee Point Detection Controls
        st.subheader("Knee Point Detection")
        show_knee_points = st.checkbox("Show Knee Points", value=True,
                                     help="Detect and display knee points (diamond markers)")

        # Knee point filter options
        if show_knee_points:
            st.write("**Knee Point Detection Groups:**")
            show_overall_knee = st.checkbox("Overall Data", value=True)
            show_sensitive_knee = st.checkbox("Sensitive Attribute Groups", value=True)
            show_environment_knee = st.checkbox("Environment Groups", value=False)
            show_correct_prediction_knee = st.checkbox("Correct Prediction Groups", value=False)

            # Knee point type options
            st.write("**Knee Point Types:**")
            show_convex_knee = st.checkbox("Convex Knee Points", value=True, help="Diamond markers - found on left side of median")
            show_concave_knee = st.checkbox("Concave Knee Points", value=True, help="Star markers - found on right side of median")

        # Additional knee point controls
        show_dedicated_viz = st.checkbox("Show Dedicated Knee Point Visualization", value=True,
                                       help="Display a separate visualization focused on knee point detection")

        # Segmented Curve Controls (only show if segmented curve is selected)
        if viz_type == "Segmented Curve":
            st.subheader("Segmented Curve Controls")

            # Make the number of segments control more prominent
            st.write("**Segment Configuration:**")
            num_segments = st.slider("Number of Segments", min_value=3, max_value=1000, value=10, step=1,
                                   help="Number of segments to divide the curve into. More segments provide finer detail but may be harder to interpret.")

            # Display current segment count
            st.write(f"**Current: {num_segments} segments**")

            # Other segmented curve options
            st.write("**Segmentation Options:**")
            adaptive_segmentation = st.checkbox("Adaptive Segmentation", value=True,
                                             help="Use adaptive segmentation based on data density for more meaningful segments")
            show_segment_analysis = st.checkbox("Show Segment Analysis", value=True,
                                             help="Display detailed analysis of segments including statistics and charts")

            # Set default color_by based on filter options
            if show_environment and not show_sensitive_attribute and not show_correct_prediction:
                color_by = 'environment'
            elif show_sensitive_attribute and not show_environment and not show_correct_prediction:
                color_by = 'sensitive_attribute'
            elif show_correct_prediction and not show_environment and not show_sensitive_attribute:
                color_by = 'correct_prediction'
            else:
                color_by = 'sensitive_attribute'  # default

            # Advanced segment controls
            st.write("**Advanced Options:**")
            show_knee_points_in_segments = st.checkbox("Show Knee Points in Segments", value=True,
                                                     help="Display knee points on the segmented curve visualization")
            show_median_lines = st.checkbox("Show Median Lines", value=True,
                                          help="Display median lines for each group in the segmented curve")

            # Set default values for missing variables
            segment_opacity = 0.8
            show_segment_boundaries = True

    # Create the appropriate visualization
    if viz_type == "Scatter Plot":
        # Create the plot with the sidebar parameters
        fig = create_filtered_scatter_plot(
            records,
            show_environment=show_environment,
            show_sensitive_attribute=show_sensitive_attribute,
            show_correct_prediction=show_correct_prediction,
            show_error_bars=show_error_bars,
            overall_sort=overall_sort,
            show_lines=show_lines,
            point_size=point_size,
            opacity=opacity,
            enable_jitter=enable_jitter,
            jitter_amount=jitter_amount,
            custom_jitter=custom_jitter,
            use_custom_jitter=use_custom_jitter,
            density_sizing=density_sizing,
            show_point_counts=show_point_counts,
            show_knee_points=show_knee_points,
            show_overall_knee=show_overall_knee if show_knee_points else False,
            show_sensitive_knee=show_sensitive_knee if show_knee_points else False,
            show_environment_knee=show_environment_knee if show_knee_points else False,
            show_correct_prediction_knee=show_correct_prediction_knee if show_knee_points else False,
            show_convex_knee=show_convex_knee if show_knee_points else False,
            show_concave_knee=show_concave_knee if show_knee_points else False
        )
        return fig, show_dedicated_viz, show_knee_points, None, None
    else:
        # Create segmented curve visualization
        fig, segment_data = create_segmented_curve_visualization(
            records,
            num_segments=num_segments,
            adaptive_segmentation=adaptive_segmentation,
            color_by=color_by,
            show_environment=show_environment,
            show_sensitive_attribute=show_sensitive_attribute,
            show_correct_prediction=show_correct_prediction,
            show_knee_points=show_knee_points_in_segments,
            show_median_lines=show_median_lines,
            segment_opacity=segment_opacity,
            show_segment_boundaries=show_segment_boundaries
        )
        return fig, False, False, segment_data, show_segment_analysis

def create_embedding_visualization(records, method='tsne'):
    """Create t-SNE or TriMap visualization."""
    df = pd.DataFrame(records)

    # Dynamically update environment colors based on actual data
    unique_environments = sorted(df['environment'].unique())
    num_environments = len(unique_environments)
    environment_colors = generate_environment_colors(num_environments)

    color_by = st.selectbox("Color points by:", ['environment', 'sensitive_attribute', 'correct_prediction', 'predicted_class'], key=f'{method}_color')

    # Point styling controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        point_size = st.slider("Point Size", min_value=1, max_value=20, value=8, step=1, key=f'{method}_size')
    with col2:
        opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1, key=f'{method}_opacity')
    with col3:
        enable_jitter = st.checkbox("Enable Jittering", value=False, help="Adds random noise to prevent overlapping points", key=f'{method}_enable_jitter')
    with col4:
        jitter_amount = st.slider("Jitter Amount", min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                 disabled=not enable_jitter, help="Amount of random noise to add", key=f'{method}_jitter')

    # Only apply jittering if enabled
    if not enable_jitter:
        jitter_amount = 0.0

    # Use absolute value of diff as correct_prediction feature, order: environment, |diff|, sensitive_attribute
    features = np.column_stack([df['environment'].values, df['diff'].values, df['sensitive_attribute'].values, df['correct_prediction'].values, df['predicted_class'].values])
    # features = np.column_stack([df['environment'].values, np.abs(df['diff'].values), df['sensitive_attribute'].values, df['correct_prediction'].values, df['predicted_class'].values])
    features_scaled = StandardScaler().fit_transform(features)

    with st.spinner(f'Computing {method.upper()} embedding...'):
        if method == 'tsne':
            perplexity = st.slider("t-SNE Perplexity:", 5, min(30, len(features)//4), min(30, len(features)//4), key=f'{method}_perplexity')
            embedding_result = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(features_scaled)
        else:
            embedding_result = trimap.TRIMAP().fit_transform(features_scaled)

    fig = go.Figure()
    colors = {'environment': environment_colors, 'sensitive_attribute': {0: '#01BEFE', 1: '#FFDD00'},
              'correct_prediction': {0: '#ADFF02', 1: '#8F00FF'}, 'predicted_class': {0: '#00CED1', 1: '#FF69B4'}}

    unique_values = sorted(df[color_by].unique())
    for value in unique_values:
        mask = df[color_by] == value
        x_coords = embedding_result[mask, 0]
        y_coords = embedding_result[mask, 1]

        # Apply jittering if specified
        if jitter_amount > 0:
            # Use a fixed seed for reproducible jittering
            np.random.seed(42)
            x_coords = x_coords + np.random.normal(0, jitter_amount, len(x_coords))
            y_coords = y_coords + np.random.normal(0, jitter_amount, len(y_coords))

        color = colors[color_by].get(value, '#808080')  # Default gray if not in scheme

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='markers',
            marker=dict(size=point_size, color=color, opacity=opacity),
            name=f'{color_by}={value} (n={mask.sum()})',
            hovertemplate=f'<b>%{{text}}</b><br>{method.upper()}1: %{{x:.2f}}<br>{method.upper()}2: %{{y:.2f}}<br>diff: %{{customdata[0]:.4f}}<br>environment: %{{customdata[1]}}<br>sensitive_attribute: %{{customdata[2]}}<br>correct_prediction: %{{customdata[3]}}<br>predicted_class: %{{customdata[4]}}<extra></extra>',
            text=df[mask]['name'],
            customdata=list(zip(df[mask]['diff'], df[mask]['environment'], df[mask]['sensitive_attribute'], df[mask]['correct_prediction'], df[mask]['predicted_class']))
        ))

    fig.update_layout(
        title=f'{method.upper()} Visualization of Data Points<br><sup>Features: environment, |diff|, sensitive_attribute, correct_prediction, predicted_class</sup>',
        xaxis_title=f'{method.upper()} Component 1', yaxis_title=f'{method.upper()} Component 2',
        showlegend=True, height=600
    )
    return fig

def create_3d_embedding_visualization(records, method='tsne'):
    """Create 3D to 2D embedding visualization."""
    df = pd.DataFrame(records)
    st.subheader(f"3D {method.upper()} Visualization")

    # Point styling controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        point_size = st.slider("Point Size", min_value=1, max_value=20, value=8, step=1, key=f'{method}_3d_size')
    with col2:
        opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1, key=f'{method}_3d_opacity')
    with col3:
        enable_jitter = st.checkbox("Enable Jittering", value=False, help="Adds random noise to prevent overlapping points", key=f'{method}_3d_enable_jitter')
    with col4:
        jitter_amount = st.slider("Jitter Amount", min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                 disabled=not enable_jitter, help="Amount of random noise to add", key=f'{method}_3d_jitter')

    # Only apply jittering if enabled
    if not enable_jitter:
        jitter_amount = 0.0

    # Use absolute value of diff, order: environment, |diff|, sensitive_attribute
    features_3d = np.column_stack([df['environment'].values, df['diff'].values, df['sensitive_attribute'].values])
    # features_3d = np.column_stack([df['environment'].values, np.abs(df['diff'].values), df['sensitive_attribute'].values])
    features_3d_scaled = StandardScaler().fit_transform(features_3d)
    colors = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF', '#00CED1', '#FF69B4']

    # Group by environment, sensitive_attribute, and correct_prediction for coloring, but use |diff| in features
    unique_combinations = df.groupby(['environment', 'sensitive_attribute', 'correct_prediction']).size().reset_index(name='count')

    if method == 'tsne':
        perplexity = st.slider("t-SNE Perplexity:", 5, min(30, len(features_3d)//4), min(30, len(features_3d)//4), key=f'{method}_3d_perplexity')
        embedding_result = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(features_3d_scaled)
    else:
        embedding_result = trimap.TRIMAP().fit_transform(features_3d_scaled)

    fig = go.Figure()
    for idx, row in unique_combinations.iterrows():
        e, s, p, count = row['environment'], row['sensitive_attribute'], row['correct_prediction'], row['count']
        mask = (df['environment'] == e) & (df['sensitive_attribute'] == s) & (df['correct_prediction'] == p)
        if mask.sum() == 0: continue

        x_coords = embedding_result[mask, 0]
        y_coords = embedding_result[mask, 1]

        # Apply jittering if specified
        if jitter_amount > 0:
            # Use a fixed seed for reproducible jittering
            np.random.seed(42)
            x_coords = x_coords + np.random.normal(0, jitter_amount, len(x_coords))
            y_coords = y_coords + np.random.normal(0, jitter_amount, len(y_coords))

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='markers',
            marker=dict(size=point_size, color=colors[idx % len(colors)], opacity=opacity),
            name=f'E={e}, S={s}, P={p} (n={count})',
            hovertemplate=f'<b>%{{text}}</b><br>{method.upper()}1: %{{x:.2f}}<br>{method.upper()}2: %{{y:.2f}}<br>environment: %{{customdata[0]}}<br>sensitive_attribute: %{{customdata[1]}}<br>correct_prediction: %{{customdata[2]}}<br>diff: %{{customdata[3]:.4f}}<br>predicted_class: %{{customdata[4]}}<extra></extra>',
            text=df[mask]['name'],
            customdata=list(zip(df[mask]['environment'], df[mask]['sensitive_attribute'], df[mask]['correct_prediction'], df[mask]['diff'], df[mask]['predicted_class']))
        ))

    fig.update_layout(
        title=f'{method.upper()}: 3D → 2D<br><sup>3D space: (environment, |diff|, sensitive_attribute)</sup>',
        xaxis_title=f'{method.upper()} Component 1', yaxis_title=f'{method.upper()} Component 2',
        showlegend=True, height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig

def load_metrics_data(data):
    """Load metrics data from either file path or uploaded file content."""
    metrics_data, summary_metrics = {}, {}

    try:
        if isinstance(data, str):
            with open(data, 'r') as f:
                lines = f.readlines()
        elif isinstance(data, bytes):
            lines = data.decode('utf-8').split('\n')

        current_metric = None
        for line in lines:
            line = line.strip()
            if line.startswith('dual_var1'):
                for part in line.split():
                    if part.startswith('env') and ('_in_' in part or '_out_' in part):
                        current_metric = part.split('_')[-1]
                        break
            elif line and current_metric and not line.startswith('dual_var1'):
                parts = line.split()
                if len(parts) >= 13:
                    metrics_data[current_metric] = {
                        'env0_in': float(parts[2]), 'env0_out': float(parts[3]),
                        'env1_in': float(parts[4]), 'env1_out': float(parts[5]),
                        'epoch': float(parts[6]), 'step': int(parts[11])
                    }

        for line in lines[-5:]:
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2:
                    summary_metrics[parts[0].strip()] = float(parts[1].strip())

    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return {}, {}

    return metrics_data, summary_metrics

def display_metrics_dashboard(metrics_data, summary_metrics):
    """Display metrics in an organized dashboard."""
    st.subheader("Model Performance Metrics")
    tab1, tab2, tab3 = st.tabs(["Summary Metrics", "Detailed Environment Metrics", "Raw Data"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy (ACC)", f"{summary_metrics.get('ACC', 0):.4f}")
            st.metric("Demographic Parity (DP)", f"{summary_metrics.get('DP', 0):.4f}")
        with col2:
            st.metric("Mean Difference (MD)", f"{summary_metrics.get('MD', 0):.4f}")
            st.metric("Equal Opportunity (EO)", f"{summary_metrics.get('EO', 0):.4f}")
        with col3:
            st.metric("AUC", f"{summary_metrics.get('AUC', 0):.4f}")
            acc_value = summary_metrics.get('ACC', 0)
            if acc_value >= 0.95: st.success("Excellent Performance")
            elif acc_value >= 0.90: st.info("Good Performance")
            elif acc_value >= 0.80: st.warning("Fair Performance")
            else: st.error("Needs Improvement")

    with tab2:
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data).T
            metrics_df.index.name = 'Metric'
            metrics_df.columns = ['Env0 (In)', 'Env0 (Out)', 'Env1 (In)', 'Env1 (Out)', 'Epoch', 'Step']

            display_df = metrics_df.copy()
            for col in ['Env0 (In)', 'Env0 (Out)', 'Env1 (In)', 'Env1 (Out)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            display_df['Epoch'] = display_df['Epoch'].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_df, use_container_width=True)

            for metric in ['acc', 'md', 'dp', 'eo', 'auc']:
                if metric in metrics_data:
                    env_names = ['Env0 (In)', 'Env0 (Out)', 'Env1 (In)', 'Env1 (Out)']
                    values = [metrics_data[metric]['env0_in'], metrics_data[metric]['env0_out'],
                             metrics_data[metric]['env1_in'], metrics_data[metric]['env1_out']]

                    fig = go.Figure(go.Bar(x=env_names, y=values, marker_color=['#c45161', '#e094a0', '#8db7d2', '#5e62a9'],
                                          text=[f'{v:.4f}' for v in values], textposition='auto'))
                    fig.update_layout(title=f'{metric.upper()} by Environment', xaxis_title='Environment',
                                    yaxis_title=f'{metric.upper()} Value', height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if metrics_data:
            raw_data = [{'Metric': metric.upper(), 'Environment 0 (In)': f"{data['env0_in']:.6f}",
                         'Environment 0 (Out)': f"{data['env0_out']:.6f}", 'Environment 1 (In)': f"{data['env1_in']:.6f}",
                         'Environment 1 (Out)': f"{data['env1_out']:.6f}", 'Epoch': f"{data['epoch']:.1f}", 'Step': data['step']}
                        for metric, data in metrics_data.items()]
            st.dataframe(pd.DataFrame(raw_data), use_container_width=True)

            summary_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
            summary_df['Value'] = summary_df['Value'].apply(lambda x: f"{x:.4f}")
            st.dataframe(summary_df, use_container_width=True)

def extract_step_from_json_name(json_name):
    """Extract the step number from the predictions JSON name."""
    match = re.search(r'step_(\d+)', json_name)
    if match:
        return int(match.group(1))
    return None

def extract_step_metrics(metrics_file_path, step):
    """
    Extract metrics for a specific step from any of the 8 .txt files.
    Dynamically finds the correct columns for env metrics and step.
    """
    step_metrics = {}
    try:
        with open(metrics_file_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            # Look for header lines that contain 'step' and at least one env metric
            if 'step' in line and any(f'env0_in_{metric}' in line for metric in ['acc', 'md', 'dp', 'eo', 'auc']):
                header_parts = line.strip().split()
                # Find indices for env metrics and step
                try:
                    step_idx = header_parts.index('step')
                except ValueError:
                    continue
                # Find indices for env metrics
                for metric in ['acc', 'md', 'dp', 'eo', 'auc']:
                    try:
                        env0_in_idx = header_parts.index(f'env0_in_{metric}')
                        env0_out_idx = header_parts.index(f'env0_out_{metric}')
                        env1_in_idx = header_parts.index(f'env1_in_{metric}')
                        env1_out_idx = header_parts.index(f'env1_out_{metric}')
                    except ValueError:
                        continue  # This metric not present in this header
                    # Next line should be the data line
                    if i+1 < len(lines):
                        data_line = lines[i+1].strip()
                        if not data_line or not any(c.isdigit() for c in data_line):
                            continue
                        data_parts = data_line.split()
                        if len(data_parts) <= max(env0_in_idx, env0_out_idx, env1_in_idx, env1_out_idx, step_idx):
                            continue
                        try:
                            step_in_data = int(float(data_parts[step_idx]))
                        except Exception:
                            continue
                        if step_in_data == step:
                            metric_name = f'env0_in_{metric}'
                            step_metrics[metric_name] = {
                                'env0_in': float(data_parts[env0_in_idx]),
                                'env0_out': float(data_parts[env0_out_idx]),
                                'env1_in': float(data_parts[env1_in_idx]),
                                'env1_out': float(data_parts[env1_out_idx]),
                            }
        return step_metrics
    except Exception as e:
        st.error(f"Error extracting step {step} metrics: {e}")
        return {}

def display_step_metrics(step_metrics, step=None):
    """Display the step metrics in a formatted way (Streamlit version)."""
    if not step_metrics:
        st.warning("No metrics found for this step.")
        return
    if step is not None:
        st.subheader(f"Metrics for Step {step}")
    else:
        st.subheader("Metrics for Step")
    metrics_summary = []
    for metric_name, values in step_metrics.items():
        metric_short = metric_name.replace('env0_in_', '').upper()
        metrics_summary.append({
            'Metric': metric_short,
            'Env0 (In)': f"{values['env0_in']:.4f}",
            'Env0 (Out)': f"{values['env0_out']:.4f}",
            'Env1 (In)': f"{values['env1_in']:.4f}",
            'Env1 (Out)': f"{values['env1_out']:.4f}"
        })
    summary_df = pd.DataFrame(metrics_summary)
    st.dataframe(summary_df, use_container_width=True)

    # Calculate averages for each metric across the four envs
    averages = {}
    for metric_name, values in step_metrics.items():
        metric_short = metric_name.replace('env0_in_', '').upper()
        avg = np.mean([values['env0_in'], values['env0_out'], values['env1_in'], values['env1_out']])
        averages[metric_short] = avg

    # Or display as metrics
    st.subheader("Average Metrics Across Environments")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Avg ACC", f"{averages.get('ACC', 0):.4f}")
    with col2:
        st.metric("Avg MD", f"{averages.get('MD', 0):.4f}")
    with col3:
        st.metric("Avg DP", f"{averages.get('DP', 0):.4f}")
    with col4:
        st.metric("Avg EO", f"{averages.get('EO', 0):.4f}")
    with col5:
        st.metric("Avg AUC", f"{averages.get('AUC', 0):.4f}")

def get_metrics_file_from_json(json_path_or_name):
    """Given a predictions JSON file path or name, return the corresponding metrics txt file path."""
    if isinstance(json_path_or_name, str):
        base = os.path.basename(json_path_or_name)
        match = re.match(r'([A-Za-z0-9]+)_step_\d+_predictions\.json', base)
        if match:
            algo = match.group(1)
            return os.path.join(DATA_DIR, f'{algo}.txt')
        # fallback: try to match x.json -> x.txt
        if base.endswith('.json'):
            algo = base.split('_')[0]
            return os.path.join(DATA_DIR, f'{algo}.txt')
    return None

def calculate_pdf_for_sensitive_attributes(records, bandwidth_method='silverman', n_points=1000):
    """
    Calculate PDF (Probability Density Function) for diff distribution across sensitive attributes.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        bandwidth_method: Method for bandwidth selection ('silverman', 'scott', or float value)
        n_points: Number of points for PDF calculation

    Returns:
        Dictionary containing PDF data for each sensitive attribute
    """
    df = pd.DataFrame(records)

    # Get unique sensitive attributes
    unique_attrs = sorted(df['sensitive_attribute'].unique())

    pdf_data = {}

    for attr_value in unique_attrs:
        subset = df[df['sensitive_attribute'] == attr_value]['diff'].values

        if len(subset) < 2:
            continue

        # Calculate bandwidth
        if bandwidth_method == 'silverman':
            # Silverman's rule of thumb: bandwidth = 0.9 * min(std, IQR/1.34) * n^(-1/5)
            n = len(subset)
            std = np.std(subset)
            iqr = np.percentile(subset, 75) - np.percentile(subset, 25)
            bandwidth = 0.9 * min(std, iqr / 1.34) * (n ** (-1/5))
        elif bandwidth_method == 'scott':
            # Scott's rule: bandwidth = 1.059 * std * n^(-1/5)
            n = len(subset)
            std = np.std(subset)
            bandwidth = 1.059 * std * (n ** (-1/5))
        else:
            bandwidth = float(bandwidth_method)

        # Create KDE
        kde = gaussian_kde(subset, bw_method=bandwidth)

        # Generate x values for PDF calculation
        x_min, x_max = subset.min(), subset.max()
        x_range = x_max - x_min
        x_padding = x_range * 0.1  # Add 10% padding

        x_values = np.linspace(x_min - x_padding, x_max + x_padding, n_points)

        # Calculate PDF values
        pdf_values = kde(x_values)

        # Store results
        pdf_data[attr_value] = {
            'x_values': x_values,
            'pdf_values': pdf_values,
            'bandwidth': bandwidth,
            'sample_size': len(subset),
            'mean': np.mean(subset),
            'std': np.std(subset),
            'min': np.min(subset),
            'max': np.max(subset)
        }

    return pdf_data

def create_pdf_line_plot(records, bandwidth_method='silverman', n_points=1000,
                        show_statistics=True, show_bandwidth_info=True):
    """
    Create a line plot showing PDF of diff distribution for different sensitive attributes.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        bandwidth_method: Method for bandwidth selection
        n_points: Number of points for PDF calculation
        show_statistics: Whether to display statistics
        show_bandwidth_info: Whether to display bandwidth information

    Returns:
        Plotly figure object
    """
    pdf_data = calculate_pdf_for_sensitive_attributes(records, bandwidth_method, n_points)

    if not pdf_data:
        st.warning("No data available for PDF calculation.")
        return None

    # Color scheme for sensitive attributes
    colors = {0: '#01BEFE', 1: '#FFDD00'}

    fig = go.Figure()

    # Add PDF lines for each sensitive attribute
    for attr_value, data in pdf_data.items():
        color = colors.get(attr_value, '#808080')

        # Create legend name with sample size
        legend_name = f'Sensitive Attribute {attr_value} (n={data["sample_size"]})'

        fig.add_trace(go.Scatter(
            x=data['x_values'],
            y=data['pdf_values'],
            mode='lines',
            name=legend_name,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>Sensitive Attribute {attr_value}</b><br>' +
                         f'x: %{{x:.4f}}<br>' +
                         f'PDF: %{{y:.4f}}<br>' +
                         f'Sample size: {data["sample_size"]}<br>' +
                         f'Mean: {data["mean"]:.4f}<br>' +
                         f'Std: {data["std"]:.4f}<extra></extra>'
        ))

    # Add vertical lines for means
    for attr_value, data in pdf_data.items():
        color = colors.get(attr_value, '#808080')
        fig.add_trace(go.Scatter(
            x=[data['mean'], data['mean']],
            y=[0, np.max(data['pdf_values'])],
            mode='lines',
            name=f'Mean (Attr {attr_value})',
            line=dict(color=color, width=1, dash='dash'),
            showlegend=False,
            hovertemplate=f'<b>Mean for Attribute {attr_value}</b><br>' +
                         f'Value: {data["mean"]:.4f}<extra></extra>'
        ))

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout
    fig.update_layout(
        title='PDF of Diff Distribution by Sensitive Attribute<br><sup>Probability Density Function using Gaussian KDE</sup>',
        xaxis_title='Diff Value',
        yaxis_title='Probability Density',
        showlegend=True,
        height=500,
        width=1200,
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode='closest'
    )

    # Display statistics if requested
    if show_statistics:
        st.subheader("PDF Statistics by Sensitive Attribute")
        stats_data = []
        for attr_value, data in pdf_data.items():
            stats_data.append({
                'Sensitive Attribute': attr_value,
                'Sample Size': data['sample_size'],
                'Mean': f"{data['mean']:.4f}",
                'Std Dev': f"{data['std']:.4f}",
                'Min': f"{data['min']:.4f}",
                'Max': f"{data['max']:.4f}",
                'Bandwidth': f"{data['bandwidth']:.4f}"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

    # Display bandwidth information if requested
    if show_bandwidth_info:
        st.info(f"Bandwidth method: {bandwidth_method}")
        if bandwidth_method == 'silverman':
            st.write("Silverman's rule of thumb: bandwidth = 0.9 * min(std, IQR/1.34) * n^(-1/5)")
        elif bandwidth_method == 'scott':
            st.write("Scott's rule: bandwidth = 1.059 * std * n^(-1/5)")
        else:
            st.write(f"Fixed bandwidth: {bandwidth_method}")

    return fig

def create_pdf_comparison_plot(records, bandwidth_method='silverman', n_points=1000):
    """
    Create a comprehensive PDF comparison plot with multiple views.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        bandwidth_method: Method for bandwidth selection
        n_points: Number of points for PDF calculation

    Returns:
        Plotly figure object with subplots
    """
    pdf_data = calculate_pdf_for_sensitive_attributes(records, bandwidth_method, n_points)

    if not pdf_data:
        st.warning("No data available for PDF calculation.")
        return None

    # Create subplots: PDF comparison and cumulative distribution
    fig = go.Figure()

    # Color scheme
    colors = {0: '#01BEFE', 1: '#FFDD00'}

    # Add PDF lines
    for attr_value, data in pdf_data.items():
        color = colors.get(attr_value, '#808080')
        legend_name = f'PDF Attr {attr_value} (n={data["sample_size"]})'

        fig.add_trace(go.Scatter(
            x=data['x_values'],
            y=data['pdf_values'],
            mode='lines',
            name=legend_name,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>PDF Attr {attr_value}</b><br>' +
                         f'x: %{{x:.4f}}<br>' +
                         f'PDF: %{{y:.4f}}<extra></extra>'
        ))

    # Add cumulative distribution lines
    for attr_value, data in pdf_data.items():
        color = colors.get(attr_value, '#808080')
        legend_name = f'CDF Attr {attr_value} (n={data["sample_size"]})'

        # Calculate cumulative distribution
        cdf_values = np.cumsum(data['pdf_values']) * (data['x_values'][1] - data['x_values'][0])
        cdf_values = cdf_values / cdf_values[-1]  # Normalize to [0,1]

        fig.add_trace(go.Scatter(
            x=data['x_values'],
            y=cdf_values,
            mode='lines',
            name=legend_name,
            line=dict(color=color, width=1, dash='dot'),
            yaxis='y2',
            hovertemplate=f'<b>CDF Attr {attr_value}</b><br>' +
                         f'x: %{{x:.4f}}<br>' +
                         f'CDF: %{{y:.4f}}<extra></extra>'
        ))

    # Add y=0 line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout with dual y-axes
    fig.update_layout(
        title='PDF and CDF Comparison by Sensitive Attribute<br><sup>Solid lines: PDF, Dotted lines: CDF</sup>',
        xaxis_title='Diff Value',
        yaxis=dict(
            title='Probability Density (PDF)',
            side='left'
        ),
        yaxis2=dict(
            title='Cumulative Probability (CDF)',
            side='right',
            overlaying='y',
            range=[0, 1]
        ),
        showlegend=True,
        height=600,
        width=1200,
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='closest'
    )

    return fig

def calculate_pdf_fairness_metrics(records, bandwidth_method='silverman', n_points=1000):
    """
    Calculate fairness metrics based on PDF analysis of diff distributions.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
        bandwidth_method: Method for bandwidth selection
        n_points: Number of points for PDF calculation

    Returns:
        Dictionary containing fairness metrics
    """
    pdf_data = calculate_pdf_for_sensitive_attributes(records, bandwidth_method, n_points)

    if len(pdf_data) < 2:
        return None

    # Get the two sensitive attribute values
    attr_values = sorted(pdf_data.keys())
    if len(attr_values) != 2:
        return None

    attr0, attr1 = attr_values[0], attr_values[1]
    data0, data1 = pdf_data[attr0], pdf_data[attr1]

    # Calculate various fairness metrics

    # 1. Mean difference
    mean_diff = abs(data0['mean'] - data1['mean'])

    # 2. Standard deviation difference
    std_diff = abs(data0['std'] - data1['std'])

    # 3. Distribution overlap (using Bhattacharyya coefficient)
    # Find common x range
    x_min = max(data0['x_values'].min(), data1['x_values'].min())
    x_max = min(data0['x_values'].max(), data1['x_values'].max())

    if x_max > x_min:
        # Interpolate both PDFs to common x values
        common_x = np.linspace(x_min, x_max, n_points)
        pdf0_interp = np.interp(common_x, data0['x_values'], data0['pdf_values'])
        pdf1_interp = np.interp(common_x, data1['x_values'], data1['pdf_values'])

        # Normalize PDFs
        pdf0_norm = pdf0_interp / np.sum(pdf0_interp)
        pdf1_norm = pdf1_interp / np.sum(pdf1_interp)

        # Calculate Bhattacharyya coefficient
        bhattacharyya = np.sum(np.sqrt(pdf0_norm * pdf1_norm))
        overlap_score = bhattacharyya
    else:
        overlap_score = 0.0

    # 4. Wasserstein distance (Earth Mover's Distance)
    try:
        from scipy.stats import wasserstein_distance
        wasserstein_dist = wasserstein_distance(
            data0['x_values'], data1['x_values'],
            data0['pdf_values'], data1['pdf_values']
        )
    except:
        wasserstein_dist = float('inf')

    # 5. Kolmogorov-Smirnov statistic
    try:
        from scipy.stats import ks_2samp
        # Sample from the distributions
        sample0 = np.random.choice(data0['x_values'], size=1000, p=data0['pdf_values']/np.sum(data0['pdf_values']))
        sample1 = np.random.choice(data1['x_values'], size=1000, p=data1['pdf_values']/np.sum(data1['pdf_values']))
        ks_stat, ks_pvalue = ks_2samp(sample0, sample1)
    except:
        ks_stat, ks_pvalue = 0.0, 1.0

    # 6. Fairness scores (0-1, where 1 is most fair)
    mean_fairness = max(0, 1 - mean_diff / 0.1)  # Normalize by 0.1 threshold
    std_fairness = max(0, 1 - std_diff / 0.1)    # Normalize by 0.1 threshold
    overlap_fairness = overlap_score
    wasserstein_fairness = max(0, 1 - wasserstein_dist / 0.1) if wasserstein_dist != float('inf') else 0
    ks_fairness = 1 - ks_stat

    # Overall fairness score
    overall_fairness = np.mean([mean_fairness, std_fairness, overlap_fairness, wasserstein_fairness, ks_fairness])

    return {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'distribution_overlap': overlap_score,
        'wasserstein_distance': wasserstein_dist,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'mean_fairness': mean_fairness,
        'std_fairness': std_fairness,
        'overlap_fairness': overlap_fairness,
        'wasserstein_fairness': wasserstein_fairness,
        'ks_fairness': ks_fairness,
        'overall_fairness': overall_fairness,
        'sample_sizes': {
            f'attr_{attr0}': data0['sample_size'],
            f'attr_{attr1}': data1['sample_size']
        },
        'statistics': {
            f'attr_{attr0}': {
                'mean': data0['mean'],
                'std': data0['std'],
                'min': data0['min'],
                'max': data0['max']
            },
            f'attr_{attr1}': {
                'mean': data1['mean'],
                'std': data1['std'],
                'min': data1['min'],
                'max': data1['max']
            }
        }
    }

def display_pdf_fairness_metrics(records):
    """Display fairness metrics based on PDF analysis."""
    metrics = calculate_pdf_fairness_metrics(records)

    if metrics is None:
        st.warning("Cannot calculate fairness metrics: need at least 2 sensitive attribute groups.")
        return

    st.subheader("PDF-based Fairness Metrics")

    # Display overall fairness score
    overall_fairness = metrics['overall_fairness']
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Fairness Score", f"{overall_fairness:.4f}")
        if overall_fairness >= 0.9:
            st.success("Excellent Fairness")
        elif overall_fairness >= 0.8:
            st.info("Good Fairness")
        elif overall_fairness >= 0.7:
            st.warning("Fair Fairness")
        else:
            st.error("Poor Fairness")

    with col2:
        st.metric("Mean Difference", f"{metrics['mean_difference']:.4f}")
        st.metric("Mean Fairness", f"{metrics['mean_fairness']:.4f}")

    with col3:
        st.metric("Distribution Overlap", f"{metrics['distribution_overlap']:.4f}")
        st.metric("Overlap Fairness", f"{metrics['overlap_fairness']:.4f}")

    with col4:
        st.metric("Wasserstein Distance", f"{metrics['wasserstein_distance']:.4f}")
        st.metric("Wasserstein Fairness", f"{metrics['wasserstein_fairness']:.4f}")

    # Detailed metrics table
    st.subheader("Detailed Fairness Metrics")
    detailed_metrics = [
        {
            'Metric': 'Mean Difference',
            'Value': f"{metrics['mean_difference']:.4f}",
            'Fairness Score': f"{metrics['mean_fairness']:.4f}",
            'Interpretation': 'Difference in average prediction errors'
        },
        {
            'Metric': 'Std Difference',
            'Value': f"{metrics['std_difference']:.4f}",
            'Fairness Score': f"{metrics['std_fairness']:.4f}",
            'Interpretation': 'Difference in prediction error variability'
        },
        {
            'Metric': 'Distribution Overlap',
            'Value': f"{metrics['distribution_overlap']:.4f}",
            'Fairness Score': f"{metrics['overlap_fairness']:.4f}",
            'Interpretation': 'Similarity of error distributions (Bhattacharyya)'
        },
        {
            'Metric': 'Wasserstein Distance',
            'Value': f"{metrics['wasserstein_distance']:.4f}",
            'Fairness Score': f"{metrics['wasserstein_fairness']:.4f}",
            'Interpretation': 'Earth mover distance between distributions'
        },
        {
            'Metric': 'KS Statistic',
            'Value': f"{metrics['ks_statistic']:.4f}",
            'Fairness Score': f"{metrics['ks_fairness']:.4f}",
            'Interpretation': 'Kolmogorov-Smirnov test statistic'
        }
    ]

    metrics_df = pd.DataFrame(detailed_metrics)
    st.dataframe(metrics_df, use_container_width=True)

    # Sample size information
    st.subheader("Sample Size Information")
    sample_sizes = metrics['sample_sizes']
    # Get the actual attribute values from the keys
    attr_keys = list(sample_sizes.keys())
    for i, key in enumerate(attr_keys):
        attr_value = key.replace('attr_', '')
        st.write(f"Group {attr_value}: {sample_sizes[key]} samples")

    # Statistical summary
    st.subheader("Statistical Summary by Group")
    stats = metrics['statistics']
    stats_data = []
    for attr, data in stats.items():
        stats_data.append({
            'Group': attr.replace('attr_', ''),
            'Mean': f"{data['mean']:.4f}",
            'Std Dev': f"{data['std']:.4f}",
            'Min': f"{data['min']:.4f}",
            'Max': f"{data['max']:.4f}"
        })

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

def create_pdf_with_sidebar(records):
    """Create PDF visualization with controls in the sidebar."""
    with st.sidebar:
        st.header("PDF Analysis Controls")

        # Bandwidth selection
        st.subheader("Bandwidth Selection")
        bandwidth_method = st.selectbox(
            "Bandwidth Method:",
            ['silverman', 'scott', '0.1', '0.05', '0.2'],
            help="Method for KDE bandwidth selection. Silverman and Scott are automatic methods."
        )

        # Number of points
        n_points = st.slider(
            "Number of PDF Points:",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Number of points used to calculate the PDF curve"
        )

        # Display options
        st.subheader("Display Options")
        show_statistics = st.checkbox("Show Statistics Table", value=True)
        show_bandwidth_info = st.checkbox("Show Bandwidth Information", value=True)
        show_cdf_comparison = st.checkbox("Show CDF Comparison", value=True)

    # Create the appropriate plot
    if show_cdf_comparison:
        fig = create_pdf_comparison_plot(records, bandwidth_method, n_points)
    else:
        fig = create_pdf_line_plot(records, bandwidth_method, n_points,
                                 show_statistics, show_bandwidth_info)

    return fig

def display_knee_positions_analysis(records):
    """
    Display detailed analysis of knee point positions.

    Args:
        records: List of records with 'diff' and 'sensitive_attribute' fields
    """
    if 'knee_positions' not in st.session_state or not st.session_state.knee_positions:
        st.warning("No knee positions data available. Please generate knee point visualization first.")
        return

    knee_positions = st.session_state.knee_positions
    df = pd.DataFrame(records)

    st.subheader("Knee Point Positions Analysis")

    # Create a comprehensive table
    analysis_data = []
    for pos in knee_positions:
        group_name = pos['group'].replace('_', ' ').title()
        if 'Sensitive Attribute' in group_name:
            group_name = group_name.replace('Sensitive Attribute', 'Attr')

        analysis_data.append({
            'Group': group_name,
            'Type': pos['type'].replace('_', ' ').title(),
            'Index': pos['x'],
            'Diff Value': f"{pos['y']:.4f}",
            'Percentile': f"{pos['percentile']:.1f}%",
            'Sample Size': len(df) if pos['group'] == 'overall' else len(df[df['sensitive_attribute'] == int(pos['group'].split('_')[-1])])
        })

    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        st.dataframe(analysis_df, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Knee Points", len(knee_positions))
            st.metric("Groups Analyzed", len(set(pos['group'] for pos in knee_positions)))

        with col2:
            avg_percentile = np.mean([pos['percentile'] for pos in knee_positions])
            st.metric("Average Percentile", f"{avg_percentile:.1f}%")
            st.metric("Min Percentile", f"{min(pos['percentile'] for pos in knee_positions):.1f}%")

        with col3:
            max_percentile = max(pos['percentile'] for pos in knee_positions)
            st.metric("Max Percentile", f"{max_percentile:.1f}%")
            st.metric("Percentile Range", f"{max_percentile - min(pos['percentile'] for pos in knee_positions):.1f}%")

        # Distribution analysis
        st.subheader("Knee Point Distribution")

        # Create a histogram of percentiles
        percentiles = [pos['percentile'] for pos in knee_positions]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=percentiles,
            nbinsx=10,
            name='Knee Point Percentiles',
            marker_color='lightblue',
            opacity=0.7
        ))
        fig_hist.update_layout(
            title='Distribution of Knee Point Percentiles',
            xaxis_title='Percentile',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Group comparison
        st.subheader("Group Comparison")
        group_data = {}
        for pos in knee_positions:
            group = pos['group']
            if group not in group_data:
                group_data[group] = []
            group_data[group].append(pos['percentile'])

        group_comparison = []
        for group, percentiles in group_data.items():
            group_name = group.replace('_', ' ').title()
            if 'Sensitive Attribute' in group_name:
                group_name = group_name.replace('Sensitive Attribute', 'Attr')

            group_comparison.append({
                'Group': group_name,
                'Count': len(percentiles),
                'Avg Percentile': f"{np.mean(percentiles):.1f}%",
                'Min Percentile': f"{min(percentiles):.1f}%",
                'Max Percentile': f"{max(percentiles):.1f}%",
                'Std Percentile': f"{np.std(percentiles):.1f}%"
            })

        comparison_df = pd.DataFrame(group_comparison)
        st.dataframe(comparison_df, use_container_width=True)

    else:
        st.warning("No knee positions data available for analysis.")

def display_segment_analysis(segment_data):
    """
    Display detailed analysis of segmented curve data.

    Args:
        segment_data: List of segment data dictionaries
    """
    if not segment_data:
        st.warning("No segment data available for analysis.")
        return

    st.subheader("Segment Analysis")

    # Create a comprehensive table
    analysis_data = []
    for seg in segment_data:
        # Get the coloring attribute and group counts
        coloring_attr = seg.get('coloring_attribute', 'sensitive_attribute')
        group_counts = seg.get('group_counts', {})
        group_percentages = seg.get('group_percentages', {})

        # Create group count and percentage strings
        group_info = []
        for group, count in group_counts.items():
            percentage = group_percentages.get(group, 0)
            group_info.append(f"{group}: {count} ({percentage:.1f}%)")

        group_info_str = ", ".join(group_info)

        analysis_data.append({
            'Segment ID': seg['segment_id'],
            'X Range': f"{seg['x_start']}-{seg['x_end']}",
            'Y Range': f"{seg['y_start']:.4f}-{seg['y_end']:.4f}",
            'Majority Group': seg['majority_group'],
            'Majority Count': seg['majority_count'],
            'Total Points': seg['total_points'],
            'Coloring Attribute': coloring_attr.replace('_', ' ').title(),
            'Group Distribution': group_info_str
        })

    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        st.dataframe(analysis_df, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Segments", len(segment_data))
            # Count segments by majority group
            majority_groups = [seg['majority_group'] for seg in segment_data]
            unique_groups = set(majority_groups)
            for group in unique_groups:
                count = majority_groups.count(group)
                st.metric(f"Group {group} Dominant Segments", count)

        with col2:
            total_points = sum(seg['total_points'] for seg in segment_data)
            st.metric("Total Points", total_points)
            avg_points_per_segment = total_points / len(segment_data) if segment_data else 0
            st.metric("Avg Points per Segment", f"{avg_points_per_segment:.1f}")

        with col3:
            # Calculate average percentages for each group
            if segment_data:
                coloring_attr = segment_data[0].get('coloring_attribute', 'sensitive_attribute')
                all_percentages = {}
                for seg in segment_data:
                    for group, percentage in seg.get('group_percentages', {}).items():
                        if group not in all_percentages:
                            all_percentages[group] = []
                        all_percentages[group].append(percentage)

                for group, percentages in all_percentages.items():
                    avg_percentage = np.mean(percentages)
                    st.metric(f"Avg {coloring_attr.replace('_', ' ').title()} {group} %", f"{avg_percentage:.1f}%")

        with col4:
            # Calculate max percentages
            if segment_data:
                for group, percentages in all_percentages.items():
                    max_percentage = max(percentages)
                    st.metric(f"Max {coloring_attr.replace('_', ' ').title()} {group} %", f"{max_percentage:.1f}%")

        # Distribution analysis
        st.subheader("Segment Distribution Analysis")

        # Create a histogram of majority groups
        majority_groups = [seg['majority_group'] for seg in segment_data]
        fig_hist = go.Figure()

        # Get unique groups and their colors
        unique_groups = sorted(set(majority_groups))
        coloring_attr = segment_data[0].get('coloring_attribute', 'sensitive_attribute')

        # Define colors based on the coloring attribute
        if coloring_attr == 'sensitive_attribute':
            colors = ['#01BEFE', '#FFDD00']
        elif coloring_attr == 'environment':
            colors = ['#FF7D00', '#FF006D', '#00CED1', '#FF69B4']
        elif coloring_attr == 'correct_prediction':
            colors = ['#ADFF02', '#8F00FF']
        else:
            colors = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D']

        fig_hist.add_trace(go.Histogram(
            x=majority_groups,
            nbinsx=len(unique_groups),
            name='Majority Groups',
            marker_color=colors[:len(unique_groups)],
            opacity=0.7
        ))
        fig_hist.update_layout(
            title=f'Distribution of Majority Groups Across Segments ({coloring_attr.replace("_", " ").title()})',
            xaxis_title='Majority Group',
            yaxis_title='Number of Segments',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Group percentage analysis
        st.subheader("Group Percentage Analysis")

        # Get all unique groups from all segments
        all_groups = set()
        for seg in segment_data:
            all_groups.update(seg.get('group_percentages', {}).keys())
        all_groups = sorted(all_groups)

        # Create percentage traces for each group
        fig_percentages = go.Figure()
        for group in all_groups:
            percentages = []
            for seg in segment_data:
                percentages.append(seg.get('group_percentages', {}).get(group, 0))

            color = colors[all_groups.index(group) % len(colors)]
            fig_percentages.add_trace(go.Scatter(
                x=list(range(1, len(segment_data) + 1)),
                y=percentages,
                mode='lines+markers',
                name=f'{coloring_attr.replace("_", " ").title()} {group} %',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))

        fig_percentages.update_layout(
            title=f'Group Percentages Across Segments ({coloring_attr.replace("_", " ").title()})',
            xaxis_title='Segment ID',
            yaxis_title='Percentage',
            height=400
        )
        st.plotly_chart(fig_percentages, use_container_width=True)

    else:
        st.warning("No segment data available for analysis.")

def main():
    st.title('FAI FDDG VIS - S.D.E. (Signed Deviation Error: p_hat_x - correct_prediction)')

    # File upload section with collapsible interface
    with st.expander("File Upload", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_json = st.file_uploader("Choose a JSON file with predictions data", type=['json'])
        with col2:
            uploaded_txt = st.file_uploader("Choose a TXT file with metrics data", type=['txt'])

    # Load data
    json_data = uploaded_json.read() if uploaded_json else (PREDICTIONS_JSON if os.path.exists(PREDICTIONS_JSON) else None)
    # Dynamically set metrics file based on predictions JSON
    metrics_file_path = None
    if uploaded_json:
        metrics_file_path = get_metrics_file_from_json(uploaded_json.name)
    elif os.path.exists(PREDICTIONS_JSON):
        metrics_file_path = get_metrics_file_from_json(PREDICTIONS_JSON)
    # If user uploads a txt file, override
    txt_data = uploaded_txt.read() if uploaded_txt else (metrics_file_path if metrics_file_path and os.path.exists(metrics_file_path) else None)

    if json_data is None:
        st.error("No JSON file available. Please upload a predictions JSON file.")
        return

    if uploaded_json: st.success(f"Loaded JSON file: {uploaded_json.name}")
    elif os.path.exists(PREDICTIONS_JSON): st.info(f"Using default JSON file: {os.path.basename(PREDICTIONS_JSON)}")

    if uploaded_txt: st.success(f"Loaded TXT file: {uploaded_txt.name}")
    elif metrics_file_path and os.path.exists(metrics_file_path): st.info(f"Using metrics file: {os.path.basename(metrics_file_path)}")
    else: st.warning("No TXT file available. Metrics dashboard will be disabled.")

    try:
        records = load_processed_predictions(json_data)
        st.success(f"Loaded {len(records)} records from predictions data")
    except Exception as e:
        st.error(f"Error loading predictions data: {e}")
        return

    # Detect step from name
    step = None
    if uploaded_json:
        step = extract_step_from_json_name(uploaded_json.name)
    elif os.path.exists(PREDICTIONS_JSON):
        step = extract_step_from_json_name(os.path.basename(PREDICTIONS_JSON))

    # Display step metrics if available
    if step is not None and metrics_file_path and os.path.exists(metrics_file_path):
        step_metrics = extract_step_metrics(metrics_file_path, step)
        if step_metrics:
            display_step_metrics(step_metrics, step)
        else:
            st.warning(f"No metrics found for step {step} in the metrics file.")

    # Visualizations
    st.subheader("Difference Distribution (Sorted)")
    fig, show_dedicated_viz, show_knee_points, segment_data, show_segment_analysis = create_scatter_plot_with_sidebar(records)
    st.plotly_chart(fig, use_container_width=True)

    # Basic statistics
    diffs = [r['diff'] for r in records]
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Mean Difference", f"{np.mean(diffs):.4f}")
    with col2: st.metric("Std Deviation", f"{np.std(diffs):.4f}")
    with col3: st.metric("Min Difference", f"{np.min(diffs):.4f}")
    with col4: st.metric("Max Difference", f"{np.max(diffs):.4f}")

    positive_count = sum(1 for d in diffs if d > 0)
    negative_count = sum(1 for d in diffs if d < 0)
    zero_count = sum(1 for d in diffs if d == 0)
    st.write(f"Positive differences: {positive_count}, Negative differences: {negative_count}, Zero differences: {zero_count}")

    # # PDF Analysis
    # st.subheader("PDF Analysis of Diff Distribution")
    # pdf_fig = create_pdf_with_sidebar(records)
    # if pdf_fig:
    #     st.plotly_chart(pdf_fig, use_container_width=True)

    # PDF Fairness Metrics
    # display_pdf_fairness_metrics(records)

    # Knee Point Analysis
    display_knee_elbow_analysis(records, show_knee=show_knee_points)


    # # Dedicated Knee Point Visualization
    # if show_dedicated_viz:
    #     st.subheader("Knee Point Detection Visualization")
    #     knee_elbow_fig = create_knee_elbow_visualization(records, show_knee=show_knee_points)
    #     st.plotly_chart(knee_elbow_fig, use_container_width=True)

    # Segment Analysis (only show if segmented curve was selected and analysis is enabled)
    if segment_data and show_segment_analysis:
        display_segment_analysis(segment_data)

    # 3D Dimensionality Reduction
    # create_3d_embedding_visualization(records, 'tsne')
    # create_3d_embedding_visualization(records, 'trimap')

    # Knee Point Positions Analysis
    # display_knee_positions_analysis(records)

if __name__ == '__main__':
    main()