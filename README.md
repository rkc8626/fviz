# FViz: Binary Visualization and Analysis Framework

A comprehensive visualization and analysis framework for binary classification model predictions with a focus on fairness analysis, knee point detection, and interactive exploration.

## Overview

FViz is a sophisticated analysis tool designed for deep learning researchers and practitioners working with binary classification models. It provides interactive visualizations, batch processing capabilities, and comprehensive fairness analysis for model predictions across different environments and sensitive attributes.

## Features

### Core Analysis Capabilities
- **Interactive Visualizations**: Real-time exploration of model predictions using Streamlit
- **Batch Processing**: Automated analysis of multiple algorithms and training steps
- **Knee Point Detection**: Advanced algorithms for identifying critical decision boundaries
- **Fairness Metrics**: Comprehensive analysis of model fairness across sensitive attributes
- **Embedding Visualization**: t-SNE and TriMap dimensionality reduction for high-dimensional data

### Visualization Types
- **Scatter Plots**: Interactive point clouds with filtering and coloring options
- **Segmented Curves**: Adaptive segmentation for trend analysis
- **Correlation Analysis**: Statistical correlation between fairness metrics

### Analysis Tools
- **Knee/Elbow Detection**: Multiple algorithms for finding critical decision points
- **Median Distance Analysis**: Statistical analysis of decision boundaries
- **Environment-specific Analysis**: Separate analysis for in-distribution and out-of-distribution data
- **Sensitive Attribute Analysis**: Fairness analysis across different protected attributes

## Project Structure

```
fviz_final_version/
├── interactive_viz.py              # Main interactive Streamlit application
├── batch_visualization_analysis.py # Batch processing and analysis
├── generate_summary.py             # Summary report generation
├── correlation_analysis.py         # Statistical correlation analysis
├── correlation_visualization.py    # Correlation visualization tools
├── dataset_specific_analysis.py    # Dataset-specific analysis functions
├── data/                          # Data directory
│   ├── CCMNIST1/                  # Dataset files
│   │   ├── MBDG_step_8000_predictions.json
│   │   └── MBDG.txt
│   └── processed_sensitive_median.csv
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, for GPU acceleration)

### Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn trimap scipy
```

## Usage

### Interactive Visualization

Start the interactive Streamlit application:

```bash
cd fviz_final_version
streamlit run interactive_viz.py
```

The interactive application provides:
- Real-time filtering and visualization of predictions
- Knee point detection and analysis
- PDF analysis for sensitive attributes
- Embedding visualizations (t-SNE, TriMap)
- Comprehensive metrics dashboard

### Batch Processing

Process multiple algorithms and training steps:

```bash
# Process specific algorithms
python batch_visualization_analysis.py --algorithms Mixup SagNet

# Process specific training steps
python batch_visualization_analysis.py --steps 0 8000

# Generate specific visualization types
python batch_visualization_analysis.py --visualizations scatter tsne trimap

# Specify input/output directories
python batch_visualization_analysis.py --input-dir data/CCMNIST1 --output-dir results

# Use GPU acceleration
python batch_visualization_analysis.py --use-gpu
```

### Advanced Options

```bash
# Generate only specific output formats
python batch_visualization_analysis.py --formats png pdf html

# Skip existing results
python batch_visualization_analysis.py --skip-existing

# Force reprocessing
python batch_visualization_analysis.py --force

# Complete example
python batch_visualization_analysis.py \
    --visualizations scatter segmented_curve \
    --input-dir data/CCMNIST1 \
    --output-dir comprehensive_analysis \
    --formats png pdf \
    --algorithms SagNet \
    --steps 8000 \
    --use-gpu
```

### Summary Report Generation

Generate comprehensive summary reports from batch results:

```bash
python generate_summary.py
```

This creates summary tables with:
- Basic statistics (mean, median, std deviation)
- Knee point analysis
- Fairness metrics (ACC, MD, DP, EO, AUC)
- Correlation analysis

## Data Format

### Input Data Structure
The framework expects JSON files containing model predictions with the following structure:

```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": 0.75,
      "ground_truth": 1,
      "environment": 0,
      "sensitive_attribute": 0,
      "diff": 0.25
    }
  ]
}
```

### Key Fields
- `prediction`: Model's predicted probability
- `ground_truth`: True binary label
- `environment`: Environment identifier (0/1 for in/out distribution)
- `sensitive_attribute`: Protected attribute value
- `diff`: Difference between prediction and ground truth

## Analysis Features

### Knee Point Detection
Advanced algorithms for identifying critical decision boundaries:
- **Kneedle Algorithm**: Robust knee point detection
- **Convex/Concave Analysis**: Separate analysis for different curve types
- **Adaptive Segmentation**: Dynamic segmentation based on data distribution

### Fairness Analysis
Comprehensive fairness metrics:
- **Accuracy (ACC)**: Overall prediction accuracy
- **Mean Difference (MD)**: Average prediction differences
- **Demographic Parity (DP)**: Equal prediction rates across groups
- **Equalized Odds (EO)**: Equal true/false positive rates
- **AUC**: Area under the ROC curve

### Visualization Features
- **Interactive Filtering**: Real-time data filtering
- **Color Coding**: Multiple color schemes for different attributes
- **Density Visualization**: Point density analysis
- **Error Bars**: Statistical uncertainty visualization
- **Jitter Control**: Adjustable point spacing

## Output Formats

The framework supports multiple output formats:
- **PNG**: High-resolution static images
- **PDF**: Vector graphics for publications
- **HTML**: Interactive web-based visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This framework is designed for research purposes and should be used in accordance with ethical AI practices and relevant regulations.
