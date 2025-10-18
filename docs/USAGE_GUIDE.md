# AI Neuro Wrangler - Usage Guide

This comprehensive guide will help you get started with the AI-Assisted Neuroimaging Data Harmonization framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Using the CLI](#using-the-cli)
5. [Using the Python API](#using-the-python-api)
6. [Data Processing Steps](#data-processing-steps)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/stefanches7/AI-assisted-Neuroimaging-harmonization.git
cd AI-assisted-Neuroimaging-harmonization
pip install -e .
```

### Verify installation

```bash
ai-neuro-wrangler --version
```

## Quick Start

### 1. Analyze Your Dataset

First, analyze your dataset to get recommendations:

```bash
ai-neuro-wrangler analyze /path/to/your/dataset
```

This will:
- Scan your dataset directory
- Count files and identify types
- Recommend appropriate processing steps

### 2. Generate Configuration

Create a configuration file to customize processing:

```bash
ai-neuro-wrangler init-config my_config.yaml
```

Edit `my_config.yaml` to adjust settings for your needs.

### 3. Run the Pipeline

Process your data with the full pipeline:

```bash
ai-neuro-wrangler wrangle \
    /path/to/input \
    /path/to/output \
    --metadata /path/to/metadata.csv \
    --config my_config.yaml \
    --report report.md
```

## Configuration

### Configuration File Structure

```yaml
# Volume normalization settings
normalization:
  method: zscore  # Options: zscore, minmax, percentile
  clip_percentiles: [1, 99]  # For percentile method

# Quality control settings
quality_control:
  min_snr: 10.0  # Minimum signal-to-noise ratio
  max_outlier_ratio: 0.05  # Maximum ratio of outlier voxels

# Outlier detection settings
outlier_detection:
  method: zscore  # Options: zscore, iqr, isolation_forest
  threshold: 3.0  # Z-score threshold

# Label encoding settings
label_encoding:
  encoding_type: label  # Options: label, onehot
  categorical_columns: null  # Auto-detect if null
```

## Using the CLI

### Analyze Command

```bash
ai-neuro-wrangler analyze /path/to/dataset
```

### Wrangle Command

```bash
ai-neuro-wrangler wrangle input/ output/ \
    --metadata metadata.csv \
    --report report.md
```

## Using the Python API

### Basic Usage

```python
from ai_neuro_wrangler import DataWranglingAgent

agent = DataWranglingAgent()
results = agent.run_pipeline(
    input_path="/path/to/input",
    output_path="/path/to/output",
    metadata_path="/path/to/metadata.csv"
)
```

## Best Practices

1. Always analyze your dataset first
2. Use configuration files
3. Generate reports for documentation
4. Validate results after processing

For more details, see the main [README](../README.md).
