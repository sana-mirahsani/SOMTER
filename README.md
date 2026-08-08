# SMOTER - SMOTE for Regression

Implementation of the SMOTE algorithm adapted for regression tasks to handle imbalanced target variables.

# Description
This project implements the **SMOTER (SMOTE for Regression)** algorithm, which generates synthetic samples for rare/important regions of the target variable in regression problems. Unlike classification SMOTE, this algorithm works with continuous target variables by using a **relevance function** to identify which target values are considered "rare" or "important" and should be oversampled.

The implementation provides a flexible, customizable solution that can be used with any tabular dataset.

# Features
- **Automatic relevance calculation** using IQR-based distribution analysis
- **Custom relevance function support** for domain-specific importance definitions
- **Flexible oversampling strategies** ('balance' or fixed ratio)
- **Pandas DataFrame integration** for seamless data processing
- **Visualization** of target distribution before and after resampling
- **Reproducible results** with random state control
- **Scikit-learn compatible** interface

# Model / Algorithm

The algorithm follows these key steps:

1. **Relevance Calculation**: Computes importance score φ(y) for each target value
2. **Rare Sample Identification**: Selects samples with relevance > threshold
3. **Neighbor Selection**: Finds k-nearest neighbors for each rare sample
4. **Synthetic Generation**: Creates new samples through interpolation between rare samples and their neighbors
5. **Target Value Assignment**: Sets target values as weighted average of parent samples

### Relevance Function (φ)

The relevance function maps target values to importance scores between 0 and 1:

| Relevance Score | Meaning |
|----------------|---------|
| φ(y) = 1 | Maximum importance (rare/extreme values) |
| φ(y) = 0 | Minimum importance (common/average values) |

## Example Results

After applying SMOTER to an imbalanced dataset:

| Metric | Before SMOTER | After SMOTER |
|--------|---------------|--------------|
| Sample Size | 900 | 1,650+ |
| Target Range | 2.0 - 18.0 | 2.0 - 18.0 |
| Standard Deviation | ~2.8 | ~4.2 |
| Distribution Balance | Skewed (rare tails) | Balanced (uniform) |

The algorithm successfully generates synthetic samples for rare target values while preserving the original data distribution.

# Dataset

The project works with any **tabular dataset** for regression tasks. The implementation:

- Accepts both NumPy arrays and Pandas DataFrames
- Handles numerical features automatically
- Requires a continuous target variable
- Works with any number of features

**Input Format:**
- Features: Numerical matrix (n_samples × n_features)
- Target: Continuous values (n_samples,)

# Demo

The project includes a built-in demonstration using synthetic data:

```python
from smoter import SMOTER
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Apply SMOTER
smoter = SMOTER(
    k=5,
    relevance_threshold=0.6,
    oversampling_ratio='balance',
    random_state=42
)

# Resample the dataset
df_resampled = smoter.fit_resample_dataframe(df, target_column='target')
```

### Before SMOTER:
![Before SMOTER](Figures/target_distribution_before.png)

### After SMOTER:
![After SMOTER](Figures/target_distribution_after.png)

# Project Structure

```
SMOTER/
│
├── main.py                     # Main implementation with SMOTER class
│
├── Figures/                    # Generated visualizations
│   └── target_distribution.png # Target distribution comparison
│
├── Data/                       # (Optional) Your datasets
│   └── your_data.csv
│
├── README.md                   # Project documentation
│
├── requirements.txt            # Python dependencies
│
└── LICENSE                     # License information

```

# Installation

## Clone the repository
```bash
git clone https://github.com/yourusername/SMOTER.git
cd SMOTER
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Requirements
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
typing-extensions>=4.0.0
```

# Usage

## Basic Usage with NumPy Arrays

```python
from smoter import SMOTER
import numpy as np

# Prepare your data
X = np.array([[...], [...], ...])  # Features
y = np.array([...])                # Target

# Initialize SMOTER
smoter = SMOTER(
    k=5,                      # Number of neighbors
    relevance_threshold=0.5,  # Threshold for rare samples
    oversampling_ratio='balance',  # Auto-balance
    random_state=42           # Reproducibility
)

# Resample
X_resampled, y_resampled = smoter.fit_resample(X, y)
```

## Usage with Pandas DataFrame

```python
import pandas as pd
from smoter import SMOTER

# Load your DataFrame
df = pd.read_csv('data.csv')

# Initialize and apply SMOTER
smoter = SMOTER(
    k=7,
    relevance_threshold=0.6,
    oversampling_ratio=2.0,  # 200% oversampling
    random_state=123
)

# Get resampled DataFrame
df_resampled = smoter.fit_resample_dataframe(df, target_column='target')
```

## Custom Relevance Function

```python
def custom_relevance(y):
    """
    Custom relevance function: give importance to values > 100
    """
    relevance = np.zeros_like(y, dtype=float)
    mask = y > 100
    if mask.any():
        relevance[mask] = (y[mask] - 100) / (y.max() - 100)
    return relevance

# Use custom function
smoter = SMOTER(
    relevance_function=custom_relevance,
    relevance_threshold=0.5,
    oversampling_ratio='balance'
)
```

## Run the Demo

```python
# Run the built-in example
python main.py
```

# Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 5 | Number of nearest neighbors |
| `relevance_threshold` | float | 0.5 | Threshold for rare sample identification |
| `oversampling_ratio` | float/str | 'balance' | Oversampling ratio or 'balance' for auto-balancing |
| `random_state` | int | None | Random seed for reproducibility |
| `relevance_function` | callable | None | Custom relevance function (φ) |

# Key Methods

| Method | Description |
|--------|-------------|
| `fit_resample(X, y)` | Apply SMOTER to NumPy arrays |
| `fit_resample_dataframe(df, target_column)` | Apply SMOTER to Pandas DataFrame |
| `_calculate_relevance_auto(y)` | Automatic relevance calculation |
| `_generate_synthetic_samples()` | Generate synthetic samples |

# Technologies Used

- **Python 3.8+** - Core programming language
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - Nearest neighbors implementation
- **Matplotlib** - Visualization
- **Typing** - Type hints for better code quality

# Mathematical Foundation

The SMOTER algorithm extends SMOTE to regression by introducing:

1. **Relevance Function (φ)**: Maps target values to importance scores
   ```
   φ: Y → [0, 1]
   ```
   where φ(y) = 1 for most important values, 0 for least important.

2. **Interpolation**: Creates new samples using linear interpolation
   ```
   x_new = x_rare + λ × (x_neighbor - x_rare)
   y_new = (y_rare + y_neighbor) / 2
   ```
   where λ ∈ [0, 1] is a random number.

3. **Oversampling Strategy**: Balances rare samples to match normal samples
   ```
   n_synthetic = n_normal - n_rare  (when oversampling_ratio='balance')
   ```

# License

MIT License - Feel free to use and modify for your projects.

# Author

**Your Name**
- Role: Machine Learning Developer
- Contact: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [Your GitHub](https://github.com/yourusername)

# Acknowledgments

- Based on the paper: "SMOTE for Regression" by Torgo, Ribeiro, et al.
- Inspired by the need for handling imbalanced regression problems

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# Citation

If you use this implementation in your research, please cite:

```bibtex
@article{torgo2013smote,
  title={SMOTE for Regression},
  author={Torgo, Lu{\'\i}s and Ribeiro, Rita P and Pfahringer, Bernhard and Branco, Paula},
  journal={Portuguese Conference on Artificial Intelligence},
  year={2013}
}
```

---

**Happy Resampling! 📊**