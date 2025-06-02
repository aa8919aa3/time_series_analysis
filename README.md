# Time Series Analysis Package

A comprehensive Python package for time series analysis using Lomb-Scargle periodograms and custom Josephson junction model fitting.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## Overview

This package provides tools for analyzing time series data, particularly focused on Josephson junction current-phase relationships. It combines classical periodogram analysis with physics-based modeling to extract meaningful parameters from experimental data.

### Key Features

- **Lomb-Scargle Periodogram Analysis**: Robust frequency domain analysis for unevenly sampled data
- **Josephson Junction Modeling**: Physics-based model with transparency effects
- **Interactive Visualizations**: Comprehensive plotting using Plotly
- **Parallel Processing**: Efficient parameter space exploration
- **Statistical Analysis**: R-squared metrics, residual analysis, and error propagation
- **Modular Design**: Clean, extensible architecture

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```markdown name=README.md
# Time Series Analysis Package

A comprehensive Python package for time series analysis using Lomb-Scargle periodograms and custom Josephson junction model fitting.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## Overview

This package provides tools for analyzing time series data, particularly focused on Josephson junction current-phase relationships. It combines classical periodogram analysis with physics-based modeling to extract meaningful parameters from experimental data.

### Key Features

- **Lomb-Scargle Periodogram Analysis**: Robust frequency domain analysis for unevenly sampled data
- **Josephson Junction Modeling**: Physics-based model with transparency effects
- **Interactive Visualizations**: Comprehensive plotting using Plotly
- **Parallel Processing**: Efficient parameter space exploration
- **Statistical Analysis**: R-squared metrics, residual analysis, and error propagation
- **Modular Design**: Clean, extensible architecture

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `astropy >= 4.3.0`
- `lmfit >= 1.0.3`
- `plotly >= 5.0.0`

### Package Installation

```bash
# Clone the repository
git clone <repository-url>
cd time_series_analysis

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from time_series_analysis import (
    AnalysisConfig,
    load_data_safe,
    LombScargleAnalyzer,
    CustomModelAnalyzer,
    PlottingManager
)

# Configure analysis
config = AnalysisConfig(
    file_path='your_data.csv',
    time_column='Time',
    value_column='Current',
    error_column='Error'
)

# Load data
times, values, errors = load_data_safe(
    config.file_path,
    config.time_column,
    config.value_column,
    config.error_column
)

# Initialize analyzers
ls_analyzer = LombScargleAnalyzer(config)
custom_analyzer = CustomModelAnalyzer(config)
plotter = PlottingManager(config)

# Perform analyses
ls_results = ls_analyzer.analyze(times, values, errors)
custom_results = custom_analyzer.analyze(times, values, errors)

# Generate plots
plotter.plot_lomb_scargle_results(ls_analyzer, times, values, errors)
plotter.plot_custom_model_results(custom_analyzer, times, values, errors, ls_analyzer)
```

### Running the Complete Analysis

```bash
# Run with default configuration
python time_series_analysis/main.py

# Run the example with synthetic data
python time_series_analysis/examples/basic_usage.py
```

## Data Format

Your CSV file should contain columns for:
- **Time/External Parameter**: Independent variable (e.g., external flux Φ_ext)
- **Measurement Values**: Dependent variable (e.g., switching current I_s)
- **Errors** (optional): Measurement uncertainties

Example CSV structure:
```csv
Time,Ic,Error
0.0,1.23e-6,0.05e-6
0.1,1.45e-6,0.05e-6
0.2,1.67e-6,0.05e-6
...
```

## Scientific Background

### Lomb-Scargle Periodogram

The Lomb-Scargle periodogram is particularly suited for unevenly sampled time series data. It calculates the spectral power as a function of frequency:

```
P(ω) = (1/2) * [|∑ᵢ xᵢ cos(ωtᵢ)|² / ∑ᵢ cos²(ωtᵢ) + |∑ᵢ xᵢ sin(ωtᵢ)|² / ∑ᵢ sin²(ωtᵢ)]
```

### Josephson Junction Model

The package implements the current-phase relationship for Josephson junctions with finite transparency:

```
I_s(Φ_ext) = I_c · sin(2πf(Φ_ext-d)-φ₀) / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

Where:
- `I_c`: Critical current
- `T`: Junction transparency (0-1)
- `f`: Conversion factor (external parameter to phase scaling)
- `d`: Horizontal shift (zero-point offset)
- `φ₀`: Intrinsic phase offset
- `k`: Quadratic background coefficient
- `r`: Linear background coefficient
- `C`: Overall current offset

## Configuration

### AnalysisConfig Parameters

```python
@dataclass
class AnalysisConfig:
    # Data configuration
    file_path: str = '164_ic.csv'
    time_column: str = 'Time'
    value_column: str = 'Ic'
    error_column: Optional[str] = None
    
    # Lomb-Scargle parameters
    min_frequency_ls: Optional[float] = None
    max_frequency_ls: float = 500000
    samples_per_peak_ls: int = 10
    fap_levels_ls: List[float] = [0.1, 0.05, 0.01]
    detrend_order_ls: int = 1
    
    # Custom model parameters
    d_multiplier: float = 100
    c_multiplier: float = 10
    d_intervals: int = 200
    c_intervals: int = 20
    
    # Plotting parameters
    plot_title_prefix: str = "Time Series Analysis"
    plot_theme: str = "plotly_white"
```

## API Reference

### Core Classes

#### LombScargleAnalyzer
```python
class LombScargleAnalyzer:
    def __init__(self, config: AnalysisConfig)
    def analyze(self, times, values, errors=None) -> LombScargleResults
    def create_smooth_model_curve(self, times, extend_factor=0.05, num_points=1000)
```

#### CustomModelAnalyzer
```python
class CustomModelAnalyzer:
    def __init__(self, config: AnalysisConfig)
    def analyze(self, times, values, errors=None, initial_frequency=None, 
                use_josephson_model=True) -> CustomModelResults
    def create_smooth_model_curve(self, times, extend_factor=0.05, num_points=1000)
```

#### PlottingManager
```python
class PlottingManager:
    def __init__(self, config: AnalysisConfig)
    def plot_lomb_scargle_results(self, analyzer, times, values, errors=None)
    def plot_custom_model_results(self, analyzer, times, values, errors=None, ls_analyzer=None)
    def plot_residuals_comparison(self, times, ls_analyzer=None, custom_analyzer=None)
```

### Utility Functions

```python
# Data loading
def load_data_safe(file_path, time_col, value_col, error_col=None)

# Statistical utilities
def calculate_r_squared(y_true, y_pred) -> float
def detrend_data(times, values, order=1) -> Tuple[np.ndarray, np.ndarray]
def compute_residuals_integral(times, residuals) -> float
```

## Output and Results

### Generated Plots

1. **Lomb-Scargle Periodogram**: Frequency vs. power with FAP thresholds
2. **Data with Fits**: Original data overlaid with model predictions
3. **Phase-Folded Data**: Data folded by the best period
4. **Parameter Correlation Heatmap**: Parameter correlations from fitting
5. **Residuals Analysis**: Residuals vs. time and histograms
6. **Parameter Matrix**: R-squared landscape for parameter exploration

### Analysis Outputs

- **Best-fit Parameters**: With uncertainties when available
- **R-squared Values**: Goodness-of-fit metrics
- **False Alarm Probabilities**: Statistical significance of detected periods
- **Residual Statistics**: Integral and distribution analysis

## Advanced Features

### Parallel Parameter Exploration

The package includes parallel computation for exploring parameter space:

```python
# Automatically uses multiple cores for parameter matrix computation
custom_results = custom_analyzer.analyze(
    times, values, errors,
    use_josephson_model=True
)

# Access parameter matrix results
if custom_results.parameter_matrix is not None:
    plotter.custom_plotter.plot_parameter_matrix_heatmap(custom_results)
```

### Custom Model Development

To add new physics models:

1. Define your model function:
```python
def my_model_func(x, param1, param2, ...):
    return model_prediction
```

2. Extend the `CustomModelAnalyzer` class
3. Add parameter estimation and setup methods
4. Include in the plotting framework

## Examples

### Example 1: Synthetic Josephson Data

```python
# Generate synthetic data
phi_ext = np.linspace(-2, 2, 200)
# ... (see examples/basic_usage.py for complete example)

# Run analysis
run_basic_analysis()
```

### Example 2: Real Experimental Data

```python
config = AnalysisConfig(
    file_path='experimental_data.csv',
    time_column='Flux_ext',
    value_column='Switching_Current',
    error_column='Current_Error',
    max_frequency_ls=1000,
    plot_title_prefix="Josephson Junction Experiment"
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

2. **Data Loading Failures**: Check CSV format and column names
```python
# Verify column names
import pandas as pd
data = pd.read_csv('your_file.csv')
print(data.columns.tolist())
```

3. **Fitting Convergence Issues**: Adjust parameter bounds or initial guesses
```python
# Check data quality
print(f"Data range: {np.ptp(values)}")
print(f"Data mean: {np.mean(values)}")
print(f"Contains NaN: {np.any(np.isnan(values))}")
```

4. **Memory Issues with Parameter Matrix**: Reduce intervals
```python
config.d_intervals = 50  # Reduce from default 200
config.c_intervals = 10  # Reduce from default 20
```

### Performance Optimization

- **Large Datasets**: Consider downsampling for initial analysis
- **Parallel Processing**: Adjust `n_jobs` parameter for your system
- **Memory Usage**: Monitor RAM usage during parameter matrix computation

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd time_series_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all public functions
- Add docstrings for all classes and methods
- Include unit tests for new features

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{time_series_analysis,
  author = {aa8919aa3},
  title = {Time Series Analysis Package for Josephson Junction Data},
  year = {2025},
  url = {<repository-url>}
}
```

## Changelog

### Version 1.0.0 (2025-06-02)
- Initial release
- Lomb-Scargle periodogram analysis
- Josephson junction model fitting
- Interactive plotting with Plotly
- Parallel parameter exploration
- Comprehensive documentation

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation
- Review the examples directory

## Acknowledgments

- Built using the excellent [astropy](https://www.astropy.org/) library for Lomb-Scargle analysis
- Model fitting powered by [lmfit](https://lmfit.github.io/lmfit-py/)
- Interactive visualizations with [Plotly](https://plotly.com/python/)
- Scientific computing foundation from [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)

---

**Author**: aa8919aa3  
**Created**: June 2, 2025  
**Last Updated**: June 2, 2025
```

I've created a comprehensive README file that includes:

## Key Sections:

1. **Overview**: Clear description of the package purpose and features
2. **Installation**: Step-by-step setup instructions
3. **Quick Start**: Basic usage examples to get users started quickly
4. **Scientific Background**: Mathematical foundations of the methods
5. **Configuration**: Detailed parameter descriptions
6. **API Reference**: Complete class and function documentation
7. **Examples**: Practical usage scenarios
8. **Troubleshooting**: Common issues and solutions
9. **Contributing**: Guidelines for developers
10. **License and Citation**: Professional project information

## Special Features:

- **Badges**: Professional status indicators
- **Code Examples**: Practical, copy-paste ready examples
- **Mathematical Formulas**: Proper equations for the Josephson model
- **Performance Tips**: Optimization guidance
- **Troubleshooting**: Common issues and solutions
- **Development Setup**: Contributing guidelines

The README is structured to serve both end-users who want to use the package and developers who want to contribute to it. It provides complete documentation while remaining accessible and well-organized.
