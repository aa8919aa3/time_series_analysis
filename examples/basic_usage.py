"""Basic usage example for time series analysis package."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from time_series_analysis import (
    AnalysisConfig,
    load_data_safe,
    LombScargleAnalyzer,
    CustomModelAnalyzer,
    PlottingManager
)


def create_sample_data():
    """Create sample Josephson junction data for testing."""
    
    # Parameters for synthetic Josephson junction
    phi_ext = np.linspace(-2, 2, 200)  # External parameter range
    
    # True parameters
    Ic_true = 1e-6  # Critical current
    T_true = 0.3    # Transparency
    f_true = 2.5    # Frequency scaling
    d_true = 0.1    # Horizontal shift
    phi0_true = 0.2 # Phase offset
    k_true = 1e-7   # Quadratic background
    r_true = 2e-7   # Linear background
    C_true = 5e-7   # Current offset
    
    # Generate synthetic data
    phase = 2 * np.pi * f_true * (phi_ext - d_true) - phi0_true
    sin_phase = np.sin(phase)
    sin_half_phase = np.sin(phase / 2)
    denominator = np.sqrt(1 - T_true * sin_half_phase**2)
    
    josephson_term = Ic_true * sin_phase / denominator
    background = k_true * (phi_ext - d_true)**2 + r_true * (phi_ext - d_true) + C_true
    
    current_clean = josephson_term + background
    
    # Add noise
    noise_level = 0.05 * np.std(current_clean)
    noise = np.random.normal(0, noise_level, len(phi_ext))
    current_noisy = current_clean + noise
    
    # Create errors
    errors = np.full_like(current_noisy, noise_level)
    
    # Save as CSV
    data = pd.DataFrame({
        'Phi_ext': phi_ext,
        'Current': current_noisy,
        'Error': errors
    })
    
    csv_path = Path(__file__).parent / 'sample_josephson_data.csv'
    data.to_csv(csv_path, index=False)
    
    print(f"Sample data created: {csv_path}")
    print(f"True parameters:")
    print(f"  Ic: {Ic_true:.3e}")
    print(f"  T: {T_true:.3f}")
    print(f"  f: {f_true:.3f}")
    print(f"  d: {d_true:.3f}")
    print(f"  phi0: {phi0_true:.3f}")
    
    return csv_path


def run_basic_analysis():
    """Run basic analysis example."""
    
    # Create sample data
    data_file = create_sample_data()
    
    # Configure analysis
    config = AnalysisConfig(
        file_path=str(data_file),
        time_column='Phi_ext',
        value_column='Current',
        error_column='Error',
        plot_title_prefix="Josephson Junction Analysis"
    )
    
    # Load data
    times, values, errors = load_data_safe(
        config.file_path,
        config.time_column,
        config.value_column,
        config.error_column
    )
    
    print(f"\nLoaded {len(times)} data points")
    
    # Initialize analyzers
    ls_analyzer = LombScargleAnalyzer(config)
    custom_analyzer = CustomModelAnalyzer(config)
    plotter = PlottingManager(config)
    
    # Lomb-Scargle analysis
    print("\n--- Lomb-Scargle Analysis ---")
    ls_results = ls_analyzer.analyze(times, values, errors)
    
    if ls_results.frequency is not None:
        print(f"Best frequency: {ls_results.best_frequency:.4f}")
        print(f"Best period: {ls_results.best_period:.4f}")
        print(f"R-squared: {ls_results.r_squared:.4f}")
        
        # Plot results
        plotter.plot_lomb_scargle_results(ls_analyzer, times, values, errors)
    
    # Josephson model analysis
    print("\n--- Josephson Junction Model Analysis ---")
    custom_results = custom_analyzer.analyze(
        times, values, errors,
        initial_frequency=ls_results.best_frequency if ls_results.frequency is not None else None,
        use_josephson_model=True
    )
    
    if custom_results.success:
        print(f"R-squared: {custom_results.r_squared:.4f}")
        
        # Print fitted parameters
        params = custom_results.fit_result.params
        print("\nFitted parameters:")
        for name, param in params.items():
            if param.stderr:
                print(f"  {name}: {param.value:.3e} ± {param.stderr:.3e}")
            else:
                print(f"  {name}: {param.value:.3e}")
        
        # Plot results
        plotter.plot_custom_model_results(
            custom_analyzer, times, values, errors, ls_analyzer
        )
    
    # Comparison plots
    print("\n--- Residuals Comparison ---")
    plotter.plot_residuals_comparison(times, ls_analyzer, custom_analyzer)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    run_basic_analysis()