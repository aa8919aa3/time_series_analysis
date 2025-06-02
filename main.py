"""Main execution script for time series analysis."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add the package directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Direct imports from module files in the current directory
from config import AnalysisConfig
from data_loader import load_data_safe
from analysis import LombScargleAnalyzer, CustomModelAnalyzer
from plotting import PlottingManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('time_series_analysis.log')
    ]
)

logger = logging.getLogger(__name__)


def run_analysis(data_file: str = None) -> None:
    """
    Run complete time series analysis pipeline.
    
    Args:
        data_file: Specific data file to analyze
    """
    logger.info("=== Starting Time Series Analysis ===")
    
    # Use specified data file or default
    file_path = data_file if data_file else '164_ic.csv'
    config = AnalysisConfig(file_path=file_path)
    
    logger.info(f"Analyzing file: {config.file_path}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Load data
    try:
        times, values, errors = load_data_safe(
            config.file_path,
            config.time_column,
            config.value_column, 
            config.error_column
        )
        logger.info(f"Loaded {len(times)} data points")
        logger.info(f"Time range: [{times.min():.6f}, {times.max():.6f}]")
        logger.info(f"Value range: [{values.min():.3e}, {values.max():.3e}]")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Initialize components
    ls_analyzer = LombScargleAnalyzer(config)
    custom_analyzer = CustomModelAnalyzer(config)
    plotter = PlottingManager(config)
    
    # Perform Lomb-Scargle analysis
    logger.info("--- Starting Lomb-Scargle Analysis ---")
    try:
        ls_results = ls_analyzer.analyze(times, values, errors)
        
        if ls_results.frequency is not None:
            logger.info("Lomb-Scargle analysis completed successfully")
            plotter.plot_lomb_scargle_results(ls_analyzer, times, values, errors)
        else:
            logger.warning("Lomb-Scargle analysis failed")
            
    except Exception as e:
        logger.error(f"Lomb-Scargle analysis failed: {e}")
        ls_results = None
    
    # Perform custom model analysis
    logger.info("--- Starting Josephson Junction Model Analysis ---")
    try:
        # Use best frequency from Lomb-Scargle if available
        initial_freq = None
        if ls_results and ls_results.best_frequency > 0:
            initial_freq = ls_results.best_frequency
        
        custom_results = custom_analyzer.analyze(
            times, values, errors, 
            initial_frequency=initial_freq,
            use_josephson_model=True
        )
        
        if custom_results.success:
            logger.info("Josephson model analysis completed successfully")
            plotter.plot_custom_model_results(
                custom_analyzer, times, values, errors, ls_analyzer
            )
        else:
            logger.warning("Josephson model analysis failed")
            
    except Exception as e:
        logger.error(f"Josephson model analysis failed: {e}")
        custom_results = None
    
    # Plot comparison of residuals
    logger.info("--- Plotting Residuals Comparison ---")
    try:
        plotter.plot_residuals_comparison(times, ls_analyzer, custom_analyzer)
    except Exception as e:
        logger.error(f"Residuals comparison plotting failed: {e}")
    
    # Summary
    logger.info("=== Analysis Summary ===")
    logger.info(f"File: {config.file_path}")
    if ls_results and ls_results.frequency is not None:
        logger.info(f"Lomb-Scargle - Best Period: {ls_results.best_period:.6f}, R²: {ls_results.r_squared:.4f}")
    
    if custom_results and custom_results.success:
        logger.info(f"Josephson Model - R²: {custom_results.r_squared:.4f}")
        
        # Log key Josephson parameters
        if custom_results.fit_result:
            params = custom_results.fit_result.params
            logger.info(f"  Critical Current (Ic): {params['Ic'].value:.3e}")
            logger.info(f"  Transparency (T): {params['T'].value:.3f}")
            logger.info(f"  Frequency (f): {params['f'].value:.3e}")
    
    logger.info("=== Analysis Complete ===\n")


def run_all_files():
    """Run analysis on both available data files."""
    data_files = ['164_ic.csv', '317_ic.csv']
    
    for data_file in data_files:
        if Path(data_file).exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"ANALYZING: {data_file}")
            logger.info(f"{'='*60}")
            run_analysis(data_file=data_file)
        else:
            logger.warning(f"File not found: {data_file}")


def main():
    """Main entry point."""
    try:
        # Check if both files exist and run analysis on them
        run_all_files()
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == '__main__':
    main()

