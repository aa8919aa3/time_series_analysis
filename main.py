"""Main execution script for time series analysis."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add the package directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from time_series_analysis import (
    AnalysisConfig, 
    load_data_safe, 
    LombScargleAnalyzer, 
    CustomModelAnalyzer,
    PlottingManager
)

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


def run_analysis(config_path: Optional[str] = None) -> None:
    """
    Run complete time series analysis pipeline.
    
    Args:
        config_path: Optional path to configuration file
    """
    logger.info("=== Starting Time Series Analysis ===")
    
    # Load configuration
    if config_path:
        # TODO: Implement config loading from file
        logger.info(f"Loading configuration from {config_path}")
        config = AnalysisConfig()  # Placeholder
    else:
        config = AnalysisConfig()
    
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
    
    logger.info("=== Analysis Complete ===")


def main():
    """Main entry point."""
    try:
        run_analysis()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == '__main__':
    main()
