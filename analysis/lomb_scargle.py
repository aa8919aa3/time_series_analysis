"""Lomb-Scargle periodogram analysis."""

import numpy as np
from astropy.timeseries import LombScargle
from astropy import units as u
from typing import Tuple, Optional, List
import numpy.typing as npt
import logging

from .utils import calculate_r_squared, detrend_data, compute_residuals_integral
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import AnalysisConfig

logger = logging.getLogger(__name__)


class LombScargleResults:
    """Container for Lomb-Scargle analysis results."""
    
    def __init__(self):
        self.frequency: Optional[npt.NDArray[np.float64]] = None
        self.power: Optional[npt.NDArray[np.float64]] = None
        self.best_frequency: float = 0.0
        self.best_period: float = float('inf')
        self.best_power: float = 0.0
        self.r_squared: float = np.nan
        self.amplitude: float = np.nan
        self.phase_rad: float = np.nan
        self.offset: float = np.nan
        self.fap_levels: Optional[List[float]] = None
        self.power_thresholds: Optional[npt.NDArray[np.float64]] = None
        self.fap_best_peak: Optional[float] = None
        self.residuals: Optional[npt.NDArray[np.float64]] = None
        self.residuals_integral: float = np.nan
        self.model_predictions: Optional[npt.NDArray[np.float64]] = None
        self.detrend_coeffs: Optional[npt.NDArray[np.float64]] = None


class LombScargleAnalyzer:
    """Lomb-Scargle periodogram analyzer."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = LombScargleResults()
    
    def analyze(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64], 
        errors: Optional[npt.NDArray[np.float64]] = None
    ) -> LombScargleResults:
        """
        Perform complete Lomb-Scargle analysis.
        
        Args:
            times: Time array
            values: Value array
            errors: Error array (optional)
            
        Returns:
            LombScargleResults object containing all results
        """
        logger.info("Starting Lomb-Scargle analysis")
        
        # Prepare data
        ls_times, ls_values, ls_errors = self._prepare_data(times, values, errors)
        
        # Create Lomb-Scargle object
        ls = LombScargle(ls_times, ls_values, dy=ls_errors, fit_mean=True, center_data=True)
        
        # Compute periodogram
        self._compute_periodogram(ls, ls_times)
        
        # Find best period and compute fit
        if self.results.frequency is not None and len(self.results.frequency) > 0:
            self._find_best_period(ls)
            self._compute_model_fit(ls, times, values)
            self._compute_false_alarm_probabilities(ls)
        
        logger.info("Lomb-Scargle analysis completed")
        return self.results
    
    def _prepare_data(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """Prepare data for Lomb-Scargle analysis including detrending."""
        
        ls_times = times.copy()
        ls_values = values.copy()
        ls_errors = errors.copy() if errors is not None else None
        
        # Detrend if requested
        if self.config.detrend_order_ls is not None and self.config.detrend_order_ls >= 0:
            try:
                ls_values, self.results.detrend_coeffs = detrend_data(
                    ls_times, ls_values, self.config.detrend_order_ls
                )
                logger.info(f"Data detrended with polynomial order {self.config.detrend_order_ls}")
            except ValueError as e:
                logger.warning(f"Detrending failed: {e}. Using original data.")
                self.results.detrend_coeffs = None
        
        return ls_times, ls_values, ls_errors
    
    def _compute_periodogram(
        self, 
        ls: LombScargle, 
        times: npt.NDArray[np.float64]
    ) -> None:
        """Compute the Lomb-Scargle periodogram."""
        
        # Determine frequency range
        min_freq = self.config.min_frequency_ls
        if min_freq is None and len(times) > 1:
            time_span = times.max() - times.min()
            if time_span > 0:
                min_freq = 0.5 / time_span
                logger.info(f"Auto-setting minimum frequency to {min_freq:.2e}")
            else:
                logger.warning("Cannot auto-determine minimum frequency")
        
        try:
            self.results.frequency, self.results.power = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=self.config.max_frequency_ls,
                samples_per_peak=self.config.samples_per_peak_ls
            )
            logger.info(f"Computed periodogram with {len(self.results.frequency)} frequency points")
        except Exception as e:
            logger.error(f"Failed to compute periodogram: {e}")
            self.results.frequency = None
            self.results.power = None
    
    def _find_best_period(self, ls: LombScargle) -> None:
        """Find the best period from the periodogram."""
        
        if self.results.power is None:
            return
        
        best_idx = np.argmax(self.results.power)
        self.results.best_frequency = self.results.frequency[best_idx]
        self.results.best_power = self.results.power[best_idx]
        
        if self.results.best_frequency != 0:
            self.results.best_period = 1.0 / self.results.best_frequency
            
            # Get fit parameters
            self.results.offset = ls.offset()
            params = ls.model_parameters(self.results.best_frequency)
            self.results.amplitude = np.sqrt(params[0]**2 + params[1]**2)
            self.results.phase_rad = np.arctan2(params[1], params[0])
            
            logger.info(f"Best frequency: {self.results.best_frequency:.6f}")
            logger.info(f"Best period: {self.results.best_period:.6f}")
            logger.info(f"Best power: {self.results.best_power:.4f}")
    
    def _compute_model_fit(
        self,
        ls: LombScargle,
        original_times: npt.NDArray[np.float64],
        original_values: npt.NDArray[np.float64]
    ) -> None:
        """Compute model predictions and residuals."""
        
        if self.results.best_frequency == 0:
            return
        
        # Model predictions on detrended scale
        model_detrended = ls.model(original_times, self.results.best_frequency)
        
        # Convert back to original scale
        self.results.model_predictions = model_detrended
        if self.results.detrend_coeffs is not None:
            trend = np.polyval(self.results.detrend_coeffs, original_times)
            self.results.model_predictions = trend + model_detrended
        
        # Compute metrics
        self.results.r_squared = calculate_r_squared(original_values, self.results.model_predictions)
        self.results.residuals = original_values - self.results.model_predictions
        self.results.residuals_integral = compute_residuals_integral(original_times, self.results.residuals)
        
        logger.info(f"R-squared: {self.results.r_squared:.4f}")
        logger.info(f"Residuals integral: {self.results.residuals_integral:.4e}")
    
    def _compute_false_alarm_probabilities(self, ls: LombScargle) -> None:
        """Compute false alarm probabilities."""
        
        if self.results.best_power <= 0:
            logger.warning("Skipping FAP calculation: best power is not positive")
            return
        
        try:
            # FAP for best peak
            self.results.fap_best_peak = ls.false_alarm_probability(
                self.results.best_power, method='baluev'
            )
            
            # Power thresholds for given FAP levels
            self.results.power_thresholds = ls.false_alarm_level(
                self.config.fap_levels_ls, method='baluev'
            )
            self.results.fap_levels = self.config.fap_levels_ls
            
            logger.info(f"FAP for best peak: {self.results.fap_best_peak:.2e}")
            for level, thresh in zip(self.results.fap_levels, self.results.power_thresholds):
                logger.info(f"Power threshold for {level*100:.1f}% FAP: {thresh:.4f}")
                
        except Exception as e:
            logger.warning(f"Failed to compute FAP: {e}")
    
    def create_smooth_model_curve(
        self, 
        times: npt.NDArray[np.float64],
        extend_factor: float = 0.05,
        num_points: int = 1000
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Create smooth model curve for plotting.
        
        Args:
            times: Original time array
            extend_factor: Factor by which to extend time range
            num_points: Number of points in smooth curve
            
        Returns:
            Tuple of (time_smooth, model_smooth)
        """
        if self.results.best_frequency == 0:
            return times, np.zeros_like(times)
        
        # Create extended time range
        time_span = times.max() - times.min()
        if time_span > 0:
            t_min = times.min() - extend_factor * time_span
            t_max = times.max() + extend_factor * time_span
            t_smooth = np.linspace(t_min, t_max, max(num_points, 2 * len(times)))
        else:
            t_smooth = np.sort(np.unique(times))
            if len(t_smooth) == 1:
                t_smooth = np.array([t_smooth[0] - 0.5, t_smooth[0], t_smooth[0] + 0.5])
        
        # Create Lomb-Scargle object for model evaluation
        # Note: This assumes we have access to the detrended data
        # In practice, you might need to store the LS object or recreate it
        
        # For now, return a simple sinusoidal model
        model_smooth = (self.results.amplitude * 
                       np.sin(2 * np.pi * self.results.best_frequency * t_smooth + self.results.phase_rad) + 
                       self.results.offset)
        
        # Add trend back if detrending was performed
        if self.results.detrend_coeffs is not None:
            trend_smooth = np.polyval(self.results.detrend_coeffs, t_smooth)
            model_smooth += trend_smooth
        
        return t_smooth, model_smooth