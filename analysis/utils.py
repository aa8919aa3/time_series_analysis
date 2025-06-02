"""Utility functions for time series analysis."""

import numpy as np
from typing import Tuple, Optional, Union
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)


def calculate_r_squared(
    y_true: npt.NDArray[np.float64], 
    y_pred: npt.NDArray[np.float64]
) -> float:
    """
    Calculate the coefficient of determination (R-squared).
    
    R² = 1 - (SS_res / SS_tot), where:
    - SS_res = Σ(y_true - y_pred)²  
    - SS_tot = Σ(y_true - mean(y_true))²
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        R-squared value. Returns NaN if calculation impossible.
        
    Note:
        Handles NaN values by excluding them from calculation.
        Returns 1.0 for perfect fit with zero total variance.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    if not np.any(valid_mask):
        return np.nan
        
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    
    # Handle edge case where all true values are identical
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0
        
    return 1.0 - (ss_res / ss_tot)


def detrend_data(
    times: npt.NDArray[np.float64], 
    values: npt.NDArray[np.float64], 
    order: int = 1
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
    """
    Detrend data by subtracting a polynomial fit.
    
    Args:
        times: Time array
        values: Value array  
        order: Polynomial order for detrending
        
    Returns:
        Tuple of (detrended_values, polynomial_coefficients)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if not isinstance(times, np.ndarray) or not isinstance(values, np.ndarray):
        raise ValueError("times and values must be numpy arrays")
    
    if len(times) != len(values):
        raise ValueError("times and values must have same length")
    
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a non-negative integer")
    
    if len(times) <= order:
        raise ValueError(f"Need at least {order + 1} data points for order {order} polynomial")
    
    try:
        poly_coeffs = np.polyfit(times, values, order)
        trend = np.polyval(poly_coeffs, times)
        detrended_values = values - trend
        logger.info(f"Data detrended using polynomial of order {order}")
        return detrended_values, poly_coeffs
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Polynomial fitting failed: {e}")


def compute_residuals_integral(
    times: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64]
) -> float:
    """
    Compute the integral of residuals using trapezoidal rule.
    
    Args:
        times: Time array (must be sorted)
        residuals: Residual array
        
    Returns:
        Integral value, or NaN if computation fails
    """
    if len(times) < 2 or np.all(np.isnan(residuals)):
        return np.nan
    
    # Sort by time
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    residuals_sorted = residuals[sort_idx]
    
    # Remove NaN values
    valid_mask = ~np.isnan(residuals_sorted)
    if not np.any(valid_mask):
        return np.nan
    
    times_clean = times_sorted[valid_mask]
    residuals_clean = residuals_sorted[valid_mask]
    
    if len(times_clean) < 2:
        return np.nan
    
    try:
        if hasattr(np, 'trapezoid'):
            return np.trapezoid(residuals_clean, times_clean)
        else:
            return np.trapz(residuals_clean, times_clean)
    except Exception as e:
        logger.warning(f"Failed to compute integral: {e}")
        return np.nan


def estimate_initial_parameters(
    times: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    best_frequency: Optional[float] = None
) -> dict:
    """
    Estimate initial parameters for model fitting.
    
    Args:
        times: Time array
        values: Value array
        best_frequency: Best frequency from Lomb-Scargle (optional)
        
    Returns:
        Dictionary of initial parameter estimates
    """
    # Linear trend estimation
    slope_init, intercept_init = np.polyfit(times, values, 1)
    
    # Amplitude estimation from detrended residuals
    residuals = values - (slope_init * times + intercept_init)
    amp_init = np.std(residuals) * np.sqrt(2)
    
    if amp_init == 0:
        amp_init = np.std(values) * np.sqrt(2)
    if amp_init == 0:
        amp_init = 1e-7
    
    # Frequency estimation
    freq_init = best_frequency if best_frequency and best_frequency > 1e-9 else 1.0
    
    return {
        'amplitude': max(amp_init, 1e-9),
        'frequency': freq_init,
        'phase': 0.0,
        'slope': slope_init,
        'offset': intercept_init,
        'd_shift': times.min()
    }