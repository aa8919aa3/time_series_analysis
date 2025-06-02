"""Data loading and validation utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)


def load_data(
    file_path: Union[str, Path], 
    time_col: str, 
    value_col: str, 
    error_col: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load time series data from a CSV file (legacy function).
    
    Args:
        file_path: Path to CSV file
        time_col: Name of time column
        value_col: Name of value column  
        error_col: Name of error column (optional)
        
    Returns:
        Tuple of (times, values, errors) or (None, None, None) if failed
    """
    try:
        return load_data_safe(file_path, time_col, value_col, error_col)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None, None, None


def load_data_safe(
    file_path: Union[str, Path],
    time_col: str,
    value_col: str, 
    error_col: Optional[str] = None,
    max_file_size_mb: float = 100
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Safely load time series data with comprehensive validation.
    
    Args:
        file_path: Path to CSV file
        time_col: Name of time column
        value_col: Name of value column
        error_col: Name of error column (optional)
        max_file_size_mb: Maximum allowed file size in MB
        
    Returns:
        Tuple of (times, values, errors)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data validation fails
    """
    # Validate file path
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB (limit: {max_file_size_mb}MB)")
    
    # Load data
    try:
        data = pd.read_csv(
            path,
            encoding='utf-8',
            engine='python',
            on_bad_lines='warn'
        )
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Validate columns
    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found. Available: {data.columns.tolist()}")
    
    if value_col not in data.columns:
        raise ValueError(f"Value column '{value_col}' not found. Available: {data.columns.tolist()}")
    
    # Extract data
    times = pd.to_numeric(data[time_col], errors='coerce').values
    values = pd.to_numeric(data[value_col], errors='coerce').values
    
    errors = None
    if error_col:
        if error_col in data.columns:
            errors_series = pd.to_numeric(data[error_col], errors='coerce')
            if errors_series.isnull().any():
                logger.warning(f"Column '{error_col}' contains non-numeric values. "
                             "Errors will be excluded for invalid points.")
            errors = errors_series.values
        else:
            logger.warning(f"Error column '{error_col}' not found. Proceeding without errors.")
    
    # Remove invalid data points
    valid_mask = ~np.isnan(times) & ~np.isnan(values)
    
    if errors is not None:
        valid_mask &= ~np.isnan(errors)
        errors = errors[valid_mask]
    
    times = times[valid_mask]
    values = values[valid_mask]
    
    if len(times) == 0:
        raise ValueError("No valid data points found after removing NaNs")
    
    logger.info(f"Successfully loaded {len(times)} data points from {file_path}")
    
    return times, values, errors


def validate_data_arrays(
    times: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64], 
    errors: Optional[npt.NDArray[np.float64]] = None
) -> None:
    """
    Validate time series data arrays.
    
    Args:
        times: Time array
        values: Value array
        errors: Error array (optional)
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(times, np.ndarray) or not isinstance(values, np.ndarray):
        raise ValueError("times and values must be numpy arrays")
    
    if len(times) != len(values):
        raise ValueError(f"Length mismatch: times({len(times)}) != values({len(values)})")
    
    if errors is not None:
        if not isinstance(errors, np.ndarray):
            raise ValueError("errors must be a numpy array")
        if len(errors) != len(times):
            raise ValueError(f"Length mismatch: errors({len(errors)}) != times({len(times)})")
    
    if len(times) == 0:
        raise ValueError("Empty data arrays")
    
    # Check for infinite values
    if np.any(np.isinf(times)) or np.any(np.isinf(values)):
        raise ValueError("Data contains infinite values")
    
    if errors is not None and np.any(np.isinf(errors)):
        raise ValueError("Errors contain infinite values")