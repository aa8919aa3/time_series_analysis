"""Analysis modules for time series data."""

from .lomb_scargle import LombScargleAnalyzer
from .custom_models import CustomModelAnalyzer
from .utils import calculate_r_squared, detrend_data

__all__ = [
    'LombScargleAnalyzer',
    'CustomModelAnalyzer', 
    'calculate_r_squared',
    'detrend_data'
]