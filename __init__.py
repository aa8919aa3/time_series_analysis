"""
Time Series Analysis Package

A comprehensive package for time series analysis using Lomb-Scargle periodograms
and custom model fitting with lmfit.
"""

__version__ = "1.0.0"
__author__ = "aa8919aa3"

from .config import AnalysisConfig
from .data_loader import load_data, load_data_safe
from .analysis import LombScargleAnalyzer, CustomModelAnalyzer
from .plotting import PlottingManager

__all__ = [
    'AnalysisConfig',
    'load_data', 
    'load_data_safe',
    'LombScargleAnalyzer',
    'CustomModelAnalyzer', 
    'PlottingManager'
]