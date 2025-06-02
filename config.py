"""Configuration management for time series analysis."""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class AnalysisConfig:
    """Configuration class for time series analysis parameters."""
    
    # Data configuration
    file_path: str = '164_ic.csv'
    time_column: str = 'Time'
    value_column: str = 'Ic'
    error_column: Optional[str] = None
    
    # Lomb-Scargle parameters
    min_frequency_ls: Optional[float] = None
    max_frequency_ls: float = 500000
    samples_per_peak_ls: int = 10
    fap_levels_ls: List[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])
    detrend_order_ls: int = 1
    
    # Custom model parameters
    d_multiplier: float = 100
    c_multiplier: float = 10
    d_intervals: int = 200
    c_intervals: int = 20
    
    # Plotting parameters
    plot_title_prefix: str = "Time Series Analysis"
    plot_theme: str = "plotly_white"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_frequency_ls <= 0:
            raise ValueError("max_frequency_ls must be positive")
        
        if self.samples_per_peak_ls <= 0:
            raise ValueError("samples_per_peak_ls must be positive")
        
        if any(level <= 0 or level >= 1 for level in self.fap_levels_ls):
            raise ValueError("FAP levels must be between 0 and 1")
        
        if self.detrend_order_ls < 0:
            raise ValueError("detrend_order_ls must be non-negative")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AnalysisConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'file_path': self.file_path,
            'time_column': self.time_column,
            'value_column': self.value_column,
            'error_column': self.error_column,
            'min_frequency_ls': self.min_frequency_ls,
            'max_frequency_ls': self.max_frequency_ls,
            'samples_per_peak_ls': self.samples_per_peak_ls,
            'fap_levels_ls': self.fap_levels_ls,
            'detrend_order_ls': self.detrend_order_ls,
            'd_multiplier': self.d_multiplier,
            'c_multiplier': self.c_multiplier,
            'd_intervals': self.d_intervals,
            'c_intervals': self.c_intervals,
            'plot_title_prefix': self.plot_title_prefix,
            'plot_theme': self.plot_theme
        }