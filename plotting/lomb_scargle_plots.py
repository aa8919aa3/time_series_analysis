"""Plotting functions for Lomb-Scargle analysis results."""

import plotly.graph_objects as go
import numpy as np
from typing import Optional
import numpy.typing as npt
import logging

from ..config import AnalysisConfig
from ..analysis.lomb_scargle import LombScargleResults
from .base import setup_plot_layout, add_data_trace

logger = logging.getLogger(__name__)


class LombScarglePlotter:
    """Plotting functions for Lomb-Scargle analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def plot_periodogram(self, results: LombScargleResults) -> None:
        """Plot the Lomb-Scargle periodogram."""
        
        fig = go.Figure()
        
        if results.frequency is not None and len(results.frequency) > 0:
            # Main periodogram
            fig.add_trace(go.Scatter(
                x=results.frequency, y=results.power,
                mode='lines', name='LS Power',
                line=dict(color='cornflowerblue')
            ))
            
            # Best frequency line
            fig.add_vline(
                x=results.best_frequency,
                line_dash="dash", line_color="red",
                annotation_text=f'Best Freq: {results.best_frequency:.4f}',
                annotation_position="top right"
            )
            
            # FAP thresholds
            if results.fap_levels and results.power_thresholds is not None:
                for level, thresh in zip(results.fap_levels, results.power_thresholds):
                    fig.add_hline(
                        y=thresh,
                        line_dash="dot", line_color="grey",
                        annotation_text=f'FAP {level*100:.1f}% ({thresh:.2f})',
                        annotation_position="bottom right"
                    )
        else:
            fig.add_annotation(
                text="Periodogram not computed.",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False
            )
        
        setup_plot_layout(
            fig, "Lomb-Scargle Periodogram",
            "Frequency (cycles / time unit)", "Lomb-Scargle Power",
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.show()
    
    def plot_data_with_fit(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]],
        t_fit: npt.NDArray[np.float64],
        y_fit: npt.NDArray[np.float64],
        results: LombScargleResults
    ) -> None:
        """Plot original data with Lomb-Scargle fit."""
        
        fig = go.Figure()
        
        # Original data
        add_data_trace(
            fig, times, values, errors,
            name="Original Data",
            marker=dict(color='grey', size=3, opacity=0.7)
        )
        
        # Lomb-Scargle fit
        ls_label = 'Lomb-Scargle Fit'
        if not np.isnan(results.r_squared):
            ls_label += f' (R²={results.r_squared:.4f})'
        
        fig.add_trace(go.Scatter(
            x=t_fit, y=y_fit,
            mode='lines', name=ls_label,
            line=dict(color='dodgerblue', width=1.5)
        ))
        
        title = f"Data with Lomb-Scargle Sinusoidal Fit (Period = {results.best_period:.4f})"
        setup_plot_layout(
            fig, title, "Time", self.config.value_column.capitalize(),
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.show()
    
    def plot_phase_folded_data(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]],
        results: LombScargleResults
    ) -> None:
        """Plot phase-folded data."""
        
        fig = go.Figure()
        
        if (results.best_period != 0 and not np.isinf(results.best_period) and 
            not np.isnan(results.best_period) and len(times) > 0):
            
            # Calculate phases
            phase = (times / results.best_period) % 1.0
            
            # Plot data points in three cycles for continuity
            for cycle_offset in [-1, 0, 1]:
                phase_shifted = phase + cycle_offset
                
                add_data_trace(
                    fig, phase_shifted, values, errors,
                    name='Data points' if cycle_offset == 0 else f'Cycle {cycle_offset}',
                    mode='markers',
                    marker=dict(color='grey', size=3, opacity=0.5),
                    showlegend=(cycle_offset == 0)
                )
            
            # Binned profile
            num_bins = 20
            phase_bins = np.linspace(0, 1, num_bins + 1)
            bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
            
            # Calculate binned means
            binned_values = []
            for i in range(num_bins):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
                if np.any(mask):
                    binned_values.append(np.mean(values[mask]))
                else:
                    binned_values.append(np.nan)
            
            # Plot binned profile
            valid_bins = ~np.isnan(binned_values)
            if np.any(valid_bins):
                # Plot in two cycles
                extended_centers = np.concatenate([bin_centers, bin_centers + 1])
                extended_values = np.concatenate([binned_values, binned_values])
                
                fig.add_trace(go.Scatter(
                    x=extended_centers, y=extended_values,
                    mode='lines', name='Mean Phase-Folded Profile',
                    line=dict(color='blue', width=1.5, dash='dash')
                ))
            
            fig.update_xaxes(
                title_text=f"Phase (Period = {results.best_period:.6f} time units)",
                range=[0, 2]
            )
            y_label = f"{self.config.value_column.capitalize()}"
            if results.detrend_coeffs is not None:
                y_label += " (Detrended for LS)"
            fig.update_yaxes(title_text=y_label)
            
        else:
            fig.add_annotation(
                text="Phase-folded plot skipped<br>due to invalid period or no data.",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False
            )
        
        setup_plot_layout(
            fig, "Phase-Folded Data", "", "",
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.show()
    
    def plot_parameters_text(self, results: LombScargleResults) -> None:
        """Display Lomb-Scargle fit parameters as text."""
        
        fig = go.Figure()
        
        param_text = (
            f"<b>Lomb-Scargle Fit Parameters:</b><br><br>"
            f"  Best Frequency: {results.best_frequency:.4f}<br>"
            f"  Best Period: {results.best_period:.6f}<br>"
            f"  R-squared: {results.r_squared:.4f}<br>"
            f"  Amplitude (sinusoid): {results.amplitude:.3e}<br>"
            f"  Phase (sinusoid, radians): {results.phase_rad:.3f}<br>"
            f"  Offset (mean term): {results.offset:.3e}<br><br>"
            f"Note: Detecting quantitative phase shift<br>"
            f"over time requires more advanced analysis."
        )
        
        fig.add_annotation(
            text=param_text,
            xref="paper", yref="paper",
            x=0.05, y=0.95, showarrow=False, align="left",
            bgcolor="wheat", opacity=0.5, bordercolor="black", borderwidth=1,
            font=dict(size=12)
        )
        
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        setup_plot_layout(
            fig, "Lomb-Scargle Fit Parameters", "", "",
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.show()