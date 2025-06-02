"""Plotting functions for custom model analysis results."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional
import numpy.typing as npt
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import AnalysisConfig
from analysis.custom_models import CustomModelResults
from analysis.lomb_scargle import LombScargleResults
from .utils import setup_plot_layout, add_data_trace, add_model_trace

logger = logging.getLogger(__name__)

# Try to import lmfit for type hints
try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    lmfit = None
    LMFIT_AVAILABLE = False


class CustomModelPlotter:
    """Plotting functions for custom model analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def plot_model_fit(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]],
        t_fit: npt.NDArray[np.float64],
        y_fit: npt.NDArray[np.float64],
        results: CustomModelResults,
        ls_t_fit: Optional[npt.NDArray[np.float64]] = None,
        ls_y_fit: Optional[npt.NDArray[np.float64]] = None,
        ls_results: Optional[LombScargleResults] = None
    ) -> None:
        """Plot original data with custom model fit and optional Lomb-Scargle comparison."""
        
        fig = go.Figure()
        
        # Original data
        add_data_trace(
            fig, times, values, errors,
            name="Original Data",
            marker=dict(color='grey', size=3, opacity=0.7)
        )
        
        # Lomb-Scargle fit (if provided)
        if ls_t_fit is not None and ls_y_fit is not None and ls_results is not None:
            ls_label = 'Lomb-Scargle Fit'
            if not np.isnan(ls_results.r_squared):
                ls_label += f' (R²={ls_results.r_squared:.4f})'
            
            fig.add_trace(go.Scatter(
                x=ls_t_fit, y=ls_y_fit,
                mode='lines', name=ls_label,
                line=dict(color='lightcoral', width=1.5, dash='dash')
            ))
            
            # Add LS parameters text box
            ls_param_text = (
                f"<b>Lomb-Scargle Parameters:</b><br>"
                f"Frequency: {ls_results.best_frequency:.4f}<br>"
                f"Period: {ls_results.best_period:.6f}<br>"
                f"R²: {ls_results.r_squared:.4f}<br>"
                f"Amplitude: {ls_results.amplitude:.3e}<br>"
                f"Phase: {ls_results.phase_rad:.3f}<br>"
                f"Offset: {ls_results.offset:.3e}"
            )
            
            fig.add_annotation(
                text=ls_param_text,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                x=1.02, y=0.75, showarrow=False, align="left",
                bgcolor="wheat", opacity=0.5, bordercolor="black", borderwidth=1,
                font=dict(size=10)
            )
        
        # Custom model fit
        custom_label = 'Josephson Junction Model'
        if not np.isnan(results.r_squared):
            custom_label += f' (R²={results.r_squared:.4f})'
        
        fig.add_trace(go.Scatter(
            x=t_fit, y=y_fit,
            mode='lines', name=custom_label,
            line=dict(color='orangered', width=1.5)
        ))
        
        # Add custom model parameters if available
        if results.fit_result is not None:
            custom_param_text = self._format_fit_parameters(results.fit_result)
            fig.add_annotation(
                text=custom_param_text,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                x=1.02, y=0.35, showarrow=False, align="left",
                bgcolor="lightblue", opacity=0.5, bordercolor="black", borderwidth=1,
                font=dict(size=10)
            )
        
        title = "Josephson Junction Model Fit"
        setup_plot_layout(
            fig, title, "External Parameter (Φ_ext)", "Current (I_s)",
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.show()
    
    def plot_correlation_heatmap(self, fit_result) -> None:
        """Plot parameter correlation matrix heatmap."""
        
        if not LMFIT_AVAILABLE:
            logger.warning("lmfit not available, skipping correlation heatmap")
            return
        
        if fit_result is None or not hasattr(fit_result, 'params'):
            logger.warning("No fit result available for correlation heatmap")
            return
        
        # Get varying parameters
        vary_params = [name for name, param in fit_result.params.items() if param.vary]
        
        if not vary_params:
            logger.warning("No varying parameters found, skipping correlation heatmap")
            return
        
        # Build correlation matrix
        n_params = len(vary_params)
        correl_matrix = np.zeros((n_params, n_params))
        
        for i, p1_name in enumerate(vary_params):
            correl_matrix[i, i] = 1.0
            for j, p2_name in enumerate(vary_params[i+1:], start=i+1):
                correlation = 0.0
                if (fit_result.params[p1_name].correl is not None and 
                    p2_name in fit_result.params[p1_name].correl):
                    correlation = fit_result.params[p1_name].correl[p2_name]
                
                correl_matrix[i, j] = correlation
                correl_matrix[j, i] = correlation
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correl_matrix,
            x=vary_params,
            y=vary_params,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation'),
            texttemplate="%{z:.2f}",
            textfont={"size": 10}
        ))
        
        setup_plot_layout(
            fig, "Parameter Correlation Heatmap (Josephson Model)",
            "Parameter", "Parameter",
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.show()
    
    def plot_parameter_matrix_heatmap(self, results: CustomModelResults) -> None:
        """Plot parameter boundaries matrix heatmap."""
        
        if results.parameter_matrix is None:
            logger.warning("No parameter matrix available for plotting")
            return
        
        # Create center values for display
        d_centers = [(d_range[0] + d_range[1]) / 2 for d_range in results.d_ranges]
        c_centers = [(c_range[0] + c_range[1]) / 2 for c_range in results.c_ranges]
        
        fig = go.Figure(data=go.Heatmap(
            z=results.parameter_matrix,
            x=c_centers,
            y=d_centers,
            colorscale='Viridis',
            colorbar=dict(title='R-squared'),
            hoverongaps=False,
            hovertemplate='d: %{y:.6f}<br>C: %{x:.6e}<br>R²: %{z:.6f}<extra></extra>'
        ))
        
        # Mark the best combination
        if (results.best_d_idx < len(d_centers) and 
            results.best_c_idx < len(c_centers)):
            best_r_squared = results.parameter_matrix[results.best_d_idx, results.best_c_idx]
            
            fig.add_scatter(
                x=[c_centers[results.best_c_idx]], 
                y=[d_centers[results.best_d_idx]], 
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name=f'Best Fit (R²={best_r_squared:.6f})',
                showlegend=True
            )
        
        setup_plot_layout(
            fig, "Parameter Boundaries Matrix: R-squared vs d and C parameters",
            "C parameter range centers", "d parameter range centers",
            title_prefix=self.config.plot_title_prefix,
            theme=self.config.plot_theme
        )
        
        fig.update_layout(width=800, height=600)
        fig.show()
    
    def _format_fit_parameters(self, fit_result) -> str:
        """Format fit parameters for display."""
        
        if not LMFIT_AVAILABLE or fit_result is None:
            return "<b>Custom Model Parameters:</b><br>Not available"
        
        param_lines = ["<b>Josephson Model Parameters:</b><br>"]
        
        param_descriptions = {
            'Ic': 'Critical Current',
            'T': 'Transparency', 
            'f': 'Frequency Scaling',
            'd': 'Horizontal Shift',
            'phi0': 'Phase Offset',
            'k': 'Quadratic Coeff',
            'r': 'Linear Coeff',
            'C': 'Current Offset'
        }
        
        for name, param in fit_result.params.items():
            desc = param_descriptions.get(name, name)
            if param.stderr is not None:
                param_lines.append(f"{desc}: {param.value:.3e} ± {param.stderr:.3e}")
            else:
                param_lines.append(f"{desc}: {param.value:.3e}")
        
        return "<br>".join(param_lines)