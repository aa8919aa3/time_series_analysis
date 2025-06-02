"""Base plotting utilities and manager."""

import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import Optional, List, Union
import numpy.typing as npt
import logging

from ..config import AnalysisConfig
from ..analysis import LombScargleAnalyzer, CustomModelAnalyzer
from .lomb_scargle_plots import LombScarglePlotter
from .custom_model_plots import CustomModelPlotter

logger = logging.getLogger(__name__)


def setup_plot_layout(
    fig: go.Figure, 
    title: str, 
    x_label: str, 
    y_label: str, 
    title_prefix: str = "",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Standard plot layout configuration.
    
    Args:
        fig: Plotly figure object
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label  
        title_prefix: Prefix for title
        theme: Plotly theme
        
    Returns:
        Configured figure
    """
    full_title = f"{title_prefix}: {title}" if title_prefix else title
    fig.update_layout(
        title_text=full_title,
        title_x=0.5,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
        template=theme
    )
    return fig


def add_data_trace(
    fig: go.Figure,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    errors: Optional[npt.NDArray[np.float64]] = None,
    name: str = "Data",
    mode: str = "lines+markers",
    **kwargs
) -> go.Figure:
    """
    Add data trace with optional error bars.
    
    Args:
        fig: Plotly figure
        x: X data
        y: Y data
        errors: Error bars (optional)
        name: Trace name
        mode: Plot mode
        **kwargs: Additional scatter plot arguments
        
    Returns:
        Figure with added trace
    """
    scatter_args = {
        'x': x, 
        'y': y, 
        'mode': mode, 
        'name': name,
        **kwargs
    }
    
    if errors is not None:
        scatter_args['error_y'] = dict(
            type='data', 
            array=errors, 
            visible=True,
            color=kwargs.get('marker', {}).get('color', 'grey'),
            thickness=0.5
        )
    
    fig.add_trace(go.Scatter(**scatter_args))
    return fig


def create_residuals_histogram(
    residuals_list: List[npt.NDArray[np.float64]],
    labels_list: List[str],
    title: str = "Histogram of Residuals",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create histogram of residuals using Plotly figure factory.
    
    Args:
        residuals_list: List of residual arrays
        labels_list: List of labels for each residual array
        title: Plot title
        theme: Plotly theme
        
    Returns:
        Plotly figure
    """
    # Filter valid data
    valid_data = []
    valid_labels = []
    
    for residuals, label in zip(residuals_list, labels_list):
        res_array = np.array(residuals)
        res_valid = res_array[~np.isnan(res_array)]
        
        if len(res_valid) > 0:
            valid_data.append(res_valid)
            valid_labels.append(label)
        else:
            logger.warning(f"No valid data for label '{label}', skipping")
    
    if not valid_data:
        logger.error("No valid data for histogram")
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return setup_plot_layout(fig, title, "Residual Value", "Density", theme=theme)
    
    # Determine bin size
    all_data = np.concatenate(valid_data)
    bin_edges = np.histogram_bin_edges(all_data, bins='rice')
    bin_size = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else None
    
    # Create distribution plot
    try:
        fig = ff.create_distplot(
            valid_data, valid_labels,
            show_hist=True, show_rug=True,
            bin_size=bin_size
        )
        fig.update_layout(template=theme)
    except Exception as e:
        logger.error(f"Failed to create distribution plot: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Plot creation failed: {e}",
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False
        )
    
    return setup_plot_layout(fig, title, "Residual Value", "Density", theme=theme)


class PlottingManager:
    """Main plotting manager that coordinates all plotting functions."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.ls_plotter = LombScarglePlotter(config)
        self.custom_plotter = CustomModelPlotter(config)
    
    def plot_lomb_scargle_results(
        self,
        analyzer: LombScargleAnalyzer,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]] = None
    ) -> None:
        """Plot all Lomb-Scargle analysis results."""
        
        results = analyzer.results
        
        # Periodogram
        if results.frequency is not None:
            self.ls_plotter.plot_periodogram(results)
        
        # Data with fit
        if results.model_predictions is not None:
            t_smooth, y_smooth = analyzer.create_smooth_model_curve(times)
            self.ls_plotter.plot_data_with_fit(
                times, values, errors, t_smooth, y_smooth, results
            )
        
        # Phase-folded data
        if results.best_period != float('inf'):
            self.ls_plotter.plot_phase_folded_data(times, values, errors, results)
        
        # Parameters text
        self.ls_plotter.plot_parameters_text(results)
    
    def plot_custom_model_results(
        self,
        analyzer: CustomModelAnalyzer,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]] = None,
        ls_analyzer: Optional[LombScargleAnalyzer] = None
    ) -> None:
        """Plot all custom model analysis results."""
        
        results = analyzer.results
        
        if not results.success:
            logger.warning("Custom model fitting was not successful, skipping plots")
            return
        
        # Model fit comparison
        t_smooth, y_smooth = analyzer.create_smooth_model_curve(times)
        
        # Include Lomb-Scargle comparison if available
        ls_t_smooth, ls_y_smooth = None, None
        if ls_analyzer and ls_analyzer.results.model_predictions is not None:
            ls_t_smooth, ls_y_smooth = ls_analyzer.create_smooth_model_curve(times)
        
        self.custom_plotter.plot_model_fit(
            times, values, errors, t_smooth, y_smooth, results,
            ls_t_smooth, ls_y_smooth, ls_analyzer.results if ls_analyzer else None
        )
        
        # Parameter correlation heatmap
        if results.fit_result is not None:
            self.custom_plotter.plot_correlation_heatmap(results.fit_result)
        
        # Parameter matrix heatmap
        if results.parameter_matrix is not None:
            self.custom_plotter.plot_parameter_matrix_heatmap(results)
    
    def plot_residuals_comparison(
        self,
        times: npt.NDArray[np.float64],
        ls_analyzer: Optional[LombScargleAnalyzer] = None,
        custom_analyzer: Optional[CustomModelAnalyzer] = None
    ) -> None:
        """Plot residuals comparison between models."""
        
        residuals_list = []
        labels_list = []
        
        # Collect residuals
        if ls_analyzer and ls_analyzer.results.residuals is not None:
            residuals_list.append(ls_analyzer.results.residuals)
            integral = ls_analyzer.results.residuals_integral
            label = f"Lomb-Scargle (∫={integral:.2e})" if not np.isnan(integral) else "Lomb-Scargle"
            labels_list.append(label)
        
        if (custom_analyzer and custom_analyzer.results.success and 
            custom_analyzer.results.residuals is not None):
            residuals_list.append(custom_analyzer.results.residuals)
            integral = custom_analyzer.results.residuals_integral
            label = f"Custom Model (∫={integral:.2e})" if not np.isnan(integral) else "Custom Model"
            labels_list.append(label)
        
        if not residuals_list:
            logger.warning("No residuals available for comparison")
            return
        
        # Plot residuals vs time
        self._plot_residuals_vs_time(times, residuals_list, labels_list)
        
        # Plot residuals histogram
        fig = create_residuals_histogram(
            residuals_list, labels_list,
            title="Comparison of Residuals Histograms",
            theme=self.config.plot_theme
        )
        fig.show()
    
    def _plot_residuals_vs_time(
        self,
        times: npt.NDArray[np.float64],
        residuals_list: List[npt.NDArray[np.float64]],
        labels_list: List[str]
    ) -> None:
        """Plot residuals vs time for comparison."""
        
        fig = go.Figure()
        
        colors = ['dodgerblue', 'orangered', 'green', 'purple']
        symbols = ['circle', 'x', 'diamond', 'square']
        
        for i, (residuals, label) in enumerate(zip(residuals_list, labels_list)):
            color = colors[i % len(colors)]
            symbol = symbols[i % len(symbols)]
            
            fig.add_trace(go.Scatter(
                x=times, y=residuals,
                mode='lines+markers',
                name=label,
                marker=dict(size=3, opacity=0.7, color=color, symbol=symbol),
                line=dict(color=color)
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
        
        setup_plot_layout(
            fig, "Comparison of Residuals vs. Time", "Time", "Residual (Data - Model)",
            title_prefix=self.config.plot_title_prefix, theme=self.config.plot_theme
        )
        
        fig.show()