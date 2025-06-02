"""Plotting utility functions."""

import plotly.graph_objects as go
import numpy as np
from typing import Optional
import numpy.typing as npt
import logging

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
    color: str = "blue",
    show_markers: bool = True,
    show_lines: bool = False
) -> go.Figure:
    """
    Add data trace to figure with optional error bars.
    
    Args:
        fig: Plotly figure object
        x: X coordinates
        y: Y coordinates
        errors: Optional error values
        name: Trace name
        color: Trace color
        show_markers: Whether to show markers
        show_lines: Whether to show lines
        
    Returns:
        Updated figure
    """
    mode = []
    if show_markers:
        mode.append("markers")
    if show_lines:
        mode.append("lines")
    mode_str = "+".join(mode) if mode else "markers"
    
    if errors is not None:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type='data',
                array=errors,
                visible=True
            ),
            mode=mode_str,
            name=name,
            marker=dict(color=color)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode=mode_str,
            name=name,
            marker=dict(color=color)
        ))
    
    return fig


def add_model_trace(
    fig: go.Figure, 
    x: npt.NDArray[np.float64], 
    y: npt.NDArray[np.float64], 
    name: str = "Model", 
    color: str = "red",
    line_width: int = 2
) -> go.Figure:
    """
    Add model trace to figure.
    
    Args:
        fig: Plotly figure object
        x: X coordinates
        y: Y coordinates
        name: Trace name
        color: Line color
        line_width: Line width
        
    Returns:
        Updated figure
    """
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=name,
        line=dict(color=color, width=line_width)
    ))
    
    return fig
