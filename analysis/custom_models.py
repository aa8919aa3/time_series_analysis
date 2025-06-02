"""Custom model fitting using lmfit with Josephson junction physics."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import numpy.typing as npt
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

from .utils import calculate_r_squared, compute_residuals_integral
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import AnalysisConfig

logger = logging.getLogger(__name__)

# Try to import lmfit
try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    logger.warning("lmfit not available. Custom model fitting will be disabled.")
    lmfit = None
    LMFIT_AVAILABLE = False


def josephson_model_func(
    phi_ext: npt.NDArray[np.float64], 
    Ic: float, 
    T: float, 
    f: float, 
    d: float,
    phi0: float, 
    k: float, 
    r: float, 
    C: float
) -> npt.NDArray[np.float64]:
    """
    Standard Josephson junction current-phase relationship.
    
    Equation:
    I_s(Φ_ext) = I_c·sin(2πf(Φ_ext-d)-φ₀) / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) 
                 + k(Φ_ext-d)² + r(Φ_ext-d) + C
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External controlling parameter
    Ic : float
        Critical current (A)
    T : float
        Junction transparency (0-1)
    f : float
        Conversion factor (Φ_ext to phase scaling)
    d : float
        Horizontal shift (zero-point offset)
    phi0 : float
        Intrinsic phase offset (radians)
    k : float
        Quadratic background coefficient
    r : float
        Linear background coefficient
    C : float
        Overall current offset
        
    Returns:
    --------
    np.ndarray
        Josephson current values
    """
    # Calculate the phase
    phase = 2 * np.pi * f * (phi_ext - d) - phi0
    
    # Josephson term with transparency effects
    sin_phase = np.sin(phase)
    sin_half_phase = np.sin(phase / 2)
    
    # Avoid division by zero and ensure valid domain for sqrt
    denominator_arg = 1 - T * sin_half_phase**2
    denominator_arg = np.maximum(denominator_arg, 1e-12)  # Prevent negative values
    
    josephson_term = Ic * sin_phase / np.sqrt(denominator_arg)
    
    # Background terms
    displacement = phi_ext - d
    quadratic_term = k * displacement**2
    linear_term = r * displacement
    
    return josephson_term + quadratic_term + linear_term + C


def legacy_model_func(x: npt.NDArray[np.float64], A: float, f: float, d: float, 
                     p: float, T: float, r: float, C: float) -> npt.NDArray[np.float64]:
    """
    Legacy simplified model function for backward compatibility.
    
    Model: y = A * sin(2*pi*f*(x-d) - p) + r*(x-d) + C
    """
    term1 = 2 * np.pi * f * (x - d) - p
    periodic_part = A * np.sin(term1)
    linear_part = r * (x - d) + C
    return periodic_part + linear_part


class CustomModelResults:
    """Container for custom model fitting results."""
    
    def __init__(self):
        self.success: bool = False
        self.fit_result: Optional[Any] = None  # lmfit.ModelResult
        self.r_squared: float = np.nan
        self.residuals: Optional[npt.NDArray[np.float64]] = None
        self.residuals_integral: float = np.nan
        self.model_predictions: Optional[npt.NDArray[np.float64]] = None
        self.parameter_matrix: Optional[npt.NDArray[np.float64]] = None
        self.best_d_idx: int = 0
        self.best_c_idx: int = 0
        self.d_ranges: Optional[list] = None
        self.c_ranges: Optional[list] = None
        self.model_type: str = "josephson"  # or "legacy"


class CustomModelAnalyzer:
    """Custom model analyzer using lmfit with Josephson junction physics."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = CustomModelResults()
    
    def analyze(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]] = None,
        initial_frequency: Optional[float] = None,
        use_josephson_model: bool = True
    ) -> CustomModelResults:
        """
        Perform custom model fitting analysis.
        
        Args:
            times: Time array (external parameter Φ_ext)
            values: Value array (measured current I_s)
            errors: Error array (optional)
            initial_frequency: Initial frequency guess (optional)
            use_josephson_model: Whether to use Josephson model (True) or legacy model (False)
            
        Returns:
            CustomModelResults object
        """
        if not LMFIT_AVAILABLE:
            logger.error("lmfit not available. Cannot perform custom model fitting.")
            return self.results
        
        model_name = "Josephson junction" if use_josephson_model else "legacy"
        logger.info(f"Starting {model_name} model fitting")
        
        self.results.model_type = "josephson" if use_josephson_model else "legacy"
        
        # Prepare weights
        weights = self._prepare_weights(times, values, errors)
        
        # Estimate initial parameters
        if use_josephson_model:
            initial_params = self._estimate_josephson_parameters(times, values, initial_frequency)
        else:
            initial_params = self._estimate_legacy_parameters(times, values, initial_frequency)
        
        # Perform fitting
        self._fit_model(times, values, weights, initial_params, use_josephson_model)
        
        # Compute parameter matrix if requested (only for Josephson model)
        if self.results.success and use_josephson_model:
            self._compute_parameter_matrix(times, values)
        
        logger.info(f"{model_name} model fitting completed")
        return self.results
    
    def _prepare_weights(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64], 
        errors: Optional[npt.NDArray[np.float64]]
    ) -> Optional[npt.NDArray[np.float64]]:
        """Prepare weights for fitting."""
        
        if errors is None:
            logger.info("No errors provided. Performing unweighted fit.")
            return None
        
        if len(errors) != len(values):
            logger.warning("Error array length mismatch. Performing unweighted fit.")
            return None
        
        # Check for valid errors
        valid_errors = errors > 1e-9
        if np.all(valid_errors):
            logger.info(f"Using measurement errors as weights for all {len(values)} points.")
            return 1.0 / errors
        elif np.any(valid_errors):
            logger.warning(f"Some errors are invalid. Using weights for {np.sum(valid_errors)} points.")
            return None
        else:
            logger.warning("All errors are invalid. Performing unweighted fit.")
            return None
    
    def _estimate_josephson_parameters(
        self,
        phi_ext: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        initial_frequency: Optional[float]
    ) -> Dict[str, float]:
        """Estimate initial parameters for Josephson junction model."""
        
        # Basic statistics
        current_range = np.ptp(current)  # peak-to-peak
        current_mean = np.mean(current)
        phi_ext_range = np.ptp(phi_ext)
        phi_ext_min = np.min(phi_ext)
        
        # Linear trend estimation
        slope_init, intercept_init = np.polyfit(phi_ext, current, 1)
        
        # Estimate critical current (amplitude)
        # Remove linear trend to estimate oscillation amplitude
        detrended = current - (slope_init * phi_ext + intercept_init)
        Ic_init = np.std(detrended) * 2  # Rough amplitude estimate
        if Ic_init == 0:
            Ic_init = current_range / 4  # Fallback
        
        # Frequency estimation
        if initial_frequency is not None and initial_frequency > 0:
            f_init = initial_frequency
        else:
            # Estimate from data periodicity (very rough)
            f_init = 1.0 / phi_ext_range if phi_ext_range > 0 else 1.0
        
        return {
            'Ic': abs(Ic_init),
            'T': 0.3,  # Moderate transparency
            'f': f_init,
            'd': phi_ext_min,  # Start from minimum
            'phi0': 0.0,  # No initial phase offset
            'k': 0.0,  # No quadratic term initially
            'r': slope_init,  # Linear background
            'C': current_mean  # Mean current level
        }
    
    def _estimate_legacy_parameters(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        initial_frequency: Optional[float]
    ) -> Dict[str, float]:
        """Estimate initial parameters for legacy model."""
        
        # Linear trend estimation
        slope_init, intercept_init = np.polyfit(times, values, 1)
        
        # Amplitude estimation from detrended residuals
        residuals_for_amp_est = values - (slope_init * times + intercept_init)
        amp_init = np.std(residuals_for_amp_est) * np.sqrt(2)
        
        if amp_init == 0:
            amp_init = np.std(values) * np.sqrt(2)
        if amp_init == 0:
            amp_init = 1e-7
        
        # Frequency estimation
        freq_init = initial_frequency if initial_frequency and initial_frequency > 1e-9 else 1.0
        
        return {
            'A': max(amp_init, 1e-9),
            'f': freq_init,
            'd': times.min(),
            'p': 0.0,
            'T': 0.5,
            'r': slope_init,
            'C': np.polyval([slope_init, intercept_init], times.min())
        }
    
    def _fit_model(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        weights: Optional[npt.NDArray[np.float64]],
        initial_params: Dict[str, float],
        use_josephson_model: bool
    ) -> None:
        """Perform the actual model fitting."""
        
        # Create model
        if use_josephson_model:
            model = lmfit.Model(josephson_model_func)
            params = self._setup_josephson_parameters(initial_params)
        else:
            model = lmfit.Model(legacy_model_func)
            params = self._setup_legacy_parameters(initial_params)
        
        logger.info("Initial parameters:")
        for name, param in params.items():
            logger.info(f"  {name}: {param.value}")
        
        try:
            # Perform fit
            if use_josephson_model:
                self.results.fit_result = model.fit(
                    values, params, phi_ext=times, weights=weights, nan_policy='omit'
                )
            else:
                self.results.fit_result = model.fit(
                    values, params, x=times, weights=weights, nan_policy='omit'
                )
            
            # Compute metrics
            self.results.r_squared = calculate_r_squared(values, self.results.fit_result.best_fit)
            self.results.model_predictions = self.results.fit_result.best_fit
            self.results.residuals = values - self.results.fit_result.best_fit
            self.results.residuals_integral = compute_residuals_integral(times, self.results.residuals)
            self.results.success = True
            
            logger.info(f"Fit successful. R-squared: {self.results.r_squared:.4f}")
            logger.info(f"Residuals integral: {self.results.residuals_integral:.4e}")
            
            # Log final parameters
            logger.info("Final parameters:")
            for name, param in self.results.fit_result.params.items():
                if param.stderr is not None:
                    logger.info(f"  {name}: {param.value:.3e} ± {param.stderr:.3e}")
                else:
                    logger.info(f"  {name}: {param.value:.3e}")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            import traceback
            traceback.print_exc()
            self.results.success = False
    
    def _setup_josephson_parameters(self, initial_params: Dict[str, float]) -> 'lmfit.Parameters':
        """Setup parameters for Josephson model."""
        
        params = lmfit.Parameters()
        
        # Critical current (positive)
        params.add('Ic', value=initial_params['Ic'], min=1e-9)
        
        # Transparency (0 to 1, but not exactly 1 to avoid singularity)
        params.add('T', value=initial_params['T'], min=0.0, max=0.99)
        
        # Frequency scaling (positive)
        params.add('f', value=initial_params['f'], min=1e-9)
        
        # Horizontal shift (unconstrained)
        params.add('d', value=initial_params['d'])
        
        # Phase offset (constrained to meaningful range)
        params.add('phi0', value=initial_params['phi0'], min=-2*np.pi, max=2*np.pi)
        
        # Background coefficients
        params.add('k', value=initial_params['k'])  # Quadratic term
        params.add('r', value=initial_params['r'])  # Linear term
        params.add('C', value=initial_params['C'])  # Constant offset
        
        return params
    
    def _setup_legacy_parameters(self, initial_params: Dict[str, float]) -> 'lmfit.Parameters':
        """Setup parameters for legacy model."""
        
        params = lmfit.Parameters()
        
        params.add('A', value=initial_params['A'], min=1e-9)
        params.add('f', value=initial_params['f'], min=1e-9)
        params.add('d', value=initial_params['d'])
        params.add('p', value=initial_params['p'], min=-2*np.pi, max=2*np.pi)
        params.add('T', value=initial_params['T'], min=0.1, max=0.9)
        params.add('r', value=initial_params['r'])
        params.add('C', value=initial_params['C'])
        
        return params
    
    def _compute_parameter_matrix(
        self,
        times: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64]
    ) -> None:
        """Compute parameter boundaries matrix for Josephson model."""
        
        logger.info("Computing parameter boundaries matrix for Josephson model...")
        
        try:
            result = create_josephson_parameter_matrix(
                times, values,
                d_multiplier=self.config.d_multiplier,
                c_multiplier=self.config.c_multiplier,
                d_intervals=self.config.d_intervals,
                c_intervals=self.config.c_intervals
            )
            
            if result is not None:
                (self.results.d_ranges, self.results.c_ranges, 
                 self.results.parameter_matrix, _, 
                 self.results.best_d_idx, self.results.best_c_idx) = result
                logger.info("Parameter matrix computation completed")
            else:
                logger.warning("Parameter matrix computation failed")
                
        except Exception as e:
            logger.error(f"Parameter matrix computation failed: {e}")
    
    def create_smooth_model_curve(
        self,
        times: npt.NDArray[np.float64],
        extend_factor: float = 0.05,
        num_points: int = 1000
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Create smooth model curve for plotting."""
        
        if not self.results.success or self.results.fit_result is None:
            return times, np.zeros_like(times)
        
        # Create extended time range
        time_span = times.max() - times.min()
        if time_span > 0:
            t_min = times.min() - extend_factor * time_span
            t_max = times.max() + extend_factor * time_span
            t_smooth = np.linspace(t_min, t_max, max(num_points, 2 * len(times)))
        else:
            t_smooth = np.sort(np.unique(times))
        
        # Evaluate model
        if self.results.model_type == "josephson":
            model_smooth = self.results.fit_result.model.eval(
                params=self.results.fit_result.params, phi_ext=t_smooth
            )
        else:
            model_smooth = self.results.fit_result.model.eval(
                params=self.results.fit_result.params, x=t_smooth
            )
        
        return t_smooth, model_smooth


def fit_josephson_combination(
    args: Tuple[int, int, Tuple[float, float], Tuple[float, float]],
    phi_ext: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64]
) -> Tuple[int, int, float]:
    """Fit a single Josephson parameter combination for parallel processing."""
    
    if not LMFIT_AVAILABLE:
        return args[0], args[1], np.nan
    
    i, j, d_range, c_range = args
    
    try:
        # Use middle point of each range
        d_val = (d_range[0] + d_range[1]) / 2
        c_val = (c_range[0] + c_range[1]) / 2
        
        # Create model analyzer instance for parameter estimation
        analyzer = CustomModelAnalyzer(AnalysisConfig())  # Default config
        initial_params = analyzer._estimate_josephson_parameters(phi_ext, current, None)
        
        # Override d and C with constrained values
        initial_params['d'] = d_val
        initial_params['C'] = c_val
        
        # Create parameters with constraints
        params = lmfit.Parameters()
        params.add('Ic', value=initial_params['Ic'], min=1e-9)
        params.add('T', value=initial_params['T'], min=0.0, max=0.99)
        params.add('f', value=initial_params['f'], min=1e-9)
        params.add('d', value=d_val, min=d_range[0], max=d_range[1])
        params.add('phi0', value=initial_params['phi0'], min=-2*np.pi, max=2*np.pi)
        params.add('k', value=initial_params['k'])
        params.add('r', value=initial_params['r'])
        params.add('C', value=c_val, min=c_range[0], max=c_range[1])
        
        # Perform fit
        model = lmfit.Model(josephson_model_func)
        result = model.fit(current, params, phi_ext=phi_ext, nan_policy='omit')
        
        # Calculate R-squared
        r_squared = calculate_r_squared(current, result.best_fit)
        return i, j, r_squared
        
    except Exception:
        return i, j, np.nan


def create_josephson_parameter_matrix(
    phi_ext: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    d_multiplier: float = 100,
    c_multiplier: float = 10,
    d_intervals: int = 200,
    c_intervals: int = 20,
    n_jobs: Optional[int] = None
) -> Optional[Tuple]:
    """
    Create parameter boundaries matrix for Josephson model with parallel processing.
    
    Args:
        phi_ext: External parameter array
        current: Current measurement array
        d_multiplier: Multiplier for d parameter range
        c_multiplier: Multiplier for C parameter range
        d_intervals: Number of d intervals
        c_intervals: Number of C intervals
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (d_ranges, c_ranges, fit_results_matrix, best_params, best_d_idx, best_c_idx)
    """
    if not LMFIT_AVAILABLE:
        logger.error("lmfit not available for parameter matrix computation")
        return None
    
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, 8)
    
    mean_phi = np.mean(phi_ext)
    mean_current = np.mean(current)
    
    logger.info(f"Mean external parameter: {mean_phi:.6f}, Mean current: {mean_current:.6e}")
    
    # Create parameter ranges
    d_min, d_max = -d_multiplier * mean_phi, d_multiplier * mean_phi
    c_min, c_max = -c_multiplier * mean_current, c_multiplier * mean_current
    
    d_step = (d_max - d_min) / d_intervals
    c_step = (c_max - c_min) / c_intervals
    
    d_ranges = [(d_min + i*d_step, d_min + (i+1)*d_step) for i in range(d_intervals)]
    c_ranges = [(c_min + j*c_step, c_min + (j+1)*c_step) for j in range(c_intervals)]
    
    logger.info(f"d parameter range: [{d_min:.6f}, {d_max:.6f}] in {d_intervals} intervals")
    logger.info(f"C parameter range: [{c_min:.6e}, {c_max:.6e}] in {c_intervals} intervals")
    
    # Create parameter combinations
    combinations = [(i, j, d_ranges[i], c_ranges[j]) 
                   for i in range(d_intervals) 
                   for j in range(c_intervals)]
    
    total_fits = len(combinations)
    logger.info(f"Starting {total_fits} Josephson fits with {n_jobs} parallel jobs...")
    
    # Parallel fitting
    fit_func = partial(fit_josephson_combination, phi_ext=phi_ext, current=current)
    
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(fit_func, combinations))
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        return None
    
    # Reconstruct matrix
    fit_results_matrix = np.full((d_intervals, c_intervals), np.nan)
    best_r_squared = -np.inf
    best_d_idx, best_c_idx = 0, 0
    
    for i, j, r_squared in results:
        fit_results_matrix[i, j] = r_squared
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_d_idx, best_c_idx = i, j
    
    logger.info(f"Josephson parameter matrix completed. Best R²: {best_r_squared:.6f}")
    logger.info(f"Best d range: [{d_ranges[best_d_idx][0]:.6f}, {d_ranges[best_d_idx][1]:.6f}]")
    logger.info(f"Best C range: [{c_ranges[best_c_idx][0]:.6e}, {c_ranges[best_c_idx][1]:.6e}]")
    
    return d_ranges, c_ranges, fit_results_matrix, None, best_d_idx, best_c_idx