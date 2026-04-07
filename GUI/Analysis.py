"""Analysis module for MOT image processing.

Provides:
  - Optical density calculation.
  - 2D Gaussian fitting with physical validation and initial-parameter
    estimation.
  - 1D scatter-plot fit models (linear, exponential, Gaussian, temperature TOF)
    and the ``fit_scatter_data`` dispatcher used by the sequence controller.
"""

import numpy as np
from scipy.optimize import curve_fit


def optical_density(I_at: np.ndarray,
                    I_0: np.ndarray,
                    I_bg: np.ndarray = None,
                    epsilon: float = 1e-8) -> np.ndarray:
    """Computes the optical density (OD) image from absorption imaging data.

    Calculates OD = ln((I_0 - I_bg) / (I_at - I_bg)), with safeguards
    against division by zero, log of non-positive numbers, and background
    noise outside the laser beam region.

    Args:
        I_at: 2D array of the image with atoms present.
        I_0: 2D array of the reference image without atoms.
        I_bg: Optional 2D array of the background (dark frame). Defaults
            to zero if not provided.
        epsilon: Small positive value used to clip denominators and
            numerators before taking the logarithm, preventing NaN/Inf.

    Returns:
        2D numpy array of optical density values, clipped to [-1, 5] and
        with low-light regions zeroed out.
    """
    Ia = np.asarray(I_at, dtype=np.float64)
    I0 = np.asarray(I_0, dtype=np.float64)

    if I_bg is None:
        Ib = 0.0
    else:
        Ib = np.asarray(I_bg, dtype=np.float64)

    num = I0 - Ib
    den = Ia - Ib

    num_clean = np.clip(num, epsilon, None)
    den_clean = np.clip(den, epsilon, None)

    OD = np.log(num_clean / den_clean)

    # Cap physical OD values to avoid background noise dominating the
    # colormap boundaries.
    OD = np.clip(OD, -1.0, 5.0)

    # Mask out areas with extremely low light (outside the laser beam).
    # Threshold: 2% of the peak reference intensity (99th percentile to
    # avoid hot pixels).
    valid_light_threshold = np.percentile(num, 99) * 0.02
    OD[num < valid_light_threshold] = 0.0

    return OD


def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, A, B):
    """Evaluates a 2D Gaussian function on a meshgrid.

    Computes:
        I(x,y) = A * exp(-((x-x0)^2/(2*sigma_x^2) +
                            (y-y0)^2/(2*sigma_y^2))) + B

    Args:
        xy: Tuple of (X, Y) meshgrid arrays, each flattened to 1D.
        x0: Center position along the x (column) axis.
        y0: Center position along the y (row) axis.
        sigma_x: Standard deviation along the x axis (pixels).
        sigma_y: Standard deviation along the y axis (pixels).
        A: Peak amplitude above the baseline.
        B: Constant baseline offset.

    Returns:
        1D numpy array of Gaussian values evaluated at each (X, Y) point.
    """
    X, Y = xy
    return A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) +
                        (Y - y0)**2 / (2 * sigma_y**2))) + B


def estimate_gaussian_params(image):
    """Estimates initial 2D Gaussian parameters from an image.

    Uses the peak pixel location for the center and the background median
    for the baseline. Sigma values default to 5 pixels as a safe starting
    point for curve_fit.

    Args:
        image: 2D numpy array to estimate parameters from.

    Returns:
        List of [x0, y0, sigma_x, sigma_y, A, B] suitable as initial
        guess (p0) for curve_fit with gaussian_2d.
    """
    background = np.median(image)
    data_centered = image - background

    y0, x0 = np.unravel_index(np.argmax(data_centered), image.shape)
    A = image[y0, x0] - background

    sigma_x = 5.0
    sigma_y = 5.0

    return [x0, y0, sigma_x, sigma_y, A, background]


def validate_fit(popt, image_shape, sigma_max_fraction=0.5):
    """Validates that fitted Gaussian parameters are physically reasonable.

    Rejects fits where:
        - Sigma is larger than a fraction of the image dimension.
        - Sigma is less than 1 pixel (sub-pixel = noise).
        - Amplitude is negative or zero (fitting a dip, not a peak).
        - Center is outside the image bounds.

    Args:
        popt: Fitted parameters [x0, y0, sigma_x, sigma_y, A, B].
        image_shape: Tuple (rows, cols) of the image that was fitted.
            When a ROI is used, this should be the ROI dimensions.
        sigma_max_fraction: Maximum allowed sigma as a fraction of the
            corresponding image dimension. Defaults to 0.5 (50%).

    Raises:
        ValueError: If any validation check fails, with a descriptive
            message indicating which parameter was out of bounds.
    """
    x0, y0, sigma_x, sigma_y, A, B = popt
    rows, cols = image_shape

    sigma_x_abs = abs(sigma_x)
    sigma_y_abs = abs(sigma_y)

    max_sigma_x = cols * sigma_max_fraction
    max_sigma_y = rows * sigma_max_fraction

    if sigma_x_abs > max_sigma_x:
        raise ValueError(
            f"Fit rejected: sigma_x={sigma_x_abs:.1f} px exceeds "
            f"{sigma_max_fraction*100:.0f}% of image width ({cols} px)."
        )
    if sigma_y_abs > max_sigma_y:
        raise ValueError(
            f"Fit rejected: sigma_y={sigma_y_abs:.1f} px exceeds "
            f"{sigma_max_fraction*100:.0f}% of image height ({rows} px)."
        )
    if sigma_x_abs < 1.0:
        raise ValueError(
            f"Fit rejected: sigma_x={sigma_x_abs:.2f} px is sub-pixel "
            f"(likely noise)."
        )
    if sigma_y_abs < 1.0:
        raise ValueError(
            f"Fit rejected: sigma_y={sigma_y_abs:.2f} px is sub-pixel "
            f"(likely noise)."
        )
    if A <= 0:
        raise ValueError(
            f"Fit rejected: amplitude A={A:.4f} is non-positive "
            f"(fitting a dip, not a peak)."
        )
    if not (0 <= x0 <= cols):
        raise ValueError(
            f"Fit rejected: center x0={x0:.1f} is outside image "
            f"bounds [0, {cols}]."
        )
    if not (0 <= y0 <= rows):
        raise ValueError(
            f"Fit rejected: center y0={y0:.1f} is outside image "
            f"bounds [0, {rows}]."
        )


def fit_function(image, func, p0=None):
    """Fits a 2D function to an image and validates the result.

    Creates a meshgrid matching the image dimensions, performs a
    least-squares curve fit, validates the fitted parameters against
    physical constraints, and returns the fitted image.

    Args:
        image: 2D numpy array to fit.
        func: Callable model function with signature
            func((X, Y), *params) -> Z.
        p0: Optional list of initial parameter guesses. If None,
            parameters are estimated automatically from the image
            using estimate_gaussian_params().

    Returns:
        Tuple of (fitted_image, popt) where:
            - fitted_image: 2D numpy array of the fitted model evaluated
              on the image grid.
            - popt: 1D array of optimal parameters from curve_fit.

    Raises:
        ValueError: If the fit result is physically unreasonable (e.g.,
            sigma too large, negative amplitude, center off-image).
        RuntimeError: If curve_fit fails to converge.
    """
    if p0 is None:
        p0 = estimate_gaussian_params(image)

    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    xy = (X.ravel(), Y.ravel())
    z = image.ravel()

    # Set bounds to help the optimizer stay in a reasonable region.
    # Parameters: [x0, y0, sigma_x, sigma_y, A, B]
    lower = [0, 0, 1.0, 1.0, 0, -np.inf]
    upper = [cols, rows, cols, rows, np.inf, np.inf]

    popt, _ = curve_fit(func, xy, z, p0, bounds=(lower, upper),
                        maxfev=10000)

    # Post-fit validation: reject physically impossible results.
    validate_fit(popt, image.shape)

    fitted_flat = func(xy, *popt)
    fitted_image = fitted_flat.reshape(rows, cols)

    return fitted_image, popt


# =============================================================================
# 1-D scatter-plot fit models
# =============================================================================

def fit_linear(x, a, b):
    """Linear model: y = a·x + b."""
    return a * x + b


def fit_exponential(x, A, tau, C):
    """Exponential decay model: y = A·exp(−x/τ) + C."""
    return A * np.exp(-x / tau) + C


def fit_gaussian_1d(x, A, x0, sigma, B):
    """1-D Gaussian model: y = A·exp(−(x−x₀)²/(2σ²)) + B."""
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + B


def fit_temperature_tof(t, sigma0_sq, kT_over_m):
    """TOF expansion: σ²(t) = σ₀² + (kT/m)·t²."""
    return sigma0_sq + kT_over_m * t ** 2
