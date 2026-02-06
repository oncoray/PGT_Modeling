from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


# =============================================================================
# Helpers / numerics
# =============================================================================

def yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
    """
    Numerically stable Yeo–Johnson transform, consistent in form with sklearn.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    lam : float
        Lambda parameter.

    Returns
    -------
    np.ndarray
        Transformed array.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    pos = x >= 0
    neg = ~pos

    if abs(lam) > 1e-12:
        out[pos] = np.expm1(lam * np.log1p(x[pos])) / lam
    else:
        out[pos] = np.log1p(x[pos])

    if abs(lam - 2.0) > 1e-12:
        a = 2.0 - lam
        out[neg] = -np.expm1(a * np.log1p(-x[neg])) / a
    else:
        out[neg] = -np.log1p(-x[neg])

    return out


def ensure_2d_energy_time(arr: np.ndarray) -> np.ndarray:
    """
    Ensure spectrum is 2D (energy,time).
    Accepts:
      - 2D: returned unchanged
      - 1D: promoted to shape (1,time)
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    raise ValueError(f"Unsupported shape {arr.shape}; expected 1D or 2D.")


def smoothed_hist_10th_percentile(
    spectrum_energy_time: np.ndarray,
    *,
    smoothing_sigma_bins: float,
    smoothed_repeat_scale: int,
    time_window_bins: Optional[List[int]] = None,
    strict_nonnegative_integer: bool = True,
) -> float:
    """
    Compute a percentile on a Gaussian-smoothed 1D time histogram derived from an
    (energy,time) spectrum using the pseudo-count repetition strategy.

    Steps:
      - optionally slice time axis with inclusive [lo, hi]
      - sum over energy -> y_hist(time)
      - smooth y_hist with gaussian_filter1d(sigma)
      - repeats = round(y_smooth * scale)
      - expand histogram by repeating bin indices and compute percentile

    Notes:
      - If strict_nonnegative_integer=True, enforce non-negative integer spectrum.
      - For numerical safety, if repeats contains negatives (unexpected for non-negative input),
        they are clipped to 0. This should not occur in normal operation.

    Returns
    -------
    float
        Percentile in bin-index units.
    """
    spec = ensure_2d_energy_time(spectrum_energy_time)

    if strict_nonnegative_integer:
        if np.any(spec < 0):
            raise ValueError("Spectrum contains negative values.")
        if not np.all(np.mod(spec, 1) == 0):
            raise ValueError("Spectrum contains non-integer values.")

    if np.sum(spec) <= 0:
        spec = np.ones_like(spec)

    if time_window_bins is not None:
        lo, hi = int(time_window_bins[0]), int(time_window_bins[1])
        spec = spec[:, lo : hi + 1]

    y_hist = np.sum(spec, axis=0).astype(float)
    n_bins = int(y_hist.shape[0])
    x = np.arange(n_bins, dtype=int)

    y_s = gaussian_filter1d(y_hist, sigma=float(smoothing_sigma_bins))
    repeats = np.round(y_s * float(smoothed_repeat_scale)).astype(int)

    if np.any(repeats < 0):
        repeats = np.maximum(repeats, 0)

    if repeats.sum() <= 0:
        repeats = np.ones_like(repeats)

    hist_all = np.repeat(x, repeats)
    return float(np.percentile(hist_all, 100.0 * float(10)))

def extract_falling_edge_gauss(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate the falling edge (half-maximum point) via Gaussian fit on the falling side.

    Procedure:
      1) Identify peak.
      2) Restrict to x > peak.
      3) Fit Gaussian to restricted region.
      4) Falling edge (half max) is: mu + sigma * sqrt(2 * ln(2)).

    Behaviour preserved:
      - If fit fails: return 2048.
      - If computed edge is outside [0, 2048]: return 2048.
      - If no falling-side data exist: fit is attempted on full spectrum.
    """
    peak_idx = int(np.argmax(y))
    mask = x > x[peak_idx]

    if not np.any(mask):
        x_fall = x
        y_fall = y
    else:
        x_fall = x[mask]
        y_fall = y[mask]

    A0 = float(y[peak_idx])
    mu0 = float(x[peak_idx])
    sigma0 = float((x_fall[-1] - mu0) / 2.0) if len(x_fall) > 1 else 1.0

    try:
        # Initial guess with strict feasibility inside bounds
        p0 = np.array([A0, mu0, sigma0], dtype=float)
        lb = np.array([0.0, 0.0, 0.0], dtype=float)
        ub = np.array([2.0 * A0, float(np.max(x)), float(np.max(x))], dtype=float)

        eps = np.finfo(float).eps
        p0 = np.minimum(np.maximum(p0, lb + eps), ub - eps)

        popt, _ = curve_fit(
            gaussian,
            x_fall,
            y_fall,
            p0=p0,
            bounds=(lb, ub),
            maxfev=10_000,
        )
    except Exception:
        return float(2048)

    _A_fit, mu_fit, sigma_fit = popt
    falling_edge = float(mu_fit + sigma_fit * np.sqrt(2.0 * np.log(2.0)))

    if (falling_edge > 2048) or (falling_edge < 0):
        falling_edge = float(2048)

    return float(falling_edge)

def extract_falling_edge(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float=40,
    lower_thresh: float = 0.3,
    upper_thresh: float = 0.7,
) -> float:
    """
    Extract falling edge using a linear fit on a (smoothed, normalised) spectrum.

    Behaviour preserved:
      - Smooth with sigma=40; normalise by max.
      - Use points after peak within [lower_thresh, upper_thresh].
      - If none: relax thresholds, else fall back to all points after peak, else all points.
      - Fit a line (polyfit); compute x at mid-intensity.
      - If out of bounds: compute a weighted spread surrogate and clamp to [0, 2048] by setting to 2048.
    """
    y_smoothed = gaussian_filter1d(y.astype(float), sigma=sigma)
    max_val = float(np.max(y_smoothed))
    assert max_val > 0, f"Max value is {max_val}. Before smoothing: {np.max(y)}."

    y_norm = y_smoothed / max_val
    peak_index = int(np.argmax(y_norm))

    indices = np.where(
        (np.arange(len(x)) > peak_index)
        & (y_norm <= upper_thresh)
        & (y_norm >= lower_thresh)
    )[0]

    if len(indices) == 0:
        alt_lower = lower_thresh * 0.8
        alt_upper = upper_thresh * 1.1
        indices = np.where(
            (np.arange(len(x)) > peak_index)
            & (y_norm <= alt_upper)
            & (y_norm >= alt_lower)
        )[0]
        if len(indices) == 0:
            indices = np.arange(peak_index + 1, len(x))
            logging.info(
                f"Falling edge calculation: Use all points after peak! Now {len(indices)} points!"
            )
            if len(indices) == 0:
                logging.info(f"Use all points! Now {len(indices)} points!")
                indices = np.arange(0, len(x))
        else:
            logging.info("Use relaxed threshold!")

    diffs = np.diff(indices)
    split_indices = np.where(diffs > 1)[0] + 1
    contiguous_groups = np.split(indices, split_indices)
    largest_group = max(contiguous_groups, key=len)

    if len(largest_group) == 0:
        largest_group = np.arange(peak_index + 1, len(x))

    x_fit = x[largest_group]
    y_fit_points = y_norm[largest_group]

    try:
        m, c = np.polyfit(x_fit, y_fit_points, 1)
    except Exception:
        m, c = 0.0, 0.0

    mid_intensity = (upper_thresh + lower_thresh) / 2.0
    falling_edge = (mid_intensity - c) / m if m != 0 else float(np.median(x_fit))

    # Out-of-range fallback (preserved)
    if (0 > falling_edge) or (falling_edge > np.max(x)):
        peak_x = x[int(np.argmax(y_smoothed))]
        falling_edge = float(
            peak_x
            + np.sqrt(2.0 * np.log(2.0))
            * np.sqrt(np.average((x - peak_x) ** 2, weights=y_smoothed))
        )

    if (falling_edge > 2048) or (falling_edge < 0):
        print(f"falling_edge {falling_edge} not within spectrum! Set to 2048.")
        falling_edge = float(2048)

    return float(falling_edge)



# =============================================================================
# Model/feature API used by evaluate_model.py
# =============================================================================

def ref_median_name(feature_name: str) -> str:
    """
    Standardized naming for reference-median features computed by the evaluator.
    """
    return f"{feature_name}_ref_median"


@dataclass
class FeatureContext:
    """
    Runtime context passed to feature functions.

    Attributes
    ----------
    args : Any
        Parsed CLI arguments from evaluate_model.py.
    time_window_bins : Optional[List[int]]
        Optional time window [lo, hi] (inclusive) for feature computations that need it.
    caches : Dict[str, Any]
        Shared mutable caches for expensive computations.
    """
    args: Any
    time_window_bins: Optional[List[int]]
    caches: Dict[str, Any]


@dataclass(frozen=True)
class FeatureSpec:
    """
    Base-feature specification.

    compute_scalar must return a single float for a single file (path).
    The evaluator is responsible for:
      - applying the feature to measured files (per-row)
      - applying the feature to reference files (per-row and per unique ref file)
      - computing the per-row reference median

    This keeps the evaluator generic and removes redundancy.
    """
    name: str
    compute_scalar: Callable[[str, FeatureContext], float]
    description: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """
    Model specification.

    base_features:
        List of base feature names that must exist in FEATURE_REGISTRY.
        For each base feature F, evaluator provides:
          - features[F]
          - features[ref_median_name(F)] == features[F_ref_median]
    predict:
        Callable mapping the feature dict to predictions (shape: n_rows).
    """
    base_features: List[str]
    predict: Callable[[Dict[str, np.ndarray]], np.ndarray]


FEATURE_REGISTRY: Dict[str, FeatureSpec] = {}
MODEL_REGISTRY: Dict[str, ModelSpec] = {}


# =============================================================================
# Default base feature(s)
# =============================================================================

def _10th_percentile_smoothed_feature_scalar(path: str, ctx: FeatureContext) -> float:
    """
    Default base feature: 10th percentule of smoothed time histogram using pseudocount repetition.

    Uses parameters from ctx.args:
      - smoothing_sigma_bins
      - smoothed_repeat_scale
      - ctx.time_window_bins
    """
    sigma = float(getattr(ctx.args, "smoothing_sigma_bins"))
    scale = int(getattr(ctx.args, "smoothed_repeat_scale"))

    spec = np.load(path)
    spec = ensure_2d_energy_time(spec)

    return smoothed_hist_10th_percentile(
        spec,
        smoothing_sigma_bins=sigma,
        smoothed_repeat_scale=scale,
        time_window_bins=ctx.time_window_bins,
        strict_nonnegative_integer=True,
    )


# Keep the original feature key used by your current pipeline/model
FEATURE_REGISTRY["Percentile_10th_smoothed"] = FeatureSpec(
    name="Percentile_10th_smoothed",
    compute_scalar=_10th_percentile_smoothed_feature_scalar,
    description="10th percentile of Gaussian-smoothed time histogram.",
)

def _position_trailing_edge_smoothed(path: str, ctx: FeatureContext) -> float:
    
    sigma = float(getattr(ctx.args, "smoothing_sigma_bins"))

    spec = np.load(path)
    spec = ensure_2d_energy_time(spec)

    time_window_bins=ctx.time_window_bins

    spec = spec[:, time_window_bins[0] : time_window_bins[1] + 1]

    number_of_bins = int(np.shape(spec)[1])
    y_hist = np.sum(spec, axis=0)
    x_hist = np.arange(number_of_bins)

    y_hist_smoothed = gaussian_filter1d(y_hist.astype(float), sigma=sigma)

    return extract_falling_edge(x_hist, y_hist_smoothed, sigma)

FEATURE_REGISTRY["Position_trailing_edge_smoothed"] = FeatureSpec(
    name="Position_trailing_edge_smoothed",
    compute_scalar=_position_trailing_edge_smoothed,
    description="Position of the trailing edge on the smoothed spectrum using linear regression of falling flank",
)

def _position_trailing_edge_gauss(path: str, ctx: FeatureContext) -> float:
    

    spec = np.load(path)
    spec = ensure_2d_energy_time(spec)

    time_window_bins=ctx.time_window_bins

    spec = spec[:, time_window_bins[0] : time_window_bins[1] + 1]

    number_of_bins = int(np.shape(spec)[1])
    y_hist = np.sum(spec, axis=0)
    x_hist = np.arange(number_of_bins)

    return extract_falling_edge_gauss(x_hist, y_hist)

FEATURE_REGISTRY["Position_trailing_edge_gauss"] = FeatureSpec(
    name="Position_trailing_edge_gauss",
    compute_scalar=_position_trailing_edge_gauss,
    description="Position of the trailing edge on the smoothed spectrum using fit of gaussian function",
)


# =============================================================================
# Default model (manuscript single-feature model)
# =============================================================================

def _paper_final_model(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Manuscript final model (Eq. (1)–(2)) written in terms of p and p_ref:
    It is a single feature model based on the relative change of the 10th 
    percentile of the smoothed spectrum.
    """
    beta0 = 0.072623
    beta1 = -4.454680
    lam = 0.0009
    mu_g = -0.000552
    sigma_g = 0.055612

    f = "Percentile_10th_smoothed"
    p = np.asarray(features[f], dtype=float)
    p_ref = np.asarray(features[ref_median_name(f)], dtype=float)

    x = (p - p_ref) / p_ref
    g = yeo_johnson(x, lam=lam)
    z = (g - mu_g) / sigma_g
    return beta0 + beta1 * z


MODEL_REGISTRY["paper_final_model"] = ModelSpec(
    base_features=["Percentile_10th_smoothed"],
    predict=_paper_final_model,
)

def _paper_trailing_edge_model(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Manuscript trailling edge model.
    """
    beta0 = 0.06942503
    beta1 = -4.69277809
    beta2 = 0.11937477
    lam1 = 0.17461968
    lam2 = 1.04637311
    mu_g1 = -0.00011688
    mu_g2 = 1.31614677
    sigma_g1 = 0.01599407
    sigma_g2 = 21.30888463

    f1 = "Position_trailing_edge_smoothed"
    f2 = "Position_trailing_edge_gauss"

    p1 = np.asarray(features[f1], dtype=float)
    p1_ref = np.asarray(features[ref_median_name(f1)], dtype=float)

    p2 = np.asarray(features[f2], dtype=float)
    p2_ref = np.asarray(features[ref_median_name(f2)], dtype=float)


    x1 = (p1 - p1_ref) / p1_ref # rel change
    x2 = p2 - p2_ref # abs change

    g1 = yeo_johnson(x1, lam=lam1)
    g2 = yeo_johnson(x2, lam=lam2)

    z1 = (g1 - mu_g1) / sigma_g1
    z2 = (g2 - mu_g2) / sigma_g2
    return beta0 + beta1 * z1 + beta2 * z2


MODEL_REGISTRY["paper_trailing_edge_model"] = ModelSpec(
    base_features=["Position_trailing_edge_smoothed", "Position_trailing_edge_gauss"],
    predict=_paper_trailing_edge_model,
)


# =============================================================================
# USER EXTENSION GUIDE
# =============================================================================
#
# 1) Add a new base feature:
#
#    def my_feature_scalar(path: str, ctx: FeatureContext) -> float:
#        ...
#
#    FEATURE_REGISTRY["MyFeature"] = FeatureSpec(
#        name="MyFeature",
#        compute_scalar=my_feature_scalar,
#        description="..."
#    )
#
# 2) Add a model using one or multiple base features:
#
#    def my_model(features: Dict[str, np.ndarray]) -> np.ndarray:
#        a = features["MyFeature"]
#        a_ref = features[ref_median_name("MyFeature")]
#        b = features["OtherFeature"]
#        b_ref = features[ref_median_name("OtherFeature")]
#        ...
#
#    MODEL_REGISTRY["my_model_name"] = ModelSpec(
#        base_features=["MyFeature", "OtherFeature"],
#        predict=my_model
#    )
#
# evaluate_model.py will compute each feature and its _ref_median automatically.
