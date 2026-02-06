# -*- coding: utf-8 -*-
"""
calculate_parameters.py

This module computes histogram-derived feature parameters from preprocessed spectra.

Original authors (2021–2022):
    Julia Wiedkamp, Sonja Schellhammer (s.schellhammer@hzdr.de)
Additional contributions (2023):
    Aaron Kieslich

"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm

from pmma.cmd_args import feature_calculation_parser
from pmma.visulisation_methods import save_features_plot

# -----------------------------------------------------------------------------
# Environment and logging
# -----------------------------------------------------------------------------
# Preserved side-effect (original code prints and sets JOBLIB_TEMP_FOLDER).
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
print("Set JOBLIB_TEMP_FOLDER to /tmp")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Spectrum axis conventions / bounds used throughout the original implementation.
SPECTRUM_MAX_BIN = 2048
EDGE_SMOOTH_SIGMA = 40  # sigma used for gaussian_filter1d in edge finders and smoothed features.


# -----------------------------------------------------------------------------
# Gaussian model and edge extraction
# -----------------------------------------------------------------------------
def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian function."""
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


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
        print("Gaussian fit failed: Set to 2048")
        return float(SPECTRUM_MAX_BIN)

    _A_fit, mu_fit, sigma_fit = popt
    falling_edge = float(mu_fit + sigma_fit * np.sqrt(2.0 * np.log(2.0)))

    if (falling_edge > SPECTRUM_MAX_BIN) or (falling_edge < 0):
        print(f"falling_edge {falling_edge} not within spectrum! Set to 2048.")
        falling_edge = float(SPECTRUM_MAX_BIN)

    return float(falling_edge)


def extract_leading_edge_gauss(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate the leading edge (half-maximum point) via Gaussian fit on the rising side.

    Procedure:
      1) Identify peak.
      2) Restrict to x < peak.
      3) Fit Gaussian to restricted region.
      4) Leading edge (half max) is: mu - sigma * sqrt(2 * ln(2)).

    Behaviour preserved:
      - If fit fails: return 2048.
      - If computed edge is outside [0, 2048]: return 0.
      - If no rising-side data exist: fit is attempted on full spectrum.
    """
    peak_idx = int(np.argmax(y))
    mask = x < x[peak_idx]

    if not np.any(mask):
        x_rise = x
        y_rise = y
    else:
        x_rise = x[mask]
        y_rise = y[mask]

    A0 = float(y[peak_idx])
    mu0 = float(x[peak_idx])
    sigma0 = float((mu0 - x_rise[0]) / 2.0) if len(x_rise) > 1 else 1.0

    try:
        eps = np.finfo(float).eps
        ub_mu = mu0 if mu0 > eps else eps
        ub_sigma = mu0 if mu0 > eps else eps

        lower_bnd = [0.0, 0.0, 0.0]
        upper_bnd = [2.0 * A0, ub_mu, ub_sigma]

        popt, _ = curve_fit(
            gaussian,
            x_rise,
            y_rise,
            p0=[A0, mu0, sigma0],
            bounds=(lower_bnd, upper_bnd),
            maxfev=10_000,
        )
    except Exception:
        print("Gaussian fit failed: Set to 2048")
        return float(SPECTRUM_MAX_BIN)

    _A_fit, mu_fit, sigma_fit = popt
    leading_edge = float(mu_fit - sigma_fit * np.sqrt(2.0 * np.log(2.0)))

    if (leading_edge > SPECTRUM_MAX_BIN) or (leading_edge < 0):
        print(f"Leading edge {leading_edge} not within spectrum! Set to 0.")
        leading_edge = 0.0

    return float(leading_edge)


def extract_falling_edge(
    x: np.ndarray,
    y: np.ndarray,
    lower_thresh: float = 0.2,
    upper_thresh: float = 0.8,
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
    y_smoothed = gaussian_filter1d(y.astype(float), sigma=EDGE_SMOOTH_SIGMA)
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

    if (falling_edge > SPECTRUM_MAX_BIN) or (falling_edge < 0):
        print(f"falling_edge {falling_edge} not within spectrum! Set to 2048.")
        falling_edge = float(SPECTRUM_MAX_BIN)

    return float(falling_edge)


def extract_leading_edge(
    x: np.ndarray,
    y: np.ndarray,
    lower_thresh: float = 0.2,
    upper_thresh: float = 0.8,
) -> float:
    """
    Extract leading edge using a linear fit on a (smoothed, normalised) spectrum.

    Behaviour preserved:
      - Smooth with sigma=40; normalise by max.
      - Use points before peak within [lower_thresh, upper_thresh].
      - If none: relax thresholds, else fall back to all points before peak, else all points.
      - Fit a line (polyfit); compute x at mid-intensity.
      - If out-of-range: compute a weighted spread surrogate.
      - Clamp to [x[0], x[peak]] (as in original).
    """
    y_smoothed = gaussian_filter1d(y.astype(float), sigma=EDGE_SMOOTH_SIGMA)
    max_val = float(np.max(y_smoothed))
    assert max_val > 0, f"Max value is {max_val}. Before smoothing: {np.max(y)}."

    y_norm = y_smoothed / max_val
    peak_index = int(np.argmax(y_norm))

    indices = np.where(
        (np.arange(len(x)) < peak_index)
        & (y_norm >= lower_thresh)
        & (y_norm <= upper_thresh)
    )[0]

    if len(indices) == 0:
        alt_lower = lower_thresh * 0.8
        alt_upper = upper_thresh * 1.1
        indices = np.where(
            (np.arange(len(x)) < peak_index)
            & (y_norm >= alt_lower)
            & (y_norm <= alt_upper)
        )[0]
        if len(indices) == 0:
            indices = np.arange(0, peak_index)
            logging.info(
                f"Leading edge calcualtion: Use all points before peak! Now {len(indices)} points!"
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
        largest_group = np.arange(0, peak_index)

    x_fit = x[largest_group]
    y_fit_points = y_norm[largest_group]

    try:
        m, c = np.polyfit(x_fit, y_fit_points, 1)
    except Exception:
        m, c = 0.0, 0.0

    mid_intensity = (upper_thresh + lower_thresh) / 2.0
    leading_edge = (mid_intensity - c) / m if m != 0 else float(np.median(x_fit))

    peak_idx_smoothed = int(np.argmax(y_smoothed))
    if (leading_edge < x[0]) or (leading_edge > x[peak_idx_smoothed]):
        peak_x = x[peak_idx_smoothed]
        leading_edge = float(
            peak_x
            - np.sqrt(2.0 * np.log(2.0))
            * np.sqrt(np.average((peak_x - x) ** 2, weights=y_smoothed))
        )

    leading_edge = float(max(min(leading_edge, x[peak_idx_smoothed]), x[0]))
    return float(leading_edge)


# -----------------------------------------------------------------------------
# Feature computation
# -----------------------------------------------------------------------------
def find_nearest(array: np.ndarray, value: float) -> int:
    """Return index of the element in `array` closest to `value`."""
    array = np.asarray(array)
    return int((np.abs(array - value)).argmin())


def compute_features(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    hist_all: np.ndarray,
    selected_features: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute statistical features from histogram arrays.

    Parameters
    ----------
    x_hist:
        Bin positions.
    y_hist:
        Counts per bin (can be float for smoothed case).
    hist_all:
        Expanded array with x repeated according to counts (or scaled counts).
    selected_features:
        Iterable of feature base names to compute. If None, computes full set.

    Returns
    -------
    Dict[str, float]
        Feature name -> value.

    Notes
    -----
    Behaviour is preserved exactly, including:
      - assertions,
      - use of raw and smoothed entropy/uniformity via p=y/sum(y),
      - NaN checks and error raising.
    """
    features: Dict[str, float] = {}

    full_set = {
        "Mean",
        "Variance",
        "Standard_deviation",
        "Skewness",
        "Kurtosis",
        "Median",
        "Percentile_10th",
        "Percentile_90th",
        "Mode",
        "IQR_25_75",
        "IQR_50_90",
        "IQR_20_80",
        "IQR_30_70",
        "IQR_40_60",
        "IQR_35_65",
        "Mean_absolute_deviation",
        "Robust_mean_absolute_deviation_10_90",
        "Robust_mean_absolute_deviation_50_90",
        "Robust_mean_absolute_deviation_20_80",
        "Median_absolute_deviation",
        "Coefficient_of_variation",
        "Quartile_coefficient_of_dispersion",
        "Quantile_coefficient_of_dispersion_35_65",
        "Entropy",
        "Uniformity",
        "T1_to_T2_distance",
        "Area_under_the_curve",
        "Position_trailing_edge",
        "Position_trailing_edge_gauss",
        "Position_leading_edge",
        "Position_leading_edge_gauss",
    }

    if selected_features is None:
        selected = full_set
    else:
        selected = set(selected_features)

    assert np.sum(y_hist) > 0

    # Common intermediates
    common_required = {
        "Mean",
        "Variance",
        "Standard_deviation",
        "Skewness",
        "Kurtosis",
        "Mean_absolute_deviation",
        "Coefficient_of_variation",
        "Mode",
        "IQR_25_75",
        "IQR_50_90",
        "IQR_20_80",
        "IQR_30_70",
        "IQR_40_60",
        "IQR_35_65",
        "Median",
        "Percentile_10th",
        "Percentile_90th",
    }

    # Mean
    if selected.intersection(common_required):
        mittelwert = float(np.average(x_hist, weights=y_hist))
        if "Mean" in selected:
            features["Mean"] = mittelwert
    else:
        # mittelwert is needed in several places; keep original dependency behaviour.
        mittelwert = float(np.average(x_hist, weights=y_hist))

    # Variance / sigma / skewness / kurtosis
    if {"Variance", "Standard_deviation", "Skewness", "Kurtosis"}.intersection(selected):
        varianz = float(np.average((x_hist - mittelwert) ** 2, weights=y_hist))
        if "Variance" in selected:
            features["Variance"] = varianz

        sigma_val = float(np.sqrt(varianz))
        if "Standard_deviation" in selected:
            features["Standard_deviation"] = sigma_val

        p = y_hist / np.sum(y_hist)

        if sigma_val == 0:
            schiefe = 0.0
        else:
            schiefe = float(np.sum((x_hist - mittelwert) ** 3 * p) / (sigma_val**3))
        if "Skewness" in selected:
            features["Skewness"] = schiefe

        if sigma_val == 0:
            kurtosis = 0.0
        else:
            kurtosis = float(np.sum((x_hist - mittelwert) ** 4 * p) / (sigma_val**4) - 3.0)
        if "Kurtosis" in selected:
            features["Kurtosis"] = kurtosis
    else:
        # p and sigma_val are used later for entropy/uniformity/cv; keep the same approach:
        p = y_hist / np.sum(y_hist)
        varianz = float(np.average((x_hist - mittelwert) ** 2, weights=y_hist))
        sigma_val = float(np.sqrt(varianz))

    # Median/percentiles/mode/IQR
    if {
        "Median",
        "Percentile_10th",
        "Percentile_90th",
        "Mode",
        "IQR_25_75",
        "IQR_50_90",
        "IQR_20_80",
        "IQR_30_70",
        "IQR_40_60",
        "IQR_35_65",
    }.intersection(selected):
        median = float(np.median(hist_all))
        if "Median" in selected:
            features["Median"] = median

        p10 = float(np.percentile(hist_all, 10))
        if "Percentile_10th" in selected:
            features["Percentile_10th"] = p10

        p90 = float(np.percentile(hist_all, 90))
        if "Percentile_90th" in selected:
            features["Percentile_90th"] = p90

        if "Mode" in selected:
            m_val = np.max(y_hist)
            idx = np.where(y_hist == m_val)[0]
            if len(idx) == 1:
                modus = float(x_hist[idx][0])
            else:
                werte = x_hist[idx]
                modus = float(werte[np.argmin(np.abs(werte - mittelwert))])
            features["Mode"] = modus

        if "IQR_25_75" in selected:
            features["IQR_25_75"] = float(np.percentile(hist_all, 75) - np.percentile(hist_all, 25))
        if "IQR_50_90" in selected:
            features["IQR_50_90"] = float(np.percentile(hist_all, 90) - np.percentile(hist_all, 50))
        if "IQR_20_80" in selected:
            features["IQR_20_80"] = float(np.percentile(hist_all, 80) - np.percentile(hist_all, 20))
        if "IQR_30_70" in selected:
            features["IQR_30_70"] = float(np.percentile(hist_all, 70) - np.percentile(hist_all, 30))
        if "IQR_40_60" in selected:
            features["IQR_40_60"] = float(np.percentile(hist_all, 60) - np.percentile(hist_all, 40))
        if "IQR_35_65" in selected:
            features["IQR_35_65"] = float(np.percentile(hist_all, 65) - np.percentile(hist_all, 35))

    # Mean absolute deviation (about mean)
    if "Mean_absolute_deviation" in selected:
        features["Mean_absolute_deviation"] = float(np.mean(np.abs(hist_all - mittelwert)))

    # Robust mean absolute deviation (within percentile window)
    robust_defs = {
        "Robust_mean_absolute_deviation_10_90": (10, 90),
        "Robust_mean_absolute_deviation_50_90": (50, 90),
        "Robust_mean_absolute_deviation_20_80": (20, 80),
    }
    for label, (low_perc, high_perc) in robust_defs.items():
        if label in selected:
            robust_idx = (hist_all >= np.percentile(hist_all, low_perc)) & (
                hist_all <= np.percentile(hist_all, high_perc)
            )
            x_hist_robust = hist_all[robust_idx]
            rmad = float(np.mean(np.abs(x_hist_robust - np.mean(x_hist_robust))))
            if rmad is None or np.isnan(rmad):
                logging.warning(f"{label} resulted in None or NaN. Setting it to 0.")
                rmad = 0.0
            features[label] = rmad

    # Median absolute deviation (about median)
    if "Median_absolute_deviation" in selected:
        features["Median_absolute_deviation"] = float(np.mean(np.abs(hist_all - np.median(hist_all))))

    # Coefficient of variation
    if "Coefficient_of_variation" in selected:
        features["Coefficient_of_variation"] = float(sigma_val / mittelwert) if mittelwert != 0 else 0.0

    # Quartile coefficient of dispersion
    if "Quartile_coefficient_of_dispersion" in selected:
        q1 = float(np.percentile(hist_all, 25))
        q3 = float(np.percentile(hist_all, 75))
        features["Quartile_coefficient_of_dispersion"] = (q3 - q1) / (q3 + q1) if (q3 + q1) != 0 else 0.0

    # Quantile coefficient of dispersion 35/65
    if "Quantile_coefficient_of_dispersion_35_65" in selected:
        q35 = float(np.percentile(hist_all, 35))
        q65 = float(np.percentile(hist_all, 65))
        features["Quantile_coefficient_of_dispersion_35_65"] = (q65 - q35) / (q65 + q35) if (q65 + q35) != 0 else 0.0

    # Entropy / uniformity
    if "Entropy" in selected:
        features["Entropy"] = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    if "Uniformity" in selected:
        features["Uniformity"] = float(np.sum(p**2))

    # T1-to-T2 distance
    if "T1_to_T2_distance" in selected:
        cs = np.cumsum(y_hist)
        cs = cs / np.max(cs)
        cs = np.flip(cs)
        i1 = find_nearest(cs, 0.8)
        i2 = find_nearest(cs, 0.2)
        features["T1_to_T2_distance"] = float(x_hist[i2] - x_hist[i1])

    # Area under curve
    if "Area_under_the_curve" in selected:
        bin_width = float(x_hist[1] - x_hist[0]) if len(x_hist) > 1 else 1.0
        features["Area_under_the_curve"] = float(int(np.sum(y_hist)) * bin_width)
        assert np.sum(y_hist) > 0, "No counts in spectrum!"

    # Edge positions (linear and Gaussian)
    if "Position_trailing_edge" in selected:
        features["Position_trailing_edge"] = float(
            extract_falling_edge(x_hist, y_hist, lower_thresh=0.3, upper_thresh=0.7)
        )
    if "Position_trailing_edge_gauss" in selected:
        features["Position_trailing_edge_gauss"] = float(extract_falling_edge_gauss(x_hist, y_hist))
    if "Position_leading_edge" in selected:
        features["Position_leading_edge"] = float(
            extract_leading_edge(x_hist, y_hist, lower_thresh=0.3, upper_thresh=0.7)
        )
    if "Position_leading_edge_gauss" in selected:
        features["Position_leading_edge_gauss"] = float(extract_leading_edge_gauss(x_hist, y_hist))

    # Final NaN check (preserved)
    if np.any(np.isnan(list(features.values()))):
        bad = [k for k, v in features.items() if np.isnan(v)]
        raise ValueError(f"NaN encountered in features: {', '.join(bad)}")

    return features


def time_features_cal(
    spectrum: np.ndarray,
    selected_features: Optional[Sequence[str]] = None,
    time_window: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Compute time-domain histogram features for a 2D (energy, time) spectrum.

    Behaviour preserved:
      - Validates spectrum non-negativity and integerness.
      - If sum == 0: replaces with ones.
      - Optional global time window slicing.
      - Computes features on raw y_hist and smoothed y_hist (sigma=40).
      - Smoothed features are stored with suffix '_smoothed'.
    """
    assert not np.any(spectrum < 0), "Spectrum contains negative values!"
    assert np.all(np.mod(spectrum, 1) == 0), "Spectrum contains non-integer values!"

    if np.sum(spectrum) == 0:
        spectrum = np.ones_like(spectrum)

    if time_window is not None:
        spectrum = spectrum[:, time_window[0] : time_window[1] + 1]

    number_of_bins = int(np.shape(spectrum)[1])
    y_hist = np.sum(spectrum, axis=0)
    x_hist = np.arange(number_of_bins)
    hist_all = np.repeat(x_hist, y_hist.astype(int))

    features_raw = compute_features(x_hist, y_hist, hist_all, selected_features=selected_features)

    y_hist_smoothed = gaussian_filter1d(y_hist.astype(float), sigma=EDGE_SMOOTH_SIGMA)
    hist_all_smoothed = np.repeat(x_hist, np.round(y_hist_smoothed * 10).astype(int))
    features_smoothed = compute_features(
        x_hist, y_hist_smoothed, hist_all_smoothed, selected_features=selected_features
    )
    features_smoothed = {f"{k}_smoothed": v for k, v in features_smoothed.items()}

    df_param = {**features_raw, **features_smoothed}

    if np.any(np.isnan(list(df_param.values()))):
        raise ValueError("Error: NaN values detected in the calculated feature table.")

    return df_param


# -----------------------------------------------------------------------------
# IO / batching helpers for multiprocessing
# -----------------------------------------------------------------------------
def process_and_save_df(df: pd.DataFrame, path: Optional[str]) -> None:
    """
    Save feature DataFrame to CSV.

    Behaviour preserved:
      - Writes ';' separated CSV.
      - Ensures 'file_path_proc' is the first column.
      - Writes even if columns are many; does not add/remove id_global.
    """
    if path is not None and not df.empty:
        cols = ["file_path_proc"] + [c for c in df.columns if c != "file_path_proc"]
        df = df[cols]
        df.to_csv(path, sep=";", index=False)
    else:
        logging.warning("No data to save. DataFrame is empty or path is None.")


def load_spectra_for_paths(path_list: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Load spectra arrays for a list of file paths into a dictionary cache.
    """
    cache: Dict[str, np.ndarray] = {}
    for path in path_list:
        cache[path] = np.load(path, allow_pickle=True)
    return cache


def process_batch_of_paths(
    batch_paths: Sequence[str],
    feature_func: Callable[..., Dict[str, float]],
    feature_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Worker: load all spectra for `batch_paths`, compute features, return list of dicts.
    """
    pid = os.getpid()
    batch_paths = list(batch_paths)

    logging.debug(
        f"[worker pid={pid}] Starting batch with {len(batch_paths)} paths. "
        f"First path: {batch_paths[0] if batch_paths else 'N/A'}"
    )

    loaded_spectra = load_spectra_for_paths(batch_paths)
    batch_results: List[Dict[str, Any]] = []

    for path in batch_paths:
        spectrum = loaded_spectra[path]
        feature_dict: Dict[str, Any] = {"file_path_proc": path}
        feature_dict.update(feature_func(spectrum, **feature_params))
        batch_results.append(feature_dict)

    logging.debug(f"[worker pid={pid}] Finished batch with {len(batch_paths)} paths.")
    return batch_results


def execute_feature_calculations(
    data_table: pd.DataFrame,
    feature_func: Callable[..., Dict[str, float]],
    feature_params: Dict[str, Any],
    output_file_path: str,
    threads: int,
    batch_size: Optional[int],
) -> None:
    """
    Compute features for all unique file_path_proc entries in `data_table`.

    Behaviour preserved:
      - Uses ProcessPoolExecutor with fork context when possible.
      - Batches inputs to reduce overhead.
      - Falls back to serial mode if PGT_FORCE_SERIAL=1 or on OSError errno 28.
      - Writes empty CSV if no samples.
    """
    if "file_path_proc" not in data_table.columns:
        raise KeyError("data_table must contain a 'file_path_proc' column.")

    logging.info("[features] Starting execute_feature_calculations()")

    paths = data_table["file_path_proc"].dropna().unique()
    n_samples = int(len(paths))

    if n_samples == 0:
        logging.warning("[features] No samples found in data_table (no file_path_proc). Nothing to compute.")
        pd.DataFrame([]).to_csv(output_file_path, sep=";", index=False)
        return

    try:
        requested = int(threads)
    except Exception:
        requested = 1

    effective_workers = max(1, min(requested, n_samples))

    # Automatic batch sizing (preserved logic)
    if batch_size is None or batch_size <= 0:
        batch_size = int(np.ceil(n_samples / max(1, effective_workers * 20)))

    batch_size = max(1, min(int(batch_size), n_samples))

    batches = [paths[i : i + batch_size] for i in range(0, n_samples, batch_size)]
    n_batches = int(len(batches))


    all_results: List[Dict[str, Any]] = []

    def run_serial() -> None:
        logging.info("[features] Running feature calculation in pure serial mode.")
        for batch_idx, batch_paths in enumerate(tqdm(batches, desc="Calculating features (serial)")):
            logging.debug(
                f"[features] [serial] Processing batch {batch_idx+1}/{n_batches} with {len(batch_paths)} paths."
            )
            all_results.extend(process_batch_of_paths(batch_paths, feature_func, feature_params))

    # Serial override
    if os.environ.get("PGT_FORCE_SERIAL", "0") == "1":
        logging.info("[features] PGT_FORCE_SERIAL=1 -> forcing serial execution.")
        run_serial()
    else:
        try:
            logging.info("[features] About to construct multiprocessing context.")
            try:
                # Preserved preference for "fork"
                import multiprocessing as mp

                ctx = mp.get_context("fork")

            except Exception:
                logging.exception(
                    "[features] Failed to create multiprocessing context; falling back to serial execution."
                )
                run_serial()
                df_features = pd.DataFrame(all_results)
                process_and_save_df(df_features, output_file_path)
                return

            logging.info("[features] About to create ProcessPoolExecutor.")
            with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
                logging.info("[features] ProcessPoolExecutor successfully created.")
                logging.info("[features] Submitting batch tasks to executor.")

                future_to_batch_index = {
                    executor.submit(process_batch_of_paths, batch_paths, feature_func, feature_params): i
                    for i, batch_paths in enumerate(batches)
                }

                logging.info("[features] All batch tasks submitted. Entering as_completed loop.")

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_batch_index),
                    total=n_batches,
                    desc="Calculating features for all batches (multi-process)",
                ):
                    batch_idx = future_to_batch_index[future]
                    try:
                        batch_results = future.result()
                    except Exception:
                        logging.exception(
                            f"[features] Exception in worker for batch index {batch_idx}. Reraising error."
                        )
                        raise
                    else:
                        logging.debug(
                            f"[features] Collected results from batch {batch_idx+1}/{n_batches} "
                            f"({len(batch_results)} samples)."
                        )
                        all_results.extend(batch_results)

                logging.info("[features] Completed all batches in multi-process mode.")

        except OSError as e:
            logging.exception("[features] OSError encountered during creation/use of ProcessPoolExecutor.")
            if getattr(e, "errno", None) == 28:
                logging.warning(
                    "[features] OSError [Errno 28] No space left on device when creating or using "
                    "ProcessPoolExecutor. Falling back to pure serial execution."
                )
                run_serial()
            else:
                raise

    df_features = pd.DataFrame(all_results)
    logging.info(f"[features] Computed features for {len(df_features)} samples. Saving to CSV.")
    process_and_save_df(df_features, output_file_path)
    logging.info("[features] execute_feature_calculations() finished.")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = feature_calculation_parser("Calculate features")

    parser.add_argument(
        "--threads",
        type=int,
        default=192,
        help="Maximum number of worker processes to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Number of spectra per batch. If omitted, it is set automatically to "
            "ceil(total_number_of_spectra / effective_threads)."
        ),
    )
    parser.add_argument(
        "--selected_features",
        nargs="+",
        type=str,
        default=None,
        help=(
            "List of feature base names to compute, separated by spaces. "
            "Example: --selected_features Position_trailing_edge Position_trailing_edge_gauss. "
            'If "all" is included, all available features will be computed. '
            "If omitted entirely, all available features will also be computed."
        ),
    )

    args = parser.parse_args()

    data_table = pd.read_csv(args.data_table_path, sep=";")

    n_unique_paths = int(data_table["file_path_proc"].nunique())
    logging.info(f"Loaded data_table from {args.data_table_path}")

    # Respect SLURM allocation if present; otherwise keep CLI value.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            args.threads = max(1, int(slurm_cpus))
        except ValueError:
            pass

    # -------------------------------------------------------------------------
    # Feature selection and feature-specific parameters
    # -------------------------------------------------------------------------
    selected_features = args.selected_features

    feature_calculation_args = json.loads(args.feature_calculation_args)
    time_window = feature_calculation_args["global_time_window"]
    print("Global time window is set to: ", time_window)

    # Interpret "all" keyword or None (behaviour preserved)
    if (selected_features is None) or ("all" in [f.lower() for f in selected_features]):
        selected_features = None
        print("All available features will be computed.")
    else:
        print(f"Selected features: {selected_features}")

    feature_params = {"selected_features": selected_features, "time_window": time_window}

    # Select feature function
    if args.feature_type == "time":
        feature_func = time_features_cal
    else:
        raise ValueError(f"Feature type '{args.feature_type}' is not implemented!")

    # -------------------------------------------------------------------------
    # Execute and save
    # -------------------------------------------------------------------------
    execute_feature_calculations(
        data_table=data_table,
        feature_func=feature_func,
        feature_params=feature_params,
        output_file_path=args.output_file_path,
        threads=args.threads,
        batch_size=args.batch_size,
    )

    logging.info(f"{args.feature_type.capitalize()} features calculated!")

    # Optional plots
    if args.plot_figures:
        save_features_plot(
            args.data_table_path,
            args.output_file_path,
            os.path.join(args.figures_path, "features", "original", args.feature_type),
        )


if __name__ == "__main__":
    main()
