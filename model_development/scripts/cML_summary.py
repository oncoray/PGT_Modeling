# -*- coding: utf-8 -*-
"""
cml_summary.py

Summarise (classical) ML model predictions and signatures across multiple runs/files.

This script:
  1) Reads signature files and prediction files (with a naming convention containing tokens: ft, fsm, ml).
  2) Joins signatures with predictions on (feature_type, feature_selection_method, model_learner).
  3) Computes grouped performance metrics (RMSE and R²) with bootstrap confidence intervals.
  4) Writes summary CSV files and produces heatmaps.

"""

import os
import numpy as np
import pandas as pd

from pmma.cmd_args import cML_summary_parser
from pmma.visulisation_methods import plot_performance_heatmaps

from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def bootstrap_confidence_interval(y_true, y_pred, stat_function, n_bootstraps=1000, random_state=1):
    """
    Bootstrap CI for a statistic between y_true and y_pred.

    Returns
    -------
    (ci_low, ci_high) : tuple[float,float]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    stats = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_function(y_true[idx], y_pred[idx])

    ci_low, ci_high = np.percentile(stats, [2.5, 97.5])
    return float(ci_low), float(ci_high)


# --------------------------------------------------------------------------------------
# Helpers: file parsing and concatenation
# --------------------------------------------------------------------------------------
def parse_filename(filename: str):
    """
    Parses the filename to extract feature_type, feature_selection_method, and model_learner.

    Expected naming convention contains tokens:
      ... _ft_<FEATURETYPE>_fsm_<FSM>_ml_<MODELLEARNER>_<something>.csv

    Notes
    -----
    - model_learner may contain underscores -> we join all parts after 'ml' up to last token.
    """
    parts = filename.split('_')

    try:
        feature_type_index = parts.index('ft') + 1
        fsm_index = parts.index('fsm') + 1
        ml_index = parts.index('ml') + 1
    except ValueError as e:
        raise ValueError(
            f"Filename '{filename}' does not follow expected convention containing "
            f"'_ft_', '_fsm_', '_ml_'. Parsed parts: {parts}"
        ) from e

    feature_type = parts[feature_type_index]
    feature_selection_method = parts[fsm_index]
    model_learner = '_'.join(parts[ml_index:-1])  # last part contains extension token-ish

    return feature_type, feature_selection_method, model_learner


def concatenate_prediction_files(file_paths):
    """
    Reads and concatenates prediction CSV files into a single DataFrame and adds metadata columns.
    """
    all_data = []
    for path in file_paths:
        filename = os.path.basename(path)
        feature_type, feature_selection_method, model_learner = parse_filename(filename)

        df = pd.read_csv(path, sep=";")
        df["feature_type"] = feature_type
        df["feature_selection_method"] = feature_selection_method
        df["model_learner"] = model_learner

        all_data.append(df)

    if len(all_data) == 0:
        raise ValueError("No prediction files provided / readable.")
    return pd.concat(all_data, ignore_index=True)


def concatenate_signature_files(file_paths):
    """
    Reads and concatenates signature files (one feature per line) into a single DataFrame.

    Output columns:
      feature_type, feature_selection_method, model_learner, signature(list[str]), signature_key(tuple[str])
    """
    all_rows = []
    for path in file_paths:
        filename = os.path.basename(path)
        feature_type, feature_selection_method, model_learner = parse_filename(filename)

        df = pd.read_csv(path, header=None)
        sig_list = list(df.iloc[:, 0].astype(str).values)

        # signature_key must be hashable -> tuple; we keep original list in 'signature'
        all_rows.append(
            {
                "feature_type": feature_type,
                "feature_selection_method": feature_selection_method,
                "model_learner": model_learner,
                "signature": sig_list,
                "signature_key": tuple(sig_list),
            }
        )

    if len(all_rows) == 0:
        raise ValueError("No signature files provided / readable.")
    return pd.DataFrame(all_rows)


# --------------------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------------------
def _validate_required_columns(df: pd.DataFrame, required: list[str], context: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {context}: {missing}")


def summarize_performances_from_predictions(df: pd.DataFrame, n_bootstraps=1000, random_state=1):
    """
    Summarize RMSE/R2 and bootstrap CIs for each group.

    Two levels are produced:
      (A) detailed groups:
          [feature_type, feature_selection_method, model_learner, cohort, cv_data_set,
           nose_orientation, proton_energy, mu, range_shift_type, repetition]
      (B) combined groups:
          [feature_type, feature_selection_method, model_learner, cohort, cv_data_set]
          with dataset fields set to "combined".
    """
    required_cols = [
        "feature_type",
        "feature_selection_method",
        "model_learner",
        "cohort",
        "cv_data_set",
        "nose_orientation",
        "proton_energy",
        "mu",
        "range_shift_type",
        "repetition",
        "range_shift",
        "predicted_range_shift",
        "signature",
        "signature_key",
    ]
    _validate_required_columns(df, required_cols, "predictions_summary")

    # Ensure numeric
    df = df.copy()
    df["range_shift"] = pd.to_numeric(df["range_shift"], errors="coerce")
    df["predicted_range_shift"] = pd.to_numeric(df["predicted_range_shift"], errors="coerce")
    df = df.dropna(subset=["range_shift", "predicted_range_shift"])

    summary_rows = []

    # -------------------------
    # (A) Detailed grouping
    # -------------------------
    detailed_group_cols = [
        "feature_type",
        "feature_selection_method",
        "model_learner",
        "cohort",
        "cv_data_set",
        "nose_orientation",
        "proton_energy",
        "mu",
        "range_shift_type",
        "repetition",
    ]

    grouped = df.groupby(detailed_group_cols, dropna=False)

    for name, group in grouped:
        y_true = group["range_shift"].values
        y_pred = group["predicted_range_shift"].values

        rmse = calculate_rmse(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)

        rmse_ci_low, rmse_ci_high = bootstrap_confidence_interval(
            y_true, y_pred, calculate_rmse, n_bootstraps=n_bootstraps, random_state=random_state
        )
        r2_ci_low, r2_ci_high = bootstrap_confidence_interval(
            y_true, y_pred, calculate_r2, n_bootstraps=n_bootstraps, random_state=random_state
        )

        # ---- Signature consistency check (FIXED) ----
        # Use signature_key (tuple, hashable) rather than raw signature (list)
        if group["signature_key"].nunique() != 1:
            raise ValueError(f"Signature is not consistent within group {name}!")

        signature = group["signature"].iloc[0]  # list[str]
        sign_size = len(signature)

        summary_rows.append(
            {
                "feature_type": name[0],
                "feature_selection_method": name[1],
                "model_learner": name[2],
                "signature": signature,
                "sign_size": sign_size,
                "cohort": name[3],
                "cv_data_set": name[4],
                "nose_orientation": name[5],
                "proton_energy": name[6],
                "mu": name[7],
                "range_shift_type": name[8],
                "repetition": name[9],
                "RMSE": float(rmse),
                "RMSE CI Low": float(rmse_ci_low),
                "RMSE CI High": float(rmse_ci_high),
                "R2": float(r2),
                "R2 CI Low": float(r2_ci_low),
                "R2 CI High": float(r2_ci_high),
            }
        )

    # -------------------------
    # (B) Combined grouping
    # -------------------------
    combined_group_cols = [
        "feature_type",
        "feature_selection_method",
        "model_learner",
        "cohort",
        "cv_data_set",
    ]

    grouped2 = df.groupby(combined_group_cols, dropna=False)

    for name, group in grouped2:
        y_true = group["range_shift"].values
        y_pred = group["predicted_range_shift"].values

        rmse = calculate_rmse(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)

        rmse_ci_low, rmse_ci_high = bootstrap_confidence_interval(
            y_true, y_pred, calculate_rmse, n_bootstraps=n_bootstraps, random_state=random_state
        )
        r2_ci_low, r2_ci_high = bootstrap_confidence_interval(
            y_true, y_pred, calculate_r2, n_bootstraps=n_bootstraps, random_state=random_state
        )

        if group["signature_key"].nunique() != 1:
            raise ValueError(f"Signature is not consistent within combined group {name}!")

        signature = group["signature"].iloc[0]
        sign_size = len(signature)

        summary_rows.append(
            {
                "feature_type": name[0],
                "feature_selection_method": name[1],
                "model_learner": name[2],
                "signature": signature,
                "sign_size": sign_size,
                "cohort": name[3],
                "cv_data_set": name[4],
                "nose_orientation": "combined",
                "proton_energy": "combined",
                "mu": "combined",
                "range_shift_type": "combined",
                "repetition": "combined",
                "RMSE": float(rmse),
                "RMSE CI Low": float(rmse_ci_low),
                "RMSE CI High": float(rmse_ci_high),
                "R2": float(r2),
                "R2 CI Low": float(r2_ci_low),
                "R2 CI High": float(r2_ci_high),
            }
        )

    return pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = cML_summary_parser("cML summary")
    args = parser.parse_args()

    print("Start summary of cML...")

    # 1) Load signatures and predictions
    signature_summary = concatenate_signature_files(args.signature_file_paths)
    predictions_summary = concatenate_prediction_files(args.individual_prediction_files)

    # 2) Merge
    predictions_summary = pd.merge(
        signature_summary,
        predictions_summary,
        on=["feature_type", "feature_selection_method", "model_learner"],
        how="inner",
    )

    # Safety: ensure signature_key exists (in case user-provided signature_summary lacks it)
    if "signature_key" not in predictions_summary.columns:
        predictions_summary["signature_key"] = predictions_summary["signature"].apply(tuple)

    # 3) Save merged predictions summary
    os.makedirs(os.path.dirname(args.summary_predictions_file_path), exist_ok=True)
    predictions_summary.to_csv(args.summary_predictions_file_path, index=False, sep=";")
    print("Summary of model predictions saved to:", args.summary_predictions_file_path)

    # 4) Summarize performances
    summary_df = summarize_performances_from_predictions(predictions_summary, n_bootstraps=1000, random_state=1)

    os.makedirs(os.path.dirname(args.summary_performance_file_path), exist_ok=True)
    summary_df.to_csv(args.summary_performance_file_path, index=False, sep=";")
    print("Summary of model performances saved to:", args.summary_performance_file_path)

    # 5) Plot heatmaps
    os.makedirs(args.heatmap_plot_dir_path, exist_ok=True)
    plot_performance_heatmaps(summary_df, args.heatmap_plot_dir_path)
    print("Heatmaps saved to:", args.heatmap_plot_dir_path)
    
