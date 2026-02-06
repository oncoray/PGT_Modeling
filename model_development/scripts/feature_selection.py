#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
feature_selection.py

Incremental feature selection based on a precomputed feature ranking.

High-level behaviour (preserved)
--------------------------------
- Iteratively builds signatures by adding the next best-ranked feature that does not violate
  multicollinearity constraints (Pearson or VIF-based).
- Two evaluation regimes:
    1) If a 'testing' cohort exists:
         - Train on all 'training', evaluate on 'testing' (no CV).
    2) Otherwise:
         - Cross-validation within 'training' (training/validation folds).
- Two model families:
    - model_learner == "iterative_linear": custom two-stage linear regression (optional)
    - otherwise: FAMILIAR pipeline via pmma.familiar_preparation

Outputs (preserved)
-------------------
- feature_selection_metrics (dict): metrics per iteration + final signature.
- final_predictions (DataFrame): predictions for best signature.
- all_predictions (DataFrame): predictions for every iteration (tagged by iteration).

"""

from __future__ import annotations

import json
import multiprocessing
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pmma.cmd_args import feature_selection_parser
from pmma.familiar_preparation import (
    create_feature_table_for_familiar,
    evaluate_familiar_experiment,
    extract_hyperparameters,
    merge_data_with_predictions,
    perform_familiar_experiment,
)
from pmma.visulisation_methods import plot_final_signature_features


def compute_mrse(df: pd.DataFrame) -> float:
    """
    Compute MRSE (mean range-shift error).

    For each unique true range_shift, compute the mean predicted outcome, then return
    the mean absolute error over these grouped values.
    """
    grouped = df.groupby("range_shift", as_index=False).agg({"predicted_outcome": "mean"})
    grouped["abs_error"] = np.abs(grouped["predicted_outcome"] - grouped["range_shift"])
    return float(grouped["abs_error"].mean())


def _bootstrap_predictions(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute point estimates and bootstrap CIs for RMSE and R².

    Returns
    -------
    (rmse, rmse_ci_low, rmse_ci_high, r2, r2_ci_low, r2_ci_high)
    """
    import numpy as _np
    from sklearn.metrics import mean_squared_error, r2_score

    rmse_true = float(_np.sqrt(mean_squared_error(df["range_shift"], df["predicted_outcome"])))
    r2_true = float(r2_score(df["range_shift"], df["predicted_outcome"]))

    rmses: List[float] = []
    r2s: List[float] = []
    for _ in range(n_bootstrap):
        samp = df.sample(n=len(df), replace=True)
        rmses.append(float(_np.sqrt(mean_squared_error(samp["range_shift"], samp["predicted_outcome"]))))
        r2s.append(float(r2_score(samp["range_shift"], samp["predicted_outcome"])))

    alpha = (100.0 - ci) / 2.0
    return (
        rmse_true,
        float(_np.percentile(rmses, alpha)),
        float(_np.percentile(rmses, 100.0 - alpha)),
        r2_true,
        float(_np.percentile(r2s, alpha)),
        float(_np.percentile(r2s, 100.0 - alpha)),
    )


def _compute_aic(df: pd.DataFrame, k: int, outcome_col: str, pred_col: str) -> float:
    """
    Compute AIC-like criterion from RSS.

    Notes
    -----
    This preserves the existing formulation:
        AIC = n * log(RSS/n) + k * log(n)
    with numeric coercion and dropping invalid rows.
    """
    y_true = pd.to_numeric(df[outcome_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(f"No valid numeric rows for AIC in '{outcome_col}'/'{pred_col}'")

    residuals = y_true[valid] - y_pred[valid]
    rss = float(np.sum(residuals**2))
    return float(n_valid * np.log(rss / n_valid) + k * np.log(n_valid))


def check_multicollinearity_additional_feature(
    normalized_features: pd.DataFrame,
    selected_features: List[str],
    feature_to_add: str,
    multicollinearity_method: str,
    multicollinearity_threshold: float,
) -> bool:
    """
    Determine whether adding `feature_to_add` to `selected_features` violates multicollinearity rules.

    Returns
    -------
    bool
        True if multicollinearity is detected (i.e., feature should be skipped).
    """
    if not selected_features:
        return False

    if multicollinearity_method == "pearson":
        for already_selected_feature in selected_features:
            correlation, _ = pearsonr(
                normalized_features[already_selected_feature].values,
                normalized_features[feature_to_add].values,
            )
            if abs(correlation) > multicollinearity_threshold:
                print(
                    f"Feature {already_selected_feature} and {feature_to_add} highly correlate "
                    f"(abs({round(correlation, 2)}) > {multicollinearity_threshold}). "
                    f"Skipping {feature_to_add}!"
                )
                return True
        return False

    if multicollinearity_method == "VIF":
        temp_selected = selected_features + [feature_to_add]
        temp_df = normalized_features[temp_selected]
        vifs = [variance_inflation_factor(temp_df.values, i) for i in range(temp_df.shape[1])]
        if vifs[-1] >= multicollinearity_threshold:
            print(
                f"Feature {feature_to_add} has high VIF ({round(vifs[-1], 2)} > "
                f"{multicollinearity_threshold}). Skipping {feature_to_add}!"
            )
            return True
        return False

    raise ValueError(
        f"The multicollinearity method {multicollinearity_method} is not implemented. "
        "Choose 'pearson' or 'VIF'."
    )


def check_multicollinearity_in_set(
    normalized_features: pd.DataFrame,
    features: List[str],
    multicollinearity_threshold: float,
    vif_bar_plot_path: str,
) -> None:
    """
    Compute VIF for the selected signature and save a bar plot.

    This function preserves the original side effects:
    - prints warnings for features above threshold
    - saves a bar plot to `vif_bar_plot_path`
    - prints the VIF DataFrame
    """
    if len(features) == 0:
        raise ValueError("No features in set to check!")

    missing_features = [f for f in features if f not in normalized_features.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

    temp_df = normalized_features[features]

    if len(features) == 1:
        vifs = pd.DataFrame({"Feature": features, "VIF": [1.0]})
    else:
        vifs = pd.DataFrame(
            {
                "Feature": features,
                "VIF": [variance_inflation_factor(temp_df.values, i) for i in range(temp_df.shape[1])],
            }
        )

    for _, row in vifs.iterrows():
        if row["VIF"] >= multicollinearity_threshold:
            print(
                f"Warning: Feature '{row['Feature']}' has high VIF ({row['VIF']}) "
                f"exceeding the threshold of {multicollinearity_threshold}!"
            )

    os.makedirs(os.path.dirname(vif_bar_plot_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(vifs["Feature"], vifs["VIF"], color="skyblue")
    plt.axhline(y=multicollinearity_threshold, color="r", linestyle="--")
    plt.title("VIF Values for Each Feature")
    plt.xlabel("Features")
    plt.ylabel("Variance Inflation Factor (VIF)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vif_bar_plot_path)

    print("Variance Inflation Factor (VIF) of all features in signature:")
    print(vifs)


def plot_rmse_r2_result(
    error_data: Dict[str, Any],
    plot_file_path: str,
    final_signature_size: int,
) -> None:
    """
    Plot RMSE (with CI), R², and AIC versus number of features and save the plot.

    Notes
    -----
    The plotting style and the underlying data fields are preserved.
    """
    os.makedirs(os.path.dirname(plot_file_path), exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # RMSE subplot (with CI)
    ax1.errorbar(
        error_data["number_of_features"],
        error_data["development"]["rmse"],
        yerr=[
            np.array(error_data["development"]["rmse"]) - np.array(error_data["development"]["rmse_ci_low"]),
            np.array(error_data["development"]["rmse_ci_high"]) - np.array(error_data["development"]["rmse"]),
        ],
        fmt="-o",
        capsize=5,
        elinewidth=2,
        markeredgewidth=2,
        label="Development",
        alpha=0.5,
    )
    ax1.errorbar(
        error_data["number_of_features"],
        error_data["validation"]["rmse"],
        yerr=[
            np.array(error_data["validation"]["rmse"]) - np.array(error_data["validation"]["rmse_ci_low"]),
            np.array(error_data["validation"]["rmse_ci_high"]) - np.array(error_data["validation"]["rmse"]),
        ],
        fmt="-o",
        capsize=5,
        elinewidth=2,
        markeredgewidth=2,
        label="Validation",
        alpha=0.5,
    )
    ax1.set_title("RMSE")
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("RMSE")
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.axvline(x=final_signature_size, color="r", linestyle="--", label="Final Signature Size")
    ax1.legend()

    # R² subplot (lines only; preserved)
    ax2.plot(
        error_data["number_of_features"],
        error_data["development"]["r2_score"],
        "-o",
        markeredgewidth=2,
        label="Development",
        alpha=0.7,
    )
    ax2.plot(
        error_data["number_of_features"],
        error_data["validation"]["r2_score"],
        "-o",
        markeredgewidth=2,
        label="Validation",
        alpha=0.7,
    )
    ax2.set_title("R2 Score")
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("R2 Score")
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.axvline(x=final_signature_size, color="r", linestyle="--", label="Final Signature Size")
    ax2.legend()

    # AIC subplot
    ax3.plot(error_data["number_of_features"], error_data["development"]["aic"], "-o", label="Development", alpha=0.7)
    ax3.plot(error_data["number_of_features"], error_data["validation"]["aic"], "-o", label="Validation", alpha=0.7)
    ax3.set_title("AIC (Full Training)")
    ax3.set_xlabel("Number of Features")
    ax3.set_ylabel("AIC")
    ax3.grid(True)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.axvline(x=final_signature_size, color="r", linestyle="--", label="Final Signature Size")
    ax3.legend()

    plt.suptitle("Model Performance Metrics vs. Number of Features")
    plt.tight_layout()
    plt.savefig(plot_file_path)
    print(f"Feature selection plot saved to {plot_file_path}")


def get_set_of_features(
    feature_table: pd.DataFrame,
    features_sorted_by_rank: List[str],
    model_learner: str,
    feature_file_path: str,
    familiar_r_file_path: str,
    feature_selection_path: str,
    feature_selection_method: str,
    feature_type: str,
    vif_bar_plot_path: str,
    n_cpus: int = 1,
    max_features: int = 10,
    multicollinearity_method: str = "pearson",
    multicollinearity_threshold: float = 0.6,
    max_no_improve_iterations: int = 3,
    multicollinearity_threshold_vif: float = 5.0,
    perform_yeo_johnson: bool = True,
    execute_two_step_fitting: bool = True,
    fit_intercept: bool = True,
    **selection_args: Any,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Incrementally build and evaluate signatures.

    Returns
    -------
    feature_selection_metrics:
        Nested dictionary containing per-iteration metrics and the final signature.
    final_predictions:
        DataFrame with columns ['sample_id', 'predicted_outcome', ...] for the best iteration.
    all_predictions:
        DataFrame with predictions for all iterations (includes 'iteration' column).
    """
    # Detect whether a 'testing' cohort exists
    cohorts = feature_table["cohort"].unique()
    has_test = "testing" in cohorts

    # Bookkeeping across iterations
    selected_features: List[str] = []
    not_selected_features: List[str] = []
    final_signature: List[str] = []
    all_iteration_predictions: List[pd.DataFrame] = []
    error_increase_count = 0

    # Prevent UnboundLocalError (preserved)
    final_predictions = pd.DataFrame(columns=["sample_id", "predicted_outcome"])

    feature_selection_metrics: Dict[str, Any] = {
        "number_of_features": [],
        "features": [],
        "development": {
            "rmse": [],
            "rmse_ci_low": [],
            "rmse_ci_high": [],
            "r2_score": [],
            "r2_score_ci_low": [],
            "r2_score_ci_high": [],
            "aic": [],
        },
        "validation": {
            "rmse": [],
            "rmse_ci_low": [],
            "rmse_ci_high": [],
            "r2_score": [],
            "r2_score_ci_low": [],
            "r2_score_ci_high": [],
            "aic": [],
        },
        "final_signature": {"features": None, "hyperparameters": None, "development": {}, "validation": {}},
    }

    selection_criterion = selection_args.get("selection_criterion", "rmse")

    # -------------------------------------------------------------------------
    # Prepare normalised features for multicollinearity checks (training only)
    # -------------------------------------------------------------------------
    pt = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
    list_features = [f for f in feature_table.columns if f not in ("id_global", "cohort", "range_shift")]

    feature_table = feature_table.dropna()

    normalized_features = pd.DataFrame(
        {
            feature: pt.fit_transform(
                feature_table.loc[feature_table["cohort"] == "training", feature].values.reshape(-1, 1)
            )[:, 0]
            for feature in list_features
        }
    )

    # If testing exists, split once for the custom branch
    if has_test:
        training_data_full = feature_table[feature_table["cohort"] == "training"].copy()
        testing_data = feature_table[feature_table["cohort"] == "testing"].copy()

    # Determine iteration list
    if max_features == 0:
        iteration_list = [len(features_sorted_by_rank)]
        selected_features = list(features_sorted_by_rank)
    else:
        iteration_list = range(1, max_features + 1)

    print(f"Iteration list: {list(iteration_list)}")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    for i in iteration_list:
        if error_increase_count >= max_no_improve_iterations:
            print("No improvement; stopping early.")
            break

        # Add next feature by rank with multicollinearity constraints
        if max_features != 0:
            for feat in features_sorted_by_rank:
                if feat in selected_features or feat in not_selected_features:
                    continue
                if check_multicollinearity_additional_feature(
                    normalized_features,
                    selected_features,
                    feat,
                    multicollinearity_method,
                    multicollinearity_threshold,
                ):
                    not_selected_features.append(feat)
                else:
                    selected_features.append(feat)
                    break

            if len(selected_features) < i:
                print("No further features available.")
                break

        print(f"Iteration {i}: testing signature {selected_features}")

        # ---------------------------------------------------------------------
        # Regime A: explicit testing cohort exists
        # ---------------------------------------------------------------------
        if has_test:
            if model_learner == "iterative_linear":
                from sklearn.linear_model import LinearRegression

                X_tr = training_data_full[selected_features]
                X_te = testing_data[selected_features]
                y_tr = training_data_full["range_shift"]
                y_te = testing_data["range_shift"]

                # Optional Yeo–Johnson transform
                if perform_yeo_johnson:
                    pt_iter = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
                    X_tr_t = pd.DataFrame(pt_iter.fit_transform(X_tr), index=X_tr.index, columns=selected_features)
                    X_te_t = pd.DataFrame(pt_iter.transform(X_te), index=X_te.index, columns=selected_features)
                    fit_intercept_stage1 = fit_intercept
                    fit_intercept_stage2 = fit_intercept
                else:
                    X_tr_t = X_tr.copy()
                    X_te_t = X_te.copy()
                    fit_intercept_stage1 = fit_intercept
                    fit_intercept_stage2 = fit_intercept

                # Stage 1 regression
                lr1 = LinearRegression(fit_intercept=fit_intercept_stage1)
                lr1.fit(X_tr_t, y_tr)
                y_tr1 = lr1.predict(X_tr_t)
                y_te1 = lr1.predict(X_te_t)

                # Optional stage 2 calibration
                if execute_two_step_fitting:
                    df_med = (
                        pd.DataFrame({"range_shift": y_tr, "pred1": y_tr1})
                        .groupby("range_shift", as_index=False)["pred1"]
                        .median()
                    )

                    lr2 = LinearRegression(fit_intercept=fit_intercept_stage2)
                    lr2.fit(df_med["pred1"].values.reshape(-1, 1), df_med["range_shift"].values)

                    y_tr_final = lr2.predict(y_tr1.reshape(-1, 1))
                    y_te_final = lr2.predict(y_te1.reshape(-1, 1))
                else:
                    lr2 = None
                    y_tr_final = y_tr1
                    y_te_final = y_te1

                mrse_train = compute_mrse(pd.DataFrame({"range_shift": y_tr, "predicted_outcome": y_tr_final}))
                mrse_test = compute_mrse(pd.DataFrame({"range_shift": y_te, "predicted_outcome": y_te_final}))

                dev_rmse, dev_rlo, dev_rhi, dev_r2, _, _ = _bootstrap_predictions(
                    pd.DataFrame({"range_shift": y_tr, "predicted_outcome": y_tr_final, "data_set": "development"})
                )
                test_rmse, test_rlo, test_rhi, test_r2, _, _ = _bootstrap_predictions(
                    pd.DataFrame({"range_shift": y_te, "predicted_outcome": y_te_final, "data_set": "validation"})
                )

                aic_full = _compute_aic(
                    pd.DataFrame({"range_shift": y_tr, "predicted_outcome": y_tr_final}),
                    k=len(selected_features),
                    outcome_col="range_shift",
                    pred_col="predicted_outcome",
                )

                print(f"Iteration {i} Performance (iterative_linear):")
                print(f"  perform_yeo_johnson     : {perform_yeo_johnson}")
                print(f"  execute_two_step_fitting: {execute_two_step_fitting}")
                print(f"  Stage-1 fit_intercept   : {fit_intercept_stage1}")
                if execute_two_step_fitting:
                    print(f"  Stage-2 fit_intercept   : {fit_intercept_stage2}")
                print(f"  Training: RMSE = {dev_rmse:.3f}, R² = {dev_r2:.3f}, MRSE = {mrse_train:.3f}")
                print(f"  Testing:  RMSE = {test_rmse:.3f}, R² = {test_r2:.3f}, MRSE = {mrse_test:.3f}")

                pred_tr = pd.DataFrame(
                    {
                        "sample_id": training_data_full["id_global"].values,
                        "predicted_outcome": y_tr_final,
                        "data_set": "development",
                    }
                )
                pred_te = pd.DataFrame(
                    {
                        "sample_id": testing_data["id_global"].values,
                        "predicted_outcome": y_te_final,
                        "data_set": "validation",
                    }
                )
                predictions = pd.concat([pred_tr, pred_te], ignore_index=True)

                predictions_iteration = predictions.copy()
                predictions_iteration["iteration"] = i
                all_iteration_predictions.append(predictions_iteration)

                # Record metrics
                feature_selection_metrics["number_of_features"].append(i)
                feature_selection_metrics["features"].append(list(selected_features))

                feature_selection_metrics["development"]["rmse"].append(dev_rmse)
                feature_selection_metrics["development"]["rmse_ci_low"].append(dev_rlo)
                feature_selection_metrics["development"]["rmse_ci_high"].append(dev_rhi)
                feature_selection_metrics["development"]["r2_score"].append(dev_r2)
                feature_selection_metrics["development"]["r2_score_ci_low"].append(dev_rlo)
                feature_selection_metrics["development"]["r2_score_ci_high"].append(dev_rhi)
                feature_selection_metrics["development"]["aic"].append(aic_full)

                feature_selection_metrics["validation"]["rmse"].append(test_rmse)
                feature_selection_metrics["validation"]["rmse_ci_low"].append(test_rlo)
                feature_selection_metrics["validation"]["rmse_ci_high"].append(test_rhi)
                feature_selection_metrics["validation"]["r2_score"].append(test_r2)
                feature_selection_metrics["validation"]["r2_score_ci_low"].append(test_rlo)
                feature_selection_metrics["validation"]["r2_score_ci_high"].append(test_rhi)
                feature_selection_metrics["validation"]["aic"].append(aic_full)

                is_best = (
                    (selection_criterion == "rmse" and test_rmse == min(feature_selection_metrics["validation"]["rmse"]))
                    or (selection_criterion == "aic" and aic_full == min(feature_selection_metrics["development"]["aic"]))
                )

                if is_best:
                    final_signature = list(selected_features)
                    final_predictions = predictions.copy()
                    error_increase_count = 0
                    feature_selection_metrics["final_signature"]["features"] = final_signature
                    feature_selection_metrics["final_signature"]["development"] = {
                        "rmse": dev_rmse,
                        "rmse_ci_low": dev_rlo,
                        "rmse_ci_high": dev_rhi,
                        "r2_score": dev_r2,
                        "r2_score_ci_low": dev_rlo,
                        "r2_score_ci_high": dev_rhi,
                        "aic": aic_full,
                    }
                    feature_selection_metrics["final_signature"]["validation"] = {
                        "rmse": test_rmse,
                        "rmse_ci_low": test_rlo,
                        "rmse_ci_high": test_rhi,
                        "r2_score": test_r2,
                        "r2_score_ci_low": test_rlo,
                        "r2_score_ci_high": test_rhi,
                        "aic": aic_full,
                    }
                    print("→ New best model on test set!")

                if i == max_features:
                    print("Reached maximum feature count.")
                continue

            # FAMILIAR branch with test set
            experiment_dir = os.path.join(
                feature_selection_path,
                "familiar",
                "experiments",
                feature_type,
                feature_selection_method,
                model_learner,
                f"iteration_{i}",
            )
            perform_familiar_experiment(
                feature_file_path=feature_file_path,
                familiar_r_file_path=familiar_r_file_path,
                model_learner=model_learner,
                experiment_dir=experiment_dir,
                experimental_design="fs + mb + ev",
                batch_id_column="cohort",
                sample_id_column="id_global",
                development_batch_id="training",
                validation_batch_id="testing",
                outcome_name="range_shift",
                outcome_column="range_shift",
                outcome_type="continuous",
                include_features=selected_features,
                parallel=True,
                parallel_nr_cores=n_cpus,
                feature_max_fraction_missing=0.01,
                transformation_method="yeo_johnson",
                normalisation_method="standardisation",
                hyperparameter={model_learner: {"sign_size": len(selected_features)}},
                skip_evaluation_elements=[
                    "auc_data",
                    "calibration_data",
                    "calibration_info",
                    "confusion_matrix",
                    "decision_curve_analyis",
                    "ice_data",
                    "permutation_vimp",
                    "univariate_analysis",
                    "model_vimp",
                    "feature_expressions",
                    "fs_vimp",
                    "sample_similarity",
                ],
            )
            results, predictions = evaluate_familiar_experiment(experiment_dir)

            predictions_iteration = predictions.copy()
            predictions_iteration["iteration"] = i
            all_iteration_predictions.append(predictions_iteration)

            rmse_dev = results["development"]["rmse"][0]
            rmse_lo_dev = results["development"]["rmse_ci_low"][0]
            rmse_hi_dev = results["development"]["rmse_ci_high"][0]
            r2_dev = results["development"]["r2_score"][0]
            r2_lo_dev = results["development"]["r2_score_ci_low"][0]
            r2_hi_dev = results["development"]["r2_score_ci_high"][0]

            rmse_test = results["validation"]["rmse"][0]
            rmse_lo_test = results["validation"]["rmse_ci_low"][0]
            rmse_hi_test = results["validation"]["rmse_ci_high"][0]
            r2_test = results["validation"]["r2_score"][0]
            r2_lo_test = results["validation"]["r2_score_ci_low"][0]
            r2_hi_test = results["validation"]["r2_score_ci_high"][0]

            print(f"Iteration {i} Performance (FAMILIAR):")
            print(f"  Training: RMSE = {rmse_dev:.3f}, R² = {r2_dev:.3f}")
            print(f"  Testing:  RMSE = {rmse_test:.3f}, R² = {r2_test:.3f}")

            try:
                train_preds = predictions[predictions["cohort"] == "training"]
                aic_full = _compute_aic(train_preds, len(selected_features), outcome_col="outcome", pred_col="predicted_outcome")
            except Exception:
                aic_full = np.nan

            feature_selection_metrics["number_of_features"].append(i)
            feature_selection_metrics["features"].append(list(selected_features))

            feature_selection_metrics["development"]["rmse"].append(rmse_dev)
            feature_selection_metrics["development"]["rmse_ci_low"].append(rmse_lo_dev)
            feature_selection_metrics["development"]["rmse_ci_high"].append(rmse_hi_dev)
            feature_selection_metrics["development"]["r2_score"].append(r2_dev)
            feature_selection_metrics["development"]["r2_score_ci_low"].append(r2_lo_dev)
            feature_selection_metrics["development"]["r2_score_ci_high"].append(r2_hi_dev)
            feature_selection_metrics["development"]["aic"].append(aic_full)

            feature_selection_metrics["validation"]["rmse"].append(rmse_test)
            feature_selection_metrics["validation"]["rmse_ci_low"].append(rmse_lo_test)
            feature_selection_metrics["validation"]["rmse_ci_high"].append(rmse_hi_test)
            feature_selection_metrics["validation"]["r2_score"].append(r2_test)
            feature_selection_metrics["validation"]["r2_score_ci_low"].append(r2_lo_test)
            feature_selection_metrics["validation"]["r2_score_ci_high"].append(r2_hi_test)
            feature_selection_metrics["validation"]["aic"].append(aic_full)

            if selection_criterion == "rmse":
                best = rmse_test == np.min(feature_selection_metrics["validation"]["rmse"])
            else:
                best = aic_full == np.nanmin(feature_selection_metrics["development"]["aic"])

            if best:
                final_signature = list(selected_features)
                final_predictions = predictions.copy()
                error_increase_count = 0
                feature_selection_metrics["final_signature"]["features"] = final_signature
                feature_selection_metrics["final_signature"]["development"] = {
                    "rmse": rmse_dev,
                    "rmse_ci_low": rmse_lo_dev,
                    "rmse_ci_high": rmse_hi_dev,
                    "r2_score": r2_dev,
                    "r2_score_ci_low": r2_lo_dev,
                    "r2_score_ci_high": r2_hi_dev,
                    "aic": aic_full,
                }
                feature_selection_metrics["final_signature"]["validation"] = {
                    "rmse": rmse_test,
                    "rmse_ci_low": rmse_lo_test,
                    "rmse_ci_high": rmse_hi_test,
                    "r2_score": r2_test,
                    "r2_score_ci_low": r2_lo_test,
                    "r2_score_ci_high": r2_hi_test,
                    "aic": aic_full,
                }
                feature_selection_metrics["final_signature"]["hyperparameters"] = extract_hyperparameters(experiment_dir)
                print("→ New best FAMILIAR model on test set!")

            if i == max_features:
                print("Reached maximum feature count.")
            continue

        # ---------------------------------------------------------------------
        # Regime B: no explicit testing cohort (CV on training)
        # ---------------------------------------------------------------------
        if model_learner == "iterative_linear":
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.model_selection import KFold

            training_data = feature_table[feature_table["cohort"] == "training"].copy()

            n_runs = 3
            n_folds = 3

            cv_results_development: List[Dict[str, float]] = []
            cv_results_validation: List[Dict[str, float]] = []
            cv_predictions_list: List[pd.DataFrame] = []

            for run in range(n_runs):
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=run)
                for fold, (train_index, val_index) in enumerate(kf.split(training_data)):
                    train_fold = training_data.iloc[train_index].copy()
                    val_fold = training_data.iloc[val_index].copy()

                    pt_cv = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
                    X_train = train_fold[selected_features]
                    X_val = val_fold[selected_features]

                    X_train_trans = pd.DataFrame(
                        pt_cv.fit_transform(X_train), index=X_train.index, columns=selected_features
                    )
                    X_val_trans = pd.DataFrame(
                        pt_cv.transform(X_val), index=X_val.index, columns=selected_features
                    )

                    # Stage 1
                    lr1 = LinearRegression()
                    y_train = train_fold["range_shift"]
                    lr1.fit(X_train_trans, y_train)
                    y_train_pred_1 = lr1.predict(X_train_trans)

                    # Stage 2 (median calibration)
                    df_group = pd.DataFrame({"range_shift": y_train, "pred1": y_train_pred_1})
                    df_median = df_group.groupby("range_shift", as_index=False)["pred1"].median()
                    lr2 = LinearRegression()
                    lr2.fit(df_median["pred1"].values.reshape(-1, 1), df_median["range_shift"].values)

                    y_train_pred_final = lr2.predict(y_train_pred_1.reshape(-1, 1))
                    y_val_pred_1 = lr1.predict(X_val_trans)
                    y_val_pred_final = lr2.predict(y_val_pred_1.reshape(-1, 1))

                    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_train_pred_final)))
                    r2_train = float(r2_score(y_train, y_train_pred_final))
                    rmse_val = float(np.sqrt(mean_squared_error(val_fold["range_shift"], y_val_pred_final)))
                    r2_val = float(r2_score(val_fold["range_shift"], y_val_pred_final))

                    cv_results_development.append({"rmse": rmse_train, "r2": r2_train})
                    cv_results_validation.append({"rmse": rmse_val, "r2": r2_val})

                    train_preds_df = train_fold[["id_global"]].copy()
                    train_preds_df["cohort"] = train_fold["cohort"]
                    train_preds_df["cv_data_set"] = "development"
                    train_preds_df["range_shift"] = y_train.values
                    train_preds_df["predicted_range_shift"] = y_train_pred_final
                    train_preds_df["cv_run"] = run
                    train_preds_df["cv_fold"] = fold

                    val_preds_df = val_fold[["id_global"]].copy()
                    val_preds_df["cohort"] = val_fold["cohort"]
                    val_preds_df["cv_data_set"] = "validation"
                    val_preds_df["range_shift"] = val_fold["range_shift"].values
                    val_preds_df["predicted_range_shift"] = y_val_pred_final
                    val_preds_df["cv_run"] = run
                    val_preds_df["cv_fold"] = fold

                    cv_predictions_list.append(train_preds_df)
                    cv_predictions_list.append(val_preds_df)

            cv_predictions = pd.concat(cv_predictions_list, axis=0, ignore_index=True)

            predictions = cv_predictions.groupby(["id_global", "cv_data_set"], as_index=False).agg(
                {"predicted_range_shift": "median", "range_shift": "first"}
            )
            predictions.rename(
                columns={
                    "id_global": "sample_id",
                    "predicted_range_shift": "predicted_outcome",
                    "cv_data_set": "data_set",
                },
                inplace=True,
            )

            predictions_iteration = predictions.copy()
            predictions_iteration["iteration"] = i
            all_iteration_predictions.append(predictions_iteration)

            dev_preds = predictions[predictions["data_set"] == "development"].copy()
            val_preds = predictions[predictions["data_set"] == "validation"].copy()

            dev_rmse, dev_rmse_ci_low, dev_rmse_ci_high, dev_r2, dev_r2_ci_low, dev_r2_ci_high = _bootstrap_predictions(
                dev_preds
            )
            val_rmse, val_rmse_ci_low, val_rmse_ci_high, val_r2, val_r2_ci_low, val_r2_ci_high = _bootstrap_predictions(
                val_preds
            )

            print("CV Performance (iterative_linear):")
            print(
                f"Development: RMSE = {round(dev_rmse,2)} [{round(dev_rmse_ci_low,2)}, {round(dev_rmse_ci_high,2)}], "
                f"R2 = {round(dev_r2,2)}"
            )
            print(
                f"Validation:  RMSE = {round(val_rmse,2)} [{round(val_rmse_ci_low,2)}, {round(val_rmse_ci_high,2)}], "
                f"R2 = {round(val_r2,2)}"
            )

            # Full training AIC (no CV), preserved
            full_training_data = training_data.copy()
            pt_full = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
            X_full = full_training_data[selected_features]
            X_full_trans = pd.DataFrame(pt_full.fit_transform(X_full), index=full_training_data.index, columns=selected_features)
            y_full = full_training_data["range_shift"].astype(float)

            lr1_full = LinearRegression()
            lr1_full.fit(X_full_trans, y_full)
            y_full_pred_1 = lr1_full.predict(X_full_trans)

            df_full_group = pd.DataFrame({"range_shift": y_full, "pred1": y_full_pred_1})
            df_full_median = df_full_group.groupby("range_shift", as_index=False)["pred1"].median()

            lr2_full = LinearRegression()
            lr2_full.fit(df_full_median["pred1"].values.reshape(-1, 1), df_full_median["range_shift"].values)
            y_full_pred_final = lr2_full.predict(y_full_pred_1.reshape(-1, 1))

            rss_full = float(np.sum((y_full - y_full_pred_final) ** 2))
            n_full = int(len(y_full))
            k = int(len(selected_features))
            aic_full = float(n_full * np.log(rss_full / n_full) + k * np.log(n_full))
            print(f"Full training AIC (iterative_linear): {aic_full}")

            feature_selection_metrics["number_of_features"].append(len(selected_features))
            feature_selection_metrics["features"].append(list(selected_features))

            feature_selection_metrics["development"]["rmse"].append(dev_rmse)
            feature_selection_metrics["development"]["rmse_ci_low"].append(dev_rmse_ci_low)
            feature_selection_metrics["development"]["rmse_ci_high"].append(dev_rmse_ci_high)
            feature_selection_metrics["development"]["r2_score"].append(dev_r2)
            feature_selection_metrics["development"]["r2_score_ci_low"].append(dev_r2_ci_low)
            feature_selection_metrics["development"]["r2_score_ci_high"].append(dev_r2_ci_high)
            feature_selection_metrics["development"]["aic"].append(aic_full)

            feature_selection_metrics["validation"]["rmse"].append(val_rmse)
            feature_selection_metrics["validation"]["rmse_ci_low"].append(val_rmse_ci_low)
            feature_selection_metrics["validation"]["rmse_ci_high"].append(val_rmse_ci_high)
            feature_selection_metrics["validation"]["r2_score"].append(val_r2)
            feature_selection_metrics["validation"]["r2_score_ci_low"].append(val_r2_ci_low)
            feature_selection_metrics["validation"]["r2_score_ci_high"].append(val_r2_ci_high)
            feature_selection_metrics["validation"]["aic"].append(aic_full)

            if selection_criterion == "rmse":
                best_condition = val_rmse_ci_high == np.min(feature_selection_metrics["validation"]["rmse"])
            elif selection_criterion == "aic":
                best_condition = aic_full == np.min(feature_selection_metrics["development"]["aic"])
            else:
                raise ValueError("Invalid selection criterion specified. Use 'rmse' or 'aic'.")

            if best_condition:
                final_predictions = predictions.copy()
                final_signature = list(selected_features)
                error_increase_count = 0
                feature_selection_metrics["final_signature"]["features"] = final_signature
                feature_selection_metrics["final_signature"]["validation"] = {
                    "rmse": val_rmse,
                    "rmse_ci_low": val_rmse_ci_low,
                    "rmse_ci_high": val_rmse_ci_high,
                    "r2_score": val_r2,
                    "r2_score_ci_low": val_r2_ci_low,
                    "r2_score_ci_high": val_r2_ci_high,
                    "aic": aic_full,
                }
                feature_selection_metrics["final_signature"]["development"] = {
                    "rmse": dev_rmse,
                    "rmse_ci_low": dev_rmse_ci_low,
                    "rmse_ci_high": dev_rmse_ci_high,
                    "r2_score": dev_r2,
                    "r2_score_ci_low": dev_r2_ci_low,
                    "r2_score_ci_high": dev_r2_ci_high,
                    "aic": aic_full,
                }
                feature_selection_metrics["final_signature"]["hyperparameters"] = {}
                print("New best iterative_linear model found based on selection criterion!")

            if i == max_features:
                print("Maximum number of features to test reached!")
                break

        else:
            # FAMILIAR CV experiment (non-iterative_linear)
            experiment_dir = os.path.join(
                feature_selection_path,
                "familiar",
                "experiments",
                feature_type,
                feature_selection_method,
                model_learner,
                f"iteration_{i}",
            )
            perform_familiar_experiment(
                feature_file_path=feature_file_path,
                model_learner=model_learner,
                experiment_dir=experiment_dir,
                familiar_r_file_path=familiar_r_file_path,
                signature=selected_features,
                experimental_design="fs + cv(mb,10,1)",
                batch_id_column="cohort",
                sample_id_column="id_global",
                development_batch_id="training",
                validation_batch_id="validation",
                outcome_name="range_shift",
                outcome_column="range_shift",
                outcome_type="continuous",
                parallel_nr_cores=n_cpus,
                parallel=True,
                feature_max_fraction_missing=0.01,
                filter_method="none",
                transformation_method="yeo_johnson",
                normalisation_method="standardisation",
                cluster_method="none",
                parallel_preprocessing="parallel_preprocessing",
                fs_method="none",
                vimp_aggregation_method="stability",
                vimp_aggregation_rank_threshold=5,
                novelty_detector="none",
                optimisation_determine_vimp=True,
                evaluation_metric=["rmse", "r2_score"],
                imputation_method="simple",
                include_features=selected_features,
                hyperparameter={model_learner: {"sign_size": len(selected_features)}},
                skip_evaluation_elements=[
                    "auc_data",
                    "calibration_data",
                    "calibration_info",
                    "confusion_matrix",
                    "decision_curve_analyis",
                    "ice_data",
                    "permutation_vimp",
                    "univariate_analysis",
                    "model_vimp",
                    "feature_expressions",
                    "fs_vimp",
                    "sample_similarity",
                ],
            )
            results, predictions = evaluate_familiar_experiment(experiment_dir)

            predictions_iteration = predictions.copy()
            predictions_iteration["iteration"] = i
            all_iteration_predictions.append(predictions_iteration)

            rmse_dev = results["development"]["rmse"][0]
            rmse_ci_low_dev = results["development"]["rmse_ci_low"][0]
            rmse_ci_high_dev = results["development"]["rmse_ci_high"][0]
            r2_score_dev = results["development"]["r2_score"][0]
            r2_score_ci_low_dev = results["development"]["r2_score_ci_low"][0]
            r2_score_ci_high_dev = results["development"]["r2_score_ci_high"][0]

            rmse_val = results["validation"]["rmse"][0]
            rmse_ci_low_val = results["validation"]["rmse_ci_low"][0]
            rmse_ci_high_val = results["validation"]["rmse_ci_high"][0]
            r2_score_val = results["validation"]["r2_score"][0]
            r2_score_ci_low_val = results["validation"]["r2_score_ci_low"][0]
            r2_score_ci_high_val = results["validation"]["r2_score_ci_high"][0]

            print("CV Performance (FAMILIAR):")
            print(
                f"Development: RMSE = {round(rmse_dev,2)} [{round(rmse_ci_low_dev,2)}, {round(rmse_ci_high_dev,2)}], "
                f"R2 = {round(r2_score_dev,2)}"
            )
            print(
                f"Validation:  RMSE = {round(rmse_val,2)} [{round(rmse_ci_low_val,2)}, {round(rmse_ci_high_val,2)}], "
                f"R2 = {round(r2_score_val,2)}"
            )

            # Separate experiment for AIC on full training
            experiment_dir_aic = os.path.join(
                feature_selection_path,
                "familiar",
                "experiments",
                feature_type,
                feature_selection_method,
                model_learner,
                f"iteration_{i}_aic",
            )
            perform_familiar_experiment(
                feature_file_path=feature_file_path,
                model_learner=model_learner,
                experiment_dir=experiment_dir_aic,
                familiar_r_file_path=familiar_r_file_path,
                signature=selected_features,
                experimental_design="fs + mb",
                batch_id_column="cohort",
                sample_id_column="id_global",
                development_batch_id="training",
                validation_batch_id="validation",
                outcome_name="range_shift",
                outcome_column="range_shift",
                outcome_type="continuous",
                parallel_nr_cores=40,
                parallel=True,
                feature_max_fraction_missing=0.01,
                filter_method="none",
                transformation_method="yeo_johnson",
                normalisation_method="standardisation",
                cluster_method="none",
                parallel_preprocessing="parallel_preprocessing",
                fs_method="none",
                vimp_aggregation_method="stability",
                vimp_aggregation_rank_threshold=5,
                novelty_detector="none",
                optimisation_determine_vimp=True,
                evaluation_metric=["rmse", "r2_score"],
                imputation_method="simple",
                include_features=selected_features,
                hyperparameter={model_learner: {"sign_size": len(selected_features)}},
                skip_evaluation_elements=[
                    "auc_data",
                    "calibration_data",
                    "calibration_info",
                    "confusion_matrix",
                    "decision_curve_analyis",
                    "ice_data",
                    "permutation_vimp",
                    "univariate_analysis",
                    "model_vimp",
                    "feature_expressions",
                    "fs_vimp",
                    "sample_similarity",
                ],
            )
            results_aic, predictions_aic = evaluate_familiar_experiment(experiment_dir_aic)
            aic_dev = _compute_aic(predictions_aic, len(selected_features), "outcome", "predicted_outcome")
            print(f"Full training AIC (FAMILIAR): {aic_dev}")

            feature_selection_metrics["number_of_features"].append(len(selected_features))
            feature_selection_metrics["features"].append(list(selected_features))

            feature_selection_metrics["development"]["rmse"].append(rmse_dev)
            feature_selection_metrics["development"]["rmse_ci_low"].append(rmse_ci_low_dev)
            feature_selection_metrics["development"]["rmse_ci_high"].append(rmse_ci_high_dev)
            feature_selection_metrics["development"]["r2_score"].append(r2_score_dev)
            feature_selection_metrics["development"]["r2_score_ci_low"].append(r2_score_ci_low_dev)
            feature_selection_metrics["development"]["r2_score_ci_high"].append(r2_score_ci_high_dev)
            feature_selection_metrics["development"]["aic"].append(aic_dev)

            feature_selection_metrics["validation"]["rmse"].append(rmse_val)
            feature_selection_metrics["validation"]["rmse_ci_low"].append(rmse_ci_low_val)
            feature_selection_metrics["validation"]["rmse_ci_high"].append(rmse_ci_high_val)
            feature_selection_metrics["validation"]["r2_score"].append(r2_score_val)
            feature_selection_metrics["validation"]["r2_score_ci_low"].append(r2_score_ci_low_val)
            feature_selection_metrics["validation"]["r2_score_ci_high"].append(r2_score_ci_high_val)
            feature_selection_metrics["validation"]["aic"].append(aic_dev)

            if selection_criterion == "rmse":
                best_condition = rmse_ci_high_val == np.min(feature_selection_metrics["validation"]["rmse"])
            elif selection_criterion == "aic":
                best_condition = aic_dev == np.min(feature_selection_metrics["development"]["aic"])
            else:
                raise ValueError("Invalid selection criterion specified. Use 'rmse' or 'aic'.")

            if best_condition:
                final_predictions = predictions.copy()
                final_signature = list(selected_features)
                error_increase_count = 0
                feature_selection_metrics["final_signature"]["features"] = final_signature
                feature_selection_metrics["final_signature"]["validation"] = {
                    "rmse": rmse_val,
                    "rmse_ci_low": rmse_ci_low_val,
                    "rmse_ci_high": rmse_ci_high_val,
                    "r2_score": r2_score_val,
                    "r2_score_ci_low": r2_score_ci_low_val,
                    "r2_score_ci_high": r2_score_ci_high_val,
                    "aic": aic_dev,
                }
                feature_selection_metrics["final_signature"]["development"] = {
                    "rmse": rmse_dev,
                    "rmse_ci_low": rmse_ci_low_dev,
                    "rmse_ci_high": rmse_ci_high_dev,
                    "r2_score": r2_score_dev,
                    "r2_score_ci_low": r2_score_ci_low_dev,
                    "r2_score_ci_high": r2_score_ci_high_dev,
                    "aic": aic_dev,
                }
                feature_selection_metrics["final_signature"]["hyperparameters"] = extract_hyperparameters(experiment_dir)
                print("New best FAMILIAR-based model found based on selection criterion!")

            if i == max_features:
                print("Maximum number of features to test reached!")
                break

    print(f"Final signature is {final_signature}.")

    # Multicollinearity report for final signature (preserved)
    check_multicollinearity_in_set(
        normalized_features,
        final_signature,
        multicollinearity_threshold_vif,
        vif_bar_plot_path,
    )

    if "range_shift" in final_predictions.columns:
        final_predictions = final_predictions.drop(["range_shift"], axis=1)

    if len(all_iteration_predictions) > 0:
        all_predictions = pd.concat(all_iteration_predictions, ignore_index=True)
    else:
        all_predictions = pd.DataFrame(columns=["sample_id", "predicted_outcome", "data_set", "iteration"])

    return feature_selection_metrics, final_predictions, all_predictions


if __name__ == "__main__":
    parser = feature_selection_parser("Feature selection")
    args = parser.parse_args()
    selection_args = json.loads(args.selection_args)

    print("Start feature selection....")
    n_cpus = multiprocessing.cpu_count()
    print(f"Number of cpus: {n_cpus}")

    data_table = pd.read_csv(args.data_table_path, sep=";")
    feature_table = pd.read_csv(args.feature_file_path, sep=";")
    feature_ranking_table = pd.read_csv(args.feature_ranking_file_path, sep=";")
    features_sorted_by_rank = feature_ranking_table.sort_values(by="score", ascending=False)["feature"].tolist()

    temp_data_dir = os.path.join(args.feature_selection_path, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)

    familiar_feature_table_path = os.path.join(
        temp_data_dir,
        f"features_{args.feature_type}_{args.feature_selection_method}_{args.model_learner}.csv",
    )
    familiar_r_file_path = os.path.join(
        args.feature_selection_path,
        f"familiar/r_files/R_file_{args.feature_type}_{args.feature_selection_method}_{args.model_learner}.R",
    )

    ranked_feature_table = create_feature_table_for_familiar(
        data_table,
        feature_table,
        features_sorted_by_rank,
        familiar_feature_table_path,
        evaluation_phase=False,
    )

    feature_selection_metrics, predictions, predictions_all = get_set_of_features(
        ranked_feature_table,
        features_sorted_by_rank,
        args.model_learner,
        familiar_feature_table_path,
        familiar_r_file_path,
        args.feature_selection_path,
        args.feature_selection_method,
        args.feature_type,
        args.vif_bar_plot_path,
        n_cpus,
        **selection_args,
    )

    # Save predictions for best signature
    os.makedirs(os.path.dirname(args.predictions_file_path), exist_ok=True)
    final_predictions_table = merge_data_with_predictions(data_table, predictions)
    final_predictions_table.to_csv(args.predictions_file_path, sep=";", index=False)
    print(f"Predictions successfully saved to {args.predictions_file_path}")

    # Save predictions for all iterations
    pred_path_iterations = args.predictions_file_path.replace(".csv", "_iterations.csv")
    all_predictions_table = merge_data_with_predictions(data_table, predictions_all)
    all_predictions_table.to_csv(pred_path_iterations, sep=";", index=False)
    print(f"All-iteration predictions successfully saved to {pred_path_iterations}")

    # Plots and signature export
    final_signature = feature_selection_metrics["final_signature"]["features"]
    plot_final_signature_features(
        ranked_feature_table,
        final_signature,
        args.signature_plot_path,
        args.feature_type,
        args.feature_selection_method,
        args.model_learner,
    )

    feature_selection_metrics["feature_type"] = args.feature_type
    feature_selection_metrics["feature_selection_method"] = args.feature_selection_method
    feature_selection_metrics["model_learner"] = args.model_learner

    plot_rmse_r2_result(feature_selection_metrics, args.feature_selection_plot_path, len(final_signature))

    os.makedirs(os.path.dirname(args.feature_selection_file_path), exist_ok=True)
    with open(args.feature_selection_file_path, "w") as file:
        json.dump(feature_selection_metrics, file, indent=4)
    print(f"Signature successfully saved to {args.feature_selection_file_path}")

    os.makedirs(os.path.dirname(args.signature_file_path), exist_ok=True)
    pd.DataFrame(final_signature).to_csv(args.signature_file_path, index=False, header=False)
    print(f"Signature successfully saved to {args.signature_file_path}")



