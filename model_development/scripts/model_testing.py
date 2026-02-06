#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:49:21 2024

@author: kiesli21
"""

from pmma.cmd_args import testing_parser
import pandas as pd
from pmma.file_preparation import evaluate_familiar_experiment, create_feature_table_for_familiar, perform_familiar_experiment, extract_hyperparameters, merge_data_with_predictions
from pmma.visulisation_methods import plot_predicted_vs_actual_range_shift
import os
import json
import multiprocessing
import numpy as np
from pathlib import Path


# Import scikit-learn modules needed for the iterative_linear approach
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

def make_json_serializable(obj):
    """
    Recursively convert numpy/pandas objects into JSON-serializable Python types.
    """
    import numpy as _np

    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)

    # Optional: support pandas scalars if they occur
    try:
        import pandas as _pd
        if isinstance(obj, (_pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass

    return obj

def perform_testing(testing_path, familiar_feature_table_path, familiar_r_file_path,
                                feature_type, feature_selection_method, model_learner, signature, perform_yeo_johnson=True,
                        execute_two_step_fitting=True, fit_intercept=True):
    """
    Conducts model testing.

    Parameters:
    - testing_path (str): Path where model testing results and artifacts will be stored.
    - familiar_feature_table_path (str): Path to the feature table used by the FAMILIAR framework.
    - familiar_r_file_path (str): Path to the R script for running the FAMILIAR model.
    - feature_type (str): Type of features used (e.g., genomic, clinical).
    - feature_selection_method (str): Method used for feature selection.
    - model_learner (str): Machine learning algorithm used for modeling.
    - signature (list): List of features constituting the model's signature.

    Returns:
    Tuple[dict, DataFrame]: A tuple containing the model testing results and the predictions DataFrame.
    """

    # Initialize the results dictionary
    testing_results = {
        'feature_type': feature_type,
        'feature_selection_method': feature_selection_method,
        'model_learner': model_learner,
        'features': signature,
        'development': {},
        'testing': {},
        'validation': {}
    }

    # If using the new iterative_linear approach, perform training on the complete training cohort
    # and testing on the testing cohort.
    if model_learner == "iterative_linear":

        external_feature_table = pd.read_csv(familiar_feature_table_path, sep=";")

        # ---------------------------------------------------------------------------
        # Sanitize modelling columns (signature + target) against NaNs
        # ---------------------------------------------------------------------------
        cols_to_check = signature + ["range_shift"]
        na_mask = external_feature_table[cols_to_check].isna().any(axis=1)

        if na_mask.any():
            n_bad = int(na_mask.sum())
            n_tot = int(len(external_feature_table))
            print(
                f"WARNING: {n_bad} of {n_tot} rows contain NaNs in {cols_to_check} and "
                f"will be excluded from model training and prediction."
            )
            external_feature_table = external_feature_table.loc[~na_mask].copy()

        if external_feature_table.empty:
            raise ValueError("All rows were dropped because of NaNs; no data left for modelling.")

        # Split the data into training and testing cohorts.
        training_data = external_feature_table[external_feature_table["cohort"] == "training"].copy()
        testing_data = external_feature_table[
            (external_feature_table["cohort"] == "testing") | (external_feature_table["cohort"] == "validation")
        ].copy()

        # -------------------------------------------------------------------------
        # Feature processing toggle
        # -------------------------------------------------------------------------
        X_train = training_data[signature]
        X_val = testing_data[signature]

        if perform_yeo_johnson:
            # --- Transformation: Yeo-Johnson and Standardisation ---
            pt_ext = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
            X_train_trans = pd.DataFrame(pt_ext.fit_transform(X_train), index=X_train.index, columns=signature)
            X_val_trans = pd.DataFrame(pt_ext.transform(X_val), index=X_val.index, columns=signature)

            fit_intercept_stage1 = fit_intercept
            fit_intercept_stage2 = fit_intercept
        else:
            # No transformation and no standardization: use raw features directly.
            pt_ext = None
            X_train_trans = X_train.copy()
            X_val_trans = X_val.copy()

            fit_intercept_stage1 = fit_intercept
            fit_intercept_stage2 = fit_intercept

        # -------------------------------------------------------------------------
        # Stage 1: Linear Regression on the Complete Training Cohort
        # -------------------------------------------------------------------------
        lr1 = LinearRegression(fit_intercept=fit_intercept_stage1)
        y_train = training_data["range_shift"].to_numpy()
        lr1.fit(X_train_trans, y_train)
 
        y_train_pred_1 = lr1.predict(X_train_trans)

        # -------------------------------------------------------------------------
        # Stage 2: Optional fine-tuning / calibration
        # -------------------------------------------------------------------------
        if execute_two_step_fitting:
            # Aggregate first-stage predictions by computing the median for each unique true range_shift.
            df_group = pd.DataFrame({"range_shift": y_train, "pred1": y_train_pred_1})
            df_median = df_group.groupby("range_shift", as_index=False)["pred1"].median()

            lr2 = LinearRegression(fit_intercept=fit_intercept_stage2)
            lr2.fit(df_median["pred1"].values.reshape(-1, 1), df_median["range_shift"].values)

            # Adjust training predictions using second-stage model.
            y_train_pred_final = lr2.predict(y_train_pred_1.reshape(-1, 1))
        else:
            # No second stage: final predictions are stage-1 predictions.
            lr2 = None
            y_train_pred_final = y_train_pred_1

        # Compute training metrics.
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_final))
        train_r2 = r2_score(y_train, y_train_pred_final)

        # -------------------------------------------------------------------------
        # Apply on the testing Cohort
        # -------------------------------------------------------------------------
        y_val_pred_1 = lr1.predict(X_val_trans)

        if execute_two_step_fitting:
            y_val_pred_final = lr2.predict(y_val_pred_1.reshape(-1, 1))
        else:
            y_val_pred_final = y_val_pred_1

        y_val = testing_data["range_shift"].to_numpy()
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_final))
        val_r2 = r2_score(y_val, y_val_pred_final)

        # -------------------------------------------------------------------------
        # Prepare the Predictions DataFrame
        # -------------------------------------------------------------------------
        train_preds_df = training_data[["id_global"]].copy()
        train_preds_df["cohort"] = training_data["cohort"]
        train_preds_df["range_shift"] = y_train
        train_preds_df["predicted_range_shift"] = y_train_pred_final
        train_preds_df["data_set"] = "development"

        val_preds_df = testing_data[["id_global"]].copy()
        val_preds_df["cohort"] = testing_data["cohort"]
        val_preds_df["range_shift"] = y_val
        val_preds_df["predicted_range_shift"] = y_val_pred_final
        val_preds_df["data_set"] = "testing"

        predictions = pd.concat([train_preds_df, val_preds_df], axis=0, ignore_index=True)

        predictions.rename(
            columns={"id_global": "sample_id", "predicted_range_shift": "predicted_outcome", "cv_data_set": "data_set"},
            inplace=True,
        )
        predictions = predictions[["sample_id", "predicted_outcome", "data_set"]]

        # Update the results dictionary with computed metrics.
        testing_results["development"].update({"rmse": train_rmse, "r2_score": train_r2})
        testing_results["testing"].update({"rmse": val_rmse, "r2_score": val_r2})

        # Hyperparameters for the iterative_linear approach can be defined as needed.
        testing_results["hyperparameters"] = {
            "perform_yeo_johnson": bool(perform_yeo_johnson),
            "execute_two_step_fitting": bool(execute_two_step_fitting),
        }
        results = testing_results

        # -------------------------------------------------------------------------
        # Bootstrap confidence intervals (training cohort)
        #   - Resamples rows of the training cohort with replacement.
        #   - Re-fits preprocessing (if enabled), stage-1, and (optionally) stage-2 per replicate.
        #   - Reports percentile CIs for:
        #       stage-1 intercept and coefficients (in the space used by stage-1),
        #       stage-2 alpha/gamma (if enabled),
        #       combined final model intercept and coefficients (if stage-2 enabled).
        # -------------------------------------------------------------------------
        bootstrap_params = {
            "n_boot": int(locals().get("n_bootstrap", 1000)),
            "ci_level": float(locals().get("bootstrap_ci_level", 0.95)),
            "random_state": int(locals().get("bootstrap_random_state", 42)),
        }

        def _bootstrap_iterative_linear_ci(
            training_df: pd.DataFrame,
            signature: list,
            perform_yeo_johnson: bool,
            execute_two_step_fitting: bool,
            fit_intercept_stage1: bool,
            fit_intercept_stage2: bool,
            n_boot: int,
            ci_level: float,
            random_state: int,
        ) -> dict:
            """
            Percentile bootstrap CIs for model parameters on the training cohort.
            Returns a dict with point estimates + CIs for stage-1, stage-2 (if enabled), and combined final model.
            """
            if n_boot < 100:
                raise ValueError("n_bootstrap should be >= 100 for stable percentile confidence intervals.")

            if not (0.0 < ci_level < 1.0):
                raise ValueError("bootstrap_ci_level must be in (0,1).")

            alpha = 1.0 - ci_level
            q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

            rng = np.random.default_rng(random_state)
            n = int(len(training_df))
            if n < 5:
                raise ValueError("Training cohort too small for bootstrapping (need at least 5 samples).")

            # Fit once on full cohort to provide point estimates in the same code path.
            X_full = training_df[signature]
            y_full = training_df["range_shift"].to_numpy()

            if perform_yeo_johnson:
                pt_full = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
                X_full_t = pd.DataFrame(pt_full.fit_transform(X_full), index=X_full.index, columns=signature)
            else:
                pt_full = None
                X_full_t = X_full.copy()

            lr1_full = LinearRegression(fit_intercept=fit_intercept_stage1)
            lr1_full.fit(X_full_t, y_full)
            pred1_full = lr1_full.predict(X_full_t)

            if execute_two_step_fitting:
                df_group_full = pd.DataFrame({"range_shift": y_full, "pred1": pred1_full})
                df_median_full = df_group_full.groupby("range_shift", as_index=False)["pred1"].median()

                lr2_full = LinearRegression(fit_intercept=fit_intercept_stage2)
                lr2_full.fit(df_median_full["pred1"].values.reshape(-1, 1), df_median_full["range_shift"].values)

                beta0_s1 = float(lr1_full.intercept_)
                betas_s1 = lr1_full.coef_.astype(float)
                alpha_s2 = float(lr2_full.intercept_)
                gamma_s2 = float(lr2_full.coef_.ravel()[0])

                final_intercept_full = alpha_s2 + gamma_s2 * beta0_s1
                final_betas_full = gamma_s2 * betas_s1
            else:
                lr2_full = None
                beta0_s1 = float(lr1_full.intercept_)
                betas_s1 = lr1_full.coef_.astype(float)
                alpha_s2 = None
                gamma_s2 = None
                final_intercept_full = None
                final_betas_full = None

            # Containers for bootstrap draws
            s1_intercepts = np.full((n_boot,), np.nan, dtype=float)
            s1_betas = np.full((n_boot, len(signature)), np.nan, dtype=float)

            if execute_two_step_fitting:
                s2_alphas = np.full((n_boot,), np.nan, dtype=float)
                s2_gammas = np.full((n_boot,), np.nan, dtype=float)
                final_intercepts = np.full((n_boot,), np.nan, dtype=float)
                final_betas = np.full((n_boot, len(signature)), np.nan, dtype=float)

            # Bootstrap
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n, endpoint=False)
                boot_df = training_df.iloc[idx].copy()

                Xb = boot_df[signature]
                yb = boot_df["range_shift"].to_numpy()

                # Fit transform within replicate (prevents leakage and matches deployment logic)
                if perform_yeo_johnson:
                    pt_b = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)
                    Xb_t = pd.DataFrame(pt_b.fit_transform(Xb), index=Xb.index, columns=signature)
                else:
                    Xb_t = Xb

                lr1_b = LinearRegression(fit_intercept=fit_intercept_stage1)
                lr1_b.fit(Xb_t, yb)

                s1_intercepts[b] = float(lr1_b.intercept_)
                s1_betas[b, :] = lr1_b.coef_.astype(float)

                if execute_two_step_fitting:
                    pred1_b = lr1_b.predict(Xb_t)

                    df_group_b = pd.DataFrame({"range_shift": yb, "pred1": pred1_b})

                    # If bootstrap sample contains only one unique range_shift, stage-2 is not identifiable.
                    if df_group_b["range_shift"].nunique() < 2:
                        continue

                    df_median_b = df_group_b.groupby("range_shift", as_index=False)["pred1"].median()
                    if len(df_median_b) < 2:
                        continue

                    lr2_b = LinearRegression(fit_intercept=fit_intercept_stage2)
                    lr2_b.fit(df_median_b["pred1"].values.reshape(-1, 1), df_median_b["range_shift"].values)

                    alpha_b = float(lr2_b.intercept_)
                    gamma_b = float(lr2_b.coef_.ravel()[0])

                    s2_alphas[b] = alpha_b
                    s2_gammas[b] = gamma_b

                    beta0_b = float(lr1_b.intercept_)
                    betas_b = lr1_b.coef_.astype(float)

                    final_intercepts[b] = alpha_b + gamma_b * beta0_b
                    final_betas[b, :] = gamma_b * betas_b

            def _pct_ci(x: np.ndarray) -> tuple:
                x = x[np.isfinite(x)]
                if x.size < max(50, int(0.1 * n_boot)):
                    return (np.nan, np.nan)
                return (float(np.percentile(x, q_lo)), float(np.percentile(x, q_hi)))

            out = {
                "bootstrap": {
                    "n_boot": int(n_boot),
                    "ci_level": float(ci_level),
                    "random_state": int(random_state),
                    "percentiles": (float(q_lo), float(q_hi)),
                },
                "stage1": {
                    "intercept_point": float(beta0_s1),
                    "intercept_ci": _pct_ci(s1_intercepts),
                    "coef_point": betas_s1.copy(),
                    "coef_ci": np.array([_pct_ci(s1_betas[:, j]) for j in range(len(signature))], dtype=float),
                    "coef_feature_order": list(signature),
                },
            }

            if execute_two_step_fitting:
                out["stage2"] = {
                    "alpha_point": float(alpha_s2),
                    "alpha_ci": _pct_ci(s2_alphas),
                    "gamma_point": float(gamma_s2),
                    "gamma_ci": _pct_ci(s2_gammas),
                }
                out["final"] = {
                    "intercept_point": float(final_intercept_full),
                    "intercept_ci": _pct_ci(final_intercepts),
                    "coef_point": final_betas_full.copy(),
                    "coef_ci": np.array([_pct_ci(final_betas[:, j]) for j in range(len(signature))], dtype=float),
                    "coef_feature_order": list(signature),
                }

            return out

        bootstrap_ci = _bootstrap_iterative_linear_ci(
            training_df=training_data,
            signature=signature,
            perform_yeo_johnson=perform_yeo_johnson,
            execute_two_step_fitting=execute_two_step_fitting,
            fit_intercept_stage1=fit_intercept_stage1,
            fit_intercept_stage2=fit_intercept_stage2,
            n_boot=bootstrap_params["n_boot"],
            ci_level=bootstrap_params["ci_level"],
            random_state=bootstrap_params["random_state"],
        )

        # Store confidence intervals in results dict (so they are logged/serialized with the run)
        testing_results["hyperparameters"].update(
            {
                "bootstrap_n": bootstrap_ci["bootstrap"]["n_boot"],
                "bootstrap_ci_level": bootstrap_ci["bootstrap"]["ci_level"],
                "bootstrap_random_state": bootstrap_ci["bootstrap"]["random_state"],
            }
        )
        testing_results["model_parameter_cis"] = bootstrap_ci

        # -------------------- Iterative Linear Model Summary (print-only) --------------------
        print("\n" + "=" * 100)
        print("Iterative Linear Model — Summary")
        print("=" * 100)
        print(f"perform_yeo_johnson     : {perform_yeo_johnson}")
        print(f"execute_two_step_fitting: {execute_two_step_fitting}")
        print(f"stage-1 fit_intercept   : {lr1.fit_intercept}")
        if execute_two_step_fitting:
            print(f"stage-2 fit_intercept   : {lr2.fit_intercept}")
        else:
            print("stage-2                : not executed")

        # --- Standard diagnostics (training) ---
        train_residuals = y_train - y_train_pred_final
        train_mae = mean_absolute_error(y_train, y_train_pred_final)
        train_med_ae = median_absolute_error(y_train, y_train_pred_final)
        train_bias = float(np.mean(train_residuals))
        train_resid_std = float(np.std(train_residuals, ddof=1))
        train_resid_q = np.percentile(train_residuals, [2.5, 25, 50, 75, 97.5])

        print("\n[Training diagnostics]")
        print(f"  RMSE             : {train_rmse:.8f}")
        print(f"  R2               : {train_r2:.8f}")
        print(f"  MAE              : {train_mae:.8f}")
        print(f"  Median AE        : {train_med_ae:.8f}")
        print(f"  Bias (mean resid): {train_bias:.8f}")
        print(f"  Residual SD      : {train_resid_std:.8f}")
        print(
            "  Residual quantiles [2.5,25,50,75,97.5]%: "
            f"[{train_resid_q[0]:.8f}, {train_resid_q[1]:.8f}, {train_resid_q[2]:.8f}, {train_resid_q[3]:.8f}, {train_resid_q[4]:.8f}]"
        )

        # --- Standard diagnostics (testing) ---
        val_residuals = y_val - y_val_pred_final
        val_mae = mean_absolute_error(y_val, y_val_pred_final)
        val_med_ae = median_absolute_error(y_val, y_val_pred_final)
        val_bias = float(np.mean(val_residuals))
        val_resid_std = float(np.std(val_residuals, ddof=1))
        val_resid_q = np.percentile(val_residuals, [2.5, 25, 50, 75, 97.5])

        print("\n[testing diagnostics]")
        print(f"  RMSE             : {val_rmse:.8f}")
        print(f"  R2               : {val_r2:.8f}")
        print(f"  MAE              : {val_mae:.8f}")
        print(f"  Median AE        : {val_med_ae:.8f}")
        print(f"  Bias (mean resid): {val_bias:.8f}")
        print(f"  Residual SD      : {val_resid_std:.8f}")
        print(
            "  Residual quantiles [2.5,25,50,75,97.5]%: "
            f"[{val_resid_q[0]:.8f}, {val_resid_q[1]:.8f}, {val_resid_q[2]:.8f}, {val_resid_q[3]:.8f}, {val_resid_q[4]:.8f}]"
        )

        # --- Coefficients of Stage-1 linear model + bootstrap CIs ---
        coef_df = (
            pd.DataFrame({"feature": signature, "beta_stage1": lr1.coef_})
            .sort_values("beta_stage1", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )

        # Map feature->CI for stage-1, aligned to signature order stored in bootstrap_ci
        s1_ci_arr = bootstrap_ci["stage1"]["coef_ci"]  # shape (p,2)
        s1_feat_order = bootstrap_ci["stage1"]["coef_feature_order"]
        s1_ci_map = {feat: (float(s1_ci_arr[i, 0]), float(s1_ci_arr[i, 1])) for i, feat in enumerate(s1_feat_order)}

        s1_int_point = float(bootstrap_ci["stage1"]["intercept_point"])
        s1_int_lo, s1_int_hi = bootstrap_ci["stage1"]["intercept_ci"]

        print("\n[Stage-1 Linear Regression Coefficients + Bootstrap CI]")
        print(
            "  Intercept (beta0): {:.8f}   CI({:.1f}%): [{:.8f}, {:.8f}]".format(
                float(lr1.intercept_),
                100.0 * bootstrap_ci["bootstrap"]["ci_level"],
                float(s1_int_lo),
                float(s1_int_hi),
            )
        )
        for j, row in coef_df.iterrows():
            feat = str(row["feature"])
            ci_lo, ci_hi = s1_ci_map.get(feat, (np.nan, np.nan))
            print(
                "  {:>2d}. {:<30s}  beta_stage1: {:+.8f}   CI({:.1f}%): [{:+.8f}, {:+.8f}]".format(
                    j + 1,
                    feat,
                    float(row["beta_stage1"]),
                    100.0 * bootstrap_ci["bootstrap"]["ci_level"],
                    float(ci_lo),
                    float(ci_hi),
                )
            )

        # --- Stage-2 calibration mapping + CI (only if executed) ---
        if execute_two_step_fitting:
            a_pt = float(bootstrap_ci["stage2"]["alpha_point"])
            a_lo, a_hi = bootstrap_ci["stage2"]["alpha_ci"]
            g_pt = float(bootstrap_ci["stage2"]["gamma_point"])
            g_lo, g_hi = bootstrap_ci["stage2"]["gamma_ci"]

            print("\n[Stage-2 Calibration (prediction adjustment) + Bootstrap CI]")
            print("  Mapping: range_shift ≈ alpha + gamma * pred1")
            print(
                "  alpha (intercept): {:.8f}   CI({:.1f}%): [{:.8f}, {:.8f}]".format(
                    float(lr2.intercept_),
                    100.0 * bootstrap_ci["bootstrap"]["ci_level"],
                    float(a_lo),
                    float(a_hi),
                )
            )
            print(
                "  gamma (slope)    : {:.8f}   CI({:.1f}%): [{:.8f}, {:.8f}]".format(
                    float(lr2.coef_.ravel()[0]),
                    100.0 * bootstrap_ci["bootstrap"]["ci_level"],
                    float(g_lo),
                    float(g_hi),
                )
            )

            # --- Combined final linear model + CI ---
            beta0_stage1 = float(lr1.intercept_)
            betas_stage1 = lr1.coef_.astype(float)
            alpha_stage2 = float(lr2.intercept_)
            gamma_stage2 = float(lr2.coef_.ravel()[0])

            final_intercept = alpha_stage2 + gamma_stage2 * beta0_stage1
            final_betas = gamma_stage2 * betas_stage1

            final_ci_arr = bootstrap_ci["final"]["coef_ci"]
            final_feat_order = bootstrap_ci["final"]["coef_feature_order"]
            final_ci_map = {feat: (float(final_ci_arr[i, 0]), float(final_ci_arr[i, 1])) for i, feat in enumerate(final_feat_order)}
            fin_int_lo, fin_int_hi = bootstrap_ci["final"]["intercept_ci"]

            final_coef_df = pd.DataFrame(
                {"feature": signature, "beta_stage1": betas_stage1, "beta_final": final_betas}
            ).sort_values("beta_final", key=np.abs, ascending=False).reset_index(drop=True)

            print("\n[Combined Final Linear Model + Bootstrap CI]")
            print("  Final model:  y ≈ final_intercept + Σ_j final_beta_j * x_j_used_by_stage1")
            if perform_yeo_johnson:
                print("  where x_j_used_by_stage1 = z_j = PowerTransformer(Yeo–Johnson, standardize=True) applied to original feature_j.")
            else:
                print("  where x_j_used_by_stage1 = raw feature_j (no transform, no standardization).")
                print("  Note: both stage-1 and stage-2 were fit with fit_intercept=False, hence intercept terms are constrained to 0.")
            print(
                "  final_intercept : {:.8f}   CI({:.1f}%): [{:.8f}, {:.8f}]".format(
                    float(final_intercept),
                    100.0 * bootstrap_ci["bootstrap"]["ci_level"],
                    float(fin_int_lo),
                    float(fin_int_hi),
                )
            )
            print("  (Reference) stage-1 intercept beta0 : {:.8f}".format(beta0_stage1))
            print("  alpha (stage-2) : {:.8f}".format(alpha_stage2))
            print("  gamma (stage-2) : {:.8f}".format(gamma_stage2))
            for j, row in final_coef_df.iterrows():
                feat = str(row["feature"])
                ci_lo, ci_hi = final_ci_map.get(feat, (np.nan, np.nan))
                print(
                    "  {:>2d}. {:<30s}  beta_stage1: {:+.8f}   beta_final: {:+.8f}   CI({:.1f}%): [{:+.8f}, {:+.8f}]".format(
                        j + 1,
                        feat,
                        float(row["beta_stage1"]),
                        float(row["beta_final"]),
                        100.0 * bootstrap_ci["bootstrap"]["ci_level"],
                        float(ci_lo),
                        float(ci_hi),
                    )
                )
        else:
            print("\n[Stage-2 Calibration]")
            print("  Not executed (execute_two_step_fitting=False).")
            print("  Final model corresponds to Stage-1 only: y ≈ beta0 + Σ_j beta_j * x_j_used_by_stage1")
            if perform_yeo_johnson:
                print("  where x_j_used_by_stage1 = z_j = PowerTransformer(Yeo–Johnson, standardize=True) applied to original feature_j.")
            else:
                print("  where x_j_used_by_stage1 = raw feature_j (no transform, no standardization).")
                print("  Note: stage-1 was fit with fit_intercept=False, hence intercept term is constrained to 0.")

        # --- Yeo–Johnson details (only if enabled) ---
        if perform_yeo_johnson and pt_ext is not None:
            print("\n[Yeo–Johnson Transformation]")
            print("  method='yeo-johnson', standardize=True")
            for feat, lam, mu, sig in zip(signature, pt_ext.lambdas_, pt_ext._scaler.mean_, pt_ext._scaler.scale_):
                print(f"  {feat:<30s}  lambda={lam: .8f}  mu_g={mu: .8f}  sigma_g={sig: .8f}")

            xz_mu = X_train_trans.mean()
            xz_sd = X_train_trans.std(ddof=1)
            print("\n  Transformed training feature moments (should be approx. mean=0, sd=1):")
            for feat in signature:
                print("   {:<30s}  mean: {:+.8f}   sd: {:.8f}".format(feat, float(xz_mu[feat]), float(xz_sd[feat])))

        print("=" * 100 + "\n")
        # ------------------ End of Iterative Linear Model Summary ------------------

    else:
        # ORIGINAL MODELLING APPROACH USING FAMILIAR:
        # Directory for storing experiment results
        experiment_dir = os.path.join(testing_path, "familiar", "experiments",
                                      feature_type, feature_selection_method, model_learner)

        # Run the experiment using the familiar framework.
        perform_familiar_experiment(
            feature_file_path=familiar_feature_table_path,
            model_learner=model_learner,
            experiment_dir=experiment_dir,
            familiar_r_file_path=familiar_r_file_path,
            signature=signature,
            experimental_design="fs + mb + ev",
            batch_id_column="cohort",
            sample_id_column="id_global",
            development_batch_id="training",
            validation_batch_id=["validation", "testing"],
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
            include_features=signature,
            hyperparameter={model_learner: {"sign_size": len(signature)}},
            skip_evaluation_elements=["auc_data", "calibration_data", "calibration_info", "confusion_matrix",
                                      "decision_curve_analyis", "ice_data", "permutation_vimp", "univariate_analysis",
                                      "model_vimp", "feature_expressions", "fs_vimp", "sample_similarity"]
        )

        # Retrieve results and predictions from the experiment.
        results, predictions = evaluate_familiar_experiment(experiment_dir)
        hyperparameters = extract_hyperparameters(experiment_dir)
        testing_results['hyperparameters'] = hyperparameters

        # Process and print model performance metrics.
        for cohort in ['development', 'testing']:
            cohort_metrics = results[cohort]
            print(f"{cohort.capitalize()} fold performance: RMSE = {cohort_metrics['rmse'][0]:.2f} "
                  f"[{cohort_metrics['rmse_ci_low'][0]:.2f}, {cohort_metrics['rmse_ci_high'][0]:.2f}], "
                  f"R2 Score = {cohort_metrics['r2_score'][0]:.2f} "
                  f"[{cohort_metrics['r2_score_ci_low'][0]:.2f}, {cohort_metrics['r2_score_ci_high'][0]:.2f}]")

            testing_results[cohort].update({
                'rmse': cohort_metrics['rmse'][0],
                'rmse_ci_low': cohort_metrics['rmse_ci_low'][0],
                'rmse_ci_high': cohort_metrics['rmse_ci_high'][0],
                'r2_score': cohort_metrics['r2_score'][0],
                'r2_score_ci_low': cohort_metrics['r2_score_ci_low'][0],
                'r2_score_ci_high': cohort_metrics['r2_score_ci_high'][0]
            })

        results = testing_results

    return results, predictions

if __name__ == "__main__":
    # Parsing command line arguments for model testing.
    parser = testing_parser("Testing")
    args = parser.parse_args()

    print("Start model testing....")
    n_cpus = multiprocessing.cpu_count()
    print(f"Number of cpus: {n_cpus}")

    # Loading data from specified file paths.
    data_table = pd.read_csv(args.data_table_path, sep=";")
    feature_table = pd.read_csv(args.feature_file_path, sep=";")

    # Define and create a temporary directory for data processing.
    temp_data_dir = os.path.join(args.testing_path, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)

    with open(args.feature_selection_file_path, 'r') as file:
        feature_selection_metrics = json.load(file)

    final_signature = feature_selection_metrics["final_signature"]["features"]
    feature_selection_method = feature_selection_metrics["feature_selection_method"]
    model_learner = feature_selection_metrics["model_learner"]
    feature_type = feature_selection_metrics["feature_type"]

    # Constructing file paths for the familiar feature table and R script.
    familiar_feature_table_path = os.path.join(temp_data_dir, f"features_{feature_type}_{feature_selection_method}_{model_learner}.csv")
    familiar_r_file_path = os.path.join(args.testing_path, f"familiar/r_files/R_file_{feature_type}_{feature_selection_method}_{model_learner}.R")

    # Creating a feature table for the FAMILIAR tool.
    familiar_feature_table = create_feature_table_for_familiar(data_table, feature_table, final_signature, familiar_feature_table_path, evaluation_phase = True)

    # Execute model testing.
    result_dict, predictions = perform_testing(args.testing_path, familiar_feature_table_path,
                                                             familiar_r_file_path, feature_type, feature_selection_method,
                                                             model_learner, final_signature)

    # Prepare predictions table and save it.
    final_predictions_table = merge_data_with_predictions(data_table, predictions)
    final_predictions_table['range_shift'] = pd.to_numeric(final_predictions_table['range_shift'])
    final_predictions_table.to_csv(args.predictions_file_path, sep=";", index=False)
    print(f"Predictions successfully saved to {args.predictions_file_path}")

    # Save the final results dictionary to a file.
    with open(args.performance_file_path, 'w') as file:
        json.dump(make_json_serializable(result_dict), file, indent=4)
        print(f"model testing results successfully saved to {args.performance_file_path}")

    # Plot the predictions.
    plot_predicted_vs_actual_range_shift(final_predictions_table, save_path=args.prediction_plot_path_training, cohort='training')
    plot_predicted_vs_actual_range_shift(final_predictions_table, save_path=args.prediction_plot_path_testing, cohort='testing')

    # Construct test path by changing only the parent directory
    val_path = Path(args.prediction_plot_path_testing)
    test_dir = val_path.parent.parent / "testing"
    test_path = test_dir / val_path.name

    plot_predicted_vs_actual_range_shift(final_predictions_table, save_path=test_path, cohort='testing')
    print("Predictions plot successfully saved!")

    # -------------------- Additional Evaluations --------------------
    # For each combination of:
    #   'cohort', 'cv_data_set', 'nose_orientation', 'proton_energy', 'mu', 'range_shift_type', 'repetition'
    # calculate and print RMSE, R2 and MRSE (with bootstrap confidence intervals).
    #
    # Ensure the grouping column exists. If 'cv_data_set' is not present but 'data_set' is, rename it.
    if 'cv_data_set' not in final_predictions_table.columns and 'data_set' in final_predictions_table.columns:
        final_predictions_table = final_predictions_table.rename(columns={'data_set': 'cv_data_set'})

    group_keys = ['cohort', 'cv_data_set', 'nose_orientation', 'proton_energy', 'mu', 'range_shift_type', 'repetition']

    def compute_mrse(df):
        """
        Computes the MRSE for a given group.
        For each unique true range_shift (air cavity thickness), the average predicted outcome is computed.
        The absolute error is then computed against the true value, and MRSE is the mean of these errors.
        """
        grouped = df.groupby('range_shift').agg({'predicted_range_shift': 'mean'}).reset_index()
        grouped['abs_error'] = np.abs(grouped['predicted_range_shift'] - grouped['range_shift'])
        return grouped['abs_error'].mean()

    def bootstrap_group_metrics(df, n_bootstrap=1000, ci=95):
        rmse_list = []
        r2_list = []
        mrse_list = []
        n = len(df)
        for _ in range(n_bootstrap):
            sample = df.sample(n=n, replace=True)
            true_vals = sample['range_shift'].values
            pred_vals = sample['predicted_range_shift'].values
            rmse_val = np.sqrt(mean_squared_error(true_vals, pred_vals))
            r2_val = r2_score(true_vals, pred_vals)
            mrse_val = compute_mrse(sample)
            rmse_list.append(rmse_val)
            r2_list.append(r2_val)
            mrse_list.append(mrse_val)
        rmse_mean = np.mean(rmse_list)
        rmse_lower = np.percentile(rmse_list, (100-ci)/2)
        rmse_upper = np.percentile(rmse_list, 100 - (100-ci)/2)

        r2_mean = np.mean(r2_list)
        r2_lower = np.percentile(r2_list, (100-ci)/2)
        r2_upper = np.percentile(r2_list, 100 - (100-ci)/2)

        mrse_mean = np.mean(mrse_list)
        mrse_lower = np.percentile(mrse_list, (100-ci)/2)
        mrse_upper = np.percentile(mrse_list, 100 - (100-ci)/2)

        return (rmse_mean, rmse_lower, rmse_upper, r2_mean, r2_lower, r2_upper, mrse_mean, mrse_lower, mrse_upper)

    print("\nAdditional Evaluations for each combination:")
    for group_values, group_df in final_predictions_table.groupby(group_keys):
        metrics = bootstrap_group_metrics(group_df, n_bootstrap=1000, ci=95)

        print("-" * 60)
        print("Group:")
        for key, value in zip(group_keys, group_values):
            print("  {}: {}".format(key, value))
        print("Metrics:")
        print("  RMSE : {:.2f} (95% CI: [{:.2f}, {:.2f}])".format(metrics[0], metrics[1], metrics[2]))
        print("  R2   : {:.2f} (95% CI: [{:.2f}, {:.2f}])".format(metrics[3], metrics[4], metrics[5]))
        print("  MRSE : {:.2f} (95% CI: [{:.2f}, {:.2f}])".format(metrics[6], metrics[7], metrics[8]))


    print("Finished!")


