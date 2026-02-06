from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback (no progress bar) if tqdm is not installed.
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# =============================================================================
# Dynamic import (unchanged)
# =============================================================================

def load_module(path: str):
    import hashlib
    import sys

    abspath = os.path.abspath(path)
    name = "user_model_module_" + hashlib.md5(abspath.encode("utf-8")).hexdigest()

    spec = importlib.util.spec_from_file_location(name, abspath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model module from: {path}")

    mod = importlib.util.module_from_spec(spec)

    # CRITICAL: register module before executing it (required by dataclasses in Py3.11)
    sys.modules[name] = mod

    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Metrics (model-agnostic; unchanged from your pipeline)
# =============================================================================

def compute_metrics(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    spot_col: str,
) -> Dict[str, float]:
    dfv = df.dropna(subset=[truth_col, pred_col, spot_col]).copy()
    if len(dfv) == 0:
        raise ValueError("No valid rows to compute metrics (missing truth/pred/spot).")

    e = (dfv[pred_col].astype(float) - dfv[truth_col].astype(float)).to_numpy()
    rmse = float(np.sqrt(np.mean(e**2)))

    dfv["error"] = e
    g = dfv.groupby(spot_col, as_index=False)

    spot_stats = g["error"].agg(
        m_s="median",
        v_s=lambda x: float(np.var(x.to_numpy(), ddof=1)) if len(x) >= 2 else np.nan,
        n="count",
    )
    spot_stats = spot_stats.dropna(subset=["v_s"]).copy()

    min_variance = 1e-6
    spot_stats["v_s"] = spot_stats["v_s"].clip(lower=min_variance)

    if len(spot_stats) == 0:
        raise ValueError("No spots with >=2 repetitions; cannot compute variance-weighted metrics.")

    w = 1.0 / spot_stats["v_s"].to_numpy()
    W = float(np.sum(w))

    wme = float(np.sum(w * spot_stats["m_s"].to_numpy()) / W)
    wrp = float(np.sqrt(np.sum(w * (spot_stats["m_s"].to_numpy() - wme) ** 2) / W))
    msp = float(np.mean(np.sqrt(spot_stats["v_s"].to_numpy())))

    return {"RMSE_mm": rmse, "WME_mm": wme, "WRP_mm": wrp, "MSP_mm": msp}


def bootstrap_metrics(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    spot_col: str,
    n_resamples: int,
    ci_percentiles: Tuple[float, float],
    random_state: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(random_state))
    spots = df[spot_col].dropna().unique()
    spots = np.asarray(spots)

    point = compute_metrics(df, truth_col, pred_col, spot_col)

    metrics_samples: Dict[str, List[float]] = {k: [] for k in point.keys()}
    for _ in tqdm(range(int(n_resamples)), desc="Bootstrapping metrics", unit="resample"):
        samp_spots = rng.choice(spots, size=len(spots), replace=True)
        dfb = pd.concat([df[df[spot_col] == s] for s in samp_spots], ignore_index=True)
        try:
            m = compute_metrics(dfb, truth_col, pred_col, spot_col)
            for k in point.keys():
                metrics_samples[k].append(float(m[k]))
        except Exception:
            continue

    out: Dict[str, Any] = {"point": point, "bootstrap": {}}
    lo, hi = float(ci_percentiles[0]), float(ci_percentiles[1])

    for k, vals in metrics_samples.items():
        if len(vals) == 0:
            out["bootstrap"][k] = {"n": 0, "ci_low": None, "ci_high": None}
            continue
        arr = np.asarray(vals, dtype=float)
        out["bootstrap"][k] = {
            "n": int(len(arr)),
            "ci_low": float(np.percentile(arr, lo)),
            "ci_high": float(np.percentile(arr, hi)),
        }
    return out


# =============================================================================
# Parsing utilities
# =============================================================================

def _parse_time_window_bins(s: Optional[str]) -> Optional[List[int]]:
    if s in (None, "", "None", "null"):
        return None
    ss = str(s).strip()

    # 1) Try YAML
    try:
        import yaml

        obj = yaml.safe_load(ss)
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            return [int(obj[0]), int(obj[1])]
        if obj is None:
            return None
    except Exception:
        pass

    # 2) Fallback: comma-separated "750,1728"
    if "," in ss:
        a, b = ss.split(",", 1)
        return [int(a.strip()), int(b.strip())]

    raise ValueError(f"Unsupported time_window_bins format: {s!r}")


def _parse_bootstrap_cfg(s: str) -> Dict[str, Any]:
    if s in (None, "", "None", "null", "{}"):
        return {}
    try:
        return json.loads(str(s).replace("'", '"'))
    except Exception:
        return eval(s)  # noqa: S307


# =============================================================================
# Core feature computation (generic, no feature name hard-coding)
# =============================================================================

def _compute_feature_and_ref_median(
    feature_name: str,
    feature_spec: Any,
    ctx: Any,
    measured_paths: List[str],
    reference_paths_per_row: List[List[str]],
    cache_by_feature: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute:
      - feature values on measured files (per row)
      - feature_ref_median values (median over references per row)

    Caching is per feature and per file path.
    Returns:
      measured_values (n_rows,),
      ref_medians (n_rows,),
      per_ref_path_value (dict path->value) for all reference paths encountered
    """
    n = len(measured_paths)
    measured_vals = np.full(n, np.nan, dtype=float)
    ref_medians = np.full(n, np.nan, dtype=float)

    per_path_cache = cache_by_feature.setdefault(feature_name, {})

    # --- helper to get scalar for a file path with caching ---
    def get_scalar(path: str) -> float:
        if path in per_path_cache:
            return per_path_cache[path]
        v = float(feature_spec.compute_scalar(path, ctx))
        per_path_cache[path] = v
        return v

    # Precompute all unique reference paths once
    unique_refs = sorted({rp for rps in reference_paths_per_row for rp in rps})
    ref_vals_by_path: Dict[str, float] = {}
    for rp in unique_refs:
        ref_vals_by_path[rp] = get_scalar(rp)

    # Compute measured + per-row ref median
    for i in range(n):
        mp = measured_paths[i]
        measured_vals[i] = get_scalar(mp)

        rps = reference_paths_per_row[i]
        vals = [ref_vals_by_path[rp] for rp in rps]
        ref_medians[i] = float(np.median(np.asarray(vals, dtype=float)))

    return measured_vals, ref_medians, ref_vals_by_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--eval_table", required=True)
    ap.add_argument("--predictions_csv", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--features_csv", required=True)

    ap.add_argument("--model_module", required=True)
    ap.add_argument("--model_name", required=True)

    # Feature extraction parameters (passed into model FeatureContext)
    ap.add_argument("--smoothing_sigma_bins", type=float, required=True)
    ap.add_argument("--smoothed_repeat_scale", type=int, required=True)
    ap.add_argument("--time_window_bins", default=None)

    # Optional external labels override
    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--labels_sep", default=",")

    # Metrics settings
    ap.add_argument("--compute_metrics", default="true")
    ap.add_argument(
        "--min_n_protons_agg_for_metrics",
        type=float,
        default=None,
        help="If set, exclude rows with n_protons_agg < threshold from metrics computation.",
    )
    ap.add_argument(
        "--n_protons_agg_col",
        default="n_protons_agg",
        help="Column name containing aggregated proton counts used for metrics filtering.",
    )
    ap.add_argument(
        "--pred_to_geo_scale_for_metrics",
        type=float,
        default=1.0,
        help="Scale factor applied to truth and prediction ONLY for metrics calculation.",
    )
    ap.add_argument("--bootstrap_cfg", default="{}")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    t0 = time.time()
    logger.info("Starting model evaluation and feature extraction.")

    time_window = _parse_time_window_bins(args.time_window_bins)
    compute_metrics_flag = str(args.compute_metrics).lower() in ("1", "true", "yes")
    bootstrap_cfg = _parse_bootstrap_cfg(args.bootstrap_cfg)

    if time_window is not None:
        logger.info("time_window_bins=%s", time_window)
    logger.info("compute_metrics=%s", compute_metrics_flag)
    if bootstrap_cfg:
        logger.info("bootstrap_cfg=%s", bootstrap_cfg)

    # -------------------------------------------------------------------------
    # Read eval table (+ optional labels)
    # -------------------------------------------------------------------------
    logger.info("Reading eval_table: %s", args.eval_table)
    df = pd.read_csv(args.eval_table)
    logger.info("Loaded eval_table with %d rows and %d columns.", len(df), len(df.columns))

    if args.labels_csv not in (None, "", "None", "null"):
        logger.info("Merging labels from: %s", args.labels_csv)
        labels = pd.read_csv(args.labels_csv, sep=args.labels_sep)
        if "sample_id" not in labels.columns:
            raise ValueError("labels_csv must contain column 'sample_id'.")
        df = df.merge(labels, on="sample_id", how="left", suffixes=("", "_lbl"))
        logger.info("After labels merge: %d rows and %d columns.", len(df), len(df.columns))

    if "measured_path" not in df.columns or "reference_paths_json" not in df.columns:
        raise ValueError("eval_table must contain columns 'measured_path' and 'reference_paths_json'.")

    measured_paths: List[str] = df["measured_path"].astype(str).tolist()
    reference_paths_per_row: List[List[str]] = [
        json.loads(s) if isinstance(s, str) else list(s)
        for s in df["reference_paths_json"].tolist()
    ]
    logger.info("Parsed measured/reference paths: n_measured=%d", len(measured_paths))

    # -------------------------------------------------------------------------
    # Load model module
    # -------------------------------------------------------------------------
    logger.info("Loading model module: %s", args.model_module)
    mod = load_module(args.model_module)

    for attr in ("MODEL_REGISTRY", "FEATURE_REGISTRY", "ModelSpec", "FeatureContext", "ref_median_name"):
        if not hasattr(mod, attr):
            raise RuntimeError(f"Model module must define '{attr}'.")

    model_registry = getattr(mod, "MODEL_REGISTRY")
    feature_registry = getattr(mod, "FEATURE_REGISTRY")
    ref_median_name = getattr(mod, "ref_median_name")

    if args.model_name not in model_registry:
        raise RuntimeError(f"Unknown model '{args.model_name}'. Available: {list(model_registry.keys())}")
    model_spec = model_registry[args.model_name]
    logger.info("Using model: %s", args.model_name)

    # Validate base features exist
    base_features: List[str] = list(getattr(model_spec, "base_features"))
    missing = [f for f in base_features if f not in feature_registry]
    if missing:
        raise RuntimeError(
            f"Model '{args.model_name}' requires missing base_features: {missing}. "
            f"Available: {sorted(feature_registry.keys())}"
        )
    logger.info("Base features (%d): %s", len(base_features), base_features)

    # -------------------------------------------------------------------------
    # Build feature context (owned by model module)
    # -------------------------------------------------------------------------
    ctx = mod.FeatureContext(
        args=args,
        time_window_bins=time_window,
        caches={},  # shared caches (optional usage by feature implementations)
    )

    # -------------------------------------------------------------------------
    # Compute features + per-feature reference medians (generic)
    # -------------------------------------------------------------------------
    logger.info("Computing features and reference medians.")
    cache_by_feature: Dict[str, Dict[str, float]] = {}
    computed_features: Dict[str, np.ndarray] = {}

    # For features.csv reference rows: map reference path -> its reference-set key (first occurrence wins)
    refset_key_for_path: Dict[str, Tuple[str, ...]] = {}
    for refs in reference_paths_per_row:
        key = tuple(sorted(map(str, refs)))
        for rp in refs:
            refset_key_for_path.setdefault(str(rp), key)

    # Also keep per-feature per-reference-file values for export
    per_feature_refvals: Dict[str, Dict[str, float]] = {}

    for f in tqdm(base_features, desc="Computing base features", unit="feature"):
        spec = feature_registry[f]
        meas_vals, ref_meds, ref_vals_by_path = _compute_feature_and_ref_median(
            feature_name=f,
            feature_spec=spec,
            ctx=ctx,
            measured_paths=measured_paths,
            reference_paths_per_row=reference_paths_per_row,
            cache_by_feature=cache_by_feature,
        )
        computed_features[f] = meas_vals
        computed_features[ref_median_name(f)] = ref_meds
        per_feature_refvals[f] = ref_vals_by_path

    logger.info("Computed %d feature columns (incl. ref medians).", len(computed_features))

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------
    logger.info("Running prediction.")
    y_pred = model_spec.predict(computed_features)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != (len(df),):
        raise ValueError(f"Model returned predictions of shape {y_pred.shape}, expected {(len(df),)}.")

    # -------------------------------------------------------------------------
    # Write predictions CSV (eval table + features + prediction)
    # -------------------------------------------------------------------------
    out = df.copy()
    out["predicted_range_shift_mm"] = y_pred

    # Attach computed feature columns (ensure no collisions)
    for name, arr in computed_features.items():
        if name in out.columns:
            raise ValueError(
                f"Computed feature column '{name}' collides with an existing column in eval_table."
            )
        out[name] = np.asarray(arr, dtype=float)

    os.makedirs(os.path.dirname(args.predictions_csv) or ".", exist_ok=True)
    out.to_csv(args.predictions_csv, index=False)
    logger.info("Wrote predictions CSV to %s", args.predictions_csv)

    # -------------------------------------------------------------------------
    # Write features CSV (measured rows + reference rows) (generic)
    # -------------------------------------------------------------------------
    logger.info("Writing features table (measured + reference diagnostic rows).")
    feature_cols = list(computed_features.keys())

    records: List[Dict[str, Any]] = []

    # measured rows
    for i, row in tqdm(out.iterrows(), total=len(out), desc="Assembling measured feature rows", unit="row"):
        rec: Dict[str, Any] = {
            "set": "measured",
            "sample_id": row.get("sample_id", f"measured_{i}"),
            "path": row["measured_path"],
        }
        for c in feature_cols:
            rec[c] = float(out.loc[i, c]) if pd.notna(out.loc[i, c]) else np.nan
        records.append(rec)

    # reference rows (diagnostic)
    ref_paths_all = sorted(refset_key_for_path.keys())

    # Precompute ref medians per reference-set key for each feature
    # using the measured-row values as authoritative (since they were computed per row).
    refset_medians: Dict[str, Dict[Tuple[str, ...], float]] = {ref_median_name(f): {} for f in base_features}
    for i, refs in enumerate(reference_paths_per_row):
        key = tuple(sorted(map(str, refs)))
        for f in base_features:
            med_col = ref_median_name(f)
            refset_medians[med_col].setdefault(key, float(out.loc[i, med_col]))

    for rp in tqdm(ref_paths_all, desc="Assembling reference diagnostic rows", unit="file"):
        key = refset_key_for_path[rp]
        rec: Dict[str, Any] = {
            "set": "reference",
            "sample_id": f"ref::{os.path.basename(rp)}",
            "path": rp,
        }

        # base feature values on that reference file
        for f in base_features:
            rec[f] = float(per_feature_refvals[f][rp])

        # ref medians (constant per reference-set)
        for f in base_features:
            med_col = ref_median_name(f)
            rec[med_col] = float(refset_medians[med_col].get(key, np.nan))

        records.append(rec)

    features_df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(args.features_csv) or ".", exist_ok=True)
    features_df.to_csv(args.features_csv, index=False)
    logger.info("Wrote features CSV to %s", args.features_csv)

    # -------------------------------------------------------------------------
    # Metrics (optional; requires ground truth)
    # -------------------------------------------------------------------------
    logger.info("Preparing metrics output.")
    metrics_out: Dict[str, Any] = {"model_name": args.model_name}

    metrics_out["metric_filtering"] = {
        "enabled": bool(args.min_n_protons_agg_for_metrics is not None),
        "n_protons_agg_col": str(args.n_protons_agg_col),
        "min_n_protons_agg_for_metrics": (
            float(args.min_n_protons_agg_for_metrics) if args.min_n_protons_agg_for_metrics is not None else None
        ),
    }

    scale_geo = float(args.pred_to_geo_scale_for_metrics)
    metrics_out["geometrical_range_shift_scaling_for_metrics"] = {
        "enabled": (abs(scale_geo - 1.0) > 0.0),
        "method": "scale",
        "scale_factor": scale_geo,
        "applied_to_truth_column": "range_shift_mm",
        "applied_to_prediction_column": "predicted_range_shift_mm",
        "effective_truth_used_in_metrics": None,
        "effective_prediction_used_in_metrics": None,
    }

    if compute_metrics_flag:
        logger.info("Computing performance metrics.")
        truth_col = "range_shift_mm"
        pred_col = "predicted_range_shift_mm"

        if truth_col not in out.columns:
            raise ValueError(
                f"Truth column '{truth_col}' missing. Provide it via filename_regex or labels_csv."
            )

        out_metrics = out.copy()
        filt_info = {"n_rows_before": int(len(out_metrics)), "n_rows_after": None, "n_removed": None}

        if args.min_n_protons_agg_for_metrics is not None:
            col = str(args.n_protons_agg_col)
            if col not in out_metrics.columns:
                raise ValueError(
                    f"Requested metrics filter uses column '{col}', but it is missing from predictions table. "
                    "Ensure create_eval_table.py writes it and it is preserved into the eval_table."
                )
            thr = float(args.min_n_protons_agg_for_metrics)
            mask = out_metrics[col].astype(float) >= thr
            out_metrics = out_metrics.loc[mask].copy()

        filt_info["n_rows_after"] = int(len(out_metrics))
        filt_info["n_removed"] = int(filt_info["n_rows_before"] - filt_info["n_rows_after"])
        metrics_out["metric_filtering"].update(filt_info)

        if len(out_metrics) == 0:
            raise ValueError("No rows left for metrics after applying min_n_protons_agg_for_metrics filter.")

        spot_group_cols = [
            "nose_orientation",
            "proton_energy",
            "mu",
            "detector",
            "layer",
            "spot_id",
        ]

        missing_cols = [c for c in spot_group_cols if c not in out_metrics.columns]
        if missing_cols:
            raise ValueError(
                "Spot grouping keys missing from table: "
                f"{missing_cols}. Available columns (first 50): {list(out_metrics.columns)[:50]}"
            )

        if out_metrics[spot_group_cols].isna().any().any():
            bad = out_metrics.loc[out_metrics[spot_group_cols].isna().any(axis=1), spot_group_cols].head(10)
            raise ValueError(
                "Spot grouping keys contain NaNs; cannot define spot identity reliably. "
                f"First problematic rows (up to 10):\n{bad.to_string(index=False)}"
            )

        spot_col = "plan_spot_id"
        spot_mi = pd.MultiIndex.from_frame(out_metrics[spot_group_cols])
        out_metrics[spot_col] = pd.factorize(spot_mi, sort=False)[0].astype(int)

        metrics_out["spot_grouping_for_metrics"] = {
            "columns": spot_group_cols,
            "derived_spot_id_column": spot_col,
            "method": "factorize_multiindex",
        }

        # Scale truth + prediction to geo units for metrics only (if requested)
        truth_col_metrics = truth_col
        pred_col_metrics = pred_col

        if abs(scale_geo - 1.0) > 0.0:
            truth_geo_col = f"{truth_col}__geo"
            pred_geo_col = f"{pred_col}__geo"
            out_metrics[truth_geo_col] = out_metrics[truth_col].astype(float) * scale_geo
            out_metrics[pred_geo_col] = out_metrics[pred_col].astype(float) * scale_geo
            truth_col_metrics = truth_geo_col
            pred_col_metrics = pred_geo_col

        metrics_out["geometrical_range_shift_scaling_for_metrics"]["effective_truth_used_in_metrics"] = truth_col_metrics
        metrics_out["geometrical_range_shift_scaling_for_metrics"]["effective_prediction_used_in_metrics"] = pred_col_metrics

        if bootstrap_cfg.get("enabled", False):
            metrics_out["performance"] = bootstrap_metrics(
                out_metrics,
                truth_col=truth_col_metrics,
                pred_col=pred_col_metrics,
                spot_col=spot_col,
                n_resamples=int(bootstrap_cfg.get("n_resamples", 1000)),
                ci_percentiles=tuple(bootstrap_cfg.get("ci_percentiles", [2.5, 97.5])),
                random_state=int(bootstrap_cfg.get("random_state", 42)),
            )
        else:
            metrics_out["performance"] = {
                "point": compute_metrics(out_metrics, truth_col_metrics, pred_col_metrics, spot_col)
            }
        logger.info("Metrics computed.")
    else:
        metrics_out["performance"] = None
        metrics_out["metric_filtering"].update(
            {"n_rows_before": int(len(out)), "n_rows_after": None, "n_removed": None}
        )
        logger.info("Metrics computation disabled by flag.")

    os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Wrote metrics JSON to %s", args.metrics_json)
    logger.info("Done in %.2f s.", time.time() - t0)


if __name__ == "__main__":
    main()
