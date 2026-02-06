# -*- coding: utf-8 -*-
"""
feature_normalization.py

Reference-based feature normalisation and (optional) detector aggregation.

Workflow
--------
1) Load:
     - data_path_table_main (df_main): unaggregated per-detector table with geometry/meta-data
     - data_path_table      (df_agg) : aggregated table (e.g., per-spot/per-layer) used as target rows
     - feature_table        (df_feat): features per file_path_proc
2) Normalise each feature relative to spatially matched reference measurements within each
   physical group (nose_orientation, proton_energy, mu, detector, layer).
3) Aggregate normalised features onto df_agg rows by matching on a set of columns.
4) Feature filtering: for each feature, retain only the _abs or _rel variant with the higher
   absolute Pearson correlation with range_shift on the training subset.

"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pmma.cmd_args import feature_normalization_parser
from pmma.visulisation_methods import save_features_plot

# -----------------------------------------------------------------------------
# Logging / warnings
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Keep original intent: suppress pandas performance warnings.
pd.options.mode.chained_assignment = None
import warnings  # noqa: E402

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def assert_no_nans(df: pd.DataFrame, *, step: str) -> None:
    """
    Fail hard if `df` contains NaNs and provide diagnostic context.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    step:
        Human-readable label of the pipeline stage.
    """
    if not df.isna().any().any():
        return

    nan_cols = df.columns[df.isna().any()].tolist()

    # Choose an identifier column if present
    if "id_global" in df.columns:
        id_col = "id_global"
    elif "file_path_proc" in df.columns:
        id_col = "file_path_proc"
    else:
        id_col = df.columns[0]

    bad_rows_series = df.loc[df.isna().any(axis=1), id_col]
    n_bad_rows = int(len(bad_rows_series))
    n_rows_total = int(len(df))
    n_rows_rel = (n_bad_rows / n_rows_total * 100.0) if n_rows_total > 0 else 0.0
    bad_rows = bad_rows_series.head(20).tolist()

    raise AssertionError(
        f"NaN values detected after step '{step}'.\n"
        f" - Columns containing NaNs : {nan_cols}\n"
        f" - Example {id_col} values : {bad_rows} (max 20 shown)\n"
        f" - {n_bad_rows} rows ({n_rows_rel:.2f} %) contain NaNs\n"
        f"Please inspect upstream computations for these rows/columns."
    )


# -----------------------------------------------------------------------------
# 1) Single-feature reference-based normalisation (worker)
# -----------------------------------------------------------------------------
def _normalize_feature_group(
    job: Tuple[str, Tuple[Any, ...], np.ndarray],
    df_main: pd.DataFrame,
    df_feat: pd.DataFrame,
    n_refs: int = 8,
    tol_xy_mm: float = 1.5,
) -> pd.DataFrame:
    """
    Worker for a single (feature, group) normalisation job.

    Parameters
    ----------
    job:
        (feature_name, key_vals, group_index)
    df_main:
        Full df_main DataFrame.
    df_feat:
        Full df_feat DataFrame.
    n_refs:
        Intended number of reference measurements used (see note below).
    tol_xy_mm:
        Spatial proximity window in mm for reference selection.

    Behaviour preserved
    -------------------
    - Only rows with cohort != "reference" are normalised and returned.
    - Reference candidates:
        * if any cohort == "reference" in the group -> those rows (and additionally range_shift_type == "grs")
        * otherwise -> rows with range_shift_type == "grs"
      Further constrained by:
        original_range_shift == sample.reference_range_shift
        and |Δx| <= tol_xy_mm and |Δy| <= tol_xy_mm.

    Notes on original behaviour
    ---------------------------
    The original code uses:
        if len(ref_vals) != n_refs and len(ref_vals) >= 3:
            ref_vals = np.random.choice(ref_vals, size=n_refs, replace=False)
    This can raise if len(ref_vals) < n_refs. This behaviour is preserved intentionally.
    """
    feature_name, key_vals, group_index = job
    gdf = df_main.loc[group_index]

    # Map file_path_proc -> feature value (rebuilt per worker; behaviour preserved)
    feat_series = df_feat.set_index("file_path_proc")[feature_name]

    # Subset to samples only (exclude explicit reference cohort)
    if "cohort" in gdf.columns:
        gdf_samples = gdf[gdf["cohort"] != "reference"].copy()
    else:
        gdf_samples = gdf.copy()

    # Prepare result table for sample rows only
    partial_df = gdf_samples[["id_global"]].copy().set_index("id_global")
    partial_df[f"{feature_name}_abs"] = np.nan
    partial_df[f"{feature_name}_rel"] = np.nan

    # Determine reference candidates
    if "cohort" in gdf.columns and (gdf["cohort"] == "reference").any():
        ref_candidates = gdf[gdf["cohort"] == "reference"].copy()
        ref_candidates = ref_candidates[ref_candidates["range_shift_type"] == "grs"]
    else:
        ref_candidates = gdf[gdf["range_shift_type"] == "grs"].copy()

    if ref_candidates.empty:
        logger.info(
            f"No reference candidates for key={key_vals}; "
            f"rows in this group remain unnormalised for '{feature_name}'."
        )
        return partial_df.reset_index()

    ref_xy = ref_candidates[["XCoord", "YCoord"]].to_numpy(dtype=float)
    ref_original_rs = ref_candidates["original_range_shift"].to_numpy(dtype=float)
    ref_file_paths = ref_candidates["file_path_proc"].to_numpy()

    if gdf_samples.empty:
        return partial_df.reset_index()

    for _, row in gdf_samples.iterrows():
        sample_id = row["id_global"]
        sample_fp = row["file_path_proc"]
        sample_x = float(row["XCoord"])
        sample_y = float(row["YCoord"])
        ref_rs_val = float(row["reference_range_shift"])

        # Sample feature value
        try:
            sample_val = float(feat_series.loc[sample_fp])
        except KeyError as e:
            raise KeyError(
                f"Feature '{feature_name}' not available for file_path_proc='{sample_fp}' "
                f"(id_global={sample_id})."
            ) from e

        if np.isnan(sample_val):
            raise ValueError(
                f"Feature '{feature_name}' is NaN for id_global={sample_id} "
                f"(file_path_proc='{sample_fp}')."
            )

        # Reference selection: match range shift and spatial window
        same_rs_mask = ref_original_rs == ref_rs_val
        if not np.any(same_rs_mask):
            logger.info(
                f"No reference rows with original_range_shift={ref_rs_val} "
                f"for id_global={sample_id}, key={key_vals}."
            )
            continue

        ref_xy_same_rs = ref_xy[same_rs_mask]
        ref_fp_same_rs = ref_file_paths[same_rs_mask]

        dx = np.abs(ref_xy_same_rs[:, 0] - sample_x)
        dy = np.abs(ref_xy_same_rs[:, 1] - sample_y)
        in_win = (dx <= tol_xy_mm) & (dy <= tol_xy_mm)

        if not in_win.any():
            logger.info(
                f"No reference within {tol_xy_mm:.2f} mm for id_global={sample_id}; "
                f"key={key_vals}, reference_range_shift={ref_rs_val}."
            )
            continue

        # Unique reference paths within spatial window
        ref_paths_unique = np.unique(ref_fp_same_rs[in_win])

        try:
            ref_vals = feat_series.loc[ref_paths_unique].astype(float).to_numpy()
        except KeyError as e:
            raise KeyError(
                f"Missing feature '{feature_name}' for one or more reference file_path_proc "
                f"entries used for id_global={sample_id}: {e}"
            ) from e

        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if ref_vals.size == 0:
            logger.info(
                f"All reference values NaN/inf for id_global={sample_id}, "
                f"key={key_vals}, reference_range_shift={ref_rs_val}."
            )
            continue

        # Preserve original selection logic (may raise when ref_vals.size < n_refs)
        if len(ref_vals) != n_refs and len(ref_vals) >= 3:
            ref_vals = np.random.choice(ref_vals, size=n_refs, replace=False)

        ref_val = float(np.median(ref_vals))
        if ref_val == 0.0:
            logger.info(
                f"For id_global={sample_id} median reference value == 0 for feature '{feature_name}', "
                f"key={key_vals}."
            )
            ref_val = 1e-4  # avoid division by zero

        abs_diff = sample_val - ref_val
        rel_diff = abs_diff / ref_val

        if np.isnan(rel_diff):
            raise ValueError(
                f"Relative difference for '{feature_name}' is NaN for id_global={sample_id} "
                f"(sample_val={sample_val}, ref_val={ref_val})."
            )

        partial_df.loc[sample_id, [f"{feature_name}_abs", f"{feature_name}_rel"]] = (abs_diff, rel_diff)

    return partial_df.reset_index()


# -----------------------------------------------------------------------------
# 2) Parallel normalisation (no aggregation)
# -----------------------------------------------------------------------------
def normalize_features_no_aggregation_parallel(
    df_main: pd.DataFrame,
    df_feat: pd.DataFrame,
    threads: int = 4,
    n_refs: int = 8,
) -> pd.DataFrame:
    """
    Perform reference-based normalisation for all features in df_feat, in parallel over
    (feature, group) jobs.

    Returns
    -------
    pd.DataFrame
        Columns:
          id_global, feat1_abs, feat1_rel, feat2_abs, feat2_rel, ...
        One row per id_global in df_main.
    """
    feat_cols = [c for c in df_feat.columns if c != "file_path_proc"]
    if not feat_cols:
        raise ValueError("No feature columns found in df_feat (only 'file_path_proc' present).")

    required_cols = [
        "id_global",
        "file_path_proc",
        "XCoord",
        "YCoord",
        "nose_orientation",
        "proton_energy",
        "mu",
        "detector",
        "layer",
        "range_shift_type",
        "original_range_shift",
        "reference_range_shift",
    ]
    missing = [c for c in required_cols if c not in df_main.columns]
    if missing:
        raise KeyError(f"df_main is missing required columns: {missing}")

    if "file_path_proc" not in df_feat.columns:
        raise KeyError("df_feat must contain a 'file_path_proc' column.")

    # Group by physical key
    key_cols = ["nose_orientation", "proton_energy", "mu", "detector", "layer"]
    grouped = list(df_main.groupby(key_cols, sort=False))

    # Build jobs (feature, group)
    jobs: List[Tuple[str, Tuple[Any, ...], np.ndarray]] = []
    for feature_name in feat_cols:
        for key_vals, gdf in grouped:
            jobs.append((feature_name, key_vals, gdf.index.to_numpy()))

    if not jobs:
        raise ValueError("No (feature, group) jobs could be constructed – check df_main/df_feat.")

    n_workers = min(int(threads), len(jobs))

    worker_func = partial(_normalize_feature_group, df_main=df_main, df_feat=df_feat, n_refs=n_refs)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        partial_dfs = list(
            tqdm(
                executor.map(worker_func, jobs),
                total=len(jobs),
                desc="Parallel feature+group normalization",
            )
        )

    # Assemble final output without merges (avoid suffixes)
    all_ids = df_main["id_global"].values
    real_features_normalized = pd.DataFrame({"id_global": all_ids}).set_index("id_global")

    for feature_name in feat_cols:
        real_features_normalized[f"{feature_name}_abs"] = np.nan
        real_features_normalized[f"{feature_name}_rel"] = np.nan

    for part_df in partial_dfs:
        if part_df.empty:
            continue
        part_df = part_df.set_index("id_global")
        common_cols = [c for c in part_df.columns if c in real_features_normalized.columns]
        if common_cols:
            real_features_normalized.update(part_df[common_cols])

    return real_features_normalized.reset_index()


# -----------------------------------------------------------------------------
# 3) Parallel aggregation (over data_path_table)
# -----------------------------------------------------------------------------
def _aggregate_chunk(
    chunk_agg: pd.DataFrame,
    df_main: pd.DataFrame,
    df_main_normed: pd.DataFrame,
    match_cols: List[str],
    normalized_cols: List[str],
) -> pd.DataFrame:
    """
    Worker for aggregation of a chunk of df_agg rows.

    For each row in chunk_agg:
      - find matching rows in df_main using match_cols equality,
      - collect corresponding normalised features (df_main_normed),
      - compute median across matched IDs for each normalised feature.
    """
    results: List[Dict[str, Any]] = []
    df_main_normed_idx = df_main_normed.set_index("id_global")

    for _, row_agg in chunk_agg.iterrows():
        agg_id = row_agg["id_global"]

        submask = pd.Series(True, index=df_main.index)
        for col in match_cols:
            submask &= df_main[col] == row_agg[col]

        matched_df = df_main.loc[submask]
        if matched_df.empty:
            raise ValueError(
                "No matching rows in data_path_table_main for aggregator row:\n"
                f"{row_agg.to_dict()}"
            )

        matched_ids = matched_df["id_global"].unique()
        sub_normed = df_main_normed_idx.loc[matched_ids, normalized_cols]
        if isinstance(sub_normed, pd.Series):
            sub_normed = sub_normed.to_frame().T

        if sub_normed.isna().any().any():
            raise ValueError(
                f"NaNs found in normalized features for aggregator row {agg_id}, "
                f"matching IDs {matched_ids}"
            )

        out_dict: Dict[str, Any] = {"id_global": agg_id}
        out_dict.update(sub_normed.median(axis=0).to_dict())
        results.append(out_dict)

    return pd.DataFrame(results)


def build_final_feature_table_parallel(
    df_agg: pd.DataFrame,
    df_main: pd.DataFrame,
    df_main_normed: pd.DataFrame,
    match_cols: List[str],
    threads: int = 4,
) -> pd.DataFrame:
    """
    Parallel aggregation driver:
      1) Split df_agg into chunks,
      2) Aggregate each chunk in a worker,
      3) Concatenate, and validate no NaNs.
    """
    normalized_cols = [c for c in df_main_normed.columns if c != "id_global"]

    chunk_size = int(np.ceil(len(df_agg) / max(1, threads)))
    chunks = [df_agg.iloc[i : i + chunk_size] for i in range(0, len(df_agg), chunk_size)]

    worker_func = partial(
        _aggregate_chunk,
        df_main=df_main,
        df_main_normed=df_main_normed,
        match_cols=match_cols,
        normalized_cols=normalized_cols,
    )

    results: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=int(threads)) as executor:
        for partial_res in tqdm(executor.map(worker_func, chunks), total=len(chunks), desc="Parallel aggregation"):
            results.append(partial_res)

    final_df = pd.concat(results, ignore_index=True)
    if final_df.isna().any().any():
        raise ValueError("Unexpected NaNs in final aggregated DataFrame.")

    return final_df


# -----------------------------------------------------------------------------
# 4) Main workflow with feature filtering
# -----------------------------------------------------------------------------
def normalize_features(
    data_path_table_main: str,
 #   data_path_table: str,
    feature_table_path: str,
    output_file_path: str,
    reference: str = "ref",
    n_refs: int = 8,
    threads: int = 4,
) -> None:
    """
    Overall workflow:
      1) Load CSV inputs.
      2) Reference-based normalisation (parallel, no aggregation).
      3) Aggregate normalised features onto df_agg rows (parallel).
      4) Save.
      5) Feature filtering: for each feature, compare Pearson correlation of its
         _abs and _rel variants with training set range_shift, and retain the
         variant with higher |correlation|.
      6) Save filtered table (same output path; behaviour preserved).
    """
    print("[1/6] Loading CSV data...")
    df_main = pd.read_csv(data_path_table_main, sep=";")
    df_agg = pd.read_csv(data_path_table_main, sep=";")

    # Preserve cohort filtering (training/validation/testing only)
    df_agg = df_agg.loc[
        (df_agg["cohort"] == "training")
        | (df_agg["cohort"] == "validation")
        | (df_agg["cohort"] == "testing"),
        :,
    ]

    df_feat = pd.read_csv(feature_table_path, sep=";")

    assert_no_nans(df_feat, step="Feature df reading")
    assert_no_nans(df_agg, step="Data table reading")

    print("[2/6] Parallel reference-based normalization of df_main...")
    df_main_normed = normalize_features_no_aggregation_parallel(
        df_main=df_main,
        df_feat=df_feat,
        threads=threads,
        n_refs=n_refs,
    )
    print("   Done. Columns in df_main_normed:", list(df_main_normed.columns))

    print("[3/6] Parallel aggregation with data_path_table...")
    match_cols = [
        "range_shift",
        "cohort",
        "nose_orientation",
        "proton_energy",
        "spot_id",
        "layer",
        "mu",
        "repetition",
    ]

    if len(np.unique(df_main["detector"].values)) == 1:
        print("No detectors to aggregate. Skip!")
        final_df = df_main_normed.copy()
    else:
        final_df = build_final_feature_table_parallel(
            df_agg=df_agg,
            df_main=df_main,
            df_main_normed=df_main_normed,
            match_cols=match_cols,
            threads=threads,
        )

    print("   Aggregation done. Final shape:", final_df.shape)

    print("[4/6] Saving final table to disk...")
    final_df.to_csv(output_file_path, sep=";", index=False)
    print(f"   Final table saved to: {output_file_path}")

    # -------------------------------------------------------------------------
    # 5) Feature filtering: retain most correlated abs/rel variant per feature
    # -------------------------------------------------------------------------
    print("[5/6] Filtering features based on Pearson correlation with range_shift in training set...")

    df_train = df_main[df_main["cohort"] == "training"][["id_global", "range_shift"]].copy()
    df_train["range_shift"] = pd.to_numeric(df_train["range_shift"], errors="coerce")

    final_train = pd.merge(final_df, df_train, on="id_global", how="inner")
    feat_cols = [c for c in df_feat.columns if c != "file_path_proc"]

    feature_choices: Dict[str, str] = {}
    for feature in feat_cols:
        abs_col = f"{feature}_abs"
        rel_col = f"{feature}_rel"

        if abs_col not in final_train.columns or rel_col not in final_train.columns:
            print(f"Warning: Missing columns for feature {feature}. Skipping correlation test.")
            continue

        abs_corr = final_train[abs_col].corr(final_train["range_shift"], method="pearson")
        rel_corr = final_train[rel_col].corr(final_train["range_shift"], method="pearson")
        print(f"Feature: {feature} -> {abs_col}: {abs_corr:.2f}, {rel_col}: {rel_corr:.2f}")

        if abs(abs_corr) >= abs(rel_corr):
            selected_variant = abs_col
            drop_variant = rel_col
        else:
            selected_variant = rel_col
            drop_variant = abs_col

        feature_choices[feature] = selected_variant

        if drop_variant in final_df.columns:
            final_df.drop(columns=[drop_variant], inplace=True)

    selected_columns = ["id_global"] + [feature_choices[f] for f in feat_cols if f in feature_choices]
    final_df = final_df[selected_columns]

    # Preserve behaviour: drop rows with NaNs and overwrite output path
    final_df = final_df.dropna()
    assert_no_nans(final_df, step="feature filtering")

    final_df.to_csv(output_file_path, sep=";", index=False)
    print(f"   Filtered final feature table saved to: {output_file_path}")

    print("[6/6] All processing steps completed successfully!")


# -----------------------------------------------------------------------------
# 5) CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = feature_normalization_parser("Feature Normalization")
    parser.add_argument(
        "--threads",
        type=int,
        default=28,
        help="Number of parallel workers for feature normalization.",
    )
    parser.add_argument(
        "--n_refs",
        type=int,
        default=8,
        help="Number of reference measurements used",
    )
    args = parser.parse_args()

    # Respect SLURM allocation if present
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            args.threads = max(1, int(slurm_cpus))
        except ValueError:
            pass

    print(f"Number of references per sample: {args.n_refs}!")

    normalize_features(
        data_path_table_main=args.data_table_path_main,
        feature_table_path=args.feature_file_path,
        output_file_path=args.normalized_feature_file_path,
 #       data_path_table=args.data_table_path,
        reference="ref",
        n_refs=args.n_refs,
        threads=args.threads,
    )

    print("Save features plot...")

    # Optional plots (behaviour preserved)
    save_features_plot(
        args.data_table_path_main,
        args.normalized_feature_file_path,
        os.path.join(args.figures_path, "features", "normalized", args.feature_type),
        aggregation_column="id_global",
    )

    print("Parallel feature normalization (with aggregation and feature filtering) completed!")


