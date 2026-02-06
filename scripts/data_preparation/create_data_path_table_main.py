#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_data_path_table_main.py

Purpose
-------
Build a “data path table” from preprocessed .npy files by:
  1) parsing metadata from the directory structure / filename,
  2) filtering samples by a cohort-assignment specification,
  3) attaching spot coordinates and spot-level meta-data (XCoord, YCoord, etc.),
  4) augmenting range-shift combinations (reference augmentation),
  5) computing Gaussian 2D aggregation of proton counts within each group, and
  6) counting available reference samples per row.

Notes
-----
- The implementation is designed for batch/cluster execution and uses multiprocessing.
- Functionality is intentionally preserved from the provided version.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pmma.cmd_args import preparation_parser

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Cohort filtering helpers
# -----------------------------------------------------------------------------
CANON_KEYS = (
    "nose_orientation",
    "proton_energy",
    "mu",
    "range_shift_type",
    "repetition",
)


def _canon_tuple(rec: Dict[str, Any]) -> Tuple[str, int, float, str, int]:
    """
    Canonical tuple for exact membership checks.

    IMPORTANT: The current filtering logic uses a 5-tuple:
        (nose_orientation, proton_energy, mu, range_shift_type, repetition)

    This is intentionally preserved to maintain the same behaviour as the original
    script, even though other parts of the pipeline may include `range_shift`.
    """
    no = str(rec["nose_orientation"]).strip()
    pe = int(rec["proton_energy"])
    mu = float(round(float(rec["mu"]), 6))
    rst = str(rec["range_shift_type"]).strip()
    rep = int(rec["repetition"])
    return (no, pe, mu, rst, rep)


def _as_list(x: Any) -> List[Any]:
    """Ensure values can be expanded via `itertools.product`."""
    return x if isinstance(x, list) else [x]


def materialise_allowed_6tuples(cohort_assignment: Dict[str, Any]) -> set[Tuple[str, int, float, str, int]]:
    """
    Expand cohort_assignment into the full set of allowed tuples.

    Behaviour preserved:
    - Only conditions specifying all keys in CANON_KEYS contribute.
    - Values can be scalars or lists.
    - Whitespace and dtypes are normalised.
    - Despite the function name, this returns a set of 5-tuples
      (see `_canon_tuple`).
    """
    allowed: set[Tuple[str, int, float, str, int]] = set()
    if not cohort_assignment:
        return allowed

    for _label, conditions in cohort_assignment.items():
        for cond in conditions:
            if not all(k in cond for k in CANON_KEYS):
                continue

            nos = [str(v).strip() for v in _as_list(cond["nose_orientation"])]
            pes = [int(v) for v in _as_list(cond["proton_energy"])]
            mus = [float(round(float(v), 6)) for v in _as_list(cond["mu"])]
            rsts = [str(v).strip() for v in _as_list(cond["range_shift_type"])]
            reps = [int(v) for v in _as_list(cond["repetition"])]

            for tup in product(nos, pes, mus, rsts, reps):
                allowed.add(tup)

    return allowed


# -----------------------------------------------------------------------------
# Numeric / physics helpers
# -----------------------------------------------------------------------------
def calculate_protons_per_spot(energy: Any, mu: Any) -> np.ndarray:
    """
    Compute number of protons per spot given spot energy (MeV) and monitor units (MU).

    Uses linear interpolation of a conversion table.
    Accepts scalar or array-like inputs.
    """
    energies = np.array(
        [
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
            99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
            114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
            159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
            174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
            189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
            219, 220, 221, 222, 223, 224, 225,
        ],
        dtype=float,
    )

    conversion_factors = np.array(
        [
            6.167e7, 6.231e7, 6.2952e7, 6.3595e7, 6.4239e7, 6.4883e7, 6.5529e7, 6.6175e7, 6.6821e7, 6.7467e7,
            6.8114e7, 6.8761e7, 6.9407e7, 7.0053e7, 7.0699e7, 7.1344e7, 7.1989e7, 7.2633e7, 7.3275e7, 7.3917e7,
            7.4558e7, 7.5197e7, 7.5835e7, 7.6471e7, 7.7106e7, 7.7738e7, 7.8369e7, 7.8998e7, 7.9625e7, 8.025e7,
            8.0873e7, 8.1493e7, 8.2111e7, 8.2726e7, 8.3339e7, 8.3949e7, 8.4557e7, 8.5161e7, 8.5763e7, 8.6362e7,
            8.6958e7, 8.7551e7, 8.8142e7, 8.8729e7, 8.9313e7, 8.9894e7, 9.0472e7, 9.1047e7, 9.1618e7, 9.2187e7,
            9.2753e7, 9.3315e7, 9.3874e7, 9.443e7, 9.4984e7, 9.5534e7, 9.6081e7, 9.6625e7, 9.7166e7, 9.7704e7,
            9.8239e7, 9.8771e7, 9.9301e7, 9.9828e7, 1.0035e8, 1.0087e8, 1.0139e8, 1.0191e8, 1.0242e8, 1.0293e8,
            1.0344e8, 1.0395e8, 1.0445e8, 1.0496e8, 1.0546e8, 1.0596e8, 1.0645e8, 1.0695e8, 1.0744e8, 1.0793e8,
            1.0842e8, 1.0891e8, 1.094e8, 1.0988e8, 1.1037e8, 1.1085e8, 1.1133e8, 1.1181e8, 1.1229e8, 1.1277e8,
            1.1325e8, 1.1372e8, 1.142e8, 1.1468e8, 1.1515e8, 1.1563e8, 1.161e8, 1.1657e8, 1.1705e8, 1.1752e8,
            1.1799e8, 1.1846e8, 1.1894e8, 1.1941e8, 1.1988e8, 1.2035e8, 1.2082e8, 1.213e8, 1.2177e8, 1.2224e8,
            1.2271e8, 1.2318e8, 1.2365e8, 1.2412e8, 1.2459e8, 1.2506e8, 1.2553e8, 1.26e8, 1.2647e8, 1.2694e8,
            1.2741e8, 1.2788e8, 1.2835e8, 1.2881e8, 1.2928e8, 1.2974e8, 1.302e8, 1.3067e8, 1.3113e8, 1.3158e8,
            1.3204e8, 1.325e8, 1.3295e8, 1.334e8, 1.3384e8, 1.3429e8, 1.3473e8, 1.3517e8, 1.356e8, 1.3603e8,
            1.3646e8, 1.3688e8, 1.3729e8, 1.377e8, 1.3811e8, 1.3851e8, 1.389e8, 1.3929e8, 1.3966e8, 1.4003e8,
            1.404e8, 1.4075e8, 1.411e8, 1.4143e8, 1.4176e8, 1.4207e8, 1.4238e8,
        ],
        dtype=float,
    )

    energy_arr = np.array(energy, dtype=float)
    mu_arr = np.array(mu, dtype=float)

    if np.any(energy_arr < energies[0]) or np.any(energy_arr > energies[-1]):
        raise ValueError(
            f"One or more energy values are outside the valid range of {energies[0]} to {energies[-1]} MeV."
        )

    conversion_factor = np.interp(energy_arr, energies, conversion_factors)
    return mu_arr * conversion_factor


def gaussian_2d_aggregation(df_group: pd.DataFrame, sigma: float = 7.8, max_dist: float = 15.6) -> pd.DataFrame:
    """
    Within a single group, compute a 2D Gaussian weighting of the proton counts.

    Expects columns:
      - 'XCoord', 'YCoord', 'n_protons'

    Adds:
      - 'n_protons_agg'
    """
    coords = df_group[["XCoord", "YCoord"]].to_numpy(dtype=float)
    n_points = len(coords)

    if n_points == 0:
        out = df_group.copy()
        out["n_protons_agg"] = np.array([], dtype=float)
        return out

    if n_points == 1:
        out = df_group.copy()
        out["n_protons_agg"] = out["n_protons"].astype(float).to_numpy()
        return out

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_sq = dx**2 + dy**2

    within_range_mask = dist_sq <= (max_dist**2)
    gauss_weights = np.exp(-dist_sq / (2.0 * sigma**2)) * within_range_mask

    proton_counts = df_group["n_protons"].to_numpy(dtype=float)
    aggregated_protons = gauss_weights.dot(proton_counts)

    out = df_group.copy()
    out["n_protons_agg"] = aggregated_protons.astype(float)
    return out


# -----------------------------------------------------------------------------
# Parsing helpers for file paths / types
# -----------------------------------------------------------------------------
def string_to_float(s: str) -> float:
    """
    Convert a string using '-' as decimal separator (e.g. '2-5' -> 2.5).

    Behaviour preserved: raises if '-' not present.
    """
    if "-" not in s:
        raise ValueError("Input string must contain a '-' as the decimal separator.")
    return float(s.replace("-", "."))


def extract_metadata_from_file_path(file_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from `file_path` relative to `output_dir`.

    Expected relative structure:
      [nose_orientation, energy, mu, range_shift_type, range_shift, det{detector}, file_name]

    If parsing fails, returns None (behaviour preserved).
    """
    rel_path = os.path.relpath(file_path, output_dir)
    parts = rel_path.split(os.sep)

    if len(parts) != 7:
        logger.info(f"Unexpected directory structure for file: {file_path}")
        return None

    energy_token = parts[1]     # e.g. "100MeV"
    mu_token = parts[2]         # e.g. "2-0MU"
    detector_dir = parts[5]     # e.g. "det1"
    file_name = parts[6]

    # Parse out integer energy (directory-level)
    try:
        _ = int(energy_token.replace("MeV", ""))
    except ValueError:
        logger.info(f"Could not parse energy value from {energy_token}")
        return None

    # Parse out MU (directory-level)
    try:
        _ = float(string_to_float(mu_token.replace("MU", "")))
    except ValueError:
        logger.info(f"Could not parse MU value from {mu_token}")
        return None

    # Parse out detector (directory-level)
    try:
        _ = int(detector_dir.replace("det", ""))
    except ValueError:
        logger.info(f"Could not parse detector from {detector_dir}")
        return None

    # Filename-level parsing (authoritative)
    pattern = re.compile(
        r"(?P<nose_orientation>[\w\d]+_withRC|[\w\d]+)_"
        r"(?P<energy>\d+)MeV_"
        r"(?P<mu>\d+-\d+)MU_"
        r"(?P<range_shift_type>\w+)_"
        r"(?P<range_shift>ref|\d+)_"
        r"det(?P<detector>\d+)_"
        r"layer(?P<layer>\d+)_"
        r"spot(?P<spot_id>\d+)_"
        r"rep(?P<repetition>\d+)\.npy"
    )

    match = pattern.match(file_name)
    if not match:
        return None

    md = match.groupdict()
    md["file_path_proc"] = file_path

    # Normalise types (behaviour preserved)
    md["proton_energy"] = int(md["energy"])
    del md["energy"]
    md["mu"] = float(string_to_float(md["mu"]))
    md["detector"] = int(md["detector"])
    md["layer"] = int(md["layer"])
    md["spot_id"] = int(md["spot_id"])
    md["repetition"] = int(md["repetition"])

    return md


def collect_preprocessed_files(output_dir: str) -> List[str]:
    """Walk through output_dir and collect all '.npy' files."""
    preprocessed_files: List[str] = []
    for root, _dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".npy"):
                preprocessed_files.append(os.path.join(root, file))
    return preprocessed_files


def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce integer/float dtypes for known columns (behaviour preserved)."""
    for col in ["proton_energy", "detector", "layer", "spot_id", "repetition"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "mu" in df.columns:
        df["mu"] = df["mu"].astype(float)
    return df


# -----------------------------------------------------------------------------
# Spot-coordinate attachment
# -----------------------------------------------------------------------------
def _harmonise_coord_key_types(coord_df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonise key-column dtypes/formatting for robust merges.
    """
    coord_df["proton_energy"] = coord_df["proton_energy"].astype(int)

    # Spot MU uses the 'x-y' decimal convention in many log files
    if coord_df["mu"].dtype == object:
        coord_df["mu"] = (
            coord_df["mu"]
            .apply(lambda s: string_to_float(s) if isinstance(s, str) else float(s))
            .astype(float)
        )

    for col in ("layer", "spot_id"):
        coord_df[col] = coord_df[col].astype(int)

    coord_df["range_shift_type"] = coord_df["range_shift_type"].astype(str)
    coord_df["range_shift"] = coord_df["range_shift"].astype(str)

    # Trim accidental whitespace
    str_cols = ["nose_orientation", "range_shift_type", "range_shift"]
    coord_df[str_cols] = coord_df[str_cols].apply(lambda s: s.str.strip())

    return coord_df


def append_spot_coordinates(
    data_df: pd.DataFrame,
    coord_csv_path: str,
    sep: str = ";",
    warn_tol_mm: float = 1.5,
) -> pd.DataFrame:
    """
    Attach spot coordinates and per-spot physical quantities to `data_df`.

    Adds:
      - XCoord, YCoord
      - Proton_energy_spot, SpotMU_spot

    Duplicate handling:
      The coordinate file can contain multiple rows mapping to the same key due to
      detector/repetition. Duplicates are collapsed using median; coordinate spread
      beyond warn_tol_mm is logged.

    Raises:
      ValueError if any row in `data_df` lacks required coordinate/meta-data after merge.
    """
    coord_df = pd.read_csv(coord_csv_path, sep=sep)

    coord_df = coord_df.rename(
        columns={
            "Proton_energy": "proton_energy",
            "SpotMU": "mu",
            "Layer": "layer",
            "SpotID": "spot_id",
            "Nose_orientation": "nose_orientation",
            "#Detector": "detector",   # may appear, ignored by key
            "Repetition": "repetition" # may appear, ignored by key
        }
    )

    coord_df = _harmonise_coord_key_types(coord_df)

    key_cols = [
        "proton_energy",
        "mu",
        "layer",
        "spot_id",
        "nose_orientation",
        "range_shift_type",
        "range_shift",
    ]

    def _collapse(group: pd.DataFrame) -> pd.Series:
        """Groupwise medians; log if coordinate spread is large (behaviour preserved)."""
        x_span = group["XCoord"].max() - group["XCoord"].min()
        y_span = group["YCoord"].max() - group["YCoord"].min()
        if (x_span > warn_tol_mm) or (y_span > warn_tol_mm):
            logger.info(
                f"Coordinate spread {x_span:.2f} mm × {y_span:.2f} mm for key {tuple(group.name)} "
                f"exceeds tolerance ({warn_tol_mm} mm); median values are used."
            )
        return pd.Series(
            {
                "XCoord": group["XCoord"].median(),
                "YCoord": group["YCoord"].median(),
                "Proton_energy_spot": group["Proton_energy_spot"].median(),
                "SpotMU_spot": group["SpotMU_spot"].median(),
            }
        )

    # NOTE: This groupby/apply/reset pattern is preserved to avoid subtle changes.
    coord_unique = (
        coord_df.groupby(key_cols, as_index=False)
        .apply(_collapse)
        .reset_index()
    )

    merged = data_df.merge(
        coord_unique,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    required_cols = ["XCoord", "YCoord", "Proton_energy_spot", "SpotMU_spot"]
    missing_mask = merged[required_cols].isnull().any(axis=1)

    if missing_mask.any():
        missing_df = merged.loc[missing_mask, key_cols]
        n_missing = len(missing_df)

        msg_hdr = (
            f"{n_missing} spot(s) still lack coordinates/spot meta-data after collapsing duplicates."
        )
        if n_missing <= 20:
            details = missing_df.to_string(index=False)
            msg = f"{msg_hdr}\nOffending rows:\n{details}"
        else:
            sample = missing_df.head(10).to_string(index=False)
            msg = f"{msg_hdr}  Showing the first 10 offending rows:\n{sample}\n … (truncated) …"
        raise ValueError(msg)

    # Normalise dtypes
    for col in required_cols:
        merged[col] = merged[col].astype(float)

    return merged


# -----------------------------------------------------------------------------
# Cluster parallelism helpers
# -----------------------------------------------------------------------------
def get_num_processes() -> int:
    """
    Determine number of processes to use under SLURM.
    Prefer SLURM_* variables; fall back to mp.cpu_count() (behaviour preserved).
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_JOB_CPUS_PER_NODE"):
        v = os.environ.get(var)
        if v:
            try:
                return int(str(v).split("(")[0])  # SLURM_JOB_CPUS_PER_NODE can be like "4(x2)"
            except ValueError:
                continue
    return mp.cpu_count()


# -----------------------------------------------------------------------------
# Range-shift augmentation + reference counting (parallel workers)
# -----------------------------------------------------------------------------
def augment_group(args: Tuple[Any, pd.DataFrame]) -> pd.DataFrame:
    """
    Worker for range-shift augmentation.

    For each row in a group and each unique reference range shift in that group:
      - original_range_shift  := original range_shift
      - reference_range_shift := chosen reference
      - range_shift           := original_range_shift - reference_range_shift
    """
    _group_key, group_df = args
    augmented_rows: List[Dict[str, Any]] = []

    unique_ref_values = group_df["range_shift"].unique()
    group_records = group_df.to_dict(orient="records")

    for ref_val in unique_ref_values:
        for row in group_records:
            rs_val = row["range_shift"]
            rel_shift = rs_val - ref_val

            new_row = row.copy()
            new_row["original_range_shift"] = rs_val
            new_row["reference_range_shift"] = ref_val
            new_row["range_shift"] = rel_shift
            augmented_rows.append(new_row)

    return pd.DataFrame(augmented_rows)


def count_references_in_group(args: Tuple[Any, pd.DataFrame]) -> pd.DataFrame:
    """
    Worker to count, for each row in a group, how many *unique reference samples*
    (unique file_path_proc) are available under the following logic:

      For a given sample row s:

        - same group key_cols:
              ['nose_orientation', 'proton_energy', 'mu', 'detector', 'layer']
        - reference rows must satisfy:
              range_shift_type == "grs"
              original_range_shift == s.reference_range_shift
        - and be within ±tol_xy_mm in both XCoord and YCoord.

    Multiple rows with the same file_path_proc in the spatial window count once.
    """
    tol_xy_mm = 1.5

    _group_key, group_df = args
    group_df = group_df.copy()

    ref_candidates = group_df[group_df["range_shift_type"] == "grs"]
    n_rows = len(group_df)

    if ref_candidates.empty or n_rows == 0:
        group_df["n_reference_samples"] = 0
        return group_df

    ref_xy = ref_candidates[["XCoord", "YCoord"]].to_numpy(dtype=float)
    ref_orig_rs = ref_candidates["original_range_shift"].to_numpy(dtype=float)
    ref_paths = ref_candidates["file_path_proc"].to_numpy()

    # Pre-index reference information by original_range_shift
    ref_info_by_rs: Dict[float, Dict[str, Any]] = {}
    for rs_val in np.unique(ref_orig_rs):
        mask = ref_orig_rs == rs_val
        ref_info_by_rs[rs_val] = {"xy": ref_xy[mask], "paths": ref_paths[mask]}

    counts = np.zeros(n_rows, dtype=int)
    sample_x = group_df["XCoord"].to_numpy(dtype=float)
    sample_y = group_df["YCoord"].to_numpy(dtype=float)
    sample_ref_rs = group_df["reference_range_shift"].to_numpy(dtype=float)

    for i in range(n_rows):
        ref_rs_val = sample_ref_rs[i]
        ref_info = ref_info_by_rs.get(ref_rs_val)
        if ref_info is None:
            counts[i] = 0
            continue

        ref_xy_rs = ref_info["xy"]
        ref_paths_rs = ref_info["paths"]
        if len(ref_xy_rs) == 0:
            counts[i] = 0
            continue

        dx = np.abs(ref_xy_rs[:, 0] - sample_x[i])
        dy = np.abs(ref_xy_rs[:, 1] - sample_y[i])
        in_win = (dx <= tol_xy_mm) & (dy <= tol_xy_mm)

        if not in_win.any():
            counts[i] = 0
            continue

        counts[i] = int(len(np.unique(ref_paths_rs[in_win])))

    group_df["n_reference_samples"] = counts
    return group_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = preparation_parser("Prepare data")

    parser.add_argument(
        "--reference",
        type=str,
        default="4",
        help='String used to designate reference range_shift (e.g. "ref").',
    )

    parser.add_argument(
        "--data_path_table_processed",
        type=str,
        required=True,
        help="Path to the CSV that supplies XCoord and YCoord for each spot.",
    )

    args = parser.parse_args()
    reference = args.reference

    # -------------------------------------------------------------------------
    # 1) Load cohort_assignment (file path or JSON string)
    # -------------------------------------------------------------------------
    if os.path.isfile(args.cohort_assignment):
        with open(args.cohort_assignment, "r") as f:
            cohort_assignment = json.load(f)
    else:
        cohort_assignment = json.loads(args.cohort_assignment)

    # Preserved stdout prints
    print(cohort_assignment)

    allowed_6tuples = materialise_allowed_6tuples(cohort_assignment)
    print("Allowed tuples:")
    print(allowed_6tuples)

    # -------------------------------------------------------------------------
    # 2) Collect preprocessed .npy files
    # -------------------------------------------------------------------------
    preprocessed_files = collect_preprocessed_files(args.root_dir)
    if len(preprocessed_files) == 0:
        logger.error(f"No preprocessed files found in {args.root_dir}.")
        raise FileNotFoundError(f"No preprocessed files found in {args.root_dir}.")
    logger.info(f"Found {len(preprocessed_files)} preprocessed files.")

    # -------------------------------------------------------------------------
    # 3) Parse metadata and filter by allowed combinations
    # -------------------------------------------------------------------------
    metadata_list: List[Dict[str, Any]] = []
    for file_path in preprocessed_files:
        meta = extract_metadata_from_file_path(file_path, args.root_dir)
        if meta is None:
            continue

        key = _canon_tuple(meta)
        if key in allowed_6tuples:
            metadata_list.append(meta)

    # -------------------------------------------------------------------------
    # 4) Build DataFrame and enforce dtypes
    # -------------------------------------------------------------------------
    data_df = pd.DataFrame(metadata_list)
    logger.info(f"DataFrame created with {len(data_df)} entries.")
    data_df = enforce_data_types(data_df)

    # -------------------------------------------------------------------------
    # 5) Cohort assignment
    # -------------------------------------------------------------------------
    data_df["cohort"] = "unassigned"

    if cohort_assignment is not None:
        logger.info("Assigning cohorts based on cohort_assignment.")
        for cohort_label, conditions_list in cohort_assignment.items():
            for condition in conditions_list:
                mask = data_df["cohort"] == "unassigned"
                for column, values in condition.items():
                    if column not in data_df.columns:
                        logger.error(f"Column '{column}' not found in DataFrame.")
                        raise KeyError(f"Column '{column}' not found in DataFrame.")
                    if isinstance(values, list):
                        mask &= data_df[column].isin(values)
                    else:
                        mask &= data_df[column] == values
                data_df.loc[mask, "cohort"] = cohort_label

    # -------------------------------------------------------------------------
    # 6) Attach spot coordinates and spot-level meta-data
    # -------------------------------------------------------------------------
    data_df = append_spot_coordinates(data_df, args.data_path_table_processed)

    # Ensure per-spot proton count exists
    if "n_protons_spot" not in data_df.columns:
        data_df["n_protons_spot"] = calculate_protons_per_spot(
            data_df["Proton_energy_spot"].values,
            data_df["SpotMU_spot"].values,
        ).astype(float)

    # -------------------------------------------------------------------------
    # 7) Remove remaining unassigned rows (before augmentation)
    # -------------------------------------------------------------------------
    unassigned_rows = data_df[data_df["cohort"] == "unassigned"]
    if not unassigned_rows.empty:
        logger.info(f"Removing {len(unassigned_rows)} unassigned rows before reference augmentation.")
        data_df = data_df[data_df["cohort"] != "unassigned"]

    # -------------------------------------------------------------------------
    # 8) Range-shift augmentation
    # -------------------------------------------------------------------------
    logger.info(
        "Augmenting range_shift combinations: each unique range_shift acts "
        "as a reference_range_shift within its (energy, mu, nose_orientation, detector) group."
    )

    def _range_shift_to_float(rs: Any) -> float:
        """
        Convert textual range_shift to numeric for difference computation.

        Behaviour preserved:
        - 'ref' is forbidden at this stage and triggers a ValueError.
        """
        s = str(rs).strip()
        if s.lower() == "ref":
            raise ValueError("Still ref in data_df range shift!?")
        try:
            return float(s)
        except ValueError as e:
            raise ValueError(f"Cannot convert range_shift value '{rs}' to float.") from e

    data_df = data_df.copy()
    data_df["range_shift"] = data_df["range_shift"].apply(_range_shift_to_float)

    group_cols_rs = ["proton_energy", "mu", "nose_orientation", "detector"]

    logger.info("Starting multiprocessing range-shift augmentation.")
    group_inputs = list(data_df.groupby(group_cols_rs))

    threads = get_num_processes()
    logger.info(f"Using {threads} CPU cores for augmentation.")

    if len(group_inputs) == 0:
        logger.warning("No groups found for augmentation; final_data_df will be empty.")
        final_data_df = data_df.copy()
    else:
        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(
                tqdm(
                    executor.map(augment_group, group_inputs),
                    total=len(group_inputs),
                    desc="Range-shift augmentation",
                )
            )
        final_data_df = pd.concat(results, ignore_index=True)

    logger.info(
        f"Multiprocessing augmentation completed. "
        f"Produced {len(final_data_df)} rows from {len(data_df)} input rows."
    )

    # -------------------------------------------------------------------------
    # 9) Apply reference selection logic
    # -------------------------------------------------------------------------
    reference_int = int(reference)
    if reference_int == -1:
        logger.info("Reference augmentation is performed!")
    else:
        logger.info(f"Reference augmentation is NOT performed! Use {reference_int} as reference!")
        final_data_df = final_data_df[final_data_df["reference_range_shift"] == reference_int]
        final_data_df.loc[
            (final_data_df["reference_range_shift"] == final_data_df["original_range_shift"])
            & (final_data_df["range_shift_type"] == "grs"),
            "cohort",
        ] = "reference"

    # Safety: remove residual unassigned rows
    if "cohort" in final_data_df.columns:
        residual_unassigned = final_data_df[final_data_df["cohort"] == "unassigned"]
        if not residual_unassigned.empty:
            logger.info(f"Removing {len(residual_unassigned)} residual unassigned rows after augmentation.")
            final_data_df = final_data_df[final_data_df["cohort"] != "unassigned"]

    # -------------------------------------------------------------------------
    # 10) Gaussian 2D aggregation of proton counts
    # -------------------------------------------------------------------------
    final_data_df["n_protons"] = final_data_df["n_protons_spot"].astype(float)

    agg_group_cols = [
        "range_shift_type",
        "range_shift",
        "nose_orientation",
        "proton_energy",
        "mu",
        "layer",
        "repetition",
        "detector",
        "original_range_shift",
        "reference_range_shift",
    ]

    def _compute_n_protons_agg(g: pd.DataFrame) -> pd.Series:
        # compute only the aggregated values, return aligned Series
        out = gaussian_2d_aggregation(g, sigma=7.8, max_dist=15.6)
        return pd.Series(out["n_protons_agg"].to_numpy(), index=g.index)

    # assign back; preserves ALL columns and index
    final_data_df["n_protons_agg"] = (
        final_data_df.groupby(agg_group_cols, group_keys=False)
        .apply(_compute_n_protons_agg)
    )

    final_data_df = final_data_df.drop(columns=["n_protons"])

    # -------------------------------------------------------------------------
    # 11) Count number of reference samples per row (parallel)
    # -------------------------------------------------------------------------
    logger.info("Starting parallel counting of reference samples per row.")

    group_cols_ref = ["nose_orientation", "proton_energy", "mu", "detector", "layer"]

    group_inputs_ref = list(final_data_df.groupby(group_cols_ref))

    threads_ref = get_num_processes()
    logger.info(f"Using {threads_ref} CPU cores for reference counting.")

    if len(group_inputs_ref) == 0:
        logger.warning("No groups found for reference counting; setting n_reference_samples = 0.")
        final_data_df["n_reference_samples"] = 0
    else:
        with ProcessPoolExecutor(max_workers=threads_ref) as executor:
            results_ref = list(
                tqdm(
                    executor.map(count_references_in_group, group_inputs_ref),
                    total=len(group_inputs_ref),
                    desc="Counting reference samples",
                )
            )
        final_data_df = pd.concat(results_ref, ignore_index=True)

    # -------------------------------------------------------------------------
    # 12) Sort by file path, assign global IDs, and save outputs
    # -------------------------------------------------------------------------
    final_data_df = final_data_df.sort_values("file_path_proc").reset_index(drop=True)
    final_data_df["id_global"] = range(1, len(final_data_df) + 1)

    final_data_df.to_csv(args.output_file_main, index=False, sep=";")
    logger.info(f"Main data path table saved to {args.output_file_main}.")

 

if __name__ == "__main__":
    print("Start")
    main()
