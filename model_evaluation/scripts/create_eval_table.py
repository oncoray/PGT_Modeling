from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback (no progress bar) if tqdm is not installed.
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


def parse_filters(s: str):
    """
    Accepts reference_filters as:
      - a Python dict (already parsed by Snakemake/YAML)
      - a YAML mapping string, e.g. "{range_shift_mm: 0.0}"
      - a JSON string, e.g. "{\"range_shift_mm\": 0.0}"
    Rejects anything else.
    """
    if s is None:
        return {}
    if isinstance(s, dict):
        return s

    if isinstance(s, str):
        ss = s.strip()
        if ss == "" or ss.lower() in ("none", "null", "{}"):
            return {}

        try:
            import yaml

            obj = yaml.safe_load(ss)
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = json.loads(ss)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError(f"Could not parse reference_filters: {s!r}")


def string_to_float(s: str) -> float:
    """
    Convert a string using '-' as decimal separator (e.g. '2-5' -> 2.5).
    """
    if "-" not in s:
        raise ValueError("Input string must contain a '-' as the decimal separator.")
    return float(s.replace("-", "."))


def parse_meta(regex: Optional[str], path: str) -> Dict[str, Any]:
    """
    Parse filename metadata and normalise to types consistent with the main pipeline:
      - proton_energy: int
      - mu: float
      - range_shift: float (mm)
      - range_shift_mm: float (mm) [alias]
      - detector/layer/spot_id/repetition: int
    """
    d: Dict[str, Any] = {}
    fn = os.path.basename(path)

    if regex:
        m = re.match(regex, fn)
        if m:
            d.update(m.groupdict())

    # Normalise & coerce types
    # energy -> proton_energy
    if "energy" in d and d["energy"] is not None:
        try:
            d["proton_energy"] = int(d["energy"])
        except Exception:
            pass
        d.pop("energy", None)

    if "mu" in d and d["mu"] is not None:
        # mu comes as '2-0' etc.
        try:
            d["mu"] = float(string_to_float(str(d["mu"])))
        except Exception:
            # fallback: keep as string
            d["mu"] = d["mu"]

    for k in ("detector", "layer", "spot_id", "repetition"):
        if k in d and d[k] is not None:
            try:
                d[k] = int(d[k])
            except Exception:
                pass

    if "range_shift" in d and d["range_shift"] is not None:
        # digits only (per your current assumption); store numeric mm
        try:
            rs = float(d["range_shift"])
        except Exception:
            rs = float(str(d["range_shift"]).strip())
        d["range_shift"] = rs
        d["range_shift_mm"] = rs

    return d


def calculate_protons_per_spot(energy: Any, mu: Any) -> np.ndarray:
    """
    Identical logic to the main pipeline: MU * conversion_factor(energy) via interpolation.
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
            f"Energy outside valid range [{energies[0]}, {energies[-1]}] MeV."
        )

    conversion_factor = np.interp(energy_arr, energies, conversion_factors)
    return mu_arr * conversion_factor

def gaussian_2d_aggregation(df_group: pd.DataFrame, sigma: float = 7.8, max_dist: float = 15.6) -> pd.DataFrame:
    """
    Identical logic to the main pipeline: compute n_protons_agg from XCoord/YCoord and n_protons.
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


def append_spot_coordinates(
    data_df: pd.DataFrame,
    coord_csv_path: str,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Minimal coordinate attachment analogous to main pipeline.
    Expects coordinate CSV contains at least:
      - XCoord, YCoord
      - Proton_energy (or proton_energy)
      - SpotMU (or mu)
      - Layer, SpotID
      - Nose_orientation
      - range_shift_type, range_shift
    """
    coord_df = pd.read_csv(coord_csv_path, sep=sep)

    coord_df = coord_df.rename(
        columns={
            "Proton_energy": "proton_energy",
            "SpotMU": "mu",
            "Layer": "layer",
            "SpotID": "spot_id",
            "Nose_orientation": "nose_orientation",
        }
    )

    # dtypes harmonisation
    coord_df["proton_energy"] = coord_df["proton_energy"].astype(int)
    if coord_df["mu"].dtype == object:
        coord_df["mu"] = coord_df["mu"].apply(lambda s: string_to_float(s) if isinstance(s, str) else float(s)).astype(float)
    else:
        coord_df["mu"] = coord_df["mu"].astype(float)

    coord_df["layer"] = coord_df["layer"].astype(int)
    coord_df["spot_id"] = coord_df["spot_id"].astype(int)
    coord_df["nose_orientation"] = coord_df["nose_orientation"].astype(str).str.strip()

    # range_shift is sometimes string in coord tables; enforce float for join consistency
    coord_df["range_shift_type"] = coord_df["range_shift_type"].astype(str).str.strip()
    coord_df["range_shift"] = coord_df["range_shift"].astype(str).str.strip()

    # EXCLUDE 'ref' rows (you stated these must not be used)
    rs_lower = coord_df["range_shift"].str.lower()
    coord_df = coord_df.loc[(rs_lower != "ref") & (rs_lower != "olduncorr")].copy()

    # Convert remaining to float
    coord_df["range_shift"] = coord_df["range_shift"].astype(float)

    key_cols = [
        "proton_energy",
        "mu",
        "layer",
        "spot_id",
        "nose_orientation",
        "range_shift_type",
        "range_shift",
    ]

    # collapse duplicates by median on coordinates (as in main pipeline logic)
    coord_unique = coord_df.groupby(key_cols, as_index=False)[["XCoord", "YCoord"]].median()

    merged = data_df.merge(coord_unique, on=key_cols, how="left", validate="many_to_one")
    if merged[["XCoord", "YCoord"]].isnull().any(axis=1).any():
        bad = merged.loc[merged[["XCoord", "YCoord"]].isnull().any(axis=1), key_cols]
        raise ValueError(
            "Missing coordinates after merge for the following keys (showing up to 20):\n"
            + bad.head(20).to_string(index=False)
        )

    merged["XCoord"] = merged["XCoord"].astype(float)
    merged["YCoord"] = merged["YCoord"].astype(float)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured_dir", required=True)
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--measured_glob", default="**/*.npy")
    ap.add_argument("--reference_glob", default="**/*.npy")
    ap.add_argument("--filename_regex", default=None)
    ap.add_argument("--key_fields", required=True, help="Comma-separated list of fields used to match reference.")
    ap.add_argument("--reference_filters", default="{}")
    ap.add_argument("--out_csv", required=True)

    # optional coordinate attachment and aggregation settings
    ap.add_argument("--coord_csv", default="")
    ap.add_argument("--coord_sep", default=";")
    ap.add_argument("--agg_sigma", type=float, default=7.8)
    ap.add_argument("--agg_max_dist", type=float, default=15.6)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    t0 = time.time()
    logger.info("Starting pair list generation.")

    key_fields = [k.strip() for k in args.key_fields.split(",") if k.strip()]
    ref_filters = parse_filters(args.reference_filters)
    logger.info("key_fields=%s", key_fields)
    if ref_filters:
        logger.info("reference_filters=%s", ref_filters)

    measured_files = sorted(glob(os.path.join(args.measured_dir, args.measured_glob), recursive=True))
    reference_files = sorted(glob(os.path.join(args.reference_dir, args.reference_glob), recursive=True))

    if len(measured_files) == 0:
        raise RuntimeError(f"No measured files found in {args.measured_dir} with glob {args.measured_glob}")
    if len(reference_files) == 0:
        raise RuntimeError(f"No reference files found in {args.reference_dir} with glob {args.reference_glob}")

    logger.info("Found %d measured files and %d reference files.", len(measured_files), len(reference_files))

    # Parse reference metadata and apply filters
    ref_rows: List[Dict[str, Any]] = []
    for fp in tqdm(reference_files, desc="Parsing reference", unit="file"):
        meta = parse_meta(args.filename_regex, fp)
        ok = True
        for fk, fv in ref_filters.items():
            if fk in meta and meta[fk] is not None:
                try:
                    ok = ok and (float(meta[fk]) == float(fv))
                except Exception:
                    ok = ok and (str(meta[fk]) == str(fv))
        if ok:
            ref_rows.append({"reference_path": fp, **meta})

    if len(ref_rows) == 0:
        raise RuntimeError("No reference files left after applying reference_filters.")

    ref_df = pd.DataFrame(ref_rows)
    logger.info("Reference candidates after filtering: %d", len(ref_df))

    out_rows: List[Dict[str, Any]] = []
    for fp in tqdm(measured_files, desc="Matching measured→reference", unit="file"):
        meta = parse_meta(args.filename_regex, fp)

        rel = os.path.relpath(fp, args.measured_dir)
        sample_id = os.path.splitext(rel)[0].replace(os.sep, "/")

        # match reference files by key_fields
        mask = pd.Series([True] * len(ref_df))
        for k in key_fields:
            if k not in meta:
                mask &= False
            else:
                mask &= (ref_df[k].astype(str) == str(meta[k]))

        matched = ref_df.loc[mask, "reference_path"].tolist()
        if len(matched) == 0:
            raise RuntimeError(
                f"No reference match for measured file '{fp}'. "
                f"Parsed meta={meta}. key_fields={key_fields}."
            )

        out_rows.append(
            {
                "sample_id": sample_id,
                "measured_path": fp,
                "reference_paths_json": json.dumps(matched),
                **meta,
            }
        )

    df = pd.DataFrame(out_rows)
    logger.info("Constructed output table with %d rows.", len(df))

    # -------------------------------------------------------------------------
    # Add n_protons and n_protons_agg (analogous to main pipeline)
    # -------------------------------------------------------------------------
    if "proton_energy" not in df.columns or "mu" not in df.columns:
        raise ValueError("Cannot compute n_protons: required columns proton_energy and mu are missing.")

    logger.info("Computing n_protons via MU→proton conversion.")
    df["n_protons"] = calculate_protons_per_spot(df["proton_energy"].to_numpy(), df["mu"].to_numpy()).astype(float)

    # Attach coordinates if provided; then compute gaussian aggregation
    if args.coord_csv and str(args.coord_csv).strip() != "":
        logger.info("Attaching spot coordinates from: %s", args.coord_csv)
        df = append_spot_coordinates(df, args.coord_csv, sep=args.coord_sep)

        logger.info(
            "Computing spatial aggregation (sigma=%.3f, max_dist=%.3f).",
            float(args.agg_sigma),
            float(args.agg_max_dist),
        )

        agg_group_cols = [
            "range_shift_type",
            "range_shift",
            "nose_orientation",
            "proton_energy",
            "mu",
            "layer",
            "repetition",
            "detector",
        ]

        def _compute_group_agg(g: pd.DataFrame) -> pd.Series:
            out = gaussian_2d_aggregation(g, sigma=float(args.agg_sigma), max_dist=float(args.agg_max_dist))
            return pd.Series(out["n_protons_agg"].to_numpy(), index=g.index)

        df["n_protons_agg"] = df.groupby(agg_group_cols, group_keys=False).apply(_compute_group_agg).astype(float)
    else:
        logger.info("No coord_csv provided; setting n_protons_agg = n_protons (fallback).")
        df["n_protons_agg"] = df["n_protons"].astype(float)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logger.info("Wrote CSV to %s", args.out_csv)
    logger.info("Done in %.2f s.", time.time() - t0)


if __name__ == "__main__":
    main()