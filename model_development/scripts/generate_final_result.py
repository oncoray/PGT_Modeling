# -*- coding: utf-8 -*-
"""
cml_final_result.py

Create a final “model card” style summary for a single chosen cML setup by combining:
  - Validation performance summaries
  - Test performance summaries

The script:
  1) Selects the final model setup as the configuration with minimal RMSE on the
     validation set (combined factors) in Validation.
  2) Extracts per-(nose_orientation, mu, proton_energy, range_shift_type, repetition) results
     from both Validation and Test for that setup.
  3) Writes a final CSV and exports a human-readable summary as TXT and PNG.

"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pmma.cmd_args import cML_final_result_parser


# =============================================================================
# Validation helpers
# =============================================================================
def _require_columns(df: pd.DataFrame, cols: List[str], *, df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def _ensure_single_value(series: pd.Series, *, label: str) -> None:
    nunique = series.nunique(dropna=False)
    if nunique != 1:
        vals = series.drop_duplicates().head(10).tolist()
        raise ValueError(f"Expected a single unique value for '{label}', found {nunique}. Examples: {vals}")


# =============================================================================
# Core logic
# =============================================================================
def get_final_results_table(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    final_model_setup: Dict[str, str],
) -> pd.DataFrame:
    """
    Generate a combined table of CV + Test results for one model setup.

    Parameters
    ----------
    df_val : pd.DataFrame
        Validation summary table (already aggregated with CIs).
    df_test : pd.DataFrame
        Test summary table (already aggregated with CIs).
    final_model_setup : dict
        Keys: ['feature_type','feature_selection_method','model_learner'].

    Returns
    -------
    pd.DataFrame
        Rows covering CV and Test for the selected setup.
        Includes: setup columns, signature, sign_size, step, cohort, factors, metrics.
    """
    setup_keys = ["feature_type", "feature_selection_method", "model_learner"]
    for k in setup_keys:
        if k not in final_model_setup:
            raise KeyError(f"final_model_setup missing key '{k}'")

    # Required columns
    common_required = [
        "feature_type",
        "feature_selection_method",
        "model_learner",
        "signature",
        "sign_size",
        "nose_orientation",
        "mu",
        "proton_energy",
        "range_shift_type",
        "repetition",
        "RMSE",
        "RMSE CI Low",
        "RMSE CI High",
        "R2",
        "R2 CI Low",
        "R2 CI High",
    ]
    _require_columns(df_val, common_required + ["cv_data_set"], df_name="df_val")
    _require_columns(df_test, common_required + ["cohort"], df_name="df_test")

    # Filter to the model setup
    df_val = df_val[
        (df_val["feature_type"] == final_model_setup["feature_type"])
        & (df_val["feature_selection_method"] == final_model_setup["feature_selection_method"])
        & (df_val["model_learner"] == final_model_setup["model_learner"])
    ].copy()

    df_test = df_test[
        (df_test["feature_type"] == final_model_setup["feature_type"])
        & (df_test["feature_selection_method"] == final_model_setup["feature_selection_method"])
        & (df_test["model_learner"] == final_model_setup["model_learner"])
    ].copy()

    if df_val.empty:
        raise ValueError("No rows found in Validation table for final_model_setup.")
    if df_test.empty:
        raise ValueError("No rows found in Test table for final_model_setup.")

    # Enforce that within each grouping tuple, metrics are unique (i.e., the table is already aggregated)
    if df_val.groupby(
        ["cv_data_set", "nose_orientation", "mu", "proton_energy", "range_shift_type", "repetition"]
    )["RMSE"].nunique().max() != 1:
        raise ValueError("Multiple RMSE values found for a CV group; input appears not fully aggregated.")

    if df_test.groupby(
        ["cohort", "nose_orientation", "mu", "proton_energy", "range_shift_type", "repetition"]
    )["RMSE"].nunique().max() != 1:
        raise ValueError("Multiple RMSE values found for an EV group; input appears not fully aggregated.")

    summary_rows: List[Dict[str, object]] = []

    def _append_rows(df: pd.DataFrame, group_cols: List[str], *, step: str, split_col: str) -> None:
        for name, group in df.groupby(group_cols):
            # name is a tuple: (split_col_value, nose_orientation, mu, proton_energy, range_shift_type, repetition)
            split_val = name[0]
            cohort_name = "training" if split_val in ("development", "training") else "validation"

            _ensure_single_value(group["signature"], label=f"{step} signature for group {name}")
            _ensure_single_value(group["sign_size"], label=f"{step} sign_size for group {name}")

            row = {
                **final_model_setup,
                "signature": group["signature"].iloc[0],
                "sign_size": int(group["sign_size"].iloc[0]),
                "step": step,
                "cohort": cohort_name,
                "nose_orientation": name[1],
                "mu": name[2],
                "proton_energy": name[3],
                "range_shift_type": name[4],
                "repetition": name[5],
                "RMSE": float(group["RMSE"].iloc[0]),
                "RMSE CI Low": float(group["RMSE CI Low"].iloc[0]),
                "RMSE CI High": float(group["RMSE CI High"].iloc[0]),
                "R2": float(group["R2"].iloc[0]),
                "R2 CI Low": float(group["R2 CI Low"].iloc[0]),
                "R2 CI High": float(group["R2 CI High"].iloc[0]),
            }
            summary_rows.append(row)

    _append_rows(
        df_val,
        ["cv_data_set", "nose_orientation", "mu", "proton_energy", "range_shift_type", "repetition"],
        step="validation",
        split_col="cv_data_set",
    )
    _append_rows(
        df_test,
        ["cohort", "nose_orientation", "mu", "proton_energy", "range_shift_type", "repetition"],
        step="external_validation",
        split_col="cohort",
    )

    final_df = pd.DataFrame(summary_rows)

    # Stable ordering: combined factors first, then lexicographic
    def _is_combined_row(r: pd.Series) -> bool:
        return (
            r["nose_orientation"] == "combined"
            and r["mu"] == "combined"
            and r["proton_energy"] == "combined"
            and r["range_shift_type"] == "combined"
            and str(r["repetition"]) == "combined"
        )

    final_df["_combined_first"] = final_df.apply(_is_combined_row, axis=1).map({True: 0, False: 1})
    final_df = final_df.sort_values(
        by=["step", "_combined_first", "nose_orientation", "mu", "proton_energy", "range_shift_type", "repetition", "cohort"]
    ).drop(columns=["_combined_first"])

    return final_df


def summarize_results_to_string(final_df: pd.DataFrame) -> str:
    """
    Convert final results table into a structured, human-readable report string.
    """
    if final_df.empty:
        return "No results available (final_df is empty)."

    # Final model setup and signature
    setup_keys = ["feature_type", "feature_selection_method", "model_learner", "signature", "sign_size"]
    model_setup = {k: final_df.iloc[0][k] for k in setup_keys}

    lines: List[str] = []
    lines.append("Final Model Setup:")
    for k, v in model_setup.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    group_cols = ["nose_orientation", "mu", "proton_energy", "range_shift_type", "repetition"]

    for step in final_df["step"].drop_duplicates().tolist():
        lines.append(f"Step: {step}")
        df_step = final_df[final_df["step"] == step].copy()

        datasets = list(df_step.groupby(group_cols).groups.keys())
        # Sort combined first, then lexicographic
        datasets.sort(key=lambda x: (x != ("combined", "combined", "combined", "combined", "combined"), x))

        for ds in datasets:
            ds_name = (
                f"nose_orientation: {ds[0]}, mu: {ds[1]}, proton_energy: {ds[2]}, "
                f"range_shift_type: {ds[3]}, rep: {ds[4]}"
            )
            lines.append(f"  Dataset: {ds_name}")

            df_ds = df_step[
                (df_step["nose_orientation"] == ds[0])
                & (df_step["mu"] == ds[1])
                & (df_step["proton_energy"] == ds[2])
                & (df_step["range_shift_type"] == ds[3])
                & (df_step["repetition"].astype(str) == str(ds[4]))
            ]

            # deterministic order: training then validation
            cohort_order = ["training", "validation"]
            for cohort in cohort_order:
                df_c = df_ds[df_ds["cohort"] == cohort]
                if df_c.empty:
                    continue

                # may be >1 row if upstream creates multiple; print all
                for _, row in df_c.iterrows():
                    rmse = (
                        f"RMSE: {row['RMSE']:.2f} "
                        f"(CI: {row['RMSE CI Low']:.2f} - {row['RMSE CI High']:.2f})"
                    )
                    r2 = (
                        f"R2: {row['R2']:.2f} "
                        f"(CI: {row['R2 CI Low']:.2f} - {row['R2 CI High']:.2f})"
                    )
                    lines.append(f"    Cohort: {cohort}")
                    lines.append(f"      {rmse}")
                    lines.append(f"      {r2}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_string_to_image_and_text(
    text: str,
    image_path: str,
    text_path: str,
    *,
    font_size: int = 28,
    wrap_width: int = 80,
    padding: int = 24,
) -> None:
    """
    Save a text report to a TXT file and a high-resolution PNG.

    Notes
    -----
    - Line breaks are preserved; each line is wrapped independently.
    - Uses Arial if available; otherwise falls back to a default bitmap font.
    """
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Wrap each line while preserving explicit line breaks
    wrapped_lines: List[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=wrap_width))

    wrapped_text = "\n".join(wrapped_lines)

    # Measure
    dummy = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(dummy)
    try:
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.multiline_textsize(wrapped_text, font=font)

    img_w = int(text_w + 2 * padding)
    img_h = int(text_h + 2 * padding)

    image = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(image)
    draw.multiline_text((padding, padding), wrapped_text, fill="black", font=font, spacing=4)

    image.save(image_path, format="PNG")


def find_experiment_setup_from_val(df: pd.DataFrame) -> Dict[str, str]:
    """
    Select the final model setup from Validation by minimising RMSE on the validation set,
    restricted to the combined setting across experimental factors.

    Returns
    -------
    dict with keys: feature_type, feature_selection_method, model_learner
    """
    required = [
        "feature_type",
        "feature_selection_method",
        "model_learner",
        "cv_data_set",
        "nose_orientation",
        "mu",
        "proton_energy",
        "range_shift_type",
        "repetition",
        "RMSE",
    ]
    _require_columns(df, required, df_name="cv_results")

    # Restrict to combined factors (full combined)
    validation_df = df[
        (df["cv_data_set"] == "validation")
        & (df["nose_orientation"] == "combined")
        & (df["mu"] == "combined")
        & (df["proton_energy"] == "combined")
        & (df["range_shift_type"] == "combined")
        & (df["repetition"].astype(str) == "combined")
    ].copy()

    if validation_df.empty:
        raise ValueError(
            "No rows found for validation combined setting. "
            "Ensure that the CV summary includes combined rows for "
            "nose_orientation/mu/proton_energy/range_shift_type/repetition."
        )

    min_idx = validation_df["RMSE"].astype(float).idxmin()
    row = validation_df.loc[min_idx]

    return {
        "feature_type": str(row["feature_type"]),
        "feature_selection_method": str(row["feature_selection_method"]),
        "model_learner": str(row["model_learner"]),
    }


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = cML_final_result_parser("cML final result")
    args = parser.parse_args()

    cv_results = pd.read_csv(args.val_performance_file_path, sep=";")
    test_results = pd.read_csv(args.test_performance_file_path, sep=";")

    print("Finding final model setup...")
    final_model_setup = find_experiment_setup_from_val(cv_results)
    print("Final model setup is:\n", final_model_setup)

    print("Calculating final results table...")
    df_final = get_final_results_table(cv_results, test_results, final_model_setup)

    os.makedirs(os.path.dirname(args.final_result_csv), exist_ok=True)
    df_final.to_csv(args.final_result_csv, sep=";", index=False)

    print("Summarizing results into a string...")
    summary_str = summarize_results_to_string(df_final)
    print(summary_str)

    save_string_to_image_and_text(summary_str, args.final_result_png, args.final_result_txt)

    print("Process finished!")


    