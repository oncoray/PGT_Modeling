#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
feature_ranking.py

Run FAMiLIA-R-based feature ranking on clustered/selected features and export:
  - a feature ranking CSV (feature, rank, score)
  - a feature importance bar plot

The script:
1) Loads clustering metadata to obtain 'cluster_representatives'.
2) Builds a FAMiLIA-R compatible feature table for the selected representatives.
3) Runs the FAMiLIA-R experiment (via pmma.familiar_preparation helpers).
4) Reads the resulting variable importance file, writes a cleaned CSV, and plots it.

"""

from __future__ import annotations

import json
import multiprocessing
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from pmma.cmd_args import feature_ranking_parser
from pmma.familiar_preparation import (
    create_feature_table_for_familiar,
    perform_familiar_experiment,
)


def plot_feature_importance(
    data: pd.DataFrame,
    save_path: str,
    feature_type: str,
    feature_selection_method: str,
) -> None:
    """
    Save a bar plot of feature importance.

    Parameters
    ----------
    data:
        DataFrame with at least columns: 'feature' and 'score'.
    save_path:
        Output path for the plot image.
    feature_type:
        Feature type label used in the plot title.
    feature_selection_method:
        Feature selection method label used in the plot title.
    """
    # Sort by descending score (most important first)
    sorted_data = data.sort_values(by="score", ascending=False)

    # Keep original heuristic for tick font size
    tick_size = max(min(10, 120 / len(sorted_data)), 6) if len(sorted_data) > 0 else 10

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data["feature"], sorted_data["score"], align="center")
    plt.xticks(rotation=90, fontsize=tick_size)
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.title(f"{feature_type} Feature Importance using {feature_selection_method} Method")
    plt.xlim(-0.5, len(sorted_data) - 0.5)
    plt.grid(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_feature_ranking(
    feature_ranking_file_path: str,
    experiment_dir: str,
    vimp_aggregation_method: str,
) -> pd.DataFrame:
    """
    Read the FAMiLIA-R variable importance output and write a cleaned ranking CSV.

    Parameters
    ----------
    feature_ranking_file_path:
        Output path for the cleaned ranking CSV (semicolon-separated).
    experiment_dir:
        Experiment directory used by FAMiLIA-R (contains the results tree).
    vimp_aggregation_method:
        Variable importance aggregation method used by FAMiLIA-R (part of file name).

    Returns
    -------
    pd.DataFrame
        Cleaned ranking table with columns: feature, rank, score.
    """
    familiar_feature_ranking_file_path = os.path.join(
        experiment_dir,
        "results",
        "pooled_data",
        "variable_importance",
        f"variable_importance_feature_selection_{vimp_aggregation_method}.csv",
    )

    if not os.path.isfile(familiar_feature_ranking_file_path):
        raise FileNotFoundError(
            "Feature ranking file does not exist here: "
            f"{familiar_feature_ranking_file_path}"
        )

    feature_ranking = pd.read_csv(familiar_feature_ranking_file_path, sep=";")

    # Preserve original column selection + rename
    feature_ranking = feature_ranking[["name", "rank", "score"]].copy()
    feature_ranking.rename(columns={"name": "feature"}, inplace=True)

    os.makedirs(os.path.dirname(feature_ranking_file_path), exist_ok=True)
    feature_ranking.to_csv(feature_ranking_file_path, index=False, sep=";")

    print(f"Feature ranking file saved to {feature_ranking_file_path}!")
    return feature_ranking


def _load_cluster_representatives(metadata_path: str) -> list[str]:
    """
    Load clustering metadata JSON and extract 'cluster_representatives'.
    """
    with open(metadata_path, "r") as file:
        cluster_metadata = json.load(file)
    return cluster_metadata.get("cluster_representatives", [])


if __name__ == "__main__":
    parser = feature_ranking_parser("Feature ranking")
    args = parser.parse_args()

    print("Start feature ranking.....")

    # 1) Load cluster representatives from metadata
    cluster_representatives = _load_cluster_representatives(args.feature_clustering_metadata_file_path)

    # 2) Prepare temp directory and paths
    temp_data_dir = os.path.join(args.feature_ranking_path, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)

    cluster_feature_file_path = os.path.join(
        temp_data_dir,
        f"features_{args.feature_type}_{args.feature_selection_method}.csv",
    )

    # 3) Load data tables
    data_table = pd.read_csv(args.data_table_path, sep=";")
    feature_table = pd.read_csv(args.feature_file_path, sep=";")

    # 4) Create feature table for FAMiLIA-R using cluster representatives
    create_feature_table_for_familiar(
        data_table,
        feature_table,
        cluster_representatives,
        cluster_feature_file_path,
        evaluation_phase=False,
    )

    # 5) Run FAMiLIA-R experiment
    familiar_r_file_path = os.path.join(
        args.feature_ranking_path,
        "familiar",
        "r_files",
        f"R_file_{args.feature_type}_{args.feature_selection_method}.R",
    )

    vimp_aggregation_method = "enhanced_borda"
    experiment_dir = os.path.join(
        args.feature_ranking_path,
        "familiar",
        "experiments",
        args.feature_type,
        f"{args.feature_selection_method}",
    )

    perform_familiar_experiment(
        feature_file_path=cluster_feature_file_path,
        model_learner="glm_gaussian",
        experiment_dir=experiment_dir,
        familiar_r_file_path=familiar_r_file_path,
        experimental_design="bs(fs,10) + mb",
        batch_id_column="cohort",
        sample_id_column="id_global",
        development_batch_id="training",
        validation_batch_id="validation",
        outcome_name="range_shift",
        outcome_column="range_shift",
        outcome_type="continuous",
        parallel_nr_cores=args.threads,
        parallel=True,
        feature_max_fraction_missing=0.01,
        filter_method="none",
        transformation_method="yeo_johnson",
        normalisation_method="standardisation",
        cluster_method="none",
        parallel_preprocessing="parallel_preprocessing",
        fs_method=args.feature_selection_method,
        vimp_aggregation_method=vimp_aggregation_method,
        vimp_aggregation_rank_threshold=10,
        novelty_detector="none",
        optimisation_determine_vimp=True,
        evaluation_metric=["rmse", "r2_score"],
        imputation_method="simple",
        # hyperparameter={"glm_gaussian": {"sign_size": ...}},
        skip_evaluation_elements=[
            "auc_data",
            "calibration_data",
            "calibration_info",
            "confusion_matrix",
            "feature_similarity",
            "sample_similarity",
            "decision_curve_analyis",
            "ice_data",
            "permutation_vimp",
            "univariate_analysis",
            "model_vimp",
            "model_performance",
            "feature_expressions",
            "hyperparameters",
            "prediction_data",
        ],
        smbo_stop_convergent_iterations=1,
    )

    # 6) Export and plot ranking
    feature_ranking = get_feature_ranking(
        args.feature_ranking_file_path,
        experiment_dir,
        vimp_aggregation_method,
    )

    plot_feature_importance(
        feature_ranking,
        args.feature_importance_plot_path,
        args.feature_type,
        args.feature_selection_method,
    )
