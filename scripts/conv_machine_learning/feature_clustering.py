#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
feature_clustering.py

Correlation-based feature clustering and selection of cluster representatives.

Workflow
--------
1) Load the data table and one or multiple feature tables.
2) Merge feature tables on 'id_global' and attach target ('range_shift') from the data table.
3) Restrict to the training cohort (as in the original implementation).
4) Preprocess features:
     - drop rows with NaNs (global row-wise)
     - Yeo–Johnson transform + standardisation per feature
     - remove features that produce errors / NaN / inf during transform
5) Compute Pearson correlation matrix among features, convert to distance:
       dist = 1 - |corr|
6) Perform agglomerative clustering using the precomputed distance matrix.
7) For each cluster select one representative feature based on:
       - "best_predictor": highest |corr(feature, target)|
       - "highest_mean_correlation": highest mean |corr(feature, others in cluster)|
8) Optionally override cluster representatives manually (validated against original feature set).
9) Save:
     - cluster_info CSV
     - metadata JSON

"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import PowerTransformer

from pmma.cmd_args import feature_clustering_parser

# Preserve original global seeding (affects any downstream random operations)
np.random.seed(1)


def process_feature_table_with_correlation_clustering(
    feature_table: pd.DataFrame,
    cluster_threshold: float = 0.1,
    cluster_representative_method: str = "best_predictor",
    cluster_representatives: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cluster features by absolute Pearson correlation and select one representative per cluster.

    Parameters
    ----------
    feature_table:
        Must contain columns: 'id_global', 'range_shift', plus feature columns.
    cluster_threshold:
        AgglomerativeClustering distance threshold applied to the precomputed distance matrix:
            distance = 1 - |corr|.
        Features with distance > threshold will not be merged.
    cluster_representative_method:
        Strategy to select a cluster representative:
            - "best_predictor": maximise |corr(feature, target)|
            - "highest_mean_correlation": maximise mean |corr(feature, others in cluster)|
    cluster_representatives:
        Optional manual override list.
        If provided and not equal to ["all"], it replaces the automatically selected representatives
        in the metadata (after validating that all are present in the original feature set).
        If None or ["all"], the automatically selected representatives are used.

    Returns
    -------
    final_table:
        Columns: id_global, range_shift, and selected representative features (transformed values).
    cluster_info:
        Per-feature metrics table with:
            Cluster, Feature, Correlation with target, Mean correlation with cluster, p-value
    metadata:
        Dictionary with clustering metadata.
    """
    print("Starting feature table processing and clustering...")

    if cluster_representative_method not in ["best_predictor", "highest_mean_correlation"]:
        raise ValueError(f"{cluster_representative_method} not implemented!")

    target_var = "range_shift"

    # Preserve original behaviour: drop any rows with NaNs before processing
    feature_table = feature_table.dropna()

    # Split target/id/features
    target = feature_table[target_var].astype(float)
    ids = feature_table["id_global"]
    features_df = feature_table.drop(columns=[target_var, "id_global"])

    initially_available_features = list(features_df.columns)

    # -------------------------------------------------------------------------
    # Yeo–Johnson transformation + standardisation per feature
    # -------------------------------------------------------------------------
    trans_error_features: List[str] = []
    pt = PowerTransformer(method="yeo-johnson", standardize=True, copy=True)

    for col in list(features_df.columns):
        try:
            features_df[col] = pt.fit_transform(features_df[col].values.reshape(-1, 1))
        except Exception:
            trans_error_features.append(col)
            features_df.drop(col, axis=1, inplace=True)
            print(
                f"Error in yeo-johnson transformation for feature {col}. "
                f"Will be skipped and added to nan_features!"
            )
            continue

        # Preserve checks and messaging (though counts for NaNs after drop were inconsistent originally)
        if np.sum(np.isinf(np.array(features_df[col].values))) > 0:
            trans_error_features.append(col)
            features_df.drop(col, axis=1, inplace=True)
            print(
                f"Inf value detected for feature {col} after transformation. "
                f"Will be skipped and added to nan_features!"
            )
        elif features_df[col].isna().values.any():
            trans_error_features.append(col)
            features_df.drop(col, axis=1, inplace=True)
            counts = int(np.sum(features_df[col].isna().values))
            print(
                f"{counts} NaN values detected for feature {col} after transformation. "
                f"Will be skipped and added to nan_features!"
            )

    # -------------------------------------------------------------------------
    # Correlation distance matrix for clustering: dist = 1 - |corr|
    # -------------------------------------------------------------------------
    corr_matrix = features_df.corr(method="pearson")
    dist_matrix = 1 - corr_matrix.abs()

    # Identify features that are all-NaN in distance matrix; include transform failures
    nan_features = dist_matrix.columns[dist_matrix.isna().all(axis=0)].tolist()
    nan_features = nan_features + trans_error_features

    # Drop all-NaN rows/cols
    dist_matrix = dist_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
    dist = dist_matrix.to_numpy()

    analyzed_features = dist_matrix.columns
    features_df = features_df[analyzed_features]

    # Preserve assertions
    assert not features_df.isna().values.any(), "DataFrame contains NaN values."
    assert not np.isinf(features_df.values).any(), "DataFrame contains infinite values."

    print(f"Features excluded due to NaN / transform errors: {nan_features}")

    # -------------------------------------------------------------------------
    # Agglomerative clustering on precomputed distance matrix
    # -------------------------------------------------------------------------
    print("Performing agglomerative clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        compute_full_tree=True,
        linkage="average",
        distance_threshold=cluster_threshold,
    )
    labels = clustering.fit_predict(dist)

    # -------------------------------------------------------------------------
    # Select representative per cluster
    # -------------------------------------------------------------------------
    cluster_features: Dict[int, List[Dict[str, Any]]] = {}
    cluster_representatives_auto: Dict[int, str] = {}

    for cluster in set(labels):
        cluster_features[cluster] = []
        analyzed_features_subcluster = analyzed_features[labels == cluster]

        # Single-feature cluster
        if len(analyzed_features_subcluster) == 1:
            feature = analyzed_features_subcluster[0]
            _slope, _intercept, r_value, p_value, _std_err = linregress(features_df[feature], target)

            corr_with_target = abs(r_value)
            mean_corr_with_cluster = 1  # not applicable for single feature

            cluster_features[cluster].append(
                {
                    "Feature": feature,
                    "Correlation with target": corr_with_target,
                    "Mean correlation with cluster": mean_corr_with_cluster,
                    "p-value": p_value,
                }
            )
            cluster_representatives_auto[cluster] = feature
            continue

        # Multi-feature cluster
        best_feature: Optional[str] = None
        best_metric = float("-inf")

        for feature in analyzed_features_subcluster:
            _slope, _intercept, r_value, p_value, _std_err = linregress(features_df[feature], target)
            corr_with_target = abs(r_value)

            correlations = []
            for other_feature in analyzed_features_subcluster:
                if feature == other_feature:
                    continue
                correlation, _ = pearsonr(features_df[feature], features_df[other_feature])
                correlations.append(abs(correlation))
            mean_corr_with_cluster = float(np.mean(correlations)) if correlations else 0.0

            cluster_features[cluster].append(
                {
                    "Feature": feature,
                    "Correlation with target": corr_with_target,
                    "p-value": p_value,
                    "Mean correlation with cluster": mean_corr_with_cluster,
                }
            )

            if cluster_representative_method == "best_predictor":
                metric = corr_with_target
            else:  # "highest_mean_correlation"
                metric = mean_corr_with_cluster

            if metric > best_metric:
                best_metric = metric
                best_feature = feature

        # Assign representative
        cluster_representatives_auto[cluster] = str(best_feature)

    # -------------------------------------------------------------------------
    # Build cluster_info output
    # -------------------------------------------------------------------------
    cluster_info = pd.DataFrame(
        [
            {
                "Cluster": cluster,
                "Feature": info["Feature"],
                "Correlation with target": info["Correlation with target"],
                "Mean correlation with cluster": info["Mean correlation with cluster"],
                "p-value": info["p-value"],
            }
            for cluster, feats in cluster_features.items()
            for info in feats
        ]
    )

    # -------------------------------------------------------------------------
    # Build final_table: id_global, range_shift, representative features (transformed values)
    # -------------------------------------------------------------------------
    final_table = pd.DataFrame({"id_global": ids, target_var: target})

    new_columns: Dict[str, Any] = {}
    for _cluster, feature in cluster_representatives_auto.items():
        new_columns[feature] = features_df[feature]

    final_table = pd.concat([final_table, pd.DataFrame(new_columns)], axis=1)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    metadata: Dict[str, Any] = {
        "initially_available_features": sorted(list(initially_available_features)),
        "excluded_features_nan": sorted(list(nan_features)),
        "cluster_representatives": sorted(list(cluster_representatives_auto.values())),
        "num_initially_available_features": len(initially_available_features),
        "num_excluded_features_nan": len(nan_features),
        "num_clusters": len(cluster_representatives_auto.values()),
    }

    # Manual override (preserved semantics: anything other than None or ["all"] triggers validation)
    if cluster_representatives is not None and not (
        isinstance(cluster_representatives, list)
        and len(cluster_representatives) == 1
        and cluster_representatives[0] == "all"
    ):
        print("Manual cluster representatives provided. Validating against feature set...")
        invalid_reps = sorted(set(cluster_representatives) - set(initially_available_features))
        if invalid_reps:
            raise ValueError(
                "The following manually specified cluster representatives are not present "
                f"in the feature set: {invalid_reps}"
            )

        metadata["cluster_representatives"] = sorted(list(cluster_representatives))
        metadata["cluster_representatives_source"] = "manual"
        print("Cluster representatives have been set manually and stored in metadata.")
    else:
        metadata["cluster_representatives_source"] = "calculated"
        print("Cluster representatives have been calculated automatically and stored in metadata.")

    print("Clustering results (metadata):")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("Clustering complete.")

    return final_table, cluster_info, metadata


def perform_feature_clustering(
    data_table_path: str,
    feature_file_paths: Sequence[str],
    cluster_info_file_path: str,
    metadata_file_path: str,
    cluster_distance_threshold: float,
    cluster_representative_method: str,
    cluster_representatives: List[str],
) -> None:
    """
    Load and merge features, restrict to training cohort, run clustering, and save outputs.

    Parameters
    ----------
    data_table_path:
        CSV with at least columns: id_global, cohort, range_shift.
    feature_file_paths:
        List of feature CSVs to merge on id_global.
    cluster_info_file_path:
        Output CSV for per-feature cluster metrics.
    metadata_file_path:
        Output JSON for clustering metadata.
    cluster_distance_threshold:
        Distance threshold used by agglomerative clustering.
    cluster_representative_method:
        "best_predictor" or "highest_mean_correlation".
    cluster_representatives:
        ["all"] to use calculated representatives, otherwise a validated manual list.
    """
    print("Loading data table and feature files...")

    data_table = pd.read_csv(data_table_path, sep=";")

    # Preserve behaviour: remove ref rows
    data_table = data_table[data_table["range_shift"] != "ref"]

    feature_table = pd.DataFrame()

    for feature_path in feature_file_paths:
        feature_type_table = pd.read_csv(feature_path, sep=";")
        if feature_table.empty:
            feature_table = feature_type_table
        else:
            feature_table = pd.merge(feature_table, feature_type_table, on="id_global", how="inner")

    # Attach cohort and target
    feature_table = pd.merge(
        feature_table,
        data_table[["id_global", "cohort", "range_shift"]],
        on="id_global",
        how="inner",
    )

    # Preserve behaviour: training only
    feature_table = feature_table[feature_table["cohort"] == "training"]
    feature_table = feature_table.drop(columns=["cohort"])

    print("Data table and feature files loaded and merged.")
    print("Processing feature table and performing clustering...")

    feature_table, cluster_info, metadata = process_feature_table_with_correlation_clustering(
        feature_table,
        cluster_distance_threshold,
        cluster_representative_method,
        cluster_representatives=cluster_representatives,
    )
    print("Feature clustering completed.")

    # Ensure output directory exists
    directory = os.path.dirname(cluster_info_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    print(f"Saving cluster information to {cluster_info_file_path}")
    cluster_info.to_csv(cluster_info_file_path, sep=";", index=False)

    print(f"Saving metadata to {metadata_file_path}")
    with open(metadata_file_path, "w") as file:
        json.dump(metadata, file, indent=4)

    print("All results saved. Feature clustering execution completed.")


if __name__ == "__main__":
    parser = feature_clustering_parser("feature clustering")
    args = parser.parse_args()

    clustering_args = json.loads(args.clustering_args)

    cluster_representatives_arg = clustering_args.get("cluster_representatives", ["all"])
    if isinstance(cluster_representatives_arg, str):
        cluster_representatives_arg = [cluster_representatives_arg]

    perform_feature_clustering(
        args.data_table_path,
        args.feature_file_paths,
        args.cluster_info_file_path,
        args.metadata_file_path,
        clustering_args["cluster_distance_threshold"],
        clustering_args["cluster_representative_method"],
        cluster_representatives_arg,
    )
