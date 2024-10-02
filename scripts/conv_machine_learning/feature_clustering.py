#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:49:25 2023

@author: kiesli21
"""

# Python module imports
import numpy as np
np.random.seed(1)
import pandas as pd

from pmma.cmd_args import feature_clustering_parser

from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr
import os
import json

def process_feature_table_with_correlation_clustering(feature_table, cluster_threshold = 0.1):
    """
    Process and cluster features in a feature table using correlation-based clustering.
    
    This function takes a DataFrame of features and performs clustering based on the correlation
    among features. It involves preprocessing steps like feature standardization, Yeo-Johnson
    transformation, and Pearson correlation computation. The function then applies agglomerative
    clustering and selects the most representative feature from each cluster based on its correlation
    with the target variable.
    
    Parameters:
    feature_table (DataFrame): A pandas DataFrame containing the feature data along with 'id_global' and 'range_shift'.
    cluster_threshold (float, optional): The threshold for the distance matrix to determine clusters. Defaults to 0.2.
    
    Returns:
    final_table (DataFrame): A pandas DataFrame containing 'id_global', 'range_shift', and the most representative features from each cluster.
    cluster_info (DataFrame): A pandas DataFrame containing cluster information, including the cluster number, feature name, and its correlation with the target variable.
    metadata (dict): A dictionary containing metadata about the clustering process, such as excluded features, number of initially available features, and cluster representatives.
    
    Notes:
    - The function applies a Yeo-Johnson transformation to the features for normalization.
    - Pearson correlation is used to compute the correlation matrix, which is then transformed into a distance matrix for clustering.
    - Clusters are determined based on the agglomerative clustering method, using the precomputed distance matrix.
    - The function identifies and selects the most representative feature from each cluster based on the highest absolute correlation with the target variable 'range_shift'.
    - Metadata about the clustering process, such as excluded features due to standard deviation or NaN values, is generated and returned.
    """
    print("Starting feature table processing and clustering...")
    
    target_var = "range_shift"
    
    # Separate target variable and id
    target = feature_table[target_var]
    ids = feature_table['id_global']
    features_df = feature_table.drop(columns=[target_var, 'id_global'])

    # Include initially available features in metadata
    initially_available_features = list(features_df.columns)


    trans_error_features = []
    # Apply Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
    for col in features_df.columns:
        try:
            features_df[col] = pt.fit_transform(features_df[col].values.reshape(-1,1))
            
        except Exception:
            trans_error_features.append(col)
            features_df.drop(col, axis = 1, inplace = True)
            print(f"Error in yeo-johnson transformation for feature {col}. Will be skipped and added to nan_features!")
            continue
        
        if np.sum(np.isinf(np.array(features_df[col].values))) > 0:
            trans_error_features.append(col)
            features_df.drop(col, axis = 1, inplace = True)
            print(f"Inf value detected for feature {col} after transformation. Will be skipped and added to nan_features!")
            
        elif features_df[col].isna().values.any():
            trans_error_features.append(col)
            features_df.drop(col, axis = 1, inplace = True)
            print(f"NaN value detected for feature {col} after transformation. Will be skipped and added to nan_features!")

    # Calculate Pearson-correlation table for clustering
    corr_matrix = features_df.corr(method='pearson')
    
    # Calculate distance matrix as 1 minus the absolute correlation values
    dist_matrix = 1 - corr_matrix.abs()
    
    # Handling NaN values in the distance matrix
    nan_features = dist_matrix.columns[dist_matrix.isna().all(axis=0)].tolist()
    nan_features = nan_features + trans_error_features
    dist_matrix = dist_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    dist = dist_matrix.to_numpy()

    analyzed_features = dist_matrix.columns
    
    features_df = features_df[analyzed_features]

    # Assertion to check for any NaN or inf values
    assert not features_df.isna().values.any(), "DataFrame contains NaN values."
    assert not np.isinf(features_df.values).any(), "DataFrame contains infinite values."

    print(f"Features excluded due to NaN values: {nan_features}")

    # Perform clustering
    print("Performing agglomerative clustering...")

    # Perform clustering using the precomputed distance matrix
    # above cluster_threshold will not be merged
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', compute_full_tree=True, linkage='average', distance_threshold=cluster_threshold)
    labels = clustering.fit_predict(dist)

    # Create structures to store cluster information and representative features
    cluster_features = {}
    cluster_representatives = {}

    # find representative feature by looking at the mean correlation to all other features in cluster
    for cluster in set(labels):
        cluster_features[cluster] = []
        analyzed_features_subcluster = analyzed_features[labels == cluster]
    
        if len(analyzed_features_subcluster) == 1:
            best_feature = analyzed_features_subcluster[0]
            cluster_representatives[cluster] = best_feature
            cluster_features[cluster].append((best_feature, 1.0))
        else:
    
            best_feature = None
            best_mean_corr = float('-inf')
        
            for feature in analyzed_features_subcluster:
                correlations = []
        
                for other_feature in analyzed_features_subcluster:
                    if feature != other_feature:
                        correlation, _ = pearsonr(features_df[feature], features_df[other_feature])
                        correlations.append(abs(correlation))
        
                mean_corr = np.mean(correlations)
                cluster_features[cluster].append((feature, mean_corr))
    
                # Identify the most representative feature (highest mean correlation)
                if mean_corr > best_mean_corr:
                    best_mean_corr = mean_corr
                    best_feature = feature
            
        cluster_representatives[cluster] = best_feature

    # Create a DataFrame to store all cluster information
    cluster_info = pd.DataFrame([(cluster, feature, corr) for cluster, features in cluster_features.items() for feature, corr in features], columns=['Cluster', 'Feature', 'Mean correlation'])

    # Create the final table with id_global, range_shift, and the representative features
    final_table = pd.DataFrame({'id_global': ids, target_var: target})
    
    # Collect new columns in a dictionary
    new_columns = {}

    for cluster, feature in cluster_representatives.items():
        new_columns[feature] = features_df[feature]

    # Concatenate all new columns to the final_table at once
    final_table = pd.concat([final_table, pd.DataFrame(new_columns)], axis=1)

    # Metadata creation (updated)
    metadata = {
        'initially_available_features': sorted(list(initially_available_features)),
        'excluded_features_nan': sorted(list(nan_features)),
        'cluster_representatives': sorted(list(cluster_representatives.values())),
        'num_initially_available_features': len(initially_available_features),
        'num_excluded_features_nan': len(nan_features),
        'num_clusters': len(cluster_representatives.values())
    }

    print("Clustering results:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("Clustering complete.")

    return final_table, cluster_info, metadata


def perform_feature_clustering(data_table_path, feature_file_paths, cluster_info_file_path, metadata_file_path, cluster_distance_threshold):
    """
    Perform feature clustering on a set of features extracted from a data table.

    This function processes multiple feature files and a data table to perform feature clustering. 
    It involves loading the data table and feature files, merging these datasets on a common identifier, 
    and then filtering and processing the data to perform correlation-based clustering of features. 
    The resulting clustered feature information and metadata are saved to specified file paths.

    Parameters:
    data_table_path (str): Path to the data table CSV file. The data table should contain columns for 'id_global', 'cohort', and 'range_shift'.
    feature_file_paths (list): A list of paths to CSV files containing different types of feature data. Each feature file is expected to contain a 'id_global' column for merging.
    cluster_info_file_path (str): File path to save the resulting cluster information as a CSV file.
    metadata_file_path (str): File path to save the clustering metadata as a JSON file.
    cluster_threshold (float, optional): The threshold for the distance matrix to determine clusters.
    
    Returns:
    None: This function does not return a value but saves the cluster information and metadata to specified file paths.

    Note:
    The function assumes a specific structure for the input data files and a correlation-based clustering process. 
    The 'cohort' column is used to filter the data for training cohort before clustering.
    """
    print("Loading data table and feature files...")
    
    # Load data table as pandas dataframe
    data_table = pd.read_csv(data_table_path, sep=";")
    
    # Initialize an empty DataFrame for the possible combination of different feature types
    feature_table = pd.DataFrame()
    
    # Loop through each feature file
    for feature_path in feature_file_paths:
        
        # Load the feature file as a DataFrame
        feature_type_table = pd.read_csv(feature_path, sep=";")
    
        # If merged_df is empty, initialize it with the first feature file
        if feature_table.empty:
            feature_table = feature_type_table
        else:
            # Merge with the existing merged_df on 'id_global'
            feature_table = pd.merge(feature_table, feature_type_table, on='id_global', how='inner')

    # Merge feature_table with the relevant columns of data_table
    feature_table = pd.merge(feature_table, data_table[['id_global', 'cohort', "range_shift"]], on='id_global', how='inner')
    
    # Filter out rows where 'cohort' is not 'training'
    feature_table = feature_table[feature_table['cohort'] == 'training']
    
    # Drop the 'cohort' column as it's no longer needed
    feature_table = feature_table.drop(columns=['cohort'])
    
    print("Data table and feature files loaded and merged.")
    
    print("Processing feature table and performing clustering...")
    # perform feature clustering
    feature_table, cluster_info, metadata = process_feature_table_with_correlation_clustering(feature_table, cluster_distance_threshold)
    print("Feature clustering completed.")
    
    # Extract the directory from the file path
    directory = os.path.dirname(cluster_info_file_path)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
      
    print(f"Saving cluster information to {cluster_info_file_path}")
    cluster_info.to_csv(cluster_info_file_path, sep = ";", index = False)
    
    print(f"Saving metadata to {metadata_file_path}")
    # Save metadata to a JSON file
    with open(metadata_file_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print("All results saved. Feature clustering execution completed.")
    return

if __name__ == "__main__":
    # Obtain arguments for feature calculation
    parser = feature_clustering_parser("feature clustering")
    args = parser.parse_args()
    
    clustering_args = json.loads(args.clustering_args)    
    
    perform_feature_clustering(args.data_table_path, args.feature_file_paths, args.cluster_info_file_path, args.metadata_file_path, clustering_args["cluster_distance_threshold"])
    
       
    
    