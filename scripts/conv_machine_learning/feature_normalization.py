#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:48:26 2024

@author: kiesli21
"""

import pandas as pd
import numpy as np
from pmma.cmd_args import feature_normalization_parser
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def normalization_abs(feature_table, relevant_ids, reference_ids, feature):
    """
    Normalizes a feature by subtracting the mean value of the reference cohort.

    Args:
    feature_table (pd.DataFrame): DataFrame containing feature values.
    relevant_ids (np.array): Array of IDs for the relevant cohort.
    reference_ids (np.array): Array of IDs for the reference cohort.
    feature (str): The name of the feature to be normalized.

    Returns:
    pd.Series: Normalized feature values.
    """
    # Calculate the mean value for the feature in the reference cohort
    ref_value = np.median(feature_table.loc[np.isin(feature_table["id_global"], reference_ids), feature].values)
    
    # Subtract the reference mean value from the feature values of the relevant cohort
    return feature_table.loc[np.isin(feature_table["id_global"], relevant_ids), feature] - ref_value

def normalization_rel(feature_table, relevant_ids, reference_ids, feature):
    """
    Normalizes a feature by subtracting the mean value of the reference cohort.

    Args:
    feature_table (pd.DataFrame): DataFrame containing feature values.
    relevant_ids (np.array): Array of IDs for the relevant cohort.
    reference_ids (np.array): Array of IDs for the reference cohort.
    feature (str): The name of the feature to be normalized.

    Returns:
    pd.Series: Normalized feature values.
    """
    # Calculate the mean value for the feature in the reference cohort
    ref_value = np.median(feature_table.loc[np.isin(feature_table["id_global"], reference_ids), feature].values)
    
    # Subtract the reference mean value from the feature values of the relevant cohort
    return (feature_table.loc[np.isin(feature_table["id_global"], relevant_ids), feature] - ref_value) / ref_value

def normalize_features(data_table_path, feature_table_path, output_file_path, reference=0):
    """
    Performs feature normalization on a dataset.

    Args:
    data_table_path (str): Path to the CSV file containing data table.
    feature_table_path (str): Path to the CSV file containing feature table.
    output_file_path (str): Path for the output normalized CSV file.
    reference (int, optional): The reference range shift value. Defaults to 0.
    """
    # Read the CSV files into DataFrames
    data_table = pd.read_csv(data_table_path, sep = ";")
    feature_table = pd.read_csv(feature_table_path, sep = ";")
    
    # Initialize a DataFrame to store normalized features
    feature_table_normalized = feature_table[['id_global']].copy()
    
    # Extract feature names, excluding the 'id_cohort' column
    features = feature_table.columns.drop(["id_global"])
    
    # Iterate over each feature and normalize it
    for feature in features:
        for spot_type in np.unique(data_table["spot_type"]):
            for proton_energy in np.unique(data_table["proton_energy"]):
                if "scanned" in spot_type:
                    for spot_number in np.unique(data_table.loc[data_table["spot_type"] == spot_type, "spot_number"]):
                        # Create masks for all relevant and reference IDs
                        mask_all = (data_table["spot_type"] == spot_type) & (data_table["proton_energy"] == proton_energy) & (data_table["spot_number"] == spot_number)
                        mask_reference = mask_all & (data_table["range_shift"] == reference)
                        # Extract relevant and reference IDs
                        relevant_ids = data_table.loc[mask_all, "id_global"].values
                        reference_ids = data_table.loc[mask_reference, "id_global"].values
                        # Normalize the feature values
                        feature_table_normalized.loc[np.isin(feature_table["id_global"], relevant_ids), feature + "_abs"] = normalization_abs(feature_table, relevant_ids, reference_ids, feature)
                        feature_table_normalized.loc[np.isin(feature_table["id_global"], relevant_ids), feature + "_rel"] = normalization_rel(feature_table, relevant_ids, reference_ids, feature)
                        
                else:
                    # Create masks for non-scanned spot types
                    mask_all = (data_table["spot_type"] == spot_type) & (data_table["proton_energy"] == proton_energy)
                    mask_reference = mask_all & (data_table["range_shift"] == reference)
                    # Extract relevant and reference IDs
                    relevant_ids = data_table.loc[mask_all, "id_global"].values
                    reference_ids = data_table.loc[mask_reference, "id_global"].values
                    # Normalize the feature values
                    feature_table_normalized.loc[np.isin(feature_table["id_global"], relevant_ids), feature + "_abs"] = normalization_abs(feature_table, relevant_ids, reference_ids, feature)
                    feature_table_normalized.loc[np.isin(feature_table["id_global"], relevant_ids), feature + "_rel"] = normalization_rel(feature_table, relevant_ids, reference_ids, feature)

    # Save the normalized feature table to a CSV file
    feature_table_normalized.to_csv(output_file_path, index=False, sep = ";")

if __name__ == "__main__":
    # Parsing command line arguments for feature normalization
    parser = feature_normalization_parser("Feature Normalization")
    args = parser.parse_args()
    
    # Perform feature normalization
    normalize_features(args.data_table_path, args.feature_file_path, args.normalized_feature_file_path, reference=0)
    
    print("Feature normalization successfull!")

