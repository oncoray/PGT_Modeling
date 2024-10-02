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

from pmma.cmd_args import feature_filtering_parser

from scipy.stats import mannwhitneyu, linregress
import os
import json


def filter_features(data_table_path, feature_table_path, output_file_path, filter_info_file_path, cohort1 = "static", cohort2 = "static_accum10", n_diff_groups = 3, p_value_statistic = 0.05, p_value_importance = 0.2, ref = 0):
    """
    Filters features from the given feature table based on statistical tests and importance.

    Parameters:
    - data_table_path (str): Path to the CSV file containing the data table.
    - feature_table_path (str): Path to the CSV file containing the feature table.
    - output_file_path (str): Path to save the filtered feature table.
    - filter_info_file_path (str): Path to save the p-values and filtering information.
    - cohort1 (str): The first cohort to compare (default is "static").
    - cohort2 (str): The second cohort to compare (default is "static_accum10").
    - n_diff_groups (int): Number of different groups required to consider a feature non-robust (default is 3).
    - p_value_statistic (float): P-value threshold for statistical tests (default is 0.05).
    - p_value_importance (float): P-value threshold for feature importance (default is 0.2).
    - ref (int): Reference range shift that does not count for important difference (default is 0).

    Returns:
    - metadata (dict): Dictionary containing metadata about the filtering process, including initially available features, excluded features, non-robust features, unimportant features, and filtered features.

    """
    
    print("Starting feature filtering...")
    
    target_var = "range_shift"
    
    # Read the CSV files into DataFrames
    data_table = pd.read_csv(data_table_path, sep = ";")
    feature_table = pd.read_csv(feature_table_path, sep = ";")
    
    list_range_shifts = sorted(list(np.unique(data_table[target_var].values)))
    list_energies = sorted(list(np.unique(data_table["proton_energy"].values)))

    # Include initially available features in metadata
    initially_available_features = list(feature_table.columns)
    initially_available_features.remove("id_global")

    ids_train = list(data_table.loc[data_table["cohort"] == "training", "id_global"].values)


    for energy in list_energies:
    # Exclude features with standard deviation of 0
        std_devs = feature_table.loc[np.isin(feature_table["id_global"].values, ids_train) & (data_table["proton_energy"] == energy),:].drop(columns=['id_global']).std()
        excluded_features_std = std_devs[std_devs == 0].index.tolist()
        feature_table = feature_table.drop(columns=excluded_features_std)
    
    filtered_features = list(feature_table.columns)
    filtered_features.remove("id_global")
    
    print(f"Features excluded due to standard deviation: {excluded_features_std}")
    
    assert cohort1 in list(data_table["spot_type"].values), f"{cohort1} not in data table!"
    assert cohort2 in list(data_table["spot_type"].values), f"{cohort2} not in data table!"
    
    features_not_robust_statistics = []
    unimportant_features = []
    
    # Initialize DataFrame to store p-values for each feature across range shifts
    p_values_df = pd.DataFrame(columns=["feature", "proton_energy"] + list_range_shifts + ["univariate_test"])
    
    
    
    for feature in filtered_features:
        
        for energy in list_energies:
        
            counter = 0
            p_values = [feature, energy]  # Start with the feature name
            
            for range_shift in list_range_shifts:
                ids_1 = list(data_table.loc[(data_table["spot_type"] == cohort1) & (data_table[target_var] == range_shift) & (data_table["proton_energy"] == energy), 'id_global'].values)
                ids_2 = list(data_table.loc[(data_table["spot_type"] == cohort2) & (data_table[target_var] == range_shift) & (data_table["proton_energy"] == energy), 'id_global'].values)
                
                values_1 = feature_table.loc[np.isin(feature_table["id_global"].values, ids_1), feature].values
                values_2 = feature_table.loc[np.isin(feature_table["id_global"].values, ids_2), feature].values
                
                _, p_value_test_statistic = mannwhitneyu(values_1, values_2)
                
                # ref does not count for important difference
                if p_value_test_statistic <= p_value_statistic and range_shift != ref:
                    counter += 1
                
                p_values.append(np.round(p_value_test_statistic, 4))  # Append p-value for the current range shift
            
            if counter >= n_diff_groups:
                features_not_robust_statistics.append(feature)
            
            ids_train = list(data_table.loc[(data_table["cohort"] == "training") & (np.isin(data_table["spot_type"], [cohort1, cohort2])) & (data_table["proton_energy"] == energy), "id_global"].values)
            
            feature_train = feature_table.loc[np.isin(feature_table["id_global"].values, ids_train), feature].values
            target_train = data_table.loc[np.isin(data_table["id_global"].values, ids_train), target_var].values
            
            _, _, _, p_value_test_importance, _ = linregress(feature_train, target_train)
            
            if p_value_test_importance >= p_value_importance:
                unimportant_features.append(feature)
            
            p_values.append(np.round(p_value_test_importance, 4))
            
            p_values_df.loc[len(p_values_df)] = p_values
            
        if len(list_energies) > 1:
            counter = 0
            p_values = [feature, "combined"]  # Start with the feature name
            
            for range_shift in list_range_shifts:
                ids_1 = list(data_table.loc[(data_table["spot_type"] == cohort1) & (data_table[target_var] == range_shift), 'id_global'].values)
                ids_2 = list(data_table.loc[(data_table["spot_type"] == cohort2) & (data_table[target_var] == range_shift), 'id_global'].values)
                
                values_1 = feature_table.loc[np.isin(feature_table["id_global"].values, ids_1), feature].values
                values_2 = feature_table.loc[np.isin(feature_table["id_global"].values, ids_2), feature].values
                
                _, p_value_test_statistic = mannwhitneyu(values_1, values_2)
                
                # ref does not count for important difference
                if p_value_test_statistic <= p_value_statistic and range_shift != ref:
                    counter += 1
                
                p_values.append(np.round(p_value_test_statistic, 4))  # Append p-value for the current range shift
            
            if counter >= n_diff_groups:
                features_not_robust_statistics.append(feature)
            
            ids_train = list(data_table.loc[(data_table["cohort"] == "training") & (np.isin(data_table["spot_type"], [cohort1, cohort2])), "id_global"].values)
            
            feature_train = feature_table.loc[np.isin(feature_table["id_global"].values, ids_train), feature].values
            target_train = data_table.loc[np.isin(data_table["id_global"].values, ids_train), target_var].values
            
            _, _, _, p_value_test_importance, _ = linregress(feature_train, target_train)
            
            if p_value_test_importance >= p_value_importance:
                unimportant_features.append(feature)
            
            p_values.append(np.round(p_value_test_importance, 4))
            
            p_values_df.loc[len(p_values_df)] = p_values
        
    features_to_exclude = np.unique(unimportant_features + features_not_robust_statistics)
    
    feature_table = feature_table.drop(columns=features_to_exclude)

    filtered_features = list(feature_table.columns)
    filtered_features.remove("id_global") 
    
    feature_table_filtered = feature_table

    # Metadata creation (updated)
    metadata = {
        'initially_available_features': sorted(list(initially_available_features)),
        'excluded_features_std': sorted(list(excluded_features_std)),
        'features_not_robust_statistics': sorted(features_not_robust_statistics),
        'unimportant_features': sorted(unimportant_features),
        'filtered_features': sorted(filtered_features),
        'num_initially_available_features': len(initially_available_features),
        'num_excluded_features_std': len(excluded_features_std),
        'num_features_not_robust_statistics': len(features_not_robust_statistics),
        'num_unimportant_features': len(unimportant_features),
        'num_filtered_features': len(filtered_features),
        'num_excluded_features': len(initially_available_features)-len(filtered_features)
    }

    print("Filtering results:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Extract the directory from the file path
    directory = os.path.dirname(output_file_path)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the normalized feature table to a CSV file
    feature_table_filtered.to_csv(output_file_path, index=False, sep = ";")

    # Save the p-values DataFrame to a CSV file
    directory = os.path.dirname(filter_info_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    p_values_df.to_csv(filter_info_file_path, index=False, sep=";")

    return metadata

if __name__ == "__main__":
    # Obtain arguments for feature calculation
    parser = feature_filtering_parser("feature filtering")
    args = parser.parse_args()

    
    filtering_args = json.loads(args.filtering_args)    
        
    metadata = filter_features(args.data_table_path, 
                               args.feature_file_path, 
                               args.filtered_feature_file_path, 
                               args.filter_info_file_path,
                               filtering_args["cohort1"], 
                               filtering_args["cohort2"], 
                               filtering_args["n_diff_groups"], 
                               filtering_args["statistic_p_value_threshold"], 
                               filtering_args["importance_p_value_threshold"], 
                               filtering_args["reference"])
    
    # Extract the directory from the file path
    directory = os.path.dirname(args.metadata_file_path)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
      
    
    print(f"Saving metadata to {args.metadata_file_path}")
    # Save metadata to a JSON file
    with open(args.metadata_file_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print("All results saved. Feature clustering execution completed.")
    
    
    