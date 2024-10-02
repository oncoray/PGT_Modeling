#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:51:56 2023

@author: kiesli21
"""

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pmma.cmd_args import feature_selection_summary_parser

def summarize_and_plot_heatmaps(holding_instance, json_file_paths, output_path):
    """
    Summarizes results and creates a heatmap for a specific holding instance.

    Parameters:
    - holding_instance (str): The feature type, feature selection method, or model learner held constant.
    - json_file_paths (list): List of paths to JSON files containing feature_selection_metrics.
    - output_path (str): Path where the heatmap will be saved.
    """

    # Dictionary to hold the combined data
    combined_data = {'feature_type': [], 'feature_selection_method': [], 'model_learner': [], 'rmse_val': []}

    # Load data from each JSON file
    for file_path in json_file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Extract relevant information
            combined_data['feature_type'].append(data['feature_type'])
            combined_data['feature_selection_method'].append(data['feature_selection_method'])
            combined_data['model_learner'].append(data['model_learner'])
            combined_data['rmse_val'].append(data["final_signature"]['validation']['rmse'])

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(combined_data)

    # Determine pivot table dimensions based on holding instance
    if holding_instance in df['feature_type'].unique():
        index, columns = 'model_learner', 'feature_selection_method'
    elif holding_instance in df['feature_selection_method'].unique():
        index, columns = 'feature_type', 'model_learner'
    elif holding_instance in df['model_learner'].unique():
        index, columns = 'feature_type', 'feature_selection_method'
    else:
        raise ValueError(f"Invalid holding instance: {holding_instance}")

    # Create the pivot table
    heatmap_df = df.pivot_table(index=index, columns=columns, values='rmse_val')

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'RMSE of internal validation folds - {holding_instance}')
    plt.ylabel(index)
    plt.xlabel(columns)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    
    parser = feature_selection_summary_parser("Feature selection summary")
    args = parser.parse_args()

    summarize_and_plot_heatmaps(args.holding_instance, args.feature_selection_file_paths, args.heatmap_plot_path)