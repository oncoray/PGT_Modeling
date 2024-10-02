#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:10:42 2023

@author: kiesli21
"""
import pandas as pd
from pmma.cmd_args import feature_combination_parser

if __name__ == "__main__":
    # Obtain arguments for feature calculation
    parser = feature_combination_parser("feature clustering")
    args = parser.parse_args()
    
    # Initialize an empty DataFrame to hold the merged result
    merged_df = None

    # Iterate over the list of CSV file paths
    for path in args.feature_file_paths:
        # Read the current CSV file into a DataFrame
        current_df = pd.read_csv(path, sep = ";")

        # If merged_df is not yet initialized, set it to the current DataFrame
        if merged_df is None:
            merged_df = current_df
        else:
            # Perform merge (left join) on the join_column
            merged_df = pd.merge(merged_df, current_df, on="id_global", how='left')

    # Save the merged DataFrame to the specified output path
    merged_df.to_csv(args.combined_feature_file_path, index=False, sep = ";")
    print(f"Merged CSV saved to {args.combined_feature_file_path}")