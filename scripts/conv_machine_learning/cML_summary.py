# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:34:00 2024

@author: Phase
"""
import pandas as pd
import os
from pmma.cmd_args import cML_summary_parser
from pmma.visulisation_methods import plot_performance_heatmaps
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)

def bootstrap_confidence_interval(y_true, y_pred, stat_function, n_bootstraps=1000):
    """
    Calculate the bootstrap confidence interval for a given statistic between y_true and y_pred.

    Parameters:
    - y_true: array-like, true values.
    - y_pred: array-like, predicted values.
    - stat_function: function, the statistic function to apply to y_true and y_pred.
                     It should take two arrays as input and return a single number (the statistic).
    - n_bootstraps: int, number of bootstrap samples to use for computing the confidence interval.

    Returns:
    - ci_low: float, the lower bound of the confidence interval.
    - ci_high: float, the upper bound of the confidence interval.
    """
    
    # Convert inputs to numpy arrays for efficient computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize a list to store the bootstrap statistics
    bootstrap_stats = []
    
    # Perform bootstrap resampling
    for _ in range(n_bootstraps):
        # Randomly sample indices with replacement
        indices = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        # Resample y_true and y_pred according to the sampled indices
        y_true_resampled, y_pred_resampled = y_true[indices], y_pred[indices]
        # Calculate the statistic for the resampled data
        statistic = stat_function(y_true_resampled, y_pred_resampled)
        # Append the calculated statistic to the list
        bootstrap_stats.append(statistic)
    
    # Convert the list of bootstrap statistics to a numpy array for efficient percentile calculation
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate the 2.5th and 97.5th percentiles of the bootstrap statistics to determine the confidence interval
    ci_low, ci_high = np.percentile(bootstrap_stats, [2.5, 97.5])
    
    return ci_low, ci_high


def summarize_performances_from_predictions(df):
    """Process df to calculate RMSE, R2, and confidence intervals for each group."""

    # Group by the specified columns
    grouped = df.groupby(['feature_type', 'feature_selection_method', 'model_learner', 'cohort', 'cv_data_set', 'spot_type', 'proton_energy'])
    
    summary_rows = []
    
    for name, group in grouped:
        y_true = group['range_shift']
        y_pred = group['predicted_range_shift']
        
        # Calculate metrics
        rmse = calculate_rmse(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)
        
        # Calculate confidence intervals through bootstrapping
        rmse_ci = bootstrap_confidence_interval(y_true, y_pred, calculate_rmse, n_bootstraps=1000)
        r2_ci = bootstrap_confidence_interval(y_true, y_pred, calculate_r2, n_bootstraps=1000)
        
        assert len(np.unique(group['signature'].values)) == 1, f"Signature is not consistent within group {name}!"
        
        signature = group['signature'].iloc[0]  # Assuming signature is consistent within the group
        
        summary_rows.append({
            'feature_type': name[0],
            'feature_selection_method': name[1],
            'model_learner': name[2],
            'signature': signature,
            'sign_size': len(signature),
            'cohort': name[3],
            'cv_data_set': name[4],
            'spot_type': name[5],
            'proton_energy': name[6],
            'RMSE': rmse,
            'RMSE CI Low': rmse_ci[0],
            'RMSE CI High': rmse_ci[1],
            'R2': r2,
            'R2 CI Low': r2_ci[0],
            'R2 CI High': r2_ci[1]
        })
        
    # Group by the specified columns
    grouped = df.groupby(['feature_type', 'feature_selection_method', 'model_learner', 'cohort', 'cv_data_set'])
    
    for name, group in grouped:
        y_true = group['range_shift']
        y_pred = group['predicted_range_shift']
        
        # Calculate metrics
        rmse = calculate_rmse(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)
        
        # Calculate confidence intervals through bootstrapping
        rmse_ci = bootstrap_confidence_interval(y_true, y_pred, calculate_rmse, n_bootstraps=1000)
        r2_ci = bootstrap_confidence_interval(y_true, y_pred, calculate_r2, n_bootstraps=1000)
        
        assert len(np.unique(group['signature'].values)) == 1, f"Signature is not consistent within group {name}!"
        signature = group['signature'].iloc[0]  # Assuming signature is consistent within the group
        
        summary_rows.append({
            'feature_type': name[0],
            'feature_selection_method': name[1],
            'model_learner': name[2],
            'signature': signature,
            'sign_size': len(signature),
            'cohort': name[3],
            'cv_data_set': name[4],
            'spot_type': "combined",
            'proton_energy': "combined",
            'RMSE': rmse,
            'RMSE CI Low': rmse_ci[0],
            'RMSE CI High': rmse_ci[1],
            'R2': r2,
            'R2 CI Low': r2_ci[0],
            'R2 CI High': r2_ci[1]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def parse_filename(filename):
    """
    Parses the filename to extract feature_type, feature_selection_method, and model_learner.
    """
    parts = filename.split('_')
    # Extract metadata based on the new naming convention
    feature_type_index = parts.index('ft') + 1
    feature_selection_method_index = parts.index('fsm') + 1
    model_learner_index = parts.index('ml') + 1
    
    feature_type = parts[feature_type_index]
    feature_selection_method = parts[feature_selection_method_index]
    model_learner = '_'.join(parts[model_learner_index:-1]) # Since model_learner may contain underscores
    
    return feature_type, feature_selection_method, model_learner

def concatenate_prediction_files(file_paths):
    """
    Reads and concatenates files into a single DataFrame.
    """
    all_data = []
    for path in file_paths:
        filename = os.path.basename(path)
        feature_type, feature_selection_method, model_learner = parse_filename(filename)
        
        df = pd.read_csv(path, sep = ";")
        df['feature_type'] = feature_type
        df['feature_selection_method'] = feature_selection_method
        df['model_learner'] = model_learner
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def concatenate_signature_files(file_paths):
    """
    Reads and concatenates files into a single DataFrame.
    """
    all_data = []
    
    for path in file_paths:
        filename = os.path.basename(path)
        feature_type, feature_selection_method, model_learner = parse_filename(filename)
        
        df = pd.read_csv(path, header = None)
        df_sign = {
            'feature_type': feature_type,
            'feature_selection_method': feature_selection_method,
            'model_learner': model_learner,
            "signature": list(df.iloc[:,0].values)
        }
        
        all_data.append(df_sign)
    
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    # Parsing command line arguments for feature ranking
    parser = cML_summary_parser("cML summary")
    args = parser.parse_args()
    
    print("Start summary of cML...")

    signature_summary = concatenate_signature_files(args.signature_file_paths)

    print(signature_summary)

    predictions_summary = concatenate_prediction_files(args.individual_prediction_files)
    
    
    
    predictions_summary = pd.merge(signature_summary, predictions_summary, 
                          on=['feature_type', 'feature_selection_method', 'model_learner'], 
                          how='inner')
    print(predictions_summary)
    
    os.makedirs(os.path.dirname(args.summary_predictions_file_path), exist_ok=True)
    
    predictions_summary.to_csv(args.summary_predictions_file_path, index=False, sep = ";")
    
    print("Summary of model predictions have been saved to: ", args.summary_predictions_file_path)
    
    summary_df = summarize_performances_from_predictions(predictions_summary)

    summary_df.to_csv(args.summary_performance_file_path, index=False, sep = ";")

    print("Summary of model performances have been saved to: ", args.summary_performance_file_path)
    
    plot_performance_heatmaps(summary_df, args.heatmap_plot_dir_path)
    
    print("Heatmaps have been saved to: ", args.heatmap_plot_dir_path)
    
