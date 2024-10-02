#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:21:21 2023

@author: kiesli21
"""
from pmma.cmd_args import feature_selection_parser
import pandas as pd
from pmma.familiar_preparation import create_familiar_r_file, evaluate_familiar_experiment, create_feature_table_for_familiar, perform_familiar_experiment, merge_data_with_predictions, extract_hyperparameters
from pmma.visulisation_methods import plot_final_signature_features
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor

    
def get_set_of_features(feature_table, features_sorted_by_rank, model_learner, feature_file_path, familiar_r_file_path, 
                              feature_selection_path, feature_selection_method, feature_type, vif_bar_plot_path, max_features=10, 
                              multicollinearity_method = "pearson", multicollinearity_threshold=0.6, max_no_improve_iterations=3,
                              multicollinearity_threshold_vif = 5):
    """
    Perform feature selection to identify the optimal set of features for a given model.
    
    This function incrementally adds features sorted by their importance to a model and evaluates their impact on model performance.
    Feature selection stops when the specified maximum number of features is reached, when there's no improvement in model
    error for a given number of iterations, or when there are no more features to add.
    
    Parameters:
    - feature_table (DataFrame): DataFrame containing the feature data.
    - features_sorted_by_rank (list): List of features sorted by their importance.
    - model_learner (callable): Learning algorithm for model training.
    - feature_file_path (str): Path to the file containing feature data.
    - familiar_r_file_path (str): Path to the R file used in the FAMILIAR tool.
    - max_features (int, optional): Maximum number of features to include. Defaults to 10.
    - multicollinearity_method (str, optional): Select the method to check for multicollinearity.
    - multicollinearity_threshold (float, optional): Threshold for multicollinearity between features. 
    - max_no_improve_iterations (int, optional): Max number of iterations without improvement. Defaults to 3.
    - args (object, optional): Additional arguments for model training or feature selection. Defaults to None.

    Returns:
    - final_signature (list): Final list of selected features.
    - error_data (dict): Dictionary containing error metrics and corresponding features at each iteration.
    """

    # Initialize variables
    selected_features = []
    not_selected_features = []
    final_signature = []
    error_increase_count = 0

    # Data structure to store error metrics
    
    feature_selection_metrics = {
        'number_of_features': [],
        'features': [],
        'development': {
            'rmse': [], 
            'rmse_ci_low': [], 
            'rmse_ci_high': [],
            'r2_score': [], 
            'r2_score_ci_low': [], 
            'r2_score_ci_high': []
        },
        'validation': {
            'rmse': [], 
            'rmse_ci_low': [], 
            'rmse_ci_high': [],
            'r2_score': [], 
            'r2_score_ci_low': [], 
            'r2_score_ci_high': []
        },
        'final_signature': {
            'features': None,
            'hyperparameters': None,
            'development': {
                'rmse': None, 
                'rmse_ci_low': None, 
                'rmse_ci_high': None,
                'r2_score': None, 
                'r2_score_ci_low': None, 
                'r2_score_ci_high': None
            },
            'validation': {
                'rmse': None, 
                'rmse_ci_low': None, 
                'rmse_ci_high': None,
                'r2_score': None, 
                'r2_score_ci_low': None, 
                'r2_score_ci_high': None
            }
        }
    }

    # Normalizing features using PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
    list_features = [feature for feature in feature_table.columns if feature not in ["id_global", "cohort", "range_shift"]]
    normalized_features = pd.DataFrame({
        feature: pt.fit_transform(feature_table.loc[feature_table["cohort"] == "training", feature].values.reshape(-1, 1))[:, 0]
        for feature in list_features
    })

    # Start the feature selection process
    for i in range(1, max_features + 1):
        if error_increase_count == max_no_improve_iterations:
            print("No decrease in error for several iterations. Stopping!")
            break

        # Iterating over features ranked by their importance
        for feature_to_add in features_sorted_by_rank:
            if feature_to_add in selected_features or feature_to_add in not_selected_features:
                continue

            found_correlation = check_multicollinearity_additional_feature(normalized_features, selected_features, feature_to_add, multicollinearity_method, multicollinearity_threshold)
    
            if found_correlation:
                not_selected_features.append(feature_to_add)
                continue
            else:
                selected_features.append(feature_to_add)
                break

        if len(selected_features) < i:
            print("No more features available to add.")
            break

        print(f"{i} features in test signature: {selected_features}")

        # Directory for storing experiment results
        experiment_dir = os.path.join(feature_selection_path, "familiar", "experiments", feature_type, feature_selection_method, model_learner, f"iteration_{i}")

        perform_familiar_experiment(
        feature_file_path=feature_file_path,
        model_learner=model_learner,
        experiment_dir=experiment_dir,
        familiar_r_file_path=familiar_r_file_path,
        signature=selected_features,
        experimental_design="fs + cv(mb,3,3)",
        batch_id_column="cohort",
        sample_id_column="id_global",
        development_batch_id="training",
        validation_batch_id="validation",
        outcome_name="range_shift",
        outcome_column="range_shift",
        outcome_type="continuous",
        parallel_nr_cores=1,  # Or use multiprocessing.cpu_count() for parallel execution
        parallel=True,
        feature_max_fraction_missing=0.01,
        filter_method="none",
        transformation_method="yeo_johnson",
        normalisation_method="standardisation",
        cluster_method="none",
        parallel_preprocessing="parallel_preprocessing",
        fs_method="none",
        vimp_aggregation_method="stability",
        vimp_aggregation_rank_threshold=5,
        novelty_detector="none",
        optimisation_determine_vimp=True,
        evaluation_metric=["rmse", "r2_score"],
        imputation_method="simple",
        include_features=selected_features,  # Optional, will default to signature if not provided
        hyperparameter={model_learner: {"sign_size": len(selected_features)}},  # Replace "YourModelLearner" with your model's name, adjust sign_size based on the number of features in your signature
        skip_evaluation_elements=["auc_data", "calibration_data", "calibration_info", "confusion_matrix",
                                  "decision_curve_analyis", "ice_data", "permutation_vimp", "univariate_analysis", 
                                  "model_vimp", "feature_expressions", "fs_vimp"]
        )


        # Evaluating the experiment and extracting error metrics
        results, predictions = evaluate_familiar_experiment(experiment_dir)
        
        # Extracting RMSE and R² score metrics for development and validation sets
        rmse_dev = results['development']['rmse'][0]
        rmse_ci_low_dev = results['development']['rmse_ci_low'][0]
        rmse_ci_high_dev = results['development']['rmse_ci_high'][0]
        
        r2_score_dev = results['development']['r2_score'][0]
        r2_score_ci_low_dev = results['development']['r2_score_ci_low'][0]
        r2_score_ci_high_dev = results['development']['r2_score_ci_high'][0]
        
        rmse_val = results['validation']['rmse'][0]
        rmse_ci_low_val = results['validation']['rmse_ci_low'][0]
        rmse_ci_high_val = results['validation']['rmse_ci_high'][0]
        
        r2_score_val = results['validation']['r2_score'][0]
        r2_score_ci_low_val = results['validation']['r2_score_ci_low'][0]
        r2_score_ci_high_val = results['validation']['r2_score_ci_high'][0]
        
        # Printing performance metrics for development and validation folds
        print(f"Development fold performance: RMSE = {round(rmse_dev,2)} [{round(rmse_ci_low_dev,2)}, {round(rmse_ci_high_dev,2)}], R2 Score = {round(r2_score_dev,2)} [{round(r2_score_ci_low_dev,2)}, {round(r2_score_ci_high_dev,2)}]")
        print(f"Validation fold performance: RMSE = {round(rmse_val,2)} [{round(rmse_ci_low_val,2)}, {round(rmse_ci_high_val,2)}], R2 Score = {round(r2_score_val,2)} [{round(r2_score_ci_low_val,2)}, {round(r2_score_ci_high_val,2)}]")
        
        # Updating error data with the results
        feature_selection_metrics['number_of_features'].append(len(selected_features))
        feature_selection_metrics["features"].append(selected_features)
        feature_selection_metrics['development']['rmse'].append(rmse_dev)
        feature_selection_metrics['development']['rmse_ci_low'].append(rmse_ci_low_dev)
        feature_selection_metrics['development']['rmse_ci_high'].append(rmse_ci_high_dev)
        feature_selection_metrics['development']['r2_score'].append(r2_score_dev)
        feature_selection_metrics['development']['r2_score_ci_low'].append(r2_score_ci_low_dev)
        feature_selection_metrics['development']['r2_score_ci_high'].append(r2_score_ci_high_dev)
        
        feature_selection_metrics['validation']['rmse'].append(rmse_val)
        feature_selection_metrics['validation']['rmse_ci_low'].append(rmse_ci_low_val)
        feature_selection_metrics['validation']['rmse_ci_high'].append(rmse_ci_high_val)
        feature_selection_metrics['validation']['r2_score'].append(r2_score_val)
        feature_selection_metrics['validation']['r2_score_ci_low'].append(r2_score_ci_low_val)
        feature_selection_metrics['validation']['r2_score_ci_high'].append(r2_score_ci_high_val)
        
        # Check if the validation error has increased
        if rmse_ci_high_val == np.min(feature_selection_metrics['validation']["rmse_ci_high"]):
            final_signature = selected_features.copy()
            error_increase_count = 0
            feature_selection_metrics['final_signature']['features'] = final_signature
            feature_selection_metrics['final_signature']['validation'] = {
            'rmse': rmse_val,
            'rmse_ci_low': rmse_ci_low_val,
            'rmse_ci_high': rmse_ci_high_val,
            'r2_score': r2_score_val,
            'r2_score_ci_low': r2_score_ci_low_val,
            'r2_score_ci_high': r2_score_ci_high_val
            }
            feature_selection_metrics['final_signature']['development'] = {
            'rmse': rmse_dev,
            'rmse_ci_low': rmse_ci_low_dev,
            'rmse_ci_high': rmse_ci_high_dev,
            'r2_score': r2_score_dev,
            'r2_score_ci_low': r2_score_ci_low_dev,
            'r2_score_ci_high': r2_score_ci_high_dev
            }
            
            hyperparameters = extract_hyperparameters(experiment_dir)
            
            final_predictions = predictions.copy()
            
            feature_selection_metrics['final_signature']["hyperparameters"] = hyperparameters
            
            print("New best model found based on lowest rmse_ci_high within tolerance!")
        
        if i == max_features:
            print("Maximum number of features to test reached!")
            break

    print(f"Final signature is {final_signature}.")
    
    check_multicollinearity_in_set(normalized_features, final_signature, multicollinearity_threshold_vif, vif_bar_plot_path)

    return feature_selection_metrics, final_predictions

def check_multicollinearity_additional_feature(normalized_features, selected_features, feature_to_add, multicollinearity_method, multicollinearity_threshold):
    """
    Check for multicollinearity in a set of selected features.

    Args:
        normalized_features (pd.DataFrame): DataFrame containing normalized features.
        selected_features (list): List of already selected features.
        feature_to_add (str): Feature to be evaluated for multicollinearity.
        multicollinearity_method (str): Method to use for multicollinearity checking ('pearson' or 'VIF').
        multicollinearity_threshold (float): Threshold for determining multicollinearity.

    Returns:
        bool: True if multicollinearity is found, False otherwise.

    Raises:
        ValueError: If an invalid multicollinearity method is specified.
    """
    found_correlation = False

    # Skip if no features are selected (first iteration)
    if len(selected_features) > 0:

        if multicollinearity_method == "pearson":
            # Checking for high correlation with already selected features
            for already_selected_feature in selected_features:
                correlation, _ = pearsonr(normalized_features[already_selected_feature].values, normalized_features[feature_to_add].values)
                if abs(correlation) > multicollinearity_threshold:
                    found_correlation = True
                    print(f"Feature {already_selected_feature} and {feature_to_add} highly correlate (abs({round(correlation, 2)})>{multicollinearity_threshold}). Skipping {feature_to_add}!")
                    break

        elif multicollinearity_method == "VIF":
            # Temporarily add feature_to_add to selected_features for VIF calculation
            temp_selected_features = selected_features + [feature_to_add]
            temp_df = normalized_features[temp_selected_features]

            # Calculate the VIFs for the features
            vifs = [variance_inflation_factor(temp_df.values, i) for i in range(temp_df.shape[1])]

            # Check the VIF for the current feature
            if vifs[-1] >= multicollinearity_threshold:
                found_correlation = True
                print(f"Feature {feature_to_add} has high VIF ({round(vifs[-1], 2)}>{multicollinearity_threshold}). Skipping {feature_to_add}!")

        else:
            raise ValueError(f'The multicollinearity method {multicollinearity_method} has not been implemented or is misspelled. Select from ["pearson", "VIF"].')

    return found_correlation

def check_multicollinearity_in_set(normalized_features, features, multicollinearity_threshold, vif_bar_plot_path):
    """
    Check for multicollinearity using Variance Inflation Factor (VIF) and plot the results.

    Args:
        normalized_features (pd.DataFrame): DataFrame containing normalized features.
        features (list): List of features to be evaluated for multicollinearity.
        multicollinearity_threshold (float): Threshold for determining multicollinearity.

    Returns:
        None: Plots the VIF values and prints warnings if multicollinearity is found.

    Raises:
        ValueError: If the DataFrame does not contain the specified features.
    """
    
    assert len(features) > 0, "No features in set to check!"
    
    # Check if all features are in the DataFrame
    missing_features = [f for f in features if f not in normalized_features.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

    # Calculate the VIFs for the features
    temp_df = normalized_features[features]
    if len(features) == 1:
        vifs = pd.DataFrame({
            "Feature": features,
            "VIF": [1.0]
        })
    else:
        vifs = pd.DataFrame({
            "Feature": features,
            "VIF": [variance_inflation_factor(temp_df.values, i) for i in range(temp_df.shape[1])]
        })

    # Check for high VIF values and print warnings
    for _, row in vifs.iterrows():
        if row['VIF'] >= multicollinearity_threshold:
            print(f"Warning: Feature '{row['Feature']}' has high VIF ({row['VIF']}) exceeding the threshold of {multicollinearity_threshold}!")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(vifs['Feature'], vifs['VIF'], color='skyblue')
    plt.axhline(y=multicollinearity_threshold, color='r', linestyle='--')
    plt.title('VIF Values for Each Feature')
    plt.xlabel('Features')
    plt.ylabel('Variance Inflation Factor (VIF)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vif_bar_plot_path)

    print("Variance Inflation Factor (VIF) of all features in signature:")
    print(vifs)



def plot_rmse_r2_result(error_data, plot_file_path, final_signature_size):
    """
    Plots RMSE, R2 score against the number of features in subplots and saves the plot to a file.

    Args:
    - error_data (dict): Dictionary containing error tracking information.
    - plot_file_path (str): File path to save the plot.
    - final_signature_size (int): The number of features in the final signature.

    The error_data dictionary is expected to have the following structure:
    - 'number_of_features': List of number of features used in the model.
    - 'development': Dictionary containing 'rmse', 'r2_score' and their respective confidence intervals.
    - 'validation': Dictionary containing 'rmse', 'r2_score' their respective confidence intervals.
    """
    
   # plt.figure(figsize=(18, 25))

    # Creating subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,12))
    

    # Plotting RMSE with error bars on the first subplot
    ax1.errorbar(error_data['number_of_features'], error_data["development"]['rmse'], 
                 yerr=[np.array(error_data["development"]['rmse']) - np.array(error_data["development"]['rmse_ci_low']), 
                       np.array(error_data["development"]['rmse_ci_high']) - np.array(error_data["development"]['rmse'])],
                 fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label="Development folds", alpha=0.5)

    ax1.errorbar(error_data['number_of_features'], error_data["validation"]['rmse'], 
                 yerr=[np.array(error_data["validation"]['rmse']) - np.array(error_data["validation"]['rmse_ci_low']), 
                       np.array(error_data["validation"]['rmse_ci_high']) - np.array(error_data["validation"]['rmse'])],
                 fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label="Validation folds", alpha=0.5)

    ax1.set_title('RMSE')
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plotting R2 Score with error bars on the second subplot
    ax2.errorbar(error_data['number_of_features'], error_data["development"]['r2_score'], 
                 yerr=[np.array(error_data["development"]['r2_score']) - np.array(error_data["development"]['r2_score_ci_low']), 
                       np.array(error_data["development"]['r2_score_ci_high']) - np.array(error_data["development"]['r2_score'])],
                 fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label="Development folds", alpha=0.5)

    ax2.errorbar(error_data['number_of_features'], error_data["validation"]['r2_score'], 
                 yerr=[np.array(error_data["validation"]['r2_score']) - np.array(error_data["validation"]['r2_score_ci_low']), 
                       np.array(error_data["validation"]['r2_score_ci_high']) - np.array(error_data["validation"]['r2_score'])],
                 fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label="Validation folds", alpha=0.5)

    ax2.set_title('R2 Score')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('R2 Score')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Highlighting the final signature size with vertical lines
    ax1.axvline(x=final_signature_size, color='r', linestyle='--', label='Final Signature Size')
    ax2.axvline(x=final_signature_size, color='r', linestyle='--', label='Final Signature Size')

    ax1.legend()
    ax2.legend()

    # Setting the main title for the figure
    plt.suptitle('RMSE, R2 Score vs. Number of Features')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_file_path)

    print(f"Feature selection plot saved to {plot_file_path}")
    
    return
 
if __name__ == "__main__":
    # Parsing command line arguments for feature ranking
    parser = feature_selection_parser("Feature selection")
    args = parser.parse_args()
    
    selection_args = json.loads(args.selection_args) 
    
    print("Start feature selection....")

    # Loading data from specified file paths
    data_table = pd.read_csv(args.data_table_path, sep=";")
    feature_table = pd.read_csv(args.feature_file_path, sep=";")
    feature_ranking_table = pd.read_csv(args.feature_ranking_file_path, sep=";")

    # Sorting features based on their ranking
    features_sorted_by_rank = feature_ranking_table.sort_values(by='score', ascending=False)['feature'].tolist()
    
    # Define and create a temporary directory for data processing
    temp_data_dir = os.path.join(args.feature_selection_path, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)

    # Constructing file paths for the familiar feature table and R script
    familiar_feature_table_path = os.path.join(temp_data_dir, f"features_{args.feature_type}_{args.feature_selection_method}.csv")
    familiar_r_file_path = os.path.join(args.feature_selection_path, f"familiar/r_files/R_file_{args.feature_type}_{args.feature_selection_method}_{args.model_learner}.R")
    
    # Creating a feature table for the FAMILIAR tool
    ranked_feature_table = create_feature_table_for_familiar(data_table, feature_table, features_sorted_by_rank, familiar_feature_table_path)
    
    # Performing forward feature selection
    feature_selection_metrics, predictions = get_set_of_features(ranked_feature_table, features_sorted_by_rank, args.model_learner, familiar_feature_table_path, familiar_r_file_path, 
                                                          args.feature_selection_path, args.feature_selection_method, args.feature_type, args.vif_bar_plot_path,
                                                          **selection_args)
    
    os.makedirs(os.path.dirname(args.predictions_file_path), exist_ok=True)
    # Prepare predictions table and save it
    final_predictions_table = merge_data_with_predictions(data_table, predictions)
    final_predictions_table.to_csv(args.predictions_file_path, sep = ";", index = False)
    print(f"Predictions successfully saved to {args.predictions_file_path}")
    
    # Plotting the final signature features
    final_signature = feature_selection_metrics["final_signature"]["features"]
    plot_final_signature_features(ranked_feature_table, final_signature, args.signature_plot_path, args.feature_type, args.feature_selection_method, args.model_learner)
    
    feature_selection_metrics["feature_type"] = args.feature_type
    feature_selection_metrics["feature_selection_method"] = args.feature_selection_method
    feature_selection_metrics["model_learner"] = args.model_learner
    
    # Plotting RMSE with error bars from the feature selection process
    plot_rmse_r2_result(feature_selection_metrics, args.feature_selection_plot_path, len(final_signature))
    
    os.makedirs(os.path.dirname(args.feature_selection_file_path), exist_ok=True)
    # Saving the feature selection results
    with open(args.feature_selection_file_path, 'w') as file:
        json.dump(feature_selection_metrics, file, indent = 4)
        print(f"Signature successfully saved to {args.feature_selection_file_path}")
        
    os.makedirs(os.path.dirname(args.signature_file_path), exist_ok=True)
    
    # Convert the list into a DataFrame with a column name (optional)
    df = pd.DataFrame(final_signature)
    
    # Write the DataFrame to a CSV file without a header and without an index
    df.to_csv(args.signature_file_path, index=False, header=False)
    
    print(f"Signature successfully saved to {args.signature_file_path}")

