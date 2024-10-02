#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:49:21 2024

@author: kiesli21
"""

from pmma.cmd_args import external_validation_parser
import pandas as pd
from pmma.familiar_preparation import evaluate_familiar_experiment, create_feature_table_for_familiar, perform_familiar_experiment, extract_hyperparameters, merge_data_with_predictions
from pmma.visulisation_methods import plot_predicted_vs_actual_range_shift
import os
import json




def perform_external_validation(external_validation_path, familiar_feature_table_path, familiar_r_file_path, feature_type, feature_selection_method, model_learner, signature):
    """
    Conducts external validation.

    Parameters:
    - external_validation_path (str): Path where external validation results and artifacts will be stored.
    - familiar_feature_table_path (str): Path to the feature table used by the FAMILIAR framework.
    - familiar_r_file_path (str): Path to the R script for running the FAMILIAR model.
    - feature_type (str): Type of features used (e.g., genomic, clinical).
    - feature_selection_method (str): Method used for feature selection.
    - model_learner (str): Machine learning algorithm used for modeling.
    - signature (list): List of features constituting the model's signature.

    Returns:
    Tuple[dict, DataFrame]: A tuple containing the external validation results and the predictions DataFrame.
    """        
    
    # Initialize the results dictionary
    external_validation_results = {
        'feature_type': feature_type,
        'feature_selection_method': feature_selection_method,
        'model_learner': model_learner,
        'features': signature,
        'development': {},
        'validation': {}
    }
    
    # Directory for storing experiment results
    experiment_dir = os.path.join(external_validation_path, "familiar", "experiments", feature_type, feature_selection_method, model_learner)

    # Run the experiment using the specified parameters
    perform_familiar_experiment(
        feature_file_path=familiar_feature_table_path,
        model_learner=model_learner,
        experiment_dir=experiment_dir,
        familiar_r_file_path=familiar_r_file_path,
        signature=signature,
        experimental_design="fs + mb + ev",
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
        include_features=signature,  
        hyperparameter={model_learner: {"sign_size": len(signature)}},  # Replace "YourModelLearner" with your model's name, adjust sign_size based on the number of features in your signature
        skip_evaluation_elements=["auc_data", "calibration_data", "calibration_info", "confusion_matrix",
                                  "decision_curve_analyis", "ice_data", "permutation_vimp", "univariate_analysis", 
                                  "model_vimp", "feature_expressions", "fs_vimp"]
    )
    
    # Retrieve results and predictions from the experiment
    results, predictions = evaluate_familiar_experiment(experiment_dir)
    
    hyperparameters = extract_hyperparameters(experiment_dir)
    external_validation_results['hyperparameters'] = hyperparameters
    
    # Process and print model performance metrics
    for cohort in ['development', 'validation']:
        cohort_metrics = results[cohort]
        print(f"{cohort.capitalize()} fold performance: RMSE = {cohort_metrics['rmse'][0]:.2f} [{cohort_metrics['rmse_ci_low'][0]:.2f}, {cohort_metrics['rmse_ci_high'][0]:.2f}], R2 Score = {cohort_metrics['r2_score'][0]:.2f} [{cohort_metrics['r2_score_ci_low'][0]:.2f}, {cohort_metrics['r2_score_ci_high'][0]:.2f}]")
     
        # Update the results dictionary with the processed metrics
        external_validation_results[cohort].update({
            'rmse': cohort_metrics['rmse'][0],
            'rmse_ci_low': cohort_metrics['rmse_ci_low'][0],
            'rmse_ci_high': cohort_metrics['rmse_ci_high'][0],
            'r2_score': cohort_metrics['r2_score'][0],
            'r2_score_ci_low': cohort_metrics['r2_score_ci_low'][0],
            'r2_score_ci_high': cohort_metrics['r2_score_ci_high'][0]
        })
    
    return external_validation_results, predictions

if __name__ == "__main__":
    # Parsing command line arguments for feature ranking
    parser = external_validation_parser("External validation")
    args = parser.parse_args()
    
    print("Start external validation....")

    # Loading data from specified file paths
    data_table = pd.read_csv(args.data_table_path, sep=";")
    feature_table = pd.read_csv(args.feature_file_path, sep=";")
    
    # Define and create a temporary directory for data processing
    temp_data_dir = os.path.join(args.external_validation_path, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)
    
    with open(args.feature_selection_file_path, 'r') as file:
        feature_selection_metrics = json.load(file)
    
    final_signature = feature_selection_metrics["final_signature"]["features"]

    feature_selection_method = feature_selection_metrics["feature_selection_method"]
    model_learner = feature_selection_metrics["model_learner"]
    feature_type = feature_selection_metrics["feature_type"]

    # Constructing file paths for the familiar feature table and R script
    familiar_feature_table_path = os.path.join(temp_data_dir, f"features_{feature_type}_{feature_selection_method}_{model_learner}.csv")
    familiar_r_file_path = os.path.join(args.external_validation_path, f"familiar/r_files/R_file_{feature_type}_{feature_selection_method}_{model_learner}.R")
    
    # Creating a feature table for the FAMILIAR tool
    familiar_feature_table = create_feature_table_for_familiar(data_table, feature_table, final_signature, familiar_feature_table_path)
    
    # Execute familiar
    result_dict, predictions = perform_external_validation(args.external_validation_path, familiar_feature_table_path, familiar_r_file_path, feature_type, feature_selection_method, model_learner, final_signature)
    
    # Prepare predictions table and save it
    final_predictions_table = merge_data_with_predictions(data_table, predictions)
    final_predictions_table.to_csv(args.predictions_file_path, sep = ";", index = False)
    print(f"Predictions successfully saved to {args.predictions_file_path}")
    
    # Saving the final results dict to a file
    with open(args.performance_file_path, 'w') as file:
        json.dump(result_dict, file, indent = 4)
        print(f"External validation results successfully saved to {args.performance_file_path}")
        
    # Plot the predictions
    plot_predicted_vs_actual_range_shift(final_predictions_table, save_path = args.prediction_plot_path, cohort = 'validation')
    print(f"Predictions plot successfully saved to {args.prediction_plot_path}")
        
