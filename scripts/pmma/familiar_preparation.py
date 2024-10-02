# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:26:30 2023

@author: kieslicaa
"""

import xml.etree.ElementTree as ET
import os
import pandas as pd
import subprocess
import shutil

def format_value_for_r(value):
    """
    Formats a Python value into a string representation that is compatible with R's syntax.

    Parameters:
    value: The value to format.

    Returns:
    str: The R-compatible string representation of the value.
    """
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    elif isinstance(value, dict):
        return f"list({', '.join(f'{k}={format_value_for_r(v)}' for k, v in value.items())})"
    elif isinstance(value, list):
        return f"c({', '.join(format_value_for_r(v) for v in value)})"
    elif isinstance(value, str):
        return f'"{value}"'
    else:
        return repr(value)

def create_familiar_r_file(familiar_r_file_path, **kwargs):
    """
    Creates and executes an R script file for the familiar package with the specified configuration,
    and prints the output of the R script.

    Parameters:
    familiar_r_file_path (str): The path where the R script file will be saved.
    config_file_path (str): The path to the configuration file to be used in the script.

    Returns:
    None
    """
    print("Create familiar R file...")
    
    # Determine the directory of the current Python script
    python_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the relative path to the R library from the Python script's directory
    r_lib_path = os.path.join(python_script_dir, "../../data/familiar/library")
    
    parameters = []
    for key, value in kwargs.items():
        if value is not None:
            formatted_value = format_value_for_r(value)
            parameters.append(f"{key}={formatted_value}")
    
    parameters_str = ",\n".join(parameters)
    summon_familiar_str = f"familiar::summon_familiar(\n{parameters_str}\n)"
    
    # R script content with the actual config file path and library location
    r_script_content = f"""
    print("Starting familiar calculations in R....")
    
    .libPaths(c("{r_lib_path}", .libPaths()))
    
    library(familiar, lib.loc = "{r_lib_path}")
    library(xml2, lib.loc = "{r_lib_path}")
    library(microbenchmark, lib.loc = "{r_lib_path}")
    
    {summon_familiar_str}
    
    print("Familiar calculations in R done!")
    """

    # Writing the R script to the specified file path
    with open(familiar_r_file_path, 'w') as file:
        file.write(r_script_content)
        
    print(f"Familiar R file saved to {familiar_r_file_path}") 

    return


def save_modified_xml(output_path, raw_file_path = r"../../data/familiar/config_raw.xml", 
                      # Paths
                      project_dir=None, experiment_dir=None, data_file=None,
                      # Data
                      experimental_design=None, imbalance_correction_method=None, imbalance_n_partitions=None,
                      batch_id_column=None, sample_id_column=None, series_id_column=None,
                      development_batch_id=None, validation_batch_id=None, outcome_name=None,
                      outcome_column=None, outcome_type=None, class_levels=None, event_indicator=None,
                      censoring_indicator=None, competing_risk_indicator=None, novelty_features=None,
                      exclude_features=None, include_features=None,
                      # Run
                      parallel=None, parallel_nr_cores=None, restart_cluster=None, cluster_type=None,
                      backend_type=None, server_port=None,
                      # Preprocessing
                      feature_max_fraction_missing=None, sample_max_fraction_missing=None, filter_method=None,
                      univariate_test_threshold=None, univariate_test_threshold_metric=None,
                      univariate_test_max_feature_set_size=None, low_var_minimum_variance_threshold=None,
                      low_var_max_feature_set_size=None, robustness_icc_type=None, robustness_threshold_metric=None,
                      robustness_threshold_value=None, transformation_method=None, normalisation_method=None,
                      batch_normalisation_method=None, imputation_method=None, cluster_method=None,
                      cluster_linkage_method=None, cluster_cut_method=None, cluster_similarity_metric=None,
                      cluster_similarity_threshold=None, cluster_representation_method=None,
                      parallel_preprocessing=None,
                      # Feature Selection
                      fs_method=None, fs_method_parameter=None, vimp_aggregation_method=None,
                      vimp_aggregation_rank_threshold=None, parallel_feature_selection=None,
                      # Model Development
                      learner=None, hyperparameter=None, novelty_detector=None, detector_parameters=None,
                      parallel_model_development=None,
                      # Hyperparameter Optimisation
                      optimisation_metric=None, optimisation_function=None, acquisition_function=None,
                      exploration_method=None, hyperparameter_learner=None, optimisation_bootstraps=None,
                      optimisation_determine_vimp=None, smbo_random_initialisation=None, smbo_n_random_sets=None,
                      max_smbo_iterations=None, smbo_step_bootstraps=None, smbo_intensify_steps=None,
                      smbo_stochastic_reject_p_value=None, smbo_stop_convergent_iterations=None,
                      smbo_stop_tolerance=None, parallel_hyperparameter_optimisation=None,
                      # Evaluation
                      evaluate_top_level_only=None, skip_evaluation_elements=None, feature_cluster_method=None,
                      feature_linkage_method=None, feature_cluster_cut_method=None, feature_similarity_metric=None,
                      feature_similarity_threshold=None, sample_cluster_method=None, sample_linkage_method=None,
                      sample_similarity_metric=None, ensemble_method=None, evaluation_metric=None, sample_limit=None,
                      detail_level=None, estimation_type=None, aggregate_results=None, confidence_level=None,
                      bootstrap_ci_method=None, eval_aggregation_method=None, eval_aggregation_rank_threshold=None,
                      eval_icc_type=None, stratification_method=None, stratification_threshold=None, time_max=None,
                      evaluation_times=None, dynamic_model_loading=None, parallel_evaluation=None, sign_size=None):
    """
    Modify and save an XML file based on input parameters.

    Args:
    - file_path (str): Path to the input XML file.
    - output_path (str): Path where the modified XML file will be saved.
    - Other parameters correspond to elements within the XML file.

    The function will modify the XML file based on the non-default input parameters and save it to the specified location.
    """

    # Load the XML file
    tree = ET.parse(raw_file_path)
    root = tree.getroot()

                
    def update_element_text(root, element_name, new_text):
        if new_text is not None:
            # Find the element with the specified name across the entire XML tree
            for element in root.iter(element_name):
                element.text = new_text
                return  # Element found and updated, no need to continue
            
    def update_hyperparameter_element(root, learner, sign_size):
        """
        Update the hyperparameter element with the learner and sign size.
    
        Args:
        - root (ET.Element): The root of the XML tree.
        - learner (str): The name of the learner.
        - sign_size (str): The sign size value.
        """
        
        for hyperparameter_element in root.iter('hyperparameter'):
        
            # Find or create the learner sub-element

            learner_element = ET.SubElement(hyperparameter_element, learner)
    
            # Create or update the sign_size sub-element

            sign_size_element = ET.SubElement(learner_element, 'sign_size')
            sign_size_element.text = sign_size

    if sign_size is not None:
        # Update the hyperparameter element if necessary
        update_hyperparameter_element(root, learner, sign_size)

    # Modify elements based on input parameters
    update_element_text(root, 'project_dir', project_dir)
    update_element_text(root, 'experiment_dir', experiment_dir)
    update_element_text(root, 'data_file', data_file)
    update_element_text(root, 'batch_id_column', batch_id_column)
    update_element_text(root, 'sample_id_column', sample_id_column)
    update_element_text(root, 'series_id_column', series_id_column)
    update_element_text(root, 'experimental_design', experimental_design)
    update_element_text(root, 'imbalance_correction_method', imbalance_correction_method)
    update_element_text(root, 'imbalance_n_partitions', imbalance_n_partitions)
    update_element_text(root, 'development_batch_id', development_batch_id)
    update_element_text(root, 'validation_batch_id', validation_batch_id)
    update_element_text(root, 'outcome_name', outcome_name)
    update_element_text(root, 'outcome_column', outcome_column)
    update_element_text(root, 'outcome_type', outcome_type)
    update_element_text(root, 'class_levels', class_levels)
    update_element_text(root, 'event_indicator', event_indicator)
    update_element_text(root, 'censoring_indicator', censoring_indicator)
    update_element_text(root, 'competing_risk_indicator', competing_risk_indicator)
    update_element_text(root, 'novelty_features', novelty_features)
    update_element_text(root, 'exclude_features', exclude_features)
    update_element_text(root, 'include_features', include_features)
    update_element_text(root, 'parallel', parallel)
    update_element_text(root, 'parallel_nr_cores', parallel_nr_cores)
    update_element_text(root, 'restart_cluster', restart_cluster)
    update_element_text(root, 'cluster_type', cluster_type)
    update_element_text(root, 'backend_type', backend_type)
    update_element_text(root, 'server_port', server_port)
    update_element_text(root, 'feature_max_fraction_missing', feature_max_fraction_missing)
    update_element_text(root, 'sample_max_fraction_missing', sample_max_fraction_missing)
    update_element_text(root, 'filter_method', filter_method)
    update_element_text(root, 'univariate_test_threshold', univariate_test_threshold)
    update_element_text(root, 'univariate_test_threshold_metric', univariate_test_threshold_metric)
    update_element_text(root, 'univariate_test_max_feature_set_size', univariate_test_max_feature_set_size)
    update_element_text(root, 'low_var_minimum_variance_threshold', low_var_minimum_variance_threshold)
    update_element_text(root, 'low_var_max_feature_set_size', low_var_max_feature_set_size)
    update_element_text(root, 'robustness_icc_type', robustness_icc_type)
    update_element_text(root, 'robustness_threshold_metric', robustness_threshold_metric)
    update_element_text(root, 'robustness_threshold_value', robustness_threshold_value)
    update_element_text(root, 'transformation_method', transformation_method)
    update_element_text(root, 'normalisation_method', normalisation_method)
    update_element_text(root, 'batch_normalisation_method', batch_normalisation_method)
    update_element_text(root, 'imputation_method', imputation_method)
    update_element_text(root, 'cluster_method', cluster_method)
    update_element_text(root, 'cluster_linkage_method', cluster_linkage_method)
    update_element_text(root, 'cluster_cut_method', cluster_cut_method)
    update_element_text(root, 'cluster_similarity_metric', cluster_similarity_metric)
    update_element_text(root, 'cluster_similarity_threshold', cluster_similarity_threshold)
    update_element_text(root, 'cluster_representation_method', cluster_representation_method)
    update_element_text(root, 'parallel_preprocessing', parallel_preprocessing)
    update_element_text(root, 'fs_method', fs_method)
    update_element_text(root, 'fs_method_parameter', fs_method_parameter)
    update_element_text(root, 'vimp_aggregation_method', vimp_aggregation_method)
    update_element_text(root, 'vimp_aggregation_rank_threshold', vimp_aggregation_rank_threshold)
    update_element_text(root, 'parallel_feature_selection', parallel_feature_selection)
    update_element_text(root, 'learner', learner)
#    update_element_text(root, 'hyperparameter', hyperparameter)
    update_element_text(root, 'novelty_detector', novelty_detector)
    update_element_text(root, 'detector_parameters', detector_parameters)
    update_element_text(root, 'parallel_model_development', parallel_model_development)
    update_element_text(root, 'optimisation_metric', optimisation_metric)
    update_element_text(root, 'optimisation_function', optimisation_function)
    update_element_text(root, 'acquisition_function', acquisition_function)
    update_element_text(root, 'exploration_method', exploration_method)
    update_element_text(root, 'hyperparameter_learner', hyperparameter_learner)
    update_element_text(root, 'optimisation_bootstraps', optimisation_bootstraps)
    update_element_text(root, 'optimisation_determine_vimp', optimisation_determine_vimp)
    update_element_text(root, 'smbo_random_initialisation', smbo_random_initialisation)
    update_element_text(root, 'smbo_n_random_sets', smbo_n_random_sets)
    update_element_text(root, 'max_smbo_iterations', max_smbo_iterations)
    update_element_text(root, 'smbo_step_bootstraps', smbo_step_bootstraps)
    update_element_text(root, 'smbo_intensify_steps', smbo_intensify_steps)
    update_element_text(root, 'smbo_stochastic_reject_p_value', smbo_stochastic_reject_p_value)
    update_element_text(root, 'smbo_stop_convergent_iterations', smbo_stop_convergent_iterations)
    update_element_text(root, 'smbo_stop_tolerance', smbo_stop_tolerance)
    update_element_text(root, 'parallel_hyperparameter_optimisation', parallel_hyperparameter_optimisation)
    update_element_text(root, 'evaluate_top_level_only', evaluate_top_level_only)
    update_element_text(root, 'skip_evaluation_elements', skip_evaluation_elements)
    update_element_text(root, 'feature_cluster_method', feature_cluster_method)
    update_element_text(root, 'feature_linkage_method', feature_linkage_method)
    update_element_text(root, 'feature_cluster_cut_method', feature_cluster_cut_method)
    update_element_text(root, 'feature_similarity_metric', feature_similarity_metric)
    update_element_text(root, 'feature_similarity_threshold', feature_similarity_threshold)
    update_element_text(root, 'sample_cluster_method', sample_cluster_method)
    update_element_text(root, 'sample_linkage_method', sample_linkage_method)
    update_element_text(root, 'sample_similarity_metric', sample_similarity_metric)
    update_element_text(root, 'ensemble_method', ensemble_method)
    update_element_text(root, 'evaluation_metric', evaluation_metric)
    update_element_text(root, 'sample_limit', sample_limit)
    update_element_text(root, 'detail_level', detail_level)
    update_element_text(root, 'estimation_type', estimation_type)
    update_element_text(root, 'aggregate_results', aggregate_results)
    update_element_text(root, 'confidence_level', confidence_level)
    update_element_text(root, 'bootstrap_ci_method', bootstrap_ci_method)
    update_element_text(root, 'eval_aggregation_method', eval_aggregation_method)
    update_element_text(root, 'eval_aggregation_rank_threshold', eval_aggregation_rank_threshold)
    update_element_text(root, 'eval_icc_type', eval_icc_type)
    update_element_text(root, 'stratification_method', stratification_method)
    update_element_text(root, 'stratification_threshold', stratification_threshold)
    update_element_text(root, 'time_max', time_max)
    update_element_text(root, 'evaluation_times', evaluation_times)
    update_element_text(root, 'dynamic_model_loading', dynamic_model_loading)
    update_element_text(root, 'parallel_evaluation', parallel_evaluation)


    # Save the modified XML to the specified output path
    tree.write(output_path)


def evaluate_familiar_experiment(experiment_dir):
    """
    Retrieves the feature ranking from the familiar feature ranking process.

    This function accesses the generated feature ranking file, reads it, and then exports 
    the relevant data (both RMSE and R² values along with their confidence intervals) to a specified path.

    Parameters:
    - experiment_dir (str): The directory where the experiment was performed.

    Returns:
    A dictionary containing RMSE and R² values along with their confidence intervals for both development and validation datasets.
    """

    # Path to the feature ranking file generated by the familiar process
    familiar_performance_file_path = os.path.join(experiment_dir, "results/pooled_data/performance/performance_metric.csv")

    # Ensure that the feature ranking file exists
    assert os.path.isfile(familiar_performance_file_path), f"Feature ranking file does not exist here {familiar_performance_file_path}!"

    # Read the feature ranking data from the file
    model_performance = pd.read_csv(familiar_performance_file_path, sep=";")

    # Initialize the results dictionary
    results = {
        'development': {'rmse': [], 'r2_score': [], 'rmse_ci_low': [], 'rmse_ci_high': [], 'r2_score_ci_low': [], 'r2_score_ci_high': []},
        'validation': {'rmse': [], 'r2_score': [], 'rmse_ci_low': [], 'rmse_ci_high': [], 'r2_score_ci_low': [], 'r2_score_ci_high': []}
    }

    # Define a helper function to extract metrics
    def extract_metric(metric_name, data_set):
        row = model_performance.loc[(model_performance["metric"] == metric_name) & (model_performance["data_set"] == data_set), :]
        assert row.shape[0] < 2, f"Evaluation row for {metric_name} in model performance file for {data_set} is ambiguous!"
        if row.shape[0] == 1:
            return row["value"].values[0], row["ci_low"].values[0], row["ci_up"].values[0]
        else:
            return None, None, None

    # Extract RMSE and R² for development and validation
    for metric in ['rmse', 'r2_score']:
        for data_set in ['development', 'validation']:
            value, ci_low, ci_high = extract_metric(metric, data_set)
            results[data_set][metric].append(value)
            results[data_set][f'{metric}_ci_low'].append(ci_low)
            results[data_set][f'{metric}_ci_high'].append(ci_high)

    # Path to the feature ranking file generated by the familiar process
    familiar_prediction_file_path = os.path.join(experiment_dir, "results/pooled_data/prediction/prediction.csv")

    # Ensure that the feature ranking file exists
    assert os.path.isfile(familiar_prediction_file_path), f"Feature ranking file does not exist here {familiar_prediction_file_path}!"

    # Read the feature ranking data from the file
    model_prediction = pd.read_csv(familiar_prediction_file_path, sep=";")

    return results, model_prediction

def create_feature_table_for_familiar(data_table, feature_table, features, familiar_feature_table_path):
    """
    Merges specified features from a feature table with data from a data table and saves the resulting table.

    This function is designed to prepare a feature table for use with the FAMILIAR tool. It extracts specified columns 
    from the data table and feature table, merges them on a common identifier, and saves the resulting table to a file.

    Parameters:
    - data_table (DataFrame): A pandas DataFrame containing the base data.
    - feature_table (DataFrame): A pandas DataFrame containing features.
    - features (list): A list of feature names to be included in the final table.
    - familiar_feature_table_path (str): File path where the merged feature table will be saved.

    Returns:
    - ranked_feature_table (DataFrame): The merged DataFrame containing the specified features and data.
    """

    # Informing the user about the operation
    print("Save feature table for familiar...") 
    
    # Extracting the necessary columns from the data table
    # Includes 'cohort', 'id_global', and 'range_shift'
    data_columns = data_table[['cohort', 'id_global', 'range_shift']]

    # Extracting the specified features from the feature table
    # Adds 'id_global' for merging purposes
    feature_columns = feature_table[features + ['id_global']]

    # Merging the data and feature tables on 'id_global'
    # Ensures that each entry in the merged table corresponds correctly
    adjusted_feature_table = pd.merge(data_columns, feature_columns, on='id_global', how='inner')

    # Saving the merged table to the specified file path
    # Uses ';' as a separator and avoids writing the index
    adjusted_feature_table.to_csv(familiar_feature_table_path, sep=';', index=False)

    # Confirming the successful saving of the file
    print(f"Familiar feature table saved to {familiar_feature_table_path}")    

    return adjusted_feature_table

def perform_familiar_experiment(feature_file_path, model_learner, experiment_dir, familiar_r_file_path, signature=None,
                                experimental_design="fs + cv(mb,3,3)", batch_id_column="cohort", sample_id_column="id_global",
                                development_batch_id="training", validation_batch_id="validation", outcome_name="range_shift",
                                outcome_column="range_shift", outcome_type="continuous", parallel_nr_cores=1,
                                parallel=True, feature_max_fraction_missing=0.01, filter_method="none",
                                transformation_method="yeo_johnson", normalisation_method="standardisation",
                                cluster_method="none", parallel_preprocessing="parallel_preprocessing", fs_method="none",
                                vimp_aggregation_method="stability", vimp_aggregation_rank_threshold=5,
                                novelty_detector="none", optimisation_determine_vimp=True,
                                evaluation_metric=["rmse", "r2_score"], imputation_method="simple", smbo_stop_convergent_iterations=None,
                                include_features=None, hyperparameter=None, skip_evaluation_elements=None):
        
    # Determine the directory of the R file
    r_file_dir = os.path.dirname(familiar_r_file_path)

    # Create the directory if it does not exist
    os.makedirs(r_file_dir, exist_ok=True)
        
    # Create the configuration for the FAMILIAR experiment
    create_familiar_r_file(familiar_r_file_path,
                           experiment_dir=experiment_dir, data_file=feature_file_path,
                           experimental_design=experimental_design, batch_id_column=batch_id_column,
                           sample_id_column=sample_id_column, development_batch_id=development_batch_id,
                           validation_batch_id=validation_batch_id, outcome_name=outcome_name,
                           outcome_column=outcome_column, outcome_type=outcome_type,
                           parallel=parallel, parallel_nr_cores=parallel_nr_cores,
                           feature_max_fraction_missing=feature_max_fraction_missing, filter_method=filter_method,
                           transformation_method=transformation_method, normalisation_method=normalisation_method,
                           cluster_method=cluster_method, parallel_preprocessing=parallel_preprocessing,
                           fs_method=fs_method, vimp_aggregation_method=vimp_aggregation_method,
                           vimp_aggregation_rank_threshold=vimp_aggregation_rank_threshold, learner=model_learner,
                           novelty_detector=novelty_detector, optimisation_determine_vimp=optimisation_determine_vimp,
                           evaluation_metric=evaluation_metric, imputation_method=imputation_method, signature=signature,
                           hyperparameter=hyperparameter, skip_evaluation_elements=skip_evaluation_elements, smbo_stop_convergent_iterations=smbo_stop_convergent_iterations,
                           include_features=include_features)

    # Check if the experiment directory exists; if so, remove it
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)

    # Execute the R script and capture the output
    try:
        result = subprocess.run(['Rscript', familiar_r_file_path], check=True, capture_output=True, text=True)
        print("Familiar R file executed successfully!")
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing the R script:", e)
        print("Error Output:\n", e.stderr)

    return

def extract_hyperparameters(experiment_dir):
    """
    Extracts hyperparameters and their values from a file located in a specified directory.

    This function navigates to a directory within the specified experiment directory, where it
    expects to find exactly one file containing hyperparameters. It reads this file and extracts
    the hyperparameters and their corresponding first value, ignoring any subsequent values in
    square brackets.

    Parameters:
    - experiment_dir (str): The path to the directory of the experiment, which contains a
                            subdirectory with the hyperparameter file.

    Returns:
    - dict: A dictionary with hyperparameter names as keys and their corresponding first values.

    Raises:
    - AssertionError: If there is not exactly one hyperparameter file in the specified directory.
    """

    hyperparameter_dir = os.path.join(experiment_dir, "results", "pooled_data", "hyperparameter")
    
    # Ensure there is exactly one hyperparameter file in the directory
    assert len(os.listdir(hyperparameter_dir)) == 1, f"Not just one hyperparameter file in {hyperparameter_dir}!"
    
    # Construct the full path to the hyperparameter file
    file_path = os.path.join(hyperparameter_dir, os.listdir(hyperparameter_dir)[0])
    
    # Dictionary to store the extracted hyperparameters
    hyperparameters = {}

    # Read and process the file to extract hyperparameters
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    for line in file_content:
        parts = line.split('\t')
        if len(parts) == 2:
            name, value = parts
            value = value.split(' ')[0]
            hyperparameters[name] = value.strip()

    return hyperparameters

def merge_data_with_predictions(data_table, predictions):
    """
    Merges the data table with prediction outcomes.
    
    Parameters:
    - data_table (DataFrame): The original data table with features and actual outcomes.
    - predictions (DataFrame): The predictions table containing predicted outcomes.
    
    Returns:
    - DataFrame: A merged table with both actual outcomes and predicted outcomes.
    """
    merged_df = pd.merge(left=data_table, right=predictions, how='left', left_on='id_global', right_on='sample_id')
    final_table = merged_df[['id_global', 'cohort','data_set', 'spot_type', 'proton_energy', 'layer', 'spot_number', 'range_shift', 'predicted_outcome']]
    final_table.rename(columns={'predicted_outcome': 'predicted_range_shift', 'data_set': 'cv_data_set'}, inplace=True)
    return final_table

   
