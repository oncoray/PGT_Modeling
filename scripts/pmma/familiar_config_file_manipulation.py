# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:26:30 2023

@author: kieslicaa
"""

import xml.etree.ElementTree as ET
import os

def create_familiar_r_file(familiar_r_file_path, config_file_path):
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
    
    # R script content with the actual config file path and library location
    r_script_content = f"""
    print("Starting familiar calculations in R....")
    
    library(familiar, lib.loc = "{r_lib_path}")
    
    config_file_path <- "{config_file_path}"
    
    familiar::summon_familiar(config = config_file_path, parallel = TRUE)
    
    print("Familiar calculations in R done!")
    """

    # Writing the R script to the specified file path
    with open(familiar_r_file_path, 'w') as file:
        file.write(r_script_content)
        
    print(f"Familiar R file saved to {familiar_r_file_pathguit}") 

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
                      evaluation_times=None, dynamic_model_loading=None, parallel_evaluation=None):
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

    # A helper function to update text of a given element
    def update_element_text(element_path, new_text):
        if new_text is not None:
            element = root.find(element_path)
            if element is not None:
                element.text = new_text

    # Modify elements based on input parameters
    update_element_text('paths/project_dir', project_dir)
    update_element_text('paths/experiment_dir', experiment_dir)
    update_element_text('paths/data_file', data_file)
    update_element_text('data/batch_id_column', batch_id_column)
    update_element_text('data/sample_id_column', sample_id_column)
    update_element_text('data/series_id_column', series_id_column)
    update_element_text('data/experimental_design', experimental_design)
    update_element_text('data/imbalance_correction_method', imbalance_correction_method)
    update_element_text('data/imbalance_n_partitions', imbalance_n_partitions)
    update_element_text('data/development_batch_id', development_batch_id)
    update_element_text('data/validation_batch_id', validation_batch_id)
    update_element_text('data/outcome_name', outcome_name)
    update_element_text('data/outcome_column', outcome_column)
    update_element_text('data/outcome_type', outcome_type)
    update_element_text('data/class_levels', class_levels)
    update_element_text('data/event_indicator', event_indicator)
    update_element_text('data/censoring_indicator', censoring_indicator)
    update_element_text('data/competing_risk_indicator', competing_risk_indicator)
    update_element_text('data/novelty_features', novelty_features)
    update_element_text('data/exclude_features', exclude_features)
    update_element_text('data/include_features', include_features)
    update_element_text('run/parallel', parallel)
    update_element_text('run/parallel_nr_cores', parallel_nr_cores)
    update_element_text('run/restart_cluster', restart_cluster)
    update_element_text('run/cluster_type', cluster_type)
    update_element_text('run/backend_type', backend_type)
    update_element_text('run/server_port', server_port)
    update_element_text('preprocessing/feature_max_fraction_missing', feature_max_fraction_missing)
    update_element_text('preprocessing/sample_max_fraction_missing', sample_max_fraction_missing)
    update_element_text('preprocessing/filter_method', filter_method)
    update_element_text('preprocessing/univariate_test_threshold', univariate_test_threshold)
    update_element_text('preprocessing/univariate_test_threshold_metric', univariate_test_threshold_metric)
    update_element_text('preprocessing/univariate_test_max_feature_set_size', univariate_test_max_feature_set_size)
    update_element_text('preprocessing/low_var_minimum_variance_threshold', low_var_minimum_variance_threshold)
    update_element_text('preprocessing/low_var_max_feature_set_size', low_var_max_feature_set_size)
    update_element_text('preprocessing/robustness_icc_type', robustness_icc_type)
    update_element_text('preprocessing/robustness_threshold_metric', robustness_threshold_metric)
    update_element_text('preprocessing/robustness_threshold_value', robustness_threshold_value)
    update_element_text('preprocessing/transformation_method', transformation_method)
    update_element_text('preprocessing/normalisation_method', normalisation_method)
    update_element_text('preprocessing/batch_normalisation_method', batch_normalisation_method)
    update_element_text('preprocessing/imputation_method', imputation_method)
    update_element_text('preprocessing/cluster_method', cluster_method)
    update_element_text('preprocessing/cluster_linkage_method', cluster_linkage_method)
    update_element_text('preprocessing/cluster_cut_method', cluster_cut_method)
    update_element_text('preprocessing/cluster_similarity_metric', cluster_similarity_metric)
    update_element_text('preprocessing/cluster_similarity_threshold', cluster_similarity_threshold)
    update_element_text('preprocessing/cluster_representation_method', cluster_representation_method)
    update_element_text('preprocessing/parallel_preprocessing', parallel_preprocessing)
    update_element_text('feature_selection/fs_method', fs_method)
    update_element_text('feature_selection/fs_method_parameter', fs_method_parameter)
    update_element_text('feature_selection/vimp_aggregation_method', vimp_aggregation_method)
    update_element_text('feature_selection/vimp_aggregation_rank_threshold', vimp_aggregation_rank_threshold)
    update_element_text('feature_selection/parallel_feature_selection', parallel_feature_selection)
    update_element_text('model_development/learner', learner)
    update_element_text('model_development/hyperparameter', hyperparameter)
    update_element_text('model_development/novelty_detector', novelty_detector)
    update_element_text('model_development/detector_parameters', detector_parameters)
    update_element_text('model_development/parallel_model_development', parallel_model_development)
    update_element_text('hyperparameter_optimisation/optimisation_metric', optimisation_metric)
    update_element_text('hyperparameter_optimisation/optimisation_function', optimisation_function)
    update_element_text('hyperparameter_optimisation/acquisition_function', acquisition_function)
    update_element_text('hyperparameter_optimisation/exploration_method', exploration_method)
    update_element_text('hyperparameter_optimisation/hyperparameter_learner', hyperparameter_learner)
    update_element_text('hyperparameter_optimisation/optimisation_bootstraps', optimisation_bootstraps)
    update_element_text('hyperparameter_optimisation/optimisation_determine_vimp', optimisation_determine_vimp)
    update_element_text('hyperparameter_optimisation/smbo_random_initialisation', smbo_random_initialisation)
    update_element_text('hyperparameter_optimisation/smbo_n_random_sets', smbo_n_random_sets)
    update_element_text('hyperparameter_optimisation/max_smbo_iterations', max_smbo_iterations)
    update_element_text('hyperparameter_optimisation/smbo_step_bootstraps', smbo_step_bootstraps)
    update_element_text('hyperparameter_optimisation/smbo_intensify_steps', smbo_intensify_steps)
    update_element_text('hyperparameter_optimisation/smbo_stochastic_reject_p_value', smbo_stochastic_reject_p_value)
    update_element_text('hyperparameter_optimisation/smbo_stop_convergent_iterations', smbo_stop_convergent_iterations)
    update_element_text('hyperparameter_optimisation/smbo_stop_tolerance', smbo_stop_tolerance)
    update_element_text('hyperparameter_optimisation/parallel_hyperparameter_optimisation', parallel_hyperparameter_optimisation)
    update_element_text('evaluation/evaluate_top_level_only', evaluate_top_level_only)
    update_element_text('evaluation/skip_evaluation_elements', skip_evaluation_elements)
    update_element_text('evaluation/feature_cluster_method', feature_cluster_method)
    update_element_text('evaluation/feature_linkage_method', feature_linkage_method)
    update_element_text('evaluation/feature_cluster_cut_method', feature_cluster_cut_method)
    update_element_text('evaluation/feature_similarity_metric', feature_similarity_metric)
    update_element_text('evaluation/feature_similarity_threshold', feature_similarity_threshold)
    update_element_text('evaluation/sample_cluster_method', sample_cluster_method)
    update_element_text('evaluation/sample_linkage_method', sample_linkage_method)
    update_element_text('evaluation/sample_similarity_metric', sample_similarity_metric)
    update_element_text('evaluation/ensemble_method', ensemble_method)
    update_element_text('evaluation/evaluation_metric', evaluation_metric)
    update_element_text('evaluation/sample_limit', sample_limit)
    update_element_text('evaluation/detail_level', detail_level)
    update_element_text('evaluation/estimation_type', estimation_type)
    update_element_text('evaluation/aggregate_results', aggregate_results)
    update_element_text('evaluation/confidence_level', confidence_level)
    update_element_text('evaluation/bootstrap_ci_method', bootstrap_ci_method)
    update_element_text('evaluation/eval_aggregation_method', eval_aggregation_method)
    update_element_text('evaluation/eval_aggregation_rank_threshold', eval_aggregation_rank_threshold)
    update_element_text('evaluation/eval_icc_type', eval_icc_type)
    update_element_text('evaluation/stratification_method', stratification_method)
    update_element_text('evaluation/stratification_threshold', stratification_threshold)
    update_element_text('evaluation/time_max', time_max)
    update_element_text('evaluation/evaluation_times', evaluation_times)
    update_element_text('evaluation/dynamic_model_loading', dynamic_model_loading)
    update_element_text('evaluation/parallel_evaluation', parallel_evaluation)


    # Save the modified XML to the specified output path
    tree.write(output_path, xml_declaration=True, encoding='utf-8')

# The function still needs to be extended with similar lines for each parameter to modify the respective XML elements.
# The paths used in update_element_text should match the structure of the XML file.
# Additional logic may be required for more complex modifications, like adding or removing elements.

