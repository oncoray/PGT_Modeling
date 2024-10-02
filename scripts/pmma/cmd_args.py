# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:04:04 2023

@author: kieslicaa
"""

from argparse import ArgumentParser

def add_dataset_args(parser):
   group = parser.add_argument_group("Dataset")
   group.add_argument(
       '--energy_bin',
       type=float, 
       default = None,
       help='Gamma energy bin size in MeV.'
   )
   
   group.add_argument(
       '--time_bin',
       type=float, 
       default = None,
       help='Finetime bin size in ns.'
   )
   
   group.add_argument(
       '--spots_to_exclude_static',
       type=int, 
       default = 0,
       help='Number of static spots which should be excluded. Will be the first x spots.'
   )
   
   group.add_argument(
       '--spots_to_include_scanned',
       nargs="*",
       type = int,
       default = [],
       help='List of scanned spot numbers which should be analysed-'
   )
   
   group.add_argument(
       '--root_dir',
       type=str, 
       default = None,
       help='Root directory for the data'
   )
   
   group.add_argument(
       '--measurement_dict',
       type=str,
       help='Measurement ID dictionary as a JSON string',
       default = None
   )
   
   group.add_argument(
       '--cohort_mapping_dict',
       type = str,
       default = None,
       help='Dict defining the analysed cohorts in JSON-format.'
   )

   
   return parser

def add_plotting_args(parser):
    group = parser.add_argument_group("Data plotting")
    group.add_argument(
        '--figures_path',
        type=str,
        help = 'Path to save figures to.',
        default = ""
        )
    
    group.add_argument(
        '--plot_figures', 
        type=bool, 
        help="Flag for figure plotting and saving.", 
        default = False)
    
    return parser

def add_preparation_args(parser):
    group = parser.add_argument_group("Data preparation")
    
    group.add_argument(
        '--output_file',
        type=str,
        help='Path to output CSV file'
    )
    
    return parser


def add_processing_args(parser):
    group = parser.add_argument_group("Data processing")
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file'
    )
    
    group.add_argument(
        '--preprocessed_dir',
        type=str,
        help = 'Path to directory of preprocessed data',
        default = None,
        required = True
    )
    
    group.add_argument(
        '--spectrum_type',
        type=str,
        help = 'Type of the spectrum which should be the output of the processing',
        default = None
    )
    
    group.add_argument(
        "--rois",
        type=str,
        nargs="*",
        choices=["full",
                 "10B",
                 "16O",
                 "11C",
                 "511keV",
                 "12C11B"
                 ],
        default=[],
        help="For which ROIs (energy lines) should 1D spectrum be processed?"
    )

    group.add_argument(
        '--completion_file_path',
        type=str,
        help = 'Path to completion file',
        default = None
        )

    group.add_argument(
        '--metadata_file_path',
        type=str,
        help = 'Path to save metadata to',
        default = None
        )
    
    group.add_argument(
        '--metadata_bg_file_path',
        type=str,
        help = 'Path to save bg metadata to',
        default = None
        )
    
    group.add_argument(
        '--data_preprocessing_args',
        type = str,
        default = None,
        help='Dict defining the arguments and parameters for the preprocessing routine.'
    )

    return parser



def add_feature_calculation_args(parser):
    group = parser.add_argument_group("Feature calculation")
    
    group.add_argument(
        '--data_dir',
        type = str,
        default = None,
        help='Path to data dir.'
    )
    group.add_argument(
        '--output_file_path',
        type = str,
        default = None,
        help='Path to save table with features.'
    )
    
    group.add_argument(
        '--feature_type',
        type=str,
        choices = ["energy", "time", "energy-time", "radiomics"],
        help = "Feature set which should be calculated."
        )
    
    group.add_argument(
        '--mirp_path',
        type = str,
        default = None,
        help='Path where config files and temp data directory for mirp are located.'
    )
    
    group.add_argument(
        "--rois",
        type=str,
        nargs="*",
        choices=["full",
                 "10B",
                 "16O",
                 "11C",
                 "511keV",
                 "12C11B"
                 ],
        default=[],
        help="For which ROIs (energy lines) should features be extracted?"
    )
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file'
    )
    
    return parser    

def add_feature_normalization_args(parser):
    # Group for feature normalization arguments
    group = parser.add_argument_group('Feature Normalization Arguments')

    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file'
    )

    # Add arguments to the group
    group.add_argument(
        '--feature_file_path',
        type=str,
        required=True,
        help='Path to the CSV file containing the original features.'
    )

    group.add_argument(
        '--normalized_feature_file_path',
        type=str,
        required=True,
        help='Path for the output CSV file to store normalized features.'
    )

    return parser

def add_feature_filtering_args(parser):
    # Group for feature normalization arguments
    group = parser.add_argument_group('Feature Filtering Arguments')

    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file'
    )

    # Add arguments to the group
    group.add_argument(
        '--feature_file_path',
        type=str,
        required=True,
        help='Path to the CSV file containing the original features.'
    )

    group.add_argument(
        '--filtered_feature_file_path',
        type=str,
        required=True,
        help='Path for the output CSV file to store filtered features.'
    )

    group.add_argument(
        '--filtering_args',
        type = str,
        default = None,
        help='Argments for the feature filtering procedure in JSON-format.'
    )
    
    group.add_argument(
        '--metadata_file_path',
        type=str,
        help = 'Path to save metadata to',
        default = None
        )
    
    group.add_argument(
        '--filter_info_file_path',
        type=str,
        required=True,
        help ='Specify the path of the file with the feature filtering info stored')
    

    return parser

def add_feature_clustering_args(parser):
    group = parser.add_argument_group('feature selection')
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file'
    )
    
    group.add_argument(
        '--feature_file_paths',
        type=str,
        nargs="+",
        default = [],
        help='List with paths to csv-files containing feature information'
        )
    
    group.add_argument(
        '--cluster_info_file_path',
        type=str,
        required=True,
        help ='Specify the path of the file with the feature clusters info stored')
    
    group.add_argument(
        '--metadata_file_path',
        type=str,
        help = 'Path to save feature clustering metadata to',
        default = None
        )
    
    group.add_argument(
        '--clustering_args',
        type = str,
        default = None,
        help='Argments for the clustering procedure in JSON-format.'
    )
    
    return parser

def add_feature_ranking_args(parser):
    group = parser.add_argument_group('feature ranking')
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file',
        required=True
    )

    group.add_argument(
        '--feature_clustering_metadata_file_path',
        type=str,
        help = 'Path to save feature selection metadata to',
        required=True
        )

    group.add_argument(
        '--feature_ranking_file_path',
        type=str,
        help = 'Path to save feature ranking data to',
        required=True
        )
    
    group.add_argument(
        '--feature_file_path',
        type=str,
        help='Path to csv-files containing feature information',
        required=True
        )
    
    group.add_argument(
        '--feature_selection_method', 
        type=str, 
        help='Feature selection methods to be applied.',
        required=True
        )
    
    group.add_argument(
        '--feature_ranking_path',
        type=str,
        help="Path to the main feature selection dir",
        required=True
        )
    
    group.add_argument(
        '--feature_type',
        type=str,
        help="Feature type to be analyzed",
        required=True
        )
    
    group.add_argument(
        '--feature_importance_plot_path',
        type=str,
        help="Path to save feature ranking plot.",
        required=True)
    
    return parser

def add_feature_selection_args(parser):
    group = parser.add_argument_group('feature selection')
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file',
        required=True
    )

    group.add_argument(
        '--feature_selection_file_path',
        type=str,
        help = 'Path to save feature selection data to',
        required=True
        )
    
    group.add_argument(
        '--feature_selection_method', 
        type=str, 
        help='Feature selection methods to be applied.',
        required=True
        )
    
    group.add_argument(
        '--feature_file_path',
        type=str,
        help='Path to csv-files containing feature information',
        required=True
        )
    
    group.add_argument(
        '--feature_ranking_file_path',
        type=str,
        help = 'Path to the feature ranking data',
        required=True
        )
    
    group.add_argument(
        '--model_learner', 
        type=str, 
        help='Feature selection methods to be applied.',
        required=True
        )
    
    group.add_argument(
        '--feature_selection_path',
        type=str,
        help="Path to the main feature selection dir",
        required=True
        )
    
    group.add_argument(
        '--vif_bar_plot_path',
        type=str,
        help="Path to the VIF bar plot of the signature",
        required=True
        )
    
    group.add_argument(
        '--signature_plot_path',
        type=str,
        help="Path to save the signature plot to",
        required=True
        )
    
    group.add_argument(
        '--feature_type',
        type=str,
        help="Feature type to be analyzed",
        required=True
        )

    group.add_argument(
        '--feature_selection_plot_path',
        type=str,
        help="Path to save feature ranking plot.",
        required=True)
    
    group.add_argument(
        '--selection_args',
        type = str,
        default = None,
        help='Argments for the selection procedure in JSON-format.'
    )
    
    group.add_argument(
        '--predictions_file_path',
        type=str,
        help="Path to csv-file to save predictions to",
        required=True)
    
    group.add_argument(
        '--signature_file_path',
        type=str,
        help="Path to json string containing the signature of the model",
        required = True)
    
    return parser

def add_external_validation_args(parser):
    group = parser.add_argument_group('external_validation')
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file',
        required=True
    )

    group.add_argument(
        '--feature_selection_file_path',
        type=str,
        help = 'Path to save feature selection data to',
        required=True
        )
    
    group.add_argument(
        '--feature_file_path',
        type=str,
        help='Path to csv-files containing feature information',
        required=True
        )
    
    group.add_argument(
        '--predictions_file_path',
        type=str,
        help="Path to csv-file to save predictions to",
        required=True)
    
    group.add_argument(
        '--performance_file_path',
        type=str,
        help="Path to json-file to save model performances to",
        required=True)
    
    group.add_argument(
        '--external_validation_path',
        type=str,
        help = 'Path to the main external validation dir',
        required = True
        )
    
    group.add_argument(
        '--prediction_plot_path',
        type=str,
        help='Path to the plot predcited vs actual range shift',
        required = True
        )
    
    return parser
    
def add_feature_combination_args(parser):
    group = parser.add_argument_group('feature combination')
    
    group.add_argument(
        '--feature_file_paths',
        type=str,
        nargs="+",
        default = [],
        help='List with paths to csv-files containing feature information'
        )
    
    group.add_argument(
        '--combined_feature_file_path',
        type=str,
        help='Paths to csv-files containing combined feature information'
        )
    
    return parser

def add_feature_selection_summary_args(parser):
    group = parser.add_argument_group('Feature Selection Summary Arguments')

    # Argument for the holding instance (e.g., specific feature type, selection method, or model learner)
    group.add_argument(
        '--holding_instance',
        type=str,
        required=True,
        help='The specific instance (feature type, feature selection method, or model learner) that is held constant for the summary.'
    )

    # Argument for the heatmap plot path
    group.add_argument(
        '--heatmap_plot_path',
        type=str,
        required=True,
        help='Path where the heatmap plot will be saved.'
    )

    # Argument for the paths of feature selection result files
    group.add_argument(
        '--feature_selection_file_paths',
        type=str,
        nargs='+',
        required=True,
        help='List of file paths to the feature selection result JSON files.'
    )

    return parser

def add_cML_summary_args(parser):
    group = parser.add_argument_group('cML summary')
    
    group.add_argument(
        '--data_table_path',
        type=str,
        help='Path to data table CSV file',
        required=True
    )
    
    group.add_argument(
        '--individual_prediction_files',
        type=str,
        nargs='+',  # Allows multiple values
        help="Paths to file(s) containing individual predictions",
        required=True)

    group.add_argument(
        '--summary_performance_file_path',
        type=str,
        help="Path to csv-file to save summary of performance to",
        required=True)
    
    group.add_argument(
        '--summary_predictions_file_path',
        type=str,
        help="Path to csv-file to save summary of predictions to",
        required=True)
    
    group.add_argument(
        '--heatmap_plot_dir_path',
        type=str, 
        help="Path to the directory containing the heatmap plots.",
        required = True
        )
    
    group.add_argument(
        '--signature_file_paths',
        type=str,
        nargs='+',  # Allows multiple values
        help="Paths to file(s) containing signatures",
        required=True)

    return parser

def add_cML_final_result_args(parser):
    group = parser.add_argument_group('cML final result')
    
    group.add_argument(
        '--ex_val_performance_file_path',
        type=str,
        help="Path to csv-file with predictions of external validation to",
        required=True)
    
    group.add_argument(
        '--cross_val_performance_file_path',
        type=str,
        help="Path to csv-file with predictions of internal cross validation",
        required=True)
    
    # Arguments for specifying the output paths for the final result in CSV, TXT, and PNG formats
    group.add_argument(
        '--final_result_csv',
        type=str,
        help='Output path for the final result CSV file',
        required=True
    )
    
    group.add_argument(
        '--final_result_txt',
        type=str,
        help='Output path for the final result text file',
        required=True
    )
    
    group.add_argument(
        '--final_result_png',
        type=str,
        help='Output path for the final result PNG image',
        required=True
    )
    
    return parser


def add_cML_args(parser):
    group = parser.add_argument_group('conventional machine learning')

    # Add argument for the directory path to familiar experiments
    group.add_argument(
        '--conventional_machine_learning_path', 
        type=str, 
        required=True, 
        help='Specify the directory path where conventional_machine_learning experiments are stored.'
    )

    # Add argument for the feature selection methods
    group.add_argument(
        '--feature_selection_methods', 
        type=str, 
        nargs='+',
        help='Define one or more feature selection methods to be applied.'
    )

    # Add argument for the model learners
    group.add_argument(
        '--model_learners', 
        type=str, 
        nargs='*',
        default = [], 
        help='List the machine learning algorithms that will be used as learners.'
    )

    group.add_argument(
        '--analyze_feature_combination',
        type=bool, 
        help="Flag for analyzing all features combined.", 
        default = False)
    
    group.add_argument(
        '--analyze_individual_features',
        type=bool, 
        help="Flag for analyzing individual features types seperatly.", 
        default = False)
    

    return parser




def preparation_parser(title):
    parser = ArgumentParser(title)
    parser = add_dataset_args(parser)
    parser = add_preparation_args(parser)
    parser = add_plotting_args(parser)
    return parser

def feature_calculation_parser(title):
    parser = ArgumentParser(title)
    parser = add_dataset_args(parser)
    parser = add_feature_calculation_args(parser)
    parser = add_plotting_args(parser)
    return parser

def dataset_parser(title):
    parser = ArgumentParser(title)
    parser = add_dataset_args(parser)
    return parser

def processing_parser(title):
    parser = ArgumentParser(title)
    parser = add_processing_args(parser)
    parser = add_dataset_args(parser)
    parser = add_plotting_args(parser)
    return parser

def feature_normalization_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_normalization_args(parser)
    return parser

def feature_filtering_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_filtering_args(parser)
    return parser

def feature_clustering_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_clustering_args(parser)
    return parser

def feature_ranking_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_ranking_args(parser)
    return parser

def feature_selection_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_selection_args(parser)
    return parser

def feature_selection_summary_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_selection_summary_args(parser)
    return parser

def feature_combination_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_combination_args(parser)
    return parser

def external_validation_parser(title):
    parser = ArgumentParser(title)
    parser = add_external_validation_args(parser)
    return parser

def cML_summary_parser(title):
    parser = ArgumentParser(title)
    parser = add_cML_summary_args(parser)
    return parser

def cML_final_result_parser(title):
    parser = ArgumentParser(title)
    parser = add_cML_final_result_args(parser)
    return parser


def cML_parser(title):
    parser = ArgumentParser(title)
    parser = add_feature_calculation_args(parser)
    parser = add_cML_args(parser)
    return parser
    



