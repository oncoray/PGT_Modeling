# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:13:23 2023

@author: Aaron Kieslich


This script is designed to create a mapping table for a set of .pkl files 
each containing 2D energy-time spectra of prompt gamma timing data. 
The data is associated with proton beam experiments with varying parameters. 

The script takes as input:
    1. A directory path containing .pkl files.
    2. Information about the organization of the files according to parameters 
    such as:
        - Proton Energy: two different energies
        - Range Shift: four different range shifts
        - Spot Type: two different spot types

The script operates by iterating over the .pkl files and retrieving the 
appropriate parameter information for each file based on the file naming 
convention and measurement ID. Just measurments with ID contained in
a dictionary will be processed.

The output is a CSV file where each row corresponds to a .pkl file.
The columns of the CSV file represent the parameters (Proton Energy, Range 
Shift, Spot Type, Layer, Spot Number), and the final column is the file path 
to the corresponding .pkl file. This table serves as a convenient lookup for 
accessing the data during further analysis.

The resulting table allows the user to filter and select subsets of the data 
based on the parameters, and provides a path to each .pkl file for loading 
and analysis as needed.

The spectra are also plotted and saved.
"""

import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
from pmma.cmd_args import preparation_parser
from pmma.visulisation_methods import save_2Dspectrum, save_timesumspectrum, save_timespectrum,save_2Dsumspectrum, save_energysumspectrum

def create_data_table(root_dir, output_file, measurement_dict, cohort_mapping_dict = None, spots_to_exclude_static = 0, \
         spots_to_include_scanned = []):
    """
    Main function to create a data map table.
    File names should be in the following structure:
        x_xxxxxxxx_1607161004xxx_layer01_spot031.pkl
    where the underscore need to be set and the measurement-ID, layer number
    and spot number can vary.
    
    Args:
        root_dir (str): Root directory to search for files.
        output_file (str): Output file path for the data map table.
        measurement_dict (dict): Dictionary mapping measurement IDs to 
        proton energy and range shift values.
        cohort_mapping_dict (list): List of dictionaries describing the training and validation cohort of the experiment.
    """
    
    assert isinstance(root_dir, str), "root_dir must be a string"
    assert isinstance(output_file, str), "output_file must be a string"
    assert isinstance(measurement_dict, dict),  \
        "measurement_dict must be a dictionary"
    
    data_map = []
    print("Start creating table with file paths...") 
    # Look at all raw data
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                file_path = os.path.join(dirpath, filename)
                measurementID = int(filename.split('_')[2][:-3])
                if measurementID not in list(measurement_dict.keys()):
                    continue
                layer = int(filename.split('layer')[1][:2])
                
                range_shift, proton_energy, spot_type = measurement_dict\
                    [measurementID]

                spot_number = int(filename.split('spot')[1][:3])
                
                if cohort_mapping_dict is not None:
                    cohort_present = any(
                        (entry['energy'] == proton_energy and entry['type'] == spot_type)
                        for entry in cohort_mapping_dict)
                    
                    if not cohort_present:
                        continue
                
                # exclude specific spots, as they have errors of don't have
                # range shifts due to scanning
                if spot_type == 'static' and spot_number <= spots_to_exclude_static:
                    continue
                if 'scanned' in spot_type and spot_number not in \
                    spots_to_include_scanned:
                    continue
                
                # Check if the combination already exists in data_map
                if any(entry[1:-1] == [proton_energy, range_shift, spot_type, \
                                       layer, spot_number] for entry in data_map):
                    continue
                
                data_map.append([measurementID, proton_energy, range_shift, \
                                 spot_type, layer, spot_number, file_path])

    
    # TODO: Add the different file paths of different stages of the data
    # Convert list of data into DataFrame
    data_map_df = pd.DataFrame(data_map, columns=['id_measurement', \
                                                  'proton_energy', 'range_shift',\
                                                  'spot_type', 'layer', \
                                                  'spot_number', 'file_path'])
    
    if cohort_mapping_dict is not None:
        
        # Split the data into training and validation sets for each energy and spot type
        for cohort in cohort_mapping_dict:
            energy = cohort['energy']
            spot_type = cohort['type']
            subset = data_map_df[(data_map_df['proton_energy'] == energy) & (data_map_df['spot_type'] == spot_type)]
            
            assert not subset.empty, f"No samples found for energy {energy} and spot type {spot_type}. Redefine cohorts!"
    
            train_size = cohort['values']['training']
    
            if train_size == 1:
                # All data goes to training
                train_subset = subset
                valid_subset = pd.DataFrame(columns=subset.columns)
            elif train_size == 0:
                # All data goes to validation
                train_subset = pd.DataFrame(columns=subset.columns)
                valid_subset = subset
            elif spot_type == 'static':
                # Sort by spot_number for static spot type
                subset = subset.sort_values(by='spot_number')
                split_index = int(len(subset) * train_size)
                train_subset = subset.iloc[:split_index]
                valid_subset = subset.iloc[split_index:]
            else:
                # Random train-test split for other spot types
                train_subset, valid_subset = train_test_split(subset, train_size=train_size, random_state=0, stratify=subset["range_shift"].values)
        
            data_map_df.loc[train_subset.index, 'cohort'] = 'training'
            data_map_df.loc[valid_subset.index, 'cohort'] = 'validation'
            
        # Check if both 'training' and 'validation' cohorts are present in the table
        assert 'training' in data_map_df['cohort'].values, "Training cohort is missing from the table."
        assert 'validation' in data_map_df['cohort'].values, "Validation cohort is missing from the table."
        
    # Generate unique IDs
    data_map_df['id_global'] = np.arange(1,len(data_map_df)+1)
    
    # Reorder the columns to make 'Global_ID' the first column
    cols = ['id_global'] + [col for col in data_map_df if col != 'id_global']
    data_map_df = data_map_df[cols]

    # Save DataFrame as CSV
    data_map_df.to_csv(output_file, index=False, sep = ";")
    print("Table with data paths saved!")



if __name__ == "__main__":
    parser = preparation_parser("Create data paths table")
    args = parser.parse_args()
    
    # Convert the JSON string back into a dictionarys
    measurement_dict = {}
    for key, value in json.loads(args.measurement_dict).items():
        measurement_dict[int(key)] = value
        
    if args.cohort_mapping_dict is not None:
        cohort_mapping_dict = json.loads(args.cohort_mapping_dict)   
    else:
        cohort_mapping_dict = None

    create_data_table(args.root_dir, args.output_file, measurement_dict, cohort_mapping_dict, args.spots_to_exclude_static,\
         args.spots_to_include_scanned)
    
