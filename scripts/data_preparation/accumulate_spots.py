# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:00:53 2024

@author: Phase
"""

import os
import pandas as pd
import numpy as np
import json
from pmma.config_file_manipulator import yaml_to_json_string, substitute_labels
from time import strftime, localtime
import pickle
from tqdm import tqdm



def accumulate_and_save(data_map, spot_type, spots_to_accumulate, output_dir):
    """
    Accumulates spectral data for specified spot types over a defined number of spots and saves the accumulated data into a new directory.

    This function processes spectral data based on a given spot type ('scanned' or 'static') and accumulates data across a specified number of spots. It then saves the new, accumulated data and a metadata JSON in a newly created directory specific to the accumulation configuration.

    Parameters:
    - data_map (pd.DataFrame): A DataFrame containing data mapping information. Expected to have columns for 'spot_type', 'id_measurement', 'proton_energy', 'range_shift', 'spot_number', 'layer', and 'file_path'.
    - spot_type (str): The type of spot to be processed (e.g., 'scanned', 'static'). Only data matching this spot type will be accumulated.
    - spots_to_accumulate (int): The number of spots to be accumulated into a single spectrum.
    - output_dir (str): The base directory path where the new accumulated data and metadata should be saved.

    Returns:
    - None: The function does not return a value but instead writes output to files.
    """

    # Initialize an empty list to hold results metadata
    result_json = []
    
    # Generate new spot type name and output directory name based on accumulation criteria
    new_spot_type_name = spot_type + f"_accum{spots_to_accumulate}"
    new_output_dir = output_dir + f"_accum{spots_to_accumulate}"
    
    # Check if the new spot type name already exists in the data map or if the output directory already exists
    if new_spot_type_name in data_map["spot_type"].values:
        print(f"{new_spot_type_name} already in data map! Either there is a mistake or the data has already been accumulated")
        return
    elif new_output_dir in os.listdir(os.path.dirname(output_dir)):
        print(f"{new_output_dir} already exists but {new_spot_type_name} not in data map! Check measurement ID in config.yaml file!")
        return
    
    # Create the new output directory
    os.makedirs(new_output_dir, exist_ok=True)
    
    # Process each measurementID in the data map
    for mID, data_map_mID in tqdm(data_map.groupby('id_measurement'), desc = "Analyse specific mID"):
        # Skip processing if the current measurementID's spot type doesn't match the target spot type
        if data_map_mID['spot_type'].iloc[0] != spot_type:
            continue
        
        # Generate a unique new measurement ID
        new_id = strftime("%d%m%y%d%M", localtime())
        already_used_ids = [i['id_measurement'] for i in result_json]
        while new_id in already_used_ids:
            new_id = str(int(new_id) + 1)

        # Extract the proton energy and range shift for the current measurementID
        proton_energy = data_map_mID['proton_energy'].iloc[0]
        range_shift = data_map_mID['range_shift'].iloc[0]
        
        # Append the new measurement data to the results metadata list
        result_json.append({
            'id_measurement': new_id, 
            'spot_type': spot_type, 
            'proton_energy': proton_energy, 
            'range_shift': range_shift, 
            'accumulations': spots_to_accumulate, 
            'new_spot_type': new_spot_type_name
        })
        

        
        # Process 'scanned' spot type
        if spot_type == "scanned":
            for spot_number in data_map_mID["spot_number"].unique():
                
                accumulated_spectrum = None  # Initialize variable to hold the accumulated spectrum
                accumulation_counter = 0  # Counter for the number of spots accumulated
                new_layer = 1  # Initialize the new layer counter
                
                # Accumulate spectra for each layer within the current spot number
                for layer in sorted(data_map_mID.loc[data_map_mID["spot_number"] == spot_number, "layer"].unique()):
                    
                    file_path = data_map_mID.loc[(data_map_mID["spot_number"] == spot_number) & (data_map_mID["layer"] == layer), "file_path"].values[0]
                    spectrum = pd.read_pickle(file_path)[2]  # Load the spectrum data
                    if accumulated_spectrum is None:
                        accumulated_spectrum = spectrum
                    else:
                        accumulated_spectrum += spectrum
                    accumulation_counter += 1
                    
                    # Save the accumulated spectrum after reaching the target number of spots
                    if accumulation_counter == spots_to_accumulate:
                        
                        save_path = os.path.join(new_output_dir, f"8_SpotPGT2D_{new_id}-17_layer{str(new_layer).zfill(2)}_spot{str(spot_number).zfill(3)}_accumulated_{spots_to_accumulate}.pkl")
                        # Assume the first two elements are consistent across files for a given key
                        metadata = pd.read_pickle(file_path)[:2] + [accumulated_spectrum]
                        with open(save_path, 'wb') as file:
                            pickle.dump(metadata, file)
                        
                        # Prepare for the next accumulation set
                        new_layer = int(new_layer) + spots_to_accumulate
                        accumulated_spectrum = None
                        accumulation_counter = 0
                        
        # Process 'static' spot type
        if spot_type == "static":
            accumulated_spectrum = None  # Initialize variable to hold the accumulated spectrum
            accumulation_counter = 0  # Counter for the number of spots accumulated
            new_layer = "01"  # Initialize the new layer counter
            new_spot_number = np.min(data_map_mID["spot_number"])  # Start with the minimum spot number
        
            # Iterate through each unique spot number to accumulate spectra
            for spot_number in sorted(data_map_mID["spot_number"].unique()):
                
                # Locate the file path for the current spot number and layer (assuming layer=1 for static spots)
                file_path = data_map_mID.loc[(data_map_mID["spot_number"] == spot_number) & (data_map_mID["layer"] == 1), "file_path"].values[0]
                spectrum = pd.read_pickle(file_path)[2]  # Load the spectrum data

                if accumulated_spectrum is None:
                    accumulated_spectrum = spectrum  # Initialize the accumulated spectrum with the first spectrum
                else:
                    accumulated_spectrum += spectrum  # Accumulate the spectrum
                accumulation_counter += 1  # Increment the accumulation counter
                
                # Save the accumulated spectrum after reaching the target number of spots
                if accumulation_counter == spots_to_accumulate:
                    
                    # Define the path where the accumulated spectrum will be saved
                    save_path = os.path.join(new_output_dir, f"8_SpotPGT2D_{new_id}-17_layer{new_layer}_spot{str(new_spot_number).zfill(3)}_accumulated_{spots_to_accumulate}.pkl")
                    # Assume the first two elements (metadata) are consistent across files for a given key
                    metadata = pd.read_pickle(file_path)[:2] + [accumulated_spectrum]
                    with open(save_path, 'wb') as file:
                        pickle.dump(metadata, file)
                    
                    
                    # Prepare for the next accumulation set
                    new_spot_number = int(new_spot_number) + spots_to_accumulate  # Update the spot number for the next batch
                    accumulated_spectrum = None  # Reset the accumulated spectrum for the next accumulation
                    accumulation_counter = 0  # Reset the counter for the next accumulation

    # Save the results metadata to a JSON file
    result_json_path = os.path.join(new_output_dir, "accumulation_info.json")
    print(result_json)
    df_result_json = pd.DataFrame(result_json)
    
    df_result_json.to_json(result_json_path, orient='records', indent=4)






def create_data_table(root_dir, measurement_dict, spots_to_exclude_static):
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
                
                if spot_type == 'static' and spot_number <= spots_to_exclude_static:
                    continue

                # exclude specific spots, as they have errors of don't have
                # range shifts due to scanning
                
                # Check if the combination already exists in data_map
                if any(entry[1:-1] == [proton_energy, range_shift, spot_type, \
                                       layer, spot_number] for entry in data_map):
                    continue
                
                data_map.append([measurementID, proton_energy, range_shift, \
                                 spot_type, layer, spot_number, file_path])

    # Convert list of data into DataFrame
    data_map_df = pd.DataFrame(data_map, columns=['id_measurement', \
                                                  'proton_energy', 'range_shift',\
                                                  'spot_type', 'layer', \
                                                  'spot_number', 'file_path'])

    return data_map_df

if __name__ == "__main__":


    configfile = "../project_pipeline/config.yaml"
    
    root_dir = "/bigdata/invivo/machine_learning/pgt-range-reconstruction/PMMA_study/data/02_Converted_Data"
    
    spots_to_accumulate = 10
    
    cohort = "static"
    
    output_dir = "/bigdata/invivo/machine_learning/pgt-range-reconstruction/PMMA_study/data/02_Converted_Data/2024_02-SingleSpot"
    
    # Substitute labels and save configuration file
    config = substitute_labels(configfile)
    
    spots_to_exclude_static = 30
    
    # get cohort mapping dict in json format, so it can be transferred via shell command
    measurement_dict_json = json.dumps(config['dataset']['measurement_dict'])
    
    # Convert the JSON string back into a dictionarys
    measurement_dict = {}
    for key, value in json.loads(measurement_dict_json).items():
        if cohort not in value:
            continue
        measurement_dict[int(key)] = value
    
    print("Create data table...")
    data_map_df = create_data_table(root_dir, measurement_dict, spots_to_exclude_static)
    
    print("Accumulate data")
    accumulate_and_save(data_map_df, cohort, spots_to_accumulate, output_dir)

