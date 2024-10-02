#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:51:28 2023

@author: kiesli21

This script processes the raw data and prepares it for feature extraction / analysis.
Follwing steps are generally performed:

1. Load the spectral data.
2. Eliminate outliers from the data set.
    - Given that the positions are already known, adjust each data point to the 
    randomized integer value of the mean of its surrounding seven neighbors.
3. Generate the summation of all individual spectras to obtain a cumulative spectral profile.
4. Identify the background (bg) region within the spectral data.
    - Perform extensive smoothing on the summed time spectrum by applying a 
    median filter, followed by two rounds of Gaussian filtering
5. Determine the largest region where the smoothed, summed time spectrum falls
below a factor of the minimum value.
6. Apply background correction to the individual spectras
7. Generate a new set of summed spectras, which are now background-corrected but not phase-adjusted.
8. Calculate the phase shifts across the spectral data.
    - Perform a linear fit of the ascending flank of the summed spectras to align the data.
9. Apply the calculated phase shifts to the individual spectras.
    - Perform random rebinning of the spectral data to consolidate data points.
10. Visualize the individual spectras, the summed spectras with various range shifts, and the summed time spectra.


"""
# Python module imports
import numpy as np
np.random.seed(42)
import pandas as pd
from scipy import ndimage
from pmma.cmd_args import processing_parser
from pmma.visulisation_methods import save_2Dspectrum, save_timesumspectrum, save_timespectrum,save_2Dsumspectrum, save_energysumspectrum, plot_activation
import os
from tqdm import tqdm
import shutil
import json
import datetime
from sklearn.linear_model import LinearRegression
from pmma.roi_definition import get_roi_array

def remove_outlier(spectrum, outlier_indices):
    """
    This function replaces outlier values in a 2D spectrum array by imputing the mean of surrounding elements.
    The surrounding elements are taken from the 7 elements before and after the outlier along the time axis.
    The function handles periodic boundary conditions by using modulo operation.
    If the new value is less than or equal to a random number between 0 and 1, it is rounded down, otherwise it is rounded up.

    Parameters:
    spectrum (numpy.ndarray): 2D array representing the spectrum. The first dimension represents energy, and the second represents time.
    outlier_indices (list): List of time indices where outlier values are present in the spectrum array.

    Returns:
    spectrum_corrected (numpy.ndarray): 2D array of the same shape as input `spectrum` where outlier values have been replaced.
    """
    # Error check for indices
    if any(idx >= spectrum.shape[1] for idx in outlier_indices):
        raise ValueError("Outlier index out of range")

    # Get the dimensions of the spectrum
    len_time = np.shape(spectrum)[1]
    len_energy = np.shape(spectrum)[0]

    # Make a copy of the spectrum to hold the corrected values
    spectrum_corrected = np.copy(spectrum)

    # Loop over each time index that contains outliers
    for idx_time in outlier_indices:
        
        # Loop over each energy level
        for idx_energy in range(len_energy):
            
            # Get the indices of the 7 elements before and after the outlier, taking into account periodicity
            left_indices = [(idx_time - i) % len_time for i in range(40, 0, -1)]
            right_indices = [(idx_time + i) % len_time for i in range(1, 41)]
        
            # Remove outlier indices from left_indices and right_indices
            left_indices = [idx for idx in left_indices if idx not in outlier_indices]
            right_indices = [idx for idx in right_indices if idx not in outlier_indices]
        
            # Compute the mean of the surrounding elements
            new_value = int(np.random.choice(np.concatenate([spectrum[idx_energy, left_indices],
                                                spectrum[idx_energy, right_indices]])))
            
            # # Generate a random number between 0 and 1
            # random_number = np.random.uniform()
            
            # # If new value is less than or equal to the random number, round down, else round up
            # if (new_value % 1) <= random_number:
            #     new_value = int(np.floor(new_value))
            # else:
            #     new_value = int(np.ceil(new_value))

            # Replace the outlier with the new value in the corrected spectrum
            spectrum_corrected[idx_energy, idx_time] = new_value
    
    # Return the corrected spectrum
    return spectrum_corrected


def apply_coarse_shift(spectrum, shift): 
    """
    Diese Funktion verschiebt das gesamte Histogramm, indem sie die 
    Ereignisse, die über die Histogrammgrenzen hinaus gehen, auf der 
    anderen Seite des Histogramms anhängt
    """
    spectrum_shifted = np.roll(spectrum, shift, axis = 0)
        
    return spectrum_shifted


def get_phase_shift(time_spectra_dict, reference, index_desired_leading_edge, 
                    filter_size_derivative, fractions_leading_edge,
                    g_median, g_gauss):
    """
    Function to calculate phase shifts in provided spectra.

    Returns:
    dict: A dictionary containing calculated phase shifts for each spectrum.

    """
    
    spectra_dict = time_spectra_dict.copy()
    
    assert reference in spectra_dict, f"Reference ({reference}) not in data."
    
    # Initialize result dict with phase shifts as zero
    result_shift = {key: 0 for key in spectra_dict.keys()}
    
    # Get the number of bins from the reference spectrum
    number_of_bins = len(spectra_dict[reference])
    
    # Generate index array for bin centering
    x_reference = np.arange(0, number_of_bins) + 0.5
    x_pitch = x_reference[1] - x_reference[0]  # pitch size for the spectra bins
    
    # Normalize all spectra based on the integral of the reference spectrum
    for key, spectrum in spectra_dict.items():    
        norm_factor = np.sum(spectra_dict[reference]) / np.sum(spectrum) 
        spectra_dict[key] = spectrum * norm_factor  # normalize the spectrum
    
    # Apply median and Gaussian filtering to smoothen the spectra
    for key, spectrum in spectra_dict.items():
        # Apply median filter
        spectra_dict[key] = ndimage.median_filter(spectrum, g_median, mode="wrap")
        # Apply gaussian filter
        spectra_dict[key] = ndimage.gaussian_filter(spectrum, g_gauss, mode="wrap")
    
    # Coarsely shift the reference spectrum to align the maximum with the center of the spectrum
    index_maximum = [np.argmax(spectra_dict[reference])]
    index_desired_maximum = int(number_of_bins / 2)
    reference_shift = int(index_desired_maximum - index_maximum[0])
    spectra_dict[reference] = apply_coarse_shift(spectra_dict[reference], reference_shift)
    result_shift[reference] += reference_shift
    
    # Coarsely shift the remaining spectra based on the maxima of their derivative
    derivative_filter = np.ones(filter_size_derivative)
    derivative_filter[filter_size_derivative//2:] = -1
    leading_edge_dict_coarse = {}
    
    for key, spectrum in spectra_dict.items():
        spectrum_derivative = ndimage.convolve(spectrum, derivative_filter, mode="wrap")
        leading_edge_dict_coarse[key] = x_reference[np.argmax(spectrum_derivative)]
    
    for key, spectrum in spectra_dict.items(): 
        spectrum_coarse_shift = int((leading_edge_dict_coarse[key] - leading_edge_dict_coarse[reference]) / x_pitch + 0.5)
        result_shift[key] += -spectrum_coarse_shift                         
        shifted_indices = (((x_reference + spectrum_coarse_shift) - 0.5) % number_of_bins).astype(int)
        spectra_dict[key] = spectrum[shifted_indices]
    
    # Apply fine shifting based on linear regression on 30% and 70% points in the reference spectrum
    min_reference = np.min(spectra_dict[reference])
    max_reference = np.max(spectra_dict[reference])
    index_max_reference = np.argmax(spectra_dict[reference])
    index_first_half = (x_reference < x_reference[index_max_reference])
    index_interval = [None, None]  # Indices of the interval for fine shift calculation
    
    # Find the indices of the 30% and 70% points in the reference spectrum
    for i, fraction in enumerate(fractions_leading_edge):
        y_fraction = fraction * (max_reference - min_reference) + min_reference
        difference_to_fraction = spectra_dict[reference] - y_fraction
        index_interval[i] = np.argmin((difference_to_fraction ** 2)[index_first_half])
    
    # Apply fine shifting based on linear regression
    leading_edge_dict_fine = {}
    for key, spectrum in spectra_dict.items():
        interval_y = spectrum[index_interval[0]:index_interval[1]]
        interval_x = x_reference[index_interval[0]:index_interval[1]]
        slope, intercept = np.polyfit(interval_x, interval_y, 1)  # Perform linear fit
        leading_edge_dict_fine[key] = (max_reference / 2 - intercept) / slope
    
    # shift spectrum so that leading edge is at specific location
    for key in spectra_dict.keys():
        fine_shift = index_desired_leading_edge - leading_edge_dict_fine[key] ###############
        result_shift[key] += fine_shift  # Add fine shift to the result
    
    return result_shift
    
def apply_phase_shift(spectrum, shift):
    """
    Apply a phase shift to a given spectrum.

    Parameters:
    spectrum (numpy.ndarray): The input array that needs phase shifting.
    shift (int or float): The phase shift amount.

    Returns:
    numpy.ndarray: The phase shifted array.
    """
    
    # If shift is zero, no need to process, return the spectrum as is.
    if shift == 0:
        return spectrum
    
    # shift is in bins.
    # First, shift for full bin values
    coarse_shift = int(shift // 1)

    spectrum_shifted = np.roll(spectrum, coarse_shift, axis=1)
    
    # If shift is fractional, calculate that part as fine shift.
    fine_shift = shift % 1
    
    # Get the shape of the input spectrum.
    shape = spectrum.shape
    
    # Use np.indices to get the indices for all values in the spectrum.
    all_indices = np.indices(shape)
    
    # This will give two 2D arrays: one for the energy coordinates and one for the time coordinates.
    # Loop through these and for each point, add it to the list the number of times specified in the spectrum_shifted.
    events = [(energy, time) for energy,time in zip(all_indices[0].ravel(), all_indices[1].ravel()) for _ in range(spectrum_shifted[energy, time])]
    
    # Compute the mid-points of the time bins.
    time_bin_mid_points = np.arange(0,shape[1]) + 0.5

    # Initialize an empty list to store shifted events.
    shifted_events = []
    
    for energy, time in events:
        
        # Shift time to the center of bin
        time_bin_shifted = time + 0.5 + fine_shift
        
        # If time bin shifted lies in mid points, no need to shift further, add to the shifted events list
        if time_bin_shifted in time_bin_mid_points:
            shifted_events.append((energy, time))
            continue
        
        # If time bin shifted has crossed the spectrum bounds, wrap it around.
        if time_bin_shifted >= shape[1]-0.5:
            idx_smaller = shape[1]-1
            idx_larger = 0
            
            smaller_bin  = time_bin_mid_points[idx_smaller]
            larger_bin = time_bin_mid_points[idx_larger]
        
        else:
            # Find the bins adjacent to the current bin.
            idx_smaller = (np.abs(time_bin_mid_points[time_bin_mid_points < time_bin_shifted] - time_bin_shifted)).argmin()
            idx_larger = (np.abs(time_bin_mid_points[time_bin_mid_points > time_bin_shifted] - time_bin_shifted)).argmin()
            
            smaller_bin = time_bin_mid_points[time_bin_mid_points < time_bin_shifted][idx_smaller]
            larger_bin =  time_bin_mid_points[time_bin_mid_points > time_bin_shifted][idx_larger]
        
        # Assign the event to one of the adjacent bins, with a probability weighted by proximity to that bin.
        if (time_bin_shifted - smaller_bin) <= np.random.uniform():
            shifted_events.append((energy,smaller_bin))
        else:
            shifted_events.append((energy,larger_bin))
        
    # Separate the list of tuples into two lists: one for energy coordinates and one for time coordinates
    energies = [event[0] for event in shifted_events]
    times = [event[1] for event in shifted_events]

    # Use np.histogram2d to generate the 2D histogram with the number of bins equal to the spectrum dimensions
    bins_energy = np.arange(0,shape[0] + 1)
    bins_time = np.arange(0,shape[1] + 1)
    spectrum_shifted, _, _ = np.histogram2d(energies, times, bins=[bins_energy, bins_time])
    
    assert np.shape(spectrum_shifted) == np.shape(spectrum), "Shape of shifted spectrum is not shape of original spectrum!"
    
    # Return the final phase shifted spectrum
    return spectrum_shifted.astype(np.int32)


def find_bg_region(spectrum_time, g_median, g_gauss, threshold_factor):
    """
    Find the largest region in a 1D histogram that represents mostly bg, considering periodic boundary conditions.

    Parameters:
    spectrum_time (numpy.ndarray): The 1D array that contains the histogram data.

    Returns:
    bg_region (tuple): The largest region that represents mostly bg. The region is represented by a tuple of (start_index, end_index).
    """
    
    # Apply median filter
    spectrum_time = ndimage.median_filter(spectrum_time, g_median, mode="wrap")
    # Apply gaussian filter two times
    spectrum_time = ndimage.gaussian_filter(spectrum_time, g_gauss, mode="wrap")

    spectrum_time = ndimage.gaussian_filter(spectrum_time, g_gauss, mode="wrap")
    
    # Estimate the bg baseline by taking the minimum of the histogram
    baseline = np.min(spectrum_time)

    # Define the threshold
    threshold = baseline * threshold_factor

    # Create a mask for the bg based on the threshold
    bg_mask = spectrum_time <= threshold
    
    len_time = len(spectrum_time)
    
    # Extend the bg mask array to handle periodic conditions
    extended_bg_mask = np.concatenate([bg_mask[-len_time:], bg_mask, bg_mask[:len_time]])

    # Label contiguous regions of bg
    labels, num_regions = ndimage.label(extended_bg_mask)

    # Find the largest region
    region_sizes = np.bincount(labels.flatten())
    largest_region_label = region_sizes[1:].argmax() + 1  # ignore 0 label
    largest_region_mask = labels == largest_region_label

    # Find the start and end index of the largest region
    start_index = np.argmax(largest_region_mask)
    end_index = len(largest_region_mask) - np.argmax(largest_region_mask[::-1])
    
    # Adjust for the added elements due to handling of periodic boundary conditions
    start_index = (start_index - len_time) % len(spectrum_time)
    end_index = (end_index - len_time) % len(spectrum_time)

    return (start_index, end_index)

def get_bg_estimation_function(df_type_energy, spot_spectra_dict, bg_regions):
    # Initialize the LUT and model parameters dictionary
    lut_list = []
    model_params = {}

    # Determine the number of energy rows from the first spectrum
    num_rows = next(iter(spot_spectra_dict.values())).shape[0]

    if np.sum(df_type_energy["spot_number"].values > 100) > 1:
        max_spots_per_layer = 225
    else:
        max_spots_per_layer = 100

    positions = np.arange(0, max_spots_per_layer * df_type_energy["layer"].max())

    for index, row in df_type_energy.iterrows():
        spot_number = row["spot_number"]
        layer = row["layer"]
        adjusted_position = max_spots_per_layer * (layer-1) + spot_number
        df_type_energy.loc[index, 'adjusted_position'] = adjusted_position

    for range_shift, df_group in df_type_energy.groupby('range_shift'):
        # Extract mean bg values for the current energy_row and range_shift
        start_index,end_index = bg_regions[range_shift]

        model_params[range_shift] = {}
    
        for energy_row in range(num_rows):
            mean_bgs = []
            positions = []
            layers = []
            spot_numbers = []

            actual_bg = {}

            for _, row in df_group.iterrows():
                spectrum = spot_spectra_dict[row['id_global']]

                if start_index < end_index:
                    bg_values = spectrum[energy_row,start_index:end_index+1]
                else:  # The region wraps around the end of the spectrum
                    bg_values = np.concatenate([spectrum[energy_row,start_index:], spectrum[energy_row,:end_index+1]]) 

                mean_bg = np.mean(bg_values)
                mean_bgs.append(mean_bg)
                
                actual_bg[row['id_global']] = mean_bg
                
                # Store the adjusted position and layer for regression
                positions.append(row['adjusted_position'])  # Normalize positions
                layers.append(row['layer'])
                spot_numbers.append(row['spot_number'])

            X = np.array(positions).reshape(-1, 1)
            y = np.array(mean_bgs)
            
            # Fit linear regression model
            model = LinearRegression().fit(X, y)
            
            # Store model parameters for the current energy_row and range_shift
            model_params[range_shift][energy_row] = (model.coef_, model.intercept_)

            for _, row in df_group.iterrows():
                position = row["adjusted_position"]
                globalid = row["id_global"]
                fitted_bg = model.predict(np.array([position]).reshape(-1, 1))[0]
                lut_list.append({
                    "globalid": globalid,
                    "range_shift": range_shift,
                    "energy_row": energy_row,
                    "irradiation_position": position, 
                    "actual_bg": actual_bg[globalid],
                    "fitted_bg": fitted_bg})
    
    return pd.DataFrame(lut_list), model_params

def extract_predicted_bg_for_globalid(bg_lut, globalid, range_shift):
    # Initialize an empty dictionary to store the result
    predicted_bg_dict = {}

    # Check if the range_shift exists in the lut
    if range_shift in bg_lut:
        # Iterate over all energy_rows for the given range_shift
        for energy_row, globalid_data in bg_lut[range_shift].items():
            # Check if the globalid exists for the current energy_row
            if globalid in globalid_data:
                # Extract the predicted_bg value
                predicted_bg = globalid_data[globalid]["fitted_bg"]
                # Map energy_row to predicted_bg in the result dictionary
                predicted_bg_dict[energy_row] = predicted_bg

    return predicted_bg_dict

def perform_data_processing(data_table_path, preprocessed_dir, spectra_type, figures_path, rois, data_preprocessing_args):
    """
    Process the data based on the given data table and save the processed data in the specified directory.
    
    Returns:
    pandas.DataFrame: A dataframe containing the process metadata for each spectrum.
    """
    
    print("Data gets processed....")
    
    reference = data_preprocessing_args["perform_data_processing"]["reference"]
    time_bin = data_preprocessing_args["perform_data_processing"]["time_bin"]
    global_time_window_set = data_preprocessing_args["perform_data_processing"]["global_time_window_set"]
    
    # Load data table as pandas dataframe
    df = pd.read_csv(data_table_path, sep=";")
    
    # list to store processing parameters
    metadata_list = []
    
    metadata_bg_list = []
    
    # perform data processing for each roi if feature_type is energy_time (=spectra_type is 1D_energy_time)
    if spectra_type == "1D":
        pass
    else:
        rois = ["full"]
    
    # Initialize empty time spectrum for later global window calculation
    summed_time_shifted_complete = None
    
    for roi in rois:
    
        print(f"Analysing roi {roi}...")
        
        # dict to store the bg regions
        shifted_bg_regions = {}    
    
        temp_data_dir = os.path.join(preprocessed_dir, "temp_data", f"{spectra_type}_{roi}")
        
        if not os.path.exists(temp_data_dir):
            os.makedirs(temp_data_dir)    
    
        # Loop over each unique spot type
        for spot_type in np.unique(df["spot_type"].values):
            df_type = df[df["spot_type"] == spot_type]
            
            shifted_bg_regions[spot_type] = {}
            
            # Loop over each unique proton energy for the current spot type
            for energy in np.unique(df_type["proton_energy"].values):
                                    
                
                print(f"Processing dataset: {spot_type},{energy}MeV")
                df_type_energy = df_type[df_type["proton_energy"] == energy]
                
                shifted_bg_regions[spot_type][energy] = {}
                
                # Check if the reference measurement ID is unique for this spot type and energy
                assert len(np.unique(df_type_energy.loc[df_type_energy["range_shift"] == reference, "id_measurement"])) == 1, "Measurement ID of reference range_shift is not unique!"
                
                # Retrieve file paths for the current spot type and energy
                data_file_paths = df_type_energy["file_path"].values
                
                # Initialize empty dictionaries to store spot spectra and time spectra for each range shift
                spot_spectra_dict = {}
                spectra_dict_time = {}
                
                # Initialize empty dict to store original spectrum for plotting
                spectra_original_dict_time = {}
                
                # Loop over each file path to get summed time spectrum
                for file_path in tqdm(data_file_paths, desc = "Load data"):
                    
                    # Load the spectrum data from file
                    spectrum = pd.read_pickle(file_path)[2]
                    
                    # convert to integer
                    spectrum = np.round(spectrum).astype(np.int32)
                    
                    roi_array = get_roi_array(roi, spectrum, data_preprocessing_args["get_roi_array"]["energy_bin_size"])
                    
                    # Get the indices of the array where the ROI is 1
                    roi_indices = np.where(roi_array == True)
                    
                    # Determine the ROI boundaries
                    min_i, max_i = np.min(roi_indices[0]), np.max(roi_indices[0])
                    min_j, max_j = np.min(roi_indices[1]), np.max(roi_indices[1])
                    
                    # Extract the ROI from the spectrum
                    spectrum = spectrum[min_i:max_i+1, min_j:max_j+1]

                    # set the base spectrum deending on the type
                    if spectra_type == "2D":
                        spectrum_original = np.copy(spectrum)
                    else:
                        spectrum_original = np.sum(spectrum, axis = 0).reshape((1, np.shape(spectrum)[1]))
                    
                    # remove outlier
                    spectrum = remove_outlier(spectrum_original, data_preprocessing_args["remove_outlier"]["outlier_indices"])       
                    
                    # Retrieve range shift and global ID from the data table
                    range_shift = df_type_energy.loc[df_type_energy["file_path"] == file_path, "range_shift"].values[0]
                    globalid = df_type_energy.loc[df_type_energy["file_path"] == file_path, "id_global"].values[0]
                    
                    # Store the loaded spectrum in the spot spectra dictionary
                    spot_spectra_dict[globalid] = np.copy(spectrum)
                    
                    # save summed time spectrum to estimate bg region
                    spectrum_time = np.sum(spectrum, axis = 0)
                    
                    # save summed time spectrum to estimate bg region
                    spectrum_time_original = np.sum(spectrum_original, axis = 0)
                    
                    if range_shift in spectra_dict_time:
                        spectra_dict_time[range_shift] += spectrum_time
                        spectra_original_dict_time[range_shift] += spectrum_time_original
                    else:
                        spectra_dict_time[range_shift] = spectrum_time
                        spectra_original_dict_time[range_shift] = spectrum_time_original
                
                figures_dir = os.path.join(figures_path,"processing_plots", f"{roi}", f"{spot_type}_{energy}MeV")
            
                if not os.path.exists(figures_dir):
                    os.makedirs(figures_dir)
                
                # save plot of original summed time spectrum
                title = title = f"summed_time_{roi}_" + spot_type + "_" + str(energy) + "MeV_original"
                save_path = os.path.join(figures_dir, "original") 
                os.makedirs(save_path, exist_ok = True)
                save_timesumspectrum(spectra_original_dict_time, time_bin, title, save_path = save_path)
    
                # save plot of summed time spectrum with outlier removed
                title = title = f"summed_time_{roi}_" + spot_type + "_" + str(energy) + "MeV_outlier_removed"
                save_path = os.path.join(figures_dir, "outlier_removed") 
                os.makedirs(save_path, exist_ok = True)
                save_timesumspectrum(spectra_dict_time, time_bin, title, save_path = save_path)
    
                bg_regions = {}
                # find background regions
                for range_shift, spectrum in tqdm(spectra_dict_time.items(), desc="Find bg region"):
                    bg_regions[range_shift] = find_bg_region(spectrum, 
                                                             data_preprocessing_args["find_bg_region"]["g_median"], 
                                                             data_preprocessing_args["find_bg_region"]["g_gauss"], 
                                                             data_preprocessing_args["find_bg_region"]["threshold_factor"])
                
                print("Background regions:", bg_regions)
                
                bg_lut, bg_model_parameters = get_bg_estimation_function(df_type_energy, spot_spectra_dict, bg_regions)
                bg_lut["spot_type"] = spot_type
                bg_lut["proton_energy"] = energy
                bg_lut["roi"] = roi
                metadata_bg_list.append(bg_lut)

                save_path_dir = os.path.join(figures_path,"activation_plots",roi,f"{spot_type}_{energy}MeV")

                os.makedirs(save_path_dir, exist_ok = True)

                plot_activation(df_type_energy, bg_lut, bg_model_parameters, save_path_dir = save_path_dir)

                # reinitiate dict to save summed time spectra to estimate phase shift 
                # on corrected spectra
                spectra_dict_time = {}
                
                # dict for the bg
                spectra_dict_time_bg = {}
                
                for globalid, spectrum in tqdm(spot_spectra_dict.items(), desc="Perform background correction"):
                    range_shift = df_type_energy.loc[df_type_energy["id_global"] == globalid, "range_shift"].values[0]
                    
                    predicted_bg_dict = bg_lut.loc[bg_lut["globalid"] == globalid, ["energy_row", "fitted_bg"]]
                    spectrum_bg_corr = perform_background_correction(spectrum, predicted_bg_dict)
                    
                    assert np.issubdtype(spectrum_bg_corr.dtype, np.integer) and np.all(spectrum_bg_corr == spectrum_bg_corr.astype(int)), "Non-integer values in spectra"
                    
                    spot_spectra_dict[globalid] = np.copy(spectrum_bg_corr)
                    # save summed time spectrum to estimate phase shift
                    spectrum_time = np.sum(spectrum_bg_corr, axis = 0)
                    
                    spectrum_orig_time = np.sum(spectrum, axis = 0)
                    
                    if range_shift in spectra_dict_time:
                        spectra_dict_time[range_shift] += spectrum_time
                    else:
                        spectra_dict_time[range_shift] = spectrum_time
                        
                    if range_shift in spectra_dict_time_bg:
                        spectra_dict_time_bg[range_shift] += spectrum_orig_time - spectrum_time
                    else:
                        spectra_dict_time_bg[range_shift] = spectrum_orig_time - spectrum_time
                        
                        
                title = title = f"summed_time_{roi}_" + spot_type + "_" + str(energy) + "MeV_bg_corrected"
                save_path = os.path.join(figures_dir, "bg_corrected") 
                os.makedirs(save_path, exist_ok = True)
                save_timesumspectrum(spectra_dict_time, time_bin, title, save_path = save_path)
                
                title = title = f"summed_time_{roi}_" + spot_type + "_" + str(energy) + "MeV_background"
                save_path = os.path.join(figures_dir, "background") 
                os.makedirs(save_path, exist_ok = True)
                save_timesumspectrum(spectra_dict_time_bg, time_bin, title, save_path = save_path)
    
                # Check if reference range shift exists in the time spectra dictionary
                assert reference in spectra_dict_time, f"No reference (range shift = {reference}) in data! Check data table!"
                            
                # Determine the phase shifts to apply
                shift_to_apply = get_phase_shift(spectra_dict_time, reference,
                                                 data_preprocessing_args["get_phase_shift"]["index_desired_leading_edge"], 
                                                 data_preprocessing_args["get_phase_shift"]["filter_size_derivative"],
                                                 tuple(data_preprocessing_args["get_phase_shift"]["fractions_leading_edge"]),
                                                 data_preprocessing_args["get_phase_shift"]["g_median"],
                                                 data_preprocessing_args["get_phase_shift"]["g_gauss"])
                
                print("Shifts to apply: ", shift_to_apply)
                
                # save bg region for later checking of itnersections with time window
                for key in shift_to_apply.keys():
                    shifted_bg_regions[spot_type][energy][key] = tuple(np.array(shift_to_apply[key] + np.array(bg_regions[key])) % np.shape(spectrum)[1])
                
                spectra_dict_time = {}
                # Loop over each file path again to apply the phase shifts
                for globalid, spectrum in tqdm(spot_spectra_dict.items(), desc = "Apply phase shift"):
                    
                    # Retrieve range shift and global ID from the data table
                    range_shift = df_type_energy.loc[df_type_energy["id_global"] == globalid, "range_shift"].values[0]
                    layer = df_type_energy.loc[df_type_energy["id_global"] == globalid, "layer"].values[0]
                    spot_number = df_type_energy.loc[df_type_energy["id_global"] == globalid, "spot_number"].values[0]
                
                    # Get the phase shift for the current range shift
                    phase_shift = shift_to_apply[range_shift]
                
                    # Apply the phase shift to the spectrum
                    spectrum_shifted = apply_phase_shift(spectrum, phase_shift)
                    
                    assert np.issubdtype(spectrum_shifted.dtype, np.integer) and np.all(spectrum_shifted == spectrum_shifted.astype(int)), "Non-integer values in spectra"
                    
                    # Save the shifted spectrum  temporarily to file
                    # needs to be done, as saving all in list would cst to much memory
                    # globalid needs to be in first place, so that feature calculations works!
                    filename = f"{globalid}_{spot_type}_{energy}MeV_{range_shift}_layer{layer}_spot{spot_number}.npy"
    
                    np.save(os.path.join(temp_data_dir, filename), spectrum_shifted)
                
                    # save time spectrum for later plotting
                    spectrum_time = np.sum(spectrum_shifted, axis = 0)
                
                    if range_shift in spectra_dict_time:
                        spectra_dict_time[range_shift] += spectrum_time
                    else:
                        spectra_dict_time[range_shift] = spectrum_time
                
                    # save complete spectrum to get optimal time window
                    if summed_time_shifted_complete is None:
                        summed_time_shifted_complete = np.sum(spectrum_shifted, axis = 0)
                    else:
                        summed_time_shifted_complete += np.sum(spectrum_shifted, axis = 0)
                        
                title = f"summed_time_{roi}_" + spot_type + "_" + str(energy) + "MeV_phase_shifted"
                save_path = os.path.join(figures_dir, "phase_shifted")
                os.makedirs(save_path, exist_ok = True)
                save_timesumspectrum(spectra_dict_time, time_bin, title, save_path = save_path)
                
                for range_shift in shift_to_apply.keys():
                    process_metadata = {
                        'ROI': roi,
                        'spot_type': spot_type,
                        'proton_energy': energy,
                        'range_shift': range_shift,
                        'background_region': bg_regions[range_shift],
                        'phase_shift': shift_to_apply[range_shift],
                        'shifted_background_region': shifted_bg_regions[spot_type][energy][range_shift]
                    }
                    metadata_list.append(process_metadata)
                            
    # Convert the list of dictionaries into a DataFrame
    process_metadata_df = pd.DataFrame(metadata_list)            
    
    print("Set global time window by preset!")
    lower_time_border, upper_time_border = global_time_window_set
    
    assert lower_time_border < upper_time_border, "Upper time border is smaller then lower time border! Check or implement that case!"
    
    global_time_window_width = upper_time_border - lower_time_border
            
    window_indices = [(lower_time_border + i) % len(summed_time_shifted_complete) for i in range(global_time_window_width)]
            
    print(f"Global time window is ({lower_time_border}, {upper_time_border})")
         
    # save global time window in metadata df
    process_metadata_df["global_time_window"] = [(lower_time_border, upper_time_border)] * len(process_metadata_df)

    # check overlap of global tie window and bg regions
    for spot_type in np.unique(df["spot_type"].values):
        df_type = df[df["spot_type"] == spot_type]
        
        for energy in np.unique(df_type["proton_energy"].values):
            print(f"Check for bg overlap: {spot_type},{energy}MeV") 
            
            for range_shift in np.unique(df_type["range_shift"].values):
               # get bg region of specific spectrum
               lower_bg_border, upper_bg_border = shifted_bg_regions[spot_type][energy][range_shift]

               # check if bg region and window region overlap
               assert check_overlap((lower_bg_border, upper_bg_border), (lower_time_border, upper_time_border)) == False,\
                   f"Bg region ({lower_bg_border}, {upper_bg_border}) and global time window ({lower_time_border}, {upper_time_border})  are overlapping!"

    for roi in rois:
        spectra_dict_time = {}
        # If the output if temporary data does not exist, create it
    
        temp_data_dir = os.path.join(preprocessed_dir, "temp_data", f"{spectra_type}_{roi}")

        
        for filename in tqdm(os.listdir(temp_data_dir), desc = "Apply global time window and save data."):
            
            # Extract globalid from filename
            globalid = int(filename.split("_")[0])
            
            # If globalid not present in dataframe, skip this file
            if globalid not in list(df["id_global"].values):
                continue
    
            # Extract the range_shift, spot_type and energy associated with the current globalid
            range_shift = df.loc[df["id_global"] == globalid, "range_shift"].values[0]
            spot_type = df.loc[df["id_global"] == globalid, "spot_type"].values[0]
            energy = df.loc[df["id_global"] == globalid, "proton_energy"].values[0]
            
            processed_data_dir = os.path.join(preprocessed_dir, f"{spot_type}_{energy}MeV", f"{spectra_type}", f"{roi}")
            
            if not os.path.exists(processed_data_dir):
                os.makedirs(processed_data_dir)

            spectrum = np.load(os.path.join(temp_data_dir, filename))
                    
            spectrum_windowed = spectrum[:, window_indices]
            
            assert np.issubdtype(spectrum_windowed.dtype, np.integer) and np.all(spectrum_windowed == spectrum_windowed.astype(int)), "Non-integer values in spectra"
        
            # save time spectrum for later plotting
            spectrum_time = np.sum(spectrum_windowed, axis = 0)

            if spot_type not in spectra_dict_time:
                spectra_dict_time[spot_type] = {}
            
            if energy not in spectra_dict_time[spot_type]:
                spectra_dict_time[spot_type][energy] = {}
        
            if range_shift in spectra_dict_time[spot_type][energy]:
                spectra_dict_time[spot_type][energy][range_shift] += spectrum_time
            else:
                spectra_dict_time[spot_type][energy][range_shift] = spectrum_time
        
            # Extracting the original filename
            filename_original = os.path.basename(df.loc[df["id_global"] == globalid, "file_path"].values[0])[:-4]
        
            # save windowed spectrum
            np.save(os.path.join(processed_data_dir, filename_original), spectrum_windowed)
            
            # delete temp spectrum
            os.remove(os.path.join(temp_data_dir, filename))
    
        # save final processed spectra plots
        for spot_type in spectra_dict_time.keys():
            for energy in spectra_dict_time[spot_type].keys():
                figures_dir = os.path.join(figures_path,"processing_plots", f"{roi}", f"{spot_type}_{energy}MeV")
                spectra_dict = spectra_dict_time[spot_type][energy]
                title = f"summed_time_{roi}_" + spot_type + "_" + str(energy) + "MeV_processed"
                save_path = os.path.join(figures_dir, "processed")
                os.makedirs(save_path, exist_ok = True)
                save_timesumspectrum(spectra_dict, time_bin, title, save_path = save_path)
                print(f"Final procesed plot saved for {spot_type}, {energy} in {save_path}")
        
        #  delete temp dir
        shutil.rmtree(os.path.join(preprocessed_dir, "temp_data", f"{spectra_type}_{roi}"))

    return process_metadata_df, pd.concat(metadata_bg_list)

def find_mean_shift(time_spectrum, desired_mean_location, max_iterations, tolerance):
    """
    Function to shift the given time_spectrum so that its time mean is at a 
    given location. The shift is done with wrap-around property using an iterative approach. 

    Args:
        time_spectrum (numpy.array): 1-D array containing the time_spectrum values.
        desired_mean_location (int): The desired location of the time mean. 
        max_iterations (int): Maximum number of iterations to perform. Defaults to 100.
        tolerance (float): Convergence tolerance. The iteration stops when the absolute difference
                           between the current and desired mean locations is below this value. 

    Returns:
        shift (int): The final shift value that brings the time mean to the desired location.
    """
    
    # Ensure the desired location is within the bounds
    assert 0 <= desired_mean_location < len(time_spectrum), 'Invalid desired location'

    original_mean_location = np.average(np.arange(len(time_spectrum)), weights=time_spectrum)

    # Perform iterative shifting
    for _ in tqdm(range(max_iterations), desc = "Find mean shift"):
        # Compute the weighted average, where the weights are the time indices
        shifted_mean_location = np.average(np.arange(len(time_spectrum)), weights=time_spectrum)

        # Check convergence
        if abs(shifted_mean_location - desired_mean_location) <= tolerance:
            shift = (shifted_mean_location - original_mean_location) % len(time_spectrum)
            break

        # Compute the necessary shift
        shift = int(np.round(desired_mean_location - shifted_mean_location)) % len(time_spectrum)

        # Shift the time_spectrum using np.roll
        time_spectrum = np.roll(time_spectrum, shift)

    return shift


def find_time_window(time_spectrum, window_size):
    """
    Function to find the starting index of a window in the given time_spectrum 
    that has the maximum sum. The window has a wrap-around property. 

    Args:
        time_spectrum (numpy.array): 1-D array containing the time_spectrum values.
        window_size (int): The size of the window.

    Returns:
        max_index (int): The starting index of the window with the maximum sum. 
    """

    assert 0 < window_size <= len(time_spectrum), 'Invalid window size'

    # Extend time_spectrum to handle wrap-around
    extended_time_spectrum = np.concatenate((time_spectrum, time_spectrum))

    # Calculate initial window sum
    window_sum = np.sum(extended_time_spectrum[:window_size])
    max_sum = window_sum
    max_index = 0

    # Slide the window over the time_spectrum
    for i in range(1, len(time_spectrum)):
        window_sum = window_sum - extended_time_spectrum[i-1] + extended_time_spectrum[i+window_size-1]

        # Update max sum and index
        if window_sum > max_sum:
            max_sum = window_sum
            max_index = i

    # The actual starting index in the original time_spectrum
    max_index %= len(time_spectrum)
    return max_index


def check_overlap(interval1, interval2):
    """
    Checks if two intervals overlap, considering periodic boundary conditions.

    :param interval1: A tuple (start, end) representing the first interval.
    :param interval2: A tuple (start, end) representing the second interval.
    :return: True if the intervals overlap, False otherwise.

    """

    start1, end1 = interval1
    start2, end2 = interval2
    
    if start1 < end1:
        if start2 >= start1 and start2 <= end1:
            return True
        elif end2 >= start1 and end2 <= end1:
            return True
        else:
            return False
    else:
        if start2 >= start1 or start2 <= end1:
            return True
        elif end2 >= start1 or end2 <= end1:
            return True
        else:
            return False

    
def perform_background_correction(spectrum, predicted_bg_dict):
    """
    This function corrects the background in a given spectrum, using lower and upper indices
    to define the region where the spectrum is considered as background.

    Parameters:
    spectrum (numpy.array): The input spectrum to be corrected.
    bg_region (tuple): Bins defining the borders of the background region

    Returns:
    numpy.array: The corrected spectrum.
    """
    
    assert np.issubdtype(spectrum.dtype, np.integer) == True, "Spectrum contains non-int values!"
    
    assert np.all(spectrum >= 0) == True, "Spectrum contains negative values!"

    # Initialize corrected spectrum with zeros
    corrected_spectrum = np.zeros_like(spectrum)
    
    total_bins = np.shape(spectrum)[1]
    
    # Loop over each spectrum to correct
    for i, counts in enumerate(spectrum):

        # Calculate the background rate in the background region
        bg_rate = predicted_bg_dict.loc[predicted_bg_dict["energy_row"] == i, "fitted_bg"].values[0]

        corrected_counts = np.copy(counts)

        # Generate a random number
        random_number = np.random.uniform()            

        # Calculate the expected number of background events in the whole spectra
        expected_background_events = total_bins * bg_rate

        # Never delete all counts as this can raise an error
        if expected_background_events >= np.sum(corrected_counts):
            expected_background_events = np.sum(corrected_counts) - 1
            if expected_background_events <= 0:
                continue

        # Round up or down the expected background events depending on the random number
        elif random_number <= (expected_background_events % 1):
            expected_background_events = int(np.ceil(expected_background_events))
        else:
            expected_background_events = int(np.floor(expected_background_events))
        
        if expected_background_events <= 0:
            continue
        
        # if expected_background_events > total_bins:
        #     expected_background_events = total_bins

        # Generate indices for background event locations
        grid_indices = np.linspace(0, total_bins, expected_background_events, endpoint = False)
        if len(grid_indices) == 1:
            step_size = total_bins
        else:
            step_size = grid_indices[1]
        random_shift = np.random.uniform() * step_size
        grid_indices = grid_indices + random_shift
        np.random.shuffle(grid_indices)
        
        # Loop over each position to eliminate background events
        for position in grid_indices:
            deleted = False
            while deleted == False:
                # Get indices of events
                event_indices = np.nonzero(corrected_counts)[0]
                
                # No events left in the corrected_counts array
                if len(event_indices) == 0:  
                    break
                
                # Calculate distances and periodic distances to random position
                distances = np.abs(event_indices - position)
                periodic_distances_1 = np.abs(event_indices + total_bins - position)
                periodic_distances_2 = np.abs(event_indices - total_bins - position)
                
                # Find the minimum distance
                combined_distances = np.minimum(distances, periodic_distances_1)
                combined_distances = np.minimum(combined_distances,periodic_distances_2)

                # Find indices with the minimum distance value
                min_distance_indices = np.where(combined_distances == np.min(combined_distances))[0]
    
                # Randomly pick one of these indices
                closest_event_index = np.random.choice(min_distance_indices)
                
                # Get the index in the original counts array
                closest_index = event_indices[closest_event_index]  

                corrected_counts[closest_index] -= 1
                deleted = True

        # Add corrected counts to the corrected spectrum
        corrected_spectrum[i,:] = corrected_counts

    assert np.sum(corrected_spectrum) > 0, "No counts left in background corrected spectrum!"

    # Return the corrected spectrum
    return corrected_spectrum


    
if __name__ == "__main__":
    # Obtain arguments for feature calculation
    parser = processing_parser("Preprocess data")
    args = parser.parse_args()
    
    data_preprocessing_args = json.loads(args.data_preprocessing_args)
    
    print(f"Processing parameters: \n {json.dumps(data_preprocessing_args, indent=4, sort_keys=True)}")
    
    # perform preprocessing
    metadata_df, metadata_bg_df = perform_data_processing(args.data_table_path, args.preprocessed_dir, args.spectrum_type, args.figures_path, args.rois, data_preprocessing_args)
    
    # save file with metadata
    directory, file_name = os.path.split(args.metadata_file_path)
    os.makedirs(directory, exist_ok = True)
    
    metadata_df.to_csv(args.metadata_file_path, sep = ";", index = False)
    
    metadata_bg_df.to_csv(args.metadata_bg_file_path, sep = ";", index = False)
    
    
    directory, file_name = os.path.split(args.completion_file_path)
    os.makedirs(directory, exist_ok = True)
    
    # make completion file for output management of snakemake
    with open(args.completion_file_path, "w") as f:
        f.write('Done')
        
    print("Preprocessing executed!!!")
    