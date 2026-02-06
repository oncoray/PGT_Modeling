#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:51:28 2023

@author: kiesli21

"""

import numpy as np
import pandas as pd
from scipy import ndimage
import uproot
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import os
import json
import logging
import matplotlib.pyplot as plt
from matplotlib import style as matplotstyle
from tqdm import tqdm
import concurrent.futures
from pmma.cmd_args import processing_parser

from scipy.spatial import cKDTree


pd.options.mode.chained_assignment = None  # default='warn'

from pmma.preprocessing_utils import perform_background_correction, calc_mean_bg_level, \
    find_mean_shift, apply_phase_shift, fit_spectrum_two_gaussians_shared_offset, fit_spectrum_single_gaussian,\
        extract_time, float_to_custom_string, load_data_table

# Matplotlib settings
matplotstyle.use('bmh')
params = {
    "figure.figsize": [18, 12],
    "figure.titlesize": 18,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "font.size": 16,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.53,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.dpi": 100
}
plt.rcParams.update(params)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -----------------------------------
# Spectrum Processing Functions
# -----------------------------------

def shift_spectrum_by_RS_peak(spectrum, desired_peak_location, preprocessing_args):
    """
    Shift the spectrum so that the Gaussian peak is at the desired location.

    Parameters:
    - spectrum: 1D numpy array representing the spectrum counts.
    - desired_peak_location: The desired location for the Gaussian peak.
    - preprocessing_args: Dictionary containing preprocessing arguments.

    Returns:
    - shifted_spectrum: The shifted spectrum.
    - shift_needed: The amount of shift applied.
    - offset: The background offset estimated from the fit.
    """
    RS_peak_shift_method = preprocessing_args.get("RS_peak_shift_method", "double_gaussian")

    if RS_peak_shift_method == "double_gaussian":
        popt, pcov = fit_spectrum_two_gaussians_shared_offset(spectrum)
        if popt is None:
            raise ValueError("Fitting failed in shift_spectrum_by_RS_peak using double_gaussian method")
        offset, A1, mu1, sigma1, A2, mu2, sigma2 = popt

        # Decide which peak to use
        if A1 < A2:
            mu = mu1
        else:
            mu = mu2

    elif RS_peak_shift_method == "single_gaussian":
        # Need to get the region from preprocessing_args
        fit_region = preprocessing_args.get("fit_region", (0, len(spectrum)-1))
        popt, pcov = fit_spectrum_single_gaussian(spectrum, fit_region)
        if popt is None:
            raise ValueError("Fitting failed in shift_spectrum_by_RS_peak using single_gaussian method")
        offset, A, mu, sigma = popt
    else:
        raise ValueError(f"Unknown RS_peak_shift_method: {RS_peak_shift_method}")

    # Calculate shift needed to bring mu to desired_peak_location
    shift_needed = desired_peak_location - mu
    shifted_spectrum = apply_phase_shift(spectrum, shift_needed)

    return shifted_spectrum, shift_needed, offset


def remove_outliers(spectrum, outlier_indices):
    """
    Replace outlier values in a 2D spectrum array with the mean of surrounding elements.
    Handles periodic boundary conditions by using modulo operation.

    Parameters:
    - spectrum: 2D numpy array representing the spectrum.
    - outlier_indices: List or array of indices representing outlier positions in time dimension.

    Returns:
    - corrected_spectrum: 2D numpy array with outliers replaced.
    """
    # Validate indices
    if any(idx >= spectrum.shape[1] for idx in outlier_indices):
        raise ValueError("Outlier index out of range")

    len_energy, len_time = spectrum.shape
    corrected_spectrum = np.copy(spectrum)

    for idx_time in outlier_indices:
        for idx_energy in range(len_energy):
            left_indices = [(idx_time - i) % len_time for i in range(40, 0, -1)]
            right_indices = [(idx_time + i) % len_time for i in range(1, 41)]

            # Remove outlier indices
            left_indices = [idx for idx in left_indices if idx not in outlier_indices]
            right_indices = [idx for idx in right_indices if idx not in outlier_indices]

            surrounding_values = np.concatenate([
                spectrum[idx_energy, left_indices],
                spectrum[idx_energy, right_indices]
            ])
            new_value = int(np.random.choice(surrounding_values))

            corrected_spectrum[idx_energy, idx_time] = new_value

    return corrected_spectrum

def find_bg_region(spectrum_original, median_filter_size, gaussian_filter_size, threshold_factor):
    """
    Find the largest region in a 1D histogram that represents mostly background.

    Parameters:
    - spectrum_original: 1D numpy array of the original spectrum.
    - median_filter_size: Size of the median filter.
    - gaussian_filter_size: Size of the Gaussian filter.
    - threshold_factor: Threshold factor to identify background region.

    Returns:
    - Tuple of (start_index, end_index) indicating the background region.
    """
    spectrum = np.copy(spectrum_original)
    spectrum = ndimage.median_filter(spectrum, median_filter_size, mode="wrap")
    spectrum = ndimage.gaussian_filter(spectrum, gaussian_filter_size, mode="wrap", output=float)

    min_spectrum = np.min(spectrum)
    max_spectrum = np.max(spectrum)

    if min_spectrum == max_spectrum:
        spectrum = np.copy(spectrum_original)
        spectrum = ndimage.gaussian_filter(spectrum, gaussian_filter_size, mode="wrap", output=float)

        min_spectrum = np.min(spectrum)
        max_spectrum = np.max(spectrum)

        assert min_spectrum != max_spectrum, "Min and max of spectrum after gaussian filtering still the same!"


    # Normalize the spectrum
    spectrum_normalized = (spectrum - min_spectrum) / (max_spectrum - min_spectrum)
    bg_mask = spectrum_normalized <= threshold_factor
    len_time = len(spectrum)
    extended_bg_mask = np.concatenate([bg_mask, bg_mask, bg_mask])
    labels, num_regions = ndimage.label(extended_bg_mask)
    region_sizes = np.bincount(labels.flatten())
    largest_region_label = region_sizes[1:].argmax() + 1
    largest_region_mask = labels == largest_region_label

    start_index = np.argmax(largest_region_mask)
    end_index = len(largest_region_mask) - np.argmax(largest_region_mask[::-1])

    start_index = (start_index - len_time) % len_time
    end_index = (end_index - len_time) % len_time

    return (start_index, end_index)



def process_spectrum(spectrum, preprocessing_args, perform_remove_outliers=True):
    """
    Process the spectrum by removing outliers and shifting.

    Parameters:
    - spectrum: 2D numpy array representing the spectrum.
    - preprocessing_args: Dictionary containing preprocessing arguments.
    - perform_remove_outliers: Boolean indicating whether to remove outliers.

    Returns:
    - shifted_spectrum: 2D numpy array of the shifted spectrum.
    - bg_region: Tuple indicating the background region.
    - mean_bg: Background mean (to be computed later).
    - shifts: Dictionary containing shifts applied.
    """
    shifts = {}  # Dictionary to store shifts

    if perform_remove_outliers:
        spectrum_no_outliers = remove_outliers(spectrum, preprocessing_args["remove_outlier"]["outlier_indices"])
    else:
        spectrum_no_outliers = np.copy(spectrum)
    spectrum_time_no_outliers = np.sum(spectrum_no_outliers, axis=0)

    # Shift spectrum based on preprocessing method
    preprocessing_method = preprocessing_args.get("preprocessing_method", "mean")
    desired_mean_location = preprocessing_args.get("desired_mean_location", 1024)
    find_mean_shift_args = preprocessing_args.get("find_mean_shift", {})
    tolerance = find_mean_shift_args.get("tolerance", 0.5)
    max_iterations = find_mean_shift_args.get("max_iterations", 100)

    if preprocessing_method == "mean":
        shift_needed = find_mean_shift(
            spectrum_time_no_outliers,
            desired_mean=desired_mean_location,
            tolerance=tolerance,
            max_iterations=max_iterations
        )
        shifts['total_shift'] = shift_needed

        if shift_needed is None:
            logging.warning("Shift could not be found. Spectrum will not be shifted.")
            shifted_spectrum = spectrum_no_outliers
            shifts['total_shift'] = 0
        else:
            shifted_spectrum = apply_phase_shift(spectrum_no_outliers, shift_needed)

        # For 'mean' method, use 'find_bg_region' to find bg_region
        bg_region = find_bg_region(
            shifted_spectrum[0, :],
            preprocessing_args["find_bg_region"]["g_median"],
            preprocessing_args["find_bg_region"]["g_gauss"],
            preprocessing_args["find_bg_region"]["threshold_factor"]
        )
        bg_region = preprocessing_args.get("bg_region", bg_region)
        mean_bg = None  # Will be computed later using bg_region

    elif preprocessing_method == "RS_peak":
        # First, shift spectrum so that mean is at desired location
        shift_needed_initial = find_mean_shift(
            spectrum_time_no_outliers,
            desired_mean=desired_mean_location,
            tolerance=tolerance,
            max_iterations=max_iterations
        )
        shifts['initial_mean_shift'] = shift_needed_initial

        if shift_needed_initial is None:
            logging.warning("Initial shift could not be found. Spectrum will not be shifted.")
            shifted_spectrum = spectrum_no_outliers
            shifts['initial_mean_shift'] = 0
        else:
            shifted_spectrum = apply_phase_shift(spectrum_no_outliers, shift_needed_initial)

        # Now, perform second shift based on RS_peak method
        shifted_spectrum_1d = shifted_spectrum[0, :]
        desired_peak_location = preprocessing_args.get("desired_peak_location", 1024)
        shifted_spectrum_1d, shift_needed_second, mean_bg = shift_spectrum_by_RS_peak(
            shifted_spectrum_1d,
            desired_peak_location=desired_peak_location,
            preprocessing_args=preprocessing_args
        )
        shifts['initial_RS_peak_shift'] = shift_needed_second

        # Apply second shift to the 2D spectrum
        shifted_spectrum = apply_phase_shift(shifted_spectrum, shift_needed_second)

        # Total shift is the sum of both shifts
        shifts['total_shift'] = shift_needed_initial + shift_needed_second

        bg_region = find_bg_region(
            shifted_spectrum[0, :],
            preprocessing_args["find_bg_region"]["g_median"],
            preprocessing_args["find_bg_region"]["g_gauss"],
            preprocessing_args["find_bg_region"]["threshold_factor"]
        )
        bg_region = preprocessing_args.get("bg_region", bg_region)

        mean_bg = None  # Will be computed later using bg_region
    else:
        raise NotImplementedError(f"Preprocessing method '{preprocessing_method}' is not implemented.")

    return shifted_spectrum, bg_region, mean_bg, shifts

def extract_spectrum(measurement_group, spectrum_type, preprocessing_args):
    """
    Extract and process the spectrum from the measurement group.

    Parameters:
    - measurement_group: DataFrame containing the measurement data.
    - spectrum_type: String indicating the type of spectrum to extract.
    - preprocessing_args: Dictionary containing preprocessing arguments.

    Returns:
    - spectrum: 2D numpy array representing the extracted spectrum.
    """
    fine_times = measurement_group["FineTimeCorrected"].values
    bins = np.linspace(0, 9.407338, 2049, endpoint=True) # 2049 as it sets histogram bins intervals
    counts, bin_edges = np.histogram(fine_times, bins=bins)

    spectrum = counts.astype(np.uint32)

    return spectrum.reshape((1, spectrum.shape[0]))



def gaussian_weight(distance, sigma):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

def stochastic_rounding(spectrum_float):
    """
    Convert a float spectrum to integer spectrum using stochastic rounding.

    Parameters:
    - spectrum_float: 1D numpy array of floats.

    Returns:
    - spectrum_int: 1D numpy array of integers.
    """
    fractional_part, integer_part = np.modf(spectrum_float)
    random_numbers = np.random.uniform(size=fractional_part.shape)
    increment = (random_numbers < fractional_part).astype(int)
    spectrum_int = integer_part.astype(int) + increment
    return spectrum_int

def aggregate_spectra(layer_group, positions, spectra, max_distance, sigma):
    """
    Aggregate spectra using Gaussian weights based on spatial distances.

    Parameters:
    - layer_group: DataFrame containing spots in the layer.
    - positions: numpy array of positions (XCoord, YCoord).
    - spectra: List of spectra corresponding to the positions.
    - max_distance: Maximum distance to consider for neighboring spots.
    - sigma: Sigma parameter for the Gaussian weight function.

    Returns:
    - aggregated_spectra_int: List of aggregated integer spectra.
    """
    nbrs = NearestNeighbors(radius=max_distance, algorithm='ball_tree').fit(positions)
    distances_array, indices_array = nbrs.radius_neighbors(positions)
    # For each spot, aggregate spectra
    aggregated_spectra = []
    for i in range(len(layer_group)):
        spot_spectrum = spectra[i]
        neighbor_indices = indices_array[i]
        neighbor_distances = distances_array[i]
        weights = gaussian_weight(neighbor_distances, sigma)

        # Aggregate spectra
        weighted_spectrum = np.zeros_like(spot_spectrum, dtype=float)
        for idx, weight in zip(neighbor_indices, weights):
            neighbor_spectrum = spectra[idx]
            weighted_spectrum += weight * neighbor_spectrum
        aggregated_spectra.append(weighted_spectrum)
    # Convert float spectra to integer spectra using stochastic rounding
    aggregated_spectra_int = [stochastic_rounding(spectrum) for spectrum in aggregated_spectra]
    return np.array(aggregated_spectra_int, dtype=np.uint32)


def calc_shift_RS_peak_to_ref_pos(
    all_spectra_df,
    detector,
    preprocessing_args,
    positional_shift_model_path=r"../../data/positional_shift_model.csv",
    reference_position=512,
    fit_region=(350, 650)
):
    """


    Parameters
    ----------
    all_spectra_df : pd.DataFrame
        A DataFrame containing (at minimum) the columns:
        ['Spectrum', 'measurement_time', 'measurement_day', 'SpotID', 'Detector',
         'shifts', 'id_global', 'Energy', 'Energy_spot', 'XCoord', 'YCoord', 'Layer',
         'total_shift_before_bg_corr', ...].
        'Spectrum' is a 1D numpy array (wrapped as an object in pandas) of size e.g. 2048 bins.

    detector : str
        The detector name or ID you want to process.

    positional_shift_model_path : str, optional
        Path to a CSV with columns ['detector','XCoords','YCoords','shift'] giving
        coordinate-based shift offsets. Default is "../../data/positional_shift_model.csv".

    reference_position : float, optional
        Desired peak position in aggregator-based approach. (Not directly used here.)

    fit_region : tuple, optional
        (start_index, end_index) region to use for the single Gaussian fit in aggregator approach.

    Returns
    -------
    final_shifts_df : pd.DataFrame
        A DataFrame with (at least) the columns:
        ['id_global', 'measurement_day', 'final_shift', 'measurement_shift', 'positional_shift'].

    Raises
    ------
    ValueError : If no shift data is found for this detector, or if data are missing.
    """
    


    # -------------------------------------------------------------------------
    # 1. Read shift data for the specified detector
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    positional_shift_model_path = os.path.join(script_dir, positional_shift_model_path)
    shift_data_detector = pd.read_csv(positional_shift_model_path, sep=";")
    shift_data_detector = shift_data_detector[shift_data_detector['detector'] == detector]
    if shift_data_detector.empty:
        raise ValueError(f"No shift data available for detector '{detector}'.")

    # -------------------------------------------------------------------------
    # 2. Filter the main data to the selected detector and prepare relevant columns
    # -------------------------------------------------------------------------
    data = all_spectra_df[all_spectra_df['Detector'] == detector].copy()
    if data.empty:
        raise ValueError(f"No data available for detector '{detector}'.")

    columns_needed = [
        'measurement_time', 'measurement_day', 'SpotID', 'Detector', "Range_shift",
        'shifts', 'id_global', 'Energy', 'Energy_spot',
        'XCoord', 'YCoord', 'Spectrum', 'Layer'
    ]
    data = data[[c for c in columns_needed if c in data.columns]].copy()


    # -------------------------------------------------------------------------
    # 3. Build KDTree to get coordinate-based shifts
    # -------------------------------------------------------------------------
    shift_coords = shift_data_detector[['XCoords', 'YCoords']].values
    shift_values = -shift_data_detector['shift'].values
    kdtree = cKDTree(shift_coords)

    # We'll store the final shifts we compute in a list of DataFrames:
    final_shifts_list = []

    # Group the data by measurement_day.
    for measurement_day, group_day in data.groupby('measurement_day'):
       
        # For each row in group_day, find the coordinate-based "positional_shift"
        positional_shifts = []
        for idx, row in group_day.iterrows():
            x_coord, y_coord = row['XCoord'], row['YCoord']
            _, index = kdtree.query([x_coord, y_coord])
            shift_map_value = shift_values[index]
            positional_shifts.append(shift_map_value)
        group_day['positional_shift'] = positional_shifts
       

        # We'll create columns to hold temporary partial-spectra
        # (the original code generally doesn't store partial steps in the DataFrame, but let's do it inline).
        group_day['final_shift'] = np.nan
        group_day['measurement_shift'] = np.nan  # Not used in aggregator approach

        for m_time, group_me in group_day.groupby('measurement_time'):

            if group_me.empty:
                continue

            # PART 1: Shift each row's original spectrum by 'shift_corrected'
            aggregated_spectrum = None
            for idx_row, row_row in group_me.iterrows():

                spec_removed_outliers = remove_outliers(row_row['Spectrum'], preprocessing_args["remove_outlier"]["outlier_indices"])

                spec_orig = spec_removed_outliers[0,:]  # 1D np.array

                pos_shift = row_row['positional_shift']
                energy_shift = calc_energy_shift(row_row['Energy_spot'])

                # Step 4: Apply combined shift
                total_shift_before_agg = pos_shift + energy_shift
                spec_shifted = apply_phase_shift(spec_orig, total_shift_before_agg)


                if aggregated_spectrum is None:
                    aggregated_spectrum = spec_shifted
                else:
                    aggregated_spectrum += spec_shifted

            size = len(aggregated_spectrum)

            assert len(aggregated_spectrum) == 2048, f"Not correct size: {size}!"
            assert np.sum(aggregated_spectrum) > 0, "Aggreagtad spectrum for shift determination is summed 0!"


            # First, shift spectrum so that mean is at desired location
            shift_spectrum_to_mean = find_mean_shift(
                aggregated_spectrum,
                desired_mean=1024,
                tolerance=0.5,
                max_iterations=10000
            )

            if shift_spectrum_to_mean is None:
                raise ValueError
            else:
                shifted_agg_spectrum = apply_phase_shift(aggregated_spectrum, shift_spectrum_to_mean)

            # Now, perform second shift based on RS_peak method
            shifted_agg_spectrum, shift_RS_peak_to_position, _ = shift_spectrum_by_RS_peak(
                shifted_agg_spectrum,
                desired_peak_location=reference_position,
                preprocessing_args=preprocessing_args
            )

            # Total shift is the sum of both shifts
            measurement_shift = shift_spectrum_to_mean + shift_RS_peak_to_position

            measurement_shift = measurement_shift % 2048

            for idx_row, row_row in group_me.iterrows():
                pos_shift = row_row['positional_shift']
                energy_shift = calc_energy_shift(row_row['Energy_spot'])
                
                final_shift = measurement_shift + pos_shift + energy_shift
                
                group_day.loc[idx_row, 'measurement_shift'] = measurement_shift
                group_day.loc[idx_row, 'final_shift'] = final_shift
                group_day.loc[idx_row, 'energy_shift'] = energy_shift
                group_day.loc[idx_row, 'positional_shift'] = pos_shift

        final_shifts_list.append(
            group_day[['id_global', 'measurement_day', 'final_shift',
                       'measurement_shift', 'positional_shift', 'energy_shift']]
        )


    # -------------------------------------------------------------------------
    # 4. Concatenate all groups and finalize
    # -------------------------------------------------------------------------
    final_shifts_df = pd.concat(final_shifts_list, ignore_index=True)

    return final_shifts_df




def calc_energy_shift(energy_value):
    """Calculate the energy shift from a polynomial model."""
    coeffs = np.array([ 2.71513606e-05, -2.01435240e-02,  5.91535881e+00, -8.81176496e+02,
        5.84110400e+04])
    return -np.polyval(coeffs, energy_value) % 2048
# -----------------------------------
# Data Processing Functions
# -----------------------------------

def extract_spectra_file(file_path, group, preprocessing_args, spectrum_type, output_dir):
    """
    Process a single measurement file.

    Parameters:
    - file_path: Path to the measurement file.
    - group: DataFrame containing the rows from the data table corresponding to this file.
    - preprocessing_args: Dictionary containing preprocessing arguments.
    - spectrum_type: Type of spectrum to process.

    Returns:
    - spectra_df: DataFrame with spectra and related data.
    """
    logging.info(f"Processing file: {file_path}")
    # Load the file data
    file_data = uproot.open(file_path)["data"].arrays(library='pd')

    if preprocessing_args["energy_range"][1] != 0:
        lower, upper = preprocessing_args["energy_range"]
        file_data = file_data[
            file_data["EnergyCalibrated"].between(lower, upper, inclusive="neither")
        ]

    grouped_data = file_data.groupby(["LayerID", "LocalSpotID", "GlobalSpotID"])

    # Build a list of spectra for this file
    spectra_list = []
    for (layer_id, local_spot_id, global_spot_id), measurement_group in grouped_data:
        # Find the matching row in the group
        matching_rows = group[
            (group['Layer'] == layer_id) &
            (group['SpotID'] == local_spot_id) &
            (group['GlobalSpotID'] == global_spot_id)
        ]
        if matching_rows.empty:
            continue
        x_coord = matching_rows['XCoord'].values[0]
        y_coord = matching_rows['YCoord'].values[0]
        id_global = matching_rows['id_global'].values[0]
        # Extract additional required parameters
        detector = matching_rows['#Detector'].values[0]
        nose_orientation = matching_rows['Nose_orientation'].values[0]
        energy = matching_rows['Proton_energy'].values[0]
        energy_spot = matching_rows["Proton_energy_spot"].values[0]
        mu = matching_rows['SpotMU'].values[0]
        mu_spot = matching_rows["SpotMU_spot"].values[0]

        range_shift = matching_rows['range_shift'].values[0]
        range_shift_type = matching_rows['range_shift_type'].values[0]
        repetition = matching_rows['repetition'].values[0]
        measurement_day = matching_rows['Measurement_day'].values[0]
        measurement_time = extract_time(matching_rows['Measurement_time'].values[0])
        triggertime = matching_rows['Triggertime'].values[0]
        # Extract the spectrum
        spectrum = extract_spectrum(measurement_group, spectrum_type, preprocessing_args)

        assert np.sum(spectrum) > 0, f"No counts in original spectrum for {file_path}"

        # Remove outliers and shift spectrum
        #shifted_spectrum, bg_region, mean_bg, shifts = process_spectrum(spectrum, preprocessing_args, perform_remove_outliers=True)

        # Store data
        spectra_list.append({
            'Layer': layer_id,
            'SpotID': local_spot_id,
            'GlobalSpotID': global_spot_id,
            'XCoord': x_coord,
            'YCoord': y_coord,
            'Spectrum': spectrum,  # Store original spectrum
            'id_global': id_global,
            'Detector': detector,
            'Nose_orientation': nose_orientation,
            'Energy': energy,
            'Energy_spot': energy_spot,
            'MU': mu,
            'MU_spot': mu_spot,
            'Range_shift_type': range_shift_type,
            'Range_shift': range_shift,
            'Repetition': repetition,
            'measurement_time': measurement_time,
            'measurement_day': measurement_day,
            'triggertime': triggertime
        })

    if not spectra_list:
        logging.info(f"No valid spectra found in file: {file_path}")
        return pd.DataFrame()

    spectra_df = pd.DataFrame(spectra_list)
    return spectra_df

def compute_average_bg_region(bg_regions, spectrum_length):
    """
    Compute the average background region from a list of background regions.

    Parameters:
    - bg_regions: List of tuples representing background regions (start_index, end_index).
    - spectrum_length: Length of the spectrum.

    Returns:
    - avg_bg_region: Tuple representing the average background region (start_index, end_index).
    """
    # Create masks for each background region
    bg_masks = []
    for start, end in bg_regions:
        bg_mask = np.zeros(spectrum_length, dtype=bool)
        if start <= end:
            bg_mask[start:end+1] = True
        else:
            bg_mask[start:] = True
            bg_mask[:end+1] = True
        bg_masks.append(bg_mask)

    # Sum the masks
    sum_bg_masks = np.sum(bg_masks, axis=0)
    majority_count = len(bg_masks) // 2 + 1  # More than half
    avg_bg_mask = sum_bg_masks >= majority_count

    # Find start and end indices of avg_bg_mask
    # Handle wrap-around
    extended_mask = np.concatenate([avg_bg_mask, avg_bg_mask, avg_bg_mask])
    labels, num_labels = ndimage.label(extended_mask)
    label_counts = np.bincount(labels)
    largest_label = label_counts[1:].argmax() + 1  # Skip label 0
    largest_region = labels == largest_label
    indices = np.where(largest_region)[0]
    start_index = indices[0] % spectrum_length
    end_index = indices[-1] % spectrum_length
    avg_bg_region = (int(start_index), int(end_index))
    return avg_bg_region

def save_spectrum(output_dir, global_id, detector, nose_orientation, energy, mu, range_shift, range_shift_type, layer, local_spot_id, repetition, spectrum, state):
    """
    Save the processed spectrum to a file.

    Parameters:
    - output_dir: Base directory to save the spectrum.
    - global_id: Global identifier for the spectrum.
    - detector: Detector number.
    - nose_orientation: Orientation of the nose.
    - energy: Energy value.
    - mu: MU value.
    - range_shift: Range shift value.
    - layer: Layer number.
    - local_spot_id: Local spot ID.
    - repetition: Repetition number.
    - spectrum: 1D numpy array representing the spectrum.
    """
    float_mu = float_to_custom_string(mu)
    sub_dir = f"{state}/{nose_orientation}/{int(energy)}MeV/{float_mu}MU/{range_shift_type}/{range_shift}/det{detector}"

    os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

    file_name = f"{nose_orientation}_{int(energy)}MeV_{float_mu}MU_{range_shift_type}_{range_shift}_det{int(detector)}_layer{int(layer)}_spot{int(local_spot_id)}_rep{int(repetition)}.npy"
    np.save(os.path.join(output_dir, sub_dir, file_name), spectrum.astype(np.uint16))




# -----------------------------------
# Main Execution
# -----------------------------------


def process_single_file(args):
    """
    Process a single group corresponding to a unique file path.

    Parameters:
    - args: Tuple containing (file_path, group, preprocessing_args, spectrum_type)

    Returns:
    - spectra_df: DataFrame containing spectra and related data
    """
    file_path, group, preprocessing_args, spectrum_type, output_dir = args

    spectra_df = extract_spectra_file(
        file_path, group, preprocessing_args, spectrum_type, output_dir
    )

    if spectra_df.empty:
        logging.info("Spectra df is empty. Skip dataset!")
        return None  # Skip this group

    return spectra_df


def bg_corr_and_save_of_final_spectrum(args):
    idx, row_data, output_dir, preprocessing_args = args

    id_global = row_data['id_global']
    aggregated_spectrum = np.array(row_data['aggregated_spectrum'], dtype=np.uint32)
    # Now aggregated_spectrum is reconstructed

    detector = row_data['Detector']
    nose_orientation = row_data['Nose_orientation']
    energy = row_data['Energy']
    energy_spot = row_data['Energy_spot']
    mu = row_data['MU']
    mu_spot = row_data['MU_spot']
    range_shift = row_data['Range_shift']
    range_shift_type = row_data['Range_shift_type']
    layer = row_data['Layer']
    local_spot_id = row_data['SpotID']
    repetition = row_data['Repetition']

    measurement_time = row_data['measurement_time']
    measurement_day = row_data['measurement_day']
    triggertime = row_data['triggertime']

    bg_region = preprocessing_args["bg_region"]
    # Calculate mean background and signal
    mean_bg_value, mean_signal = calc_mean_bg_level(np.sum(aggregated_spectrum, axis = 0), bg_region)
    mean_relative_bg = mean_bg_value / np.sum(aggregated_spectrum) if np.sum(aggregated_spectrum) != 0 else 0

    # Proceed with background correction
    aggregated_spectrum_bg_corr = perform_background_correction(aggregated_spectrum, mean_bg_value)


    if np.sum(aggregated_spectrum_bg_corr) == 0:
        print("No counts in final processed spectrum before global windowing:")
        print(f"{nose_orientation}, {energy}, {mu}, rs: {range_shift}, {range_shift_type}, layer: {layer}, spot: {local_spot_id}, rep: {repetition}")
        print(f"mean_bg: {mean_bg_value}, mean_signal:{mean_signal}, mean_relative_bg: {mean_relative_bg}")



    spectrum_processed = aggregated_spectrum_bg_corr

    if np.sum(spectrum_processed) == 0:
        print("No counts in final processed spectrum after global windowing:")
        print(f"{nose_orientation}, {energy}, {mu}, rs: {range_shift}, {range_shift_type}, layer: {layer}, spot: {local_spot_id}, rep: {repetition}")
        print(f"mean_bg: {mean_bg_value}, mean_signal:{mean_signal}, mean_relative_bg: {mean_relative_bg}, sum_spectrum_non_windowed_after_bgcorr: {np.sum(aggregated_spectrum_bg_corr)}")

    save_spectrum(
        output_dir=output_dir,
        global_id=id_global,
        detector=detector,
        nose_orientation=nose_orientation,
        energy=energy,
        mu=mu,
        range_shift=range_shift,
        range_shift_type=range_shift_type,
        layer=layer,
        local_spot_id=local_spot_id,
        repetition=repetition,
        spectrum=spectrum_processed,
        state="Processed"
    )

    preprocessing_metadata = {
        "id_global": id_global,
        "Layer": layer,
        "local_spotID": local_spot_id,
        "XCoord": row_data["XCoord"],
        "YCoord": row_data["YCoord"],
        "avg_bg_region": bg_region,
        "bg_region": bg_region,
        "mean_bg": mean_bg_value,
        "mean_relative_bg": mean_relative_bg,
        "mean_signal": mean_signal,
        "SNR": mean_signal / mean_bg_value if mean_bg_value != 0 else np.inf,
        "measurement_time": measurement_time,
        "measurement_day": measurement_day,
        "triggertime": triggertime,
        'measurement_shift': row_data["measurement_shift"],
        'positional_shift': row_data["positional_shift"],
        'final_shift': row_data["final_shift"],
        'Detector': detector,
        'Nose_orientation': nose_orientation,
        'Energy': energy,
        'Energy_spot': energy_spot,
        'MU': mu,
        'MU_spot': mu_spot,
        'Range_shift': range_shift,
        'Range_shift_type': range_shift_type,
        'Repetition': repetition,
        "mean_bg": mean_bg_value
    }

    return preprocessing_metadata


def perform_gaussian_aggregation_of_close_neighbours(args):
    group, perform_aggregation, max_distance, sigma = args
    try:
        positions = group[['XCoord', 'YCoord']].values
        spectra = [spectrum for spectrum in group['Final_shifted_Spectrum'].values]

        if perform_aggregation:
            aggregated_spectra_int = aggregate_spectra(
                group, positions, spectra, max_distance, sigma
            )
            # Ensure alignment
            aggregated_spectra_int = aggregated_spectra_int[np.argsort(group.index)]
            return list(zip(group.index, aggregated_spectra_int))
        else:
            # If aggregation is disabled, use the adjusted shifted spectra directly
            return list(zip(group.index, spectra))
    except Exception as e:
        logging.error(f"Error aggregating group {group.name}: {e}")
        return []

def perform_data_processing(
    data_table_path,
    output_dir,
    spectrum_type,
    figures_path,
    preprocessing_args,
    threads,
    detector
):
    logging.info(spectrum_type)

    if spectrum_type != "1D":
        raise NotImplementedError(f"{spectrum_type} not implemented. Only 1D spectra processing is implemented.")

    data_frame = load_data_table(data_table_path)

    data_frame = data_frame[data_frame["#Detector"] == detector]

    # Ensure 'repetition' column exists
    if 'repetition' not in data_frame.columns:
        data_frame['repetition'] = 1  # Default to 1 if not present

    # Prepare arguments for each group
    args_list = [
        (file_path, group, preprocessing_args, spectrum_type, output_dir)
        for file_path, group in data_frame.groupby('File path')
    ]

    all_spectra_df_list = []

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        # Map the process_group function to the arguments list
        results = list(tqdm(
            executor.map(process_single_file, args_list),
            total=len(args_list),
            desc="Processing datasets in parallel..."
        ))

    # Collect spectra_df from each result
    for result in results:
        if result is not None:
            spectra_df = result
            all_spectra_df_list.append(spectra_df)

    if not all_spectra_df_list:
        logging.error("No data to process after initial extraction.")
        return

    all_spectra_df = pd.concat(all_spectra_df_list, ignore_index=True)

    # Perform optimization
    logging.info("Calculation shifts needed so RS peak aligns with ref position...")
    final_shifted_df = calc_shift_RS_peak_to_ref_pos(all_spectra_df, detector, preprocessing_args)

    all_spectra_df = all_spectra_df.merge(final_shifted_df, on=['id_global', 'measurement_day'], how='left')

    # Now apply adjusted shifts
    logging.info("Applying adjusted shifts to spectra...")

    # Initialize columns to avoid KeyError
    all_spectra_df['Final_shifted_Spectrum'] = None
    all_spectra_df['Aggregated_Spectrum'] = None  

    # Apply adjusted shifts to the spectra
    for idx, row in all_spectra_df.iterrows():
        final_shift = row['final_shift']
        if pd.isnull(final_shift):
            logging.warning(f"No adjusted shift found for id_global {row['id_global']}. Skipping.")
            continue
        original_spectrum = row['Spectrum']
        # Apply adjusted shift
        adjusted_spectrum = apply_phase_shift(original_spectrum, final_shift)

        assert np.sum(adjusted_spectrum) > 0, "Adjusted spectrum has no events!"
        all_spectra_df.at[idx, 'Final_shifted_Spectrum'] = adjusted_spectrum

    # Now perform aggregation after adjusted shifts
    logging.info("Aggregate spectra from close neigbours using gaussian aggregation...")

    all_spectra_df["Range_shift"] = all_spectra_df["Range_shift"].astype(str)

    grouping_columns = [
        'Detector', 'Nose_orientation', 'Energy', 'MU',
        'Range_shift', 'Range_shift_type', 'Repetition', 'Layer'
    ]
    
    # Extract aggregation settings
    aggregation_args = preprocessing_args.get("neighbour_aggregation", {})
    perform_neighbour_aggregation = aggregation_args.get("perform_neighbour_aggregation", True)
    max_distance = aggregation_args.get("max_distance", 15.6)
    sigma = aggregation_args.get("sigma", 7.8)
    
    # Build the list of subgroup DataFrames in one pass
    groups = [g for _, g in all_spectra_df.groupby(grouping_columns, sort=False)]

    # Use ProcessPoolExecutor to parallelize the aggregation
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        # Map the aggregate_group function to each group
        agg_results = list(tqdm(
            executor.map(perform_gaussian_aggregation_of_close_neighbours, [(group, perform_neighbour_aggregation, max_distance, sigma)  for group in groups]),
            total=len(groups),
            desc="Perform gaussian aggregation of close neighbors in parallel..."
        ))

    # Update the Aggregated_Spectrum column with the aggregated results
    for group_result in agg_results:
        for idx, aggregated_spectrum in group_result:
            all_spectra_df.at[idx, 'Aggregated_Spectrum'] = aggregated_spectrum
            assert np.sum(aggregated_spectrum) > 0, "Aggregated_Spectrum has no events!"

    # Now proceed with background correction, final shifting, and saving
    logging.info("Proceeding with background correction, final shifting, and saving spectra...")

    # Prepare arguments for each row
    row_args_list = []
    for idx, row in all_spectra_df.iterrows():
        id_global = row['id_global']
        final_shift = row['final_shift']

        # Collect necessary data
        aggregated_spectrum = np.array(row['Aggregated_Spectrum'], dtype=np.uint32)
        aggregated_spectrum_list = aggregated_spectrum.tolist()

        row_data = {
            'id_global': id_global,
            'aggregated_spectrum': aggregated_spectrum_list,
            'Detector': row['Detector'],
            'Nose_orientation': row['Nose_orientation'],
            'Energy': row['Energy'],
            'Energy_spot': row['Energy_spot'],
            'MU': row['MU'],
            'MU_spot': row['MU_spot'],
            'Range_shift': row['Range_shift'],
            'Range_shift_type': row['Range_shift_type'],
            'Layer': row['Layer'],
            'SpotID': row['SpotID'],
            'XCoord': row['XCoord'],
            'YCoord': row['YCoord'],
            'Repetition': row['Repetition'],
            'final_shift': row["final_shift"],
            'measurement_shift': row["measurement_shift"],
            'energy_shift': row["energy_shift"],
            'positional_shift': row["positional_shift"],
            'measurement_time': row['measurement_time'],
            'measurement_day': row['measurement_day'],
            'triggertime': row['triggertime'],
        }

        row_args_list.append((idx, row_data, output_dir, preprocessing_args))

    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        results = list(tqdm(
            executor.map(bg_corr_and_save_of_final_spectrum, row_args_list),
            total=len(row_args_list),
            desc=f"Processing spectra in parallel..."
        ))


    # Collect bg_region_data
    preprocessing_metadata_list = [result for result in results if result is not None]

    # Convert bg_region_list to DataFrame
    preprocessing_metadata_df = pd.DataFrame(preprocessing_metadata_list)

    # Save all bg regions data
    filename = f"preprocessing_metadata_{spectrum_type}_det{detector}.csv"
    output_path = os.path.join(output_dir, filename)
    preprocessing_metadata_df.to_csv(output_path, sep=";", index=False)
    logging.info("Data processing and optimization completed.")

def main():
    """
    Main function to execute data processing.
    """
    logging.info("Starting data processing...")

    parser = processing_parser("Preprocess data")

    args = parser.parse_args()

    print(f"Threads: {args.threads}")

    data_preprocessing_args = json.loads(args.data_preprocessing_args)

    logging.info(f"Processing parameters: \n{json.dumps(data_preprocessing_args, indent=4, sort_keys=True)}")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(args)
    perform_data_processing(
        args.data_table_path,
        args.output_dir,
        args.spectrum_type,
        args.figures_path,
        data_preprocessing_args,
        args.threads,
        args.detector
    )

    # Create the completion file
    completion_content = {
        "status": "success"
    }

    try:
        with open(args.completion_file_path, 'w') as completion_file:
            json.dump(completion_content, completion_file, indent=4)
        logging.info(f"Completion file created at {args.completion_file_path}")
    except Exception as e:
        logging.error(f"Failed to create completion file: {e}")
        raise

    logging.info("Everything finished!")

if __name__ == "__main__":
    main()


