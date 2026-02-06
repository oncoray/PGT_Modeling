#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
from scipy.optimize import curve_fit
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

def two_gaussians_shared_offset(x, offset, A1, mu1, sigma1, A2, mu2, sigma2):
    """
    Model function: sum of two Gaussian functions with a shared offset.
    """
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    return offset + gauss1 + gauss2

def fit_spectrum_single_gaussian(spectrum, fit_region):
    """
    Fit the spectrum to a single Gaussian function with offset within a specified region.

    Parameters:
    - spectrum: 1D numpy array representing the spectrum counts.
    - fit_region: Tuple (start_index, end_index) defining the region to fit.

    Returns:
    - popt: Optimal values for the parameters.
    - pcov: The estimated covariance of popt.
    """
    start_idx, end_idx = fit_region

    # Handle wrap-around if necessary
    if start_idx > end_idx:
        indices = np.concatenate((np.arange(start_idx, len(spectrum)), np.arange(0, end_idx + 1)))
        x_data = indices
        y_data = spectrum[indices]
    else:
        x_data = np.arange(start_idx, end_idx + 1)
        y_data = spectrum[start_idx:end_idx + 1]

    # Use a simple Gaussian with offset
    def gaussian_with_offset(x, offset, A, mu, sigma):
        return offset + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    assert np.min(y_data) < np.max(y_data), "Min and max are the same in single gaussian fit region!"

    # Initial guesses
    offset_guess = np.min(y_data)
    A_guess = np.max(y_data) - offset_guess
    mu_guess = x_data[int(len(x_data) / 2)]
    sigma_guess = (end_idx - start_idx) / 4

    initial_guesses = [offset_guess, A_guess, mu_guess, sigma_guess]

    lower_bounds = [0, 0, start_idx, 1]
    upper_bounds = [np.max(y_data), np.max(y_data)*2, end_idx, len(spectrum)]

    try:
        popt, pcov = curve_fit(
            gaussian_with_offset,
            x_data,
            y_data,
            p0=initial_guesses,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=100000
        )
    except Exception as e:
        logging.info(f"Initial guesses: {initial_guesses}")
        logging.error(f"Error in fitting spectrum with single Gaussian: {e}")
        return None, None

    return popt, pcov

def fit_spectrum_two_gaussians_shared_offset(spectrum):
    """
    Fit the spectrum to the sum of two Gaussian functions with shared offset (background).

    Parameters:
    - spectrum: 1D numpy array representing the spectrum counts.

    Returns:
    - popt: Optimal values for the parameters.
    - pcov: The estimated covariance of popt.
    """
    x_data = np.arange(len(spectrum))
    y_data = spectrum

    # Apply heavy smoothing to the spectrum
    sigma_smoothing = 20  # Adjust sigma as needed for heavy smoothing
    y_filtered = median_filter(y_data, size=10)
    y_smoothed = gaussian_filter1d(y_filtered, sigma=sigma_smoothing, mode='wrap')
    y_smoothed = gaussian_filter1d(y_data, sigma=sigma_smoothing, mode='wrap')

    # Initial guesses for the parameters
    # Offset: estimate as the minimum of y_data
    offset_guess = np.min(y_smoothed)
    # A1, A2: estimate as (max(y_data) - offset) / 2
    A2_guess = np.max(y_smoothed) - offset_guess
    A1_guess = A2_guess / 2

    mu1_guess = 600
    mu2_guess = 1200

    # Sigma guesses
    sigma1_guess = 100
    sigma2_guess = 150

    initial_guesses = [offset_guess, A1_guess, mu1_guess, sigma1_guess, A2_guess, mu2_guess, sigma2_guess]

    # Bounds for the parameters
    lower_bounds = [0, 0, 0, 1, 0, 0, 1]
    upper_bounds = [np.max(spectrum), np.max(spectrum)*1.2, len(spectrum), len(spectrum), np.max(spectrum)*1.2, len(spectrum), len(spectrum)]

    try:
        popt, pcov = curve_fit(
            two_gaussians_shared_offset,
            x_data,
            y_data,
            p0=initial_guesses,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=100000
        )
    except Exception as e:
        logging.info(f"Initial guesses: {initial_guesses}")
        logging.error(f"Error in fitting spectrum: {e}")
        return None, None

    return popt, pcov


def extract_time(time_str):
    """
    Convert time string to numerical time in seconds.
    """
    hours, minutes, seconds = map(float, time_str.split('.'))
    return hours * 3600 + minutes * 60 + seconds

def float_to_custom_string(value):
    """
    Convert a float value to a custom string representation with one decimal place, replacing '.' with '-'.
    """
    rounded_value = round(value, 1)
    custom_string = str(rounded_value).replace('.', '-')
    return custom_string


def load_data_table(data_table_path):
    """
    Load the data table as a pandas DataFrame.

    Parameters:
    - data_table_path: Path to the data table CSV file.

    Returns:
    - DataFrame containing the data table.
    """
    return pd.read_csv(data_table_path, sep=";")

def apply_phase_shift(spectrum, shift):
    """
    Apply a phase shift to a given spectrum. The spectrum can be either 1D or 2D.
    For a 2D spectrum, the shift is applied along the time axis (axis=1).

    The phase shift is decomposed into an integer (coarse) shift and a fractional (fine) shift.
    The integer shift is applied using np.roll, and the fractional shift is applied by redistributing
    counts probabilistically: each event has a probability equal to the absolute value of the fractional
    part to be shifted one additional bin in the appropriate direction (right for positive, left for negative).

    Parameters:
        spectrum (numpy.ndarray): The input array (1D or 2D) representing the spectrum.
        shift (float): The total phase shift amount in bins. A positive value shifts to the right and
                       a negative value shifts to the left.

    Returns:
        numpy.ndarray: The phase shifted spectrum (same shape as input) with integer counts.
    """

    # Determine the axis for applying the shift based on the dimensionality.
    # For 1D, the only axis is used; for 2D, the shift is applied along the time axis (axis=1).
    if spectrum.ndim == 1:
        shift_axis = 0
    elif spectrum.ndim == 2:
        shift_axis = 1
    else:
        raise ValueError("Spectrum must be either 1D or 2D.")

    # If no shift is required, return the original spectrum.
    if shift == 0:
        return spectrum

    # Decompose the shift into its integer (coarse) and fractional (fine) components.
    # Here, int() truncates toward zero. For example, int(2.2) == 2 and int(-0.2) == 0.
    coarse_shift = int(shift)
    fine_shift = shift - coarse_shift

    # Apply the coarse (integer) shift using np.roll.
    spectrum_coarse = np.roll(spectrum, shift=coarse_shift, axis=shift_axis)

    # If there is no fractional part, return the coarse shifted spectrum.
    if fine_shift == 0:
        return spectrum_coarse.astype(np.int32)

    # Define the probability to remain in the current bin.
    # For a positive fractional shift, each event has probability 1 - fine_shift to stay.
    # For a negative fractional shift, use the absolute value.
    p_remain = 1 - abs(fine_shift)

    # Vectorized binomial sampling: for each bin count, determine how many events remain.
    lower_counts = np.random.binomial(spectrum_coarse, p_remain)
    upper_counts = spectrum_coarse - lower_counts

    # For the fractional shift, the "upper" events are shifted one additional bin.
    # The direction of this extra shift is determined by the sign of fine_shift.
    additional_shift = 1 if fine_shift > 0 else -1
    upper_shifted = np.roll(upper_counts.astype(np.int32), shift=additional_shift, axis=shift_axis)

    # Combine the counts that remain and those shifted by one extra bin.
    fine_spectrum = lower_counts.astype(np.int32) + upper_shifted

    return fine_spectrum

def find_mean_shift(time_spectrum, desired_mean, tolerance=0.5, max_iterations=100):
    """
    Shift the given time spectrum so that its time mean is at a desired location.

    Parameters:
    - time_spectrum (numpy array): The input time spectrum to be shifted.
    - desired_mean (float): The desired mean location.
    - tolerance (float): The tolerance for the mean location difference.
    - max_iterations (int): Maximum number of iterations to attempt shift.

    Returns:
    - int: The shift index if a valid shift is found, otherwise None.
    """

    if max_iterations is None:
        logging.info("Max iterations is set to None. No shift will be performed!")
        return None

    assert np.sum(time_spectrum) > 0, "Spectrum has no counts. Can't find mean shift!"

    spectrum_length = len(time_spectrum)
    iterations = spectrum_length if max_iterations is None else max_iterations

    for shift in range(iterations):
        # Roll the spectrum by the current shift value
        shifted_spectrum = np.roll(time_spectrum, shift)
        # Calculate the weighted mean of the shifted spectrum
        current_mean = np.average(np.arange(spectrum_length), weights=shifted_spectrum)

        # Determine mid and outer indices for comparison
        outer_indices = np.arange(spectrum_length) < spectrum_length / 4
        outer_indices = outer_indices | (np.arange(spectrum_length) > 3 * spectrum_length / 4)
        mid_indices = ~outer_indices

        if abs(current_mean - desired_mean) <= tolerance and np.sum(shifted_spectrum[mid_indices]) > np.sum(shifted_spectrum[outer_indices]):
            return shift

    # If no valid shift is found, raise a ValueError
    raise ValueError(f"No valid shift found to reach mean location {desired_mean} within tolerance {tolerance} after {iterations} iterations")

def calc_mean_bg_level(spectrum, interval):
    """
    Calculate mean values inside and outside a specified interval in a 1D spectrum.

    Parameters:
    - spectrum: 1D numpy array of spectrum counts.
    - interval: Tuple of (start_index, end_index) defining the interval.

    Returns:
    - mean_inside: Mean value of spectrum within the interval.
    - mean_outside: Mean value of spectrum outside the interval.
    """
    start, end = interval
    n = len(spectrum)

    if start > end:
        interval_indices = np.concatenate((np.arange(start, n), np.arange(0, end + 1)))
    else:
        interval_indices = np.arange(start, end + 1)

    interval_values = spectrum[interval_indices]
    outside_values = np.delete(spectrum, interval_indices)

    mean_inside = np.mean(interval_values)
    mean_outside = np.mean(outside_values)

    return mean_inside, mean_outside

def perform_background_correction(spectrum, predicted_bg):
    """
    Correct the background in a given spectrum.

    Parameters:
    - spectrum: 2D numpy array representing the spectrum.
    - predicted_bg: Predicted background rate.

    Returns:
    - corrected_spectrum: 2D numpy array of the background-corrected spectrum.
    """
    assert np.issubdtype(spectrum.dtype, np.integer), "Spectrum contains non-int values!"
    assert np.all(spectrum >= 0), "Spectrum contains negative values!"
    assert np.sum(spectrum) > 0, "No counts in spectrum!"

    corrected_spectrum = np.zeros_like(spectrum)
    total_bins = spectrum.shape[1]

    for i, counts in enumerate(spectrum):
        bg_rate = predicted_bg
        corrected_counts = np.copy(counts)
        random_number = np.random.uniform()
        expected_bg_events = total_bins * bg_rate

        if expected_bg_events >= np.sum(corrected_counts):
            expected_bg_events = np.sum(corrected_counts) - 1
            if expected_bg_events <= 0:
                corrected_spectrum[i, :] = corrected_counts
                continue

        elif random_number <= (expected_bg_events % 1):
            expected_bg_events = int(np.ceil(expected_bg_events))
        else:
            expected_bg_events = int(np.floor(expected_bg_events))

        if expected_bg_events <= 0:
            corrected_spectrum[i, :] = corrected_counts
            continue

        grid_indices = np.linspace(0, total_bins, expected_bg_events, endpoint=False)
        step_size = grid_indices[1] if len(grid_indices) > 1 else total_bins
        random_shift = np.random.uniform() * step_size
        grid_indices += random_shift
        np.random.shuffle(grid_indices)

        for position in grid_indices:
            deleted = False
            while not deleted:
                event_indices = np.nonzero(corrected_counts)[0]

                if len(event_indices) == 0:
                    break

                distances = np.abs(event_indices - position)
                periodic_distances_1 = np.abs(event_indices + total_bins - position)
                periodic_distances_2 = np.abs(event_indices - total_bins - position)
                combined_distances = np.minimum(distances, periodic_distances_1)
                combined_distances = np.minimum(combined_distances, periodic_distances_2)
                min_distance_indices = np.where(combined_distances == np.min(combined_distances))[0]
                closest_event_index = np.random.choice(min_distance_indices)
                closest_index = event_indices[closest_event_index]

                corrected_counts[closest_index] -= 1
                deleted = True

        corrected_spectrum[i, :] = corrected_counts

    #assert np.sum(corrected_spectrum) > 0, f"No counts left in background corrected spectrum! Bg_rate: {bg_rate}, Exp bg events: {expected_bg_events},  Sum original: {np.sum(spectrum)}"

    return corrected_spectrum
