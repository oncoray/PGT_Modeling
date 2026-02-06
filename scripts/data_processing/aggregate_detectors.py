#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:38:53 2024

@author: kiesli21
"""

import os
import numpy as np
import logging
import argparse
import concurrent.futures
from functools import partial
from tqdm import tqdm
import json
import pandas as pd  # Import pandas to handle DataFrame operations
import re  # Import re for regular expressions


def custom_string_to_float(custom_string):
    """
    Convert a float value to a custom string representation with one decimal place, replacing '.' with '-'.
    """
    value = float(custom_string.replace('-', '.'))
    return value

def aggregate_detectors(output_dir, threads, data_preprocessing_args):
    """
    Aggregate spectra from different detectors into one for each measurement.

    Parameters:
    - output_dir: Base directory where the processed spectra are saved.
    - threads: Number of processes to use for parallel processing.
    - data_preprocessing_args: Dictionary containing preprocessing arguments.
    """
    logging.info("Starting detector aggregation...")

    # Extract parameters from data_preprocessing_args
    max_iterations = data_preprocessing_args["find_mean_shift"]["max_iterations"]
    tolerance = data_preprocessing_args["find_mean_shift"]["tolerance"]
    desired_mean_location = data_preprocessing_args["desired_mean_location"]
    bg_region = data_preprocessing_args.get("bg_region", (0, 100))
    global_time_window = data_preprocessing_args.get("global_time_window", (700, 1728))

    data_states = ["Processed", "Original", "Shifted", "Background"]

    for state in data_states:

        processed_dir = os.path.join(output_dir, state)
        if not os.path.exists(processed_dir):
            logging.warning(f"Processed directory does not exist: {processed_dir}")
            continue
        else:
            logging.info(f"State: {state}. Directory to search for the data: {processed_dir}")

        spectra_groups = {}
        # Walk through the directory structure
        for root, dirs, files in os.walk(processed_dir):
            # Collect spectra files
            spectra_files = [f for f in files if f.endswith('.npy')]

            if not spectra_files:
                continue

            # Group spectra by measurement parameters excluding detector number

            for file_name in spectra_files:
                # Skip aggregated detectors (e.g., det99, det88)
                if 'det99' in file_name or 'det88' in file_name:
                    continue

                # Use the file path to extract identifiers using the regex pattern
                file_path = os.path.join(root, file_name)
                # Prepare the regex pattern
                pattern = re.compile(
                    r'^([a-zA-Z0-9_]+)_(\d+)MeV_([\d\-\.]+)MU_([\w\-]+)_([\w\-\.]+)_det(\d+)_layer(\d+)_spot(\d+)_rep(\d+)\.npy$'
                )
                match = pattern.match(os.path.basename(file_path))
                if not match:
                    logging.warning(f"File name does not match expected pattern: {file_name}")
                    continue

                # Extract identifiers
                nose_orientation = match.group(1)
                energy = int(match.group(2))
                mu = custom_string_to_float(match.group(3))
                range_shift_type = match.group(4)
                range_shift = match.group(5)
                detector = int(match.group(6))
                layer = int(match.group(7))
                local_spot_id = int(match.group(8))
                repetition = int(match.group(9))

                # Create a measurement key excluding detector number
                measurement_key = (
                    nose_orientation,
                    energy,
                    mu,
                    range_shift_type,
                    range_shift,
                    layer,
                    local_spot_id,
                    repetition
                )

                # Store file paths and metadata
                if measurement_key not in spectra_groups:
                    spectra_groups[measurement_key] = {
                        'file_paths': [],
                        'metadata': {
                            'nose_orientation': nose_orientation,
                            'energy': energy,
                            'mu': mu,
                            'range_shift_type': range_shift_type,
                            'range_shift': range_shift,
                            'detector': detector,  # Original detector number
                            'layer': layer,
                            'local_spot_id': local_spot_id,
                            'repetition': repetition
                        }
                    }
                spectra_groups[measurement_key]['file_paths'].append(file_path)

        # Prepare tasks for parallel processing
        tasks = list(spectra_groups.values())

        # Use the specified number of processes
        if threads < 1:
            threads = None  # Use default number of processes (os.cpu_count())
        logging.info(f"Using {threads} processes for aggregation.")

        # Define a partial function to pass the additional parameters to the worker function
        worker_func = partial(
            process_measurement,
            processed_dir=processed_dir,
            state=state,
            desired_mean_location=desired_mean_location,
            tolerance=tolerance,
            max_iterations=max_iterations,
            bg_region=bg_region,
            global_time_window=global_time_window,
            output_dir=output_dir,
            data_preprocessing_args=data_preprocessing_args
        )

        # Use ProcessPoolExecutor to process in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            # Use a tqdm progress bar
            with tqdm(total=len(tasks), desc="Aggregate detectors....") as pbar:
                # Submit tasks to the executor
                futures = {
                    executor.submit(worker_func, task['file_paths'], task['metadata']): task['metadata']
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    metadata = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error processing measurement {metadata}: {e}")
                        raise

        logging.info(f"Detector aggregation completed for state: {state}.")

def process_measurement(
    file_paths,
    metadata,
    processed_dir,
    state,
    desired_mean_location,
    tolerance,
    max_iterations,
    bg_region,
    global_time_window,
    output_dir,
    data_preprocessing_args):


    # Remove pattern matching; use metadata directly

    detector = metadata['detector']


    if len(file_paths) == 1:
        logging.warning(f"Just one file path for {metadata}. No need to aggregate!")
        return  # No need to aggregate if only one file
    if len(file_paths) == 0:
        logging.warning(f"No file path for {metadata}. No need to aggregate")
        return  # No need to aggregate if only one file

    combined_spectrum = None
    for file_path in file_paths:
        spectrum = np.load(file_path)
        if combined_spectrum is None:
            combined_spectrum = spectrum.copy()
        else:
            combined_spectrum += spectrum

    # Define variables for file naming and paths
    original_file_name = os.path.basename(file_paths[0])

    # For aggregated spectrum, replace with 'det99' initially
    detector_str = f'det{detector}'
    new_detector_str = 'det99'
    new_file_name = original_file_name.replace(detector_str, new_detector_str)

    # Reconstruct the subdirectory path with det99
    relative_path = os.path.relpath(os.path.dirname(file_paths[0]), processed_dir)
    path_parts = relative_path.split(os.sep)
    det_replaced = False
    for i, part in enumerate(path_parts):
        if part.startswith('det'):
            path_parts[i] = new_detector_str
            det_replaced = True
            break
    if not det_replaced:
        # If 'det{detector}' is not found, append 'det99'
        path_parts.append(new_detector_str)
    # Create the new subdirectory path
    new_sub_dir = os.path.join(processed_dir, *path_parts)

    # Ensure the new subdirectory exists
    os.makedirs(new_sub_dir, exist_ok=True)

    # Save the combined spectrum in the new subdirectory
    combined_file_path = os.path.join(new_sub_dir, new_file_name)
    np.save(combined_file_path, combined_spectrum)


def main():
    parser = argparse.ArgumentParser(description='Aggregate spectra from different detectors into one for each measurement.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory where the processed spectra are saved.')
    parser.add_argument('--threads', type=int, default=1, help='Number of processes to use for parallel processing.')
    parser.add_argument(
        '--data_preprocessing_args',
        type=str,
        default=None,
        help='Dict defining the arguments and parameters for the preprocessing routine.'
    )
    args = parser.parse_args()

    data_preprocessing_args = json.loads(args.data_preprocessing_args)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Call the aggregation function
    aggregate_detectors(
        output_dir=args.output_dir,
        threads=args.threads,
        data_preprocessing_args=data_preprocessing_args
    )

    logging.info("Detector aggregation completed!")

if __name__ == '__main__':
    main()
