# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:13:23 2023

@author: Aaron Kieslich
"""
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import re
import uproot
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom import for argument parsing
from pmma.cmd_args import preparation_parser

def extract_data(root_file_path):
    """Unchanged: Extract spot‑level information from a ROOT file."""
    start_time = time.time()
    try:
        file_data = uproot.open(root_file_path)["data"].arrays(library='pd')
        if file_data.empty:
            logging.warning(f"No data found in {root_file_path}")
            return pd.DataFrame()

        spot_mu_unique = file_data["SpotMU"].unique()
        spot_energy_unique = file_data["SpotEnergy"].unique()
        detector = os.path.basename(root_file_path)[8:10]

        if len(spot_mu_unique) == 0 or len(spot_energy_unique) == 0:
            logging.warning(f"No SpotMU or SpotEnergy found in {root_file_path}")
            return pd.DataFrame()

        spot_mu_median = np.round(np.median(spot_mu_unique), 1)
        spot_energy_median = np.round(np.max(spot_energy_unique), 0)

        file_data = file_data[(file_data["SpotMU"] != 0) & (file_data["LayerID"] != 0)]
        file_data = file_data[(file_data["SpotXCoordinate"] != 10000) & (file_data["GlobalSpotID"] != 0)]
        if file_data.empty:
            logging.warning(f"No valid data after filtering in {root_file_path}")
            return pd.DataFrame()

        file_data = file_data.drop_duplicates(subset='GlobalSpotID')

        spot_mu = file_data["SpotMU"].copy()
        spot_energy = file_data["SpotEnergy"].copy()

        # Add fixed columns
        file_data["File path"] = root_file_path
        file_data["Proton_energy"] = spot_energy_median
        file_data["SpotMU"] = spot_mu_median
        file_data["#Detector"] = detector
        file_data["Proton_energy_spot"] = spot_energy
        file_data["SpotMU_spot"] = spot_mu

        # Rename columns
        file_data = file_data.rename(columns={
            "LayerID": "Layer",
            "LocalSpotID": "SpotID",
            "SpotXCoordinate": "XCoord",
            "SpotYCoordinate": "YCoord",
            "Triggertime": "Triggertime",
        })

        selected_columns = [
            "File path", "Proton_energy", "SpotMU", "#Detector", "Layer", "SpotID",
            "GlobalSpotID", "XCoord", "YCoord", "Proton_energy_spot", "SpotMU_spot",
            "Triggertime",
        ]
        return file_data[selected_columns]

    except Exception as e:
        logging.error(f"Error extracting data from {root_file_path}: {e}")
        return pd.DataFrame()


def find_potential_files(main_data_dir, extraction_info):
    """Unchanged: Walk directory tree and identify candidate ROOT files."""
    file_paths = []
    logging.info("Searching for potential files…")

    for measurement_day in os.listdir(main_data_dir):
        m_day_dir = os.path.join(main_data_dir, measurement_day)
        if not os.path.isdir(m_day_dir):
            continue

        for nose_orientation in os.listdir(m_day_dir):
            if nose_orientation not in extraction_info["nose_orientations"]:
                continue
            print(nose_orientation)

            m_day_nose_dir = os.path.join(m_day_dir, nose_orientation)
            if not os.path.isdir(m_day_nose_dir):
                continue

            for proton_energy in os.listdir(m_day_nose_dir):
                m_day_nose_energy_dir = os.path.join(m_day_nose_dir, proton_energy)
                if not os.path.isdir(m_day_nose_energy_dir):
                    continue

                try:
                    proton_energy_int = int(proton_energy.replace("MeV", ""))
                except ValueError:
                    continue


                if proton_energy_int not in extraction_info["proton_energies"]:
                    continue

                print(proton_energy_int)

                for range_shift in os.listdir(m_day_nose_energy_dir):
                    # Accept directories containing any specified range‑shift type **or** exactly 'ref'.
                    if not (range_shift.lower() == "ref" or any(rs_type in range_shift for rs_type in extraction_info["range_shift_types"])):
                        continue

                    print(range_shift)

                    m_day_nose_energy_rs_dir = os.path.join(m_day_nose_energy_dir, range_shift)
                    if not os.path.isdir(m_day_nose_energy_rs_dir):
                        continue

                    for root_file in os.listdir(m_day_nose_energy_rs_dir):
                        file_path = os.path.join(m_day_nose_energy_rs_dir, root_file)
                        if file_path.endswith(".root"):
                            detector_str = os.path.basename(file_path)[8:10]
                            try:
                                detector = int(detector_str)
                            except ValueError:
                                continue
                            if detector not in extraction_info["detectors"]:
                                continue
                            file_paths.append(file_path)

    logging.info(f"Found {len(file_paths)} potential files.")
    return file_paths


def process_single_file(file_path, extraction_info, time_pattern):
    """Process one ROOT file and attach metadata parsed from its path."""
    try:
        logging.info(f"Processing {file_path}…")
        result_file_df = extract_data(file_path)
        if result_file_df.empty:
            logging.warning(f"No data extracted for file {file_path}")
            return pd.DataFrame()

        # Validate energy & MU against extraction_info
        spotmu_unique = result_file_df["SpotMU"].unique()
        proton_energy_unique = result_file_df["Proton_energy"].unique()

        if len(spotmu_unique) != 1 or len(proton_energy_unique) != 1:
            logging.warning(f"Inconsistent SpotMU or Proton_energy in file {file_path}")
            return pd.DataFrame()

        file_spotmu = spotmu_unique[0]
        if not any(np.isclose(file_spotmu, mu, atol=1e-5) for mu in extraction_info["SpotMUs"]):
            logging.warning(f"SpotMU {file_spotmu} not in allowed set {extraction_info['SpotMUs']}")
            return pd.DataFrame()

        file_energy = proton_energy_unique[0]
        if file_energy not in extraction_info["proton_energies"]:
            logging.warning(f"Energy {file_energy} not in allowed set {extraction_info['proton_energies']}")
            return pd.DataFrame()

        # Measurement time from filename (unchanged)
        match = time_pattern.search(file_path)
        result_file_df["Measurement_time"] = match.group(1) if match else "Unknown"

        # ------------------------------------------------------------------
        # Parse metadata from directory structure
        # ------------------------------------------------------------------
        parts = file_path.split(os.sep)
        if len(parts) >= 5:
            result_file_df["Measurement_day"] = parts[-5]
            result_file_df["Nose_orientation"] = parts[-4]

            range_shift_full = parts[-2]  # Directory directly above the ROOT file
            # NEW: Recognise special directory "ref" (case‑insensitive)
            if range_shift_full.lower() == "ref":
                result_file_df["range_shift_type"] = "ref"
                result_file_df["range_shift"] = "ref"
            else:
                # Fall back to historic pattern "(<type>)_(<shift>)mm"
                range_shift_pattern = re.compile(r"^(?P<range_shift_type>[^_]+)_(?P<range_shift>.+?)(mm)?$")
                match_rs = range_shift_pattern.match(range_shift_full)
                if match_rs:
                    result_file_df["range_shift_type"] = match_rs.group("range_shift_type")
                    result_file_df["range_shift"] = match_rs.group("range_shift")
                else:
                    logging.warning(f"Could not parse range_shift '{range_shift_full}' in file {file_path}")
                    result_file_df["range_shift_type"] = "Unknown"
                    result_file_df["range_shift"] = "Unknown"
        else:
            logging.warning(f"Unexpected file path structure: {file_path}")
            result_file_df["Measurement_day"] = "Unknown"
            result_file_df["Nose_orientation"] = "Unknown"
            result_file_df["range_shift_type"] = "Unknown"
            result_file_df["range_shift"] = "Unknown"

        return result_file_df

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return pd.DataFrame()


def create_data_table(main_data_dir, output_file, extraction_info=None, threads=1):
    """
    Creates a data table by extracting data from ROOT files in the specified directory,
    filtering based on extraction_info, and optionally assigns training and validation cohorts.
    Utilizes a pool of processes (rather than threads) to process files in parallel.

    Parameters:
    main_data_dir (str): Path to the main data directory.
    output_file (str): Path to the output CSV file.
    extraction_info (dict, optional): Dictionary containing criteria to filter files.
    threads (int): Number of worker processes to use for parallel file processing.
    """
    assert isinstance(main_data_dir, str), "main_data_dir must be a string"
    assert isinstance(output_file, str), "output_file must be a string"
    assert isinstance(threads, int) and threads > 0, "threads must be a positive integer"

    if extraction_info is None:
        logging.error("No extraction_info provided. Cannot proceed.")
        return

    logging.info(f"Using {threads} worker processes for parallel processing.")

    file_paths = find_potential_files(main_data_dir, extraction_info)
    if not file_paths:
        logging.error("No potential files found. Exiting.")
        return

    results = []
    time_pattern = re.compile(r"_(\d{2}\.\d{2}\.\d{2})\+")

    # Process files in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(process_single_file, fp, extraction_info, time_pattern): fp
            for fp in file_paths
        }

        # Use tqdm to monitor progress of completed futures
        for future in tqdm(as_completed(futures), desc="Processing files", total=len(futures)):
            file_path = futures[future]
            try:
                result_file_df = future.result()
                if not result_file_df.empty:
                    results.append(result_file_df)
            except Exception as e:
                logging.error(f"Error in future processing {file_path}: {e}")

    if not results:
        logging.error("No data extracted. Exiting.")
        return

    logging.info("Creating final data table...")

    # Concatenate all results into a single DataFrame
    data_map_df = pd.concat(results, ignore_index=True)

    # Generate unique IDs
    data_map_df['id_global'] = np.arange(1, len(data_map_df) + 1)

    # Reorder columns to make 'id_global' the first column
    cols = ['id_global'] + [col for col in data_map_df.columns if col != 'id_global']
    data_map_df = data_map_df[cols]

    logging.info("Assigning repetition numbers based on Measurement_time and File path...")

    # Define characteristics to group by (excluding 'Measurement_time')
    characteristics_columns = [
        "Proton_energy", "SpotMU", "#Detector",
        "Nose_orientation", "range_shift_type", "range_shift",
    ]

    # Step 1: For each group and File path, find the earliest Measurement_time
    file_repetition_df = (
        data_map_df
        .groupby(characteristics_columns + ['File path'])['Measurement_time']
        .min()
        .reset_index()
    )

    # Step 2: Sort the File paths within each group by Measurement_time
    file_repetition_df = file_repetition_df.sort_values(
        by=characteristics_columns + ['Measurement_time']
    )

    # Step 3: Assign repetition numbers based on sorted order within each group
    file_repetition_df['repetition'] = (
        file_repetition_df
        .groupby(characteristics_columns)
        .cumcount() + 1
    )

    # Step 4: Merge the repetition numbers back into the main DataFrame
    data_map_df = data_map_df.merge(
        file_repetition_df[characteristics_columns + ['File path', 'repetition']],
        on=characteristics_columns + ['File path'],
        how='left'
    )

    logging.info("Repetition numbers assigned based on Measurement_time and File path.")

    # Assertion to ensure that within each group and repetition, there is only one File path
    logging.info("Verifying that each repetition within a group has a unique File path...")
    group_cols = characteristics_columns + ['repetition']
    file_path_counts = data_map_df.groupby(group_cols)['File path'].nunique()
    problematic_groups = file_path_counts[file_path_counts > 1]
    assert problematic_groups.empty, (
        f"Some repetitions within groups have multiple File paths:\n{problematic_groups}"
    )
    logging.info("Each repetition within a group is associated with a unique File path.")

    # Assertion to ensure uniqueness of specified combinations
    unique_columns = [
        "Proton_energy", "SpotMU", "#Detector", "Layer",
        "SpotID", "GlobalSpotID", "Measurement_day",
        "Nose_orientation", "range_shift_type", "range_shift", "repetition"
    ]
    duplicates = data_map_df.duplicated(subset=unique_columns, keep=False)
    assert not duplicates.any(), (
        "There are duplicate rows with the same combination of specified columns."
    )

    logging.info("All specified combinations are unique.")

    # Save DataFrame as CSV
    data_map_df.to_csv(output_file, index=False, sep=";")
    logging.info(f"Data table saved to {output_file}.")

if __name__ == "__main__":
    parser = preparation_parser("Create data paths table (updated)")

    args = parser.parse_args()

    extraction_info = json.loads(args.extraction_info) if args.extraction_info is not None else None
    if extraction_info is None:
        logging.error("Extraction info not provided. Exiting.")
        exit(1)

    assert isinstance(extraction_info, dict), "extraction_info must be a dict."

    create_data_table(args.root_dir, args.output_file, extraction_info, threads=args.threads)



