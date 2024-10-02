# -*- coding: utf-8 -*-
"""
This module performs the calculation of histogram parameters
    
Authors: Julia Wiedkamp, Sonja Schellhammer, 2021-2022
         Contact: s.schellhammer@hzdr.de
         Aaron Kieslich, 2023
"""

import numpy as np
import pandas as pd
from pmma.cmd_args import feature_calculation_parser
from pmma.visulisation_methods import plot_features
from tqdm import tqdm
import itertools
import os
import xml.etree.ElementTree as ET
from pmma.mirp.mainFunctions import extract_features
import shutil
import glob
import logging
from pmma.roi_definition import get_roi_array
logging.getLogger().setLevel(logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def energy_features_cal(spectra_dict, energy_bin_size, n_slices=5):
    """
    Function to calculate the absolute and relative intensities of regions of interest (rois) 
    in a spectrum, as well as the peak ratios of absolute intensities between each pair of rois
    and for different time slices of the spectrum.

    Parameters:
    spectrum (np.array): The spectrum data.
    rois (list of np.array): The regions of interest within the spectrum.
    energy_bin_size (float): Size of the energy bin.
    n_slices (int): Number of slices along axis=1 of the spectrum. Default is 5.

    Returns:
    dict: Dictionary of absolute and relative intensities, and ratios. 
          Keys are the parameter names and values are the calculated parameters.
    """

    # Initialize an empty dictionary to store the results
    df_param = {}

    # Calculate the total number of events in the spectrum
    n_events = np.sum(spectra_dict["full"])

    # Calculate the absolute and relative intensity for each roi
    for roi, spectrum in spectra_dict.items():
        abs_intensity = np.sum(spectrum)
        df_param[f"Absolute_intensity_{roi}"] = abs_intensity
        df_param[f"Relative_intensity_{roi}"] = abs_intensity / n_events

    rois = spectra_dict.keys()

    # Calculate the ratio of absolute intensities for each pair of rois
    for roi1, roi2 in itertools.combinations(rois, 2):  # 2 for pairs
        ratio = df_param[f"Absolute_intensity_{roi1}"] / df_param[f"Absolute_intensity_{roi2}"] if df_param[f"Absolute_intensity_{roi2}"] != 0 else np.inf
        df_param[f"Peak_ratio_{roi1}_{roi2}"] = ratio

    # Calculate peak ratios for different slices
    slice_width = int(spectrum.shape[1] / n_slices)
    for slice_idx in range(n_slices):
        # Define the beginning and ending bin of each slice
        beginning_bin = slice_idx * slice_width
        ending_bin = (slice_idx + 1) * slice_width if slice_idx < n_slices - 1 else spectrum.shape[1]

        # Calculate absolute intensities for each roi within the slice
        abs_intensities_slice = {}
        for roi, spectrum in spectra_dict.items():
            roi_spectrum_slice = spectrum[0,beginning_bin:ending_bin]
            abs_intensities_slice[roi] = np.sum(roi_spectrum_slice)

        # Calculate and store peak ratios for each pair current slice
        for roi1, roi2 in itertools.combinations(rois, 2):
            # Calculate the peak ratio for the current 
            ratio_slice = (abs_intensities_slice[roi1] / abs_intensities_slice[roi2] if abs_intensities_slice[roi2] != 0 else np.inf)

            df_param[f"Time_slice_{beginning_bin}_{ending_bin}_peak_ratio_{roi1}_{roi2}"] = ratio_slice

    return df_param 


def time_features_cal(spectrum):
        """
        This function takes a 2D array being the (energy,time) spectrum. 
        It calculates statistical parameters and returns a DataFrame with the calculated parameters.
    
        :param spectrum: 2D numpy array being the spectrum
        :return df_param: DataFrame with statistical parameters
        """

        assert not np.any(spectrum < 0), "spectrum contains negative values!"
        assert np.all(np.mod(spectrum,1) == 0), "spectrum contains non-integer values!"
        assert np.sum(spectrum) != 0, "No events in spectrum"
        
        number_of_bins = np.shape(spectrum)[1]

        # Calculate y_hist by summing along axis 0 of the input 2D array
        # it corresponds to the counts of the timing histogram
        events_time = np.sum(spectrum, axis=0)
        
        y_hist = events_time
        
        # make array for time bin values
        x_hist = np.arange(number_of_bins)
        
        # Generate array, in which all time values occur as often as there were events at that time
        hist_all = np.concatenate([np.array(int(y_hist[i]) * [x_hist[i]]) for i in range(len(x_hist))])
    
        # Calculate probabilities for the occurrence of individual times
        p = y_hist / np.sum(y_hist)
    
        # Create an empty DataFrame for storing the calculated parameters
        df_param = {} 
        
        # Insert all your calculations here
    
        # Arithmetic mean
        mittelwert = np.average(x_hist, weights=y_hist)
        df_param["Mean"] = mittelwert

        # Variance
        varianz = np.average((x_hist - mittelwert)**2, weights=y_hist)
        df_param["Variance"] = varianz
        
        # Standard deviation
        sigma = np.sqrt(varianz)
        df_param["Standard_deviation"] = sigma
        
        # Schiefe
        if sigma == 0.0:
            schiefe = 0.0
        else:
            schiefe = np.sum((x_hist - mittelwert) ** 3.0 * p)/(sigma ** 3.0)
        df_param["Skewness"] = schiefe
        
        # Kurtosis/Wölbung
        if sigma == 0.0:
            kurtosis = 0.0
        else:
            kurtosis = np.sum((x_hist-mittelwert) ** 4.0 * p)/(sigma ** 4.0) - 3.0
        df_param["Kurtosis"] = kurtosis
        
        # Median
        median = np.median(hist_all)
        df_param["Median"] = median
        
        # # Minimum
        # minimum = np.min(x_hist)
        # df_param["Minimum"] = minimum
        
        # 10th percentile
        ten_percentile = np.percentile(hist_all, 10)
        df_param["Percentile_10th"] = ten_percentile
        
        # 90th percentile
        ninety_percentile = np.percentile(hist_all, 90)
        df_param["Percentile_90th"] = ninety_percentile
        
        # # Maximum
        # maximum = np.max(x_hist)
        # df_param["Maximum"] = maximum
        
        # Mode
        m = np.max(y_hist)
        index = np.where(y_hist == m)[0]
        if len(index) == 1:
            modus = x_hist[index][0]
        else:
            werte = x_hist[index]
            index_1 = np.argmin(np.abs(werte-mittelwert))
            modus = werte[index_1]
        df_param["Mode"] = modus
        
        # Interquartil-Abstand 25%-75%
        iqr_2575 = np.percentile(hist_all, q=75) - np.percentile(hist_all, q=25)
        df_param["IQR_25_75"] = iqr_2575
        
        # Interquartil-Abstand 50%-90%
        iqr_5090 = np.percentile(hist_all, q=90) - np.percentile(hist_all, q=50)
        df_param["IQR_50_90"] = iqr_5090
        
        # Interquartil-Abstand 20%-80%
        iqr_2080 = np.percentile(hist_all, q=80) - np.percentile(hist_all, q=20)
        df_param["IQR_20_80"] = iqr_2080
        
        # Interquartil-Abstand 30%-70%
        iqr_3070 = np.percentile(hist_all, q=70) - np.percentile(hist_all, q=30)
        df_param["IQR_30_70"] = iqr_3070
        
        # Interquartil-Abstand 40%-60%
        iqr_4060 = np.percentile(hist_all, q=60) - np.percentile(hist_all, q=40)
        df_param["IQR_40_60"] = iqr_4060
        
        # Interquartil-Abstand 35%-65%
        iqr_3565 = np.percentile(hist_all, q=65) - np.percentile(hist_all, q=35)
        df_param["IQR_35_65"] = iqr_3565
        
        # Mean absolute deviation
        mad = np.mean(np.abs(hist_all - mittelwert))
        df_param["Mean_absolute_deviation"] = mad
        
        # Robust mean absolut deviation 10%-90%
        x_hist_robust = hist_all[(hist_all >= np.percentile(hist_all, 10)) & 
                              (hist_all <= np.percentile(hist_all, 90))]
        rmad = np.mean(np.abs(x_hist_robust - np.mean(x_hist_robust)))
        df_param["Robust_mean_absolute_deviation_10_90"] = rmad
        
        # Robus mean absolut deviation 50%-90%
        x_hist_robust = hist_all[(hist_all >= np.percentile(hist_all, 50)) & 
                              (hist_all <= np.percentile(hist_all, 90))]
        rmad = np.mean(np.abs(x_hist_robust - np.mean(x_hist_robust)))
        df_param["Robust_mean_absolute_deviation_50_90"] = rmad
        
        # Robus mean absolut deviation 20%-80%
        x_hist_robust = hist_all[(hist_all >= np.percentile(hist_all, 20)) & 
                              (hist_all <= np.percentile(hist_all, 80))]
        rmad = np.mean(np.abs(x_hist_robust - np.mean(x_hist_robust)))
        df_param["Robust_mean_absolute_deviation_20_80"] = rmad
        
        # Median absolute deviation
        medad = np.mean(np.abs(hist_all - np.median(hist_all)))
        df_param["Median_absolute_deviation"] = medad
        
        # Variance Koeffizient
        if sigma == 0.0:
            vc = 0.0
        else:
            vc = sigma/mittelwert
        df_param["Coefficient_of_variation"] = vc
        
        # Quartil coefficient of dispersion
        qcd = (np.percentile(hist_all, q=75) - np.percentile(hist_all, q=25))/(
            np.percentile(hist_all, q=75) + np.percentile(hist_all, q=25))
        df_param["Quartile_coefficient_of_dispersion"] = qcd
        
        # Quantil coefficient of dispersion (35-65)
        qcd = (np.percentile(hist_all, q=65) - np.percentile(hist_all, q=35))/(
            np.percentile(hist_all, q=65) + np.percentile(hist_all, q=35))
        df_param["Quantile_coefficient_of_dispersion_35_65"] = qcd
        
        # Entropy
        entropie = -np.sum(p[p > 0.0] * np.log2(p[p > 0.0]))
        df_param["Entropy"] = entropie
        
        # Uniformity
        uni = np.sum(p ** 2.0)
        df_param["Uniformity"] = uni
        
        # Marcatili
        cs = np.cumsum(y_hist)
        cs = cs/np.max(cs)
        cs = np.flip(cs)
        i1 = find_nearest(cs, 0.8)
        i2 = find_nearest(cs, 0.2)
        t = x_hist[i2] - x_hist[i1]
        df_param["T1_to_T2_distance"] = t
        
        # Area under the curve = Total number of events * bin width
        auc = len(hist_all)*(x_hist[1]-x_hist[0])
        df_param["Area_under_the_curve"] = auc
        
        # Position of the trailing edge
        interval_size = 64
        gradients = []
        for i in np.arange(number_of_bins-interval_size):
            gradient_i = y_hist[i+interval_size-1] - y_hist[i]
            gradients.append(gradient_i)
        trailing_edge_beginning = np.argmin(gradients)
        trailing_edge_centre = int(trailing_edge_beginning + interval_size/2)
        df_param["Position_trailing_edge"] = x_hist[trailing_edge_centre]
         
        return df_param

def radiomics_features_cal(spectrum, rois, energy_bin_size, mirp_path, id_global):
    """
    This function calculates the radiomics features for given spectrum data.

    Parameters:
    spectrum (np.array): The input spectrum data.
    rois (list): Regions of interest for the radiomics feature calculation.
    energy_bin_size (int): Size of the energy bin.
    mirp_path (str): The path to the directory for the mirp feature extraction.
    id_global (int): A unique identifier for the current data sample.

    Returns:
    None. The function creates temporary directories, performs feature extraction, and then removes the temporary directories.
    """
    
    # Determine the directory of the current Python script
    python_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # copy mirp dir to experiment. Overwrite if needed.
    shutil.copytree(os.path.join(python_script_dir, "../../data/mirp_feature_extraction"), mirp_path, dirs_exist_ok = True)
    
    id_global_str = str(id_global)
    
    print("Calculate features for {}...".format(id_global))

    # Define base paths
    base_path = os.path.join(mirp_path, "temp_data", "samples", "project_" + id_global_str)
    roi_dir = os.path.join(base_path, id_global_str,"roi")
    spectrum_dir = os.path.join(base_path, id_global_str, "spectrum")

    # Create these directories if they do not exist
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(spectrum_dir, exist_ok=True)

    # Adjust dimensions of spectrum for MIRP and save it
    spectrum_new = np.copy(spectrum.reshape(1, *spectrum.shape))    
    np.save(os.path.join(spectrum_dir, "spectrum.npy"), spectrum_new)

    list_rois = rois.copy()

    # Initialize paths
    config_data_path_load = os.path.join(mirp_path, "raw_config_files", "config_data.xml")
    config_settings_path = os.path.join(mirp_path, "config_settings.xml")

    # Iterate over regions of interest and save arrays for each ROI
    for roi in list_rois:
        feature_dir = os.path.join(mirp_path, "temp_data", "features", roi)
        roi_array = get_roi_array(roi, spectrum, energy_bin_size)

        # Adjust dimensions of spectrum for MIRP and save it
        roi_array = roi_array.reshape(1, np.shape(roi_array)[0], np.shape(roi_array)[1])
        np.save(os.path.join(roi_dir, roi + ".npy"), roi_array)

        # Adjust MIRP configuration for current spectrum and ROI data
        config_data_path = os.path.join(mirp_path, "config_data_{}.xml".format(id_global_str))
        config_data = ET.parse(config_data_path_load)
        root = config_data.getroot()
        for paths in root.iter('paths'):
            paths.find('project_folder').text = base_path
            paths.find('cohort').text = roi
            paths.find('write_folder').text = feature_dir
        
        for data in root.iter('data'):
            data.find('image_folder').text = 'spectrum'
            data.find('roi_folder').text = 'roi'
            data.find('roi_names').text = roi
            
        config_data.write(config_data_path)
    
        # start mirp function for feature extraction
        extract_features(config_data_path, config_settings_path)
    
    # delete config data file
    os.remove(config_data_path)
    
    # Remove temporary directories after feature extraction
    shutil.rmtree(base_path)
    
    print("Features for {} calculated!".format(id_global))
    return

def find_nearest(array, value):
        """
        This function takes an array and a value and returns the index of
        the element closest to the given value.
    
        :param array: Input array
        :param value: Value to find the nearest element to
        :return idx: Index of the element nearest to the given value
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

def process_and_save_df(df, path):
    """Helper function to process a DataFrame and save it to a CSV file.

    Args:
        df (DataFrame): Feature DataFrame to save.
        path (str): The path where to save the CSV file.
    """
    if path is not None and not df.empty:
        # Reorder the columns to have 'id_global' as the first column
        cols = ['id_global'] + [col for col in df.columns if col != 'id_global']
        df = df[cols]
        # Save the DataFrame to a csv file
        df.to_csv(path, sep=";", index=False)
    else:
        print("Nothing is saved!")



def execute_time_feature_cal(data_dir, data_table, time_features_path):
    """
    Calculate time features for a given set of data.

    Args:
        data_dir (str): The path to the directory containing the data.
        time_features_path (str): Path to save the calculated time features.
    """
    
    # time features can be calculated on 1D spectras
    spectrum_type = "1D"
    
    # Initialize list to store features
    feature_table_time = []
    
    # Iterate over each file in the directory, displaying a progress bar
    for id_global in tqdm(data_table["id_global"].values, desc = "Calculating radiomics features..."):
        
        spot_type = data_table.loc[data_table["id_global"] == id_global, "spot_type"].values[0]
        energy = data_table.loc[data_table["id_global"] == id_global, "proton_energy"].values[0]
        
        filename = os.path.basename(data_table.loc[data_table["id_global"] == id_global, "file_path"].values[0])[:-4] + ".npy"

        processed_data_dir = os.path.join(data_dir, f"{spot_type}_{energy}MeV", f"{spectrum_type}", "full")

        # Construct the full file path by joining the directory and the file name
        path = os.path.join(processed_data_dir, filename)

        spectrum = np.array(np.load(path, allow_pickle=True))

        # Calculate time parameters
        feature_dict = time_features_cal(spectrum)
        
        # Append the measurement ID to the feature dictionary
        feature_dict['id_global'] = id_global  
        feature_table_time.append(feature_dict)

    df_feature_table_time = pd.DataFrame(feature_table_time)

    # Process and save the time features dataframe
    process_and_save_df(df_feature_table_time, time_features_path)


def execute_energy_time_feature_cal(data_dir, data_table, energy_time_features_path, rois, energy_bin_size):
    """
    Calculate energy and time features for a given set of data and regions of interest (ROIs).

    Args:
        data_dir (str): The path to the directory containing the data.
        energy_time_features_path (str): Path to save the calculated energy-time features.
        rois (list): List of regions of interest for feature calculation.
        energy_bin_size (float): The energy bin size.
    """

    # energy features can be calculated on 1D spectras
    spectrum_type = "1D"
    
    # Initialize dictionary to store features
    feature_table_energy_time = {}
    
    # Iterate over each file in the directory, displaying a progress bar
    for id_global in tqdm(data_table["id_global"].values, desc = "Calculating energy-time features..."):
        
        spot_type = data_table.loc[data_table["id_global"] == id_global, "spot_type"].values[0]
        energy = data_table.loc[data_table["id_global"] == id_global, "proton_energy"].values[0]
        
        filename = os.path.basename(data_table.loc[data_table["id_global"] == id_global, "file_path"].values[0])[:-4] + ".npy"
     
        for roi in rois:
     
            if roi == "full":
                print("Energy-time features will not be calculated for full spectrum!")
                continue
            
            processed_data_dir = os.path.join(data_dir, f"{spot_type}_{energy}MeV", f"{spectrum_type}", f"{roi}")
     
            # Construct the full file path by joining the directory and the file name
            path = os.path.join(processed_data_dir, filename)
     
            spectrum = np.array(np.load(path, allow_pickle=True))
    
            feature_dict = time_features_cal(spectrum)
            feature_dict = {f'{k}_{roi}': v for k, v in feature_dict.items()}
            
            if id_global not in feature_table_energy_time:
                feature_table_energy_time[id_global] = {}

            feature_table_energy_time[id_global].update(feature_dict)
    
    # Convert the dictionary to a DataFrame
    final_df = pd.DataFrame.from_dict(feature_table_energy_time, orient='index').reset_index()
    final_df.rename(columns={'index': 'id_global'}, inplace=True)
    
    # Process and save the energy-time features dataframe
    process_and_save_df(final_df, energy_time_features_path)


def execute_energy_feature_cal(data_dir, data_table, energy_features_path, rois, energy_bin_size):
    """
    Calculate energy features for a given set of data.

    Args:
        data_dir (str): The path to the directory containing the data.
        energy_features_path (str): Path to save the calculated energy features.
        rois (list): List of regions of interest for feature calculation.
        energy_bin_size (float): energy_bin_size
    """

    # Initialize list to store features
    feature_table_energy = []
    
    # energy features can be calculated on 1D spectras
    spectrum_type = "1D"
    
    # Iterate over each file in the directory, displaying a progress bar
    for id_global in tqdm(data_table["id_global"].values, desc = "Calculating energy features..."):
        
        spot_type = data_table.loc[data_table["id_global"] == id_global, "spot_type"].values[0]
        energy = data_table.loc[data_table["id_global"] == id_global, "proton_energy"].values[0]
        
        filename = os.path.basename(data_table.loc[data_table["id_global"] == id_global, "file_path"].values[0])[:-4] + ".npy"
        
        spectra_rois = {}
        
        for roi in rois:
            processed_data_dir = os.path.join(data_dir, f"{spot_type}_{energy}MeV", f"{spectrum_type}", f"{roi}")
            
            # Construct the full file path by joining the directory and the file name
            path = os.path.join(processed_data_dir, filename)

            # Load the spectrum data from the provided path
            spectrum = np.array(np.load(path, allow_pickle=True))
            spectra_rois[roi] = spectrum


        # Calculate energy parameters
        feature_dict = energy_features_cal(spectra_rois, energy_bin_size)
        
        # Append the measurement ID to the feature dictionary
        feature_dict['id_global'] = id_global  
        feature_table_energy.append(feature_dict)

    df_feature_table_energy = pd.DataFrame(feature_table_energy)

    # Process and save the energy features dataframe
    process_and_save_df(df_feature_table_energy, energy_features_path)

def execute_radiomics_feature_cal(data_dir, data_table,
                                  radiomics_features_path, 
                                  rois, energy_bin_size, 
                                  mirp_path):
    """
    This function is used to calculate the radiomics features of the given data.

    Parameters:
    data_dir (str): The path to the directory containing the data.
    radiomics_features_path (str): The path where the calculated radiomics features should be saved.
    rois (list of strings): Regions of interest for the radiomics feature calculation.
    energy_bin_size (float): energy_bin_size 
    mirp_path (str): The path to the MIRP directory.

    Returns:
    None. The function saves the calculated features to the path specified by `radiomics_features_path`.
    """
    
    # energy features can be calculated on 1D spectras
    spectrum_type = "2D"
    
    # Iterate over each file in the directory, displaying a progress bar
    for id_global in tqdm(data_table["id_global"].values, desc = "Calculating radiomics features..."):
        
        spot_type = data_table.loc[data_table["id_global"] == id_global, "spot_type"].values[0]
        energy = data_table.loc[data_table["id_global"] == id_global, "proton_energy"].values[0]
        
        filename = os.path.basename(data_table.loc[data_table["id_global"] == id_global, "file_path"].values[0])[:-4] + ".npy"

        processed_data_dir = os.path.join(data_dir, f"{spot_type}_{energy}MeV", f"{spectrum_type}", "full")

        # Construct the full file path by joining the directory and the file name
        path = os.path.join(processed_data_dir, filename)

        spectrum = np.array(np.load(path, allow_pickle=True))
    
        # Calculate radiomics features
        radiomics_features_cal(spectrum, rois, energy_bin_size, mirp_path, id_global)

    feature_dir = os.path.join(mirp_path,"temp_data", "features")

    # Get a list of all csv files in all directories
    all_files = glob.glob(os.path.join(feature_dir, '*/*.csv'))
    
    # Create a dictionary to store dataframes
    dfs = {}
    
    # Define columns to exclude
    exclude_columns = ['id_cohort', 'img_data_roi', 'img_data_config', 'img_data_modality', 'img_data_settings_id']
    
    # Process each csv file
    for filename in tqdm(all_files, desc = "Combining features files..."):
        # Load csv file to a DataFrame
        df_param_roi = pd.read_csv(filename, sep = ";")
        
        # Reset index
        df_param_roi.reset_index(drop=True, inplace=True)
        
        roi = df_param_roi["id_cohort"].values[0]
    
        # Remove id_cohort column
        df_param_roi = df_param_roi.drop(exclude_columns, axis=1, errors = 'ignore')
    
        # Append ROI to feature column names, excluding id_subject
        df_param_roi.columns = [f"{col}_{roi}" if col != 'id_subject' else col for col in df_param_roi.columns]
        
        # If subject is not in dfs, add it. If it is, concatenate horizontally
        id_subject = df_param_roi['id_subject'].values[0]
        if id_subject not in dfs.keys():
            dfs[id_subject] = df_param_roi
        else:
            dfs[id_subject] = pd.concat([dfs[id_subject], df_param_roi.drop('id_subject', axis=1)], axis=1)
    
    # Combine all dataframes in the dictionary into one dataframe
    df_param = pd.concat(dfs.values(), ignore_index=True)
    
    # Sort the final dataframe by 'id_subject'
    df_param = df_param.sort_values(by='id_subject')
    
    # Rename id_subject to id_global
    df_param = df_param.rename(columns={'id_subject': 'id_global'})
    
    # Replace '.' with '_' in column names
    df_param.columns = df_param.columns.str.replace('.', '_')
    
    # Save final DataFrame to a csv file
    df_param.to_csv(radiomics_features_path, index=False, sep = ";")

    # remove individual feature data
    shutil.rmtree(feature_dir)

    return


def save_features_plot(data_table, feature_table_path, figures_path, feature_type):
    """
    This function merges a data table and a feature table, creates feature plots for each feature,
    and saves them to the specified directory. Existing plots in the directory are deleted.
    
    Parameters:
    - data_table_path: String, path to the data table csv file
    - feature_table_path: String, path to the feature table csv file
    - figures_path: String, path to the directory where the plots will be saved
    - feature_type: String, the type of feature. Used in the save path.
    """
    
    df_features = pd.read_csv(feature_table_path, sep=";")
    
    df_merge = pd.merge(df_features, data_table[['id_global', 'range_shift', 'proton_energy', 'spot_type']], on='id_global', how='left')
    features = df_merge.columns.drop(['id_global', 'range_shift', 'proton_energy', 'spot_type'])
    
    save_path = os.path.join(figures_path, "features", feature_type)
    
    os.makedirs(save_path, exist_ok=True)
    
    print("Creating and saving feature plots...")
    # Loop over each unique spot type
    for spot_type in np.unique(df_merge["spot_type"].values):
        print(f"Spot type: {spot_type}")
        df_type = df_merge[df_merge["spot_type"] == spot_type]
        
        # Loop over each unique proton energy for the current spot type
        for energy in np.unique(df_type["proton_energy"].values):
            print(f"Proton energy: {energy}")
            df_type_energy = df_type[df_type["proton_energy"] == energy]
               
            for feature in features:
                feature_values = df_type_energy[feature].values.flatten()
                range_shifts = df_type_energy["range_shift"].values.flatten()
                title = spot_type + "_" + str(energy) + "MeV_" + feature
                plot_features(feature_values, range_shifts, title=title, save_path=os.path.join(save_path, title + ".png"))
    
    print(f"Feature plots saved in {save_path}")
    return


    
    
if __name__ == "__main__":
    # Obtain arguments for feature calculation
    parser = feature_calculation_parser("Calculate features")
    args = parser.parse_args()

    data_table = pd.read_csv(args.data_table_path, sep = ";")

    assert args.feature_type in ["energy", "time", "energy-time", "radiomics"], f"{args.feature_type} missspelled or not implemented!"
    
    # Execute calculation of different features if paths are provided
    if args.feature_type == "energy":
        print("Calculate energy features.")
        execute_energy_feature_cal(args.data_dir, data_table, args.output_file_path, args.rois, args.energy_bin)
        print("Energy features calculated!")

    elif args.feature_type == "time":
        print("Calculate time features.")
        execute_time_feature_cal(args.data_dir, data_table, args.output_file_path)
        print("Time features calculated!")

    elif args.feature_type == "energy-time":
        print("Calculate energy-time features.")
        execute_energy_time_feature_cal(args.data_dir, data_table, args.output_file_path, args.rois, args.energy_bin)
        print("Energy-time features calculated!")

    elif args.feature_type == "radiomics":
        print("Calculate radiomics features.")
        execute_radiomics_feature_cal(args.data_dir, data_table, args.output_file_path, args.rois, args.energy_bin, args.mirp_path)
        print("Radiomics features calculated!")
    
    if args.plot_figures == True:
        save_features_plot(data_table, args.output_file_path, args.figures_path, args.feature_type)

        
