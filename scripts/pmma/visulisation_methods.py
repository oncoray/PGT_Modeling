#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:37:28 2023

@author: kiesli21
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style as matplotstyle
matplotstyle.use('bmh')
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy import ndimage
from tqdm import tqdm

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

def aggregate_lut(lut):
    # Initialize a new dictionary for the aggregated data
    aggregated_lut = {}

    # Iterate over the range_shift, energy_row, and globalid to aggregate actual_bg and fitted_bg
    for range_shift in lut.keys():
        if range_shift not in aggregated_lut:
            aggregated_lut[range_shift] = {}
        
        for energy_row in lut[range_shift].keys():
            for globalid, data in lut[range_shift][energy_row].items():
                # If globalid not in the new structure, initialize it
                if globalid not in aggregated_lut[range_shift]:
                    aggregated_lut[range_shift][globalid] = {
                        "irradiation_position": data["irradiation_position"],
                        "actual_bg": 0,  # Initialize sum of actual_bg
                        "fitted_bg": 0   # Initialize sum of fitted_bg
                    }
                
                # Sum the actual_bg and fitted_bg values across energy rows
                aggregated_lut[range_shift][globalid]["actual_bg"] += data["actual_bg"]
                aggregated_lut[range_shift][globalid]["fitted_bg"] += data["fitted_bg"]

    return aggregated_lut

def plot_activation(df_type_energy, bg_lut, bg_model_parameters, save_path_dir):
    print("Plot activation function")

    num_rows = np.max(bg_lut["energy_row"].values) + 1

    number_of_range_shifts = len(bg_model_parameters)
    cmap = plt.cm.get_cmap('inferno')
    
    for energy_row in tqdm(range(num_rows), desc="Plot activation function"):
        plt.figure(figsize=(10, 6))  # Set the figure size for clarity

        max_position = 0
        # Get a color for each range_shift from the color cycle
        for idx, (range_shift,params) in enumerate(bg_model_parameters.items()):

                color_index = idx / number_of_range_shifts
                color = cmap(color_index)
                coef, intercept = bg_model_parameters[range_shift][energy_row]

                for _, data in bg_lut.loc[(bg_lut["range_shift"] == range_shift) * (bg_lut["energy_row"] == energy_row),:].iterrows():
                    position = data["irradiation_position"] 
                    actual_bg = data["actual_bg"]
                    
                    plt.scatter(position, actual_bg, marker='o', s=25, color=color, alpha = 0.75)

                    if position > max_position:
                        max_position = position

                # Create x values for the regression line
                x_vals = np.arange(0, max_position+1, 1)  # Adjusted for detailed plotting
                y_vals = intercept + coef * x_vals

                # Plot regression line with a dashed style
                plt.plot(x_vals, y_vals, label=f'Linear regression (Range Shift {range_shift})', linestyle='--', color=color)

        plt.title(f"Background Estimation for Energy Row {energy_row}")
        plt.xlabel('Irradiation position')
        plt.ylabel('Mean Background / counts per bin')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(f"{save_path_dir}/bg_estimation_energy_row_{energy_row}.png")
        plt.close()  # Close the plot to free up memory
        
    # # save the aggregated bg of all energy rows
    
    # plt.figure(figsize=(10, 6))  # Set the figure size for clarity
    # positions = []
    # fitted_bgs = []
    # # Get a color for each range_shift from the color cycle
    # for idx, (range_shift,params) in enumerate(bg_model_parameters.items()):

    #         color_index = idx / number_of_range_shifts
    #         color = cmap(color_index)
    #         for globalid, data in bg_lut.groupby("globalid"):
    #             position = np.unqiue(data["irradiation_position"].values)[0]
    #             actual_bg = np.sum(data["actual_bg"].values)
    #             fitted_bg = np.sum(data["fitted_bg"].values)
                
    #             positions.append(position)
    #             fitted_bgs.append(fitted_bg)
    #             plt.scatter(position, actual_bg, marker='o', s=25, color=color, alpha = 0.75)

    #         # Plot regression line with a dashed style
    #         plt.plot(positions, fitted_bgs, label=f'Linear regression (Range Shift {range_shift})', linestyle='--', color=color)

    # plt.title("Background Estimation for Full Spectrum")
    # plt.xlabel('Irradiation position')
    # plt.ylabel('Aggregated Mean Background / counts per bin')
    # plt.legend()
    # plt.grid(True)

    # # Save the plot
    # plt.savefig(f"{save_path_dir}/bg_estimation_energy_row_full.png")
    # plt.close()  # Close the plot to free up memory
    

def plot_performance_heatmaps(summary_df, output_path):
    """
    Plots heatmaps of RMSE and R2 for the validation cohort across different spot data sets.

    Parameters:
    - summary_df (DataFrame): Summary DataFrame containing performance metrics.
    - output_path (str): Base path where the heatmaps will be saved.
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    # if just training is in cohort table, then its cross validation
    if len(np.unique(summary_df['cohort'].values)) == 1:
        validation_df = summary_df[summary_df['cv_data_set'] == 'validation']
    else:
        # Filter for validation cohort
        validation_df = summary_df[summary_df['cohort'] == 'validation']

    for (spot_type, proton_energy), df_subset in validation_df.groupby(['spot_type', 'proton_energy']):
        # Create pivot tables for RMSE and R2
        pivot_rmse = df_subset.pivot_table(index=['feature_type', 'feature_selection_method'], columns='model_learner', values='RMSE')
        pivot_r2 = df_subset.pivot_table(index=['feature_type', 'feature_selection_method'], columns='model_learner', values='R2')

        # Setup the matplotlib figure with two subplots (side by side)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
        
        fig.suptitle(f'Spot Type {spot_type}, Proton Energy {proton_energy}')
        
        # Plotting RMSE heatmap
        sns.heatmap(pivot_rmse, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=axes[0])
        axes[0].set_title('RMSE')
        axes[0].set_ylabel('Feature Type / Feature Selection Method')
        axes[0].set_xlabel('Model Learner')

        # Plotting R2 heatmap
        sns.heatmap(pivot_r2, annot=True, fmt=".2f", cmap='coolwarm_r', cbar=True, ax=axes[1])
        axes[1].set_title('R2')
        axes[1].set_xlabel('Model Learner')
        # The y-axis label is shared across subplots, already set by the first subplot
        
        # Adjust layout
        plt.tight_layout()

        # Save the combined figure
        plt.savefig(os.path.join(f"{output_path}", f"heatmap_{spot_type}_{proton_energy}.png"))
        plt.close()

    return



def save_2Dspectrum(spectrum, time_bin_size, energy_bin_size, title, save_path):
    """
    Function to create and save a 2D plot of a spectrum.
    
    Parameters:
    spectrum : ndarray
        The 2D spectrum to be plotted.
    time_bin_size : float
        The time bin size for the x-axis.
    energy_bin_size : float
        The energy bin size for the y-axis.
    title : str
        The title of the plot.
    save_path : str
        The directory where the plot should be saved.
    dpi : int, optional
        The resolution of the saved image.
    """
    
    max_time = time_bin_size * np.shape(spectrum)[1]
    max_energy = energy_bin_size * np.shape(spectrum)[0]
    shape = np.shape(spectrum)
    
    # Check if spectrum is 2D
    assert len(shape) == 2, "Spectrum is not 2D!"

    # create an array for energy and time axis
    energy = np.linspace(0, max_energy, shape[0], endpoint=False)
    time = np.linspace(0, max_time, shape[1], endpoint=False)
    
    # create a new figure
    fig, ax = plt.subplots()

    # create 2D plot using pcolormesh
    im = ax.pcolormesh(time, energy, spectrum, shading='auto')

    # set the x, y labels and title for the plot
    ax.set_xlabel('Time / ns')
    ax.set_ylabel('Energy / MeV')
    ax.set_title(title)
    
    # add a colorbar to the plot
    fig.colorbar(im, label='number of detected events')
    
    # save the figure
    plt.savefig(os.path.join(save_path, title))
    plt.close()

def save_timespectrum(spectrum, time_bin_size, title, save_path):
    """
    Function to create and save a time spectrum plot.
    
    Parameters:
    spectrum : ndarray
        The 2D spectrum to be plotted.
    time_bin_size : float
        The mtime_bin_size for the x-axis.
    title : str
        The title of the plot.
    save_path : str
        The directory where the plot should be saved.
    dpi : int, optional
        The resolution of the saved image.
    """
    
    # Get the shape of the spectrum
    shape = np.shape(spectrum)
    
    max_time = time_bin_size * shape[1]
    
    # Compute time spectrum by summing over energy axis
    time_spectrum = np.sum(spectrum, axis=0)
    
    # create an array for time axis
    time = np.linspace(0, max_time, shape[1], endpoint=False)
    
    # create a new figure
    fig, ax = plt.subplots()

    # set the x, y labels and title for the plot
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Events per bin')
    ax.set_title(title)

    # plot the time spectrum
    ax.plot(time, time_spectrum)

    # save the figure
    plt.savefig(os.path.join(save_path, title))
    plt.close()

def save_timesumspectrum(spectrum_dict, time_bin_size, title, g_median = 7, g_gauss = 11, save_path=None):
    """
    Function to create and save a plot of multiple time spectra.
    
    Parameters:
    spectrum_dict : dict
        A dictionary of 2D spectra to be plotted.
    time_bin_size : float
        The time_bin_size for the x-axis.
    title : str
        The title of the plot.
    save_path : str
        The directory where the plot should be saved.
    """
    
    # define the colormap
    cmap = plt.cm.get_cmap('inferno')
    
    # create a new figure
    fig, ax = plt.subplots()

    # set the x, y labels and title for the plot
    ax.set_xlabel('Time / ns')
    ax.set_ylabel('Events per bin')
    ax.set_title(title)

    # Separate range shifts into strings and numbers
    str_keys = [key for key in spectrum_dict.keys() if isinstance(key, str)]
    num_keys = [key for key in spectrum_dict.keys() if isinstance(key, str) == False]
    
    # Sort them
    str_keys.sort()
    num_keys.sort()
    
    # Combine them back into single list
    sorted_keys = str_keys + num_keys
    
    total_keys = len(sorted_keys)

    # plot all the spectra in the input dictionary
    for i, key in enumerate(sorted_keys):
        if len(np.shape(spectrum_dict[key])) == 1:
            spectrum = spectrum_dict[key]
        else:
            spectrum = np.sum(spectrum_dict[key], axis=0)
            
        max_time = np.shape(spectrum)[0] * time_bin_size
            
        # create an array for time axis
        time = np.linspace(0, max_time, np.shape(spectrum)[0], endpoint=False)

        # label for the plot
        plot_label = str(key)

        # get normalized color index
        color_index = i / total_keys
        
        # Apply median filter
        smoothed_spectrum = ndimage.median_filter(spectrum, g_median, mode="wrap")
        # Apply gaussian filter
        smoothed_spectrum = ndimage.gaussian_filter(spectrum, g_gauss, mode="wrap")

        # plot the time spectrum
        ax.plot(time, spectrum, label=plot_label, ms=0, color=cmap(color_index), alpha = 0.3)
        
        ax.plot(time, smoothed_spectrum, color=cmap(color_index), alpha=0.8, linestyle='-', ms = 0)

    # add a legend to the plot
    ax.legend()

    if save_path is not None:
         # save the figure
         plt.savefig(os.path.join(save_path, title))
    plt.close()
    
def save_energysumspectrum(spectrum_dict, energy_bin_size, title, save_path):
    """
    Function to create and save a plot of multiple time spectra.
    
    Parameters:
    spectrum_dict : dict
        A dictionary of 2D spectra to be plotted.
    energy_bin_size : float
        The energy bin size for the x-axis.
    title : str
        The title of the plot.
    save_path : str
        The directory where the plot should be saved.
    """
    
    
    # define the colormap
    cmap = plt.cm.get_cmap('inferno')
    
    # create a new figure
    fig, ax = plt.subplots()

    # set the x, y labels and title for the plot
    ax.set_xlabel('Energy / MeV')
    ax.set_ylabel('Events per bin')
    ax.set_title(title)

    # Separate range shifts into strings and numbers
    str_keys = [key for key in spectrum_dict.keys() if isinstance(key, str)]
    num_keys = [key for key in spectrum_dict.keys() if isinstance(key, str) == False]
    
    # Sort them
    str_keys.sort()
    num_keys.sort()
    
    # Combine them back into single list
    sorted_keys = str_keys + num_keys
    
    total_keys = len(sorted_keys)

    # plot all the spectra in the input dictionary
    for i, key in enumerate(sorted_keys):
        if len(np.shape(spectrum_dict[key])) == 1:
            spectrum = spectrum_dict[key]
        else:
            spectrum = np.sum(spectrum_dict[key], axis=1)
        
        max_energy = np.shape(spectrum)[0] * energy_bin_size
        # create an array for energy axis
        energy = np.linspace(0, max_energy, np.shape(spectrum)[0], endpoint=False)

        # label for the plot
        plot_label = str(key)

        # get normalized color index
        color_index = i / total_keys

        # plot the time spectrum
        ax.plot(energy, spectrum, label=plot_label, ms=0, color=cmap(color_index), alpha = 0.7)

    # add a legend to the plot
    ax.legend()

    # save the figure
    plt.savefig(os.path.join(save_path, title))
    plt.close()
    
def save_2Dsumspectrum(spectrum_dict, time_bin_size, energy_bin_size, title, save_path):
    """
    Function to create and save a plot of multiple 2D spectra.
    
    Parameters:
    spectrum_dict : dict
        A dictionary of 2D spectra to be plotted.
    time_bin_size : float
        The time bin size for the x-axis.
    energy_bin_size: float
        The energy bin size for the y-axis.
    title : str
        The title of the plot.
    save_path : str
        The directory where the plot should be saved.
    """
    
    
    # plot all the spectra in the input dictionary
    for key in spectrum_dict.keys():
        spectrum = spectrum_dict[key]
        
        save_2Dspectrum(spectrum, time_bin_size, energy_bin_size, title + "_" + str(key), save_path)
        

def plot_features(feature_values, range_shifts, title, save_path):
    """
    Plot a boxplot of feature values grouped by their numeric range shifts.
    Raises an error if range shifts cannot be converted to numeric values.

    Parameters:
    feature_values (list): The feature values to be plotted.
    range_shifts (list): Corresponding range shift values for the feature values. 
                         All range shifts must be convertible to numeric values.
    title (str): The title of the plot.
    save_path (str): The directory where the plot should be saved.
    dpi (int): The resolution of the saved image.

    The function does not return anything. A plot is displayed as output.
    """

    # Create a DataFrame from the feature values and range shifts
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Range Shift': range_shifts
    })

    # Convert Range Shifts to numeric and assert all are converted
    df['Range Shift'] = pd.to_numeric(df['Range Shift'], errors='coerce')
    assert not df['Range Shift'].isnull().any(), "Non-numeric range shifts found."

    # Sort numeric Range Shifts in ascending order
    df.sort_values(by='Range Shift', ascending=True, inplace=True)

    # Prepare boxplot data groups based on unique range shifts
    unique_shifts = sorted(df['Range Shift'].unique())
    data_groups = [df[df['Range Shift'] == shift]['Feature Value'].dropna() for shift in unique_shifts]
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Generate boxplots
    box = ax.boxplot(data_groups, positions=unique_shifts, widths=2, patch_artist=True, notch=False)

    # Customize boxplot appearance
    for patch in box['boxes']:
        patch.set_facecolor('grey')  # Set boxplot color
    
    # Provide additional plot decorations
    ax.set_title(title)
    ax.set_xlabel('Range Shift')
    ax.set_ylabel('Feature Value')

    # Save the figure
    plt.savefig(save_path)
    plt.close()
    
    
def plot_predicted_vs_actual_range_shift(data, title=None, save_path=None, cohort='training'):
    """
    Generates boxplots comparing predicted range shifts to actual range shifts for each spot type within a specified cohort.
    
    Parameters:
    - data (DataFrame): The dataset containing 'cohort', 'spot_type', 'range_shift', and 'predicted_range_shift' columns.
    - title (str, optional): The base title for each subplot. Additional details will be appended to this title.
    - save_path (str, optional): Path to save the generated plot. If None, the plot is not saved.
    - cohort (str): The cohort within the data to focus on. Defaults to 'training'.
    
    The function creates a subplot for each unique 'spot_type' within the specified 'cohort', plotting boxplots that
    compare the distribution of 'predicted_range_shift' values against the actual 'range_shift' values. Each subplot
    includes a black dashed line representing the optimal prediction (where predicted equals actual) and displays the
    Root Mean Squared Error (RMSE) and R^2 score to quantify prediction accuracy.
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Filter data based on the specified cohort
    data_cohort = data[data["cohort"] == cohort]
    
    # Extract unique spot types within the cohort
    spot_types = np.unique(data_cohort["spot_type"].values)

    # Setup the figure for plotting
    fig, axs = plt.subplots(len(spot_types), figsize=(18, 8 * len(spot_types)), constrained_layout=True, sharex=True, sharey=True)

    # Ensure axs is iterable for consistent handling
    if len(spot_types) == 1:
        axs = np.array([axs])
    
    for ax, spot_type in zip(axs, spot_types):
        # Filter data for the current spot type
        subset = data_cohort[data_cohort["spot_type"] == spot_type]
        
        # Prepare boxplot data groups based on unique range shifts
        unique_shifts = sorted(subset['range_shift'].unique())
        data_groups = [subset[subset['range_shift'] == shift]['predicted_range_shift'].dropna() for shift in unique_shifts]
        
        # Generate boxplots
        box = ax.boxplot(data_groups, positions=unique_shifts, widths=2, patch_artist=True, notch=False)

        # Customize boxplot appearance
        for patch in box['boxes']:
            patch.set_facecolor('grey')  # Set boxplot color

        # Calculate and display RMSE and R^2 score, handling cases with NaN values
        valid_subset = subset.dropna(subset=['predicted_range_shift'])
        rmse = np.sqrt(mean_squared_error(valid_subset['range_shift'], valid_subset['predicted_range_shift']))
        r2 = r2_score(valid_subset['range_shift'], valid_subset['predicted_range_shift'])

        # Add a black dashed line for optimal prediction
        ax.plot([0, max(unique_shifts, default=0)], [0, max(unique_shifts, default=0)], '--', color='black', label='Optimal Prediction')

        # Set subplot title, labels, and legend
        plot_title = f'{title + " | " if title else ""}Spot Type: {spot_type} | RMSE: {rmse:.2f}, R2: {r2:.2f}'
        ax.set_title(plot_title)
        ax.set_xlabel('Range Shift')
        ax.set_ylabel('Predicted Range Shift')
        ax.legend()
        ax.grid(True)

    # Optionally save the plot to a file
    if save_path:
        plt.savefig(save_path)
    
    plt.close()
    
def plot_final_signature_features(ranked_feature_table, final_signature, save_path, feature_type, feature_selection_method, model_learner):
    """
    Plots the features in the final signature as boxplots, with separate plots for training and validation cohorts.

    Parameters:
    - ranked_feature_table (DataFrame): DataFrame containing the feature data.
    - final_signature (list): List of features in the final signature.
    - save_path (str): The directory where the plot should be saved.
    - feature_type (str): The type of feature.
    - feature_selection_method (str): The feature selection method used.
    - model_learner (str): The model learning algorithm used.
    - dpi (int): The resolution of the saved image.
    """
    num_features = len(final_signature)
    num_rows = int(np.sqrt(num_features))
    num_cols = int(np.ceil(num_features / num_rows))

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))

    for i, feature in enumerate(final_signature):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(x='range_shift', y=feature, hue='cohort', hue_order = ["training", "validation"], data=ranked_feature_table, palette='Set2', order = np.arange(0,25,5))
        plt.title(feature)
        plt.xlabel('Range Shift')
        plt.ylabel('Feature Value')

        # Adding a legend to each subplot
        if i == 0:  # Adding the legend only to the first subplot for clarity
            ax.legend(title='Cohort', loc='upper right')
        else:
            ax.legend([],[], frameon=False)

    plt.suptitle(f'Feature Type: {feature_type}; Selection Method: {feature_selection_method}; Model Learner: {model_learner}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


    
    



        
        
        
        
        
        
        
        
        
        
        