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
import math
import logging

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

def plot_bg_data(bg_region_df, figures_path):
    """
    Plot background data and save the figures.

    Parameters:
    - bg_region_df: DataFrame containing background region data.
    - figures_path: Directory to save the figures.
    """
    os.makedirs(figures_path, exist_ok=True)

    # Plot mean_bg vs. triggertime
    plt.figure()
    for layer, group in bg_region_df.groupby('Layer'):
        plt.scatter(group['triggertime'], group['mean_bg'], label=f'Layer {layer}')
    plt.xlabel('Triggertime')
    plt.ylabel('Mean Background')
    plt.title('Mean Background vs. Triggertime')
    plt.legend()
    plt.savefig(os.path.join(figures_path, 'mean_bg_vs_triggertime.png'))
    plt.close()

    # Plot histogram of SNR
    plt.figure()
    plt.hist(bg_region_df['SNR'], bins=50, edgecolor='black')
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.title('Histogram of SNR')
    plt.savefig(os.path.join(figures_path, 'snr_histogram.png'))
    plt.close()

    # Plot histogram of mean_relative_bg
    plt.figure()
    plt.hist(bg_region_df['mean_relative_bg'], bins=50, edgecolor='black')
    plt.xlabel('Mean Relative Background')
    plt.ylabel('Count')
    plt.title('Histogram of Mean Relative Background')
    plt.savefig(os.path.join(figures_path, 'mean_relative_bg_histogram.png'))
    plt.close()

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
    Plots heatmaps of RMSE and R2 for the validation cohort across different datasets.

    Parameters:
    - summary_df (DataFrame): Summary DataFrame containing performance metrics.
    - output_path (str): Base path where the heatmaps will be saved.
    """

    os.makedirs(output_path, exist_ok=True)

    # Determine if cross-validation or external validation is used
    if len(summary_df['cohort'].unique()) == 1:
        validation_df = summary_df[summary_df['cv_data_set'] == 'validation']
    else:
        validation_df = summary_df[summary_df['cohort'] == 'validation']

    # Group by the new variables
    grouping_vars = ['nose_orientation', 'proton_energy', 'mu', 'range_shift_type']
    for group_values, df_subset in validation_df.groupby(grouping_vars):
        nose_orientation, proton_energy, mu, range_shift_type = group_values

        # Create pivot tables for RMSE and R2
        pivot_rmse = df_subset.pivot_table(
            index=['feature_type', 'feature_selection_method'],
            columns='model_learner',
            values='RMSE'
        )
        pivot_r2 = df_subset.pivot_table(
            index=['feature_type', 'feature_selection_method'],
            columns='model_learner',
            values='R2'
        )

        # Setup the matplotlib figure with two subplots (side by side)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)

        fig.suptitle(
            f'Nose Orientation: {nose_orientation}, Proton Energy: {proton_energy}, '
            f'Mu: {mu}, Range Shift Type: {range_shift_type}'
        )

        # Plotting RMSE heatmap
        sns.heatmap(
            pivot_rmse, annot=True, fmt=".2f", cmap='winter_r', cbar=True, ax=axes[0], vmin=0, vmax=5, cbar_kws={'label': 'RMSE / mm'}
        )

        axes[0].set_title('RMSE')
        axes[0].set_ylabel('Feature Type / Feature Selection Method')
        axes[0].set_xlabel('Model Learner')

        # Plotting R2 heatmap
        sns.heatmap(
            pivot_r2, annot=True, fmt=".2f", cmap='winter_r', cbar=True, ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'R2'}
        )
        axes[1].set_title('R2')
        axes[1].set_xlabel('Model Learner')

        # Adjust layout
        plt.tight_layout()

        # Save the combined figure with a descriptive filename
        filename = (
            f"heatmap_nose-{nose_orientation}_energy-{proton_energy}_mu-{mu}_"
            f"shift-{range_shift_type}.png"
        )
        plt.savefig(os.path.join(output_path, filename))
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



def plot_ref_features(feature_values, layers, title, save_path):
    """
    Plot a boxplot of feature values grouped by their range shifts, handling both 'ref' and numeric values.
    The boxplots will be ordered starting with 'ref' followed by numeric range shifts in ascending order.

    Parameters:
    feature_values (list or array-like): The feature values to be plotted.
    range_shifts (list or array-like): Corresponding range shift values for the feature values.
                                       Can contain 'ref' and numeric values.
    title (str): The title of the plot.
    save_path (str): The path where the plot should be saved.

    The function does not return anything. It saves the plot to the specified path.
    """

    # Create a DataFrame from the feature values and range shifts
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Layers': layers
    })

    # Get the unique range shifts and sort them using the custom key
    unique_shifts = df['Layers'].unique()
    unique_shifts_sorted = sorted(unique_shifts)

    # Prepare data groups for boxplot
    data_groups = [df[df['Layers'] == shift]['Feature Value'].dropna() for shift in unique_shifts_sorted]

    # Create positions for boxplots
    positions = list(range(len(unique_shifts_sorted)))

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True, notch=False)

    # Customize boxplot appearance
    for patch in box['boxes']:
        patch.set_facecolor('grey')  # Set boxplot color

    # Set x-ticks to show 'ref' and numeric range shifts
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_shifts_sorted)

    # Provide additional plot decorations
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Feature Value')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close()


def plot_features(feature_values, range_shifts, title, save_path):
    """
    Plot a boxplot of feature values grouped by their range shifts, handling both 'ref' and numeric values.
    The boxplots will be ordered starting with 'ref' followed by numeric range shifts in ascending order.

    Parameters:
    feature_values (list or array-like): The feature values to be plotted.
    range_shifts (list or array-like): Corresponding range shift values for the feature values.
                                       Can contain 'ref' and numeric values.
    title (str): The title of the plot.
    save_path (str): The path where the plot should be saved.

    The function does not return anything. It saves the plot to the specified path.
    """

    # Create a DataFrame from the feature values and range shifts
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Range Shift': range_shifts
    })

    # Convert numeric range shifts to float, keep 'ref' as is
    def convert_range_shift(x):
        try:
            return float(x)
        except ValueError:
            return x  # Keep 'ref' as is

    df['Range Shift Converted'] = df['Range Shift'].apply(convert_range_shift)

    # Create a custom sort key that orders 'ref' first, followed by numeric values in ascending order
    def sort_key(x):
        if x == 'ref':
            return -1  # 'ref' will come first
        else:
            return float(x)

    # Get the unique range shifts and sort them using the custom key
    unique_shifts = df['Range Shift Converted'].unique()
    unique_shifts_sorted = sorted(unique_shifts, key=sort_key)

    # Prepare data groups for boxplot
    data_groups = [df[df['Range Shift Converted'] == shift]['Feature Value'].dropna() for shift in unique_shifts_sorted]

    # Create positions for boxplots
    positions = list(range(len(unique_shifts_sorted)))

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True, notch=False)

    # Customize boxplot appearance
    for patch in box['boxes']:
        patch.set_facecolor('grey')  # Set boxplot color

    # Set x-ticks to show 'ref' and numeric range shifts
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_shifts_sorted)

    # Provide additional plot decorations
    ax.set_title(title)
    ax.set_xlabel('Range Shift')
    ax.set_ylabel('Feature Value')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close()

def plot_predicted_vs_actual_range_shift(data, title=None, save_path=None, cohort='training', col_name_data='cohort'):
    """
    Generates a multi-subplot figure comparing predicted range shifts to actual range shifts for each unique combination
    of nose_orientation, proton_energy, mu, detector, and range_shift_type within a specified cohort. In addition, if there
    are multiple subplots due to grouping, a combined subplot (without grouping) is added as the first subplot.

    Parameters:
    - data (DataFrame): The dataset containing 'cohort', 'nose_orientation', 'proton_energy', 'mu', 'detector',
      'range_shift', and 'predicted_range_shift' columns.
    - title (str, optional): The title for the entire figure.
    - save_path (str, optional): Full path to save the generated plot. If None, the plot is not saved.
    - cohort (str): The cohort within the data to focus on. Defaults to 'training'.

    The function creates a figure with subplots arranged in a grid. If more than one subgroup is present,
    the first subplot presents a combined boxplot of all data (filtered by the cohort). Each subsequent subplot
    corresponds to a unique combination of 'nose_orientation', 'proton_energy', 'mu', 'detector', and 'range_shift_type'
    and includes a boxplot of predicted versus actual range shifts. Each subplot also displays a black dashed line representing
    the optimal prediction (where predicted equals actual) along with the computed Root Mean Squared Error (RMSE) and R^2 score.
    """

    # Filter data based on the specified cohort
    data_cohort = data[data[col_name_data] == cohort]

    # Define the grouping columns for detailed subplots
    group_columns = ['nose_orientation', 'proton_energy', 'mu', 'detector', "range_shift_type"]

    # Group data by the specified columns
    grouped = data_cohort.groupby(group_columns)
    num_groups = len(grouped)

    if num_groups == 0:
        print("No data available for the specified cohort and grouping.")
        return

    # Determine whether to add a combined subplot (only if there are multiple subplots)
    add_combined = num_groups > 1
    total_plots = num_groups + 1 if add_combined else num_groups

    # Determine the layout of subplots (up to 3 per row)
    num_cols = min(3, total_plots)
    num_rows = math.ceil(total_plots / num_cols)

    # Create the figure and axes
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    axs = np.array(axs).reshape(-1)  # Flatten in case of multiple rows/columns

    plot_index = 0  # Index to track current axis

    # If multiple subplots are generated, plot an additional combined subplot first
    if add_combined:
        ax = axs[plot_index]
        plot_index += 1

        # Prepare boxplot data for the combined dataset
        unique_shifts = sorted(data_cohort['range_shift'].unique())
        data_groups = [
            data_cohort[data_cohort['range_shift'] == shift]['predicted_range_shift'].dropna()
            for shift in unique_shifts
        ]

        # Generate the boxplot for the combined data
        box = ax.boxplot(data_groups, positions=unique_shifts, widths=2, patch_artist=True,
                         notch=False, showmeans=True)
        for patch in box['boxes']:
            patch.set_facecolor('grey')

        # Calculate RMSE and R^2 for the combined dataset
        valid_data = data_cohort.dropna(subset=['predicted_range_shift'])
        if not valid_data.empty:
            rmse = np.sqrt(mean_squared_error(valid_data['range_shift'], valid_data['predicted_range_shift']))
            r2 = r2_score(valid_data['range_shift'], valid_data['predicted_range_shift'])
        else:
            rmse = np.nan
            r2 = np.nan

        # Add a black dashed line representing the optimal prediction (predicted equals actual)
        if unique_shifts:
            min_shift = min(unique_shifts)
            max_shift = max(unique_shifts)
        else:
            min_shift, max_shift = 0, 0
        ax.plot([min_shift, max_shift], [min_shift, max_shift], '--', color='black', label='Optimal Prediction')

        # Set subplot title and labels for the combined plot
        ax.set_title(f'Combined Data\nRMSE: {rmse:.2f}, R2: {r2:.2f}')
        ax.set_xlabel('Actual Range Shift')
        ax.set_ylabel('Predicted Range Shift')
        ax.legend()
        ax.grid(True)

    # Iterate over each group and create corresponding subplots
    for group_keys, group_data in grouped:
        ax = axs[plot_index]
        plot_index += 1

        nose_orientation, proton_energy, mu, detector, range_shift_type = group_keys

        # Prepare boxplot data groups for the current subgroup
        unique_shifts = sorted(group_data['range_shift'].unique())
        data_groups = [
            group_data[group_data['range_shift'] == shift]['predicted_range_shift'].dropna()
            for shift in unique_shifts
        ]

        # Generate the boxplot for the current group
        box = ax.boxplot(data_groups, positions=unique_shifts, widths=2, patch_artist=True,
                         notch=False, showmeans=True)
        for patch in box['boxes']:
            patch.set_facecolor('grey')

        # Calculate RMSE and R^2 for the subgroup
        valid_data = group_data.dropna(subset=['predicted_range_shift'])
        if not valid_data.empty:
            rmse = np.sqrt(mean_squared_error(valid_data['range_shift'], valid_data['predicted_range_shift']))
            r2 = r2_score(valid_data['range_shift'], valid_data['predicted_range_shift'])
        else:
            rmse = np.nan
            r2 = np.nan

        # Add a black dashed line for optimal prediction
        if unique_shifts:
            min_shift = min(unique_shifts)
            max_shift = max(unique_shifts)
        else:
            min_shift, max_shift = 0, 0
        ax.plot([min_shift, max_shift], [min_shift, max_shift], '--', color='black', label='Optimal Prediction')

        # Set subplot title, labels, and legend for the subgroup
        plot_title = (f'Nose: {nose_orientation}, Energy: {proton_energy} MeV,\n'
                      f'MU: {mu}, Detector: {detector}, RST: {range_shift_type}\n'
                      f'RMSE: {rmse:.2f}, R2: {r2:.2f}')
        ax.set_title(plot_title)
        ax.set_xlabel('Actual Range Shift')
        ax.set_ylabel('Predicted Range Shift')
        ax.legend()
        ax.grid(True)

    # Remove any unused subplots
    for idx in range(plot_index, len(axs)):
        fig.delaxes(axs[idx])

    # Adjust layout and optionally set the overall title
    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Optionally save the plot to a file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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

    plt.figure(figsize=(6 * num_cols, 6 * num_rows))

    for i, feature in enumerate(final_signature):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(x='range_shift', y=feature, hue='cohort', hue_order = ["training", "validation"], data=ranked_feature_table, palette='Set2')
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


def save_features_plot(data_table_path, feature_table_path, save_path, aggregation_column = "file_path_proc"):
    """
    This function merges a data table and a feature table, creates feature plots for each feature,
    and saves them to the specified directory.

    Parameters:
    - data_table: DataFrame containing the data table
    - feature_table_path: String, path to the feature table csv file
    - save_path: String, path to the directory where the plots will be saved
    """

    df_features = pd.read_csv(feature_table_path, sep=";")

    data_table = pd.read_csv(data_table_path, sep=";")

    # Merge the data table and features on 'id_global'
    df_merge = pd.merge(df_features, data_table, on=aggregation_column, how='left')

    df_merge = df_merge[df_merge["range_shift_type"] == "grs"]

    # Get all feature columns by excluding 'id_global'
    features = [col for col in df_features.columns if col != aggregation_column]

    os.makedirs(save_path, exist_ok=True)

    logging.info("Creating and saving feature plots...")

    # Group by 'cohort', 'proton_energy', 'mu', and 'detector'
    group_cols = ['cohort', 'proton_energy', 'mu', 'detector']

    grouped = df_merge.groupby(group_cols)

    for group_keys, group_df in grouped:
        # group_keys is a tuple: (cohort, energy, mu, detector)
        cohort, energy, mu, detector = group_keys
        logging.info(f"Cohort: {cohort}, Energy: {energy}, MU: {mu}, Detector: {detector}")

        for feature in features:
            feature_values = group_df[feature].values.flatten()
            title = f"{cohort}_{energy}MeV_{mu}MU_det{detector}_{feature}"
            if cohort.startswith("ref"):
                layers = group_df["layer"].values.flatten()
                plot_ref_features(feature_values, layers, title=title, save_path=os.path.join(save_path, title + ".png"))
            else:
                range_shifts = group_df["range_shift"].values.flatten()

                plot_features(feature_values, range_shifts, title=title, save_path=os.path.join(save_path, title + ".png"))

    logging.info(f"Feature plots saved in {save_path}")





def plot_loss_curve(train_hist, eval_hist, save_path, title='Training vs. Eval Loss',
                    train_hist_smoothed=None, eval_hist_smoothed=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_hist) + 1)

    plt.plot(epochs, train_hist, marker='o', linewidth=1.5, label='Train Loss', alpha=0.7)
    plt.plot(epochs, eval_hist, marker='o', linewidth=1.5, label='Test Loss', alpha=0.7)

    if train_hist_smoothed is not None:
        plt.plot(epochs, train_hist_smoothed, linewidth=2.0, linestyle='-', label='Train Loss (smoothed)')
    if eval_hist_smoothed is not None:
        plt.plot(epochs, eval_hist_smoothed, linewidth=2.0, linestyle='-', label='Test Loss (smoothed)')

    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(title)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()




