import numpy as np

def get_roi_array(roi, spectrum, energy_bin_size):
    """
    This function creates an array that represents the region of interest (ROI) within the given spectrum data. 
    The ROI is specified by the `roi` argument. It can be "10B", "11C", "16O", or "511keV".
    The ROI array is the same shape as the input spectrum data, and it contains ones where the energy bins fall 
    within the range specified for the ROI, and zeros elsewhere.

    Parameters:
    roi (str): Specifies the region of interest. Can be "10B", "11C", "16O", "12C+11B", or "511keV".
    spectrum (np.array): The spectrum data.
    energy_bin_size (float): Energy bin size.

    Returns:
    np.array: The ROI array, of the same shape as the input spectrum data.
    """

    energy_range = energy_bin_size * np.shape(spectrum)[0]

    # Define the energy bins.
    energy_bins = np.linspace(0, energy_range, num = np.shape(spectrum)[0], endpoint = True)

    # Specify the energy range for each possible ROI.
    if roi == "10B":
        lower_energy = 0.6
        upper_energy = 0.8

    elif roi == "11C":
        lower_energy = 1.8
        upper_energy = 2.2

    elif roi == "12C11B":
        lower_energy = 3.2
        upper_energy = 4.7

    elif roi == "16O":
        lower_energy = 4.9
        upper_energy = 6.3

    elif roi == "511keV":
        lower_energy = 0.3
        upper_energy = 0.6
    elif roi == "full":
        return np.ones(np.shape(spectrum), dtype = bool)


    # Find indices of energy_bins within the range specified for the ROI.
    roi_indices = np.where((energy_bins >= lower_energy) & (energy_bins <= upper_energy))
    
    # Initialize an array of zeros with the same shape as the input spectrum data.
    roi_array = np.zeros(np.shape(spectrum))
    
    # Set values at the ROI indices to 1.
    roi_array[roi_indices, :] = 1

    return roi_array.astype(bool)