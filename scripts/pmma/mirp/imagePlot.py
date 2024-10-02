import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from mirp.utilities import check_string, makedirs_check


def plot_image(img_obj, roi_list=None, slice_id="all", roi_mask=None, file_path=None, file_name="plot", g_range=None):
    """

    :param img_obj:
    :param roi_list:
    :param slice_id:
    :param roi_mask:
    :param file_path:
    :param file_name
    :param g_range
    :return:
    """

    # Skip if the input image is not available
    if img_obj.is_missing:
        return

    # Determine grey level range
    if g_range is None:
        g_range = [np.nan, np.nan]
    else:
        g_range = deepcopy(g_range)

    # Adapt unset intensity ranges
    if img_obj.modality == "PT" and not img_obj.is_normalised:
        # PET specific settings
        if np.isnan(g_range[0]):
            g_range[0] = 0.0

        if np.isnan(g_range[1]):
            g_range[1] = np.max(img_obj.get_voxel_grid())

    elif img_obj.modality == "CT" and not img_obj.is_normalised:
        # CT specific settings
        if np.isnan(g_range[0]):
            g_range[0] = -1024.0

        if np.isnan(g_range[1]):
            g_range[1] = np.max(img_obj.get_voxel_grid())

    elif img_obj.modality == "MR" and not img_obj.is_normalised:
        # MR specific settings
        if np.isnan(g_range[0]):
            g_range[0] = 0.0

        if np.isnan(g_range[1]):
            g_range[1] = np.max(img_obj.get_voxel_grid())

    elif img_obj.is_normalised:
        g_range[0] = np.min(img_obj.get_voxel_grid())
        g_range[1] = np.max(img_obj.get_voxel_grid())

    # Create custom colour map for rois ###################################

    # Import custom color map function
    from matplotlib.colors import LinearSegmentedColormap

    # Define color map. The custom color map goes from transparent black to semi-transparent green and is used as an overlay.
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.6, 0.6)),
             'blue': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
             'alpha': ((0.0, 0.0, 0.0),
                       (1.0, 0.4, 0.4))
             }

    # Create map and register
    custom_roi_cmap = LinearSegmentedColormap("colour_roi", cdict)
    plt.register_cmap(cmap=custom_roi_cmap)

    # Define colour map for base images ################################
    # Determine modality and set colour map used.
    if img_obj.modality == "PT":
        img_colour_map = "gist_yarg"
    elif img_obj.modality == "CT":
        img_colour_map = "gist_gray"
    else:
        img_colour_map = "gist_gray"

    # Check type of slice_id variable
    if isinstance(slice_id, str):
        if slice_id == "all":
            roi_flag = None
            slice_id = np.arange(start=0, stop=img_obj.size[0])
        elif slice_id == "all_roi":
            roi_flag = "all"
            slice_id = None
        elif slice_id == "roi_center":
            roi_flag = "center"
            slice_id = None
        elif slice_id == "img_center":
            roi_flag = None
            slice_id = int(np.floor(img_obj.size[0]/2))
        else:
            raise ValueError("%s is not a valid entry for identifying one or more slices for plotting.", slice_id)
    else:
        roi_flag = None

    # create directory for the given patient
    img_descriptor = check_string(img_obj.get_export_descriptor())
    plot_path = os.path.join(file_path, img_descriptor) if file_path is not None else None
    if plot_path is not None:
        makedirs_check(plot_path)

    # Plot without roi ################################################
    # If no rois are present, iterate over slices only
    if roi_list is None:

        # Cast slice_id to list, if it isn't a list. This allows iteration.
        if not type(slice_id) is list:
            slice_id = [slice_id]

        # Iterate over slices
        for curr_slice in slice_id:

            # Set file name
            if plot_path is not None:
                plot_file_name = os.path.join(plot_path, check_string(file_name + "_" + str(curr_slice) + ".png"))
            else:
                plot_file_name = None

            # Get image and roi slices
            img_slice = img_obj.get_voxel_grid()[curr_slice, :, :]

            # Determine minimum and maximum luminance:
            img_g_range = crop_image_intensity(img_slice=img_slice, g_range=g_range, modality=img_obj.modality)

            # Determine the extent of the image in mm so that it maintains the pixel size.
            extent = [img_obj.spacing[1] * img_obj.size[1],
                      img_obj.spacing[2] * img_obj.size[2]]

            # Plot
            plotter(slice_list=[img_slice],
                    colour_map_list=[img_colour_map],
                    file_name=plot_file_name,
                    intensity_range=img_g_range,
                    extent=extent)

    # Plot with roi ##################################################
    else:

        # Iterate over rois in roi_list
        for curr_roi in roi_list:
            roi_descriptor = check_string(curr_roi.get_export_descriptor())
            plot_path_roi = os.path.join(plot_path, roi_descriptor) if plot_path is not None else None
            if plot_path_roi is not None:
                makedirs_check(plot_path_roi)

            # Find roi center slice
            if slice_id is None:
                if roi_flag == "center":
                    slice_id = curr_roi.get_center_slice()
                elif roi_flag == "all":
                    slice_id = curr_roi.get_all_slices()

            # Cast slice_id to list, if it isn't a list. This allows iteration.
            if not (isinstance(slice_id, list) or isinstance(slice_id, np.ndarray)):
                slice_id = [slice_id]

            # Iterate over slices
            for curr_slice in slice_id:

                # Set file name
                if plot_path_roi is not None:
                    plot_file_name = os.path.join(plot_path_roi, check_string(file_name + "_" + str(curr_slice) + ".png"))
                else:
                    plot_file_name = None

                # Get image and roi slices
                img_slice = img_obj.get_voxel_grid()[curr_slice, :, :]
                if roi_mask == "intensity" and curr_roi.roi_intensity is not None:
                    roi_slice = curr_roi.roi_intensity.get_voxel_grid()[curr_slice, :, :]
                elif roi_mask == "morphology" and curr_roi.roi_morphology is not None:
                    roi_slice = curr_roi.roi_morphology.get_voxel_grid()[curr_slice, :, :]
                else:
                    roi_slice = curr_roi.roi.get_voxel_grid()[curr_slice, :, :]

                # Determine minimum and maximum luminance:
                if img_obj.is_normalised:
                    img_g_range = deepcopy(g_range)
                else:
                    img_g_range = crop_image_intensity(img_slice=img_slice, g_range=g_range, modality=img_obj.modality)

                # Determine the extent of the image in mm so that it maintains the pixel size.
                extent = [img_obj.spacing[1] * img_obj.size[1],
                          img_obj.spacing[2] * img_obj.size[2]]

                # Create figure
                plotter(slice_list=[img_slice, roi_slice],
                        colour_map_list=[img_colour_map, "colour_roi"],
                        file_name=plot_file_name,
                        intensity_range=img_g_range,
                        extent=extent)


def plotter(slice_list, colour_map_list, file_name=None, overlay_alpha=1.0, intensity_range=None, extent=(1.0, 1.0)):
    """
    This is the background plotting function that does the actual plotting of images.

    :param slice_list: List of 2D images with the same size. The first image [0] will be plotted in the background
    :param colour_map_list: List of colour maps. The first colour map corresponds to the first image. The number of colour maps should match the number of slices in slice_list
    :param file_name: Full path with file name for writing the image to. If set to None (default), the image is not saved to file, but displayed in a separate window instead.
    :param overlay_alpha: Transparency of overlay images, i.e. the second, third etc. images in the slice list.
    :param intensity_range: Intensity range for the intensities in the plot.
    :return: None
    """

    # Create axes and figure
    fig, ax = plt.subplots(1)

    # Set figure space in axis
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Invisible axis
    ax.axis('off')

    plt.ioff()

    # Plot input image
    for ii in np.arange(len(slice_list)):
        plot_data = deepcopy(slice_list[ii])

        if ii == 0 and intensity_range is not None:
            # Plot image.
            plt.imshow(plot_data,
                       cmap=plt.get_cmap(colour_map_list[ii]),
                       extent=[0, extent[1], 0, extent[0]],
                       vmin=intensity_range[0],
                       vmax=intensity_range[1],
                       alpha=1.0,
                       interpolation="none")

        else:
            # Plot mask. First check if the slice contains any part of the mask.
            if np.sum(plot_data) == 0:
                continue

            plt.imshow(plot_data,
                       cmap=plt.get_cmap(colour_map_list[ii]),
                       extent=[0, extent[1], 0, extent[0]],
                       vmin=0,
                       vmax=1,
                       alpha=overlay_alpha,
                       interpolation="none")

    # Save plot
    plt.savefig(file_name, pad_inches=0.0, bbox_inches="tight", dpi=150)

    plt.close(fig)


def crop_image_intensity(img_slice, g_range, modality):

    # Get intensity range present in the image
    img_g_range = [np.min(img_slice), np.max(img_slice)]

    # Update with re-segmentation range
    if not np.isnan(g_range[0]):
        img_g_range[0] = g_range[0]
    if not np.isnan(g_range[1]):
        img_g_range[1] = g_range[1]

    # Extend 20% outside of range
    if not np.isnan(g_range[0]):
        img_g_range[0] = img_g_range[0] - 0.20 * (img_g_range[1] - img_g_range[0])
    if not np.isnan(g_range[1]):
        img_g_range[1] = img_g_range[1] + 0.20 * (img_g_range[1] - img_g_range[0])

    # Modality specific settings
    if modality == "PT" and img_g_range[0] < 0.0:
        img_g_range[0] = 0.0

    if modality == "CT" and img_g_range[0] < -1000.0:
        img_g_range[0] = -1000.0
    elif modality == "CT" and img_g_range[1] > 3000.0:
        img_g_range[1] = 3000.0

    if modality == "MR" and img_g_range[0] < 0.0:
        img_g_range[0] = 0.0

    return img_g_range
