# from mirp.configThreading import disable_multi_threading
# disable_multi_threading()

import logging
import multiprocessing as mp
import os
import time
from inspect import stack, getmodule

import numpy as np
import pandas as pd

from pmma.mirp.importSettings import import_configuration_settings, import_data_settings


def get_roi_names(data_config=None, settings_config=None, to_file=True):
    """ Allows automatic extraction of roi names """

    if settings_config is not None:
        settings_list = import_configuration_settings(settings_config)
        data_obj_list = import_data_settings(data_config, settings_list)
    else:
        data_obj_list = import_data_settings(data_config, None)

    roi_info_list = []

    # Iterate over data and
    for data_obj in data_obj_list:
        roi_info_list += [data_obj.get_roi_list()]

    # Concatenate list
    df_roi = pd.concat(roi_info_list)

    # Export list to project path
    if to_file:
        write_path = data_obj_list[0].write_path
        file_path = os.path.normpath(os.path.join(write_path, "project_roi.csv"))
        df_roi.to_csv(path_or_buf=file_path, sep=";", na_rep="NA", index=False, decimal=".")

        logging.info(f"Writing list of ROI names to {file_path}.")

    else:
        return df_roi


def get_image_acquisition_parameters(data_config=None, settings_config=None, plot_images="single", to_file=True):
    """ Allows automatic extraction of imaging parameters """

    # Read settings from settings files and create the appropriate data objects.
    if settings_config is not None:
        settings_list = import_configuration_settings(settings_config)
        data_obj_list = import_data_settings(data_config, settings_list, plot_images=plot_images)
    else:
        data_obj_list = import_data_settings(data_config, None, plot_images=plot_images)

    # Initiate an empty list
    meta_list = []

    # Iterate over all data and extract imaging parameters
    for data_obj in data_obj_list:
        meta_list += [data_obj.get_imaging_parameter_table()]

    # Concatenate list
    df_meta = pd.concat(meta_list)

    # Export list to project path
    if to_file:
        write_path = data_obj_list[0].write_path
        file_path = os.path.normpath(os.path.join(write_path, "project_image_meta_data.csv"))
        df_meta.to_csv(path_or_buf=file_path, sep=";", na_rep="NA", index=False, decimal=".")

        logging.info(f"Writing list of acquisition parameters to {file_path}.")

    else:
        return df_meta


def get_file_structure_parameters(data_config, to_file=True):

    # Create separate data object for each subject
    data_obj_list = import_data_settings(data_config, None, file_structure=True)

    # Initiate an empty list
    meta_list = []

    # Iterate over all data and extract imaging parameters
    for data_obj in data_obj_list:
        meta_list += [data_obj.get_file_structure_information()]

    # Concatenate list
    df_meta = pd.concat(meta_list, sort=False)

    # Write to file
    if to_file:
        write_path = data_obj_list[0].write_path
        file_path = os.path.normpath(os.path.join(write_path, "file_meta_data.csv"))
        df_meta.to_csv(path_or_buf=file_path, sep=";", na_rep="NA", index=False, decimal=".")

        logging.info(f"Writing overview of images files to {file_path}.")

    else:
        return df_meta


def parse_file_structure(data_config, file, use_folder_name=True):
    """
    Parse
    :param data_config: Path to the data configuration xml file.
    :param file: Path to the assignment csv file.
    :param use_folder_name: Flag to use the original folder name (True) or the DICOM patient name entry (False) as the name for the main patient-level folder.
    :return:
    """

    # Create separate data object for each subject
    data_obj_list = import_data_settings(data_config, None, file_structure=True)

    # Iterate over all data and extract imaging parameters
    for data_obj in data_obj_list:
        data_obj.restructure_files(file=file, use_folder_name=use_folder_name)


def extract_images_for_deep_learning(data_config,
                                     settings_config,
                                     output_slices=False,
                                     crop_size=None,
                                     center_crops_per_slice=True,
                                     remove_empty_crops=True,
                                     intensity_range=None,
                                     normalisation="none",
                                     as_numpy=False,
                                     plot_images=False):
    """
    Extract images for deep learning.

    :param data_config: full path to a data configuration xml file.
    :param settings_config: full path to a settings configuration xml file.
    :param output_slices: whether to produce image slices (True) or an image volume (False) as output.
    :param crop_size: set size of the crop. If two values are specified, crops are created in the axial
    plane (y, x). In case three values are provided, the slice stack is cropped to the given size (z, y, x). Providing
    three values allows for generating three-dimensional crops. If only one value is provided, an axial crop with
    equal x and y dimensions is created.
    :param center_crops_per_slice: Specifies whether crops are centered the ROI on each individual slice (True) or on
    the overall center of mass (False). This setting is ignored (False) if crop_size is specified to generate
    three-dimensional crops.
    :param remove_empty_crops: Specifies whether slices without a mask are removed (True) or kept (False). This
    setting is ignored (False) if crop_size is specified to generate three-dimensional crops.
    :param intensity_range: Range (if any) for saturating image intensities. Image intensities outside this range
    receive the nearest valid value.
    :param normalisation: Method used to normalise image intensity. Can be one of "none", "range" or
    "standardisation". Image intensities are normalised after saturating the image.
    :param as_numpy: Specify whether the slices and masks should be returned as numpy ndarrays (True) or MIRP image
    and ROI objects (False).
    :param plot_images: flag to plot images and masks as .png (default: False)
    :return: List of (ImageClass) images with (RoiClass) masks

    This function allows unified processing of images for input into deep learning networks.
    """

    # Parse data
    settings_list = import_configuration_settings(path=settings_config)
    data_obj_list = import_data_settings(path=data_config,
                                         config_settings=settings_list,
                                         keep_images_in_memory=False,
                                         compute_features=False,
                                         extract_images=False,
                                         plot_images=plot_images)

    # Process images for deep learning
    image_list = []
    for data_obj in data_obj_list:
        image_list += data_obj.process_deep_learning(output_slices=output_slices,
                                                     crop_size=crop_size,
                                                     center_crops_per_slice=center_crops_per_slice,
                                                     remove_empty_crops=remove_empty_crops,
                                                     normalisation=normalisation,
                                                     as_numpy=as_numpy,
                                                     intensity_range=intensity_range)

    return image_list


def parallel_process(data_obj):
    """
    Function for parallel feature extraction

    :param data_obj: input ExperimentClass object.
    :return:

    This function is used internally.
    """

    data_obj.process()


def extract_features(data_config, settings_config, n_processes=1):
    """
    Automates feature extraction
    :param data_config: full path to a data configuration xml file.
    :param settings_config: full path to a settings configuration xml file.
    :param n_processes: number of simultaneous processes. For n_processes > 1, parallel processes are started.
    :return:

    This function calls the process_images function with presets.
    """
    process_images(data_config=data_config, settings_config=settings_config, n_processes=n_processes,
                   keep_images_in_memory=False, compute_features=True, extract_images=False, plot_images=False)


def extract_images_to_nifti(data_config, settings_config, n_processes=1):
    """
    Automates extraction of images to nifti format.
    :param data_config: full path to a data configuration xml file.
    :param settings_config: full path to a settings configuration xml file.
    :param n_processes: number of simultaneous processes. For n_processes > 1, parallel processes are started.
    :return:

    This function calls the process_images function with presets.
    """

    process_images(data_config=data_config, settings_config=settings_config, n_processes=n_processes,
                   keep_images_in_memory=False, compute_features=False, extract_images=True, plot_images=False)


def process_images(data_config, settings_config, n_processes=1, keep_images_in_memory=False, compute_features=True,
                   extract_images=False, plot_images=False):
    """
    Process images for various tasks.

    :param data_config: full path to a data configuration xml file.
    :param settings_config: full path to a settings configuration xml file.
    :param n_processes: number of simultaneous processes. For n_processes > 1, parallel processes are started.
    :param keep_images_in_memory: flag to keep images in memory. This avoids repeated loading of images, but at the expense of memory.
    :param compute_features: flag to compute features (default: False)
    :param extract_images: flag to extract images and mask in Nifti format (default: False)
    :param plot_images: flag to plot images and masks as .png (default: False)
    :return:
    """

    if not compute_features and not extract_images:
        return None

    # Extract features
    if n_processes > 1:

        # Usually, we would only run this code if __name__ == "__main__".
        # However, as this function is nested in a module, the __name__ is always "mirp.mainFunctions" instead of "__main__", thus prohibiting any multiprocessing.
        # Therefore we have to see if "__main__" is the name that appears in one of the parent modules. This is not always the top module, and therefore we search
        # the stack. The trick is that on windows OS "__main__" does appear in the calling environment from where the script is executed, but is called "__mp_main__"
        # in the multiprocessing spawning process. We perform this check to prevent infinite spawning on Windows OS, as it doesn't fork neatly but spawns identical
        # separate process by repeating the stack call in each process. The alternative would be to include a __name__ == "__main__" check in the calling script,
        # and a switch variable (e.g. as_master) in the function call, but that puts the onus on the end-user, and is a terrible idea due to the principle of least
        # astonishment.
        module_names = ["none"]
        for stack_entry in stack():
            current_module = getmodule(stack_entry[0])
            if current_module is not None:
                module_names += [current_module.__name__]

        if "__main__" in module_names:

            # Parse data
            settings_list = import_configuration_settings(path=settings_config)
            data_obj_list = import_data_settings(path=data_config, config_settings=settings_list, keep_images_in_memory=keep_images_in_memory,
                                                 compute_features=compute_features, extract_images=extract_images, plot_images=plot_images)

            # Initate process manager
            df_mngr = pd.DataFrame({"job_id": np.arange(len(data_obj_list)),
                                    "job_processed": np.zeros(shape=len(data_obj_list), dtype=bool),
                                    "job_in_process": np.zeros(shape=len(data_obj_list), dtype=bool),
                                    "assigned_worker": -np.ones(shape=len(data_obj_list), dtype=int),
                                    "error_iteration": np.zeros(shape=len(data_obj_list), dtype=int)})

            # Initiate worker list
            worker_list = []
            for ii in np.arange(n_processes):

                # Check if enough jobs are available
                if ii >= len(data_obj_list): break

                # Add job to worker
                process_name = data_obj_list[ii].subject + "_" + data_obj_list[ii].modality + "_" + data_obj_list[ii].data_str + "_" +\
                               data_obj_list[ii].settings.general.config_str
                worker_list.append(mp.Process(target=parallel_process, args=(data_obj_list[ii],), name=process_name))
                worker_list[ii].daemon = True
                df_mngr.loc[ii, "assigned_worker"] = ii

            # Initiate a list that keeps track of repeated errors and skips those samples.
            error_skip_list = []

            # Iterate and process all jobs
            while np.any(~df_mngr.job_processed):

                # Start jobs
                for ii in np.arange(len(worker_list)):
                    # Check if worker is assigned
                    if ~np.any(df_mngr.assigned_worker == ii):
                        continue

                    # Get current job id
                    curr_job_id = df_mngr.job_id[df_mngr.assigned_worker == ii]

                    # Check if job is still in progress or was completed
                    if df_mngr.job_processed[curr_job_id].values or df_mngr.job_in_process[curr_job_id].values: continue

                    # Start process
                    df_mngr.loc[curr_job_id, "job_in_process"] = True
                    worker_list[ii].start()

                # No more workers are available
                free_workers = []

                # Check finished jobs - every 5 seconds
                while len(free_workers) == 0:
                    time.sleep(5)
                    for ii in np.arange(len(worker_list)):

                        # Check if worker is assigned
                        if ~np.any(df_mngr.assigned_worker == ii):
                            free_workers.append(ii)
                            continue

                        # Get current job id
                        curr_job_id = df_mngr.job_id[df_mngr.assigned_worker == ii].values[0]

                        # Check if worker is still processing
                        if worker_list[ii].is_alive(): continue

                        # Check exit code of the stopped worker
                        if worker_list[ii].exitcode == 0:
                            # Normal exit - update table and set worker
                            df_mngr.loc[curr_job_id, "job_processed"] = True
                            df_mngr.loc[curr_job_id, "job_in_process"] = False
                            df_mngr.loc[curr_job_id, "assigned_worker"] = -1

                            free_workers.append(ii)

                        else:
                            # This indicates some fault (e.g. segmentation fault)
                            df_mngr.loc[curr_job_id, "error_iteration"] += 1

                            # Stop after 2 iterations that produce errors
                            if df_mngr.loc[curr_job_id, "error_iteration"] < 2:
                                df_mngr.loc[curr_job_id, "job_in_process"] = False
                                df_mngr.loc[curr_job_id, "assigned_worker"] = -1

                                logging.warning("Process ended prematurely, attempting to restart.")
                            else:
                                df_mngr.loc[curr_job_id, "job_processed"] = True
                                df_mngr.loc[curr_job_id, "job_in_process"] = False
                                df_mngr.loc[curr_job_id, "assigned_worker"] = -1

                                error_skip_list.append(curr_job_id)

                                logging.warning("Process ended prematurely, no attempt to restart again.")

                            # Free up the worker
                            free_workers.append(ii)

                # Check remaining available jobs
                available_jobs = df_mngr.job_id[~np.logical_or(df_mngr.job_processed, df_mngr.job_in_process)]

                # Add new jobs to workers
                for jj in np.arange(len(free_workers)):

                    # Check if enough jobs are available
                    if jj >= len(available_jobs): break

                    # Add job to worker
                    sel_job_id = available_jobs.values[jj]
                    process_name = data_obj_list[sel_job_id].subject + "_" + data_obj_list[sel_job_id].modality + "_" + \
                                   data_obj_list[sel_job_id].data_str + "_" + data_obj_list[sel_job_id].settings.general.config_str
                    worker_list[free_workers[jj]] = mp.Process(target=parallel_process,
                                                               args=(data_obj_list[sel_job_id],), name=process_name)
                    worker_list[free_workers[jj]].daemon = True
                    df_mngr.loc[sel_job_id, "assigned_worker"] = free_workers[jj]

            # Exit statement
            logging.info("Feature calculation has been completed.")

            if len(error_skip_list) > 0:
                names = [data_obj_list[ii].modality + "_" + data_obj_list[ii].data_str + " of " + data_obj_list[ii].subject + " (" +
                         data_obj_list[ii].cohort + ")" for ii in error_skip_list]
                logging.info("No features could be calculated for %s due to errors.", ", ".join(sample_name for sample_name in names))

    else:
        # Parse data
        settings_list = import_configuration_settings(path=settings_config)
        data_obj_list = import_data_settings(path=data_config, config_settings=settings_list, keep_images_in_memory=keep_images_in_memory,
                                             compute_features=compute_features, extract_images=extract_images)
        for data_obj in data_obj_list:
            print("Processing")
            data_obj.process()
