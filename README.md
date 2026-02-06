# Prompt Gamma Timing Modeling Workflow (PGT_Modeling)

## Overview

This repository provides a reproducible workflow for prompt gamma timing (PGT) model development. The workflow is divided into three parts:

1. **Data processing** – converts raw prompt-gamma timing measurements (root) and associated metadata into processed .npy-files.
2. **Model development** – trains and selects statistical/ML models on the processed data, including feature selection and hyper-parameter tuning.
3. **Model evaluation** – applies trained models to evaluation data and produces performance metrics, visualisations and reports.

Each of these stages is orchestrated by **Snakemake**, and a dedicated configuration (`config.yaml`) and execution script (`*.sh`) are provided for each stage. To reproduce the pipeline you must adjust the configuration files to your local environment and then invoke the appropriate shell script to run the workflow on a **SLURM** cluster or locally.

## Environment setup

Before running any workflow, create and activate the conda environment provided with this repository. This example uses **mamba** for fast package installation and assumes that you are inside the project root (`PGT_Modeling`).

```bash
cd PGT_Modeling
mamba create -n pgt_ml_workflow --override-channels -c conda-forge --strict-channel-priority python=3.11 pip -y
mamba activate pgt_ml_workflow
# install conda and pip requirements
mamba install --override-channels -c conda-forge --strict-channel-priority --file requirements_conda.txt -y
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_pip.txt
# install the repository as editable package
python -m pip install -e .
```

After installation the `pgt_ml_workflow` environment contains all dependencies required by the Snakemake workflows. Activate this environment (`mamba activate pgt_ml_workflow`) whenever you wish to run the pipeline.

## Data processing

The data processing workflow lives in `data_processing/` and is configured via `data_processing/config.yaml`. This configuration defines where raw data are located, where intermediate and final results should be stored, and experiment parameters such as spectrum types and proton energies.

After editing the `paths` section to point to your directories and adjusting experiment settings as needed, you can process the data using Snakemake.

The pipeline ingests measurement files in the **ROOT** format using `uproot`, extracts prompt-gamma timing spectra and applies a sequence of preprocessing steps such as background correction, peak alignment and Gaussian aggregation. The processed spectra are written as NumPy arrays (`*.npy`) to the output directory. Subsequent scripts aggregate detectors and build a data path table from these `.npy` files for model development.

### Running on a cluster

For **SLURM** clusters a ready-made submission script is provided. On the **ROSI** cluster use `execute_data_preprocessing_rosi.sh`; on the **HEMERA** cluster use `execute_data_preprocessing_hemera.sh`.

These SBATCH scripts set SLURM job parameters and activate the workflow environment, detect the cluster name, and then call Snakemake with the cluster-generic executor. They parse the directories defined in the configuration file, copy the Snakefile and resolved `config.yaml` into the experiment directory, unlock the workflow if necessary and dispatch Snakemake rules as individual SLURM jobs.

```bash
sbatch data_processing/execute_data_preprocessing_rosi.sh
```
```bash
sbatch data_processing/execute_data_preprocessing_hemera.sh
```

## Model development

The model development workflow is located in `model_development/`. The configuration file `model_development/config.yaml` specifies paths to the processed data, the experiment name, selected features, cohort definitions and candidate models. Adjust these parameters so that they reflect your environment and desired modelling choices.

### Running on a cluster

To train models on a SLURM cluster use `model_development/execute_snakefile_cluster.sh`. This script activates the `pgt_ml_workflow` environment, detects the cluster name and loads appropriate modules, prepares a resolved configuration, and copies the Snakefile and resolved `config.yaml` into the experiment directory. It then unlocks the workflow and invokes Snakemake with the cluster-generic executor so that each rule is submitted as a SLURM job. After completion it generates a Snakemake report and copies SLURM logs back to the configured logs directory.

Once your configuration file has been edited, submit the script via `sbatch` to run model development on the cluster.

## Model evaluation

The evaluation workflow is in `model_evaluation/` and configured through `model_evaluation/config.yaml`. The configuration defines the paths to the trained model, validation data and output directory, as well as evaluation settings such as the metrics to compute and labels. Edit these fields to match your environment and chosen models.

### Running on a cluster

Modify the evaluation configuration file to point to your model and validation data, then adapt a cluster submission script analogous to those used for data processing and model development. The script should activate the environment, resolve the configuration and call Snakemake with the cluster-generic executor.

```bash
sbatch model_evaluation/execute_model_evaluation_cluster.sh
```

### Running locally

Model evaluation can also be executed locally:

```bash
sbatch model_evaluation/execute_model_evaluation_local.sh
```

