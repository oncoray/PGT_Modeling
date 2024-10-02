#!/bin/bash --login

#SBATCH --job-name=PMMA
#SBATCH --time=8:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=1GB

conda activate pmma
module load gcc/10.2.0
module load R/4.3.0

# module to install xml2 R-package
module load libxml2-devel/2.9.1

# Read the logs_dir from the substituted config file
logs_dir=$(python -c "from pmma.config_file_manipulator import substitute_labels; print(substitute_labels('config.yaml')['paths']['logs_dir'])" | tail -n 1)

# Construct the experiment directory path
experiment_dir=$(python -c "from pmma.config_file_manipulator import substitute_labels; print(substitute_labels('config.yaml')['paths']['experiment_dir'])" | tail -n 1)

# Create the experiment directory if it does not exist
mkdir -p "${experiment_dir}"

# Copy the Snakefile to the experiment directory
cp Snakefile "${experiment_dir}/"

cp config.yaml "${experiment_dir}/"

# Change the current working directory to the experiment directory
cd "${experiment_dir}"

# Execute snakemake commands in the experiment directory
snakemake --unlock
snakemake --cores ${SLURM_CPUS_PER_TASK} --rerun-incomplete
snakemake --report report.html

# copy slurm-*.out file to desired directory after job finishes
cd -  # Return to the original directory
cp slurm-$SLURM_JOBID.out $logs_dir/

