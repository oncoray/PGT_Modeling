#!/bin/bash --login

###############################################################################
#                          SLURM MASTER JOB SETTINGS
###############################################################################
#SBATCH --job-name=PGT_head
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3GB

echo "Job started at: $(date)"


eval "$(mamba shell hook --shell bash --root-prefix=~/.conda/envs/pgt_head_conda/bin/python)"

mamba activate pgt_ml_workflow


# --- detect Slurm cluster name (authoritative) ---
CLUSTER_NAME="$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/ {gsub(/[[:space:]]/,"",$2); print $2; exit}')"


# --- set partitions depending on cluster ---
if [[ "${CLUSTER_NAME}" == "hemera" ]]; then
  module load gcc/10.2.0
  module load libxml2-devel/2.9.1
  PARTITIONS="defq"
elif [[ "${CLUSTER_NAME}" == "cluster-rosi" ]]; then
  PARTITIONS="cpu-rome,cpu-milan,cpu-genoa"
  module load gcc/14.2.0
  module load libxml2/2.15.0
else
  echo "ERROR: wrong cluster. Detected ClusterName='${CLUSTER_NAME}'." >&2
  exit 1
fi



###############################################################################
#                          PARSE DIRECTORIES FROM CONFIG
###############################################################################
# Example: reading from a Python function that returns the 'logs_dir' and 'experiment_dir'
logs_dir=$(python -c "from pmma.config_file_manipulator import substitute_labels; print(substitute_labels('config.yaml')['paths']['logs_dir'])" | tail -n 1)
experiment_dir=$(python -c "from pmma.config_file_manipulator import substitute_labels; print(substitute_labels('config.yaml')['paths']['experiment_dir'])" | tail -n 1)

mkdir -p "${logs_dir}"
mkdir -p "${experiment_dir}"

# Copy relevant files to the experiment directory
cp Snakefile "${experiment_dir}/"
cp config.yaml "${experiment_dir}/"

# Move into the experiment directory
cd "${experiment_dir}"

echo "Created directories and copied workflow files at: $(date)"
echo "Current directory: $(pwd)"

###############################################################################
#                          SNAKEMAKE IN CLUSTER MODE
###############################################################################
# 1) snakemake --unlock (if needed)
# 2) snakemake --cluster  (each rule is submitted as its own SLURM job)

# Unlock the workflow in case it's locked from previous runs
snakemake --unlock

# The --jobs value (here: 8) sets how many SLURM jobs can be active simultaneously.
# Adjust as needed. The --latency-wait adds a delay (in seconds) to handle file system lags.
# The --cluster-status uses a Python helper to monitor job states under SLURM.

echo "Snakemake version: $(snakemake --version)"
which snakemake


snakemake \
  --executor cluster-generic \
  --latency-wait 10 \
  --cluster-generic-submit-cmd "sleep 2; sbatch \
    --partition ${PARTITIONS} \
    --time={resources.time} \
    --cpus-per-task={threads} \
    --mem-per-cpu={resources.mem_per_cpu} \
    --job-name={rule}.{wildcards} \
    --output=logs/slurm-%j_{rule}_{wildcards}.out" \
  --cluster-generic-cancel-cmd "scancel" \
  --jobs 8 \
  --retries 0 \
  --cores 128 \
  --rerun-incomplete





# Optionally generate a Snakemake HTML report for the entire workflow
snakemake --report report.html

###############################################################################
#                          POST-JOB CLEANUP
###############################################################################
cd -  # Return to the original directory

# Copy the SLURM master job log to the logs_dir
cp slurm-${SLURM_JOBID}.out "${logs_dir}/"

echo "Job completed at: $(date)"
