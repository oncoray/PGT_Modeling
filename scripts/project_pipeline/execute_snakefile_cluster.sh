#!/bin/bash --login

#SBATCH --job-name=pgt_mod_dev
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

eval "$(mamba shell hook --shell bash --root-prefix=~/.conda/envs/pgt_head_conda/bin/python)"

mamba activate pgt_ml_workflow

# --- detect Slurm cluster name (authoritative) ---
CLUSTER_NAME="$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/ {gsub(/[[:space:]]/,"",$2); print $2; exit}')"


# --- set partitions depending on cluster ---
if [[ "${CLUSTER_NAME}" == "hemera" ]]; then
  module load gcc/10.2.0
  module load libxml2-devel/2.9.1
  PARTITIONS="defq,rome,genoa,milan"
elif [[ "${CLUSTER_NAME}" == "cluster-rosi" ]]; then
  PARTITIONS="cpu-rome,cpu-milan,cpu-genoa"
  ml genoa
  ml GCC/12.3.0
  ml R/4.3.2
  module load libxml2/2.15.0
  module load CMake
else
  echo "ERROR: wrong cluster. Detected ClusterName='${CLUSTER_NAME}'." >&2
  exit 1
fi

############################################
# 1) Create a resolved config on the submit node
############################################
python - << 'EOF'
from pmma.config_file_manipulator import substitute_labels
import yaml

# Load and substitute labels from the template config
cfg = substitute_labels("config.yaml")

# Write a resolved config file next to the original
with open("config_resolved.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

############################################
# 2) Read paths from the resolved config
############################################
logs_dir=$(python - << 'EOF'
import yaml
with open("config_resolved.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["logs_dir"])
EOF
)

experiment_dir=$(python - << 'EOF'
import yaml
with open("config_resolved.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["experiment_dir"])
EOF
)

# Create the experiment directory if it does not exist
mkdir -p "${experiment_dir}"

############################################
# 3) Copy Snakefile and resolved config into experiment_dir
############################################
cp Snakefile "${experiment_dir}/"

# Inside the experiment_dir, the config file will be named config.yaml
# and already be fully resolved (no placeholders).
cp config_resolved.yaml "${experiment_dir}/config.yaml"

# Change the current working directory to the experiment directory
cd "${experiment_dir}"

mkdir -p logs

# Unlock the workflow in case it's locked from previous runs
snakemake --unlock

# The --jobs value (here: 8) sets how many SLURM jobs can be active simultaneously.
# Adjust as needed. The --latency-wait adds a delay (in seconds) to handle file system lags.
# The --cluster-status uses a Python helper to monitor job states under SLURM.

echo "Snakemake version: $(snakemake --version)"
which snakemake

snakemake \
  --executor cluster-generic \
  --rerun-incomplete \
  --latency-wait 10 \
  --cluster-generic-submit-cmd "sleep 2; sbatch \
    --partition ${PARTITIONS} \
    --time={resources.time} \
    --cpus-per-task={threads} \
    --mem-per-cpu={resources.mem_per_cpu} \
    --job-name={rule}.{wildcards} \
    --output=logs/slurm-%j_{rule}_{wildcards}.out" \
  --cluster-generic-cancel-cmd "scancel" \
  --jobs 40 \
  --retries 0 \
  --cores 2000

snakemake --report report.html

# copy slurm-*.out file to desired directory after job finishes
cd -  # Return to the original directory
cp slurm-$SLURM_JOBID.out $logs_dir/

