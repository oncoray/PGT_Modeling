#!/bin/bash --login

#SBATCH --job-name=model_eval
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

set -euo pipefail

############################################
# 0) Environment
############################################
eval "$(mamba shell hook --shell bash --root-prefix=~/.conda/envs/pgt_head_conda/bin/python)"
mamba activate pgt_ml_workflow

PARTITIONS="cpu-rome,cpu-milan,cpu-genoa"
ml genoa
ml GCC/12.3.0
ml R/4.3.2
module load libxml2/2.15.0

############################################
# 3) Create a resolved config on the submit node
############################################
python - << 'EOF'
from pmma.config_file_manipulator import substitute_labels
import yaml

cfg = substitute_labels("config.yaml")

with open("config_resolved.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

############################################
# 4) Read paths from the resolved config
############################################
out_dir=$(python - << 'EOF'
import yaml
with open("config_resolved.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["out_dir"])
EOF
)

logs_dir=$(python - << 'EOF'
import yaml
with open("config_resolved.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["logs_dir"])
EOF
)

# Create the run/output directory if it does not exist
mkdir -p "${out_dir}"
mkdir -p "${logs_dir}"

############################################
# 5) Copy Snakefile and resolved config into out_dir (run directory)
############################################
cp Snakefile "${out_dir}/"
cp config_resolved.yaml "${out_dir}/config.yaml"

cd "${out_dir}"
mkdir -p logs

############################################
# 6) Unlock in case it is locked from previous runs
############################################
snakemake --unlock || true

echo "Snakemake version: $(snakemake --version)"
which snakemake

############################################
# 7) Execute Snakemake on SLURM (cluster-generic)
############################################
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

############################################
# 8) Create a Snakemake HTML report
############################################
snakemake --report report.html

############################################
# 9) Copy the submit script SLURM output to logs_dir (optional)
############################################
cd - >/dev/null
if [[ -n "${SLURM_JOBID:-}" ]] && [[ -f "slurm-${SLURM_JOBID}.out" ]]; then
  cp "slurm-${SLURM_JOBID}.out" "${logs_dir}/"
fi
