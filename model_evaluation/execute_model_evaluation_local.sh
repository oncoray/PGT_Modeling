#!/bin/bash
set -euo pipefail

###############################################################################
# Local runner for the model evaluation Snakemake pipeline (no SLURM)
#
# Usage:
#   chmod +x execute_snakefile_model_eval_local.sh
#   ./execute_snakefile_model_eval_local.sh
#
# Optional environment variables:
#   CORES=8     # number of local cores Snakemake may use (default: nproc)
#   JOBS=8      # max parallel jobs (default: same as CORES)
#   CONFIG=config.yaml
#   SNAKEFILE=Snakefile
###############################################################################

module load python

eval "$(mamba shell hook --shell bash)"
mamba activate pgt_ml_workflow

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
SNAKEFILE="${SNAKEFILE:-${SCRIPT_DIR}/Snakefile}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${SNAKEFILE}" ]]; then
  echo "ERROR: Snakefile not found: ${SNAKEFILE}" >&2
  exit 1
fi

###############################################################################
# 0) Environment (best-effort; keeps behavior similar but avoids cluster modules)
###############################################################################
# If you want strict behavior: remove the '|| true' parts and enforce activation.

if command -v mamba >/dev/null 2>&1; then
  # Use mamba hook if available
  eval "$(mamba shell hook --shell bash)" || true
  # Activate your workflow env if it exists
  mamba activate pgt_ml_workflow || true
fi

if ! command -v snakemake >/dev/null 2>&1; then
  echo "ERROR: snakemake not found in PATH. Activate the environment that contains Snakemake." >&2
  exit 1
fi

###############################################################################
# 1) Create a resolved config on the local machine (same as cluster script)
###############################################################################
RESOLVED_CFG="${SCRIPT_DIR}/config_resolved.yaml"

python - << 'EOF'
from pmma.config_file_manipulator import substitute_labels
import yaml
cfg = substitute_labels("config.yaml")
with open("config_resolved.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

###############################################################################
# 2) Read paths from the resolved config (same as cluster script)
###############################################################################
out_dir="$(python - << 'EOF'
import yaml
with open("config_resolved.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["out_dir"])
EOF
)"

logs_dir="$(python - << 'EOF'
import yaml
with open("config_resolved.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["logs_dir"])
EOF
)"

mkdir -p "${out_dir}"
mkdir -p "${logs_dir}"

###############################################################################
# 3) Copy Snakefile and resolved config into out_dir (run directory)
###############################################################################
cp "${SNAKEFILE}" "${out_dir}/Snakefile"
cp "${RESOLVED_CFG}" "${out_dir}/config.yaml"

cd "${out_dir}"
mkdir -p logs

###############################################################################
# 4) Unlock in case it is locked from previous runs
###############################################################################
snakemake --unlock || true

echo "Snakemake version: $(snakemake --version)"
echo "Snakemake path:    $(command -v snakemake)"

###############################################################################
# 5) Execute Snakemake locally
###############################################################################
CORES="${CORES:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}"
JOBS="${JOBS:-$CORES}"

snakemake \
  --rerun-incomplete \
  --latency-wait 10 \
  --retries 0 \
  --cores "${CORES}" \
  --jobs "${JOBS}"

###############################################################################
# 6) Create a Snakemake HTML report (same behavior)
###############################################################################
snakemake --report report.html

echo "Done. Outputs are in: ${out_dir}"
echo "Report: ${out_dir}/report.html"
