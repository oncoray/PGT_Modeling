# range_reconstruction_PMMA_phantom

We want to reconstruct the air cavity size in a PMMA phantom based on timing and energy information of prompt gammas.


Usage
-----

## Installation

First, create a (conda) virtual environment like this:
```bash
conda create -n pmma python=3.8
```

Install the required libraries and pmma package
```bash
conda activate pmma
cd scripts
pip install -r requirements.txt
pip install -e .
```

Adjust config file and setup experiment:
```bash
nano project_pipeline/config.yaml
```

Submit batch job to excute experiment
```bash
sbatch project_pipeline/execute_snakefile.sh
```

