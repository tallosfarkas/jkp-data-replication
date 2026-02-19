#!/bin/bash
#SBATCH --job-name=jkp_py_mom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=jkp_py_mom_%j.log
#SBATCH --error=jkp_py_mom_%j.err

echo "Job Started at $(date)"

# Run the Python script using uv
uv run --with polars 1_select_factors_median.py

echo "Job Finished at $(date)"
