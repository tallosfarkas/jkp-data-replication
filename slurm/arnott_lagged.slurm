#!/bin/bash
#SBATCH --job-name=arnott_lagged
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --nodelist=clustergpu03
#SBATCH --time=12:00:00
#SBATCH --output=logs/arnott_lagged_%j.log
#SBATCH --error=logs/arnott_lagged_%j.err

module load python
cd $HOME/jkp-data

echo "Job started on $(hostname) at $(date)"
echo "--------------------------------------"

# --- Run the Strategy ---
# This runs the lagged version we just created
uv run --with polars code/arnott_strategy.py

echo "--------------------------------------"
echo "Job finished at $(date)"
