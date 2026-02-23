#!/bin/bash
#SBATCH --job-name=arnott_phase2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=logs/arnott_phase2_%j.log
#SBATCH --error=logs/arnott_phase2_%j.err

echo "Job started on $(hostname) at $(date)"
echo "--------------------------------------"

cd $HOME/jkp-data-replication

# Make sure logs dir exists
mkdir -p logs

# Run Phase 2 full grid (30 combinations, stock-level decomposition)
uv run python code/arnott_phase2_full_grid.py

echo "--------------------------------------"
echo "Job finished at $(date)"
