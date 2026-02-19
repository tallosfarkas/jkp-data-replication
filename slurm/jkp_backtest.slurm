#!/bin/bash
#SBATCH --job-name=jkp_backtest
#SBATCH --nodes=1
#SBATCH --partition=hicpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=jkp_backtest_%j.log
#SBATCH --error=jkp_backtest_%j.err

echo "Job Started at $(date)"
uv run --with polars code/3_run_backtest.py
echo "Job Finished at $(date)"
