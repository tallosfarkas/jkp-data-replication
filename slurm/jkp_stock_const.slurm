#!/bin/bash
#SBATCH --job-name=jkp_stock_const
#SBATCH --nodes=1
#SBATCH --partition=hicpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --output=jkp_stock_%j.log
#SBATCH --error=jkp_stock_%j.err

echo "Job Started at $(date)"
uv run --with polars code/2_construct_portfolios.py
echo "Job Finished at $(date)"
