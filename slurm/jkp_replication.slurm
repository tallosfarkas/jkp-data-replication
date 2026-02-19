#!/bin/bash
#SBATCH --job-name=jkp_replication
#SBATCH --partition=hicpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=48:00:00
#SBATCH --output=jkp_output_%j.log
#SBATCH --error=jkp_error_%j.err

echo "------------------------------------------------"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Start:     $(date)"
echo "Resources: 64 CPUs, 450GB RAM, hicpu partition"
echo "------------------------------------------------"

# Ensure we are in the correct directory
cd $SLURM_SUBMIT_DIR

# 1. Run Data Generation (Stocks & Characteristics)
echo ">>> [Step 1/2] Running main.py..."
start_time=$(date +%s)
uv run code/main.py
end_time=$(date +%s)
echo ">>> main.py finished in $((end_time - start_time)) seconds."

if [ $? -ne 0 ]; then
    echo "!!! Error: main.py failed. Stopping execution."
    exit 1
fi

# 2. Run Portfolio Construction (Factors)
echo ">>> [Step 2/2] Running portfolio.py..."
start_time=$(date +%s)
uv run code/portfolio.py
end_time=$(date +%s)
echo ">>> portfolio.py finished in $((end_time - start_time)) seconds."

if [ $? -ne 0 ]; then
    echo "!!! Error: portfolio.py failed."
    exit 1
fi

echo "------------------------------------------------"
echo "All steps completed successfully at $(date)"
echo "------------------------------------------------"
