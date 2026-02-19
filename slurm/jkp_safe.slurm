#!/bin/bash
#SBATCH --job-name=jkp_safe
#SBATCH --partition=hicpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=48:00:00
#SBATCH --output=jkp_safe_%j.log
#SBATCH --error=jkp_safe_%j.err

echo "------------------------------------------------"
echo "Job ID:    $SLURM_JOB_ID"
echo "Start:     $(date)"
echo "------------------------------------------------"

cd $SLURM_SUBMIT_DIR

# 1. Run Data Generation (Stocks & Characteristics)
echo ">>> [Step 1/2] Running main.py..."
uv run code/main.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "!!! Error: main.py failed with exit code $EXIT_CODE."
    echo "Check jkp_safe_$SLURM_JOB_ID.err for python tracebacks."
    exit $EXIT_CODE
fi

echo ">>> main.py finished successfully."

# 2. Run Portfolio Construction (Factors)
echo ">>> [Step 2/2] Running portfolio.py..."
uv run code/portfolio.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "!!! Error: portfolio.py failed with exit code $EXIT_CODE."
    exit $EXIT_CODE
fi

echo "------------------------------------------------"
echo "All steps completed successfully at $(date)"
echo "------------------------------------------------"
