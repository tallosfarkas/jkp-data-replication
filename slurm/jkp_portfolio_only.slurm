#!/bin/bash
#SBATCH --job-name=jkp_portfolio_only
#SBATCH --partition=hicpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=logs/jkp_portfolio_%j.log
#SBATCH --error=logs/jkp_portfolio_%j.err

echo "------------------------------------------------"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Start:     $(date)"
echo "Task:      Portfolio Construction & Weight Extraction (USA Only)"
echo "------------------------------------------------"

# Ensure we are in the correct directory (jkp-data root)
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist to avoid error writing
mkdir -p logs

# 1. Skip Main Data Generation
echo ">>> [Skipping Step 1] Data generation (main.py) skipped as requested."

# 2. Run Portfolio Construction
echo ">>> [Step 2/2] Running portfolio.py..."
start_time=$(date +%s)

# Using 'uv run' as per your environment setup
uv run code/portfolio_USA_weights.py
EXIT_CODE=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

if [ $EXIT_CODE -ne 0 ]; then
    echo "!!! Error: portfolio.py failed with exit code $EXIT_CODE."
    echo "!!! Check logs/jkp_portfolio_$SLURM_JOB_ID.err for Python tracebacks."
    exit $EXIT_CODE
fi

echo ">>> portfolio.py finished successfully in $duration seconds."
echo "------------------------------------------------"
echo "Job completed at $(date)"
echo "------------------------------------------------"
