#!/bin/bash
#SBATCH --job-name=jkp_factor_mom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=jkp_factor_mom_%j.log
#SBATCH --error=jkp_factor_mom_%j.err

echo "------------------------------------------------"
echo "Job Started at $(date)"
echo "------------------------------------------------"

# Load R module (adjust version if needed, e.g., 'module load r/4.3.0')
module load r

# Run the R script
Rscript code/Factor_momentum.R

echo "------------------------------------------------"
echo "Job Finished at $(date)"
echo "------------------------------------------------"
