#!/bin/bash
#SBATCH --job-name=factor_mom_r
#SBATCH --partition=hicpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --output=logs/analysis_r_%j.log
#SBATCH --error=logs/analysis_r_%j.err

# 1. Load Modules (Same as installation)
module purge
module load gcc
module load r
module load cmake
module load openssl

# 2. Point to your local library
export R_LIBS_USER=~/R_libs

echo "Running Factor Momentum Analysis in R..."

# 3. Run the script
Rscript code/1_factor_momentum_cluster.R

echo "Done."
