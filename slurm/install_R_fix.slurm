#!/bin/bash
#SBATCH --job-name=install_R_fix
#SBATCH --partition=hicpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_%j.log
#SBATCH --error=logs/install_%j.err

module purge

# LOAD DEPENDENCIES (CRITICAL)
module load gcc
module load r
module load cmake
module load openssl  # <-- This fixes the "OpenSSL not found" error

export R_LIBS_USER=~/R_libs
mkdir -p $R_LIBS_USER

echo "Starting installation with OpenSSL loaded..."
Rscript install_packages_fast.R
