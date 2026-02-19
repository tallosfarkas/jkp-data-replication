#!/bin/bash
#SBATCH --job-name=repair_arrow
#SBATCH --partition=hicpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=logs/repair_%j.log
#SBATCH --error=logs/repair_%j.err

module purge
module load gcc
module load r
module load cmake
module load openssl

export R_LIBS_USER=~/R_libs

echo "Repairing Arrow Installation..."
Rscript repair_arrow.R
