#!/bin/bash
#SBATCH --job-name=jkp_factor_momentum
#SBATCH --ntasks=1
#SBATCH --partition=medium
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=logs/jkp_factor_momentum_%j.out
#SBATCH --error=logs/jkp_factor_momentum_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@wu.ac.at

# =============================================================
# JKP Factor Momentum Pipeline
#
# Step 1: portfolio_USA_weights.py
#         Reads:  data/processed/characteristics/USA.parquet
#         Writes: data/processed/portfolios/all_factor_constituents.parquet
#                 data/processed/portfolios/pfs.parquet
#                 data/processed/portfolios/pfs_daily.parquet
#                 data/processed/portfolios/lms.parquet
#                 data/processed/portfolios/lms_daily.parquet
#                 ... (hml, clusters, industry, regional)
#
# Step 2: arnott_strategy_master.py
#         Reads:  data/processed/portfolios/all_factor_constituents.parquet
#                 data/processed/return_data/daily_rets_by_country/USA.parquet
#         Writes: data/processed/portfolios/arnott_master.parquet
#
# Prerequisites:
#   - main.py must have been run first (characteristics + return data must exist)
#   - uv must be installed and project virtualenv initialised
# =============================================================

echo "======================================================"
echo "  JKP Factor Momentum Pipeline"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Start    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

cd $HOME/jkp-data || { echo "ERROR: jkp-data directory not found"; exit 1; }
mkdir -p logs

# ---------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------
echo ""
echo "[PRE-FLIGHT] Checking required input files..."

MISSING=0
for f in \
    "data/processed/characteristics/USA.parquet" \
    "data/processed/other_output/nyse_cutoffs.parquet" \
    "data/processed/other_output/return_cutoffs.parquet" \
    "data/processed/other_output/return_cutoffs_daily.parquet" \
    "data/processed/other_output/market_returns.parquet" \
    "data/processed/other_output/market_returns_daily.parquet" \
    "data/processed/return_data/daily_rets_by_country/USA.parquet" \
    "data/factor_details.xlsx" \
    "data/country_classification.xlsx" \
    "data/cluster_labels.csv"
do
    if [ ! -f "$f" ]; then
        echo "  [MISSING] $f"
        MISSING=1
    else
        echo "  [OK]      $f"
    fi
done

if [ $MISSING -ne 0 ]; then
    echo "ERROR: Missing input files. Run main.py first."
    exit 1
fi

# ---------------------------------------------------------
# STEP 1: Build Constituent Bible
#   - NYSE breakpoints (bps=nyse, bp_min_n=1)
#   - All 3 portfolio buckets saved (pf = 1, 2, 3)
#   - All JKP dividend yield columns attached
#   - Output: all_factor_constituents.parquet
# ---------------------------------------------------------
echo ""
echo "[STEP 1/2] portfolio_USA_weights.py"
echo "           Start: $(date '+%H:%M:%S')"

uv run python code/portfolio_USA_weights.py
STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: portfolio_USA_weights.py failed (exit $STEP1_EXIT)"
    exit $STEP1_EXIT
fi

echo "           Done:  $(date '+%H:%M:%S')"

# Verify the key output that Step 2 depends on
if [ ! -f "data/processed/portfolios/all_factor_constituents.parquet" ]; then
    echo "ERROR: all_factor_constituents.parquet was not created. Check logs."
    exit 1
fi

if [ ! -f "data/processed/portfolios/lms_daily.parquet" ]; then
    echo "ERROR: lms_daily.parquet was not created. Check logs."
    exit 1
fi

# ---------------------------------------------------------
# STEP 2: Arnott Factor Momentum — Single-Stock Netting
#   - Reads Constituent Bible
#   - Applies sign corrections (FACTORS_TO_FLIP list)
#   - Computes 1-month momentum signal per factor
#   - Long top 50%, Short bottom 50% (dollar-neutral)
#   - 1-month signal lag: signal(T) → trade(T+1)
#   - 1-day execution lag: weight=0 on first trading day of trade month
#   - Output: arnott_master.parquet
# ---------------------------------------------------------
echo ""
echo "[STEP 2/2] arnott_strategy_master.py"
echo "           Start: $(date '+%H:%M:%S')"

uv run python code/arnott_strategy_master.py
STEP2_EXIT=$?

if [ $STEP2_EXIT -ne 0 ]; then
    echo "ERROR: arnott_strategy_master.py failed (exit $STEP2_EXIT)"
    exit $STEP2_EXIT
fi

echo "           Done:  $(date '+%H:%M:%S')"

# ---------------------------------------------------------
# Final output summary
# ---------------------------------------------------------
echo ""
echo "======================================================"
echo "  Output files:"
for f in \
    "data/processed/portfolios/all_factor_constituents.parquet" \
    "data/processed/portfolios/pfs.parquet" \
    "data/processed/portfolios/pfs_daily.parquet" \
    "data/processed/portfolios/lms.parquet" \
    "data/processed/portfolios/lms_daily.parquet" \
    "data/processed/portfolios/arnott_master.parquet"
do
    if [ -f "$f" ]; then
        SIZE=$(du -sh "$f" | cut -f1)
        echo "  [OK]      $f  ($SIZE)"
    else
        echo "  [MISSING] $f"
    fi
done

echo "======================================================"
echo "  Pipeline complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

exit 0
