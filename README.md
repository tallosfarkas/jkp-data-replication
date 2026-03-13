# JKP Factor Momentum — ILab Replication Project

This repository contains two connected contributions:

**Part A — JKP Python Migration:** A full Python migration of the Jensen, Kelly & Pedersen (2023, *Journal of Finance*) factor investing pipeline, originally written in SAS/R. The pipeline downloads raw WRDS data, constructs 100+ firm characteristics, and builds long-short factor portfolios for global equity markets.

**Part B — Arnott Factor Momentum Extension:** An implementation of the Arnott, Kalesnik & Linnainmaa (2023, *Review of Financial Studies*) factor momentum strategy at the individual stock level, built on top of the JKP constituent weights produced in Part A.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Execution Order](#execution-order)
4. [Part A: JKP Replication](#part-a-jkp-replication)
5. [Part B: Factor Momentum](#part-b-factor-momentum)
6. [Data Reference Files](#data-reference-files)
7. [Dependencies](#dependencies)
8. [Key Design Choices](#key-design-choices)

---

## Repository Structure

```
jkp-data-replication/
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
│
├── jkp_replication/                   ← Part A: JKP Python migration
│   ├── main.py                        # Pipeline orchestration
│   ├── aux_functions.py               # 100+ helper functions
│   ├── portfolio.py                   # Portfolio construction (all countries)
│   ├── portfolio_USA_weights.py       # Portfolio construction + USA constituent weights  [MODIFIED]
│   └── wrds_credentials.py            # WRDS credential management
│
├── factor_momentum/                   ← Part B: Arnott (2023) extension
│   ├── factor_mom_arnott.py           # Factor-level momentum with 1-day lag  [NEW]
│   ├── arnott_strategy_master.py      # Builds master backtest file (stock-level)  [NEW]
│   ├── arnott_phase2_full_grid.py     # Full 30-combination parameter grid  [NEW]
│   └── factor_mom_backtest.py         # Integrated backtest: factor + stock + diagnostics  [NEW]
│
├── slurm/                             ← HPC job scripts
│   ├── 1_jkp_replication.slurm       # Full JKP pipeline: main.py + portfolio.py
│   ├── 2_jkp_usa_weights.slurm       # USA portfolio + weights only
│   ├── 3_arnott_phase1.slurm         # Arnott Phase 1: weights + strategy master
│   └── 4_arnott_phase2.slurm         # Arnott Phase 2: full grid
│
├── data/
│   ├── factor_details.xlsx
│   ├── country_classification.xlsx
│   └── cluster_labels.csv
│
└── documentation/
    ├── release_notes.qmd
    └── release_notes.html
```

---

## Setup

### Requirements

- A valid **WRDS account** with access to CRSP and Compustat
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) installed on your system
- A machine with sufficient resources: **64+ CPU cores, 450 GB RAM** recommended for the full pipeline

### Store WRDS credentials

```sh
uv run python jkp_replication/wrds_credentials.py
```

Follow the prompts. Credentials are stored in the OS keyring — never written to disk in plaintext.

To reset:

```sh
uv run python jkp_replication/wrds_credentials.py --reset
uv run python jkp_replication/wrds_credentials.py
```

> **Note:** When starting the pipeline, WRDS may prompt for two-factor authentication (e.g., a Duo push). Approve within a few seconds or the script will fail.

---

## Execution Order

The full pipeline runs in four sequential steps:

| Step | Script | Description | Prerequisites |
|------|--------|-------------|---------------|
| 1 | `jkp_replication/main.py` | Build firm characteristics + stock returns | WRDS credentials |
| 2 | `jkp_replication/portfolio_USA_weights.py` | Construct USA factor portfolios + extract constituent weights | Step 1 output |
| 3 | `factor_momentum/arnott_strategy_master.py` | Build Arnott master backtest file (stock-level) | Step 2 output |
| 4 | `factor_momentum/arnott_phase2_full_grid.py` | Run full 30-combination parameter grid | Step 2 output |

### On a Slurm cluster

```sh
# Part A: full JKP replication
sbatch slurm/1_jkp_replication.slurm

# Part A: USA portfolios + weights only (if main.py already ran)
sbatch slurm/2_jkp_usa_weights.slurm

# Part B: Arnott Phase 1
sbatch slurm/3_arnott_phase1.slurm

# Part B: Arnott Phase 2 full grid
sbatch slurm/4_arnott_phase2.slurm
```

### Locally / interactively

```sh
# Step 1
uv run python jkp_replication/main.py

# Step 2
uv run python jkp_replication/portfolio_USA_weights.py

# Step 3
uv run python factor_momentum/arnott_strategy_master.py

# Step 4
ARNOTT_ENV=local uv run python factor_momentum/arnott_phase2_full_grid.py
```

---

## Part A: JKP Replication

### main.py

Orchestrates the full data processing pipeline by calling functions from `aux_functions.py`.

**Pipeline steps:**

1. Load WRDS credentials and connect
2. Set data end date (default: `2024-12-31`)
3. Download ~30 raw WRDS tables (CRSP, Compustat, etc.)
4. Augment the CRSP monthly security file (MSF)
5. Build market cap and trading info (MCTI)
6. Prepare and combine CRSP and Compustat security files
7. Classify industries (CRSP, Compustat, Fama-French 49)
8. Calculate NYSE size breakpoints and return winsorization cutoffs
9. Generate daily and monthly market returns
10. Standardize accounting data
11. Create 100+ firm characteristics
12. Compute rolling metrics (volatility, skewness, beta)
13. Calculate the Quality-Minus-Junk (QMJ) factor
14. Save all output to `data/processed/`

---

### aux_functions.py

Library of 100+ helper functions (~309 KB) supporting `main.py`. Organized by purpose:

- **Setup & I/O:** Folder creation, parquet read/write
- **WRDS loading:** CRSP, Compustat, security file tables
- **Data augmentation:** Merging raw tables into interim frames
- **Characteristic calculation:** 100+ individual factor computations
- **Market data:** Beta, return winsorization, market returns
- **Cleaning:** Delistings, missing data, outlier treatment

---

### portfolio.py

Constructs factor portfolios for all countries in the dataset.

**For each country and each of 160+ characteristics:**

1. Load characteristics parquet, filter by size group / data source / minimum count
2. Compute ECDF-based portfolio assignments (bottom / mid / top terciles)
3. Calculate equal-weighted, value-weighted, and VW-capped returns
4. Create industry portfolios (GICS and Fama-French 49)
5. Generate characteristics-managed portfolios (CMP)
6. Aggregate to long-short (HML), sign by factor direction, cluster, and regional level

**Output:** `data/processed/portfolios/`

---

### portfolio_USA_weights.py *(MODIFIED)*

A USA-only variant of `portfolio.py` with two key additions: stock-level constituent weight extraction and daily portfolio return aggregation. See [Key Design Choices](#key-design-choices) for the rationale.

**Covers 41 selected characteristics** across size, valuation, profitability, momentum, liquidity, quality, risk, investment, and accruals.

**Key output — `usa_factor_weights.parquet`:**

| Column | Type | Description |
|--------|------|-------------|
| `characteristic` | String | Factor name (e.g., `be_me`) |
| `id` | String | PERMNO stock identifier |
| `eom` | Date | Portfolio formation month-end |
| `weight` | Float64 | VW-capped weight within portfolio leg |
| `leg` | Int8 | `+1` = long (top), `-1` = short (bottom) |
| `excntry` | String | Always `USA` |

**All output files:** `pfs.parquet`, `hml.parquet`, `lms.parquet`, `cmp.parquet`, `clusters.parquet`, daily variants, industry portfolios, and `usa_factor_weights.parquet`.

---

### wrds_credentials.py

Manages WRDS authentication.

- Stores **username** in `~/.wrds_user` (not sensitive)
- Stores **password** in the OS keyring via the `keyring` library
- Returns a `Credentials` dataclass consumed by `main.py`

---

## Part B: Factor Momentum

All four files implement the Arnott, Kalesnik & Linnainmaa (2023, RFS) strategy, which applies 1-month factor momentum signals to construct stock-level long-short portfolios.

**1-day implementation lag mechanics (applied throughout):**
- Signal observed at close of month-end `t`
- Trade executes at close of **first trading day** of month `t+1`
- Holding return = full-month gross / day-1 gross − 1

---

### factor_mom_arnott.py

**Inputs:** `data/processed/portfolios/lms_daily.parquet`, `data/processed/return_data/daily_rets_by_country/USA.parquet`

Implements the Arnott factor momentum strategy with a configurable portfolio construction grid:

| Dimension | Options |
|-----------|---------|
| Selection rule | top/bottom 50%, 33%, 25% of factors |
| Position type | long-short or long-only |
| Benchmark | time-series strategy (always included) |

Optional volatility scaling to a 10% annualized target.

**Output:** `data/processed/portfolios/factor_momentum/grid_summary.csv`

---

### arnott_strategy_master.py

**Inputs:** `usa_factor_weights.parquet`, `lms.parquet`, `data/factor_details.xlsx`, `data/processed/return_data/daily_rets_by_country/USA.parquet`

Builds the master backtest file for stock-level analysis:

1. Computes 1-month momentum signal per factor from `lms.parquet`
2. Selects top-25% winner factors each month
3. Joins with stock-level constituent weights from `usa_factor_weights.parquet`
4. Applies sign corrections using `factor_details.xlsx` direction column
5. Nets out each stock's combined position across all selected factors
6. Attaches daily stock returns and adjusts for 1-day execution lag

**Output:** `data/processed/portfolios/arnott_master.parquet`

---

### arnott_phase2_full_grid.py

**Inputs:** `usa_factor_weights.parquet`, daily factor + stock return files

Full 30-combination parameter grid at the stock level, mirroring the Phase 1 R script exactly:

| Parameter | Values |
|-----------|--------|
| Lookback windows | 1m, 2m, 3m, 6m, 12m |
| Strategy types | CS_LS_50/33/25 (long-short), CS_LO_50/33/25 (long-only) |

Auto-detects environment (`ARNOTT_ENV=cluster` for WU HPC, local Mac paths otherwise).

**Outputs (all in `data/processed/phase2_output/`):**
- `phase2_factor_returns.parquet` — factor-level monthly returns for all 30 combinations
- `phase2_stock_weights.parquet` — stock-level netted weights
- `phase2_master.parquet` — weights + returns + market cap + dividend yield (ready for frictions analysis)
- `phase2_summary.csv` / `phase2_stock_summary.csv` — performance tables (Ann. Return, Sharpe)

---

### factor_mom_backtest.py

**Inputs:** `data/processed/portfolios/pfs_daily.parquet`, `data/processed/portfolios/arnott_master.parquet`

Integrated backtest running three components in one script:

| Component | Description | Source | Target |
|-----------|-------------|--------|--------|
| A — Factor-level | Phase 1 baseline from daily factor returns | `pfs_daily.parquet` | ~3.74% Ann. Return, ~0.60 Sharpe |
| B — Stock-level | Phase 2 from master file | `arnott_master.parquet` | monthly weight × return aggregation |
| C — Diagnostics | Schema + quality checks | `arnott_master.parquet` | null rates, weight distribution, coverage |

**Outputs:** `factor_momentum_phase1.parquet`, `factor_momentum_phase2.parquet`, `factor_momentum_report.txt`

---

## Data Reference Files

| File | Purpose |
|------|---------|
| `data/factor_details.xlsx` | Factor names, abbreviations, long-short sign directions |
| `data/country_classification.xlsx` | MSCI country classifications (developed / emerging / frontier) |
| `data/cluster_labels.csv` | Thematic cluster assignments for each factor |

All generated data files (`.parquet`, `.csv`, `.xlsx`, archives) are excluded from version control via `.gitignore`.

---

## Dependencies

Managed via `uv` using `pyproject.toml` (Python 3.11+ required):

| Package | Version | Purpose |
|---------|---------|---------|
| `polars` | 1.34.0 | Primary dataframe library (multithreaded, fast) |
| `polars-ds` | 0.10.2 | Data science extensions for Polars |
| `polars-ols` | 0.3.5 | OLS regression inside Polars expressions |
| `ibis-framework` | 11.0 | Dataframe abstraction layer |
| `duckdb` | 1.4.3 | In-process SQL queries over parquet files |
| `pyarrow` | 16.1.0 | Columnar data I/O |
| `numpy` | 1.26.4 | Numerical computing |
| `tqdm` | 4.66.5 | Progress bars |
| `keyring` | 25.6.0 | Secure OS keyring credential storage |
| `sqlalchemy` | 1.4.49 | Database ORM (WRDS connection) |
| `fastexcel` | >=0.16.0 | Fast Excel file I/O |

---

## Key Design Choices

### Why `portfolio_USA_weights.py` instead of `portfolio.py`

The original `portfolio.py` outputs only aggregate factor returns — it does not retain which stocks were in which portfolio leg or what weight they held. The Arnott strategy requires stock-level positions (PERMNO × month × weight × leg) to net out each stock's combined exposure across multiple factors. `portfolio_USA_weights.py` was created specifically to extract this constituent weight table while keeping the full portfolio construction logic intact. Scope was restricted to USA because the Arnott (2023) analysis covers the US market.

### Why a 1-day implementation lag

Arnott, Kalesnik & Linnainmaa (2023) document that factor momentum profits are substantially reduced once realistic transaction timing is accounted for. Their key finding: if you could trade at the same closing price at which the signal is formed (zero lag), the strategy looks strong; with one day of lag — the earliest realistic execution — returns drop significantly. This project implements their exact mechanics: signal at close of `eom_t`, trade at close of first trading day of month `t+1`, holding return adjusted for the day-1 return drag.

### References

- Jensen, T., Kelly, B., & Pedersen, L. H. (2023). Is there a Replication Crisis in Finance? *Journal of Finance*. Original codebase: [github.com/bkelly-lab/ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis)
- Arnott, R., Kalesnik, V., & Linnainmaa, J. T. (2023). Factor Momentum. *Review of Financial Studies*.
