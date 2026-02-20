# Global Factor, Stock, and Firm Data

This repository contains Python code to generate a global dataset of factor returns, stock returns, and firm characteristics from ["Is there a Replication Crisis in Finance?"](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249) by Jensen, Kelly, and Pedersen (Journal of Finance, 2023).

The codebase is a full Python migration of the original SAS/R pipeline. It downloads raw financial data from WRDS (Wharton Research Data Services), processes 100+ firm characteristics, constructs long-short factor portfolios, and optionally outputs individual stock-level constituent weights for US portfolios.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites & Setup](#prerequisites--setup)
4. [Running the Pipeline](#running-the-pipeline)
5. [Code Files](#code-files)
   - [main.py](#mainpy)
   - [aux_functions.py](#aux_functionspy)
   - [portfolio.py](#portfoliopy)
   - [portfolio_USA_weights.py](#portfolio_usa_weightspy)
   - [wrds_credentials.py](#wrds_credentialspy)
6. [Slurm Scripts](#slurm-scripts)
7. [Output Files](#output-files)
8. [Data Organization](#data-organization)
9. [Dependencies](#dependencies)
10. [Notes](#notes)

---

## Project Overview

The pipeline generates three main outputs:

| Output | Description |
|--------|-------------|
| **Firm characteristics** | 100+ monthly accounting and market-based metrics per stock |
| **Stock returns** | Daily and monthly returns with supporting data |
| **Factor returns** | Long-short portfolio returns derived from the characteristics |

Data is sourced from WRDS and spans global equity markets. The pipeline is designed to run on a high-performance computing cluster, requesting **128 CPU cores and 450 GB RAM**, and takes approximately **6 hours** to complete.

---

## Repository Structure

```
jkp-data/
├── code/                         # Main Python scripts
│   ├── main.py                   # Full data pipeline orchestration
│   ├── aux_functions.py          # 100+ helper functions
│   ├── portfolio.py              # Portfolio construction (all countries)
│   ├── portfolio_USA_weights.py  # Portfolio construction + USA weights
│   └── wrds_credentials.py       # Secure WRDS credential management
├── data/
│   ├── factor_details.xlsx       # Factor names, abbreviations, directions
│   ├── country_classification.xlsx  # MSCI country classifications
│   ├── cluster_labels.csv        # Factor cluster assignments
│   ├── raw/                      # Downloaded WRDS tables
│   ├── interim/                  # Intermediate processed files
│   └── processed/                # Final output (characteristics, portfolios)
├── slurm/                        # Slurm HPC job submission scripts
├── documentation/
│   ├── release_notes.qmd         # Quarto markdown source
│   └── release_notes.html        # HTML documentation & comparison plots
├── logs/                         # Execution logs
├── pyproject.toml                # Python dependencies (uv-based)
└── recover.py                    # Git blob recovery utility
```

---

## Prerequisites & Setup

### Requirements

- A valid **WRDS account** with access to CRSP and Compustat
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) installed on your system (used for dependency management)
- A machine with sufficient resources: **128 CPU cores, 450 GB RAM** recommended for the full pipeline

### Setup Steps

**1. Clone the repository**

```sh
git clone git@github.com:bkelly-lab/jkp-data.git
cd jkp-data
```

**2. Store your WRDS credentials**

```sh
uv run python code/wrds_credentials.py
```

Follow the prompts to enter your username and password. Credentials are stored securely using the OS keyring — never written to disk in plaintext.

To reset stored credentials:

```sh
uv run python code/wrds_credentials.py --reset
uv run python code/wrds_credentials.py
```

---

## Running the Pipeline

### On a Slurm Cluster (recommended)

```sh
sbatch slurm/submit_job_som_hpc.slurm
```

This runs `main.py` followed by `portfolio.py` using 128 CPUs and 450 GB RAM.

To run only the portfolio construction step (e.g., after characteristics already exist):

```sh
sbatch slurm/jkp_portfolio_only.slurm
```

### Interactive / Local Session

```sh
# Step 1: Build characteristics and stock returns (~6 hours)
uv run python code/main.py

# Step 2: Construct factor portfolios (all countries)
uv run python code/portfolio.py

# OR: Construct USA portfolios with constituent weights
uv run python code/portfolio_USA_weights.py
```

> **Important:** When starting the pipeline, WRDS may prompt for two-factor authentication (e.g., a Duo push notification). You must approve this within a few seconds or the script will fail. You should see files appearing in `data/raw/` shortly after approval.

---

## Code Files

### main.py

The orchestration script. It drives the entire data processing pipeline by calling a sequence of functions from `aux_functions.py`.

**Pipeline steps:**

1. Load WRDS credentials
2. Set data end date (default: `2024-12-31`)
3. Download ~30 raw WRDS data tables (CRSP, Compustat, etc.)
4. Augment the CRSP monthly security file (MSF)
5. Build market cap and trading info (MCTI)
6. Prepare CRSP and Compustat security files
7. Combine CRSP and Compustat data
8. Classify industries (CRSP, Compustat, Fama-French 49)
9. Calculate NYSE size breakpoints
10. Calculate return winsorization cutoffs
11. Generate daily and monthly market returns
12. Standardize accounting data
13. Create 100+ firm characteristics (accounting and market-based)
14. Compute rolling window metrics (volatility, skewness, beta, etc.)
15. Calculate the Quality-Minus-Junk (QMJ) factor
16. Save all processed data to `data/processed/`

To change the end date, edit line 4 of `main.py`:

```python
end_date = pl.datetime(2024, 12, 31)
```

---

### aux_functions.py

A library of 100+ helper functions (~309 KB) supporting `main.py`. Functions are grouped by purpose:

- **Setup & I/O:** Folder structure, parquet read/write helpers
- **WRDS data loading:** Pulling CRSP, Compustat, and security-file tables
- **Data augmentation:** Merging raw tables into interim frames
- **Characteristic calculation:** 100+ individual factor computations
- **Market data:** Beta estimation, return winsorization, market returns
- **Cleaning:** Delistings, missing data handling, outlier treatment (CRSP vs. Compustat)

---

### portfolio.py

Constructs factor portfolios for all countries in the dataset.

**For each country and each of 160+ characteristics:**

1. Load the characteristics parquet file
2. Filter by size group, data source, and minimum stock count
3. Compute ECDF-based portfolio assignments (bottom / mid / top terciles by default)
4. Calculate equal-weighted, value-weighted, and value-weighted-capped returns
5. Create industry portfolios (GICS and Fama-French 49)
6. Generate characteristics-managed portfolios (CMP)

**Aggregation steps:**

- Create long-short (HML) returns (top minus bottom)
- Sign returns by factor direction using `data/factor_details.xlsx`
- Generate cluster-level portfolio returns
- Build regional portfolios: Developed ex-US, Emerging, Frontier, World, World ex-US

**Output:** All results written to `data/processed/portfolios/`

---

### portfolio_USA_weights.py

A specialized variant of `portfolio.py` that focuses exclusively on **USA stocks** and additionally extracts **individual stock-level constituent weights** for each factor portfolio. This enables full portfolio replication.

#### What makes it different

While `portfolio.py` outputs aggregate returns, `portfolio_USA_weights.py` captures, for each stock in the top and bottom portfolios of each factor: its identifier, the month, its value-weighted-capped weight, and which leg (long or short) it belongs to.

#### Characteristics covered

The script processes **41 carefully selected characteristics** covering:

| Category | Examples |
|----------|---------|
| Size & age | `market_equity`, `age` |
| Valuation | `be_me`, `sale_me`, `ebitda_mev`, `at_me` |
| Profitability | `gp_at`, `ope_be`, `ebit_sale`, `niq_be` |
| Momentum | `ret_12_1`, `ret_1_0`, `ret_60_12` |
| Liquidity | `ami_126d`, `dolvol_126d` |
| Quality | `f_score`, `o_score`, `z_score`, `qmj` |
| Risk | `beta_60m`, `ivol_ff3_21d`, `betabab_1260d` |
| Investment | `at_gr1`, `capex_abn`, `inv_gr1` |
| Accruals | `oaccruals_at` |
| Other | 20+ further accounting and market metrics |

#### Configuration

The `settings` dictionary at the top of the script controls key parameters:

| Setting | Value | Description |
|---------|-------|-------------|
| `end_date` | `2025-12-31` | Last date included |
| `n_portfolios` | `3` | Number of portfolio buckets (bottom / mid / top) |
| `bp_min_n` | `10` | Minimum stocks required to form breakpoints |
| `bp_stocks` | `"non_mc"` | Breakpoint universe (non-microcap stocks) |
| `daily_pf` | `True` | Also compute daily portfolio returns |
| `cmp` | `True` | Compute characteristics-managed portfolios |

#### Detailed step-by-step logic

**1. Data loading**

Loads the USA characteristics parquet file and selects only the 41 relevant columns plus stock identifiers and industry codes.

**2. Data preparation**

- Caps market equity at the NYSE 80th percentile
- Converts all numeric columns to `Float64`
- Filters by data source (CRSP, Compustat, or both)
- Drops rows missing: size group, market equity, or forward returns

**3. Return winsorization**

Winsorizes extreme returns at the 0.1st and 99.9th percentiles. Applied only to Compustat-sourced observations (`source_crsp == 0`). If daily returns are also requested, loads daily return cutoffs for the same treatment.

**4. ECDF-based portfolio assignment**

For each characteristic and each month:

- Identify the breakpoint stocks (non-microcap universe with at least 10 observations)
- Compute the empirical CDF (ECDF) within breakpoint stocks
- Multiply ECDF by the number of portfolios (3) and assign each stock to portfolio 1, 2, or 3
- This ensures consistent tercile assignment relative to the non-microcap universe

**5. Constituent weight extraction (key feature)**

After portfolio assignment, for stocks in the top (portfolio 3) and bottom (portfolio 1):

- Calculate each stock's weight = its capped market equity / sum of capped market equity in the same portfolio-month
- Assign a leg direction: `+1` for long (top), `-1` for short (bottom)
- Store the record as `(characteristic, stock_id, month, weight, leg)`

This weight table accumulates across all 41 characteristics and is written to `usa_factor_weights.parquet` at the end.

**6. Portfolio return calculation**

Within each portfolio-month group, calculate:

- Number of constituent stocks
- Median characteristic value (as a signal proxy)
- Equal-weighted return
- Value-weighted return
- Value-weighted-capped return (using capped market equity)

Month-end dates are shifted forward by one month to align portfolio formation with the subsequent holding-period return.

**7. Daily portfolio returns**

If `daily_pf = True`, portfolio constituent weights from month-end are joined to daily returns for each trading day within the holding period. Daily portfolio returns are aggregated at the portfolio-day level.

**8. Industry portfolios**

Calculates both GICS-based and Fama-French 49 industry portfolio returns (equal-weighted and value-weighted) with a minimum of 10 stocks per group per month.

**9. Long-short (HML) construction**

Top portfolio return minus bottom portfolio return, giving a raw long-short factor return. Returns are then **signed** using the `direction` column from `data/factor_details.xlsx`:

- `+1`: higher characteristic values predict outperformance → keep sign
- `-1`: lower characteristic values predict outperformance → flip sign

**10. Clustering**

Factors are grouped into thematic clusters based on `data/cluster_labels.csv`. Cluster-level returns are computed as equal-weighted averages of their constituent factor long-short returns.

**11. Regional aggregation**

Builds five regional factor sets by weighting country-level returns by market capitalization:

- Developed markets (ex-US)
- Emerging markets
- Frontier markets
- World (all countries)
- World ex-US

**12. Output writing**

Files written to `data/processed/portfolios/`:

| File | Contents |
|------|---------|
| `pfs.parquet` | All portfolio returns (by factor, portfolio number, month) |
| `hml.parquet` | Raw long-short returns |
| `lms.parquet` | Signed long-short returns |
| `cmp.parquet` | Characteristics-managed portfolio returns |
| `clusters.parquet` | Cluster-level returns |
| `pfs_daily.parquet` | Daily portfolio returns |
| `hml_daily.parquet` | Daily long-short returns |
| `lms_daily.parquet` | Daily signed long-short returns |
| `clusters_daily.parquet` | Daily cluster returns |
| `industry_gics.parquet` | GICS industry portfolio returns |
| `industry_ff49.parquet` | Fama-French 49 industry returns |
| `usa_factor_weights.parquet` | **Stock-level constituent weights for all 41 USA factors** |
| `regional_factors/` | Regional aggregations by factor and cluster |
| `country_factors/` | Country-level factor returns |

#### usa_factor_weights.parquet schema

| Column | Type | Description |
|--------|------|-------------|
| `characteristic` | String | Factor name (e.g., `be_me`) |
| `id` | String | Stock identifier (PERMNO or similar) |
| `eom` | Date | End-of-month date (portfolio formation month) |
| `weight` | Float64 | Value-weighted-capped weight within the portfolio leg |
| `leg` | Int8 | `+1` = long leg (top portfolio), `-1` = short leg (bottom portfolio) |
| `excntry` | String | Country code (`USA`) |

---

### wrds_credentials.py

Manages WRDS authentication securely.

- Stores the **username** in `~/.wrds_user` (plain text, not sensitive)
- Stores the **password** in the OS keyring (never on disk)
- Uses the `keyring` library for cross-platform secure storage
- Returns a `Credentials` dataclass consumed by `main.py`

---

## Slurm Scripts

Located in `slurm/`. All scripts use `uv run python` for execution.

| Script | Purpose | CPUs | RAM | Est. Time |
|--------|---------|------|-----|-----------|
| `submit_job_som_hpc.slurm` | Full pipeline: `main.py` + `portfolio.py` | 128 | 450 GB | ~6 hours |
| `jkp_replication.slurm` | Full replication | 64 | 450 GB | ~48 hours |
| `jkp_portfolio_only.slurm` | Portfolio only (`portfolio_USA_weights.py`) | 64 | 450 GB | ~12 hours |
| `jkp_safe.slurm` | Partial pipeline with checkpoints | 64 | 450 GB | ~24 hours |
| `jkp_factor_mom.slurm` | Momentum factor specific | 32 | 200 GB | ~4 hours |

---

## Output Files

All final output lives in `data/processed/`:

```
data/processed/
├── characteristics/          # Per-country firm characteristics (parquet)
├── return_data/
│   └── daily_rets_by_country/  # Daily stock returns by country
├── accounting_data/          # Standardized accounting data
├── other_output/             # NYSE cutoffs, market returns, winsorization cutoffs
└── portfolios/
    ├── pfs.parquet           # All portfolio returns
    ├── hml.parquet           # Raw long-short returns
    ├── lms.parquet           # Signed long-short returns
    ├── cmp.parquet           # Characteristics-managed portfolios
    ├── clusters.parquet      # Cluster returns
    ├── pfs_daily.parquet     # Daily portfolio returns
    ├── hml_daily.parquet     # Daily long-short returns
    ├── lms_daily.parquet     # Daily signed long-short
    ├── clusters_daily.parquet
    ├── industry_gics.parquet
    ├── industry_ff49.parquet
    ├── usa_factor_weights.parquet   # USA stock-level weights
    ├── country_factors/
    └── regional_factors/
```

See `documentation/release_notes.html` for a full description of each output file and a comparison of the Python output against the original SAS/R codebase.

---

## Data Organization

Reference data files tracked in git:

| File | Purpose |
|------|---------|
| `data/factor_details.xlsx` | Factor names, abbreviations, long-short sign directions |
| `data/country_classification.xlsx` | MSCI country classifications (developed / emerging / frontier) |
| `data/cluster_labels.csv` | Thematic cluster assignments for each factor |

All generated data files (`.parquet`, `.csv`, `.xlsx`, database files, archives) are excluded from version control via `.gitignore`.

---

## Dependencies

Managed via `uv` using `pyproject.toml`:

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

Python 3.11+ is required.

---

## Notes

- **End date:** The default data end date is `2024-12-31`. To change it, edit line 4 of `code/main.py`:
  ```python
  end_date = pl.datetime(1992, 5, 6)  # example
  ```
- **Factor documentation:** Full variable definitions are in the [JKP documentation PDF](https://jkpfactors.s3.amazonaws.com/documents/Documentation.pdf).
- **Data distribution:** Factor returns are available at [jkpfactors.com](https://jkpfactors.com). Stock returns and firm characteristics are distributed via [WRDS](https://wrds-www.wharton.upenn.edu/pages/get-data/contributed-data-forms/global-factor-data/).
- **Original SAS/R codebase:** Still available at [github.com/bkelly-lab/ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis), but this Python codebase is recommended for future work.
- **Polars vs Pandas:** The pipeline uses Polars instead of Pandas because it is natively multithreaded and significantly faster on large datasets, which is critical given the scale of the data.
