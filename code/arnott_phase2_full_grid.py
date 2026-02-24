"""
arnott_phase2_full_grid.py
==========================
Phase 2 Replication: Arnott (2023) Factor Momentum → Single-Stock Portfolios

Strictly mirrors the Phase 1 R script (Arnott_Replication_Full_Grid.R) but
decomposes every strategy to individual stock positions using the JKP
constituent weights (usa_factor_weights.parquet).

PARAMETER GRID (matches Phase 1 exactly)
-----------------------------------------
  Lookback windows  :  1m (1 month), 2m, 3m, 6m, 12m
  Strategy types    :
    CS_LS_50   Cross-Sec Long-Short, top/bottom 50%  (median split, Arnott canonical)
    CS_LS_33   Cross-Sec Long-Short, top/bottom 33%
    CS_LS_25   Cross-Sec Long-Short, top/bottom 25%
    CS_LO_50   Cross-Sec Long-Only,  top 50%
    CS_LO_33   Cross-Sec Long-Only,  top 33%
    CS_LO_25   Cross-Sec Long-Only,  top 25%
  Total combinations : 5 × 6 = 30

1-DAY IMPLEMENTATION LAG
--------------------------
  Signal observed at close of eom_t.
  Portfolio traded at CLOSE of first trading day of month t+1 (Day 1).
  Holding period: close(Day1_{t+1}) → close(eom_{t+1})
  Factor lag1 return: full_month_gross / day1_gross - 1

INPUTS  (relative to jkp-data-replication/ working directory)
--------------------------------------------------------------
  ../01_Data/Processed/USA_daily_rets.parquet        Daily L-S factor portfolio returns
  ../01_Data/Processed/usa_factor_weights.parquet    Stock-level factor constituents
  ../01_Data/Processed/USA_stocks_char.parquet       Monthly stock characteristics (me, div_yield, ret)

OUTPUTS
-------
  ../01_Data/Processed/Phase2/phase2_factor_returns.parquet
      Factor-level monthly returns for all 30 strategy-lookback combinations
      (used to validate against Phase 1 R results)

  ../01_Data/Processed/Phase2/phase2_stock_weights.parquet
      Stock-level (PERMNO) netted weights for all 30 combinations
      Columns: strategy, lookback, eom_signal, eom_hold, id, net_weight, gross_weight, n_factors

  ../01_Data/Processed/Phase2/phase2_master.parquet
      Master file: weights + returns + market cap + dividend yield
      Ready for frictions analysis (shorting costs, dividend tax, transaction costs)
      Columns: strategy, lookback, eom_signal, eom_hold, id,
               net_weight, gross_weight, n_factors,
               me, div12m_me, ret_exc_lead1m, ret_exc_lag1

  ../01_Data/Processed/Phase2/phase2_summary.csv
      FACTOR-level performance: Ann_Ret, Ann_Vol, Sharpe (full 1963+ period)

  ../01_Data/Processed/Phase2/phase2_stock_summary.csv
      STOCK-level performance: Ann_Ret, Ann_Vol, Sharpe (months where stock data available)
      Note: N_months < factor-level N because USA_stocks_char.parquet has gaps in early years.
      Missing months have negative factor momentum on average, so stock SR > full-period factor SR.

EXECUTION
---------
  cd jkp-data-replication
  uv run python code/arnott_phase2_full_grid.py
"""

import os
import csv
import time
import warnings
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

# --- Environment detection -------------------------------------------
# Set ENV = "cluster" when running on WU HPC (~/jkp-data/).
# Set ENV = "local"   when running on local Mac (../01_Data/Processed/).
# Can also override via environment variable: ARNOTT_ENV=cluster uv run python ...
import socket
_hostname = socket.gethostname()
ENV = os.environ.get(
    "ARNOTT_ENV",
    "cluster" if ("cluster" in _hostname or "slurm" in _hostname or
                  _hostname.startswith("compute") or _hostname.startswith("node"))
    else "local"
)
print(f"  Environment: {ENV}  (hostname: {_hostname})")

if ENV == "cluster":
    # WU HPC: repo lives at ~/jkp-data/, data inside data/processed/
    _BASE        = os.path.expanduser("~/jkp-data")
    DATA_PATH    = f"{_BASE}/data/processed"
    OUTPUT_PATH  = f"{_BASE}/data/processed/phase2_output"

    DAILY_FACTOR_FILE  = f"{DATA_PATH}/portfolios/lms_daily.parquet"
    STOCK_WEIGHTS_FILE = f"{DATA_PATH}/portfolios/usa_factor_weights.parquet"
    STOCK_CHARS_FILE   = f"{DATA_PATH}/characteristics/USA.parquet"
else:
    # Local Mac: repo lives at jkp-data-replication/, data at ../01_Data/Processed/
    DATA_PATH    = "../01_Data/Processed"
    OUTPUT_PATH  = "../01_Data/Processed/Phase2"

    DAILY_FACTOR_FILE  = f"{DATA_PATH}/USA_daily_rets.parquet"
    STOCK_WEIGHTS_FILE = f"{DATA_PATH}/usa_factor_weights.parquet"
    STOCK_CHARS_FILE   = f"{DATA_PATH}/USA_stocks_char.parquet"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Output files
OUT_FACTOR_RETS   = f"{OUTPUT_PATH}/phase2_factor_returns.parquet"
OUT_STOCK_WEIGHTS = f"{OUTPUT_PATH}/phase2_stock_weights.parquet"
OUT_MASTER        = f"{OUTPUT_PATH}/phase2_master.parquet"
OUT_SUMMARY       = f"{OUTPUT_PATH}/phase2_summary.csv"

# Strategy parameters
START_YEAR   = 1963
EXCNTRY      = "USA"
RET_TYPE     = "ret_vw_cap"  # Value-weighted-capped returns (JKP standard)

# Lookback grid: number of months
LOOKBACK_GRID   = [1, 2, 3, 6, 12]
LOOKBACK_LABELS = {1: "1M", 2: "2M", 3: "3M", 6: "6M", 12: "12M"}

# Cross-sectional strategy grid: (label, top_pct, bot_pct, long_only)
GRID = [
    ("CS_LS_50", 0.50, 0.50, False),   # top 50% long, bottom 50% short (Arnott median)
    ("CS_LS_33", 0.33, 0.33, False),   # top 33% long, bottom 33% short
    ("CS_LS_25", 0.25, 0.25, False),   # top 25% long, bottom 25% short
    ("CS_LO_50", 0.50, 0.00, True),    # top 50% long only
    ("CS_LO_33", 0.33, 0.00, True),    # top 33% long only
    ("CS_LO_25", 0.25, 0.00, True),    # top 25% long only
]
STRATEGY_NAMES = [g[0] for g in GRID]

MONTHS_PER_YEAR = 12

t_start = time.time()
print("=" * 70)
print("  ARNOTT (2023) PHASE 2 — Single-Stock Factor Momentum")
print(f"  Grid: {len(LOOKBACK_GRID)} lookbacks × {len(GRID)} strategies = "
      f"{len(LOOKBACK_GRID)*len(GRID)} combinations")
print(f"  1-Day Implementation Lag | Start Year: {START_YEAR}")
print("=" * 70)


# ============================================================
# STEP 1: Monthly Factor Returns with 1-Day Lag
# ============================================================
print("\n[1/7] Computing 1-day-lagged monthly factor returns...", flush=True)

# Load daily L-S factor portfolio returns (one signed return per factor per day)
daily = (
    pl.scan_parquet(DAILY_FACTOR_FILE)
    .filter(pl.col("excntry") == EXCNTRY)
    .select(["characteristic", "direction", "date", RET_TYPE])
    .rename({RET_TYPE: "daily_ret"})
    # NOTE: ret_vw_cap in USA_daily_rets.parquet (= lms_daily) is ALREADY sign-corrected
    # (positive = factor doing well in its natural / economically-meaningful direction).
    # Do NOT multiply by direction here — that would double-sign direction=-1 factors.
    # Direction correction is applied only at the stock-constituent level via true_leg = leg × direction.
    .with_columns(pl.col("daily_ret").alias("signed_ret"))
    .with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    .sort(["characteristic", "date"])
    # Day rank within each factor-month (1 = first trading day)
    .with_columns(
        pl.col("date")
        .rank(method="ordinal")
        .over(["characteristic", "year", "month"])
        .alias("day_rank")
    )
    .collect()
)

# Extract direction map for later use with stock weights
direction_map = (
    daily.select(["characteristic", "direction"])
    .unique()
    .sort("characteristic")
)

# Aggregate to monthly: full compound return and first-day gross
monthly_factors = (
    daily
    .group_by(["characteristic", "year", "month"])
    .agg(
        (pl.col("signed_ret") + 1).product().alias("full_month_gross"),
        (pl.col("signed_ret") + 1)
        .filter(pl.col("day_rank") == 1)
        .first()
        .alias("day1_gross"),
        # Use calendar end-of-month (not last trading day) so that eom matches
        # the eom column in usa_factor_weights.parquet, which uses calendar month-end.
        # When month-end falls on a weekend, last trading day != calendar EOM → join fails.
        pl.col("date").max().dt.month_end().alias("eom"),  # calendar end-of-month
    )
    # 1-day-lagged return: skip first trading day
    .with_columns(
        (pl.col("full_month_gross") / pl.col("day1_gross") - 1).alias("factor_ret_lag1"),
        (pl.col("full_month_gross") - 1).alias("factor_ret_full"),
    )
    .sort(["characteristic", "year", "month"])
)

n_factors = monthly_factors["characteristic"].n_unique()
print(f"      {len(monthly_factors):,} factor-months | {n_factors} factors")
print(f"      Date range: {monthly_factors['eom'].min()} → {monthly_factors['eom'].max()}")


# ============================================================
# STEP 2: Signals and Forward Returns (All Lookback Periods)
# ============================================================
print("\n[2/7] Building signals for all lookback windows...", flush=True)

# For each factor, shift factor_ret_full back by N months to get N-month signal.
# Forward return = factor_ret_lag1 of next month (what we earn with 1-day lag).

signals_all = {}

for lb in LOOKBACK_GRID:
    lb_label = LOOKBACK_LABELS[lb]

    # For 1-month lookback: signal = prior month full return (shift by 1)
    # For N-month lookback: signal = cumulative log return over N months
    if lb == 1:
        sig_df = (
            monthly_factors
            .sort(["characteristic", "year", "month"])
            .with_columns(
                pl.col("factor_ret_full").shift(1).over("characteristic").alias("signal"),
                pl.col("factor_ret_lag1").shift(-1).over("characteristic").alias("factor_ret_fwd"),
                pl.col("eom").shift(-1).over("characteristic").alias("eom_hold"),
            )
            .drop_nulls(subset=["signal", "factor_ret_fwd"])
            .rename({"eom": "eom_signal"})
        )
    else:
        # N-month rolling cumulative log return as signal
        sig_df = (
            monthly_factors
            .sort(["characteristic", "year", "month"])
            .with_columns(
                pl.col("factor_ret_full").log1p()
                  .rolling_sum(window_size=lb, min_samples=lb)
                  .over("characteristic")
                  .alias("signal_raw")
            )
            # Shift signal by 1 (use signal from prior month, not current)
            .with_columns(
                pl.col("signal_raw").shift(1).over("characteristic").alias("signal"),
                pl.col("factor_ret_lag1").shift(-1).over("characteristic").alias("factor_ret_fwd"),
                pl.col("eom").shift(-1).over("characteristic").alias("eom_hold"),
            )
            .drop_nulls(subset=["signal", "factor_ret_fwd"])
            .rename({"eom": "eom_signal"})
        )

    sig_df = sig_df.filter(pl.col("eom_signal").dt.year() >= START_YEAR)
    signals_all[lb_label] = sig_df
    print(f"      {lb_label}: {len(sig_df):,} obs, "
          f"{sig_df['eom_signal'].min()} → {sig_df['eom_signal'].max()}")


# ============================================================
# STEP 3: Factor-Level Strategy Weights & Returns
# ============================================================
print("\n[3/7] Computing factor-level strategy returns for all combinations...", flush=True)


def build_cs_weights(df: pl.DataFrame, label: str, top: float, bot: float, long_only: bool):
    """
    Cross-sectional strategy weights.
    Rank factors by signal within each eom_signal:
      - Long:  top `top` fraction (pctile > 1-top)
      - Short: bottom `bot` fraction (pctile <= bot)  [ignored if long_only]
    """
    ranked = (
        df
        .with_columns(
            pl.col("signal").rank(method="average").over("eom_signal").alias("rank"),
            pl.col("signal").count().over("eom_signal").alias("n_factors"),
        )
        .with_columns((pl.col("rank") / pl.col("n_factors")).alias("pctile"))
    )

    if long_only:
        selected = (
            ranked
            .with_columns(
                pl.when(pl.col("pctile") > (1 - top))
                  .then(pl.lit(1))
                  .otherwise(pl.lit(None))
                  .alias("fm_side")
            )
            .filter(pl.col("fm_side").is_not_null())
        )
    else:
        selected = (
            ranked
            .with_columns(
                pl.when(pl.col("pctile") > (1 - top)).then(pl.lit(1))
                  .when(pl.col("pctile") <= bot).then(pl.lit(-1))
                  .otherwise(pl.lit(None)).alias("fm_side")
            )
            .filter(pl.col("fm_side").is_not_null())
        )

    # Equal-weight within each leg
    weighted = (
        selected
        .with_columns(
            pl.when(pl.col("fm_side") == 1)
              .then(pl.lit(1.0) / (pl.col("fm_side") == 1).sum().over("eom_signal").cast(pl.Float64))
              .otherwise(pl.lit(1.0) / (pl.col("fm_side") == -1).sum().over("eom_signal").cast(pl.Float64))
              .alias("fm_weight_abs")
        )
        .with_columns((pl.col("fm_side") * pl.col("fm_weight_abs")).alias("fm_weight"))
        .with_columns(pl.lit(label).alias("strategy"))
        # Keep only the columns needed downstream (ensures consistent schema)
        .select([
            "characteristic", "eom_signal", "eom_hold",
            "signal", "fm_side", "fm_weight_abs", "fm_weight",
            "factor_ret_fwd", "strategy",
        ])
    )
    return weighted


# Build returns for every lookback × strategy combination
all_factor_returns = []
all_factor_weights = []

for lb_label, sig_df in signals_all.items():
    for strat_label, top, bot, long_only in GRID:
        key = f"{strat_label}_{lb_label}"

        # Build weights
        wts = build_cs_weights(sig_df, strat_label, top, bot, long_only)
        wts = wts.with_columns(pl.lit(lb_label).alias("lookback"))
        all_factor_weights.append(wts)

        # Aggregate to monthly strategy return
        ret = (
            wts
            .group_by(["strategy", "lookback", "eom_signal", "eom_hold"])
            .agg(
                (pl.col("fm_weight") * pl.col("factor_ret_fwd")).sum().alias("ret"),
                pl.len().alias("n_factors_active"),
            )
            .sort(["strategy", "lookback", "eom_signal"])
        )
        all_factor_returns.append(ret)

factor_returns = pl.concat(all_factor_returns)
factor_weights_all = pl.concat(all_factor_weights)

print(f"      Built {len(factor_returns):,} strategy-month observations")
print(f"      Strategies × lookbacks: {factor_returns['strategy'].n_unique()} × "
      f"{factor_returns['lookback'].n_unique()}")


# ============================================================
# STEP 4: Performance Validation (Factor-Level)
# ============================================================
print("\n[4/7] Computing performance summary (factor-level)...", flush=True)

summary_rows = []

for strat_label in STRATEGY_NAMES:
    for lb_label in [LOOKBACK_LABELS[lb] for lb in LOOKBACK_GRID]:
        sub = (
            factor_returns
            .filter((pl.col("strategy") == strat_label) & (pl.col("lookback") == lb_label))
            .sort("eom_signal")
        )
        if len(sub) < 12:
            continue

        rets = sub["ret"].to_numpy()
        n    = len(rets)

        ann_ret  = float(np.mean(rets)) * MONTHS_PER_YEAR
        ann_vol  = float(np.std(rets, ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        cum_ret  = float(np.prod(1 + rets)) - 1
        cum      = np.cumprod(1 + rets)
        peak     = np.maximum.accumulate(cum)
        max_dd   = float(((cum - peak) / peak).min())
        calmar   = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")

        summary_rows.append({
            "strategy":  strat_label,
            "lookback":  lb_label,
            "n_months":  n,
            "ann_ret":   round(ann_ret, 6),
            "ann_vol":   round(ann_vol, 6),
            "sharpe":    round(sharpe, 4),
            "cum_ret":   round(cum_ret, 6),
            "max_dd":    round(max_dd, 6),
            "calmar":    round(calmar, 4),
        })

# Print grid
print(f"\n  {'Strategy':<12} {'Lookback':<8} {'Sharpe':>7} {'Ann_Ret':>9} {'Ann_Vol':>9} {'Max_DD':>9}")
print("  " + "-" * 58)
for row in summary_rows:
    print(f"  {row['strategy']:<12} {row['lookback']:<8} "
          f"{row['sharpe']:>7.2f} {row['ann_ret']:>8.2%} "
          f"{row['ann_vol']:>8.2%} {row['max_dd']:>8.2%}")

# Save CSV
with open(OUT_SUMMARY, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\n  Saved: {OUT_SUMMARY}")

# Save factor-level returns parquet
factor_returns.write_parquet(OUT_FACTOR_RETS)
print(f"  Saved: {OUT_FACTOR_RETS}  ({os.path.getsize(OUT_FACTOR_RETS)/1e6:.1f} MB)")


# ============================================================
# STEP 5: Stock Decomposition
# ============================================================
print("\n[5/7] Decomposing to single stocks...", flush=True)

# Load stock-level factor constituent weights
stock_weights_raw = (
    pl.read_parquet(STOCK_WEIGHTS_FILE)
    .filter(pl.col("excntry") == EXCNTRY)
    # Apply direction sign correction: true_leg = leg × direction
    .join(direction_map, on="characteristic", how="left")
    .with_columns((pl.col("leg") * pl.col("direction")).alias("true_leg"))
    # Add year/month for joining with holding period
    .with_columns(
        pl.col("eom").dt.year().alias("year_signal"),
        pl.col("eom").dt.month().alias("month_signal"),
    )
    # Next month (holding period) year/month
    .with_columns(
        pl.when(pl.col("month_signal") == 12)
          .then(pl.col("year_signal") + 1)
          .otherwise(pl.col("year_signal"))
          .alias("year_hold"),
        pl.when(pl.col("month_signal") == 12)
          .then(pl.lit(1))
          .otherwise(pl.col("month_signal") + 1)
          .alias("month_hold"),
    )
)

n_stocks_total = stock_weights_raw["id"].n_unique()
print(f"      Stock weight file: {len(stock_weights_raw):,} rows | "
      f"{n_stocks_total:,} unique stocks | "
      f"{stock_weights_raw['characteristic'].n_unique()} factors")
print(f"      Date range: {stock_weights_raw['eom'].min()} → {stock_weights_raw['eom'].max()}")

# Decompose for every strategy × lookback combo
all_netted = []

for lb_label in [LOOKBACK_LABELS[lb] for lb in LOOKBACK_GRID]:
    # Get factor weights for this lookback
    fw_lb = factor_weights_all.filter(pl.col("lookback") == lb_label)

    for strat_label in STRATEGY_NAMES:
        fw = (
            fw_lb
            .filter(pl.col("strategy") == strat_label)
            .select(["characteristic", "eom_signal", "eom_hold", "fm_weight", "fm_side"])
        )

        if len(fw) == 0:
            continue

        # Join factor weights to stock constituent weights
        # eom (formation month) in stock_weights = eom_signal in factor_weights
        detail = (
            stock_weights_raw
            .join(fw, left_on=["characteristic", "eom"], right_on=["characteristic", "eom_signal"],
                  how="inner")
            # Stock contribution to strategy: fm_weight × signed_leg × constituent_weight
            # fm_weight   = ±1/N_factors (positive = long factor, negative = short factor)
            # true_leg    = ±1 (direction-corrected: +1 means this stock is on the "good" side of factor)
            # weight      = stock's share within its leg (value-weighted-capped)
            .with_columns(
                (pl.col("fm_weight") * pl.col("true_leg") * pl.col("weight"))
                .alias("stock_fm_contrib")
            )
        )

        # Net across factors per (eom_signal, eom_hold, stock)
        netted = (
            detail
            .group_by(["eom", "eom_hold", "id"])
            .agg(
                pl.col("stock_fm_contrib").sum().alias("net_weight"),
                pl.col("stock_fm_contrib").abs().sum().alias("gross_weight"),
                pl.len().alias("n_factors"),
            )
            .rename({"eom": "eom_signal"})
            .with_columns(
                pl.lit(strat_label).alias("strategy"),
                pl.lit(lb_label).alias("lookback"),
            )
            .filter(pl.col("eom_signal").dt.year() >= START_YEAR)
        )

        # Normalize net weights so gross exposure = 1.0 per month
        # (ensures returns are correctly scaled)
        netted = netted.with_columns(
            (pl.col("net_weight") / pl.col("net_weight").abs().sum().over("eom_signal"))
            .alias("net_weight"),
        )

        all_netted.append(netted)

all_netted_df = pl.concat(all_netted)

n_combos = all_netted_df.select(["strategy", "lookback"]).unique().height
print(f"      Decomposed {n_combos} strategy×lookback combinations")
print(f"      Total stock-month-strategy rows: {len(all_netted_df):,}")

# Save stock-level weights
all_netted_df.write_parquet(OUT_STOCK_WEIGHTS)
print(f"  Saved: {OUT_STOCK_WEIGHTS}  ({os.path.getsize(OUT_STOCK_WEIGHTS)/1e6:.1f} MB)")


# ============================================================
# STEP 6: Attach Stock Characteristics (Returns, ME, Dividend Yield)
# ============================================================
print("\n[6/7] Attaching stock characteristics (returns, ME, dividends)...", flush=True)

# Load only the columns needed for friction analysis
# eom = end of signal month (formation date)
# ret_exc_lead1m = full next-month excess return (holding period return, no 1-day lag)
# me = market equity at formation (USD millions)
# div12m_me = 12-month trailing dividend yield
stock_chars = (
    pl.scan_parquet(STOCK_CHARS_FILE)
    .select(["id", "eom", "me", "ret_exc_lead1m", "div12m_me"])
    .filter(pl.col("eom").dt.year() >= START_YEAR)
    .collect()
)

print(f"      Stock chars loaded: {len(stock_chars):,} rows | "
      f"{stock_chars['id'].n_unique():,} unique stocks")

# Join stock characteristics to netted weights
# eom_signal (weights) = eom (stock chars) — formation month
master = (
    all_netted_df
    .join(
        stock_chars.rename({"eom": "eom_signal"}),
        on=["id", "eom_signal"],
        how="left",
    )
)

null_ret  = master["ret_exc_lead1m"].is_null().sum()
null_me   = master["me"].is_null().sum()
null_div  = master["div12m_me"].is_null().sum()
total_rows = len(master)

print(f"      Merge results ({total_rows:,} rows):")
print(f"        ret_exc_lead1m nulls: {null_ret:,} ({100*null_ret/total_rows:.1f}%)")
print(f"        me nulls            : {null_me:,} ({100*null_me/total_rows:.1f}%)")
print(f"        div12m_me nulls     : {null_div:,} ({100*null_div/total_rows:.1f}%)  "
      f"← expected (many stocks don't pay dividends)")

# ---- Sanity check for CS_LS_50 1M ----
print("\n      Sanity check — stock-level vs factor-level returns (CS_LS_50, 1M):")
check_strat = "CS_LS_50"
check_lb    = "1M"

stock_rets_chk = (
    master
    .filter(
        (pl.col("strategy") == check_strat)
        & (pl.col("lookback") == check_lb)
        & pl.col("ret_exc_lead1m").is_not_null()
    )
    .group_by("eom_signal")
    .agg((pl.col("net_weight") * pl.col("ret_exc_lead1m")).sum().alias("stock_ret"))
)
factor_rets_check = (
    factor_returns
    .filter((pl.col("strategy") == check_strat) & (pl.col("lookback") == check_lb))
    .select(["eom_signal", "ret"])
)
cmp = stock_rets_chk.join(factor_rets_check, on="eom_signal", how="inner")
if len(cmp) > 10:
    corr = float(np.corrcoef(cmp["stock_ret"].to_numpy(), cmp["ret"].to_numpy())[0, 1])
    flag = "*** WARNING: Low correlation — check direction/sign logic ***" if corr < 0.90 else "✓ Good alignment"
    print(f"        Pearson correlation (stock vs factor return): {corr:.4f}  {flag}")

    r_chk = stock_rets_chk.sort("eom_signal")["stock_ret"].to_numpy()
    ar_s = float(np.mean(r_chk)) * 12
    av_s = float(np.std(r_chk, ddof=1)) * np.sqrt(12)
    sr_s = ar_s / av_s if av_s > 0 else float("nan")
    print(f"        Stock-level   | Ann Ret: {ar_s:+.2%}  Ann Vol: {av_s:.2%}  Sharpe: {sr_s:.2f}  N={len(r_chk)}")
    factor_row = next((x for x in summary_rows if x["strategy"]==check_strat and x["lookback"]==check_lb), None)
    if factor_row:
        print(f"        Factor-level  | Ann Ret: {factor_row['ann_ret']:+.2%}  Ann Vol: {factor_row['ann_vol']:.2%}  Sharpe: {factor_row['sharpe']:.2f}  N={factor_row['n_months']}")
        print(f"        NOTE: Factor N={factor_row['n_months']} > Stock N={len(r_chk)} because")
        print(f"              {factor_row['n_months']-len(r_chk)} months lack stock characteristic data.")
        print(f"              Factor SR for same {len(r_chk)} months is typically higher than full-period SR.")

# ---- Stock-level performance summary for ALL strategy × lookback combos ----
print("\n      Computing stock-level performance for all 30 strategy×lookback combinations...")

stock_summary_rows = []
for lb_label in [LOOKBACK_LABELS[lb] for lb in LOOKBACK_GRID]:
    for strat_label in STRATEGY_NAMES:
        stock_pnl = (
            master
            .filter(
                (pl.col("strategy") == strat_label)
                & (pl.col("lookback") == lb_label)
                & pl.col("ret_exc_lead1m").is_not_null()
            )
            .group_by("eom_signal")
            .agg((pl.col("net_weight") * pl.col("ret_exc_lead1m")).sum().alias("ret"))
            .sort("eom_signal")
        )
        if len(stock_pnl) < 12:
            continue
        r = stock_pnl["ret"].to_numpy()
        n = len(r)
        ann_ret  = float(np.mean(r)) * MONTHS_PER_YEAR
        ann_vol  = float(np.std(r, ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        cum_ret  = float(np.prod(1 + r)) - 1
        cum      = np.cumprod(1 + r)
        peak     = np.maximum.accumulate(cum)
        max_dd   = float(((cum - peak) / peak).min())
        calmar   = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")
        stock_summary_rows.append({
            "strategy":  strat_label,
            "lookback":  lb_label,
            "n_months":  n,
            "ann_ret":   round(ann_ret, 6),
            "ann_vol":   round(ann_vol, 6),
            "sharpe":    round(sharpe, 4),
            "cum_ret":   round(cum_ret, 6),
            "max_dd":    round(max_dd, 6),
            "calmar":    round(calmar, 4),
        })

# Print stock-level grid
print(f"\n  {'Strategy':<12} {'Lookback':<8} {'Sharpe':>7} {'Ann_Ret':>9} {'Ann_Vol':>9} {'N':>6}  (stock-level, ret_exc_lead1m)")
print("  " + "-" * 60)
for row in stock_summary_rows:
    print(f"  {row['strategy']:<12} {row['lookback']:<8} "
          f"{row['sharpe']:>7.2f} {row['ann_ret']:>8.2%} "
          f"{row['ann_vol']:>8.2%} {row['n_months']:>6}")

# Save stock-level summary
OUT_STOCK_SUMMARY = f"{OUTPUT_PATH}/phase2_stock_summary.csv"
if stock_summary_rows:
    with open(OUT_STOCK_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stock_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stock_summary_rows)
    print(f"\n  Saved stock-level summary: {OUT_STOCK_SUMMARY}")

# Save master file
master.write_parquet(OUT_MASTER)
print(f"\n  Saved: {OUT_MASTER}  ({os.path.getsize(OUT_MASTER)/1e6:.1f} MB)")


# ============================================================
# STEP 7: Final Report
# ============================================================
print("\n[7/7] Final summary...", flush=True)

elapsed = time.time() - t_start

print(f"""
{'='*70}
  PHASE 2 COMPLETE  ({elapsed:.0f}s)
{'='*70}

  INPUT DATA
    Factors        : {n_factors} (JKP USA, direction-corrected)
    Stocks         : {n_stocks_total:,} unique PERMNOs
    Sample period  : {START_YEAR} onward

  PARAMETER GRID
    Lookback windows : {', '.join(LOOKBACK_LABELS[lb] for lb in LOOKBACK_GRID)}
    Strategy types   : {', '.join(STRATEGY_NAMES)}
    Total combos     : {len(LOOKBACK_GRID)*len(STRATEGY_NAMES)}

  IMPLEMENTATION
    Signal lag : 1-day (skip first trading day of holding month)
    Rebalancing: Monthly (signal observed at eom_t, trade at day1_t+1)
    Weights    : Equal-weight within each leg, normalized to gross=1.0

  OUTPUT FILES
    {OUT_FACTOR_RETS}
      ↳ Factor-level monthly returns for all 30 combos (validate vs Phase 1)
    {OUT_STOCK_WEIGHTS}
      ↳ Stock-level netted weights (PERMNO, eom_signal, net_weight, gross_weight, n_factors)
    {OUT_MASTER}
      ↳ MASTER FILE: weights + me + div12m_me + ret_exc_lead1m
         Ready for friction analysis (shorting costs, dividend tax, transaction costs)
    {OUT_SUMMARY}
      ↳ Performance CSV (Ann_Ret, Ann_Vol, Sharpe, Max_DD, Calmar)

  MASTER FILE SCHEMA (phase2_master.parquet)
    strategy         : CS_LS_50 | CS_LS_33 | CS_LS_25 | CS_LO_50 | CS_LO_33 | CS_LO_25
    lookback         : 1M | 2M | 3M | 6M | 12M
    eom_signal       : end of signal month (portfolio formation date, PERMNO observed here)
    eom_hold         : end of holding month (unwind date)
    id               : PERMNO (stock identifier)
    net_weight       : netted signed portfolio weight (+long, -short)
    gross_weight     : sum of |factor contributions| before netting (≥ |net_weight|)
    n_factors        : number of factors this stock appears in this month
    me               : market equity at formation (USD millions, for position sizing)
    div12m_me        : trailing 12-month dividend yield (for dividend tax modeling)
    ret_exc_lead1m   : full holding-month excess return (for performance/friction calculation)

  NEXT STEPS (FRICTION ANALYSIS)
    1. Shorting costs  : apply to rows where net_weight < 0
       Cost ≈ annual_borrow_rate × |net_weight| / 12 per month
    2. Dividend tax     : apply to long positions in dividend-paying stocks
       div_tax_drag ≈ tax_rate × div12m_me × net_weight / 12 per month
    3. Transaction costs: apply to |Δweight| at each rebalancing date
       tc_drag ≈ round_trip_cost × sum(|w_t - w_t-1|) / 2 per month
{'='*70}
""")
