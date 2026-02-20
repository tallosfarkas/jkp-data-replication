"""
factor_momentum_backtest.py
============================
Python equivalent of 04_real_world_factor_momentum_implementation.R

Runs THREE things in one script:

  [A] FACTOR-LEVEL BACKTEST  (Phase 1 baseline, from pfs_daily.parquet)
      Replicates R Section 1 exactly:
        - Daily factor returns from pfs_daily (VW_CAP L-S)
        - Sign corrections (hardcoded FACTORS_TO_FLIP list)
        - 1-month momentum signal
        - Top 50% Long / Bottom 50% Short, dollar-neutral
        - 1-month signal lag + 1-day execution lag (Day 1 = 0 weight)
        Target: Ann. Return ~3.74%, Sharpe ~0.60

  [B] STOCK-LEVEL BACKTEST   (Phase 2, from arnott_master.parquet)
      Uses the pre-built master file to compute:
        - Monthly strategy returns (weight × ret_exc_lead1m)
        - Implementation-lag-adjusted returns (subtract Day-1 drag)
        - Ann. Return, Sharpe, Max DD, Monthly Turnover

  [C] DIAGNOSTIC REPORT      (arnott_master.parquet schema + quality checks)
      Prints column list, dtypes, null rates, weight distribution,
      and a month-by-month coverage check.

Output files (saved to OUTPUT_PATH):
  factor_momentum_phase1.parquet   ← factor-level daily returns
  factor_momentum_phase2.parquet   ← stock-level monthly returns
  factor_momentum_report.txt       ← full text report
"""

import polars as pl
import numpy as np
import os
import time
from datetime import date

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH    = "data/processed"
OUTPUT_PATH  = "data/processed/portfolios"
PFS_DAILY    = f"{OUTPUT_PATH}/pfs_daily.parquet"
MASTER_FILE  = f"{OUTPUT_PATH}/arnott_master.parquet"

START_DATE   = date(1963, 1, 1)
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR       = 12

FACTORS_TO_FLIP = {
    "betabab_1260d", "market_equity", "beta_60m", "ivol_ff3_21d",
    "age", "prc", "ret_1_0", "sale_gr1", "at_gr1", "at_be",
    "ret_60_12", "chcsho_12m", "netis_at", "o_score", "capx_gr1",
    "oaccruals_at", "dolvol_126d", "inv_gr1", "cowc_gr1a",
    "capex_abn", "dbnetis_at", "noa_at",
}


# =============================================================================
# HELPERS
# =============================================================================

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def performance_stats(returns: pl.Series, freq: str = "daily") -> dict:
    """Ann. return, vol, Sharpe, max drawdown from a return series."""
    r = returns.drop_nulls().to_numpy()
    if len(r) == 0:
        return {}
    scale = TRADING_DAYS_PER_YEAR if freq == "daily" else MONTHS_PER_YEAR
    ann_ret  = (1 + r).prod() ** (scale / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(scale)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = np.cumprod(1 + r)
    peak     = np.maximum.accumulate(cum)
    max_dd   = ((cum - peak) / peak).min()
    return {"Ann_Return": ann_ret, "Ann_Vol": ann_vol,
            "Sharpe": sharpe, "Max_DD": max_dd, "N_obs": len(r)}

def fmt(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if k in ("Ann_Return", "Ann_Vol", "Max_DD"):
            lines.append(f"  {k:<14}: {v*100:+.2f}%")
        elif k == "Sharpe":
            lines.append(f"  {k:<14}: {v:.3f}")
        else:
            lines.append(f"  {k:<14}: {v}")
    return "\n".join(lines)


# =============================================================================
# [C] DIAGNOSTIC — arnott_master.parquet
# =============================================================================

def diagnose_master(report_lines: list) -> None:
    ts("[C] Diagnosing arnott_master.parquet...")
    sep = "=" * 65

    if not os.path.exists(MASTER_FILE):
        msg = f"  FILE NOT FOUND: {MASTER_FILE}"
        print(msg); report_lines.append(msg)
        return

    df = pl.read_parquet(MASTER_FILE)

    lines = [
        sep,
        "  ARNOTT MASTER FILE — SCHEMA & QUALITY REPORT",
        sep,
        f"  Path        : {MASTER_FILE}",
        f"  Size on disk: {os.path.getsize(MASTER_FILE)/1e6:.1f} MB",
        f"  Rows        : {df.height:,}",
        f"  Columns     : {df.width}",
        "",
        "  COLUMNS & DTYPES:",
    ]
    for col in df.columns:
        dtype    = str(df[col].dtype)
        n_null   = df[col].is_null().sum()
        pct_null = 100 * n_null / df.height
        lines.append(f"    {col:<22} {dtype:<15} nulls: {n_null:>8,} ({pct_null:4.1f}%)")

    # Date range
    lines += [
        "",
        f"  DATE RANGE  : {df['eom'].min()} → {df['eom'].max()}",
        f"  STOCKS/MONTH: {df.group_by('eom').len()['len'].mean():.0f} avg, "
        f"{df.group_by('eom').len()['len'].min()} min, "
        f"{df.group_by('eom').len()['len'].max()} max",
        f"  FACTORS HIT : n_factors_hit range "
        f"{df['n_factors_hit'].min()} – {df['n_factors_hit'].max()}, "
        f"mean {df['n_factors_hit'].mean():.1f}",
    ]

    # Weight distribution
    w = df["weight"]
    long_w  = w.filter(w > 0)
    short_w = w.filter(w < 0)
    lines += [
        "",
        "  WEIGHT DISTRIBUTION:",
        f"    Long  stocks: {long_w.len():>8,}  gross long  = {long_w.sum():+.4f}  (avg {long_w.mean():+.6f})",
        f"    Short stocks: {short_w.len():>8,}  gross short = {short_w.sum():+.4f}  (avg {short_w.mean():+.6f})",
        f"    Net exposure: {w.sum():+.6f}  (should be ~0 for dollar-neutral)",
        f"    Gross exp.  : {w.abs().sum():.4f}  (should be ~1.0 per month total)",
    ]

    # Monthly gross exposure check
    monthly_gross = (
        df.group_by("eom")
        .agg(pl.col("weight").abs().sum().alias("gross"))
        .sort("eom")
    )
    g = monthly_gross["gross"]
    lines += [
        "",
        "  MONTHLY GROSS EXPOSURE (should be 1.0 each month):",
        f"    Mean: {g.mean():.4f}  Std: {g.std():.4f}  "
        f"Min: {g.min():.4f}  Max: {g.max():.4f}",
    ]

    # ret_exc_lead1m null check by year
    if "ret_exc_lead1m" in df.columns:
        null_by_year = (
            df.with_columns(pl.col("eom").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("ret_exc_lead1m").is_null().mean().alias("null_pct")
            )
            .sort("year")
            .filter(pl.col("null_pct") > 0.01)
        )
        if null_by_year.height > 0:
            lines += ["", "  YEARS WITH >1% NULL ret_exc_lead1m:"]
            for row in null_by_year.iter_rows(named=True):
                lines.append(f"    {row['year']}: {row['null_pct']*100:.1f}%")
        else:
            lines.append("\n  ret_exc_lead1m: no years with >1% nulls ✓")

    lines.append(sep)

    for l in lines:
        print(l)
    report_lines.extend(lines)


# =============================================================================
# [A] FACTOR-LEVEL BACKTEST (Phase 1 — from pfs_daily.parquet)
# =============================================================================

def run_phase1_backtest(report_lines: list) -> pl.DataFrame:
    ts("[A] Phase 1 — Factor-level daily backtest (R replication)...")

    if not os.path.exists(PFS_DAILY):
        msg = f"  MISSING: {PFS_DAILY} — skipping Phase 1"
        print(msg); report_lines.append(msg)
        return None

    # Load daily portfolio returns (VW_CAP)
    pfs = (
        pl.scan_parquet(PFS_DAILY)
        .filter(pl.col("excntry") == "USA")
        .select(["date", "characteristic", "pf", "ret_vw_cap"])
        .collect()
    )

    # Long - Short daily factor returns
    factors = (
        pfs.group_by(["date", "characteristic"])
        .agg([
            pl.col("ret_vw_cap").filter(pl.col("pf") == pl.col("pf").max()).first().alias("ret_long"),
            pl.col("ret_vw_cap").filter(pl.col("pf") == pl.col("pf").min()).first().alias("ret_short"),
        ])
        .with_columns(
            (pl.col("ret_long") - pl.col("ret_short")).alias("factor_ret")
        )
    )

    # Sign corrections
    factors = factors.with_columns(
        pl.when(pl.col("characteristic").is_in(list(FACTORS_TO_FLIP)))
        .then(-pl.col("factor_ret"))
        .otherwise(pl.col("factor_ret"))
        .alias("factor_ret")
    )

    # Monthly momentum signal = sum(log(1 + factor_ret)) within month
    factors = factors.with_columns(
        pl.col("date").dt.month_start().dt.offset_by("-1d").alias("month")
    )
    monthly_sig = (
        factors.group_by(["month", "characteristic"])
        .agg(pl.col("factor_ret").log1p().sum().alias("mom_signal"))
    )

    # Factor weights: top 50% → +1, bottom 50% → -1, normalize
    weights = (
        monthly_sig
        .with_columns(
            pl.col("mom_signal").median().over("month").alias("median_sig")
        )
        .with_columns(
            pl.when(pl.col("mom_signal") >= pl.col("median_sig"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(-1.0))
            .alias("raw_w")
        )
        .with_columns(
            (pl.col("raw_w") / pl.col("raw_w").abs().sum().over("month"))
            .alias("weight_ls")
        )
        # 1-month signal lag: signal(month T) → trade in month T+1
        .with_columns(
            pl.col("month").dt.offset_by("1mo").dt.month_end().alias("trade_month")
        )
        .select(["trade_month", "characteristic", "weight_ls"])
    )

    # Join weights to daily factor returns
    factors = factors.with_columns(
        pl.col("date").dt.month_start().dt.offset_by("-1d").alias("month_end")
    )
    bt = factors.join(
        weights,
        left_on=["month_end", "characteristic"],
        right_on=["trade_month", "characteristic"],
        how="inner",
    )

    # Filter start date
    bt = bt.filter(pl.col("date").dt.date() >= START_DATE)

    # 1-Day execution lag: zero weight on first trading day of month
    bt = (
        bt.sort(["characteristic", "date"])
        .with_columns(
            pl.col("date").min().over("month_end").alias("first_trade_day")
        )
        .with_columns(
            pl.when(pl.col("date") == pl.col("first_trade_day"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("weight_ls"))
            .alias("weight_ls")
        )
    )

    # Aggregate to daily strategy return
    strategy = (
        bt.group_by("date")
        .agg((pl.col("weight_ls") * pl.col("factor_ret")).sum().alias("ret_daily"))
        .sort("date")
    )

    # Save
    out = f"{OUTPUT_PATH}/factor_momentum_phase1.parquet"
    strategy.write_parquet(out)

    stats = performance_stats(strategy["ret_daily"], freq="daily")
    lines = [
        "",
        "=" * 65,
        "  [A] PHASE 1 — FACTOR-LEVEL BACKTEST (R REPLICATION)",
        "=" * 65,
        f"  Source     : pfs_daily.parquet",
        f"  Period     : {strategy['date'].min()} → {strategy['date'].max()}",
        f"  Trading days: {strategy.height:,}",
        "",
        fmt(stats),
        "",
        "  BENCHMARK (from R script): Ann ~3.74%, Sharpe ~0.60",
        f"  Saved → {out}",
        "=" * 65,
    ]
    for l in lines:
        print(l)
    report_lines.extend(lines)
    return strategy


# =============================================================================
# [B] STOCK-LEVEL BACKTEST (Phase 2 — from arnott_master.parquet)
# =============================================================================

def run_phase2_backtest(report_lines: list) -> pl.DataFrame:
    ts("[B] Phase 2 — Stock-level monthly backtest...")

    if not os.path.exists(MASTER_FILE):
        msg = f"  MISSING: {MASTER_FILE} — skipping Phase 2"
        print(msg); report_lines.append(msg)
        return None

    df = pl.read_parquet(MASTER_FILE)
    df = df.filter(pl.col("eom").dt.date() >= START_DATE)

    # --- Monthly return: weight × ret_exc_lead1m ---
    monthly_raw = (
        df.group_by("eom")
        .agg(
            (pl.col("weight") * pl.col("ret_exc_lead1m")).sum().alias("ret_raw"),
            pl.col("weight").abs().sum().alias("gross_exp"),
            pl.len().alias("n_stocks"),
        )
        .sort("eom")
    )

    # --- Implementation lag adjustment ---
    # Subtract Day-1 drag: abs(weight) × ret_day1 per stock, summed
    if "ret_day1" in df.columns:
        day1_drag = (
            df.group_by("eom")
            .agg(
                (pl.col("weight").abs() * pl.col("ret_day1")).sum().alias("day1_drag")
            )
        )
        monthly = monthly_raw.join(day1_drag, on="eom", how="left")
        monthly = monthly.with_columns(
            (pl.col("ret_raw") - pl.col("day1_drag")).alias("ret_lagged")
        )
    else:
        monthly = monthly_raw.with_columns(
            pl.col("ret_raw").alias("ret_lagged"),
            pl.lit(0.0).alias("day1_drag"),
        )

    # --- Turnover ---
    # Turnover = sum(|w_t - w_{t-1}|) / 2 per month (assuming full rebalance)
    # For a monthly strategy where positions are fully rebuilt each month,
    # turnover ≈ gross_exp (since all old positions are closed and new ones opened).
    # We compute the true figure using weight changes.
    df_sorted = df.sort(["id", "eom"])
    df_to = (
        df_sorted
        .with_columns(
            pl.col("weight").shift(1).over("id").alias("weight_prev")
        )
        .with_columns(
            pl.col("weight_prev").fill_null(0.0)
        )
        .with_columns(
            (pl.col("weight") - pl.col("weight_prev")).abs().alias("trade_size")
        )
        .group_by("eom")
        .agg(
            (pl.col("trade_size").sum() / 2).alias("turnover")
        )
    )
    monthly = monthly.join(df_to, on="eom", how="left")

    # Stats
    stats_raw    = performance_stats(monthly["ret_raw"],    freq="monthly")
    stats_lagged = performance_stats(monthly["ret_lagged"], freq="monthly")
    avg_turnover = monthly["turnover"].mean()
    ann_turnover = avg_turnover * 12

    # Save
    out = f"{OUTPUT_PATH}/factor_momentum_phase2.parquet"
    monthly.write_parquet(out)

    lines = [
        "",
        "=" * 65,
        "  [B] PHASE 2 — STOCK-LEVEL MONTHLY BACKTEST",
        "=" * 65,
        f"  Source : arnott_master.parquet",
        f"  Period : {monthly['eom'].min()} → {monthly['eom'].max()}",
        f"  Months : {monthly.height:,}",
        "",
        "  WITHOUT implementation lag (raw):",
        fmt(stats_raw),
        "",
        "  WITH 1-day implementation lag (ret - Day1 drag):",
        fmt(stats_lagged),
        "",
        f"  TURNOVER:",
        f"    Monthly avg : {avg_turnover*100:.1f}%",
        f"    Annualised  : {ann_turnover*100:.1f}%  (target: <200%)",
        f"    Cost @ 10bps: -{ann_turnover*0.001*100:.2f}% alpha drag/year",
        "",
        "  PHASE 1 TARGET: Ann ~3.74%, Sharpe ~0.60",
        f"  Saved → {out}",
        "=" * 65,
    ]
    for l in lines:
        print(l)
    report_lines.extend(lines)
    return monthly


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    report_lines = []

    print("=" * 65)
    print("  FACTOR MOMENTUM — FULL DIAGNOSTIC & BACKTEST")
    print(f"  Run date : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # [C] Schema + quality check first so you know what you're working with
    diagnose_master(report_lines)

    # [A] Factor-level (Phase 1 baseline)
    run_phase1_backtest(report_lines)

    # [B] Stock-level (Phase 2)
    run_phase2_backtest(report_lines)

    # Save full report to text file
    report_path = f"{OUTPUT_PATH}/factor_momentum_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    ts(f"Full report saved → {report_path}")


if __name__ == "__main__":
    main()