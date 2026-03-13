"""
factor_momentum_arnott.py
Arnott, Kalesnik & Linnainmaa (2023, RFS) factor momentum with realistic 1-day lag.

1-DAY LAG MECHANICS:
  Signal observed at close of eom_t.
  Trades execute at CLOSE of FIRST TRADING DAY of month t+1.
  Holding period: close(day1_t+1) -> close(eom_{t+1})
  lag1_ret = (1 + full_month_ret) / (1 + day1_ret) - 1

Computed for both factors (lms_daily.parquet) and stocks (daily_rets_by_country/USA.parquet).

PORTFOLIO CONSTRUCTION OPTIMIZATION GRID
-----------------------------------------
  Dimension 1 – Selection rule (fraction of factors selected):
      "50pct"   top/bottom 50%  (median split)
      "33pct"   top/bottom 33%
      "25pct"   top/bottom 25%

  Dimension 2 – Position type:
      "long_short"   equal-weight long winners, equal-weight short losers
      "long_only"    equal-weight long winners only (no short leg)

  Additionally, the original Time-Series strategy (TS) is always included as benchmark.

  All strategies go through the same vol-scaling step when VOL_SCALE=True.
  Summary performance table is printed at the end and saved to
      {output_dir}/grid_summary.csv
"""

import os
import csv
import numpy as np
import polars as pl
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
data_path  = "data/processed"
pf_path    = "data/processed/portfolios"
output_dir = "data/processed/portfolios/factor_momentum"
os.makedirs(output_dir, exist_ok=True)

LOOKBACK_MONTHS    = 1
RET_TYPE           = "ret_vw_cap"
VOL_SCALE          = True
VOL_SCALE_TARGET   = 0.10
VOL_SCALE_LOOKBACK = 36

# ── Optimization grid definition ───────────────────────────
# Each entry: (label, top_pctile, bot_pctile, long_only)
#   top_pctile  : fraction selected for long leg   (e.g. 0.50 = top 50%)
#   bot_pctile  : fraction selected for short leg  (0.0  = long-only, no short)
#   long_only   : if True, weights are purely positive (no short leg)
GRID = [
    # Long-Short strategies
    ("CS_LS_50", 0.50, 0.50, False),   # top 50% long, bottom 50% short
    ("CS_LS_33", 0.33, 0.33, False),   # top 33% long, bottom 33% short
    ("CS_LS_25", 0.25, 0.25, False),   # top 25% long, bottom 25% short
    # Long-Only strategies
    ("CS_LO_50", 0.50, 0.00, True),    # top 50% long only
    ("CS_LO_33", 0.33, 0.00, True),    # top 33% long only
    ("CS_LO_25", 0.25, 0.00, True),    # top 25% long only
]

print(f"\n{'='*60}")
print("  Arnott Factor Momentum  -  Realistic 1-Day Lag")
print(f"  Lookback={LOOKBACK_MONTHS}m  RetType={RET_TYPE}  VolScale={VOL_SCALE}")
print(f"  Grid: {len(GRID)} CS strategies + 1 TS benchmark")
print(f"{'='*60}\n")
t0 = time.time()

# ============================================================
# STEP 1: 1-day-lagged FACTOR returns from daily data
# lag1_ret = prod(1+r_d for d in month) / (1+r_day1) - 1
# ============================================================
print("[1/7] 1-day-lagged factor returns from lms_daily.parquet...", flush=True)

lms_daily = (
    pl.read_parquet(f"{pf_path}/lms_daily.parquet")
    .filter(pl.col("excntry") == "USA")
    .select(["characteristic", "date", RET_TYPE])
    .rename({RET_TYPE: "daily_ret"})
    .sort(["characteristic", "date"])
    .with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").rank(method="ordinal").over(["characteristic", "year", "month"]).alias("day_rank"),
    )
)

monthly_factors = (
    lms_daily
    .group_by(["characteristic", "year", "month"])
    .agg(
        (pl.col("daily_ret") + 1).product().alias("full_month_gross"),
        (pl.col("daily_ret") + 1).filter(pl.col("day_rank") == 1).first().alias("day1_gross"),
        pl.col("date").max().alias("eom"),
    )
    .with_columns(
        (pl.col("full_month_gross") / pl.col("day1_gross") - 1).alias("factor_ret_lag1"),
        (pl.col("full_month_gross") - 1).alias("factor_ret_full"),
    )
    .sort(["characteristic", "year", "month"])
)

print(f"      {len(monthly_factors)} obs, {monthly_factors['characteristic'].n_unique()} factors")
print(f"      Range: {monthly_factors['eom'].min()} - {monthly_factors['eom'].max()}")

# ============================================================
# STEP 2: Signal + forward return pairs
# signal          = full-month return at month t    (known at eom_t)
# factor_ret_fwd  = lag1-adjusted return at month t+1  (what we earn)
# ============================================================
print("[2/7] Building signals (1-month lookback)...", flush=True)

signals = (
    monthly_factors
    .sort(["characteristic", "year", "month"])
    .with_columns(
        pl.col("factor_ret_full").shift(LOOKBACK_MONTHS).over("characteristic").alias("signal"),
        pl.col("factor_ret_lag1").shift(-1).over("characteristic").alias("factor_ret_fwd_lag1"),
        pl.col("eom").shift(-1).over("characteristic").alias("eom_hold"),
    )
    .drop_nulls(subset=["signal", "factor_ret_fwd_lag1"])
    .rename({"eom": "eom_signal"})
)

print(f"      {len(signals)} obs, range: {signals['eom_signal'].min()} - {signals['eom_signal'].max()}")

# ============================================================
# STEP 3: Build weights for ALL strategies
# ============================================================
print("[3/7] Building factor momentum weights for all grid strategies...", flush=True)


def build_ts(df):
    """
    Time-Series strategy: long factors with positive prior return,
    short factors with negative prior return. All factors included.
    """
    return (
        df
        .with_columns(
            pl.when(pl.col("signal") > 0).then(pl.lit(1))
              .when(pl.col("signal") < 0).then(pl.lit(-1))
              .otherwise(pl.lit(None)).alias("fm_side")
        )
        .filter(pl.col("fm_side").is_not_null())
        .with_columns(
            pl.when(pl.col("fm_side") == 1)
              .then(pl.lit(1.0) / (pl.col("fm_side") == 1).sum().over("eom_signal").cast(pl.Float64))
              .otherwise(pl.lit(1.0) / (pl.col("fm_side") == -1).sum().over("eom_signal").cast(pl.Float64))
              .alias("fm_weight_abs")
        )
        .with_columns((pl.col("fm_side") * pl.col("fm_weight_abs")).alias("fm_weight"))
        .with_columns(pl.lit("TS").alias("strategy"))
    )


def build_cs(df, label, top=0.50, bot=0.50, long_only=False):
    """
    Cross-Sectional strategy with configurable selection thresholds.

    Parameters
    ----------
    label       : strategy name stored in the 'strategy' column (e.g. "CS_LS_50")
    top         : fraction of factors selected for the LONG leg  (e.g. 0.50 = top half)
    bot         : fraction of factors selected for the SHORT leg (set 0.0 for long-only)
    long_only   : if True, only the long leg is built; weights sum to +1
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
        # Only assign side = 1 to the top `top` fraction
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
        # Long top `top` fraction, short bottom `bot` fraction
        selected = (
            ranked
            .with_columns(
                pl.when(pl.col("pctile") > (1 - top)).then(pl.lit(1))
                  .when(pl.col("pctile") <= bot).then(pl.lit(-1))
                  .otherwise(pl.lit(None)).alias("fm_side")
            )
            .filter(pl.col("fm_side").is_not_null())
        )

    return (
        selected
        .with_columns(
            pl.when(pl.col("fm_side") == 1)
              .then(pl.lit(1.0) / (pl.col("fm_side") == 1).sum().over("eom_signal").cast(pl.Float64))
              .otherwise(pl.lit(1.0) / (pl.col("fm_side") == -1).sum().over("eom_signal").cast(pl.Float64))
              .alias("fm_weight_abs")
        )
        .with_columns((pl.col("fm_side") * pl.col("fm_weight_abs")).alias("fm_weight"))
        .with_columns(pl.lit(label).alias("strategy"))
    )


# Build TS benchmark + all CS grid strategies
ts_weights = build_ts(signals)

cs_weights_list = [
    build_cs(signals, label=label, top=top, bot=bot, long_only=long_only)
    for label, top, bot, long_only in GRID
]

all_weights = pl.concat([ts_weights] + cs_weights_list)
strategy_names = ["TS"] + [g[0] for g in GRID]

print(f"      Strategies built: {strategy_names}")

# ============================================================
# STEP 4: Strategy returns (1-day-lagged) + Vol Scaling
# ============================================================
print("[4/7] Computing strategy returns and vol-scaling...", flush=True)

strat_returns = (
    all_weights
    .group_by(["strategy", "eom_signal", "eom_hold"])
    .agg(
        (pl.col("fm_weight") * pl.col("factor_ret_fwd_lag1")).sum().alias("ret"),
        pl.len().alias("n_factors_active"),
    )
    .sort(["strategy", "eom_signal"])
)

if VOL_SCALE:
    strat_returns = (
        strat_returns
        .with_columns(
            (pl.col("ret").rolling_std(window_size=VOL_SCALE_LOOKBACK, min_samples=12)
               .over("strategy") * (12**0.5)).alias("ann_vol_est")
        )
        .with_columns(pl.col("ann_vol_est").shift(1).over("strategy").alias("ann_vol_lagged"))
        .with_columns(
            pl.when(pl.col("ann_vol_lagged").is_not_null() & (pl.col("ann_vol_lagged") > 0))
              .then(pl.min_horizontal(pl.lit(5.0), pl.lit(float(VOL_SCALE_TARGET)) / pl.col("ann_vol_lagged")))
              .otherwise(pl.lit(1.0))
              .alias("leverage")
        )
        .with_columns((pl.col("ret") * pl.col("leverage")).alias("ret_scaled"))
    )
else:
    strat_returns = strat_returns.with_columns(
        pl.col("ret").alias("ret_scaled"),
        pl.lit(1.0).alias("leverage"),
        pl.lit(None).cast(pl.Float64).alias("ann_vol_est"),
        pl.lit(None).cast(pl.Float64).alias("ann_vol_lagged"),
    )

strat_returns = strat_returns.with_columns(
    ((1 + pl.col("ret")).cum_prod().over("strategy")).alias("cum_ret"),
    ((1 + pl.col("ret_scaled")).cum_prod().over("strategy")).alias("cum_ret_scaled"),
)

# ============================================================
# STEP 4b: OPTIMIZATION GRID SUMMARY TABLE
# ============================================================
print("\n" + "="*80)
print("  PORTFOLIO CONSTRUCTION OPTIMIZATION GRID")
print("  (Unscaled returns — raw strategy performance)")
print("="*80)

grid_display = [
    # (strategy_label, selection_rule, position_type)
    ("TS",       "All factors",  "Long-Short (time-series)"),
    ("CS_LS_50", "Top/Bot 50%",  "Long-Short"),
    ("CS_LS_33", "Top/Bot 33%",  "Long-Short"),
    ("CS_LS_25", "Top/Bot 25%",  "Long-Short"),
    ("CS_LO_50", "Top 50%",      "Long-Only"),
    ("CS_LO_33", "Top 33%",      "Long-Only"),
    ("CS_LO_25", "Top 25%",      "Long-Only"),
]

summary_rows = []

for strat_label, sel_rule, pos_type in grid_display:
    s = strat_returns.filter(pl.col("strategy") == strat_label).sort("eom_signal")
    if len(s) == 0:
        continue

    rets = s["ret"].to_numpy()

    ann_ret = float(rets.mean()) * 12
    ann_vol = float(rets.std()) * (12**0.5)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    # Max drawdown on unscaled cumulative return
    cum         = s["cum_ret"].to_numpy()
    running_max = np.maximum.accumulate(cum)
    drawdowns   = cum / running_max - 1
    max_dd      = float(drawdowns.min())
    calmar      = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")

    # Skewness
    n    = len(rets)
    skew = float(((rets - rets.mean())**3).mean() / (rets.std()**3)) if n > 2 else float("nan")

    print(
        f"  {strat_label:<12}  {sel_rule:<15}  {pos_type:<30}"
        f"  Ret={ann_ret:+.2%}  Vol={ann_vol:.2%}  SR={sharpe:.2f}"
        f"  MaxDD={max_dd:.2%}  Calmar={calmar:.2f}  Skew={skew:.2f}"
    )

    summary_rows.append({
        "strategy":       strat_label,
        "selection_rule": sel_rule,
        "position_type":  pos_type,
        "ann_ret":        round(ann_ret, 6),
        "ann_vol":        round(ann_vol, 6),
        "sharpe":         round(sharpe, 4),
        "max_dd":         round(max_dd, 6),
        "calmar":         round(calmar, 4),
        "skewness":       round(skew, 4),
        "n_months":       n,
    })

print("="*80)

# Sharpe heat-map table
print("\n  SHARPE RATIO HEAT MAP\n")
print(f"  {'Selection':<15}  {'Long-Short':>12}  {'Long-Only':>12}")
print(f"  {'-'*15}  {'-'*12}  {'-'*12}")
for sel_label, ls_key, lo_key in [
    ("Top/Bot 50%", "CS_LS_50", "CS_LO_50"),
    ("Top/Bot 33%", "CS_LS_33", "CS_LO_33"),
    ("Top/Bot 25%", "CS_LS_25", "CS_LO_25"),
]:
    ls_row = next((r for r in summary_rows if r["strategy"] == ls_key), None)
    lo_row = next((r for r in summary_rows if r["strategy"] == lo_key), None)
    ls_sr  = f"{ls_row['sharpe']:.2f}" if ls_row else "n/a"
    lo_sr  = f"{lo_row['sharpe']:.2f}" if lo_row else "n/a"
    print(f"  {sel_label:<15}  {ls_sr:>12}  {lo_sr:>12}")
print()

# Save summary to CSV
if summary_rows:
    csv_path = f"{output_dir}/grid_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  [OK] Grid summary saved to {csv_path}")

# ============================================================
# STEP 5: 1-day-lagged STOCK returns
# Same mechanics: skip day 1 return for each stock in each month
# ============================================================
print("\n[5/7] 1-day-lagged stock returns from daily_rets_by_country/USA.parquet...", flush=True)

daily_stocks = (
    pl.read_parquet(
        f"{data_path}/return_data/daily_rets_by_country/USA.parquet",
        columns=["id", "date", "ret_exc"],
    )
    .sort(["id", "date"])
    .with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").rank(method="ordinal").over(["id", "year", "month"]).alias("day_rank"),
    )
)

stock_monthly = (
    daily_stocks
    .group_by(["id", "year", "month"])
    .agg(
        (pl.col("ret_exc") + 1).product().alias("full_month_gross"),
        (pl.col("ret_exc") + 1).filter(pl.col("day_rank") == 1).first().alias("day1_gross"),
    )
    .with_columns(
        (pl.col("full_month_gross") / pl.col("day1_gross") - 1).alias("ret_exc_lag1"),
    )
)

print(f"      {len(stock_monthly)} obs, {stock_monthly['id'].n_unique()} unique stocks")

# ============================================================
# STEP 6: Decompose FMom to stocks + attach lag1 returns
# (Run for TS and all grid strategies)
# ============================================================
print("[6/7] Decomposing to single stocks...", flush=True)

char_info = (
    pl.read_excel("data/factor_details.xlsx", sheet_name="details")
    .filter(pl.col("abr_jkp").is_not_null())
    .select([pl.col("abr_jkp").alias("characteristic"), pl.col("direction").cast(pl.Int8)])
)

stock_weights = (
    pl.read_parquet(f"{pf_path}/usa_factor_weights.parquet")
    .join(char_info, on="characteristic", how="left")
    .with_columns((pl.col("leg") * pl.col("direction")).alias("signed_leg"))
    .with_columns(
        pl.col("eom").dt.year().alias("year_form"),
        pl.col("eom").dt.month().alias("month_form"),
    )
    .with_columns(
        pl.when(pl.col("month_form") == 12).then(pl.col("year_form") + 1).otherwise(pl.col("year_form")).alias("year_hold"),
        pl.when(pl.col("month_form") == 12).then(pl.lit(1)).otherwise(pl.col("month_form") + 1).alias("month_hold"),
    )
    .join(
        stock_monthly.select(["id", "year", "month", "ret_exc_lag1"])
                     .rename({"year": "year_hold", "month": "month_hold"}),
        on=["id", "year_hold", "month_hold"],
        how="left",
    )
)

missing = stock_weights["ret_exc_lag1"].is_null().sum()
pct     = 100 * missing / len(stock_weights)
print(f"      Stocks missing lag1 return: {missing:,} ({pct:.1f}%) - delistings/data gaps")


def decompose(fw_all, sw, strategy):
    fw = fw_all.filter(pl.col("strategy") == strategy).select(
        ["characteristic", "eom_signal", "eom_hold", "fm_weight", "fm_side", "fm_weight_abs"]
    )
    detail = (
        sw
        .join(fw, left_on=["characteristic", "eom"], right_on=["characteristic", "eom_signal"], how="inner")
        .with_columns(
            (pl.col("fm_weight") * pl.col("signed_leg") * pl.col("weight")).alias("stock_fm_weight")
        )
    )
    netted = (
        detail
        .group_by(["eom", "eom_hold", "id"])
        .agg(
            pl.col("stock_fm_weight").sum().alias("net_weight"),
            pl.col("stock_fm_weight").abs().sum().alias("gross_weight"),
            pl.len().alias("n_factors"),
            pl.col("me").first().alias("me"),
            pl.col("me_cap").first().alias("me_cap"),
            pl.col("div_yield").first().alias("div_yield"),
            pl.col("ret_exc_lag1").first().alias("ret_exc_lag1"),
            pl.col("ret_exc_lead1m").first().alias("ret_exc_lead1m_full"),
        )
        .rename({"eom": "eom_signal"})
        .with_columns(pl.lit(strategy).alias("strategy"))
    )
    return detail, netted


# Decompose all strategies
all_detail_list = []
all_netted_list = []

for strat in strategy_names:
    try:
        det, net = decompose(all_weights, stock_weights, strat)
        all_detail_list.append(det)
        all_netted_list.append(net)
    except Exception as e:
        print(f"      Warning: decompose failed for {strat}: {e}")

all_detail = pl.concat(all_detail_list)
all_netted = pl.concat(all_netted_list)

# Sanity check + lag cost (shown for TS and CS_LS_50 as representative cases)
print("\n      Sanity check (stock portfolio vs factor portfolio return):")
for strat in ["TS", "CS_LS_50"]:
    s_ret = (
        all_netted
        .filter((pl.col("strategy") == strat) & pl.col("ret_exc_lag1").is_not_null())
        .group_by("eom_signal")
        .agg((pl.col("net_weight") * pl.col("ret_exc_lag1")).sum().alias("stock_ret"))
    )
    f_ret = strat_returns.filter(pl.col("strategy") == strat).select(["eom_signal", "ret"])
    cmp   = s_ret.join(f_ret, on="eom_signal", how="inner")
    if len(cmp) > 0:
        corr = cmp.select(pl.pearsonr("stock_ret", "ret"))[0, 0]
        mad  = (cmp["stock_ret"] - cmp["ret"]).abs().mean()
        flag = "  *** LOW CORR ***" if corr < 0.99 else ""
        print(f"      {strat}: corr={corr:.4f}  mean_abs_diff={mad:.6f}{flag}")

print("\n      Annual return cost of 1-day implementation lag:")
for strat in ["TS", "CS_LS_50"]:
    chk = (
        all_netted
        .filter(
            (pl.col("strategy") == strat)
            & pl.col("ret_exc_lag1").is_not_null()
            & pl.col("ret_exc_lead1m_full").is_not_null()
        )
        .group_by("eom_signal")
        .agg(
            (pl.col("net_weight") * pl.col("ret_exc_lag1")).sum().alias("ret_lag1"),
            (pl.col("net_weight") * pl.col("ret_exc_lead1m_full")).sum().alias("ret_full"),
        )
    )
    if len(chk) > 0:
        cost_bps = (chk["ret_full"] - chk["ret_lag1"]).mean() * 12 * 10_000
        print(f"      {strat}: lag cost = {cost_bps:.1f} bps/year  (= what you give up by not trading at eom_t close)")

# ============================================================
# STEP 7: Write outputs
# ============================================================
print("\n[7/7] Writing outputs...", flush=True)

strat_returns.write_parquet(f"{output_dir}/factor_momentum_returns.parquet")
all_weights.write_parquet(f"{output_dir}/factor_momentum_weights.parquet")
all_netted.write_parquet(f"{output_dir}/factor_momentum_stocks.parquet")
all_detail.write_parquet(f"{output_dir}/factor_momentum_stocks_detail.parquet")

for fname in ["factor_momentum_returns", "factor_momentum_weights",
              "factor_momentum_stocks", "factor_momentum_stocks_detail"]:
    path = f"{output_dir}/{fname}.parquet"
    size = os.path.getsize(path) / 1024**2 if os.path.exists(path) else 0
    print(f"      [OK] {fname}.parquet  ({size:.1f} MB)")

print(f"\nDone in {time.time()-t0:.1f}s\n{'='*60}")
print("""
STRATEGIES IN OUTPUT FILES
  strategy column values:
    TS          Time-Series benchmark (all factors, long/short on sign of prior return)
    CS_LS_50    Cross-Sec Long-Short, top/bottom 50%  (median split)
    CS_LS_33    Cross-Sec Long-Short, top/bottom 33%
    CS_LS_25    Cross-Sec Long-Short, top/bottom 25%
    CS_LO_50    Cross-Sec Long-Only,  top 50%
    CS_LO_33    Cross-Sec Long-Only,  top 33%
    CS_LO_25    Cross-Sec Long-Only,  top 25%

EXECUTION PROTOCOL:
  eom_signal (e.g. Jan 31): observe factor returns -> compute signals -> determine net_weight per stock
  Day 1 of Feb (e.g. Feb 1): generate order list, EXECUTE at CLOSE
  Hold until close of eom_hold (Feb 28/29)
  Realised portfolio return = sum_i(net_weight_i * ret_exc_lag1_i)

OUTPUT SCHEMA (factor_momentum_stocks.parquet):
  eom_signal           end of signal month (positions determined here)
  eom_hold             end of holding month (unwind here)
  id                   PERMNO
  net_weight           netted signed position (+long / -short)
  gross_weight         sum |factor contributions| before netting
  n_factors            number of factors this stock appears in
  me / me_cap          market equity at formation
  div_yield            dividend yield (null if unavailable)
  ret_exc_lag1         actual lag1 excess return (day2 to eom_hold)
  ret_exc_lead1m_full  full-month return (for lag cost analysis)
  strategy             TS | CS_LS_50 | CS_LS_33 | CS_LS_25 | CS_LO_50 | CS_LO_33 | CS_LO_25
""")
