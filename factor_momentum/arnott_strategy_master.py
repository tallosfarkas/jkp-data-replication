import polars as pl
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_PATH = "data/processed/portfolios"
WEIGHTS_PATH = f"{OUTPUT_PATH}/usa_factor_weights.parquet"
LMS_PATH = f"{OUTPUT_PATH}/lms.parquet" 
DETAILS_PATH = "data/factor_details.xlsx"
STOCK_RET_PATH = "data/processed/characteristics/USA.parquet"
DAILY_RET_PATH = "data/processed/return_data/daily_rets_by_country/USA.parquet"

# Standard Arnott Logic
START_YEAR = 1963
LOOKBACK_MONTHS = 1
TOP_PERCENTILE = 0.25 

def generate_master_file():
    print(f"--- Building Master Backtest File (Weights + Returns + Lag) ---")
    
    # 1. Load & Net Weights (Standard Process)
    print("1. Calculating Net Weights...")
    details = pl.read_excel(DETAILS_PATH, sheet_name="details").select(
        [pl.col("abr_jkp").alias("characteristic"), pl.col("direction").cast(pl.Int32)]
    )
    lms = pl.read_parquet(LMS_PATH).filter(pl.col("excntry") == "USA")
    
    # Generate Signal & Winners
    lms = lms.sort(["characteristic", "eom"])
    lms = lms.with_columns(
        pl.col("ret_vw_cap").shift(LOOKBACK_MONTHS).over("characteristic").alias("mom_signal")
    )
    lms = lms.filter(pl.col("eom").dt.year() >= START_YEAR)
    lms = lms.filter(pl.col("mom_signal").is_not_null())

    ranked = lms.with_columns([
        pl.col("mom_signal").rank(descending=True, method="min").over("eom").alias("rank"),
        pl.len().over("eom").alias("n_factors")
    ])
    winners = ranked.filter(pl.col("rank") <= (pl.col("n_factors") * TOP_PERCENTILE))
    winners = winners.with_columns((1.0 / pl.count("characteristic").over("eom")).alias("strat_weight"))

    # Join Weights
    weights = pl.read_parquet(WEIGHTS_PATH).filter(pl.col("eom").dt.year() >= START_YEAR)
    weights = weights.join(details, on="characteristic", how="left")
    weights = weights.with_columns((pl.col("leg") * pl.col("direction")).alias("true_leg"))
    
    # Netting
    active = weights.join(winners, on=["eom", "characteristic"], how="inner")
    active = active.with_columns((pl.col("strat_weight") * pl.col("weight") * pl.col("true_leg")).alias("net_contrib"))
    
    master = active.group_by(["eom", "id"]).agg(
        pl.sum("net_contrib").alias("weight"),
        pl.n_unique("characteristic").alias("n_factors_hit"),
    )
    
    # Normalize (Gross Exposure = 1.0)
    master = master.with_columns(
         (pl.col("weight") / pl.col("weight").abs().sum()).over("eom").alias("weight")
    )

    # 2. Attach Monthly Returns
    print("2. Attaching Monthly Returns...")
    rets = pl.read_parquet(STOCK_RET_PATH, columns=["id", "eom", "ret_exc_lead1m"])
    master = master.join(rets, on=["eom", "id"], how="inner")

    # 3. Attach Day 1 Lag Returns
    print("3. Calculating & Attaching Day 1 Returns...")
    daily = pl.read_parquet(DAILY_RET_PATH, columns=["id", "date", "ret_exc"])
    
    # Find first trading day for each EOM
    calendar = daily.select("date").unique().sort("date")
    calendar = calendar.with_columns(pl.col("date").dt.month_start().dt.offset_by("-1d").alias("eom_lag"))
    first_days = calendar.group_by("eom_lag").agg(pl.col("date").min().alias("first_trade_date"))
    
    # Get stock-specific returns on that first day
    day1 = daily.join(first_days, left_on="date", right_on="first_trade_date", how="inner")
    day1 = day1.select([
        pl.col("id"), 
        pl.col("eom_lag").alias("eom"), 
        pl.col("ret_exc").alias("ret_day1")
    ])
    
    # Join (Left join because some stocks might not trade on day 1, fill with 0)
    master = master.join(day1, on=["eom", "id"], how="left").fill_null(0)

    # 4. Save
    save_path = f"{OUTPUT_PATH}/arnott_master.parquet"
    print(f"4. Saving Master File ({master.estimated_size() / 1e6:.1f} MB)...")
    master.write_parquet(save_path)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    generate_master_file()
