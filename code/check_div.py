"""
check_dividends.py
------------------
Quick diagnostic: traces where divi / retx are available or missing
across the JKP pipeline data files.

Run from project root:
    uv run python code/check_dividends.py

Takes ~10 seconds. No data is modified.
"""

import os
import polars as pl

data_path = "data/processed"

DIV_COLS = ["divi", "retx", "prc_raw", "shrout", "dolvol_126d"]

def check_cols(path, label, cols=DIV_COLS):
    if not os.path.exists(path):
        print(f"  [MISSING FILE]  {label}  ->  {path}")
        return
    schema = pl.read_parquet_schema(path)
    available = [c for c in cols if c in schema]
    missing   = [c for c in cols if c not in schema]
    total_cols = len(schema)
    print(f"\n  {label}")
    print(f"    Path       : {path}")
    print(f"    Total cols : {total_cols}")
    print(f"    Found      : {available if available else 'none'}")
    print(f"    Missing    : {missing if missing else 'none'}")

    # If divi found, show basic stats (non-null rate)
    if "divi" in schema:
        sample = pl.read_parquet(path, columns=["divi"]).head(100_000)
        non_null_pct = 100 * sample["divi"].is_not_null().mean()
        print(f"    divi non-null (first 100k rows): {non_null_pct:.1f}%")

print("=" * 60)
print("  DIVIDEND / RETURN COLUMN DIAGNOSTIC")
print("=" * 60)

# ── 1. Characteristics (final output of main.py) ─────────────
check_cols(f"{data_path}/characteristics/USA.parquet",
           "characteristics/USA.parquet  [output of main.py]")

# ── 2. World return data (monthly) ───────────────────────────
check_cols(f"{data_path}/return_data/world_ret_monthly.parquet",
           "return_data/world_ret_monthly.parquet",
           cols=["id", "eom", "excntry", "divi", "retx", "ret_exc", "me"])

# ── 3. Daily stock returns (USA) ──────────────────────────────
check_cols(f"{data_path}/return_data/daily_rets_by_country/USA.parquet",
           "return_data/daily_rets_by_country/USA.parquet",
           cols=["id", "date", "ret_exc", "divi", "retx", "prc"])

# ── 4. AP factors (may carry divi as a characteristic) ────────
check_cols(f"{data_path}/other_output/ap_factors_monthly.parquet",
           "other_output/ap_factors_monthly.parquet",
           cols=["divi", "retx", "ret_exc"])

# ── 5. usa_factor_weights (output of portfolio_USA_weights.py) 
check_cols(f"{data_path}/portfolios/usa_factor_weights.parquet",
           "portfolios/usa_factor_weights.parquet",
           cols=["div_yield", "divi", "retx", "me", "me_cap"])

print()
print("=" * 60)
print("  DIAGNOSIS & RECOMMENDED FIX")
print("=" * 60)

usa_schema = pl.read_parquet_schema(f"{data_path}/characteristics/USA.parquet") \
    if os.path.exists(f"{data_path}/characteristics/USA.parquet") else {}

world_schema = pl.read_parquet_schema(f"{data_path}/return_data/world_ret_monthly.parquet") \
    if os.path.exists(f"{data_path}/return_data/world_ret_monthly.parquet") else {}

divi_in_chars  = "divi"  in usa_schema
divi_in_monthly = "divi" in world_schema

if divi_in_monthly and not divi_in_chars:
    print("""
  RESULT: divi IS in world_ret_monthly.parquet but NOT in characteristics/USA.parquet.

  ROOT CAUSE: main.py merges characteristics with return data but does not
  carry divi / retx through to the final characteristics parquet.

  FIX: In main.py, find where the characteristics parquet is written for each
  country (typically something like `data.write_parquet(...)`) and ensure
  'divi' and 'retx' are joined in from the monthly return data BEFORE writing.

  Concretely, add a left-join step such as:

      monthly_rets = pl.read_parquet("data/processed/return_data/world_ret_monthly.parquet")
                       .filter(pl.col("excntry") == "USA")
                       .select(["id", "eom", "divi", "retx"])

      data = data.join(monthly_rets, on=["id", "eom"], how="left")

  Then re-run main.py for USA and re-run the full factor momentum pipeline.
""")

elif not divi_in_monthly and not divi_in_chars:
    print("""
  RESULT: divi is NOT found in either characteristics or world_ret_monthly.parquet.

  ROOT CAUSE: The CRSP dividend columns (divi, retx) were not downloaded /
  retained during the original WRDS pull in main.py.

  FIX: Re-run the WRDS data download in main.py ensuring the following CRSP
  DSF / MSIX columns are included in the SELECT:
      divi   (CRSP monthly dividend income per share)
      retx   (CRSP monthly ex-dividend return)

  These are standard CRSP MSIX columns. Check your WRDS query in main.py
  and add them to the column list, then reprocess.
""")

elif divi_in_chars:
    print("""
  RESULT: divi IS present in characteristics/USA.parquet.

  The warning in portfolio_USA_weights.py may be a false alarm caused by
  column-name mismatch. Check that the column is named exactly 'divi' (lowercase)
  and that the optional_columns list in portfolio_USA_weights.py matches.
""")

print()
print("  Run this script again after any fix to confirm columns appear.")
print("=" * 60)