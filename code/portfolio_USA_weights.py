import os
import polars as pl
from datetime import date
import numpy as np
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Sortedness.*by.*provided",
)

# setting data path and output path
data_path = "data/processed"
output_path = "data/processed/portfolios"

countries = ['USA']

# characteristics
# characteristics filtered for QFin Lab replication
chars = [
    "age",              # Firm_Age
    "ami_126d",         # Amihud_Illiquidity
    "at_be",            # Leverage_Factor (Assets / Book Equity)
    "at_gr1",           # Asset_Growth_CMA
    "at_turnover",      # Change_in_Asset_Turnover
    "be_me",            # Book_to_Market_HML
    "beta_60m",         # ADDED: Market Beta (Standard CAPM Beta is usually needed for alphas)
    "betabab_1260d",    # Low_Beta_BAB
    "capex_abn",        # Abnormal_Investment
    "capx_gr1",         # CAPX_Growth_Rate
    "chcsho_12m",       # One_Year_Share_Issuance
    "cowc_gr1a",        # Net_Working_Capital_Changes
    "dbnetis_at",       # Debt_Issuance_Factor
    "dolvol_126d",      # High_Volume_Premium
    "dsale_dinv",       # Growth_in_Sales_Inventory
    "ebit_sale",        # Profit_Margin
    "ebitda_mev",       # Enterprise_Multiple
    "f_score",          # Piotroski_F_Score
    "gp_at",            # Gross_Profitability
    "inv_gr1",          # Growth_in_Inventory
    "ivol_ff3_21d",     # Residual_Variance_RVAR
    "market_equity",    # Size_SMB
    "netis_at",         # Total_External_Financing
    "ni_be",            # Return_on_Equity
    "ni_me",            # Earnings_to_Price
    "niq_at",           # Return_on_Assets
    "noa_at",           # Net_Operating_Assets
    "o_score",          # Ohlson_O_Score
    "oaccruals_at",     # Accruals_Factor
    "ocf_me",           # Cash_Flow_to_Price
    "ope_be",           # Operating_Profitability_RMW
    "prc",              # Nominal_Price
    "qmj",              # Quality_Minus_Junk_QMJ
    "ret_1_0",          # ADDED: Short Term Reversal (Critical control)
    "ret_12_1",         # ADDED: Standard Momentum (MOM) - CRITICAL FOR BASELINE
    "ret_60_12",        # Long_Term_Reversals_LTREV
    "sale_gr1",         # Sales_Growth
    "sale_me",          # Sales_to_Price
    "z_score",          # Altman_Z_Score
]

# a dictionary which has the parameters for constructing portfolios
settings = {
    "end_date": date(2025, 12, 31),
    "pfs": 3,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": 10,
    "cmp": {"us": True, "int": False},
    "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
    "regional_pfs": {
        "ret_type": "vw_cap",
        "country_excl": ["ZWE", "VEN"],
        "country_weights": "market_cap",
        "stocks_min": 5,
        "months_min": 5 * 12,
        "countries_min": 3,
    },
    "daily_pf": True,
    "ind_pf": True,
}


def add_ecdf(df: pl.DataFrame, group_cols: list[str] = ["eom"]) -> pl.DataFrame:
    # 1) counts of reference sample per distinct var within each group
    ref_counts = (
        df.filter(pl.col("bp_stock")).group_by(group_cols + ["var"]).agg(n_ref=pl.len())
    )

    # 2) ECDF steps: cumulative share within each group
    ref_steps = (
        ref_counts.sort(group_cols + ["var"])
        .with_columns(
            # apply the window to the whole fraction to ensure same partition
            cdf_val=(pl.cum_sum("n_ref") / pl.sum("n_ref")).over(group_cols)
        )
        .select(group_cols + ["var", "cdf_val"])
    )

    # 3) MUST pre-sort both sides by group_cols + ["var"] for join_asof with 'by'
    left = df.sort(group_cols + ["var"])
    right = ref_steps.sort(group_cols + ["var"])  # already sorted above

    out = (
        left.join_asof(
            right,
            on="var",
            by=group_cols,
            strategy="backward",
        )
        .with_columns(pl.col("cdf_val").fill_null(0.0).alias("cdf"))
        .drop("cdf_val")
    )
    return out


# main portfolios function to create the portfolios
def portfolios(
    data_path,
    excntry,
    chars,
    pfs,  # Number of portfolios
    bps,  # What should breakpoints be based on? Non-Microcap stocks ("non_mc") or NYSE stocks "nyse"
    bp_min_n,  # Minimum number of stocks used for breakpoints
    nyse_size_cutoffs,  # Data frame with NYSE size breakpoints
    source=[
        "CRSP",
        "COMPUSTAT",
    ],  # Use data from "CRSP", "Compustat" or both: c("CRSP", "COMPUSTAT")
    wins_ret=True,  # Should Compustat returns be winsorized at the 0.1% and 99.9% of CRSP returns?
    cmp_key=False,  # Create characteristics managed size portfolios?
    signals=False,  # Create portfolio signals?
    signals_standardize=False,  # Map chars to [-0.5, +0.5]?,
    signals_w="vw_cap",  # Weighting for signals: in c("ew", "vw", "vw_cap")
    daily_pf=False,  # Should daily return be estimated
    ind_pf=True,  # Should industry portfolio returns be estimated
    ret_cutoffs=None,  # Data frame for monthly winsorization. Neccesary when wins_ret=T
    ret_cutoffs_daily=None,  # Data frame for daily winsorization. Neccesary when wins_ret=T and daily_pf=T
):
    # characerteristics data
    file_path = f"{data_path}/characteristics/{excntry}.parquet"

    # Select the required columns
    columns = (
        [
            "id",
            "eom",
            "source_crsp",
            "comp_exchg",
            "crsp_exchcd",
            "size_grp",
            "ret_exc",
            "ret_exc_lead1m",
            "me",
            "gics",
            "ff49",
        ]
        + chars
        + ["excntry"]
    )

    # Load the data
    data = pl.read_parquet(file_path, columns=columns)
    data = data

    # capping me at nyse cut-off
    data = data.join(
        nyse_size_cutoffs.select(["eom", "nyse_p80"]), on="eom", how="left"
    )
    data = data.with_columns(
        pl.min_horizontal(pl.col("me"), pl.col("nyse_p80")).alias("me_cap")
    ).drop("nyse_p80")

    # ensuring numerical columns are float-added later:
    exclude = ["id", "eom", "source_crsp", "size_grp", "excntry"]
    for i in data.columns:
        if i not in exclude:
            data = data.with_columns(pl.col(i).cast(pl.Float64))

    # Screens
    if len(source) == 1:
        if source[0] == "CRSP":
            data = data.filter(pl.col("source_crsp") == 1)
        elif source[0] == "COMPUSTAT":
            data = data.filter(pl.col("source_crsp") == 0)
    data = data.filter(
        (pl.col("size_grp").is_not_null())
        & (pl.col("me").is_not_null())
        & (pl.col("ret_exc_lead1m").is_not_null())
    )

    # Daily Returns
    if daily_pf:
        daily_file_path = (
            f"{data_path}/return_data/daily_rets_by_country/{excntry}.parquet"
        )
        daily = pl.read_parquet(daily_file_path, columns=["id", "date", "ret_exc"])
        # daily = daily.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date"))
        daily = daily.with_columns(
            (pl.col("date").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
        )
        # ensuring numerical columns are float-added later:
        daily = daily.with_columns(pl.col("ret_exc").cast(pl.Float64))

    if wins_ret:
        data = data.join(
            ret_cutoffs.select(["eom_lag1", "ret_exc_0_1", "ret_exc_99_9"]).rename(
                {"eom_lag1": "eom", "ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}
            ),
            on="eom",
            how="left",
        )
        data = data.with_columns(
            pl.when(
                (pl.col("source_crsp") == 0)
                & (pl.col("ret_exc_lead1m") > pl.col("p999"))
            )
            .then(pl.col("p999"))
            .when(
                (pl.col("source_crsp") == 0)
                & (pl.col("ret_exc_lead1m") < pl.col("p001"))
            )
            .then(pl.col("p001"))
            .otherwise(pl.col("ret_exc_lead1m"))
            .alias("ret_exc_lead1m")
        ).drop(["source_crsp", "p001", "p999"])

        if daily_pf:
            # Extracting year and month from the date column
            daily = daily.with_columns(
                [
                    pl.col("date").dt.year().cast(pl.Int64).alias("year"),
                    pl.col("date").dt.month().cast(pl.Int64).alias("month"),
                ]
            )

            # Joining with daily return cutoffs
            daily = daily.join(
                ret_cutoffs_daily.with_columns(
                    [pl.col("year").cast(pl.Int64), pl.col("month").cast(pl.Int64)]
                )
                .select(["year", "month", "ret_exc_0_1", "ret_exc_99_9"])
                .rename({"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}),
                on=["year", "month"],
                how="left",
            )

            # Applying winsorization to daily returns for Compustat data (id > 99999)
            daily = daily.with_columns(
                pl.when((pl.col("id") > 99999) & (pl.col("ret_exc") > pl.col("p999")))
                .then(pl.col("p999"))
                .when((pl.col("id") > 99999) & (pl.col("ret_exc") < pl.col("p001")))
                .then(pl.col("p001"))
                .otherwise(pl.col("ret_exc"))
                .alias("ret_exc")
            ).drop(["p001", "p999", "year", "month"])

    # standardizing signals
    if signals_standardize and signals:
        data = (
            data
            # Ranking within groups defined by 'eom'
            .with_columns(
                [
                    (pl.col(char).rank(method="min").over("eom").cast(pl.Int64)).alias(
                        char
                    )
                    for char in chars
                ]
            )
            # normalizing ranks
            .with_columns(
                [
                    (
                        ((pl.col(char) / pl.col(char).max()) - pl.lit(0.5)).over("eom")
                    ).alias(char)
                    for char in chars
                ]
            )
        )

    if ind_pf:
        # Filter data where 'gics' is not null and select required columns
        ind_data = data.filter(
            pl.col("gics").is_not_null()
        ).select(
            ["eom", "gics", "excntry", "ret_exc_lead1m", "me", "me_cap"]
        )  # original code didn;t select 'excntry' in the above steps in the data table, updating that

        # Process GICS codes (extract first 2 digits and convert to numeric)
        ind_data = ind_data.with_columns(
            (pl.col("gics").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64)).alias("gics")
        )

        # Calculate industry returns based on GICS
        ind_gics = ind_data.group_by(["gics", "eom"]).agg(
            [
                pl.len().alias("n"),
                (pl.col("ret_exc_lead1m").mean()).alias("ret_ew"),
                (
                    (pl.col("ret_exc_lead1m") * pl.col("me")).sum() / pl.col("me").sum()
                ).alias("ret_vw"),
                (
                    (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum()
                    / pl.col("me_cap").sum()
                ).alias("ret_vw_cap"),
            ]
        )
        ind_gics = ind_gics.with_columns(
            pl.lit(excntry).str.to_uppercase().alias("excntry")
        )
        ind_gics = ind_gics.with_columns(
            (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
        )
        ind_gics = ind_gics.filter(pl.col("n") >= bp_min_n)

        # Estimate industry portfolios by Fama-French portfolios for US data
        if excntry == "usa":
            ind_data = data.filter(pl.col("ff49").is_not_null()).select(
                ["eom", "ff49", "ret_exc_lead1m", "me", "me_cap"]
            )
            ind_ff49 = ind_data.group_by(["ff49", "eom"]).agg(
                [
                    pl.len().alias("n"),
                    (pl.col("ret_exc_lead1m").mean()).alias("ret_ew"),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me")).sum()
                        / pl.col("me").sum()
                    ).alias("ret_vw"),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum()
                        / pl.col("me_cap").sum()
                    ).alias("ret_vw_cap"),
                ]
            )
            ind_ff49 = ind_ff49.with_columns(
                pl.lit(excntry).str.to_uppercase().alias("excntry")
            )
            ind_ff49 = ind_ff49.with_columns(
                (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
            )
            ind_ff49 = ind_ff49.filter(pl.col("n") >= bp_min_n)

    # creating portfolios for all the characteristics
    char_pfs = []
    for i, x in enumerate(tqdm(chars, desc="Processing chars", unit="char", ncols=80)):
        op = {}

        data = data.with_columns(pl.col(x).cast(pl.Float64).alias("var"))
        if not signals:
            # Select rows where 'var' is not missing and only specific columns
            sub = (
                data.lazy()
                .filter(pl.col("var").is_not_null())
                .select(
                    [
                        "id",
                        "eom",
                        "var",
                        "size_grp",
                        "ret_exc_lead1m",
                        "me",
                        "me_cap",
                        "crsp_exchcd",
                        "comp_exchg",
                    ]
                )
            )
        else:
            # Select rows where 'var' is not missing, retaining all columns
            sub = data.lazy().filter(pl.col("var").is_not_null())

        if bps == "nyse":
            # Create 'bp_stock' column for NYSE criteria
            sub = sub.with_columns(
                (
                    ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
                    | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
                ).alias("bp_stock")
            )

        elif bps == "non_mc":
            # Create 'bp_stock' column for non-microcap criteria
            sub = sub.with_columns(
                pl.col("size_grp").is_in(["mega", "large", "small"]).alias("bp_stock")
            )

        sub = sub.with_columns(bp_n=pl.sum("bp_stock").over("eom")).filter(
            pl.col("bp_n") >= bp_min_n
        )

        # Ensure that 'sub' is not empty
        if sub.limit(1).collect().height > 0:
            sub = add_ecdf(sub)

            # Step 1: Find the minimum CDF value within each 'eom' group
            sub = sub.with_columns(pl.col("cdf").min().over("eom").alias("min_cdf"))

            # Step 2: Adjust CDF values for the lowest value in each group
            sub = sub.with_columns(
                pl.when(pl.col("cdf") == pl.col("min_cdf"))
                .then(0.00000001)
                .otherwise(pl.col("cdf"))
                .alias("cdf")
            )

            # Step 3: Calculate portfolio assignments and adjust portfolio numbers (Happens when non-bp stocks extend beyond the bp stock range)
            sub = sub.with_columns(
                (pl.col("cdf") * pfs)
                .ceil()
                .clip(lower_bound=1, upper_bound=pfs)
                .alias("pf")
            )
		# --- START MODIFICATION: EXTRACT CONSTITUENT WEIGHTS ---
            # Filter for Long (Top Portfolio) and Short (Bottom Portfolio) only
            # pf=1 is the lowest bin, pf=pfs (e.g., 3 or 10) is the highest bin
            weights_df = sub.filter(pl.col("pf").is_in([1, pfs]))
            
            # Calculate VW_CAP weights relative to that specific portfolio bin
            weights_df = weights_df.with_columns(
                (pl.col("me_cap") / pl.col("me_cap").sum().over(["eom", "pf"])).alias("weight")
            )
            
            # Assign Leg Direction: +1 for Top Bin, -1 for Bottom Bin
            # NOTE: You must multiply this by the factor direction from factor_details.xlsx later!
            weights_df = weights_df.with_columns(
                pl.when(pl.col("pf") == pfs).then(pl.lit(1))
                .otherwise(pl.lit(-1)).cast(pl.Int8).alias("leg")
            )

            # Select only essential columns to save memory
            weights_df = weights_df.select([
                pl.lit(x).alias("characteristic"),
                "id", 
                "eom", 
                "weight", 
                "leg" # +1 or -1
            ])
            
            # Store in the output dictionary
            op["weights"] = weights_df.collect()
            # --- END MODIFICATION ---

            pf_returns = sub.group_by(["pf", "eom"]).agg(
                [
                    pl.lit(x).alias("characteristic"),
                    pl.len().alias("n"),
                    pl.median("var").alias("signal"),
                    pl.mean("ret_exc_lead1m").alias("ret_ew"),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me")).sum()
                        / pl.col("me").sum()
                    ).alias("ret_vw"),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum()
                        / pl.col("me_cap").sum()
                    ).alias("ret_vw_cap"),
                ]
            )
            pf_returns = pf_returns.with_columns(
                pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom")
            )
            op["pf_returns"] = pf_returns.collect()

            if signals:
                if signals_w == "ew":
                    sub = sub.with_columns(
                        (1 / pl.col("eom").len()).over(["pf", "eom"]).alias("w")
                    )
                elif signals_w == "vw":
                    sub = sub.with_columns(
                        (pl.col("me") / pl.col("me").sum())
                        .over(["pf", "eom"])
                        .alias("w")
                    )
                elif signals_w == "vw_cap":
                    sub = sub.with_columns(
                        (pl.col("me_cap") / pl.col("me_cap").sum())
                        .over(["pf", "eom"])
                        .alias("w")
                    )

                sub = sub.with_columns(
                    [
                        pl.when(pl.col(var).is_null())
                        .then(pl.lit(0))
                        .otherwise(pl.col(var))
                        .alias(var)
                        for var in chars
                    ]
                )
                pf_signals = sub.with_columns(
                    [
                        (pl.col("w") * pl.col(var)).sum().over(["pf", "eom"])
                        for var in chars
                    ]
                )

                pf_signals = pf_signals.with_columns(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"),
                    ]
                )
                signals = pf_signals.clone()  # store in dictionary later
                op["signals"] = signals.collect()

            if daily_pf:
                weights = (
                    sub.group_by(["eom", "pf"])
                    .agg(
                        [
                            pl.col("id"),
                            (1 / pl.len()).alias("w_ew"),
                            (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                            (pl.col("me_cap") / pl.col("me_cap").sum()).alias(
                                "w_vw_cap"
                            ),
                        ]
                    )
                    .explode("id", "w_vw", "w_vw_cap")
                )

                daily_sub = weights.join(
                    daily.lazy(),
                    left_on=["id", "eom"],
                    right_on=["id", "eom_lag1"],
                    how="left",
                ).filter(
                    (pl.col("pf").is_not_null()) & (pl.col("ret_exc").is_not_null())
                )

                pf_daily = daily_sub.group_by(["pf", "date"]).agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n"),
                        ((pl.col("w_ew") * pl.col("ret_exc")).sum()).alias("ret_ew"),
                        ((pl.col("w_vw") * pl.col("ret_exc")).sum()).alias("ret_vw"),
                        ((pl.col("w_vw_cap") * pl.col("ret_exc")).sum()).alias(
                            "ret_vw_cap"
                        ),
                    ]
                )
                op["pf_daily"] = pf_daily.collect()

            char_pfs.append(op)

    output = {}
# ... existing output aggregation ...
    
    # Aggregate weights
    if len([op["weights"] for op in char_pfs if "weights" in op]) > 0:
        output["weights"] = pl.concat([op["weights"] for op in char_pfs if "weights" in op])
        
        # Add country info (USA)
        if output["weights"].height > 0:
            output["weights"] = output["weights"].with_columns(
                pl.lit(excntry).str.to_uppercase().alias("excntry")
            )
    # Aggregate pf_returns
    if len([op["pf_returns"] for op in char_pfs]) > 0:
        output["pf_returns"] = pl.concat([op["pf_returns"] for op in char_pfs])
    else:
        pass
    # Aggregate pf_daily if daily_pf is true
    if (daily_pf) and len([op["pf_daily"] for op in char_pfs]) > 0:
        output["pf_daily"] = pl.concat([op["pf_daily"] for op in char_pfs])
    else:
        pass
    # Handle industry portfolio returns if ind_pf is true
    if ind_pf:
        output["gics_returns"] = ind_gics  # Assuming ind_gics is a DataFrame
        if excntry == "usa":
            output["ff49_returns"] = (
                ind_ff49.clone()
            )  # Assuming ind_ff49 is a DataFrame

    # Add excntry to pf_returns and pf_daily, and aggregate signals
    if len(output) > 0:
        if "pf_returns" in output and output["pf_returns"].height > 0:
            output["pf_returns"] = output["pf_returns"].with_columns(
                pl.lit(excntry).str.to_uppercase().alias("excntry")
            )
            if daily_pf and "pf_daily" in output:
                output["pf_daily"] = output["pf_daily"].with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )
            if signals and "signals" in output:
                output["signals"] = pl.concat([op["signals"] for op in char_pfs])
                output["signals"] = output["signals"].with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )

    results = []
    # if (excntry=='usa' and cmp_key['us']) or (excntry!='usa' and cmp_key['int']):
    if cmp_key:
        for x in chars:
            print(f"   CMP - {x}: {chars.index(x) + 1} out of {len(chars)}")

            # Create a new column 'var' based on the current 'x'
            data = data.with_columns(pl.col(x).alias("var"))

            # Subsetting and ranking
            sub = data.filter(pl.col("var").is_not_null()).select(
                ["eom", "var", "size_grp", "ret_exc_lead1m"]
            )

            # Calculate ranks, rank deviations, and weights
            sub = (
                sub.with_columns(
                    (
                        (pl.col("var").rank("average").over("size_grp", "eom"))
                        / (pl.len().over("size_grp", "eom") + 1)
                    ).alias("p_rank")
                )
                .with_columns(
                    pl.col("p_rank").mean().over("size_grp", "eom").alias("mean_p_rank")
                )
                .with_columns(
                    (pl.col("p_rank") - pl.col("mean_p_rank")).alias("p_rank_dev")
                )
                .with_columns(
                    (pl.col("p_rank_dev") / ((pl.col("p_rank_dev").abs().sum()) / 2))
                    .over("size_grp", "eom")
                    .alias("weight")
                )
            )

            # Aggregation
            cmp = (
                sub.group_by(["size_grp", "eom"])
                .agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n_stocks"),
                        ((pl.col("ret_exc_lead1m") * pl.col("weight")).sum()).alias(
                            "ret_weighted"
                        ),
                        ((pl.col("var") * pl.col("weight")).sum()).alias(
                            "signal_weighted"
                        ),
                        pl.col("var").std().alias("sd_var"),
                    ]
                )
                .with_columns(pl.lit(excntry).alias("excntry"))
            )

            # Post-processing
            cmp = cmp.filter(pl.col("sd_var") != 0).drop("sd_var")
            cmp = cmp.with_columns(
                (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
            )

            results.append(cmp)

    if len(results) > 0:
        output_cmp = pl.concat(results)
        output_cmp = output_cmp.with_columns(
            pl.col("excntry").str.to_uppercase().alias("excntry")
        )
        output["cmp"] = output_cmp

    return output


# function for regional grouping of portfolios etc
def regional_data(
    data,
    mkt,
    date_col,
    char_col,
    countries,
    weighting,
    countries_min,
    periods_min,
    stocks_min,
):
    # Determine Country Weights
    weights = mkt.select(
        [
            pl.col("excntry"),
            pl.col(date_col).alias(date_col),
            pl.col("mkt_vw_exc"),
            pl.when(weighting == "market_cap")
            .then(pl.col("me_lag1"))
            .when(weighting == "stocks")
            .then(pl.col("stocks").cast(pl.Float64))
            .when(weighting == "ew")
            .then(1)
            .alias("country_weight"),
        ]
    )
    # Portfolio Return
    pf = data.filter(
        (pl.col("excntry").is_in(countries.implode()))
        & (pl.col("n_stocks_min") >= stocks_min)
    )
    pf = pf.join(weights, on=["excntry", date_col], how="left")
    pf = (
        pf.filter(pl.col("mkt_vw_exc").is_not_null())
        .group_by([char_col, date_col])
        .agg(
            [
                pl.len().alias("n_countries"),
                pl.col("direction").first().alias("direction"),
                (pl.col("ret_ew") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_ew"),
                (pl.col("ret_vw") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_vw"),
                (pl.col("ret_vw_cap") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_vw_cap"),
                (pl.col("mkt_vw_exc") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("mkt_vw_exc"),
            ]
        )
    )

    # Minimum Requirement: Countries
    pf = pf.filter(pl.col("n_countries") >= countries_min)

    # Minimum Requirement: Months
    pf = (
        pf.with_columns(pl.len().over(char_col).alias("periods"))
        .filter(pl.col("periods") >= periods_min)
        .drop("periods")
        .sort([char_col, date_col])
    )

    return pf


print(
    f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
    flush=True,
)
# Extract Neccesary Information
# Read Factor details from Excel file
char_info = (
    pl.read_excel(
        "data/factor_details.xlsx",
        sheet_name="details",
    )
    .filter(pl.col("abr_jkp").is_not_null())
    .select(
        [pl.col("abr_jkp").alias("characteristic"), pl.col("direction").cast(pl.Int32)]
    )
)

# Read country classification details from Excel file
country_classification = pl.read_excel(
    "data/country_classification.xlsx",
    sheet_name="countries",
)

# getting relevannt information from country classification file loaded at the start loaded at the start.
# Select columns
country_classification = country_classification.select(
    ["excntry", "msci_development", "region"]
)
# Filter out rows with NA in 'excntry' and exclude specific countries
country_classification = country_classification.filter(
    (pl.col("excntry").is_not_null())
    & (~pl.col("excntry").is_in(settings["regional_pfs"]["country_excl"]))
)

# Creating the regions DataFrame
regions = pl.DataFrame(
    {
        "name": ["developed", "emerging", "frontier", "world", "world_ex_us"],
        "country_codes": [
            country_classification.filter(
                (pl.col("msci_development") == "developed")
                & (pl.col("excntry") != "USA")
            )["excntry"].to_list(),
            country_classification.filter(pl.col("msci_development") == "emerging")[
                "excntry"
            ].to_list(),
            country_classification.filter(pl.col("msci_development") == "frontier")[
                "excntry"
            ].to_list(),
            country_classification["excntry"].to_list(),
            country_classification.filter(pl.col("excntry") != "USA")[
                "excntry"
            ].to_list(),
        ],
        "countries_min": [settings["regional_pfs"]["countries_min"]] * 3 + [1, 3],
    }
)


# Read cluster lables details from Excel file
cluster_labels = pl.read_csv(
    "data/cluster_labels.csv",
    infer_schema_length=int(1e10),
)


# nyse_cutoffs
nyse_size_cutoffs = pl.read_parquet(f"{data_path}/other_output/nyse_cutoffs.parquet")
# nyse_size_cutoffs = nyse_size_cutoffs.with_columns(pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom"))

# return_cutoffs
ret_cutoffs = pl.read_parquet(f"{data_path}/other_output/return_cutoffs.parquet")
# ret_cutoffs = ret_cutoffs.with_columns(pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom"))
ret_cutoffs = ret_cutoffs.with_columns(
    (pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
)
if settings["daily_pf"]:
    ret_cutoffs_daily = pl.read_parquet(
        f"{data_path}/other_output/return_cutoffs_daily.parquet"
    )

# market_returns
market = pl.read_parquet(f"{data_path}/other_output/market_returns.parquet")
# market = market.with_columns(pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom"))

# daily_market_returns
if settings["daily_pf"]:
    market_daily = pl.read_parquet(
        f"{data_path}/other_output/market_returns_daily.parquet"
    )
    # market_daily = market_daily.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date"))


# creating portfolios by using the portfolios function
portfolio_data = {}
for ex in countries:
    print(f"{ex}: {countries.index(ex) + 1} out of {len(countries)}")
    result = portfolios(
        data_path=data_path,
        excntry=ex,
        chars=chars,
        pfs=settings["pfs"],  # Number of portfolios
        bps=settings[
            "bps"
        ],  # What should breakpoints be based on? Non-Microcap stocks ("non_mc") or NYSE stocks "nyse"
        bp_min_n=settings["bp_min_n"],  # Minimum number of stocks used for breakpoints
        nyse_size_cutoffs=nyse_size_cutoffs,  # Data frame with NYSE size breakpoints
        source=settings[
            "source"
        ],  # Use data from "CRSP", "Compustat" or both: c("CRSP", "COMPUSTAT")
        wins_ret=settings[
            "wins_ret"
        ],  # Should Compustat returns be winsorized at the 0.1% and 99.9% of CRSP returns?
        cmp_key=settings["cmp"]["us"]
        if ex == "usa"
        else settings["cmp"]["int"],  # Create characteristics managed size portfolios?
        signals=settings["signals"]["us"]
        if ex == "usa"
        else settings["signals"]["int"],  # Create portfolio signals?
        signals_standardize=settings["signals"][
            "standardize"
        ],  # Map chars to [-0.5, +0.5]?,
        signals_w=settings["signals"][
            "weight"
        ],  # Weighting for signals: in c("ew", "vw", "vw_cap")
        daily_pf=settings["daily_pf"],  # Should daily return be estimated
        ind_pf=settings["ind_pf"],  # Should industry portfolio returns be estimated
        ret_cutoffs=ret_cutoffs,  # Data frame for monthly winsorization. Neccesary when wins_ret=T
        ret_cutoffs_daily=ret_cutoffs_daily,  # Data frame for daily winsorization. Neccesary when wins_ret=T and daily_pf=T
    )
    portfolio_data[ex] = result


# aggregating portfolio returns
# pf_returns = pl.concat([portfolio_data[sub_dict]['pf_returns']
#                         for sub_dict in portfolio_data
#                         if 'pf_returns' in portfolio_data[sub_dict]])

# pf_returns = pl.concat([portfolio_data[sub_dict]['pf_returns']
#                         for sub_dict in portfolio_data
#                          if sub_dict in portfolio_data and 'pf_returns' in portfolio_data[sub_dict]])


if any(
    sub_data and "pf_returns" in sub_data
    for sub_key, sub_data in portfolio_data.items()
):
    pf_returns = pl.concat(
        [
            sub_data["pf_returns"]
            for sub_key, sub_data in portfolio_data.items()
            if sub_data and "pf_returns" in sub_data
        ]
    )
    pf_returns = pf_returns.select(
        [
            "excntry",
            "characteristic",
            "pf",
            "eom",
            "n",
            "signal",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
        ]
    )
    pf_returns = pf_returns.sort(["excntry", "characteristic", "pf", "eom"])
else:
    pf_returns = None


if settings["daily_pf"] and any(
    sub_data and "pf_daily" in sub_data for sub_key, sub_data in portfolio_data.items()
):
    pf_daily = pl.concat(
        [
            sub_data["pf_daily"]
            for sub_key, sub_data in portfolio_data.items()
            if sub_data and "pf_returns" in sub_data
        ]
    )
    # pf_daily = pl.concat([portfolio_data[sub_dict]['pf_daily']
    #                     for sub_dict in portfolio_data
    #                     if 'pf_daily' in portfolio_data[sub_dict]])
    pf_daily = pf_daily.sort(["excntry", "characteristic", "pf", "date"])
else:
    pf_daily = None


# aggregating industry classification returns
# GICS Returns
if settings["ind_pf"] and any(
    sub_data and "gics_returns" in sub_data
    for sub_key, sub_data in portfolio_data.items()
):
    gics_returns = pl.concat(
        [
            sub_data["gics_returns"]
            for sub_key, sub_data in portfolio_data.items()
            if sub_data and "gics_returns" in sub_data
        ]
    )

    # gics_returns =pl.concat([portfolio_data[sub_dict]['gics_returns']
    #                     for sub_dict in portfolio_data
    #                     if 'gics_returns' in portfolio_data[sub_dict]])
    gics_returns = gics_returns.sort(["excntry", "gics", "eom"])

    # FF49 Returns
    if "usa" in countries:
        ff49_returns = portfolio_data["usa"]["ff49_returns"]
        ff49_returns = ff49_returns.sort(["excntry", "ff49", "eom"])
    else:
        ff49_returns = None
else:
    gics_returns = None


# Create HML Returns
if pf_returns is not None and pf_returns.height > 0:
    hml_returns = pf_returns.group_by(["excntry", "characteristic", "eom"]).agg(
        [
            pl.col("pf").is_in([settings["pfs"], 1]).sum().alias("pfs"),
            (
                pl.col("signal").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("signal").filter(pl.col("pf") == 1).first()
            ).alias("signal"),
            (
                pl.col("n").filter(pl.col("pf") == settings["pfs"]).first()
                + pl.col("n").filter(pl.col("pf") == 1).first()
            ).alias("n_stocks"),
            (pl.col("n").filter(pl.col("pf").is_in([settings["pfs"], 1])).min()).alias(
                "n_stocks_min"
            ),
            (
                pl.col("ret_ew").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("ret_ew").filter(pl.col("pf") == 1).first()
            ).alias("ret_ew"),
            (
                pl.col("ret_vw").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("ret_vw").filter(pl.col("pf") == 1).first()
            ).alias("ret_vw"),
            (
                pl.col("ret_vw_cap").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("ret_vw_cap").filter(pl.col("pf") == 1).first()
            ).alias("ret_vw_cap"),
        ]
    )

    hml_returns = hml_returns.filter(pl.col("pfs") == 2).drop("pfs")
    hml_returns = hml_returns.sort(["excntry", "characteristic", "eom"])

    # Create Long-Short Factors [Sign Returns to be consistent with original paper]
    lms_returns = char_info.join(hml_returns, on="characteristic", how="left")

    # Define columns to be modified
    resign_cols = ["signal", "ret_ew", "ret_vw", "ret_vw_cap"]
    lms_returns = lms_returns.with_columns(
        [pl.col(var) * pl.col("direction").alias(var) for var in resign_cols]
    )
else:
    hml_returns = None
    lms_returns = None

# daily hml and lms
if settings["daily_pf"] and pf_daily is not None and pf_daily.height > 0:
    hml_daily = pf_daily.group_by(["excntry", "characteristic", "date"]).agg(
        [
            pl.col("pf").is_in([settings["pfs"], 1]).sum().alias("pfs"),
            (
                pl.col("n").filter(pl.col("pf") == settings["pfs"]).first()
                + pl.col("n").filter(pl.col("pf") == 1).first()
            ).alias("n_stocks"),
            (pl.col("n").filter(pl.col("pf").is_in([settings["pfs"], 1])).min()).alias(
                "n_stocks_min"
            ),
            (
                pl.col("ret_ew").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("ret_ew").filter(pl.col("pf") == 1).first()
            ).alias("ret_ew"),
            (
                pl.col("ret_vw").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("ret_vw").filter(pl.col("pf") == 1).first()
            ).alias("ret_vw"),
            (
                pl.col("ret_vw_cap").filter(pl.col("pf") == settings["pfs"]).first()
                - pl.col("ret_vw_cap").filter(pl.col("pf") == 1).first()
            ).alias("ret_vw_cap"),
        ]
    )

    hml_daily = hml_daily.filter(pl.col("pfs") == 2).drop("pfs")
    hml_daily = hml_daily.sort(["excntry", "characteristic", "date"])

    lms_daily = char_info.join(hml_daily, on="characteristic", how="left")
    resign_cols = ["ret_ew", "ret_vw", "ret_vw_cap"]

    lms_daily = lms_daily.with_columns(
        [(pl.col(var) * pl.col("direction")).alias(var) for var in resign_cols]
    )
else:
    hml_daily = None
    lms_daily = None


# Extract CMP returns
cmp_list = [
    portfolio_data[sub_dict]["cmp"]
    for sub_dict in portfolio_data
    if "cmp" in portfolio_data[sub_dict]
]
if cmp_list:
    cmp_returns = pl.concat(cmp_list)
else:
    # Handle the empty list case here
    print("No 'cmp' keys found")


# Create Clustered Portfolios
if lms_returns is not None:
    cluster_pfs = (
        lms_returns.join(cluster_labels, on="characteristic", how="left")
        .group_by(["excntry", "cluster", "eom"])
        .agg(
            [
                pl.len().alias("n_factors"),
                pl.col("ret_ew").mean().alias("ret_ew"),
                pl.col("ret_vw").mean().alias("ret_vw"),
                pl.col("ret_vw_cap").mean().alias("ret_vw_cap"),
            ]
        )
    )
else:
    cluster_pfs = None

# Conditional Operation for Daily Clustered Portfolios
if settings["daily_pf"] and lms_daily is not None:
    cluster_pfs_daily = (
        lms_daily.join(cluster_labels, on="characteristic", how="left")
        .group_by(["excntry", "cluster", "date"])
        .agg(
            [
                pl.len().alias("n_factors"),
                pl.col("ret_ew").mean().alias("ret_ew"),
                pl.col("ret_vw").mean().alias("ret_vw"),
                pl.col("ret_vw_cap").mean().alias("ret_vw_cap"),
            ]
        )
    )
else:
    cluster_pfs_daily = None

# creating regional portfolios
if lms_returns is not None:
    regional_pfs = []
    for i in range(regions.height):
        info = regions[i][0]
        reg_pf = regional_data(
            data=lms_returns,
            mkt=market,
            countries=info["country_codes"][0],
            date_col="eom",
            char_col="characteristic",
            weighting=settings["regional_pfs"]["country_weights"],
            countries_min=info["countries_min"][0],
            periods_min=settings["regional_pfs"]["months_min"],
            stocks_min=settings["regional_pfs"]["stocks_min"],
        )
        reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
        reg_pf = reg_pf.select(
            [
                "region",
                "characteristic",
                "direction",
                "eom",
                "n_countries",
                "ret_ew",
                "ret_vw",
                "ret_vw_cap",
                "mkt_vw_exc",
            ]
        )
        regional_pfs.append(reg_pf)

    regional_pfs = pl.concat(regional_pfs)  # .explode('direction')

else:
    regional_pfs = None


if settings["daily_pf"] and lms_daily is not None:
    regional_pfs_daily = []
    for i in range(regions.height):
        info = regions[i][0]
        reg_pf = regional_data(
            data=lms_daily,
            mkt=market_daily,
            countries=info["country_codes"][0],
            date_col="date",
            char_col="characteristic",
            weighting=settings["regional_pfs"]["country_weights"],
            countries_min=info["countries_min"][0],
            periods_min=settings["regional_pfs"]["months_min"] * 21,
            stocks_min=settings["regional_pfs"]["stocks_min"],
        )
        reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
        reg_pf = reg_pf.select(
            [
                "region",
                "characteristic",
                "direction",
                "date",
                "n_countries",
                "ret_ew",
                "ret_vw",
                "ret_vw_cap",
                "mkt_vw_exc",
            ]
        )

        regional_pfs_daily.append(reg_pf)  # .explode('direction')

    regional_pfs_daily = pl.concat(regional_pfs_daily)

else:
    regional_pfs_daily = None


# creating regional clusters
if cluster_pfs is not None:
    regional_clusters = []
    for i in range(regions.height):
        info = regions[i][0]
        reg_pf = cluster_pfs.rename({"n_factors": "n_stocks_min"})
        reg_pf = reg_pf.with_columns(
            pl.lit(None).cast(pl.Float64).alias("direction")
        )  # Adding 'direction' column with NA values
        reg_pf = regional_data(
            data=reg_pf,
            mkt=market,
            countries=info["country_codes"][0],
            date_col="eom",
            char_col="cluster",
            weighting=settings["regional_pfs"]["country_weights"],
            countries_min=info["countries_min"][0],
            periods_min=settings["regional_pfs"]["months_min"],
            stocks_min=1,
        )
        reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
        reg_pf = reg_pf.select(
            [
                "region",
                "cluster",
                "eom",
                "n_countries",
                "ret_ew",
                "ret_vw",
                "ret_vw_cap",
                "mkt_vw_exc",
            ]
        )

        regional_clusters.append(reg_pf)

    regional_clusters = pl.concat(regional_clusters)
else:
    regional_clusters = None


if settings["daily_pf"] and cluster_pfs_daily is not None:
    regional_clusters_daily = []
    for i in range(regions.height):
        info = regions[i][0]
        reg_pf = cluster_pfs_daily.rename({"n_factors": "n_stocks_min"})
        reg_pf = reg_pf.with_columns(
            pl.lit(None).cast(pl.Float64).alias("direction")
        )  # Adding 'direction' column with NA values
        reg_pf = regional_data(
            data=reg_pf,
            mkt=market_daily,
            countries=info["country_codes"][0],
            date_col="date",
            char_col="cluster",
            weighting=settings["regional_pfs"]["country_weights"],
            countries_min=info["countries_min"][0],
            periods_min=settings["regional_pfs"]["months_min"] * 21,
            stocks_min=1,
        )
        reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
        reg_pf = reg_pf.select(
            [
                "region",
                "cluster",
                "date",
                "n_countries",
                "ret_ew",
                "ret_vw",
                "ret_vw_cap",
                "mkt_vw_exc",
            ]
        )
        regional_clusters_daily.append(reg_pf)

    regional_clusters_daily = pl.concat(regional_clusters_daily)
else:
    regional_clusters_daily = None


# Writing output
# if "market" in globals():
# market.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
#     f"{output_path}/market_returns.parquet"
# )
if "pf_returns" in globals() and pf_returns is not None:
    pf_returns.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
        f"{output_path}/pfs.parquet"
    )
if "hml_returns" in globals() and hml_returns is not None:
    hml_returns.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
        f"{output_path}/hml.parquet"
    )
if "lms_returns" in globals() and lms_returns is not None:
    lms_returns.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
        f"{output_path}/lms.parquet"
    )
if "cmp_returns" in globals() and cmp_returns is not None:
    cmp_returns.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
        f"{output_path}/cmp.parquet"
    )
if "cluster_pfs" in globals() and cluster_pfs is not None:
    cluster_pfs.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
        f"{output_path}/clusters.parquet"
    )


if settings["daily_pf"]:
    # if "market_daily" in globals():
    #     market_daily.filter(pl.col("date") <= settings["end_date"]).write_parquet(
    #         f"{output_path}/market_returns_daily.parquet"
    #     )
    if "pf_daily" in globals() and pf_daily is not None:
        pf_daily.filter(pl.col("date") <= settings["end_date"]).write_parquet(
            f"{output_path}/pfs_daily.parquet"
        )
    if "hml_daily" in globals() and hml_daily is not None:
        hml_daily.filter(pl.col("date") <= settings["end_date"]).write_parquet(
            f"{output_path}/hml_daily.parquet"
        )
    if "lms_daily" in globals() and lms_daily is not None:
        lms_daily.filter(pl.col("date") <= settings["end_date"]).write_parquet(
            f"{output_path}/lms_daily.parquet"
        )
    if "cluster_pfs_daily" in globals() and cluster_pfs_daily is not None:
        cluster_pfs_daily.filter(pl.col("date") <= settings["end_date"]).write_parquet(
            f"{output_path}/clusters_daily.parquet"
        )







# ... existing write_parquet calls ...

# Save Single Stock Constituent Weights
if "portfolio_data" in globals():
    # Extract weights from the nested dictionary structure
    all_weights_list = [
        sub_data["weights"] 
        for sub_key, sub_data in portfolio_data.items() 
        if sub_data and "weights" in sub_data
    ]
    
    if all_weights_list:
        print("Saving constituent weights...", flush=True)
        all_weights = pl.concat(all_weights_list)
        
        # Filter by date
        all_weights = all_weights.filter(pl.col("eom") <= settings["end_date"])
        
        # Save to parquet
        all_weights.write_parquet(f"{output_path}/usa_factor_weights.parquet")
        print(f"Weights saved to {output_path}/usa_factor_weights.parquet")

print(
    f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
    flush=True,
)






if settings["ind_pf"]:
    if "gics_returns" in globals() and gics_returns is not None:
        gics_returns.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
            f"{output_path}/industry_gics.parquet"
        )
    if "ff49_returns" in globals() and ff49_returns is not None:
        ff49_returns.filter(pl.col("eom") <= settings["end_date"]).write_parquet(
            f"{output_path}/industry_ff49.parquet"
        )


# Create directory for Regional Factors
if "regional_pfs" in globals() and regional_pfs is not None:
    reg_folder = os.path.join(output_path, "regional_factors")
    if not os.path.exists(reg_folder):
        os.makedirs(reg_folder)

    # Write regional portfolios to parquet files
    for reg in regional_pfs["region"].unique():
        filtered_df = regional_pfs.filter(
            (pl.col("eom") <= settings["end_date"]) & (pl.col("region") == reg)
        )
        file_path = os.path.join(reg_folder, f"{reg}.parquet")
        filtered_df.write_parquet(file_path)

# Conditional block for daily regional factors
if settings["daily_pf"]:
    if "regional_pfs_daily" in globals() and regional_pfs_daily is not None:
        # Create directory for Daily Regional Factors
        reg_folder_daily = os.path.join(output_path, "regional_factors_daily")
        if not os.path.exists(reg_folder_daily):
            os.makedirs(reg_folder_daily)

        # Write daily regional portfolios to parquet files
        for reg in regional_pfs_daily["region"].unique():
            filtered_df_daily = regional_pfs_daily.filter(
                (pl.col("date") <= settings["end_date"]) & (pl.col("region") == reg)
            )
            file_path_daily = os.path.join(reg_folder_daily, f"{reg}.parquet")
            filtered_df_daily.write_parquet(file_path_daily)


# Create directory for Regional Clusters
if "regional_clusters" in globals() and regional_clusters is not None:
    reg_folder = os.path.join(output_path, "regional_clusters")
    if not os.path.exists(reg_folder):
        os.makedirs(reg_folder)

    # Write regional clusters to parquet files
    for reg in regional_clusters["region"].unique():
        filtered_df = regional_clusters.filter(
            (pl.col("eom") <= settings["end_date"]) & (pl.col("region") == reg)
        )
        file_path = os.path.join(reg_folder, f"{reg}.parquet")
        filtered_df.write_parquet(file_path)

# Conditional block for daily regional clusters
if settings["daily_pf"]:
    if "regional_clusters_daily" in globals() and regional_clusters_daily is not None:
        # Create directory for Daily Regional Clusters
        reg_folder_daily = os.path.join(output_path, "regional_clusters_daily")
        if not os.path.exists(reg_folder_daily):
            os.makedirs(reg_folder_daily)

        # Write daily regional clusters to parquet files
        for reg in regional_clusters_daily["region"].unique():
            filtered_df_daily = regional_clusters_daily.filter(
                (pl.col("date") <= settings["end_date"]) & (pl.col("region") == reg)
            )
            file_path_daily = os.path.join(reg_folder_daily, f"{reg}.parquet")
            filtered_df_daily.write_parquet(file_path_daily)


# Create directory for Country Factors
if "lms_returns" in globals() and lms_returns is not None:
    cnt_folder = os.path.join(output_path, "country_factors")
    if not os.path.exists(cnt_folder):
        os.makedirs(cnt_folder)

    # Write country factors to parquet files
    for exc in lms_returns["excntry"].unique():
        if exc:
            filtered_df = lms_returns.filter(
                (pl.col("eom") <= settings["end_date"]) & (pl.col("excntry") == exc)
            )
            file_path = os.path.join(cnt_folder, f"{exc}.parquet")
            filtered_df.write_parquet(file_path)

# Conditional block for daily country factors
if settings["daily_pf"]:
    if "lms_daily" in globals() and lms_daily is not None:
        # Create directory for Daily Country Factors
        cnt_folder_daily = os.path.join(output_path, "country_factors_daily")
        if not os.path.exists(cnt_folder_daily):
            os.makedirs(cnt_folder_daily)

        # Write daily country factors to parquet files
        for exc in lms_daily["excntry"].unique():
            if exc:
                filtered_df_daily = lms_daily.filter(
                    (pl.col("date") <= settings["end_date"])
                    & (pl.col("excntry") == exc)
                )
                file_path_daily = os.path.join(cnt_folder_daily, f"{exc}.parquet")
                filtered_df_daily.write_parquet(file_path_daily)

print(
    f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
    flush=True,
)

