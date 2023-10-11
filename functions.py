def populate_own(inset, idvar, datevar, datename, forward_max, period):
    # sort the input according to id and date respectively and remove duplicates: arg: inset=, datevar=, idvar
    temp_df = inset.sort([idvar, datevar]).unique([idvar, datevar], keep='last')

    if period == 'month':

        # add a new date column that has month end dates: arg: datename=:
        temp_df = temp_df.with_columns(pl.col(datevar).dt.month_end().alias(datename))

        # create an empty dataframe
        output = temp_df.clear()

        # sort into groups according to ids to forward fill:
        for idx, group in temp_df.group_by(idvar):
            min_date = group.select(pl.min(datename))[0,0]
            max_date = group.select(pl.max(datename))[0,0] + relativedelta(months=forward_max)
            max_date = max_date.replace(day=1)
            max_date = max_date - datetime.timedelta(days=1)

            all_dates = []
            current_date = min_date

            while current_date <= max_date:
                all_dates.append(current_date)
                current_date += relativedelta(months=1)

            all_dates_df = pl.DataFrame({datename: all_dates})
            all_dates_df = all_dates_df.select(pl.col(datename).dt.month_end())
            all_dates_df = all_dates_df.join(group, on=datename, how="left")
            all_dates_df = all_dates_df.select(pl.all().forward_fill(limit=forward_max))
            all_dates_df = all_dates_df.select(output.columns)

            output = output.vstack(all_dates_df)

        return output

    else:

        # add a new date column that has daily dates: arg: datename=:
        temp_df = temp_df.with_columns(pl.col(datevar).alias(datename))

        # create an empty dataframe
        output = temp_df.clear()

        # sort into groups according to ids to forward fill:
        for idx, group in temp_df.group_by(idvar):
            min_date = group.select(pl.min(datename))[0,0]
            max_date = group.select(pl.max(datename))[0,0] + relativedelta(days=forward_max)
            # max_date = max_date.replace(day=1)
            # max_date = max_date - datetime.timedelta(days=1)

            all_dates = []
            current_date = min_date

            while current_date <= max_date:
                all_dates.append(current_date)
                current_date += relativedelta(days=1)
            
            all_dates_df = pl.DataFrame({datename: all_dates})
            all_dates_df = all_dates_df.join(group, on=datename, how="left")
            all_dates_df = all_dates_df.select(pl.all().forward_fill(limit=forward_max))
            all_dates_df = all_dates_df.select(output.columns)

            output = output.vstack(all_dates_df)

        return output




def compustat_fx(exrt_dly):
    usd_curcdd = pl.DataFrame({
        'curcdd': ['USD'],
        'datadate': [19500101],
        'fx': [1.00]
    })

    
    a = exrt_dly.filter(pl.col('fromcurd') == 'GBP')
    b = exrt_dly.filter(pl.col('tocurd') == 'USD')

    # Convert the 'datadate' column to Date type
    usd_curcdd = usd_curcdd.with_columns(
        pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
    )

    fx1 = (
        a.join(b, on=['fromcurd', 'datadate'])
        .with_columns((pl.col('exratd') / pl.col('exratd_right')).alias('fx'))
        .select([pl.col('tocurd').alias('curcdd'), 'datadate', 'fx'])
    )

    # Step 4: Merging USD Base Data with Extracted FX Data
    fx2 = fx1.vstack(usd_curcdd)

    # Step 5: Sorting the Data
    fx2 = fx2.sort(['curcdd', 'datadate'], descending=[False, False])
    fx3 = populate_own(inset=fx2, idvar='curcdd', datevar='datadate', datename='date', forward_max=12, period='daily')
    fx3 = fx3.filter(~pl.all_horizontal(pl.col("curcdd").is_null()))
    fx3=fx3.drop("datadate").select([pl.col('curcdd'), pl.col('date').alias('datadate'), pl.col("fx")])

   

    
    # Step 7: Final Sorting and Deduplication
    output = fx3.sort(['curcdd', 'datadate']).unique(['curcdd', 'datadate'])

    return output


def parse_dates(df, column_name="datadate", specified_format=None):
    """
    Parse a date column in the given DataFrame using Polars API with a specified format or multiple default formats.
    
    Parameters:
    - df: The input Polars DataFrame.
    - column_name: The name of the column to parse. Default is "datadate".
    - specified_format: A specific date format to try first. If not provided, defaults will be used.
    
    Returns:
    - A Polars DataFrame with the parsed date column.
    """
    formats = ["%Y%m%d", "%Y-%m-%d", "%d-%m-%Y"]
    
    # If a specified format is provided, prioritize it
    if specified_format:
        formats.insert(0, specified_format)
    
    for fmt in formats:
        try:
            # Attempt to parse the date column with the current format
            df = df.with_columns(
                pl.col(column_name).cast(pl.Utf8).str.strptime(pl.Date, format=fmt).alias(column_name)
            )
            # If parsing succeeds without errors, break out of the loop
            break
        except:
            # If parsing fails, continue to the next format
            continue
    
    return df

def CRSP_INDUSTRY():
    """
    Create daily historical SIC and NAICS industry identifiers dataset from CRSP data 
    
    Parameters:
    - None

    Returns:
    - Historical SIC and NAICS industry identifiers
    """
    
    # Load the CSV file into a DataFrame
    CRSP_DSENAMES = pl.read_csv("path", ignore_errors=True)
    
    # Convert date columns from string to date format
    CRSP_DSENAMES = CRSP_DSENAMES.with_columns(
        pl.col("NAMEDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("NAMEDT"),
        pl.col("NAMEENDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("NAMEENDT")
    )

    # Select relevant columns, rename, sort, and remove duplicates
    permno0 = CRSP_DSENAMES.select(
        pl.col("PERMNO"),
        pl.col("PERMCO"),
        pl.col("NAMEDT"),
        pl.col("NAMEENDT"),
        pl.col("SICCD").alias("SIC"),
        pl.col("NAICS").cast(pl.Int32).alias("NAICS"),
    ).unique().sort(["PERMNO", "NAMEDT", "NAMEENDT"])

    # Replace missing or zero values in SIC and NAICS with -999
    permno1 = permno0.with_columns(
        pl.when(pl.col("SIC").is_null() | (pl.col("SIC") == 0))
        .then(-999)
        .otherwise(pl.col("SIC"))
        .alias("SIC")
    ).with_columns(
        pl.when(pl.col("NAICS").is_null())
        .then(-999)
        .otherwise(pl.col("NAICS"))
        .alias("NAICS")        
    )
    
    # Calculate duration between NAMEENDT and NAMEDT dates
    permno2 = permno1.with_columns(
        (pl.col("NAMEENDT") - pl.col("NAMEDT")).alias("PERMNO_DIFF")
    )

    # Expand rows for each date in the range between NAMEDT and NAMEENDT
    permno3 = permno2.with_columns(pl.date_ranges("NAMEDT", "NAMEENDT")).explode("date_range")
    permno3 = permno3.select(["PERMNO", "PERMCO", "SIC", "NAICS", pl.col("date_range").alias("NAMEDT")])

    # Convert the placeholder value -999 back to None for SIC and NAICS columns and rename the NAMEDT column to DATE
    permno4 = permno3.with_columns(
        pl.when(pl.col("SIC") == -999)
        .then(pl.lit(None).cast(pl.Int64))
        .otherwise(pl.col("SIC"))
        .alias("SIC")
    ).with_columns(
        pl.when(pl.col("NAICS") == -999)
        .then(pl.lit(None).cast(pl.Int64))
        .otherwise(pl.col("NAICS"))
        .alias("NAICS")
    ).rename({"NAMEDT": "DATE"})

    # Sort the DataFrame by PERMNO and DATE and remove duplicates based on these columns
    out = permno4.sort(["PERMNO", "DATE"]).unique(["PERMNO", "DATE"])
    
    return out

def COMP_SIC_NAICS():
    """
    Create a daily historical SIC and NAICS industry identifiers dataset using NA and global annual reports
    
    Parameters:
    - None

    Returns:
    - Historical SIC and NAICS industry identifiers dataset
    """
    
    # Load the COMP_FUNDA CSV file into a DataFrame and format relevant columns
    COMP_FUNDA = pl.read_csv("path", infer_schema_length=int(1e13), ignore_errors=True)
    COMP_FUNDA = COMP_FUNDA.with_columns(
        pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
    )
    COMP_FUNDA = COMP_FUNDA.with_columns(pl.col("gvkey").cast(pl.Utf8).alias("gvkey"))

    # Load the COMP_G_FUNDA CSV file into a DataFrame and format relevant columns
    COMP_G_FUNDA = pl.read_csv("path", infer_schema_length=int(1e10), ignore_errors=True)
    COMP_G_FUNDA = COMP_G_FUNDA.with_columns(
        pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
    )
    COMP_G_FUNDA = COMP_G_FUNDA.with_columns(pl.col("gvkey").cast(pl.Utf8).alias("gvkey"))

    # Select relevant columns from COMP_FUNDA and remove duplicates
    comp1 = COMP_FUNDA.select(["gvkey", "datadate", pl.col("sich").alias("sic"), pl.col("naicsh").alias("naics")]).unique()

    # Filter out a specific set of data based on given conditions
    comp2 = comp1.filter(~((pl.col("gvkey") == "175650") & 
                           (pl.col("datadate") == pl.date(2005,12,31)) & 
                           (pl.col("naics").is_null())))

    # Select relevant columns from COMP_G_FUNDA and remove duplicates
    comp3 = COMP_G_FUNDA.select(["gvkey", "datadate", pl.col("sich").alias("sic"), pl.col("naicsh").alias("naics")]).unique()

    # Outer join comp2 and comp3 DataFrames on 'gvkey' and 'datadate'
    comp4 = comp2.join(comp3, on=["gvkey", "datadate"], how="outer")

    # Create a DataFrame selecting and re-assigning columns based on conditions
    comp5 = comp4.select([
        "gvkey", 
        pl.col("datadate").alias("date"), 
        pl.when(pl.col("sic").is_null()).then(pl.col("sic_right")).otherwise(pl.col("sic")).alias("sic"),
        pl.when(pl.col("naics").is_null()).then(pl.col("naics_right")).otherwise(pl.col("naics")).alias("naics")
    ])

    # Sort DataFrame by 'gvkey' and 'date', and prepare for date range calculations
    comp6 = comp5.sort(["gvkey", "date"])
    comp6 = comp6.with_columns(
        pl.when(pl.col('gvkey') != pl.col('gvkey').shift(-1))
        .then(True)
        .otherwise(False)
        .alias("mask")
    )
    comp6 = comp6.with_columns(pl.col("date").shift(-1).alias("end_date"))
    comp6 = comp6.with_columns(
        pl.when(pl.col('mask'))
        .then(pl.lit(None))
        .otherwise(pl.col("end_date"))
        .alias("updated_end_date")
    )
    comp6 = comp6.with_columns(
        pl.when(pl.col('updated_end_date').is_null())
        .then(pl.col("date"))
        .otherwise(pl.col("updated_end_date"))
        .alias("valid_to")
    )

    # Expand rows for each date in the range between 'date' and 'valid_to'
    comp7 = comp6.with_columns(pl.date_ranges("date", "valid_to")).explode("date_range")

    # Select and rename the necessary columns
    comp8 = comp7.select(["gvkey", "date_range", "sic", "naics"])
    comp8 = comp8.rename({"date_range": "date"})

    # Remove duplicates based on 'gvkey' and 'date' and sort the DataFrame
    out = comp8.unique(subset=["gvkey", "date"], keep="last").sort(["gvkey", "date"])

    return out

def COMP_HGICS(lib):
    """
    Create a daily historical gics dataset from COMPUSTAT, either from the NA or global dataset 
    
    Parameters:
    - lib: A string, if 'global' will process COMP_G_CO_HGIC dataset, otherwise will process COMP_CO_HGIC dataset.

    Returns:
    - Daily historical gics dataset from COMPUSTAT
    """
    
    # If lib parameter is 'global', read COMP_G_CO_HGIC dataset, else read COMP_CO_HGIC dataset
    if lib == "global":
        COMP_G_CO_HGIC = pl.read_csv("path", infer_schema_length=int(1e13), ignore_errors=True)
        COMP_G_CO_HGIC = COMP_G_CO_HGIC.with_columns(
            pl.col("indfrom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indfrom"),
            pl.col("indthru").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indthru"),
            pl.col("gvkey").cast(pl.Utf8),
            pl.col("indtype").cast(pl.Utf8),
            pl.col("ggroup").cast(pl.Utf8),
            pl.col("gind").cast(pl.Utf8),
            pl.col("gsector").cast(pl.Utf8),
            pl.col("gsubind").cast(pl.Utf8)
        )
        data = COMP_G_CO_HGIC
    else:
        COMP_CO_HGIC = pl.read_csv("path", infer_schema_length=int(1e13), ignore_errors=True)
        COMP_CO_HGIC = COMP_CO_HGIC.with_columns(
            pl.col("indfrom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indfrom"),
            pl.col("indthru").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indthru"),
            pl.col("gvkey").cast(pl.Utf8),
            pl.col("indtype").cast(pl.Utf8),
            pl.col("ggroup").cast(pl.Utf8),
            pl.col("gind").cast(pl.Utf8),
            pl.col("gsector").cast(pl.Utf8),
            pl.col("gsubind").cast(pl.Utf8)
        )
        data = COMP_CO_HGIC
        

    # Filter rows where 'gvkey' is not null, select relevant columns, and remove duplicates
    gic1 = data.filter(pl.col("gvkey").is_not_null()).select([
        "gvkey", "indfrom", "indthru", pl.col("gsubind").alias("gics")]
    ).unique().sort(["gvkey", "indfrom"])

    # Replace null 'gics' values with -999
    gic2 = gic1.with_columns(
        pl.when(pl.col("gics").is_null()).then(-999).otherwise(pl.col("gics")).alias("gics")
    )

    # Handle missing 'indthru' values
    gic3 = gic2.with_columns(
        pl.when((pl.col("indthru").is_null()) & (pl.col("gvkey") != pl.col("gvkey").shift(-1)))
        .then(pl.lit(date.today()))
        .otherwise(pl.col("indthru"))
        .alias("indthru")
    )

    # Handle missing 'indthru' for the last row
    gic3 = gic3.with_row_count().with_columns(
        pl.when((pl.col("row_nr") == (gic3.height-1)) & (pl.col("indthru").is_null()))
        .then(pl.lit(date.today()))
        .otherwise(pl.col("indthru"))
        .alias("indthru")
    ).drop("row_nr")

    # Expand rows for each date in the range between 'indfrom' and 'indthru'
    gic4 = gic3.with_columns(pl.date_ranges("indfrom", "indthru")).explode("date_range")

    # Select and rename the necessary columns
    gic5 = gic4.select(["gvkey", "gics", pl.col("date_range").alias("date")])

    # Remove duplicates based on 'gvkey' and 'date' and sort the DataFrame
    out = gic5.sort(["gvkey", "date"]).unique(subset=["gvkey", "date"], keep="last")

    return out



def HGICS_JOIN():
    """
    Join NA and global daily historical gics data from COMPUSTAT.

    Parameters:
    - None

    Returns:
    - A Polars DataFrame with joined and cleaned GICS and date information.
    """
    g_hgics = COMP_HGICS(lib="global")
    na_hgics = COMP_HGICS(lib="local")

    # Join the local_data and global_data based on 'gvkey' and 'date' using an outer join
    gjoin1 = local_data.join(global_data, on=["gvkey", "date"], how="outer")

    # Handle missing 'gics' values. If 'gics' from local_data is missing, use 'gics' from global_data
    gjoin2 = gjoin1.with_columns(
        pl.when(pl.col("gics").is_null()).then(pl.col("gics_right")).otherwise(pl.col("gics")).alias("gics")
    ).select(["gvkey", "date", "gics"])

    # Remove duplicates based on 'gvkey' and 'date' and sort the DataFrame
    out = gjoin2.sort(["gvkey", "date"]).unique(subset=["gvkey", "date"], keep="last")

    return out



def comp_industry():
    """
    Joins SIC and NAICS industry identifiers to GICS identifiers constructed from COMPUSTAT data

    Parameters:
    - None

    Returns:
    - Combined SIC, NAICS, and GICS identifiers constructed from COMPUSTAT data
    """
    comp_other= COMP_SIC_NAICS()
    comp_gics=HGICS_JOIN()

    # Step 1: Perform an outer join between gics_table and other_table based on 'gvkey' and 'date' and sort the resulting DataFrame
    join1 = gics_table.join(other_table, on=["gvkey", "date"], how="outer")
    join1 = join1.sort(["gvkey", "date"])

    # Step 2: Create a mask column to identify the last row for each 'gvkey' group
    join2 = join1.with_columns(
        pl.when(pl.col('gvkey') != pl.col('gvkey').shift(-1))
        .then(True)
        .otherwise(False)
        .alias("mask")
    )

    # Step 3: Shift the 'date' column to compute end dates
    join3 = join2.with_columns(pl.col("date").shift(-1).alias("end_date"))

    # Step 4: Update the end date based on the mask created in Step 2
    join4 = join3.with_columns(
        pl.when(pl.col('mask'))
        .then(pl.col("date"))
        .otherwise(pl.col("end_date")) 
        .alias("updated_end_date")
    )

    # Step 5: Handle the end date for the very last row
    join5 = join4.with_row_count().with_columns(
        pl.when((pl.col("row_nr") == (join4.height-1)) & (pl.col("updated_end_date").is_null()))
        .then(pl.col("date"))
        .otherwise(pl.col("updated_end_date"))
        .alias("final_end_date")
    )

    # Step 6: Generate a range of dates between the 'date' and 'final_end_date'
    join6 = join5.with_columns(pl.date_ranges("date", "final_end_date", closed="left"))

    # Step 7: Explode the DataFrame based on the date range
    join7 = join6.explode("date_range")

    # Step 8: Create a mask to identify rows with a change in 'gvkey'
    join8 = join7.with_columns(
        pl.when(pl.col('row_nr') != pl.col('row_nr').shift(1))
        .then(False)
        .otherwise(True)
        .alias("mask_orig")
    ).drop("row_nr")

    # Step 9: Adjust the mask for the first row
    join9 = join8.with_row_count().with_columns(
        pl.when(pl.col("row_nr") == 0).then(False)
        .otherwise(pl.col("mask_orig"))
        .alias("mask_orig")
    ).drop("row_nr")

    # Step 10: Reset industry classification columns based on the mask
    join10 = join9.with_columns([
        pl.when(pl.col("mask_orig"))
        .then(pl.lit(None))
        .otherwise(pl.col("gics"))
        .alias("gics"),
        
        pl.when(pl.col("mask_orig"))
        .then(pl.lit(None))
        .otherwise(pl.col("sic"))
        .alias("sic"),
        
        pl.when(pl.col("mask_orig"))
        .then(pl.lit(None))
        .otherwise(pl.col("naics"))
        .alias("naics"),
    ])

    # Step 11: Rename the 'date_range' column and filter the relevant columns
    join11 = join10.select(["gvkey", pl.col("date_range").alias("date"), "gics", "sic", "naics"])

    # Step 12: Remove duplicates and sort the final DataFrame
    join12 = join11.unique(["gvkey", "date"], keep="first")
    out = join12.sort(["gvkey", "date"])

    return out