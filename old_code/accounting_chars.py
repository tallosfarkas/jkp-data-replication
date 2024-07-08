#importing all the required packages:
import polars as pl
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import math
import numpy as np
import statsmodels.api as sm


COMP_FUNDQ = pl.read_csv("path.csv", infer_schema_length=int(1e13), ignore_errors=True)
COMP_FUNDQ = COMP_FUNDQ.with_columns(
    pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
)

COMP_G_FUNDQ = pl.read_csv("path.csv", infer_schema_length=int(1e13), ignore_errors=True)
COMP_G_FUNDQ = COMP_G_FUNDQ.with_columns(
    pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
)COMP_FUNDA = pl.read_csv("path.csv", infer_schema_length=int(1e13), ignore_errors=True)
COMP_FUNDA = COMP_FUNDA.with_columns(
    pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
)

COMP_G_FUNDA = pl.read_csv("path.csv", infer_schema_length=int(1e13), ignore_errors=True)
COMP_G_FUNDA = COMP_G_FUNDA.with_columns(
    pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
)
COMP_EXRT_DLY = pl.read_csv("path.csv", infer_schema_length=int(1e10), ignore_errors=True)
COMP_EXRT_DLY = COMP_EXRT_DLY.with_columns(
    pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
)
fx = pl.read_csv("path.csv", infer_schema_length=int(1e10), ignore_errors=True)
fx = fx.with_columns(
    pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date")
)
world_msf = pl.read_csv("path.csv", infer_schema_length=int(1e13), ignore_errors=True)
world_msf = world_msf.with_columns([
    pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("DATE"), 
    pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom")]
)


def standardized_accounting_data(coverage, convert_to_usd, me_data, include_helpers_vars=1, start_date=date(1950,1,1)):
    """
    Standardizes the accounting data from compustat along two dimensions: geographical coverage and frequency. It harmonizes the columns in na and global data and between annual and quarterly data.
    Dependant on functions: quarterize, expand, and add_helper_vars

    Parameters:
    - coverage: A string. Options: 'na', 'global', and 'world'
    - convert_to_usd: An binary. Options: 0 and 1 (default=0)
    - me_data: dataframe containing the market data
    - include_helpers_vars: A binary. Options 0 and 1 (default=1)
    - start_date. Data format, specifying the date to start. (default date(1950,1,1))

    Returns:
    - two datasets, annual and quarterly. Each contains standardized accounting data of relevant frequancy.
    """
    
    # Compustat Accounting Vars to Extract
    avars_inc = [
    "sale", "revt", "gp", "ebitda", "oibdp", "ebit", "oiadp", "pi", "ib", "ni", "mii",
    "cogs", "xsga", "xopr", "xrd", "xad", "xlr", "dp", "xi", "do", "xido", "xint", "spi", "nopi", "txt",
    "dvt"
    ]

    avars_cf = [
    # Operating
    "oancf", "ibc", "dpc", "xidoc", "capx", "wcapt",
    
    # Financing
    "fincf", "fiao", "txbcof", "ltdch", "dltis", "dltr", "dlcch", "purtshr", "prstkc", "sstk",
    "dv", "dvc"
    ]

    avars_bs = [
    # Assets
    "at", "act", "aco", "che", "invt", "rect", "ivao", "ivst", "ppent", "ppegt", "intan", "ao", "gdwl", "re",
    
    # Liabilities
    "lt", "lct", "dltt", "dlc", "txditc", "txdb", "itcb", "txp", "ap", "lco", "lo",
    "seq", "ceq", "pstkrv", "pstkl", "pstk", "mib", "icapt"
    ]

    # Variables in avars_other are not measured in currency units, and only available in annual data
    avars_other = ["emp"]

    avars = avars_inc + avars_cf + avars_bs
    print(f"INCOME STATEMENT: {len(avars_inc)} || CASH FLOW STATEMENT: {len(avars_cf)} || BALANCE SHEET: {len(avars_bs)} || OTHER: {len(avars_other)}")

    #finding which variables of interest are available in the quarterly data
    combined_columns = COMP_FUNDQ.columns + COMP_G_FUNDQ.columns
    qvars_q = list({col for col in combined_columns if col[:-1].lower() in avars and col.endswith("q")}) #different from above to get only unique values
    qvars_y = list({col for col in combined_columns if col[:-1].lower() in avars and col.endswith("y")})
    qvars = qvars_q + qvars_y

    #preparing global data
    #annual
    if coverage in ['global', 'world']:
        compcond = (pl.col('indfmt').is_in(['INDL', 'FS']) & 
                    (pl.col('datafmt') == 'HIST_STD') & 
                    (pl.col('popsrc') == 'I') & 
                    (pl.col('consol') == 'C') & 
                    (pl.col('datadate') >= start_date))
    
    # For g_funda dataset:
    g_funda = COMP_G_FUNDA.clone()  
# Filtering and creating new columns for g_funda dataset
# Variables Not Available in G_FUNDA with Replacement
    g_funda1 = (
        g_funda.filter(compcond)
        .with_columns(pl.lit('GLOBAL').alias('source'))
        .with_columns(
            (pl.col('ib') + pl.col('xi').fill_null(0) + pl.col('do').fill_null(0)).alias('ni')
        )
    # Variables in NA data nut Not Available in G_FUNDA- without Replacement
        .with_columns(pl.lit(None).alias('gp'))
        .with_columns(pl.lit(None).alias('pstkrv'))
        .with_columns(pl.lit(None).alias('pstkl'))
        .with_columns(pl.lit(None).alias('itcb'))
        .with_columns(pl.lit(None).alias('xad'))
        .with_columns(pl.lit(None).alias('txbcof'))
        .select(['gvkey', 'datadate', 'indfmt', 'curcd', 'source'] + avars + avars_other)
    )

    # Grouping and filtering for g_funda1


    __gfunda =  g_funda1.with_columns(pl.count().over(['gvkey', 'datadate']).alias('count_indfmt'))
    __gfunda = __gfunda.filter(
            (pl.col('count_indfmt') == 1) | ((pl.col('count_indfmt') == 2) & (pl.col('indfmt') == 'INDL'))
        ).drop(['indfmt', 'count_indfmt'])

    #quarterly:

    g_fundq = COMP_G_FUNDQ.clone()  

    g_fundq1 = (
        g_fundq.filter(compcond & (pl.col('datadate') >= start_date))
        .with_columns(pl.lit('GLOBAL').alias('source'))
    
    # Variables in NA data but Not Available in G_FUNDQ- with Replacement
        .with_columns(
            (pl.col('ibq') + pl.col('xiq').fill_null(0)).alias('niq'),
            (pl.col('ppentq') + pl.col('dpactq')).alias('ppegtq')
        )
    
    # Variables in NA data but Not Available in G_FUNDQ- without Replacement
        .with_columns(
            pl.lit(None).alias('icaptq'),
            pl.lit(None).alias('niy'),
            pl.lit(None).alias('txditcq'),
            pl.lit(None).alias('txpq'),
            pl.lit(None).alias('xidoq'),
            pl.lit(None).alias('xidoy'),
            pl.lit(None).alias('xrdq'),
            pl.lit(None).alias('xrdy'),
            pl.lit(None).alias('txbcofy'),
            pl.lit(None).alias('doq'),
            pl.lit(None).alias('doy'),
        )
    
        .select(['gvkey', 'datadate', 'indfmt', 'fyr', 'fyearq', 'fqtr', 'curcdq', 'source'] + qvars)
    )

    # Grouping and filtering for g_fundq1
    __gfundq =  g_fundq1.with_columns(pl.count().over(['gvkey', 'datadate']).alias('count_indfmt'))
    __gfundq = __gfundq.filter(
            (pl.col('count_indfmt') == 1) | ((pl.col('count_indfmt') == 2) & (pl.col('indfmt') == 'INDL'))
        ).drop(['indfmt', 'count_indfmt'])


    #preparing north american data(annual and quarterly):

    if coverage in ['na', 'world']:
        compcond = (
            (pl.col('indfmt') == 'INDL') & 
            (pl.col('datafmt') == 'STD') & 
            (pl.col('popsrc') == 'D') & 
            (pl.col('consol') == 'C') & 
            (pl.col('datadate') >= start_date)
        )


    funda = COMP_FUNDA.clone()
    

    __funda = (
            funda
            .filter(compcond)
            .with_columns(
                pl.lit('NA').alias('source'),
                pl.lit(None).cast(pl.Float64).alias('wcapt'),
                pl.lit(None).cast(pl.Float64).alias('ltdch'),
                pl.lit(None).cast(pl.Float64).alias('purtshr'))
            .select(['gvkey', 'datadate', 'curcd', 'source'] + avars + avars_other)
        )

    fundq = COMP_FUNDQ.clone()
    
    __fundq = (
            fundq
            .filter(compcond)
            .with_columns(
                pl.lit('NA').alias('source'),
                pl.lit(None).cast(pl.Float64).alias('dvtq'),
                pl.lit(None).cast(pl.Float64).alias('gpq'),
                pl.lit(None).cast(pl.Float64).alias('dvty'),
                pl.lit(None).cast(pl.Float64).alias('gpy'),
                pl.lit(None).cast(pl.Float64).alias('ltdchy'),
                pl.lit(None).cast(pl.Float64).alias('purtshry'),
                pl.lit(None).cast(pl.Float64).alias('wcapty'))
            .select(['gvkey', 'datadate', 'fyr', 'fyearq', 'fqtr', 'curcdq', 'source'] + qvars)
        )


    #changing data types of columns in global data to be equal to na, otherwise vertical staking using vstack does not work (check: if the datatypes have been
    #correctly set, especially for columns created in the NA data, because we use them for datatypes.
    for name, dtype in zip(__funda.columns, __funda.dtypes):
        __gfunda = __gfunda.with_columns(pl.col(name).cast(dtype))
    
    for name, dtype in zip(__fundq.columns, __fundq.dtypes):
        __gfundq = __gfundq.with_columns(pl.col(name).cast(dtype))



    if coverage == 'world':
        __wfunda = __gfunda.vstack(__funda)
        __wfundq = __gfundq.vstack(__fundq)



    
    if coverage == 'na':
        aname = __funda
        qname = __fundq
    elif coverage == 'global':
        aname = __gfunda
        qname = __gfundq
    else:
        aname = __wfunda
        qname = __wfundq


    #converting to usd if required
    if convert_to_usd == 1:
        # fx = compustat_fx(exrt_dly=COMP_EXRT_DLY)
        # #Join to get the fx rates
        __tempa = aname.join(fx, left_on=["datadate", "curcd"], right_on=["date", "curcdd"], how="left").select(aname.columns+ ["fx"])
        __tempq = qname.join(fx, left_on=["datadate", "curcdq"], right_on=["date", "curcdd"], how="left").select(qname.columns+ ["fx"])


    
        # Apply the fx rate conversion for avars
        for col_name in avars:
            __tempa = __tempa.with_columns((pl.col(col_name) * pl.col('fx')).alias(col_name))
        __tempa = __tempa.with_columns(pl.lit("USD").alias("curcd")).drop('fx')
        
        # Apply the fx rate conversion for qvars
        for col_name in qvars:
            __tempq = __tempq.with_columns((pl.col(col_name) * pl.col('fx')).alias(col_name))
        __tempq = __tempq.with_columns(pl.lit("USD").alias("curcdq")).drop('fx')
        
        
        # Rename final dataframes after conversion
        __compa1 = __tempa.clone()
        __compq1 = __tempq.clone()

    else:
        # Rename the dataframes
        __compa1 = aname.clone()
        __compq1 = qname.clone()


#quarterizing year-to-date data:

    __compq2 = quarterize(df=__compq1, var_list=qvars_y)
    __compq2 = __compq2.sort(['gvkey', 'fyr', 'fyearq', 'fqtr']).unique(['gvkey', 'fyr', 'fyearq', 'fqtr'], keep='first')


    #we quarterized some variables that were already available quarterized. Now just updating them if they have missing values
    __compq3 = __compq2
    for var_ytd in qvars_y:
        var = var_ytd[:-1]
        if (var + "q") in qvars_q:
            __compq3 = __compq3.with_columns(
            pl.col(var + "q").fill_null(pl.col(var_ytd + "_q"))
        ).drop(var_ytd + "_q")
        else:
            __compq3 = __compq3.rename({var_ytd + "_q": var + "q"})


    #creating some variables that need in quarterly forms
    __compq3 = __compq3.with_columns([
        pl.col("ibq").alias("ni_qtr"),
        pl.col("saleq").alias("sale_qtr"),
        (pl.coalesce(["oancfq", (pl.col('ibq') + pl.col('dpq') - pl.coalesce([pl.col('ibq'), 0]))])).alias("ocf_qtr")
    ])


    #replaing quarterly variables with ttm:
    
    yrl_vars = [
        "cogsq", "xsgaq", "xintq", "dpq", "txtq", "xrdq", "dvq", "spiq", "saleq", "revtq",
        "xoprq", "oibdpq", "oiadpq", "ibq", "niq", "xidoq", "nopiq", "miiq", "piq", "xiq",
        "xidocq", "capxq", "oancfq", "ibcq", "dpcq", "wcaptq",
        "prstkcq", "sstkq", "purtshrq",
        # "dsq",
        "dltrq", "ltdchq", "dlcchq",
        "fincfq", "fiaoq", "txbcofq", "dvtq",
        #we are missingk three variables here
        #adding here
        "gpq", "doq", "dltisq"
    ]



    for var_yrl in yrl_vars:
        var_yrl_name = var_yrl[:-1]
        __compq3 = __compq3.with_columns(
            pl.when((pl.col("fqtr") == 4) &
                   (pl.col("gvkey") != pl.col("gvkey").shift(3)) |
                   (pl.col("fyr") != pl.col("fyr").shift(3)) |
                   (pl.col("curcdq") != pl.col("curcdq").shift(3)) |
                   ((pl.col("fqtr") + pl.col("fqtr").shift(1) + pl.col("fqtr").shift(2) + pl.col("fqtr").shift(3)) != 10))
            .then(pl.col(var_yrl_name + 'y'))
            .when((pl.col("fqtr") != 4) &
                   (pl.col("gvkey") != pl.col("gvkey").shift(3)) |
                   (pl.col("fyr") != pl.col("fyr").shift(3)) |
                   (pl.col("curcdq") != pl.col("curcdq").shift(3)) |
                   ((pl.col("fqtr") + pl.col("fqtr").shift(1) + pl.col("fqtr").shift(2) + pl.col("fqtr").shift(3)) != 10))
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise((pl.col(var_yrl) + pl.col(var_yrl).shift(1) + pl.col(var_yrl).shift(2) + pl.col(var_yrl).shift(3)))
            .alias(var_yrl_name)
                ).drop([var_yrl, var_yrl_name + 'y'])

    #renaming bs variables by removing the q suffix
    bs_vars = [
        "seqq", "ceqq", "pstkq", "icaptq", "mibq", "gdwlq", "req",
        "atq", "actq", "invtq", "rectq", "ppegtq", "ppentq", "aoq", "acoq", 
        "intanq", "cheq", "ivaoq", "ivstq", "ltq", "lctq", "dlttq", 
        "dlcq", "txpq", "apq", "lcoq", "loq", "txditcq", "txdbq"
    ]
    
    bs_vars_updated = list(col[:-1] for col in bs_vars)

    __compq3 = __compq3.rename(dict(zip(bs_vars, bs_vars_updated)))
    __compq3 = __compq3.rename({"curcdq": "curcd"})
    __compq4 = __compq3.sort(["gvkey", "datadate", "fyr"]).unique(keep='first')

    #creating the newly created quarterly variables in the annual data:
    
    __compa2 = __compa1.with_columns(
                pl.lit(None).cast(pl.Float64).alias('ni_qtr'),
                pl.lit(None).cast(pl.Float64).alias('sale_qtr'),
                pl.lit(None).cast(pl.Float64).alias('ocf_qtr') 
        )

    
    #preparing market equity data:
    
    __me_data = me_data.filter(
            (pl.col("gvkey").is_not_null()) & 
            (pl.col("primary_sec") == 1) & 
            (pl.col("me_company").is_not_null()) & 
            (pl.col("common") == 1) & 
            (pl.col("obs_main") == 1)
        ).select(
        ['gvkey', 'eom', pl.col('me_company').alias('me_fiscal')]).group_by(
        ["gvkey", "eom"]).agg(pl.col("me_fiscal").max())


    #adding market equity data to accounting data:
    __compa3 = __compa2.join(__me_data, left_on=["gvkey", "datadate"], right_on=["gvkey", "eom"], how="left").select(__compa2.columns+ ["me_fiscal"])
    __compq5 = __compq4.join(__me_data, left_on=["gvkey", "datadate"], right_on=["gvkey", "eom"], how="left").select(__compq4.columns+ ["me_fiscal"])


    ##standardizing annual and quarterly data
    __compa3 = __compa3.with_columns([
        pl.lit(None).alias('fqtr'),
        pl.lit(None).alias('fyearq'),
        pl.lit(None).alias('fyr')
    ])

    __compq5 = __compq5.with_columns([
        pl.lit(None).alias('dvc'),
        pl.lit(None).alias('ebit'),
        pl.lit(None).alias('ebitda'),
        pl.lit(None).alias('itcb'),
        pl.lit(None).alias('pstkl'),
        pl.lit(None).alias('pstkrv'),
        pl.lit(None).alias('xad'),
        pl.lit(None).alias('xlr'),
        pl.lit(None).alias('emp')
    ])


    #adding helper variables if required:
    if include_helpers_vars==1:
        __compq6 = add_helper_vars(data=__compq5)
        __compa4 = add_helper_vars(data=__compa3)
    else:
        __compq6 = __compq5.clone()
        __compa4 = __compa3.clone()



    acc_std_ann = __compa4.sort(['gvkey', 'datadate']).unique(['gvkey', 'datadate'])
    acc_std_qtr = __compq6.sort(['gvkey', 'datadate']).unique(['gvkey', 'datadate'])


    return acc_std_ann, acc_std_qtr

acc_std_ann, acc_std_qtr = standardized_accounting_data(coverage='world', convert_to_usd=1, me_data = world_msf, include_helpers_vars=1, start_date=date(2020,1,1))

def quarterize(df, var_list):
    quarterized_df = df.clone()
    for var in var_list:
        quarterized_df = quarterized_df.with_columns(
            pl.when(pl.col("fqtr") == 1)
                .then(pl.col(var))
                .when((pl.col("fqtr") != 1) & 
                    (pl.col("gvkey") == pl.col("gvkey").shift(1)) & 
                    ((pl.col("fqtr") - pl.col("fqtr").shift(1)) == 1))
                .then((pl.col(var) - pl.col(var).shift(1)))
                .otherwise(pl.lit(None))
                .alias(var + "_q")
            )
    return quarterized_df


def expand(data, id_vars, start_date, end_date, freq='day', new_date_name='date'):
    if freq =='day':
        __expanded = data.with_columns(pl.date_ranges(start=start_date, end=end_date, interval='1d')).rename({"date_range": new_date_name}).explode(new_date_name).drop([start_date, end_date])
    elif freq == 'month':
         __expanded = data.with_columns(pl.date_ranges(start=start_date, end=end_date, interval='1mo_saturating')).rename({"date_range": new_date_name}).explode(new_date_name).with_columns(pl.col(new_date_name).dt.month_end()).drop([start_date, end_date])


    __expanded = __expanded.sort(id_vars + [new_date_name]).unique(id_vars + [new_date_name])
    return __expanded


def add_helper_vars(data):
    __comp_dates1 = data.select(['gvkey', 'curcd', 'datadate']).group_by(
    ["gvkey", "curcd"]).agg(
    pl.col("datadate").min().alias('start_date'),
    pl.col("datadate").max().alias('end_date'))

    __comp_dates2 = expand(data=__comp_dates1, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='datadate')

    
    temp_data = data.with_columns(pl.lit(1).cast(pl.Float64).alias('data_available'))
    __helpers1 = __comp_dates2.join(temp_data, left_on=["gvkey", "curcd", "datadate"], right_on=["gvkey", "curcd", "datadate"], how="left").with_columns(pl.col("data_available").fill_null(strategy="zero")).select(temp_data.columns)
    __helpers1 = __helpers1.sort(["gvkey", "curcd", "datadate"]).unique(["gvkey", "curcd", "datadate"]) 


    __helpers2 = __helpers1.with_columns(pl.col('curcd').cumcount().over(['gvkey', 'curcd']).alias('count')).with_columns((pl.col('count') + pl.lit(1)).alias("count"))


    var_pos= ['at', 'sale', 'revt', 'dv', 'che']
    for var in var_pos:
        output = __helpers2.with_columns(pl.when(pl.col(var)<0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(var)).alias(var))

    data_with_helper_variables = output.clone()


    data_with_helper_variables = (data_with_helper_variables
    .with_columns(pl.coalesce([pl.col("sale"), pl.col("revt")]).alias("sale_x"))
    .with_columns(pl.coalesce([pl.col("gp"), pl.col("sale_x") - pl.col("cogs")]).alias("gp_x"))
    .with_columns(pl.coalesce([pl.col("xopr"), pl.col("cogs") + pl.col("xsga")]).alias("opex_x"))
    .with_columns(pl.coalesce([pl.col("ebitda"), pl.col("oibdp"), pl.col("sale_x") - pl.col("opex_x"), pl.col("gp_x") - pl.col("xsga")]).alias("ebitda_x"))
    .with_columns(pl.coalesce([pl.col("ebit"), pl.col("oiadp"), pl.col("ebitda_x") - pl.col("dp")]).alias("ebit_x"))
    .with_columns((pl.col("ebitda_x") + pl.coalesce([pl.col("xrd"), pl.lit(0)])).alias("op_x"))
    .with_columns((pl.col("ebitda_x") - pl.col("xint")).alias("ope_x"))
    .with_columns(pl.coalesce([pl.col("pi"), (pl.col("ebit_x") - pl.col("xint") + pl.coalesce([pl.col("spi"), pl.lit(0)]) + pl.coalesce([pl.col("nopi"), pl.lit(0)]))]).alias("pi_x"))
    .with_columns(pl.coalesce([pl.col("xido"), (pl.col("xi") + pl.coalesce([pl.col("do"), pl.lit(0)]))]).alias("xido_x"))
    .with_columns(pl.coalesce([pl.col("ib"), (pl.col("ni") - pl.col("xido_x")), (pl.col("pi_x") - pl.col("txt") - pl.coalesce([pl.col("mii"), pl.lit(0)]))]).alias("ni_x"))
    .with_columns(pl.coalesce([pl.col("ni"), (pl.col("ni_x") + pl.coalesce([pl.col("xido_x"), pl.lit(0)])), (pl.col("ni_x") + pl.col("xi") + pl.col("do"))]).alias("nix_x"))
    .with_columns((pl.col("nix_x") + pl.col("xint")).alias("fi_x"))
    .with_columns(pl.coalesce([pl.col("dvt"), pl.col("dv")]).alias("div_x"))
)


    data_with_helper_variables = (data_with_helper_variables
    .with_columns((pl.col("prstkc") + pl.col("purtshr")).alias("eqbb_x"))
    .with_columns(pl.col("sstk").alias("eqis_x"))
    .with_columns((pl.col("eqis_x") - pl.col("eqbb_x")).alias("eqnetis_x"))
    .with_columns((pl.col("div_x") + pl.col("eqbb_x")).alias("eqpo_x"))
    .with_columns((pl.col("div_x") - pl.col("eqnetis_x")).alias("eqnpo_x"))
    .with_columns(pl.when((pl.col("dltis").is_null()) & (pl.col("dltr").is_null()) & (pl.col("ltdch").is_null()) & (pl.col("count") <= 12)).then(pl.lit(None))
                  .otherwise(pl.coalesce([(pl.col("dltis") - pl.col("dltr")), pl.col("ltdch"), (pl.col("dltt") - pl.col("dltt").shift(12))])).alias("dltnetis_x"))
    .with_columns(pl.when(pl.col("dlcch").is_null() & (pl.col("count") <= 12)).then(pl.lit(None))
                  .otherwise(pl.coalesce([pl.col("dlcch"), (pl.col("dlc") - pl.col("dlc").shift(12))])).alias("dstnetis_x"))
    .with_columns((pl.col("dstnetis_x") + pl.col("dltnetis_x")).alias("dbnetis_x"))
    .with_columns((pl.col("eqnetis_x") + pl.col("dbnetis_x")).alias("netis_x"))
    .with_columns(pl.coalesce([pl.col("fincf"), (pl.col("netis_x") - pl.col("dv") + pl.coalesce([pl.col("fiao"), pl.lit(0)]) + pl.coalesce([pl.col("txbcof"), pl.lit(0)]))]).alias("fincf_x"))
)


    data_with_helper_variables = (data_with_helper_variables
    .with_columns((pl.col("dltt") + pl.col("dlc")).alias("debt_x"))
    .with_columns(pl.coalesce([pl.col("pstkrv"), pl.col("pstkl"), pl.col("pstk")]).alias("pstk_x"))
    .with_columns(pl.coalesce([pl.col("seq"), pl.col("ceq") + pl.col("pstk_x"), pl.col("at") - pl.col("lt")]).alias("seq_x"))
    .with_columns(pl.coalesce([pl.col("at"), pl.col("seq_x") + pl.col("dltt") + pl.coalesce(pl.col("lct"), pl.lit(0)) + pl.coalesce(pl.col("lo"), pl.lit(0)) + pl.coalesce(pl.col("txditc"), pl.lit(0))]).alias("at_x"))
    .with_columns(pl.coalesce([pl.col("act"), pl.col("rect") + pl.col("invt") + pl.col("che") + pl.col("aco")]).alias("ca_x"))
    .with_columns(pl.coalesce([pl.col("lct"), pl.col("ap") + pl.col("dlc") + pl.col("txp") + pl.col("lco")]).alias("cl_x"))
    .with_columns((pl.col("at_x") - pl.col("ca_x")).alias("nca_x"))
    .with_columns((pl.col("lt") - pl.col("cl_x")).alias("ncl_x"))
    .with_columns((pl.col("debt_x") - pl.coalesce(pl.col("che"), pl.lit(0))).alias("netdebt_x"))
    .with_columns(pl.coalesce([pl.col("txditc"), pl.col("txdb") + pl.col("itcb")]).alias("txditc_x"))
    .with_columns((pl.col("seq_x") + pl.col("txditc_x") - pl.col("pstk_x")).alias("be_x"))
    .with_columns(pl.coalesce([pl.col("icapt") + pl.coalesce(pl.col("dlc"), pl.lit(0)) - pl.coalesce(pl.col("che"), pl.lit(0)), pl.col("netdebt_x") + pl.col("seq_x") + pl.coalesce(pl.col("mib"), pl.lit(0))]).alias("bev_x"))
    .with_columns((pl.col("ca_x") - pl.col("che")).alias("coa_x"))
    .with_columns((pl.col("cl_x") - pl.coalesce(pl.col("dlc"), pl.lit(0))).alias("col_x"))
    .with_columns((pl.col("coa_x") - pl.col("col_x")).alias("cowc_x"))
    .with_columns((pl.col("at_x") - pl.col("ca_x") - pl.coalesce(pl.col("ivao"), pl.lit(0))).alias("ncoa_x"))
    .with_columns((pl.col("lt") - pl.col("cl_x") - pl.col("dltt")).alias("ncol_x"))
    .with_columns((pl.col("ncoa_x") - pl.col("ncol_x")).alias("nncoa_x"))
    .with_columns((pl.coalesce(pl.col("ivst"), pl.lit(0)) + pl.coalesce(pl.col("ivao"), pl.lit(0))).alias("fna_x"))
    .with_columns((pl.col("debt_x") + pl.coalesce(pl.col("pstk_x"), pl.lit(0))).alias("fnl_x"))
    .with_columns((pl.col("fna_x") - pl.col("fnl_x")).alias("nfna_x"))
    .with_columns((pl.col("coa_x") + pl.col("ncoa_x")).alias("oa_x"))
    .with_columns((pl.col("col_x") + pl.col("ncol_x")).alias("ol_x"))
    .with_columns((pl.col("oa_x") - pl.col("ol_x")).alias("noa_x"))
    .with_columns((pl.col("ppent") + pl.col("intan") + pl.col("ao") - pl.col("lo") + pl.col("dp")).alias("lnoa_x"))
    .with_columns(pl.coalesce([pl.col("ca_x") - pl.col("invt"), pl.col("che") + pl.col("rect")]).alias("caliq_x"))
    .with_columns((pl.col("ca_x") - pl.col("cl_x")).alias("nwc_x"))
    .with_columns((pl.col("ppegt") + pl.col("invt")).alias("ppeinv_x"))
    .with_columns((pl.col("che") + 0.75 * pl.col("coa_x") + 0.5 * (pl.col("at_x") - pl.col("ca_x") - pl.coalesce(pl.col("intan"), 0))).alias("aliq_x")))

    var_bs= ['be_x', 'bev_x']
    for var in var_bs:
        data_with_helper_variables = data_with_helper_variables.with_columns(pl.when(pl.col(var)<0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(var)).alias(var))

    data_with_helper_variables = (data_with_helper_variables
    .with_columns(pl.when(pl.col("count") <= 12).then(None).otherwise(pl.coalesce([pl.col("ni_x") - pl.col("oancf"), pl.col("cowc_x") - pl.col("cowc_x").shift(12) + pl.col("nncoa_x") - pl.col("nncoa_x").shift(12)])).alias("oacc_x"))
    .with_columns(pl.when(pl.col("count") <= 12).then(None).otherwise(pl.col("oacc_x") + pl.col("nfna_x") - pl.col("nfna_x").shift(12)).alias("tacc_x"))
    .with_columns(pl.coalesce([pl.col("oancf"), pl.col("ni_x") - pl.col("oacc_x"), pl.col("ni_x") + pl.col("dp") - pl.coalesce([pl.col("wcapt"), pl.lit(0)])]).alias("ocf_x"))
    .with_columns((pl.col("ocf_x") - pl.col("capx")).alias("fcf_x"))
    .with_columns((pl.col("ebitda_x") + pl.coalesce([pl.col("xrd"), pl.lit(0)]) - pl.col("oacc_x")).alias("cop_x"))
    )

    data_with_helper_variables = data_with_helper_variables.drop('count')

    return data_with_helper_variables

# Complete list of accounting characteristics
acc_chars = [
    # Accounting Based Size Measures
    "assets", "sales", "book_equity", "net_income", "enterprise_value",
    
    # 1yr Growth
    "at_gr1", "ca_gr1", "nca_gr1", "lt_gr1", "cl_gr1", "ncl_gr1", "be_gr1", "pstk_gr1", "debt_gr1",
    "sale_gr1", "cogs_gr1", "sga_gr1", "opex_gr1",
    
    # 3yr Growth
    "at_gr3", "ca_gr3", "nca_gr3", "lt_gr3", "cl_gr3", "ncl_gr3", "be_gr3", "pstk_gr3", "debt_gr3",
    "sale_gr3", "cogs_gr3", "sga_gr3", "opex_gr3",
    
    # 1yr Growth Scaled by Assets
    "cash_gr1a", "inv_gr1a", "rec_gr1a", "ppeg_gr1a", "lti_gr1a", "intan_gr1a", "debtst_gr1a", "ap_gr1a",
    "txp_gr1a", "debtlt_gr1a", "txditc_gr1a", "coa_gr1a", "col_gr1a", "cowc_gr1a", "ncoa_gr1a", "ncol_gr1a", "nncoa_gr1a",
    "oa_gr1a", "ol_gr1a", "noa_gr1a", "fna_gr1a", "fnl_gr1a", "nfna_gr1a", "gp_gr1a", "ebitda_gr1a", "ebit_gr1a",
    "ope_gr1a", "ni_gr1a", "nix_gr1a", "dp_gr1a", "ocf_gr1a", "fcf_gr1a", "nwc_gr1a",
    "eqnetis_gr1a", "dltnetis_gr1a", "dstnetis_gr1a", "dbnetis_gr1a", "netis_gr1a", "fincf_gr1a", "eqnpo_gr1a",
    "tax_gr1a", "div_gr1a", "eqbb_gr1a", "eqis_gr1a", "eqpo_gr1a", "capx_gr1a",
    
    # 3yr Growth Scaled by Assets
    "cash_gr3a", "inv_gr3a", "rec_gr3a", "ppeg_gr3a", "lti_gr3a", "intan_gr3a", "debtst_gr3a", "ap_gr3a",
    "txp_gr3a", "debtlt_gr3a", "txditc_gr3a", "coa_gr3a", "col_gr3a", "cowc_gr3a", "ncoa_gr3a", "ncol_gr3a", "nncoa_gr3a",
    "oa_gr3a", "ol_gr3a", "noa_gr3a", "fna_gr3a", "fnl_gr3a", "nfna_gr3a", "gp_gr3a", "ebitda_gr3a", "ebit_gr3a",
    "ope_gr3a", "ni_gr3a", "nix_gr3a", "dp_gr3a", "ocf_gr3a", "fcf_gr3a", "nwc_gr3a",
    "eqnetis_gr3a", "dltnetis_gr3a", "dstnetis_gr3a", "dbnetis_gr3a", "netis_gr3a", "fincf_gr3a", "eqnpo_gr3a",
    "tax_gr3a", "div_gr3a", "eqbb_gr3a", "eqis_gr3a", "eqpo_gr3a", "capx_gr3a",
    
    # Investment
    "capx_at", "rd_at",
    
    # Profitability
    "gp_sale", "ebitda_sale", "ebit_sale", "pi_sale", "ni_sale", "nix_sale", "ocf_sale", "fcf_sale",
    "gp_at", "ebitda_at", "ebit_at", "fi_at", "cop_at",
    "ope_be", "ni_be", "nix_be", "ocf_be", "fcf_be",
    "gp_bev", "ebitda_bev", "ebit_bev", "fi_bev", "cop_bev",
    "gp_ppen", "ebitda_ppen", "fcf_ppen",
    
    # Issuance
    "fincf_at", "netis_at", "eqnetis_at", "eqis_at", "dbnetis_at", "dltnetis_at", "dstnetis_at",
    
    # Equity Payout
    "eqnpo_at", "eqbb_at", "div_at",
    
    # Accruals
    "oaccruals_at", "oaccruals_ni", "taccruals_at", "taccruals_ni", "noa_at",
    
    # Capitalization/Leverage Ratios
    "be_bev", "debt_bev", "cash_bev", "pstk_bev", "debtlt_bev", "debtst_bev",
    "debt_mev", "pstk_mev", "debtlt_mev", "debtst_mev",
    
    # Financial Soundness Ratios
    "int_debtlt", "int_debt", "cash_lt", "inv_act", "rec_act",
    "ebitda_debt", "debtst_debt", "cl_lt", "debtlt_debt", "profit_cl", "ocf_cl",
    "ocf_debt", "lt_ppen", "debtlt_be", "fcf_ocf",
    "opex_at", "nwc_at",
    
    # Solvency Ratios
    "debt_at", "debt_be", "ebit_int",
    
    # Liquidity Ratios
    "cash_cl", "caliq_cl", "ca_cl",
    "inv_days", "rec_days", "ap_days", "cash_conversion",
    
    # Activity/Efficiency Ratio
    "inv_turnover", "at_turnover", "rec_turnover", "ap_turnover",
    
    # Non-Recurring Items
    "spi_at", "xido_at", "nri_at",
    
    # Miscellaneous
    "adv_sale", "staff_sale", "rd_sale", "div_ni", "sale_bev", "sale_be", "sale_nwc", "tax_pi",
    
    # Balance Sheet Fundamentals to Market Equity
    "be_me", "at_me", "cash_me",
    
    # Income Fundamentals to Market Equity
    "gp_me", "ebitda_me", "ebit_me", "ope_me", "ni_me", "nix_me", "sale_me", "ocf_me", "fcf_me", "cop_me",
    "rd_me",
    
    # Equity Payout/issuance to Market Equity
    "div_me", "eqbb_me", "eqis_me", "eqpo_me", "eqnpo_me", "eqnetis_me",
    
    # Debt Issuance to Market Enterprise Value
    "dltnetis_mev", "dstnetis_mev", "dbnetis_mev",
    
    # Firm Payout/issuance to Market Enterprise Value
    "netis_mev",
    
    # Balance Sheet Fundamentals to Market Enterprise Value
    "at_mev", "be_mev", "bev_mev", "ppen_mev", "cash_mev",
    
    # Income/CF Fundamentals to Market Enterprise Value
    "gp_mev", "ebitda_mev", "ebit_mev", "cop_mev", "sale_mev", "ocf_mev", "fcf_mev", "fincf_mev",
    
    # New Variables from HXZ
    "ni_inc8q", "ppeinv_gr1a", "lnoa_gr1a", "capx_gr1", "capx_gr2", "capx_gr3", "sti_gr1a",
    "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1", "saleq_gr1", "rd5_at",
    "dsale_dinv", "dsale_drec", "dgp_dsale", "dsale_dsga",
    "saleq_su", "niq_su", "debt_me", "netdebt_me", "capex_abn", "inv_gr1", "be_gr1a",
    "op_at", "pi_nix", "op_atl1", "gp_atl1", "ope_bel1", "cop_atl1",
    "at_be", "ocfq_saleq_std",
    "aliq_at", "aliq_mat", "tangibility",
    "eq_dur", "f_score", "o_score", "z_score", "kz_index", "intrinsic_value", "ival_me",
    "sale_emp_gr1", "emp_gr1", "cash_at"
    "earnings_variability", "ni_ar1", "ni_ivol"

    #New Variables not in HXZ
    "niq_saleq_std", "ni_emp", "sale_emp", "ni_at",
    "ocf_at", "ocf_at_chg1", "roeq_be_std", "roe_be_std",
    "gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"

]


def create_acc_chars(data, lag_to_public, max_data_lag, __keep_vars, me_data, suffix):
    
    #sorting the data
    __chars3 = data.sort(['gvkey', 'curcd', 'datadate'])

    #adding a count column that keeps a count of the number of the obs for a given gvkey (and curcd)
    __chars4 = __chars3.with_columns(pl.col('curcd').cumcount().over(['gvkey', 'curcd']).alias('count')).with_columns((pl.col('count') + pl.lit(1)).alias("count"))

    #accounting based size measures
    __chars5 = (__chars4
        .with_columns(pl.col("at_x").alias("assets"))
       .with_columns(pl.col("sale_x").alias("sales"))
       .with_columns(pl.col("be_x").alias("book_equity"))
       .with_columns(pl.col("ni_x").alias("net_income")))
    
    
    #growth characteristics
    growth_vars = [
    "at_x", "ca_x", "nca_x",                 # Assets - Aggregated
    "lt", "cl_x", "ncl_x",                   # Liabilities - Aggregated
    "be_x", "pstk_x", "debt_x",              # Financing Book Values
    "sale_x", "cogs", "xsga", "opex_x",      # Sales and Operating Costs
    "capx", "invt"
    ]

    #1-yr growth
    for i in growth_vars:
        __chars5 = var_growth(df=__chars5, var_gr=i, horizon=12)
    
    #3-yr growth
    for i in growth_vars:
        __chars5 = var_growth(df=__chars5, var_gr=i, horizon=36)


    #Change Scaled by Asset Characteristics 
    
    ch_asset_vars = [
        # Assets - Individual Items
        "che", "invt", "rect", "ppegt", "ivao", "ivst", "intan",
        
        # Liabilities - Individual Items
        "dlc", "ap", "txp", "dltt", "txditc",
        
        # Operating Assets/Liabilities
        "coa_x", "col_x", "cowc_x", "ncoa_x", "ncol_x", "nncoa_x", "oa_x", "ol_x",
        
        # Financial Assets/Liabilities
        "fna_x", "fnl_x", "nfna_x",
        
        # Income Statement
        "gp_x", "ebitda_x", "ebit_x", "ope_x", "ni_x", "nix_x", "dp",
        
        # Aggregated Cash Flow
        "fincf_x", "ocf_x", "fcf_x", "nwc_x",
        
        # Financing Cash Flow
        "eqnetis_x", "dltnetis_x", "dstnetis_x", "dbnetis_x", "netis_x", "eqnpo_x",
        
        # Tax Change
        "txt",
        
        # Financing Cash Flow
        "eqbb_x", "eqis_x", "div_x", "eqpo_x",
        
        # Other
        "capx", "be_x"
    ]
    
    
    
    #1yr Change Scaled by Assets
    for i in ch_asset_vars:
        __chars5 = chg_to_assets(df=__chars5, var_gr=i, horizon=12)
    
    
    #3yr Change Scaled by Assets
    for i in ch_asset_vars:
        __chars5 = chg_to_assets(df=__chars5, var_gr=i, horizon=36)

    
    #Investment Measure
    __chars5 = (__chars5
            .with_columns((pl.col("capx")/pl.col('at_x')).alias("capx_at"))
            .with_columns((pl.col("xrd")/pl.col('at_x')).alias("rd_at")))



    #Non-Recurring Items
    __chars5 = (__chars5
            .with_columns((pl.col("spi")/pl.col('at_x')).alias("spi_at"))
            .with_columns((pl.col("xido_x")/pl.col('at_x')).alias("xido_at"))
            .with_columns(((pl.col("spi") + pl.col("xido_x"))/pl.col('at_x')).alias("nri_at")) )


    #profitability margins
    __chars5 = (__chars5
            .with_columns((pl.col("gp_x")/pl.col('sale_x')).alias("gp_sale"))                             #Gross Profit Margin
            .with_columns((pl.col("ebitda_x")/pl.col('sale_x')).alias("ebitda_sale"))                     #Operating Profit Margin before Depreciation
            .with_columns((pl.col("ebit_x")/pl.col('sale_x')).alias("ebit_sale"))                         #Operating profit Margin after Depreciation
            .with_columns((pl.col("pi_x")/pl.col('sale_x')).alias("pi_sale"))                             #Pretax Profit Margin
            .with_columns((pl.col("ni_x")/pl.col('sale_x')).alias("ni_sale"))                             #Net Profit Margin Before XI
            .with_columns((pl.col("ni")/pl.col('sale_x')).alias("nix_sale"))                              #Net Profit Margin
            .with_columns((pl.col("ocf_x")/pl.col('sale_x')).alias("ocf_sale"))                           #Operating Cash Flow Margin
            .with_columns((pl.col("fcf_x")/pl.col('sale_x')).alias("fcf_sale")))                          #Free Cash Flow Margin



    #Return on assets:
    __chars5 = (__chars5
            .with_columns((pl.col("gp_x")/pl.col('at_x')).alias("gp_at"))
            .with_columns((pl.col("ebitda_x")/pl.col('at_x')).alias("ebitda_at"))
            .with_columns((pl.col("ebit_x")/pl.col('at_x')).alias("ebit_at"))
            .with_columns((pl.col("fi_x")/pl.col('at_x')).alias("fi_at"))
            .with_columns((pl.col("cop_x")/pl.col('at_x')).alias("cop_at"))
            .with_columns((pl.col("ni_x")/pl.col('at_x')).alias("ni_at")))



    #Return on book equity:
    __chars5 = (__chars5
            .with_columns((pl.col("ope_x")/pl.col('be_x')).alias("ope_be"))
            .with_columns((pl.col("ni_x")/pl.col('be_x')).alias("ni_be"))
            .with_columns((pl.col("nix_x")/pl.col('be_x')).alias("nix_be"))
            .with_columns((pl.col("ocf_x")/pl.col('be_x')).alias("ocf_be"))
            .with_columns((pl.col("fcf_x")/pl.col('be_x')).alias("fcf_be")))


    #Return on invested book capital:
    __chars5 = (__chars5
            .with_columns((pl.col("gp_x")/pl.col('bev_x')).alias("gp_bev"))
            .with_columns((pl.col("ebitda_x")/pl.col('bev_x')).alias("ebitda_bev"))
            .with_columns((pl.col("ebit_x")/pl.col('bev_x')).alias("ebit_bev"))                            #Pre tax Return on Book Enterprise Value
            .with_columns((pl.col("fi_x")/pl.col('bev_x')).alias("fi_bev"))                                #ROIC
            .with_columns((pl.col("cop_x")/pl.col('bev_x')).alias("cop_bev")))                             #Cash Based Operating Profit to Invested Capital


    #Return on Physical Capital:
    __chars5 = (__chars5
            .with_columns((pl.col("gp_x")/pl.col('ppent')).alias("gp_ppen"))
            .with_columns((pl.col("ebitda_x")/pl.col('ppent')).alias("ebitda_ppen"))
            .with_columns((pl.col("fcf_x")/pl.col('ppent')).alias("fcf_ppen")))




    #Issuance Variables:
    __chars5 = (__chars5
            .with_columns((pl.col("fincf_x")/pl.col('at_x')).alias("fincf_at"))
            .with_columns((pl.col("netis_x")/pl.col('at_x')).alias("netis_at"))
            .with_columns((pl.col("eqnetis_x")/pl.col('at_x')).alias("eqnetis_at"))
            .with_columns((pl.col("eqis_x")/pl.col('at_x')).alias("eqis_at"))
            .with_columns((pl.col("dbnetis_x")/pl.col('at_x')).alias("dbnetis_at"))
            .with_columns((pl.col("dltnetis_x")/pl.col('at_x')).alias("dltnetis_at"))
            .with_columns((pl.col("dstnetis_x")/pl.col('at_x')).alias("dstnetis_at")))


    #Equity Payout:
    __chars5 = (__chars5
            .with_columns((pl.col("eqnpo_x")/pl.col('at_x')).alias("eqnpo_at"))
            .with_columns((pl.col("eqbb_x")/pl.col('at_x')).alias("eqbb_at"))
            .with_columns((pl.col("div_x")/pl.col('at_x')).alias("div_at")))


    #accruals:
    __chars5 = (__chars5
            .with_columns((pl.col("oacc_x")/pl.col('at_x')).alias("oaccruals_at"))                                  # Operating Accruals
            .with_columns((pl.col("oacc_x")/pl.col('nix_x').abs()).alias("oaccruals_ni"))                           #Percent Operating Accruals
            .with_columns((pl.col("tacc_x")/pl.col('at_x')).alias("taccruals_at"))                                  #Total Accruals
            .with_columns((pl.col("tacc_x")/pl.col('nix_x').abs()).alias("taccruals_ni"))                           #Percent Total Accruals
            .with_columns((pl.col("noa_x")/pl.col('at_x').shift(12)).alias("noa_at"))                               #Net Operating Asset to Total Assets
            .with_columns(
                pl.when((pl.col("count") <= 12) | (pl.col("at_x").shift(12) <= 0))
                .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col("noa_at")).alias("noa_at"))
    )



    #Capitalization/Leverage Ratios Book:
    __chars5 = (__chars5
            .with_columns((pl.col("be_x")/pl.col('bev_x')).alias("be_bev"))                                         # Common Equity as % of Book Enterprise Value
            .with_columns((pl.col("debt_x")/pl.col('bev_x')).alias("debt_bev"))                                     #Total Debt as % of Book Enterprise Value
            .with_columns((pl.col("che")/pl.col('bev_x')).alias("cash_bev"))                                        #Cash and Short-Term Investments to Book Enterprise Value 
            .with_columns((pl.col("pstk_x")/pl.col('bev_x')).alias("pstk_bev"))                                     #Prefered Stock to Book Enterprise Value 
            .with_columns((pl.col("dltt")/pl.col('bev_x')).alias("debtlt_bev"))                                     #Long-term debt as % of Book Enterprise Value
            .with_columns((pl.col("dlc")/pl.col('bev_x')).alias("debtst_bev")))                                     #Short-term debt as % of Book Enterprise Value



    #Financial Soundness Ratios:
    __chars5 = (__chars5
            .with_columns((pl.col("xint")/pl.col('debt_x')).alias("int_debt"))                                      #Interest as % of average total debt
            .with_columns((pl.col("xint")/pl.col('dltt')).alias("int_debtlt"))                                      #Interest as % of average long-term debt
            .with_columns((pl.col("ebitda_x")/pl.col('debt_x')).alias("ebitda_debt"))                               #Ebitda to total debt
            .with_columns((pl.col("ebitda_x")/pl.col('cl_x')).alias("profit_cl"))                                   #Profit before D&A to current liabilities
            .with_columns((pl.col("ocf_x")/pl.col('cl_x')).alias("ocf_cl"))                                         #Operating cash flow to current liabilities
            .with_columns((pl.col("ocf_x")/pl.col('debt_x')).alias("ocf_debt"))                                     #Operating cash flow to total debt
            .with_columns((pl.col("che")/pl.col('lt')).alias("cash_lt"))                                            #Cash balance to Total Liabilities
            .with_columns((pl.col("invt")/pl.col('act')).alias("inv_act"))                                          #inventory as % of current assets
            .with_columns((pl.col("rect")/pl.col('act')).alias("rec_act"))                                          #receivables as % of current assets
            .with_columns((pl.col("dlc")/pl.col('debt_x')).alias("debtst_debt"))                                    #short term term as % of total debt
            .with_columns((pl.col("cl_x")/pl.col('lt')).alias("cl_lt"))                                             #current liabilities as % of total liabilities
            .with_columns((pl.col("dltt")/pl.col('debt_x')).alias("debtlt_debt"))                                   #long-term debt as % of total liabilities
            .with_columns((pl.col("lt")/pl.col('ppent')).alias("lt_ppen"))                                          #total liabilities to total tangible assets
            .with_columns((pl.col("dltt")/pl.col('be_x')).alias("debtlt_be"))                                       #long-term debt to book equity
            .with_columns((pl.col("opex_x")/pl.col('at_x')).alias("opex_at"))                                       #Operating Leverage ala Novy-Marx (2011)
            .with_columns((pl.col("nwc_x")/pl.col('at_x')).alias("nwc_at"))
            .with_columns((pl.col("fcf_x")/pl.col('ocf_x')).alias("fcf_ocf"))                                       #Free Cash Flow/Operating Cash Flow
            .with_columns(
                pl.when(pl.col("ocf_x") <= 0)
                .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col("fcf_ocf")).alias("fcf_ocf"))
    )


    #Solvency Ratios:
    __chars5 = (__chars5
            .with_columns((pl.col("debt_x")/pl.col('at_x')).alias("debt_at"))                                       #Debt-to-assets
            .with_columns((pl.col("debt_x")/pl.col('be_x')).alias("debt_be"))                                       #debt to shareholders' equity ratio
            .with_columns((pl.col("ebit_x")/pl.col('xint')).alias("ebit_int")))                                     #interest coverage ratio




    #Liquidity Ratios:
    __chars5 = (__chars5
            #Days Inventory Outstanding 
            .with_columns(((pl.concat_list(['invt', pl.col('invt').shift(12)]).list.mean()/pl.col('cogs'))*365).alias('inv_days')).with_columns(pl.when(pl.col('count') < 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('inv_days')).alias('inv_days'))
            #Days Sales Outstanding
            .with_columns(((pl.concat_list(['rect', pl.col('rect').shift(12)]).list.mean()/pl.col('sale_x'))*365).alias('rec_days')).with_columns(pl.when(pl.col('count') < 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('rec_days')).alias('rec_days'))
            #Days Accounts Payable Outstanding
            .with_columns(((pl.concat_list(['ap', pl.col('ap').shift(12)]).list.mean()/pl.col('cogs'))*365).alias('ap_days')).with_columns(pl.when(pl.col('count') < 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('ap_days')).alias('ap_days'))
            #Cash Conversion Cycle
            .with_columns((pl.col('inv_days') + pl.col('rec_days') + pl.col('ap_days') ).alias('cash_conversion')).with_columns(pl.when(pl.col('cash_conversion') < 0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('cash_conversion')).alias('cash_conversion'))
            #Cash Ratio
            .with_columns((pl.col('che')/ pl.col('cl_x')).alias('cash_cl')).with_columns(pl.when(pl.col('cl_x') <= 0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('cash_cl')).alias('cash_cl'))
            #Quick Ratio (acid test)
            .with_columns((pl.col('caliq_x')/ pl.col('cl_x')).alias('caliq_cl')).with_columns(pl.when(pl.col('cl_x') <= 0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('caliq_cl')).alias('caliq_cl'))
            #Current Ratio
            .with_columns((pl.col('ca_x')/ pl.col('cl_x')).alias('ca_cl')).with_columns(pl.when(pl.col('cl_x') <= 0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('ca_cl')).alias('ca_cl'))
    )




    #Activity/Efficiency Ratios:
    __chars5 = (__chars5
            #Inventory Turnover
            .with_columns((pl.col('cogs')/pl.concat_list(['invt', pl.col('invt').shift(12)]).list.mean()).alias('inv_turnover')).with_columns(pl.when(pl.col('count') <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('inv_turnover')).alias('inv_turnover'))
            #Asset Turnover
            .with_columns((pl.col('sale_x')/pl.concat_list(['at_x', pl.col('at_x').shift(12)]).list.mean()).alias('at_turnover')).with_columns(pl.when(pl.col('count') <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('at_turnover')).alias('at_turnover'))
            #Receivables Turnover 
            .with_columns((pl.col('sale_x')/pl.concat_list(['rect', pl.col('rect').shift(12)]).list.mean()).alias('rec_turnover')).with_columns(pl.when(pl.col('count') <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('rec_turnover')).alias('rec_turnover'))
            #Account Payables Turnover
            .with_columns(((pl.col('cogs') + pl.col('invt'))/pl.concat_list(['ap', pl.col('ap').shift(12)]).list.mean()).alias('ap_turnover')).with_columns(pl.when(pl.col('count') <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('ap_turnover')).alias('ap_turnover'))
    )




    #Miscellaneous Ratios
    __chars5 = (__chars5
                #advertising as % of sales
               .with_columns((pl.col('xad') / pl.col('sale_x')).alias('adv_sale')) 
                #labor expense as % of sales
               .with_columns((pl.col('xlr') / pl.col('sale_x')).alias('staff_sale'))
                #sale per $ Book Enterprise Value
               .with_columns((pl.col('sale_x') / pl.col('bev_x')).alias('sale_bev'))
               .with_columns((pl.col('xrd') / pl.col('sale_x')).alias('rd_sale'))
                #sales per $ total stockholders' equity
               .with_columns((pl.col('sale_x') / pl.col('be_x')).alias('sale_be'))
               
               # Calculate div_ni and apply condition
               .with_columns((pl.col('div_x') / pl.coalesce(['nix_x','ni_x'])).alias('div_ni'))
               .with_columns(pl.when(pl.coalesce(['nix_x','ni_x']) <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('div_ni')).alias('div_ni'))
               
               # Calculate sale_nwc and apply condition
               .with_columns((pl.col('sale_x') / pl.col('nwc_x')).alias('sale_nwc'))
               .with_columns(pl.when(pl.col('nwc_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('sale_nwc')).alias('sale_nwc'))
               
               # Calculate tax_pi and apply condition
               .with_columns((pl.col('txt') / pl.col('pi_x')).alias('tax_pi'))
               .with_columns(pl.when(pl.col('pi_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('tax_pi')).alias('tax_pi'))
              )


    #New variables:
    __chars5 = (__chars5
         
               # Calculate cash_at and apply condition
               .with_columns((pl.col('che') / pl.col('at_x')).alias('cash_at'))  #emp not available in quarterly data created by sas code.
               .with_columns(pl.when(pl.col('at_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('cash_at')).alias('cash_at'))
               
               # Calculate ni_emp and apply condition
               .with_columns((pl.col('ni_x') / pl.col('emp')).alias('ni_emp'))
               .with_columns(pl.when(pl.col('emp') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ni_emp')).alias('ni_emp'))
    
               # Calculate sale_emp and apply condition
               .with_columns((pl.col('sale_x') / pl.col('emp')).alias('sale_emp'))
               .with_columns(pl.when(pl.col('emp') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('sale_emp')).alias('sale_emp'))
    
               # Calculate sale_emp_gr1 and apply condition
               .with_columns(((pl.col('sale_emp') / pl.col('sale_emp').shift(12)) - 1).alias('sale_emp_gr1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('sale_emp').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('sale_emp_gr1')).alias('sale_emp_gr1'))
    
               # Calculate emp_gr1 and apply condition
               .with_columns(((pl.col('emp') - pl.col('emp').shift(12)) / (0.5 * pl.col('emp') + 0.5 * pl.col('emp').shift(12))).alias('emp_gr1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('emp_gr1') == 0) | ((0.5 * pl.col('emp') + 0.5 * pl.col('emp').shift(12)) == 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('emp_gr1')).alias('emp_gr1'))
              )



    #Number of Consecutive Earnings Increases:
    __chars5 = (__chars5
         
               # checking increase from last year
               .with_columns((pl.col('ni_x') > pl.col('ni_x').shift(12)).alias('check_increase'))
               .with_columns(pl.concat_list([pl.col('check_increase').shift(i) for i in range(0, 22, 3)]).list.sum().alias('ni_inc8q'))
               .with_columns(pl.when((pl.col('count') < 33) | (pl.col('ni_inc8q') != 8))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ni_inc8q')).alias('ni_inc8q'))
                .drop('check_increase')
               )


    #1yr Change Scaled by Lagged Assets
    ch_asset_lag_vars = ['noa_x', 'ppeinv_x']
    for i in ch_asset_lag_vars:
        __chars5 = chg_to_lagassets(df=__chars5, var_gr=i)


    #1yr Change Scaled by Average Assets
    ch_asset_avg_vars = ['lnoa_x']
    for i in ch_asset_avg_vars:
        __chars5 = chg_to_avgassets(df=__chars5, var_gr=i)


    #CAPEX growth over 2 years
    __chars5 = var_growth(df=__chars5, var_gr='capx', horizon=24)


    #Quarterly profitability measures:
    __chars5 = (__chars5
                
               # Calculate saleq_gr1 and apply condition
               .with_columns(((pl.col('sale_qtr') / pl.col('sale_qtr').shift(12)) - 1).alias('saleq_gr1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('sale_qtr').shift(12) < 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('saleq_gr1')).alias('saleq_gr1'))
    
               # Calculate niq_be and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('be_x').shift(3)).alias('niq_be'))
               .with_columns(pl.when((pl.col('count') <= 3) | (pl.col('be_x').shift(3) < 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_be')).alias('niq_be'))
    
               # Calculate niq_at and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('at_x').shift(3)).alias('niq_at'))
               .with_columns(pl.when((pl.col('count') <= 3) | (pl.col('at_x').shift(3) < 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_at')).alias('niq_at'))
    
               # Calculate niq_be_chg1 and apply condition
               .with_columns((pl.col('niq_be') - pl.col('niq_be').shift(12)).alias('niq_be_chg1'))
               .with_columns(pl.when(pl.col('count') <= 12)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_be_chg1')).alias('niq_be_chg1'))
    
               # Calculate niq_at_chg1 and apply condition
               .with_columns((pl.col('niq_at') - pl.col('niq_at').shift(12)).alias('niq_at_chg1'))
               .with_columns(pl.when(pl.col('count') <= 12)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_at_chg1')).alias('niq_at_chg1'))
              )


    #R&D capital-to-assets
    __chars5 = (__chars5
    
               # Calculate rd5_at
               .with_columns(
                   ((pl.col('xrd') + pl.col('xrd').shift(12) * 0.8 + pl.col('xrd').shift(24) * 0.6 + pl.col('xrd').shift(36) * 0.4 + pl.col('xrd').shift(48) * 0.2) / pl.col('at_x')).alias('rd5_at')
               )
               # Apply condition to rd5_at
               .with_columns(
                   pl.when((pl.col('count') <= 48) | (pl.col('at_x') <= 0))
                   .then(pl.lit(None).cast(pl.Float64))
                   .otherwise(pl.col('rd5_at')).alias('rd5_at'))
               )



    #Abarbanell and Bushee (1998)
    ch_asset_AandB = ['sale_x', 'invt', 'rect', 'gp_x', 'xsga']
    for i in ch_asset_AandB:
        __chars5 = chg_to_exp(df=__chars5, var_ce=i)


    __chars5 = (__chars5
    
               # Calculate dsale_dinv
               .with_columns((pl.col('sale_ce') - pl.col('invt_ce')).alias('dsale_dinv'))
    
               # Calculate dsale_drec
               .with_columns((pl.col('sale_ce') - pl.col('rect_ce')).alias('dsale_drec'))
    
               # Calculate dgp_dsale
               .with_columns((pl.col('gp_ce') - pl.col('sale_ce')).alias('dgp_dsale'))
    
               # Calculate dsale_dsga
               .with_columns((pl.col('sale_ce') - pl.col('xsga_ce')).alias('dsale_dsga'))
              ).drop(['sale_ce', 'invt_ce', 'rect_ce', 'gp_ce', 'xsga_ce'])



    #Earnings and Revenue 'Surpise'
    __chars5 = standardized_unexpected(df=__chars5, var='sale_qtr', qtrs=8, qtrs_min=6)
    __chars5 = standardized_unexpected(df=__chars5, var='ni_qtr', qtrs=8, qtrs_min=6)



    #Abnormal Corporate Investment
    __chars5 = (__chars5
               # Calculate __capex_sale and its condition
               .with_columns((pl.col('capx') / pl.col('sale_x')).alias('__capex_sale'))
               .with_columns(pl.when(pl.col('sale_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__capex_sale')).alias('__capex_sale'))
    
               # Calculate capex_abn
               .with_columns((
                   pl.col('__capex_sale') / (
                       (pl.col('__capex_sale').shift(12) +
                        pl.col('__capex_sale').shift(24) +
                        pl.col('__capex_sale').shift(36)) / 3
                   ) - 1).alias('capex_abn'))
               # Apply condition to capex_abn
               .with_columns(
                   pl.when(pl.col('count') <= 36)
                   .then(pl.lit(None).cast(pl.Float64))
                   .otherwise(pl.col('capex_abn'))
               .alias('capex_abn'))
                
               .drop('__capex_sale'))



    #Profit scaled by lagged 
    __chars5 = (__chars5
    
               # Calculate op_atl1 and apply its conditions
               .with_columns((pl.col('op_x') / pl.col('at_x').shift(12)).alias('op_atl1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('op_atl1')).alias('op_atl1'))
    
               # Calculate gp_atl1 and apply its conditions
               .with_columns((pl.col('gp_x') / pl.col('at_x').shift(12)).alias('gp_atl1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('gp_atl1')).alias('gp_atl1'))
    
               # Calculate ope_bel1 and apply its conditions
               .with_columns((pl.col('ope_x') / pl.col('be_x').shift(12)).alias('ope_bel1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('be_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ope_bel1')).alias('ope_bel1'))
    
               # Calculate cop_atl1 and apply its conditions
               .with_columns((pl.col('cop_x') / pl.col('at_x').shift(12)).alias('cop_atl1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('cop_atl1')).alias('cop_atl1'))
              )



    #Profitability Measures
    __chars5 = (__chars5
    
               # Calculate pi_nix and apply its conditions
               .with_columns((pl.col('pi_x') / pl.col('nix_x')).alias('pi_nix'))
               .with_columns(pl.when((pl.col('pi_x') <= 0) | (pl.col('nix_x') <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('pi_nix')).alias('pi_nix'))
    
               # Calculate ocf_at and apply its conditions
               .with_columns((pl.col('ocf_x') / pl.col('at_x')).alias('ocf_at'))
               .with_columns(pl.when(pl.col('at_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ocf_at')).alias('ocf_at'))
    
               # Calculate op_at and apply its conditions
               .with_columns((pl.col('op_x') / pl.col('at_x')).alias('op_at'))
               .with_columns(pl.when(pl.col('at_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('op_at')).alias('op_at'))
    
               # Calculate ocf_at_chg1 and apply its conditions
               .with_columns((pl.col('ocf_at') - pl.col('ocf_at').shift(12)).alias('ocf_at_chg1'))
               .with_columns(pl.when(pl.col('count') <= 12)
                             .then(pl.lit(None).cast(pl.Float64))
                           .otherwise(pl.col('ocf_at_chg1')).alias('ocf_at_chg1'))
              )
  


    #Book Leverage:
    __chars5 = (__chars5.with_columns((pl.col('at_x') / pl.col('be_x')).alias('at_be')))



    #Volatility Quarterly Items
    __chars5 = (__chars5
    
               # Calculate __ocfq_saleq and apply condition
               .with_columns((pl.col('ocf_qtr') / pl.col('sale_qtr')).alias('__ocfq_saleq'))
               .with_columns(pl.when(pl.col('sale_qtr') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__ocfq_saleq')).alias('__ocfq_saleq'))
    
               # Calculate __niq_saleq and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('sale_qtr')).alias('__niq_saleq'))
               .with_columns(pl.when(pl.col('sale_qtr') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__niq_saleq')).alias('__niq_saleq'))
    
               # Calculate __roeq and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('be_x')).alias('__roeq'))
               .with_columns(pl.when(pl.col('be_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__roeq')).alias('__roeq'))
              )
    
    __chars5 = volq(df=__chars5, name='ocfq_saleq_std', var='__ocfq_saleq', qtrs=16, qtrs_min=8)
    __chars5 = volq(df=__chars5, name='niq_saleq_std', var='__niq_saleq', qtrs=16, qtrs_min=8)
    __chars5 = volq(df=__chars5, name='roeq_be_std', var='__roeq', qtrs=20, qtrs_min=12)
    __chars5 = __chars5.drop(['__ocfq_saleq', '__niq_saleq', '__roeq'])



    #Volatility Annual Items:
    __chars5 = (__chars5
    
               # Calculate __roe and apply condition
               .with_columns((pl.col('ni_x') / pl.col('be_x')).alias('__roe'))
               .with_columns(pl.when(pl.col('be_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__roe')).alias('__roe'))
               )
    
    __chars5 = vola(df=__chars5, name='roe_be_std', var='__roe', yrs=5, yrs_min=5)
    __chars5 = __chars5.drop('__roe')


    #Earnings Smoothness
    __chars5 = earnings_variability(df=__chars5, esm_h=5)



    #Asset Liquidity:
    __chars5 = (__chars5
    
               .with_columns((pl.col('aliq_x') / pl.col('at_x').shift(12)).alias('aliq_at'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <=0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('aliq_at')).alias('aliq_at'))
               )

    #Equity Duration
    __chars5 = equity_duration_cd(df=__chars5, horizon=10, r=0.12, roe_mean=0.12, roe_ar1=0.57, g_mean=0.06, g_ar1=0.24)

    #F-score
    __chars5 = pitroski_f(df=__chars5, name='f_score')

    #O-score
    __chars5 =  ohlson_o(df=__chars5, name='o_score')

    #Z-score
    __chars5 =  altman_z(df=__chars5, name='z_score')

    #Intrinsics value
    __chars5 = intrinsic_value(df= __chars5, name ='intrinsic_value', r=0.12)

    #Kz-index
    __chars5 = kz_index(df= __chars5, name ='kz_index')


    #5 year ratio change (For quality minus junk variables)
    __chars5 = chg_var1_to_var2(df=__chars5, name='gpoa_ch5', var1='gp_x', var2='at_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='roe_ch5', var1='ni_x', var2='be_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='roa_ch5', var1='ni_x', var2='at_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='cfoa_ch5', var1='ocf_x', var2='at_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='gmar_ch5', var1='gp_x', var2='sale_x', horizon=60);


    #delete helper variables
    __chars5 = __chars5.drop('count')


    #Earning's persistence
    earnings_pers = earnings_persistence(data=__chars5, __n=2, __min=2)
    __chars6 = __chars5.join(earnings_pers, left_on=['gvkey', 'curcd', 'datadate'], right_on=['gvkey', 'curcd', 'datadate'], how='left').select(__chars5.columns + ['ni_ar1', 'ni_ivol'])

    #Keep only dates with accounting data
    __chars7  = __chars6.filter(pl.col('data_available')==1).sort(["gvkey", "datadate"])


    #lagging for public availability of data
    __chars8 =  __chars7.with_columns(pl.col('datadate').dt.offset_by(f'{lag_to_public}mo_saturating').dt.month_end().alias('start_date'))
    __chars8 =  __chars8.with_columns(pl.col('start_date').shift(-1).over(['gvkey']).alias('next_start_date'))
    __chars8 =  __chars8.with_columns(pl.min_horizontal((pl.col('next_start_date').dt.offset_by('-1mo_saturating').dt.month_end()),(pl.col('datadate').dt.offset_by(f'{max_data_lag}mo_saturating').dt.month_end())).alias('end_date'))
    __chars8 = __chars8.drop('next_start_date')


    __chars9 = expand(data=__chars8, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='public_date')

    #Convert All Raw (non-scaled) Variables to USD
    __chars10 = __chars9.join(fx, left_on=['curcd', 'public_date'], right_on=['curcdd', 'date'], how='left').select(__chars9.columns + ['fx'])

    var_raw = ['assets', 'sales', 'book_equity', 'net_income']
    for i in var_raw:
        __chars11 = __chars10.with_columns((pl.col(i)*pl.col('fx')).alias(i))
    
    __chars11 = __chars11.drop('curcd')


    #adding and filtering market return data
    __me_data1 = me_data.filter(
            (pl.col("gvkey").is_not_null()) & 
            (pl.col("primary_sec") == 1) & 
            (pl.col("me_company").is_not_null()) & 
            (pl.col("common") == 1) & 
            (pl.col("obs_main") == 1)
        ).select(
        ['gvkey', 'eom', 'me_company']).group_by(
        ["gvkey", "eom"]).agg(pl.col("me_company").max())

    __chars12 = __chars11.join(__me_data1, left_on=['gvkey', 'public_date'], right_on=['gvkey', 'eom'], how='left').select(__chars11.columns + ['me_company'])
    __chars13 = __chars12.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'])

    #Create Ratios using both Accounting and Market Value
    __chars14 = (__chars13

             #calculting market enterprise value    
            .with_columns((pl.col('me_company') + (pl.col('netdebt_x')*pl.col('fx'))).alias('mev'))
            .with_columns(pl.when((pl.col('mev') <= 0))
                          .then(None)
                          .otherwise(pl.col('mev')).alias('mev'))

             #calculating market asset value    
                         .with_columns((pl.col('at_x') * pl.col('fx') - pl.col('be_x') * pl.col('fx') + pl.col('me_company')).alias('mat'))
            .with_columns(pl.when((pl.col('mat') <= 0))
                          .then(None)
                          .otherwise(pl.col('mat')).alias('mat'))

            #correcting market value in case it is negative (should we do it before calculating the above two?)
            .with_columns(pl.when((pl.col('me_company') <= 0))
                          .then(None)
                          .otherwise(pl.col('me_company')).alias('me_company'))       
         )


    #Characteristics Scaled by Market Equity
    me_vars = [
        "at_x", "be_x", "debt_x", "netdebt_x", "che", "sale_x", "gp_x", "ebitda_x",
        "ebit_x", "ope_x", "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "div_x",
        "eqbb_x", "eqis_x", "eqpo_x", "eqnpo_x", "eqnetis_x", "xrd"
    ]
    
    for i in me_vars:
        __chars14 = scale_me(df=__chars14, var=i)
    
    #Characteristics Scaled by Market Enterprise Value
    
    mev_vars = [
        "at_x", "bev_x", "ppent", "be_x", "che", "sale_x", "gp_x", "ebitda_x",
        "ebit_x", "ope_x", "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "debt_x",
        "pstk_x", "dltt", "dlc", "dltnetis_x", "dstnetis_x", "dbnetis_x", 
        "netis_x", "fincf_x"
    ]
    
    for i in mev_vars:
        __chars14 = scale_mev(df=__chars14, var=i)


    __chars14 = (__chars14
        .with_columns(
            ((pl.col('intrinsic_value') * pl.col('fx')) / (pl.col('me_company'))).alias('ival_me')
        )
                )


    #Characteristics Scaled by Market Assets
    __chars14 = (__chars14
        .with_columns(
            ((pl.col('aliq_x') * pl.col('fx')) / (pl.col('mat').shift(12))).alias('aliq_mat')
        )
        .with_columns(
            pl.when(pl.col('gvkey') != pl.col('gvkey').shift(12))
            .then(None)
            .otherwise(pl.col('aliq_mat')).alias('aliq_mat')
        )
    )
    
    #Size Measure
    __chars14 = (__chars14
        .with_columns(
            (pl.col('mev')).alias('enterprise_value')
        )
    )
    
    #Equity Duration
    __chars14 = (__chars14
        .with_columns(
            ((pl.col('ed_cd_w') * pl.col('fx')) / (pl.col('me_company')) + pl.col('ed_constant') * (pl.col('me_company') - pl.col('ed_cd') * pl.col('fx'))/pl.col('me_company')).alias('eq_dur')
        )
        .with_columns(
            pl.when((pl.col('ed_err') ==1) | (pl.col('eq_dur') <=0))
            .then(None)
            .otherwise(pl.col('eq_dur')).alias('eq_dur')
        )
    )



    #renaming columns:
    __chars15 = __chars14
    rename_dict = {
        "xrd": "rd",
        "xsga": "sga",
        "dlc": "debtst",
        "dltt": "debtlt",
        "oancf": "ocf",
        "ppegt": "ppeg",
        "ppent": "ppen",
        "che": "cash",
        "invt": "inv",
        "rect": "rec",
        "txt": "tax",
        "ivao": "lti",
        "ivst": "sti",
        "sale_qtr": "saleq",
        "ni_qtr": "niq",
        "ocf_qtr": "ocfq"
    }
    
    
    for a, b in rename_dict.items():
        __chars15 = __chars15.rename({col: col.replace(a, b) for col in __chars15.columns})


    #selecting variable columns of interest
    __chars16 = __chars15.select(['source', 'gvkey', 'public_date', 'datadate'] + __keep_vars)


    #addinf sufiix if mentioned
    if suffix is None:
        __chars16 = __chars16
    else:
        for i in keep_vars:
        __chars16 = __chars16.rename({i:i+suffix})


    output = __chars16.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'])

    return output

def chg_to_lagassets(df, var_gr):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    
    # Appending '_gr' and '1a' to the name
    name_gr = f"{name_gr}_gr1a"
    
    # Calculating the growth rate
    df = df.with_columns(
        ((pl.col(var_gr) - pl.col(var_gr).shift(12))/pl.col('at_x').shift(12)).alias(name_gr)
    )
    
    # Applying conditions to set certain values to NaN
    df = df.with_columns(
        pl.when((pl.col('at_x').shift(12) <= 0) | 
                (pl.col("count") <= 12))
        .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name_gr)).alias(name_gr)
    )
    
    return df


def chg_to_avgassets(df, var_gr):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    
    # Appending '_gr' and '1a' to the name
    name_gr = f"{name_gr}_gr1a"
    
    # Calculating the growth rate
    df = df.with_columns(
        ((pl.col(var_gr) - pl.col(var_gr).shift(12))/pl.col('at_x').shift(12)).alias(name_gr)
    )
    
    # Applying conditions to set certain values to NaN
    df = df.with_columns(
        pl.when((pl.col('at_x').shift(12) <= 0) | 
                (pl.col("count") <= 12))
        .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name_gr)).alias(name_gr)
    )
    
    return df


def chg_to_exp(df, var_ce):
    # Removing '_x' from the column name
    name_ce = var_ce.replace('_x', '')
    
    # Appending '_gr' and '1a' to the name
    name_ce = f"{name_ce}_ce"
    
    # Calculating the growth rate
    df = df.with_columns(
        ((pl.col(var_ce)/((pl.col(var_ce).shift(12) + pl.col(var_ce).shift(24))/2)) -1).alias(name_ce)
    )
    
    # Applying conditions to set certain values to NaN
    df = df.with_columns(
        pl.when((pl.col('count') <= 24) | 
                (((pl.col(var_ce).shift(12) + pl.col(var_ce).shift(24))/2) <= 0))
        .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name_ce)).alias(name_ce)
    )
    
    return df


def standardized_unexpected(df, var, qtrs, qtrs_min):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    
    # Appending '_su' to the name
    name = f"{name}_su"

    #creating helping variables
    df = df.with_columns((pl.col(var) - pl.col(var).shift(12)).alias('__chg'))
    df = df.with_columns(pl.concat_list([pl.col('__chg').shift(i) for i in range(0, (3*qtrs), 3)]).list.mean().alias('__chg_mean'))
    df = df.with_columns(pl.concat_list([pl.col('__chg').shift(i) for i in range(0, (3*qtrs), 3)]).list.eval(pl.element().std()).alias('__chg_std')).explode('__chg_std')
    df = df.with_columns(pl.concat_list([pl.col('__chg').shift(i) for i in range(0, (3*qtrs), 3)]).list.eval(pl.element().is_not_null().sum()).alias('__chg_n')).explode('__chg_n')
    df = df.with_columns(pl.when(pl.col('__chg_n') <= qtrs_min).then(pl.lit(None)).otherwise(pl.col("__chg_mean")).alias('__chg_mean'))
    df = df.with_columns(pl.when(pl.col('__chg_n') <= qtrs_min).then(pl.lit(None)).otherwise(pl.col("__chg_std")).alias('__chg_std'))

    #creating the characteristic
    df = df.with_columns(
        ((pl.col(var) + pl.col(var).shift(12) + pl.col('__chg_mean').shift(3)) /(pl.col('__chg_std').shift(3))).alias(name)
    )
    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when(pl.col('count') <= (12 + qtrs*3)).then(pl.lit(None)).otherwise(pl.col(name)).alias(name))
    df = df.drop(['__chg', '__chg_mean', '__chg_std', '__chg_n'])
    
    return df



def volq(df, name, var, qtrs, qtrs_min):

    #creating helping variables
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (3*qtrs), 3)]).list.eval(pl.element().is_not_null().sum()).alias('__n')).explode('__n')
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (3*qtrs), 3)]).list.eval(pl.element().std()).alias(name)).explode(name)

    
    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when((pl.col('count') <= ((qtrs-1)*3)) | (pl.col('__n') < qtrs_min)).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name)).alias(name))
    df = df.drop('__n')
    
    return df


def vola(df, name, var, yrs, yrs_min):

    #creating helping variables
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (12*yrs), 12)]).list.eval(pl.element().is_not_null().sum()).alias('__n')).explode('__n')
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (12*yrs), 12)]).list.eval(pl.element().std()).alias(name)).explode(name)

    
    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when((pl.col('count') <= ((yrs-1)*12)) | (pl.col('__n') < yrs_min)).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name)).alias(name))
    df = df.drop('__n')
    
    return df

__chars5 = (__chars5

           # Calculate __roe and apply condition
           .with_columns((pl.col('che') + 0.715 * pl.col('rect') + 0.547 * pl.col('invt') + 0.535 * pl.col('ppegt')).alias('tangibility'))
           )


def earnings_variability(df, esm_h):

    #creating helping variables
    df = df.with_columns((pl.col('ni_x')/pl.col('at_x').shift(12)).alias('__roa'))
    df = df.with_columns((pl.col('ocf_x')/pl.col('at_x').shift(12)).alias('__croa'))
    df = df.with_columns(pl.concat_list([pl.col('__roa').shift(i) for i in range(0, (12*esm_h), 12)]).list.eval(pl.element().is_not_null().sum()).alias('__roa_n')).explode('__roa_n')
    df = df.with_columns(pl.concat_list([pl.col('__croa').shift(i) for i in range(0, (12*esm_h), 12)]).list.eval(pl.element().is_not_null().sum()).alias('__croa_n')).explode('__croa_n')
    df = df.with_columns(pl.concat_list([pl.col('__roa').shift(i) for i in range(0, (12*esm_h), 12)]).list.eval(pl.element().std()).alias('__roa_std')).explode('__roa_std')
    df = df.with_columns(pl.concat_list([pl.col('__croa').shift(i) for i in range(0, (12*esm_h), 12)]).list.eval(pl.element().std()).alias('__croa_std')).explode('__croa_std')


    #calculating earning variability:
    df = df.with_columns((pl.col('__roa_std')/pl.col('__croa_std')).alias('earnings_variability'))

    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when((pl.col('count') <= (12*esm_h)) | (pl.col('__croa_std') <= 0) | (pl.col('__roa_n') < esm_h) | (pl.col('__croa_n') < esm_h))
                         .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('earnings_variability')).alias('earnings_variability'))
    df = df.drop(['__roa', '__croa', '__roa_n', '__croa_n', '__roa_std', '__croa_std'])
    
    return df


def equity_duration_cd(df, horizon, r, roe_mean, roe_ar1, g_mean, g_ar1):

    #creating initial variables
    df = df.with_columns((pl.col('ni_x')/pl.col('be_x').shift(12)).alias('__roe0')).with_columns(pl.when((pl.col('count') <= 12) | (pl.col('be_x').shift(12) <=1)).then(pl.lit(None)).otherwise(pl.col('__roe0')).alias('__roe0'))
    df = df.with_columns(((pl.col('sale_x')/pl.col('sale_x').shift(12))-1).alias('__g0')).with_columns(pl.when((pl.col('count') <= 12) | (pl.col('sale_x').shift(12) <=1)).then(pl.lit(None)).otherwise(pl.col('__g0')).alias('__g0'))
    df = df.with_columns(pl.col('be_x').alias('__be0'))

    #forecast cash distributions
    roe_c = roe_mean*(1 - roe_ar1)
    g_c = g_mean*(1 - g_ar1)
    for i in range(1, horizon+1):
        j = i-1
        df = df.with_columns((roe_c + roe_ar1*pl.col(f'__roe{j}')).alias(f'__roe{i}'))
        df = df.with_columns((g_c + g_ar1*pl.col(f'__g{j}')).alias(f'__g{i}'))
        df = df.with_columns((pl.col(f'__be{j}') * (1 + pl.col(f'__g{i}'))).alias(f'__be{i}'))
        df = df.with_columns((pl.col(f'__be{j}') * (pl.col(f'__roe{i}') - pl.col(f'__g{i}'))).alias(f'__cd{i}'))

    #Create Duration Helper Variables
    df = df.with_columns((pl.lit(horizon) + ((1 + r) / r)).alias('ed_constant'))
    df = df.with_columns(pl.lit(0).alias('ed_cd_w'))
    df = df.with_columns(pl.lit(0).alias('ed_cd'))
    df = df.with_columns(pl.lit(0).alias('ed_err'))
    for t in range(1, horizon+1):
         df = df.with_columns((pl.col('ed_cd_w') + t *pl.col(f'__cd{t}')/((1 + r)**t)).alias('ed_cd_w'))
         df = df.with_columns((pl.col('ed_cd') + pl.col(f'__cd{t}')/((1 + r)**t)).alias('ed_cd'))
         df = df.with_columns(pl.when(pl.col(f'__be{t}') < 0).then(pl.col('ed_err') == pl.lit(1)).alias('ed_err'))
        
    df = df.drop(['__roe', '__g', '__be', '__cd'])
    
    return df


def pitroski_f(df, name):
    df = (df

           # Calculate __f_roa and apply condition
           .with_columns((pl.col('ni_x') / pl.col('at_x').shift(12)).alias('__f_roa'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_roa')).alias('__f_roa'))

           # Calculate __f_croa and apply condition
           .with_columns((pl.col('ocf_x') / pl.col('at_x').shift(12)).alias('__f_croa'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_croa')).alias('__f_croa'))

           # Calculate __f_droa and apply condition
           .with_columns((pl.col('__f_roa') - pl.col('__f_roa').shift(12)).alias('__f_droa'))
           .with_columns(pl.when(pl.col('count') <= 12)
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_droa')).alias('__f_droa'))

           # Calculate __f_acc without condition
           .with_columns((pl.col('__f_croa') - pl.col('__f_roa')).alias('__f_acc'))

           # Calculate __f_lev and apply condition
           .with_columns(((pl.col('dltt') / pl.col('at_x')) - (pl.col('dltt') / pl.col('at_x').shift(12))).alias('__f_lev'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x') <= 0) | (pl.col('at_x').shift(12) <= 0))
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_lev')).alias('__f_lev'))

           # Calculate __f_liq and apply condition
           .with_columns(((pl.col('ca_x') / pl.col('cl_x')) - (pl.col('ca_x') / pl.col('cl_x').shift(12))).alias('__f_liq'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('cl_x') <= 0) | (pl.col('cl_x').shift(12) <= 0))
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_liq')).alias('__f_liq'))

           # Calculate __f_eqis without condition
           .with_columns(pl.col('eqis_x').alias('__f_eqis'))

           # Calculate __f_gm and apply condition
           .with_columns(((pl.col('gp_x') / pl.col('sale_x')) - (pl.col('gp_x') / pl.col('sale_x').shift(12))).alias('__f_gm'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('sale_x') <= 0) | (pl.col('sale_x').shift(12) <= 0))
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_gm')).alias('__f_gm'))

           # Calculate __f_aturn and apply condition
           .with_columns(((pl.col('sale_x') / pl.col('at_x').shift(12)) - (pl.col('sale_x').shift(12) / pl.col('at_x').shift(24))).alias('__f_aturn'))
           .with_columns(pl.when((pl.col('count') <= 24) | (pl.col('at_x').shift(12) <= 0) | (pl.col('at_x').shift(24) <= 0))
                         .then(pl.lit(None).cast(pl.Float64))
                         .otherwise(pl.col('__f_aturn')).alias('__f_aturn'))

           # Calculate &name. and apply conditions
           .with_columns(((pl.col('__f_roa') > 0).cast(pl.Int32)
                         + (pl.col('__f_croa') > 0).cast(pl.Int32)
                         + (pl.col('__f_droa') > 0).cast(pl.Int32)
                         + (pl.col('__f_acc') > 0).cast(pl.Int32)
                         + (pl.col('__f_lev') < 0).cast(pl.Int32)
                         + (pl.col('__f_liq') > 0).cast(pl.Int32)
                         + (pl.coalesce([pl.col('__f_eqis'), 0]) == 0).cast(pl.Int32)
                         + (pl.col('__f_gm') > 0).cast(pl.Int32)
                         + (pl.col('__f_aturn') > 0).cast(pl.Int32))
                         .alias(name)
           )
           .with_columns(pl.when(
                pl.col('__f_roa').is_null() |
                pl.col('__f_croa').is_null() |
                pl.col('__f_droa').is_null() |
                pl.col('__f_acc').is_null() |
                pl.col('__f_lev').is_null() |
                pl.col('__f_liq').is_null() |
                pl.col('__f_gm').is_null() |
                pl.col('__f_aturn').is_null()
            ).then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col(name)).alias(name))
         )

    df = df.drop(['__f_roa', '__f_croa', '__f_droa', '__f_acc', '__f_lev', '__f_liq', '__f_eqis', '__f_gm', '__f_aturn',])


    return df


def ohlson_o(df, name):
    df = (df
           # Calculate __o_lat
           .with_columns((pl.col('at_x').log()).alias('__o_lat'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_lat')).alias('__o_lat'))

           # Calculate __o_lev
           .with_columns((pl.col('debt_x') / pl.col('at_x')).alias('__o_lev'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_lev')).alias('__o_lev'))

           # Calculate __o_wc
           .with_columns(((pl.col('ca_x') - pl.col('cl_x')) / pl.col('at_x')).alias('__o_wc'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_wc')).alias('__o_wc'))

           # Calculate __o_roe
           .with_columns((pl.col('nix_x') / pl.col('at_x')).alias('__o_roe'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_roe')).alias('__o_roe'))

           # Calculate __o_cacl
           .with_columns((pl.col('cl_x') / pl.col('ca_x')).alias('__o_cacl'))
           .with_columns(pl.when(pl.col('ca_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_cacl')).alias('__o_cacl'))

           # Calculate __o_ffo
           .with_columns(((pl.col('pi_x') + pl.col('dp')) / pl.col('lt')).alias('__o_ffo'))
           .with_columns(pl.when(pl.col('lt') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_ffo')).alias('__o_ffo'))

           # Calculate __o_neg_eq
           .with_columns((pl.col('lt') > pl.col('at_x')).cast(pl.Int32).alias('__o_neg_eq'))
           .with_columns(pl.when(pl.col('lt').is_null() | pl.col('at_x').is_null())
                         .then(None)
                         .otherwise(pl.col('__o_neg_eq')).alias('__o_neg_eq'))

           # Calculate __o_neg_earn
           .with_columns(((pl.col('nix_x') < 0) & (pl.col('nix_x').shift(12) < 0)).cast(pl.Int32).alias('__o_neg_earn'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('nix_x').is_null()) | (pl.col('nix_x').shift(12).is_null()))
                         .then(None)
                         .otherwise(pl.col('__o_neg_earn')).alias('__o_neg_earn'))

           # Calculate __o_nich
           .with_columns(((pl.col('nix_x') - pl.col('nix_x').shift(12)) / (pl.col('nix_x').abs() + pl.col('nix_x').shift(12).abs())).alias('__o_nich'))
           .with_columns(pl.when((pl.col('count') <= 12) | ((pl.col('nix_x').abs() + pl.col('nix_x').shift(12).abs()) == 0))
                         .then(None)
                         .otherwise(pl.col('__o_nich')).alias('__o_nich'))

           # Calculate O-score using the variables and their conditions
           .with_columns((-1.32 - 0.407 * pl.col('__o_lat') + 6.03 * pl.col('__o_lev') 
                         + 1.43 * pl.col('__o_wc') + 0.076 * pl.col('__o_cacl') 
                         - 1.72 * pl.col('__o_neg_eq') - 2.37 * pl.col('__o_roe') 
                         - 1.83 * pl.col('__o_ffo') + 0.285 * pl.col('__o_neg_earn') 
                         - 0.52 * pl.col('__o_nich')).alias(name))
         )
    return df


def altman_z(df, name):
    df = (df
            # creating helper variables
           # Calculate __z_wc
           .with_columns(((pl.col('ca_x') - pl.col('cl_x')) / pl.col('at_x')).alias('__z_wc'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_wc')).alias('__z_wc'))

           # Calculate __z_re
           .with_columns((pl.col('re') / pl.col('at_x')).alias('__z_re'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_re')).alias('__z_re'))

           # Calculate __z_eb
           .with_columns((pl.col('ebitda_x') / pl.col('at_x')).alias('__z_eb'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_eb')).alias('__z_eb'))

           # Calculate __z_sa
           .with_columns((pl.col('sale_x') / pl.col('at_x')).alias('__z_sa'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_sa')).alias('__z_sa'))

           # Calculate __z_me
           .with_columns((pl.col('me_fiscal') / pl.col('lt')).alias('__z_me'))
           .with_columns(pl.when(pl.col('lt') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_me')).alias('__z_me'))

           # Calculate temporary z-score
           .with_columns((1.2 * pl.col('__z_wc') + 1.4 * pl.col('__z_re') 
                         + 3.3 * pl.col('__z_eb') + 0.6 * pl.col('__z_me') 
                         + 1.0 * pl.col('__z_sa')).alias(name))
         )
    df =df.drop(['__z_wc', '__z_re', '__z_eb', '__z_sa', '__z_me'])
    
    return df

__chars5 =  altman_z(df=__chars5, name='z_score')

def intrinsic_value(df, name, r):
    df = (df
            # creating helper variables
           # Calculate __iv_po
           .with_columns((pl.col('div_x')/ pl.col('nix_x')).alias('__iv_po'))
           .with_columns(pl.when(pl.col('nix_x') <= 0)
                         .then((pl.col('div_x')/ (pl.col('at_x') * 0.06)))
                         .otherwise(pl.col('__iv_po')).alias('__iv_po'))

           # Calculate __iv_roe
           .with_columns((pl.col('nix_x') / ((pl.col('be_x') +  pl.col('be_x').shift(12))/2)).alias('__iv_roe'))
           .with_columns(pl.when((pl.col('count') <= 12) | ((pl.col('be_x') +  pl.col('be_x').shift(12)) <= 0))
                         .then(None)
                         .otherwise(pl.col('__iv_roe')).alias('__iv_roe'))

           # Calculate __iv_be1
           .with_columns(((1 + (1 - pl.col('__iv_po')) * pl.col('__iv_roe')) * pl.col('be_x')).alias('__iv_be1'))


           # Calculate intrinsic value
           .with_columns(( pl.col('be_x') + (((pl.col('__iv_roe') - r)/(1+ r)) * pl.col('be_x')) + (((pl.col('__iv_roe') - r)/((1+ r) * r)) * pl.col('__iv_be1'))).alias(name))
           .with_columns(pl.when(pl.col(name) <= 0)
                         .then(None)
                         .otherwise(pl.col(name)).alias(name))
         )
    df =df.drop(['__iv_po', '__iv_roe', '__iv_be1'])
    
    return df


def kz_index(df, name):

# Assume that __chars5 is your initial DataFrame and you have added the appropriate columns.
    df = (df
            # Calculate __kz_cf
            .with_columns(((pl.col('ni_x') + pl.col('dp')) / pl.col('ppent').shift(12)).alias('__kz_cf'))
            .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('ppent').shift(12) <= 0))
                          .then(None)
                          .otherwise(pl.col('__kz_cf')).alias('__kz_cf'))
            
            # Calculate __kz_q
            .with_columns(((pl.col('at_x') + pl.col('me_fiscal') - pl.col('be_x')) / pl.col('at_x')).alias('__kz_q'))
            .with_columns(pl.when(pl.col('at_x') <= 0)
                          .then(None)
                          .otherwise(pl.col('__kz_q')).alias('__kz_q'))
            
            # Calculate __kz_db
            .with_columns((pl.col('debt_x') / (pl.col('debt_x') + pl.col('seq_x'))).alias('__kz_db'))
            .with_columns(pl.when((pl.col('debt_x') + pl.col('seq_x')) == 0)
                          .then(None)
                          .otherwise(pl.col('__kz_db')).alias('__kz_db'))
            
            # Calculate __kz_dv
            .with_columns((pl.col('div_x') / pl.col('ppent').shift(12)).alias('__kz_dv'))
            .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('ppent').shift(12) <= 0))
                          .then(None)
                          .otherwise(pl.col('__kz_dv')).alias('__kz_dv'))
            
            # Calculate __kz_cs
            .with_columns((pl.col('che') / pl.col('ppent').shift(12)).alias('__kz_cs'))
            .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('ppent').shift(12) <= 0))
                          .then(None)
                          .otherwise(pl.col('__kz_cs')).alias('__kz_cs'))
            
            # Calculate the kz_index using the helper variables
            .with_columns((- 1.002 * pl.col('__kz_cf') + 0.283 * pl.col('__kz_q') 
                           + 3.139 * pl.col('__kz_db') - 39.368 * pl.col('__kz_dv') 
                           - 1.315 * pl.col('__kz_cs')).alias(name))
)

    return df

def chg_var1_to_var2(df, name, var1, var2, horizon):

# Assume that __chars5 is your initial DataFrame and you have added the appropriate columns.
    df = (df
            # Calculate __x
            .with_columns((pl.col(var1) / pl.col(var2)).alias('__x'))
            .with_columns(pl.when(pl.col(var2) <= 0)
                          .then(None)
                          .otherwise(pl.col('__x')).alias('__x'))
            
            # Calculate main
            .with_columns((pl.col('__x') - pl.col('__x').shift(horizon)).alias(name))
            .with_columns(pl.when(pl.col('count') <= horizon)
                          .then(None)
                          .otherwise(pl.col(name)).alias(name))
         )
    df = df.drop('__x')

    return df

__chars5 = __chars5.drop('count')

def earnings_persistence(data, __n, __min):
    __months = __n*12  
    __acc1 = data.sort(['gvkey', 'curcd', 'datadate'])
    __acc2 = __acc1.with_columns(pl.col('curcd').cumcount().over(['gvkey', 'curcd']).alias('count')).with_columns((pl.col('count') + pl.lit(1)).alias("count"))
    __acc3 = (__acc2.with_columns((pl.col('ni_x')/pl.col('at_x')).alias('__ni_at'))
             .with_columns(pl.when(pl.col('at_x') <= 0).then(None).otherwise(pl.col('__ni_at')).alias('__ni_at'))

             .with_columns(pl.col('__ni_at').shift(12).alias('__ni_at_l1'))
             .with_columns(pl.when(pl.col('count') <= 12).then(None).otherwise(pl.col('__ni_at_l1')).alias('__ni_at_l1'))
             )
    __acc4 = __acc3.filter((pl.col('__ni_at').is_not_null()) & (pl.col('__ni_at_l1').is_not_null())).select(['gvkey', 'curcd', 'datadate', '__ni_at', '__ni_at_l1'])
    __acc4 = __acc4.with_columns(pl.col('datadate').dt.month().alias('month'))
    
    datadate=[]
    gvkey = []
    curcd = []
    beta = []
    alpha = []

    grouped_df = __acc4.group_by(['month', 'gvkey', 'curcd'])
    for group_key, subgroup in grouped_df:
        start_date = subgroup['datadate'].min()
        end_date = subgroup['datadate'].max()
        date_data = pl.date_range(start_date,  end_date, "1y", eager=True).dt.month_end()
        date_data = pl.DataFrame(date_data)
        data2 = date_data.join(subgroup, left_on=["date"], right_on=["datadate"], how="left").rename({"date": "datadate"}).sort('datadate')
        for l in range(0, len(data2), __n):
            data3 = data2.slice(l,__n)
            gvkey.append(group_key[1])
            curcd.append(group_key[2])
            datadate.append(subgroup['datadate'].max())
            y = subgroup.select('__ni_at').to_numpy().flatten()
            X = subgroup.select('__ni_at_l1').to_numpy().flatten()
            if ((np.count_nonzero(~np.isnan(y)) >= __min) & (np.count_nonzero(~np.isnan(X)) >= __min)):
                X = sm.add_constant(X)
                model = sm.OLS(y, X)
                results = model.fit()
                alpha.append(results.params[0])
                beta.append(results.params[1])
            else:
                alpha.append(None)
                beta.append(None)
                        
    output = pl.DataFrame({'datadate' : datadate, 'gvkey' : gvkey, 'curcd' : curcd, 'ni_ar1' : alpha, 'ni_ivol' : beta})
    return output

__chars5 = __chars5.with_columns(
    pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
)

def scale_me(df, var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    
    # Appending '_me' to the name
    name = f"{name}_me"
    
    # Scaling
    df = df.with_columns(
        ((pl.col(var) * pl.col('fx'))/pl.col('me_company')).alias(name)
    )
    
    return df

def scale_mev(df, var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    
    # Appending '_me' to the name
    name = f"{name}_mev"
    
    # Scaling
    df = df.with_columns(
        ((pl.col(var) * pl.col('fx'))/pl.col('mev')).alias(name)
    )
    
    return df


if suffix is None:
    __chars16 = __chars16
else:
    for i in keep_vars:
    __chars16 = __chars16.rename({i:i+suffix})


output = __chars16.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'])



def combine_ann_qtr_chars(ann_df, qtr_df, char_vars, q_suffix):
    # Create a combined DataFrame by left joining on 'gvkey' and 'public_date'
    combined_df = ann_df.join(qtr_df, left_on=['gvkey', 'public_date'], right_on=['gvkey', 'public_date'], how='left', suffix=q_suffix)
    
    # Define the logic to update annual data with quarterly data if it is more recent
    for char_var in char_vars:
        combined_df = (combined_df.with_columns(
            pl.when((pl.col(char_var).is_null()) | ((pl.col(f"{char_var}{q_suffix}").is_not_null()) & (pl.col(f"datadate{q_suffix}") > pl.col('datadate'))))
            .then(pl.col(f"{char_var}{q_suffix}"))
            .otherwise(pl.col(char_var))
            .alias(char_var)
        ))
        
        # Drop the quarterly variable after the update
        combined_df = combined_df.drop(f"{char_var}{q_suffix}")
    
    # Drop the no longer needed 'datadate' fields
    combined_df = combined_df.drop(['datadate', f'datadate{q_suffix}'])
    
    # Remove duplicates based on 'gvkey' and 'public_date' and sort the DataFrame
    combined_df = combined_df.unique(subset=['gvkey', 'public_date']).sort(['gvkey', 'public_date'])
    
    return combined_df
__keep_vars = acc_chars