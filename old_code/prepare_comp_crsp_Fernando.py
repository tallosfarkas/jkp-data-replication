import polars as pl
import numpy as np
from time import time
from tqdm import tqdm

def shift_func(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(pl.col('datadate').shift(-1).alias("following"))
def process_group(df: pl.DataFrame) -> pl.DataFrame:
    return df.upsample(time_column="ddate", every="1d").fill_null(strategy="forward")
def upsample_fx(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([pl.col('curcdd').set_sorted(),pl.col('datadate').set_sorted()])
    return df.upsample(time_column='datadate', every='1d').fill_null(strategy='forward')
def upsample_fx(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([pl.col('curcdd').set_sorted(),pl.col('datadate').set_sorted()])
    return df.upsample(time_column='datadate', every='1d').fill_null(strategy='forward')
def populate_own(inset, idvar, datevar, datename):
    inset = inset.unique([idvar, datevar])
    inset_flags = inset.clone().select([pl.col(idvar), pl.col(datevar).alias(datename), pl.lit(1).alias('flag_orig')])
    inset = inset.with_columns(pl.col(datevar).alias(datename))
    inset = inset.sort([idvar, datevar]).group_by(idvar, maintain_order = True).map_groups(shift_func)
    inset = inset.with_columns(pl.col(datevar).dt.offset_by('12mo_saturating').dt.month_end().alias('forward_max'))
    inset = inset.with_columns(inset[['following','forward_max']].min(axis=1).dt.offset_by('-1d').alias('n')).drop(['following','forward_max'])
    inset_aux = inset.clone().drop(['csho_fund', 'ajex_fund', datename, datevar]).rename({'n': datename})
    inset = pl.concat([inset, inset_aux], how = 'diagonal').sort([idvar, datename])
    inset = inset.with_columns([pl.col(idvar).set_sorted(),pl.col(datename).set_sorted()])
    inset = inset.group_by("gvkey", maintain_order=True).map_groups(process_group)
    #Option that might be faster once there's more data
    #inset = inset.group_by("gvkey").map_groups(process_group).sort([idvar, datename])
    inset = inset.join(inset_flags, on = [idvar, datename], how = 'left')
    inset = inset.filter(~((pl.col('flag_orig').is_null()) & (pl.col(datename) > pl.col('n')))).drop(['n','flag_orig'])
    return inset
def compustat_fx():
    __fx1 = pl.read_ipc('Raw data/__fx1.ft')
    __fx1 = pl.concat([__fx1,pl.DataFrame({'curcdd': 'USD','datadate': '1958-01-01','fx': 1.0}).with_columns(pl.col('datadate').str.to_date('%Y-%m-%d'))]).sort(['curcdd', 'datadate'])
    __fx1 = __fx1.group_by('curcdd').map_groups(upsample_fx).sort(['curcdd', 'datadate'])
    __fx1 = __fx1.unique(subset=['datadate','curcdd']).sort(['curcdd', 'datadate'])
    return __fx1
def gen_comp_dsf(__comp_dsf_na, __comp_dsf_global):
    __comp_dsf = pl.concat([__comp_dsf_na,__comp_dsf_global], how = 'diagonal')
    fx = compustat_fx()
    fx_div = fx.clone().with_columns(pl.col('fx').alias('fx_div')).drop('fx')
    __comp_dsf =  __comp_dsf.join(fx, how = 'left', on = ['datadate', 'curcdd']) #fx datadate in our terminology is the same as date in Theis' code
    __comp_dsf = __comp_dsf.join(fx_div, how = 'left', left_on = ['datadate', 'curcddv'], right_on = ['datadate', 'curcdd'])
    __comp_dsf = __comp_dsf.with_columns([
                                        (pl.col('prc_local')*pl.col('fx')).alias('prc'),
                                        (pl.col('prc_high_lcl')*pl.col('fx')).alias('prc_high'),
                                        (pl.col('prc_low_lcl')*pl.col('fx')).alias('prc_low'),
                                        (pl.col('ri_local')*pl.col('fx')).alias('ri'),
                                       ])
    __comp_dsf = __comp_dsf.with_columns([
                                        (pl.col('prc')*pl.col('cshoc')).alias('me'),
                                        (pl.col('prc')*pl.col('cshtrd')).alias('dolvol'),
                                        ((pl.col('div').fill_null(strategy='zero'))*pl.col('fx_div')).alias('div_tot'),
                                        ((pl.col('divd').fill_null(strategy='zero'))*pl.col('fx_div')).alias('div_cash'),
                                        ((pl.col('divsp').fill_null(strategy='zero'))*pl.col('fx_div')).alias('div_spc'),
                                        pl.col('datadate').dt.month_end().alias('eom')
                                       ]).drop(['div', 'divd', 'divsp', 'fx_div', 'curcddv', 'prc_high_lcl', 'prc_low_lcl'])    
    return __comp_dsf
def adj_trd_vol_NASDAQ(datevar, col_to_adjust, exchg_var, exchg_val):
    c1 = pl.col(exchg_var) == exchg_val
    c2 = pl.col(datevar) < pl.datetime(2001, 2, 1)
    c3 = pl.col(datevar) < pl.datetime(2001, 12, 31)
    c4 = pl.col(datevar) < pl.datetime(2003, 12, 31)
    adj_trd_vol = (
                    pl.when(c1 & c2).then(pl.col(col_to_adjust) / 2)
                    .when(c1 & c3 & ~c2).then(pl.col(col_to_adjust) / 1.8)
                    .when(c1 & c4 & ~c3 & ~c2).then(pl.col(col_to_adjust) / 1.6)
                    .otherwise(pl.col(col_to_adjust))).alias(col_to_adjust)
    return adj_trd_vol
def gen_comp_msf(__comp_dsf, __firm_shares2):
    __comp_msf = __comp_dsf.clone()
    aux = __comp_msf.group_by(['gvkey', 'iid', 'eom']).agg(
                                                  aux_prc_highm = (pl.max_horizontal('prc', 'prc_high') / pl.col('ajexdi')).max(),
                                                  aux_prc_lowm = (pl.min_horizontal('prc', 'prc_low') / pl.col('ajexdi')).min(),
                                                  aux_div_totm = (pl.col('div_tot') / pl.col('ajexdi')).sum(),
                                                  aux_div_cashm = (pl.col('div_cash') / pl.col('ajexdi')).sum(),
                                                  aux_div_spcm = (pl.col('div_spc') / pl.col('ajexdi')).sum(),
                                                  aux_div_cshtrm = (pl.col('cshtrd') / pl.col('ajexdi')).sum(),
                                                  dolvolm = pl.sum('dolvol')
                                                  )
    __comp_msf = __comp_msf.join(aux, how = 'left', on = ['gvkey', 'iid', 'eom'])
    __comp_msf = __comp_msf.with_columns([
                                        (pl.col('aux_prc_highm') * pl.col('ajexdi')).alias('prc_highm'),
                                        (pl.col('aux_prc_lowm') * pl.col('ajexdi')).alias('prc_lowm'),
                                        (pl.col('aux_div_totm') * pl.col('ajexdi')).alias('div_totm'),
                                        (pl.col('aux_div_cashm') * pl.col('ajexdi')).alias('div_cashm'),
                                        (pl.col('aux_div_spcm') * pl.col('ajexdi')).alias('div_spcm'),
                                        (pl.col('aux_div_cshtrm') * pl.col('ajexdi')).alias('cshtrm')
                                        ])
    __comp_msf = __comp_msf.drop(['aux_prc_highm', 'aux_prc_lowm', 'aux_div_totm', 'aux_div_cashm', 'aux_div_spcm',
                                    'aux_div_cshtrm','cshtrd', 'div_tot', 'div_cash',
                                    'div_spc', 'dolvol', 'prc_high', 'prc_low'])
    __comp_msf = __comp_msf.filter((pl.col('prc_local').is_not_null()) & (pl.col('curcdd').is_not_null()) & (pl.col('prcstd').is_in([3, 4, 10])))
    dict_aux = {'div_totm': 'div_tot', 'div_cashm': 'div_cash', 'div_spcm': 'div_spc', 'dolvolm': 'dolvol', 'prc_highm': 'prc_high', 'prc_lowm': 'prc_low'}
    __comp_msf = __comp_msf.rename(dict_aux).sort(['gvkey', 'iid', 'eom', 'datadate'])
    #This is faster than: __comp_msf1.group_by(['gvkey', 'iid', 'eom'], maintain_order=True).last()
    __comp_msf = __comp_msf.group_by(['gvkey', 'iid', 'eom']).last().sort(['gvkey', 'iid', 'eom', 'datadate'])
    __comp_msf = __comp_msf.with_columns(pl.lit('secd').alias('source'))
    __comp_secm = pl.read_ipc('Raw data/__comp_secm1.ft')
    fx = compustat_fx()
    fx_div = fx.clone().with_columns(pl.col('fx').alias('fx_div')).drop('fx')
    __comp_secm = __comp_secm.join(__firm_shares2, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'ddate'])
    __comp_secm = __comp_secm.join(fx, how = 'left', on = ['datadate', 'curcdd']) #fx datadate in our terminology is the same as date in Theis' code
    __comp_secm = __comp_secm.join(fx_div, how = 'left', left_on = ['datadate', 'curcddvm'], right_on = ['datadate', 'curcdd'])
    __comp_secm = __comp_secm.with_columns([pl.col('datadate').dt.month_end().alias('eom'),
                                              pl.coalesce(pl.col('cshom') / 1e6, pl.col('csfsm') / 1e3, pl.col('cshoq'), (pl.col('csho_fund') * pl.col('ajex_fund')) / pl.col('ajexm')).alias('cshoc')
                                             ]).drop(['cshom', 'csfsm', 'cshoq', 'csho_fund', 'ajex_fund', 'ajexm'])
    __comp_secm = __comp_secm.with_columns(adj_trd_vol_NASDAQ('datadate', 'cshtrm', 'exchg', 14))
    __comp_secm = __comp_secm.with_columns([
                                          pl.when(pl.col('curcdd') == 'USD').then(pl.lit(1)).otherwise(pl.col('fx')).alias('fx'),
                                          pl.when(pl.col('curcddvm') == 'USD').then(pl.lit(1)).otherwise(pl.col('fx_div')).alias('fx_div')
                                         ])
    __comp_secm = __comp_secm.with_columns([
                                        (pl.col('prc_local')*pl.col('fx')).alias('prc'),
                                        (pl.col('prc_high')*pl.col('fx')).alias('prc_high'),
                                        (pl.col('prc_low')*pl.col('fx')).alias('prc_low'),
                                        (pl.col('ri_local')*pl.col('fx')).alias('ri'),
                                        (pl.col('dvpsxm')*pl.col('fx_div')).alias('div_tot')
                                         ])
    __comp_secm = __comp_secm.with_columns([
                                        (pl.col('prc')*pl.col('cshoc')).alias('me'),
                                        (pl.col('prc')*pl.col('cshtrm')).alias('dolvol'),
                                        (pl.lit(None)).alias('div_cash'),
                                        (pl.lit(None)).alias('div_spc'),
                                        (pl.lit(10.)).alias('prcstd'),
                                        (pl.lit('secm')).alias('source'),
                                         ]).drop(['dvpsxm', 'fx_div', 'curcddvm'])
    common_vars = [
    'gvkey', 'iid', 'datadate', 'eom', 'tpci', 'exchg', 'curcdd', 'prc_local', 
    'prc_high', 'prc_low', 'ajexdi', 'cshoc', 'ri_local', 'fx', 'prc', 'me', 
    'cshtrm', 'dolvol', 'ri', 'div_tot', 'div_cash', 'div_spc', 'prcstd', 'source']
    __comp_secm = __comp_secm.select(pl.col(common_vars)).with_columns(pl.col('div_cash').cast(pl.Float64),pl.col('div_spc').cast(pl.Float64))
    __comp_msf = __comp_msf.select(pl.col(common_vars))
    __comp_msf = pl.concat([__comp_msf, __comp_secm]).unique(subset = ['gvkey', 'iid', 'eom'], keep = 'first').drop('source').sort(['gvkey', 'iid', 'eom'])
    return __comp_msf
def comp_exchanges():
    __ex_country = pl.read_ipc('Raw data/__ex_country1.ft')
    special_exchanges = [
        0, 1, 2, 3, 4, 15, 16, 17, 18, 21,
        13, 19, 20, 127, 150, 157, 229, 263, 269, 281,
        283, 290, 320, 326, 341, 342, 347, 348, 349, 352]
    #15, 16, 17, 18, 21 US exchanges not in NYSE, Amex and NASDAQ
    #150 AIAF Mercado De Renta Fija --> Spanish exchange for trading debt securities https://practiceguides.chambers.com/practice-guides/capital-markets-debt-2019/spain/1-debt-marketsexchanges 
    #349 BATS Chi-X Europe --> Trades stocks from various european exchanges. Should we keep it?
    #352 CHI-X Australia --> Only Trades securities listed on ASX (exchg=106). Should we keep it?
    __ex_country = pl.SQLContext(frame=__ex_country).execute(
        """
        SELECT DISTINCT exchg,
            CASE 
                WHEN COUNT(DISTINCT excntry) > 1 THEN 'multi national'
                ELSE MAX(excntry)
            END AS excntry
        FROM frame
        WHERE excntry IS NOT NULL AND exchg IS NOT NULL
        GROUP BY exchg
        """).collect().sort('exchg')
    comp_r_ex_codes = pl.read_ipc('Raw data/comp_r_ex_codes.ft')
    __ex_country = __ex_country.join(comp_r_ex_codes, how = 'left', left_on = 'exchg', right_on = 'exchgcd')
    __ex_country = __ex_country.with_columns(pl.col('exchg').cast(pl.Int64))
    exch_main = pl.when((pl.col('excntry') != 'multi national') & pl.col('exchg').is_in(special_exchanges).not_()).then(pl.lit(1)).otherwise(pl.lit(0)).alias('exch_main')
    __ex_country = __ex_country.with_columns(exch_main)
    return __ex_country
def add_primary_sec(data, datevar):
    __prihistrow = pl.read_ipc('Raw data/__prihistrow.ft')
    __prihistusa = pl.read_ipc('Raw data/__prihistusa.ft')
    __prihistcan = pl.read_ipc('Raw data/__prihistcan.ft')
    __header = pl.read_ipc('Raw data/__header.ft').unique() #Header has duplicates
    c1 = pl.col(datevar) >= pl.col('effdate')
    c2 = pl.col(datevar) <= pl.col('thrudate')
    c3 = pl.col('thrudate').is_null()
    filter_join = (c1 & (c2 | c3))
    aux_data = data[['gvkey', datevar]].unique()
    __prihistrow_join = aux_data.join(__prihistrow, how = 'left', on = 'gvkey').filter(filter_join).drop(['effdate','thrudate'])
    __prihistusa_join = aux_data.join(__prihistusa, how = 'left', on = 'gvkey').filter(filter_join).drop(['effdate','thrudate'])
    __prihistcan_join = aux_data.join(__prihistcan, how = 'left', on = 'gvkey').filter(filter_join).drop(['effdate','thrudate'])
    data = data.join(__prihistrow_join, how = 'left', on = ['gvkey', datevar])
    data = data.join(__prihistusa_join, how = 'left', on = ['gvkey', datevar])
    data = data.join(__prihistcan_join, how = 'left', on = ['gvkey', datevar])
    data = data.join(__header, how = 'left', on = 'gvkey')
    c1 = pl.col('iid').is_not_null()
    c2 = pl.col('iid') == pl.col('prihistrow')
    c3 = pl.col('iid') == pl.col('prihistusa')
    c4 = pl.col('iid') == pl.col('prihistcan')
    primary_sec = pl.when(c1 & (c2 | c3 | c4)).then(pl.lit(1)).otherwise(pl.lit(0)).alias('primary_sec')
    data = data.with_columns([pl.coalesce(['prihistrow', 'prirow']).alias('prihistrow'),
                              pl.coalesce(['prihistusa', 'priusa']).alias('prihistusa'),
                              pl.coalesce(['prihistcan', 'prican']).alias('prihistcan'),
                            ])
    data = data.with_columns(pl.coalesce(primary_sec))
    data = data.drop(['prihistrow','prihistusa','prihistcan', 'prirow', 'priusa', 'prican'])
    return data
def read_crsp_mcti_and_ff_monthly():
    crsp_mcti = add_yr_month(pl.read_ipc('Raw data/crsp_mcti_t30ret.ft'), 'caldt')
    ff_factors_monthly = add_yr_month(pl.read_ipc('Raw data/ff_factors_monthly.ft'), 'date')
    return crsp_mcti, ff_factors_monthly
def add_yr_month(df, datevar): 
    return df.with_columns([pl.col(datevar).dt.year().alias('yr'),pl.col(datevar).dt.month().alias('m')])
def prepare_crsp_sf(freq):
    __crsp_sf = pl.read_ipc(f'Raw data/__crsp_sf_{freq}.ft')
    __crsp_sf = __crsp_sf.with_columns(adj_trd_vol_NASDAQ('date', 'vol', 'exchcd', 3)).sort(['permno', 'date'])
    __crsp_sf = __crsp_sf.with_columns([
                                         (pl.col('prc').abs() * pl.col('vol')).alias('dolvol'),
                                         ((pl.col('ret') - pl.col('retx')) * pl.col('prc').shift(1) * (pl.col('cfacshr')/pl.col('cfacshr').shift(1))).alias('div_tot') 
                                        ])
    aux__crsp_sf = __crsp_sf.clone().unique('permno', keep = 'first').with_columns(pl.lit(None).cast(pl.Float64).alias('div_tot'))
    __crsp_sf = aux__crsp_sf.vstack(__crsp_sf).unique(['permno', 'date'], keep = 'first').sort(['permno', 'date'])
    del aux__crsp_sf
    crsp_sedelist = pl.read_ipc(f'Raw data/crsp_{freq}sedelist.ft')
    if freq == 'm':
        crsp_sedelist = add_yr_month(crsp_sedelist, 'dlstdt').drop('dlstdt')
        __crsp_sf = add_yr_month(__crsp_sf, 'date')
        __crsp_sf = __crsp_sf.join(crsp_sedelist, how = 'left', on = ['permno', 'yr','m']).drop(['yr', 'm'])
    else:
        __crsp_sf = __crsp_sf.with_columns([pl.col('permno').cast(pl.Int64).alias('permno')])
        __crsp_sf = __crsp_sf.join(crsp_sedelist, how = 'left', left_on = ['permno', 'date'], right_on = ['permno', 'dlstdt'])
    c1 = pl.col('dlret').is_null()
    c2 = pl.col('dlstcd') == 500
    c3 = pl.col('dlstcd') >= 520
    c4 = pl.col('dlstcd') <= 584
    c5 = (c1 & (c2 | (c3 & c4)))
    c6 = pl.col('ret').is_null()
    c7 = pl.col('dlret').is_not_null()
    c8 = (c6 & c7)
    __crsp_sf = __crsp_sf.with_columns([
                                          pl.when(c5).then(pl.lit(-0.3)).otherwise(pl.col('dlret')).alias('dlret'),
                                          pl.when(c8).then(pl.lit(0.)).otherwise(pl.col('ret')).alias('ret')
                                        ])
    __crsp_sf = __crsp_sf.with_columns((((pl.col('ret') + 1) * (pl.coalesce(['dlret', 0.]) + 1)) -1).alias('ret')).drop(['dlret', 'dlstcd'])
    crsp_mcti, ff_factors_monthly = read_crsp_mcti_and_ff_monthly()
    ff_factors_monthly = ff_factors_monthly.drop('date') 
    scale = 1 if (freq == 'm') else 21
    __crsp_sf = add_yr_month(__crsp_sf, 'date')
    __crsp_sf = __crsp_sf.join(crsp_mcti, how = 'left', on = ['yr', 'm']).join(ff_factors_monthly, how = 'left', on = ['yr', 'm']).drop(['yr', 'm', 'caldt'])
    __crsp_sf = __crsp_sf.with_columns(((pl.col('ret') - pl.coalesce(['t30ret','rf']))/scale).alias('ret_exc')).drop(['rf', 't30ret'])
    aux__crsp_sf =  __crsp_sf.group_by(['permco','date']).agg(pl.sum('me').alias('me_company'))
    __crsp_sf = __crsp_sf.join(aux__crsp_sf, how = 'left', on = ['permco','date'])
    del aux__crsp_sf
    if freq == 'm':__crsp_sf = __crsp_sf.with_columns([(pl.col('vol') * 100).alias('vol'), (pl.col('dolvol') * 100).alias('dolvol')])
    else: pass
    return __crsp_sf.unique(['permno', 'date']).sort(['permno', 'date'])
def gen_ret_lag_dif(freq):
    if freq == 'd':
        ret_lag_dif = (pl.col('datadate')-pl.col('datadate').shift(1)).alias('ret_lag_dif')
        ret_lag_dif_cast_type = pl.lit(None).cast(pl.Duration(time_unit = 'ms')).alias('ret_lag_dif')
    if freq == 'm':
        months_date = pl.col('datadate').dt.year()*12 + pl.col('datadate').dt.month()
        months_date_lag = pl.col('datadate').shift(1).dt.year()*12 + pl.col('datadate').shift(1).dt.month()
        ret_lag_dif = (months_date - months_date_lag).alias('ret_lag_dif')
        ret_lag_dif_cast_type = pl.lit(None).cast(pl.Int64).alias('ret_lag_dif')
    return ret_lag_dif, ret_lag_dif_cast_type
def prepare_comp_sf(freq):
    __firm_shares1 = pl.read_ipc('Raw data/__firm_shares1.ft')
    __firm_shares2 = populate_own(__firm_shares1, 'gvkey', 'datadate', 'ddate')
    del __firm_shares1
    __comp_dsf_na = pl.read_ipc('Raw data/__comp_dsf_na.ft')
    __comp_dsf_na = __comp_dsf_na.with_columns((pl.col('cshoc')/1e6).alias('cshoc'))
    __comp_dsf_na = __comp_dsf_na.join(__firm_shares2, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'ddate'])
    __comp_dsf_na = __comp_dsf_na.with_columns(pl.col('cshoc').fill_null(pl.col('csho_fund')*pl.col('ajex_fund')/pl.col('ajexdi')))
    __comp_dsf_na = __comp_dsf_na.with_columns(adj_trd_vol_NASDAQ('datadate', 'cshtrd', 'exchg', 14))
    __comp_dsf_global = pl.read_ipc('Raw data/__comp_dsf_global.ft')
    __comp_dsf = gen_comp_dsf(__comp_dsf_na,__comp_dsf_global).drop('datadate_right')
    del __comp_dsf_na, __comp_dsf_global
    if freq == 'd':
        del __firm_shares2
        base = __comp_dsf
        __comp_sf1 = __comp_dsf.unique(['gvkey','iid','datadate']).clone()
        __comp_dsf_res = process_comp_sf1(base, __comp_sf1, freq)
        del base, __comp_sf1, __comp_dsf
        res = {'d' : __comp_dsf_res}
    if freq == 'm':
        base = gen_comp_msf(__comp_dsf, __firm_shares2)
        del __firm_shares2, __comp_dsf
        __comp_sf1 = base.clone().unique(['gvkey','iid','datadate'])
        __comp_msf_res = process_comp_sf1(base, __comp_sf1, freq)
        del base, __comp_sf1
        res = {'m' : __comp_msf_res}
    if freq == 'both':
        freq = 'd'
        base = __comp_dsf
        __comp_sf1 = __comp_dsf.unique(['gvkey','iid','datadate']).clone()
        __comp_dsf_res = process_comp_sf1(base, __comp_sf1, freq)
        freq = 'm'
        base = gen_comp_msf(__comp_dsf, __firm_shares2)
        del __comp_dsf, __firm_shares2
        __comp_sf1 = base.clone().unique(['gvkey','iid','datadate'])
        __comp_msf_res = process_comp_sf1(base, __comp_sf1, freq)
        del base, __comp_sf1
        res = {'m' : __comp_msf_res, 'd' : __comp_dsf_res}
    return res
def process_comp_sf1(base, __comp_sf1, freq):
    ret_lag_dif, ret_lag_dif_cast_type = gen_ret_lag_dif(freq)
    __returns = __comp_sf1.select([pl.col('gvkey'), pl.col('iid'), pl.col('datadate'), pl.col('ri'), pl.col('ri_local'), pl.col('prcstd'),pl.col('curcdd')]).clone().sort(['gvkey', 'iid', 'datadate'])
    del __comp_sf1
    __returns = __returns.filter((pl.col('ri').is_not_null()) & (pl.col('prcstd').is_in([3, 4, 10]))).clone()
    __returns = __returns.with_columns([pl.col('ri').pct_change().alias('ret'), pl.col('ri_local').pct_change().alias('ret_local'), ret_lag_dif, pl.col('curcdd').shift(1).alias('lag_curcdd'), pl.lit(0).alias('first_flag')])
    __returns_aux = __returns.clone().unique(['gvkey', 'iid'], keep = 'first')
    __returns_aux = __returns_aux.with_columns([pl.lit(None).cast(pl.Float64).alias('ret'),pl.lit(None).cast(pl.Float64).alias('ret_local'), ret_lag_dif_cast_type, pl.lit(1).alias('first_flag')])
    __returns = __returns_aux.vstack(__returns).unique(['gvkey', 'iid','datadate'], keep = 'first')
    del __returns_aux
    c1 = (pl.col('first_flag') == 0)
    c2 = (pl.col('curcdd') != pl.col('lag_curcdd'))
    __returns = __returns.with_columns(pl.when(c1 & c2).then(pl.col('ret')).otherwise(pl.col('ret_local')).alias('ret_local')).sort(['gvkey', 'iid', 'datadate']).drop(['lag_curcdd','first_flag','curcdd','prcstd','ri','ri_local'])
    __delist = __returns.filter((pl.col('ret_local').is_not_null()) & (pl.col('ret_local') != 0)).clone().sort(['gvkey', 'iid', 'datadate']).unique(['gvkey', 'iid'], keep = 'last').clone()
    __sec_info = pl.read_ipc('Raw data/__sec_info.ft')
    __delist = __delist[['gvkey', 'iid', 'datadate']].join(__sec_info, how = 'left', on = ['gvkey', 'iid']).rename({'datadate': 'date_delist'})
    __delist = __delist.filter(pl.col('secstat') == 'I').with_columns(pl.when(pl.col('dlrsni').is_in(['02', '03'])).then(pl.lit(-0.3)).otherwise(pl.lit(0)).alias('dlret'))
    __comp_sf2 = base.join(__returns[['gvkey','iid','datadate','ret','ret_local','ret_lag_dif']], how = 'left', on = ['gvkey','iid', 'datadate'])
    __comp_sf2 = __comp_sf2.join(__delist[['gvkey','iid','date_delist','dlret']], how = 'left', on = ['gvkey', 'iid'])
    __comp_sf2 = __comp_sf2.filter((pl.col('datadate') <= pl.col('date_delist')) | (pl.col('date_delist').is_null()))
    __comp_sf2 = __comp_sf2.with_columns([
                                            pl.when(pl.col('datadate') == pl.col('date_delist')).then((1+pl.col('ret'))*(1+pl.col('dlret')) - 1).otherwise(pl.col('ret')).alias('ret'),
                                            pl.when(pl.col('datadate') == pl.col('date_delist')).then((1+pl.col('ret_local'))*(1+pl.col('dlret')) - 1).otherwise(pl.col('ret_local')).alias('ret_local')
                                            ]).drop(['ri', 'ri_local', 'date_delist', 'dlret'])
    crsp_mcti, ff_factors_monthly = read_crsp_mcti_and_ff_monthly()
    __comp_sf2 = add_yr_month(__comp_sf2, 'datadate')
    scale = 1 if (freq == 'm') else 21
    __comp_sf2 = __comp_sf2.join(crsp_mcti, how = 'left', on = ['yr', 'm']).join(ff_factors_monthly, how = 'left', on = ['yr', 'm']).drop(['date','caldt'])
    __comp_sf2 = __comp_sf2.with_columns(((pl.col('ret') - pl.coalesce(['t30ret','rf']))/scale).alias('ret_exc')).drop(['m', 'yr', 'rf', 't30ret'])
    __exchanges = comp_exchanges()
    __comp_sf2 = __comp_sf2.with_columns(pl.col('exchg').cast(pl.Int64))
    __comp_sf2 = __comp_sf2.join(__exchanges, how = 'left', on = ['exchg'])
    __comp_sf2 = add_primary_sec(__comp_sf2, 'datadate').unique(['gvkey', 'iid', 'datadate'])
    return __comp_sf2

comp_sf = prepare_comp_sf('both')
comp_sf['m'].write_ipc('comp_msf.ft')
comp_sf['d'].write_ipc('comp_dsf.ft')
del comp_sf
crsp_msf = prepare_crsp_sf('m')
crsp_dsf = prepare_crsp_sf('d')
crsp_msf.write_ipc('crsp_msf.ft')
crsp_dsf.write_ipc('crsp_dsf.ft')
del crsp_msf, crsp_dsf
