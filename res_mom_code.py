#Credit: Fernando Reyes
#Adapted from original SAS code by: Jensen, T., Kelly, B., and Pedersen, L. “Is There a Replication Crisis in Finance?” Journal of Finance (2023)
#References: OpenAI. (2024). ChatGPT (Mar 14 version) [Large language model]. Retrieved from https://chat.openai.com/chat
def regression_3vars(y, x1, x2, x3, __n, __min):
    den = (-((pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.col(x3).rolling_var(window_size=__n, min_periods=__min))) +
       (pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min))**2 +
       (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min))**2 -
       2 * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) +
       (pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))**2)
    beta1 = ((pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) -
         (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * ((pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min))) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) +
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) +
         (pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min))) / den
    beta2 = ((pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) -
         (pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * ((pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min))) +
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) +
         (pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))) / den
    beta3 = (-((pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min))) +
         (pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) +
         (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) +
         (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min))) / den
    alpha = pl.col(y).rolling_mean(window_size=__n, min_periods = __min) - beta1 * pl.col(x1).rolling_mean(window_size=__n, min_periods = __min) - beta2 * pl.col(x2).rolling_mean(window_size=__n, min_periods = __min) - beta3 * pl.col(x3).rolling_mean(window_size=__n, min_periods = __min)
    return alpha, beta1, beta2, beta3
def get_rolling_residuals(a,b1,b2,b3,x1,x2,x3,y,w):
    squared_res_sum = w*(pl.col(a)**2) + 2 * pl.col(a) * pl.col(b1) * pl.col(x1).rolling_sum(window_size=w) + (pl.col(b1)**2) * (pl.col(x1)**2).rolling_sum(window_size=w) + 2 * pl.col(a) * pl.col(b2)* pl.col(x2).rolling_sum(window_size=w) + 2 * pl.col(b1) * pl.col(b2) * (pl.col(x1) * pl.col(x2)).rolling_sum(window_size=w) + (pl.col(b2)**2) * (pl.col(x2)**2).rolling_sum(window_size=w) + 2 * pl.col(a) * pl.col(b3) * pl.col(x3).rolling_sum(window_size=w) + 2 * pl.col(b1) * pl.col(b3) * (pl.col(x1) * pl.col(x3)).rolling_sum(window_size=w) + 2 * pl.col(b2) * pl.col(b3) * (pl.col(x2) *  pl.col(x3)).rolling_sum(window_size=w) + (pl.col(b3)**2) * (pl.col(x3)**2).rolling_sum(window_size=w) - 2 * pl.col(a) * pl.col(y).rolling_sum(window_size=w) -  2 * pl.col(b1) * (pl.col(x1) * pl.col(y)).rolling_sum(window_size=w) - 2 * pl.col(b2) * (pl.col(x2) * pl.col(y)).rolling_sum(window_size=w) - 2 * pl.col(b3) * (pl.col(x3) * pl.col(y)).rolling_sum(window_size=w) + (pl.col(y)**2).rolling_sum(window_size=w)
    res_mean = (pl.col(y).rolling_mean(window_size=w) - pl.col(a) - pl.col(b1) * pl.col(x1).rolling_mean(window_size=w) - pl.col(b2) * pl.col(x2).rolling_mean(window_size=w) - pl.col(b3) * pl.col(x3).rolling_mean(window_size=w))
    var = (squared_res_sum - w*(res_mean**2))/(w-1)
    return (res_mean/(var**0.5))
def residual_momentum(output_path, data_path, fcts_path, __n, __min, incl, skip):
    w = incl - skip
    data = pl.scan_ipc(data_path)
    fcts = pl.scan_ipc(fcts_path).select(['excntry', 'eom','mktrf', 'hml', 'smb_ff'])
    c1 = pl.col('ret_local') != 0
    c2 = pl.col('ret_exc').is_not_null()
    c3 = pl.col('ret_lag_dif') == 1
    c4 = pl.col('mktrf').is_not_null()
    #Select depending on if it's eager or lazy mode
    #data = data.filter(c1 & c2 & c3).select(['excntry','id','eom','ret_exc'])
    data = data.filter(c1).filter(c2).filter(c3).select(['excntry','id','eom','ret_exc'])
    fcts = fcts.filter(c4).select(['excntry', 'eom','mktrf', 'hml', 'smb_ff'])
    __msf = data.join(fcts, how = 'left', on = ['excntry', 'eom']).drop('excntry')
    wins_data = winsorize_own(__msf,['eom'],'ret_exc',0.1/100,99.9/100)
    __msf = __msf.join(wins_data, how = 'left', on = 'eom')
    __msf = __msf.with_columns(ret_exc = pl.when(pl.col('ret_exc')<= pl.col('low')).then(pl.col('low')).when(pl.col('ret_exc')>= pl.col('high')).then(pl.col('high')).otherwise(pl.col('ret_exc'))).select(['id','eom','ret_exc','mktrf','hml','smb_ff'])
    __msf2 = __msf.select(['id','eom']).sort(['id','eom']).with_columns(aux = (pl.col('eom').shift(-1).over('id')).dt.offset_by(f'-1mo').dt.month_end())
    __msf2 = __msf2.with_columns(date_range = pl.date_ranges(start='eom', end='aux', interval='1mo')).with_columns(date_range = pl.coalesce(['date_range', 'eom'])).select(['id','date_range']).explode('date_range').rename({'date_range': 'eom'})
    __msf = __msf2.join(__msf, on = ['id','eom'], how = 'left')
    alpha, beta1, beta2, beta3 = regression_3vars('ret_exc','mktrf','smb_ff','hml',__n,__min)
    __msf = __msf.sort(['id','eom']).with_columns([alpha.over('id').alias('alpha'),beta1.over('id').alias('beta1'),beta2.over('id').alias('beta2'),beta3.over('id').alias('beta3')])
    residual_exp = lambda i:  pl.col('ret_exc').shift(i) - (pl.col('alpha')*pl.lit(1.) + pl.col('beta1') * pl.col('mktrf').shift(i) + pl.col('beta2') * pl.col('smb_ff').shift(i) + pl.col('beta3') * pl.col('hml').shift(i))
    __msf = __msf.sort(['id','eom']).with_columns(std_res = pl.concat_list([residual_exp(i).over(['id']) for i in range(1, w+1)]).list.drop_nulls()).select(['id','eom','std_res'])
    __msf = __msf.collect()
    __msf = __msf.with_columns(std_res =  pl.col('std_res').list.eval(pl.element().mean()/pl.element().std())).explode('std_res')
    __msf = __msf.select(['id','eom',pl.col('std_res').alias(f'resff3_{incl}_{skip}')])
    __msf.sort(['id','eom']).write_ipc(output_path + f'_{incl}_{skip}.ft')
    del __msf
