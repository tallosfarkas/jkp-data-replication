import polars as pl
import wrds_Fernando
wrds_session = wrds_Fernando.Connection(wrds_username= "jkpfactors_user", wrds_password="jkpfactors_password")
comp_cond = "indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'"
__firm_shares1 = wrds_session.raw_sql_polars_uri(f"""
            SELECT gvkey, datadate, cshoq AS csho_fund, ajexq AS ajex_fund
            FROM comp.fundq WHERE {comp_cond} AND cshoq IS NOT NULL AND ajexq IS NOT NULL
            UNION ALL
            SELECT gvkey, datadate, csho AS csho_fund, ajex AS ajex_fund
            FROM comp.funda WHERE {comp_cond} AND csho IS NOT NULL AND ajex IS NOT NULL
            """)
__firm_shares1 = __firm_shares1.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__firm_shares1.write_ipc('Raw data/__firm_shares1.ft')
del __firm_shares1, comp_cond
__comp_dsf_na = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, iid, datadate, tpci, exchg, prcstd, curcdd, prccd AS prc_local, ajexdi,
            CASE WHEN prcstd!=5 then prchd ELSE NULL END AS prc_high_lcl,
            CASE WHEN prcstd!=5 then prcld ELSE NULL END AS prc_low_lcl,
            cshtrd, cshoc,
            (prccd/ajexdi*trfd) AS ri_local,
            curcddv, div, divd, divsp
            FROM comp.secd
             """)
__comp_dsf_na = __comp_dsf_na.with_columns(pl.col("gvkey").cast(pl.Int64))
__comp_dsf_na.write_ipc('Raw data/__comp_dsf_na.ft')
del __comp_dsf_na
__comp_dsf_global = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, iid, datadate, tpci, exchg, prcstd, curcdd, 
            prccd/qunit AS prc_local, ajexdi, cshoc/1e6 AS cshoc,
            CASE WHEN prcstd!=5 then prchd/qunit ELSE NULL END AS prc_high_lcl,
            CASE WHEN prcstd!=5 then prcld/qunit ELSE NULL END AS prc_low_lcl,
            cshtrd, ((prccd/qunit)/ajexdi*trfd) AS ri_local,
            curcddv, div, divd, divsp
            FROM comp.g_secd
             """)
__comp_dsf_global = __comp_dsf_global.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__comp_dsf_global.write_ipc('Raw data/__comp_dsf_global.ft')
del __comp_dsf_global
__fx1 = wrds_session.raw_sql_polars_uri("""
            SELECT DISTINCT a.tocurd AS curcdd, a.datadate,  b.exratd/a.exratd AS fx
            FROM comp.exrt_dly a, comp.exrt_dly b
            WHERE a.fromcurd = 'GBP' AND b.tocurd = 'USD'
            AND a.fromcurd = b.fromcurd AND a.datadate = b.datadate
             """)
__fx1.write_ipc('Raw data/__fx1.ft')
del __fx1
__comp_secm1 = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, iid, datadate, tpci, exchg, curcdm AS curcdd,
            prccm AS prc_local, prchm AS prc_high, prclm AS prc_low, ajexm AS ajexdi, 
            cshom, csfsm, cshoq, ajexm, dvpsxm, cshtrm, curcddvm, prccm/ajexm*trfm AS ri_local
            FROM comp.secm
             """)
__comp_secm1 = __comp_secm1.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__comp_secm1.write_ipc('Raw data/__comp_secm1.ft')
del __comp_secm1
crsp_mcti_t30ret = wrds_session.raw_sql_polars_uri("""
            SELECT caldt, t30ret
            FROM crsp.mcti
             """)
crsp_mcti_t30ret.write_ipc('Raw data/crsp_mcti_t30ret.ft')
del crsp_mcti_t30ret
ff_factors_monthly = wrds_session.raw_sql_polars_uri("""
            SELECT date, rf 
            FROM ff.factors_monthly
             """)
ff_factors_monthly.write_ipc('Raw data/ff_factors_monthly.ft')
del ff_factors_monthly
__ex_country1 = wrds_session.raw_sql_polars_uri("""
            SELECT DISTINCT exchg, excntry
            FROM comp.g_security
            UNION ALL
            SELECT DISTINCT exchg, excntry
            FROM comp.security
             """)
__ex_country1.write_ipc('Raw data/__ex_country1.ft')
del __ex_country1
comp_r_ex_codes = wrds_session.raw_sql_polars_uri("""
            SELECT exchgdesc, exchgcd 
            FROM comp.r_ex_codes
             """)
comp_r_ex_codes.write_ipc('Raw data/comp_r_ex_codes.ft')
del comp_r_ex_codes
__prihistrow = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, itemvalue as prihistrow, effdate, thrudate
        	FROM comp.g_sec_history WHERE item = 'PRIHISTROW'
             """)
__prihistrow = __prihistrow.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__prihistrow.write_ipc('Raw data/__prihistrow.ft')
del __prihistrow
__prihistusa = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, itemvalue as prihistusa, effdate, thrudate
        	FROM comp.sec_history WHERE item = 'PRIHISTUSA'
             """)
__prihistusa = __prihistusa.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__prihistusa.write_ipc('Raw data/__prihistusa.ft')
del __prihistusa
__prihistcan = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, itemvalue as prihistcan, effdate, thrudate
        	FROM comp.sec_history WHERE item = 'PRIHISTCAN'
             """)
__prihistcan = __prihistcan.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__prihistcan.write_ipc('Raw data/__prihistcan.ft')
del __prihistcan
__header = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, prirow, priusa, prican
            FROM comp.company
    		UNION ALL
    		SELECT gvkey, prirow, priusa, prican
            FROM comp.g_company
             """)
__header = __header.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__header.write_ipc('Raw data/__header.ft')
del __header
__crsp_sf_m = wrds_session.raw_sql_polars_uri("""
            SELECT a.permno, a.permco, a.date, (a.prc < 0) AS bidask, abs(a.prc) AS prc, a.shrout/1000 AS shrout, abs(a.prc) * a.shrout/1000 AS me,
                a.ret, a.retx, a.cfacshr, a.vol, 
                CASE WHEN a.prc > 0 AND a.askhi > 0 THEN a.askhi ELSE NULL END AS prc_high,
                CASE WHEN a.prc > 0 AND a.bidlo > 0 THEN a.bidlo ELSE NULL END AS prc_low,
                b.shrcd, b.exchcd, c.gvkey, c.liid AS iid,
                b.exchcd in (1, 2, 3) AS exch_main			
            FROM crsp.msf AS a 
            LEFT JOIN crsp.msenames AS b
                ON a.permno=b.permno AND a.date>=namedt AND a.date<=b.nameendt
            LEFT JOIN crsp.ccmxpf_lnkhist AS c
                ON a.permno=c.lpermno AND (a.date>=c.linkdt OR c.linkdt IS NULL) AND 
                (a.date<=c.linkenddt OR c.linkenddt IS NULL) AND c.linktype in ('LC', 'LU', 'LS')
            """)
__crsp_sf_m = __crsp_sf_m.with_columns([
                                        pl.col('gvkey').cast(pl.Int64).alias('gvkey'),
                                        pl.col('permno').cast(pl.Int64).alias('permno'),
                                        pl.col('permco').cast(pl.Int64).alias('permco')
                                       ])
__crsp_sf_m.write_ipc('Raw data/__crsp_sf_m.ft')
del __crsp_sf_m
__crsp_sf_d = wrds_session.raw_sql_polars_uri("""
            SELECT a.permno, a.permco, a.date, (a.prc < 0) AS bidask, abs(a.prc) AS prc, a.shrout/1000 AS shrout, abs(a.prc) * a.shrout/1000 AS me,
                a.ret, a.retx, a.cfacshr, a.vol, 
                CASE WHEN a.prc > 0 AND a.askhi > 0 THEN a.askhi ELSE NULL END AS prc_high,
                CASE WHEN a.prc > 0 AND a.bidlo > 0 THEN a.bidlo ELSE NULL END AS prc_low,
                b.shrcd, b.exchcd, c.gvkey, c.liid AS iid,
                b.exchcd in (1, 2, 3) AS exch_main			
            FROM crsp.dsf AS a 
            LEFT JOIN crsp.dsenames AS b
                ON a.permno=b.permno AND a.date>=namedt AND a.date<=b.nameendt
            LEFT JOIN crsp.ccmxpf_lnkhist AS c
                ON a.permno=c.lpermno AND (a.date>=c.linkdt OR c.linkdt IS NULL) AND 
                (a.date<=c.linkenddt OR c.linkenddt IS NULL) AND c.linktype in ('LC', 'LU', 'LS')
            """)
__crsp_sf_d = __crsp_sf_d.with_columns([
                                        pl.col('gvkey').cast(pl.Int64).alias('gvkey'),
                                        pl.col('permno').cast(pl.Int64).alias('permno'),
                                        pl.col('permco').cast(pl.Int64).alias('permco')
                                        ])
__crsp_sf_d.write_ipc('Raw data/__crsp_sf_d.ft')
del __crsp_sf_d
crsp_dsedelist = wrds_session.raw_sql_polars_uri("""
            SELECT dlret, dlstcd, permno, dlstdt
            FROM crsp.dsedelist
            """)
crsp_dsedelist = crsp_dsedelist.with_columns([pl.col('permno').cast(pl.Int64).alias('permno')])
crsp_dsedelist.write_ipc('Raw data/crsp_dsedelist.ft')
del crsp_dsedelist
crsp_msedelist = wrds_session.raw_sql_polars_uri("""
            SELECT dlret, dlstcd, permno, dlstdt
            FROM crsp.msedelist
            """)
crsp_msedelist = crsp_msedelist.with_columns([pl.col('permno').cast(pl.Int64).alias('permno')])
crsp_msedelist.write_ipc('Raw data/crsp_msedelist.ft')
del crsp_msedelist
__sec_info = wrds_session.raw_sql_polars_uri("""
            SELECT gvkey, iid, secstat, dlrsni
            FROM comp.security
            UNION ALL
            SELECT gvkey, iid, secstat, dlrsni
            FROM comp.g_security
            """)
__sec_info = __sec_info.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
__sec_info.write_ipc('Raw data/__sec_info.ft')
del __sec_info