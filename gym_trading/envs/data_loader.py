# coding: utf-8
import os
import logging
from datetime import datetime, timedelta, date
import pandas as pd
import pandas_datareader.data as web
import tushare as ts

logger = logging.getLogger(__name__)


def cached_filename(name, date):
    return '{name}_{d}_data.pkl.gz'.format(name=name, d=date)


def get_all_hist_data(code, stock_basics, year_interval=3):
    def format_date(d):
        return d.strftime('%Y-%m-%d')

    proxies = ts.get_proxies(count=10)
    now_date = datetime.now().date()
    # 获取上市时间
    ipo_date = datetime.strptime(str(stock_basics.loc[code]['timeToMarket']), '%Y%m%d').date() \
        if stock_basics.loc[code]['timeToMarket'] else date(2000, 1, 1)
    start_date = ipo_date
    end_date = date(start_date.year+year_interval, 1, 1)
    data_frames = []
    while now_date >= start_date:
        try:
            batch_df = ts.get_h_data(
                code, start=format_date(start_date), end=format_date(end_date), proxies=proxies
            )
        except:
            continue
        data_frames.append(batch_df)
        start_date = end_date
        end_date = date(start_date.year+year_interval, 1, 1)
    return pd.concat(data_frames)


def get_data_tushare(code):
    stock_basics = ts.get_stock_basics()  # 获取股票基本情况
    return get_all_hist_data(code, stock_basics)


def data_loader(name, cache_days=10, compression='gzip'):
    df = None
    cache_dir = './tmp_data/'
    try:
        os.mkdir(cache_dir)
        logger.info('tmp data dir: {p} created'.format(p=cache_dir))
    except OSError:
        logger.info('tmp data dir: {p} existed'.format(p=cache_dir))
    now_date = datetime.now().date()
    for i in range(cache_days):
        target_file = cached_filename(name, now_date+timedelta(days=i))
        cache_path = os.path.join(cache_dir, target_file)
        if os.path.exists(cache_path):
            df = pd.read_pickle(cache_path, compression=compression)
    if df is None:
        df = web.DataReader(name, 'yahoo', start=datetime(2012,8,29), end=datetime.now())
        # df = get_data_tushare(name)
        # cache data
        df.to_pickle(cached_filename(name, now_date), compression)
    return df
