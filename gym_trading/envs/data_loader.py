# coding: utf-8
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import pandas_datareader as pdr

logger = logging.getLogger(__name__)


def cached_filename(name, date):
    return '{name}_{d}_data.pkl.gz'.format(name=name, d=date)


def data_loader(name, cache_days=3, compression='gzip'):
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
        df = pdr.get_data_yahoo(name)
        # cache data
        df.to_pickle(cache_path, compression)
    return df
