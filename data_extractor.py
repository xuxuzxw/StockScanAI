# data_extractor.py
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# 导入项目模块
import data
from logger_config import log

# 定义缓存文件存储路径
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(trade_date: str) -> str:
    """生成基于日期的缓存文件名"""
    return os.path.join(CACHE_DIR, f"raw_data_{trade_date}.h5")

def extract_data_for_date(trade_date: str):
    """
    步骤一：执行所有耗时的数据抽取和预处理工作，并将结果存入本地HDF5缓存文件。
    """
    log.info(f"===== 开始执行 {trade_date} 的数据抽取任务 =====")
    start_time = time.time()
    
    cache_file = get_cache_filename(trade_date)
    if os.path.exists(cache_file):
        log.info(f"检测到缓存文件 {cache_file} 已存在，跳过数据抽取。")
        return

    dm = data.DataManager()
    stock_list = dm.get_stock_basic()
    ts_codes = stock_list['ts_code'].tolist()

    # --- 1. 批量获取截面数据 ---
    log.info(f"  正在获取 {trade_date} 的全市场每日指标...")
    daily_basics_df = dm.pro.daily_basic(trade_date=trade_date)

    log.info(f"  正在获取 {trade_date} 的全市场资金流数据...")
    money_flow_df = dm.pro.moneyflow(trade_date=trade_date)

    # --- 2. 循环获取时序数据 ---
    log.info("  开始循环获取各股票的历史价格...")
    start_date_lookback = (pd.to_datetime(trade_date) - timedelta(days=90)).strftime('%Y%m%d')
    prices_dict = {}
    for i, code in enumerate(ts_codes):
        if (i + 1) % 100 == 0:
            log.info(f"    价格获取进度: {i+1}/{len(ts_codes)}")
        prices = dm.get_daily(code, start_date_lookback, trade_date)
        if prices is not None and not prices.empty:
            prices_dict[code] = prices.set_index('trade_date')['close']
    daily_prices_df = pd.DataFrame(prices_dict)
    daily_prices_df.index = pd.to_datetime(daily_prices_df.index)

    # --- 3. 循环获取财务数据 ---
    log.info("  开始循环获取所有股票的财务数据...")
    all_fina_data_cache = {}
    for i, code in enumerate(ts_codes):
        if (i + 1) % 100 == 0:
            log.info(f"    财务数据获取进度: {i+1}/{len(ts_codes)}")
        fina_df = dm.get_fina_indicator(ts_code=code)
        if fina_df is not None and not fina_df.empty:
            all_fina_data_cache[code] = fina_df
    
    # --- 4. 将所有数据存入HDF5文件 ---
    log.info(f"所有数据获取完毕，正在写入缓存文件: {cache_file}")
    with pd.HDFStore(cache_file, mode='w', complevel=9) as store:
        store.put('stock_list', stock_list, format='table')
        store.put('daily_prices', daily_prices_df, format='table')
        if daily_basics_df is not None:
            store.put('daily_basics', daily_basics_df, format='table')
        if money_flow_df is not None:
            store.put('money_flow', money_flow_df, format='table')
        # HDF5不支持字典直接存储，我们将其合并为一个大的DataFrame
        if all_fina_data_cache:
            all_fina_df = pd.concat(all_fina_data_cache, names=['ts_code']).reset_index(level=1, drop=True)
            store.put('all_fina', all_fina_df, format='table')
            
    duration = time.time() - start_time
    log.info(f"===== {trade_date} 的数据抽取任务完成！缓存文件已生成。总耗时: {duration:.2f} 秒。=====")

if __name__ == '__main__':
    dm = data.DataManager()
    cal_df = dm.pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
    latest_trade_date = cal_df[cal_df['is_open'] == 1]['cal_date'].max()
    
    extract_data_for_date(latest_trade_date)