# factor_calculator.py
# 【V4.0 架构分离版】
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sqlalchemy import text
import os

# 导入项目模块
import data
from logger_config import log
from data_extractor import get_cache_filename # 导入缓存文件名函数

# ... (FACTORS_TO_CALCULATE 列表保持不变)
FACTORS_TO_CALCULATE = [
    'pe_ttm', 'roe', 'growth_revenue_yoy', 'debt_to_assets',
    'momentum', 'volatility', 'net_inflow_ratio'
]

class FactorCalculatorFromCache:
    """
    职责分离后的因子计算器，只从本地HDF5缓存文件读取数据并进行计算。
    """
    def __init__(self, trade_date: str):
        self.trade_date = trade_date
        self.trade_date_dt = pd.to_datetime(trade_date)
        self.cache_file = get_cache_filename(trade_date)
        
        self.dm = data.DataManager()
        
        # 待加载的数据
        self.stock_list = None
        self.daily_prices_df = None
        self.daily_basics_df = None
        self.money_flow_df = None
        self.all_fina_df = None

    def _load_data_from_cache(self) -> bool:
        """步骤1: 从本地HDF5文件加载数据，速度极快。"""
        log.info(f"【数据加载】正在从缓存文件 {self.cache_file} 加载数据...")
        if not os.path.exists(self.cache_file):
            log.error(f"错误：缓存文件不存在！请先运行 data_extractor.py 来生成当日数据。")
            return False
        
        with pd.HDFStore(self.cache_file, mode='r') as store:
            self.stock_list = store.get('stock_list')
            self.daily_prices_df = store.get('daily_prices')
            self.daily_basics_df = store.get('daily_basics')
            self.money_flow_df = store.get('money_flow')
            self.all_fina_df = store.get('all_fina')
        
        self.ts_codes = self.stock_list['ts_code'].tolist()
        log.info("【数据加载】所有数据从缓存加载完毕。")
        return True

    def calculate_factors(self) -> pd.DataFrame:
        """步骤2: 在内存中进行向量化计算。"""
        if not self._load_data_from_cache():
            return pd.DataFrame()
            
        log.info("【因子计算】开始向量化计算所有因子...")
        # (这部分计算逻辑与上一个版本完全相同)
        results = {}
        # ... (省略与上一版完全相同的向量化计算代码) ...
        # --- 向量化计算：价格类因子 (动量, 波动率) ---
        if self.daily_prices_df is not None and not self.daily_prices_df.empty:
            log.info("  正在计算价格类因子...")
            prices = self.daily_prices_df.ffill()
            if len(prices) >= 21:
                results['momentum'] = prices.iloc[-1] / prices.iloc[-21] - 1
            if len(prices) >= 20:
                log_returns = np.log(prices / prices.shift(1))
                results['volatility'] = log_returns.iloc[-20:].std() * np.sqrt(252)

        # --- 向量化计算：基于当日截面数据的因子 ---
        log.info("  正在计算截面类因子...")
        if self.daily_basics_df is not None and not self.daily_basics_df.empty:
            basics = self.daily_basics_df.set_index('ts_code')
            results['pe_ttm'] = basics['pe_ttm']
            
            if self.money_flow_df is not None and not self.money_flow_df.empty:
                flow = self.money_flow_df.set_index('ts_code')
                combined = basics.join(flow, how='inner')
                amount_yuan = combined['amount'] * 1000
                net_inflow_yuan = (combined['buy_lg_amount'] - combined['sell_lg_amount']) * 10000
                results['net_inflow_ratio'] = net_inflow_yuan.divide(amount_yuan).fillna(0)
        
        # --- 循环计算（已优化）：财务类因子 ---
        log.info("  正在计算财务类因子...")
        fina_factors = {
            'roe': {}, 'growth_revenue_yoy': {}, 'debt_to_assets': {}
        }
        if self.all_fina_df is not None:
            for code in self.ts_codes:
                all_fina = self.all_fina_df.loc[[code]]
                if all_fina.empty:
                    continue
                
                pit_fina_row = self.dm.get_pit_financial_data(all_fina, as_of_date=self.trade_date)
                if pit_fina_row is not None and not pit_fina_row.empty:
                    row = pit_fina_row.iloc[0]
                    fina_factors['roe'][code] = row.get('roe')
                    fina_factors['growth_revenue_yoy'][code] = row.get('or_yoy', row.get('netprofit_yoy'))
                    fina_factors['debt_to_assets'][code] = row.get('debt_to_assets')
        
        for name, data_dict in fina_factors.items():
            results[name] = pd.Series(data_dict)

        # --- 整合结果 ---
        final_df = pd.DataFrame(results).reset_index().rename(columns={'index': 'ts_code'})
        long_df = final_df.melt(id_vars='ts_code', var_name='factor_name', value_name='factor_value').dropna()
        long_df['trade_date'] = self.trade_date
        
        log.info("【因子计算】所有因子计算完毕。")
        return long_df

    def save_to_db(self, final_df: pd.DataFrame):
        # (这部分入库逻辑与上一版完全相同)
        log.info("【数据入库】开始存储因子数据...")
        if final_df.empty:
            log.warning("没有有效的因子数据可以存入数据库。")
            return
        # ... (省略与上一版完全相同的入库代码) ...
        final_df['trade_date'] = pd.to_datetime(final_df['trade_date'])
        try:
            with self.dm.engine.connect() as connection:
                with connection.begin():
                    delete_sql = text("DELETE FROM factors_exposure WHERE trade_date = :trade_date")
                    connection.execute(delete_sql, {'trade_date': self.trade_date})
                    final_df.to_sql('factors_exposure', connection, if_exists='append', index=False, chunksize=10000)
            log.info(f"成功为 {self.trade_date} 写入 {len(final_df)} 条因子数据。")
        except Exception as e:
            log.critical(f"因子数据写入数据库时发生严重错误: {e}", exc_info=True)


def run_daily_calculation(trade_date: str):
    """
    执行每日因子计算的主工作流。
    """
    log.info(f"===== 开始执行 {trade_date} 的因子计算任务 (从缓存加载) =====")
    start_time = time.time()
    
    # --- 前置检查机制 (检查最终结果，而非缓存) ---
    dm = data.DataManager()
    try:
        with dm.engine.connect() as connection:
            check_sql = text("SELECT 1 FROM factors_exposure WHERE trade_date = :trade_date LIMIT 1")
            result = connection.execute(check_sql, {'trade_date': trade_date}).scalar_one_or_none()
        if result == 1:
            log.info(f"检测到数据库中已存在 {trade_date} 的因子数据。任务无需重复执行。")
            return
    except Exception as e:
        log.error(f"在执行前置检查时发生数据库错误: {e}", exc_info=True)
        return
        
    calculator = FactorCalculatorFromCache(trade_date=trade_date)
    final_df = calculator.calculate_factors()
    
    if not final_df.empty:
        calculator.save_to_db(final_df)
    
    duration = time.time() - start_time
    log.info(f"===== {trade_date} 的因子计算任务完成！总耗时: {duration:.2f} 秒。=====")

if __name__ == '__main__':
    dm = data.DataManager()
    cal_df = dm.pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
    latest_trade_date = cal_df[cal_df['is_open'] == 1]['cal_date'].max()
    
    # 新的工作流：先确保数据抽取完成，再进行计算
    log.info("--- 工作流步骤1: 检查并执行数据抽取 ---")
    # 导入并运行数据抽取器
    from data_extractor import extract_data_for_date
    extract_data_for_date(latest_trade_date)
    
    log.info("--- 工作流步骤2: 执行因子计算 ---")
    run_daily_calculation(latest_trade_date)