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
import quant_engine
from logger_config import log
from data_extractor import get_cache_filename # 导入缓存文件名函数

# ... (FACTORS_TO_CALCULATE 列表保持不变)
FACTORS_TO_CALCULATE = [
    # 基本面
    'pe_ttm', 'roe', 'growth_revenue_yoy', 'debt_to_assets',
    # 技术面
    'momentum', 'volatility',
    # 资金面
    'net_inflow_ratio',
    # 筹码面 (V2.1新增)
    'holder_num_change_ratio', 'major_shareholder_net_buy_ratio', 'top_list_net_buy_amount',
    # 价值回报 (V2.1新增)
    'dividend_yield', 'forecast_growth_rate', 'repurchase_ratio',
    # 交易行为 (V2.1新增)
    'block_trade_ratio'
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
        self.top_list_df = None
        self.block_trade_df = None

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
            if 'top_list' in store.keys():
                self.top_list_df = store.get('top_list')
            if 'block_trade' in store.keys():
                self.block_trade_df = store.get('block_trade')
        
        self.ts_codes = self.stock_list['ts_code'].tolist()
        log.info("【数据加载】所有数据从缓存加载完毕。")
        return True

    def calculate_factors(self) -> pd.DataFrame:
        """【V2.1重构】步骤2: 统一化、自动化计算所有因子。"""
        if not self._load_data_from_cache():
            return pd.DataFrame()
            
        log.info("【因子计算】开始统一化计算所有因子...")
        ff = quant_engine.FactorFactory(_data_manager=self.dm)
        
        # 准备一个包含所有计算所需参数的字典
        params = {
            'date': self.trade_date,
            'start_date': (self.trade_date_dt - pd.Timedelta(days=90)).strftime('%Y%m%d'),
            'end_date': self.trade_date,
            'top_list_df': self.top_list_df,
            'block_trade_df': self.block_trade_df,
            # 未来可以加入更多全局参数...
        }

        # 统一的循环计算逻辑
        all_factor_data = {}
        for factor in FACTORS_TO_CALCULATE:
            log.info(f"  正在计算因子: {factor}...")
            # 为每个因子准备一个Series来存放结果
            factor_series = pd.Series(index=self.ts_codes, dtype=float)
            for code in self.ts_codes:
                params['ts_code'] = code
                factor_series[code] = ff.calculate(factor, **params)
            
            all_factor_data[factor] = factor_series

        # --- 整合结果 ---
        # （注意：之前的向量化部分被这个更通用的循环取代了，虽然部分计算效率可能略降，但可维护性大幅提升）
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