# quant_project/quant_engine.py
# --- 这是合并后的文件，整合了 quant_engine.py, quant_engine_v2.py, 和 quant_engine_v3.py 的所有功能 ---

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import cvxpy as cp
from queue import Queue, Empty
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
import sqlite3
from functools import lru_cache

# 导入日志记录器
from logger_config import log

# ############################################################################
# --- 模块来源: quant_engine.py (原始基础模块) ---
# ############################################################################

def run_main_workflow():
    """
    【源于 main.py】
    执行一个完整的、自动化的量化研究与回测流程。
    """
    log.info("========== 主控流程开始 ==========")

    try:
        # --- 1. 初始化核心组件 ---
        log.info("[步骤 1/5] 初始化所有管理器和引擎...")
        # 注意：此处需要一个在外部定义的 data_manager 实例
        # 为使其可独立运行，我们在 __main__ 部分进行实例化
        import data
        data_manager = data.DataManager()
        factor_factory = FactorFactory(_data_manager=data_manager)
        task_runner = AutomatedTasks(data_manager, factor_factory)
        fa = FactorAnalyzer(_data_manager=data_manager)

        # --- 2. 数据更新 ---
        log.info("[步骤 2/5] 执行每日数据更新任务...")
        update_status = task_runner.run_daily_data_update()
        log.info(f"数据更新完成，状态: {update_status}")

        # --- 3. 因子计算与分析 ---
        log.info("[步骤 3/5] 计算并分析示例因子（动量因子）...")
        stock_pool = ['000001.SZ', '600519.SH', '300750.SZ']
        # 此处仅为演示，计算一个截面的因子值
        date = '20250715' # 假设我们分析今天的因子
        start_date = '20250515'
        
        raw_values = {s: factor_factory.calc_momentum(s, start_date, date) for s in stock_pool}
        factor_series = pd.Series(raw_values, name=pd.to_datetime(date)).dropna()
        log.info(f"计算得到因子值:\n{factor_series}")

        ic, p_value = fa.calculate_ic(factor_series)
        log.info(f"因子在 {date} 的IC值为: {ic:.4f} (p-value: {p_value:.4f})")

        # --- 4. 策略执行与组合构建 ---
        # 简单策略：选择动量因子值最高的股票
        log.info("[步骤 4/5] 基于因子值构建投资组合...")
        if not factor_series.empty:
            target_stock = factor_series.idxmax()
            log.info(f"策略决定买入动量最高的股票: {target_stock}")
            # 在实际应用中，这里会调用组合优化器来生成权重
            portfolio_weights = pd.DataFrame({'weight': [1.0]}, index=[target_stock])
        else:
            log.warning("没有有效的因子值，无法构建投资组合。")
            portfolio_weights = pd.DataFrame()


        # --- 5. 执行回测（使用简化的向量化回测） ---
        log.info("[步骤 5/5] 对选股结果进行一个简单的回测...")
        if not portfolio_weights.empty:
            backtest_start = '20240101'
            backtest_end = '20250716'
            
            all_prices = pd.DataFrame()
            prices = data_manager.get_adjusted_daily(target_stock, backtest_start, backtest_end)
            if prices is not None and not prices.empty:
                all_prices[target_stock] = prices.set_index('trade_date')['close']
            
            # 向量化回测器需要一个历史因子表，此处简化
            if not all_prices.empty:
                # 简化为直接计算持有期收益，以演示概念。
                log.info(f"对单一目标股票 {target_stock} 进行简单的持有期收益计算...")
                returns = all_prices[target_stock].pct_change().dropna()
                cumulative_return = (1 + returns).prod() - 1
                log.info(f"从 {backtest_start} 到 {backtest_end} 的持有期累计收益率为: {cumulative_return:.2%}")
            else:
                log.warning("无法获取价格数据，跳过回测。")
        else:
            log.warning("无法执行回测，因为投资组合为空。")

    except Exception as e:
        log.critical("主控流程发生严重错误!", exc_info=True) # exc_info=True会记录完整的错误堆栈

    log.info("========== 主控流程结束 ==========")


class FactorFactory:
    """
    【增强版】模块化因子工厂
    - 基于复权数据计算技术指标。
    - 新增资金流、基本面、筹码结构等多维度因子。
    """
    def __init__(self, _data_manager):
        self.data_manager = _data_manager

    # --- A. 价格类因子 (基于复权数据) ---
    def calc_momentum(self, ts_code: str, start_date: str, end_date: str, window=20) -> float:
        """计算N日动量 (收益率)"""
        df = self.data_manager.get_adjusted_daily(ts_code, start_date, end_date, adj='hfq')
        if df is None or len(df) < window:
            return np.nan
        return df['close'].iloc[-1] / df['close'].iloc[-window] - 1

    def calc_volatility(self, ts_code: str, start_date: str, end_date: str, window=20) -> float:
        """计算N日年化波动率"""
        df = self.data_manager.get_adjusted_daily(ts_code, start_date, end_date, adj='hfq')
        if df is None or len(df) < window:
            return np.nan
        log_returns = np.log(df['close'] / df['close'].shift(1))
        return log_returns.iloc[-window:].std() * np.sqrt(252)

    # --- B. 资金类因子 ---
    def calc_net_inflow_ratio(self, ts_code: str, start_date: str, end_date: str, window=5) -> float:
        """计算N日净流入额占成交额比例"""
        df_flow = self.data_manager.get_moneyflow(ts_code, start_date, end_date)
        if df_flow is None or df_flow.empty or len(df_flow) < window:
            return np.nan
        
        recent_flow = df_flow.tail(window)
        net_mf_amount = recent_flow['net_mf_amount'].sum()
        trade_amount = recent_flow['buy_lg_amount'].sum() + recent_flow['sell_lg_amount'].sum() + recent_flow['buy_md_amount'].sum() + recent_flow['sell_md_amount'].sum() + recent_flow['buy_sm_amount'].sum() + recent_flow['sell_sm_amount'].sum()
        return net_mf_amount / trade_amount if trade_amount != 0 else 0

    def calc_north_hold_change(self, ts_code: str, start_date: str, end_date: str) -> float:
        """计算区间内北向资金持股比例(ratio)的变动"""
        df_hk = self.data_manager.get_hk_hold(ts_code, start_date, end_date)
        if df_hk is None or df_hk.empty or len(df_hk) < 2:
            return np.nan
        df_hk['trade_date'] = pd.to_datetime(df_hk['trade_date'])
        df_hk = df_hk.sort_values('trade_date')
        return df_hk['ratio'].iloc[-1] - df_hk['ratio'].iloc[0]

    # --- C. 基本面因子 ---
    def calc_roe(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.1重构-健壮性】获取指定日期可得的最新财报的ROE(摊薄)"""
        try:
            all_fina = self.data_manager.get_fina_indicator(ts_code)
            if all_fina is None or all_fina.empty:
                return np.nan

            latest_fina = self.data_manager.get_pit_financial_data(all_fina, as_of_date=date)
            if latest_fina is None or latest_fina.empty:
                # log.debug(f"在日期 {date} 未找到 {ts_code} 的有效PIT财务数据。")
                return np.nan
                
            return latest_fina['roe'].iloc[0]
        except (KeyError, IndexError) as e:
            log.warning(f"计算ROE因子 for {ts_code} on {date} 时出错: {e}")
            return np.nan

    def calc_pe_ttm(self, ts_code: str, date: str) -> float:
        """获取指定日期的PE TTM"""
        df_basic = self.data_manager.get_daily_basic(ts_code, date, date)
        if df_basic is None or df_basic.empty or 'pe_ttm' not in df_basic.columns:
            return np.nan
        return df_basic['pe_ttm'].iloc[0]

    def calc_growth_revenue_yoy(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.1重构-健壮性】获取指定日期可得的最新财报的营业收入同比增长率(%)"""
        try:
            all_fina = self.data_manager.get_fina_indicator(ts_code)
            if all_fina is None or all_fina.empty:
                return np.nan

            latest_fina = self.data_manager.get_pit_financial_data(all_fina, as_of_date=date)
            if latest_fina is None or latest_fina.empty:
                # log.debug(f"在日期 {date} 未找到 {ts_code} 的有效PIT财务数据。")
                return np.nan

            latest_fina_row = latest_fina.iloc[0]
            
            # 优先使用 'or_yoy' 字段
            if 'or_yoy' in latest_fina_row.index and pd.notna(latest_fina_row['or_yoy']):
                return latest_fina_row['or_yoy']
            
            # 回退到 'netprofit_yoy'
            if 'netprofit_yoy' in latest_fina_row.index and pd.notna(latest_fina_row['netprofit_yoy']):
                return latest_fina_row['netprofit_yoy']
            
            return np.nan
        except (KeyError, IndexError) as e:
            log.warning(f"计算营收增长率因子 for {ts_code} on {date} 时出错: {e}")
            return np.nan

    def calc_debt_to_assets(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.1重构-健壮性】获取指定日期可得的最新财报的资产负债率(%)"""
        try:
            all_fina = self.data_manager.get_fina_indicator(ts_code)
            if all_fina is None or all_fina.empty:
                return np.nan
                
            latest_fina = self.data_manager.get_pit_financial_data(all_fina, as_of_date=date)
            if latest_fina is None or latest_fina.empty:
                # log.debug(f"在日期 {date} 未找到 {ts_code} 的有效PIT财务数据。")
                return np.nan
                
            return latest_fina['debt_to_assets'].iloc[0]
        except (KeyError, IndexError) as e:
            log.warning(f"计算资产负债率因子 for {ts_code} on {date} 时出错: {e}")
            return np.nan

    # --- D. 筹码类因子 (V2.1新增) ---
    def calc_holder_num_change_ratio(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.1新增】计算最新一期股东人数相对上一期的变化率(%)。负值表示筹码集中。"""
        df_holder_num = self.data_manager.get_holder_number(ts_code)
        if df_holder_num is None or df_holder_num.empty or len(df_holder_num) < 2:
            return np.nan

        # 筛选出公告日在指定日期之前的数据
        df_holder_num['ann_date'] = pd.to_datetime(df_holder_num['ann_date'])
        as_of_date_dt = pd.to_datetime(date)
        available_data = df_holder_num[df_holder_num['ann_date'] <= as_of_date_dt].copy()

        if len(available_data) < 2:
            return np.nan

        # 按报告期（end_date）降序排序，获取最新的两期财报
        available_data = available_data.sort_values(by='end_date', ascending=False)
        latest_num = available_data.iloc[0]['holder_num']
        previous_num = available_data.iloc[1]['holder_num']

        if previous_num is None or previous_num == 0:
            return np.nan

        return (latest_num - previous_num) / previous_num

    def calc_major_shareholder_net_buy_ratio(self, ts_code: str, date: str, lookback_days=90, **kwargs) -> float:
        """【V2.1新增】计算近N日重要股东增减持金额占自由流通市值的比例(%)。"""
        end_date_dt = pd.to_datetime(date)
        start_date_str = (end_date_dt - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')
        end_date_str = end_date_dt.strftime('%Y%m%d')
        
        df_trade = self.data_manager.get_holder_trade(ts_code, start_date_str, end_date_str)
        if df_trade is None or df_trade.empty:
            return 0.0

        # Tushare中，增持的in_de='IN', 减持='DE'
        df_trade['change_value'] = df_trade['change_vol'] * df_trade['avg_price']
        net_buy_value = df_trade[df_trade['in_de'] == 'IN']['change_value'].sum() - \
                        df_trade[df_trade['in_de'] == 'DE']['change_value'].sum()
        
        # 获取最新的流通市值
        df_basic = self.data_manager.get_daily_basic(ts_code, end_date_str, end_date_str)
        if df_basic is None or df_basic.empty or 'float_share' not in df_basic.columns:
            return np.nan
        
        # 流通股本单位是万股，总市值单位是万元，需统一
        float_cap = df_basic['float_share'].iloc[0] * df_basic['close'].iloc[0] # 单位: 万*元
        if float_cap == 0:
            return np.nan

        # 增减持金额单位是万元，流通市值单位也是万元
        return (net_buy_value / float_cap) * 100 # 返回百分比

    def calc_top_list_net_buy_amount(self, ts_code: str, date: str, top_list_df: pd.DataFrame) -> float:
        """【V2.1新增】从当日龙虎榜数据中，获取该股的净买入额(万元)。"""
        if top_list_df is None or top_list_df.empty:
            return 0.0
        
        stock_on_list = top_list_df[top_list_df['ts_code'] == ts_code]
        if stock_on_list.empty:
            return 0.0
            
        # 一个票可能因为多个原因上榜，这里简单求和
        return stock_on_list['net_amount'].sum()
    
    # --- F. 风险因子 (Beta Factors) (V2.3新增) ---
    def calc_size(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.3新增】计算市值因子（总市值的自然对数）。"""
        df_basic = self.data_manager.get_daily_basic(ts_code, date, date)
        if df_basic is None or df_basic.empty or 'total_mv' not in df_basic.columns:
            return np.nan
        # 总市值单位是万元
        total_mv = df_basic['total_mv'].iloc[0] * 10000
        if total_mv <= 0:
            return np.nan
        return np.log(total_mv)

    def calc_pb(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.3新增】计算市净率因子（作为价值因子的代表）。"""
        df_basic = self.data_manager.get_daily_basic(ts_code, date, date)
        if df_basic is None or df_basic.empty or 'pb' not in df_basic.columns:
            return np.nan
        pb = df_basic['pb'].iloc[0]
        return pb if pb > 0 else np.nan

    def calc_block_trade_ratio(self, ts_code: str, date: str, block_trade_df: pd.DataFrame, lookback_days=90, **kwargs) -> float:
        """【V2.1新增】计算近N日大宗交易成交额占期间总成交额的比例(%)。"""
        # 1. 从传入的DataFrame中筛选出该股的数据
        if block_trade_df is None or block_trade_df.empty:
            return 0.0
        
        stock_block_trade = block_trade_df[block_trade_df['ts_code'] == ts_code]
        if stock_block_trade.empty:
            return 0.0

        block_trade_amount_sum = stock_block_trade['amount'].sum() # 单位：万元

        # 2. 获取区间总成交额
        end_date_dt = pd.to_datetime(date)
        start_date_str = (end_date_dt - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')
        df_daily = self.data_manager.get_daily(ts_code, start_date_str, date)
        
        if df_daily is None or df_daily.empty:
            return np.nan
            
        total_amount = df_daily['amount'].sum() # 单位：千元
        total_amount_wan = total_amount / 10 # 转换为万元

        if total_amount_wan == 0:
            return np.nan

        return (block_trade_amount_sum / total_amount_wan) * 100

    # --- E. 价值与回报类因子 (V2.1新增) ---
    def calc_dividend_yield(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.1新增】计算最新的TTM股息率(%)。"""
        # 1. 获取近12个月的每股分红
        end_date_dt = pd.to_datetime(date)
        start_date_12m = (end_date_dt - pd.Timedelta(days=365)).strftime('%Y%m%d')
        df_dividend = self.data_manager.get_dividend(ts_code)
        
        if df_dividend is None or df_dividend.empty:
            return 0.0
            
        # 筛选出在过去一年内、且在观察日期前已公告的分红
        df_dividend['ann_date'] = pd.to_datetime(df_dividend['ann_date'])
        recent_dividends = df_dividend[
            (df_dividend['ann_date'] >= start_date_12m) &
            (df_dividend['ann_date'] <= end_date_dt)
        ].copy()
        
        if recent_dividends.empty:
            return 0.0
        
        # 计算每股现金分红总额
        total_cash_div = recent_dividends['cash_div_tax'].sum()
        
        # 2. 获取当日股价
        df_basic = self.data_manager.get_daily_basic(ts_code, date, date)
        if df_basic is None or df_basic.empty:
            return np.nan
        
        price = df_basic['close'].iloc[0]
        if price == 0:
            return np.nan
            
        return (total_cash_div / price) * 100

    def calc_forecast_growth_rate(self, ts_code: str, date: str, **kwargs) -> float:
        """【V2.1新增】获取最新一期已公告的业绩预告净利润增长率(%)。取预告区间的平均值。"""
        end_date_dt = pd.to_datetime(date)
        start_date_hist = (end_date_dt - pd.Timedelta(days=365)).strftime('%Y%m%d') # 回看一年内的预告
        
        df_forecast = self.data_manager.get_forecast(ts_code, start_date_hist, date)
        if df_forecast is None or df_forecast.empty:
            return np.nan
            
        # 筛选出在指定日期或之前已公告的数据，并按公告日排序
        df_forecast['ann_date'] = pd.to_datetime(df_forecast['ann_date'])
        available_forecast = df_forecast[df_forecast['ann_date'] <= end_date_dt].sort_values(by='ann_date', ascending=False)
        
        if available_forecast.empty:
            return np.nan
            
        latest_forecast = available_forecast.iloc[0]
        # 取预告增长率的中间值
        avg_growth = (latest_forecast['p_change_min'] + latest_forecast['p_change_max']) / 2
        return avg_growth

    def calc_repurchase_ratio(self, ts_code: str, date: str, lookback_days=365, **kwargs) -> float:
        """【V2.1新增】计算近N日累计回购金额占当前总市值的比例(%)。"""
        end_date_dt = pd.to_datetime(date)
        start_date_str = (end_date_dt - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')
        end_date_str = end_date_dt.strftime('%Y%m%d')

        df_repurchase = self.data_manager.get_repurchase(ts_code, start_date_str, end_date_str)
        if df_repurchase is None or df_repurchase.empty:
            return 0.0
            
        total_repurchase_amount = df_repurchase['amount'].sum() # 单位: 万元
        
        # 获取最新的总市值
        df_basic = self.data_manager.get_daily_basic(ts_code, end_date_str, end_date_str)
        if df_basic is None or df_basic.empty or 'total_mv' not in df_basic.columns:
            return np.nan
        
        total_market_value = df_basic['total_mv'].iloc[0] # 单位: 万元
        if total_market_value == 0:
            return np.nan
            
        return (total_repurchase_amount / total_market_value) * 100

    def calculate(self, factor_name: str, **kwargs) -> float:
        """
        【V2.1重构】统一的因子计算入口。
        根据因子名称，自动调用对应的 calc_ 方法。
        :param factor_name: 因子名称，如 'momentum'。
        :param kwargs: 计算所需的所有参数，如 ts_code, date, start_date, end_date 等。
        :return: 因子值。
        """
        try:
            # 构造计算方法的名称，例如 'calc_momentum'
            method_name = f"calc_{factor_name}"
            # 使用 getattr 获取对应的计算方法
            calc_method = getattr(self, method_name)
            # 直接传递所有可能的参数，由具体的计算方法按需取用
            return calc_method(**kwargs)
        except AttributeError:
            log.error(f"因子 '{factor_name}' 的计算方法 '{method_name}' 在 FactorFactory 中未定义。")
            return np.nan
        except Exception as e:
            # log.debug(f"计算因子 '{factor_name}' for {kwargs.get('ts_code')} 时出错: {e}")
            return np.nan

    def calc_top10_holder_ratio(self, ts_code: str, date: str, **kwargs) -> float:
        """【PIT修正】获取指定日期可得的最新报告期的前十大流通股东持股比例总和(%)"""
        target_date = pd.to_datetime(date, format='%Y%m%d')
        # 这是一个近似实现，通过反向迭代可能的报告期。
        # 一个更鲁棒的方案是预先获取并缓存全部历史数据。
        for year_offset in range(3): # 回看3年
            year = target_date.year - year_offset
            periods = [f"{year}1231", f"{year}0930", f"{year}0630", f"{year}0331"]
            for p in periods:
                if p > target_date.strftime('%Y%m%d'):
                    continue
                
                df_holders = self.data_manager.get_top10_floatholders(ts_code=ts_code, period=p)
                
                if df_holders is not None and not df_holders.empty and 'ann_date' in df_holders.columns:
                    df_holders['ann_date'] = pd.to_datetime(df_holders['ann_date'], format='%Y%m%d')
                    # 查找在目标日期或之前已公告的数据
                    df_pit = df_holders[df_holders['ann_date'] <= target_date].copy()
                    if not df_pit.empty:
                        # 对于该报告期，找到已公告的最新数据
                        latest_ann_date = df_pit['ann_date'].max()
                        # 筛选出该次公告的完整数据
                        df_final = df_pit[df_pit['ann_date'] == latest_ann_date]
                        return df_final['hold_ratio'].sum()
        return np.nan

    # --- E. 宏观类因子 (V2.0新增) ---
    def calc_m1_m2_scissors_gap(self, date: str, **kwargs) -> float:
        """
        获取指定日期的最新M1-M2剪刀差。
        注意：这是一个宏观因子，对所有股票在当期暴露值相同。
        """
        target_date = pd.to_datetime(date, format='%Y%m%d')
        start_m = (target_date - pd.Timedelta(days=90)).strftime('%Y%m') # 获取近3个月的数据确保覆盖
        end_m = target_date.strftime('%Y%m')

        df_m = self.data_manager.get_cn_m(start_m, end_m)
        if df_m is None or df_m.empty:
            return np.nan
        
        latest_m = df_m.iloc[-1]
        return latest_m['m1_yoy'] - latest_m['m2_yoy']

    # --- F. 语义因子 ---
    def create_sentiment_factor(self, sentiment_score: float) -> float:
        """将AI情绪评分直接作为因子值"""
        return sentiment_score

    def create_event_dummy_factor(self, extracted_events: str) -> int:
        """将特定正面/负面事件作为0/1虚拟变量因子"""
        if any(keyword in extracted_events for keyword in ["增持", "新品发布", "业绩增长"]):
            return 1
        elif any(keyword in extracted_events for keyword in ["调查", "诉讼", "监管提示"]):
            return -1
        return 0

    # --- G. V2.4新增：经典技术指标计算 ---
    def _get_price_series(self, ts_code: str, end_date: str, lookback_days: int) -> pd.Series | None:
        """获取用于计算技术指标的后复权价格序列的辅助函数"""
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')
        df_adj = self.data_manager.get_adjusted_daily(ts_code, start_date, end_date, adj='hfq')
        if df_adj is None or df_adj.empty:
            return None
        return df_adj.set_index('trade_date')['close']

    def calc_macd(self, ts_code: str, end_date: str) -> dict:
        """计算MACD指标 (DIF, DEA, MACD柱)"""
        prices = self._get_price_series(ts_code, end_date, 100)
        if prices is None or len(prices) < 30:
            return {'diff': np.nan, 'dea': np.nan, 'macd_hist': np.nan}
        
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        diff = exp12 - exp26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_hist = (diff - dea) * 2
        
        return {'diff': diff.iloc[-1], 'dea': dea.iloc[-1], 'macd_hist': macd_hist.iloc[-1]}

    def calc_rsi(self, ts_code: str, end_date: str, window: int = 14) -> float:
        """计算RSI指标"""
        prices = self._get_price_series(ts_code, end_date, 50)
        if prices is None or len(prices) < window + 1:
            return np.nan
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calc_boll(self, ts_code: str, end_date: str, window: int = 20) -> dict:
        """计算布林带指标 (上轨, 中轨, 下轨)"""
        prices = self._get_price_series(ts_code, end_date, 50)
        if prices is None or len(prices) < window:
            return {'upper': np.nan, 'middle': np.nan, 'lower': np.nan, 'close': np.nan}
            
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        return {'upper': upper.iloc[-1], 'middle': middle.iloc[-1], 'lower': lower.iloc[-1], 'close': prices.iloc[-1]}


class FactorProcessor:
    """
    因子预处理流水线
    - 负责对原始因子（raw factors）进行科学、系统的处理。
    """
    def __init__(self, _data_manager):
        self.data_manager = _data_manager

    def _winsorize(self, factor_series: pd.Series, method='mad', n=3) -> pd.Series:
        """去极值处理"""
        if method == 'mad':
            median = factor_series.median()
            mad = (factor_series - median).abs().median()
            upper_bound = median + n * 1.4826 * mad
            lower_bound = median - n * 1.4826 * mad
        elif method == 'quantile':
            upper_bound = factor_series.quantile(1 - n/100)
            lower_bound = factor_series.quantile(n/100)
        else:
            raise ValueError("Method not supported. Use 'mad' or 'quantile'.")
        return factor_series.clip(lower=lower_bound, upper=upper_bound)

    def _standardize(self, factor_series: pd.Series) -> pd.Series:
        """标准化处理 (z-score)"""
        mean = factor_series.mean()
        std = factor_series.std()
        if std == 0:
            return pd.Series(0, index=factor_series.index)
        return (factor_series - mean) / std

    def _neutralize_industry(self, factor_series: pd.Series, stock_basics: pd.DataFrame) -> pd.Series:
        """行业中性化处理"""
        df = pd.DataFrame({'factor': factor_series})
        df_basics_indexed = stock_basics.set_index('ts_code')
        df = pd.merge(df, df_basics_indexed[['industry']], left_index=True, right_index=True)
        
        industry_dummies = pd.get_dummies(df['industry'], drop_first=True, dtype=float)
        df = pd.concat([df, industry_dummies], axis=1)
        df.fillna(0, inplace=True)
        
        X = df[industry_dummies.columns]
        y = df['factor']

        try:
            # Add constant for intercept
            X_with_const = np.c_[np.ones(X.shape[0]), X.values]
            # OLS: beta = (X'X)^-1 * X'y
            beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y.values
            residuals = y - (X_with_const @ beta)
            return pd.Series(residuals, index=y.index)
        except np.linalg.LinAlgError: # if matrix is singular
            return factor_series # return original if neutralization fails

    def process_factor(self, factor_series: pd.Series, neutralize=True) -> pd.Series:
        """执行完整的因子预处理流水线"""
        processed_series = factor_series.dropna()
        if processed_series.empty:
            return processed_series
        
        processed_series = self._winsorize(processed_series)
        processed_series = self._standardize(processed_series)
        
        if neutralize:
            all_stocks_info = self.data_manager.get_stock_basic()
            # Filter stock_info to only include stocks present in the factor series
            relevant_stocks_info = all_stocks_info[all_stocks_info['ts_code'].isin(processed_series.index)]
            processed_series = self._neutralize_industry(processed_series, relevant_stocks_info)
            # Standardize again after neutralization
            processed_series = self._standardize(processed_series)
            
        return processed_series

class FactorAnalyzer:
    """
    【源于 factor_analyzer.py】
    【因子分析器】
    - 负责因子的存储、加载和有效性分析。
    - 核心分析工具包括 IC/IR 计算和分层回测。
    """
    def __init__(self, _data_manager):
        self.data_manager = _data_manager
        # 【V2.0 架构统一】移除旧的db_path和sqlite3连接逻辑，
        # 完全依赖 data_manager 提供的 SQLAlchemy engine connection。
        self.conn = _data_manager.conn


    def save_factors_to_db(self, factors_df: pd.DataFrame, table_name: str):
        """
        将包含多因子暴露值的 DataFrame 存入数据库。
        :param factors_df: DataFrame，index为日期，columns为股票代码，values为因子值。
        :param table_name: 要存入的数据库表名。
        """
        try:
            # 确保索引名为 'trade_date' 以便正确加载
            factors_df.index.name = 'trade_date'
            factors_df.to_sql(table_name, self.conn, if_exists='replace', index=True)
            log.info(f"因子数据已成功保存到表 '{table_name}'。")
        except Exception as e:
            log.error(f"保存因子数据时出错: {e}", exc_info=True)

    def load_factors_from_db(self, table_name: str) -> pd.DataFrame:
        """
        从数据库加载因子数据。
        :param table_name: 数据库表名。
        :return: 包含因子数据的 DataFrame。
        """
        try:
            df = pd.read_sql(f'SELECT * FROM {table_name}', self.conn, index_col='trade_date')
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            log.error(f"从数据库加载因子 '{table_name}' 时出错: {e}", exc_info=True)
            return pd.DataFrame()

    def calculate_ic(self, factor_series: pd.Series, forward_return_period=20) -> tuple:
        """
        计算单个截面日期的信息系数 (Information Coefficient, IC)。
        IC衡量的是因子值与未来收益率之间的秩相关性。
        :param factor_series: 单个日期、多支股票的因子值 Series。
        :param forward_return_period: 用于计算未来收益率的时间窗口（交易日）。
        :return: IC值和p-value。
        """
        # 1. 获取未来价格数据
        stock_list = factor_series.index.tolist()
        start_date = pd.to_datetime(factor_series.name)
        end_date = start_date + pd.Timedelta(days=forward_return_period * 2) # 预留足够的时间窗口
        
        prices_df = pd.DataFrame()
        for stock in stock_list:
            prices = self.data_manager.get_adjusted_daily(stock, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
            if prices is not None and not prices.empty:
                prices_df[stock] = prices.set_index('trade_date')['close']
        
        if len(prices_df) < forward_return_period:
            return np.nan, np.nan # 数据不足

        # 2. 计算未来收益率
        future_returns = prices_df.pct_change(periods=forward_return_period).shift(-forward_return_period)
        
        # 3. 对齐因子和收益率数据
        aligned_factors = factor_series
        if future_returns.empty:
            return np.nan, np.nan
        aligned_returns = future_returns.iloc[0] # 取第一个截面日的未来收益率
        
        common_index = aligned_factors.index.intersection(aligned_returns.dropna().index)
        if len(common_index) < 10: # 如果共同的股票太少，则结果无意义
            return np.nan, np.nan
            
        aligned_factors = aligned_factors.loc[common_index]
        aligned_returns = aligned_returns.loc[common_index]

        # 4. 计算 Spearman 秩相关系数
        if aligned_factors.empty or aligned_returns.empty:
            return np.nan, np.nan
        ic, p_value = spearmanr(aligned_factors, aligned_returns)
        return ic, p_value

    def calculate_ir(self, ic_series: pd.Series) -> float:
        """
        根据IC时间序列计算信息比率 (Information Ratio, IR)。
        IR = IC均值 / IC标准差，衡量因子表现的稳定性和风险调整后收益。
        """
        if ic_series.empty or ic_series.std() == 0:
            return np.nan
        return ic_series.mean() / ic_series.std()

    def run_layered_backtest(self, factor_df: pd.DataFrame, num_quantiles=5, forward_return_period=20) -> tuple:
        """
        执行分层回测，评估因子的单调性。
        :param factor_df: DataFrame，index为日期，columns为股票代码。
        :param num_quantiles: 分层数量。
        :param forward_return_period: 持仓周期（交易日）。
        :return: 包含各分层累计收益的DataFrame和Plotly图表对象。
        """
        # 1. 获取所有需要的价格数据
        all_stocks = factor_df.columns.tolist()
        start_date = factor_df.index.min().strftime('%Y%m%d')
        end_date = (factor_df.index.max() + pd.Timedelta(days=forward_return_period * 2)).strftime('%Y%m%d')
        
        all_prices = pd.DataFrame()
        # A more efficient way to get prices
        # prices_dict = self.data_manager.run_batch_download(all_stocks, start_date, end_date)
        # for stock, prices in prices_dict.items():
        #    if prices is not None and not prices.empty:
        #        all_prices[stock] = prices.set_index('trade_date')['close']

        for stock in all_stocks: # Fallback to loop if async fails or not preferred
            prices = self.data_manager.get_adjusted_daily(stock, start_date, end_date)
            if prices is not None and not prices.empty:
                all_prices[stock] = prices.set_index('trade_date')['close']

        # 2. 计算每日收益率
        daily_returns = all_prices.pct_change().fillna(0)
        
        # 3. 按周期进行分层和收益计算
        layered_returns = []
        rebalance_dates = factor_df.index[::forward_return_period]

        for date in rebalance_dates:
            if date not in daily_returns.index: continue
            
            factor_slice = factor_df.loc[date].dropna()
            if factor_slice.empty: continue
            
            # 制作分层标签
            labels = [f'Q{i+1}' for i in range(num_quantiles)]
            try:
                quantiles = pd.qcut(factor_slice, num_quantiles, labels=labels, duplicates='drop')
            except ValueError: # Not enough unique values to form quantiles
                continue
            
            period_start_idx = daily_returns.index.searchsorted(date)
            period_end_idx = min(period_start_idx + forward_return_period, len(daily_returns))
            
            period_returns = daily_returns.iloc[period_start_idx:period_end_idx]
            
            # 计算每个分层的等权平均收益
            quantile_returns = period_returns.groupby(quantiles, axis=1).mean().iloc[1:]
            layered_returns.append(quantile_returns)

        if not layered_returns:
            return pd.DataFrame(), go.Figure().update_layout(title="没有足够数据进行分层回测。")

        # 4. 合并并计算累计收益
        final_returns = pd.concat(layered_returns)
        cumulative_returns = (1 + final_returns).cumprod()
        
        # 5. 绘图
        fig = go.Figure()
        for col in cumulative_returns.columns:
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[col], mode='lines', name=col))
        
        fig.update_layout(
            title=f'因子分层回测 (持仓周期: {forward_return_period}天)',
            xaxis_title='日期',
            yaxis_title='累计净值',
            template='plotly_dark'
        )
        
        return cumulative_returns, fig

class AlphaStrategy:
    """
    高级策略框架
    - 融合量化因子和AI语义因子，生成最终投资决策。
    """
    def __init__(self, _data_manager, _factor_factory, _ai_orchestrator):
        self.data_manager = _data_manager
        self.factor_factory = _factor_factory
        self.ai_orchestrator = _ai_orchestrator

    def generate_scores(self, ts_code: str, date: str, ai_analysis_results: dict = None) -> dict:
        """为单个股票生成综合评分。"""
        scores = {}
        pe_factor = self.factor_factory.calc_pe_ttm(ts_code, date)
        roe_factor = self.factor_factory.calc_roe(ts_code, date)
        pe_score = 100 - min(pe_factor, 100) if pd.notna(pe_factor) and pe_factor > 0 else 50
        roe_score = min(roe_factor * 5, 100) if pd.notna(roe_factor) else 50
        quant_score = (pe_score * 0.4) + (roe_score * 0.6)
        scores['quant_score'] = quant_score
        scores['final_score'] = quant_score
        scores['decision'] = "持有" if quant_score > 60 else "中性"
        scores['reasoning'] = f"量化综合评分: {quant_score:.2f} (PE贡献: {pe_score*0.4:.2f}, ROE贡献: {roe_score*0.6:.2f})."

        if ai_analysis_results:
            event_factor = self.factor_factory.create_event_dummy_factor(ai_analysis_results.get('key_events', ''))
            if event_factor == -1:
                scores['final_score'] = 0
                scores['decision'] = "卖出/规避"
                scores['reasoning'] += "\nAI信号: 检测到重大负面事件，一票否决。"
                return scores
            
            sentiment_score_str = ai_analysis_results.get('sentiment_score', '0.0')
            try:
                sentiment_score = float(sentiment_score_str)
            except ValueError:
                sentiment_score = 0.0
            sentiment_factor = self.factor_factory.create_sentiment_factor(sentiment_score)
            sentiment_score_mapped = (sentiment_factor + 1) * 50
            final_score = (quant_score * 0.8) + (sentiment_score_mapped * 0.2)
            scores['final_score'] = final_score
            scores['reasoning'] += f"\nAI融合: 情绪得分为 {sentiment_score_mapped:.2f}，融合后最终得分为 {final_score:.2f}。"
            if final_score > 70:
                scores['decision'] = "买入/增持"
            elif final_score < 50:
                scores['decision'] = "卖出/减持"
            else:
                scores['decision'] = "持有"
        return scores

class AdaptiveAlphaStrategy:
    """
    自适应Alpha策略
    - 核心逻辑: 基于因子历史ICIR值动态调整因子权重。
    """
    def __init__(self, factor_factory, factor_processor, factor_analyzer, stock_prices_df):
        self.factor_factory = factor_factory
        self.factor_processor = factor_processor
        self.factor_analyzer = factor_analyzer
        self.stock_prices_df = stock_prices_df # 需要全周期的价格数据来计算因子和未来收益

    @lru_cache(maxsize=16) # 缓存计算过的权重，提高重复调用速度
    def calculate_dynamic_weights(self, end_date_str: str, lookback_days: int, factors_to_use: tuple) -> pd.Series:
        """
        在给定的end_date，回顾lookback_days，计算各因子的IR，并以此作为权重。
        """
        log.info(f"[{end_date_str}] 开始计算动态权重，回看周期: {lookback_days}天...")
        end_date = pd.to_datetime(end_date_str)
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        # 获取此期间的交易日
        all_trade_dates = self.stock_prices_df.index
        trade_dates_in_period = all_trade_dates[(all_trade_dates >= start_date) & (all_trade_dates <= end_date)]
        
        # 我们按月计算IC，以平衡计算量和时效性
        monthly_dates = trade_dates_in_period.to_period('M').to_timestamp('M', how='end').unique()
        monthly_dates = monthly_dates.intersection(trade_dates_in_period)

        ic_history = {}
        stock_pool = self.stock_prices_df.columns.tolist()

        for factor_name in factors_to_use:
            factor_ics = []
            for date in monthly_dates:
                date_str = date.strftime('%Y%m%d')
                factor_start_date = (date - pd.Timedelta(days=60)).strftime('%Y%m%d')
                
                # 计算当天的因子值
                if factor_name in ['momentum', 'volatility']:
                    prices_slice = self.stock_prices_df.loc[:date]
                    raw_values = {}
                    for s in prices_slice.columns:
                        df_temp = prices_slice[[s]].reset_index()
                        df_temp.columns = ['trade_date', 'close']
                        if factor_name == 'momentum':
                             if len(df_temp) >= 20: raw_values[s] = df_temp['close'].iloc[-1] / df_temp['close'].iloc[-20] - 1
                        elif factor_name == 'volatility':
                             if len(df_temp) >= 20: raw_values[s] = np.log(df_temp['close'] / df_temp['close'].shift(1)).iloc[-20:].std() * np.sqrt(252)
                else: # 其他非价格序列因子
                    raw_values = {s: getattr(self.factor_factory, f"calc_{factor_name}")(s, factor_start_date, date_str) for s in stock_pool}
                
                raw_series = pd.Series(raw_values, name=date).dropna()
                if raw_series.empty: continue
                
                processed_factor = self.factor_processor.process_factor(raw_series, neutralize=True)
                
                # 计算IC
                ic, _ = self.factor_analyzer.calculate_ic(processed_factor)
                if not np.isnan(ic):
                    factor_ics.append(ic)
            
            ic_history[factor_name] = pd.Series(factor_ics)
        
        # 计算IR作为权重
        ir_values = {name: self.factor_analyzer.calculate_ir(ics) for name, ics in ic_history.items()}
        ir_series = pd.Series(ir_values).fillna(0)
        
        # 将IR值转化为权重（只考虑正向IR，并归一化）
        weights = ir_series.clip(lower=0)
        if weights.sum() == 0: # 如果所有IR都为负，则等权
            weights = pd.Series(1.0 / len(factors_to_use), index=factors_to_use)
        else:
            weights = weights / weights.sum()
        
        log.info(f"[{end_date_str}] 动态权重计算完成:\n{weights}")
        return weights

    def generate_composite_factor(self, trade_date: pd.Timestamp, stock_pool: list, factors_to_use: tuple, ic_lookback_days: int) -> pd.Series:
        """
        为给定的交易日生成加权合成因子。
        """
        # 1. 获取当前周期的动态权重
        dynamic_weights = self.calculate_dynamic_weights(trade_date.strftime('%Y%m%d'), ic_lookback_days, factors_to_use)
        
        # 2. 计算并合成因子
        composite_factor = pd.Series(dtype=float)
        factor_start_date = (trade_date - pd.Timedelta(days=60)).strftime('%Y%m%d')
        date_str = trade_date.strftime('%Y%m%d')

        for factor_name, weight in dynamic_weights.items():
            if weight == 0: continue
            
            # (与app.py中的因子计算逻辑保持一致)
            if factor_name in ['momentum', 'volatility']:
                prices_slice = self.stock_prices_df.loc[:trade_date]
                raw_values = {}
                for s in prices_slice.columns:
                    df_temp = prices_slice[[s]].reset_index()
                    df_temp.columns = ['trade_date', 'close']
                    if factor_name == 'momentum':
                         if len(df_temp) >= 20: raw_values[s] = df_temp['close'].iloc[-1] / df_temp['close'].iloc[-20] - 1
                    elif factor_name == 'volatility':
                         if len(df_temp) >= 20: raw_values[s] = np.log(df_temp['close'] / df_temp['close'].shift(1)).iloc[-20:].std() * np.sqrt(252)
            else:
                raw_values = {s: getattr(self.factor_factory, f"calc_{factor_name}")(s, factor_start_date, date_str) for s in stock_pool}

            raw_series = pd.Series(raw_values).dropna()
            if raw_series.empty: continue
            processed_factor = self.factor_processor.process_factor(raw_series, neutralize=True)

            if composite_factor.empty:
                composite_factor = processed_factor.mul(weight).reindex(stock_pool).fillna(0)
            else:
                composite_factor = composite_factor.add(processed_factor.mul(weight), fill_value=0)
        
        return composite_factor, dynamic_weights


class PortfolioOptimizer:
    """
    【增强版】投资组合优化器
    - 使用 SciPy.optimize.minimize 实现均值-方差优化。
    """
    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.num_assets = len(expected_returns)

    def _portfolio_performance(self, weights):
        """计算投资组合表现（收益、标准差、夏普比率）"""
        returns = np.sum(self.expected_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = returns / std if std != 0 else 0
        return returns, std, sharpe

    def _negative_sharpe(self, weights):
        """目标函数：最小化负的夏普比率"""
        return -self._portfolio_performance(weights)[2]

    def optimize_max_sharpe(self, max_weight_per_stock=0.1):
        """
        执行优化，目标是最大化夏普比率。
        :param max_weight_per_stock: 单个股票的最大权重限制。
        """
        if self.num_assets == 0:
            return pd.DataFrame(columns=['weight'])

        # 约束条件：所有权重之和为1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # 边界条件：每个权重在0到max_weight_per_stock之间
        bounds = tuple((0, max_weight_per_stock) for _ in range(self.num_assets))
        # 初始猜测：等权重
        initial_weights = np.array(self.num_assets * [1. / self.num_assets,])

        # 执行优化
        opts = minimize(self._negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not opts.success:
            weights = np.clip(initial_weights, 0, max_weight_per_stock)
            weights /= np.sum(weights)
            return pd.DataFrame(weights, index=self.expected_returns.index, columns=['weight'])

        optimized_weights = pd.DataFrame(opts.x, index=self.expected_returns.index, columns=['weight'])
        return optimized_weights[optimized_weights['weight'] > 1e-4]

class VectorizedBacktester:
    """
    【增强版】向量化回测引擎
    - 新增交易成本和止损功能。
    """
    def __init__(self, all_prices: pd.DataFrame, all_factors: pd.DataFrame = None, rebalance_freq='M', num_groups=10, 
                 commission=0.0003, slippage=0.0002, stop_loss_pct=None):
        self.prices = all_prices.copy()
        # 只有在传入了 all_factors 时才进行处理
        self.factors = all_factors.copy() if all_factors is not None else None
        self.rebalance_freq = rebalance_freq
        self.num_groups = num_groups
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.results = {}

    def _get_rebalance_dates(self):
        """【最终版】获取调仓日期，使用更稳健的resample方法"""
        # 使用 'ME' (Month-End) 来消除 FutureWarning
        freq = 'W' if self.rebalance_freq == 'W' else 'ME'
        return self.prices.resample(freq).last().index

    def run(self, weights_df: pd.DataFrame = None):
        """
        【最终加固版】执行回测。
        :param weights_df: 如果提供了权重DataFrame，则使用该权重进行回测，否则执行因子分组策略。
        """
        rebalance_dates = self._get_rebalance_dates()
        log.info(f"回测期间共找到 {len(rebalance_dates)} 个调仓日。")

        if len(rebalance_dates) < 2:
            log.warning("回测期间的有效调仓日不足2个，无法构成交易区间，回测终止。")
            self.results = {'performance': pd.Series({"错误": "有效调仓日不足2个"}), 'cumulative_returns': pd.Series([1.0])}
            return self.results
            
        portfolio_value = 1.0
        daily_returns = []
        last_weights = pd.Series(dtype=float)

        for i in range(len(rebalance_dates) - 1):
            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i+1]
            
            # --- 核心修正：寻找调仓日对应的最后一个有效交易日 (使用asof，兼容旧版pandas) ---
            # asof() 会返回在 start_date 当天或之前的最后一个有效索引值
            actual_weight_date = weights_df.index.asof(start_date)
            
            # 如果调仓日在所有数据之前，asof会返回NaT (Not a Time)
            if pd.isna(actual_weight_date):
                log.warning(f"调仓日 {start_date.date()} 早于数据起始日，跳过。")
                continue

            target_weights = weights_df.loc[actual_weight_date].dropna()
            target_weights = target_weights[target_weights > 0]

            # 如果当天没有选出任何股票，则空仓并跳到下一个周期
            if target_weights.empty:
                # 空仓期的收益是0
                period_dates = self.prices.loc[actual_weight_date:end_date].index
                # 我们需要填充与价格序列对齐的零收益序列
                if len(period_dates) > 1:
                    daily_returns.append(pd.Series(0, index=period_dates[1:]))
                last_weights = pd.Series(dtype=float)
                continue
            
            # --- 后续逻辑保持不变，但使用 actual_weight_date ---
            period_prices = self.prices.loc[actual_weight_date:end_date]
            
            turnover = (target_weights.subtract(last_weights.reindex(target_weights.index, fill_value=0), fill_value=0)).abs().sum()
            trade_cost = turnover * (self.commission + self.slippage)
            
            period_portfolio_value = portfolio_value * (1 - trade_cost)

            period_daily_returns = period_prices[target_weights.index].pct_change().iloc[1:]
            
            if self.stop_loss_pct is not None:
                cumulative_period_returns = (1 + period_daily_returns).cumprod()
                stop_loss_triggered = (cumulative_period_returns < (1 - self.stop_loss_pct)).any(axis=1)
                if stop_loss_triggered.any():
                    trigger_date = stop_loss_triggered.idxmax()
                    period_daily_returns = period_daily_returns.loc[:trigger_date]
                    target_weights = pd.Series()
            
            weighted_returns = period_daily_returns.dot(target_weights.reindex(period_daily_returns.columns, fill_value=0))
            
            daily_returns.append(weighted_returns)
            portfolio_value = period_portfolio_value * (1 + weighted_returns).prod()
            last_weights = target_weights

        self.results['daily_returns'] = pd.concat(daily_returns).dropna()
        self._calculate_performance()
        return self.results

    def _calculate_performance(self):
        """计算详细的绩效指标"""
        returns = self.results['daily_returns']
        if returns.empty:
            self.results['performance'] = pd.Series(dtype=float)
            self.results['cumulative_returns'] = pd.Series(dtype=float)
            return
            
        cumulative_returns = (1 + returns).cumprod()
        self.results['cumulative_returns'] = cumulative_returns
        
        total_days = len(returns)
        if total_days == 0:
            self.results['performance'] = pd.Series(dtype=float)
            return

        trading_days_per_year = 252
        
        annual_return = cumulative_returns.iloc[-1]**(trading_days_per_year / total_days) - 1
        annual_volatility = returns.std() * np.sqrt(trading_days_per_year)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        self.results['performance'] = pd.Series({
            '年化收益率': f"{annual_return:.2%}",
            '年化波动率': f"{annual_volatility:.2%}",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '累计收益率': f"{(cumulative_returns.iloc[-1] - 1):.2%}"
        })

    def plot_results(self):
        """绘制回测结果图表"""
        if 'cumulative_returns' not in self.results or self.results['cumulative_returns'].empty:
            return go.Figure().update_layout(title="没有足够的数据来绘制图表。")
            
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("净值曲线", "回撤"))
        
        fig.add_trace(go.Scatter(x=self.results['cumulative_returns'].index, y=self.results['cumulative_returns'], name='策略净值'), row=1, col=1)
        
        drawdown = (self.results['cumulative_returns'] - self.results['cumulative_returns'].cummax()) / self.results['cumulative_returns'].cummax()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='回撤', fill='tozeroy', line=dict(color='red')), row=2, col=1)
        
        fig.update_layout(title_text="策略回测表现", template="plotly_dark", height=600)
        return fig

class AutomatedTasks:
    """定义需要周期性执行的自动化任务。"""
    def __init__(self, data_manager, factor_factory):
        self.data_manager = data_manager
        self.factor_factory = factor_factory

    def run_daily_data_update(self):
        """任务一：每日数据更新。"""
        print(f"[{datetime.now()}] Running daily data update...")
        try:
            self.data_manager.get_stock_basic(force_update=True)
            print("  - Stock basic updated.")
            self.data_manager.get_daily('000001.SZ', '20200101', datetime.now().strftime('%Y%m%d'))
            print("  - Sample daily data updated.")
            print("Daily data update finished.")
            return "Daily update successful."
        except Exception as e:
            error_msg = f"Daily update failed: {e}"
            print(error_msg)
            return error_msg

    def run_factor_calculation(self, output_path='data/processed_factors.csv'):
        """任务二：全市场因子计算。"""
        print(f"[{datetime.now()}] Running factor calculation...")
        try:
            all_stocks = self.data_manager.get_stock_basic()['ts_code'].tolist()
            latest_date = datetime.now().strftime('%Y%m%d')
            start_date = (pd.to_datetime(latest_date) - pd.Timedelta(days=50)).strftime('%Y%m%d')
            
            factor_data = {}
            for ts_code in all_stocks[:50]: 
                try:
                    momentum = self.factor_factory.calc_momentum(ts_code, start_date, latest_date)
                    factor_data[ts_code] = {'momentum': momentum}
                except Exception:
                    continue
            
            df_factors = pd.DataFrame.from_dict(factor_data, orient='index')
            df_factors.to_csv(output_path)
            
            success_msg = f"Factor calculation finished. {len(df_factors)} stocks processed. Saved to {output_path}."
            print(success_msg)
            return success_msg
        except Exception as e:
            error_msg = f"Factor calculation failed: {e}"
            print(error_msg)
            return error_msg

# ############################################################################
# --- 模块来源: quant_engine_v2.py (高级功能模块) ---
# ############################################################################

class MarketProfile:
    """
    分析宏观经济指标以确定当前的市场状态。
    实现了‘动态市场状态感知’(功能需求 3.1)的要求。
    """
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def get_market_regime(self, end_date: datetime) -> str:
        """
        基于PMI和M1-M2剪刀差判断市场状态。
        """
        start_m = f"{end_date.year-1}{end_date.month:02d}"
        end_m = f"{end_date.year}{end_date.month:02d}"
        
        df_pmi = self.data_manager.get_cn_pmi(start_m, end_m)
        df_m = self.data_manager.get_cn_m(start_m, end_m)

        if df_pmi is None or df_m is None or df_pmi.empty or df_m.empty:
            return "未知"

        latest_pmi = df_pmi.iloc[-1]
        latest_m = df_m.iloc[-1]

        # 修复大小写问题：将所有列名转为小写
        latest_pmi.index = [idx.lower() for idx in latest_pmi.index]
        
        # 根据最新的Tushare接口文档，制造业PMI字段为'pmi010000'
        pmi_col = 'pmi010000'

        # 如果找不到任何有效的PMI列，则记录警告并返回未知状态
        if pmi_col not in latest_pmi.index:
            log.warning(f"无法在PMI数据中找到 '{pmi_col}' 列。可用列: {latest_pmi.index.tolist()}")
            return "未知(数据列缺失)"

        pmi_value = pd.to_numeric(latest_pmi[pmi_col])
        m1_m2_gap = latest_m['m1_yoy'] - latest_m['m2_yoy']

        if pmi_value > 51 and m1_m2_gap > 1:
            return "牛市"
        elif pmi_value < 49 and m1_m2_gap < -1:
            return "熊市"
        else:
            return "震荡市"

class MLAlphaStrategy:
    """
    【V2.3 增强版】一个使用机器学习模型（LightGBM）进行选股的策略。
    实现了‘多模型融合引擎’(功能需求 3.2)的要求。
    """
    def __init__(self):
        self.model = LGBMClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.features_columns = []

    def _prepare_data(self, all_prices: pd.DataFrame, all_factors: pd.DataFrame, forward_return_period=20, quintile=5):
        """
        【V2.3 修复版】准备用于模型训练的特征(X)和目标(y)。
        :param all_prices: 包含所有股票价格的时间序列DataFrame (wide format: dates x stocks)。
        :param all_factors: 包含所有因子暴露的长格式DataFrame (long format: date, ts_code, factor_name, factor_value)。
        :param forward_return_period: 预测的未来收益周期。
        :param quintile: 收益率分组数量，5表示取收益率最高的20%的股票作为正样本。
        :return: X (特征), y (标签)
        """
        log.info("【ML数据准备】开始准备训练数据...")
        # 1. 计算未来收益率作为标签 (y)
        future_returns = all_prices.pct_change(periods=forward_return_period).shift(-forward_return_period)
        ranks = future_returns.rank(axis=1, pct=True, na_option='keep') # 保留NaN
        target = (ranks > (1 - 1/quintile))
        y = target.stack().rename('target')

        # 2. 准备特征矩阵 (X)
        # 将长格式的因子数据转换为宽格式
        X = all_factors.pivot_table(index='trade_date', columns='ts_code', values='factor_value', aggfunc='first')
        X = X.stack().rename('factor_value').reset_index()
        # 再次透视，构建最终的特征矩阵
        X = X.pivot_table(index=['trade_date', 'ts_code'], columns='factor_name', values='factor_value')
        
        # 3. 对齐特征和标签
        # 合并X和y，dropna确保每个样本都有完整的特征和标签
        combined_data = X.join(y, how='inner').dropna()

        if combined_data.empty:
            log.warning("对齐和去空值后，没有剩余的有效训练数据。")
            return pd.DataFrame(), pd.Series()

        y_final = combined_data['target']
        X_final = combined_data.drop(columns=['target'])

        self.features_columns = X_final.columns.tolist()
        log.info(f"【ML数据准备】数据准备完毕。共 {len(X_final)} 个样本，{len(self.features_columns)} 个特征。")
        return X_final, y_final

    def train(self, all_prices: pd.DataFrame, all_factors_long: pd.DataFrame, model_path="ml_model.pkl"):
        """
        【V2.3 修正版】在准备好的数据上训练机器学习模型，并保存到本地。
        :param all_factors_long: 长格式的因子数据。
        :param model_path: 模型保存路径。
        :return: 训练结果的字典。
        """
        import joblib
        X, y = self._prepare_data(all_prices, all_factors_long)
        
        if X.empty:
            return {"status": "error", "message": "没有有效的训练数据。"}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
        
        log.info("【ML模型训练】开始训练LightGBM分类器...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        accuracy = self.model.score(X_test, y_test)
        log.info(f"【ML模型训练】训练成功！测试集准确率: {accuracy:.2%}")

        # 保存模型
        joblib.dump(self.model, model_path)
        log.info(f"模型已保存至: {model_path}")

        return {
            "status": "success",
            "accuracy": accuracy,
            "features": self.features_columns,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "model_path": model_path
        }

    def load_model(self, model_path="ml_model.pkl"):
        """从本地加载已训练好的模型。"""
        import joblib
        import os
        if not os.path.exists(model_path):
            log.error(f"模型文件不存在: {model_path}")
            self.is_trained = False
            return False
        
        self.model = joblib.load(model_path)
        self.is_trained = True
        log.info(f"模型已从 {model_path} 加载成功。")
        return True

    def predict_top_stocks(self, factor_snapshot: pd.DataFrame, top_n: int = 20) -> pd.Index:
        """
        【V2.3 增强版】使用加载的模型，对单日因子截面数据进行预测，返回得分最高的N只股票。
        :param factor_snapshot: DataFrame, index为ts_code, columns为因子名称。
        :return: 包含得分最高的N只股票代码的Index。
        """
        if not self.is_trained:
            log.warning("模型尚未训练或加载，无法进行预测。")
            return pd.Index([])

        if factor_snapshot.empty:
            return pd.Index([])

        # 【鲁棒性修正】在预测前，确保输入DataFrame的列顺序与模型训练时完全一致
        try:
            # LightGBM模型会通过 model.feature_name_ 保存特征名
            model_features = self.model.feature_name_
            factor_snapshot_aligned = factor_snapshot[model_features]
        except Exception as e:
            log.error(f"特征对齐失败: {e}。模型需要特征: {self.model.feature_name_}，但输入数据只有: {factor_snapshot.columns.tolist()}")
            return pd.Index([])

        # 预测成为正样本（即未来高收益）的概率
        probabilities = self.model.predict_proba(factor_snapshot_aligned)[:, 1]
        prob_series = pd.Series(probabilities, index=factor_snapshot_aligned.index)
        
        return prob_series.nlargest(top_n).index


class RiskManager:
    """
    【V2.3 新增】投资组合级风险管理器
    - 负责分析一个投资组合在不同风险因子上的暴露度。
    """
    def __init__(self, factor_factory: FactorFactory, factor_processor: FactorProcessor):
        self.ff = factor_factory
        self.fp = factor_processor
        self.risk_factors = ['size', 'pb', 'momentum', 'volatility'] # 定义要分析的风险因子

    def calculate_risk_exposure(self, portfolio_weights: pd.Series, trade_date: str) -> pd.Series:
        """
        计算给定投资组合在指定日期的风险暴露。
        :param portfolio_weights: Series, index为ts_code, values为权重。
        :param trade_date: 分析日期 YYYYMMDD
        :return: Series, index为风险因子名称, values为组合在该因子上的加权暴露值。
        """
        if portfolio_weights.empty:
            return pd.Series(dtype=float)

        stock_pool = portfolio_weights.index.tolist()
        
        # 1. 计算股票池中所有股票在当期的风险因子暴露
        risk_factor_exposure = {}
        for factor in self.risk_factors:
            params = {
                'date': trade_date,
                'start_date': (pd.to_datetime(trade_date) - pd.Timedelta(days=365)).strftime('%Y%m%d'),
                'end_date': trade_date,
            }
            raw_values = {code: self.ff.calculate(factor, ts_code=code, **params) for code in stock_pool}
            raw_series = pd.Series(raw_values).dropna()

            # 2. 对风险因子进行标准化处理，使其具有可比性
            if not raw_series.empty:
                processed_series = self.fp.process_factor(raw_series, neutralize=False) # 风险因子通常不进行行业中性化
                risk_factor_exposure[factor] = processed_series

        risk_df = pd.DataFrame(risk_factor_exposure)
        
        # 3. 计算投资组合的加权风险暴露
        # 将持仓权重与股票的风险因子暴露对齐
        aligned_weights, aligned_risk_df = portfolio_weights.align(risk_df, join='inner', axis=0)
        
        if aligned_weights.empty:
            return pd.Series(dtype=float)
            
        # 核心计算：权重向量 与 风险暴露矩阵 的点积
        portfolio_exposure = aligned_risk_df.mul(aligned_weights, axis=0).sum()
        
        return portfolio_exposure
        """
        使用训练好的模型预测指定日期的表现最优股票。
        """
        if not self.is_trained or trade_date not in self.features.index:
            return pd.Index([])

        X_pred = self.features.loc[trade_date].dropna().to_frame()
        if X_pred.empty:
            return pd.Index([])
            
        probabilities = self.model.predict_proba(X_pred)[:, 1]
        prob_series = pd.Series(probabilities, index=X_pred.index)
        
        return prob_series.nlargest(top_n).index

class PerformanceAttribution:
    """
    实现用于业绩归因的 Brinson-Fachler 模型。
    分析收益以确定其来自“资产配置”和“个股选择”的贡献。
    实现了‘全方位绩效归因’(功能需求 4.2)的要求。
    """
    def __init__(self, portfolio_returns: pd.Series, portfolio_weights: pd.DataFrame, 
                 benchmark_returns: pd.Series, benchmark_weights: pd.DataFrame, 
                 stock_industry_map: pd.DataFrame):
        self.pr = portfolio_returns
        self.pw = portfolio_weights
        self.br = benchmark_returns
        self.bw = benchmark_weights
        self.stock_industry_map = stock_industry_map.set_index('ts_code')

    def _aggregate_by_industry(self, weights, returns):
        """按行业聚合权重和收益的辅助函数。"""
        df = pd.DataFrame({'weight': weights, 'return': returns}).join(self.stock_industry_map)
        industry_return = df.groupby('industry').apply(lambda x: np.average(x['return'], weights=x['weight']))
        industry_weight = df.groupby('industry')['weight'].sum()
        return industry_weight, industry_return

    def run_brinson_attribution(self) -> pd.DataFrame:
        """执行 Brinson 模型计算。"""
        pw_ind, pr_ind = self._aggregate_by_industry(self.pw, self.pr)
        bw_ind, br_ind = self._aggregate_by_industry(self.bw, self.br)
        
        industries = bw_ind.index.union(pw_ind.index)
        pw_ind, bw_ind = pw_ind.align(bw_ind, join='outer', axis=0, fill_value=0)
        pr_ind, br_ind = pr_ind.align(br_ind, join='outer', axis=0, fill_value=0)

        total_benchmark_return = (bw_ind * br_ind).sum()

        allocation = (pw_ind - bw_ind) * (br_ind - total_benchmark_return)
        selection = bw_ind * (pr_ind - br_ind)
        interaction = (pw_ind - bw_ind) * (pr_ind - br_ind)
        
        results = pd.DataFrame({
            '资产配置': allocation,
            '个股选择': selection,
            '交叉效应': interaction,
            '总效应': allocation + selection + interaction
        })

        results.loc['总计'] = results.sum()
        return results.applymap(lambda x: f"{x:.4%}")

# ############################################################################
# --- 模块来源: quant_engine_v3.py (CVXPY优化器和事件驱动回测框架) ---
# ############################################################################

class AdvancedPortfolioOptimizer:
    """
    【高级组合优化器】
    - 使用 CVXPY 库，一个为凸优化问题设计的强大工具。
    - 支持复杂的投资组合约束，如行业中性、风险因子暴露等。
    """
    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, stock_industry_map: pd.DataFrame):
        self.returns = expected_returns
        self.cov = cov_matrix
        self.stock_industry_map = stock_industry_map.set_index('ts_code')
        self.num_assets = len(expected_returns)
        self.tickers = expected_returns.index

    def optimize(self, max_weight_per_stock=0.1, industry_max_exposure=0.2, gamma=0.5):
        """
        执行均值-方差优化，并加入行业约束。
        目标：最大化 (预期收益 - gamma * 组合风险)
        """
        weights = cp.Variable(self.num_assets)

        portfolio_return = self.returns.values @ weights
        portfolio_risk = cp.quad_form(weights, self.cov.values)
        objective = cp.Maximize(portfolio_return - gamma * portfolio_risk)

        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= max_weight_per_stock
        ]

        industries = self.stock_industry_map['industry'].unique()
        for industry in industries:
            industry_stocks_idx = [i for i, t in enumerate(self.tickers) if self.stock_industry_map.loc[t, 'industry'] == industry]
            if industry_stocks_idx:
                constraints.append(cp.sum(weights[industry_stocks_idx]) <= industry_max_exposure)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"优化未找到最优解，状态: {problem.status}")
                return None
            
            optimized_weights = pd.DataFrame(weights.value, index=self.tickers, columns=['weight'])
            return optimized_weights[optimized_weights['weight'] > 1e-5]
        except Exception as e:
            print(f"CVXPY 优化过程中发生错误: {e}")
            return None

class Event(ABC):
    """事件基类"""
    @property
    def type(self):
        return self.__class__.__name__

class MarketEvent(Event):
    """标记新一轮市场数据的事件"""
    def __init__(self, timestamp):
        self.timestamp = timestamp

class SignalEvent(Event):
    """由策略对象产生的交易信号事件"""
    def __init__(self, ts_code, direction, strength=1.0):
        self.ts_code = ts_code
        self.direction = direction # 'LONG' or 'SHORT' or 'EXIT'
        self.strength = strength

class OrderEvent(Event):
    """由投资组合对象发出的订单事件"""
    def __init__(self, ts_code, order_type, quantity, direction):
        self.ts_code = ts_code
        self.order_type = order_type # 'MKT' or 'LMT'
        self.quantity = quantity
        self.direction = direction # 'BUY' or 'SELL'

class FillEvent(Event):
    """模拟订单被执行的成交事件"""
    def __init__(self, timestamp, ts_code, quantity, direction, fill_cost, commission):
        self.timestamp = timestamp
        self.ts_code = ts_code
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission

class DataHandler(ABC):
    """数据处理器基类"""
    @abstractmethod
    def get_latest_bar(self, ts_code): pass
    
    @abstractmethod
    def update_bars(self): pass

class Strategy(ABC):
    """策略基类"""
    @abstractmethod
    def calculate_signals(self, event): pass

class Portfolio(ABC):
    """投资组合基类"""
    @abstractmethod
    def update_signal(self, event): pass
    
    @abstractmethod
    def update_fill(self, event): pass
    
    @abstractmethod
    def update_timeindex(self, event): pass

class SimplePortfolio(Portfolio):
    """【修正版】一个简单的投资组合实现，负责仓位和P&L管理"""
    def __init__(self, data_handler: "SimpleDataHandler", events_queue, initial_capital=1000000.0, commission=0.0003, slippage=0.0002, sizing_method='fixed_cash', trade_cash_pct=0.05):
        self.data_handler = data_handler
        self.events = events_queue
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.sizing_method = sizing_method
        self.trade_cash_pct = trade_cash_pct

        self.all_positions = []  # 记录每个时间点的仓位
        self.current_positions = {code: 0 for code in self.data_handler.all_prices.columns}

        self.all_holdings = []  # 记录每个时间点的资产组合价值
        self.current_holdings = {'cash': initial_capital, 'total': initial_capital, 'commission': 0, 'slippage': 0}
        
        self.trade_log = []

    def update_timeindex(self, event: MarketEvent):
        """在每个时间点更新投资组合的总价值"""
        dt = event.timestamp
        holdings_update = {'timestamp': dt, 'cash': self.current_holdings['cash'], 'commission': 0, 'slippage': 0}
        
        total_value = self.current_holdings['cash']
        for code, quantity in self.current_positions.items():
            if quantity > 0:
                price = self.data_handler.get_latest_bar_value(code, 'close')
                market_value = price * quantity
                holdings_update[code] = market_value
                total_value += market_value
        
        holdings_update['total'] = total_value
        self.all_holdings.append(holdings_update)
        self.current_holdings['total'] = total_value
        # 记录仓位快照
        self.all_positions.append({'timestamp': dt, **self.current_positions})

    def update_signal(self, event: SignalEvent):
        """【修正版】根据策略信号生成订单"""
        if not isinstance(event, SignalEvent):
            return
        
        if event.direction == 'LONG':
            # --- 修正核心逻辑：使用更真实的资金管理 ---
            # 不再使用总资产的百分比，而是使用当前可用现金的百分比
            # 防止因同一天出现多个买入信号而过度开仓
            if self.sizing_method == 'fixed_cash':
                target_value = self.current_holdings['cash'] * self.trade_cash_pct
            else: # 默认为总资产百分比
                target_value = self.current_holdings['total'] * self.trade_cash_pct
                
            price = self.data_handler.get_latest_bar_value(event.ts_code, 'close')
            if pd.isna(price) or price == 0: return
            
            # 确保有足够现金执行交易
            if self.current_holdings['cash'] < target_value:
                log.debug(f"[{self.data_handler.latest_data['timestamp'].strftime('%Y-%m-%d')}] 跳过买入 {event.ts_code}: 现金不足。")
                return

            quantity = int(target_value / price // 100 * 100) # 按手买入
            if quantity > 0:
                order = OrderEvent(event.ts_code, 'MKT', quantity, 'BUY')
                self.events.put(order)
                log.info(f"[{self.data_handler.latest_data['timestamp'].strftime('%Y-%m-%d')}] 生成买入订单: {event.ts_code}, 数量: {quantity}")

        elif event.direction == 'EXIT':
            quantity = self.current_positions.get(event.ts_code, 0)
            if quantity > 0:
                order = OrderEvent(event.ts_code, 'MKT', quantity, 'SELL')
                self.events.put(order)

    def update_fill(self, event: FillEvent):
        """根据成交事件更新仓位和现金"""
        if not isinstance(event, FillEvent):
            return

        if event.direction == 'BUY':
            self.current_positions[event.ts_code] += event.quantity
            self.current_holdings['cash'] -= (event.fill_cost + event.commission)
        elif event.direction == 'SELL':
            self.current_positions[event.ts_code] -= event.quantity
            self.current_holdings['cash'] += (event.fill_cost - event.commission)
        
        self.trade_log.append({
            'timestamp': event.timestamp,
            'ts_code': event.ts_code,
            'direction': event.direction,
            'quantity': event.quantity,
            'fill_cost': event.fill_cost,
            'commission': event.commission
        })

    def get_performance_report(self) -> dict:
        """回测结束后，生成绩效报告"""
        if not self.all_holdings:
            return {}
            
        report_df = pd.DataFrame(self.all_holdings).set_index('timestamp')
        report_df['returns'] = report_df['total'].pct_change().fillna(0)
        
        total_days = len(report_df)
        trading_days_per_year = 252
        
        cumulative_return = (report_df['total'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + cumulative_return) ** (trading_days_per_year / total_days) - 1
        annual_volatility = report_df['returns'].std() * np.sqrt(trading_days_per_year)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        rolling_max = report_df['total'].cummax()
        drawdown = (report_df['total'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'performance': pd.Series({
                '初始资金': f"{self.initial_capital:,.2f}",
                '最终总资产': f"{report_df['total'].iloc[-1]:,.2f}",
                '累计收益率': f"{cumulative_return:.2%}",
                '年化收益率': f"{annual_return:.2%}",
                '年化波动率': f"{annual_volatility:.2%}",
                '夏普比率': f"{sharpe_ratio:.2f}",
                '最大回撤': f"{max_drawdown:.2%}",
                '总交易次数': len(self.trade_log)
            }),
            'equity_curve': report_df[['total']],
            'trade_log': pd.DataFrame(self.trade_log)
        }

class ExecutionHandler(ABC):
    """执行处理器基类"""
    @abstractmethod
    def execute_order(self, event): pass

class MockExecutionHandler(ExecutionHandler):
    """【增强版】模拟执行处理器，实现基于成交量的动态滑点模型"""
    def __init__(self, events_queue, data_handler: SimpleDataHandler, portfolio: SimplePortfolio, impact_factor=0.1):
        self.events = events_queue
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.impact_factor = impact_factor # 市场冲击系数

    def execute_order(self, event: OrderEvent):
        """【增强版】模拟订单执行，并创建 FillEvent，包含动态滑点"""
        if not isinstance(event, OrderEvent):
            return

        price = self.data_handler.get_latest_bar_value(event.ts_code, 'close')
        if pd.isna(price) or price == 0:
            return
        
        # --- 动态滑点模型 ---
        # 1. 固定滑点 (模拟买卖价差)
        base_slippage_pct = self.portfolio.slippage 
        
        # 2. 市场冲击滑点 (与交易量和流动性相关)
        avg_vol_20d = self.data_handler.get_average_volume(event.ts_code, 20, self.data_handler.latest_data['timestamp'])
        
        market_impact_pct = 0
        if avg_vol_20d > 0:
            # Tushare成交量单位是“手”，需乘以100
            volume_ratio = event.quantity / (avg_vol_20d * 100)
            # 冲击成本与交易量占比的平方根成正比是一个常用模型
            market_impact_pct = self.impact_factor * np.sqrt(volume_ratio)
            
        total_slippage_pct = base_slippage_pct + market_impact_pct
        slippage_amount = price * total_slippage_pct
        
        fill_price = price + slippage_amount if event.direction == 'BUY' else price - slippage_amount
        
        fill_cost = fill_price * event.quantity
        commission = fill_cost * self.portfolio.commission
        
        fill_event = FillEvent(
            timestamp=self.data_handler.latest_data['timestamp'], # 使用当前时间戳
            ts_code=event.ts_code,
            quantity=event.quantity,
            direction=event.direction,
            fill_cost=fill_cost,
            commission=commission
        )
        self.events.put(fill_event)

class EventDrivenBacktester:
    """
    【事件驱动回测引擎】
    - 通过事件队列驱动，模拟真实交易环境。
    """
    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = Queue()
        
        self.strategy.events = self.events
        self.portfolio.events = self.events
        # 新增：从data_handler获取事件队列的引用
        if hasattr(self.data_handler, 'events'):
            self.data_handler.events = self.events


    def _run_loop(self):
        """【增强版】主事件循环，增加日志记录"""
        while True:
            # 1. 数据先行，生成市场事件
            self.data_handler.update_bars()
            if not self.data_handler.continue_backtest:
                break
            
            # 2. 处理事件队列
            while True:
                try:
                    event = self.events.get(block=False)
                except Empty:
                    break # 事件队列处理完毕，进入下一时间点
                else:
                    log.debug(f"处理事件: {event.type}")
                    if event.type == 'MarketEvent':
                        self.strategy.calculate_signals(event)
                        self.portfolio.update_timeindex(event)
                    
                    elif event.type == 'SignalEvent':
                        self.portfolio.update_signal(event)
                        
                    elif event.type == 'OrderEvent':
                        self.execution_handler.execute_order(event)
                        
                    elif event.type == 'FillEvent':
                        self.portfolio.update_fill(event)

    def run_backtest(self):
        log.info("事件驱动回测开始...")
        self._run_loop()
        log.info("事件驱动回测结束。")
        return self.portfolio.get_performance_report()


# --- 使用示例 (来自 v3) ---
# 为了演示，我们需要创建上述抽象基类的具体实现
# 这部分代码通常会更复杂，这里提供一个极简的框架

class SimpleDataHandler(DataHandler):
    """【增强版】一个简单的历史数据处理器，包含价格和成交量"""
    def __init__(self, events_queue, all_prices_df: pd.DataFrame, all_volumes_df: pd.DataFrame):
        self.events = events_queue
        # 确保时间索引是 datetime 类型
        all_prices_df.index = pd.to_datetime(all_prices_df.index)
        all_volumes_df.index = pd.to_datetime(all_volumes_df.index)
        
        self.all_prices = all_prices_df
        self.all_volumes = all_volumes_df # 新增成交量数据

        self.latest_prices = {}
        self.latest_volumes = {}
        self.latest_data = {} # 保持兼容性
        self.price_iterator = self.all_prices.iterrows()
        self.volume_iterator = self.all_volumes.iterrows()
        self.continue_backtest = True

    def update_bars(self):
        """【增强版】推送下一个时间点的MarketEvent，同步价格和成交量"""
        try:
            timestamp, price_series = next(self.price_iterator)
            _, volume_series = next(self.volume_iterator)
            
            # 存储最新的bar数据
            self.latest_prices = price_series.to_dict()
            self.latest_volumes = volume_series.to_dict()
            self.latest_data = {'timestamp': timestamp, **self.latest_prices} # For compatibility
            
            # 推送带有时间戳的事件
            self.events.put(MarketEvent(timestamp))
        except StopIteration:
            self.continue_backtest = False

    def get_latest_bar_value(self, ts_code, val_type='close'):
        """【增强版】获取最新一个时间点的数据值"""
        if val_type == 'close':
            return self.latest_prices.get(ts_code, np.nan)
        elif val_type == 'volume':
            return self.latest_volumes.get(ts_code, np.nan)
        return np.nan

    def get_historical_bars(self, ts_code, N: int, timestamp, bar_type='price'):
        """【增强版】获取到指定时间戳为止的N条历史数据"""
        try:
            if bar_type == 'price':
                return self.all_prices[ts_code].loc[:timestamp].tail(N)
            elif bar_type == 'volume':
                return self.all_volumes[ts_code].loc[:timestamp].tail(N)
        except (KeyError, IndexError):
            return None
        return None

    def get_average_volume(self, ts_code: str, N: int, timestamp) -> float:
        """【新增】获取到指定时间戳为止的N日平均成交量"""
        volumes = self.get_historical_bars(ts_code, N, timestamp, bar_type='volume')
        if volumes is not None and not volumes.empty:
            return volumes.mean()
        return 0.0

class SimpleMovingAverageStrategy(Strategy):
    """一个简单的双均线交叉策略"""
    def __init__(self, data_handler: SimpleDataHandler, short_window=10, long_window=30):
        self.data_handler = data_handler
        self.short_window = short_window
        self.long_window = long_window
        # 持仓状态机，避免重复发送信号
        self.bought = {code: 'OUT' for code in self.data_handler.all_prices.columns}

    def calculate_signals(self, event: MarketEvent):
        """在每个市场事件发生时，为所有股票计算均线交叉信号"""
        if not isinstance(event, MarketEvent):
            return

        for code in self.data_handler.all_prices.columns:
            # 使用 data_handler 获取历史数据
            hist_prices = self.data_handler.get_historical_bars(code, self.long_window, event.timestamp, bar_type='price')

            if hist_prices is None or len(hist_prices) < self.long_window:
                continue

            short_ma = hist_prices.tail(self.short_window).mean()
            long_ma = hist_prices.mean() # long_window is the length of hist_prices

            # 金叉买入信号
            if short_ma > long_ma and self.bought[code] == 'OUT':
                log.info(f"[{event.timestamp}] 买入信号: {code} (金叉)")
                self.events.put(SignalEvent(code, 'LONG'))
                self.bought[code] = 'IN'
            # 死叉卖出信号
            elif short_ma < long_ma and self.bought[code] == 'IN':
                log.info(f"[{event.timestamp}] 卖出信号: {code} (死叉)")
                self.events.put(SignalEvent(code, 'EXIT'))
                self.bought[code] = 'OUT'

# --- V2.0新增：行业轮动策略 ---
class IndustryRotationStrategy:
    """
    一个简单的行业轮动策略。
    1. 使用IndustryAnalyzer选出排名靠前的行业。
    2. 在选出的行业内，使用另一个因子（如动量）选出个股。
    """
    def __init__(self, industry_analyzer, factor_factory, factor_processor):
        self.ia = industry_analyzer
        self.ff = factor_factory
        self.fp = factor_processor

    def select_portfolio(self, date: str, industry_factor: str, stock_factor: str, top_n_industries: int, top_n_stocks_per_industry: int) -> pd.Series:
        """
        执行选股逻辑。
        :return: 包含最终股票池和综合得分的Series
        """
        # 1. 行业筛选
        top_industries = self.ia.get_industry_factor_rank(date, industry_factor).head(top_n_industries)
        log.info(f"选出的Top行业:\n{top_industries}")
        
        # 2. 在选定行业内进行个股选择
        stock_pool_df = self.ia.stock_basics[self.ia.stock_basics['industry'].isin(top_industries.index)]
        
        # 计算个股因子
        start_date = (pd.to_datetime(date) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        stock_factor_values = {}
        for code in stock_pool_df['ts_code']:
            try:
                stock_factor_values[code] = getattr(self.ff, f"calc_{stock_factor}")(ts_code=code, start_date=start_date, end_date=date)
            except Exception:
                continue
        
        stock_factor_series = pd.Series(stock_factor_values).dropna()
        
        # 3. 预处理并选出最终股票
        if stock_factor_series.empty:
            return pd.Series(dtype=float)
            
        processed_stock_factor = self.fp.process_factor(stock_factor_series, neutralize=False) # 在行业内选股，通常不进行行业中性化
        
        final_portfolio = pd.Series(dtype=float)
        df_for_selection = pd.DataFrame({'factor': processed_stock_factor}).merge(
            self.ia.stock_basics[['ts_code', 'industry']], left_index=True, right_on='ts_code'
        )
        
        for industry in top_industries.index:
            industry_stocks = df_for_selection[df_for_selection['industry'] == industry]
            top_stocks = industry_stocks.sort_values('factor', ascending=False).head(top_n_stocks_per_industry)
            final_portfolio = pd.concat([final_portfolio, top_stocks.set_index('ts_code')['factor']])
            
        return final_portfolio

# 主程序入口示例
class HistoricalScorer:
    """【V2.4 新增】历史得分计算器"""
    def __init__(self, data_manager: data.DataManager, factor_processor: FactorProcessor):
        self.dm = data_manager
        self.fp = factor_processor
        self.stock_basics = self.dm.get_stock_basic()

    def calculate_financial_score_series(self, ts_code: str) -> pd.Series:
        """计算单只股票的历史财务得分序列"""
        all_fina = self.dm.get_fina_indicator(ts_code=ts_code)
        if all_fina is None or all_fina.empty:
            return pd.Series(dtype=float)

        # 筛选核心财务因子并处理方向性
        # 得分越高越好
        score_factors = {
            'roe': 1, 
            'netprofit_yoy': 1, 
            'or_yoy': 1,
            'debt_to_assets': -1 # 负债率越低越好
        }
        
        # 仅保留需要的列并转换为数值
        fina_df = all_fina[['end_date'] + list(score_factors.keys())].copy()
        for col in score_factors:
            fina_df[col] = pd.to_numeric(fina_df[col], errors='coerce')
        
        fina_df = fina_df.dropna().set_index('end_date').sort_index()
        if fina_df.empty:
            return pd.Series(dtype=float)

        # 对每个因子在历史序列中进行百分位排名 (0-1)
        ranked_df = fina_df.rank(pct=True)
        
        # 根据方向性调整得分
        for factor, direction in score_factors.items():
            if direction == -1:
                ranked_df[factor] = 1 - ranked_df[factor]
        
        # 等权合成最终得分 (0-100)
        final_score = ranked_df.mean(axis=1) * 100
        return final_score.rename("财务得分")

    def calculate_fund_score_series(self, ts_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算单只股票的历史资金得分序列"""
        query = f"""
            SELECT trade_date, factor_name, factor_value
            FROM factors_exposure
            WHERE ts_code = '{ts_code}' 
              AND trade_date BETWEEN '{start_date}' AND '{end_date}'
              AND factor_name IN ('momentum', 'net_inflow_ratio', 'volatility')
        """
        with self.dm.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return pd.Series(dtype=float)

        df_wide = df.pivot(index='trade_date', columns='factor_name', values='factor_value')
        df_wide['volatility'] = df_wide['volatility'] * -1 # 波动率越低越好
        
        # 对每日的因子值进行横向加权（此处简化为等权）
        daily_score = df_wide.mean(axis=1)
        
        # 对日度得分进行平滑处理，例如7日移动平均
        smoothed_score = daily_score.rolling(window=7, min_periods=1).mean()
        
        # 归一化到 0-100
        normalized_score = (smoothed_score - smoothed_score.min()) / (smoothed_score.max() - smoothed_score.min()) * 100
        return normalized_score.rename("资金得分")

    def calculate_macro_score_series(self, start_date: str, end_date: str) -> pd.Series:
        """计算宏观景气度得分序列"""
        start_m = pd.to_datetime(start_date).strftime('%Y%m')
        end_m = pd.to_datetime(end_date).strftime('%Y%m')

        df_pmi = self.dm.get_cn_pmi(start_m, end_m)
        df_m = self.dm.get_cn_m(start_m, end_m)

        if df_pmi is None or df_m is None or df_pmi.empty or df_m.empty:
            return pd.Series(dtype=float)

        # 【鲁棒性修复】将列名统一转为小写，以兼容Tushare接口可能的大小写变动
        df_pmi.columns = [col.lower() for col in df_pmi.columns]
        df_m.columns = [col.lower() for col in df_m.columns]

        df_pmi['month'] = pd.to_datetime(df_pmi['month'], format='%Y%m')
        df_m['month'] = pd.to_datetime(df_m['month'], format='%Y%m')
        df_macro = pd.merge(df_pmi[['month', 'pmi010000']], df_m[['month', 'm1_yoy', 'm2_yoy']], on='month')
        
        df_macro = df_macro.set_index('month').sort_index()
        df_macro['pmi'] = pd.to_numeric(df_macro['pmi010000'])
        df_macro['m1_m2_gap'] = df_macro['m1_yoy'] - df_macro['m2_yoy']

        # 对PMI和M1-M2剪刀差进行Z-Score标准化
        pmi_z = (df_macro['pmi'] - df_macro['pmi'].mean()) / df_macro['pmi'].std()
        m_gap_z = (df_macro['m1_m2_gap'] - df_macro['m1_m2_gap'].mean()) / df_macro['m1_m2_gap'].std()

        # 等权合成宏观分
        macro_score = (pmi_z + m_gap_z) / 2
        
        # 归一化到 0-100
        normalized_score = (macro_score - macro_score.min()) / (macro_score.max() - macro_score.min()) * 100
        return normalized_score.rename("宏观景气分")


class LSTMAlphaStrategy:
    """
    【V2.4新增】基于LSTM的时序预测策略框架。
    - 实现了 "AI策略大脑进化" 的规划。
    """
    def __init__(self, data_manager: data.DataManager, lookback_window=60, prediction_horizon=5):
        self.dm = data_manager
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.model = None # 此处应为一个PyTorch或TensorFlow模型实例
        self.scaler = None # 用于数据归一化的Scaler

    def _prepare_lstm_data(self, ts_code: str, end_date: str) -> np.ndarray | None:
        """为单只股票准备LSTM模型的输入数据 (X)"""
        from sklearn.preprocessing import MinMaxScaler
        
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=self.lookback_window * 2)).strftime('%Y%m%d')
        df = self.dm.get_adjusted_daily(ts_code, start_date, end_date, adj='hfq')
        
        if df is None or len(df) < self.lookback_window:
            return None
            
        # 使用收盘价和成交量作为特征
        features = df[['close', 'vol']].tail(self.lookback_window)
        
        # 归一化
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = self.scaler.fit_transform(features)
        
        # 塑造成LSTM需要的 [samples, timesteps, features] 形状
        return np.array([scaled_features])

    def train(self, ts_code: str, training_data: pd.DataFrame):
        """
        （示意）训练LSTM模型的逻辑。
        在实际应用中，这将是一个复杂的过程，包括构建模型、定义损失函数和优化器。
        """
        log.info(f"正在为 {ts_code} 训练LSTM模型... (示意流程)")
        # 1. 数据准备 (创建 X 和 y)
        # 2. 构建PyTorch/TensorFlow模型
        # 3. 循环训练
        # self.model = trained_model
        pass

    def predict_future_return(self, ts_code: str, end_date: str) -> float:
        """
        【V2.4优化】使用“已训练好”的LSTM模型，预测未来N天的预期收益率。
        优化点：将scaler的使用本地化，避免类级别的状态污染。
        """
        # _prepare_lstm_data 现在应该返回 X_pred 和 它所使用的 scaler
        X_pred_data = self._prepare_lstm_data(ts_code, end_date)
        if X_pred_data is None:
            return np.nan

        X_pred, local_scaler = X_pred_data # 解包
        if X_pred is None:
            return np.nan
        
        # 假设模型已加载并可进行预测
        # predicted_scaled_price = self.model.predict(X_pred)
        # 此处使用一个模拟值代替真实模型预测
        predicted_scaled_price = X_pred[0, -1, 0] * (1 + np.random.randn() * 0.05)
        
        # 反归一化得到预测的价格
        current_price = local_scaler.inverse_transform(np.array([[X_pred[0, -1, 0], 0]]))[0, 0]
        predicted_price = local_scaler.inverse_transform(np.array([[predicted_scaled_price, 0]]))[0, 0]

        return (predicted_price / current_price) - 1

    # 同时需要修改 _prepare_lstm_data 以返回 scaler
    def _prepare_lstm_data(self, ts_code: str, end_date: str) -> tuple[np.ndarray, any] | None:
        """为单只股票准备LSTM模型的输入数据 (X)"""
        from sklearn.preprocessing import MinMaxScaler
        
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=self.lookback_window * 2)).strftime('%Y%m%d')
        df = self.dm.get_adjusted_daily(ts_code, start_date, end_date, adj='hfq')
        
        if df is None or len(df) < self.lookback_window:
            return None, None
            
        features = df[['close', 'vol']].tail(self.lookback_window)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # 塑造成LSTM需要的 [samples, timesteps, features] 形状
        return np.array([scaled_features]), scaler


if __name__ == '__main__':
    # #################################################
    # 演示【源于 main.py】的主工作流
    # #################################################
    print("\n" + "#"*20 + "  演示主控工作流  " + "#"*20)
    run_main_workflow()
    print("#"*60 + "\n")


    # #################################################
    # 演示【源于 factor_analyzer.py】的因子分析功能
    # #################################################
    print("\n" + "#"*20 + "  演示因子分析功能  " + "#"*20)
    import data
    # 1. 准备工作：实例化组件并计算一个因子
    log.info("【步骤1】初始化组件...")
    dm = data.DataManager()
    ff = FactorFactory(_data_manager=dm)
    fp = FactorProcessor(_data_manager=dm)
    analyzer = FactorAnalyzer(_data_manager=dm)

    log.info("\n【步骤2】计算一个动量因子作为示例...")
    stock_pool = ['000001.SZ', '600519.SH', '300750.SZ', '601318.SH', '000651.SZ']
    dates = pd.to_datetime(pd.date_range('2024-01-01', '2024-06-30', freq='M'))
    
    momentum_factor = pd.DataFrame()
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        start_date_str = (date - pd.Timedelta(days=60)).strftime('%Y%m%d')
        raw_values = {s: ff.calc_momentum(s, start_date_str, date_str) for s in stock_pool}
        momentum_factor[date] = pd.Series(raw_values)
    
    momentum_factor = momentum_factor.T
    momentum_factor.index.name = 'trade_date'
    log.info("动量因子原始数据 (部分):\n%s", momentum_factor.head())

    # 2. 核心功能演示
    log.info("\n【功能1】演示因子存储与加载...")
    analyzer.save_factors_to_db(momentum_factor, 'factor_momentum_example')
    loaded_factor = analyzer.load_factors_from_db('factor_momentum_example')
    log.info("从数据库加载的因子数据 (部分):\n%s", loaded_factor.head())

    log.info("\n【功能2】演示IC/IR分析...")
    ic_values = {}
    for date, factor_slice in loaded_factor.iterrows():
        ic, p = analyzer.calculate_ic(factor_slice.dropna())
        if not np.isnan(ic):
            ic_values[date] = ic
    
    ic_series = pd.Series(ic_values)
    information_ratio = analyzer.calculate_ir(ic_series)
    log.info(f"动量因子IC均值: {ic_series.mean():.4f}")
    log.info(f"动量因子IR: {information_ratio:.4f}")
    log.info("IC时间序列:\n%s", ic_series)

    log.info("\n【功能3】演示分层回测...")
    layered_results, fig = analyzer.run_layered_backtest(loaded_factor, num_quantiles=3, forward_return_period=20)
    log.info("分层回测结果 (部分):\n%s", layered_results.head())
    log.info("分层回测图表已生成。在实际应用中，此图表可直接在Streamlit中展示。")
    print("#"*60 + "\n")


    # #################################################
    # 演示【高级组合优化器】
    # #################################################
    print("\n" + "#"*20 + "  演示高级组合优化器  " + "#"*20)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    np.random.seed(42)
    mock_returns = pd.Series(np.random.rand(5) / 10, index=tickers)
    mock_cov = pd.DataFrame(np.random.rand(5, 5) / 100, index=tickers, columns=tickers).abs()
    mock_cov = mock_cov + mock_cov.T / 2
    np.fill_diagonal(mock_cov.values, np.random.rand(5)/20)
    
    mock_industry_map = pd.DataFrame({
        'ts_code': tickers,
        'industry': ['科技', '科技', '科技', '零售', '汽车']
    })
    
    optimizer = AdvancedPortfolioOptimizer(mock_returns, mock_cov, mock_industry_map)
    optimized_w = optimizer.optimize(max_weight_per_stock=0.3, industry_max_exposure=0.6)
    
    log.info("优化后的权重:\n%s", optimized_w)
    if optimized_w is not None:
        tech_sum = optimized_w.join(mock_industry_map.set_index('ts_code')).groupby('industry')['weight'].sum().get('科技', 0)
        log.info(f"\n科技行业总权重: {tech_sum:.4f}")
    print("#"*60 + "\n")

    log.info("事件驱动回测框架已定义，可供后续扩展使用。")