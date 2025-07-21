# StockScanAI/run_daily_pipeline.py
#
# 【V2.2 核心优化】
# 目的：将数据抽取和因子计算两个独立的后台任务，合并为一个统一、健壮的数据管道。
#      消除了对本地HDF5缓存文件的依赖，实现了数据在内存中的无缝流转。

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text

# 导入项目模块
import data
import quant_engine
from logger_config import log
from progress_tracker import ProgressTracker
from system_check_error_handler import safe_execute, retry_on_failure, CheckResult

# 【V2.2 重构】将因子列表的定义移入此统一管道脚本中，移除对旧文件的依赖
@dataclass
class PipelineConfig:
    """数据管道配置常量 - 使用dataclass提高类型安全性"""
    
    # 因子列表
    FACTORS_TO_CALCULATE: List[str] = field(default_factory=lambda: [
        "pe_ttm", "roe", "growth_revenue_yoy", "debt_to_assets",
        "momentum", "volatility", "net_inflow_ratio",
        "holder_num_change_ratio", "major_shareholder_net_buy_ratio",
        "top_list_net_buy_amount", "dividend_yield", "forecast_growth_rate",
        "repurchase_ratio", "block_trade_ratio"
    ])
    
    # 时间配置
    MARKET_CLOSE_TIME: str = "15:30"
    LOOKBACK_DAYS: int = 90
    CALENDAR_BUFFER_DAYS: int = 30
    CALENDAR_FUTURE_DAYS: int = 5
    
    # 数据质量阈值
    MIN_TRADING_DAYS: int = 55
    MIN_FACTOR_COUNT_THRESHOLD: int = 100
    
    # 批处理配置
    DOWNLOAD_CHUNK_SIZE: int = 150
    DB_CHUNK_SIZE: int = 500
    DB_WRITE_CHUNK_SIZE: int = 10000
    MAX_RETRIES: int = 3
    
    # 历史数据起始日期
    HISTORICAL_START_DATE: str = "20100101"
    
    # 因子计算常量
    MOMENTUM_LOOKBACK_DAYS: int = 21
    VOLATILITY_LOOKBACK_DAYS: int = 20
    TRADING_DAYS_PER_YEAR: int = 252
    
    @classmethod
    def create_for_environment(cls, env: str = "production") -> 'PipelineConfig':
        """Create configuration for specific environment"""
        if env == "development":
            return cls(
                DOWNLOAD_CHUNK_SIZE=50,  # Smaller chunks for dev
                DB_CHUNK_SIZE=100,
                MAX_RETRIES=1,
                MIN_TRADING_DAYS=20  # Less strict for dev
            )
        elif env == "testing":
            return cls(
                FACTORS_TO_CALCULATE=["pe_ttm", "momentum"],  # Only basic factors
                DOWNLOAD_CHUNK_SIZE=10,
                DB_CHUNK_SIZE=50,
                MAX_RETRIES=1,
                LOOKBACK_DAYS=30
            )
        else:  # production
            return cls()

def _batch_fetch_financial_data(dm, ts_codes: List[str]) -> pd.DataFrame:
    """
    优化的批量财务数据获取函数
    使用并发处理和智能缓存
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from progress_tracker import ProgressTracker
    
    log.info("【性能优化】开始批量预取全市场财务数据...")
    
    try:
        all_fina_data_list = []
        failed_codes = []
        
        # 使用进度跟踪器
        tracker = ProgressTracker(len(ts_codes), "财务数据预取")
        
        def fetch_single_stock_financial_data(code: str) -> Optional[pd.DataFrame]:
            """获取单只股票的财务数据"""
            try:
                df_fina = dm.get_fina_indicator(ts_code=code, force_update=False)
                return df_fina if df_fina is not None and not df_fina.empty else None
            except Exception as e:
                log.debug(f"股票 {code} 财务数据获取失败: {e}")
                return None
        
        # 使用线程池并发处理（I/O密集型任务）
        max_workers = min(4, len(ts_codes) // 100 + 1)  # 动态调整线程数
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(fetch_single_stock_financial_data, code): code 
                for code in ts_codes
            }
            
            # 收集结果
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_fina_data_list.append(result)
                        tracker.update(1, success=True)
                    else:
                        failed_codes.append(code)
                        tracker.update(1, success=False)
                except Exception as e:
                    log.debug(f"处理股票 {code} 结果时出错: {e}")
                    failed_codes.append(code)
                    tracker.update(1, success=False)
        
        # 完成进度跟踪
        summary = tracker.finish()
        
        # 合并数据
        if all_fina_data_list:
            all_fina_data = pd.concat(all_fina_data_list, ignore_index=True)
            # 去重并排序
            all_fina_data = all_fina_data.drop_duplicates(subset=['ts_code', 'end_date']).sort_values(['ts_code', 'end_date'])
            log.info(f"✅ 批量预取完成: {len(all_fina_data)} 条财务记录，成功率: {summary['success_rate']:.1%}")
        else:
            all_fina_data = pd.DataFrame()
            log.warning("⚠️ 未能成功预取任何财务数据")
        
        if failed_codes:
            log.warning(f"⚠️ {len(failed_codes)} 只股票财务数据获取失败")
        
        return all_fina_data
        
    except Exception as e:
        log.error(f"❌ 批量预取财务数据失败: {e}", exc_info=True)
        return pd.DataFrame()


FACTORS_TO_CALCULATE = PipelineConfig().FACTORS_TO_CALCULATE


def extract_data(trade_date: str, config: Optional[PipelineConfig] = None) -> dict:
    """
    步骤一：执行所有耗时的数据抽取和预处理工作。
    【V2.7 增强版 - 增加进度显示】
    
    Args:
        trade_date: 交易日期 (YYYYMMDD格式)
        config: 管道配置，如果为None则使用默认配置
        
    Returns:
        dict: 包含所有抽取数据的字典
    """
    if config is None:
        config = PipelineConfig()
    log.info("=" * 60)
    log.info(f"📊 开始为 {trade_date} 抽取全市场原始数据")
    log.info("=" * 60)

    dm = data.DataManager()
    stock_list = dm.get_stock_basic()
    ts_codes = stock_list["ts_code"].tolist()
    
    log.info(f"🎯 目标股票数量: {len(ts_codes)} 只")

    # --- 1. 批量获取截面数据 ---
    log.info("📈 获取当日截面数据...")
    
    data_sources = [
        ("基本指标", lambda: dm.pro.daily_basic(trade_date=trade_date)),
        ("资金流向", lambda: dm.pro.moneyflow(trade_date=trade_date)),
        ("龙虎榜", lambda: dm.pro.top_list(trade_date=trade_date)),
        ("大宗交易", lambda: dm.pro.block_trade(trade_date=trade_date))
    ]
    
    results = {}
    for name, func in data_sources:
        try:
            log.info(f"  📊 获取{name}数据...")
            result_data = func()
            results[name] = result_data
            count = len(result_data) if result_data is not None and not result_data.empty else 0
            log.info(f"  ✅ {name}: {count} 条记录")
        except Exception as e:
            log.warning(f"  ⚠️  {name}获取失败: {e}")
            results[name] = pd.DataFrame()
    
    daily_basics_df = results.get("基本指标", pd.DataFrame())
    money_flow_df = results.get("资金流向", pd.DataFrame())
    top_list_df = results.get("龙虎榜", pd.DataFrame())
    block_trade_df = results.get("大宗交易", pd.DataFrame())

    # --- 2. 【缓存优先】获取时序价格数据 ---
    log.info("  开始获取各股票的历史价格 (缓存优先)...")
    start_date_lookback = (pd.to_datetime(trade_date) - timedelta(days=90)).strftime("%Y%m%d")
    prices_dict = {}

# 步骤 A: 检查数据库缓存
    try:
        min_trading_days = 55
        # V3.1 终极健壮性修复：在SQL查询中明确进行日期类型转换，彻底解决缓存检查失效问题
        query = text("""
            SELECT ts_code FROM ts_daily
            WHERE trade_date BETWEEN TO_DATE(:start_date, 'YYYYMMDD') AND TO_DATE(:end_date, 'YYYYMMDD') 
              AND ts_code = ANY(:ts_codes)
            GROUP BY ts_code 
            HAVING COUNT(trade_date) >= :min_days
        """)
        with dm.engine.connect() as conn:
            cached_stocks_result = conn.execute(query, {
                "start_date": start_date_lookback, "end_date": trade_date,
                "ts_codes": ts_codes, "min_days": PipelineConfig.MIN_TRADING_DAYS
            }).fetchall()
        cached_stocks = {row[0] for row in cached_stocks_result}
        log.info(f"    数据库缓存检查完成：{len(cached_stocks)}/{len(ts_codes)} 只股票已有完整本地数据。")
    except Exception as e:
        log.warning(f"    检查缓存失败: {e}。将尝试全量下载。")
        cached_stocks = set()

    # 步骤 B: 仅下载无缓存的数据 (带重试机制)
    stocks_to_download = sorted(list(set(ts_codes) - cached_stocks))
    if stocks_to_download:
        log.info(f"    需要从网络下载 {len(stocks_to_download)} 只股票的数据...")
        
        # --- 【新增】完整性校验与重试逻辑 ---
        downloaded_data_raw = {}
        max_retries = PipelineConfig.MAX_RETRIES  # V2.9 提升健壮性：增加上层重试次数
        for attempt in range(max_retries + 1):
            # V2.9 修正：每次重试时，都需要重新计算还需要下载的列表
            needed = [
                code
                for code in stocks_to_download
                if code not in downloaded_data_raw
                or downloaded_data_raw[code] is None
                or downloaded_data_raw[code].empty
            ]
            if not needed:
                break
            
            if attempt > 0:
                log.warning(f"    第 {attempt} 次重试，下载剩余的 {len(needed)} 只股票...")

            chunk_size = PipelineConfig.DOWNLOAD_CHUNK_SIZE
            for i in range(0, len(needed), chunk_size):
                chunk = needed[i : i + chunk_size]
                log.info(f"      正在下载块 {i//chunk_size + 1}/{len(needed)//chunk_size + 1} (股票 {i+1}-{i+len(chunk)})...")
                chunk_results = dm.run_batch_download(chunk, start_date_lookback, trade_date)
                downloaded_data_raw.update(chunk_results)
            
        # 校验最终结果
        missing_stocks = [code for code in stocks_to_download if code not in downloaded_data_raw or downloaded_data_raw[code] is None or downloaded_data_raw[code].empty]
        if missing_stocks:
            log.error(f"    警告：经过 {max_retries} 次重试后，仍有 {len(missing_stocks)} 只股票数据下载失败。")
        # --- 校验逻辑结束 ---

        for code, df in downloaded_data_raw.items():
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                prices_dict[code] = df.set_index("trade_date")["close"]

    # 步骤 C: 从数据库加载已缓存的数据
    if cached_stocks:
        log.info(f"    正在从数据库分批加载 {len(cached_stocks)} 只已缓存股票的数据...")
        # (此处逻辑不变)
        cached_list = list(cached_stocks)
        chunk_size = PipelineConfig.DB_CHUNK_SIZE
        for i in range(0, len(cached_list), chunk_size):
            chunk = cached_list[i:i+chunk_size]
            query = text("""
                SELECT ts_code, trade_date, close FROM ts_daily
                WHERE trade_date BETWEEN :start_date AND :end_date AND ts_code = ANY(:ts_codes)
            """)
            with dm.engine.connect() as conn:
                cached_df = pd.read_sql(query, conn, params={
                    "start_date": start_date_lookback, "end_date": trade_date, "ts_codes": chunk
                })
            
            if not cached_df.empty:
                cached_df['trade_date'] = pd.to_datetime(cached_df['trade_date'])
                for code, group in cached_df.groupby('ts_code'):
                    prices_dict[code] = group.set_index('trade_date')['close']

    # --- 3. 合并与格式化 ---
    daily_prices_df = pd.DataFrame(prices_dict)

    log.info("【数据抽取】所有原始数据提取完毕。")

    return {
        "stock_list": stock_list,
        "ts_codes": ts_codes,
        "daily_prices_df": daily_prices_df,
        "daily_basics_df": daily_basics_df,
        "money_flow_df": money_flow_df,
        "top_list_df": top_list_df,
        "block_trade_df": block_trade_df,
    }


def calculate_and_save_factors(
    trade_date: str, 
    raw_data: Dict[str, Any], 
    all_fina_data: pd.DataFrame,
    config: Optional[PipelineConfig] = None
) -> bool:
    """
    步骤二：执行因子计算与存储。
    
    Args:
        trade_date: 交易日期 YYYYMMDD格式
        raw_data: 从 extract_data 函数获取的原始数据字典
        all_fina_data: 预先获取的全市场财务数据
        config: 管道配置，如果为None则使用默认配置
        
    Returns:
        bool: 成功返回True，失败返回False
        
    Raises:
        Exception: 当数据写入数据库失败时
    """
    if config is None:
        config = PipelineConfig()
    log.info("【因子计算】开始混合模式计算所有因子...")
    dm = data.DataManager()
    ff = quant_engine.FactorFactory(_data_manager=dm)
    trade_date_dt = pd.to_datetime(trade_date)

    # 从传入的字典中解包数据
    ts_codes = raw_data["ts_codes"]
    daily_prices_df = raw_data["daily_prices_df"]
    daily_basics_df = raw_data["daily_basics_df"]
    money_flow_df = raw_data["money_flow_df"]

    results = {}

    # --- 第一部分：高性能的向量化计算 ---
    log.info("  正在向量化计算：价格类与截面类因子...")
    vectorized_factors = ["pe_ttm", "momentum", "volatility", "net_inflow_ratio"]
    financial_factors = ["roe", "growth_revenue_yoy", "debt_to_assets"] # 定义财务因子

    if daily_prices_df is not None and not daily_prices_df.empty:
        prices = daily_prices_df.ffill()
        # 动量因子计算 (21日动量)
        if len(prices) >= config.MOMENTUM_LOOKBACK_DAYS:
            results["momentum"] = (
                prices.iloc[-1] / prices.iloc[-config.MOMENTUM_LOOKBACK_DAYS] - 1
            )
        
        # 波动率因子计算 (20日波动率，年化)
        if len(prices) >= config.VOLATILITY_LOOKBACK_DAYS:
            log_returns = np.log(prices / prices.shift(1))
            results["volatility"] = (
                log_returns.iloc[-config.VOLATILITY_LOOKBACK_DAYS:].std() * 
                np.sqrt(config.TRADING_DAYS_PER_YEAR)
            )

    if daily_basics_df is not None and not daily_basics_df.empty:
        basics = daily_basics_df.set_index("ts_code")
        results["pe_ttm"] = basics["pe_ttm"]
        if money_flow_df is not None and not money_flow_df.empty:
            flow = money_flow_df.drop(columns=["trade_date"]).set_index("ts_code")
            combined = basics.join(flow, how="inner")
            if not combined.empty and "amount" in combined.columns:
                amount_yuan = combined["amount"] * 1000
                net_inflow_yuan = (
                    combined["buy_lg_amount"] - combined["sell_lg_amount"]
                ) * 10000
                results["net_inflow_ratio"] = net_inflow_yuan.divide(
                    amount_yuan
                ).fillna(0)

    # --- 【性能优化】第二部分：基于预取数据的财务因子计算 ---
    log.info("  正在基于预取数据计算：财务类因子...")
    if all_fina_data is not None and not all_fina_data.empty:
        # 对预取的所有财务数据进行一次PIT筛选
        try:
            # Use group_keys=False for pandas compatibility and better performance
            pit_fina_data = all_fina_data.groupby('ts_code', group_keys=False).apply(
                lambda x: dm.get_pit_financial_data(x, trade_date)
            ).reset_index(drop=True)
            
            if pit_fina_data.empty:
                log.warning("PIT筛选后无有效财务数据")
            else:
                log.info(f"PIT筛选完成: {len(pit_fina_data)} 条有效财务记录")
                
        except Exception as e:
            log.error(f"PIT筛选失败: {e}")
            pit_fina_data = pd.DataFrame()
        
        pit_fina_data = pit_fina_data.set_index('ts_code')
        results["roe"] = pit_fina_data["roe"]
        results["growth_revenue_yoy"] = pit_fina_data["or_yoy"] # Tushare中字段为 or_yoy
        results["debt_to_assets"] = pit_fina_data["debt_to_assets"]


    # --- 第三部分：剩余因子统一循环计算 ---
    log.info("  正在循环计算：筹码与价值类因子...")
    params = {
        "date": trade_date,
        "start_date": (trade_date_dt - timedelta(days=90)).strftime("%Y%m%d"),
        "end_date": trade_date,
        "top_list_df": raw_data["top_list_df"],
        "block_trade_df": raw_data["block_trade_df"],
    }
    loop_factors = [f for f in FACTORS_TO_CALCULATE if f not in vectorized_factors and f not in financial_factors]

    for factor in loop_factors:
        log.info(f"    正在计算因子: {factor}...")
        factor_series = pd.Series(index=ts_codes, dtype=float)
        for code in ts_codes:
            params["ts_code"] = code
            factor_series[code] = ff.calculate(factor, **params)
        results[factor] = factor_series

    # --- 第四部分：整合与存储 ---
    log.info("【数据入库】开始存储因子数据...")
    final_df = pd.DataFrame(results).reset_index().rename(columns={"index": "ts_code"})
    long_df = final_df.melt(
        id_vars="ts_code", var_name="factor_name", value_name="factor_value"
    ).dropna()
    long_df["trade_date"] = pd.to_datetime(trade_date)

    if long_df.empty:
        log.warning("没有有效的因子数据可以存入数据库")
        return False

    try:
        with dm.engine.connect() as connection:
            with connection.begin():
                delete_sql = text(
                    "DELETE FROM factors_exposure WHERE trade_date = :trade_date"
                )
                connection.execute(delete_sql, {"trade_date": trade_date})
                long_df.to_sql(
                    "factors_exposure",
                    connection,
                    if_exists="append",
                    index=False,
                    chunksize=config.DB_WRITE_CHUNK_SIZE,
                )
        log.info(f"✅ 成功为 {trade_date} 写入 {len(long_df)} 条因子数据")
        return True
        
    except Exception as e:
        log.critical(f"❌ 因子数据写入数据库时发生严重错误: {e}", exc_info=True)
        return False


@retry_on_failure(max_retries=2, delay=1.0)
def get_latest_available_trade_date(dm: data.DataManager) -> str:
    """
    智能获取最新可用的交易日期
    
    逻辑：
    1. 如果今天是交易日且已过收盘时间(15:30)，返回今天
    2. 如果今天是交易日但未到收盘时间，返回上一个交易日
    3. 如果今天不是交易日（周末/节假日），返回最近的交易日
    
    Returns:
        str: 最新可用交易日期 (YYYYMMDD格式)
    """
    now = datetime.now()
    current_date = now.strftime("%Y%m%d")
    
    # 获取交易日历
    trade_dates = _get_trade_calendar(dm, now)
    if not trade_dates:
        log.warning("无法获取交易日历，使用当前日期")
        return current_date
    
    # 判断交易日期
    if current_date in trade_dates:
        return _handle_trading_day(now, current_date, trade_dates)
    else:
        return _handle_non_trading_day(current_date, trade_dates)


def _get_trade_calendar(dm: data.DataManager, now: datetime) -> List[str]:
    """获取交易日历"""
    start_date = (now - timedelta(days=PipelineConfig.CALENDAR_BUFFER_DAYS)).strftime("%Y%m%d")
    end_date = (now + timedelta(days=PipelineConfig.CALENDAR_FUTURE_DAYS)).strftime("%Y%m%d")
    
    cal_df = dm.pro.trade_cal(exchange="", start_date=start_date, end_date=end_date)
    
    if cal_df is None or cal_df.empty:
        return []
    
    trade_dates = cal_df[cal_df["is_open"] == 1]["cal_date"].tolist()
    trade_dates.sort()
    return trade_dates


def _handle_trading_day(now: datetime, current_date: str, trade_dates: List[str]) -> str:
    """处理当天是交易日的情况"""
    current_time = now.time()
    market_close_time = datetime.strptime(PipelineConfig.MARKET_CLOSE_TIME, "%H:%M").time()
    
    if current_time >= market_close_time:
        log.info(f"今天({current_date})是交易日且已过收盘时间，使用今天作为目标日期")
        return current_date
    else:
        # 获取上一个交易日
        try:
            current_index = trade_dates.index(current_date)
            if current_index > 0:
                prev_trade_date = trade_dates[current_index - 1]
                log.info(f"今天({current_date})是交易日但未到收盘时间，使用上一交易日: {prev_trade_date}")
                return prev_trade_date
            else:
                log.info(f"今天是第一个交易日，使用今天: {current_date}")
                return current_date
        except ValueError:
            log.warning(f"在交易日列表中未找到今天({current_date})，使用最近交易日")
            return max([d for d in trade_dates if d <= current_date], default=current_date)


def _handle_non_trading_day(current_date: str, trade_dates: List[str]) -> str:
    """处理当天不是交易日的情况"""
    recent_trade_dates = [d for d in trade_dates if d <= current_date]
    if recent_trade_dates:
        latest_trade_date = max(recent_trade_dates)
        log.info(f"今天({current_date})不是交易日，使用最近的交易日: {latest_trade_date}")
        return latest_trade_date
    else:
        log.warning("未找到合适的交易日，使用当前日期")
        return current_date


def check_data_exists(dm: data.DataManager, trade_date: str) -> dict:
    """
    检查指定交易日的数据是否已存在
    
    Args:
        dm: DataManager实例
        trade_date: 交易日期 (YYYYMMDD格式)
    
    Returns:
        dict: 包含各类数据存在状态的字典
    """
    result = {
        'trade_date': trade_date,
        'daily_data_exists': False,
        'factor_data_exists': False,
        'daily_data_count': 0,
        'factor_data_count': 0,
        'should_skip': False
    }
    
    try:
        with dm.engine.connect() as conn:
            # 检查日线数据
            daily_count = conn.execute(text("""
                SELECT COUNT(*) FROM ts_daily 
                WHERE trade_date = :trade_date
            """), {"trade_date": trade_date}).scalar()
            
            result['daily_data_count'] = daily_count
            result['daily_data_exists'] = daily_count > 0
            
            # 检查因子数据
            factor_count = conn.execute(text("""
                SELECT COUNT(*) FROM factors_exposure 
                WHERE trade_date = :trade_date
            """), {"trade_date": trade_date}).scalar()
            
            result['factor_data_count'] = factor_count
            result['factor_data_exists'] = factor_count > 0
            
            # 判断是否应该跳过
            # 如果因子数据已存在且数量合理（>100），则跳过
            result['should_skip'] = result['factor_data_exists'] and result['factor_data_count'] > PipelineConfig.MIN_FACTOR_COUNT_THRESHOLD
            
    except Exception as e:
        log.error(f"检查数据存在性失败: {e}")
    
    return result


def _determine_dates_to_process(dm: data.DataManager) -> list:
    """
    智能确定需要处理的交易日列表
    
    Returns:
        list: 需要处理的交易日列表
    """
    log.info("===== 智能确定需要处理的交易日 =====")
    
    try:
        # 首先尝试智能模式：只处理最新的交易日
        latest_trade_date = get_latest_available_trade_date(dm)
        log.info(f"确定最新可用交易日: {latest_trade_date}")
        
        # 检查该日期的数据是否已存在
        data_status = check_data_exists(dm, latest_trade_date)
        
        log.info(f"数据存在状态检查:")
        log.info(f"  - 日线数据: {'存在' if data_status['daily_data_exists'] else '不存在'} ({data_status['daily_data_count']} 条)")
        log.info(f"  - 因子数据: {'存在' if data_status['factor_data_exists'] else '不存在'} ({data_status['factor_data_count']} 条)")
        
        if data_status['should_skip']:
            log.info(f"✅ {latest_trade_date} 的数据已存在且完整，跳过处理")
            return []
        
        # 如果需要处理，则只处理这一个日期
        log.info(f"📊 将处理交易日: {latest_trade_date}")
        return [latest_trade_date]
        
    except Exception as e:
        log.error(f"智能日期确定失败，回退到传统模式: {e}")
        # 回退到传统的历史补漏模式
        return get_missing_dates_traditional(dm)


def get_missing_dates_traditional(dm: data.DataManager) -> list:
    """
    传统的缺失日期获取方法（作为备用）
    """
    log.info("===== 使用传统方法确定需要处理的交易日列表 =====")
    try:
        # 获取数据库中factors_exposure表的最新日期
        with dm.engine.connect() as connection:
            query_max_date = text("SELECT MAX(trade_date) FROM factors_exposure")
            last_processed_date_db = connection.execute(query_max_date).scalar_one_or_none()

        # 确定开始获取交易日历的日期
        if last_processed_date_db:
            # 从数据库最新日期的下一天开始检查
            start_cal_date = (pd.to_datetime(last_processed_date_db) + timedelta(days=1)).strftime("%Y%m%d")
        else:
            # 如果数据库为空，则从一个很早的日期开始（例如2010年）
            start_cal_date = PipelineConfig.HISTORICAL_START_DATE
        
        end_cal_date = datetime.now().strftime("%Y%m%d")

        # 获取Tushare交易日历
        cal_df = dm.pro.trade_cal(
            exchange="",
            start_date=start_cal_date,
            end_date=end_cal_date,
        )
        all_trade_dates_ts = set(cal_df[cal_df["is_open"] == 1]["cal_date"].tolist())

        # 获取数据库中已有的factors_exposure日期
        existing_factors_dates = set()
        if last_processed_date_db:
            with dm.engine.connect() as connection:
                query_existing_dates = text(
                    "SELECT DISTINCT trade_date FROM factors_exposure WHERE trade_date >= :start_date AND trade_date <= :end_date"
                )
                result = connection.execute(query_existing_dates, {
                    "start_date": start_cal_date, "end_date": end_cal_date
                }).fetchall()
                existing_factors_dates = {row[0] for row in result}

        # 找出所有缺失的交易日
        dates_to_process = sorted(list(all_trade_dates_ts - existing_factors_dates))

        if not dates_to_process:
            log.info("所有历史数据均已是最新，无需补漏。任务结束。")
            return []
        else:
            log.info(f"发现 {len(dates_to_process)} 个缺失或待处理的交易日。")
            return dates_to_process

    except Exception as e:
        log.error(f"传统方法确定交易日列表失败: {e}", exc_info=True)
        return []


def run_daily_pipeline():
    """
    【V3.3 智能优化】统一数据管道主函数。
    新增智能日期判断，避免重复获取已存在的数据。
    """
    log.info("===== 启动智能统一数据管道 =====")
    dm = data.DataManager()

    # 确定需要处理的交易日列表
    dates_to_process = _determine_dates_to_process(dm)
    if not dates_to_process:
        log.info("===== 智能数据管道任务完成（无需处理）=====")
        return

    # 2. 循环处理每个交易日
    for current_trade_date in dates_to_process:
        log.info(f"===== 开始执行 {current_trade_date} 的统一数据管道任务 =====")
        start_time = time.time()

        # --- 前置检查机制 (针对当前日期) ---
        try:
            with dm.engine.connect() as connection:
                check_sql = text(
                    "SELECT 1 FROM factors_exposure WHERE trade_date = :trade_date LIMIT 1"
                )
                result = connection.execute(
                    check_sql, {"trade_date": current_trade_date}
                ).scalar_one_or_none()
            if result == 1:
                log.info(
                    f"检测到数据库中已存在 {current_trade_date} 的因子数据。跳过。"
                )
                continue # 跳过当前日期，处理下一个
        except Exception as e:
            log.error(f"在执行前置检查时发生数据库错误: {e}", exc_info=True)
            continue # 发生错误则跳过当前日期

        # --- 执行数据管道 ---
        # 步骤一：抽取数据
        raw_data_dict = extract_data(current_trade_date)

        # 【性能优化】步骤二：批量预取全市场财务数据
        all_fina_data = _batch_fetch_financial_data(dm, raw_data_dict["ts_codes"])

        # 步骤三：计算并存储因子
        calculate_and_save_factors(current_trade_date, raw_data_dict, all_fina_data)

        duration = time.time() - start_time
        log.info(
            f"===== {current_trade_date} 的统一数据管道任务完成！总耗时: {duration:.2f} 秒。====="
        )

    log.info("===== 所有待处理交易日的统一数据管道任务全部完成！ =====")


if __name__ == "__main__":
    run_daily_pipeline()
    # input("\n任务执行完毕，按 Enter 键退出...") # 移除input，避免非交互式运行挂起
