# StockScanAI/run_daily_pipeline.py
#
# 【V2.2 核心优化】
# 目的：将数据抽取和因子计算两个独立的后台任务，合并为一个统一、健壮的数据管道。
#      消除了对本地HDF5缓存文件的依赖，实现了数据在内存中的无缝流转。

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

# 导入项目模块
import data
import quant_engine
from logger_config import log
from progress_tracker import ProgressTracker

# 【V2.2 重构】将因子列表的定义移入此统一管道脚本中，移除对旧文件的依赖
FACTORS_TO_CALCULATE = [
    "pe_ttm",
    "roe",
    "growth_revenue_yoy",
    "debt_to_assets",
    "momentum",
    "volatility",
    "net_inflow_ratio",
    "holder_num_change_ratio",
    "major_shareholder_net_buy_ratio",
    "top_list_net_buy_amount",
    "dividend_yield",
    "forecast_growth_rate",
    "repurchase_ratio",
    "block_trade_ratio",
]


def extract_data(trade_date: str) -> dict:
    """
    步骤一：执行所有耗时的数据抽取和预处理工作。
    【V2.7 增强版 - 增加进度显示】
    """
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
            data = func()
            results[name] = data
            count = len(data) if data is not None and not data.empty else 0
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
                "ts_codes": ts_codes, "min_days": min_trading_days
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
        max_retries = 3  # V2.9 提升健壮性：增加上层重试次数
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

            chunk_size = 150
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
        chunk_size = 500
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


def calculate_and_save_factors(trade_date: str, raw_data: dict, all_fina_data: pd.DataFrame):
    """
    步骤二：执行因子计算与存储。
    原 factor_calculator.py 的核心逻辑。
    :param trade_date: 交易日期 YYYYMMDD
    :param raw_data: 从 extract_data 函数获取的原始数据字典。
    :param all_fina_data: 【新增】预先获取的全市场财务数据。
    """
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
        if len(prices) >= 21:
            results["momentum"] = prices.iloc[-1] / prices.iloc[-21] - 1
        if len(prices) >= 20:
            results["volatility"] = np.log(prices / prices.shift(1)).iloc[
                -20:
            ].std() * np.sqrt(252)

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
        pit_fina_data = all_fina_data.groupby('ts_code', include_groups=False).apply(
            lambda x: dm.get_pit_financial_data(x, trade_date)
        ).reset_index(drop=True)
        
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
        log.warning("没有有效的因子数据可以存入数据库。")
        return

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
                    chunksize=10000,
                )
        log.info(f"成功为 {trade_date} 写入 {len(long_df)} 条因子数据。")
    except Exception as e:
        log.critical(f"因子数据写入数据库时发生严重错误: {e}", exc_info=True)


def run_daily_pipeline():
    """
    执行每日数据管道的主工作流。
    - 具备智能补漏功能，自动补齐历史缺失数据。
    """
    dm = data.DataManager()

    # 1. 确定需要处理的交易日列表
    log.info("===== 正在确定需要处理的交易日列表（包括历史补漏） =====")
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
            start_cal_date = "20100101"
        
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
            return
        else:
            log.info(f"发现 {len(dates_to_process)} 个缺失或待处理的交易日。")

    except Exception as e:
        log.error(f"确定交易日列表失败: {e}", exc_info=True)
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
        log.info("【性能优化】开始批量预取全市场财务数据...")
        try:
            all_fina_data_list = []
            ts_codes_for_fina = raw_data_dict["ts_codes"]
            log.info(f"  正在为 {len(ts_codes_for_fina)} 只股票批量预取财务数据...")
            
            # 优化：只对需要更新的股票获取财务数据
            # 财务数据已在data.py中实现智能缓存，此处无需额外判断
            for i, code in enumerate(ts_codes_for_fina):
                if i % 100 == 0: 
                    log.info(f"    已处理 {i}/{len(ts_codes_for_fina)} 只股票的财务数据...")
                try:
                    df_fina = dm.get_fina_indicator(ts_code=code, force_update=False) # 不强制更新，利用内部缓存
                    if df_fina is not None and not df_fina.empty:
                        all_fina_data_list.append(df_fina)
                except Exception as e:
                    log.warning(f"    为股票 {code} 预取财务数据失败: {e}")
                    continue
            
            if all_fina_data_list:
                all_fina_data = pd.concat(all_fina_data_list).drop_duplicates(subset=['ts_code', 'end_date'])
                log.info(f"批量预取完成，共获取 {len(all_fina_data)} 条财务记录。")
            else:
                all_fina_data = pd.DataFrame()
                log.warning("未能成功预取任何财务数据。")
        except Exception as e:
            log.error(f"批量预取财务数据失败: {e}", exc_info=True)
            all_fina_data = pd.DataFrame()

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
