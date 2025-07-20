# quant_project/run_daily_pipeline.py
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
    原 data_extractor.py 的核心逻辑。
    :param trade_date: 交易日期 YYYYMMDD
    :return: 包含所有原始数据DataFrame的字典。
    """
    log.info(f"【数据抽取】开始为 {trade_date} 抽取全市场原始数据...")

    dm = data.DataManager()
    stock_list = dm.get_stock_basic()
    ts_codes = stock_list["ts_code"].tolist()

    # --- 1. 批量获取截面数据 ---
    log.info(f"  正在获取每日指标...")
    daily_basics_df = dm.pro.daily_basic(trade_date=trade_date)

    log.info(f"  正在获取资金流数据...")
    money_flow_df = dm.pro.moneyflow(trade_date=trade_date)

    log.info(f"  正在获取龙虎榜数据...")
    top_list_df = dm.pro.top_list(trade_date=trade_date)

    log.info(f"  正在获取大宗交易数据...")
    block_trade_df = dm.pro.block_trade(trade_date=trade_date)

    # --- 2. 【性能优化】异步并发获取时序数据 (价格) ---
    log.info("  开始异步并发获取各股票的历史价格...")
    start_date_lookback = (pd.to_datetime(trade_date) - timedelta(days=90)).strftime(
        "%Y%m%d"
    )

    # 【V2.4 UX优化】将大任务拆分为小批次，并增加进度条
    prices_dict_raw = {}
    chunk_size = 150  # 每批次处理150只股票
    total_stocks = len(ts_codes)
    for i in range(0, total_stocks, chunk_size):
        chunk = ts_codes[i : i + chunk_size]
        chunk_results = dm.run_batch_download(chunk, start_date_lookback, trade_date)
        prices_dict_raw.update(chunk_results)

        progress = min(i + chunk_size, total_stocks)
        log.info(f"    价格获取进度: {progress}/{total_stocks}")

    # 将原始DataFrame字典转换为所需的Series字典
    prices_dict = {
        code: df.set_index("trade_date")["close"]
        for code, df in prices_dict_raw.items()
        if df is not None and not df.empty
    }

    daily_prices_df = pd.DataFrame(prices_dict)
    if not daily_prices_df.empty:
        daily_prices_df.index = pd.to_datetime(daily_prices_df.index)

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


def calculate_and_save_factors(trade_date: str, raw_data: dict):
    """
    步骤二：执行因子计算与存储。
    原 factor_calculator.py 的核心逻辑。
    :param trade_date: 交易日期 YYYYMMDD
    :param raw_data: 从 extract_data 函数获取的原始数据字典。
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
            # 【Bug修复】在join前，丢弃flow表中的冗余trade_date列，避免列名冲突
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

    # --- 第二部分：统一循环计算 ---
    log.info("  正在循环计算：财务、筹码与价值类因子...")
    params = {
        "date": trade_date,
        "start_date": (trade_date_dt - timedelta(days=90)).strftime("%Y%m%d"),
        "end_date": trade_date,
        "top_list_df": raw_data["top_list_df"],
        "block_trade_df": raw_data["block_trade_df"],
    }
    loop_factors = [f for f in FACTORS_TO_CALCULATE if f not in vectorized_factors]

    for factor in loop_factors:
        log.info(f"    正在计算因子: {factor}...")
        factor_series = pd.Series(index=ts_codes, dtype=float)
        for code in ts_codes:
            params["ts_code"] = code
            factor_series[code] = ff.calculate(factor, **params)
        results[factor] = factor_series

    # --- 第三部分：整合与存储 ---
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
    """
    dm = data.DataManager()
    try:
        cal_df = dm.pro.trade_cal(
            exchange="",
            start_date=(datetime.now() - timedelta(days=5)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        latest_trade_date = cal_df[cal_df["is_open"] == 1]["cal_date"].max()
    except Exception as e:
        log.error(f"获取最新交易日失败: {e}")
        return

    log.info(f"===== 开始执行 {latest_trade_date} 的统一数据管道任务 =====")
    start_time = time.time()

    # --- 前置检查机制 ---
    try:
        with dm.engine.connect() as connection:
            check_sql = text(
                "SELECT 1 FROM factors_exposure WHERE trade_date = :trade_date LIMIT 1"
            )
            result = connection.execute(
                check_sql, {"trade_date": latest_trade_date}
            ).scalar_one_or_none()
        if result == 1:
            log.info(
                f"检测到数据库中已存在 {latest_trade_date} 的因子数据。任务无需重复执行。"
            )
            return
    except Exception as e:
        log.error(f"在执行前置检查时发生数据库错误: {e}", exc_info=True)
        return

    # --- 执行数据管道 ---
    # 步骤一：抽取数据
    raw_data_dict = extract_data(latest_trade_date)

    # 步骤二：计算并存储因子
    calculate_and_save_factors(latest_trade_date, raw_data_dict)

    duration = time.time() - start_time
    log.info(
        f"===== {latest_trade_date} 的统一数据管道任务完成！总耗时: {duration:.2f} 秒。====="
    )


if __name__ == "__main__":
    run_daily_pipeline()
    input("\n任务执行完毕，按 Enter 键退出...")
