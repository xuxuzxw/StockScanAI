# StockScanAI/initialize_database.py
from datetime import datetime, timedelta

import pandas as pd

import config

# 确保在项目根目录运行
import data
from logger_config import log


def initialize_all_data():
    """
    执行一次性的全量数据初始化。
    1. 实例化 DataManager，自动创建数据库和表结构。
    2. 下载并缓存所有A股的基本信息。
    3. 下载并缓存所有交易日的日历数据。
    4. 为核心宽基指数（如沪深300）和部分重要个股预加载足够长的历史日线数据。
    """
    log.info("===== 开始执行数据库和基础数据初始化 =====")

    try:
        data_manager = data.DataManager(
            token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL
        )
        log.info(f"数据库已连接: {config.DATABASE_URL}")

        # 1. 更新股票基本信息
        log.info("正在下载和缓存所有A股基本信息...")
        stock_basic = data_manager.get_stock_basic(force_update=True)
        if stock_basic is not None and not stock_basic.empty:
            log.info(f"成功获取 {len(stock_basic)} 条股票基本信息。")
        else:
            log.error("获取股票基本信息失败！")
            return

        # 2. 更新交易日历
        # (Tushare的trade_cal接口目前不是必需的，因为get_daily等会自动处理非交易日，此处省略)

        # 3. 预加载核心指数和个股的历史行情数据
        log.info("正在为核心标的预加载历史行情数据...")

        core_assets = {
            "indices": ["000300.SH", "000905.SH", "399006.SZ"],
            "stocks": ["600519.SH", "000001.SZ", "300750.SZ"],
        }

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=10 * 365)).strftime(
            "%Y%m%d"
        )  # 拉取10年数据

        for asset_type, codes in core_assets.items():
            for code in codes:
                log.info(f"  正在处理 {asset_type}: {code}...")

                df = None
                # 【修正】根据资产类型调用不同的数据获取函数
                if asset_type == "indices":
                    # 对指数，调用专用的指数日线接口
                    df = data_manager.get_index_daily(
                        ts_code=code, start_date=start_date, end_date=end_date
                    )
                elif asset_type == "stocks":
                    # 对股票，调用复权行情接口
                    df = data_manager.get_adjusted_daily(
                        ts_code=code, start_date=start_date, end_date=end_date
                    )

                if df is not None and not df.empty:
                    log.info(f"  > 成功为 {code} 加载了 {len(df)} 条日线数据。")
                else:
                    log.warning(f"  > 未能为 {code} 加载数据。")

        log.info("===== 数据库初始化完成！ =====")

    except Exception as e:
        log.critical("数据库初始化过程中发生严重错误！", exc_info=True)


if __name__ == "__main__":
    # 在第一次部署或需要重建缓存时，运行此脚本
    # python initialize_database.py
    initialize_all_data()
