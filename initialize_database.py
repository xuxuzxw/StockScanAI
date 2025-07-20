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
    log.info("=" * 60)
    log.info("🚀 开始执行数据库和基础数据初始化")
    log.info("=" * 60)

    try:
        # 首先进行数据库健康检查
        from database_health_checker import DatabaseHealthChecker
        health_checker = DatabaseHealthChecker()
        
        if not health_checker.run_full_check():
            log.error("❌ 数据库健康检查失败，请先修复数据库问题")
            return
        
        data_manager = data.DataManager(
            token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL
        )
        log.info(f"✅ 数据库连接成功: {config.DATABASE_URL}")

        # 1. 更新股票基本信息
        log.info("📊 下载和缓存所有A股基本信息...")
        stock_basic = data_manager.get_stock_basic(force_update=True)
        if stock_basic is not None and not stock_basic.empty:
            log.info(f"✅ 成功获取 {len(stock_basic)} 条股票基本信息")
        else:
            log.error("❌ 获取股票基本信息失败！")
            return

        # 2. 更新交易日历
        # (Tushare的trade_cal接口目前不是必需的，因为get_daily等会自动处理非交易日，此处省略)

        # 3. 预加载核心指数和个股的历史行情数据
        log.info("📈 预加载核心标的历史行情数据...")

        core_assets = {
            "indices": ["000300.SH", "000905.SH", "399006.SZ"],
            "stocks": ["600519.SH", "000001.SZ", "300750.SZ"],
        }

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=10 * 365)).strftime(
            "%Y%m%d"
        )  # 拉取10年数据

        # 使用进度跟踪器
        from progress_tracker import ProgressTracker
        total_assets = sum(len(codes) for codes in core_assets.values())
        progress_tracker = ProgressTracker(total_assets, "核心资产数据加载")

        for asset_type, codes in core_assets.items():
            log.info(f"  📊 处理{asset_type}...")
            for code in codes:
                try:
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
                        log.info(f"    ✅ {code}: {len(df)} 条记录")
                    else:
                        log.warning(f"    ⚠️  {code}: 无数据")
                        
                    progress_tracker.update(1, code)
                    
                except Exception as e:
                    log.error(f"    ❌ {code}: 加载失败 - {e}")
                    progress_tracker.mark_failed(1, f"{code}加载失败")

        progress_tracker.finish()
        
        log.info("=" * 60)
        log.info("🎉 数据库初始化完成！")
        log.info("=" * 60)

    except Exception as e:
        log.critical("数据库初始化过程中发生严重错误！", exc_info=True)


if __name__ == "__main__":
    # 在第一次部署或需要重建缓存时，运行此脚本
    # python initialize_database.py
    initialize_all_data()
