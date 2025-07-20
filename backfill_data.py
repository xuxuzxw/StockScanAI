# StockScanAI/backfill_data.py
import argparse
import time

import pandas as pd
from sqlalchemy import text

import config
import data
from logger_config import log
from progress_tracker import ProgressTracker, DataQualityChecker


def backfill_data_optimized(
    start_date: str,
    end_date: str,
    ts_codes: list = None,
    chunk_size: int = 200,
):
    """
    【V2.9 增强版】具备断点续传、停牌填充和详细进度显示的高性能数据回填工具。
    - 启动时自动检测已完成的股票（包括已填充停牌标记的），并跳过它们。
    - 新增停牌日填充逻辑：对成功返回但部分日期无数据的股票，用占位符填充，确保数据完整性。
    - 利用异步并发下载，大幅提升全市场数据回填速度。
    - 将串行处理改为分块批量处理，减少IO开销。
    - 增强进度显示和错误处理机制。
    :param start_date: 开始日期 YYYYMMDD
    :param end_date: 结束日期 YYYYMMDD
    :param ts_codes: 股票代码列表，如果为None则处理全市场
    :param chunk_size: 每次并发下载的股票数量
    """
    log.info("=" * 60)
    log.info(f"🚀 开始执行高性能数据回填任务 (v2.9)")
    log.info(f"📅 时间范围: {start_date} → {end_date}")
    log.info(f"⚙️  并发设置: 每批 {chunk_size} 只股票")
    log.info("=" * 60)
    
    dm = data.DataManager(token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL)

    # --- V3.1 最终版：恢复并修复断点续传检查逻辑 ---
    # 1. 获取全市场或指定的股票列表
    if ts_codes is None:
        log.info("📊 获取全市场股票列表...")
        all_stocks = dm.get_stock_basic()
        initial_ts_codes = all_stocks["ts_code"].tolist()
        log.info(f"✓ 初始目标共 {len(initial_ts_codes)} 只股票")
    else:
        initial_ts_codes = ts_codes
        log.info(f"📊 使用指定股票列表，共 {len(initial_ts_codes)} 只股票")

    # 2. 获取指定时间范围内的总交易日数和日期列表
    log.info("📅 获取交易日历...")
    all_trade_dates = set()
    total_trade_days = 0
    try:
        trade_cal_df = dm.pro.trade_cal(start_date=start_date, end_date=end_date)
        open_trade_days_df = trade_cal_df[trade_cal_df["is_open"] == 1]
        total_trade_days = len(open_trade_days_df)
        all_trade_dates = set(open_trade_days_df["cal_date"])
        log.info(f"✓ 时间范围内共有 {total_trade_days} 个交易日")
    except Exception as e:
        log.error(f"✗ 获取交易日历失败: {e}")
        log.warning("⚠ 无法执行断点续传检查和停牌填充，将尝试全量回填")

    # 3. 查询数据库，找出已完成的股票
    completed_codes = set()
    if total_trade_days > 0:
        log.info("🔍 检查数据库中已完成回填的股票...")
        try:
            with dm.engine.connect() as conn:
                # 修复：使用 TO_DATE 确保日期比较的健壮性
                query = text(
                    """
                    SELECT ts_code FROM ts_daily
                    WHERE trade_date BETWEEN TO_DATE(:start, 'YYYYMMDD') AND TO_DATE(:end, 'YYYYMMDD')
                    GROUP BY ts_code
                    HAVING COUNT(1) >= :count
                """
                )
                result = conn.execute(
                    query,
                    {
                        "start": start_date,
                        "end": end_date,
                        "count": total_trade_days,
                    },
                )
                completed_codes = {row[0] for row in result}
                completion_rate = len(completed_codes) / len(initial_ts_codes) * 100
                log.info(f"✓ 发现 {len(completed_codes)} 只股票数据已完整 ({completion_rate:.1f}%)")
        except Exception as e:
            log.error(f"✗ 检查数据库时发生错误: {e}")
            log.warning("⚠ 将尝试全量回填")

    # 4. 确定最终需要处理的股票列表
    ts_codes_to_process = sorted(list(set(initial_ts_codes) - completed_codes))

    if not ts_codes_to_process:
        log.info("所有目标股票的数据均已是最新，无需回填。任务结束。")
        return
    else:
        log.info(f"将处理 {len(ts_codes_to_process)} 只尚未完成或数据不完整的股票。")
    # --- 预检查结束 ---

    total_stocks = len(ts_codes_to_process)
    start_time_total = time.time()
    
    # 初始化进度跟踪器
    progress_tracker = ProgressTracker(total_stocks, "数据回填")
    log.info(f"🎯 开始处理 {total_stocks} 只股票的数据回填")

    max_chunk_retries = 3  # V3.0 终极优化：为每个块增加重试机制
    processed_count = 0
    
    for i in range(0, total_stocks, chunk_size):
        chunk = ts_codes_to_process[i : i + chunk_size]

        # --- 新增：块级重试循环 ---
        chunk_success = False
        for attempt in range(max_chunk_retries + 1):
            chunk_start_time = time.time()
            chunk_num = i//chunk_size + 1
            total_chunks = (total_stocks + chunk_size - 1) // chunk_size
            
            log.info(f"📦 处理批次 {chunk_num}/{total_chunks} ({len(chunk)} 只股票)")
            if attempt > 0:
                log.warning(f"  🔄 重试 {attempt}/{max_chunk_retries}")

            try:
                # 并发下载日线和复权因子数据
                log.info(f"  > 正在并发下载 {len(chunk)} 只股票的日线数据...")
                daily_data_dict = dm.run_batch_download(chunk, start_date, end_date)

                log.info(f"  > 正在增量下载 {len(chunk)} 只股票的复权因子...")
                adj_factor_dict = {}
                for code in chunk:
                    adj_df = dm.get_adj_factor(code, start_date, end_date)
                    if adj_df is not None and not adj_df.empty:
                        adj_factor_dict[code] = adj_df

                # 数据处理与入库
                log.info(f"  > 正在处理数据并批量写入数据库...")
                processed_daily_dfs = []
                adj_dfs = []
                for code in chunk:
                    daily_df = daily_data_dict.get(code)
                    # V3.0 健壮性修复：只有在下载成功时才获取复权因子和填充停牌
                    if daily_df is None:
                        log.warning(f"  > {code} 日线数据下载失败，跳过后续处理。")
                        continue
                    
                    adj_df = adj_factor_dict.get(code)

                    # --- 停牌数据填充逻辑 ---
                    if all_trade_dates:
                        existing_dates = set()
                        if not daily_df.empty:
                            existing_dates = set(pd.to_datetime(daily_df["trade_date"]).dt.strftime('%Y%m%d'))

                        missing_dates = all_trade_dates - existing_dates

                        if missing_dates:
                            log.debug(f"  > 填充 {code} 的 {len(missing_dates)} 个缺失/停牌日...")
                            placeholder_df = pd.DataFrame({ "ts_code": code, "trade_date": list(missing_dates), "vol": 0, "amount": 0 })
                            daily_df = pd.concat([daily_df, placeholder_df], ignore_index=True)

                    if not daily_df.empty:
                        processed_daily_dfs.append(daily_df)
                    if adj_df is not None and not adj_df.empty:
                        adj_dfs.append(adj_df)

                if processed_daily_dfs:
                    dm._upsert_data(
                        "ts_daily",
                        pd.concat(processed_daily_dfs, ignore_index=True),
                        ["trade_date", "ts_code"],
                    )
                if adj_dfs:
                    dm._upsert_data(
                        "ts_adj_factor",
                        pd.concat(adj_dfs, ignore_index=True),
                        ["trade_date", "ts_code"],
                    )

                chunk_duration = time.time() - chunk_start_time
                processed_count += len(chunk)
                
                # 更新进度
                progress_tracker.update(len(chunk), f"批次{chunk_num}")
                
                log.info(f"  ✅ 批次处理成功，耗时: {chunk_duration:.2f}秒，平均: {chunk_duration/len(chunk):.2f}秒/股")
                chunk_success = True
                break  # 成功处理，跳出重试循环

            except Exception as e:
                log.error(f"❌ 批次 {chunk_num} (第 {attempt+1} 次尝试) 处理失败: {e}")
                if attempt == max_chunk_retries:
                    log.critical(f"  💥 批次 {chunk_num} 在尝试 {max_chunk_retries+1} 次后彻底失败，跳过此批次")
                    progress_tracker.mark_failed(len(chunk), f"批次{chunk_num}彻底失败")
                else:
                    sleep_time = 2 ** (attempt + 1)  # 指数退避
                    log.info(f"  ⏳ 等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
        
        if not chunk_success:
            log.error(f"⚠️  批次 {chunk_num} 最终失败，已跳过")

    # 完成进度跟踪
    progress_tracker.finish()
    
    # 执行数据质量检查
    log.info("🔍 开始数据质量检查...")
    quality_checker = DataQualityChecker(dm)
    quality_results = quality_checker.batch_quality_check(
        ts_codes_to_process[:100],  # 抽样检查前100只股票
        start_date, 
        end_date
    )
    
    total_duration = time.time() - start_time_total
    log.info("=" * 60)
    log.info(f"🎉 全部数据回填任务完成！")
    log.info(f"⏱️  总耗时: {total_duration/60:.2f} 分钟")
    log.info(f"📊 处理统计: {processed_count}/{total_stocks} 只股票")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股历史数据回填工具 (高性能-断点续传版)")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--stocks", nargs="*", default=None, help="股票代码列表，不填则为全市场"
    )
    parser.add_argument(
        "--chunk", type=int, default=200, help="每次并发请求的股票数量"
    )

    args = parser.parse_args()

    backfill_data_optimized(args.start, args.end, args.stocks, args.chunk)