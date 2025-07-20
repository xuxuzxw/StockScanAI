# quant_project/backfill_data.py
import argparse

import config
import data
from logger_config import log


def backfill_data(
    start_date: str, end_date: str, ts_codes: list = None, data_type: str = "daily"
):
    """
    一个更强大的数据回填工具。
    :param start_date: 开始日期 YYYYMMDD
    :param end_date: 结束日期 YYYYMMDD
    :param ts_codes: 股票代码列表，如果为None则处理全市场
    :param data_type: 要回填的数据类型, e.g., 'daily', 'adj_factor'
    """
    log.info(f"开始执行数据回填任务: {data_type} from {start_date} to {end_date}")
    dm = data.DataManager(token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL)

    if ts_codes is None:
        log.info("未指定股票列表，将回填全市场股票数据...")
        all_stocks = dm.get_stock_basic()
        ts_codes = all_stocks["ts_code"].tolist()
        log.info(f"全市场共 {len(ts_codes)} 只股票。")

    total = len(ts_codes)
    for i, code in enumerate(ts_codes):
        try:
            if data_type == "daily":
                # 调用 get_adjusted_daily 会自动处理日线和复权因子
                dm.get_adjusted_daily(
                    ts_code=code, start_date=start_date, end_date=end_date
                )
            # 未来可以扩展其他数据类型
            # elif data_type == 'financials':
            #     dm.get_fina_indicator(ts_code=code, force_update=True)

            if (i + 1) % 100 == 0:
                log.info(f"处理进度: {i + 1}/{total}")

        except Exception as e:
            log.error(f"处理 {code} 时发生错误: {e}")
            continue

    log.info("数据回填任务完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股历史数据回填工具")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--stocks", nargs="*", default=None, help="股票代码列表，不填则为全市场"
    )
    parser.add_argument("--type", default="daily", help="数据类型 (e.g., daily)")

    args = parser.parse_args()

    # 示例用法: python backfill_data.py --start 20220101 --end 20241231
    backfill_data(args.start, args.end, args.stocks, args.type)
