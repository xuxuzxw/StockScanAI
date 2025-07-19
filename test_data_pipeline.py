# quant_project/test_data_pipeline.py
import pandas as pd
from logger_config import log
import data
import config

def run_data_tests():
    """
    执行数据管道的单元测试和集成测试。
    """
    log.info("===== [测试开始] 数据管道验证 =====")
    
    try:
        dm = data.DataManager(token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL)
        log.info("1. DataManager 实例化成功。")
    except Exception as e:
        log.critical("1. DataManager 实例化失败！测试终止。", exc_info=True)
        return

    # 测试用例 1: 获取单只股票的后复权日线数据
    try:
        log.info("2. 正在测试 [get_adjusted_daily]...")
        df_adj = dm.get_adjusted_daily(ts_code='000001.SZ', start_date='20250101', end_date='20250717')
        assert df_adj is not None and not df_adj.empty
        assert 'close' in df_adj.columns and 'vol' in df_adj.columns
        log.info("   [成功] get_adjusted_daily 功能正常，获取到复权数据。")
    except Exception as e:
        log.error("   [失败] get_adjusted_daily 测试未通过。", exc_info=True)

    # 测试用例 2: 异步批量下载多只股票日线数据
    try:
        log.info("3. 正在测试 [run_batch_download]...")
        stock_pool = ['600519.SH', '300750.SZ', '601318.SH'] # 修正股票代码 .SS -> .SH
        prices_dict = dm.run_batch_download(stock_pool, '20250701', '20250717')
        assert len(prices_dict) == 3
        assert not prices_dict['600519.SH'].empty
        log.info("   [成功] run_batch_download 功能正常，获取到批量数据。")
    except Exception as e:
        log.error("   [失败] run_batch_download 测试未通过。", exc_info=True)

    # 测试用例 3: Point-in-Time (PIT) 财务数据获取
    try:
        log.info("4. 正在测试 [get_pit_financial_data]...")
        all_fina = dm.get_fina_indicator(ts_code='600519.SH') # 修正股票代码 .SS -> .SH
        # 假设我们站在 2024-05-01，获取当时能看到的最新财报
        pit_fina = dm.get_pit_financial_data(all_fina, as_of_date='20240501')
        assert pit_fina is not None and not pit_fina.empty
        # 2024年一季报(end_date=20240331)通常在4月底发布，所以此时应该能取到
        assert pit_fina.iloc[0]['end_date'] == '20240331'
        log.info("   [成功] get_pit_financial_data 功能正常，获取到正确的PIT数据。")
    except Exception as e:
        log.error("   [失败] get_pit_financial_data 测试未通过。", exc_info=True)
        
    log.info("===== [测试结束] 数据管道验证 =====")

if __name__ == "__main__":
    # 执行命令: python test_data_pipeline.py
    run_data_tests()