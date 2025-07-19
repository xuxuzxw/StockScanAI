# quant_project/test_analyzers.py
from logger_config import log
import data
import config
import quant_engine as qe
import index_analyzer as ia
import industry_analyzer as ind_a # 别名以防冲突

def run_analyzer_tests():
    """
    执行V2.0新增分析模块的单元测试。
    """
    log.info("===== [测试开始] V2.0分析器模块验证 =====")
    
    try:
        dm = data.DataManager(token=config.TUSHARE_TOKEN, db_path=config.DB_PATH)
        ff = qe.FactorFactory(_data_manager=dm)
        index_analyzer = ia.IndexAnalyzer(data_manager=dm)
        industry_analyzer = ind_a.IndustryAnalyzer(data_manager=dm, factor_factory=ff)
        log.info("1. V2.0分析器实例化成功。")
    except Exception as e:
        log.critical("1. V2.0分析器实例化失败！测试终止。", exc_info=True)
        return

    date = '20250717'
    
    # 测试用例 1: 大盘估值分析
    try:
        log.info("2. 正在测试 [IndexAnalyzer.get_index_valuation_percentile]...")
        val = index_analyzer.get_index_valuation_percentile('000300.SH', date)
        assert 'pe_percentile' in val and 0 <= val['pe_percentile'] <= 1
        log.info(f"   [成功] 沪深300估值百分位: PE={val['pe_percentile']:.2%}, PB={val['pb_percentile']:.2%}")
    except Exception as e:
        log.error("   [失败] 大盘估值分析测试未通过。", exc_info=True)

    # 测试用例 2: 行业因子排名
    try:
        log.info("3. 正在测试 [IndustryAnalyzer.get_industry_factor_rank]...")
        # 使用一个相对稳定的基本面因子进行测试
        rank_df = industry_analyzer.get_industry_factor_rank(date, 'pe_ttm', ascending=True)
        assert not rank_df.empty
        assert '银行' in rank_df.index
        log.info(f"   [成功] 行业PE排名 (部分):\n{rank_df.head()}")
    except Exception as e:
        log.error("   [失败] 行业因子排名测试未通过。", exc_info=True)
        
    log.info("===== [测试结束] V2.0分析器模块验证 =====")

if __name__ == "__main__":
    run_analyzer_tests()