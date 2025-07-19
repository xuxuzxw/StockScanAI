# quant_project/test_quant_engine.py
import pandas as pd
from logger_config import log
import data
import config
import quant_engine as qe

def run_engine_tests():
    """
    执行核心量化引擎的单元测试。
    """
    log.info("===== [测试开始] 核心量化引擎验证 =====")
    
    try:
        dm = data.DataManager(token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL)
        ff = qe.FactorFactory(_data_manager=dm)
        fp = qe.FactorProcessor(_data_manager=dm)
        log.info("1. 引擎组件实例化成功。")
    except Exception as e:
        log.critical("1. 引擎组件实例化失败！测试终止。", exc_info=True)
        return

    ts_code = '600519.SH' # 修正股票代码 .SS -> .SH
    date = '20250717'
    start_date = '20250101'
    
    # 测试用例 1: 计算技术类因子 (动量)
    try:
        log.info(f"2. 正在测试计算技术因子 [calc_momentum] for {ts_code}...")
        momentum = ff.calc_momentum(ts_code, start_date, date, window=20)
        assert isinstance(momentum, float)
        log.info(f"   [成功] 动量因子计算结果: {momentum:.4f}")
    except Exception as e:
        log.error("   [失败] 技术因子计算测试未通过。", exc_info=True)

    # 测试用例 2: 计算基本面因子 (ROE - Point-in-Time)
    try:
        log.info(f"3. 正在测试计算PIT基本面因子 [calc_roe] for {ts_code}...")
        roe = ff.calc_roe(ts_code, date=date)
        assert isinstance(roe, float)
        log.info(f"   [成功] ROE因子(PIT)计算结果: {roe:.4f}")
    except Exception as e:
        log.error("   [失败] 基本面因子计算测试未通过。", exc_info=True)

    # 测试用例 3: 因子预处理流水线
    try:
        log.info("4. 正在测试 [FactorProcessor] 预处理流水线...")
        # 构造一个截面因子Series
        stock_pool = ['600519.SH', '000001.SZ', '300750.SZ', '601318.SH'] # 修正股票代码
        raw_pe_values = {code: ff.calc_pe_ttm(code, date=date) for code in stock_pool}
        factor_series = pd.Series(raw_pe_values).dropna()        
        log.info(f"   原始PE_TTM因子值:\n{factor_series}")
        processed_factor = fp.process_factor(factor_series, neutralize=True)
        
        assert not processed_factor.empty
        # 检查标准化后的均值是否接近0，标准差是否接近1
        assert abs(processed_factor.mean()) < 1e-10
        assert abs(processed_factor.std() - 1.0) < 1e-1
        log.info(f"   [成功] 因子预处理完成，标准化和中性化正常。处理后因子值:\n{processed_factor}")
    except Exception as e:
        log.error("   [失败] 因子预处理测试未通过。", exc_info=True)

    log.info("===== [测试结束] 核心量化引擎验证 =====")

if __name__ == "__main__":
    # 执行命令: python test_quant_engine.py
    run_engine_tests()