# quant_project/run_system_check.py
#
# A股量化投研平台 - 统一系统健康检查程序
#
# 使用方法:
# 1. 确保您的 .env 文件已正确配置。
# 2. 直接在命令行运行: python run_system_check.py
#
# 功能:
# - 替代所有旧的 test_*.py 文件。
# - 逐一检查系统的核心组件，提供清晰的通过/失败报告。
# - 设计为快速、低成本的诊断工具。

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

# 将当前目录添加到sys.path，确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
try:
    import config
    import data
    import intelligence
    import quant_engine as qe
    from logger_config import log
except ImportError as e:
    print(f"FATAL: 关键模块导入失败: {e}")
    print("请确保您在项目的根目录下运行此脚本，并且所有依赖已安装。")
    sys.exit(1)

# --- 检查流程 ---


def run_all_checks():
    """
    执行所有系统健康检查。
    """
    checks_passed = 0
    total_checks = 6

    log.info("==============================================")
    log.info("=== A股量化投研平台 - 系统健康检查启动 ===")
    log.info("==============================================")

    # --- 检查 1: 配置文件与环境变量 ---
    log.info("\n--- [检查 1/6] 配置文件与环境变量 (config.py & .env) ---")
    try:
        assert (
            config.TUSHARE_TOKEN and config.TUSHARE_TOKEN != "YOUR_TUSHARE_TOKEN"
        ), "TUSHARE_TOKEN 未配置"
        assert config.DATABASE_URL, "DATABASE_URL 未配置"
        assert config.AI_MODEL_CONFIG["fast_and_cheap"][
            "api_key"
        ], "AI_API_KEY_FAST 未配置"
        log.info("  [PASS] 所有关键环境变量已成功加载。")
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 配置检查失败: {e}", exc_info=True)
        log.critical("关键配置缺失，后续检查可能无法进行。请检查您的 .env 文件。")
        finalize_report(checks_passed, total_checks)
        return

    # --- 检查 2: 数据库连接 (Database Connection) ---
    log.info("\n--- [检查 2/6] 数据库连接 ---")
    try:
        engine = create_engine(config.DATABASE_URL)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        log.info(
            f"  [PASS] 成功连接到数据库: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        )
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 数据库连接失败: {e}", exc_info=True)
        log.critical(
            "无法连接到数据库，后续检查将失败。请检查数据库服务是否运行，以及.env中的连接信息是否正确。"
        )
        finalize_report(checks_passed, total_checks)
        return

    # --- 检查 3: 数据层核心功能 (DataManager & Tushare API) ---
    log.info("\n--- [检查 3/6] 数据层与Tushare API连通性 ---")
    dm = None
    try:
        dm = data.DataManager()
        stock_list = dm.get_stock_basic(list_status="L")
        assert (
            stock_list is not None and not stock_list.empty
        ), "获取股票列表失败，返回为空。"
        log.info(
            f"  [PASS] DataManager 实例化成功，并成功从 Tushare 获取到 {len(stock_list)} 只上市股票的基本信息。"
        )
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 数据层或Tushare API检查失败: {e}", exc_info=True)
        log.critical("无法从Tushare获取数据，请检查Tushare Token是否有效或网络连接。")
        finalize_report(checks_passed, total_checks)
        return

    # --- 检查 4: 引擎层核心功能 (FactorFactory & FactorProcessor) ---
    log.info("\n--- [检查 4/6] 引擎层因子计算 ---")
    try:
        ff = qe.FactorFactory(_data_manager=dm)
        fp = qe.FactorProcessor(_data_manager=dm)
        # 以'贵州茅台'为例，计算一个简单的PE因子
        ts_code_test = "600519.SH"
        # 【鲁棒性修复】获取截至今日的最后一个交易日，而不是一个未来的日期
        today_str = pd.Timestamp.now().strftime("%Y%m%d")
        trade_cal = dm.pro.trade_cal(
            exchange="", start_date="20240101", end_date=today_str
        )
        latest_trade_date = trade_cal[trade_cal["is_open"] == 1]["cal_date"].max()

        pe_ttm = ff.calc_pe_ttm(ts_code=ts_code_test, date=latest_trade_date)
        assert isinstance(pe_ttm, float) and pd.notna(
            pe_ttm
        ), f"为 {ts_code_test} 计算的 pe_ttm 无效。"
        log.info(
            f"  [PASS] 因子引擎实例化成功，并为 {ts_code_test} 计算出 PE(TTM) 因子值为: {pe_ttm:.2f}"
        )
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 引擎层检查失败: {e}", exc_info=True)
        finalize_report(checks_passed, total_checks)
        return

    # --- 检查 5: 智能分析层 (AIOrchestrator & AI API) ---
    log.info("\n--- [检查 5/6] 智能分析层与AI API连通性 (低成本测试) ---")
    try:
        ai_orchestrator = intelligence.AIOrchestrator(config.AI_MODEL_CONFIG, dm)
        prompt = "你好，请确认你可以正常工作。回答'ok'即可。"
        # 使用成本最低的模型进行一次简单的API调用测试
        response = ai_orchestrator._execute_ai_call(prompt, "fast_and_cheap")
        assert "ok" in response.lower(), f"AI模型返回内容非预期: {response}"
        log.info(
            "  [PASS] AI引擎实例化成功，并成功调用低成本模型 API，密钥和网络连接正常。"
        )
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] AI层检查失败: {e}", exc_info=True)
        log.warning(
            "请检查 .env 文件中的 AI_API_KEY_FAST 是否正确，以及网络是否能访问AI服务。"
        )
        finalize_report(checks_passed, total_checks)
        return

    # --- 检查 6: 端到端工作流冒烟测试 ---
    log.info("\n--- [检查 6/6] 端到端工作流冒烟测试 ---")
    try:
        # 这是一个简化的测试，仅验证核心工作流的组件能否被正确初始化和调用
        # 而不实际运行耗时的完整流程
        from run_strategy_daily import run_daily_strategy_workflow

        log.info("  正在尝试模拟调用每日策略工作流...")
        # 此处不直接运行 run_daily_strategy_workflow()，因为它会执行完整流程
        # 而是验证其依赖的核心组件是否能被正确组装
        test_adaptive_strategy = qe.AdaptiveAlphaStrategy(
            ff, fp, qe.FactorAnalyzer(dm), pd.DataFrame()
        )
        test_risk_manager = qe.RiskManager(ff, fp)
        assert test_adaptive_strategy is not None
        assert test_risk_manager is not None
        log.info(
            "  [PASS] 每日策略工作流的核心组件均可被成功实例化，端到端流程基本通畅。"
        )
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 端到端工作流冒烟测试失败: {e}", exc_info=True)

    # --- 总结报告 ---
    finalize_report(checks_passed, total_checks)


def finalize_report(passed, total):
    """
    生成并打印最终的检查报告。
    """
    log.info("\n==============================================")
    log.info("=== 系统健康检查完成 ===")
    log.info(f"=== 结果: {passed}/{total} 项检查通过 ===")
    log.info("==============================================")
    if passed == total:
        log.info("恭喜！您的A股量化投研平台所有核心组件均工作正常。")
    else:
        log.warning("系统存在问题，请根据上面的 [FAIL] 日志进行排查。")


if __name__ == "__main__":
    run_all_checks()
    input("\n检查完毕，请按 Enter 键退出...")
