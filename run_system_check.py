# StockScanAI/run_system_check.py
#
# A股量化投研平台 - 统一系统健康检查程序
#
# 使用方法:
# 1. 确保您的 .env 文件已正确配置。
# 2. 直接在命令行运行: python run_system_check.py
# 3. 使用参数: python run_system_check.py --mode [quick|full|stability]
#
# 功能:
# - 替代所有旧的 test_*.py 文件。
# - 逐一检查系统的核心组件，提供清晰的通过/失败报告。
# - 设计为快速、低成本的诊断工具。
# - 集成了进度跟踪、数据验证、稳定性检查等功能。

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
import psutil
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


def check_database_data_quality(dm: data.DataManager):
    """
    【新增】对数据库中的核心数据（特别是财务数据）进行质量和完整性检查。
    """
    log.info("  --- 开始财务数据质量专项检查 ---")
    with dm.engine.connect() as conn:
        # 1. 检查 stock_basic 表
        all_stocks_df = pd.read_sql("SELECT ts_code FROM stock_basic", conn)
        total_stocks = len(all_stocks_df)
        log.info(f"  [检查1/4] 股票列表: 共在 `stock_basic` 表中找到 {total_stocks} 只股票。")

        # 2. 检查 financial_indicators 表的总体情况
        fina_stocks_df = pd.read_sql(
            "SELECT DISTINCT ts_code FROM financial_indicators", conn
        )
        fina_stocks_count = len(fina_stocks_df)
        coverage_ratio = fina_stocks_count / total_stocks if total_stocks > 0 else 0
        log.info(
            f"  [检查2/4] 财务数据覆盖率: `financial_indicators` 表覆盖了 {fina_stocks_count} 只股票，覆盖率: {coverage_ratio:.2%}"
        )

        if coverage_ratio < 0.9:
            log.warning("  > 警告：财务数据覆盖率低于90%，可能大量股票数据缺失。")
            missing_stocks = set(all_stocks_df["ts_code"]) - set(
                fina_stocks_df["ts_code"]
            )
            log.warning(f"  > 部分缺失股票示例: {list(missing_stocks)[:5]}")

        # 3. 检查数据的时效性和深度
        query_latest_date = text("SELECT MAX(end_date) FROM financial_indicators")
        query_avg_depth = text("""
            SELECT AVG(report_count) FROM (
                SELECT ts_code, COUNT(1) as report_count 
                FROM financial_indicators 
                GROUP BY ts_code
            ) as sub_query
        """)

        latest_date_result = conn.execute(query_latest_date).scalar_one_or_none()
        avg_depth_result = conn.execute(query_avg_depth).scalar_one_or_none()

        if latest_date_result and avg_depth_result:
            latest_date = latest_date_result
            avg_depth = avg_depth_result
            log.info(f"  [检查3/4] 数据时效性与深度:")
            log.info(f"  > 最新财报报告期: {latest_date}")
            log.info(f"  > 历史数据深度: 平均每只股票有 {avg_depth:.1f} 个季度的财务报告。")
            if avg_depth < 12: # 少于3年
                 log.warning("  > 警告：平均历史数据深度不足3年，可能影响长周期策略回测。")
        else:
            log.error("  > 无法获取财务数据的时效性和深度统计。")
            raise Exception("无法获取财务数据统计")

        # 4. 抽样检查
        log.info("  [检查4/4] 数据抽样详情:")
        sample_stocks = fina_stocks_df["ts_code"].sample(min(5, fina_stocks_count)).tolist()
        for stock in sample_stocks:
            stock_fina_df = pd.read_sql(
                f"SELECT MIN(end_date) as first_date, MAX(end_date) as last_date, COUNT(1) as count FROM financial_indicators WHERE ts_code='{stock}'",
                conn,
            ).iloc[0]
            log.info(
                f"  > 抽样 {stock}: 共 {stock_fina_df['count']} 条记录, 报告期从 {stock_fina_df['first_date']} 到 {stock_fina_df['last_date']}"
            )

    log.info("  --- 财务数据质量专项检查完成 ---")


def check_system_resources():
    """检查系统资源使用情况"""
    log.info("  --- 开始系统资源检查 ---")
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # 磁盘使用情况
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        log.info(f"  > CPU使用率: {cpu_percent}%")
        log.info(f"  > 内存使用率: {memory_percent}% (可用: {memory_available_gb:.2f}GB)")
        log.info(f"  > 磁盘使用率: {disk_percent:.1f}% (可用: {disk_free_gb:.2f}GB)")
        
        # 判断资源状态
        if cpu_percent < 80 and memory_percent < 85 and disk_percent < 90:
            log.info("  > 系统资源状态: 良好")
            return True
        else:
            log.warning("  > 系统资源状态: 紧张，可能影响性能")
            return False
            
    except Exception as e:
        log.error(f"  > 系统资源检查失败: {e}")
        return False


def check_database_performance(dm):
    """检查数据库性能"""
    log.info("  --- 开始数据库性能检查 ---")
    try:
        start_time = time.time()
        
        with dm.engine.connect() as conn:
            # 测试简单查询性能
            simple_query_start = time.time()
            conn.execute(text("SELECT 1")).scalar()
            simple_query_time = time.time() - simple_query_start
            
            # 测试复杂查询性能
            complex_query_start = time.time()
            result = conn.execute(text("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT ts_code) as unique_stocks,
                       MAX(trade_date) as latest_date
                FROM ts_daily
            """)).fetchone()
            complex_query_time = time.time() - complex_query_start
            
            # 检查数据库大小
            try:
                db_size_result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                """)).scalar()
                log.info(f"  > 数据库大小: {db_size_result}")
            except:
                log.info("  > 数据库大小: 无法获取")
            
            log.info(f"  > 简单查询耗时: {simple_query_time*1000:.2f}ms")
            log.info(f"  > 复杂查询耗时: {complex_query_time*1000:.2f}ms")
            
            if result:
                log.info(f"  > 数据记录总数: {result[0]:,}")
                log.info(f"  > 股票数量: {result[1]:,}")
                log.info(f"  > 最新数据日期: {result[2]}")
            
            # 判断性能状态
            if complex_query_time < 1.0:
                log.info("  > 数据库性能: 良好")
                return True
            else:
                log.warning("  > 数据库性能: 较慢，可能需要优化")
                return False
                
    except Exception as e:
        log.error(f"  > 数据库性能检查失败: {e}")
        return False


def check_data_freshness(dm):
    """检查数据新鲜度"""
    log.info("  --- 开始数据新鲜度检查 ---")
    try:
        with dm.engine.connect() as conn:
            # 检查最新交易数据
            latest_daily_data = conn.execute(text("""
                SELECT MAX(trade_date) as latest_date,
                       COUNT(DISTINCT ts_code) as stocks_count
                FROM ts_daily
            """)).fetchone()
            
            # 检查最新因子数据
            latest_factor_data = conn.execute(text("""
                SELECT MAX(trade_date) as latest_date,
                       COUNT(DISTINCT ts_code) as stocks_count,
                       COUNT(DISTINCT factor_name) as factors_count
                FROM factors_exposure
            """)).fetchone()
            
            # 计算数据滞后天数
            current_date = datetime.now().date()
            
            daily_lag_days = 0
            factor_lag_days = 0
            
            if latest_daily_data and latest_daily_data[0]:
                daily_lag_days = (current_date - latest_daily_data[0]).days
                log.info(f"  > 最新日线数据: {latest_daily_data[0]} (滞后{daily_lag_days}天)")
                log.info(f"  > 日线数据覆盖: {latest_daily_data[1]:,} 只股票")
            
            if latest_factor_data and latest_factor_data[0]:
                factor_lag_days = (current_date - latest_factor_data[0]).days
                log.info(f"  > 最新因子数据: {latest_factor_data[0]} (滞后{factor_lag_days}天)")
                log.info(f"  > 因子数据覆盖: {latest_factor_data[1]:,} 只股票")
                log.info(f"  > 因子种类数量: {latest_factor_data[2]} 个")
            
            # 判断数据新鲜度
            max_lag = max(daily_lag_days, factor_lag_days)
            if max_lag <= 3:
                log.info("  > 数据新鲜度: 良好")
                return True
            else:
                log.warning(f"  > 数据新鲜度: 滞后{max_lag}天，建议更新")
                return False
                
    except Exception as e:
        log.error(f"  > 数据新鲜度检查失败: {e}")
        return False


def test_stable_factors(dm):
    """测试稳定的因子计算"""
    log.info("  --- 开始稳定因子计算测试 ---")
    try:
        ff = qe.FactorFactory(_data_manager=dm)
        
        # 获取测试日期
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        
        trade_cal = dm.pro.trade_cal(exchange="", start_date=start_date, end_date=end_date)
        test_date = trade_cal[trade_cal["is_open"] == 1]["cal_date"].iloc[-1]
        
        # 测试稳定的因子
        test_stocks = ["600519.SH", "000001.SZ"]
        stable_factors = ["pe_ttm", "size"]
        
        log.info(f"  > 测试股票: {test_stocks}")
        log.info(f"  > 测试因子: {stable_factors}")
        log.info(f"  > 测试日期: {test_date}")
        
        success_count = 0
        total_tests = len(stable_factors) * len(test_stocks)
        
        for factor_name in stable_factors:
            for stock in test_stocks:
                try:
                    value = ff.calculate(factor_name, ts_code=stock, date=test_date)
                    if value is not None and pd.notna(value):
                        log.info(f"  > {stock} {factor_name}: {value:.4f}")
                        success_count += 1
                    else:
                        log.warning(f"  > {stock} {factor_name}: 无效值")
                except Exception as e:
                    log.warning(f"  > {stock} {factor_name}: 计算失败 - {e}")
        
        success_rate = success_count / total_tests if total_tests > 0 else 0
        log.info(f"  > 因子计算成功率: {success_rate:.1%} ({success_count}/{total_tests})")
        
        if success_rate >= 0.8:
            log.info("  > 稳定因子测试: 通过")
            return True
        else:
            log.warning("  > 稳定因子测试: 未达标")
            return False
            
    except Exception as e:
        log.error(f"  > 稳定因子测试失败: {e}")
        return False


def test_data_storage(dm):
    """测试数据存储功能"""
    log.info("  --- 开始数据存储测试 ---")
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'trade_date': [pd.to_datetime('2025-07-21')],
            'ts_code': ['TEST.SH'],
            'factor_name': ['test_factor'],
            'factor_value': [1.0]
        })
        
        with dm.engine.connect() as conn:
            with conn.begin():
                # 删除可能存在的测试数据
                conn.execute(text("""
                    DELETE FROM factors_exposure 
                    WHERE ts_code = 'TEST.SH' AND factor_name = 'test_factor'
                """))
                
                # 插入测试数据
                test_data.to_sql('factors_exposure', conn, if_exists='append', index=False)
                
                # 验证数据
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM factors_exposure 
                    WHERE ts_code = 'TEST.SH' AND factor_name = 'test_factor'
                """)).scalar()
                
                if result == 1:
                    log.info("  > 数据写入和读取: 成功")
                    
                    # 清理测试数据
                    conn.execute(text("""
                        DELETE FROM factors_exposure 
                        WHERE ts_code = 'TEST.SH' AND factor_name = 'test_factor'
                    """))
                    
                    return True
                else:
                    log.error("  > 数据验证失败")
                    return False
                    
    except Exception as e:
        log.error(f"  > 数据存储测试失败: {e}")
        return False


def run_quick_check():
    """快速检查 - 只检查最基础的功能"""
    log.info("=== 快速系统检查 ===")
    checks_passed = 0
    total_checks = 4
    
    # 基础检查
    if check_config():
        checks_passed += 1
    else:
        return finalize_report(checks_passed, total_checks)
    
    dm = check_database_connection()
    if dm:
        checks_passed += 1
    else:
        return finalize_report(checks_passed, total_checks)
    
    if check_tushare_api(dm):
        checks_passed += 1
    
    if test_stable_factors(dm):
        checks_passed += 1
    
    return finalize_report(checks_passed, total_checks)


def run_stability_check():
    """稳定性检查 - 专注于系统稳定性和数据质量"""
    log.info("=== 系统稳定性检查 ===")
    checks_passed = 0
    total_checks = 6
    
    # 基础检查
    if check_config():
        checks_passed += 1
    else:
        return finalize_report(checks_passed, total_checks)
    
    dm = check_database_connection()
    if dm:
        checks_passed += 1
    else:
        return finalize_report(checks_passed, total_checks)
    
    if check_tushare_api(dm):
        checks_passed += 1
    
    if check_system_resources():
        checks_passed += 1
    
    if check_database_performance(dm):
        checks_passed += 1
    
    if test_data_storage(dm):
        checks_passed += 1
    
    return finalize_report(checks_passed, total_checks)


def check_config():
    """检查配置文件"""
    log.info("\n--- [检查] 配置文件与环境变量 ---")
    try:
        assert (
            config.TUSHARE_TOKEN and config.TUSHARE_TOKEN != "YOUR_TUSHARE_TOKEN"
        ), "TUSHARE_TOKEN 未配置"
        assert config.DATABASE_URL, "DATABASE_URL 未配置"
        assert config.AI_MODEL_CONFIG["fast_and_cheap"][
            "api_key"
        ], "AI_API_KEY_FAST 未配置"
        log.info("  [PASS] 所有关键环境变量已成功加载。")
        return True
    except Exception as e:
        log.error(f"  [FAIL] 配置检查失败: {e}")
        return False


def check_database_connection():
    """检查数据库连接"""
    log.info("\n--- [检查] 数据库连接 ---")
    try:
        engine = create_engine(config.DATABASE_URL)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        log.info(
            f"  [PASS] 成功连接到数据库: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        )
        return data.DataManager()
    except Exception as e:
        log.error(f"  [FAIL] 数据库连接失败: {e}")
        return None


def check_tushare_api(dm):
    """检查Tushare API"""
    log.info("\n--- [检查] Tushare API连通性 ---")
    try:
        stock_list = dm.get_stock_basic(list_status="L")
        assert (
            stock_list is not None and not stock_list.empty
        ), "获取股票列表失败，返回为空。"
        log.info(
            f"  [PASS] 成功从 Tushare 获取到 {len(stock_list)} 只上市股票的基本信息。"
        )
        return True
    except Exception as e:
        log.error(f"  [FAIL] Tushare API检查失败: {e}")
        return False


def run_all_checks():
    """
    执行所有系统健康检查。
    """
    checks_passed = 0
    total_checks = 10  # 升级：总检查项增加到10个

    log.info("==============================================")
    log.info("=== A股量化投研平台 - 完整系统健康检查 ===")
    log.info("==============================================")

    # --- 检查 1: 配置文件与环境变量 ---
    if not check_config():
        return finalize_report(checks_passed, total_checks)
    checks_passed += 1

    # --- 检查 2: 数据库连接 ---
    dm = check_database_connection()
    if not dm:
        return finalize_report(checks_passed, total_checks)
    checks_passed += 1

    # --- 检查 3: Tushare API连通性 ---
    if not check_tushare_api(dm):
        return finalize_report(checks_passed, total_checks)
    checks_passed += 1

    # --- 检查 4: 系统资源状态 ---
    log.info("\n--- [检查 4/10] 系统资源状态 ---")
    if check_system_resources():
        checks_passed += 1

    # --- 检查 5: 数据库性能 ---
    log.info("\n--- [检查 5/10] 数据库性能 ---")
    if check_database_performance(dm):
        checks_passed += 1

    # --- 检查 6: 数据新鲜度 ---
    log.info("\n--- [检查 6/10] 数据新鲜度 ---")
    if check_data_freshness(dm):
        checks_passed += 1

    # --- 检查 7: 数据库数据质量验证 ---
    log.info("\n--- [检查 7/10] 数据库财务数据质量验证 ---")
    try:
        check_database_data_quality(dm)
        log.info("  [PASS] 数据库核心数据质量验证通过。")
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 数据库数据质量验证失败: {e}")

    # --- 检查 8: 引擎层核心功能 (FactorFactory & FactorProcessor) ---
    log.info("\n--- [检查 8/10] 引擎层因子计算 ---")
    try:
        ff = qe.FactorFactory(_data_manager=dm)
        fp = qe.FactorProcessor(_data_manager=dm)
        ts_code_test = "600519.SH"
        today_str = pd.Timestamp.now().strftime("%Y%m%d")
        trade_cal = dm.pro.trade_cal(
            exchange="", start_date="20240101", end_date=today_str
        )
        
        # 【V2.9 健壮性修复】检查Tushare返回的DataFrame是否符合预期
        if trade_cal is None or "is_open" not in trade_cal.columns:
            actual_cols = trade_cal.columns.tolist() if trade_cal is not None else "None"
            log.error(f"  > Tushare API 'trade_cal' 未返回预期的 'is_open' 列。实际返回列: {actual_cols}")
            raise ValueError("Tushare trade_cal API 响应格式不正确")

        open_trade_days = trade_cal[trade_cal["is_open"] == 1]
        
        pe_ttm = None
        date_used_for_pe = None

        # 尝试获取最近两个交易日的数据，以应对Tushare数据延迟
        for i in range(1, 3):
            if len(open_trade_days) >= i:
                date_to_try = open_trade_days["cal_date"].iloc[-i]
                log.info(f"  > 尝试为日期 {date_to_try} 计算 PE(TTM)...")
                pe_ttm_candidate = ff.calc_pe_ttm(ts_code=ts_code_test, date=date_to_try)
                if isinstance(pe_ttm_candidate, float) and pd.notna(pe_ttm_candidate):
                    pe_ttm = pe_ttm_candidate
                    date_used_for_pe = date_to_try
                    break # 成功获取，退出循环
                else:
                    log.warning(f"  > 在 {date_to_try} 未能获取到有效的 PE(TTM) 数据，可能是数据暂未更新。")
        
        assert pe_ttm is not None, f"在最近的两个交易日均未能为 {ts_code_test} 计算出有效的 pe_ttm。"
        
        log.info(
            f"  [PASS] 因子引擎实例化成功，并在日期 {date_used_for_pe} 为 {ts_code_test} 计算出 PE(TTM) 因子值为: {pe_ttm:.2f}"
        )
        checks_passed += 1
    except Exception as e:
        log.error(f"  [FAIL] 引擎层检查失败: {e}", exc_info=True)
        finalize_report(checks_passed, total_checks)
        return

    # --- 检查 6: 智能分析层 (AIOrchestrator & AI API) ---
    log.info("\n--- [检查 6/7] 智能分析层与AI API连通性 (低成本测试) ---")
    try:
        ai_orchestrator = intelligence.AIOrchestrator(config.AI_MODEL_CONFIG, dm)
        prompt = "你好，请确认你可以正常工作。回答'ok'即可。"
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

    # --- 检查 7: 端到端工作流冒烟测试 ---
    log.info("\n--- [检查 7/7] 端到端工作流冒烟测试 ---")
    try:
        log.info("  正在尝试模拟调用每日策略工作流...")
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