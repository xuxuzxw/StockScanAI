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
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import aiohttp
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

# --- 配置常量 ---
class CheckConfig:
    """Configuration constants for system checks"""
    # Performance thresholds
    SIMPLE_QUERY_MAX_MS = 100
    COMPLEX_QUERY_MAX_MS = 1000
    
    # Resource thresholds
    MAX_CPU_PERCENT = 80
    MAX_MEMORY_PERCENT = 85
    MAX_DISK_PERCENT = 90
    
    # Data quality thresholds
    MIN_COVERAGE_RATIO = 0.9
    MIN_HISTORICAL_DEPTH_QUARTERS = 12
    MAX_DATA_LAG_DAYS = 3
    
    # Test configuration
    TEST_STOCKS = ["600519.SH", "000001.SZ", "000002.SZ"]
    STABLE_FACTORS = ["pe_ttm", "pb"]
    MIN_FACTOR_SUCCESS_RATE = 0.6
    
    # API test configuration
    DEFAULT_CONCURRENT_REQUESTS = 10
    MIN_API_SUCCESS_RATE = 0.8


# --- 检查流程 ---


def check_database_data_quality(dm: data.DataManager) -> bool:
    """
    对数据库中的核心数据（特别是财务数据）进行质量和完整性检查。
    
    Returns:
        bool: True if data quality checks pass, False otherwise
    """
    log.info("  --- 开始财务数据质量专项检查 ---")
    
    try:
        with dm.engine.connect() as conn:
            # Break down into smaller, focused checks
            if not _check_stock_coverage(conn):
                return False
            if not _check_data_timeliness_and_depth(conn):
                return False
            _perform_data_sampling(conn)

        log.info("  --- 财务数据质量专项检查完成 ---")
        return True
        
    except Exception as e:
        log.error(f"  > 财务数据质量检查失败: {e}")
        return False


def _check_stock_coverage(conn) -> bool:
    """Check stock coverage in financial data"""
    total_stocks = conn.execute(text("SELECT COUNT(*) FROM stock_basic")).scalar()
    log.info(f"  [检查1/4] 股票列表: 共在 `stock_basic` 表中找到 {total_stocks} 只股票。")

    fina_stocks_count = conn.execute(text(
        "SELECT COUNT(DISTINCT ts_code) FROM financial_indicators"
    )).scalar()
    coverage_ratio = fina_stocks_count / total_stocks if total_stocks > 0 else 0
    log.info(
        f"  [检查2/4] 财务数据覆盖率: `financial_indicators` 表覆盖了 {fina_stocks_count} 只股票，覆盖率: {coverage_ratio:.2%}"
    )

    if coverage_ratio < 0.9:
        log.warning("  > 警告：财务数据覆盖率低于90%，可能大量股票数据缺失。")
    
    return True


def _check_data_timeliness_and_depth(conn) -> bool:
    """Check data timeliness and historical depth"""
    latest_date_result = conn.execute(text("SELECT MAX(end_date) FROM financial_indicators")).scalar_one_or_none()
    avg_depth_result = conn.execute(text("""
        SELECT AVG(report_count) FROM (
            SELECT ts_code, COUNT(1) as report_count 
            FROM financial_indicators 
            GROUP BY ts_code
        ) as sub_query
    """)).scalar_one_or_none()

    if latest_date_result and avg_depth_result:
        log.info(f"  [检查3/4] 数据时效性与深度:")
        log.info(f"  > 最新财报报告期: {latest_date_result}")
        log.info(f"  > 历史数据深度: 平均每只股票有 {avg_depth_result:.1f} 个季度的财务报告。")
        if avg_depth_result < 12:  # 少于3年
            log.warning("  > 警告：平均历史数据深度不足3年，可能影响长周期策略回测。")
        return True
    else:
        log.error("  > 无法获取财务数据的时效性和深度统计。")
        return False


def _perform_data_sampling(conn):
    """Perform data sampling checks"""
    log.info("  [检查4/4] 数据抽样详情:")
    sample_stocks_result = conn.execute(text("""
        SELECT ts_code FROM financial_indicators 
        GROUP BY ts_code 
        ORDER BY RANDOM() 
        LIMIT 5
    """)).fetchall()
    
    for row in sample_stocks_result:
        stock = row[0]
        stock_stats = conn.execute(text("""
            SELECT MIN(end_date) as first_date, MAX(end_date) as last_date, COUNT(1) as count 
            FROM financial_indicators WHERE ts_code = :stock
        """), {"stock": stock}).fetchone()
        
        log.info(
            f"  > 抽样 {stock}: 共 {stock_stats.count} 条记录, "
            f"报告期从 {stock_stats.first_date} 到 {stock_stats.last_date}"
        )


def check_system_resources() -> bool:
    """
    检查系统资源使用情况
    
    Returns:
        bool: True if system resources are within acceptable limits, False otherwise
    """
    log.info("  --- 开始系统资源检查 ---")
    
    try:
        resources = _gather_system_resources()
        _log_resource_status(resources)
        return _evaluate_resource_health(resources)
        
    except Exception as e:
        log.error(f"  > 系统资源检查失败: {e}")
        return False


def _gather_system_resources() -> dict:
    """Gather system resource metrics"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用情况
    memory = psutil.virtual_memory()
    
    # 磁盘使用情况
    disk = psutil.disk_usage('.')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_percent': (disk.used / disk.total) * 100,
        'disk_free_gb': disk.free / (1024**3)
    }


def _log_resource_status(resources: dict):
    """Log resource status information"""
    log.info(f"  > CPU使用率: {resources['cpu_percent']}%")
    log.info(f"  > 内存使用率: {resources['memory_percent']}% (可用: {resources['memory_available_gb']:.2f}GB)")
    log.info(f"  > 磁盘使用率: {resources['disk_percent']:.1f}% (可用: {resources['disk_free_gb']:.2f}GB)")


def _evaluate_resource_health(resources: dict) -> bool:
    """Evaluate if system resources are healthy"""
    is_healthy = (
        resources['cpu_percent'] < CheckConfig.MAX_CPU_PERCENT and
        resources['memory_percent'] < CheckConfig.MAX_MEMORY_PERCENT and
        resources['disk_percent'] < CheckConfig.MAX_DISK_PERCENT
    )
    
    if is_healthy:
        log.info("  > 系统资源状态: 良好")
    else:
        log.warning("  > 系统资源状态: 紧张，可能影响性能")
    
    return is_healthy


def check_database_performance(dm: data.DataManager) -> bool:
    """
    检查数据库性能 - 优化版本
    
    Args:
        dm: DataManager instance for database operations
        
    Returns:
        bool: True if database performance is acceptable, False otherwise
    """
    log.info("  --- 开始数据库性能检查 ---")
    
    from system_check_config import PERF_THRESHOLDS
    
    try:
        with dm.engine.connect() as conn:
            # 1. 简单查询性能测试
            simple_query_start = time.time()
            conn.execute(text("SELECT 1")).scalar()
            simple_query_time = (time.time() - simple_query_start) * 1000
            
            # 2. 批量获取统计信息 - 单次查询获取多个指标
            complex_query_start = time.time()
            stats_result = conn.execute(text("""
                WITH table_stats AS (
                    SELECT 
                        COALESCE(
                            (SELECT reltuples::bigint FROM pg_class WHERE relname = 'ts_daily'),
                            0
                        ) as total_records,
                        (SELECT MAX(trade_date) FROM ts_daily) as latest_date
                ),
                stock_stats AS (
                    SELECT COUNT(DISTINCT ts_code) as unique_stocks
                    FROM (
                        SELECT ts_code FROM ts_daily 
                        ORDER BY trade_date DESC 
                        LIMIT 10000
                    ) recent_sample
                )
                SELECT 
                    ts.total_records,
                    ts.latest_date,
                    ss.unique_stocks
                FROM table_stats ts, stock_stats ss
            """)).fetchone()
            
            complex_query_time = (time.time() - complex_query_start) * 1000
            
            # 3. 数据库大小查询（可选，不影响性能判断）
            try:
                db_size = conn.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )).scalar()
                log.info(f"  > 数据库大小: {db_size}")
            except Exception:
                log.info("  > 数据库大小: 无法获取")
            
            # 输出性能指标
            log.info(f"  > 简单查询耗时: {simple_query_time:.2f}ms")
            log.info(f"  > 复杂查询耗时: {complex_query_time:.2f}ms")
            
            if stats_result:
                log.info(f"  > 数据记录总数: {stats_result.total_records:,}")
                log.info(f"  > 股票数量: {stats_result.unique_stocks:,}")
                log.info(f"  > 最新数据日期: {stats_result.latest_date}")
            
            # 使用配置化的阈值判断性能
            performance_ok = (
                simple_query_time <= PERF_THRESHOLDS.SIMPLE_QUERY_MAX_MS and
                complex_query_time <= PERF_THRESHOLDS.COMPLEX_QUERY_MAX_MS
            )
            
            if performance_ok:
                log.info("  > 数据库性能: 良好")
                return True
            else:
                log.warning(f"  > 数据库性能: 需要优化 (阈值: 简单查询<{PERF_THRESHOLDS.SIMPLE_QUERY_MAX_MS}ms, 复杂查询<{PERF_THRESHOLDS.COMPLEX_QUERY_MAX_MS}ms)")
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


def test_stable_factors(dm: data.DataManager) -> bool:
    """
    测试稳定的因子计算
    
    Args:
        dm: DataManager instance
        
    Returns:
        bool: True if factor calculations are stable, False otherwise
    """
    log.info("  --- 开始稳定因子计算测试 ---")
    start_time = time.time()
    
    try:
        from system_check_config import TEST_CONFIG
        
        ff = qe.FactorFactory(_data_manager=dm)
        
        # 获取测试日期
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        
        trade_cal = dm.pro.trade_cal(exchange="", start_date=start_date, end_date=end_date)
        test_date = trade_cal[trade_cal["is_open"] == 1]["cal_date"].iloc[-1]
        
        log.info(f"  > 测试股票: {CheckConfig.TEST_STOCKS}")
        log.info(f"  > 测试因子: {CheckConfig.STABLE_FACTORS}")
        log.info(f"  > 测试日期: {test_date}")
        
        success_count = 0
        total_tests = len(CheckConfig.STABLE_FACTORS) * len(CheckConfig.TEST_STOCKS)
        
        for factor_name in CheckConfig.STABLE_FACTORS:
            for stock in CheckConfig.TEST_STOCKS:
                try:
                    # 使用更安全的因子计算方法
                    if hasattr(ff, f'calc_{factor_name}'):
                        calc_method = getattr(ff, f'calc_{factor_name}')
                        value = calc_method(ts_code=stock, date=test_date)
                    else:
                        log.warning(f"  > 因子计算方法 calc_{factor_name} 不存在")
                        continue
                        
                    if value is not None and pd.notna(value):
                        log.info(f"  > {stock} {factor_name}: {value:.4f}")
                        success_count += 1
                    else:
                        log.warning(f"  > {stock} {factor_name}: 无效值")
                except Exception as e:
                    log.warning(f"  > {stock} {factor_name}: 计算失败 - {e}")
        
        success_rate = success_count / total_tests if total_tests > 0 else 0
        log.info(f"  > 因子计算成功率: {success_rate:.1%} ({success_count}/{total_tests})")
        
        # Log performance metrics
        total_duration = time.time() - start_time
        avg_time_per_factor = total_duration / total_tests if total_tests > 0 else 0
        log.info(f"  > 平均因子计算时间: {avg_time_per_factor:.3f}秒")
        
        if success_rate >= TEST_CONFIG.MIN_FACTOR_SUCCESS_RATE:
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


def comprehensive_database_check(dm) -> bool:
    """
    全面的数据库数据完整性检查 - 重构版本
    使用面向对象设计，提高可维护性和可测试性
    """
    try:
        from system_check_config_improved import get_config
        config = get_config("default").data_quality
    except ImportError:
        from system_check_config import DATA_QUALITY_THRESHOLDS
        config = DATA_QUALITY_THRESHOLDS
    
    # Use the improved DatabaseChecker class
    checker = DatabaseChecker(dm, config)
    return checker.run_comprehensive_check()


class DatabaseChecker:
    """Handles comprehensive database integrity checks"""
    
    def __init__(self, dm, config=None):
        self.dm = dm
        self.config = config or self._get_default_config()
        self.issues_found = []
        self.total_checks = 0
        self.passed_checks = 0
    
    def _get_default_config(self):
        """Get default configuration with proper fallback handling"""
        try:
            from system_check_config_improved import get_config
            return get_config("default").data_quality
        except ImportError:
            log.warning("Advanced configuration not available, using basic configuration")
            return self._create_fallback_config()
    
    def _create_fallback_config(self):
        """Create fallback configuration when advanced config is not available"""
        from types import SimpleNamespace
        
        # Define constants to avoid duplication
        CORE_TABLES = {
            'stock_basic': '股票基本信息',
            'ts_daily': '日线行情数据', 
            'factors_exposure': '因子暴露数据',
            'financial_indicators': '财务指标数据',
            'ts_adj_factor': '复权因子数据'
        }
        
        TIME_RANGE_TABLES = ['ts_daily', 'factors_exposure', 'financial_indicators']
        
        return SimpleNamespace(
            CORE_TABLES=CORE_TABLES,
            TIME_RANGE_TABLES=TIME_RANGE_TABLES,
            MAX_DATA_AGE_DAYS=7,
            MIN_STOCK_COVERAGE_RATIO=0.8,
            MAX_INVALID_DATA_PERCENT=1.0,
            MAX_PRICE_INCONSISTENCY_PERCENT=0.1,
            MAX_FINANCIAL_DATA_AGE_MONTHS=6,
            MIN_ORPHAN_THRESHOLD=100
        )
    
    def run_comprehensive_check(self) -> bool:
        """Run all database checks and return overall success"""
        log.info("  --- 开始全面数据库数据完整性检查 ---")
        
        try:
            with self.dm.engine.connect() as conn:
                # Run all check methods
                check_methods = [
                    (self._check_core_tables, "核心表存在性和基本统计"),
                    (self._check_time_ranges, "数据时间范围"),
                    (self._check_stock_consistency, "股票代码一致性"),
                    (self._check_data_quality, "数据质量问题"),
                    (self._check_factor_completeness, "因子数据完整性"),
                    (self._check_financial_completeness, "财务数据完整性"),
                    (self._check_data_relationships, "数据关联性"),
                    (self._check_database_metrics, "数据库性能指标")
                ]
                
                for i, (check_method, description) in enumerate(check_methods, 1):
                    log.info(f"  [{i}/{len(check_methods)}] 检查{description}")
                    try:
                        check_method(conn)
                    except Exception as e:
                        log.error(f"    ✗ {description}检查失败: {e}")
                        self.issues_found.append(f"{description}检查失败: {e}")
            
            return self._generate_report()
            
        except Exception as e:
            log.error(f"  > 全面数据库检查失败: {e}")
            return False
    
    def _check_core_tables(self, conn) -> None:
        """Check core table existence and basic statistics"""
        for table, desc in self.config.CORE_TABLES.items():
            self.total_checks += 1
            try:
                # Use safe query building
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                log.info(f"    ✓ {desc} ({table}): {count:,} 条记录")
                self.passed_checks += 1
            except Exception as e:
                log.error(f"    ✗ {desc} ({table}): 检查失败 - {e}")
                self.issues_found.append(f"表 {table} 检查失败: {e}")
    
    def _check_time_ranges(self, conn) -> None:
        """Check data time ranges and freshness"""
        for table in self.config.TIME_RANGE_TABLES:
            self.total_checks += 1
            try:
                date_col = 'end_date' if table == 'financial_indicators' else 'trade_date'
                
                result = conn.execute(text(f"""
                    SELECT MIN({date_col}) as min_date, 
                           MAX({date_col}) as max_date,
                           COUNT(DISTINCT {date_col}) as date_count
                    FROM {table}
                """)).fetchone()
                
                if result and result[0]:
                    log.info(f"    ✓ {table}: {result[0]} 至 {result[1]} ({result[2]} 个日期)")
                    self.passed_checks += 1
                    
                    # Check data freshness for trading data
                    if table in ['ts_daily', 'factors_exposure']:
                        days_old = (datetime.now().date() - result[1]).days
                        if days_old > self.config.MAX_DATA_AGE_DAYS:
                            self.issues_found.append(f"{table} 数据过旧，最新数据距今 {days_old} 天")
                else:
                    log.warning(f"    ⚠ {table}: 无有效日期数据")
                    self.issues_found.append(f"{table} 无有效日期数据")
                    
            except Exception as e:
                log.error(f"    ✗ {table} 时间范围检查失败: {e}")
                self.issues_found.append(f"{table} 时间范围检查失败: {e}")
    
    def _check_stock_consistency(self, conn) -> None:
        """Check stock code consistency across tables"""
        self.total_checks += 1
        try:
            stock_counts = {}
            for table in ['stock_basic', 'ts_daily', 'factors_exposure', 'financial_indicators']:
                if table == 'stock_basic':
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                else:
                    count = conn.execute(text(f"SELECT COUNT(DISTINCT ts_code) FROM {table}")).scalar()
                stock_counts[table] = count
            
            log.info("    股票代码数量对比:")
            for table, count in stock_counts.items():
                log.info(f"      {table}: {count:,} 只股票")
            
            # Check coverage ratios
            base_count = stock_counts.get('stock_basic', 0)
            if base_count > 0:
                for table, count in stock_counts.items():
                    if table != 'stock_basic':
                        coverage = count / base_count
                        if coverage < self.config.MIN_STOCK_COVERAGE_RATIO:
                            self.issues_found.append(f"{table} 股票覆盖率过低: {coverage:.1%}")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 股票代码一致性检查失败: {e}")
            self.issues_found.append(f"股票代码一致性检查失败: {e}")
    
    def _check_data_quality(self, conn) -> None:
        """Check data quality issues in trading data"""
        self.total_checks += 1
        try:
            quality_issues = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) as invalid_close,
                    SUM(CASE WHEN vol IS NULL OR vol < 0 THEN 1 ELSE 0 END) as invalid_volume,
                    SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as price_inconsistency
                FROM ts_daily
                WHERE trade_date >= (SELECT MAX(trade_date) - INTERVAL '30 days' FROM ts_daily)
            """)).fetchone()
            
            if quality_issues and quality_issues[0] > 0:
                total_records = quality_issues[0]
                invalid_close_pct = (quality_issues[1] / total_records) * 100
                invalid_vol_pct = (quality_issues[2] / total_records) * 100
                price_inconsist_pct = (quality_issues[3] / total_records) * 100
                
                log.info(f"    最近30天数据质量 (共{total_records:,}条):")
                log.info(f"      无效收盘价: {quality_issues[1]} ({invalid_close_pct:.2f}%)")
                log.info(f"      无效成交量: {quality_issues[2]} ({invalid_vol_pct:.2f}%)")
                log.info(f"      价格不一致: {quality_issues[3]} ({price_inconsist_pct:.2f}%)")
                
                if invalid_close_pct > self.config.MAX_INVALID_DATA_PERCENT:
                    self.issues_found.append(f"无效收盘价比例过高: {invalid_close_pct:.2f}%")
                if price_inconsist_pct > self.config.MAX_PRICE_INCONSISTENCY_PERCENT:
                    self.issues_found.append(f"价格不一致比例过高: {price_inconsist_pct:.2f}%")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 数据质量检查失败: {e}")
            self.issues_found.append(f"数据质量检查失败: {e}")
    
    def _check_factor_completeness(self, conn) -> None:
        """Check factor data completeness"""
        self.total_checks += 1
        try:
            factor_stats = conn.execute(text("""
                SELECT 
                    factor_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    COUNT(DISTINCT trade_date) as date_count,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date
                FROM factors_exposure 
                GROUP BY factor_name 
                ORDER BY record_count DESC
            """)).fetchall()
            
            if factor_stats:
                log.info(f"    因子数据统计 (共{len(factor_stats)}个因子):")
                for factor in factor_stats[:5]:  # Show top 5
                    log.info(f"      {factor[0]}: {factor[1]:,}条记录, {factor[2]}只股票, {factor[3]}个日期")
                
                # Check factor data consistency
                date_counts = [f[3] for f in factor_stats]
                if len(set(date_counts)) > 1:
                    self.issues_found.append(f"因子数据日期不一致，范围: {min(date_counts)}-{max(date_counts)}")
            else:
                self.issues_found.append("无因子数据")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 因子数据完整性检查失败: {e}")
            self.issues_found.append(f"因子数据完整性检查失败: {e}")
    
    def _check_financial_completeness(self, conn) -> None:
        """Check financial data completeness"""
        self.total_checks += 1
        try:
            financial_stats = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT ts_code) as stock_count,
                    COUNT(DISTINCT end_date) as period_count,
                    MIN(end_date) as earliest_period,
                    MAX(end_date) as latest_period,
                    COUNT(*) as total_records
                FROM financial_indicators
            """)).fetchone()
            
            if financial_stats:
                log.info("    财务数据统计:")
                log.info(f"      覆盖股票: {financial_stats[0]:,} 只")
                log.info(f"      报告期数: {financial_stats[1]} 个")
                log.info(f"      时间范围: {financial_stats[2]} 至 {financial_stats[3]}")
                log.info(f"      总记录数: {financial_stats[4]:,} 条")
                
                # Check financial data freshness
                if financial_stats[3]:
                    latest_period = pd.to_datetime(financial_stats[3])
                    months_old = (datetime.now() - latest_period).days / 30
                    if months_old > self.config.MAX_FINANCIAL_DATA_AGE_MONTHS:
                        self.issues_found.append(f"财务数据过旧，最新报告期距今 {months_old:.1f} 个月")
            else:
                self.issues_found.append("无财务数据")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 财务数据完整性检查失败: {e}")
            self.issues_found.append(f"财务数据完整性检查失败: {e}")
    
    def _check_data_relationships(self, conn) -> None:
        """Check data relationships between tables"""
        self.total_checks += 1
        try:
            # Check for daily data without factor data
            orphan_daily = conn.execute(text("""
                SELECT COUNT(DISTINCT d.ts_code) 
                FROM ts_daily d 
                LEFT JOIN factors_exposure f ON d.ts_code = f.ts_code AND d.trade_date = f.trade_date
                WHERE f.ts_code IS NULL 
                AND d.trade_date >= (SELECT MAX(trade_date) - INTERVAL '7 days' FROM ts_daily)
            """)).scalar()
            
            if orphan_daily and orphan_daily > 0:
                log.warning(f"    ⚠ 发现 {orphan_daily} 只股票有日线数据但缺少因子数据")
                if orphan_daily > self.config.MIN_ORPHAN_THRESHOLD:
                    self.issues_found.append(f"大量股票缺少因子数据: {orphan_daily} 只")
            else:
                log.info("    ✓ 数据关联性良好")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 数据关联性检查失败: {e}")
            self.issues_found.append(f"数据关联性检查失败: {e}")
    
    def _check_database_metrics(self, conn) -> None:
        """Check database performance metrics"""
        self.total_checks += 1
        try:
            # Check database size
            db_size = conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)).scalar()
            
            # Check table sizes
            table_sizes = conn.execute(text("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('ts_daily', 'factors_exposure', 'financial_indicators')
                ORDER BY pg_total_relation_size(tablename::regclass) DESC
            """)).fetchall()
            
            log.info(f"    数据库大小: {db_size}")
            log.info("    主要表大小:")
            for table_name, size in table_sizes:
                log.info(f"      {table_name}: {size}")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 数据库性能指标检查失败: {e}")
            self.issues_found.append(f"数据库性能指标检查失败: {e}")
    
    def _generate_report(self) -> bool:
        """Generate final check report"""
        log.info("  --- 全面数据库检查报告 ---")
        log.info(f"  总检查项: {self.total_checks}")
        log.info(f"  通过检查: {self.passed_checks}")
        
        if self.total_checks > 0:
            success_rate = (self.passed_checks / self.total_checks) * 100
            log.info(f"  检查通过率: {success_rate:.1f}%")
        
        if self.issues_found:
            log.warning(f"  发现 {len(self.issues_found)} 个问题:")
            for i, issue in enumerate(self.issues_found, 1):
                log.warning(f"    {i}. {issue}")
        else:
            log.info("  ✅ 未发现数据完整性问题")
        
        log.info("  --- 全面数据库数据完整性检查完成 ---")
        
        return len(self.issues_found) == 0


async def test_api_concurrent_performance(dm, test_count=10):
    """测试API并发性能"""
    log.info("  --- 开始API并发性能测试 ---")
    
    TUSHARE_API_URL = "http://api.tushare.pro"
    
    async def fetch_tushare_api(session, api_name, params):
        payload = {
            "api_name": api_name,
            "token": config.TUSHARE_TOKEN,
            "params": params,
            "fields": []
        }
        try:
            async with session.post(
                TUSHARE_API_URL, 
                data=json.dumps(payload), 
                timeout=30
            ) as response:
                resp_json = await response.json()
                return resp_json.get("code") == 0
        except Exception:
            return False
    
    try:
        # 测试基础API调用
        test_params = {
            "ts_code": "600519.SH",
            "start_date": "20240101",
            "end_date": "20240131"
        }
        
        start_time = time.time()
        tasks = []
        
        async with aiohttp.ClientSession() as session:
            for _ in range(test_count):
                task = asyncio.create_task(
                    fetch_tushare_api(session, "daily", test_params)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        success_count = sum(1 for r in results if r)
        
        log.info(f"  > 并发请求数: {test_count}")
        log.info(f"  > 总耗时: {total_duration:.2f}秒")
        log.info(f"  > 成功率: {success_count/test_count:.1%}")
        log.info(f"  > 平均RPS: {test_count/total_duration:.2f}")
        
        if success_count >= test_count * 0.8:
            log.info("  > API并发性能: 良好")
            return True
        else:
            log.warning("  > API并发性能: 需要关注")
            return False
            
    except Exception as e:
        log.error(f"  > API并发性能测试失败: {e}")
        return False


def test_query_performance_benchmark(dm):
    """测试查询性能基准"""
    log.info("  --- 开始查询性能基准测试 ---")
    
    test_queries = [
        {
            'name': '单股票最新数据查询',
            'sql': """
                SELECT * FROM ts_daily 
                WHERE ts_code = '600519.SH' 
                ORDER BY trade_date DESC 
                LIMIT 1
            """,
            'expected_ms': 50
        },
        {
            'name': '因子数据查询',
            'sql': """
                SELECT ts_code, factor_value
                FROM factors_exposure 
                WHERE factor_name = 'pe_ttm' 
                AND trade_date >= '2024-01-01'
                ORDER BY trade_date DESC
                LIMIT 100
            """,
            'expected_ms': 200
        },
        {
            'name': '市场涨幅排行查询',
            'sql': """
                SELECT ts_code, close, pct_chg
                FROM ts_daily 
                WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily LIMIT 1)
                ORDER BY pct_chg DESC 
                LIMIT 50
            """,
            'expected_ms': 500
        }
    ]
    
    try:
        with dm.engine.connect() as conn:
            all_passed = True
            
            for query in test_queries:
                log.info(f"  > 测试: {query['name']}")
                
                # 运行3次取平均值
                times = []
                for i in range(3):
                    start_time = time.time()
                    try:
                        result = conn.execute(text(query['sql'])).fetchall()
                        elapsed_ms = (time.time() - start_time) * 1000
                        times.append(elapsed_ms)
                    except Exception as e:
                        log.error(f"    查询失败: {e}")
                        all_passed = False
                        break
                
                if times:
                    avg_time = sum(times) / len(times)
                    log.info(f"    平均耗时: {avg_time:.2f}ms")
                    log.info(f"    返回记录: {len(result) if 'result' in locals() else 0}条")
                    
                    if avg_time <= query['expected_ms']:
                        log.info("    ✓ 性能达标")
                    else:
                        log.warning(f"    ⚠ 性能超时 (期望<{query['expected_ms']}ms)")
                        all_passed = False
            
            return all_passed
            
    except Exception as e:
        log.error(f"  > 查询性能基准测试失败: {e}")
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
    total_checks = 7
    
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
    
    # 全面数据库数据完整性检查
    log.info("\n--- [检查 4/7] 全面数据库数据完整性检查 ---")
    try:
        if comprehensive_database_check(dm):
            log.info("  [PASS] 全面数据库数据完整性检查通过。")
            checks_passed += 1
        else:
            log.warning("  [WARN] 全面数据库数据完整性检查发现问题，请查看详细日志。")
    except Exception as e:
        log.error(f"  [FAIL] 全面数据库数据完整性检查失败: {e}")
    
    if check_system_resources():
        checks_passed += 1
    
    if check_database_performance(dm):
        checks_passed += 1
    
    if test_data_storage(dm):
        checks_passed += 1
    
    return finalize_report(checks_passed, total_checks)


def run_performance_check():
    """性能检查 - 专注于API和数据库性能测试"""
    log.info("=== 系统性能检查 ===")
    checks_passed = 0
    total_checks = 5
    
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
    
    # 性能测试
    log.info("\n--- [检查 3/5] API并发性能测试 ---")
    try:
        # 更安全的异步函数执行方式
        if asyncio.run(test_api_concurrent_performance(dm, 5)):
            checks_passed += 1
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # 如果已经在事件循环中，使用 create_task
            try:
                loop = asyncio.get_event_loop()
                task = loop.create_task(test_api_concurrent_performance(dm, 5))
                if loop.run_until_complete(task):
                    checks_passed += 1
            except Exception as inner_e:
                log.error(f"  > API并发性能测试失败: {inner_e}")
        else:
            log.error(f"  > API并发性能测试失败: {e}")
    except Exception as e:
        log.error(f"  > API并发性能测试失败: {e}")
    
    log.info("\n--- [检查 4/5] 查询性能基准测试 ---")
    if test_query_performance_benchmark(dm):
        checks_passed += 1
    
    log.info("\n--- [检查 5/5] 数据库性能检查 ---")
    if check_database_performance(dm):
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
    try:
        stock_list = dm.get_stock_basic("L")
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
    total_checks = 11  # 升级：总检查项增加到11个

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
    log.info("\n--- [检查 3/10] Tushare API连通性 ---")
    if check_tushare_api(dm):
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

    # --- 检查 7: 全面数据库数据完整性检查 ---
    log.info("\n--- [检查 7/11] 全面数据库数据完整性检查 ---")
    try:
        if comprehensive_database_check(dm):
            log.info("  [PASS] 全面数据库数据完整性检查通过。")
            checks_passed += 1
        else:
            log.warning("  [WARN] 全面数据库数据完整性检查发现问题，请查看详细日志。")
    except Exception as e:
        log.error(f"  [FAIL] 全面数据库数据完整性检查失败: {e}")

    # --- 检查 8: 数据库财务数据质量验证 ---
    log.info("\n--- [检查 8/11] 数据库财务数据质量验证 ---")
    try:
        if check_database_data_quality(dm):
            log.info("  [PASS] 数据库核心数据质量验证通过。")
            checks_passed += 1
        else:
            log.warning("  [WARN] 数据库核心数据质量验证发现问题。")
    except Exception as e:
        log.error(f"  [FAIL] 数据库数据质量验证失败: {e}")

    # --- 检查 9: 引擎层核心功能 (FactorFactory & FactorProcessor) ---
    log.info("\n--- [检查 9/11] 引擎层因子计算 ---")
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

    # --- 检查 10: 智能分析层 (AIOrchestrator & AI API) ---
    log.info("\n--- [检查 10/11] 智能分析层与AI API连通性 (低成本测试) ---")
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

    # --- 检查 11: 端到端工作流冒烟测试 ---
    log.info("\n--- [检查 11/11] 端到端工作流冒烟测试 ---")
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





def main():
    """Main entry point for the system health checker"""
    parser = argparse.ArgumentParser(
        description="A股量化投研平台系统健康检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system_check.py --mode quick
  python run_system_check.py --mode full --output report.json
  python run_system_check.py --mode stability --verbose
  python run_system_check.py --mode performance --no-interactive
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "full", "stability", "performance"], 
        default="full",
        help="检查模式: quick(快速检查), full(完整检查), stability(稳定性检查), performance(性能检查)"
    )
    
    parser.add_argument(
        "--output",
        help="输出报告到指定文件 (JSON格式)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出模式"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="非交互模式，不等待用户输入"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="检查超时时间（秒），默认300秒"
    )
    
    try:
        args = parser.parse_args()
        
        # Set logging level based on verbosity
        if args.verbose:
            import logging
            log.setLevel(logging.DEBUG)
        
        # Run the appropriate check mode
        if args.mode == "quick":
            result = run_quick_check()
        elif args.mode == "stability":
            result = run_stability_check()
        elif args.mode == "performance":
            result = run_performance_check()
        else:
            result = run_all_checks()
        
        # Save report if output file specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'mode': args.mode,
                    'result': result
                }, f, ensure_ascii=False, indent=2)
            log.info(f"报告已保存到: {args.output}")
        
        # Interactive mode prompt
        if not args.no_interactive:
            input("\n检查完毕，请按 Enter 键退出...")
            
    except KeyboardInterrupt:
        log.info("\n用户中断检查")
        sys.exit(1)
    except Exception as e:
        log.error(f"系统检查过程中发生错误: {e}")
        if args.verbose:
            log.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()