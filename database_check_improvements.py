"""
Improved database check implementation with better structure and maintainability
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
from sqlalchemy import text
from logger_config import log


@dataclass
class CheckResult:
    """Represents the result of a database check"""
    name: str
    success: bool
    message: str = ""
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class DatabaseCheckConfig:
    """Configuration for database checks"""
    CORE_TABLES = {
        'stock_basic': '股票基本信息',
        'ts_daily': '日线行情数据', 
        'factors_exposure': '因子暴露数据',
        'financial_indicators': '财务指标数据',
        'ts_adj_factor': '复权因子数据'
    }
    
    TIME_RANGE_TABLES = ['ts_daily', 'factors_exposure', 'financial_indicators']
    
    # Quality thresholds
    MAX_DATA_AGE_DAYS = 7
    MIN_COVERAGE_RATIO = 0.8
    MAX_INVALID_DATA_PERCENT = 1.0
    MAX_PRICE_INCONSISTENCY_PERCENT = 0.1
    MAX_FINANCIAL_DATA_AGE_MONTHS = 6


class DatabaseChecker:
    """Handles comprehensive database integrity checks"""
    
    def __init__(self, dm, config: DatabaseCheckConfig = None):
        self.dm = dm
        self.config = config or DatabaseCheckConfig()
        self.issues_found = []
        self.total_checks = 0
        self.passed_checks = 0
    
    def run_comprehensive_check(self) -> bool:
        """Run all database checks and return overall success"""
        log.info("  --- 开始全面数据库数据完整性检查 ---")
        
        try:
            with self.dm.engine.connect() as conn:
                # Run all check methods
                check_methods = [
                    self._check_core_tables,
                    self._check_time_ranges,
                    self._check_stock_consistency,
                    self._check_data_quality,
                    self._check_factor_completeness,
                    self._check_financial_completeness,
                    self._check_data_relationships,
                    self._check_database_metrics
                ]
                
                for i, check_method in enumerate(check_methods, 1):
                    log.info(f"  [{i}/{len(check_methods)}] {check_method.__name__.replace('_check_', '检查')}")
                    try:
                        check_method(conn)
                    except Exception as e:
                        log.error(f"    ✗ {check_method.__name__} 失败: {e}")
                        self.issues_found.append(f"{check_method.__name__} 检查失败: {e}")
            
            return self._generate_report()
            
        except Exception as e:
            log.error(f"  > 全面数据库检查失败: {e}")
            return False
    
    def _check_core_tables(self, conn) -> None:
        """Check core table existence and basic statistics"""
        for table, desc in self.config.CORE_TABLES.items():
            self.total_checks += 1
            try:
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
                        if coverage < self.config.MIN_COVERAGE_RATIO:
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
                if orphan_daily > 100:
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


# Updated function to use the new class-based approach
def comprehensive_database_check(dm) -> bool:
    """
    全面的数据库数据完整性检查 - 重构版本
    使用面向对象设计，提高可维护性和可测试性
    """
    checker = DatabaseChecker(dm)
    return checker.run_comprehensive_check()