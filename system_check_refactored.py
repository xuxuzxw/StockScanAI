"""
Refactored system check with improved design patterns and maintainability
"""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import aiohttp
import pandas as pd
import psutil
from sqlalchemy import create_engine, text

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
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


class CheckMode(Enum):
    """System check modes"""
    QUICK = "quick"
    FULL = "full"
    STABILITY = "stability"
    PERFORMANCE = "performance"


@dataclass
class CheckResult:
    """Result of a system check"""
    name: str
    success: bool
    duration_seconds: float
    message: str = ""
    details: Dict[str, Any] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'success': self.success,
            'duration_seconds': self.duration_seconds,
            'message': self.message,
            'details': self.details,
            'error': str(self.error) if self.error else None
        }


class SystemCheckConfig:
    """Centralized configuration for system checks"""
    
    # Performance thresholds
    SIMPLE_QUERY_MAX_MS = 100
    COMPLEX_QUERY_MAX_MS = 1000
    FACTOR_CALCULATION_MAX_MS = 5000
    
    # Resource thresholds
    MAX_CPU_PERCENT = 80
    MAX_MEMORY_PERCENT = 85
    MAX_DISK_PERCENT = 90
    
    # Data quality thresholds
    MIN_COVERAGE_RATIO = 0.9
    MAX_DATA_AGE_DAYS = 7
    MAX_INVALID_DATA_PERCENT = 1.0
    MAX_PRICE_INCONSISTENCY_PERCENT = 0.1
    MAX_FINANCIAL_DATA_AGE_MONTHS = 6
    MIN_ORPHAN_THRESHOLD = 100
    
    # Core tables
    CORE_TABLES = {
        'stock_basic': '股票基本信息',
        'ts_daily': '日线行情数据', 
        'factors_exposure': '因子暴露数据',
        'financial_indicators': '财务指标数据',
        'ts_adj_factor': '复权因子数据'
    }
    
    TIME_RANGE_TABLES = ['ts_daily', 'factors_exposure', 'financial_indicators']
    
    # Test configuration
    TEST_STOCKS = ["600519.SH", "000001.SZ", "000002.SZ"]
    STABLE_FACTORS = ["pe_ttm", "pb"]
    MIN_FACTOR_SUCCESS_RATE = 0.6
    
    @classmethod
    def get_config(cls, profile: str = "default") -> 'SystemCheckConfig':
        """Get configuration by profile"""
        try:
            from system_check_config_improved import get_config
            return get_config(profile)
        except ImportError:
            log.warning("Advanced configuration not available, using basic configuration")
            return cls()


class BaseChecker(ABC):
    """Base class for all system checkers"""
    
    def __init__(self, dm: data.DataManager, config: SystemCheckConfig):
        self.dm = dm
        self.config = config
        self.results: List[CheckResult] = []
    
    @abstractmethod
    def run_checks(self) -> List[CheckResult]:
        """Run all checks and return results"""
        pass
    
    def _execute_check(self, check_name: str, check_func: Callable) -> CheckResult:
        """Execute a single check with timing and error handling"""
        start_time = time.time()
        
        try:
            success = check_func()
            duration = time.time() - start_time
            
            return CheckResult(
                name=check_name,
                success=bool(success),
                duration_seconds=duration,
                message="检查完成" if success else "检查失败"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            log.error(f"检查 {check_name} 失败: {e}")
            
            return CheckResult(
                name=check_name,
                success=False,
                duration_seconds=duration,
                message=str(e),
                error=e
            )


class DatabaseChecker(BaseChecker):
    """Enhanced database checker with improved design"""
    
    def __init__(self, dm: data.DataManager, config: SystemCheckConfig):
        super().__init__(dm, config)
        self.issues_found = []
        self.total_checks = 0
        self.passed_checks = 0
    
    def run_checks(self) -> List[CheckResult]:
        """Run comprehensive database checks"""
        log.info("  --- 开始数据库检查 ---")
        
        check_methods = [
            ("core_tables", "核心表检查", self._check_core_tables),
            ("time_ranges", "时间范围检查", self._check_time_ranges),
            ("stock_consistency", "股票一致性检查", self._check_stock_consistency),
            ("data_quality", "数据质量检查", self._check_data_quality),
            ("factor_completeness", "因子完整性检查", self._check_factor_completeness),
            ("financial_completeness", "财务完整性检查", self._check_financial_completeness),
            ("data_relationships", "数据关联性检查", self._check_data_relationships),
            ("database_metrics", "数据库指标检查", self._check_database_metrics)
        ]
        
        try:
            with self.dm.engine.connect() as conn:
                for check_id, description, check_method in check_methods:
                    log.info(f"  执行{description}")
                    result = self._execute_check(check_id, lambda: check_method(conn))
                    self.results.append(result)
                    
                    if result.success:
                        self.passed_checks += 1
                    else:
                        self.issues_found.append(f"{description}: {result.message}")
                    
                    self.total_checks += 1
            
            self._generate_report()
            return self.results
            
        except Exception as e:
            log.error(f"数据库检查失败: {e}")
            error_result = CheckResult(
                name="database_check",
                success=False,
                duration_seconds=0,
                message=str(e),
                error=e
            )
            return [error_result]
    
    def _check_core_tables(self, conn) -> bool:
        """Check core table existence and basic statistics"""
        try:
            for table, desc in self.config.CORE_TABLES.items():
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                log.info(f"    ✓ {desc} ({table}): {count:,} 条记录")
            return True
        except Exception as e:
            log.error(f"    ✗ 核心表检查失败: {e}")
            return False
    
    def _check_time_ranges(self, conn) -> bool:
        """Check data time ranges and freshness"""
        try:
            for table in self.config.TIME_RANGE_TABLES:
                date_col = 'end_date' if table == 'financial_indicators' else 'trade_date'
                
                result = conn.execute(text(f"""
                    SELECT MIN({date_col}) as min_date, 
                           MAX({date_col}) as max_date,
                           COUNT(DISTINCT {date_col}) as date_count
                    FROM {table}
                """)).fetchone()
                
                if result and result[0]:
                    log.info(f"    ✓ {table}: {result[0]} 至 {result[1]} ({result[2]} 个日期)")
                    
                    # Check data freshness
                    if table in ['ts_daily', 'factors_exposure']:
                        days_old = (datetime.now().date() - result[1]).days
                        if days_old > self.config.MAX_DATA_AGE_DAYS:
                            self.issues_found.append(f"{table} 数据过旧，最新数据距今 {days_old} 天")
                else:
                    log.warning(f"    ⚠ {table}: 无有效日期数据")
                    self.issues_found.append(f"{table} 无有效日期数据")
            
            return True
        except Exception as e:
            log.error(f"    ✗ 时间范围检查失败: {e}")
            return False
    
    def _check_stock_consistency(self, conn) -> bool:
        """Check stock code consistency across tables"""
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
            
            return True
        except Exception as e:
            log.error(f"    ✗ 股票一致性检查失败: {e}")
            return False
    
    def _check_data_quality(self, conn) -> bool:
        """Check data quality issues in trading data"""
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
                price_inconsist_pct = (quality_issues[3] / total_records) * 100
                
                log.info(f"    最近30天数据质量 (共{total_records:,}条):")
                log.info(f"      无效收盘价: {quality_issues[1]} ({invalid_close_pct:.2f}%)")
                log.info(f"      价格不一致: {quality_issues[3]} ({price_inconsist_pct:.2f}%)")
                
                if invalid_close_pct > self.config.MAX_INVALID_DATA_PERCENT:
                    self.issues_found.append(f"无效收盘价比例过高: {invalid_close_pct:.2f}%")
                if price_inconsist_pct > self.config.MAX_PRICE_INCONSISTENCY_PERCENT:
                    self.issues_found.append(f"价格不一致比例过高: {price_inconsist_pct:.2f}%")
            
            return True
        except Exception as e:
            log.error(f"    ✗ 数据质量检查失败: {e}")
            return False
    
    def _check_factor_completeness(self, conn) -> bool:
        """Check factor data completeness"""
        try:
            factor_stats = conn.execute(text("""
                SELECT 
                    factor_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    COUNT(DISTINCT trade_date) as date_count
                FROM factors_exposure 
                GROUP BY factor_name 
                ORDER BY record_count DESC
            """)).fetchall()
            
            if factor_stats:
                log.info(f"    因子数据统计 (共{len(factor_stats)}个因子):")
                for factor in factor_stats[:5]:  # Show top 5
                    log.info(f"      {factor[0]}: {factor[1]:,}条记录, {factor[2]}只股票")
            else:
                self.issues_found.append("无因子数据")
            
            return True
        except Exception as e:
            log.error(f"    ✗ 因子完整性检查失败: {e}")
            return False
    
    def _check_financial_completeness(self, conn) -> bool:
        """Check financial data completeness"""
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
                log.info(f"      总记录数: {financial_stats[4]:,} 条")
            else:
                self.issues_found.append("无财务数据")
            
            return True
        except Exception as e:
            log.error(f"    ✗ 财务完整性检查失败: {e}")
            return False
    
    def _check_data_relationships(self, conn) -> bool:
        """Check data relationships between tables"""
        try:
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
            
            return True
        except Exception as e:
            log.error(f"    ✗ 数据关联性检查失败: {e}")
            return False
    
    def _check_database_metrics(self, conn) -> bool:
        """Check database performance metrics"""
        try:
            db_size = conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)).scalar()
            
            log.info(f"    数据库大小: {db_size}")
            return True
        except Exception as e:
            log.error(f"    ✗ 数据库指标检查失败: {e}")
            return False
    
    def _generate_report(self) -> None:
        """Generate comprehensive report"""
        log.info("  --- 数据库检查报告 ---")
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


class SystemResourceChecker(BaseChecker):
    """System resource checker"""
    
    def run_checks(self) -> List[CheckResult]:
        """Run system resource checks"""
        log.info("  --- 开始系统资源检查 ---")
        
        result = self._execute_check("system_resources", self._check_resources)
        return [result]
    
    def _check_resources(self) -> bool:
        """Check system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            
            log.info(f"    CPU使用率: {cpu_percent}%")
            log.info(f"    内存使用率: {memory.percent}%")
            log.info(f"    磁盘使用率: {disk_percent:.1f}%")
            
            # Check thresholds
            is_healthy = (
                cpu_percent < self.config.MAX_CPU_PERCENT and
                memory.percent < self.config.MAX_MEMORY_PERCENT and
                disk_percent < self.config.MAX_DISK_PERCENT
            )
            
            if is_healthy:
                log.info("    ✓ 系统资源状态良好")
            else:
                log.warning("    ⚠ 系统资源紧张")
            
            return is_healthy
            
        except Exception as e:
            log.error(f"    ✗ 系统资源检查失败: {e}")
            return False


class SystemCheckOrchestrator:
    """Main orchestrator for system checks"""
    
    def __init__(self, dm: data.DataManager, mode: CheckMode = CheckMode.FULL):
        self.dm = dm
        self.mode = mode
        self.config = SystemCheckConfig.get_config()
        self.checkers: List[BaseChecker] = []
        self.all_results: List[CheckResult] = []
    
    def register_checkers(self) -> None:
        """Register all checkers based on mode"""
        # Always include database checker
        self.checkers.append(DatabaseChecker(self.dm, self.config))
        
        # Add resource checker for full and performance modes
        if self.mode in [CheckMode.FULL, CheckMode.PERFORMANCE]:
            self.checkers.append(SystemResourceChecker(self.dm, self.config))
    
    def run_all_checks(self) -> bool:
        """Run all registered checks"""
        log.info(f"=== 系统检查 ({self.mode.value.upper()}) ===")
        
        self.register_checkers()
        
        total_success = True
        
        for checker in self.checkers:
            results = checker.run_checks()
            self.all_results.extend(results)
            
            # Check if any critical checks failed
            for result in results:
                if not result.success:
                    total_success = False
        
        self._generate_final_report()
        return total_success
    
    def _generate_final_report(self) -> None:
        """Generate final comprehensive report"""
        total_checks = len(self.all_results)
        passed_checks = sum(1 for r in self.all_results if r.success)
        
        log.info("\n" + "=" * 50)
        log.info("系统检查完成")
        log.info("=" * 50)
        log.info(f"检查模式: {self.mode.value.upper()}")
        log.info(f"总检查项: {total_checks}")
        log.info(f"通过检查: {passed_checks}")
        
        if total_checks > 0:
            success_rate = (passed_checks / total_checks) * 100
            log.info(f"成功率: {success_rate:.1f}%")
        
        # Show timing summary
        total_duration = sum(r.duration_seconds for r in self.all_results)
        log.info(f"总耗时: {total_duration:.2f}秒")
        
        # Show failed checks
        failed_checks = [r for r in self.all_results if not r.success]
        if failed_checks:
            log.info("\n失败的检查:")
            for result in failed_checks:
                log.info(f"  - {result.name}: {result.message}")
        else:
            log.info("🎉 所有检查均通过！系统状态良好。")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="A股量化投研平台系统检查")
    parser.add_argument(
        "--mode", 
        choices=["quick", "full", "stability", "performance"],
        default="full",
        help="检查模式"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize data manager
        dm = data.DataManager()
        
        # Create and run orchestrator
        mode = CheckMode(args.mode)
        orchestrator = SystemCheckOrchestrator(dm, mode)
        
        success = orchestrator.run_all_checks()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        log.error(f"系统检查失败: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()