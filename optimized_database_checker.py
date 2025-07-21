"""
Highly optimized database checker with minimal queries and maximum efficiency
"""
from typing import Dict, List, Tuple, Any, Optional
from sqlalchemy import text
import time
import pandas as pd
from datetime import datetime
from logger_config import log
from system_check_config_improved import get_config


class OptimizedDatabaseChecker:
    """Ultra-efficient database checker using single comprehensive queries"""
    
    def __init__(self, dm, config_profile: str = "default"):
        self.dm = dm
        self.config = get_config(config_profile)
        self.issues_found = []
        self.metrics = {}
        
    def run_comprehensive_check(self) -> bool:
        """Run all checks in minimal database queries"""
        log.info("  --- 开始优化数据库检查 ---")
        start_time = time.time()
        
        try:
            with self.dm.engine.connect() as conn:
                # Single mega-query to get all basic statistics
                basic_stats = self._get_all_basic_stats(conn)
                self._analyze_basic_stats(basic_stats)
                
                # Single quality check query
                quality_stats = self._get_quality_stats(conn)
                self._analyze_quality_stats(quality_stats)
                
                # Single relationship check
                relationship_stats = self._get_relationship_stats(conn)
                self._analyze_relationship_stats(relationship_stats)
                
                # Database metadata in one query
                db_metadata = self._get_database_metadata(conn)
                self._analyze_database_metadata(db_metadata)
                
            total_time = time.time() - start_time
            self.metrics['total_check_time'] = total_time
            
            return self._generate_optimized_report()
            
        except Exception as e:
            log.error(f"  > 优化数据库检查失败: {e}")
            return False
    
    def _get_all_basic_stats(self, conn) -> Dict[str, Any]:
        """Get all basic table statistics in a single query"""
        log.info("  [1/4] 获取基础统计信息...")
        
        # Mega query to get all table stats at once
        query = text("""
            WITH table_stats AS (
                -- Stock basic stats
                SELECT 
                    'stock_basic' as table_name,
                    COUNT(*) as record_count,
                    COUNT(*) as stock_count,
                    NULL::date as min_date,
                    NULL::date as max_date,
                    0 as date_count,
                    0 as factor_count
                FROM stock_basic
                
                UNION ALL
                
                -- Daily data stats
                SELECT 
                    'ts_daily' as table_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT trade_date) as date_count,
                    0 as factor_count
                FROM ts_daily
                
                UNION ALL
                
                -- Factor exposure stats
                SELECT 
                    'factors_exposure' as table_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT trade_date) as date_count,
                    COUNT(DISTINCT factor_name) as factor_count
                FROM factors_exposure
                
                UNION ALL
                
                -- Financial indicators stats
                SELECT 
                    'financial_indicators' as table_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    MIN(end_date) as min_date,
                    MAX(end_date) as max_date,
                    COUNT(DISTINCT end_date) as date_count,
                    0 as factor_count
                FROM financial_indicators
            )
            SELECT * FROM table_stats ORDER BY table_name
        """)
        
        results = conn.execute(query).fetchall()
        
        # Convert to dictionary for easy access
        stats = {}
        for row in results:
            stats[row[0]] = {
                'record_count': row[1],
                'stock_count': row[2],
                'min_date': row[3],
                'max_date': row[4],
                'date_count': row[5],
                'factor_count': row[6]
            }
        
        return stats
    
    def _get_quality_stats(self, conn) -> Dict[str, Any]:
        """Get data quality statistics in a single query"""
        log.info("  [2/4] 获取数据质量统计...")
        
        query = text("""
            WITH quality_checks AS (
                -- Daily data quality (last 30 days)
                SELECT 
                    'daily_quality' as check_type,
                    COUNT(*) as total_records,
                    SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) as invalid_close,
                    SUM(CASE WHEN vol IS NULL OR vol < 0 THEN 1 ELSE 0 END) as invalid_volume,
                    SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as price_inconsistency,
                    0 as orphan_count
                FROM ts_daily
                WHERE trade_date >= (SELECT MAX(trade_date) - INTERVAL '30 days' FROM ts_daily)
                
                UNION ALL
                
                -- Orphaned data check
                SELECT 
                    'orphan_data' as check_type,
                    0 as total_records,
                    0 as invalid_close,
                    0 as invalid_volume,
                    0 as price_inconsistency,
                    COUNT(DISTINCT d.ts_code) as orphan_count
                FROM ts_daily d 
                LEFT JOIN factors_exposure f ON d.ts_code = f.ts_code AND d.trade_date = f.trade_date
                WHERE f.ts_code IS NULL 
                AND d.trade_date >= (SELECT MAX(trade_date) - INTERVAL '7 days' FROM ts_daily)
            )
            SELECT * FROM quality_checks
        """)
        
        results = conn.execute(query).fetchall()
        
        quality_stats = {}
        for row in results:
            quality_stats[row[0]] = {
                'total_records': row[1],
                'invalid_close': row[2],
                'invalid_volume': row[3],
                'price_inconsistency': row[4],
                'orphan_count': row[5]
            }
        
        return quality_stats
    
    def _get_relationship_stats(self, conn) -> Dict[str, Any]:
        """Get data relationship statistics"""
        log.info("  [3/4] 获取数据关联统计...")
        
        query = text("""
            SELECT 
                'coverage_analysis' as analysis_type,
                (SELECT COUNT(*) FROM stock_basic) as total_stocks,
                (SELECT COUNT(DISTINCT ts_code) FROM ts_daily) as daily_stocks,
                (SELECT COUNT(DISTINCT ts_code) FROM factors_exposure) as factor_stocks,
                (SELECT COUNT(DISTINCT ts_code) FROM financial_indicators) as financial_stocks
        """)
        
        result = conn.execute(query).fetchone()
        
        return {
            'total_stocks': result[1],
            'daily_coverage': result[2] / result[1] if result[1] > 0 else 0,
            'factor_coverage': result[3] / result[1] if result[1] > 0 else 0,
            'financial_coverage': result[4] / result[1] if result[1] > 0 else 0
        }
    
    def _get_database_metadata(self, conn) -> Dict[str, Any]:
        """Get database metadata and performance metrics"""
        log.info("  [4/4] 获取数据库元数据...")
        
        query = text("""
            SELECT 
                pg_size_pretty(pg_database_size(current_database())) as db_size,
                (
                    SELECT json_object_agg(
                        tablename,
                        pg_size_pretty(pg_total_relation_size(tablename::regclass))
                    )
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename IN ('stock_basic', 'ts_daily', 'factors_exposure', 'financial_indicators')
                ) as table_sizes,
                (
                    SELECT COUNT(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                ) as active_connections
        """)
        
        result = conn.execute(query).fetchone()
        
        return {
            'database_size': result[0],
            'table_sizes': result[1] or {},
            'active_connections': result[2]
        }
    
    def _analyze_basic_stats(self, stats: Dict[str, Any]):
        """Analyze basic statistics and identify issues"""
        log.info("  分析基础统计信息:")
        
        for table_name, table_stats in stats.items():
            record_count = table_stats['record_count']
            stock_count = table_stats['stock_count']
            
            if record_count == 0:
                self.issues_found.append(f"表 {table_name} 为空")
                log.warning(f"    ⚠ {table_name}: 表为空")
            else:
                log.info(f"    ✓ {table_name}: {record_count:,} 条记录, {stock_count:,} 只股票")
                
                # Check data freshness for time-series tables
                if table_stats['max_date'] and table_name in ['ts_daily', 'factors_exposure']:
                    days_old = (datetime.now().date() - table_stats['max_date']).days
                    if days_old > self.config.data_quality.MAX_DATA_AGE_DAYS:
                        self.issues_found.append(f"{table_name} 数据过旧，最新数据距今 {days_old} 天")
                        log.warning(f"    ⚠ {table_name}: 数据过旧 ({days_old} 天)")
                
                # Store metrics
                self.metrics[f'{table_name}_records'] = record_count
                self.metrics[f'{table_name}_stocks'] = stock_count
    
    def _analyze_quality_stats(self, quality_stats: Dict[str, Any]):
        """Analyze data quality statistics"""
        log.info("  分析数据质量:")
        
        # Analyze daily data quality
        if 'daily_quality' in quality_stats:
            daily_quality = quality_stats['daily_quality']
            total_records = daily_quality['total_records']
            
            if total_records > 0:
                invalid_close_pct = (daily_quality['invalid_close'] / total_records) * 100
                invalid_vol_pct = (daily_quality['invalid_volume'] / total_records) * 100
                price_inconsist_pct = (daily_quality['price_inconsistency'] / total_records) * 100
                
                log.info(f"    最近30天数据质量 (共{total_records:,}条):")
                log.info(f"      无效收盘价: {daily_quality['invalid_close']} ({invalid_close_pct:.2f}%)")
                log.info(f"      无效成交量: {daily_quality['invalid_volume']} ({invalid_vol_pct:.2f}%)")
                log.info(f"      价格不一致: {daily_quality['price_inconsistency']} ({price_inconsist_pct:.2f}%)")
                
                # Check against thresholds
                if invalid_close_pct > self.config.data_quality.MAX_INVALID_DATA_PERCENT:
                    self.issues_found.append(f"无效收盘价比例过高: {invalid_close_pct:.2f}%")
                if price_inconsist_pct > self.config.data_quality.MAX_PRICE_INCONSISTENCY_PERCENT:
                    self.issues_found.append(f"价格不一致比例过高: {price_inconsist_pct:.2f}%")
                
                # Store metrics
                self.metrics['invalid_close_percent'] = invalid_close_pct
                self.metrics['price_inconsistency_percent'] = price_inconsist_pct
        
        # Analyze orphaned data
        if 'orphan_data' in quality_stats:
            orphan_count = quality_stats['orphan_data']['orphan_count']
            if orphan_count > self.config.data_quality.MIN_ORPHAN_THRESHOLD:
                self.issues_found.append(f"大量股票缺少因子数据: {orphan_count} 只")
                log.warning(f"    ⚠ 发现 {orphan_count} 只股票缺少因子数据")
            else:
                log.info(f"    ✓ 数据关联性良好 (孤立数据: {orphan_count} 只)")
            
            self.metrics['orphan_stocks'] = orphan_count
    
    def _analyze_relationship_stats(self, relationship_stats: Dict[str, Any]):
        """Analyze data relationship statistics"""
        log.info("  分析数据覆盖率:")
        
        total_stocks = relationship_stats['total_stocks']
        daily_coverage = relationship_stats['daily_coverage']
        factor_coverage = relationship_stats['factor_coverage']
        financial_coverage = relationship_stats['financial_coverage']
        
        log.info(f"    基准股票数: {total_stocks:,}")
        log.info(f"    日线数据覆盖率: {daily_coverage:.1%}")
        log.info(f"    因子数据覆盖率: {factor_coverage:.1%}")
        log.info(f"    财务数据覆盖率: {financial_coverage:.1%}")
        
        # Check coverage thresholds
        min_coverage = self.config.data_quality.MIN_COVERAGE_RATIO
        
        if daily_coverage < min_coverage:
            self.issues_found.append(f"日线数据覆盖率过低: {daily_coverage:.1%}")
        if factor_coverage < min_coverage:
            self.issues_found.append(f"因子数据覆盖率过低: {factor_coverage:.1%}")
        if financial_coverage < min_coverage * 0.8:  # Financial data can be lower
            self.issues_found.append(f"财务数据覆盖率过低: {financial_coverage:.1%}")
        
        # Store metrics
        self.metrics['daily_coverage'] = daily_coverage
        self.metrics['factor_coverage'] = factor_coverage
        self.metrics['financial_coverage'] = financial_coverage
    
    def _analyze_database_metadata(self, metadata: Dict[str, Any]):
        """Analyze database metadata"""
        log.info("  分析数据库元数据:")
        
        log.info(f"    数据库大小: {metadata['database_size']}")
        log.info(f"    活跃连接数: {metadata['active_connections']}")
        
        if metadata['table_sizes']:
            log.info("    主要表大小:")
            for table, size in metadata['table_sizes'].items():
                log.info(f"      {table}: {size}")
        
        # Store metrics
        self.metrics['database_size'] = metadata['database_size']
        self.metrics['active_connections'] = metadata['active_connections']
    
    def _generate_optimized_report(self) -> bool:
        """Generate optimized report with metrics"""
        log.info("  --- 优化数据库检查报告 ---")
        
        # Calculate overall health score
        health_score = self._calculate_health_score()
        
        log.info(f"  检查耗时: {self.metrics.get('total_check_time', 0):.2f}秒")
        log.info(f"  系统健康评分: {health_score:.1f}/100")
        
        if self.issues_found:
            log.warning(f"  发现 {len(self.issues_found)} 个问题:")
            for i, issue in enumerate(self.issues_found, 1):
                log.warning(f"    {i}. {issue}")
        else:
            log.info("  ✅ 未发现数据完整性问题")
        
        # Performance summary
        log.info("  性能指标:")
        log.info(f"    数据库大小: {self.metrics.get('database_size', 'N/A')}")
        log.info(f"    日线数据覆盖率: {self.metrics.get('daily_coverage', 0):.1%}")
        log.info(f"    因子数据覆盖率: {self.metrics.get('factor_coverage', 0):.1%}")
        
        log.info("  --- 优化数据库检查完成 ---")
        
        return len(self.issues_found) == 0 and health_score >= 80
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Deduct points for each issue
        score -= len(self.issues_found) * 10
        
        # Deduct points for low coverage
        daily_coverage = self.metrics.get('daily_coverage', 1.0)
        factor_coverage = self.metrics.get('factor_coverage', 1.0)
        
        if daily_coverage < 0.9:
            score -= (0.9 - daily_coverage) * 50
        if factor_coverage < 0.9:
            score -= (0.9 - factor_coverage) * 50
        
        # Deduct points for data quality issues
        invalid_close_pct = self.metrics.get('invalid_close_percent', 0)
        if invalid_close_pct > 1.0:
            score -= invalid_close_pct * 5
        
        return max(0, score)


# Updated function for the optimized checker
def run_optimized_database_check(dm, config_profile: str = "default") -> bool:
    """
    Run optimized database check with minimal queries and maximum efficiency
    """
    checker = OptimizedDatabaseChecker(dm, config_profile)
    return checker.run_comprehensive_check()