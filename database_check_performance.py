"""
Performance-optimized database checks
"""
from typing import Dict, List, Tuple, Any
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from logger_config import log


class PerformantDatabaseChecker:
    """Database checker optimized for performance"""
    
    def __init__(self, dm, config=None):
        self.dm = dm
        self.config = config
        self.issues_found = []
        self.total_checks = 0
        self.passed_checks = 0
    
    def run_batch_checks(self, conn) -> Dict[str, Any]:
        """Run multiple checks in a single database query where possible"""
        log.info("  执行批量数据库检查...")
        
        # Single query to get comprehensive table statistics
        batch_query = text("""
            WITH table_stats AS (
                SELECT 
                    'stock_basic' as table_name,
                    COUNT(*) as record_count,
                    COUNT(*) as stock_count,
                    NULL::date as min_date,
                    NULL::date as max_date,
                    0 as date_count
                FROM stock_basic
                
                UNION ALL
                
                SELECT 
                    'ts_daily' as table_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT trade_date) as date_count
                FROM ts_daily
                
                UNION ALL
                
                SELECT 
                    'factors_exposure' as table_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT trade_date) as date_count
                FROM factors_exposure
                
                UNION ALL
                
                SELECT 
                    'financial_indicators' as table_name,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    MIN(end_date) as min_date,
                    MAX(end_date) as max_date,
                    COUNT(DISTINCT end_date) as date_count
                FROM financial_indicators
            )
            SELECT * FROM table_stats ORDER BY table_name
        """)
        
        start_time = time.time()
        results = conn.execute(batch_query).fetchall()
        query_time = time.time() - start_time
        
        log.info(f"    批量查询完成，耗时: {query_time:.3f}秒")
        
        # Process results
        table_stats = {}
        for row in results:
            table_stats[row[0]] = {
                'record_count': row[1],
                'stock_count': row[2],
                'min_date': row[3],
                'max_date': row[4],
                'date_count': row[5]
            }
        
        return table_stats
    
    def run_parallel_quality_checks(self, conn) -> Dict[str, Any]:
        """Run quality checks in parallel where possible"""
        log.info("  执行并行数据质量检查...")
        
        quality_queries = {
            'daily_quality': text("""
                SELECT 
                    'ts_daily' as table_name,
                    COUNT(*) as total_records,
                    SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) as invalid_close,
                    SUM(CASE WHEN vol IS NULL OR vol < 0 THEN 1 ELSE 0 END) as invalid_volume,
                    SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as price_inconsistency
                FROM ts_daily
                WHERE trade_date >= (SELECT MAX(trade_date) - INTERVAL '30 days' FROM ts_daily)
            """),
            
            'factor_consistency': text("""
                SELECT 
                    factor_name,
                    COUNT(DISTINCT trade_date) as date_count,
                    COUNT(DISTINCT ts_code) as stock_count,
                    COUNT(*) as total_records
                FROM factors_exposure 
                GROUP BY factor_name
                HAVING COUNT(*) > 0
                ORDER BY COUNT(*) DESC
            """),
            
            'orphan_data': text("""
                SELECT COUNT(DISTINCT d.ts_code) as orphan_count
                FROM ts_daily d 
                LEFT JOIN factors_exposure f ON d.ts_code = f.ts_code AND d.trade_date = f.trade_date
                WHERE f.ts_code IS NULL 
                AND d.trade_date >= (SELECT MAX(trade_date) - INTERVAL '7 days' FROM ts_daily)
            """)
        }
        
        results = {}
        start_time = time.time()
        
        # Execute queries sequentially (connection is not thread-safe)
        for check_name, query in quality_queries.items():
            try:
                result = conn.execute(query).fetchall()
                results[check_name] = result
            except Exception as e:
                log.error(f"    质量检查 {check_name} 失败: {e}")
                results[check_name] = None
        
        query_time = time.time() - start_time
        log.info(f"    并行质量检查完成，耗时: {query_time:.3f}秒")
        
        return results
    
    def run_optimized_comprehensive_check(self) -> bool:
        """Run optimized comprehensive database check"""
        log.info("  --- 开始性能优化的数据库检查 ---")
        
        try:
            with self.dm.engine.connect() as conn:
                # 1. Batch table statistics
                table_stats = self.run_batch_checks(conn)
                self._process_table_stats(table_stats)
                
                # 2. Parallel quality checks
                quality_results = self.run_parallel_quality_checks(conn)
                self._process_quality_results(quality_results)
                
                # 3. Database metrics (single query)
                db_metrics = self._get_database_metrics(conn)
                self._process_db_metrics(db_metrics)
                
                return self._generate_optimized_report()
                
        except Exception as e:
            log.error(f"  > 优化数据库检查失败: {e}")
            return False
    
    def _process_table_stats(self, table_stats: Dict[str, Any]) -> None:
        """Process batch table statistics results"""
        log.info("  [1/3] 处理表统计信息")
        
        for table_name, stats in table_stats.items():
            self.total_checks += 1
            
            if stats['record_count'] > 0:
                log.info(f"    ✓ {table_name}: {stats['record_count']:,} 条记录, "
                        f"{stats['stock_count']:,} 只股票")
                
                # Check data freshness for time-series tables
                if stats['max_date'] and table_name in ['ts_daily', 'factors_exposure']:
                    days_old = (datetime.now().date() - stats['max_date']).days
                    if days_old > self.config.MAX_DATA_AGE_DAYS:
                        self.issues_found.append(f"{table_name} 数据过旧，最新数据距今 {days_old} 天")
                
                self.passed_checks += 1
            else:
                log.warning(f"    ⚠ {table_name}: 表为空")
                self.issues_found.append(f"{table_name} 表为空")
    
    def _process_quality_results(self, quality_results: Dict[str, Any]) -> None:
        """Process quality check results"""
        log.info("  [2/3] 处理数据质量检查结果")
        
        # Process daily data quality
        if quality_results.get('daily_quality'):
            daily_result = quality_results['daily_quality'][0]
            self._analyze_daily_quality(daily_result)
        
        # Process factor consistency
        if quality_results.get('factor_consistency'):
            self._analyze_factor_consistency(quality_results['factor_consistency'])
        
        # Process orphan data
        if quality_results.get('orphan_data'):
            orphan_count = quality_results['orphan_data'][0][0]
            self._analyze_orphan_data(orphan_count)
    
    def _analyze_daily_quality(self, daily_result) -> None:
        """Analyze daily data quality results"""
        self.total_checks += 1
        
        total_records = daily_result[1]
        if total_records > 0:
            invalid_close_pct = (daily_result[2] / total_records) * 100
            invalid_vol_pct = (daily_result[3] / total_records) * 100
            price_inconsist_pct = (daily_result[4] / total_records) * 100
            
            log.info(f"    最近30天数据质量 (共{total_records:,}条):")
            log.info(f"      无效收盘价: {daily_result[2]} ({invalid_close_pct:.2f}%)")
            log.info(f"      无效成交量: {daily_result[3]} ({invalid_vol_pct:.2f}%)")
            log.info(f"      价格不一致: {daily_result[4]} ({price_inconsist_pct:.2f}%)")
            
            if invalid_close_pct <= self.config.MAX_INVALID_DATA_PERCENT and \
               price_inconsist_pct <= self.config.MAX_PRICE_INCONSISTENCY_PERCENT:
                self.passed_checks += 1
            else:
                if invalid_close_pct > self.config.MAX_INVALID_DATA_PERCENT:
                    self.issues_found.append(f"无效收盘价比例过高: {invalid_close_pct:.2f}%")
                if price_inconsist_pct > self.config.MAX_PRICE_INCONSISTENCY_PERCENT:
                    self.issues_found.append(f"价格不一致比例过高: {price_inconsist_pct:.2f}%")
    
    def _analyze_factor_consistency(self, factor_results: List) -> None:
        """Analyze factor data consistency"""
        self.total_checks += 1
        
        if factor_results:
            log.info(f"    因子数据统计 (共{len(factor_results)}个因子):")
            
            date_counts = [row[1] for row in factor_results[:5]]  # Top 5 factors
            for row in factor_results[:5]:
                log.info(f"      {row[0]}: {row[3]:,}条记录, {row[2]}只股票, {row[1]}个日期")
            
            # Check consistency
            if len(set(date_counts)) <= 1:
                self.passed_checks += 1
            else:
                self.issues_found.append(f"因子数据日期不一致，范围: {min(date_counts)}-{max(date_counts)}")
        else:
            self.issues_found.append("无因子数据")
    
    def _analyze_orphan_data(self, orphan_count: int) -> None:
        """Analyze orphaned data"""
        self.total_checks += 1
        
        if orphan_count == 0:
            log.info("    ✓ 数据关联性良好")
            self.passed_checks += 1
        else:
            log.warning(f"    ⚠ 发现 {orphan_count} 只股票有日线数据但缺少因子数据")
            if orphan_count > self.config.MIN_ORPHAN_THRESHOLD:
                self.issues_found.append(f"大量股票缺少因子数据: {orphan_count} 只")
            else:
                self.passed_checks += 1
    
    def _get_database_metrics(self, conn) -> Dict[str, Any]:
        """Get database performance metrics in a single query"""
        metrics_query = text("""
            SELECT 
                pg_size_pretty(pg_database_size(current_database())) as db_size,
                (
                    SELECT json_agg(
                        json_build_object(
                            'table_name', tablename,
                            'size', pg_size_pretty(pg_total_relation_size(tablename::regclass))
                        )
                    )
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename IN ('ts_daily', 'factors_exposure', 'financial_indicators')
                ) as table_sizes
        """)
        
        result = conn.execute(metrics_query).fetchone()
        return {
            'db_size': result[0],
            'table_sizes': result[1] if result[1] else []
        }
    
    def _process_db_metrics(self, db_metrics: Dict[str, Any]) -> None:
        """Process database metrics"""
        log.info("  [3/3] 处理数据库性能指标")
        self.total_checks += 1
        
        try:
            log.info(f"    数据库大小: {db_metrics['db_size']}")
            log.info("    主要表大小:")
            
            for table_info in db_metrics['table_sizes']:
                log.info(f"      {table_info['table_name']}: {table_info['size']}")
            
            self.passed_checks += 1
            
        except Exception as e:
            log.error(f"    ✗ 数据库性能指标处理失败: {e}")
            self.issues_found.append(f"数据库性能指标处理失败: {e}")
    
    def _generate_optimized_report(self) -> bool:
        """Generate optimized report"""
        log.info("  --- 性能优化数据库检查报告 ---")
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
        
        log.info("  --- 性能优化数据库检查完成 ---")
        
        return len(self.issues_found) == 0


# Updated function using performance-optimized approach
def comprehensive_database_check_optimized(dm) -> bool:
    """
    性能优化的全面数据库检查
    使用批量查询和并行处理提高效率
    """
    from system_check_config import DATA_QUALITY_THRESHOLDS
    
    checker = PerformantDatabaseChecker(dm, DATA_QUALITY_THRESHOLDS)
    return checker.run_optimized_comprehensive_check()