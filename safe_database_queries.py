"""
Safe database query utilities to prevent SQL injection
"""
from typing import Dict, List, Optional, Any
from sqlalchemy import text
from logger_config import log


class SafeQueryBuilder:
    """Builder for safe database queries"""
    
    # Whitelist of allowed table names
    ALLOWED_TABLES = {
        'stock_basic', 'ts_daily', 'factors_exposure', 
        'financial_indicators', 'ts_adj_factor'
    }
    
    # Whitelist of allowed column names
    ALLOWED_COLUMNS = {
        'ts_code', 'trade_date', 'end_date', 'close', 'vol', 
        'high', 'low', 'open', 'factor_name', 'factor_value'
    }
    
    @classmethod
    def validate_table_name(cls, table_name: str) -> bool:
        """Validate table name against whitelist"""
        return table_name in cls.ALLOWED_TABLES
    
    @classmethod
    def validate_column_name(cls, column_name: str) -> bool:
        """Validate column name against whitelist"""
        return column_name in cls.ALLOWED_COLUMNS
    
    @classmethod
    def build_count_query(cls, table_name: str) -> text:
        """Build safe count query"""
        if not cls.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        
        # Use parameterized query with identifier
        return text(f"SELECT COUNT(*) FROM {table_name}")
    
    @classmethod
    def build_time_range_query(cls, table_name: str, date_column: str) -> text:
        """Build safe time range query"""
        if not cls.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        if not cls.validate_column_name(date_column):
            raise ValueError(f"Invalid column name: {date_column}")
        
        return text(f"""
            SELECT MIN({date_column}) as min_date, 
                   MAX({date_column}) as max_date,
                   COUNT(DISTINCT {date_column}) as date_count
            FROM {table_name}
        """)
    
    @classmethod
    def build_stock_count_query(cls, table_name: str) -> text:
        """Build safe stock count query"""
        if not cls.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        
        if table_name == 'stock_basic':
            return text(f"SELECT COUNT(*) FROM {table_name}")
        else:
            return text(f"SELECT COUNT(DISTINCT ts_code) FROM {table_name}")
    
    @classmethod
    def build_data_quality_query(cls) -> text:
        """Build data quality check query"""
        return text("""
            SELECT 
                COUNT(*) as total_records,
                SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) as invalid_close,
                SUM(CASE WHEN vol IS NULL OR vol < 0 THEN 1 ELSE 0 END) as invalid_volume,
                SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as price_inconsistency
            FROM ts_daily
            WHERE trade_date >= (SELECT MAX(trade_date) - INTERVAL '30 days' FROM ts_daily)
        """)
    
    @classmethod
    def build_factor_stats_query(cls) -> text:
        """Build factor statistics query"""
        return text("""
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
        """)
    
    @classmethod
    def build_financial_stats_query(cls) -> text:
        """Build financial data statistics query"""
        return text("""
            SELECT 
                COUNT(DISTINCT ts_code) as stock_count,
                COUNT(DISTINCT end_date) as period_count,
                MIN(end_date) as earliest_period,
                MAX(end_date) as latest_period,
                COUNT(*) as total_records
            FROM financial_indicators
        """)
    
    @classmethod
    def build_orphan_data_query(cls) -> text:
        """Build query to find orphaned daily data"""
        return text("""
            SELECT COUNT(DISTINCT d.ts_code) 
            FROM ts_daily d 
            LEFT JOIN factors_exposure f ON d.ts_code = f.ts_code AND d.trade_date = f.trade_date
            WHERE f.ts_code IS NULL 
            AND d.trade_date >= (SELECT MAX(trade_date) - INTERVAL '7 days' FROM ts_daily)
        """)
    
    @classmethod
    def build_table_size_query(cls) -> text:
        """Build query to get table sizes"""
        table_list = "', '".join(cls.ALLOWED_TABLES)
        return text(f"""
            SELECT 
                tablename,
                pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('{table_list}')
            ORDER BY pg_total_relation_size(tablename::regclass) DESC
        """)


class SafeDatabaseChecker:
    """Database checker using safe query methods"""
    
    def __init__(self, dm, config=None):
        self.dm = dm
        self.config = config
        self.query_builder = SafeQueryBuilder()
        self.issues_found = []
        self.total_checks = 0
        self.passed_checks = 0
    
    def check_core_tables_safe(self, conn) -> None:
        """Safely check core tables"""
        for table, desc in self.config.CORE_TABLES.items():
            self.total_checks += 1
            try:
                query = self.query_builder.build_count_query(table)
                count = conn.execute(query).scalar()
                log.info(f"    ✓ {desc} ({table}): {count:,} 条记录")
                self.passed_checks += 1
            except ValueError as e:
                log.error(f"    ✗ 查询构建错误: {e}")
                self.issues_found.append(f"查询构建错误: {e}")
            except Exception as e:
                log.error(f"    ✗ {desc} ({table}): 检查失败 - {e}")
                self.issues_found.append(f"表 {table} 检查失败: {e}")
    
    def check_time_ranges_safe(self, conn) -> None:
        """Safely check time ranges"""
        for table in self.config.TIME_RANGE_TABLES:
            self.total_checks += 1
            try:
                date_col = 'end_date' if table == 'financial_indicators' else 'trade_date'
                query = self.query_builder.build_time_range_query(table, date_col)
                result = conn.execute(query).fetchone()
                
                if result and result[0]:
                    log.info(f"    ✓ {table}: {result[0]} 至 {result[1]} ({result[2]} 个日期)")
                    self.passed_checks += 1
                    
                    # Check data freshness
                    if table in ['ts_daily', 'factors_exposure']:
                        days_old = (datetime.now().date() - result[1]).days
                        if days_old > self.config.MAX_DATA_AGE_DAYS:
                            self.issues_found.append(f"{table} 数据过旧，最新数据距今 {days_old} 天")
                else:
                    log.warning(f"    ⚠ {table}: 无有效日期数据")
                    self.issues_found.append(f"{table} 无有效日期数据")
                    
            except ValueError as e:
                log.error(f"    ✗ 查询构建错误: {e}")
                self.issues_found.append(f"查询构建错误: {e}")
            except Exception as e:
                log.error(f"    ✗ {table} 时间范围检查失败: {e}")
                self.issues_found.append(f"{table} 时间范围检查失败: {e}")
    
    def check_data_quality_safe(self, conn) -> None:
        """Safely check data quality"""
        self.total_checks += 1
        try:
            query = self.query_builder.build_data_quality_query()
            quality_issues = conn.execute(query).fetchone()
            
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