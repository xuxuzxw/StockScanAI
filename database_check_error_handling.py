"""
Enhanced error handling for database checks
"""
from typing import Optional, Callable, Any
import functools
from logger_config import log
from system_check_error_handler import CheckResult, safe_execute


class DatabaseCheckError(Exception):
    """Base exception for database check errors"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class TableNotFoundError(DatabaseCheckError):
    """Raised when a required table is not found"""
    pass


class DataQualityError(DatabaseCheckError):
    """Raised when data quality issues are detected"""
    pass


class ConnectionError(DatabaseCheckError):
    """Raised when database connection fails"""
    pass


class TimeoutError(DatabaseCheckError):
    """Raised when operation times out"""
    pass


def database_check_handler(check_name: str):
    """
    Decorator for database check methods that provides consistent error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, conn, *args, **kwargs):
            self.total_checks += 1
            try:
                result = func(self, conn, *args, **kwargs)
                self.passed_checks += 1
                return result
            except TableNotFoundError as e:
                error_msg = f"表不存在: {e}"
                log.error(f"    ✗ {check_name}: {error_msg}")
                self.issues_found.append(error_msg)
            except DataQualityError as e:
                error_msg = f"数据质量问题: {e}"
                log.warning(f"    ⚠ {check_name}: {error_msg}")
                self.issues_found.append(error_msg)
            except Exception as e:
                error_msg = f"{check_name} 检查失败: {e}"
                log.error(f"    ✗ {check_name}: {error_msg}")
                self.issues_found.append(error_msg)
        return wrapper
    return decorator


def safe_database_operation(operation_name: str):
    """
    Decorator for safe database operations with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            success, result = safe_execute(func, *args, **kwargs)
            if not success:
                log.error(f"数据库操作失败: {operation_name}")
                return None
            return result
        return wrapper
    return decorator


class DatabaseHealthChecker:
    """
    Enhanced database health checker with better error handling
    """
    
    def __init__(self, dm, config=None):
        self.dm = dm
        self.config = config
        self.issues_found = []
        self.total_checks = 0
        self.passed_checks = 0
        self.check_results = []
    
    @database_check_handler("核心表检查")
    def _check_core_tables(self, conn):
        """Check core tables with enhanced error handling"""
        missing_tables = []
        
        for table, desc in self.config.CORE_TABLES.items():
            try:
                # Check if table exists first
                exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": table}).scalar()
                
                if not exists:
                    raise TableNotFoundError(f"表 {table} 不存在")
                
                # Get record count
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                
                if count == 0:
                    raise DataQualityError(f"表 {table} 为空")
                
                log.info(f"    ✓ {desc} ({table}): {count:,} 条记录")
                
                # Store result for reporting
                self.check_results.append(CheckResult(
                    name=f"{table}_count",
                    success=True,
                    message=f"{desc}: {count:,} 条记录"
                ))
                
            except (TableNotFoundError, DataQualityError):
                raise  # Re-raise specific errors
            except Exception as e:
                raise DatabaseCheckError(f"检查表 {table} 时发生错误: {e}")
    
    @safe_database_operation("获取表统计信息")
    def _get_table_stats(self, conn, table: str) -> Optional[dict]:
        """Safely get table statistics"""
        try:
            result = conn.execute(text(f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ts_code) as unique_stocks
                FROM {table}
            """)).fetchone()
            
            return {
                'total_records': result[0],
                'unique_stocks': result[1]
            } if result else None
            
        except Exception as e:
            log.error(f"获取表 {table} 统计信息失败: {e}")
            return None
    
    def generate_detailed_report(self) -> dict:
        """Generate a detailed report with structured results"""
        return {
            'summary': {
                'total_checks': self.total_checks,
                'passed_checks': self.passed_checks,
                'success_rate': self.passed_checks / max(1, self.total_checks),
                'issues_count': len(self.issues_found)
            },
            'issues': self.issues_found,
            'check_results': [result.to_dict() for result in self.check_results],
            'timestamp': datetime.now().isoformat()
        }