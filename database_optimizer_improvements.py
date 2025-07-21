# Additional improvements for DatabaseFinalOptimizer

# 1. Add connection pooling for better performance
from sqlalchemy.pool import QueuePool

class DatabaseFinalOptimizer:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or config.DATABASE_URL
        # Add connection pooling
        self.engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True  # Validate connections before use
        )

# 2. Add caching for repeated queries
from functools import lru_cache
from threading import Lock

class QueryCache:
    def __init__(self, max_size: int = 100):
        self._cache = {}
        self._lock = Lock()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value

# 3. Add async support for concurrent operations
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

class AsyncDatabaseOptimizer:
    async def run_parallel_tests(self, queries: List[QueryTestCase]) -> List[PerformanceResult]:
        """Run performance tests in parallel"""
        tasks = [self._test_query_async(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 4. Add configuration management
@dataclass
class OptimizerConfig:
    """优化器配置"""
    max_limit_value: int = 10000
    performance_test_iterations: int = 3
    connection_pool_size: int = 5
    cache_size: int = 100
    timeout_seconds: int = 30
    
    @classmethod
    def from_env(cls) -> 'OptimizerConfig':
        """从环境变量加载配置"""
        import os
        return cls(
            max_limit_value=int(os.getenv('DB_MAX_LIMIT', 10000)),
            performance_test_iterations=int(os.getenv('DB_TEST_ITERATIONS', 3)),
            connection_pool_size=int(os.getenv('DB_POOL_SIZE', 5)),
            cache_size=int(os.getenv('DB_CACHE_SIZE', 100)),
            timeout_seconds=int(os.getenv('DB_TIMEOUT', 30))
        )

# 5. Add comprehensive logging and monitoring
import logging
from datetime import datetime
from typing import ContextManager

class PerformanceMonitor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
    
    @contextmanager
    def measure(self, operation: str) -> ContextManager:
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics[operation] = duration
            self.logger.info(f"{operation} completed in {duration:.3f}s")
    
    def get_summary(self) -> Dict[str, float]:
        return self.metrics.copy()

# 6. Add retry mechanism for database operations
from tenacity import retry, stop_after_attempt, wait_exponential

class DatabaseFinalOptimizer:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _execute_with_retry(self, sql: str) -> Any:
        """Execute SQL with retry logic"""
        with self.get_connection() as conn:
            return conn.execute(text(sql)).fetchall()

# 7. Add validation and error handling
class ValidationError(Exception):
    """自定义验证错误"""
    pass

class DatabaseFinalOptimizer:
    def _validate_sql_query(self, sql: str) -> None:
        """验证SQL查询安全性"""
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        sql_upper = sql.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ValidationError(f"Dangerous SQL keyword detected: {keyword}")
        
        # Check for potential SQL injection patterns
        injection_patterns = [
            r"';.*--",  # Comment injection
            r"UNION.*SELECT",  # Union injection
            r"OR.*1=1",  # Boolean injection
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_upper):
                raise ValidationError(f"Potential SQL injection detected: {pattern}")

# 8. Add comprehensive testing framework
import unittest
from unittest.mock import Mock, patch

class TestDatabaseOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = DatabaseFinalOptimizer("sqlite:///:memory:")
    
    def test_extract_ts_code_valid(self):
        sql = "SELECT * FROM ts_daily WHERE ts_code = '600519.SH'"
        result = self.optimizer._extract_ts_code(sql)
        self.assertEqual(result, "600519.SH")
    
    def test_extract_ts_code_invalid(self):
        sql = "SELECT * FROM ts_daily WHERE ts_code = 'INVALID'"
        result = self.optimizer._extract_ts_code(sql)
        self.assertIsNone(result)
    
    @patch('database_final_optimizer.log')
    def test_performance_test_with_mock(self, mock_log):
        # Test with mocked database
        pass

# 9. Add health check functionality
class HealthChecker:
    def __init__(self, optimizer: DatabaseFinalOptimizer):
        self.optimizer = optimizer
    
    def check_database_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with self.optimizer.get_connection() as conn:
                conn.execute(text("SELECT 1")).fetchone()
            return True
        except Exception as e:
            log.error(f"Database connection failed: {e}")
            return False
    
    def check_materialized_views(self) -> Dict[str, bool]:
        """检查物化视图状态"""
        results = {}
        with self.optimizer.get_connection() as conn:
            for view in self.optimizer.MATERIALIZED_VIEWS:
                try:
                    conn.execute(text(f"SELECT COUNT(*) FROM {view}")).fetchone()
                    results[view] = True
                except Exception:
                    results[view] = False
        return results
    
    def comprehensive_health_check(self) -> Dict[str, Any]:
        """全面健康检查"""
        return {
            'database_connection': self.check_database_connection(),
            'materialized_views': self.check_materialized_views(),
            'timestamp': datetime.now().isoformat()
        }