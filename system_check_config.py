"""
Configuration constants for system health checks
"""
from dataclasses import dataclass
from typing import List


@dataclass
class PerformanceThresholds:
    """Performance-related thresholds"""
    SIMPLE_QUERY_MAX_MS: int = 100
    COMPLEX_QUERY_MAX_MS: int = 1000
    API_RESPONSE_MAX_MS: int = 5000
    FACTOR_CALC_MAX_MS: int = 2000


@dataclass
class ResourceThresholds:
    """System resource thresholds"""
    MAX_CPU_PERCENT: int = 80
    MAX_MEMORY_PERCENT: int = 85
    MAX_DISK_PERCENT: int = 90
    MIN_AVAILABLE_MEMORY_GB: float = 1.0
    MIN_AVAILABLE_DISK_GB: float = 5.0


@dataclass
class DataQualityThresholds:
    """Data quality thresholds"""
    MIN_COVERAGE_RATIO: float = 0.9
    MIN_HISTORICAL_DEPTH_QUARTERS: int = 12
    MAX_DATA_LAG_DAYS: int = 3
    MIN_SAMPLE_SIZE: int = 100
    
    # Database check specific thresholds
    MAX_DATA_AGE_DAYS: int = 7
    MIN_STOCK_COVERAGE_RATIO: float = 0.8
    MAX_INVALID_DATA_PERCENT: float = 1.0
    MAX_PRICE_INCONSISTENCY_PERCENT: float = 0.1
    MAX_FINANCIAL_DATA_AGE_MONTHS: int = 6
    MIN_ORPHAN_THRESHOLD: int = 100


@dataclass
class TestConfiguration:
    """Test configuration settings"""
    TEST_STOCKS: List[str] = None
    STABLE_FACTORS: List[str] = None
    MIN_FACTOR_SUCCESS_RATE: float = 0.6
    DEFAULT_CONCURRENT_REQUESTS: int = 10
    MIN_API_SUCCESS_RATE: float = 0.8
    SAMPLE_STOCK_COUNT: int = 5
    
    def __post_init__(self):
        if self.TEST_STOCKS is None:
            self.TEST_STOCKS = ["600519.SH", "000001.SZ", "000002.SZ"]
        if self.STABLE_FACTORS is None:
            self.STABLE_FACTORS = ["pe_ttm", "pb"]


@dataclass
class CheckModeThresholds:
    """Success rate thresholds for different check modes"""
    QUICK_MODE_THRESHOLD: float = 0.75
    FULL_MODE_THRESHOLD: float = 0.8
    STABILITY_MODE_THRESHOLD: float = 0.85
    PERFORMANCE_MODE_THRESHOLD: float = 0.75


# Global configuration instances
PERF_THRESHOLDS = PerformanceThresholds()
RESOURCE_THRESHOLDS = ResourceThresholds()
DATA_QUALITY_THRESHOLDS = DataQualityThresholds()
TEST_CONFIG = TestConfiguration()
MODE_THRESHOLDS = CheckModeThresholds()


# Database connection settings
DB_CONNECTION_TIMEOUT = 30  # seconds
DB_QUERY_TIMEOUT = 60  # seconds

# API settings
API_TIMEOUT = 30  # seconds
API_RETRY_COUNT = 3
API_RETRY_DELAY = 1.0  # seconds

# Logging settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File paths
REPORT_OUTPUT_DIR = "reports"
LOG_OUTPUT_DIR = "logs"