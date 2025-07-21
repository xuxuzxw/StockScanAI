"""
Centralized configuration for system checks
"""
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


@dataclass
class PerformanceThresholds:
    """Performance-related thresholds"""
    SIMPLE_QUERY_MAX_MS: int = 100
    COMPLEX_QUERY_MAX_MS: int = 1000
    FACTOR_CALCULATION_MAX_MS: int = 5000
    API_RESPONSE_MAX_MS: int = 3000
    
    # Resource thresholds
    MAX_CPU_PERCENT: float = 80.0
    MAX_MEMORY_PERCENT: float = 85.0
    MAX_DISK_PERCENT: float = 90.0


@dataclass
class DataQualityThresholds:
    """Data quality thresholds"""
    MIN_COVERAGE_RATIO: float = 0.8
    MIN_STOCK_COVERAGE_RATIO: float = 0.8
    MAX_DATA_AGE_DAYS: int = 7
    MAX_INVALID_DATA_PERCENT: float = 1.0
    MAX_PRICE_INCONSISTENCY_PERCENT: float = 0.1
    MAX_FINANCIAL_DATA_AGE_MONTHS: int = 6
    MIN_ORPHAN_THRESHOLD: int = 100
    
    # Core tables that must exist
    CORE_TABLES: Dict[str, str] = None
    TIME_RANGE_TABLES: List[str] = None
    
    def __post_init__(self):
        if self.CORE_TABLES is None:
            self.CORE_TABLES = {
                'stock_basic': '股票基本信息',
                'ts_daily': '日线行情数据', 
                'factors_exposure': '因子暴露数据',
                'financial_indicators': '财务指标数据',
                'ts_adj_factor': '复权因子数据'
            }
        
        if self.TIME_RANGE_TABLES is None:
            self.TIME_RANGE_TABLES = ['ts_daily', 'factors_exposure', 'financial_indicators']


@dataclass
class TestConfiguration:
    """Test configuration"""
    TEST_STOCKS: List[str] = None
    STABLE_FACTORS: List[str] = None
    MIN_FACTOR_SUCCESS_RATE: float = 0.6
    DEFAULT_CONCURRENT_REQUESTS: int = 10
    MIN_API_SUCCESS_RATE: float = 0.8
    
    def __post_init__(self):
        if self.TEST_STOCKS is None:
            self.TEST_STOCKS = ["600519.SH", "000001.SZ", "000002.SZ"]
        
        if self.STABLE_FACTORS is None:
            self.STABLE_FACTORS = ["pe_ttm", "pb"]


@dataclass
class SystemCheckConfig:
    """Main configuration class"""
    performance: PerformanceThresholds = None
    data_quality: DataQualityThresholds = None
    test: TestConfiguration = None
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    DETAILED_LOGGING: bool = True
    
    # Timeout configuration
    DEFAULT_CHECK_TIMEOUT: int = 30
    CRITICAL_CHECK_TIMEOUT: int = 60
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = PerformanceThresholds()
        if self.data_quality is None:
            self.data_quality = DataQualityThresholds()
        if self.test is None:
            self.test = TestConfiguration()


# Global configuration instance
CONFIG = SystemCheckConfig()


# Configuration validation
def validate_config(config: SystemCheckConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Validate performance thresholds
    if config.performance.SIMPLE_QUERY_MAX_MS <= 0:
        issues.append("SIMPLE_QUERY_MAX_MS must be positive")
    
    if config.performance.MAX_CPU_PERCENT <= 0 or config.performance.MAX_CPU_PERCENT > 100:
        issues.append("MAX_CPU_PERCENT must be between 0 and 100")
    
    # Validate data quality thresholds
    if config.data_quality.MIN_COVERAGE_RATIO <= 0 or config.data_quality.MIN_COVERAGE_RATIO > 1:
        issues.append("MIN_COVERAGE_RATIO must be between 0 and 1")
    
    # Validate test configuration
    if not config.test.TEST_STOCKS:
        issues.append("TEST_STOCKS cannot be empty")
    
    if config.test.MIN_FACTOR_SUCCESS_RATE <= 0 or config.test.MIN_FACTOR_SUCCESS_RATE > 1:
        issues.append("MIN_FACTOR_SUCCESS_RATE must be between 0 and 1")
    
    return issues


# Environment-specific configurations
class ConfigProfiles:
    """Predefined configuration profiles"""
    
    @staticmethod
    def development() -> SystemCheckConfig:
        """Development environment configuration"""
        config = SystemCheckConfig()
        config.performance.SIMPLE_QUERY_MAX_MS = 200  # More lenient
        config.data_quality.MIN_COVERAGE_RATIO = 0.7  # Lower requirement
        config.test.MIN_FACTOR_SUCCESS_RATE = 0.5  # Lower requirement
        return config
    
    @staticmethod
    def production() -> SystemCheckConfig:
        """Production environment configuration"""
        config = SystemCheckConfig()
        config.performance.SIMPLE_QUERY_MAX_MS = 50  # Stricter
        config.data_quality.MIN_COVERAGE_RATIO = 0.95  # Higher requirement
        config.test.MIN_FACTOR_SUCCESS_RATE = 0.8  # Higher requirement
        return config
    
    @staticmethod
    def testing() -> SystemCheckConfig:
        """Testing environment configuration"""
        config = SystemCheckConfig()
        config.performance.SIMPLE_QUERY_MAX_MS = 500  # Very lenient
        config.data_quality.MIN_COVERAGE_RATIO = 0.5  # Low requirement
        config.test.TEST_STOCKS = ["600519.SH"]  # Single stock for speed
        config.test.STABLE_FACTORS = ["pe_ttm"]  # Single factor
        return config


def get_config(profile: str = "default") -> SystemCheckConfig:
    """Get configuration by profile name"""
    profiles = {
        "development": ConfigProfiles.development,
        "production": ConfigProfiles.production,
        "testing": ConfigProfiles.testing,
        "default": lambda: CONFIG
    }
    
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(profiles.keys())}")
    
    config = profiles[profile]()
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        raise ValueError(f"Configuration validation failed: {issues}")
    
    return config