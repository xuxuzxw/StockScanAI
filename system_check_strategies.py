"""
System check strategies using Strategy Pattern
"""
from abc import ABC, abstractmethod
from typing import List, Callable
import logging

log = logging.getLogger(__name__)


class CheckStrategy(ABC):
    """Abstract base class for check strategies"""
    
    @abstractmethod
    def get_description(self) -> str:
        """Get strategy description"""
        pass
    
    @abstractmethod
    def get_checks(self) -> List[Callable]:
        """Get list of check functions to execute"""
        pass
    
    @abstractmethod
    def get_success_threshold(self) -> float:
        """Get minimum success rate threshold"""
        pass


class QuickCheckStrategy(CheckStrategy):
    """Quick check strategy - essential checks only"""
    
    def get_description(self) -> str:
        return "快速系统检查 - 仅检查核心功能"
    
    def get_checks(self) -> List[Callable]:
        from run_system_check import (
            check_config, check_database_connection, 
            check_tushare_api, test_stable_factors
        )
        return [
            check_config,
            check_database_connection,
            check_tushare_api,
            test_stable_factors
        ]
    
    def get_success_threshold(self) -> float:
        return 0.75  # 75% success rate required


class FullCheckStrategy(CheckStrategy):
    """Full comprehensive check strategy"""
    
    def get_description(self) -> str:
        return "完整系统检查 - 全面检查所有组件"
    
    def get_checks(self) -> List[Callable]:
        from run_system_check import (
            check_config, check_database_connection, check_tushare_api,
            check_system_resources, check_database_performance,
            check_data_freshness, check_database_data_quality,
            test_stable_factors, test_data_storage
        )
        return [
            check_config,
            check_database_connection,
            check_tushare_api,
            check_system_resources,
            check_database_performance,
            check_data_freshness,
            check_database_data_quality,
            test_stable_factors,
            test_data_storage
        ]
    
    def get_success_threshold(self) -> float:
        return 0.8  # 80% success rate required


class StabilityCheckStrategy(CheckStrategy):
    """Stability-focused check strategy"""
    
    def get_description(self) -> str:
        return "系统稳定性检查 - 专注于稳定性和数据质量"
    
    def get_checks(self) -> List[Callable]:
        from run_system_check import (
            check_config, check_database_connection, check_tushare_api,
            check_system_resources, check_database_performance,
            test_data_storage
        )
        return [
            check_config,
            check_database_connection,
            check_tushare_api,
            check_system_resources,
            check_database_performance,
            test_data_storage
        ]
    
    def get_success_threshold(self) -> float:
        return 0.85  # 85% success rate required


class PerformanceCheckStrategy(CheckStrategy):
    """Performance-focused check strategy"""
    
    def get_description(self) -> str:
        return "系统性能检查 - 专注于API和数据库性能"
    
    def get_checks(self) -> List[Callable]:
        from run_system_check import (
            check_config, check_database_connection,
            check_database_performance, test_query_performance_benchmark
        )
        return [
            check_config,
            check_database_connection,
            check_database_performance,
            test_query_performance_benchmark
        ]
    
    def get_success_threshold(self) -> float:
        return 0.75  # 75% success rate required


class CheckStrategyFactory:
    """Factory for creating check strategies"""
    
    _strategies = {
        'quick': QuickCheckStrategy,
        'full': FullCheckStrategy,
        'stability': StabilityCheckStrategy,
        'performance': PerformanceCheckStrategy
    }
    
    @classmethod
    def create_strategy(cls, mode: str) -> CheckStrategy:
        """Create strategy instance for given mode"""
        if mode not in cls._strategies:
            raise ValueError(f"Unknown check mode: {mode}")
        
        return cls._strategies[mode]()
    
    @classmethod
    def get_available_modes(cls) -> List[str]:
        """Get list of available check modes"""
        return list(cls._strategies.keys())