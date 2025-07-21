# Example refactored structure
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import asyncio
import time

class CheckStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"

@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    duration: float
    details: Optional[Dict] = None

class SystemChecker(ABC):
    """Base class for all system checkers"""
    
    @abstractmethod
    async def check(self) -> CheckResult:
        pass
    
    def _time_execution(self, func):
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            return result, duration
        return wrapper

class DatabaseChecker(SystemChecker):
    def __init__(self, data_manager):
        self.dm = data_manager
    
    async def check(self) -> CheckResult:
        try:
            start_time = time.time()
            # Database connection check logic here
            duration = time.time() - start_time
            return CheckResult(
                name="Database Connection",
                status=CheckStatus.PASSED,
                message="Database connection successful",
                duration=duration
            )
        except Exception as e:
            return CheckResult(
                name="Database Connection",
                status=CheckStatus.FAILED,
                message=f"Database connection failed: {e}",
                duration=time.time() - start_time
            )

class SystemHealthChecker:
    def __init__(self):
        self.checkers: List[SystemChecker] = []
        self.results: List[CheckResult] = []
    
    def add_checker(self, checker: SystemChecker):
        self.checkers.append(checker)
    
    async def run_all_checks(self) -> List[CheckResult]:
        """Run all checks concurrently where possible"""
        tasks = [checker.check() for checker in self.checkers]
        self.results = await asyncio.gather(*tasks, return_exceptions=True)
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive report"""
        passed = sum(1 for r in self.results if r.status == CheckStatus.PASSED)
        total = len(self.results)
        
        report = f"System Health Check Report\n"
        report += f"{'='*50}\n"
        report += f"Results: {passed}/{total} checks passed\n\n"
        
        for result in self.results:
            report += f"{result.name}: {result.status.value}\n"
            report += f"  Message: {result.message}\n"
            report += f"  Duration: {result.duration:.3f}s\n\n"
        
        return report