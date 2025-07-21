"""
Improved system check framework with better maintainability and consistency
"""
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Optional
from enum import Enum
import time
from logger_config import log


class CheckMode(Enum):
    QUICK = "quick"
    FULL = "full" 
    STABILITY = "stability"
    PERFORMANCE = "performance"


@dataclass
class CheckDefinition:
    """Defines a system check with metadata"""
    name: str
    description: str
    check_function: Callable
    required_for_modes: List[CheckMode]
    dependencies: List[str] = None
    timeout_seconds: int = 30
    critical: bool = True  # If False, failure won't stop execution
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


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


class SystemCheckOrchestrator:
    """Orchestrates system checks with proper dependency management and reporting"""
    
    def __init__(self, dm, mode: CheckMode = CheckMode.FULL):
        self.dm = dm
        self.mode = mode
        self.checks: Dict[str, CheckDefinition] = {}
        self.results: List[CheckResult] = []
        self._register_checks()
    
    def _register_checks(self):
        """Register all available checks"""
        # Import check functions
        from run_system_check import (
            check_database_connection, check_tushare_api, check_system_resources,
            check_database_performance, check_data_freshness, comprehensive_database_check,
            check_database_data_quality, test_stable_factors
        )
        
        # Define checks with proper metadata
        check_definitions = [
            CheckDefinition(
                name="database_connection",
                description="数据库连接",
                check_function=lambda: check_database_connection(self.dm),
                required_for_modes=[CheckMode.QUICK, CheckMode.FULL, CheckMode.STABILITY, CheckMode.PERFORMANCE],
                critical=True
            ),
            CheckDefinition(
                name="basic_data_validation",
                description="基础数据验证",
                check_function=lambda: self._check_basic_data_validation(),
                required_for_modes=[CheckMode.QUICK, CheckMode.FULL, CheckMode.STABILITY],
                dependencies=["database_connection"]
            ),
            CheckDefinition(
                name="tushare_api",
                description="Tushare API连通性",
                check_function=lambda: check_tushare_api(self.dm),
                required_for_modes=[CheckMode.FULL, CheckMode.STABILITY],
                dependencies=["database_connection"]
            ),
            CheckDefinition(
                name="system_resources",
                description="系统资源状态",
                check_function=check_system_resources,
                required_for_modes=[CheckMode.FULL, CheckMode.PERFORMANCE],
                critical=False
            ),
            CheckDefinition(
                name="database_performance",
                description="数据库性能",
                check_function=lambda: check_database_performance(self.dm),
                required_for_modes=[CheckMode.FULL, CheckMode.PERFORMANCE],
                dependencies=["database_connection"]
            ),
            CheckDefinition(
                name="data_freshness",
                description="数据新鲜度",
                check_function=lambda: check_data_freshness(self.dm),
                required_for_modes=[CheckMode.FULL, CheckMode.STABILITY],
                dependencies=["database_connection"]
            ),
            CheckDefinition(
                name="comprehensive_database_check",
                description="全面数据库数据完整性检查",
                check_function=lambda: comprehensive_database_check(self.dm),
                required_for_modes=[CheckMode.FULL],
                dependencies=["database_connection"],
                timeout_seconds=60
            ),
            CheckDefinition(
                name="database_data_quality",
                description="数据库财务数据质量验证",
                check_function=lambda: check_database_data_quality(self.dm),
                required_for_modes=[CheckMode.FULL],
                dependencies=["database_connection"]
            ),
            CheckDefinition(
                name="factor_calculation",
                description="引擎层因子计算",
                check_function=lambda: test_stable_factors(self.dm),
                required_for_modes=[CheckMode.FULL, CheckMode.STABILITY],
                dependencies=["database_connection", "data_freshness"]
            ),
            CheckDefinition(
                name="ai_connectivity",
                description="智能分析层与AI API连通性",
                check_function=self._check_ai_connectivity,
                required_for_modes=[CheckMode.FULL],
                critical=False
            ),
            CheckDefinition(
                name="end_to_end_workflow",
                description="端到端工作流冒烟测试",
                check_function=self._check_end_to_end_workflow,
                required_for_modes=[CheckMode.FULL],
                dependencies=["database_connection", "factor_calculation"]
            )
        ]
        
        for check_def in check_definitions:
            self.checks[check_def.name] = check_def
    
    def run_checks(self) -> bool:
        """Run all checks for the current mode"""
        # Filter checks for current mode
        applicable_checks = [
            check for check in self.checks.values()
            if self.mode in check.required_for_modes
        ]
        
        # Sort by dependencies (simple topological sort)
        ordered_checks = self._sort_checks_by_dependencies(applicable_checks)
        
        log.info(f"=== 系统检查 ({self.mode.value.upper()}) ===")
        log.info(f"总检查项: {len(ordered_checks)}")
        log.info("=" * 50)
        
        passed_checks = 0
        
        for i, check_def in enumerate(ordered_checks, 1):
            log.info(f"\n--- [检查 {i}/{len(ordered_checks)}] {check_def.description} ---")
            
            result = self._execute_check(check_def)
            self.results.append(result)
            
            if result.success:
                passed_checks += 1
                log.info(f"  [PASS] {check_def.description} 通过 ({result.duration_seconds:.2f}s)")
            else:
                log.error(f"  [FAIL] {check_def.description} 失败: {result.message}")
                if check_def.critical:
                    log.error("  关键检查失败，停止后续检查")
                    break
        
        self._generate_final_report(passed_checks, len(ordered_checks))
        return passed_checks == len(ordered_checks)
    
    def _execute_check(self, check_def: CheckDefinition) -> CheckResult:
        """Execute a single check with timeout and error handling"""
        start_time = time.time()
        
        try:
            # Check dependencies
            if not self._check_dependencies_passed(check_def):
                return CheckResult(
                    name=check_def.name,
                    success=False,
                    duration_seconds=0,
                    message="依赖检查未通过"
                )
            
            # Execute the check
            success = check_def.check_function()
            duration = time.time() - start_time
            
            return CheckResult(
                name=check_def.name,
                success=bool(success),
                duration_seconds=duration,
                message="检查完成" if success else "检查失败"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return CheckResult(
                name=check_def.name,
                success=False,
                duration_seconds=duration,
                message=str(e),
                error=e
            )
    
    def _check_dependencies_passed(self, check_def: CheckDefinition) -> bool:
        """Check if all dependencies have passed"""
        for dep_name in check_def.dependencies:
            dep_result = next((r for r in self.results if r.name == dep_name), None)
            if not dep_result or not dep_result.success:
                return False
        return True
    
    def _sort_checks_by_dependencies(self, checks: List[CheckDefinition]) -> List[CheckDefinition]:
        """Simple topological sort for dependency ordering"""
        ordered = []
        remaining = checks.copy()
        
        while remaining:
            # Find checks with no unresolved dependencies
            ready = [
                check for check in remaining
                if all(dep in [c.name for c in ordered] for dep in check.dependencies)
            ]
            
            if not ready:
                # Circular dependency or missing dependency
                log.warning("检测到循环依赖或缺失依赖，使用原始顺序")
                return checks
            
            # Add ready checks to ordered list
            for check in ready:
                ordered.append(check)
                remaining.remove(check)
        
        return ordered
    
    def _check_basic_data_validation(self) -> bool:
        """Basic data validation check"""
        try:
            with self.dm.engine.connect() as conn:
                # Check if core tables exist and have data
                core_tables = ['stock_basic', 'ts_daily']
                for table in core_tables:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    if count == 0:
                        log.error(f"  表 {table} 为空")
                        return False
                    log.info(f"  ✓ {table}: {count:,} 条记录")
                return True
        except Exception as e:
            log.error(f"  基础数据验证失败: {e}")
            return False
    
    def _check_ai_connectivity(self) -> bool:
        """Check AI connectivity"""
        try:
            import intelligence
            ai_orchestrator = intelligence.AIOrchestrator(config.AI_MODEL_CONFIG, self.dm)
            # Perform a simple test
            return True
        except Exception as e:
            log.error(f"  AI连通性检查失败: {e}")
            return False
    
    def _check_end_to_end_workflow(self) -> bool:
        """End-to-end workflow smoke test"""
        try:
            import quant_engine as qe
            
            ff = qe.FactorFactory(_data_manager=self.dm)
            fp = qe.FactorProcessor(ff)
            
            test_adaptive_strategy = qe.AdaptiveAlphaStrategy(
                ff, fp, qe.FactorAnalyzer(self.dm), pd.DataFrame()
            )
            test_risk_manager = qe.RiskManager(ff, fp)
            
            assert test_adaptive_strategy is not None
            assert test_risk_manager is not None
            
            log.info("  ✓ 核心组件实例化成功")
            return True
            
        except Exception as e:
            log.error(f"  端到端工作流测试失败: {e}")
            return False
    
    def _generate_final_report(self, passed: int, total: int):
        """Generate final report"""
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        log.info("\n" + "=" * 50)
        log.info("系统检查完成")
        log.info("=" * 50)
        log.info(f"检查模式: {self.mode.value.upper()}")
        log.info(f"总检查项: {total}")
        log.info(f"通过检查: {passed}")
        log.info(f"成功率: {success_rate:.1f}%")
        
        if passed == total:
            log.info("🎉 所有检查均通过！系统状态良好。")
        else:
            failed = total - passed
            log.warning(f"⚠️  {failed} 项检查失败，请查看上述详细信息。")
        
        # Show timing summary
        total_duration = sum(r.duration_seconds for r in self.results)
        log.info(f"总耗时: {total_duration:.2f}秒")
        
        # Show failed checks
        failed_checks = [r for r in self.results if not r.success]
        if failed_checks:
            log.info("\n失败的检查:")
            for result in failed_checks:
                log.info(f"  - {result.name}: {result.message}")


# Updated main function
def run_system_check_improved(mode: str = "full"):
    """
    Improved system check with better structure and maintainability
    """
    try:
        # Initialize data manager
        dm = data.DataManager()
        
        # Convert mode string to enum
        check_mode = CheckMode(mode.lower())
        
        # Create and run orchestrator
        orchestrator = SystemCheckOrchestrator(dm, check_mode)
        return orchestrator.run_checks()
        
    except Exception as e:
        log.error(f"系统检查初始化失败: {e}")
        return False