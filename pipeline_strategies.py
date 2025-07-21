"""
Strategy pattern implementation for different pipeline processing modes
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from logger_config import log


class PipelineStrategy(ABC):
    """Abstract strategy for pipeline processing"""
    
    @abstractmethod
    def determine_dates_to_process(self, dm) -> List[str]:
        """Determine which dates need processing"""
        pass
    
    @abstractmethod
    def should_skip_date(self, dm, trade_date: str) -> bool:
        """Check if a date should be skipped"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name for logging"""
        pass


class IncrementalStrategy(PipelineStrategy):
    """Strategy for processing only the latest missing data"""
    
    def __init__(self, min_factor_threshold: int = 100):
        self.min_factor_threshold = min_factor_threshold
    
    def determine_dates_to_process(self, dm) -> List[str]:
        """Only process the latest available trade date if data is missing"""
        log.info("===== 使用增量策略确定处理日期 =====")
        
        try:
            from run_daily_pipeline import get_latest_available_trade_date, check_data_exists
            
            latest_trade_date = get_latest_available_trade_date(dm)
            log.info(f"确定最新可用交易日: {latest_trade_date}")
            
            data_status = check_data_exists(dm, latest_trade_date)
            
            if data_status['should_skip']:
                log.info(f"✅ {latest_trade_date} 的数据已存在且完整，跳过处理")
                return []
            
            log.info(f"📊 将处理交易日: {latest_trade_date}")
            return [latest_trade_date]
            
        except Exception as e:
            log.error(f"增量策略失败: {e}")
            return []
    
    def should_skip_date(self, dm, trade_date: str) -> bool:
        """Check if date should be skipped based on existing data"""
        try:
            from run_daily_pipeline import check_data_exists
            data_status = check_data_exists(dm, trade_date)
            return data_status['should_skip']
        except Exception:
            return False
    
    def get_strategy_name(self) -> str:
        return "增量处理策略"


class BackfillStrategy(PipelineStrategy):
    """Strategy for backfilling missing historical data"""
    
    def __init__(self, start_date: str = "20100101"):
        self.start_date = start_date
    
    def determine_dates_to_process(self, dm) -> List[str]:
        """Find all missing dates in historical data"""
        log.info("===== 使用回填策略确定处理日期 =====")
        
        try:
            # Get last processed date from database
            with dm.engine.connect() as connection:
                from sqlalchemy import text
                query_max_date = text("SELECT MAX(trade_date) FROM factors_exposure")
                last_processed_date = connection.execute(query_max_date).scalar_one_or_none()
            
            # Determine start date for calendar
            if last_processed_date:
                start_cal_date = (pd.to_datetime(last_processed_date) + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start_cal_date = self.start_date
            
            end_cal_date = datetime.now().strftime("%Y%m%d")
            
            # Get trading calendar
            cal_df = dm.pro.trade_cal(exchange="", start_date=start_cal_date, end_date=end_cal_date)
            all_trade_dates = set(cal_df[cal_df["is_open"] == 1]["cal_date"].tolist())
            
            # Get existing dates
            existing_dates = set()
            if last_processed_date:
                with dm.engine.connect() as connection:
                    from sqlalchemy import text
                    query_existing = text(
                        "SELECT DISTINCT trade_date FROM factors_exposure "
                        "WHERE trade_date >= :start_date AND trade_date <= :end_date"
                    )
                    result = connection.execute(query_existing, {
                        "start_date": start_cal_date, "end_date": end_cal_date
                    }).fetchall()
                    existing_dates = {row[0] for row in result}
            
            # Find missing dates
            missing_dates = sorted(list(all_trade_dates - existing_dates))
            
            if not missing_dates:
                log.info("所有历史数据均已是最新，无需回填")
                return []
            
            log.info(f"发现 {len(missing_dates)} 个需要回填的交易日")
            return missing_dates
            
        except Exception as e:
            log.error(f"回填策略失败: {e}")
            return []
    
    def should_skip_date(self, dm, trade_date: str) -> bool:
        """Always process dates in backfill mode"""
        return False
    
    def get_strategy_name(self) -> str:
        return "历史回填策略"


class ForceUpdateStrategy(PipelineStrategy):
    """Strategy for forcing updates of specific dates"""
    
    def __init__(self, target_dates: List[str]):
        self.target_dates = target_dates
    
    def determine_dates_to_process(self, dm) -> List[str]:
        """Process specified dates regardless of existing data"""
        log.info(f"===== 使用强制更新策略，目标日期: {self.target_dates} =====")
        return sorted(self.target_dates)
    
    def should_skip_date(self, dm, trade_date: str) -> bool:
        """Never skip in force update mode"""
        return False
    
    def get_strategy_name(self) -> str:
        return "强制更新策略"


class PipelineContext:
    """Context class for pipeline strategy pattern"""
    
    def __init__(self, strategy: PipelineStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: PipelineStrategy):
        """Change strategy at runtime"""
        self.strategy = strategy
    
    def execute_pipeline(self, dm):
        """Execute pipeline using current strategy"""
        log.info(f"使用策略: {self.strategy.get_strategy_name()}")
        
        dates_to_process = self.strategy.determine_dates_to_process(dm)
        
        if not dates_to_process:
            log.info("===== 管道任务完成（无需处理）=====")
            return
        
        # Process each date
        for trade_date in dates_to_process:
            if self.strategy.should_skip_date(dm, trade_date):
                log.info(f"策略决定跳过日期: {trade_date}")
                continue
            
            log.info(f"===== 开始处理 {trade_date} =====")
            # Execute actual pipeline logic here
            self._process_single_date(dm, trade_date)
    
    def _process_single_date(self, dm, trade_date: str):
        """Process a single date (placeholder for actual logic)"""
        # This would contain the actual pipeline processing logic
        # from the original run_daily_pipeline function
        pass


# Factory for creating strategies
class StrategyFactory:
    """Factory for creating pipeline strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> PipelineStrategy:
        """Create strategy based on type"""
        strategies = {
            "incremental": IncrementalStrategy,
            "backfill": BackfillStrategy,
            "force": ForceUpdateStrategy
        }
        
        if strategy_type not in strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategies[strategy_type](**kwargs)
    
    @staticmethod
    def create_auto_strategy(dm) -> PipelineStrategy:
        """Automatically choose the best strategy based on data state"""
        try:
            # Check if we have any data
            with dm.engine.connect() as conn:
                from sqlalchemy import text
                count = conn.execute(text("SELECT COUNT(*) FROM factors_exposure")).scalar()
            
            if count == 0:
                log.info("数据库为空，选择回填策略")
                return BackfillStrategy()
            else:
                log.info("数据库有数据，选择增量策略")
                return IncrementalStrategy()
                
        except Exception as e:
            log.warning(f"自动策略选择失败，使用增量策略: {e}")
            return IncrementalStrategy()