"""
Refactored factor calculation with better separation of concerns
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from logger_config import log


@dataclass
class FactorCalculationConfig:
    """Configuration for factor calculations"""
    MOMENTUM_LOOKBACK_DAYS: int = 21
    VOLATILITY_LOOKBACK_DAYS: int = 20
    TRADING_DAYS_PER_YEAR: int = 252
    MIN_TRADING_DAYS_FOR_MOMENTUM: int = 21
    MIN_TRADING_DAYS_FOR_VOLATILITY: int = 20
    
    # Database configuration
    DB_WRITE_CHUNK_SIZE: int = 10000
    
    # Factor groups
    VECTORIZED_FACTORS: List[str] = None
    FINANCIAL_FACTORS: List[str] = None
    LOOP_FACTORS: List[str] = None
    
    def __post_init__(self):
        if self.VECTORIZED_FACTORS is None:
            self.VECTORIZED_FACTORS = ["pe_ttm", "momentum", "volatility", "net_inflow_ratio"]
        if self.FINANCIAL_FACTORS is None:
            self.FINANCIAL_FACTORS = ["roe", "growth_revenue_yoy", "debt_to_assets"]
        if self.LOOP_FACTORS is None:
            self.LOOP_FACTORS = [
                "holder_num_change_ratio", "major_shareholder_net_buy_ratio",
                "top_list_net_buy_amount", "dividend_yield", "forecast_growth_rate",
                "repurchase_ratio", "block_trade_ratio"
            ]


class FactorCalculator(ABC):
    """Abstract base class for factor calculators"""
    
    @abstractmethod
    def calculate(self, data: Dict[str, Any], config: FactorCalculationConfig) -> pd.Series:
        """Calculate factor values"""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Return list of required data keys"""
        pass


class VectorizedFactorCalculator(FactorCalculator):
    """Calculator for vectorized factors (price-based, cross-sectional)"""
    
    def calculate(self, data: Dict[str, Any], config: FactorCalculationConfig) -> Dict[str, pd.Series]:
        """Calculate all vectorized factors at once"""
        results = {}
        
        # Price-based factors
        if "daily_prices_df" in data:
            price_factors = self._calculate_price_factors(data["daily_prices_df"], config)
            results.update(price_factors)
        
        # Cross-sectional factors
        if "daily_basics_df" in data:
            cross_sectional_factors = self._calculate_cross_sectional_factors(
                data["daily_basics_df"], data.get("money_flow_df"), config
            )
            results.update(cross_sectional_factors)
        
        return results
    
    def _calculate_price_factors(self, prices_df: pd.DataFrame, config: FactorCalculationConfig) -> Dict[str, pd.Series]:
        """Calculate price-based factors efficiently"""
        results = {}
        
        if prices_df is None or prices_df.empty:
            return results
        
        prices = prices_df.ffill()
        
        # Momentum calculation
        if len(prices) >= config.MIN_TRADING_DAYS_FOR_MOMENTUM:
            results["momentum"] = (
                prices.iloc[-1] / prices.iloc[-config.MOMENTUM_LOOKBACK_DAYS] - 1
            )
            log.info(f"    ✓ 计算动量因子: {len(results['momentum'].dropna())} 只股票")
        
        # Volatility calculation
        if len(prices) >= config.MIN_TRADING_DAYS_FOR_VOLATILITY:
            log_returns = np.log(prices / prices.shift(1))
            results["volatility"] = (
                log_returns.iloc[-config.VOLATILITY_LOOKBACK_DAYS:].std() * 
                np.sqrt(config.TRADING_DAYS_PER_YEAR)
            )
            log.info(f"    ✓ 计算波动率因子: {len(results['volatility'].dropna())} 只股票")
        
        return results
    
    def _calculate_cross_sectional_factors(
        self, 
        basics_df: pd.DataFrame, 
        money_flow_df: Optional[pd.DataFrame],
        config: FactorCalculationConfig
    ) -> Dict[str, pd.Series]:
        """Calculate cross-sectional factors"""
        results = {}
        
        if basics_df is None or basics_df.empty:
            return results
        
        basics = basics_df.set_index("ts_code")
        results["pe_ttm"] = basics["pe_ttm"]
        log.info(f"    ✓ 提取PE_TTM因子: {len(results['pe_ttm'].dropna())} 只股票")
        
        # Money flow factors
        if money_flow_df is not None and not money_flow_df.empty:
            flow_factor = self._calculate_money_flow_factor(basics, money_flow_df)
            if flow_factor is not None:
                results["net_inflow_ratio"] = flow_factor
                log.info(f"    ✓ 计算资金流入比例: {len(results['net_inflow_ratio'].dropna())} 只股票")
        
        return results
    
    def _calculate_money_flow_factor(self, basics: pd.DataFrame, money_flow_df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate money flow factor safely"""
        try:
            flow = money_flow_df.drop(columns=["trade_date"]).set_index("ts_code")
            combined = basics.join(flow, how="inner")
            
            if combined.empty or "amount" not in combined.columns:
                return None
            
            amount_yuan = combined["amount"] * 1000
            net_inflow_yuan = (
                combined["buy_lg_amount"] - combined["sell_lg_amount"]
            ) * 10000
            
            return net_inflow_yuan.divide(amount_yuan).fillna(0)
        except Exception as e:
            log.warning(f"资金流向因子计算失败: {e}")
            return None
    
    def get_required_data(self) -> List[str]:
        return ["daily_prices_df", "daily_basics_df", "money_flow_df"]


class FinancialFactorCalculator(FactorCalculator):
    """Calculator for financial factors using pre-fetched data"""
    
    def calculate(self, data: Dict[str, Any], config: FactorCalculationConfig) -> Dict[str, pd.Series]:
        """Calculate financial factors from pre-fetched data"""
        results = {}
        
        if "all_fina_data" not in data or data["all_fina_data"] is None or data["all_fina_data"].empty:
            log.warning("无财务数据，跳过财务因子计算")
            return results
        
        try:
            # Apply PIT filtering with improved error handling
            pit_fina_data = self._apply_pit_filtering(data["all_fina_data"], data["trade_date"], data["dm"])
            
            if pit_fina_data.empty:
                log.warning("PIT筛选后无有效财务数据")
                return results
            
            pit_fina_data = pit_fina_data.set_index('ts_code')
            
            # Calculate financial factors
            factor_mappings = {
                "roe": "roe",
                "growth_revenue_yoy": "or_yoy",  # Tushare field mapping
                "debt_to_assets": "debt_to_assets"
            }
            
            for factor_name, field_name in factor_mappings.items():
                if field_name in pit_fina_data.columns:
                    results[factor_name] = pit_fina_data[field_name]
                    log.info(f"    ✓ 提取财务因子 {factor_name}: {len(results[factor_name].dropna())} 只股票")
                else:
                    log.warning(f"财务数据中缺少字段: {field_name}")
            
            return results
            
        except Exception as e:
            log.error(f"财务因子计算失败: {e}")
            return results
    
    def _apply_pit_filtering(self, all_fina_data: pd.DataFrame, trade_date: str, dm) -> pd.DataFrame:
        """Apply Point-in-Time filtering with better error handling"""
        try:
            # Use pandas groupby with proper error handling
            pit_fina_data = all_fina_data.groupby('ts_code', group_keys=False).apply(
                lambda x: dm.get_pit_financial_data(x, trade_date)
            ).reset_index(drop=True)
            
            log.info(f"    PIT筛选完成: {len(pit_fina_data)} 条有效财务记录")
            return pit_fina_data
            
        except Exception as e:
            log.error(f"PIT筛选失败: {e}")
            return pd.DataFrame()
    
    def get_required_data(self) -> List[str]:
        return ["all_fina_data", "trade_date", "dm"]


class LoopFactorCalculator(FactorCalculator):
    """Calculator for factors that require individual stock processing"""
    
    def calculate(self, data: Dict[str, Any], config: FactorCalculationConfig) -> Dict[str, pd.Series]:
        """Calculate factors that require loop processing"""
        results = {}
        
        if "ff" not in data or "ts_codes" not in data:
            log.warning("缺少因子工厂或股票代码，跳过循环因子计算")
            return results
        
        ff = data["ff"]
        ts_codes = data["ts_codes"]
        
        # Prepare parameters
        params = self._prepare_calculation_params(data)
        
        # Calculate each factor
        for factor in config.LOOP_FACTORS:
            log.info(f"    正在计算因子: {factor}...")
            factor_series = self._calculate_single_factor(ff, factor, ts_codes, params)
            if factor_series is not None:
                results[factor] = factor_series
                log.info(f"    ✓ 完成因子 {factor}: {len(factor_series.dropna())} 只股票")
        
        return results
    
    def _prepare_calculation_params(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for factor calculation"""
        from datetime import datetime, timedelta
        
        trade_date = data["trade_date"]
        trade_date_dt = pd.to_datetime(trade_date)
        
        return {
            "date": trade_date,
            "start_date": (trade_date_dt - timedelta(days=90)).strftime("%Y%m%d"),
            "end_date": trade_date,
            "top_list_df": data.get("top_list_df", pd.DataFrame()),
            "block_trade_df": data.get("block_trade_df", pd.DataFrame()),
        }
    
    def _calculate_single_factor(
        self, 
        ff, 
        factor: str, 
        ts_codes: List[str], 
        params: Dict[str, Any]
    ) -> Optional[pd.Series]:
        """Calculate a single factor for all stocks"""
        try:
            factor_series = pd.Series(index=ts_codes, dtype=float)
            
            for code in ts_codes:
                try:
                    params["ts_code"] = code
                    factor_series[code] = ff.calculate(factor, **params)
                except Exception as e:
                    log.debug(f"股票 {code} 的因子 {factor} 计算失败: {e}")
                    factor_series[code] = np.nan
            
            return factor_series
            
        except Exception as e:
            log.error(f"因子 {factor} 计算失败: {e}")
            return None
    
    def get_required_data(self) -> List[str]:
        return ["ff", "ts_codes", "trade_date", "top_list_df", "block_trade_df"]


class FactorCalculationOrchestrator:
    """Orchestrates the entire factor calculation process"""
    
    def __init__(self, config: Optional[FactorCalculationConfig] = None):
        self.config = config or FactorCalculationConfig()
        self.calculators = {
            "vectorized": VectorizedFactorCalculator(),
            "financial": FinancialFactorCalculator(),
            "loop": LoopFactorCalculator()
        }
    
    def calculate_all_factors(self, trade_date: str, raw_data: dict, all_fina_data: pd.DataFrame, dm, ff) -> Dict[str, pd.Series]:
        """Calculate all factors using appropriate calculators"""
        log.info("【因子计算】开始分层计算所有因子...")
        
        # Prepare data for calculators
        calculation_data = {
            **raw_data,
            "all_fina_data": all_fina_data,
            "trade_date": trade_date,
            "dm": dm,
            "ff": ff
        }
        
        all_results = {}
        
        # Execute calculations in order
        for calc_type, calculator in self.calculators.items():
            log.info(f"  执行{calc_type}因子计算...")
            try:
                results = calculator.calculate(calculation_data, self.config)
                all_results.update(results)
                log.info(f"  ✓ {calc_type}因子计算完成: {len(results)} 个因子")
            except Exception as e:
                log.error(f"  ✗ {calc_type}因子计算失败: {e}")
        
        log.info(f"【因子计算】总计算完成: {len(all_results)} 个因子")
        return all_results
    
    def save_factors_to_database(self, factors: Dict[str, pd.Series], trade_date: str, dm) -> bool:
        """Save calculated factors to database"""
        log.info("【数据入库】开始存储因子数据...")
        
        try:
            # Convert to long format
            final_df = pd.DataFrame(factors).reset_index().rename(columns={"index": "ts_code"})
            long_df = final_df.melt(
                id_vars="ts_code", 
                var_name="factor_name", 
                value_name="factor_value"
            ).dropna()
            long_df["trade_date"] = pd.to_datetime(trade_date)
            
            if long_df.empty:
                log.warning("没有有效的因子数据可以存入数据库")
                return False
            
            # Save to database with transaction
            with dm.engine.connect() as connection:
                with connection.begin():
                    # Delete existing data for the date
                    delete_sql = text("DELETE FROM factors_exposure WHERE trade_date = :trade_date")
                    connection.execute(delete_sql, {"trade_date": trade_date})
                    
                    # Insert new data
                    long_df.to_sql(
                        "factors_exposure",
                        connection,
                        if_exists="append",
                        index=False,
                        chunksize=self.config.DB_WRITE_CHUNK_SIZE,
                    )
            
            log.info(f"✓ 成功为 {trade_date} 写入 {len(long_df)} 条因子数据")
            return True
            
        except Exception as e:
            log.error(f"因子数据写入数据库失败: {e}")
            return False


# Updated main function
def calculate_and_save_factors_improved(trade_date: str, raw_data: dict, all_fina_data: pd.DataFrame):
    """
    Improved factor calculation and storage function
    """
    import data
    import quant_engine
    
    dm = data.DataManager()
    ff = quant_engine.FactorFactory(_data_manager=dm)
    
    # Use the orchestrator
    config = FactorCalculationConfig()
    orchestrator = FactorCalculationOrchestrator(config)
    
    # Calculate all factors
    factors = orchestrator.calculate_all_factors(trade_date, raw_data, all_fina_data, dm, ff)
    
    # Save to database
    success = orchestrator.save_factors_to_database(factors, trade_date, dm)
    
    if not success:
        raise Exception(f"Failed to save factors for {trade_date}")