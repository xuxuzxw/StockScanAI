"""
Builder pattern for flexible data extraction configuration
"""
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import pandas as pd
from logger_config import log


@dataclass
class DataExtractionConfig:
    """Configuration for data extraction"""
    trade_date: str
    lookback_days: int = 90
    min_trading_days: int = 55
    download_chunk_size: int = 150
    db_chunk_size: int = 500
    max_retries: int = 3
    
    # Data sources to extract
    extract_cross_sectional: bool = True
    extract_price_history: bool = True
    extract_financial_data: bool = True
    
    # Specific data types
    include_daily_basic: bool = True
    include_money_flow: bool = True
    include_top_list: bool = True
    include_block_trade: bool = True
    
    # Custom data sources
    custom_extractors: Dict[str, Callable] = field(default_factory=dict)


class DataExtractionBuilder:
    """Builder for configuring data extraction"""
    
    def __init__(self, trade_date: str):
        self.config = DataExtractionConfig(trade_date=trade_date)
    
    def with_lookback_days(self, days: int) -> 'DataExtractionBuilder':
        """Set lookback period for historical data"""
        self.config.lookback_days = days
        return self
    
    def with_chunk_sizes(self, download_chunk: int, db_chunk: int) -> 'DataExtractionBuilder':
        """Set chunk sizes for batch processing"""
        self.config.download_chunk_size = download_chunk
        self.config.db_chunk_size = db_chunk
        return self
    
    def with_retry_config(self, max_retries: int) -> 'DataExtractionBuilder':
        """Set retry configuration"""
        self.config.max_retries = max_retries
        return self
    
    def include_cross_sectional_data(self, include: bool = True) -> 'DataExtractionBuilder':
        """Include/exclude cross-sectional data"""
        self.config.extract_cross_sectional = include
        return self
    
    def include_price_history(self, include: bool = True) -> 'DataExtractionBuilder':
        """Include/exclude price history"""
        self.config.extract_price_history = include
        return self
    
    def include_financial_data(self, include: bool = True) -> 'DataExtractionBuilder':
        """Include/exclude financial data"""
        self.config.extract_financial_data = include
        return self
    
    def with_specific_data_types(
        self, 
        daily_basic: bool = True,
        money_flow: bool = True,
        top_list: bool = True,
        block_trade: bool = True
    ) -> 'DataExtractionBuilder':
        """Configure specific data types to extract"""
        self.config.include_daily_basic = daily_basic
        self.config.include_money_flow = money_flow
        self.config.include_top_list = top_list
        self.config.include_block_trade = block_trade
        return self
    
    def add_custom_extractor(self, name: str, extractor: Callable) -> 'DataExtractionBuilder':
        """Add custom data extractor"""
        self.config.custom_extractors[name] = extractor
        return self
    
    def build(self) -> DataExtractionConfig:
        """Build the final configuration"""
        return self.config


class DataExtractor:
    """Handles data extraction based on configuration"""
    
    def __init__(self, dm, config: DataExtractionConfig):
        self.dm = dm
        self.config = config
        self.results = {}
    
    def extract_all_data(self) -> Dict[str, Any]:
        """Extract all configured data"""
        log.info("=" * 60)
        log.info(f"📊 开始为 {self.config.trade_date} 抽取数据")
        log.info("=" * 60)
        
        # Get stock list
        stock_list = self.dm.get_stock_basic()
        ts_codes = stock_list["ts_code"].tolist()
        log.info(f"🎯 目标股票数量: {len(ts_codes)} 只")
        
        self.results = {
            "stock_list": stock_list,
            "ts_codes": ts_codes
        }
        
        # Extract different types of data based on configuration
        if self.config.extract_cross_sectional:
            self._extract_cross_sectional_data()
        
        if self.config.extract_price_history:
            self._extract_price_history(ts_codes)
        
        if self.config.extract_financial_data:
            self._extract_financial_data(ts_codes)
        
        # Extract custom data
        self._extract_custom_data()
        
        log.info("【数据抽取】所有配置的数据提取完毕")
        return self.results
    
    def _extract_cross_sectional_data(self):
        """Extract cross-sectional data for the trade date"""
        log.info("📈 获取当日截面数据...")
        
        data_sources = []
        
        if self.config.include_daily_basic:
            data_sources.append(("基本指标", lambda: self.dm.pro.daily_basic(trade_date=self.config.trade_date)))
        
        if self.config.include_money_flow:
            data_sources.append(("资金流向", lambda: self.dm.pro.moneyflow(trade_date=self.config.trade_date)))
        
        if self.config.include_top_list:
            data_sources.append(("龙虎榜", lambda: self.dm.pro.top_list(trade_date=self.config.trade_date)))
        
        if self.config.include_block_trade:
            data_sources.append(("大宗交易", lambda: self.dm.pro.block_trade(trade_date=self.config.trade_date)))
        
        # Extract data with error handling
        for name, func in data_sources:
            try:
                log.info(f"  📊 获取{name}数据...")
                result_data = func()
                count = len(result_data) if result_data is not None and not result_data.empty else 0
                log.info(f"  ✅ {name}: {count} 条记录")
                
                # Store with standardized names
                key_mapping = {
                    "基本指标": "daily_basics_df",
                    "资金流向": "money_flow_df",
                    "龙虎榜": "top_list_df",
                    "大宗交易": "block_trade_df"
                }
                self.results[key_mapping[name]] = result_data if result_data is not None else pd.DataFrame()
                
            except Exception as e:
                log.warning(f"  ⚠️  {name}获取失败: {e}")
                key_mapping = {
                    "基本指标": "daily_basics_df",
                    "资金流向": "money_flow_df",
                    "龙虎榜": "top_list_df",
                    "大宗交易": "block_trade_df"
                }
                self.results[key_mapping[name]] = pd.DataFrame()
    
    def _extract_price_history(self, ts_codes: List[str]):
        """Extract price history with caching optimization"""
        log.info("💰 获取历史价格数据 (缓存优先)...")
        
        from datetime import timedelta
        start_date_lookback = (
            pd.to_datetime(self.config.trade_date) - timedelta(days=self.config.lookback_days)
        ).strftime("%Y%m%d")
        
        prices_dict = {}
        
        # Check cache
        cached_stocks = self._check_price_cache(ts_codes, start_date_lookback)
        
        # Download missing data
        stocks_to_download = sorted(list(set(ts_codes) - cached_stocks))
        if stocks_to_download:
            downloaded_data = self._download_price_data(stocks_to_download, start_date_lookback)
            prices_dict.update(downloaded_data)
        
        # Load cached data
        if cached_stocks:
            cached_data = self._load_cached_price_data(list(cached_stocks), start_date_lookback)
            prices_dict.update(cached_data)
        
        self.results["daily_prices_df"] = pd.DataFrame(prices_dict)
    
    def _check_price_cache(self, ts_codes: List[str], start_date: str) -> set:
        """Check which stocks have sufficient cached data"""
        try:
            from sqlalchemy import text
            query = text("""
                SELECT ts_code FROM ts_daily
                WHERE trade_date BETWEEN TO_DATE(:start_date, 'YYYYMMDD') AND TO_DATE(:end_date, 'YYYYMMDD') 
                  AND ts_code = ANY(:ts_codes)
                GROUP BY ts_code 
                HAVING COUNT(trade_date) >= :min_days
            """)
            
            with self.dm.engine.connect() as conn:
                result = conn.execute(query, {
                    "start_date": start_date,
                    "end_date": self.config.trade_date,
                    "ts_codes": ts_codes,
                    "min_days": self.config.min_trading_days
                }).fetchall()
            
            cached_stocks = {row[0] for row in result}
            log.info(f"    缓存检查: {len(cached_stocks)}/{len(ts_codes)} 只股票已有完整数据")
            return cached_stocks
            
        except Exception as e:
            log.warning(f"    缓存检查失败: {e}")
            return set()
    
    def _download_price_data(self, stocks_to_download: List[str], start_date: str) -> Dict[str, pd.Series]:
        """Download price data for stocks not in cache"""
        log.info(f"    下载 {len(stocks_to_download)} 只股票的价格数据...")
        
        downloaded_data = {}
        
        for attempt in range(self.config.max_retries + 1):
            needed = [
                code for code in stocks_to_download
                if code not in downloaded_data or downloaded_data[code] is None
            ]
            
            if not needed:
                break
            
            if attempt > 0:
                log.warning(f"    第 {attempt} 次重试，下载剩余 {len(needed)} 只股票...")
            
            # Download in chunks
            chunk_size = self.config.download_chunk_size
            for i in range(0, len(needed), chunk_size):
                chunk = needed[i:i + chunk_size]
                log.info(f"      下载块 {i//chunk_size + 1}/{(len(needed)-1)//chunk_size + 1}")
                
                chunk_results = self.dm.run_batch_download(chunk, start_date, self.config.trade_date)
                
                # Process results
                for code, df in chunk_results.items():
                    if df is not None and not df.empty:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        downloaded_data[code] = df.set_index("trade_date")["close"]
        
        return downloaded_data
    
    def _load_cached_price_data(self, cached_stocks: List[str], start_date: str) -> Dict[str, pd.Series]:
        """Load cached price data from database"""
        log.info(f"    从数据库加载 {len(cached_stocks)} 只股票的缓存数据...")
        
        cached_data = {}
        chunk_size = self.config.db_chunk_size
        
        for i in range(0, len(cached_stocks), chunk_size):
            chunk = cached_stocks[i:i + chunk_size]
            
            from sqlalchemy import text
            query = text("""
                SELECT ts_code, trade_date, close FROM ts_daily
                WHERE trade_date BETWEEN :start_date AND :end_date AND ts_code = ANY(:ts_codes)
            """)
            
            with self.dm.engine.connect() as conn:
                cached_df = pd.read_sql(query, conn, params={
                    "start_date": start_date,
                    "end_date": self.config.trade_date,
                    "ts_codes": chunk
                })
            
            if not cached_df.empty:
                cached_df['trade_date'] = pd.to_datetime(cached_df['trade_date'])
                for code, group in cached_df.groupby('ts_code'):
                    cached_data[code] = group.set_index('trade_date')['close']
        
        return cached_data
    
    def _extract_financial_data(self, ts_codes: List[str]):
        """Extract financial data for all stocks"""
        log.info("📊 批量预取财务数据...")
        
        all_fina_data_list = []
        
        for i, code in enumerate(ts_codes):
            if i % 100 == 0:
                log.info(f"    处理进度: {i}/{len(ts_codes)} 只股票")
            
            try:
                df_fina = self.dm.get_fina_indicator(ts_code=code, force_update=False)
                if df_fina is not None and not df_fina.empty:
                    all_fina_data_list.append(df_fina)
            except Exception as e:
                log.debug(f"    股票 {code} 财务数据获取失败: {e}")
        
        if all_fina_data_list:
            all_fina_data = pd.concat(all_fina_data_list).drop_duplicates(subset=['ts_code', 'end_date'])
            log.info(f"    财务数据预取完成: {len(all_fina_data)} 条记录")
        else:
            all_fina_data = pd.DataFrame()
            log.warning("    未能获取任何财务数据")
        
        self.results["all_fina_data"] = all_fina_data
    
    def _extract_custom_data(self):
        """Extract custom data using provided extractors"""
        if not self.config.custom_extractors:
            return
        
        log.info("🔧 执行自定义数据提取...")
        
        for name, extractor in self.config.custom_extractors.items():
            try:
                log.info(f"    执行自定义提取器: {name}")
                result = extractor(self.dm, self.config.trade_date)
                self.results[f"custom_{name}"] = result
                log.info(f"    ✅ 自定义提取器 {name} 完成")
            except Exception as e:
                log.error(f"    ✗ 自定义提取器 {name} 失败: {e}")
                self.results[f"custom_{name}"] = None


# Usage example
def create_extraction_config(trade_date: str, mode: str = "full") -> DataExtractionConfig:
    """Factory function to create common extraction configurations"""
    builder = DataExtractionBuilder(trade_date)
    
    if mode == "minimal":
        # Only basic data for quick processing
        config = (builder
                 .include_cross_sectional_data(True)
                 .include_price_history(False)
                 .include_financial_data(False)
                 .with_specific_data_types(daily_basic=True, money_flow=False, top_list=False, block_trade=False)
                 .build())
    
    elif mode == "price_only":
        # Only price data for technical analysis
        config = (builder
                 .include_cross_sectional_data(False)
                 .include_price_history(True)
                 .include_financial_data(False)
                 .with_lookback_days(252)  # One year of data
                 .build())
    
    elif mode == "full":
        # Complete data extraction
        config = (builder
                 .include_cross_sectional_data(True)
                 .include_price_history(True)
                 .include_financial_data(True)
                 .with_lookback_days(90)
                 .with_chunk_sizes(150, 500)
                 .with_retry_config(3)
                 .build())
    
    else:
        raise ValueError(f"Unknown extraction mode: {mode}")
    
    return config