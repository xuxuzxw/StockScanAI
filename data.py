# quant_project/data.py
import tushare as ts
import pandas as pd
import sqlalchemy
from sqlalchemy.exc import ProgrammingError
import os
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import time
import asyncio
import aiohttp
import json

import config

# --- 装饰器：用于缓存和API频率控制 ---
def api_rate_limit(calls_per_minute=500):
    """Tushare API调用频率控制装饰器"""
    min_interval = 60.0 / calls_per_minute
    last_call_time = {'value': 0}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_time['value']
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_call_time['value'] = time.time()
            return result
        return wrapper
    return decorator

class DataManager:
    """
    【V2.0 数据库升级版】智能数据中心 (Data Hub)
    - 全面覆盖Tushare核心接口。
    - 内置复权逻辑，提供精确的价格数据。
    - 基于 TimescaleDB 的高性能时序数据库存储。
    - 集成了异步数据引擎，支持高并发数据下载。
    """
    _API_URL = 'http://api.tushare.pro'

    def __init__(self, token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL, concurrency_limit=20, requests_per_minute=400):
        if not token or token == "YOUR_TUSHARE_TOKEN":
            raise ValueError("Tushare Token未在config.py中配置。")
        self.pro = ts.pro_api(token)
        self.token = token # For async calls
        
        if not db_url:
            raise ValueError("数据库连接URL未在config.py中配置。")
            
        self.engine = sqlalchemy.create_engine(db_url)
        self._initialize_db()

        # 【修正】异步组件，增加速率控制
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.session = None
        # 根据每分钟请求数，计算出每次请求间的最小延迟（秒）
        # 留出一些安全边际，所以用400而不是800
        self.request_delay = 60.0 / requests_per_minute

    def _initialize_db(self):
        """【V2.0核心改造】初始化TimescaleDB，创建Hypertable"""
        with self.engine.begin() as connection:
            try:
                # 步骤1: 启用TimescaleDB扩展
                connection.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                
                # 步骤2: 创建一个通用的时序表（以日线行情为例）
                # 注意：Pandas to_sql 会自动创建表，但我们需要预先定义主键和索引以获得最佳性能
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS ts_daily (
                        trade_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        open NUMERIC,
                        high NUMERIC,
                        low NUMERIC,
                        close NUMERIC,
                        pre_close NUMERIC,
                        "change" NUMERIC,
                        pct_chg NUMERIC,
                        vol BIGINT,
                        amount NUMERIC,
                        PRIMARY KEY (trade_date, ts_code)
                    );
                """))
                
                # 步骤3: 将普通表转化为Hypertable（TimescaleDB的核心）
                # 只有在表是新创建的情况下才需要执行
                connection.execute(sqlalchemy.text("SELECT create_hypertable('ts_daily', 'trade_date', if_not_exists => TRUE);"))

                # 【增强】为复权因子表（ts_adj_factor）创建表结构和Hypertable
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS ts_adj_factor (
                        trade_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        adj_factor NUMERIC,
                        PRIMARY KEY (trade_date, ts_code)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('ts_adj_factor', 'trade_date', if_not_exists => TRUE);"))

                # 【修正版】为非结构化文本数据创建表，主键符合TimescaleDB要求
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS text_corpus (
                        publish_date DATE NOT NULL,
                        content_hash VARCHAR(64) NOT NULL,
                        ts_code TEXT,
                        title TEXT,
                        source VARCHAR(255),
                        content TEXT,
                        PRIMARY KEY (publish_date, content_hash)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('text_corpus', 'publish_date', if_not_exists => TRUE);"))

                # 【新增】为预计算的因子暴露值创建表结构和Hypertable
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS factors_exposure (
                        trade_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        factor_name VARCHAR(50) NOT NULL,
                        factor_value NUMERIC,
                        PRIMARY KEY (trade_date, ts_code, factor_name)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('factors_exposure', 'trade_date', if_not_exists => TRUE);"))

                # 【新增】为AI分析报告创建缓存表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS ai_reports (
                        trade_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        report_content TEXT,
                        model_used VARCHAR(50),
                        estimated_cost NUMERIC,
                        PRIMARY KEY (trade_date, ts_code)
                    );
                """))
                # 这张表也可以从Hypertable中受益，便于按时间管理
                connection.execute(sqlalchemy.text("SELECT create_hypertable('ai_reports', 'trade_date', if_not_exists => TRUE);"))

            except ProgrammingError as e:
                print(f"数据库初始化时发生已知错误（可能已初始化），可忽略: {e}")
            except Exception as e:
                print(f"数据库初始化失败: {e}")
                raise e
        print("TimescaleDB 初始化检查完成。")

    @property
    def conn(self):
        # 让 conn 成为一个属性，每次访问时都从引擎获取一个新的连接
        # 这比在__init__中创建一个长期连接更健壮
        return self.engine.connect()

    def _fetch_and_cache(self, api_func, table_name, force_update=False, **kwargs):
        """通用数据获取与缓存逻辑"""
        inspector = sqlalchemy.inspect(self.engine)
        table_exists = inspector.has_table(table_name)

        if not force_update:
            try:
                if table_exists:
                    # 使用 with 语句确保连接被正确关闭
                    with self.engine.connect() as connection:
                        df = pd.read_sql(f'SELECT * FROM {table_name}', connection)
                    if not df.empty:
                        return df
            except Exception:
                pass
        
        df_api = api_func(**kwargs)
        if df_api is not None and not df_api.empty:
            with self.engine.connect() as connection:
                # 使用事务来确保写入的原子性
                with connection.begin():
                    df_api.to_sql(table_name, connection, if_exists='replace', index=False)
        return df_api

    # --- (一) 核心基础数据 (同步) ---
    @api_rate_limit()
    def get_stock_basic(self, list_status='L', force_update=False):
        return self._fetch_and_cache(self.pro.stock_basic, 'stock_basic', force_update=force_update, list_status=list_status)
    
    def _fetch_and_cache_timeseries(self, table_name, date_column, api_func, ts_code, start_date, end_date, **kwargs):
        """
        【V2.0修正】为时序数据设计的增量更新缓存逻辑 (适配SQLAlchemy)。
        """
        max_date_in_db_str = None
        df_db = pd.DataFrame()
        # 使用SQLAlchemy的Inspector来检查表是否存在
        inspector = sqlalchemy.inspect(self.engine)
        table_exists = inspector.has_table(table_name)

        # 1. 如果表已存在，则查询现有数据，确定增量更新的起始点
        if table_exists:
            # 使用参数化查询防止SQL注入
            query = sqlalchemy.text(f"""
                SELECT * FROM {table_name} 
                WHERE ts_code = :ts_code AND {date_column} >= :start_date AND {date_column} <= :end_date
            """)
            try:
                with self.engine.connect() as connection:
                    df_db = pd.read_sql(query, connection, params={'ts_code': ts_code, 'start_date': start_date, 'end_date': end_date})
                if not df_db.empty:
                    # 需要确保日期列是字符串格式或可以被正确比较
                    if pd.api.types.is_datetime64_any_dtype(df_db[date_column]):
                         max_date_in_db = df_db[date_column].max()
                         max_date_in_db_str = max_date_in_db.strftime('%Y%m%d')
                    else:
                         max_date_in_db_str = df_db[date_column].max()

            except Exception as e:
                print(f"读取表 {table_name} 失败: {e}。将尝试全量更新。")
                df_db = pd.DataFrame()

        # 2. 确定需要从API下载的日期范围
        fetch_start_date = start_date
        if max_date_in_db_str:
            # 从数据库最新日期的下一天开始获取
            fetch_start_date = (pd.to_datetime(max_date_in_db_str) + pd.Timedelta(days=1)).strftime('%Y%m%d')

        # 3. 如果需要，从API下载新数据
        if fetch_start_date <= end_date:
            df_api = api_func(ts_code=ts_code, start_date=fetch_start_date, end_date=end_date, **kwargs)
            
            if df_api is not None and not df_api.empty:
                try:
                    # PostgreSQL 支持 ON CONFLICT (UPSERT)，但 to_sql 不直接支持。
                    # 'append' 是最简单的方式，依靠主键来防止重复。
                    # 我们在 _initialize_db 中为 ts_daily 设置了主键 (trade_date, ts_code)。
                    # 对于没有主键的表，需要先创建。
                    with self.engine.connect() as connection:
                        with connection.begin(): # 开启事务
                            df_api.to_sql(table_name, connection, if_exists='append', index=False)
                    
                    # 合并数据库中已有的数据和新获取的数据
                    if not df_db.empty:
                        df_db = pd.concat([df_db, df_api]).drop_duplicates(subset=['ts_code', date_column], keep='last')
                    else:
                        df_db = df_api
                except Exception as e:
                    print(f"写入数据库失败: {e}, 将重新读取全量数据。")
                    # 如果写入失败，回退到重新读取全量
                    with self.engine.connect() as connection:
                         df_db = pd.read_sql(f"SELECT * FROM {table_name} WHERE ts_code = '{ts_code}'", connection)

        # 4. 排序并返回最终结果
        if not df_db.empty:
            # 筛选出最终需要的日期范围
            df_db[date_column] = pd.to_datetime(df_db[date_column])
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            df_db = df_db[(df_db[date_column] >= start_date_dt) & (df_db[date_column] <= end_date_dt)]
            df_db = df_db.sort_values(by=date_column).reset_index(drop=True)
        return df_db

    @api_rate_limit()
    def get_daily(self, ts_code, start_date, end_date):
        """【优化】获取日线行情，已集成增量缓存"""
        return self._fetch_and_cache_timeseries(
            table_name='ts_daily',
            date_column='trade_date',
            api_func=self.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

    @api_rate_limit()
    def get_index_daily(self, ts_code, start_date, end_date):
        """【新增】获取指数日线行情，已集成增量缓存"""
        # 注意：指数行情与个股行情使用相同的表(ts_daily)，因为字段兼容
        return self._fetch_and_cache_timeseries(
            table_name='ts_daily',
            date_column='trade_date',
            api_func=self.pro.index_daily, # 调用Tushare的指数日线接口
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

    @api_rate_limit()
    def get_adj_factor(self, ts_code, start_date, end_date):
        """【优化】获取复权因子，已集成增量缓存"""
        return self._fetch_and_cache_timeseries(
            table_name='ts_adj_factor',
            date_column='trade_date',
            api_func=self.pro.adj_factor,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )    
    @lru_cache(maxsize=128)
    def get_adjusted_daily(self, ts_code, start_date, end_date, adj='hfq'):
        """
        【核心工具】获取复权日线行情。
        自动处理日线和复权因子的合并与计算。
        :param adj: 'qfq' (前复权), 'hfq' (后复权)
        """
        df_daily = self.get_daily(ts_code, start_date, end_date)
        df_adj = self.get_adj_factor(ts_code, start_date, end_date)
        
        if df_daily is None or df_daily.empty or df_adj is None or df_adj.empty:
            return None
        
        # 确保 trade_date 列是 object/string 类型以便合并
        df_daily['trade_date'] = df_daily['trade_date'].astype(str)
        df_adj['trade_date'] = df_adj['trade_date'].astype(str)

        df = pd.merge(df_daily, df_adj, on='trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)

        if adj == 'qfq':
            # 后复权因子是向前累乘的，所以最后一个是基准。前复权则相反。
            # 此处应使用最后一个因子值作为基准来计算前复权因子
            if not df.empty:
                last_factor = df['adj_factor'].iloc[-1]
                df['adj_factor'] = df['adj_factor'] / last_factor
        
        # 确保价格列是数值类型
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in price_cols:
            df[col] = df[col] * df['adj_factor']
        
        return df.drop(columns=['adj_factor'])

    # --- (二) 日度行情与指标 (同步) ---
    @api_rate_limit()
    def get_daily_basic(self, ts_code, start_date, end_date):
        return self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)

    @api_rate_limit()
    def get_moneyflow(self, ts_code, start_date, end_date):
        return self.pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)

    # --- (三) 财务与基本面 (同步) ---
    @api_rate_limit()
    def get_fina_indicator(self, ts_code, force_update=False):
        table_name = f"fina_indicator_{ts_code.replace('.', '_')}"
        return self._fetch_and_cache(self.pro.fina_indicator, table_name, force_update=force_update, ts_code=ts_code)

    @api_rate_limit()
    def get_income(self, ts_code, period):
        return self.pro.income(ts_code=ts_code, period=period, report_type='1')

    @api_rate_limit()
    def get_balancesheet(self, ts_code, period):
        return self.pro.balancesheet(ts_code=ts_code, period=period, report_type='1')

    @api_rate_limit()
    def get_cashflow(self, ts_code, period):
        return self.pro.cashflow(ts_code=ts_code, period=period, report_type='1')

    # --- (四) 资金流与筹码 (同步) ---
    @api_rate_limit()
    def get_hk_hold(self, ts_code, start_date, end_date):
        return self.pro.hk_hold(ts_code=ts_code, start_date=start_date, end_date=end_date)

    @api_rate_limit()
    def get_top_list(self, trade_date):
        return self.pro.top_list(trade_date=trade_date)

    @api_rate_limit()
    def get_margin_detail(self, ts_code, start_date, end_date):
        return self.pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date=end_date)

    @api_rate_limit(calls_per_minute=100)
    def get_top10_floatholders(self, ts_code, period):
        return self.pro.top10_floatholders(ts_code=ts_code, period=period)

    # --- (五) 宏观经济 (同步) ---
    @api_rate_limit()
    def get_cn_m(self, start_m, end_m, force_update=False):
        return self._fetch_and_cache(self.pro.cn_m, 'macro_cn_m', force_update=force_update, start_m=start_m, end_m=end_m)

    @api_rate_limit()
    def get_cn_pmi(self, start_m, end_m, force_update=False):
        return self._fetch_and_cache(self.pro.cn_pmi, 'macro_cn_pmi', force_update=force_update, start_m=start_m, end_m=end_m)
    
    @api_rate_limit()
    def get_cn_cpi(self, start_m, end_m, force_update=False):
        """【V2.0新增】获取全国居民消费价格指数CPI"""
        return self._fetch_and_cache(self.pro.cn_cpi, 'macro_cn_cpi', force_update=force_update, start_m=start_m, end_m=end_m)

    @api_rate_limit()
    def get_shibor(self, start_date, end_date, force_update=False):
        """【V2.0新增】获取Shibor同业拆放利率"""
        return self._fetch_and_cache(self.pro.shibor, 'macro_shibor', force_update=force_update, start_date=start_date, end_date=end_date)

    # --- (六) 异步数据获取模块 ---
    async def _get_async_session(self):
        """获取或创建 aiohttp Session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _close_async_session(self):
        """关闭 aiohttp Session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _fetch_async(self, api_name: str, **kwargs) -> pd.DataFrame:
        """核心的异步数据获取方法"""
        session = await self._get_async_session()
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": kwargs,
            "fields": []
        }
        
        async with self.semaphore:
            await asyncio.sleep(self.request_delay)
            try:
                async with session.post(self._API_URL, data=json.dumps(payload)) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    
                    if resp_json['code'] != 0:
                        raise ConnectionError(f"Tushare API Error: {resp_json['msg']}")
                    
                    data = resp_json['data']
                    columns = data['fields']
                    items = data['items']
                    
                    return pd.DataFrame(items, columns=columns)
            except aiohttp.ClientError as e:
                print(f"Async request for {api_name} failed: {e}")
                return pd.DataFrame()

    async def _batch_get_daily_async(self, ts_codes: list, start_date: str, end_date: str) -> dict:
        """并发获取多支股票的日线行情"""
        tasks = [self._fetch_async('daily', ts_code=code, start_date=start_date, end_date=end_date) for code in ts_codes]
        results_list = await asyncio.gather(*tasks)
        return {code: df for code, df in zip(ts_codes, results_list)}

    def run_batch_download(self, ts_codes: list, start_date: str, end_date: str) -> dict:
        """同步方法来运行异步批量下载任务"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(self._batch_get_daily_async(ts_codes, start_date, end_date))
        loop.run_until_complete(self._close_async_session())
        return results
        
    # --- (七) 非结构化数据处理 ---
    def update_text_corpus(self, ts_code):
        if ts_code not in config.SIMULATED_NEWS_SOURCE: return
        news_list = config.SIMULATED_NEWS_SOURCE[ts_code]
        with self.engine.connect() as connection:
            with connection.begin(): # 开启事务
                for news in news_list:
                    content_hash = hash(news['content'])
                    # PostgreSQL 使用 %s作为参数占位符
                    res = connection.execute(sqlalchemy.text("SELECT doc_id FROM text_corpus WHERE content_hash = :chash"), {'chash': str(content_hash)}).fetchone()
                    if res is None:
                        connection.execute(sqlalchemy.text('''
                            INSERT INTO text_corpus (ts_code, publish_date, title, source, content, content_hash)
                            VALUES (:tsc, :pd, :ti, :so, :co, :ch)
                        '''), {
                            'tsc': ts_code, 'pd': datetime.now().strftime('%Y%m%d'), 'ti': news['title'],
                            'so': 'Simulated Source', 'co': news['content'], 'ch': str(content_hash)
                        })

    def get_text_for_analysis(self, ts_code, date_range=None, limit=3):
        self.update_text_corpus(ts_code)
        query = f"SELECT title, content FROM text_corpus WHERE ts_code = '{ts_code}' ORDER BY publish_date DESC LIMIT {limit}"
        with self.engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df

    def get_pit_financial_data(self, all_financial_data: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        """
        【新增】过滤财务数据以确保其在时间点（Point-in-Time, PIT）上是正确的。
        防止"未来函数"或"前视偏差"(lookahead bias)。
        """
        if 'ann_date' not in all_financial_data.columns or 'end_date' not in all_financial_data.columns:
            # log.warning("财务数据必须包含 'ann_date' 和 'end_date' 列才能进行PIT正确性处理。")
            return None
        
        # 确保日期列是datetime类型
        all_financial_data['ann_date'] = pd.to_datetime(all_financial_data['ann_date'])
        as_of_date_dt = pd.to_datetime(as_of_date)

        # 过滤出公告日在指定日期之前的数据
        available_data = all_financial_data[all_financial_data['ann_date'] <= as_of_date_dt].copy()
        
        if available_data.empty:
            return None
        
        # 按报告期（end_date）降序排序，获取最新的那份财报
        available_data.sort_values(by='end_date', ascending=False, inplace=True)
        
        return available_data.head(1)
    
    def __del__(self):
        """关闭数据库连接和异步会话"""
        # SQLAlchemy 引擎通常不需要手动关闭，连接池会管理
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._close_async_session())
        except (RuntimeError, AttributeError):
            pass