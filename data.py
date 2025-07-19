# quant_project/data.py
import tushare as ts
import akshare as ak
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
from logger_config import log

# --- 装饰器：用于缓存和API频率控制 ---
def api_rate_limit(calls_per_minute=200):
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

    def __init__(self, token=config.TUSHARE_TOKEN, db_url=config.DATABASE_URL, concurrency_limit=4, requests_per_minute=50):
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

    def _upsert_data(self, table_name: str, df: pd.DataFrame, primary_keys: list[str]):
        """
        【V2.1重构】通用的数据插入或更新（UPSERT）方法。
        利用 PostgreSQL 的 ON CONFLICT 特性高效写入数据。
        """
        if df is None or df.empty:
            return

        from sqlalchemy.dialects.postgresql import insert

        with self.engine.connect() as connection:
            with connection.begin():
                for index, row in df.iterrows():
                    # 将Pandas的row转换为字典
                    row_dict = row.to_dict()
                    # 构造 insert 语句
                    stmt = insert(sqlalchemy.table(table_name, *[sqlalchemy.column(c) for c in df.columns])) \
                        .values(**row_dict)
                    
                    # 定义 ON CONFLICT DO UPDATE 的逻辑
                    # 更新除主键外的所有列
                    update_cols = {col.name: col for col in stmt.excluded if col.name not in primary_keys}
                    if update_cols:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=primary_keys,
                            set_=update_cols
                        )
                    else: # 如果没有其他列可以更新，则只执行 DO NOTHING
                         stmt = stmt.on_conflict_do_nothing(index_elements=primary_keys)

                    # 执行语句
                    connection.execute(stmt)

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

                # --- V2.1 新增表 ---
                # 【V2.1新增】股东人数表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS stk_holdernumber (
                        ts_code TEXT NOT NULL,
                        ann_date DATE,
                        end_date DATE NOT NULL,
                        holder_num BIGINT,
                        PRIMARY KEY (ts_code, end_date)
                    );
                """))
                # 【V2.1新增】重要股东增减持表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS stk_holdertrade (
                        ann_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        holder_name TEXT,
                        holder_type TEXT,
                        in_de TEXT,
                        change_vol NUMERIC,
                        change_ratio NUMERIC,
                        after_share NUMERIC,
                        after_ratio NUMERIC,
                        avg_price NUMERIC,
                        total_share NUMERIC,
                        -- 复合主键，防止同一天同一股东的多笔交易记录重复
                        PRIMARY KEY (ann_date, ts_code, holder_name, change_vol)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('stk_holdertrade', 'ann_date', if_not_exists => TRUE);"))
                # 【V2.1新增】龙虎榜每日详情表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS top_list (
                        trade_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        name TEXT,
                        close NUMERIC,
                        pct_change NUMERIC,
                        turnover_rate NUMERIC,
                        amount NUMERIC,
                        l_sell NUMERIC,
                        l_buy NUMERIC,
                        l_amount NUMERIC,
                        net_amount NUMERIC,
                        net_rate NUMERIC,
                        amount_rate NUMERIC,
                        float_values NUMERIC,
                        reason TEXT,
                        PRIMARY KEY (trade_date, ts_code, reason)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('top_list', 'trade_date', if_not_exists => TRUE);"))

                # 【V2.1新增】业绩快报表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS express (
                        ts_code TEXT NOT NULL,
                        ann_date DATE,
                        end_date DATE NOT NULL,
                        revenue NUMERIC,
                        operate_profit NUMERIC,
                        total_profit NUMERIC,
                        n_income NUMERIC,
                        total_assets NUMERIC,
                        total_hldr_eqy_exc_min_int NUMERIC,
                        diluted_eps NUMERIC,
                        diluted_roe NUMERIC,
                        yoy_op NUMERIC,
                        yoy_gr NUMERIC,
                        yoy_net_profit NUMERIC,
                        bps NUMERIC,
                        perf_summary TEXT,
                        update_flag TEXT,
                        PRIMARY KEY (ts_code, end_date)
                    );
                """))

                # 【V2.2 新增】统一的财务指标主表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS financial_indicators (
                        ts_code TEXT NOT NULL,
                        ann_date DATE,
                        end_date DATE NOT NULL,
                        -- 这里只列举部分核心字段作为示例，实际字段会很多
                        roe NUMERIC,
                        netprofit_yoy NUMERIC,
                        debt_to_assets NUMERIC,
                        or_yoy NUMERIC,
                        -- 考虑到财务指标字段繁多，未来可以采用JSONB类型存储以增加灵活性
                        -- 但当前为保持结构清晰，仍采用列式存储
                        PRIMARY KEY (ts_code, end_date)
                    );
                """))
                # 【V2.1新增】分红送股表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS financial_dividend (
                        ts_code TEXT NOT NULL,
                        end_date DATE NOT NULL,
                        ann_date DATE,
                        div_proc TEXT,
                        stk_div NUMERIC,
                        stk_bo_rate NUMERIC,
                        stk_co_rate NUMERIC,
                        cash_div NUMERIC,
                        cash_div_tax NUMERIC,
                        record_date DATE,
                        ex_date DATE,
                        pay_date DATE,
                        div_listdate DATE,
                        imp_ann_date DATE,
                        PRIMARY KEY (ts_code, end_date)
                    );
                """))
                # 【V2.1新增】股票回购表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS financial_repurchase (
                        ts_code TEXT NOT NULL,
                        ann_date DATE NOT NULL,
                        end_date DATE,
                        proc TEXT,
                        exp_date DATE,
                        vol NUMERIC,
                        amount NUMERIC,
                        high_limit NUMERIC,
                        low_limit NUMERIC,
                        PRIMARY KEY (ts_code, ann_date)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('financial_repurchase', 'ann_date', if_not_exists => TRUE);"))

                # 【V2.1新增】大宗交易表
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS block_trade (
                        trade_date DATE NOT NULL,
                        ts_code TEXT NOT NULL,
                        price NUMERIC,
                        vol NUMERIC,
                        amount NUMERIC,
                        buyer TEXT,
                        seller TEXT,
                        PRIMARY KEY (trade_date, ts_code, amount, buyer, seller)
                    );
                """))
                connection.execute(sqlalchemy.text("SELECT create_hypertable('block_trade', 'trade_date', if_not_exists => TRUE);"))

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
    
    @lru_cache(maxsize=1)
    def get_industry_index_map(self):
        """【V2.4新增】获取申万一级行业名称到指数代码的映射"""
        try:
            # Tushare的申万行业分类 level='L1', src='SW'
            df = self.pro.index_classify(level='L1', src='SW')
            if df is not None and not df.empty:
                return df.set_index('industry_name')['index_code'].to_dict()
        except Exception as e:
            log.error(f"获取行业指数映射失败: {e}")
        return {}

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
        
        # 【鲁棒性修复】确保 trade_date 列是 datetime 类型以便正确合并
        df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
        df_adj['trade_date'] = pd.to_datetime(df_adj['trade_date'])

        df = pd.merge(df_daily, df_adj, on='trade_date')
        if df.empty:
            return None # 如果合并后为空，直接返回
            
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
    def get_fina_indicator(self, ts_code: str, force_update=False):
        """【V2.2重构完成】获取单只股票的财务指标，适配统一大表 financial_indicators。"""
        table_name = 'financial_indicators'
        primary_keys = ['ts_code', 'end_date']

        # 强制更新或首次获取时，从API获取
        if force_update:
            df_api = self.pro.fina_indicator(ts_code=ts_code)
            if df_api is not None and not df_api.empty:
                # 为了防止列不匹配，只选择我们在新表中定义的列
                # 注意：这是一个简化处理，实际生产中需要一个完整的列映射
                db_cols = ['ts_code', 'ann_date', 'end_date', 'roe', 'netprofit_yoy', 'debt_to_assets', 'or_yoy']
                api_cols_to_keep = [col for col in db_cols if col in df_api.columns]
                self._upsert_data(table_name, df_api[api_cols_to_keep], primary_keys)

        # 无论如何，都从数据库返回该股票的完整历史数据
        try:
            query = sqlalchemy.text(f"SELECT * FROM {table_name} WHERE ts_code = :ts_code")
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query, connection, params={'ts_code': ts_code})
            if not df_db.empty:
                return df_db.sort_values(by='end_date', ascending=False).reset_index(drop=True)
            else: # 如果数据库为空，则触发一次API更新
                df_api = self.pro.fina_indicator(ts_code=ts_code)
                if df_api is not None and not df_api.empty:
                    db_cols = ['ts_code', 'ann_date', 'end_date', 'roe', 'netprofit_yoy', 'debt_to_assets', 'or_yoy']
                    api_cols_to_keep = [col for col in db_cols if col in df_api.columns]
                    self._upsert_data(table_name, df_api[api_cols_to_keep], primary_keys)
                    return df_api[api_cols_to_keep].sort_values(by='end_date', ascending=False).reset_index(drop=True)
                
        except Exception as e:
            # log.error(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return pd.DataFrame()
        
        return pd.DataFrame()


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

    @api_rate_limit(calls_per_minute=100)
    def get_holder_number(self, ts_code: str, force_update=False):
        """【V2.1重构-效率】获取股东人数，使用UPSERT逻辑高效缓存。"""
        table_name = 'stk_holdernumber'
        primary_keys = ['ts_code', 'end_date']
        
        # 即使非强制更新，也从API获取一次，以抓取最新的数据
        # 因为API不支持增量查询，但我们的UPSERT逻辑可以处理增量写入
        df_api = self.pro.stk_holdernumber(ts_code=ts_code)
        if df_api is not None and not df_api.empty:
            self._upsert_data(table_name, df_api, primary_keys)

        # 无论如何，最后都从数据库返回最完整的数据
        try:
            query = sqlalchemy.text(f"SELECT * FROM {table_name} WHERE ts_code = :ts_code")
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query, connection, params={'ts_code': ts_code})
            return df_db.sort_values(by='end_date').reset_index(drop=True)
        except Exception as e:
            print(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return pd.DataFrame()

    @api_rate_limit()
    def get_holder_trade(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取重要股东增减持数据，带缓存。"""
        # 此接口数据量较小，可以简化缓存逻辑：优先从API获取，然后补充写入。
        table_name = 'stk_holdertrade'
        df_api = self.pro.stk_holdertrade(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df_api is not None and not df_api.empty:
            try:
                with self.engine.connect() as connection:
                    with connection.begin():
                        # to_sql 配合复合主键，重复数据将因违反主键约束而无法插入
                        df_api.to_sql(table_name, connection, if_exists='append', index=False)
            except sqlalchemy.exc.IntegrityError:
                # 这是一个预期的行为，当尝试插入重复数据时发生，可以安全地忽略
                pass
            except Exception as e:
                print(f"写入 {table_name} for {ts_code} 失败: {e}")
        return df_api

    @api_rate_limit()
    def get_top_list(self, trade_date: str = None, start_date: str = None, end_date: str = None, force_update=False):
        """【V2.1修正】获取龙虎榜数据，支持单日或多日查询，带缓存。"""
        table_name = 'top_list'

        if trade_date:
            start_date = end_date = trade_date

        if start_date and end_date:
            if not force_update:
                try:
                    query = sqlalchemy.text("SELECT * FROM top_list WHERE trade_date BETWEEN :start AND :end")
                    with self.engine.connect() as connection:
                        df_db = pd.read_sql(query, connection, params={'start': start_date, 'end': end_date})
                    if not df_db.empty:
                        return df_db
                except Exception as e:
                    print(f"从 {table_name} 读取 {start_date}-{end_date} 数据失败: {e}。将从API获取。")
            
            # 按日循环获取
            all_dfs = []
            trade_dates = self.pro.trade_cal(start_date=start_date, end_date=end_date)
            open_dates = trade_dates[trade_dates['is_open'] == 1]['cal_date'].tolist()

            for date in open_dates:
                df_api = self.pro.top_list(trade_date=date)
                if df_api is not None and not df_api.empty:
                    all_dfs.append(df_api)
            
            if not all_dfs:
                return pd.DataFrame()

            final_df = pd.concat(all_dfs)
            # 写入数据库
            with self.engine.connect() as connection:
                with connection.begin():
                    # 这里简化为追加，依赖主键防止重复。更稳妥的方式是先删后写。
                    final_df.to_sql(table_name, connection, if_exists='append', index=False)
            return final_df
        
        # 原有的单日逻辑
        if not force_update:
            try:
                query = sqlalchemy.text("SELECT * FROM top_list WHERE trade_date = :trade_date")
                with self.engine.connect() as connection:
                    df_db = pd.read_sql(query, connection, params={'trade_date': trade_date})
                if not df_db.empty:
                    return df_db
            except Exception as e:
                print(f"从 {table_name} 读取 {trade_date} 数据失败: {e}。将从API获取。")

        df_api = self.pro.top_list(trade_date=trade_date)
        if df_api is not None and not df_api.empty:
            with self.engine.connect() as connection:
                with connection.begin():
                    delete_sql = sqlalchemy.text("DELETE FROM top_list WHERE trade_date = :trade_date")
                    connection.execute(delete_sql, {'trade_date': trade_date})
                    df_api.to_sql(table_name, connection, if_exists='append', index=False)
        return df_api

    @api_rate_limit()
    def get_forecast(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取业绩预告数据，带缓存。"""
        table_name = 'financial_forecast'
        # 此接口数据量较小，可以简化缓存逻辑：优先从API获取，然后补充写入。
        df_api = self.pro.forecast(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df_api is not None and not df_api.empty:
            try:
                with self.engine.connect() as connection:
                    with connection.begin():
                        # to_sql 配合复合主键，重复数据将因违反主键约束而无法插入
                        df_api.to_sql(table_name, connection, if_exists='append', index=False)
            except sqlalchemy.exc.IntegrityError:
                # 这是一个预期的行为，当尝试插入重复数据时发生，可以安全地忽略
                pass
            except Exception as e:
                print(f"写入 {table_name} for {ts_code} 失败: {e}")
        return df_api

    @api_rate_limit()
    def get_dividend(self, ts_code: str, force_update=False):
        """【V2.1重构-效率】获取分红送股数据，使用UPSERT逻辑高效缓存。"""
        table_name = 'financial_dividend'
        primary_keys = ['ts_code', 'end_date']
        
        df_api = self.pro.dividend(ts_code=ts_code)
        if df_api is not None and not df_api.empty:
             self._upsert_data(table_name, df_api, primary_keys)

        # 无论如何，最后都从数据库返回最完整的数据
        try:
            query = sqlalchemy.text(f"SELECT * FROM {table_name} WHERE ts_code = :ts_code")
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query, connection, params={'ts_code': ts_code})
            return df_db.sort_values(by='end_date').reset_index(drop=True)
        except Exception as e:
            print(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return pd.DataFrame()

    @api_rate_limit()
    def get_repurchase(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取股票回购数据，带缓存。"""
        table_name = 'financial_repurchase'
        df_api = self.pro.repurchase(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df_api is not None and not df_api.empty:
            try:
                with self.engine.connect() as connection:
                    with connection.begin():
                        # to_sql 配合复合主键，重复数据将因违反主键约束而无法插入
                        df_api.to_sql(table_name, connection, if_exists='append', index=False)
            except sqlalchemy.exc.IntegrityError:
                # 这是一个预期的行为，当尝试插入重复数据时发生，可以安全地忽略
                pass
            except Exception as e:
                print(f"写入 {table_name} for {ts_code} 失败: {e}")
        return df_api

    @api_rate_limit()
    def get_express(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取业绩快报数据，带缓存。"""
        table_name = 'express'
        df_api = self.pro.express(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df_api is not None and not df_api.empty:
            self._upsert_data(table_name, df_api, ['ts_code', 'end_date'])
        
        # 从数据库读取返回，确保数据完整性
        try:
            query = sqlalchemy.text(f"SELECT * FROM {table_name} WHERE ts_code = :ts_code AND end_date BETWEEN :start AND :end")
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query, connection, params={'ts_code': ts_code, 'start': start_date, 'end': end_date})
            return df_db
        except Exception as e:
            print(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return df_api # 如果读取失败，返回刚从API获取的数据

    @api_rate_limit()
    def get_block_trade(self, trade_date: str = None, start_date: str = None, end_date: str = None):
        """【V2.1修正】获取大宗交易数据，支持单日或多日查询，使用UPSERT缓存。"""
        table_name = 'block_trade'
        
        if trade_date:
            start_date = end_date = trade_date

        if start_date and end_date:
            # 按日循环获取
            all_dfs = []
            trade_dates = self.pro.trade_cal(start_date=start_date, end_date=end_date)
            open_dates = trade_dates[trade_dates['is_open'] == 1]['cal_date'].tolist()

            for date in open_dates:
                df_api_daily = self.pro.block_trade(trade_date=date)
                if df_api_daily is not None and not df_api_daily.empty:
                    all_dfs.append(df_api_daily)
            
            if not all_dfs:
                return pd.DataFrame()

            final_df = pd.concat(all_dfs)
            self._upsert_data(table_name, final_df, ['trade_date', 'ts_code', 'amount', 'buyer', 'seller'])
            return final_df

        return pd.DataFrame()


    # --- (五) 宏观经济 (同步) ---
    @api_rate_limit()
    def get_cn_m(self, start_m, end_m, force_update=False):
        return self._fetch_and_cache(self.pro.cn_m, 'macro_cn_m', force_update=force_update, start_m=start_m, end_m=end_m)

    @api_rate_limit()
    def get_cn_pmi(self, start_m, end_m, force_update=False):
        return self._fetch_and_cache(self.pro.cn_pmi, 'macro_cn_pmi', force_update=force_update, start_m=start_m, end_m=end_m)
    
    @api_rate_limit()
    def get_cn_cpi(self, start_m: str = None, end_m: str = None, force_update=False):
        """【V2.1激活】获取全国居民消费价格指数CPI，使用通用缓存"""
        return self._fetch_and_cache(self.pro.cn_cpi, 'macro_cn_cpi', force_update=force_update, start_m=start_m, end_m=end_m)

    @api_rate_limit()
    def get_cn_gdp(self, start_q: str = None, end_q: str = None, force_update=False):
        """【V2.1新增】获取季度GDP数据，使用通用缓存"""
        return self._fetch_and_cache(self.pro.cn_gdp, 'macro_cn_gdp', force_update=force_update, start_q=start_q, end_q=end_q)

    @api_rate_limit()
    def get_shibor(self, start_date: str = None, end_date: str = None, force_update=False):
        """【V2.1激活】获取Shibor同业拆放利率，使用通用缓存"""
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
    
    def purge_old_ai_reports(self, ts_code: str, keep_latest: int = 10):
        """【V2.1新增】清理指定股票的旧AI报告，仅保留最新的N份。"""
        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    # 使用窗口函数和子查询来确定要删除的旧报告
                    # 这是一个健壮的、在各种SQL方言中都有效的方法
                    delete_sql = sqlalchemy.text(f"""
                        DELETE FROM ai_reports
                        WHERE ctid IN (
                            SELECT ctid FROM (
                                SELECT ctid, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                                FROM ai_reports
                                WHERE ts_code = :ts_code
                            ) as sub
                            WHERE sub.rn > :keep_latest
                        )
                    """)
                    result = connection.execute(delete_sql, {'ts_code': ts_code, 'keep_latest': keep_latest})
                    if result.rowcount > 0:
                        log.info(f"为 {ts_code} 清理了 {result.rowcount} 份旧的AI报告。")
        except Exception as e:
            log.error(f"清理 {ts_code} 的AI报告时出错: {e}", exc_info=True)

    def __del__(self):
        """关闭数据库连接和异步会话"""
        # SQLAlchemy 引擎通常不需要手动关闭，连接池会管理
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._close_async_session())
        except (RuntimeError, AttributeError):
            pass