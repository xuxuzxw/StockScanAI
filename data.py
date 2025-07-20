# StockScanAI/data.py
import asyncio
import json
import time
from datetime import datetime
from functools import lru_cache, wraps

import aiohttp
import akshare as ak
import pandas as pd
import sqlalchemy
import tushare as ts
from sqlalchemy.exc import ProgrammingError

import config
from logger_config import log

# --- 装饰器：用于缓存和API频率控制 ---

# 用于存储每个API最后调用时间戳的全局字典
_last_call_times = {}


def api_rate_limit(api_key: str):
    """
    【V2.5 重构版】Tushare API 精细化调用频率控制装饰器。
    - 能根据 config.py 中 API_RATE_LIMITS 的配置，为不同接口应用不同的速率限制。
    - 每个接口有独立的计时器，互不影响。
    """
    # 从配置中获取限制，如果未定义则使用默认值
    calls_per_minute = config.API_RATE_LIMITS.get(
        api_key, config.API_RATE_LIMITS.get("default", 400)
    )
    min_interval = 60.0 / calls_per_minute

    def decorator(func):
        # 初始化该API的最后调用时间
        if func.__name__ not in _last_call_times:
            _last_call_times[func.__name__] = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取当前时间
            current_time = time.time()
            # 计算距离上次调用的时间差
            elapsed = current_time - _last_call_times[func.__name__]

            # 如果时间间隔不够，则等待
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                log.debug(
                    f"API '{func.__name__}' 触发限速 (规则: {api_key}, 上限: {calls_per_minute}/min)，等待 {sleep_time:.4f} 秒..."
                )
                time.sleep(sleep_time)

            # 执行实际的函数调用
            result = func(*args, **kwargs)

            # 更新该API的最后调用时间
            _last_call_times[func.__name__] = time.time()

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

    _API_URL = "http://api.tushare.pro"

    def __init__(
        self,
        token=config.TUSHARE_TOKEN,
        db_url=config.DATABASE_URL,
        # 【V2.9 优化】从config.py读取异步下载的默认配置，不再硬编码
        concurrency_limit=config.ASYNC_DOWNLOAD_CONFIG.get("concurrency_limit", 5),
        requests_per_minute=config.ASYNC_DOWNLOAD_CONFIG.get(
            "requests_per_minute", 400
        ),
    ):
        if not token or token == "YOUR_TUSHARE_TOKEN":
            raise ValueError("Tushare Token未在config.py中配置。")
        self.pro = ts.pro_api(token)
        self.token = token  # For async calls

        if not db_url:
            raise ValueError("数据库连接URL未在config.py中配置。")

        self.engine = sqlalchemy.create_engine(db_url)
        self._initialize_db()

        # 【V2.9.1 修正】异步组件，增加速率控制
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.session = None
        # 根据每分钟请求数和并发数，计算出每次请求间的最小延迟（秒）
        # 速率(RPS) = 并发数 / 延迟时间
        # 目标速率(RPS) = 每分钟请求数 / 60
        # -> 延迟时间 = (并发数 * 60) / 每分钟请求数
        self.request_delay = (60.0 * concurrency_limit) / requests_per_minute

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
                    stmt = insert(
                        sqlalchemy.table(
                            table_name, *[sqlalchemy.column(c) for c in df.columns]
                        )
                    ).values(**row_dict)

                    # 定义 ON CONFLICT DO UPDATE 的逻辑
                    # 更新除主键外的所有列
                    update_cols = {
                        col.name: col
                        for col in stmt.excluded
                        if col.name not in primary_keys
                    }
                    if update_cols:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=primary_keys, set_=update_cols
                        )
                    else:  # 如果没有其他列可以更新，则只执行 DO NOTHING
                        stmt = stmt.on_conflict_do_nothing(index_elements=primary_keys)

                    # 执行语句
                    connection.execute(stmt)

    def _initialize_db(self):
        """【V2.0核心改造】初始化TimescaleDB，创建Hypertable - 增强版"""
        log.info("开始初始化数据库...")
        
        with self.engine.begin() as connection:
            try:
                self._setup_timescale_extension(connection)
                self._create_core_tables(connection)
                self._create_additional_tables(connection)
                
            except ProgrammingError as e:
                log.warning(f"数据库初始化时发生已知错误（可能已初始化），可忽略: {e}")
            except Exception as e:
                log.error(f"数据库初始化失败: {e}")
                raise
        
        log.info("TimescaleDB 初始化检查完成。")

    def _setup_timescale_extension(self, connection):
        """设置TimescaleDB扩展"""
        log.info("  检查TimescaleDB扩展...")
        try:
            connection.execute(
                sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            )
            log.info("  ✓ TimescaleDB扩展已启用")
            return True
        except Exception as e:
            log.error(f"  ✗ TimescaleDB扩展启用失败: {e}")
            log.warning("  将使用普通PostgreSQL表结构继续...")
            return False

    def _create_table_with_hypertable(self, connection, table_config):
        """创建表并可选地转换为Hypertable"""
        table_name = table_config["name"]
        log.info(f"  创建表: {table_name}")
        
        try:
            connection.execute(sqlalchemy.text(table_config["sql"]))
            log.info(f"  ✓ 表 {table_name} 创建成功")
            
            if table_config.get("is_hypertable", False):
                self._create_hypertable(connection, table_name, table_config.get("time_column", "trade_date"))
                
        except Exception as e:
            log.error(f"  ✗ 表 {table_name} 创建失败: {e}")
            raise

    def _create_hypertable(self, connection, table_name, time_column):
        """将表转换为Hypertable"""
        try:
            connection.execute(
                sqlalchemy.text(
                    f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);"
                )
            )
            log.info(f"  ✓ {table_name} 已转换为Hypertable")
        except Exception as e:
            log.warning(f"  ⚠ {table_name} Hypertable转换失败，使用普通表: {e}")

    def _get_core_table_definitions(self):
        """获取核心表定义"""
        return [
            {
                "name": "ts_daily",
                "sql": """
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
                """,
                "is_hypertable": True,
                "time_column": "trade_date"
            },
            {
                "name": "stock_basic",
                "sql": """
                CREATE TABLE IF NOT EXISTS stock_basic (
                    ts_code TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    area TEXT,
                    industry TEXT,
                    market TEXT,
                    list_date DATE,
                    list_status TEXT,
                    is_hs TEXT
                );
                """,
                "is_hypertable": False
            }
        ]

    def _create_core_tables(self, connection):
        """创建核心表结构"""
        log.info("  创建核心表结构...")
        for table_config in self._get_core_table_definitions():
            self._create_table_with_hypertable(connection, table_config)

    def _create_additional_tables(self, connection):
        """创建其他业务表"""
        log.info("  创建其他业务表...")
        
        additional_tables = self._get_additional_table_definitions()
        for table_config in additional_tables:
            self._create_table_with_hypertable(connection, table_config)

    def _get_additional_table_definitions(self):
        """获取其他业务表定义"""
        return [
            {
                "name": "ts_adj_factor",
                "sql": """
                CREATE TABLE IF NOT EXISTS ts_adj_factor (
                    trade_date DATE NOT NULL,
                    ts_code TEXT NOT NULL,
                    adj_factor NUMERIC,
                    PRIMARY KEY (trade_date, ts_code)
                );
                """,
                "is_hypertable": True,
                "time_column": "trade_date"
            },
            {
                "name": "text_corpus",
                "sql": """
                CREATE TABLE IF NOT EXISTS text_corpus (
                    publish_date DATE NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    ts_code TEXT,
                    title TEXT,
                    source VARCHAR(255),
                    content TEXT,
                    PRIMARY KEY (publish_date, content_hash)
                );
                """,
                "is_hypertable": True,
                "time_column": "publish_date"
            },
            {
                "name": "factors_exposure",
                "sql": """
                CREATE TABLE IF NOT EXISTS factors_exposure (
                    trade_date DATE NOT NULL,
                    ts_code TEXT NOT NULL,
                    factor_name VARCHAR(50) NOT NULL,
                    factor_value NUMERIC,
                    PRIMARY KEY (trade_date, ts_code, factor_name)
                );
                """,
                "is_hypertable": True,
                "time_column": "trade_date"
            },
            {
                "name": "ai_reports",
                "sql": """
                CREATE TABLE IF NOT EXISTS ai_reports (
                    trade_date DATE NOT NULL,
                    ts_code TEXT NOT NULL,
                    report_content TEXT,
                    model_used VARCHAR(50),
                    estimated_cost NUMERIC,
                    PRIMARY KEY (trade_date, ts_code)
                );
                """,
                "is_hypertable": True,
                "time_column": "trade_date"
            },
            {
                "name": "stk_holdernumber",
                "sql": """
                CREATE TABLE IF NOT EXISTS stk_holdernumber (
                    ts_code TEXT NOT NULL,
                    ann_date DATE,
                    end_date DATE NOT NULL,
                    holder_num BIGINT,
                    PRIMARY KEY (ts_code, end_date)
                );
                """,
                "is_hypertable": False
            },
            {
                "name": "stk_holdertrade",
                "sql": """
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
                    PRIMARY KEY (ann_date, ts_code, holder_name, change_vol)
                );
                """,
                "is_hypertable": True,
                "time_column": "ann_date"
            },
            {
                "name": "top_list",
                "sql": """
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
                """,
                "is_hypertable": True,
                "time_column": "trade_date"
            },
            {
                "name": "express",
                "sql": """
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
                """,
                "is_hypertable": False
            },
            {
                "name": "financial_income",
                "sql": """
                CREATE TABLE IF NOT EXISTS financial_income (
                    ts_code TEXT NOT NULL,
                    ann_date DATE,
                    f_ann_date DATE,
                    end_date DATE NOT NULL,
                    report_type TEXT,
                    comp_type TEXT,
                    basic_eps NUMERIC,
                    diluted_eps NUMERIC,
                    total_revenue NUMERIC,
                    n_income NUMERIC,
                    total_profit NUMERIC,
                    PRIMARY KEY (ts_code, end_date, report_type)
                );
                """,
                "is_hypertable": False
            },
            {
                "name": "financial_balancesheet",
                "sql": """
                CREATE TABLE IF NOT EXISTS financial_balancesheet (
                    ts_code TEXT NOT NULL,
                    ann_date DATE,
                    f_ann_date DATE,
                    end_date DATE NOT NULL,
                    report_type TEXT,
                    comp_type TEXT,
                    total_assets NUMERIC,
                    total_liab NUMERIC,
                    total_hldr_eqy_inc_min_int NUMERIC,
                    PRIMARY KEY (ts_code, end_date, report_type)
                );
                """,
                "is_hypertable": False
            },
            {
                "name": "financial_cashflow",
                "sql": """
                CREATE TABLE IF NOT EXISTS financial_cashflow (
                    ts_code TEXT NOT NULL,
                    ann_date DATE,
                    f_ann_date DATE,
                    end_date DATE NOT NULL,
                    report_type TEXT,
                    comp_type TEXT,
                    n_cashflow_act NUMERIC,
                    n_add_capital NUMERIC,
                    PRIMARY KEY (ts_code, end_date, report_type)
                );
                """,
                "is_hypertable": False
            },
            {
                "name": "top10_floatholders",
                "sql": """
                CREATE TABLE IF NOT EXISTS top10_floatholders (
                    ts_code TEXT NOT NULL,
                    ann_date DATE,
                    end_date DATE NOT NULL,
                    holder_name TEXT,
                    hold_ratio NUMERIC,
                    hold_amount NUMERIC,
                    PRIMARY KEY (ts_code, end_date, holder_name)
                );
                """,
                "is_hypertable": False
            },
            {
                "name": "financial_indicators",
                "sql": """
                CREATE TABLE IF NOT EXISTS financial_indicators (
                    ts_code TEXT NOT NULL,
                    ann_date DATE,
                    end_date DATE NOT NULL,
                    roe NUMERIC,
                    netprofit_yoy NUMERIC,
                    debt_to_assets NUMERIC,
                    or_yoy NUMERIC,
                    PRIMARY KEY (ts_code, end_date)
                );
                """,
                "is_hypertable": False
            },
            {
                "name": "financial_dividend",
                "sql": """
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
                """,
                "is_hypertable": False
            },
            {
                "name": "financial_repurchase",
                "sql": """
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
                """,
                "is_hypertable": True,
                "time_column": "ann_date"
            },
            {
                "name": "block_trade",
                "sql": """
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
                """,
                "is_hypertable": True,
                "time_column": "trade_date"
            }
        ]

    @property
    def conn(self):
        # 让 conn 成为一个属性，每次访问时都从引擎获取一个新的连接
        # 这比在__init__中创建一个长期连接更健壮
        return self.engine.connect()

    def _fetch_and_cache(self, api_func, table_name, force_update=False, frequency="daily", **kwargs):
        """
        通用数据获取与缓存逻辑。
        新增 frequency 参数，用于判断数据的“新鲜度”。
        """
        inspector = sqlalchemy.inspect(self.engine)
        table_exists = inspector.has_table(table_name)

        if not force_update:
            try:
                if table_exists:
                    with self.engine.connect() as connection:
                        # 【V2.9.1 修正】为 stock_basic 表增加特殊处理，避免因缺少日期列而报错
                        if table_name == "stock_basic":
                            df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
                            if not df.empty:
                                log.info(f"数据表 '{table_name}' 已存在，直接从数据库返回。如需更新请使用 force_update=True。")
                                return df

                        # 尝试获取最新一条数据的时间戳
                        query_latest = sqlalchemy.text(f"SELECT MAX(trade_date) FROM {table_name}")
                        if table_name == "macro_cn_gdp":  # GDP表没有trade_date，用quarter
                            query_latest = sqlalchemy.text(f"SELECT MAX(quarter) FROM {table_name}")
                        elif table_name in ["macro_cn_m", "macro_cn_cpi", "macro_cn_pmi"]:  # 月度数据
                            query_latest = sqlalchemy.text(f"SELECT MAX(month) FROM {table_name}")

                        latest_date_in_db = connection.execute(query_latest).scalar_one_or_none()

                    if latest_date_in_db:
                        is_fresh = False
                        current_date = datetime.now()

                        if frequency == "daily":
                            if pd.to_datetime(latest_date_in_db).date() == current_date.date():
                                is_fresh = True
                        elif frequency == "monthly":
                            # 假设 monthly 数据在每月15号之后更新
                            latest_month_in_db = pd.to_datetime(str(latest_date_in_db) + "01").month
                            latest_year_in_db = pd.to_datetime(str(latest_date_in_db) + "01").year
                            if (current_date.year == latest_year_in_db and current_date.month == latest_month_in_db and current_date.day <= 15) or \
                               (current_date.year == latest_year_in_db and current_date.month > latest_month_in_db) or \
                               (current_date.year > latest_year_in_db):
                                is_fresh = False # 已过更新时间，需要更新
                            else:
                                is_fresh = True # 还在当月15号之前，数据新鲜

                        elif frequency == "quarterly":
                            # 假设季度数据在季末后45天内更新
                            latest_quarter_end = pd.to_datetime(str(latest_date_in_db) + "01") # 转换为日期
                            if latest_quarter_end.month == 3: # Q1
                                next_report_due = latest_quarter_end.replace(month=5, day=15)
                            elif latest_quarter_end.month == 6: # Q2
                                next_report_due = latest_quarter_end.replace(month=8, day=15)
                            elif latest_quarter_end.month == 9: # Q3
                                next_report_due = latest_quarter_end.replace(month=11, day=15)
                            elif latest_quarter_end.month == 12: # Q4
                                next_report_due = latest_quarter_end.replace(year=latest_quarter_end.year + 1, month=3, day=31)
                            else:
                                next_report_due = current_date # 无法判断，强制更新

                            if current_date < next_report_due:
                                is_fresh = True

                        if is_fresh:
                            log.info(f"数据库中 {table_name} 表的最新数据 ({latest_date_in_db}) 仍是新鲜的，直接从数据库返回。")
                            df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
                            if not df.empty:
                                return df

            except Exception as e:
                log.warning(f"从 {table_name} 读取最新数据失败: {e}。将尝试从API获取。")

        df_api = api_func(**kwargs)
        if df_api is not None and not df_api.empty:
            with self.engine.connect() as connection:
                with connection.begin():
                    df_api.to_sql(
                        table_name, connection, if_exists="replace", index=False
                    )
        return df_api

    # --- (一) 核心基础数据 (同步) ---
    @api_rate_limit("stock_basic")
    def get_stock_basic(self, list_status="L", force_update=False):
        return self._fetch_and_cache(
            self.pro.stock_basic,
            "stock_basic",
            force_update=force_update,
            list_status=list_status,
        )

    @lru_cache(maxsize=1)
    def get_industry_index_map(self):
        """【V2.4新增】获取申万一级行业名称到指数代码的映射"""
        try:
            # Tushare的申万行业分类 level='L1', src='SW'
            df = self.pro.index_classify(level="L1", src="SW")
            if df is not None and not df.empty:
                return df.set_index("industry_name")["index_code"].to_dict()
        except Exception as e:
            log.error(f"获取行业指数映射失败: {e}")
        return {}

    def _fetch_and_cache_timeseries(
        self, table_name, date_column, api_func, ts_code, start_date, end_date, **kwargs
    ):
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
            query = sqlalchemy.text(
                f"""
                SELECT * FROM {table_name}
                WHERE ts_code = :ts_code AND {date_column} >= :start_date AND {date_column} <= :end_date
            """
            )
            try:
                with self.engine.connect() as connection:
                    df_db = pd.read_sql(
                        query,
                        connection,
                        params={
                            "ts_code": ts_code,
                            "start_date": start_date,
                            "end_date": end_date,
                        },
                    )
                if not df_db.empty:
                    # 需要确保日期列是字符串格式或可以被正确比较
                    if pd.api.types.is_datetime64_any_dtype(df_db[date_column]):
                        max_date_in_db = df_db[date_column].max()
                        max_date_in_db_str = max_date_in_db.strftime("%Y%m%d")
                    else:
                        max_date_in_db_str = df_db[date_column].max()

            except Exception as e:
                print(f"读取表 {table_name} 失败: {e}。将尝试全量更新。")
                df_db = pd.DataFrame()

        # 2. 确定需要从API下载的日期范围
        fetch_start_date = start_date
        if max_date_in_db_str:
            # 从数据库最新日期的下一天开始获取
            fetch_start_date = (
                pd.to_datetime(max_date_in_db_str) + pd.Timedelta(days=1)
            ).strftime("%Y%m%d")

        # 3. 如果需要，从API下载新数据
        if fetch_start_date <= end_date:
            df_api = api_func(
                ts_code=ts_code,
                start_date=fetch_start_date,
                end_date=end_date,
                **kwargs,
            )

            if df_api is not None and not df_api.empty:
                try:
                    # PostgreSQL 支持 ON CONFLICT (UPSERT)，但 to_sql 不直接支持。
                    # 'append' 是最简单的方式，依靠主键来防止重复。
                    # 我们在 _initialize_db 中为 ts_daily 设置了主键 (trade_date, ts_code)。
                    # 对于没有主键的表，需要先创建。
                    with self.engine.connect() as connection:
                        with connection.begin():  # 开启事务
                            df_api.to_sql(
                                table_name, connection, if_exists="append", index=False
                            )

                    # 合并数据库中已有的数据和新获取的数据
                    if not df_db.empty:
                        df_db = pd.concat([df_db, df_api]).drop_duplicates(
                            subset=["ts_code", date_column], keep="last"
                        )
                    else:
                        df_db = df_api
                except Exception as e:
                    print(f"写入数据库失败: {e}, 将重新读取全量数据。")
                    # 如果写入失败，回退到重新读取全量
                    with self.engine.connect() as connection:
                        df_db = pd.read_sql(
                            f"SELECT * FROM {table_name} WHERE ts_code = '{ts_code}'",
                            connection,
                        )

        # 4. 排序并返回最终结果
        if not df_db.empty:
            # 筛选出最终需要的日期范围
            df_db[date_column] = pd.to_datetime(df_db[date_column])
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            df_db = df_db[
                (df_db[date_column] >= start_date_dt)
                & (df_db[date_column] <= end_date_dt)
            ]
            df_db = df_db.sort_values(by=date_column).reset_index(drop=True)
        return df_db

    def _fetch_and_cache_quarterly(
        self, table_name, api_func, ts_code, report_type="1", **kwargs
    ):
        """
        【V2.5新增】为季度财务数据设计的增量更新缓存逻辑。
        - 检查数据库中是否存在最新季度数据。
        - 如果存在且未到下一个报告期，则直接从数据库返回。
        - 否则，从API获取数据并UPSERT到数据库。
        """
        inspector = sqlalchemy.inspect(self.engine)
        table_exists = inspector.has_table(table_name)
        df_db = pd.DataFrame()

        # 1. 尝试从数据库获取最新数据
        if table_exists:
            try:
                query = sqlalchemy.text(
                    f"""
                    SELECT * FROM {table_name}
                    WHERE ts_code = :ts_code AND report_type = :report_type
                    ORDER BY end_date DESC
                    LIMIT 1
                """
                )
                with self.engine.connect() as connection:
                    latest_in_db = pd.read_sql(
                        query,
                        connection,
                        params={"ts_code": ts_code, "report_type": report_type},
                    )

                if not latest_in_db.empty:
                    latest_end_date_db = pd.to_datetime(latest_in_db["end_date"].iloc[0])
                    current_date = datetime.now()

                    # 判断是否已到下一个报告期
                    # 假设报告期是 3.31, 6.30, 9.30, 12.31
                    # 如果当前日期在下一个报告期之前，则认为数据库数据是新鲜的
                    is_fresh = False
                    if latest_end_date_db.month == 3 and current_date.month <= 6:
                        is_fresh = True
                    elif latest_end_date_db.month == 6 and current_date.month <= 9:
                        is_fresh = True
                    elif latest_end_date_db.month == 9 and current_date.month <= 12:
                        is_fresh = True
                    elif latest_end_date_db.month == 12 and current_date.month <= 3:
                        is_fresh = True

                    if is_fresh:
                        log.info(
                            f"数据库中 {table_name} 表 {ts_code} 的最新季度数据 ({latest_end_date_db.strftime('%Y%m%d')}) 仍是新鲜的，直接从数据库返回。"
                        )
                        # 从数据库获取所有历史数据
                        query_all = sqlalchemy.text(
                            f"""
                            SELECT * FROM {table_name}
                            WHERE ts_code = :ts_code AND report_type = :report_type
                            ORDER BY end_date DESC
                        """
                        )
                        with self.engine.connect() as connection:
                            df_db = pd.read_sql(
                                query_all,
                                connection,
                                params={"ts_code": ts_code, "report_type": report_type},
                            )
                        return df_db

            except Exception as e:
                log.warning(f"从 {table_name} 读取最新季度数据失败: {e}，将尝试从API获取。")

        # 2. 从API获取数据并UPSERT
        log.info(f"从API获取 {table_name} 表 {ts_code} 的季度数据...")
        df_api = api_func(ts_code=ts_code, report_type=report_type, **kwargs)

        if df_api is not None and not df_api.empty:
            # 确保 end_date 和 ann_date 是日期格式
            if 'end_date' in df_api.columns:
                df_api['end_date'] = pd.to_datetime(df_api['end_date'], errors='coerce').dt.strftime('%Y%m%d')
            if 'ann_date' in df_api.columns:
                df_api['ann_date'] = pd.to_datetime(df_api['ann_date'], errors='coerce').dt.strftime('%Y%m%d')

            self._upsert_data(table_name, df_api, ["ts_code", "end_date", "report_type"])
            # 重新从数据库读取所有数据以确保完整性
            query_all = sqlalchemy.text(
                f"""
                SELECT * FROM {table_name}
                WHERE ts_code = :ts_code AND report_type = :report_type
                ORDER BY end_date DESC
            """
            )
            with self.engine.connect() as connection:
                df_db = pd.read_sql(
                    query_all,
                    connection,
                    params={"ts_code": ts_code, "report_type": report_type},
                )
            return df_db
        return pd.DataFrame()

    @api_rate_limit("daily")
    def get_daily(self, ts_code, start_date, end_date):
        """【优化】获取日线行情，已集成增量缓存"""
        return self._fetch_and_cache_timeseries(
            table_name="ts_daily",
            date_column="trade_date",
            api_func=self.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    @api_rate_limit("index_daily")
    def get_index_daily(self, ts_code, start_date, end_date):
        """【新增】获取指数日线行情，已集成增量缓存"""
        # 注意：指数行情与个股行情使用相同的表(ts_daily)，因为字段兼容
        return self._fetch_and_cache_timeseries(
            table_name="ts_daily",
            date_column="trade_date",
            api_func=self.pro.index_daily,  # 调用Tushare的指数日线接口
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    @api_rate_limit("adj_factor")
    def get_adj_factor(self, ts_code, start_date, end_date):
        """【优化】获取复权因子，已集成增量缓存"""
        return self._fetch_and_cache_timeseries(
            table_name="ts_adj_factor",
            date_column="trade_date",
            api_func=self.pro.adj_factor,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    @lru_cache(maxsize=128)
    def get_adjusted_daily(self, ts_code, start_date, end_date, adj="hfq"):
        """
        【V2.4优化-核心工具】获取复权日线行情（支持双数据源热备）。
        自动处理日线和复权因子的合并与计算。
        :param adj: 'qfq' (前复权), 'hfq' (后复权)
        """
        df_daily = self.get_daily(ts_code, start_date, end_date)
        df_adj = self.get_adj_factor(ts_code, start_date, end_date)

        # 主数据源Tushare获取失败时，启动备用数据源AkShare
        if df_daily is None or df_daily.empty:
            log.warning(
                f"主数据源(Tushare)未能获取 {ts_code} 的日线数据，尝试启动备用数据源(AkShare)..."
            )
            # AkShare自带前复权，因此直接返回，不再需要与adj_factor合并
            df_ak = self._get_daily_ak(ts_code, start_date, end_date)
            if df_ak is not None and not df_ak.empty:
                log.info(f"备用数据源(AkShare)成功获取到 {ts_code} 的数据。")
                return df_ak
            else:
                log.error(f"主备数据源均未能获取 {ts_code} 的日线数据。")
                return None

        if df_adj is None or df_adj.empty:
            return None

        # 【鲁棒性修复】确保 trade_date 列是 datetime 类型以便正确合并
        df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
        df_adj["trade_date"] = pd.to_datetime(df_adj["trade_date"])

        df = pd.merge(df_daily, df_adj, on="trade_date")
        if df.empty:
            return None  # 如果合并后为空，直接返回

        df = df.sort_values("trade_date", ascending=True).reset_index(drop=True)

        if adj == "qfq":
            # 后复权因子是向前累乘的，所以最后一个是基准。前复权则相反。
            # 此处应使用最后一个因子值作为基准来计算前复权因子
            if not df.empty:
                last_factor = df["adj_factor"].iloc[-1]
                df["adj_factor"] = df["adj_factor"] / last_factor

        # 确保价格列是数值类型
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in price_cols:
            df[col] = df[col] * df["adj_factor"]

        return df.drop(columns=["adj_factor"])

    # --- (二) 日度行情与指标 (同步) ---
    @api_rate_limit("daily_basic")
    def get_daily_basic(self, ts_code, start_date, end_date):
        return self.pro.daily_basic(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    @api_rate_limit("moneyflow")
    def get_moneyflow(self, ts_code, start_date, end_date):
        return self.pro.moneyflow(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    # --- (三) 财务与基本面 (同步) ---
    @api_rate_limit("fina_indicator")
    def get_fina_indicator(self, ts_code: str, force_update=False):
        """【V2.2重构完成】获取单只股票的财务指标，适配统一大表 financial_indicators。"""
        table_name = "financial_indicators"
        primary_keys = ["ts_code", "end_date"]

        # 1. 尝试从数据库获取最新数据
        if not force_update:
            try:
                query = sqlalchemy.text(
                    f"""
                    SELECT * FROM {table_name}
                    WHERE ts_code = :ts_code
                    ORDER BY end_date DESC
                    LIMIT 1
                """
                )
                with self.engine.connect() as connection:
                    latest_in_db = pd.read_sql(
                        query,
                        connection,
                        params={"ts_code": ts_code},
                    )

                if not latest_in_db.empty:
                    latest_end_date_db = pd.to_datetime(latest_in_db["end_date"].iloc[0])
                    current_date = datetime.now()

                    # 判断是否已到下一个报告期
                    is_fresh = False
                    if latest_end_date_db.month == 3 and current_date.month <= 6:
                        is_fresh = True
                    elif latest_end_date_db.month == 6 and current_date.month <= 9:
                        is_fresh = True
                    elif latest_end_date_db.month == 9 and current_date.month <= 12:
                        is_fresh = True
                    elif latest_end_date_db.month == 12 and current_date.month <= 3:
                        is_fresh = True

                    if is_fresh:
                        log.info(
                            f"数据库中 {table_name} 表 {ts_code} 的最新财务指标 ({latest_end_date_db.strftime('%Y%m%d')}) 仍是新鲜的，直接从数据库返回。"
                        )
                        # 从数据库获取所有历史数据
                        query_all = sqlalchemy.text(
                            f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
                        )
                        with self.engine.connect() as connection:
                            df_db = pd.read_sql(
                                query_all,
                                connection,
                                params={"ts_code": ts_code},
                            )
                        return df_db.sort_values(by="end_date", ascending=False).reset_index(drop=True)

            except Exception as e:
                log.warning(f"从 {table_name} 读取最新财务指标失败: {e}，将尝试从API获取。")

        # 2. 从API获取数据并UPSERT
        log.info(f"从API获取 {table_name} 表 {ts_code} 的财务指标...")
        
        # 确定增量获取的起始日期
        fetch_start_date = None
        if not force_update: # 只有在非强制更新模式下才尝试增量
            try:
                query_max_date = sqlalchemy.text(
                    f"SELECT MAX(end_date) FROM {table_name} WHERE ts_code = :ts_code"
                )
                with self.engine.connect() as connection:
                    max_date_in_db = connection.execute(query_max_date, {"ts_code": ts_code}).scalar_one_or_none()
                
                if max_date_in_db:
                    # 从数据库最新日期的下一天开始获取
                    fetch_start_date = (pd.to_datetime(max_date_in_db) + pd.Timedelta(days=1)).strftime("%Y%m%d")
            except Exception as e:
                log.warning(f"获取数据库最大日期失败: {e}，将尝试全量获取。")

        # 如果没有指定起始日期（例如数据库为空或获取最大日期失败），则从一个很早的日期开始
        if not fetch_start_date:
            fetch_start_date = "19900101" # 确保获取所有历史数据

        # 调用Tushare API，传入增量日期参数
        df_api = self.pro.fina_indicator(ts_code=ts_code, start_date=fetch_start_date)
        
        if df_api is not None and not df_api.empty:
            # 确保 end_date 和 ann_date 是日期格式
            if 'end_date' in df_api.columns:
                df_api['end_date'] = pd.to_datetime(df_api['end_date'], errors='coerce').dt.strftime('%Y%m%d')
            if 'ann_date' in df_api.columns:
                df_api['ann_date'] = pd.to_datetime(df_api['ann_date'], errors='coerce').dt.strftime('%Y%m%d')

            db_cols = [
                "ts_code",
                "ann_date",
                "end_date",
                "roe",
                "netprofit_yoy",
                "debt_to_assets",
                "or_yoy",
            ]
            api_cols_to_keep = [col for col in db_cols if col in df_api.columns]
            self._upsert_data(table_name, df_api[api_cols_to_keep], primary_keys)

            # 重新从数据库读取所有数据以确保完整性
            query_all = sqlalchemy.text(
                f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
            )
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query_all, connection, params={"ts_code": ts_code})
            return df_db.sort_values(by="end_date", ascending=False).reset_index(drop=True)
        return pd.DataFrame()

    @api_rate_limit("income")
    def get_income(self, ts_code, period):
        return self._fetch_and_cache_quarterly(
            table_name="financial_income",
            api_func=self.pro.income,
            ts_code=ts_code,
            period=period,
            report_type="1",
        )

    @api_rate_limit("balancesheet")
    def get_balancesheet(self, ts_code, period):
        return self._fetch_and_cache_quarterly(
            table_name="financial_balancesheet",
            api_func=self.pro.balancesheet,
            ts_code=ts_code,
            period=period,
            report_type="1",
        )

    @api_rate_limit("cashflow")
    def get_cashflow(self, ts_code, period):
        return self._fetch_and_cache_quarterly(
            table_name="financial_cashflow",
            api_func=self.pro.cashflow,
            ts_code=ts_code,
            period=period,
            report_type="1",
        )

    # --- (四) 资金流与筹码 (同步) ---
    @api_rate_limit("hk_hold")
    def get_hk_hold(self, ts_code, start_date, end_date):
        return self.pro.hk_hold(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    @api_rate_limit("margin_detail")
    def get_margin_detail(self, ts_code, start_date, end_date):
        return self.pro.margin_detail(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    @api_rate_limit("top10_floatholders")
    def get_top10_floatholders(self, ts_code: str, period: str = None, force_update=False):
        """
        【V2.5优化】获取十大流通股东，支持按季度缓存。
        :param ts_code: 股票代码
        :param period: 报告期，如 '20240331'。如果为None，则尝试获取最新。
        :param force_update: 是否强制从API更新。
        """
        table_name = "top10_floatholders"
        primary_keys = ["ts_code", "end_date", "holder_name"]

        # 1. 尝试从数据库获取最新数据
        if not force_update:
            try:
                query = sqlalchemy.text(
                    f"""
                    SELECT * FROM {table_name}
                    WHERE ts_code = :ts_code
                    ORDER BY end_date DESC
                    LIMIT 1
                """
                )
                with self.engine.connect() as connection:
                    latest_in_db = pd.read_sql(
                        query,
                        connection,
                        params={"ts_code": ts_code},
                    )

                if not latest_in_db.empty:
                    latest_end_date_db = pd.to_datetime(latest_in_db["end_date"].iloc[0])
                    current_date = datetime.now()

                    # 判断是否已到下一个报告期
                    is_fresh = False
                    if latest_end_date_db.month == 3 and current_date.month <= 6:
                        is_fresh = True
                    elif latest_end_date_db.month == 6 and current_date.month <= 9:
                        is_fresh = True
                    elif latest_end_date_db.month == 9 and current_date.month <= 12:
                        is_fresh = True
                    elif latest_end_date_db.month == 12 and current_date.month <= 3:
                        is_fresh = True

                    if is_fresh:
                        log.info(
                            f"数据库中 {table_name} 表 {ts_code} 的最新十大流通股东 ({latest_end_date_db.strftime('%Y%m%d')}) 仍是新鲜的，直接从数据库返回。"
                        )
                        # 从数据库获取所有历史数据
                        query_all = sqlalchemy.text(
                            f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
                        )
                        with self.engine.connect() as connection:
                            df_db = pd.read_sql(
                                query_all,
                                connection,
                                params={"ts_code": ts_code},
                            )
                        return df_db.sort_values(by="end_date", ascending=False).reset_index(drop=True)

            except Exception as e:
                log.warning(f"从 {table_name} 读取最新十大流通股东失败: {e}，将尝试从API获取。")

        # 2. 从API获取数据并UPSERT
        log.info(f"从API获取 {table_name} 表 {ts_code} 的十大流通股东...")
        df_api = self.pro.top10_floatholders(ts_code=ts_code, period=period)

        if df_api is not None and not df_api.empty:
            # 确保 end_date 和 ann_date 是日期格式
            if 'end_date' in df_api.columns:
                df_api['end_date'] = pd.to_datetime(df_api['end_date'], errors='coerce').dt.strftime('%Y%m%d')
            if 'ann_date' in df_api.columns:
                df_api['ann_date'] = pd.to_datetime(df_api['ann_date'], errors='coerce').dt.strftime('%Y%m%d')

            self._upsert_data(table_name, df_api, primary_keys)
            # 重新从数据库读取所有数据以确保完整性
            query_all = sqlalchemy.text(
                f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
            )
            with self.engine.connect() as connection:
                df_db = pd.read_sql(
                    query_all,
                    connection,
                    params={"ts_code": ts_code},
                )
            return df_db.sort_values(by="end_date", ascending=False).reset_index(drop=True)
        return pd.DataFrame()

    @api_rate_limit("holder_number")
    def get_holder_number(self, ts_code: str, force_update=False):
        """【V2.1重构-效率】获取股东人数，使用UPSERT逻辑高效缓存。"""
        table_name = "stk_holdernumber"
        primary_keys = ["ts_code", "end_date"]

        # 1. 尝试从数据库获取最新数据
        if not force_update:
            try:
                query = sqlalchemy.text(
                    f"""
                    SELECT * FROM {table_name}
                    WHERE ts_code = :ts_code
                    ORDER BY end_date DESC
                    LIMIT 1
                """
                )
                with self.engine.connect() as connection:
                    latest_in_db = pd.read_sql(
                        query,
                        connection,
                        params={"ts_code": ts_code},
                    )

                if not latest_in_db.empty:
                    latest_end_date_db = pd.to_datetime(latest_in_db["end_date"].iloc[0])
                    current_date = datetime.now()

                    # 判断是否已到下一个报告期
                    is_fresh = False
                    if latest_end_date_db.month == 3 and current_date.month <= 6:
                        is_fresh = True
                    elif latest_end_date_db.month == 6 and current_date.month <= 9:
                        is_fresh = True
                    elif latest_end_date_db.month == 9 and current_date.month <= 12:
                        is_fresh = True
                    elif latest_end_date_db.month == 12 and current_date.month <= 3:
                        is_fresh = True

                    if is_fresh:
                        log.info(
                            f"数据库中 {table_name} 表 {ts_code} 的最新股东人数 ({latest_end_date_db.strftime('%Y%m%d')}) 仍是新鲜的，直接从数据库返回。"
                        )
                        # 从数据库获取所有历史数据
                        query_all = sqlalchemy.text(
                            f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
                        )
                        with self.engine.connect() as connection:
                            df_db = pd.read_sql(
                                query_all,
                                connection,
                                params={"ts_code": ts_code},
                            )
                        return df_db.sort_values(by="end_date").reset_index(drop=True)

            except Exception as e:
                log.warning(f"从 {table_name} 读取最新股东人数失败: {e}，将尝试从API获取。")

        # 2. 从API获取数据并UPSERT
        log.info(f"从API获取 {table_name} 表 {ts_code} 的股东人数...")
        df_api = self.pro.stk_holdernumber(ts_code=ts_code)
        if df_api is not None and not df_api.empty:
            self._upsert_data(table_name, df_api, primary_keys)

        # 无论如何，最后都从数据库返回最完整的数据
        try:
            query = sqlalchemy.text(
                f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
            )
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query, connection, params={"ts_code": ts_code})
            return df_db.sort_values(by="end_date").reset_index(drop=True)
        except Exception as e:
            print(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return pd.DataFrame()

    @api_rate_limit("holder_trade")
    def get_holder_trade(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取重要股东增减持数据，带缓存。"""
        # 此接口数据量较小，可以简化缓存逻辑：优先从API获取，然后补充写入。
        table_name = "stk_holdertrade"
        df_api = self.pro.stk_holdertrade(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
        if df_api is not None and not df_api.empty:
            try:
                with self.engine.connect() as connection:
                    with connection.begin():
                        # to_sql 配合复合主键，重复数据将因违反主键约束而无法插入
                        df_api.to_sql(
                            table_name, connection, if_exists="append", index=False
                        )
            except sqlalchemy.exc.IntegrityError:
                # 这是一个预期的行为，当尝试插入重复数据时发生，可以安全地忽略
                pass
            except Exception as e:
                print(f"写入 {table_name} for {ts_code} 失败: {e}")
        return df_api

    @api_rate_limit("top_list")
    def get_top_list(
        self,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None,
        force_update=False,
    ):
        """【V2.1修正】获取龙虎榜数据，支持单日或多日查询，带缓存。"""
        table_name = "top_list"

        if trade_date:
            start_date = end_date = trade_date

        if start_date and end_date:
            if not force_update:
                try:
                    query = sqlalchemy.text(
                        "SELECT * FROM top_list WHERE trade_date BETWEEN :start AND :end"
                    )
                    with self.engine.connect() as connection:
                        df_db = pd.read_sql(
                            query,
                            connection,
                            params={"start": start_date, "end": end_date},
                        )
                    if not df_db.empty:
                        return df_db
                except Exception as e:
                    print(
                        f"从 {table_name} 读取 {start_date}-{end_date} 数据失败: {e}。将从API获取。"
                    )

            # 按日循环获取
            all_dfs = []
            trade_dates = self.pro.trade_cal(start_date=start_date, end_date=end_date)
            open_dates = trade_dates[trade_dates["is_open"] == 1]["cal_date"].tolist()

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
                    final_df.to_sql(
                        table_name, connection, if_exists="append", index=False
                    )
            return final_df

        # 原有的单日逻辑
        if not force_update:
            try:
                query = sqlalchemy.text(
                    "SELECT * FROM top_list WHERE trade_date = :trade_date"
                )
                with self.engine.connect() as connection:
                    df_db = pd.read_sql(
                        query, connection, params={"trade_date": trade_date}
                    )
                if not df_db.empty:
                    return df_db
            except Exception as e:
                print(f"从 {table_name} 读取 {trade_date} 数据失败: {e}。将从API获取。")

        df_api = self.pro.top_list(trade_date=trade_date)
        if df_api is not None and not df_api.empty:
            with self.engine.connect() as connection:
                with connection.begin():
                    delete_sql = sqlalchemy.text(
                        "DELETE FROM top_list WHERE trade_date = :trade_date"
                    )
                    connection.execute(delete_sql, {"trade_date": trade_date})
                    df_api.to_sql(
                        table_name, connection, if_exists="append", index=False
                    )
        return df_api

    @api_rate_limit("forecast")
    def get_forecast(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取业绩预告数据，带缓存。"""
        table_name = "financial_forecast"
        # 此接口数据量较小，可以简化缓存逻辑：优先从API获取，然后补充写入。
        df_api = self.pro.forecast(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
        if df_api is not None and not df_api.empty:
            try:
                with self.engine.connect() as connection:
                    with connection.begin():
                        # to_sql 配合复合主键，重复数据将因违反主键约束而无法插入
                        df_api.to_sql(
                            table_name, connection, if_exists="append", index=False
                        )
            except sqlalchemy.exc.IntegrityError:
                # 这是一个预期的行为，当尝试插入重复数据时发生，可以安全地忽略
                pass
            except Exception as e:
                print(f"写入 {table_name} for {ts_code} 失败: {e}")
        return df_api

    @api_rate_limit("dividend")
    def get_dividend(self, ts_code: str, force_update=False):
        """【V2.1重构-效率】获取分红送股数据，使用UPSERT逻辑高效缓存。"""
        table_name = "financial_dividend"
        primary_keys = ["ts_code", "end_date"]

        df_api = self.pro.dividend(ts_code=ts_code)
        if df_api is not None and not df_api.empty:
            self._upsert_data(table_name, df_api, primary_keys)

        # 无论如何，最后都从数据库返回最完整的数据
        try:
            query = sqlalchemy.text(
                f"SELECT * FROM {table_name} WHERE ts_code = :ts_code"
            )
            with self.engine.connect() as connection:
                df_db = pd.read_sql(query, connection, params={"ts_code": ts_code})
            return df_db.sort_values(by="end_date").reset_index(drop=True)
        except Exception as e:
            print(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return pd.DataFrame()

    @api_rate_limit("repurchase")
    def get_repurchase(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取股票回购数据，带缓存。"""
        table_name = "financial_repurchase"
        df_api = self.pro.repurchase(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
        if df_api is not None and not df_api.empty:
            try:
                with self.engine.connect() as connection:
                    with connection.begin():
                        # to_sql 配合复合主键，重复数据将因违反主键约束而无法插入
                        df_api.to_sql(
                            table_name, connection, if_exists="append", index=False
                        )
            except sqlalchemy.exc.IntegrityError:
                # 这是一个预期的行为，当尝试插入重复数据时发生，可以安全地忽略
                pass
            except Exception as e:
                print(f"写入 {table_name} for {ts_code} 失败: {e}")
        return df_api

    @api_rate_limit("express")
    def get_express(self, ts_code: str, start_date: str, end_date: str):
        """【V2.1新增】获取业绩快报数据，带缓存。"""
        table_name = "express"
        primary_keys = ["ts_code", "end_date"]

        # 1. 尝试从数据库获取最新数据
        try:
            query = sqlalchemy.text(
                f"""
                SELECT * FROM {table_name}
                WHERE ts_code = :ts_code AND end_date BETWEEN :start AND :end
                ORDER BY end_date DESC
                LIMIT 1
                """
            )
            with self.engine.connect() as connection:
                latest_in_db = pd.read_sql(
                    query,
                    connection,
                    params={"ts_code": ts_code, "start": start_date, "end": end_date},
                )

            if not latest_in_db.empty:
                latest_end_date_db = pd.to_datetime(latest_in_db["end_date"].iloc[0])
                current_date = datetime.now()

                # 判断是否已到下一个报告期
                is_fresh = False
                if latest_end_date_db.month == 3 and current_date.month <= 6:
                    is_fresh = True
                elif latest_end_date_db.month == 6 and current_date.month <= 9:
                    is_fresh = True
                elif latest_end_date_db.month == 9 and current_date.month <= 12:
                    is_fresh = True
                elif latest_end_date_db.month == 12 and current_date.month <= 3:
                    is_fresh = True

                if is_fresh:
                    log.info(
                        f"数据库中 {table_name} 表 {ts_code} 的最新业绩快报 ({latest_end_date_db.strftime('%Y%m%d')}) 仍是新鲜的，直接从数据库返回。"
                    )
                    # 从数据库获取所有历史数据
                    query_all = sqlalchemy.text(
                        f"SELECT * FROM {table_name} WHERE ts_code = :ts_code AND end_date BETWEEN :start AND :end"
                    )
                    with self.engine.connect() as connection:
                        df_db = pd.read_sql(
                            query_all,
                            connection,
                            params={"ts_code": ts_code, "start": start_date, "end": end_date},
                        )
                    return df_db

        except Exception as e:
            log.warning(f"从 {table_name} 读取最新业绩快报失败: {e}，将尝试从API获取。")

        # 2. 从API获取数据并UPSERT
        log.info(f"从API获取 {table_name} 表 {ts_code} 的业绩快报...")
        df_api = self.pro.express(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
        if df_api is not None and not df_api.empty:
            self._upsert_data(table_name, df_api, primary_keys)

        # 从数据库读取返回，确保数据完整性
        try:
            query = sqlalchemy.text(
                f"SELECT * FROM {table_name} WHERE ts_code = :ts_code AND end_date BETWEEN :start AND :end"
            )
            with self.engine.connect() as connection:
                df_db = pd.read_sql(
                    query,
                    connection,
                    params={"ts_code": ts_code, "start": start_date, "end": end_date},
                )
            return df_db
        except Exception as e:
            print(f"从 {table_name} 读取 {ts_code} 数据失败: {e}。")
            return pd.DataFrame() # 如果读取失败，返回空DataFrame

    @api_rate_limit("block_trade")
    def get_block_trade(
        self, trade_date: str = None, start_date: str = None, end_date: str = None
    ):
        """【V2.1修正】获取大宗交易数据，支持单日或多日查询，使用UPSERT缓存。"""
        table_name = "block_trade"

        if trade_date:
            start_date = end_date = trade_date

        if start_date and end_date:
            # 按日循环获取
            all_dfs = []
            trade_dates = self.pro.trade_cal(start_date=start_date, end_date=end_date)
            open_dates = trade_dates[trade_dates["is_open"] == 1]["cal_date"].tolist()

            for date in open_dates:
                df_api_daily = self.pro.block_trade(trade_date=date)
                if df_api_daily is not None and not df_api_daily.empty:
                    all_dfs.append(df_api_daily)

            if not all_dfs:
                return pd.DataFrame()

            final_df = pd.concat(all_dfs)
            self._upsert_data(
                table_name,
                final_df,
                ["trade_date", "ts_code", "amount", "buyer", "seller"],
            )
            return final_df

        return pd.DataFrame()

    # --- (五) 宏观经济 (同步) ---
    @api_rate_limit("cn_m")
    def get_cn_m(self, start_m, end_m, force_update=False):
        return self._fetch_and_cache(
            self.pro.cn_m,
            "macro_cn_m",
            force_update=force_update,
            frequency="monthly",
            start_m=start_m,
            end_m=end_m,
        )

    @api_rate_limit("cn_pmi")
    def get_cn_pmi(self, start_m, end_m, force_update=False):
        return self._fetch_and_cache(
            self.pro.cn_pmi,
            "macro_cn_pmi",
            force_update=force_update,
            frequency="monthly",
            start_m=start_m,
            end_m=end_m,
        )

    @api_rate_limit("cn_cpi")
    def get_cn_cpi(self, start_m: str = None, end_m: str = None, force_update=False):
        """【V2.1激活】获取全国居民消费价格指数CPI，使用通用缓存"""
        return self._fetch_and_cache(
            self.pro.cn_cpi,
            "macro_cn_cpi",
            force_update=force_update,
            frequency="monthly",
            start_m=start_m,
            end_m=end_m,
        )

    @api_rate_limit("cn_gdp")
    def get_cn_gdp(self, start_q: str = None, end_q: str = None, force_update=False):
        """【V2.1新增】获取季度GDP数据，使用通用缓存"""
        return self._fetch_and_cache(
            self.pro.cn_gdp,
            "macro_cn_gdp",
            force_update=force_update,
            frequency="quarterly",
            start_q=start_q,
            end_q=end_q,
        )

    @api_rate_limit("shibor")
    def get_shibor(
        self, start_date: str = None, end_date: str = None, force_update=False
    ):
        """【V2.1激活】获取Shibor同业拆放利率，使用通用缓存"""
        return self._fetch_and_cache(
            self.pro.shibor,
            "macro_shibor",
            force_update=force_update,
            start_date=start_date,
            end_date=end_date,
        )

    @lru_cache(maxsize=256)
    def get_minute_bars_ak(
        self, ts_code: str, period: str = "5", adjust: str = "qfq"
    ) -> pd.DataFrame:
        """【V2.4新增】使用AkShare获取分钟级别K线数据。"""
        # ts_code格式转换为akshare格式： e.g., '600519.SH' -> 'sh600519'
        ak_code = f"{ts_code[-2:].lower()}{ts_code[:6]}"
        try:
            df = ak.stock_zh_a_hist_min_em(symbol=ak_code, period=period, adjust=adjust)
            if df is not None and not df.empty:
                # 重命名列以适配内部标准
                df = df.rename(
                    columns={
                        "时间": "trade_time",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "vol",
                        "成交额": "amount",
                        "换手率": "turnover",
                    }
                )
                df["trade_time"] = pd.to_datetime(df["trade_time"])
                return df[
                    [
                        "trade_time",
                        "open",
                        "close",
                        "high",
                        "low",
                        "vol",
                        "amount",
                        "turnover",
                    ]
                ]
        except Exception as e:
            log.warning(f"使用AkShare获取 {ts_code} 的分钟线数据失败: {e}")
        return pd.DataFrame()

    def _get_daily_ak(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """【V2.4优化】使用AkShare获取日线行情作为备用数据源。"""
        ak_code = f"{ts_code[-2:].lower()}{ts_code[:6]}"
        start_date_ak = pd.to_datetime(start_date).strftime("%Y%m%d")
        end_date_ak = pd.to_datetime(end_date).strftime("%Y%m%d")
        try:
            df = ak.stock_zh_a_hist(
                symbol=ak_code,
                start_date=start_date_ak,
                end_date=end_date_ak,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                # 重命名列以适配内部标准
                df = df.rename(
                    columns={
                        "日期": "trade_date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "vol",
                        "成交额": "amount",
                        "涨跌幅": "pct_chg",
                        "换手率": "turnover_rate",
                    }
                )
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                # AkShare的vol单位是股，Tushare是手，需要统一
                df["vol"] = df["vol"] / 100
                return df
        except Exception as e:
            log.warning(f"后备方案：使用AkShare获取 {ts_code} 的日线数据失败: {e}")
        return None

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

    async def _fetch_async(
        self, api_name: str, max_retries: int = 3, **kwargs
    ) -> pd.DataFrame:
        """【V2.7 增强版】核心的异步数据获取方法，增加了重试与指数退避机制。"""
        session = await self._get_async_session()
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": kwargs,
            "fields": [],
        }

        # --- 新增重试逻辑 ---
        for attempt in range(max_retries + 1):
            async with self.semaphore:
                # 在首次尝试前不延迟，在重试前延迟
                if attempt > 0:
                    # 指数退避策略: 1s, 2s, 4s...
                    sleep_time = 2**attempt
                    log.warning(
                        f"  > API '{api_name}' 请求失败 (参数: {kwargs}), 第 {attempt}/{max_retries} 次重试，等待 {sleep_time} 秒..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(self.request_delay)

                try:
                    async with session.post(
                        self._API_URL, data=json.dumps(payload), timeout=60
                    ) as response:
                        response.raise_for_status()
                        resp_json = await response.json()

                        if resp_json["code"] != 0:
                            # Tushare 返回了业务错误，这种错误通常重试也无效，直接抛出
                            raise ConnectionError(
                                f"Tushare API Error: {resp_json['msg']}"
                            )

                        data = resp_json["data"]
                        columns = data["fields"]
                        items = data["items"]
                        return pd.DataFrame(items, columns=columns)

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # 捕获网络层错误 (如 Server disconnected) 或超时错误，这些适合重试
                    log.debug(f"  > 捕获到网络或超时错误: {e}")
                    if attempt == max_retries:
                        log.error(
                            f"  > API '{api_name}' (参数: {kwargs}) 在尝试 {max_retries} 次后彻底失败。"
                        )
                        return pd.DataFrame()  # 返回空DataFrame表示失败
                    continue  # 继续下一次重试

                except ConnectionError as e:
                    # 捕获Tushare业务逻辑错误，直接失败
                    log.error(f"  > Tushare业务逻辑错误: {e}")
                    return pd.DataFrame()

                except Exception as e:
                    # 捕获其他未知异常
                    log.error(
                        f"  > API '{api_name}' (参数: {kwargs}) 发生未知严重错误: {e}",
                        exc_info=True,
                    )
                    return pd.DataFrame()
        # --- 重试逻辑结束 ---
        return pd.DataFrame()  # 如果循环结束仍未成功，返回空

    async def _batch_get_daily_async(
        self, ts_codes: list, start_date: str, end_date: str
    ) -> dict:
        """并发获取多支股票的日线行情"""
        tasks = [
            self._fetch_async(
                "daily", ts_code=code, start_date=start_date, end_date=end_date
            )
            for code in ts_codes
        ]
        results_list = await asyncio.gather(*tasks)
        return {code: df for code, df in zip(ts_codes, results_list)}

    def run_batch_download(
        self, ts_codes: list, start_date: str, end_date: str
    ) -> dict:
        """同步方法来运行异步批量下载任务"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(
            self._batch_get_daily_async(ts_codes, start_date, end_date)
        )
        loop.run_until_complete(self._close_async_session())
        return results

    # --- (七) 非结构化数据处理 ---
    def update_text_corpus(self, ts_code):
        if ts_code not in config.SIMULATED_NEWS_SOURCE:
            return
        news_list = config.SIMULATED_NEWS_SOURCE[ts_code]
        with self.engine.connect() as connection:
            with connection.begin():  # 开启事务
                for news in news_list:
                    content_hash = hash(news["content"])
                    # PostgreSQL 使用 %s作为参数占位符
                    res = connection.execute(
                        sqlalchemy.text(
                            "SELECT doc_id FROM text_corpus WHERE content_hash = :chash"
                        ),
                        {"chash": str(content_hash)},
                    ).fetchone()
                    if res is None:
                        connection.execute(
                            sqlalchemy.text(
                                """
                            INSERT INTO text_corpus (ts_code, publish_date, title, source, content, content_hash)
                            VALUES (:tsc, :pd, :ti, :so, :co, :ch)
                        """
                            ),
                            {
                                "tsc": ts_code,
                                "pd": datetime.now().strftime("%Y%m%d"),
                                "ti": news["title"],
                                "so": "Simulated Source",
                                "co": news["content"],
                                "ch": str(content_hash),
                            },
                        )

    def get_text_for_analysis(self, ts_code, date_range=None, limit=3):
        self.update_text_corpus(ts_code)
        query = f"SELECT title, content FROM text_corpus WHERE ts_code = '{ts_code}' ORDER BY publish_date DESC LIMIT {limit}"
        with self.engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df

    def get_pit_financial_data(
        self, all_financial_data: pd.DataFrame, as_of_date: str
    ) -> pd.DataFrame:
        """
        【新增】过滤财务数据以确保其在时间点（Point-in-Time, PIT）上是正确的。
        防止"未来函数"或"前视偏差"(lookahead bias)。
        """
        if (
            "ann_date" not in all_financial_data.columns
            or "end_date" not in all_financial_data.columns
        ):
            # log.warning("财务数据必须包含 'ann_date' 和 'end_date' 列才能进行PIT正确性处理。")
            return None

        # 确保日期列是datetime类型
        all_financial_data["ann_date"] = pd.to_datetime(all_financial_data["ann_date"])
        as_of_date_dt = pd.to_datetime(as_of_date)

        # 过滤出公告日在指定日期之前的数据
        available_data = all_financial_data[
            all_financial_data["ann_date"] <= as_of_date_dt
        ].copy()

        if available_data.empty:
            return None

        # 按报告期（end_date）降序排序，获取最新的那份财报
        available_data.sort_values(by="end_date", ascending=False, inplace=True)

        return available_data.head(1)

    def purge_old_ai_reports(self, ts_code: str, keep_latest: int = 10):
        """【V2.1新增】清理指定股票的旧AI报告，仅保留最新的N份。"""
        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    # 使用窗口函数和子查询来确定要删除的旧报告
                    # 这是一个健壮的、在各种SQL方言中都有效的方法
                    delete_sql = sqlalchemy.text(
                        f"""
                        DELETE FROM ai_reports
                        WHERE ctid IN (
                            SELECT ctid FROM (
                                SELECT ctid, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                                FROM ai_reports
                                WHERE ts_code = :ts_code
                            ) as sub
                            WHERE sub.rn > :keep_latest
                        )
                    """
                    )
                    result = connection.execute(
                        delete_sql, {"ts_code": ts_code, "keep_latest": keep_latest}
                    )
                    if result.rowcount > 0:
                        log.info(
                            "为 %s 清理了 %s 份旧的AI报告。", ts_code, result.rowcount
                        )
        except Exception:
            log.error("清理 %s 的AI报告时出错:", ts_code, exc_info=True)

    def __del__(self):
        """关闭数据库连接和异步会话"""
        # SQLAlchemy 引擎通常不需要手动关闭，连接池会管理
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._close_async_session())
        except (RuntimeError, AttributeError):
            pass
