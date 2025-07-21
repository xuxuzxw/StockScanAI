#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量操作优化器 - 专门优化数据存取效率

核心优化策略：
1. 批量插入优化 - 使用COPY和批量事务
2. 数据更新优化 - 使用UPSERT和批量更新
3. 查询缓存策略 - 减少重复查询
4. 连接池优化 - 提高并发性能
5. 内存使用优化 - 减少内存占用
"""

import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import psycopg2
from io import StringIO
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from logger_config import log


class BatchOperationsOptimizer:
    def __init__(self):
        # 优化连接池配置
        self.engine = create_engine(
            config.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,          # 连接池大小
            max_overflow=10,      # 最大溢出连接
            pool_pre_ping=True,   # 连接前检查
            pool_recycle=3600     # 连接回收时间(秒)
        )
    
    def optimize_bulk_insert(self):
        """优化批量插入操作"""
        log.info("=== 批量插入优化测试 ===")
        
        # 创建测试数据
        test_data = self._generate_test_data(10000)
        
        # 测试不同的插入方法
        methods = [
            ('pandas.to_sql', self._test_pandas_insert),
            ('COPY命令', self._test_copy_insert),
            ('批量INSERT', self._test_batch_insert),
            ('单条INSERT', self._test_single_insert)
        ]
        
        results = {}
        for method_name, method_func in methods:
            try:
                log.info(f"\n测试方法: {method_name}")
                elapsed_time = method_func(test_data.copy())
                results[method_name] = elapsed_time
                log.info(f"耗时: {elapsed_time:.2f}秒")
            except Exception as e:
                log.error(f"{method_name} 测试失败: {e}")
                results[method_name] = float('inf')
        
        # 性能对比
        log.info("\n=== 批量插入性能对比 ===")
        fastest = min(results.values())
        for method, time_taken in sorted(results.items(), key=lambda x: x[1]):
            if time_taken != float('inf'):
                speedup = fastest / time_taken if time_taken > 0 else 1
                log.info(f"{method}: {time_taken:.2f}秒 (相对最快: {speedup:.1f}x)")
    
    def _generate_test_data(self, rows=10000):
        """生成测试数据"""
        import random
        from datetime import datetime, timedelta
        
        base_date = datetime(2024, 1, 1)
        data = []
        
        for i in range(rows):
            data.append({
                'trade_date': base_date + timedelta(days=i % 365),
                'ts_code': f'TEST{i%100:03d}.SH',
                'open': round(random.uniform(10, 100), 2),
                'high': round(random.uniform(10, 100), 2),
                'low': round(random.uniform(10, 100), 2),
                'close': round(random.uniform(10, 100), 2),
                'vol': random.randint(1000, 1000000),
                'amount': round(random.uniform(1000000, 100000000), 2)
            })
        
        return pd.DataFrame(data)
    
    def _test_pandas_insert(self, data):
        """测试pandas.to_sql插入"""
        start_time = time.time()
        
        # 创建临时表
        table_name = 'test_pandas_insert'
        self._create_test_table(table_name)
        
        try:
            data.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
        finally:
            self._drop_test_table(table_name)
        
        return time.time() - start_time
    
    def _test_copy_insert(self, data):
        """测试COPY命令插入"""
        start_time = time.time()
        
        table_name = 'test_copy_insert'
        self._create_test_table(table_name)
        
        try:
            # 使用psycopg2的COPY命令
            conn = psycopg2.connect(config.DATABASE_URL)
            cursor = conn.cursor()
            
            # 将DataFrame转换为CSV格式
            output = StringIO()
            data.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
            output.seek(0)
            
            # 使用COPY命令
            cursor.copy_from(
                output, 
                table_name,
                columns=('trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount'),
                sep='\t'
            )
            conn.commit()
            
        finally:
            if 'conn' in locals():
                conn.close()
            self._drop_test_table(table_name)
        
        return time.time() - start_time
    
    def _test_batch_insert(self, data):
        """测试批量INSERT"""
        start_time = time.time()
        
        table_name = 'test_batch_insert'
        self._create_test_table(table_name)
        
        try:
            with self.engine.connect() as conn:
                # 批量插入，每批1000条
                batch_size = 1000
                for i in range(0, len(data), batch_size):
                    batch = data.iloc[i:i+batch_size]
                    values = []
                    for _, row in batch.iterrows():
                        values.append(f"('{row['trade_date']}', '{row['ts_code']}', {row['open']}, {row['high']}, {row['low']}, {row['close']}, {row['vol']}, {row['amount']})")
                    
                    sql = f"""
                        INSERT INTO {table_name} 
                        (trade_date, ts_code, open, high, low, close, vol, amount)
                        VALUES {','.join(values)}
                    """
                    conn.execute(text(sql))
                conn.commit()
        finally:
            self._drop_test_table(table_name)
        
        return time.time() - start_time
    
    def _test_single_insert(self, data):
        """测试单条INSERT（仅测试前100条）"""
        start_time = time.time()
        
        table_name = 'test_single_insert'
        self._create_test_table(table_name)
        
        try:
            with self.engine.connect() as conn:
                # 只测试前100条，避免太慢
                test_data = data.head(100)
                for _, row in test_data.iterrows():
                    sql = f"""
                        INSERT INTO {table_name} 
                        (trade_date, ts_code, open, high, low, close, vol, amount)
                        VALUES ('{row['trade_date']}', '{row['ts_code']}', {row['open']}, {row['high']}, {row['low']}, {row['close']}, {row['vol']}, {row['amount']})
                    """
                    conn.execute(text(sql))
                conn.commit()
                
                # 按比例估算完整时间
                estimated_time = (time.time() - start_time) * (len(data) / 100)
                return estimated_time
        finally:
            self._drop_test_table(table_name)
        
        return time.time() - start_time
    
    def _create_test_table(self, table_name):
        """创建测试表"""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    trade_date DATE,
                    ts_code VARCHAR(20),
                    open DECIMAL(10,2),
                    high DECIMAL(10,2),
                    low DECIMAL(10,2),
                    close DECIMAL(10,2),
                    vol BIGINT,
                    amount DECIMAL(20,2)
                )
            """))
            conn.commit()
    
    def _drop_test_table(self, table_name):
        """删除测试表"""
        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()
    
    def optimize_upsert_operations(self):
        """优化UPSERT操作"""
        log.info("\n=== UPSERT操作优化 ===")
        
        # PostgreSQL的UPSERT语法示例
        upsert_examples = {
            'ts_daily数据更新': """
                INSERT INTO ts_daily (trade_date, ts_code, open, high, low, close, vol, amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_date, ts_code) 
                DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    vol = EXCLUDED.vol,
                    amount = EXCLUDED.amount
            """,
            '因子数据更新': """
                INSERT INTO factors_exposure (trade_date, ts_code, factor_name, factor_value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (trade_date, ts_code, factor_name)
                DO UPDATE SET factor_value = EXCLUDED.factor_value
            """
        }
        
        log.info("推荐的UPSERT模式:")
        for name, sql in upsert_examples.items():
            log.info(f"\n{name}:")
            log.info(f"  {sql.strip()}")
    
    def optimize_query_caching(self):
        """优化查询缓存策略"""
        log.info("\n=== 查询缓存优化 ===")
        
        caching_strategies = [
            {
                'strategy': '应用层缓存',
                'description': '在Python应用中缓存常用查询结果',
                'example': 'Redis或内存缓存股票基本信息、最新价格等'
            },
            {
                'strategy': '物化视图',
                'description': '数据库层面的查询结果缓存',
                'example': '每日市场统计、股票排行榜等'
            },
            {
                'strategy': '查询结果集缓存',
                'description': '缓存复杂查询的中间结果',
                'example': '因子计算结果、技术指标等'
            }
        ]
        
        for strategy in caching_strategies:
            log.info(f"\n{strategy['strategy']}:")
            log.info(f"  描述: {strategy['description']}")
            log.info(f"  应用: {strategy['example']}")
    
    def memory_usage_optimization(self):
        """内存使用优化"""
        log.info("\n=== 内存使用优化 ===")
        
        optimization_tips = [
            "1. 分块处理大数据集:",
            "   - 使用pandas.read_sql(chunksize=10000)分块读取",
            "   - 避免一次性加载全部数据到内存",
            "",
            "2. 数据类型优化:",
            "   - 使用合适的数据类型(int32 vs int64, float32 vs float64)",
            "   - 使用category类型处理重复字符串",
            "",
            "3. 及时释放内存:",
            "   - 使用del删除不需要的DataFrame",
            "   - 使用gc.collect()强制垃圾回收",
            "",
            "4. 流式处理:",
            "   - 对于ETL操作，使用生成器而非列表",
            "   - 边读边处理，避免数据积累"
        ]
        
        for tip in optimization_tips:
            log.info(tip)
    
    def connection_pool_optimization(self):
        """连接池优化"""
        log.info("\n=== 连接池优化 ===")
        
        with self.engine.connect() as conn:
            # 检查当前连接状态
            connections = conn.execute(text("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity 
                WHERE datname = current_database()
            """)).fetchone()
            
            if connections:
                log.info(f"当前数据库连接状态:")
                log.info(f"  总连接数: {connections[0]}")
                log.info(f"  活跃连接: {connections[1]}")
                log.info(f"  空闲连接: {connections[2]}")
            
            # 连接池配置建议
            log.info(f"\n当前连接池配置:")
            log.info(f"  pool_size: {self.engine.pool.size()}")
            log.info(f"  max_overflow: {self.engine.pool._max_overflow}")
            
            log.info(f"\n连接池优化建议:")
            log.info(f"  - 根据并发需求调整pool_size (建议5-20)")
            log.info(f"  - 设置合理的max_overflow (建议pool_size的2倍)")
            log.info(f"  - 启用pool_pre_ping避免连接超时")
            log.info(f"  - 设置pool_recycle定期回收连接")


def main():
    optimizer = BatchOperationsOptimizer()
    
    try:
        log.info("=== 批量操作优化分析 ===")
        
        # 1. 批量插入优化
        optimizer.optimize_bulk_insert()
        
        # 2. UPSERT操作优化
        optimizer.optimize_upsert_operations()
        
        # 3. 查询缓存优化
        optimizer.optimize_query_caching()
        
        # 4. 内存使用优化
        optimizer.memory_usage_optimization()
        
        # 5. 连接池优化
        optimizer.connection_pool_optimization()
        
        log.info("\n=== 批量操作优化完成 ===")
        
    except Exception as e:
        log.error(f"优化过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())