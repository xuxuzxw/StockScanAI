#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级查询优化 - 针对特定慢查询的深度优化

重点解决：
1. 单股票最新数据查询慢的问题
2. 市场排行查询慢的问题
3. 创建更有效的索引策略
"""

import time
from sqlalchemy import create_engine, text
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from logger_config import log


class AdvancedQueryOptimizer:
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
    
    def analyze_slow_queries(self):
        """分析慢查询的根本原因"""
        log.info("=== 慢查询根因分析 ===")
        
        with self.engine.connect() as conn:
            # 1. 分析单股票最新数据查询为什么慢
            log.info("\n1. 分析单股票最新数据查询:")
            
            # 检查查询执行计划
            plan = conn.execute(text("""
                EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
                SELECT * FROM ts_daily 
                WHERE ts_code = '600519.SH' 
                ORDER BY trade_date DESC 
                LIMIT 1
            """)).fetchall()
            
            log.info("执行计划:")
            for row in plan:
                log.info(f"  {row[0]}")
            
            # 2. 分析市场排行查询为什么慢
            log.info("\n2. 分析市场排行查询:")
            
            plan2 = conn.execute(text("""
                EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
                SELECT ts_code, close, pct_chg, vol
                FROM ts_daily 
                WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily)
                ORDER BY pct_chg DESC 
                LIMIT 50
            """)).fetchall()
            
            log.info("执行计划:")
            for row in plan2:
                log.info(f"  {row[0]}")
    
    def create_advanced_indexes(self):
        """创建高级索引优化"""
        log.info("\n=== 创建高级索引 ===")
        
        advanced_indexes = [
            {
                'name': 'idx_ts_daily_latest_by_stock',
                'sql': """
                    CREATE INDEX CONCURRENTLY idx_ts_daily_latest_by_stock 
                    ON ts_daily (ts_code, trade_date DESC) 
                    WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
                """,
                'description': '部分索引：只索引最近30天的数据，优化最新数据查询'
            },
            {
                'name': 'idx_ts_daily_pct_chg_latest',
                'sql': """
                    CREATE INDEX CONCURRENTLY idx_ts_daily_pct_chg_latest
                    ON ts_daily (trade_date, pct_chg DESC)
                    WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days'
                """,
                'description': '部分索引：最近7天的涨跌幅排序索引'
            }
        ]
        
        with self.engine.connect() as conn:
            for idx in advanced_indexes:
                try:
                    # 检查索引是否已存在
                    exists = conn.execute(text(f"""
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = '{idx['name']}'
                    """)).fetchone()
                    
                    if exists:
                        log.info(f"✓ 索引 {idx['name']} 已存在")
                        continue
                    
                    log.info(f"创建高级索引: {idx['name']}")
                    log.info(f"说明: {idx['description']}")
                    
                    # 注意：TimescaleDB可能不支持CONCURRENTLY，我们去掉它
                    sql_without_concurrent = idx['sql'].replace('CONCURRENTLY ', '')
                    
                    start_time = time.time()
                    conn.execute(text(sql_without_concurrent))
                    elapsed = time.time() - start_time
                    
                    log.info(f"✓ 索引创建成功 ({elapsed:.2f}秒)")
                    
                except Exception as e:
                    log.error(f"✗ 创建索引失败: {e}")
    
    def create_optimized_queries(self):
        """创建优化后的查询版本"""
        log.info("\n=== 优化查询策略 ===")
        
        optimized_queries = {
            'single_stock_latest_v1': {
                'name': '单股票最新数据 - 版本1（子查询优化）',
                'sql': """
                    SELECT * FROM ts_daily 
                    WHERE ts_code = '600519.SH' 
                    AND trade_date = (
                        SELECT MAX(trade_date) 
                        FROM ts_daily 
                        WHERE ts_code = '600519.SH'
                    )
                """
            },
            'single_stock_latest_v2': {
                'name': '单股票最新数据 - 版本2（窗口函数）',
                'sql': """
                    SELECT * FROM (
                        SELECT *, 
                               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                        FROM ts_daily 
                        WHERE ts_code = '600519.SH'
                        AND trade_date >= CURRENT_DATE - INTERVAL '30 days'
                    ) ranked 
                    WHERE rn = 1
                """
            },
            'market_movers_v1': {
                'name': '市场排行 - 版本1（预计算最新日期）',
                'sql': """
                    WITH latest_date AS (
                        SELECT MAX(trade_date) as max_date FROM ts_daily
                    )
                    SELECT ts_code, close, pct_chg, vol
                    FROM ts_daily, latest_date
                    WHERE trade_date = latest_date.max_date
                    ORDER BY pct_chg DESC 
                    LIMIT 50
                """
            },
            'market_movers_v2': {
                'name': '市场排行 - 版本2（直接指定日期）',
                'sql': """
                    SELECT ts_code, close, pct_chg, vol
                    FROM ts_daily 
                    WHERE trade_date = '2025-07-18'  -- 直接使用已知的最新日期
                    ORDER BY pct_chg DESC 
                    LIMIT 50
                """
            }
        }
        
        with self.engine.connect() as conn:
            for query_id, query_info in optimized_queries.items():
                log.info(f"\n测试: {query_info['name']}")
                
                times = []
                for i in range(3):
                    start_time = time.time()
                    try:
                        result = conn.execute(text(query_info['sql'])).fetchall()
                        elapsed_ms = (time.time() - start_time) * 1000
                        times.append(elapsed_ms)
                    except Exception as e:
                        log.error(f"查询失败: {e}")
                        break
                
                if times:
                    avg_time = sum(times) / len(times)
                    log.info(f"平均耗时: {avg_time:.2f}ms")
                    log.info(f"返回记录: {len(result) if 'result' in locals() else 0} 条")
    
    def create_materialized_view_for_latest_data(self):
        """为最新数据创建物化视图"""
        log.info("\n=== 创建最新数据物化视图 ===")
        
        with self.engine.connect() as conn:
            try:
                # 检查物化视图是否已存在
                exists = conn.execute(text("""
                    SELECT 1 FROM pg_matviews 
                    WHERE matviewname = 'mv_latest_stock_data'
                """)).fetchone()
                
                if exists:
                    log.info("✓ 物化视图 mv_latest_stock_data 已存在")
                    
                    # 刷新物化视图
                    log.info("刷新物化视图...")
                    start_time = time.time()
                    conn.execute(text("REFRESH MATERIALIZED VIEW mv_latest_stock_data"))
                    elapsed = time.time() - start_time
                    log.info(f"✓ 物化视图刷新完成 ({elapsed:.2f}秒)")
                    
                else:
                    log.info("创建最新股票数据物化视图...")
                    start_time = time.time()
                    
                    conn.execute(text("""
                        CREATE MATERIALIZED VIEW mv_latest_stock_data AS
                        SELECT DISTINCT ON (ts_code)
                            ts_code, trade_date, open, high, low, close, vol, amount, pct_chg
                        FROM ts_daily 
                        ORDER BY ts_code, trade_date DESC
                    """))
                    
                    # 创建索引
                    conn.execute(text("""
                        CREATE INDEX idx_mv_latest_stock_data_ts_code 
                        ON mv_latest_stock_data (ts_code)
                    """))
                    
                    conn.execute(text("""
                        CREATE INDEX idx_mv_latest_stock_data_pct_chg 
                        ON mv_latest_stock_data (pct_chg DESC)
                    """))
                    
                    elapsed = time.time() - start_time
                    log.info(f"✓ 物化视图创建完成 ({elapsed:.2f}秒)")
                
                # 测试物化视图查询性能
                log.info("\n测试物化视图查询性能:")
                
                # 单股票最新数据
                start_time = time.time()
                result1 = conn.execute(text("""
                    SELECT * FROM mv_latest_stock_data 
                    WHERE ts_code = '600519.SH'
                """)).fetchall()
                time1 = (time.time() - start_time) * 1000
                log.info(f"单股票查询: {time1:.2f}ms")
                
                # 涨幅排行
                start_time = time.time()
                result2 = conn.execute(text("""
                    SELECT ts_code, close, pct_chg 
                    FROM mv_latest_stock_data 
                    ORDER BY pct_chg DESC 
                    LIMIT 50
                """)).fetchall()
                time2 = (time.time() - start_time) * 1000
                log.info(f"涨幅排行查询: {time2:.2f}ms")
                
            except Exception as e:
                log.error(f"物化视图操作失败: {e}")
    
    def benchmark_all_optimizations(self):
        """对比所有优化方案的性能"""
        log.info("\n=== 全面性能对比 ===")
        
        test_cases = [
            {
                'name': '原始单股票查询',
                'sql': """
                    SELECT * FROM ts_daily 
                    WHERE ts_code = '600519.SH' 
                    ORDER BY trade_date DESC 
                    LIMIT 1
                """
            },
            {
                'name': '物化视图单股票查询',
                'sql': """
                    SELECT * FROM mv_latest_stock_data 
                    WHERE ts_code = '600519.SH'
                """
            },
            {
                'name': '原始市场排行查询',
                'sql': """
                    SELECT ts_code, close, pct_chg 
                    FROM ts_daily 
                    WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily)
                    ORDER BY pct_chg DESC 
                    LIMIT 50
                """
            },
            {
                'name': '物化视图市场排行查询',
                'sql': """
                    SELECT ts_code, close, pct_chg 
                    FROM mv_latest_stock_data 
                    ORDER BY pct_chg DESC 
                    LIMIT 50
                """
            }
        ]
        
        with self.engine.connect() as conn:
            results = {}
            
            for test in test_cases:
                log.info(f"\n测试: {test['name']}")
                
                times = []
                for i in range(3):
                    start_time = time.time()
                    try:
                        result = conn.execute(text(test['sql'])).fetchall()
                        elapsed_ms = (time.time() - start_time) * 1000
                        times.append(elapsed_ms)
                    except Exception as e:
                        log.error(f"查询失败: {e}")
                        break
                
                if times:
                    avg_time = sum(times) / len(times)
                    results[test['name']] = avg_time
                    log.info(f"平均耗时: {avg_time:.2f}ms")
            
            # 性能对比总结
            log.info("\n=== 性能对比总结 ===")
            for name, time_ms in results.items():
                if time_ms < 50:
                    status = "✓ 优秀"
                elif time_ms < 200:
                    status = "✓ 良好"
                elif time_ms < 1000:
                    status = "⚠ 一般"
                else:
                    status = "⚠ 较慢"
                
                log.info(f"{name}: {time_ms:.2f}ms {status}")


def main():
    optimizer = AdvancedQueryOptimizer()
    
    try:
        # 1. 分析慢查询原因
        optimizer.analyze_slow_queries()
        
        # 2. 创建高级索引
        optimizer.create_advanced_indexes()
        
        # 3. 测试优化查询
        optimizer.create_optimized_queries()
        
        # 4. 创建物化视图
        optimizer.create_materialized_view_for_latest_data()
        
        # 5. 全面性能对比
        optimizer.benchmark_all_optimizations()
        
        log.info("\n=== 高级查询优化完成 ===")
        
    except Exception as e:
        log.error(f"优化过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())