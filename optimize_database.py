#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库性能优化脚本

功能：
1. 分析当前数据库性能瓶颈
2. 创建关键索引
3. 优化查询性能
4. 提供性能对比报告

使用方法：
python optimize_database.py --analyze    # 分析性能
python optimize_database.py --optimize   # 执行优化
python optimize_database.py --benchmark  # 性能基准测试
"""

import argparse
import time
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from logger_config import log


class DatabaseOptimizer:
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
    
    def analyze_table_stats(self):
        """分析表统计信息"""
        log.info("=== 数据库表统计分析 ===")
        
        with self.engine.connect() as conn:
            # 获取主要表的统计信息
            tables_to_analyze = [
                'ts_daily', 'factors_exposure', 'financial_indicators', 
                'stock_basic', 'ts_adj_factor'
            ]
            
            for table in tables_to_analyze:
                try:
                    # 表大小和记录数
                    result = conn.execute(text(f"""
                        SELECT 
                            pg_size_pretty(pg_total_relation_size('{table}')) as table_size,
                            pg_size_pretty(pg_relation_size('{table}')) as data_size,
                            pg_size_pretty(pg_total_relation_size('{table}') - pg_relation_size('{table}')) as index_size,
                            (SELECT COUNT(*) FROM {table}) as row_count
                    """)).fetchone()
                    
                    if result:
                        log.info(f"表 {table}:")
                        log.info(f"  - 总大小: {result[0]}")
                        log.info(f"  - 数据大小: {result[1]}")
                        log.info(f"  - 索引大小: {result[2]}")
                        log.info(f"  - 记录数: {result[3]:,}")
                        
                except Exception as e:
                    log.warning(f"分析表 {table} 失败: {e}")
    
    def analyze_slow_queries(self):
        """分析慢查询"""
        log.info("\n=== 慢查询分析 ===")
        
        # 测试常见的慢查询
        slow_queries = [
            {
                'name': '全表统计查询',
                'sql': """
                    SELECT COUNT(*) as total_records,
                           COUNT(DISTINCT ts_code) as unique_stocks,
                           MAX(trade_date) as latest_date
                    FROM ts_daily
                """,
                'expected_time': 1.0
            },
            {
                'name': '因子数据统计',
                'sql': """
                    SELECT COUNT(*) as total_records,
                           COUNT(DISTINCT ts_code) as unique_stocks,
                           COUNT(DISTINCT factor_name) as factors_count,
                           MAX(trade_date) as latest_date
                    FROM factors_exposure
                """,
                'expected_time': 0.5
            },
            {
                'name': '最新交易日查询',
                'sql': """
                    SELECT DISTINCT trade_date 
                    FROM ts_daily 
                    ORDER BY trade_date DESC 
                    LIMIT 10
                """,
                'expected_time': 0.1
            }
        ]
        
        with self.engine.connect() as conn:
            for query in slow_queries:
                start_time = time.time()
                try:
                    result = conn.execute(text(query['sql'])).fetchall()
                    elapsed = time.time() - start_time
                    
                    status = "✓" if elapsed < query['expected_time'] else "⚠"
                    log.info(f"{status} {query['name']}: {elapsed*1000:.2f}ms (期望: <{query['expected_time']*1000:.0f}ms)")
                    
                except Exception as e:
                    log.error(f"✗ {query['name']}: 查询失败 - {e}")
    
    def check_existing_indexes(self):
        """检查现有索引"""
        log.info("\n=== 现有索引分析 ===")
        
        with self.engine.connect() as conn:
            # 查询现有索引
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE schemaname = 'public'
                AND tablename IN ('ts_daily', 'factors_exposure', 'financial_indicators', 'ts_adj_factor')
                ORDER BY tablename, indexname
            """)).fetchall()
            
            current_table = None
            for row in result:
                if row[1] != current_table:
                    current_table = row[1]
                    log.info(f"\n表 {current_table} 的索引:")
                
                log.info(f"  - {row[2]}: {row[3]}")
    
    def create_performance_indexes(self):
        """创建性能优化索引"""
        log.info("\n=== 创建性能优化索引 ===")
        
        # 定义需要创建的索引（过滤掉已有的索引）
        indexes_to_create = [
            {
                'name': 'idx_ts_daily_ts_code',
                'table': 'ts_daily',
                'columns': 'ts_code',
                'description': '股票代码索引，优化按股票查询'
            },
            {
                'name': 'idx_factors_exposure_ts_code',
                'table': 'factors_exposure',
                'columns': 'ts_code',
                'description': '因子数据股票代码索引'
            },
            {
                'name': 'idx_financial_indicators_end_date',
                'table': 'financial_indicators',
                'columns': 'end_date',
                'description': '财务指标日期索引'
            }
        ]
        
        for idx in indexes_to_create:
            try:
                with self.engine.connect() as conn:
                    # 检查索引是否已存在
                    exists = conn.execute(text(f"""
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = '{idx['name']}'
                    """)).fetchone()
                    
                    if exists:
                        log.info(f"✓ 索引 {idx['name']} 已存在")
                        continue
                    
                    # 创建索引（TimescaleDB不支持CONCURRENTLY）
                    log.info(f"创建索引: {idx['name']} ({idx['description']})")
                    start_time = time.time()
                    
                    conn.execute(text(f"""
                        CREATE INDEX {idx['name']} 
                        ON {idx['table']} ({idx['columns']})
                    """))
                    
                    elapsed = time.time() - start_time
                    log.info(f"✓ 索引 {idx['name']} 创建成功 ({elapsed:.2f}秒)")
                    
            except Exception as e:
                log.error(f"✗ 创建索引 {idx['name']} 失败: {e}")
    
    def update_table_statistics(self):
        """更新表统计信息"""
        log.info("\n=== 更新表统计信息 ===")
        
        tables = ['ts_daily', 'factors_exposure', 'financial_indicators', 'ts_adj_factor']
        
        with self.engine.connect() as conn:
            with conn.begin():
                for table in tables:
                    try:
                        log.info(f"更新表 {table} 的统计信息...")
                        conn.execute(text(f"ANALYZE {table}"))
                        log.info(f"✓ 表 {table} 统计信息更新完成")
                    except Exception as e:
                        log.error(f"✗ 更新表 {table} 统计信息失败: {e}")
    
    def benchmark_performance(self):
        """性能基准测试"""
        log.info("\n=== 性能基准测试 ===")
        
        benchmark_queries = [
            {
                'name': '单股票历史数据查询',
                'sql': """
                    SELECT * FROM ts_daily 
                    WHERE ts_code = '600519.SH' 
                    AND trade_date >= '2024-01-01'
                    ORDER BY trade_date DESC
                """,
                'target': 0.05
            },
            {
                'name': '最新交易日全市场数据',
                'sql': """
                    SELECT ts_code, close, pct_chg 
                    FROM ts_daily 
                    WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily)
                    ORDER BY pct_chg DESC 
                    LIMIT 100
                """,
                'target': 0.5
            },
            {
                'name': '因子数据查询',
                'sql': """
                    SELECT * FROM factors_exposure 
                    WHERE factor_name = 'pe_ttm' 
                    AND trade_date >= '2024-01-01'
                    ORDER BY trade_date DESC, factor_value DESC
                    LIMIT 1000
                """,
                'target': 0.1
            }
        ]
        
        with self.engine.connect() as conn:
            for query in benchmark_queries:
                times = []
                for i in range(3):  # 运行3次取平均值
                    start_time = time.time()
                    try:
                        result = conn.execute(text(query['sql'])).fetchall()
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                    except Exception as e:
                        log.error(f"基准测试失败: {query['name']} - {e}")
                        break
                
                if times:
                    avg_time = sum(times) / len(times)
                    status = "✓" if avg_time < query['target'] else "⚠"
                    log.info(f"{status} {query['name']}: {avg_time*1000:.2f}ms (目标: <{query['target']*1000:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(description="数据库性能优化工具")
    parser.add_argument("--analyze", action="store_true", help="分析数据库性能")
    parser.add_argument("--optimize", action="store_true", help="执行优化操作")
    parser.add_argument("--benchmark", action="store_true", help="性能基准测试")
    parser.add_argument("--all", action="store_true", help="执行所有操作")
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.optimize, args.benchmark, args.all]):
        parser.print_help()
        return
    
    optimizer = DatabaseOptimizer()
    
    try:
        if args.analyze or args.all:
            optimizer.analyze_table_stats()
            optimizer.check_existing_indexes()
            optimizer.analyze_slow_queries()
        
        if args.optimize or args.all:
            optimizer.create_performance_indexes()
            optimizer.update_table_statistics()
        
        if args.benchmark or args.all:
            optimizer.benchmark_performance()
            
        log.info("\n=== 数据库优化完成 ===")
        
    except Exception as e:
        log.error(f"数据库优化过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())