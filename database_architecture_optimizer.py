#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库架构优化器 - 针对新建数据库的全面优化方案

优化策略：
1. 数据分区策略 - 基于时间和股票代码
2. 索引策略优化 - 最小化但最有效的索引
3. 数据压缩和存储优化
4. 查询模式优化
5. 批量操作优化
6. 内存和缓存配置
"""

import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from logger_config import log


class DatabaseArchitectureOptimizer:
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
    
    def analyze_data_patterns(self):
        """分析数据访问模式，为优化提供依据"""
        log.info("=== 数据访问模式分析 ===")
        
        with self.engine.connect() as conn:
            # 1. 分析数据分布
            log.info("\n1. 数据时间分布分析:")
            try:
                date_range = conn.execute(text("""
                    SELECT 
                        MIN(trade_date) as earliest_date,
                        MAX(trade_date) as latest_date,
                        COUNT(DISTINCT trade_date) as trading_days,
                        COUNT(*) as total_records
                    FROM ts_daily
                """)).fetchone()
                
                if date_range and date_range[0]:
                    log.info(f"  - 数据时间范围: {date_range[0]} 到 {date_range[1]}")
                    log.info(f"  - 交易日数量: {date_range[2]:,} 天")
                    log.info(f"  - 总记录数: {date_range[3]:,}")
                    log.info(f"  - 平均每日记录: {date_range[3]/date_range[2]:.0f}")
                
            except Exception as e:
                log.warning(f"数据分布分析失败: {e}")
            
            # 2. 分析股票分布
            log.info("\n2. 股票数据分布分析:")
            try:
                stock_stats = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT ts_code) as total_stocks,
                        AVG(stock_records) as avg_records_per_stock,
                        MIN(stock_records) as min_records,
                        MAX(stock_records) as max_records
                    FROM (
                        SELECT ts_code, COUNT(*) as stock_records
                        FROM ts_daily 
                        GROUP BY ts_code
                    ) stock_counts
                """)).fetchone()
                
                if stock_stats:
                    log.info(f"  - 股票总数: {stock_stats[0]:,}")
                    log.info(f"  - 平均每股记录数: {stock_stats[1]:.0f}")
                    log.info(f"  - 记录数范围: {stock_stats[2]} - {stock_stats[3]}")
                
            except Exception as e:
                log.warning(f"股票分布分析失败: {e}")
            
            # 3. 分析查询热点
            log.info("\n3. 数据访问热点分析:")
            try:
                # 最近30天的数据访问频率更高
                recent_data = conn.execute(text("""
                    SELECT COUNT(*) as recent_records
                    FROM ts_daily 
                    WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
                """)).scalar()
                
                log.info(f"  - 最近30天记录数: {recent_data:,}")
                log.info(f"  - 热点数据比例: {recent_data/date_range[3]*100:.1f}%")
                
            except Exception as e:
                log.warning(f"热点分析失败: {e}")
    
    def optimize_table_structure(self):
        """优化表结构和分区策略"""
        log.info("\n=== 表结构优化 ===")
        
        optimizations = [
            {
                'name': '启用TimescaleDB压缩',
                'description': '对历史数据启用压缩以节省存储空间',
                'sql': """
                    ALTER TABLE ts_daily SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'ts_code',
                        timescaledb.compress_orderby = 'trade_date DESC'
                    )
                """,
                'check_sql': """
                    SELECT compression_state FROM timescaledb_information.chunks 
                    WHERE hypertable_name = 'ts_daily' LIMIT 1
                """
            },
            {
                'name': '设置数据保留策略',
                'description': '自动清理过期数据',
                'sql': """
                    SELECT add_retention_policy('ts_daily', INTERVAL '5 years')
                """,
                'check_sql': """
                    SELECT * FROM timescaledb_information.drop_chunks_policies 
                    WHERE hypertable = 'ts_daily'
                """
            }
        ]
        
        with self.engine.connect() as conn:
            for opt in optimizations:
                try:
                    # 检查是否已经应用
                    if opt.get('check_sql'):
                        existing = conn.execute(text(opt['check_sql'])).fetchone()
                        if existing:
                            log.info(f"✓ {opt['name']} 已启用")
                            continue
                    
                    log.info(f"应用优化: {opt['name']}")
                    conn.execute(text(opt['sql']))
                    log.info(f"✓ {opt['name']} 应用成功")
                    
                except Exception as e:
                    log.warning(f"⚠ {opt['name']} 应用失败: {e}")
    
    def create_optimal_indexes(self):
        """创建最优索引策略"""
        log.info("\n=== 最优索引策略 ===")
        
        # 基于实际查询模式的索引策略
        index_strategy = [
            {
                'name': 'idx_ts_daily_stock_lookup',
                'table': 'ts_daily',
                'columns': 'ts_code, trade_date DESC',
                'description': '单股票时间序列查询优化',
                'usage': '单股票历史数据查询、最新价格查询'
            },
            {
                'name': 'idx_ts_daily_market_snapshot',
                'table': 'ts_daily', 
                'columns': 'trade_date, pct_chg DESC',
                'description': '市场快照查询优化',
                'usage': '涨跌幅排行、市场概览'
            },
            {
                'name': 'idx_factors_exposure_lookup',
                'table': 'factors_exposure',
                'columns': 'factor_name, trade_date DESC, factor_value DESC',
                'description': '因子查询优化',
                'usage': '因子排名、因子历史'
            }
        ]
        
        with self.engine.connect() as conn:
            for idx in index_strategy:
                try:
                    # 检查表是否存在
                    table_exists = conn.execute(text(f"""
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = '{idx['table']}'
                    """)).fetchone()
                    
                    if not table_exists:
                        log.info(f"⚠ 表 {idx['table']} 不存在，跳过索引 {idx['name']}")
                        continue
                    
                    # 检查索引是否已存在
                    exists = conn.execute(text(f"""
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = '{idx['name']}'
                    """)).fetchone()
                    
                    if exists:
                        log.info(f"✓ 索引 {idx['name']} 已存在")
                        continue
                    
                    log.info(f"创建索引: {idx['name']}")
                    log.info(f"  用途: {idx['usage']}")
                    
                    start_time = time.time()
                    conn.execute(text(f"""
                        CREATE INDEX {idx['name']} 
                        ON {idx['table']} ({idx['columns']})
                    """))
                    
                    elapsed = time.time() - start_time
                    log.info(f"✓ 索引创建成功 ({elapsed:.2f}秒)")
                    
                except Exception as e:
                    log.error(f"✗ 创建索引 {idx['name']} 失败: {e}")
    
    def optimize_database_settings(self):
        """优化数据库配置参数"""
        log.info("\n=== 数据库配置优化 ===")
        
        # PostgreSQL性能优化参数
        performance_settings = [
            {
                'name': 'shared_buffers',
                'recommended': '256MB',
                'description': '共享缓冲区大小'
            },
            {
                'name': 'effective_cache_size', 
                'recommended': '1GB',
                'description': '系统缓存大小估计'
            },
            {
                'name': 'work_mem',
                'recommended': '4MB',
                'description': '查询工作内存'
            },
            {
                'name': 'maintenance_work_mem',
                'recommended': '64MB', 
                'description': '维护操作内存'
            },
            {
                'name': 'checkpoint_completion_target',
                'recommended': '0.9',
                'description': '检查点完成目标'
            }
        ]
        
        with self.engine.connect() as conn:
            log.info("当前数据库配置:")
            for setting in performance_settings:
                try:
                    current_value = conn.execute(text(f"""
                        SELECT setting, unit FROM pg_settings 
                        WHERE name = '{setting['name']}'
                    """)).fetchone()
                    
                    if current_value:
                        unit = current_value[1] or ''
                        log.info(f"  {setting['name']}: {current_value[0]}{unit} (推荐: {setting['recommended']})")
                    
                except Exception as e:
                    log.warning(f"无法获取 {setting['name']} 配置: {e}")
            
            log.info("\n注意: 数据库配置需要在postgresql.conf中修改并重启数据库")
    
    def create_efficient_queries(self):
        """创建高效查询模板"""
        log.info("\n=== 高效查询模板 ===")
        
        query_templates = {
            'single_stock_latest': {
                'description': '单股票最新数据',
                'sql': """
                    SELECT * FROM ts_daily 
                    WHERE ts_code = :ts_code 
                    ORDER BY trade_date DESC 
                    LIMIT 1
                """,
                'optimization': '利用复合索引 (ts_code, trade_date)'
            },
            'market_movers': {
                'description': '市场涨跌幅排行',
                'sql': """
                    SELECT ts_code, close, pct_chg, vol
                    FROM ts_daily 
                    WHERE trade_date = :trade_date
                    ORDER BY pct_chg DESC 
                    LIMIT :limit_count
                """,
                'optimization': '利用复合索引 (trade_date, pct_chg)'
            },
            'stock_history_range': {
                'description': '股票历史区间数据',
                'sql': """
                    SELECT trade_date, close, vol, pct_chg
                    FROM ts_daily 
                    WHERE ts_code = :ts_code 
                    AND trade_date BETWEEN :start_date AND :end_date
                    ORDER BY trade_date DESC
                """,
                'optimization': '利用主键索引范围扫描'
            },
            'factor_ranking': {
                'description': '因子排名查询',
                'sql': """
                    SELECT ts_code, factor_value
                    FROM factors_exposure 
                    WHERE factor_name = :factor_name 
                    AND trade_date = :trade_date
                    ORDER BY factor_value DESC 
                    LIMIT :limit_count
                """,
                'optimization': '利用因子专用索引'
            }
        }
        
        log.info("推荐的高效查询模板:")
        for name, template in query_templates.items():
            log.info(f"\n{name}: {template['description']}")
            log.info(f"  优化策略: {template['optimization']}")
    
    def benchmark_optimized_performance(self):
        """基准测试优化后的性能"""
        log.info("\n=== 优化后性能基准测试 ===")
        
        test_queries = [
            {
                'name': '单股票最新数据',
                'sql': """
                    SELECT * FROM ts_daily 
                    WHERE ts_code = '600519.SH' 
                    ORDER BY trade_date DESC 
                    LIMIT 1
                """,
                'target_ms': 10
            },
            {
                'name': '最新交易日涨幅前100',
                'sql': """
                    SELECT ts_code, close, pct_chg 
                    FROM ts_daily 
                    WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily)
                    ORDER BY pct_chg DESC 
                    LIMIT 100
                """,
                'target_ms': 100
            },
            {
                'name': '股票近一年历史',
                'sql': """
                    SELECT trade_date, close, vol 
                    FROM ts_daily 
                    WHERE ts_code = '600519.SH' 
                    AND trade_date >= CURRENT_DATE - INTERVAL '1 year'
                    ORDER BY trade_date DESC
                """,
                'target_ms': 50
            }
        ]
        
        with self.engine.connect() as conn:
            for test in test_queries:
                times = []
                for i in range(3):  # 运行3次取平均
                    start_time = time.time()
                    try:
                        result = conn.execute(text(test['sql'])).fetchall()
                        elapsed_ms = (time.time() - start_time) * 1000
                        times.append(elapsed_ms)
                    except Exception as e:
                        log.error(f"查询失败: {test['name']} - {e}")
                        break
                
                if times:
                    avg_time = sum(times) / len(times)
                    status = "✓" if avg_time < test['target_ms'] else "⚠"
                    log.info(f"{status} {test['name']}: {avg_time:.2f}ms (目标: <{test['target_ms']}ms)")
    
    def generate_optimization_report(self):
        """生成优化报告和建议"""
        log.info("\n=== 数据库优化报告 ===")
        
        recommendations = [
            "1. 数据分区策略:",
            "   - TimescaleDB已自动按时间分区",
            "   - 考虑按股票代码进一步分区（如果单表超过1000万记录）",
            "",
            "2. 索引策略:",
            "   - 保持最小但最有效的索引集合",
            "   - 定期监控索引使用情况，删除未使用的索引",
            "",
            "3. 数据压缩:",
            "   - 启用TimescaleDB压缩节省70-90%存储空间",
            "   - 对超过7天的历史数据自动压缩",
            "",
            "4. 查询优化:",
            "   - 使用参数化查询避免SQL注入和提高缓存命中",
            "   - 避免SELECT *，只查询需要的列",
            "   - 合理使用LIMIT限制结果集大小",
            "",
            "5. 批量操作:",
            "   - 使用COPY或批量INSERT提高数据导入效率",
            "   - 在非交易时间进行大批量数据更新",
            "",
            "6. 监控和维护:",
            "   - 定期运行ANALYZE更新统计信息",
            "   - 监控慢查询日志",
            "   - 定期检查表膨胀和碎片化"
        ]
        
        for rec in recommendations:
            log.info(rec)


def main():
    optimizer = DatabaseArchitectureOptimizer()
    
    try:
        # 1. 分析数据模式
        optimizer.analyze_data_patterns()
        
        # 2. 优化表结构
        optimizer.optimize_table_structure()
        
        # 3. 创建最优索引
        optimizer.create_optimal_indexes()
        
        # 4. 检查数据库配置
        optimizer.optimize_database_settings()
        
        # 5. 提供查询模板
        optimizer.create_efficient_queries()
        
        # 6. 性能基准测试
        optimizer.benchmark_optimized_performance()
        
        # 7. 生成优化报告
        optimizer.generate_optimization_report()
        
        log.info("\n=== 数据库架构优化完成 ===")
        
    except Exception as e:
        log.error(f"优化过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())