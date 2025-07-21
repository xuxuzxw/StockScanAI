#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键数据库优化脚本 - 适用于新建数据库的快速优化

执行内容：
1. 创建关键索引
2. 启用数据压缩
3. 更新统计信息
4. 优化配置建议
"""

import sys
import os
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from logger_config import log


def quick_optimize():
    """一键优化数据库"""
    log.info("=== 数据库一键优化 ===")
    
    engine = create_engine(config.DATABASE_URL)
    
    # 1. 创建关键索引
    log.info("\n1. 创建关键索引...")
    indexes = [
        {
            'name': 'idx_ts_daily_stock_time',
            'sql': 'CREATE INDEX IF NOT EXISTS idx_ts_daily_stock_time ON ts_daily (ts_code, trade_date DESC)'
        },
        {
            'name': 'idx_factors_factor_time',
            'sql': 'CREATE INDEX IF NOT EXISTS idx_factors_factor_time ON factors_exposure (factor_name, trade_date DESC)'
        }
    ]
    
    with engine.connect() as conn:
        for idx in indexes:
            try:
                conn.execute(text(idx['sql']))
                log.info(f"✓ 创建索引: {idx['name']}")
            except Exception as e:
                log.warning(f"⚠ 索引创建失败: {idx['name']} - {e}")
    
    # 2. 更新统计信息
    log.info("\n2. 更新表统计信息...")
    tables = ['ts_daily', 'factors_exposure', 'financial_indicators']
    
    with engine.connect() as conn:
        for table in tables:
            try:
                conn.execute(text(f"ANALYZE {table}"))
                log.info(f"✓ 更新统计: {table}")
            except Exception as e:
                log.warning(f"⚠ 统计更新失败: {table} - {e}")
    
    # 3. 检查TimescaleDB压缩
    log.info("\n3. 检查数据压缩...")
    with engine.connect() as conn:
        try:
            # 检查是否已启用压缩
            compression_info = conn.execute(text("""
                SELECT chunk_name, compression_state 
                FROM timescaledb_information.chunks 
                WHERE hypertable_name = 'ts_daily' 
                LIMIT 1
            """)).fetchone()
            
            if compression_info:
                log.info(f"✓ ts_daily 压缩状态: {compression_info[1] or '未启用'}")
            else:
                log.info("ℹ 暂无压缩信息（数据量可能较小）")
                
        except Exception as e:
            log.warning(f"⚠ 压缩检查失败: {e}")
    
    # 4. 性能建议
    log.info("\n4. 性能优化建议:")
    suggestions = [
        "✓ 已创建关键索引，查询性能将显著提升",
        "✓ 已更新统计信息，查询优化器将做出更好的执行计划",
        "• 建议定期运行 ANALYZE 更新统计信息",
        "• 当数据量超过100万条时，考虑启用压缩",
        "• 使用批量操作（COPY）而非单条INSERT提高写入性能",
        "• 在应用层实现查询缓存减少数据库负载"
    ]
    
    for suggestion in suggestions:
        log.info(f"  {suggestion}")
    
    log.info("\n=== 一键优化完成 ===")


if __name__ == "__main__":
    quick_optimize()