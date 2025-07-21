#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库索引管理器

负责数据库索引的创建、删除、监控和优化
这是数据库性能优化的核心组件之一
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import pandas as pd

import config
from logger_config import log

# Constants
HIGH_EFFICIENCY_THRESHOLD = 100
LOW_EFFICIENCY_THRESHOLD = 1
MAX_REPORT_ITEMS = 5
MAX_UNUSED_INDEXES_REPORT = 10
BYTES_TO_MB = 1024 * 1024

# Supported tables for index operations
SUPPORTED_TABLES = {'ts_daily', 'factors_exposure', 'financial_indicators', 'stock_basic'}


@dataclass
class IndexConfig:
    """索引配置数据类"""
    name: str
    table: str
    columns: str
    description: str
    usage: str
    is_unique: bool = False
    is_partial: bool = False
    where_clause: Optional[str] = None
    priority: int = 1  # 1=高优先级, 2=中优先级, 3=低优先级


@dataclass
class IndexStats:
    """索引统计信息数据类"""
    name: str
    table: str
    size_mb: float
    scans: int
    tup_read: int
    tup_fetch: int
    last_used: Optional[datetime]
    efficiency_score: float


class IndexManager:
    """数据库索引管理器"""
    
    def __init__(self, db_url: str = None):
        """初始化索引管理器"""
        self.db_url = db_url or config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        self.index_configs = self._load_index_configs()
    
    def _validate_table_name(self, table_name: str) -> bool:
        """验证表名是否在支持的表列表中"""
        return table_name in SUPPORTED_TABLES
    
    def _validate_index_name(self, index_name: str) -> bool:
        """验证索引名称格式"""
        if not index_name or not isinstance(index_name, str):
            return False
        # 索引名应该只包含字母、数字和下划线
        return index_name.replace('_', '').replace('-', '').isalnum()
        
    def _load_index_configs(self) -> Dict[str, IndexConfig]:
        """加载索引配置"""
        # 核心业务索引配置
        configs = {
            # 单股票查询优化索引
            'idx_ts_daily_stock_lookup': IndexConfig(
                name='idx_ts_daily_stock_lookup',
                table='ts_daily',
                columns='ts_code, trade_date DESC',
                description='单股票时间序列查询优化',
                usage='单股票历史数据查询、最新价格查询',
                priority=1
            ),
            
            # 市场排行查询索引
            'idx_ts_daily_market_snapshot': IndexConfig(
                name='idx_ts_daily_market_snapshot',
                table='ts_daily',
                columns='trade_date, pct_chg DESC',
                description='市场快照查询优化',
                usage='涨跌幅排行、市场概览',
                priority=1
            ),
            
            # 因子查询专用索引
            'idx_factors_exposure_lookup': IndexConfig(
                name='idx_factors_exposure_lookup',
                table='factors_exposure',
                columns='factor_name, trade_date DESC, factor_value DESC',
                description='因子查询优化',
                usage='因子排名、因子历史',
                priority=1
            ),
            
            # 财务数据查询索引
            'idx_financial_indicators_lookup': IndexConfig(
                name='idx_financial_indicators_lookup',
                table='financial_indicators',
                columns='ts_code, end_date DESC',
                description='财务指标查询优化',
                usage='财务数据时间序列查询',
                priority=2
            ),
            
            # 热点数据部分索引（最近数据）
            'idx_ts_daily_recent': IndexConfig(
                name='idx_ts_daily_recent',
                table='ts_daily',
                columns='ts_code, trade_date DESC',
                description='热点数据部分索引',
                usage='最近数据快速查询',
                is_partial=True,
                where_clause="trade_date >= '2024-01-01'",
                priority=2
            ),
            
            # 成交量排序索引
            'idx_ts_daily_volume': IndexConfig(
                name='idx_ts_daily_volume',
                table='ts_daily',
                columns='trade_date, vol DESC',
                description='成交量排序索引',
                usage='成交量排行查询',
                priority=3
            ),
            
            # 股票基本信息索引
            'idx_stock_basic_industry': IndexConfig(
                name='idx_stock_basic_industry',
                table='stock_basic',
                columns='industry, ts_code',
                description='行业分类索引',
                usage='按行业查询股票',
                priority=2
            )
        }
        
        return configs
    
    def check_table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """), {"table_name": table_name}).scalar()
            return result
    
    def check_index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = :index_name
                )
            """), {"index_name": index_name}).scalar()
            return result
    
    def create_index(self, config: IndexConfig, force: bool = False) -> bool:
        """创建单个索引"""
        try:
            # 验证输入
            if not self._validate_table_name(config.table):
                log.error(f"不支持的表名: {config.table}")
                return False
            
            if not self._validate_index_name(config.name):
                log.error(f"无效的索引名称: {config.name}")
                return False
            
            # 检查表是否存在
            if not self.check_table_exists(config.table):
                log.warning(f"表 {config.table} 不存在，跳过索引 {config.name}")
                return False
            
            # 检查索引是否已存在
            if self.check_index_exists(config.name) and not force:
                log.info(f"索引 {config.name} 已存在")
                return True
            
            log.info(f"创建索引: {config.name}")
            log.info(f"  表: {config.table}")
            log.info(f"  列: {config.columns}")
            log.info(f"  用途: {config.usage}")
            
            start_time = time.time()
            
            with self.engine.connect() as conn:
                with conn.begin():
                    # 如果强制重建，先删除现有索引
                    if force and self.check_index_exists(config.name):
                        conn.execute(text(f"DROP INDEX IF EXISTS {config.name}"))
                    
                    # 构建CREATE INDEX语句
                    create_sql = f"CREATE INDEX {config.name} ON {config.table} ({config.columns})"
                    if config.where_clause:
                        create_sql += f" WHERE {config.where_clause}"
                    
                    conn.execute(text(create_sql))
            
            elapsed = time.time() - start_time
            log.info(f"✓ 索引 {config.name} 创建成功 ({elapsed:.2f}秒)")
            return True
            
        except Exception as e:
            log.error(f"✗ 创建索引 {config.name} 失败: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """删除索引"""
        try:
            if not self._validate_index_name(index_name):
                log.error(f"无效的索引名称: {index_name}")
                return False
                
            if not self.check_index_exists(index_name):
                log.info(f"索引 {index_name} 不存在")
                return True
            
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text(f"DROP INDEX {index_name}"))
            
            log.info(f"✓ 索引 {index_name} 删除成功")
            return True
            
        except Exception as e:
            log.error(f"✗ 删除索引 {index_name} 失败: {e}")
            return False
    
    def get_index_stats(self) -> List[IndexStats]:
        """获取索引使用统计"""
        stats = []
        
        try:
            with self.engine.connect() as conn:
                # 获取索引统计信息
                result = conn.execute(text("""
                    SELECT 
                        s.indexrelname as index_name,
                        s.relname as table_name,
                        pg_size_pretty(pg_relation_size(s.indexrelid)) as size,
                        pg_relation_size(s.indexrelid) as size_bytes,
                        COALESCE(s.idx_scan, 0) as scans,
                        COALESCE(s.idx_tup_read, 0) as tup_read,
                        COALESCE(s.idx_tup_fetch, 0) as tup_fetch
                    FROM pg_stat_user_indexes s
                    WHERE s.schemaname = 'public'
                    AND s.relname IN ('ts_daily', 'factors_exposure', 'financial_indicators', 'stock_basic')
                    ORDER BY COALESCE(s.idx_scan, 0) DESC
                """)).fetchall()
                
                for row in result:
                    # 计算效率分数
                    scans = row[4] or 0
                    size_mb = (row[3] or 0) / BYTES_TO_MB
                    
                    if size_mb > 0:
                        efficiency_score = scans / size_mb  # 扫描次数/MB
                    else:
                        efficiency_score = 0
                    
                    stats.append(IndexStats(
                        name=row[0],
                        table=row[1],
                        size_mb=size_mb,
                        scans=scans,
                        tup_read=row[5] or 0,
                        tup_fetch=row[6] or 0,
                        last_used=None,  # PostgreSQL不直接提供最后使用时间
                        efficiency_score=efficiency_score
                    ))
                    
        except Exception as e:
            log.error(f"获取索引统计失败: {e}")
        
        return stats
    
    def analyze_index_usage(self) -> Dict[str, any]:
        """分析索引使用情况"""
        stats = self.get_index_stats()
        
        if not stats:
            return {"error": "无法获取索引统计信息"}
        
        # 分析结果
        total_indexes = len(stats)
        used_indexes = len([s for s in stats if s.scans > 0])
        unused_indexes = [s for s in stats if s.scans == 0]
        high_efficiency = [s for s in stats if s.efficiency_score > HIGH_EFFICIENCY_THRESHOLD]
        low_efficiency = [s for s in stats if s.efficiency_score < LOW_EFFICIENCY_THRESHOLD and s.scans > 0]
        
        total_size_mb = sum(s.size_mb for s in stats)
        
        analysis = {
            "summary": {
                "total_indexes": total_indexes,
                "used_indexes": used_indexes,
                "unused_indexes": len(unused_indexes),
                "usage_rate": f"{used_indexes/total_indexes*100:.1f}%" if total_indexes > 0 else "0%",
                "total_size_mb": f"{total_size_mb:.2f}"
            },
            "high_efficiency_indexes": [
                {"name": s.name, "table": s.table, "scans": s.scans, "efficiency": f"{s.efficiency_score:.2f}"}
                for s in high_efficiency[:MAX_REPORT_ITEMS]
            ],
            "low_efficiency_indexes": [
                {"name": s.name, "table": s.table, "scans": s.scans, "efficiency": f"{s.efficiency_score:.2f}"}
                for s in low_efficiency[:MAX_REPORT_ITEMS]
            ],
            "unused_indexes": [
                {"name": s.name, "table": s.table, "size_mb": f"{s.size_mb:.2f}"}
                for s in unused_indexes[:MAX_UNUSED_INDEXES_REPORT]
            ]
        }
        
        return analysis
    
    def create_all_indexes(self, priority_filter: Optional[int] = None) -> Dict[str, bool]:
        """创建所有配置的索引"""
        results = {}
        
        configs_to_create = self.index_configs.values()
        if priority_filter:
            configs_to_create = [c for c in configs_to_create if c.priority <= priority_filter]
        
        # 按优先级排序
        configs_to_create = sorted(configs_to_create, key=lambda x: x.priority)
        
        log.info(f"开始创建 {len(configs_to_create)} 个索引...")
        
        for config in configs_to_create:
            results[config.name] = self.create_index(config)
        
        # 统计结果
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        log.info(f"索引创建完成: {success_count}/{total_count} 成功")
        
        return results
    
    def optimize_indexes(self) -> Dict[str, any]:
        """索引优化建议"""
        analysis = self.analyze_index_usage()
        recommendations = []
        
        # 建议删除未使用的索引
        if analysis.get("unused_indexes"):
            recommendations.append({
                "type": "删除未使用索引",
                "description": f"发现 {len(analysis['unused_indexes'])} 个未使用的索引，建议删除以节省空间",
                "indexes": [idx["name"] for idx in analysis["unused_indexes"][:MAX_REPORT_ITEMS]]
            })
        
        # 建议重建低效索引
        if analysis.get("low_efficiency_indexes"):
            recommendations.append({
                "type": "重建低效索引",
                "description": f"发现 {len(analysis['low_efficiency_indexes'])} 个低效索引，建议重建或优化",
                "indexes": [idx["name"] for idx in analysis["low_efficiency_indexes"]]
            })
        
        # 检查缺失的关键索引
        missing_indexes = []
        for config in self.index_configs.values():
            if config.priority == 1 and not self.check_index_exists(config.name):
                missing_indexes.append(config.name)
        
        if missing_indexes:
            recommendations.append({
                "type": "创建缺失的关键索引",
                "description": f"发现 {len(missing_indexes)} 个缺失的高优先级索引",
                "indexes": missing_indexes
            })
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def maintenance_report(self) -> str:
        """生成维护报告"""
        optimization = self.optimize_indexes()
        
        report_lines = [
            "=== 数据库索引维护报告 ===",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "索引使用统计:",
            f"  总索引数: {optimization['analysis']['summary']['total_indexes']}",
            f"  已使用: {optimization['analysis']['summary']['used_indexes']}",
            f"  未使用: {optimization['analysis']['summary']['unused_indexes']}",
            f"  使用率: {optimization['analysis']['summary']['usage_rate']}",
            f"  总大小: {optimization['analysis']['summary']['total_size_mb']} MB",
            ""
        ]
        
        if optimization["recommendations"]:
            report_lines.append("优化建议:")
            for i, rec in enumerate(optimization["recommendations"], 1):
                report_lines.append(f"  {i}. {rec['type']}")
                report_lines.append(f"     {rec['description']}")
                if rec["indexes"]:
                    report_lines.append(f"     涉及索引: {', '.join(rec['indexes'][:3])}")
                    if len(rec["indexes"]) > 3:
                        report_lines.append(f"     等 {len(rec['indexes'])} 个索引")
                report_lines.append("")
        else:
            report_lines.append("✓ 所有索引状态良好，无需优化")
        
        return "\n".join(report_lines)


def main():
    """主函数 - 用于测试和维护"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库索引管理器")
    parser.add_argument("--create", action="store_true", help="创建所有索引")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3], help="只创建指定优先级的索引")
    parser.add_argument("--analyze", action="store_true", help="分析索引使用情况")
    parser.add_argument("--report", action="store_true", help="生成维护报告")
    parser.add_argument("--drop", type=str, help="删除指定索引")
    
    args = parser.parse_args()
    
    manager = IndexManager()
    
    try:
        if args.create:
            results = manager.create_all_indexes(priority_filter=args.priority)
            print(f"索引创建结果: {results}")
            
        elif args.analyze:
            analysis = manager.analyze_index_usage()
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
            
        elif args.report:
            report = manager.maintenance_report()
            print(report)
            
        elif args.drop:
            success = manager.drop_index(args.drop)
            print(f"删除索引 {args.drop}: {'成功' if success else '失败'}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        log.error(f"执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())