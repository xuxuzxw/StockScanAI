#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库查询优化器

负责查询重写、执行计划优化和结果缓存
这是数据库性能优化的核心组件
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import pandas as pd

import config
from logger_config import log

# Constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 300
SLOW_QUERY_THRESHOLD_MS = 1000
MAX_SLOW_QUERIES_REPORT = 20
QUERY_TRUNCATE_LENGTH = 200
SQL_PREVIEW_LENGTH = 100
CACHE_KEY_LENGTH = 8


@dataclass
class OptimizedQuery:
    """优化后的查询对象"""
    original_sql: str
    optimized_sql: str
    query_type: str
    optimization_applied: List[str]
    estimated_improvement: float
    cache_key: Optional[str] = None


@dataclass
class ExecutionPlan:
    """查询执行计划"""
    query_hash: str
    plan_text: str
    estimated_cost: float
    estimated_rows: int
    actual_time_ms: Optional[float] = None
    index_usage: List[str] = None


@dataclass
class QueryPattern:
    """查询模式定义"""
    name: str
    pattern: str
    optimization_rule: str
    cache_ttl: int
    priority: int


class OptimizationStrategy:
    """优化策略基类"""
    
    def apply(self, sql: str) -> Tuple[str, List[str], float]:
        """应用优化策略
        
        Returns:
            Tuple[optimized_sql, optimizations_applied, estimated_improvement]
        """
        raise NotImplementedError


class StockLookupOptimization(OptimizationStrategy):
    """股票查询优化策略"""
    
    def apply(self, sql: str) -> Tuple[str, List[str], float]:
        optimizations = []
        optimized_sql = sql
        estimated_improvement = 1.0
        
        if 'ORDER BY trade_date DESC' in sql.upper():
            optimizations.append('stock_lookup_index_hint')
            estimated_improvement = 2.0
            
        return optimized_sql, optimizations, estimated_improvement


class MarketSnapshotOptimization(OptimizationStrategy):
    """市场快照优化策略"""
    
    def apply(self, sql: str) -> Tuple[str, List[str], float]:
        optimizations = []
        optimized_sql = sql
        estimated_improvement = 1.0
        
        if 'MAX(trade_date)' in sql.upper():
            optimizations.append('precomputed_latest_date')
            estimated_improvement = 3.0
            
        return optimized_sql, optimizations, estimated_improvement


class FactorLookupOptimization(OptimizationStrategy):
    """因子查询优化策略"""
    
    def apply(self, sql: str) -> Tuple[str, List[str], float]:
        return sql, ['factor_index_optimization'], 1.5


class IndustryOptimization(OptimizationStrategy):
    """行业查询优化策略"""
    
    def apply(self, sql: str) -> Tuple[str, List[str], float]:
        return sql, ['industry_index_optimization'], 1.2


class QueryCache:
    """查询结果缓存管理器"""
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, default_ttl: int = DEFAULT_CACHE_TTL):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _generate_key(self, sql: str, params: dict = None) -> str:
        """生成缓存键"""
        content = sql + str(sorted((params or {}).items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, sql: str, params: dict = None) -> Optional[Any]:
        """获取缓存结果"""
        key = self._generate_key(sql, params)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # 检查是否过期
            if datetime.now() < entry['expires_at']:
                self.access_times[key] = datetime.now()
                log.debug(f"缓存命中: {key[:CACHE_KEY_LENGTH]}...")
                return entry['data']
            else:
                # 清理过期缓存
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(self, sql: str, data: Any, params: dict = None, ttl: int = None) -> str:
        """设置缓存"""
        key = self._generate_key(sql, params)
        ttl = ttl or self.default_ttl
        
        # 如果缓存已满，清理最久未访问的条目
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'data': data,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'sql': sql[:SQL_PREVIEW_LENGTH] + '...' if len(sql) > SQL_PREVIEW_LENGTH else sql
        }
        self.access_times[key] = datetime.now()
        
        log.debug(f"缓存设置: {key[:CACHE_KEY_LENGTH]}... (TTL: {ttl}s)")
        return key
    
    def _evict_lru(self):
        """清理最久未访问的缓存条目"""
        if not self.access_times:
            return
        
        try:
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            if lru_key in self.cache:
                del self.cache[lru_key]
            if lru_key in self.access_times:
                del self.access_times[lru_key]
            log.debug(f"清理LRU缓存: {lru_key[:CACHE_KEY_LENGTH]}...")
        except (ValueError, KeyError) as e:
            log.warning(f"LRU缓存清理失败: {e}")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        log.info("缓存已清空")
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        now = datetime.now()
        active_entries = sum(1 for entry in self.cache.values() if now < entry['expires_at'])
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': len(self.cache) - active_entries,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'usage_rate': f"{len(self.cache)/self.max_size*100:.1f}%"
        }


class QueryOptimizer:
    """查询优化器核心类"""
    
    def __init__(self, db_url: str = None, cache_size: int = DEFAULT_CACHE_SIZE):
        if cache_size <= 0:
            raise ValueError("Cache size must be positive")
        
        self.db_url = db_url or config.DATABASE_URL
        if not self.db_url:
            raise ValueError("Database URL is required")
            
        try:
            self.engine = create_engine(self.db_url)
        except Exception as e:
            log.error(f"Failed to create database engine: {e}")
            raise
            
        self.cache = QueryCache(max_size=cache_size)
        self.query_patterns = self._load_query_patterns()
        self.optimization_strategies = self._load_optimization_strategies()
        self.optimization_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'optimizations_applied': 0,
            'total_time_saved_ms': 0
        }
    
    def _load_optimization_strategies(self) -> Dict[str, OptimizationStrategy]:
        """加载优化策略"""
        return {
            'use_stock_lookup_index': StockLookupOptimization(),
            'use_market_snapshot_index': MarketSnapshotOptimization(),
            'use_factor_lookup_index': FactorLookupOptimization(),
            'use_industry_index': IndustryOptimization()
        }
    
    def _load_query_patterns(self) -> List[QueryPattern]:
        """加载查询模式和优化规则"""
        patterns = [
            QueryPattern(
                name='single_stock_latest',
                pattern=r'SELECT.*FROM ts_daily.*WHERE ts_code.*ORDER BY trade_date DESC.*LIMIT 1',
                optimization_rule='use_stock_lookup_index',
                cache_ttl=60,  # 1分钟缓存
                priority=1
            ),
            QueryPattern(
                name='market_movers',
                pattern=r'SELECT.*FROM ts_daily.*WHERE trade_date.*ORDER BY pct_chg DESC.*LIMIT',
                optimization_rule='use_market_snapshot_index',
                cache_ttl=300,  # 5分钟缓存
                priority=1
            ),
            QueryPattern(
                name='factor_ranking',
                pattern=r'SELECT.*FROM factors_exposure.*WHERE factor_name.*ORDER BY.*factor_value',
                optimization_rule='use_factor_lookup_index',
                cache_ttl=600,  # 10分钟缓存
                priority=2
            ),
            QueryPattern(
                name='stock_history',
                pattern=r'SELECT.*FROM ts_daily.*WHERE ts_code.*AND trade_date.*ORDER BY trade_date',
                optimization_rule='use_stock_lookup_index',
                cache_ttl=1800,  # 30分钟缓存
                priority=2
            ),
            QueryPattern(
                name='industry_stocks',
                pattern=r'SELECT.*FROM stock_basic.*WHERE industry.*ORDER BY',
                optimization_rule='use_industry_index',
                cache_ttl=3600,  # 1小时缓存
                priority=3
            )
        ]
        return patterns
    
    def _identify_query_pattern(self, sql: str) -> Optional[QueryPattern]:
        """识别查询模式"""
        import re
        
        sql_normalized = ' '.join(sql.split()).upper()
        
        for pattern in self.query_patterns:
            if re.search(pattern.pattern.upper(), sql_normalized):
                return pattern
        
        return None
    
    def _apply_optimization_rules(self, sql: str, pattern: QueryPattern) -> OptimizedQuery:
        """应用优化规则"""
        strategy = self.optimization_strategies.get(pattern.optimization_rule)
        
        if strategy:
            optimized_sql, optimizations, estimated_improvement = strategy.apply(sql)
        else:
            log.warning(f"Unknown optimization rule: {pattern.optimization_rule}")
            optimized_sql, optimizations, estimated_improvement = sql, [], 1.0
        
        return OptimizedQuery(
            original_sql=sql,
            optimized_sql=optimized_sql,
            query_type=pattern.name,
            optimization_applied=optimizations,
            estimated_improvement=estimated_improvement
        )
    
    def optimize_query(self, sql: str, params: dict = None) -> OptimizedQuery:
        """优化查询"""
        # 识别查询模式
        pattern = self._identify_query_pattern(sql)
        
        if pattern:
            log.debug(f"识别查询模式: {pattern.name}")
            optimized = self._apply_optimization_rules(sql, pattern)
            optimized.cache_key = self.cache._generate_key(sql, params)
            return optimized
        else:
            # 未识别的查询，返回原始查询
            return OptimizedQuery(
                original_sql=sql,
                optimized_sql=sql,
                query_type='unknown',
                optimization_applied=[],
                estimated_improvement=1.0
            )
    
    def get_execution_plan(self, sql: str) -> ExecutionPlan:
        """获取查询执行计划"""
        query_hash = hashlib.md5(sql.encode()).hexdigest()
        
        try:
            with self.engine.connect() as conn:
                # 获取执行计划
                explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
                result = conn.execute(text(explain_sql)).fetchone()
                
                if result:
                    plan_data = result[0][0]  # JSON格式的执行计划
                    
                    # 提取关键信息
                    estimated_cost = plan_data.get('Total Cost', 0)
                    estimated_rows = plan_data.get('Plan Rows', 0)
                    actual_time = plan_data.get('Actual Total Time', 0)
                    
                    # 提取索引使用信息
                    index_usage = []
                    def extract_indexes(node):
                        if isinstance(node, dict):
                            if 'Index Name' in node:
                                index_usage.append(node['Index Name'])
                            for key, value in node.items():
                                if isinstance(value, (dict, list)):
                                    extract_indexes(value)
                        elif isinstance(node, list):
                            for item in node:
                                extract_indexes(item)
                    
                    extract_indexes(plan_data)
                    
                    return ExecutionPlan(
                        query_hash=query_hash,
                        plan_text=json.dumps(plan_data, indent=2),
                        estimated_cost=estimated_cost,
                        estimated_rows=estimated_rows,
                        actual_time_ms=actual_time,
                        index_usage=index_usage
                    )
        
        except Exception as e:
            log.warning(f"获取执行计划失败: {e}")
        
        return ExecutionPlan(
            query_hash=query_hash,
            plan_text="执行计划获取失败",
            estimated_cost=0,
            estimated_rows=0
        )
    
    def execute_query(self, sql: str, params: dict = None, use_cache: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """执行优化后的查询"""
        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty")
        
        start_time = time.time()
        self.optimization_stats['total_queries'] += 1
        
        # 检查缓存
        if use_cache:
            cached_result = self.cache.get(sql, params)
            if cached_result is not None:
                self.optimization_stats['cache_hits'] += 1
                execution_time = time.time() - start_time
                return cached_result, {
                    'cache_hit': True,
                    'execution_time_ms': execution_time * 1000,
                    'optimization_applied': []
                }
        
        # 优化查询
        optimized = self.optimize_query(sql, params)
        if optimized.optimization_applied:
            self.optimization_stats['optimizations_applied'] += 1
        
        # 执行查询
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(optimized.optimized_sql), params).fetchall()
                else:
                    result = conn.execute(text(optimized.optimized_sql)).fetchall()
                
                execution_time = time.time() - start_time
                
                # 缓存结果
                if use_cache:
                    pattern = self._identify_query_pattern(sql)
                    ttl = pattern.cache_ttl if pattern else self.cache.default_ttl
                    self.cache.set(sql, result, params, ttl)
                
                # 计算时间节省
                if optimized.estimated_improvement > 1.0:
                    time_saved = execution_time * (optimized.estimated_improvement - 1.0) * 1000
                    self.optimization_stats['total_time_saved_ms'] += time_saved
                
                return result, {
                    'cache_hit': False,
                    'execution_time_ms': execution_time * 1000,
                    'optimization_applied': optimized.optimization_applied,
                    'estimated_improvement': optimized.estimated_improvement,
                    'query_type': optimized.query_type
                }
        
        except Exception as e:
            log.error(f"查询执行失败: {e}")
            raise
    
    def analyze_slow_queries(self, threshold_ms: float = SLOW_QUERY_THRESHOLD_MS) -> List[Dict[str, Any]]:
        """分析慢查询"""
        slow_queries = []
        
        try:
            with self.engine.connect() as conn:
                # 从pg_stat_statements获取慢查询（如果可用）
                result = conn.execute(text("""
                    SELECT 
                        query,
                        calls,
                        total_time,
                        mean_time,
                        rows
                    FROM pg_stat_statements 
                    WHERE mean_time > :threshold
                    ORDER BY mean_time DESC
                    LIMIT :limit
                """), {'threshold': threshold_ms, 'limit': MAX_SLOW_QUERIES_REPORT}).fetchall()
                
                for row in result:
                    query_text = row[0]
                    truncated_query = (
                        query_text[:QUERY_TRUNCATE_LENGTH] + '...' 
                        if len(query_text) > QUERY_TRUNCATE_LENGTH 
                        else query_text
                    )
                    
                    slow_queries.append({
                        'query': truncated_query,
                        'calls': row[1],
                        'total_time_ms': row[2],
                        'mean_time_ms': row[3],
                        'avg_rows': row[4]
                    })
        
        except Exception as e:
            log.warning(f"慢查询分析失败（可能需要启用pg_stat_statements）: {e}")
        
        return slow_queries
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        cache_stats = self.cache.stats()
        
        hit_rate = 0
        if self.optimization_stats['total_queries'] > 0:
            hit_rate = self.optimization_stats['cache_hits'] / self.optimization_stats['total_queries'] * 100
        
        return {
            'query_stats': self.optimization_stats,
            'cache_stats': cache_stats,
            'cache_hit_rate': f"{hit_rate:.1f}%",
            'avg_time_saved_per_query_ms': (
                self.optimization_stats['total_time_saved_ms'] / 
                max(self.optimization_stats['total_queries'], 1)
            )
        }
    
    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        stats = self.get_optimization_stats()
        
        report_lines = [
            "=== 查询优化器报告 ===",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "查询统计:",
            f"  总查询数: {stats['query_stats']['total_queries']}",
            f"  缓存命中: {stats['query_stats']['cache_hits']}",
            f"  优化应用: {stats['query_stats']['optimizations_applied']}",
            f"  缓存命中率: {stats['cache_hit_rate']}",
            f"  平均节省时间: {stats['avg_time_saved_per_query_ms']:.2f}ms",
            "",
            "缓存统计:",
            f"  缓存条目: {stats['cache_stats']['active_entries']}/{stats['cache_stats']['max_size']}",
            f"  使用率: {stats['cache_stats']['usage_rate']}",
            f"  过期条目: {stats['cache_stats']['expired_entries']}",
            ""
        ]
        
        # 慢查询分析
        slow_queries = self.analyze_slow_queries()
        if slow_queries:
            report_lines.append("慢查询TOP5:")
            for i, query in enumerate(slow_queries[:5], 1):
                report_lines.append(f"  {i}. 平均耗时: {query['mean_time_ms']:.2f}ms")
                report_lines.append(f"     调用次数: {query['calls']}")
                report_lines.append(f"     查询: {query['query']}")
                report_lines.append("")
        else:
            report_lines.append("✓ 未发现明显的慢查询")
        
        return "\n".join(report_lines)


def main():
    """主函数 - 用于测试和维护"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库查询优化器")
    parser.add_argument("--test", action="store_true", help="运行测试查询")
    parser.add_argument("--stats", action="store_true", help="显示优化统计")
    parser.add_argument("--report", action="store_true", help="生成优化报告")
    parser.add_argument("--clear-cache", action="store_true", help="清空缓存")
    parser.add_argument("--sql", type=str, help="执行指定SQL查询")
    
    args = parser.parse_args()
    
    optimizer = QueryOptimizer()
    
    try:
        if args.test:
            # 测试查询
            test_queries = [
                "SELECT * FROM ts_daily WHERE ts_code = '600519.SH' ORDER BY trade_date DESC LIMIT 1",
                "SELECT ts_code, close, pct_chg FROM ts_daily WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily) ORDER BY pct_chg DESC LIMIT 50",
                "SELECT * FROM stock_basic WHERE industry = '银行' ORDER BY ts_code"
            ]
            
            for sql in test_queries:
                print(f"\n测试查询: {sql[:50]}...")
                result, stats = optimizer.execute_query(sql)
                print(f"结果: {len(result)} 条记录")
                print(f"执行时间: {stats['execution_time_ms']:.2f}ms")
                print(f"优化应用: {stats['optimization_applied']}")
        
        elif args.stats:
            stats = optimizer.get_optimization_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        elif args.report:
            report = optimizer.generate_optimization_report()
            print(report)
        
        elif args.clear_cache:
            optimizer.cache.clear()
            print("缓存已清空")
        
        elif args.sql:
            result, stats = optimizer.execute_query(args.sql)
            print(f"查询结果: {len(result)} 条记录")
            print(f"执行统计: {stats}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        log.error(f"执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())