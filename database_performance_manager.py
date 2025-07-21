#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库性能管理器

集成索引管理器和查询优化器，提供统一的数据库性能优化接口
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text

import config
from logger_config import log
from database_index_manager import IndexManager
from database_query_optimizer import QueryOptimizer


class DatabasePerformanceManager:
    """数据库性能管理器"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        self.index_manager = IndexManager(db_url)
        self.query_optimizer = QueryOptimizer(db_url)
        
    def initialize_performance_optimization(self) -> Dict[str, Any]:
        """初始化性能优化"""
        log.info("=== 初始化数据库性能优化 ===")
        
        results = {
            'index_creation': {},
            'optimization_setup': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. 创建核心索引
        log.info("1. 创建核心性能索引...")
        index_results = self.index_manager.create_all_indexes(priority_filter=1)
        results['index_creation'] = index_results
        
        success_count = sum(1 for success in index_results.values() if success)
        total_count = len(index_results)
        log.info(f"   核心索引创建: {success_count}/{total_count} 成功")
        
        # 2. 初始化查询优化器
        log.info("2. 初始化查询优化器...")
        self.query_optimizer.cache.clear()  # 清空缓存重新开始
        log.info("   查询优化器初始化完成")
        
        log.info("=== 性能优化初始化完成 ===")
        return results
    
    def execute_optimized_query(self, sql: str, params: dict = None, 
                               use_cache: bool = True) -> tuple:
        """执行优化查询"""
        return self.query_optimizer.execute_query(sql, params, use_cache)
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """性能基准测试"""
        log.info("=== 数据库性能基准测试 ===")
        
        benchmark_queries = [
            {
                'name': '单股票最新数据',
                'sql': "SELECT * FROM ts_daily WHERE ts_code = '600519.SH' ORDER BY trade_date DESC LIMIT 1",
                'target_ms': 50,
                'category': 'stock_lookup'
            },
            {
                'name': '市场涨幅排行',
                'sql': """
                    SELECT ts_code, close, pct_chg, vol
                    FROM ts_daily 
                    WHERE trade_date = (SELECT MAX(trade_date) FROM ts_daily)
                    ORDER BY pct_chg DESC 
                    LIMIT 50
                """,
                'target_ms': 200,
                'category': 'market_ranking'
            },
            {
                'name': '股票历史数据',
                'sql': """
                    SELECT trade_date, close, vol, pct_chg
                    FROM ts_daily 
                    WHERE ts_code = '600519.SH' 
                    AND trade_date >= '2024-01-01'
                    ORDER BY trade_date DESC
                    LIMIT 100
                """,
                'target_ms': 100,
                'category': 'stock_history'
            },
            {
                'name': '行业股票查询',
                'sql': """
                    SELECT ts_code, name, industry
                    FROM stock_basic 
                    WHERE industry = '银行'
                    ORDER BY ts_code
                """,
                'target_ms': 10,
                'category': 'industry_lookup'
            },
            {
                'name': '因子数据查询',
                'sql': """
                    SELECT ts_code, factor_value
                    FROM factors_exposure 
                    WHERE factor_name = 'pe_ttm' 
                    ORDER BY trade_date DESC, factor_value DESC
                    LIMIT 100
                """,
                'target_ms': 50,
                'category': 'factor_lookup'
            }
        ]
        
        results = {
            'test_results': [],
            'summary': {
                'total_tests': len(benchmark_queries),
                'passed_tests': 0,
                'failed_tests': 0,
                'avg_performance_ratio': 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        performance_ratios = []
        
        for query in benchmark_queries:
            log.info(f"\n测试: {query['name']}")
            
            # 运行查询3次取平均值
            times = []
            for i in range(3):
                try:
                    result, stats = self.execute_optimized_query(query['sql'])
                    times.append(stats['execution_time_ms'])
                except Exception as e:
                    log.error(f"查询失败: {e}")
                    break
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                # 计算性能比率（目标时间/实际时间，>1表示超过目标）
                performance_ratio = query['target_ms'] / avg_time
                performance_ratios.append(performance_ratio)
                
                test_result = {
                    'name': query['name'],
                    'category': query['category'],
                    'avg_time_ms': avg_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'target_ms': query['target_ms'],
                    'performance_ratio': performance_ratio,
                    'status': 'passed' if avg_time <= query['target_ms'] else 'failed',
                    'record_count': len(result) if 'result' in locals() else 0
                }
                
                results['test_results'].append(test_result)
                
                if avg_time <= query['target_ms']:
                    results['summary']['passed_tests'] += 1
                    log.info(f"  ✓ 通过: {avg_time:.2f}ms (目标: <{query['target_ms']}ms)")
                else:
                    results['summary']['failed_tests'] += 1
                    log.info(f"  ✗ 超时: {avg_time:.2f}ms (目标: <{query['target_ms']}ms)")
                
                log.info(f"  记录数: {len(result) if 'result' in locals() else 0}")
        
        # 计算总体性能
        if performance_ratios:
            results['summary']['avg_performance_ratio'] = sum(performance_ratios) / len(performance_ratios)
        
        pass_rate = results['summary']['passed_tests'] / results['summary']['total_tests'] * 100
        log.info(f"\n基准测试完成: {results['summary']['passed_tests']}/{results['summary']['total_tests']} 通过 ({pass_rate:.1f}%)")
        
        return results
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """综合性能分析"""
        log.info("=== 综合性能分析 ===")
        
        analysis = {
            'index_analysis': {},
            'query_optimization': {},
            'performance_benchmark': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. 索引分析
        log.info("1. 分析索引使用情况...")
        analysis['index_analysis'] = self.index_manager.analyze_index_usage()
        
        # 2. 查询优化统计
        log.info("2. 分析查询优化情况...")
        analysis['query_optimization'] = self.query_optimizer.get_optimization_stats()
        
        # 3. 性能基准测试
        log.info("3. 执行性能基准测试...")
        analysis['performance_benchmark'] = self.benchmark_performance()
        
        # 4. 生成优化建议
        log.info("4. 生成优化建议...")
        recommendations = []
        
        # 基于索引分析的建议
        if analysis['index_analysis'].get('summary', {}).get('usage_rate', '0%') < '50%':
            recommendations.append({
                'type': '索引优化',
                'priority': 'high',
                'description': '索引使用率较低，建议清理未使用的索引',
                'action': '运行索引清理工具'
            })
        
        # 基于性能测试的建议
        failed_tests = analysis['performance_benchmark']['summary']['failed_tests']
        if failed_tests > 0:
            recommendations.append({
                'type': '查询性能',
                'priority': 'high',
                'description': f'{failed_tests}个查询未达到性能目标',
                'action': '检查慢查询并优化索引策略'
            })
        
        # 基于缓存的建议
        cache_hit_rate = float(analysis['query_optimization']['cache_hit_rate'].rstrip('%'))
        if cache_hit_rate < 30:
            recommendations.append({
                'type': '缓存优化',
                'priority': 'medium',
                'description': f'缓存命中率较低({cache_hit_rate:.1f}%)',
                'action': '调整缓存策略和TTL设置'
            })
        
        analysis['recommendations'] = recommendations
        
        log.info("=== 综合性能分析完成 ===")
        return analysis
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        analysis = self.comprehensive_analysis()
        
        report_lines = [
            "=== 数据库性能综合报告 ===",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 索引使用情况:",
            f"  总索引数: {analysis['index_analysis']['summary']['total_indexes']}",
            f"  使用率: {analysis['index_analysis']['summary']['usage_rate']}",
            f"  总大小: {analysis['index_analysis']['summary']['total_size_mb']} MB",
            "",
            "🚀 查询优化统计:",
            f"  总查询数: {analysis['query_optimization']['query_stats']['total_queries']}",
            f"  缓存命中率: {analysis['query_optimization']['cache_hit_rate']}",
            f"  优化应用次数: {analysis['query_optimization']['query_stats']['optimizations_applied']}",
            "",
            "⚡ 性能基准测试:",
            f"  测试通过率: {analysis['performance_benchmark']['summary']['passed_tests']}/{analysis['performance_benchmark']['summary']['total_tests']}",
            f"  平均性能比率: {analysis['performance_benchmark']['summary']['avg_performance_ratio']:.2f}",
            ""
        ]
        
        # 详细测试结果
        report_lines.append("详细测试结果:")
        for test in analysis['performance_benchmark']['test_results']:
            status_icon = "✓" if test['status'] == 'passed' else "✗"
            report_lines.append(f"  {status_icon} {test['name']}: {test['avg_time_ms']:.2f}ms (目标: {test['target_ms']}ms)")
        
        report_lines.append("")
        
        # 优化建议
        if analysis['recommendations']:
            report_lines.append("🔧 优化建议:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                report_lines.append(f"  {i}. {priority_icon} {rec['type']}: {rec['description']}")
                report_lines.append(f"     建议操作: {rec['action']}")
                report_lines.append("")
        else:
            report_lines.append("✅ 系统性能良好，暂无优化建议")
        
        return "\n".join(report_lines)
    
    def quick_optimize(self) -> Dict[str, Any]:
        """快速优化"""
        log.info("=== 执行快速优化 ===")
        
        results = {
            'actions_taken': [],
            'improvements': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. 创建缺失的高优先级索引
        log.info("1. 检查并创建缺失的关键索引...")
        index_results = self.index_manager.create_all_indexes(priority_filter=1)
        created_indexes = [name for name, success in index_results.items() if success]
        if created_indexes:
            results['actions_taken'].append(f"创建了 {len(created_indexes)} 个关键索引")
        
        # 2. 清理缓存
        log.info("2. 清理查询缓存...")
        self.query_optimizer.cache.clear()
        results['actions_taken'].append("清理了查询缓存")
        
        # 3. 更新表统计信息
        log.info("3. 更新表统计信息...")
        try:
            with self.engine.connect() as conn:
                tables = ['ts_daily', 'factors_exposure', 'financial_indicators', 'stock_basic']
                for table in tables:
                    conn.execute(text(f"ANALYZE {table}"))
                results['actions_taken'].append("更新了表统计信息")
        except Exception as e:
            log.warning(f"更新统计信息失败: {e}")
        
        log.info("=== 快速优化完成 ===")
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库性能管理器")
    parser.add_argument("--init", action="store_true", help="初始化性能优化")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--analyze", action="store_true", help="综合性能分析")
    parser.add_argument("--report", action="store_true", help="生成性能报告")
    parser.add_argument("--quick-optimize", action="store_true", help="执行快速优化")
    
    args = parser.parse_args()
    
    manager = DatabasePerformanceManager()
    
    try:
        if args.init:
            result = manager.initialize_performance_optimization()
            print(f"初始化结果: {result}")
        
        elif args.benchmark:
            result = manager.benchmark_performance()
            print(f"基准测试结果: {result['summary']}")
        
        elif args.analyze:
            result = manager.comprehensive_analysis()
            print(f"分析完成，发现 {len(result['recommendations'])} 个优化建议")
        
        elif args.report:
            report = manager.generate_performance_report()
            print(report)
        
        elif args.quick_optimize:
            result = manager.quick_optimize()
            print(f"快速优化完成: {result}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        log.error(f"执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())