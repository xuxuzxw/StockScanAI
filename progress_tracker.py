# StockScanAI/progress_tracker.py
#
# 进度跟踪和数据质量检查模块

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
from logger_config import log


@dataclass
class BatchQualityConfig:
    """批量质量检查配置"""
    sample_size: int = 50
    batch_size: int = 100
    progress_interval: int = 50
    max_workers: int = 4
    max_retries: int = 2
    fail_fast: bool = False
    
    # Performance tuning
    conservative_batch_size: int = 50
    conservative_max_workers: int = 2
    
    def get_retry_config(self) -> 'BatchQualityConfig':
        """获取重试时的保守配置"""
        return BatchQualityConfig(
            sample_size=self.sample_size,
            batch_size=self.conservative_batch_size,
            progress_interval=self.progress_interval,
            max_workers=self.conservative_max_workers,
            max_retries=0,  # No nested retries
            fail_fast=False
        )


class ProgressTracker:
    """进度跟踪器 - 用于监控数据回填进度"""
    
    def __init__(self, total_items: int, description: str = "处理中"):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = time.time()
        
    def update(self, increment: int = 1, success: bool = True):
        """更新进度"""
        if success:
            self.completed_items += increment
        else:
            self.failed_items += increment
            
        current_time = time.time()
        # 每5秒或每完成10%更新一次日志
        if (current_time - self.last_update_time > 5.0 or 
            (self.completed_items + self.failed_items) % max(1, self.total_items // 10) == 0):
            self._log_progress()
            self.last_update_time = current_time
    
    def _log_progress(self):
        """记录进度日志"""
        total_processed = self.completed_items + self.failed_items
        if total_processed == 0:
            return
            
        progress_pct = (total_processed / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if total_processed > 0:
            avg_time_per_item = elapsed_time / total_processed
            eta_seconds = avg_time_per_item * (self.total_items - total_processed)
            eta_str = f"{eta_seconds/60:.1f}分钟" if eta_seconds > 60 else f"{eta_seconds:.0f}秒"
        else:
            eta_str = "未知"
        
        log.info(f"  {self.description}: {progress_pct:.1f}% "
                f"({total_processed}/{self.total_items}) "
                f"成功:{self.completed_items} 失败:{self.failed_items} "
                f"预计剩余:{eta_str}")
    
    def finish(self):
        """完成进度跟踪"""
        total_time = time.time() - self.start_time
        total_processed = self.completed_items + self.failed_items
        
        log.info(f"  {self.description}完成: "
                f"总计{total_processed}项 "
                f"成功{self.completed_items}项 "
                f"失败{self.failed_items}项 "
                f"耗时{total_time/60:.1f}分钟")
        
        return {
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'failed_items': self.failed_items,
            'total_time': total_time,
            'success_rate': self.completed_items / max(1, total_processed)
        }


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, dm):
        self.dm = dm
        
    def check_data_completeness(self, ts_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """检查数据完整性"""
        try:
            # 获取交易日历
            trade_cal = self.dm.pro.trade_cal(
                exchange='', 
                start_date=start_date, 
                end_date=end_date
            )
            
            if trade_cal is None or trade_cal.empty:
                return {'status': 'error', 'message': '无法获取交易日历'}
            
            # 获取该股票的实际数据
            with self.dm.engine.connect() as conn:
                actual_data = pd.read_sql(
                    f"""
                    SELECT DISTINCT trade_date 
                    FROM ts_daily 
                    WHERE ts_code = '{ts_code}' 
                    AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY trade_date
                    """,
                    conn
                )
            
            # 计算完整性
            expected_days = len(trade_cal[trade_cal['is_open'] == 1])
            actual_days = len(actual_data)
            completeness_ratio = actual_days / max(1, expected_days)
            
            return {
                'status': 'success',
                'ts_code': ts_code,
                'expected_days': expected_days,
                'actual_days': actual_days,
                'completeness_ratio': completeness_ratio,
                'is_complete': completeness_ratio >= 0.95  # 95%以上认为完整
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'ts_code': ts_code,
                'message': str(e)
            }
    
    def check_data_quality(self, ts_code: str, sample_size: int = 100) -> Dict[str, Any]:
        """检查数据质量"""
        try:
            with self.dm.engine.connect() as conn:
                # 获取样本数据
                sample_data = pd.read_sql(
                    f"""
                    SELECT * FROM ts_daily 
                    WHERE ts_code = '{ts_code}' 
                    ORDER BY trade_date DESC 
                    LIMIT {sample_size}
                    """,
                    conn
                )
            
            if sample_data.empty:
                return {'status': 'error', 'message': '无数据'}
            
            # 检查数据质量指标
            quality_checks = {
                'null_values': sample_data.isnull().sum().sum(),
                'negative_prices': (sample_data[['open', 'high', 'low', 'close']] < 0).sum().sum(),
                'zero_volume': (sample_data['vol'] == 0).sum(),
                'price_consistency': self._check_price_consistency(sample_data),
                'data_count': len(sample_data)
            }
            
            # 计算质量分数
            quality_score = self._calculate_quality_score(quality_checks, len(sample_data))
            
            return {
                'status': 'success',
                'ts_code': ts_code,
                'quality_checks': quality_checks,
                'quality_score': quality_score,
                'is_good_quality': quality_score >= 0.9
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'ts_code': ts_code,
                'message': str(e)
            }
    
    def _check_price_consistency(self, data: pd.DataFrame) -> int:
        """检查价格一致性（high >= low, close在high和low之间等）"""
        inconsistencies = 0
        
        # 检查 high >= low
        inconsistencies += (data['high'] < data['low']).sum()
        
        # 检查 close 在 high 和 low 之间
        inconsistencies += ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
        
        # 检查 open 在合理范围内
        inconsistencies += ((data['open'] > data['high']) | (data['open'] < data['low'])).sum()
        
        return inconsistencies
    
    def _calculate_quality_score(self, checks: Dict[str, Any], total_records: int) -> float:
        """计算数据质量分数 (0-1)"""
        if total_records == 0:
            return 0.0
        
        # 各项检查的权重
        penalties = 0
        
        # 空值惩罚
        penalties += (checks['null_values'] / total_records) * 0.3
        
        # 负价格惩罚
        penalties += (checks['negative_prices'] / total_records) * 0.4
        
        # 零成交量惩罚（轻微）
        penalties += (checks['zero_volume'] / total_records) * 0.1
        
        # 价格不一致惩罚
        penalties += (checks['price_consistency'] / total_records) * 0.2
        
        # 计算最终分数
        score = max(0.0, 1.0 - penalties)
        return score
    
    def batch_quality_check(
        self, 
        ts_codes: List[str], 
        config: Optional[BatchQualityConfig] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量数据质量检查 - 优化版本
        
        Args:
            ts_codes: 股票代码列表
            config: 批量处理配置，如果为None则使用默认配置
            **kwargs: 覆盖配置的特定参数
            
        Returns:
            List[Dict[str, Any]]: 质量检查结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # Initialize configuration
        if config is None:
            config = BatchQualityConfig()
        
        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        results = []
        failed_stocks = []
        start_time = time.time()
        
        log.info(f"开始批量质量检查，共{len(ts_codes)}只股票...")
        log.info(f"配置: 批大小={config.batch_size}, 并发数={config.max_workers}, 采样={config.sample_size}")
        
        def check_single_stock(ts_code: str) -> Dict[str, Any]:
            """检查单只股票的数据质量"""
            try:
                result = self.check_data_quality(ts_code, config.sample_size)
                result['ts_code'] = ts_code
                result['status'] = 'success'
                return result
            except Exception as e:
                error_result = {
                    'ts_code': ts_code,
                    'status': 'error',
                    'error': str(e),
                    'score': 0.0
                }
                if config.fail_fast:
                    raise
                return error_result
        
        # Process in batches with parallel execution
        for batch_start in range(0, len(ts_codes), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(ts_codes))
            batch_codes = ts_codes[batch_start:batch_end]
            
            log.info(f"处理批次 {batch_start//config.batch_size + 1}/{(len(ts_codes)-1)//config.batch_size + 1}")
            
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Submit all tasks in current batch
                future_to_code = {
                    executor.submit(check_single_stock, ts_code): ts_code 
                    for ts_code in batch_codes
                }
                
                # Collect results as they complete
                batch_results = []
                for i, future in enumerate(as_completed(future_to_code)):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result['status'] == 'error':
                            failed_stocks.append(result['ts_code'])
                        
                        # Progress reporting
                        if (i + 1) % config.progress_interval == 0 or (i + 1) == len(batch_codes):
                            completed = batch_start + i + 1
                            progress = completed / len(ts_codes) * 100
                            elapsed = time.time() - start_time
                            eta = elapsed / completed * (len(ts_codes) - completed) if completed > 0 else 0
                            
                            log.info(f"进度: {completed}/{len(ts_codes)} ({progress:.1f}%) "
                                   f"- 耗时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
                    
                    except Exception as e:
                        failed_code = future_to_code[future]
                        log.error(f"股票 {failed_code} 质量检查失败: {e}")
                        failed_stocks.append(failed_code)
                        
                        if config.fail_fast:
                            raise
                
                results.extend(batch_results)
        
        # Generate summary
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('status') == 'success'])
        
        log.info(f"批量质量检查完成:")
        log.info(f"  总股票数: {len(ts_codes)}")
        log.info(f"  成功检查: {success_count}")
        log.info(f"  失败股票: {len(failed_stocks)}")
        log.info(f"  总耗时: {total_time:.2f}秒")
        log.info(f"  平均速度: {len(ts_codes)/total_time:.1f} 股票/秒")
        
        if failed_stocks:
            log.warning(f"失败的股票代码: {failed_stocks[:10]}{'...' if len(failed_stocks) > 10 else ''}")
        
        return results
    
    def batch_quality_check_with_retry(
        self,
        ts_codes: List[str],
        config: Optional[BatchQualityConfig] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        带重试机制的批量质量检查
        
        Args:
            ts_codes: 股票代码列表
            config: 批量处理配置
            **kwargs: 覆盖配置的特定参数
            
        Returns:
            List[Dict[str, Any]]: 质量检查结果
        """
        if config is None:
            config = BatchQualityConfig()
        
        results = self.batch_quality_check(ts_codes, config, **kwargs)
        
        # Find failed stocks for retry
        failed_stocks = [r['ts_code'] for r in results if r.get('status') == 'error']
        
        retry_count = 0
        while failed_stocks and retry_count < config.max_retries:
            retry_count += 1
            log.info(f"重试第 {retry_count} 次，重试 {len(failed_stocks)} 只股票...")
            
            # Use conservative retry configuration
            retry_config = config.get_retry_config()
            
            retry_results = self.batch_quality_check(
                failed_stocks, 
                retry_config,
                **kwargs
            )
            
            # Update original results
            retry_dict = {r['ts_code']: r for r in retry_results}
            for i, result in enumerate(results):
                if result['ts_code'] in retry_dict:
                    results[i] = retry_dict[result['ts_code']]
            
            # Update failed stocks list
            failed_stocks = [r['ts_code'] for r in retry_results if r.get('status') == 'error']
        
        return results
    
    def get_quality_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取质量检查摘要统计
        
        Args:
            results: 质量检查结果列表
            
        Returns:
            Dict[str, Any]: 摘要统计信息
        """
        if not results:
            return {'error': '无检查结果'}
        
        successful_results = [r for r in results if r.get('status') == 'success']
        failed_results = [r for r in results if r.get('status') == 'error']
        
        if not successful_results:
            return {
                'total_stocks': len(results),
                'successful_checks': 0,
                'failed_checks': len(failed_results),
                'success_rate': 0.0,
                'error': '所有检查均失败'
            }
        
        scores = [r.get('score', 0.0) for r in successful_results]
        
        return {
            'total_stocks': len(results),
            'successful_checks': len(successful_results),
            'failed_checks': len(failed_results),
            'success_rate': len(successful_results) / len(results),
            'quality_stats': {
                'mean_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'scores_above_80': len([s for s in scores if s >= 0.8]),
                'scores_below_50': len([s for s in scores if s < 0.5])
            },
            'top_quality_stocks': [
                r['ts_code'] for r in sorted(successful_results, 
                key=lambda x: x.get('score', 0), reverse=True)[:10]
            ],
            'low_quality_stocks': [
                r['ts_code'] for r in sorted(successful_results, 
                key=lambda x: x.get('score', 0))[:10]
            ]
        }
    
    def generate_quality_report(self, results: List[Dict[str, Any]]) -> str:
        """生成数据质量报告"""
        if not results:
            return "无数据质量检查结果"
        
        successful_checks = [r for r in results if r.get('status') == 'success']
        failed_checks = [r for r in results if r.get('status') == 'error']
        
        if not successful_checks:
            return f"所有{len(results)}项检查都失败了"
        
        # 统计质量指标
        avg_quality_score = sum(r['quality_score'] for r in successful_checks) / len(successful_checks)
        good_quality_count = sum(1 for r in successful_checks if r['is_good_quality'])
        
        report = []
        report.append(f"数据质量检查报告:")
        report.append(f"  检查股票数: {len(results)}")
        report.append(f"  成功检查: {len(successful_checks)}")
        report.append(f"  失败检查: {len(failed_checks)}")
        report.append(f"  平均质量分数: {avg_quality_score:.2%}")
        report.append(f"  高质量数据: {good_quality_count}/{len(successful_checks)} ({good_quality_count/len(successful_checks):.1%})")
        
        return "\n".join(report)