# quant_project/scheduler.py
#
# 这是一个独立的、可长期运行的自动化任务调度器。
# 它使用 APScheduler 库，根据预设的时间规则，自动执行数据更新、因子计算等任务。
# 实现了“功能需求规格说明书”中的 1.2 健壮的数据调度器。
# 实现了“功能需求规格说明书”中的 5.3 自动化调度。
# 作为一个独立的后台服务运行，用于定时执行自动化任务。

import time
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone
from functools import partial

# 导入项目模块
import data
import quant_engine
import intelligence
import config
from logger_config import log
import factor_calculator # 【新增】导入新的因子计算模块
from datetime import datetime

# --- 启动调度器 ---

def main():
    log.info("初始化自动化任务调度器...")

    try:
        # 实例化所有需要的核心组件
        data_manager = data.DataManager()
        factor_factory = quant_engine.FactorFactory(_data_manager=data_manager)
        ai_orchestrator = intelligence.AIOrchestrator(config.AI_MODEL_CONFIG)
        task_runner = quant_engine.AutomatedTasks(data_manager, factor_factory)
    except Exception as e:
        log.critical("调度器初始化失败!", exc_info=True)
        return

    scheduler = BlockingScheduler(timezone=timezone('Asia/Shanghai'))

    # 任务1: 每日数据更新
    scheduler.add_job(task_runner.run_daily_data_update, 'cron', day_of_week='mon-fri', hour=17, minute=30, id='daily_update_job')
    log.info("已注册任务：'每日数据更新' (周一至周五 17:30)")
    
    # 任务2: 【升级】每日全市场因子预计算
    # 在每个交易日收盘后执行
    scheduler.add_job(
        factor_calculator.calculate_all_factors_for_date, 
        'cron', 
        args=[datetime.now().strftime('%Y%m%d')], # 传入当天日期作为参数
        day_of_week='mon-fri', 
        hour=18, 
        minute=0, 
        id='daily_factor_calculation_job'
    )
    log.info("已注册任务：'每日全市场因子预计算' (周一至周五 18:00)")

    # 任务3: 每周自动生成并发送AI投研报告
    # 使用 partial 绑定工作流所需的参数
    report_workflow_func = partial(
        intelligence.generate_and_send_report_workflow,
        orchestrator=ai_orchestrator,
        data_manager=data_manager,
        factor_factory=factor_factory
    )
    # 为配置中的每支股票创建一个独立的调度任务
    for stock_code in config.AUTOMATED_REPORT_STOCKS:
        job_id = f'weekly_report_job_{stock_code}'
        scheduler.add_job(
            report_workflow_func, 
            'cron', 
            args=[stock_code],
            day_of_week='sun', # 每周日
            hour=3, 
            minute=0, 
            id=job_id
        )
        log.info(f"已注册任务：'每周AI报告 - {stock_code}' (周日 03:00)")


    log.info("调度器已启动，等待任务触发...")
    scheduler.print_jobs()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("调度器已手动停止。")
        scheduler.shutdown()

if __name__ == '__main__':
    main()