# quant_project/run_strategy_daily.py
#
# 【V2.3 核心自动化引擎】
# 目的：将所有模块串联起来，形成一个每日自动运行、输出实战级投研晨报的闭环工作流。

import pandas as pd
from datetime import datetime, timedelta
import os

# 导入项目模块
import config
import data
import intelligence
import quant_engine
from logger_config import log

def run_daily_strategy_workflow():
    """
    执行每日策略分析与晨报生成的核心工作流。
    """
    log.info("===== 开始执行每日策略自动化工作流 =====")

    # --- 1. 初始化核心组件 ---
    try:
        dm = data.DataManager()
        ai_orchestrator = intelligence.AIOrchestrator(config.AI_MODEL_CONFIG, dm)
        ff = quant_engine.FactorFactory(_data_manager=dm)
        fp = quant_engine.FactorProcessor(_data_manager=dm)
        fa = quant_engine.FactorAnalyzer(_data_manager=dm)
        risk_manager = quant_engine.RiskManager(ff, fp)
    except Exception as e:
        log.critical("初始化核心组件失败！工作流终止。", exc_info=True)
        return

    # --- 2. 确定分析日期 ---
    try:
        cal_df = dm.pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
        trade_date = cal_df[cal_df['is_open'] == 1]['cal_date'].max()
        log.info(f"确定分析日期为最新的交易日: {trade_date}")
    except Exception as e:
        log.error(f"获取最新交易日失败: {e}")
        return

    # --- 3. 定义并执行策略 ---
    #    (这是一个预设的、经过验证的策略配置示例)
    strategy_config = {
        'name': '自适应权重策略 (动量+价值+质量)',
        'type': 'adaptive_weight',
        'params': {
            'factors_to_use': ('momentum', 'pe_ttm', 'roe'),
            'ic_lookback_days': 180,
            'top_n_stocks': 20, # 初步筛选的股票数量
            'max_weight_per_stock': 0.10 # 单票最大权重
        }
    }
    log.info(f"正在执行策略: {strategy_config['name']}")

    try:
        # a. 数据准备
        stock_pool = dm.get_stock_basic()['ts_code'].tolist()[:300] # 为提高演示速度，缩小范围
        prices_start_date = (pd.to_datetime(trade_date) - timedelta(days=strategy_config['params']['ic_lookback_days'] + 60)).strftime('%Y%m%d')
        prices_dict = dm.run_batch_download(stock_pool, prices_start_date, trade_date)
        all_prices_df = pd.DataFrame({
            stock: df.set_index('trade_date')['close']
            for stock, df in prices_dict.items() if df is not None and not df.empty
        }).sort_index()
        all_prices_df.index = pd.to_datetime(all_prices_df.index)
        all_prices_df.dropna(axis=1, how='all', inplace=True)
        stock_pool = all_prices_df.columns.tolist()

        # b. 生成当期信号/权重
        adaptive_strategy = quant_engine.AdaptiveAlphaStrategy(ff, fp, fa, all_prices_df)
        composite_factor, dynamic_weights = adaptive_strategy.generate_composite_factor(
            pd.to_datetime(trade_date),
            stock_pool,
            strategy_config['params']['factors_to_use'],
            strategy_config['params']['ic_lookback_days']
        )
        
        # c. 优化得到最终持仓
        selected_stocks = composite_factor.nlargest(strategy_config['params']['top_n_stocks']).index
        cov_matrix = all_prices_df[selected_stocks].loc[:trade_date].pct_change().iloc[-252:].cov() * 252
        expected_returns = composite_factor[selected_stocks]
        optimizer = quant_engine.PortfolioOptimizer(expected_returns, cov_matrix)
        target_portfolio = optimizer.optimize_max_sharpe(max_weight_per_stock=strategy_config['params']['max_weight_per_stock'])
        log.info(f"策略执行完毕，生成目标持仓 {len(target_portfolio)} 只。")

    except Exception as e:
        log.error("策略执行过程中发生错误！", exc_info=True)
        return

    # --- 4. 执行风险分析 ---
    log.info("正在执行投资组合风险分析...")
    try:
        risk_exposure = risk_manager.calculate_risk_exposure(target_portfolio['weight'], trade_date)
        log.info(f"风险分析完毕。")
    except Exception as e:
        log.error("风险分析过程中发生错误！", exc_info=True)
        risk_exposure = pd.Series(dtype=float)

    # --- 5. 生成AI晨报 ---
    log.info("正在调用AI引擎生成投研晨报...")
    try:
        # a. 准备给AI的数据包
        report_payload = {
            "报告日期": trade_date,
            "执行策略": strategy_config,
            "核心动态因子权重": dynamic_weights.to_dict(),
            "最终目标持仓": target_portfolio.to_dict()['weight'],
            "组合风险暴露": risk_exposure.to_dict(),
            "市场关键新闻": "（此处未来可接入新闻模块）"
        }

        # b. 构建AI Prompt
        prompt = f"""
        作为A股量化投研平台的首席策略师，请根据以下今日策略执行的最终结果，撰写一份专业的、面向内部基金经理的《每日AI投研晨报》。

        **今日核心数据:**
        ```json
        {json.dumps(report_payload, indent=2, ensure_ascii=False)}
        ```

        **晨报撰写要求:**
        1.  **开篇明义:** 首先对今日策略的核心逻辑和市场观点进行总结。
        2.  **持仓解读:** 清晰地列出最终的目标持仓股票及权重，并结合“核心动态因子权重”数据，解释为什么今天会选出这些股票（例如：“今日模型超配了动量因子，因此选出的股票普遍呈现强势上涨趋势”）。
        3.  **风险洞察:** **【核心要求】** 结合“组合风险暴露”数据，对当前持仓的整体风险风格进行精准分析和提示。例如：“请注意，当前持仓在小市值因子上有较高暴露，可能会在市场风格切换时产生较大波动。”
        4.  **格式清晰:** 使用Markdown格式，结构清晰，重点突出。
        5.  **结尾:** 附上必要的风险提示。
        """

        # c. 调用AI生成报告 (使用中端模型以平衡成本和效果)
        ai_report_content = ai_orchestrator._execute_ai_call(prompt, 'medium_balanced')
        
        # d. 将报告存入数据库，以供前端展示
        #    我们复用 ai_reports 表，但使用一个特殊的 ts_code 来标识晨报
        with dm.engine.connect() as conn:
            with conn.begin():
                upsert_sql = text("""
                    INSERT INTO ai_reports (trade_date, ts_code, report_content, model_used, estimated_cost)
                    VALUES (:date, :code, :content, :model, :cost)
                    ON CONFLICT (trade_date, ts_code) DO UPDATE 
                    SET report_content = EXCLUDED.report_content,
                        model_used = EXCLUDED.model_used,
                        estimated_cost = EXCLUDED.estimated_cost;
                """)
                cost_summary = ai_orchestrator.get_session_costs()
                conn.execute(upsert_sql, {
                    'date': trade_date,
                    'code': 'STRATEGY_MORNING_REPORT', # 特殊代码
                    'content': ai_report_content,
                    'model': cost_summary['model_used'],
                    'cost': cost_summary['estimated_cost']
                })
        log.info("AI晨报生成并存储成功！")

    except Exception as e:
        log.error("生成AI晨报过程中发生错误！", exc_info=True)

    log.info("===== 每日策略自动化工作流执行完毕 =====")


if __name__ == '__main__':
    run_daily_strategy_workflow()
    input("\n任务执行完毕，按 Enter 键退出...")