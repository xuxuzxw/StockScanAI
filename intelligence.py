# StockScanAI/intelligence.py
# 【V2.0 智能优化版】
import json
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Tuple

import markdown2
import pandas as pd
import requests
from sqlalchemy import text

# 导入项目模块
import config
import data
import quant_engine as factors
from logger_config import log


def _real_ai_call(prompt: str, model_choice: str) -> str:
    """【已激活】执行对真实大模型API的HTTP请求。"""
    model_config = config.AI_MODEL_CONFIG.get(model_choice)
    if not model_config:
        raise ValueError(f"未知的模型选择: {model_choice}")

    api_key = model_config.get("api_key")
    endpoint = model_config.get("endpoint")
    model_name = model_config.get("model_name")

    if not all([api_key, endpoint, model_name]):
        raise ValueError(
            f"模型 '{model_choice}' 的配置不完整 (api_key, endpoint, model_name)。"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        **(model_config.get("params", {})),
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        response_json = response.json()

        if "choices" in response_json and response_json["choices"]:
            content = response_json["choices"][0]["message"]["content"]
            return content.strip()
        else:
            raise ValueError(f"API响应格式不符合预期: {response_json}")

    except requests.exceptions.RequestException as e:
        error_message = f"AI API请求失败 (模型: {model_name}): {e}"
        log.error(error_message)
        return f"错误：调用AI模型失败。详情: {error_message}"
    except Exception as e:
        error_message = f"处理AI响应时发生错误 (模型: {model_name}): {e}"
        log.error(error_message)
        return f"错误：处理AI响应失败。详情: {error_message}"


class AITools:
    """将每个AI能力封装成一个独立的、可被编排器调用的“工具”。"""

    def __init__(self, orchestrator: "AIOrchestrator"):
        self.orchestrator = orchestrator
        self.data_manager = orchestrator.data_manager

    def summarize_text(
        self, text: str, task_description: str, model_choice="fast_and_cheap"
    ) -> str:
        prompt = f"Please summarize the following text for the purpose of '{task_description}':\n\n{text}"
        return self.orchestrator._execute_ai_call(prompt, model_choice)

    def extract_events(self, text: str, model_choice="medium_balanced") -> str:
        prompt = f"Please extract key events (like earnings, M&A, new products) from the text:\n\n{text}"
        return self.orchestrator._execute_ai_call(prompt, model_choice)

    def analyze_sentiment(self, text: str, model_choice="medium_balanced") -> str:
        prompt = f"Please analyze the sentiment of the text and provide a score from -1 (very negative) to 1 (very positive):\n\n{text}"
        return self.orchestrator._execute_ai_call(prompt, model_choice)

    def generate_narrative_report(
        self,
        structured_data: Dict,
        old_reports: list = None,
        model_choice="powerful_and_expensive",
    ) -> str:
        """【V2.1增强版】生成报告前，查询上下文信息，并参考历史报告。"""
        ts_code = structured_data["股票代码"]
        end_date = structured_data["分析日期"]

        # --- V2.1 新增：构建历史报告上下文 ---
        historical_context = "无历史报告可供参考。"
        if old_reports:
            historical_context = ""
            for i, report in enumerate(old_reports):
                historical_context += (
                    f"--- 历史报告 {i+1} (日期: {report['trade_date']}) ---\n"
                )
                historical_context += f"{report['report_content']}\n\n"

        # --- 1. 获取行业对比上下文 ---
        try:
            stock_info = self.data_manager.get_stock_basic()
            industry = stock_info[stock_info["ts_code"] == ts_code].iloc[0]["industry"]

            query = text(
                f"""
                SELECT factor_name, factor_value FROM factors_exposure
                WHERE trade_date = '{end_date}'
                  AND ts_code IN (SELECT ts_code FROM stock_basic WHERE industry = '{industry}')
            """
            )
            with self.data_manager.engine.connect() as conn:
                industry_data = pd.read_sql(query, conn)

            industry_avg = (
                industry_data.groupby("factor_name")["factor_value"].mean().to_dict()
            )

            # 将行业均值整合进原始数据
            for key, val in structured_data["技术面指标"].items():
                avg = industry_avg.get(key.lower(), "N/A")
                if isinstance(avg, (int, float)):
                    avg = f"{avg:.2f}"
                structured_data["技术面指标"][key] = f"{val} (行业均值: {avg})"

            for key, val in structured_data["基本面指标"].items():
                avg = industry_avg.get(key.lower(), "N/A")
                if isinstance(avg, (int, float)):
                    avg = f"{avg:.2f}"
                structured_data["基本面指标"][key] = f"{val} (行业均值: {avg})"

        except Exception as e:
            log.warning(f"为 {ts_code} 获取行业对比上下文失败: {e}")

        # --- 2. 获取历史趋势上下文 (以ROE为例) ---
        try:
            prev_date = (pd.to_datetime(end_date) - timedelta(days=90)).strftime(
                "%Y%m%d"
            )
            query = text(
                f"""
                SELECT factor_value FROM factors_exposure
                WHERE ts_code = '{ts_code}' AND factor_name = 'roe' AND trade_date < '{end_date}'
                ORDER BY trade_date DESC LIMIT 1
            """
            )
            with self.data_manager.engine.connect() as conn:
                prev_roe = conn.execute(query).scalar_one_or_none()

            if prev_roe:
                roe_str = structured_data["基本面指标"].get("roe", "N/A")
                structured_data["基本面指标"][
                    "roe"
                ] = f"{roe_str} (上一季度: {prev_roe:.2f})"

        except Exception as e:
            log.warning(f"为 {ts_code} 获取历史趋势上下文失败: {e}")

        # --- 3. 构建最终的Prompt ---
        prompt = f"""
        作为一名顶级的A股量化分析师，请根据以下信息，生成一份全面、深入、专业的投资研究报告。

        **第一部分：历史报告回顾**
        这是最近的2份关于该公司的历史分析报告，请你参考其中的观点和数据变化，以形成对公司发展趋势的动态认知。
        {historical_context}

        **第二部分：最新的结构化数据**
        这是截止到 {end_date} 的最新数据，其中括号内为行业或历史对比信息。
        {json.dumps(structured_data, indent=2, ensure_ascii=False)}

        **报告撰写要求**:
        1.  **动态视角**: **【核心要求】** 你的分析必须体现出动态和发展的视角。结合历史报告和最新数据，明确指出公司的积极或消极变化趋势。例如，“与上一份报告相比，公司的ROE出现显著改善，表明其盈利能力正在增强”，或者“注意到股东人数连续两个报告期持续集中，这是一个积极信号”。
        2.  **核心结论**: 在报告开头以“投资要点”的形式，明确给出“强烈推荐”、“推荐”、“中性”或“回避”的核心投资建议。
        3.  **归因式论述**: 必须详细解释得出核心结论的理由。明确指出是哪个/哪些维度的信息（特别是对比历史后发现的变化趋势）共同支撑了你的判断。
        4.  **【关键】完整性要求**: 你的报告必须覆盖“最新的结构化数据”中提供的**所有指标**。如果某个指标的值为 "N/A"，你必须明确指出该项指标，并说明“近期无相关数据”或“数据不适用”。**严禁在报告中忽略任何值为 "N/A" 的指标。**
        5.  **结构清晰**: 分别对技术面、资金面、基本面、筹码面等进行分点论述。
        6.  **风险提示**: 在报告末尾，必须包含对潜在风险的客观提示。
        """
        return self.orchestrator._execute_ai_call(prompt, model_choice)


class AIOrchestrator:
    """智能体的大脑，负责任务编排、模型选择和成本控制。"""

    def __init__(self, model_config: Dict, data_manager: data.DataManager):
        self.model_config = model_config
        self.data_manager = data_manager
        self.tools = AITools(self)
        self.call_log = []

    def _execute_ai_call(self, prompt: str, model_choice: str) -> str:
        """执行AI调用并记录日志"""
        start_time = time.time()
        response_text = _real_ai_call(prompt, model_choice)
        duration = time.time() - start_time
        cost = (
            (len(prompt.split()) + len(response_text.split()))
            / 1000
            * self.model_config[model_choice]["cost_per_1k_tokens"]
        )
        self.call_log.append(
            {
                "model": model_choice,
                "duration": duration,
                "cost": cost,
                "model_name": self.model_config[model_choice]["model_name"],
            }
        )
        return response_text

    def get_session_costs(self) -> Dict[str, Any]:
        """获取当前会话的成本统计"""
        if not self.call_log:
            return {"total_calls": 0, "estimated_cost": 0.0, "model_used": "N/A"}

        total_cost = sum(log["cost"] for log in self.call_log)
        model_used = self.call_log[-1]["model_name"] if self.call_log else "N/A"
        return {
            "total_calls": len(self.call_log),
            "estimated_cost": total_cost,
            "model_used": model_used,
        }

    def reset_costs(self):
        """重置成本记录器"""
        self.call_log = []


def _send_email(subject: str, html_content: str, smtp_config: Dict):
    """【增强版】邮件发送辅助函数"""
    required_keys = ["host", "port", "user", "password", "sender", "recipients"]
    if not all(smtp_config.get(key) for key in required_keys) or not smtp_config.get(
        "recipients"
    ):
        missing_keys = [key for key in required_keys if not smtp_config.get(key)]
        print(
            f"SMTP配置不完整，跳过邮件发送。缺失或为空的配置项: {missing_keys}。请检查 .env 文件。"
        )
        return False

    msg = MIMEMultipart()
    msg["From"] = smtp_config["sender"]
    msg["To"] = ", ".join(smtp_config["recipients"])
    msg["Subject"] = subject
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
        server.starttls()
        server.login(smtp_config["user"], smtp_config["password"])
        server.sendmail(
            smtp_config["sender"], smtp_config["recipients"], msg.as_string()
        )
        server.quit()
        print(f"邮件成功发送至: {msg['To']}")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False


def full_analysis_workflow(
    orchestrator: AIOrchestrator,
    data_manager: data.DataManager,
    factor_factory: factors.FactorFactory,
    ts_code: str,
    date_range: Tuple[str, str],
) -> Tuple[str, Dict]:
    """【缓存与智能优化版】工作流"""
    orchestrator.reset_costs()
    start_date, end_date = date_range

    # --- 1. 【增强版】前置缓存检查 ---
    log.info(f"正在为 {ts_code} on {end_date} 检查AI报告缓存...")
    try:
        with data_manager.engine.connect() as conn:
            # 检查当日报告
            query_today = text(
                "SELECT report_content, model_used, estimated_cost FROM ai_reports WHERE trade_date = :date AND ts_code = :code"
            )
            cached_report_today = conn.execute(
                query_today, {"date": end_date, "code": ts_code}
            ).fetchone()

            if cached_report_today:
                log.info(f"命中当日缓存！直接从数据库加载报告。")
                report_content, model_used, cost = cached_report_today
                cost_summary = {
                    "total_calls": 0,
                    "estimated_cost": cost,
                    "model_used": model_used,
                }
                return f"**(报告来自当日缓存)**\n\n{report_content}", cost_summary

            # 获取最近的2份旧报告
            query_old = text(
                """
                SELECT trade_date, report_content FROM ai_reports 
                WHERE ts_code = :code AND trade_date < :date 
                ORDER BY trade_date DESC 
                LIMIT 2
            """
            )
            old_reports_cursor = conn.execute(
                query_old, {"code": ts_code, "date": end_date}
            )
            old_reports = [dict(row) for row in old_reports_cursor.mappings()]

    except Exception as e:
        log.error(f"检查或获取AI报告缓存时出错: {e}", exc_info=True)
        old_reports = []

    log.info("未命中当日缓存，开始执行实时AI分析...")
    # --- 2. 实时分析流程 ---
    latest_date_dt = datetime.strptime(end_date, "%Y%m%d")
    try:
        momentum = factor_factory.calc_momentum(ts_code, start_date, end_date)
        volatility = factor_factory.calc_volatility(ts_code, start_date, end_date)
        net_inflow = factor_factory.calc_net_inflow_ratio(ts_code, start_date, end_date)
        north_hold = factor_factory.calc_north_hold_change(
            ts_code, start_date, end_date
        )
        roe = factor_factory.calc_roe(ts_code, date=end_date)  # 新增ROE获取
        growth = factor_factory.calc_growth_revenue_yoy(ts_code, date=end_date)
        debt_ratio = factor_factory.calc_debt_to_assets(ts_code, date=end_date)
        holder_ratio = factor_factory.calc_top10_holder_ratio(ts_code, date=end_date)
        df_pmi = data_manager.get_cn_pmi(
            f"{latest_date_dt.year-2}{latest_date_dt.month:02d}",
            f"{latest_date_dt.year}{latest_date_dt.month:02d}",
        )
        latest_pmi = (
            df_pmi.iloc[-1].to_dict()
            if df_pmi is not None and not df_pmi.empty
            else "N/A"
        )
        news_df = data_manager.get_text_for_analysis(ts_code=ts_code, limit=3)
        news_text = (
            "\n".join(
                [
                    f"标题: {row['title']}\n内容: {row['content'][:100]}..."
                    for _, row in news_df.iterrows()
                ]
            )
            if not news_df.empty
            else "无新闻数据"
        )

        summary = orchestrator.tools.summarize_text(
            news_text, "新闻摘要", "fast_and_cheap"
        )
        if "错误" in summary:
            return (
                f"AI工作流在“摘要”步骤失败: {summary}",
                orchestrator.get_session_costs(),
            )

        events = orchestrator.tools.extract_events(summary, "medium_balanced")
        sentiment = orchestrator.tools.analyze_sentiment(summary, "medium_balanced")

        # V2.1 新增：获取筹码类因子
        holder_num_change = factor_factory.calc_holder_num_change_ratio(
            ts_code, date=end_date
        )
        major_shareholder_trade = factor_factory.calc_major_shareholder_net_buy_ratio(
            ts_code, date=end_date
        )
        # 获取当日龙虎榜数据
        top_list_df = data_manager.get_top_list(trade_date=end_date)
        top_list_net_buy = factor_factory.calc_top_list_net_buy_amount(
            ts_code, end_date, top_list_df
        )

        # V2.1 新增：获取价值回报类因子
        dividend_yield = factor_factory.calc_dividend_yield(ts_code, date=end_date)
        forecast_growth = factor_factory.calc_forecast_growth_rate(
            ts_code, date=end_date
        )
        repurchase_ratio = factor_factory.calc_repurchase_ratio(ts_code, date=end_date)
        # 获取大宗交易数据
        block_trade_df = data_manager.get_block_trade(trade_date=end_date)
        block_trade_ratio = factor_factory.calc_block_trade_ratio(
            ts_code, end_date, block_trade_df
        )

        # --- V2.4 新增：计算并解读技术指标 ---
        tech_signals = {}
        try:
            # MACD
            macd_data = factor_factory.calc_macd(ts_code, end_date)
            if all(pd.notna(v) for v in macd_data.values()):
                status = (
                    "金叉"
                    if macd_data["macd_hist"] > 0
                    and (
                        factor_factory.calc_macd(
                            ts_code,
                            (
                                datetime.strptime(end_date, "%Y%m%d")
                                - timedelta(days=5)
                            ).strftime("%Y%m%d"),
                        )["macd_hist"]
                        <= 0
                    )
                    else (
                        "死叉"
                        if macd_data["macd_hist"] < 0
                        and (
                            factor_factory.calc_macd(
                                ts_code,
                                (
                                    datetime.strptime(end_date, "%Y%m%d")
                                    - timedelta(days=5)
                                ).strftime("%Y%m%d"),
                            )["macd_hist"]
                            >= 0
                        )
                        else "多头" if macd_data["diff"] > macd_data["dea"] else "空头"
                    )
                )
                tech_signals["MACD"] = (
                    f"DIFF: {macd_data['diff']:.2f}, DEA: {macd_data['dea']:.2f}, 状态: {status}"
                )

            # RSI
            rsi_value = factor_factory.calc_rsi(ts_code, end_date)
            if pd.notna(rsi_value):
                rsi_status = (
                    "超买区"
                    if rsi_value > 70
                    else "超卖区" if rsi_value < 30 else "适中区"
                )
                tech_signals["RSI(14)"] = f"数值: {rsi_value:.2f} ({rsi_status})"

            # BOLL
            boll_data = factor_factory.calc_boll(ts_code, end_date)
            if all(pd.notna(v) for v in boll_data.values()):
                boll_status = (
                    "触及上轨"
                    if boll_data["close"] >= boll_data["upper"]
                    else (
                        "触及下轨"
                        if boll_data["close"] <= boll_data["lower"]
                        else "通道内运行"
                    )
                )
                tech_signals["BOLL(20)"] = (
                    f"上轨: {boll_data['upper']:.2f}, 中轨: {boll_data['middle']:.2f}, 下轨: {boll_data['lower']:.2f}, 状态: {boll_status}"
                )

        except Exception as e:
            log.warning(f"为 {ts_code} 计算V2.4技术指标时出错: {e}")

        final_data_payload = {
            "股票代码": ts_code,
            "分析日期": end_date,
            "技术面指标": {
                "momentum": f"{momentum:.2%}",
                "volatility": f"{volatility:.2%}",
            },
            "技术信号解读": tech_signals,  # V2.4 新增
            "资金面指标": {
                "net_inflow_ratio": f"{net_inflow:.2%}",
                "north_hold_change": f"{north_hold:.2f}%",
            },
            "基本面指标": {
                "roe": f"{roe:.2%}",
                "growth_revenue_yoy": f"{growth:.2%}",
                "debt_to_assets": f"{debt_ratio:.2%}",
                # V2.1 新增
                "dividend_yield_ttm": (
                    f"{dividend_yield:.2%}" if pd.notna(dividend_yield) else "N/A"
                ),
                "forecast_growth_rate": (
                    f"{forecast_growth:.2%}" if pd.notna(forecast_growth) else "N/A"
                ),
                "repurchase_ratio_1y": (
                    f"{repurchase_ratio:.4f}%" if pd.notna(repurchase_ratio) else "N/A"
                ),
            },
            "筹码面指标": {
                "top10_holder_ratio": f"{holder_ratio:.2%}",
                "holder_num_change_ratio": (
                    f"{holder_num_change:.2%}" if pd.notna(holder_num_change) else "N/A"
                ),
                "major_shareholder_net_buy_ratio": (
                    f"{major_shareholder_trade:.4f}%"
                    if pd.notna(major_shareholder_trade)
                    else "N/A"
                ),
                "top_list_net_buy_amount_sum": (
                    f"{top_list_net_buy:,.2f} 万元"
                    if pd.notna(top_list_net_buy)
                    else "0.00 万元"
                ),
                "block_trade_ratio_90d": (
                    f"{block_trade_ratio:.2f}%"
                    if pd.notna(block_trade_ratio)
                    else "N/A"
                ),
            },
            "宏观环境": {"最新PMI数据": latest_pmi},
            "语义分析结果": {"市场情绪": sentiment, "关键事件": events},
        }

        final_report = orchestrator.tools.generate_narrative_report(
            structured_data=final_data_payload,
            old_reports=old_reports,
            model_choice="powerful_and_expensive",
        )

    except Exception as e:
        final_report = f"执行AI工作流时发生未知内部错误: {e}"

    cost_summary = orchestrator.get_session_costs()

    # --- 3. 【新增】结果存入缓存 ---
    if "错误" not in final_report:
        log.info(f"正在将新生成的报告存入缓存...")
        try:
            with data_manager.engine.connect() as conn:
                with conn.begin():
                    upsert_sql = text(
                        """
                        INSERT INTO ai_reports (trade_date, ts_code, report_content, model_used, estimated_cost)
                        VALUES (:date, :code, :content, :model, :cost)
                        ON CONFLICT (trade_date, ts_code) DO UPDATE 
                        SET report_content = EXCLUDED.report_content,
                            model_used = EXCLUDED.model_used,
                            estimated_cost = EXCLUDED.estimated_cost;
                    """
                    )
                    conn.execute(
                        upsert_sql,
                        {
                            "date": end_date,
                            "code": ts_code,
                            "content": final_report,
                            "model": cost_summary["model_used"],
                            "cost": cost_summary["estimated_cost"],
                        },
                    )
            log.info("报告缓存成功。")

            # --- V2.1 新增：调用报告清理任务 ---
            data_manager.purge_old_ai_reports(ts_code, keep_latest=10)

        except Exception as e:
            log.error(f"AI报告写入或清理失败: {e}", exc_info=True)

    return final_report, cost_summary


def generate_and_send_report_workflow(
    orchestrator: AIOrchestrator,
    data_manager: data.DataManager,
    factor_factory: factors.FactorFactory,
    ts_code: str,
):
    """
    【增强版自动化工作流】为单个股票生成AI分析与量化数据综合报告，并将其通过邮件发送。
    """
    print(f"开始为 {ts_code} 生成自动化投研报告...")

    # 1. 定义分析周期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

    # 2. 调用核心AI分析流程
    try:
        markdown_report, cost = full_analysis_workflow(
            orchestrator, data_manager, factor_factory, ts_code, date_range
        )
    except Exception as e:
        print(f"为 {ts_code} 执行AI分析时出错: {e}")
        return

    # 3. 抓取额外的量化数据
    try:
        stock_basics = data_manager.get_stock_basic()
        stock_info = stock_basics[stock_basics["ts_code"] == ts_code].iloc[0]
        stock_name = stock_info["name"]

        # 获取最新财务指标
        df_fina = (
            data_manager.get_fina_indicator(ts_code)
            .sort_values("end_date", ascending=False)
            .head(1)
            .T
        )
        df_fina.columns = ["最新财报"]
        fina_html = df_fina.to_html(classes="table table-striped", border=0)

        # 获取最新十大股东
        df_holders = pd.DataFrame()
        end_date_dt = datetime.now()
        for year_offset in range(2):
            year = end_date_dt.year - year_offset
            periods = [f"{year}1231", f"{year}0930", f"{year}0630", f"{year}0331"]
            for p in periods:
                if p <= end_date_dt.strftime("%Y%m%d"):
                    df_temp = data_manager.get_top10_floatholders(ts_code, p)
                    if df_temp is not None and not df_temp.empty:
                        df_holders = df_temp
                        break
            if not df_holders.empty:
                break

        holders_html = (
            df_holders.to_html(classes="table table-striped", index=False, border=0)
            if not df_holders.empty
            else "<p>股东数据获取失败。</p>"
        )

    except Exception as e:
        print(f"抓取 {ts_code} 的额外量化数据时出错: {e}")
        fina_html = "<p>财务数据获取失败。</p>"
        holders_html = "<p>股东数据获取失败。</p>"

    # 4. 渲染并发送邮件
    subject = (
        f"A股智能投研周报 - {stock_name} ({ts_code}) - {end_date.strftime('%Y-%m-%d')}"
    )

    html_body = markdown2.markdown(
        markdown_report, extras=["fenced-code-blocks", "tables", "header-ids"]
    )

    html_report = f"""
    <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; }}
                h1, h2, h3, h4 {{ color: #1a1a1a; border-bottom: 1px solid #eee; padding-bottom: 5px;}}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; border: 1px solid #ddd; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f7f7f7; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #fdfdfd; }}
                .container {{ padding: 20px; max-width: 800px; margin: auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
                .footer {{ font-size: 0.8em; color: #888; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI智能投研报告 - {stock_name} ({ts_code})</h1>
                <hr>
                
                <h2>Part 1: AI 混合智能体分析</h2>
                {html_body}
                
                <h2>Part 2: 核心量化数据快照</h2>
                
                <h3>最新财务指标</h3>
                {fina_html}
                
                <h3>最新前十大流通股东</h3>
                {holders_html}

                <div class="footer">
                    <p>本报告由A股量化投研平台于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 自动生成。</p>
                    <p>AI分析成本: {cost.get('total_calls', 'N/A')} 次调用, 预估 ${cost.get('estimated_cost', 0):.4f}。仅供研究参考, 不构成投资建议。</p>
                </div>
            </div>
        </body>
    </html>
    """

    _send_email(subject, html_report, config.SMTP_CONFIG)
    print(f"为 {ts_code} 生成并发送综合报告的工作流已完成。")


def triggered_analysis_workflow(
    orchestrator: "AIOrchestrator",
    data_manager: "data.DataManager",
    ts_code: str,
    trigger_event: str,
) -> str:
    """
    【V2.4新增】由量化信号触发的轻量级、低成本AI分析工作流。
    - 实现了 "量化信号触发式AI分析" 的规划。
    """
    log.info(f"接收到 {ts_code} 的触发事件: '{trigger_event}'，启动轻量级AI分析...")
    orchestrator.reset_costs()

    # 1. 根据触发事件，采集必要信息
    # 此处为简化示例，实际应根据事件类型采集不同数据
    df_minute = data_manager.get_minute_bars_ak(ts_code)
    if df_minute.empty:
        return f"无法获取 {ts_code} 的分钟线数据，分析中止。"

    latest_bar = df_minute.iloc[-1]
    price_change_5min = (latest_bar["close"] / df_minute.iloc[-2]["close"] - 1) * 100

    context = f"""
    股票 {ts_code} 在最新的5分钟K线上价格剧烈变动 {price_change_5min:.2f}%，
    且成交量放大至 {latest_bar['vol']}手。
    """

    # 2. 构建针对性的、低成本的AI Prompt
    prompt = f"""
    作为一名盘中交易分析师，请根据以下实时异动信息，快速分析可能的原因，并给出简短的操作建议。

    **异动信息:**
    {context}

    **分析要求:**
    1.  推测可能导致该异动的原因（如：突发新闻、板块联动、大单买入/卖出）。
    2.  给出简明扼要的应对策略（如：观察后续走势、检查相关新闻、小仓位试探）。
    3.  回答必须简短、直接、适合盘中快速决策。
    """

    # 3. 使用低成本模型进行分析
    report = orchestrator._execute_ai_call(prompt, "fast_and_cheap")
    cost = orchestrator.get_session_costs()

    log.info(f"轻量级AI分析完成。成本: ${cost['estimated_cost']:.5f}")
    return report
