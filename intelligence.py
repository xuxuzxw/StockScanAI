# quant_project/intelligence.py
# 【V2.0 智能优化版】
import time
import pandas as pd
from typing import Dict, Any, Tuple
import json
from datetime import datetime, timedelta
from sqlalchemy import text
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import markdown2
import requests

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

    api_key = model_config.get('api_key')
    endpoint = model_config.get('endpoint')
    model_name = model_config.get('model_name')

    if not all([api_key, endpoint, model_name]):
        raise ValueError(f"模型 '{model_choice}' 的配置不完整 (api_key, endpoint, model_name)。")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **(model_config.get('params', {}))
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        response.raise_for_status()       
        response_json = response.json()
        
        if 'choices' in response_json and response_json['choices']:
            content = response_json['choices'][0]['message']['content']
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

    def summarize_text(self, text: str, task_description: str, model_choice='fast_and_cheap') -> str:
        prompt = f"Please summarize the following text for the purpose of '{task_description}':\n\n{text}"
        return self.orchestrator._execute_ai_call(prompt, model_choice)

    def extract_events(self, text: str, model_choice='medium_balanced') -> str:
        prompt = f"Please extract key events (like earnings, M&A, new products) from the text:\n\n{text}"
        return self.orchestrator._execute_ai_call(prompt, model_choice)

    def analyze_sentiment(self, text: str, model_choice='medium_balanced') -> str:
        prompt = f"Please analyze the sentiment of the text and provide a score from -1 (very negative) to 1 (very positive):\n\n{text}"
        return self.orchestrator._execute_ai_call(prompt, model_choice)

    def generate_narrative_report(self, structured_data: Dict, model_choice='powerful_and_expensive') -> str:
        """【智能升级版】生成报告前，先查询上下文信息"""
        ts_code = structured_data["股票代码"]
        end_date = structured_data["分析日期"]
        
        # --- 1. 获取行业对比上下文 ---
        try:
            stock_info = self.data_manager.get_stock_basic()
            industry = stock_info[stock_info['ts_code'] == ts_code].iloc[0]['industry']
            
            query = text(f"""
                SELECT factor_name, factor_value FROM factors_exposure
                WHERE trade_date = '{end_date}'
                  AND ts_code IN (SELECT ts_code FROM stock_basic WHERE industry = '{industry}')
            """)
            with self.data_manager.engine.connect() as conn:
                industry_data = pd.read_sql(query, conn)
            
            industry_avg = industry_data.groupby('factor_name')['factor_value'].mean().to_dict()
            
            # 将行业均值整合进原始数据
            for key, val in structured_data["技术面指标"].items():
                avg = industry_avg.get(key.lower(), 'N/A')
                if isinstance(avg, (int, float)): avg = f"{avg:.2f}"
                structured_data["技术面指标"][key] = f"{val} (行业均值: {avg})"
            
            for key, val in structured_data["基本面指标"].items():
                avg = industry_avg.get(key.lower(), 'N/A')
                if isinstance(avg, (int, float)): avg = f"{avg:.2f}"
                structured_data["基本面指标"][key] = f"{val} (行业均值: {avg})"

        except Exception as e:
            log.warning(f"为 {ts_code} 获取行业对比上下文失败: {e}")

        # --- 2. 获取历史趋势上下文 (以ROE为例) ---
        try:
            prev_date = (pd.to_datetime(end_date) - timedelta(days=90)).strftime('%Y%m%d')
            query = text(f"""
                SELECT factor_value FROM factors_exposure
                WHERE ts_code = '{ts_code}' AND factor_name = 'roe' AND trade_date < '{end_date}'
                ORDER BY trade_date DESC LIMIT 1
            """)
            with self.data_manager.engine.connect() as conn:
                prev_roe = conn.execute(query).scalar_one_or_none()
            
            if prev_roe:
                # 注意：此处应更新ROE，而不是营收同比增长
                roe_str = structured_data["基本面指标"].get("roe", "N/A")
                structured_data["基本面指标"]["roe"] = f"{roe_str} (上一季度: {prev_roe:.2%})"

        except Exception as e:
            log.warning(f"为 {ts_code} 获取历史趋势上下文失败: {e}")

        # --- 3. 构建最终的Prompt ---
        prompt = f"""
        请根据以下给出的、包含了**横向行业对比**和**纵向历史趋势**的结构化数据，生成一份全面、深入、专业的投资研究报告。

        **报告要求**:
        1.  **语言风格**: 专业、客观、逻辑清晰，模仿顶级券商分析师的口吻。
        2.  **核心结论**: 在报告开头以“投资要点”的形式，明确给出“强烈推荐”、“推荐”、“中性”或“回避”的核心投资建议。
        3.  **【关键要求】归因式论述**: 必须详细解释得出核心结论的理由。明确指出是哪个/哪些维度的信息（例如：技术面的“动量强于行业平均”、基本面的“ROE连续改善”或资金面的“主力资金持续流入”）共同支撑了你的判断。如果存在矛盾信号（如基本面优秀但估值过高），必须指出并进行深入分析。
        4.  **结构清晰**: 分别对技术面、资金面、基本面、筹码面、宏观环境和市场情绪进行分点论述，并在论述中自然地融入括号里的对比数据。
        5.  **风险提示**: 在报告末尾，必须包含对潜在风险的客观提示。

        **结构化数据 (括号内为对比信息)**:
        {json.dumps(structured_data, indent=2, ensure_ascii=False)}
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
        cost = (len(prompt.split()) + len(response_text.split())) / 1000 * self.model_config[model_choice]['cost_per_1k_tokens']
        self.call_log.append({
            "model": model_choice,
            "duration": duration,
            "cost": cost,
            "model_name": self.model_config[model_choice]['model_name']
        })
        return response_text

    def get_session_costs(self) -> Dict[str, Any]:
        """获取当前会话的成本统计"""
        if not self.call_log:
            return {"total_calls": 0, "estimated_cost": 0.0, "model_used": "N/A"}
        
        total_cost = sum(log['cost'] for log in self.call_log)
        model_used = self.call_log[-1]['model_name'] if self.call_log else "N/A"
        return {
            "total_calls": len(self.call_log),
            "estimated_cost": total_cost,
            "model_used": model_used
        }
    
    def reset_costs(self):
        """重置成本记录器"""
        self.call_log = []

def _send_email(subject: str, html_content: str, smtp_config: Dict):
    """【增强版】邮件发送辅助函数"""
    required_keys = ['host', 'port', 'user', 'password', 'sender', 'recipients']
    if not all(smtp_config.get(key) for key in required_keys) or not smtp_config.get('recipients'):
        missing_keys = [key for key in required_keys if not smtp_config.get(key)]
        print(f"SMTP配置不完整，跳过邮件发送。缺失或为空的配置项: {missing_keys}。请检查 .env 文件。")
        return False

    msg = MIMEMultipart()
    msg['From'] = smtp_config['sender']
    msg['To'] = ", ".join(smtp_config['recipients'])
    msg['Subject'] = subject
    msg.attach(MIMEText(html_content, 'html', 'utf-8'))

    try:
        server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
        server.starttls()
        server.login(smtp_config['user'], smtp_config['password'])
        server.sendmail(smtp_config['sender'], smtp_config['recipients'], msg.as_string())
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
    date_range: Tuple[str, str]
) -> Tuple[str, Dict]:
    """【缓存与智能优化版】工作流"""
    orchestrator.reset_costs()
    start_date, end_date = date_range

    # --- 1. 【新增】前置缓存检查 ---
    log.info(f"正在为 {ts_code} on {end_date} 检查AI报告缓存...")
    try:
        with data_manager.engine.connect() as conn:
            query = text("SELECT report_content, model_used, estimated_cost FROM ai_reports WHERE trade_date = :date AND ts_code = :code")
            cached_report = conn.execute(query, {'date': end_date, 'code': ts_code}).fetchone()
        
        if cached_report:
            log.info(f"命中缓存！直接从数据库加载报告。")
            report_content, model_used, cost = cached_report
            cost_summary = {"total_calls": 0, "estimated_cost": cost, "model_used": model_used}
            return f"**(报告来自缓存)**\n\n{report_content}", cost_summary
    except Exception as e:
        log.error(f"检查AI报告缓存时出错: {e}", exc_info=True)

    log.info("未命中缓存，开始执行实时AI分析...")
    # --- 2. 实时分析流程 ---
    latest_date_dt = datetime.strptime(end_date, '%Y%m%d')
    try:
        momentum = factor_factory.calc_momentum(ts_code, start_date, end_date)
        volatility = factor_factory.calc_volatility(ts_code, start_date, end_date)
        net_inflow = factor_factory.calc_net_inflow_ratio(ts_code, start_date, end_date)
        north_hold = factor_factory.calc_north_hold_change(ts_code, start_date, end_date)
        roe = factor_factory.calc_roe(ts_code, date=end_date) # 新增ROE获取
        growth = factor_factory.calc_growth_revenue_yoy(ts_code, date=end_date)
        debt_ratio = factor_factory.calc_debt_to_assets(ts_code, date=end_date)
        holder_ratio = factor_factory.calc_top10_holder_ratio(ts_code, date=end_date)
        df_pmi = data_manager.get_cn_pmi(f"{latest_date_dt.year-2}{latest_date_dt.month:02d}", f"{latest_date_dt.year}{latest_date_dt.month:02d}")
        latest_pmi = df_pmi.iloc[-1].to_dict() if df_pmi is not None and not df_pmi.empty else "N/A"
        news_df = data_manager.get_text_for_analysis(ts_code=ts_code, limit=3)
        news_text = "\n".join([f"标题: {row['title']}\n内容: {row['content'][:100]}..." for _, row in news_df.iterrows()]) if not news_df.empty else "无新闻数据"

        summary = orchestrator.tools.summarize_text(news_text, "新闻摘要", 'fast_and_cheap')
        if "错误" in summary:
            return f"AI工作流在“摘要”步骤失败: {summary}", orchestrator.get_session_costs()

        events = orchestrator.tools.extract_events(summary, 'medium_balanced')
        sentiment = orchestrator.tools.analyze_sentiment(summary, 'medium_balanced')
        
        final_data_payload = {
            "股票代码": ts_code,
            "分析日期": end_date,
            "技术面指标": {"momentum": f"{momentum:.2%}", "volatility": f"{volatility:.2%}"},
            "资金面指标": {"net_inflow_ratio": f"{net_inflow:.2%}", "north_hold_change": f"{north_hold:.2f}%"},
            "基本面指标": {"roe": f"{roe:.2%}", "growth_revenue_yoy": f"{growth:.2%}", "debt_to_assets": f"{debt_ratio:.2%}"},
            "筹码面指标": {"top10_holder_ratio": f"{holder_ratio:.2%}"},
            "宏观环境": {"最新PMI数据": latest_pmi},
            "语义分析结果": {"市场情绪": sentiment, "关键事件": events}
        }
        
        final_report = orchestrator.tools.generate_narrative_report(final_data_payload, 'powerful_and_expensive')
    
    except Exception as e:
        final_report = f"执行AI工作流时发生未知内部错误: {e}"
    
    cost_summary = orchestrator.get_session_costs()

    # --- 3. 【新增】结果存入缓存 ---
    if "错误" not in final_report:
        log.info(f"正在将新生成的报告存入缓存...")
        try:
            with data_manager.engine.connect() as conn:
                with conn.begin():
                    upsert_sql = text("""
                        INSERT INTO ai_reports (trade_date, ts_code, report_content, model_used, estimated_cost)
                        VALUES (:date, :code, :content, :model, :cost)
                        ON CONFLICT (trade_date, ts_code) DO UPDATE 
                        SET report_content = EXCLUDED.report_content,
                            model_used = EXCLUDED.model_used,
                            estimated_cost = EXCLUDED.estimated_cost;
                    """)
                    conn.execute(upsert_sql, {
                        'date': end_date,
                        'code': ts_code,
                        'content': final_report,
                        'model': cost_summary['model_used'],
                        'cost': cost_summary['estimated_cost']
                    })
            log.info("报告缓存成功。")
        except Exception as e:
            log.error(f"AI报告写入缓存失败: {e}", exc_info=True)

    return final_report, cost_summary

def generate_and_send_report_workflow(
    orchestrator: AIOrchestrator,
    data_manager: data.DataManager,
    factor_factory: factors.FactorFactory,
    ts_code: str
):
    """
    【增强版自动化工作流】为单个股票生成AI分析与量化数据综合报告，并将其通过邮件发送。
    """
    print(f"开始为 {ts_code} 生成自动化投研报告...")
    
    # 1. 定义分析周期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = (start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
    
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
        stock_info = stock_basics[stock_basics['ts_code'] == ts_code].iloc[0]
        stock_name = stock_info['name']
        
        # 获取最新财务指标
        df_fina = data_manager.get_fina_indicator(ts_code).sort_values('end_date', ascending=False).head(1).T
        df_fina.columns = ['最新财报']
        fina_html = df_fina.to_html(classes='table table-striped', border=0)

        # 获取最新十大股东
        df_holders = pd.DataFrame()
        end_date_dt = datetime.now()
        for year_offset in range(2):
            year = end_date_dt.year - year_offset
            periods = [f"{year}1231", f"{year}0930", f"{year}0630", f"{year}0331"]
            for p in periods:
                if p <= end_date_dt.strftime('%Y%m%d'):
                    df_temp = data_manager.get_top10_floatholders(ts_code, p)
                    if df_temp is not None and not df_temp.empty:
                        df_holders = df_temp
                        break
            if not df_holders.empty:
                break
        
        holders_html = df_holders.to_html(classes='table table-striped', index=False, border=0) if not df_holders.empty else "<p>股东数据获取失败。</p>"

    except Exception as e:
        print(f"抓取 {ts_code} 的额外量化数据时出错: {e}")
        fina_html = "<p>财务数据获取失败。</p>"
        holders_html = "<p>股东数据获取失败。</p>"

    # 4. 渲染并发送邮件
    subject = f"A股智能投研周报 - {stock_name} ({ts_code}) - {end_date.strftime('%Y-%m-%d')}"
    
    html_body = markdown2.markdown(
        markdown_report, 
        extras=["fenced-code-blocks", "tables", "header-ids"]
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