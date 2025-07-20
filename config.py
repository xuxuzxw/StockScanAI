# quant_project/config.py

import os

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# --- Tushare API 配置 ---
# 请在项目根目录创建 .env 文件，并填入 TUSHARE_TOKEN
# 格式: TUSHARE_TOKEN="your_token_here"
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")

# --- 数据库配置 (V2.0 TimescaleDB) ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# 构建SQLAlchemy数据库连接URL
# postgresql+psycopg2://<user>:<password>@<host>:<port>/<dbname>
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# 旧的 DB_PATH 不再需要
# DB_PATH = "data/a_stock_quant_v2.db"

# --- AI 模型配置 ---
# 设计目标是支持多模型、分级调用
AI_MODEL_CONFIG = {
    "fast_and_cheap": {
        "api_key": os.getenv("AI_API_KEY_FAST"),
        "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
        "model_name": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "cost_per_1k_tokens": 0.000,
        "params": {
            "max_tokens": 8192,  # 控制最大输出长度
            "temperature": 0.5,  # 控制创造性，0.0最确定，1.0最随机
        },
    },
    "medium_balanced": {
        "api_key": os.getenv("AI_API_KEY_MEDIUM"),
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model_name": "qwen-plus-latest",
        "cost_per_1k_tokens": 0.002,
        "params": {
            "max_tokens": 8192,
            "temperature": 0.7,
        },
    },
    "powerful_and_expensive": {
        "api_key": os.getenv("AI_API_KEY_POWERFUL"),
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model_name": "deepseek-reasoner",
        "cost_per_1k_tokens": 0.016,
        "params": {
            "max_tokens": 8192,
            "temperature": 0.75,
        },
    },
}

# --- 模拟非结构化数据源 ---
# 在实际项目中，这可能是一个爬虫模块或API
SIMULATED_NEWS_SOURCE = {
    "000001.SZ": [
        {
            "title": "平安银行发布年度财报，净利润超预期增长",
            "content": "平安银行今日公布了其年度财务报告，报告显示，公司净利润同比增长15%，显著高于市场分析师的普遍预期。报告将增长归因于其零售银行业务的强劲扩张和有效的成本控制措施...",
        },
        {
            "title": "行业分析：银行业面临数字化转型挑战",
            "content": "一份最新的行业研究报告指出，传统银行业正面临来自金融科技公司的激烈竞争，数字化转型已成为所有银行的当务之急。报告强调了移动支付、人工智能客服和大数据风控的重要性...",
        },
        {
            "title": "宏观经济数据发布，M2增速放缓",
            "content": "国家统计局今日发布了最新的宏观经济数据，其中广义货币（M2）供应量增速有所放缓，这可能预示着未来货币政策将趋于谨慎。分析人士认为，这将对银行的信贷扩张构成一定压力...",
        },
    ],
    "600519.SH": [
        {
            "title": "贵州茅台宣布扩大产能计划",
            "content": "为满足持续增长的市场需求，贵州茅台今日宣布了一项重大的产能扩张计划。该计划预计将在未来五年内，将其基酒产能提升20%。公司管理层表示，此举旨在巩固其在高端白酒市场的领导地位...",
        },
        {
            "title": "高端消费市场趋势报告",
            "content": "一份关于中国高端消费市场的报告显示，尽管宏观经济存在不确定性，但对奢侈品和高端服务的需求依然强劲，尤其是在高净值人群中。报告特别指出，具有强大品牌护城河和文化价值的产品持续受到追捧...",
        },
        {
            "title": "监管机构提示白酒行业关注渠道库存风险",
            "content": "近期，相关监管机构向白酒行业发出风险提示，要求各大酒企密切关注渠道库存水平，防止过度压货导致市场价格波动。此举旨在促进该行业的健康、可持续发展...",
        },
    ],
}

# --- 自动化报告配置 ---
# 用于定时任务，生成并发送报告的股票列表
AUTOMATED_REPORT_STOCKS = ["600519.SH", "000001.SZ", "300750.SZ"]

# --- SMTP 邮件发送配置 ---
# 请在 .env 文件中配置以下环境变量
# SMTP_HOST="smtp.example.com"
# SMTP_PORT=587
# SMTP_USER="your_email@example.com"
# SMTP_PASSWORD="your_password"
# EMAIL_SENDER="your_email@example.com"
# EMAIL_RECIPIENTS="recipient1@example.com,recipient2@example.com"
SMTP_CONFIG = {
    "host": os.getenv("SMTP_HOST"),
    "port": int(os.getenv("SMTP_PORT", 587)),
    "user": os.getenv("SMTP_USER"),
    "password": os.getenv("SMTP_PASSWORD"),
    "sender": os.getenv("EMAIL_SENDER"),
    "recipients": [rec for rec in os.getenv("EMAIL_RECIPIENTS", "").split(",") if rec],
}

# --- Tushare API 频率限制配置 (基于官方限制的90%安全速率) ---
# key: data.py 中定义的函数名 (不含 get_)
# value: 每分钟允许的调用次数
API_RATE_LIMITS = {
    # --- 特殊限制接口 (100次/分 * 0.9) ---
    "holder_number": 300,
    # --- 标准限制接口 (500次/分 * 0.9) ---
    "default": 180,  # 为所有未明确列出的接口设置默认安全速率
    "daily": 450,
    "index_daily": 450,
    "adj_factor": 450,
    "daily_basic": 450,
    "index_dailybasic": 450,
    "moneyflow": 450,
    "hk_hold": 450,
    "moneyflow_hsgt": 450,
    "top_list": 450,
    "top_inst": 450,
    "margin_detail": 450,
    "block_trade": 450,
    "top10_floatholders": 450,
    "holder_trade": 450,
    "fina_indicator": 450,
    "income": 450,
    "balancesheet": 450,
    "cashflow": 450,
    "forecast": 450,
    "express": 450,
    "dividend": 450,
    "repurchase": 450,
    "cn_m": 450,
    "cn_pmi": 450,
    "cn_cpi": 450,
    "cn_gdp": 450,
    "shibor": 450,
}
