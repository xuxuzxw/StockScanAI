# quant_project/app_v3.py
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sqlalchemy import text

# 导入所有项目模块
import config
import data
import intelligence
import quant_engine

# --- V2.1 UI/UX 优化：定义全局英中表头翻译字典 (白金版) ---
COLUMN_MAPPING = {
    # 通用
    'ts_code': '股票代码', 'name': '股票名称', 'industry': '所属行业', 'ann_date': '公告日期',
    'end_date': '报告期', 'trade_date': '交易日期', 'close': '收盘价', 'price': '成交价格',
    'update_flag': '更新标识', 'f_ann_date': '首次公告日',

    # 排名与因子
    '综合得分': '综合得分', '行业综合得分': '行业综合得分',

    # 筹码类
    'holder_name': '股东名称', 'holder_type': '股东类型', 'hold_amount': '持有数量(股)',
    'hold_ratio': '持股比例(%)', 'hold_float_ratio': '占流通股比例(%)', 'hold_change': '变动数量(股)',
    'in_de': '增减', 'change_ratio': '变动比例(%)', 'avg_price': '均价', 'proc': '进度',
    'vol': '成交量(手)', 'amount': '成交额(千元)', 'high_limit': '回购最高价', 'low_limit': '回购最低价',
    'buyer': '买方', 'seller': '卖方', 'net_amount': '净买入额(万元)', 'reason': '上榜原因',

    # 财务类 - 通用
    'report_type': '报告类型', 'comp_type': '公司类型', 'end_type': '报告期类型',
    'revenue': '营业收入', 'operate_profit': '营业利润', 'total_profit': '利润总额',
    'n_income': '净利润', 'total_assets': '总资产', 'total_liab': '总负债',
    'total_hldr_eqy_exc_min_int': '归母股东权益', 'total_hldr_eqy_inc_min_int': '股东权益合计',
    'basic_eps': '基本每股收益', 'diluted_eps': '稀释每股收益', 'diluted_roe': '稀释ROE(%)',
    'bps': '每股净资产', 'yoy_op': '营业利润同比增长(%)', 'yoy_gr': '营业总收入同比增长(%)',
    'yoy_net_profit': '净利润同比增长(%)',

    # 财务类 - 业绩预告/快报
    'type': '预告类型', 'p_change_min': '业绩变动(最小%)', 'p_change_max': '业绩变动(最大%)',
    'net_profit_min': '净利润(最小)', 'net_profit_max': '净利润(最大)',
    'last_parent_net': '上年同期归母净利润', 'first_ann_date': '首次公告日',
    'summary': '业绩简述', 'change_reason': '变动原因', 'perf_summary': '业绩摘要',

    # 财务类 - 分红送股
    'div_proc': '分红方案', 'stk_div': '送股(股)', 'cash_div_tax': '现金分红(元)',

    # 财务类 - 利润表
    'total_revenue': '营业总收入', 'int_income': '利息收入', 'prem_earned': '已赚保费', 'comm_income': '手续费及佣金收入',
    'n_commis_income': '手续费及佣金净收入', 'n_oth_income': '其他经营净收益', 'n_oth_b_income': '其他业务净收益',
    'prem_income': '保险业务收入', 'out_prem': '分出保费', 'une_prem_reser': '提取未到期责任准备金',
    'reins_income': '分保费收入', 'n_sec_tb_income': '代理买卖证券业务净收入', 'n_sec_uw_income': '证券承销业务净收入',
    'n_asset_mg_income': '受托客户资产管理业务净收入', 'oth_b_income': '其他业务收入', 'fv_value_chg_gain': '公允价值变动收益',
    'invest_income': '投资收益', 'ass_invest_income': '对联营和合营企业的投资收益', 'forex_gain': '汇兑收益',
    'total_cogs': '营业总成本', 'oper_cost': '营业成本', 'int_exp': '利息支出', 'comm_exp': '手续费及佣金支出',
    'biz_tax_surchg': '营业税金及附加', 'sell_exp': '销售费用', 'admin_exp': '管理费用', 'fin_exp': '财务费用',
    'assets_impair_loss': '资产减值损失', 'prem_refund': '退保金', 'compens_payout': '赔付支出净额', 'compens_payout_refu': '摊回赔付支出',
    'reser_insur_liab': '提取保险责任准备金净额', 'insur_reser_refu': '摊回保险责任准备金', 'reins_exp': '分保费用', 'reins_cost_refund': '摊回分保费用',
    'oper_exp': '营业支出', 'other_bus_cost': '其他业务成本', 'non_oper_income': '营业外收入', 'non_oper_exp': '营业外支出',
    'nca_disploss': '非流动资产处置损失', 'income_tax': '所得税费用', 'n_income_attr_p': '归属于母公司所有者的净利润',
    'minority_gain': '少数股东损益', 'oth_compr_income': '其他综合收益', 't_compr_income': '综合收益总额',
    'compr_inc_attr_p': '归母综合收益总额', 'compr_inc_attr_m_s': '归属少数股东综合收益总额',
    'ebit': '息税前利润', 'ebitda': '息税折旧摊销前利润', 'insurance_exp': '保险合同准备金', 'undist_profit': '未分配利润',
    'distable_profit': '可分配利润', 'rd_exp': '研发费用', 'fin_exp_int_exp': '财务费用:利息费用', 'fin_exp_int_inc': '财务费用:利息收入',
    'div_payt': '应付股利', 'transfer_surplus_rese': '盈余公积转入', 'transfer_housing_imprest': '住房周转金转入',
    'transfer_oth': '其他转入', 'adj_lossgain': '调整损益', 'withdra_legal_surplus': '提取法定盈余公积',
    'withdra_legal_pubfund': '提取法定公益金', 'withdra_biz_devfund': '提取企业发展基金', 'withdra_rese_fund': '提取储备基金',
    'withdra_oth_ersu': '提取其他盈余公积', 'workers_welfare': '职工奖金福利', 'distr_profit_shrhder': '分配给股东的利润',
    'prfshare_payable_dvd': '应付优先股股利', 'comshare_payable_dvd': '应付普通股股利', 'capit_comstock_div': '转作股本的普通股股利',
    'continued_net_profit': '持续经营净利润',

    # 财务类 - 资产负债表
    'total_share': '总股本', 'cap_rese': '资本公积金', 'surplus_rese': '盈余公积金', 'special_rese': '专项储备',
    'money_cap': '货币资金', 'trad_asset': '交易性金融资产', 'notes_receiv': '应收票据', 'accounts_receiv': '应收账款',
    'oth_receiv': '其他应收款', 'prepayment': '预付款项', 'div_receiv': '应收股利', 'int_receiv': '应收利息',
    'inventories': '存货', 'amor_exp': '长期待摊费用', 'nca_within_1y': '一年内到期的非流动资产', 'sett_rsrv': '结算备付金',
    'loanto_oth_bank_fi': '拆出资金', 'premium_receiv': '应收保费', 'reinsur_receiv': '应收分保账款',
    'reinsur_res_receiv': '应收分保合同准备金', 'pur_resale_fa': '买入返售金融资产', 'oth_cur_assets': '其他流动资产',
    'total_cur_assets': '流动资产合计', 'fa_avail_for_sale': '可供出售金融资产', 'htm_invest': '持有至到期投资',
    'lt_eqt_invest': '长期股权投资', 'invest_real_estate': '投资性房地产', 'time_deposits': '定期存款',
    'oth_assets': '其他资产', 'lt_rec': '长期应收款', 'fix_assets': '固定资产', 'cip': '在建工程',
    'const_materials': '工程物资', 'fixed_assets_disp': '固定资产清理', 'produc_bio_assets': '生产性生物资产',
    'oil_and_gas_assets': '油气资产', 'intan_assets': '无形资产', 'r_and_d': '研发支出', 'goodwill': '商誉',
    'lt_amor_exp': '长期待摊费用', 'defer_tax_assets': '递延所得税资产', 'decr_in_disbur': '发放贷款及垫款',
    'oth_nca': '其他非流动资产', 'total_nca': '非流动资产合计', 'cash_reser_cb': '存放中央银行款项',
    'depos_in_oth_bfi': '存放同业款项', 'prec_metals': '贵金属', 'deriv_assets': '衍生金融资产',
    'rr_reins_une_prem': '应收分保未到期责任准备金', 'rr_reins_outstd_cla': '应收分保未决赔款准备金',
    'rr_reins_lins_liab': '应收分保寿险责任准备金', 'rr_reins_lthins_liab': '应收分保长期健康险责任准备金',
    'refund_depos': '存出保证金', 'ph_pledge_loans': '保户质押贷款', 'refund_cap_depos': '存出资本保证金',
    'indep_acct_assets': '独立账户资产', 'client_depos': '其中:客户资金存款', 'client_prov': '其中:客户备付金',
    'transac_seat_fee': '交易席位费', 'invest_as_receiv': '应收款项类投资', 'st_borr': '短期借款',
    'lt_borr': '长期借款', 'cb_borr': '向中央银行借款', 'depos_ib_deposits': '同业及其他金融机构存放款项',
    'loan_oth_bank': '拆入资金', 'trading_fl': '交易性金融负债', 'notes_payable': '应付票据',
    'acct_payable': '应付账款', 'adv_receipts': '预收款项', 'sold_for_repur_fa': '卖出回购金融资产款',
    'comm_payable': '应付手续费及佣金', 'payroll_payable': '应付职工薪酬', 'taxes_payable': '应交税费',
    'int_payable': '应付利息', 'div_payable': '应付股利', 'oth_payable': '其他应付款', 'acc_exp': '预提费用',
    'deferred_inc': '递延收益', 'st_bonds_payable': '应付短期债券', 'payable_to_reinsurer': '应付分保账款',
    'rsrv_insur_cont': '保险合同准备金', 'acting_trading_sec': '代理买卖证券款', 'acting_uw_sec': '代理承销证券款',
    'non_cur_liab_due_1y': '一年内到期的非流动负债', 'oth_cur_liab': '其他流动负债', 'total_cur_liab': '流动负债合计',
    'bond_payable': '应付债券', 'lt_payable': '长期应付款', 'specific_payables': '专项应付款',
    'estimated_liab': '预计负债', 'defer_tax_liab': '递延所得税负债', 'defer_inc_non_cur_liab': '非流动负债:递延收益',
    'oth_ncl': '其他非流动负债', 'total_ncl': '非流动负债合计', 'depos_oth_bfi': '同业及其他金融机构存放款项',
    'deriv_liab': '衍生金融负债', 'depos': '吸收存款', 'agency_bus_liab': '代理业务负债', 'oth_liab': '其他负债',
    'prem_receiv_adva': '预收保费', 'depos_received': '存入保证金', 'ph_invest': '保户储金及投资款',
    'reser_une_prem': '未到期责任准备金', 'reser_outstd_claims': '未决赔款准备金', 'reser_lins_liab': '寿险责任准备金',
    'reser_lthins_liab': '长期健康险责任准备金', 'indept_acc_liab': '独立账户负债', 'pledge_borr': '质押借款',
    'indem_payable': '应付赔付款', 'policy_div_payable': '应付保单红利', 'treasury_share': '库存股',
    'ordin_risk_reser': '一般风险准备', 'forex_differ': '外币报表折算差额', 'invest_loss_unconf': '未确认的投资损失',
    'minority_int': '少数股东权益', 'total_hldr_eqy_inc_min_int': '股东权益合计',
    'total_liab_hldr_eqy': '负债和股东权益总计', 'lt_payroll_payable': '长期应付职工薪酬', 'oth_comp_income': '其他综合收益',
    'oth_eqt_tools': '其他权益工具', 'oth_eqt_tools_p_shr': '其他权益工具:优先股', 'lending_funds': '融出资金',
    'acc_receivable': '应收款项', 'st_fin_payable': '应付短期融资款', 'payables': '应付款项',
    'hfs_assets': '持有待售的资产', 'hfs_sales': '持有待售的负债', 'cost_fin_assets': '融出资金',
    'fair_value_fin_assets': '以公允价值计量的金融资产', 'contract_assets': '合同资产', 'contract_liab': '合同负债',
    'accounts_receiv_bill': '应收票据及应收账款', 'accounts_pay': '应付票据及应付账款', 'oth_rcv_total': '其他应收款合计',
    'fix_assets_total': '固定资产合计', 'cip_total': '在建工程合计', 'oth_pay_total': '其他应付款合计',
    'long_pay_total': '长期应付款合计', 'debt_invest': '债权投资', 'oth_debt_invest': '其他债权投资',
}

# --- 页面基础设置 ---
st.set_page_config(page_title="AI量化投研平台 V3 - 全功能版", page_icon="🏆", layout="wide")

# --- 初始化核心组件 (增强版，使用新的引擎文件) ---
@st.cache_resource
def initialize_components():
    """初始化所有核心服务和管理器"""
    try:
        data_manager = data.DataManager()
        # 【修正】初始化AIOrchestrator时传入data_manager
        ai_orchestrator = intelligence.AIOrchestrator(config.AI_MODEL_CONFIG, data_manager)
        # 从新的quant_engine中初始化所有类
        factor_factory = quant_engine.FactorFactory(_data_manager=data_manager)
        factor_processor = quant_engine.FactorProcessor(_data_manager=data_manager)
        factor_analyzer = quant_engine.FactorAnalyzer(_data_manager=data_manager)
        task_runner = quant_engine.AutomatedTasks(data_manager, factor_factory)
        market_profiler = quant_engine.MarketProfile(data_manager=data_manager)
        return data_manager, factor_factory, ai_orchestrator, factor_processor, task_runner, market_profiler, factor_analyzer
    except Exception as e:
        st.error(f"初始化失败: {e}")
        return None, None, None, None, None, None, None

data_manager, factor_factory, ai_orchestrator, factor_processor, task_runner, market_profiler, factor_analyzer = initialize_components()
if not data_manager:
    st.stop()

# AdaptiveAlphaStrategy 需要一个包含历史价格的DataFrame，我们在后台准备好
@st.cache_resource
def preload_prices_for_adaptive_strategy(stock_codes, start, end):
    prices_dict = data_manager.run_batch_download(stock_codes, start, end)
    all_prices_df = pd.DataFrame({
        stock: df.set_index('trade_date')['close']
        for stock, df in prices_dict.items() if df is not None and not df.empty
    }).sort_index()
    all_prices_df.index = pd.to_datetime(all_prices_df.index)
    all_prices_df.dropna(axis=1, how='all', inplace=True)
    return all_prices_df

# --- 会话状态初始化 ---
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

# --- 侧边栏 ---
st.sidebar.title("投研平台控制台 V3")
st.sidebar.markdown("---")

@st.cache_data(ttl=3600)
def get_stock_list():
    # 增加过滤，去除ST和未上市的
    stocks = data_manager.get_stock_basic()

    # 增加健壮性检查：确保关键列存在
    if stocks is None or stocks.empty:
        st.error("无法获取股票基础数据。")
        return pd.DataFrame()

    # 【修正】移除对 list_status 的重复过滤。
    # 因为在 data.py 的 get_stock_basic 中已通过参数 list_status='L' 在API层面完成过滤。
    # Tushare API 因此不再返回该列，此处的检查和警告可以安全移除。

    if 'name' in stocks.columns:
        stocks = stocks[~stocks['name'].str.contains('ST')]
    else:
        st.warning("警告：'name' 列不存在，无法过滤ST股票。")

    return stocks

stock_list = get_stock_list()
if stock_list is not None and not stock_list.empty:
    stock_options = stock_list['ts_code'] + " " + stock_list['name']

    # 【重构】使用 st.session_state 来控制 selectbox
    # 如果 session_state 中有来自“智能选股排名”的点击，就用它
    if 'selected_stock' in st.session_state and st.session_state.selected_stock and st.session_state.selected_stock in stock_options.tolist():
        default_index = stock_options.tolist().index(st.session_state.selected_stock)
    else: # 否则使用默认值
        default_index = stock_options[stock_options.str.contains("贵州茅台")].index[0] if any(stock_options.str.contains("贵州茅台")) else 0

    selected_stock_str = st.sidebar.selectbox("选择股票:", options=stock_options, index=int(default_index), key="stock_selector")
    ts_code = selected_stock_str.split(" ")[0]
else:
    ts_code = st.sidebar.text_input("输入股票代码 (如 600519.SH):", "600519.SH")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)
start_date_input = st.sidebar.date_input("开始日期", start_date)
end_date_input = st.sidebar.date_input("结束日期", end_date)
start_date_str = start_date_input.strftime('%Y%m%d')
end_date_str = end_date_input.strftime('%Y%m%d')
st.sidebar.markdown("---")

# --- 主页面 ---
st.title(f"🏆 {ts_code} - 全功能深度投研")
if stock_list is not None and not stock_list[stock_list['ts_code'] == ts_code].empty:
    info = stock_list[stock_list['ts_code'] == ts_code].iloc[0]
    st.markdown(f"**名称:** {info['name']} | **行业:** {info['industry']} | **上市:** {info['list_date']}")


# --- V2.0 新增分析模块 ---
try:
    import index_analyzer
    import industry_analyzer
    index_analyzer_client = index_analyzer.IndexAnalyzer(data_manager)
    # industry_analyzer_client = industry_analyzer.IndustryAnalyzer(data_manager, factor_factory) # 待实现
    V2_MODULES_LOADED = True
except ImportError:
    V2_MODULES_LOADED = False


# --- 创建多标签页 (V2.2 UX 优化) ---
# V2.2 重构：明确定义所有标签页的最终理想顺序
all_tabs_ordered = [
    "📈 策略看板",   # V2.3 新增
    "🎯 市场全景",
    "🏭 行业透视",
    "🏆 智能选股排名",
    "📈 行情总览",
    "💰 资金与筹码",
    "🧾 深度财务",
    "🌐 宏观环境",
    "🤖 AI综合报告",
    "🔬 因子分析器",
    "🚀 回测实验室",
    "🔬 模型训练室", # V2.3 新增
    "⚙️ 系统任务"
]

# 定义哪些标签页依赖 V2 模块
V2_TABS = ["🎯 市场全景", "🏭 行业透视"]

# 根据模块加载情况，动态生成最终的标签页列表
if V2_MODULES_LOADED:
    tab_list = all_tabs_ordered
else:
    # 如果 V2 模块加载失败，则从理想顺序中移除对应的标签页
    tab_list = [tab for tab in all_tabs_ordered if tab not in V2_TABS]

tabs = st.tabs(tab_list)

# 根据最终生成的 tab_list 动态解包，更加健壮
# 使用 dict comprehension 和 globals() 来动态创建变量，避免复杂的 if/else
tab_mapping = {tab.replace(" ", "_").replace("🏆_", "").replace("📈_", "").replace("💰_", "").replace("🧾_", "").replace("🌐_", "").replace("🎯_", "").replace("🏭_", "").replace("🤖_", "").replace("🔬_", "").replace("🚀_", "").replace("⚙️_", ""): tab_obj for tab, tab_obj in zip(tab_list, tabs)}
globals().update(tab_mapping)

# 为 V2 模块创建占位符，以防加载失败
if not V2_MODULES_LOADED:
    tab_market, tab_industry = None, None
else:
    tab_market = tab_mapping.get('市场全景')
    tab_industry = tab_mapping.get('行业透视')

# 为了代码可读性，为几个核心tab创建别名
tab_strategy_board = tab_mapping.get('策略看板') # V2.3 新增
tab_ranker = tab_mapping.get('智能选股排名')
tab_main = tab_mapping.get('行情总览')
tab_funds = tab_mapping.get('资金与筹码')
tab_finance = tab_mapping.get('深度财务')
tab_macro = tab_mapping.get('宏观环境')
tab_ai = tab_mapping.get('AI综合报告')
tab_analyzer = tab_mapping.get('因子分析器')
tab_backtest = tab_mapping.get('回测实验室')
tab_trainer = tab_mapping.get('模型训练室') # V2.3 新增
tab_tasks = tab_mapping.get('系统任务')


# --- 0. 策略看板 (V2.3 新增) ---
if tab_strategy_board:
    with tab_strategy_board:
        st.subheader("每日AI投研晨报与策略持仓")
        st.markdown("此看板展示由后台自动化工作流在每日开盘前（默认 08:00）生成的最新策略分析结果。")

        # 获取最新交易日
        try:
            cal_df = data_manager.pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
            report_date = cal_df[cal_df['is_open'] == 1]['cal_date'].max()
        except Exception:
            report_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        st.info(f"正在加载 **{report_date}** 的晨报...")

        # 从数据库加载晨报
        try:
            with data_manager.engine.connect() as conn:
                query = text("SELECT report_content FROM ai_reports WHERE trade_date = :date AND ts_code = 'STRATEGY_MORNING_REPORT'")
                report_content = conn.execute(query, {'date': report_date}).scalar_one_or_none()

            if report_content:
                st.markdown(report_content)
            else:
                st.warning(f"未能在数据库中找到 {report_date} 的晨报。请确认后台 `run_strategy_daily.py` 任务是否已成功执行。")
        except Exception as e:
            st.error(f"加载晨报时发生数据库错误: {e}")


# --- 1. 智能选股排名 ---
# V2.3 健壮性优化：全面使用 tab_objects.get()
if tab_ranker:
    with tab_ranker:
        st.subheader("智能选股与行业轮动分析")
        st.markdown("构建您的专属多因子模型，系统将从**行业**和**个股**两个层面进行综合打分排名，助您实现“先选赛道、再选赛马”的专业投研。")
    # --- 1. 获取最新交易日 ---
    try:
        cal_df = data_manager.pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
        latest_trade_date = cal_df[cal_df['is_open'] == 1]['cal_date'].max()
        st.info(f"数据基于最新已计算交易日: **{latest_trade_date}**")
    except Exception as e:
        st.error(f"无法获取最新交易日: {e}")
        latest_trade_date = None

    if latest_trade_date:
        # --- 2. 用户选择因子与权重 ---
        st.markdown("#### (1) 配置您的多因子模型")
        # 【V2.2 重构】从统一的数据管道脚本中导入因子列表，确保源唯一
        from run_daily_pipeline import FACTORS_TO_CALCULATE as available_factors

        cols = st.columns(4)
        factor_direction = {
            'pe_ttm': -1, 'roe': 1, 'growth_revenue_yoy': 1, 'debt_to_assets': -1,
            'momentum': 1, 'volatility': -1, 'net_inflow_ratio': 1,
            'holder_num_change_ratio': -1, # 股东人数变化率越小越好
            'major_shareholder_net_buy_ratio': 1, # 重要股东净增持比率越大越好
            'top_list_net_buy_amount': 1, # 龙虎榜净买入额越大越好
            'dividend_yield': 1, # 股息率越高越好
            'forecast_growth_rate': 1, # 预告增长率越高越好
            'repurchase_ratio': 1, # 回购比例越高越好
            'block_trade_ratio': 1 # 大宗交易占比越高，说明该股可能在机构间关注度高
        }

        # --- V2.1 重构：明确定义因子分类列表 ---
        VALUE_FACTORS = ['pe_ttm', 'dividend_yield', 'repurchase_ratio']
        QUALITY_GROWTH_FACTORS = ['roe', 'growth_revenue_yoy', 'debt_to_assets', 'forecast_growth_rate']
        TECH_FINANCE_FACTORS = ['momentum', 'volatility', 'net_inflow_ratio', 'block_trade_ratio']
        CHIP_FACTORS = ['holder_num_change_ratio', 'major_shareholder_net_buy_ratio', 'top_list_net_buy_amount']

        with cols[0]:
            st.multiselect("价值/回报因子", [f for f in available_factors if f in VALUE_FACTORS], default=['pe_ttm', 'dividend_yield', 'repurchase_ratio'], key="value_factors")
        with cols[1]:
            st.multiselect("质量/成长因子", [f for f in available_factors if f in QUALITY_GROWTH_FACTORS], default=['roe', 'growth_revenue_yoy', 'forecast_growth_rate'], key="quality_factors")
        with cols[2]:
            st.multiselect("技术/资金因子", [f for f in available_factors if f in TECH_FINANCE_FACTORS], default=['momentum', 'net_inflow_ratio'], key="tech_factors")
        with cols[3]:
            st.multiselect("筹码因子", [f for f in available_factors if f in CHIP_FACTORS], default=['holder_num_change_ratio', 'major_shareholder_net_buy_ratio'], key="chip_factors")

        user_selection = st.session_state.value_factors + st.session_state.quality_factors + st.session_state.tech_factors + st.session_state.chip_factors
        # --- 3. 执行排名 ---
        if st.button("🚀 开始智能排名", use_container_width=True):
            if not user_selection:
                st.warning("请至少选择一个因子。")
            else:
                with st.spinner("正在从因子库提取数据并计算行业与个股综合得分..."):
                    try:
                        # --- A. 从数据库查询所有选中因子的数据 ---
                        query = text(f"""
                            SELECT ts_code, factor_name, factor_value
                            FROM factors_exposure
                            WHERE trade_date = '{latest_trade_date}'
                            AND factor_name IN ({','.join([f"'{f}'" for f in user_selection])})
                        """)
                        with data_manager.engine.connect() as conn:
                            all_factor_data = pd.read_sql(query, conn)

                        # --- B. 数据处理：将长表转换为宽表 ---
                        factor_table = all_factor_data.pivot(index='ts_code', columns='factor_name', values='factor_value').dropna()

                        # --- C. 合并行业信息 ---
                        full_stock_list = get_stock_list()
                        factor_table_with_industry = factor_table.merge(full_stock_list[['ts_code', 'name', 'industry']], on='ts_code')

                        # --- D. 【新增】计算行业综合得分 ---
                        st.markdown("---")
                        st.markdown("#### (2) 行业综合得分排名")
                        industry_factors = factor_table_with_industry.groupby('industry')[user_selection].mean()
                        processed_industry_factors = industry_factors.apply(lambda x: (x - x.mean()) / x.std())
                        for factor, direction in factor_direction.items():
                            if factor in processed_industry_factors.columns:
                                processed_industry_factors[factor] *= direction
                        processed_industry_factors['行业综合得分'] = processed_industry_factors.mean(axis=1)
                        industry_rank = processed_industry_factors.sort_values('行业综合得分', ascending=False)

                        st.dataframe(industry_rank.style.format('{:.2f}'))
                        st.bar_chart(industry_rank['行业综合得分'].head(15))

                        # --- E. 计算个股综合得分 ---
                        st.markdown("---")
                        st.markdown("#### (3) 个股综合得分排名")
                        processed_stock_factors = factor_table.apply(lambda x: (x - x.mean()) / x.std())
                        for factor, direction in factor_direction.items():
                            if factor in processed_stock_factors.columns:
                                processed_stock_factors[factor] *= direction
                        processed_stock_factors['综合得分'] = processed_stock_factors.mean(axis=1)

                        final_rank = processed_stock_factors.merge(full_stock_list[['ts_code', 'name', 'industry']], on='ts_code')
                        final_rank = final_rank.sort_values('综合得分', ascending=False).reset_index(drop=True)

                        # --- F. 个股结果展示与交互 ---
                        final_rank_display = final_rank[['ts_code', 'name', 'industry', '综合得分']].head(100)
                        
                        st.caption("💡 小提示：直接点击下方表格中的任意一行，系统将自动跳转到该股票的深度分析页面。")

                        # 【交互修复】使用 st.data_editor 替代 st.dataframe，以捕获行选择事件
                        # 将选择状态存储在 session_state 中
                        if "selected_rank_row" not in st.session_state:
                            st.session_state.selected_rank_row = None
                        
                        edited_df = st.data_editor(
                            final_rank_display, 
                            hide_index=True, 
                            disabled=final_rank_display.columns, # 设置所有列为不可编辑
                            on_select="rerun", # 当选择变化时，重新运行脚本
                            key="rank_selector"
                        )

                        # 检查是否有行被选中
                        if st.session_state.rank_selector and st.session_state.rank_selector.get("selection", {}).get("rows"):
                            selected_index = st.session_state.rank_selector["selection"]["rows"][0]
                            selected_row = final_rank_display.iloc[selected_index]
                            
                            selected_ts_code = selected_row['ts_code']
                            selected_name = selected_row['name']
                            
                            # 更新全局 session_state 并触发重跑，让侧边栏的 selectbox 更新
                            st.session_state.selected_stock = f"{selected_ts_code} {selected_name}"
                            st.rerun()

                    except Exception as e:
                        st.error(f"排名计算过程中发生错误: {e}")
                        st.exception(e)

# --- 2. 行情总览 ---
if tab_main:
    with tab_main:
        st.subheader("日K线图 (后复权) & 综合指标")
        df_adj = data_manager.get_adjusted_daily(ts_code, start_date_str, end_date_str, adj='hfq')
        if df_adj is not None and not df_adj.empty:
            # --- 1. 数据获取与合并 ---
            # 获取每日基本面指标（PE、换手率等）
            df_basic = data_manager.get_daily_basic(ts_code, start_date_str, end_date_str)
            if df_basic is not None and not df_basic.empty:
                # 【修正】在合并前，确保两个DataFrame的'trade_date'列都是datetime类型
                df_adj['trade_date'] = pd.to_datetime(df_adj['trade_date'])
                df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])
                # 将基础指标合并到主数据框中
                df_adj = pd.merge(df_adj, df_basic[['trade_date', 'pe_ttm', 'turnover_rate']], on='trade_date', how='left')

            # --- 2. 计算技术指标 ---
            df_adj['EMA20'] = df_adj['close'].ewm(span=20, adjust=False).mean()
            df_adj['EMA60'] = df_adj['close'].ewm(span=60, adjust=False).mean()
            df_adj['EMA120'] = df_adj['close'].ewm(span=120, adjust=False).mean()
            df_adj['Vol_EMA20'] = df_adj['vol'].ewm(span=20, adjust=False).mean()

            # --- 3. 绘图 (增强版，4个子图) ---
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.55, 0.15, 0.15, 0.15] # 调整各子图高度占比
            )

            # 图1: K线与均线
            fig.add_trace(go.Candlestick(x=df_adj['trade_date'], open=df_adj['open'], high=df_adj['high'], low=df_adj['low'], close=df_adj['close'], name='K线'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_adj['trade_date'], y=df_adj['EMA20'], mode='lines', name='EMA20', line=dict(color='yellow', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_adj['trade_date'], y=df_adj['EMA60'], mode='lines', name='EMA60', line=dict(color='cyan', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_adj['trade_date'], y=df_adj['EMA120'], mode='lines', name='EMA120', line=dict(color='magenta', width=1)), row=1, col=1)

            # 图2: 成交量
            fig.add_trace(go.Bar(x=df_adj['trade_date'], y=df_adj['vol'], name='成交量', marker_color='lightblue'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_adj['trade_date'], y=df_adj['Vol_EMA20'], mode='lines', name='成交量EMA20', line=dict(color='orange', width=1)), row=2, col=1)

            # 图3: 市盈率 (PE-TTM)
            if 'pe_ttm' in df_adj.columns:
                fig.add_trace(go.Scatter(x=df_adj['trade_date'], y=df_adj['pe_ttm'], mode='lines', name='市盈率PE(TTM)', line=dict(color='lightgreen', width=1.5)), row=3, col=1)
                fig.update_yaxes(title_text="PE(TTM)", row=3, col=1)

            # 图4: 换手率
            if 'turnover_rate' in df_adj.columns:
                fig.add_trace(go.Bar(x=df_adj['trade_date'], y=df_adj['turnover_rate'], name='换手率(%)', marker_color='violet'), row=4, col=1)
                fig.update_yaxes(title_text="换手率(%)", row=4, col=1)

            fig.update_layout(
                title_text=f"{ts_code} - 技术、估值与情绪综合视图",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=800  # 增加图表总高度以容纳更多子图
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("无法获取复权行情数据。")

# --- 2. 资金与筹码 ---
if tab_funds:
    with tab_funds:
        st.subheader("资金流向 & 股东结构 (V2.1 增强)")

    # --- Part 1: 原有资金流分析 ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**主力资金流 (近30日)**")
        df_flow = data_manager.get_moneyflow(ts_code, (end_date - timedelta(days=30)).strftime('%Y%m%d'), end_date_str)
        if df_flow is not None and not df_flow.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_flow['trade_date'], y=df_flow['net_mf_amount'], name='净流入额'))
            fig.update_layout(title="主力资金净流入(万元)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**北向资金持股比例**")
        df_hk = data_manager.get_hk_hold(ts_code, start_date_str, end_date_str)
        if df_hk is not None and not df_hk.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hk['trade_date'], y=df_hk['ratio'], mode='lines', name='持股比例(%)'))
            fig.update_layout(title="北向资金持股比例(%)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # --- Part 2: V2.1 新增筹码分析 ---
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**股东人数变化趋势**")
        df_holder_num = data_manager.get_holder_number(ts_code)
        if df_holder_num is not None and not df_holder_num.empty and len(df_holder_num) > 1:
            df_holder_num['end_date'] = pd.to_datetime(df_holder_num['end_date'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_holder_num['end_date'], y=df_holder_num['holder_num'], mode='lines+markers', name='股东人数'))
            fig.update_layout(title="股东人数", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暂无足够的股东人数数据。")

    with col4:
        st.markdown("**龙虎榜净买入 (近90日)**")
        df_top_list_hist = data_manager.get_top_list(start_date=(end_date - timedelta(days=90)).strftime('%Y%m%d'), end_date=end_date_str)
        if df_top_list_hist is not None and not df_top_list_hist.empty:
            df_top_list_hist['trade_date'] = pd.to_datetime(df_top_list_hist['trade_date'])
            stock_top_list = df_top_list_hist[df_top_list_hist['ts_code'] == ts_code]
            if not stock_top_list.empty:
                 fig = go.Figure()
                 fig.add_trace(go.Bar(x=stock_top_list['trade_date'], y=stock_top_list['net_amount'], name='龙虎榜净买入额'))
                 fig.update_layout(title="龙虎榜净买入(万元)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("该股近90日未登上龙虎榜。")
        else:
            st.warning("暂无龙虎榜数据。")

    st.markdown("**重要股东增减持 (近一年)**")
    df_holder_trade = data_manager.get_holder_trade(ts_code, start_date= (end_date - timedelta(days=365)).strftime('%Y%m%d'), end_date=end_date_str)
    if df_holder_trade is not None and not df_holder_trade.empty:
        df_display = df_holder_trade.sort_values(by='ann_date', ascending=True)
        df_display = df_display[['ann_date', 'holder_name', 'in_de', 'change_ratio', 'avg_price']].rename(columns=COLUMN_MAPPING)
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("该股近一年无重要股东增减持记录。")

    st.markdown("**大宗交易明细 (近90日)**")
    df_block_trade = data_manager.get_block_trade(start_date=(end_date - timedelta(days=90)).strftime('%Y%m%d'), end_date=end_date_str)
    if df_block_trade is not None and not df_block_trade.empty:
        stock_block_trade = df_block_trade[df_block_trade['ts_code'] == ts_code]
        if not stock_block_trade.empty:
            df_display = stock_block_trade.sort_values(by='trade_date', ascending=True)
            # 修正：'price'列已加入翻译字典
            df_display = df_display[['trade_date', 'price', 'vol', 'amount', 'buyer', 'seller']].rename(columns=COLUMN_MAPPING)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("该股近90日无大宗交易记录。")
    else:
        st.info("近90日无大宗交易记录。")


    st.markdown("**前十大流通股东 (最新报告期)**")
    # ... (原有获取十大股东的代码保持不变) ...
    latest_period = ""
    for year_offset in range(2):
        year = end_date.year - year_offset
        periods = [f"{year}1231", f"{year}0930", f"{year}0630", f"{year}0331"]
        for p in periods:
            if p <= end_date.strftime('%Y%m%d'):
                df_holders = data_manager.get_top10_floatholders(ts_code, p)
                if df_holders is not None and not df_holders.empty:
                    latest_period = p
                    break
        if latest_period:
            break

    if latest_period:
        st.info(f"当前显示财报周期: {latest_period}")
        # 修正：确保所有列都被翻译
        df_display = df_holders.rename(columns=COLUMN_MAPPING)
        st.dataframe(df_display, use_container_width=True, height=385, hide_index=True)
    else:
        st.warning("未能获取前十大流通股东数据。")

# --- 3. 深度财务 ---
if tab_finance:
    with tab_finance:
        st.subheader("财务报表与前瞻指标 (V2.1 增强)")

    # --- Part 1: V2.1 新增财务前瞻 ---
    st.markdown(f"**业绩快报 (最新)**")
    # --- V2.1 重构：创建一个统一的函数来处理转置和翻译 ---
    def display_transposed_df(df: pd.DataFrame):
        if df is None or df.empty:
            return
        # 确保只处理单行数据
        if len(df) > 1:
             df = df.sort_values(by='ann_date', ascending=False).head(1)

        df_display = df.T.reset_index()
        df_display.columns = ['指标', '数值']
        # 核心修正：使用我们强大的新字典来翻译“指标”列
        df_display['指标'] = df_display['指标'].map(COLUMN_MAPPING).fillna(df_display['指标'])
        df_display['数值'] = df_display['数值'].astype(str)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    df_express = data_manager.get_express(ts_code, start_date=(end_date - timedelta(days=365)).strftime('%Y%m%d'), end_date=end_date_str)
    if df_express is not None and not df_express.empty:
        df_express['ann_date'] = pd.to_datetime(df_express['ann_date'])
        latest_pit_express = df_express[df_express['ann_date'] <= end_date].sort_values(by='ann_date', ascending=False).head(1)
        if not latest_pit_express.empty:
            display_transposed_df(latest_pit_express)
        else:
            st.info("近一年无已披露的业绩快报。")
    else:
        st.info("近一年无业绩快报。")

    st.markdown(f"**业绩预告 (最新)**")
    df_forecast = data_manager.get_forecast(ts_code, start_date=(end_date - timedelta(days=365)).strftime('%Y%m%d'), end_date=end_date_str)
    if df_forecast is not None and not df_forecast.empty:
        df_forecast['ann_date'] = pd.to_datetime(df_forecast['ann_date'])
        latest_pit_forecast = df_forecast[df_forecast['ann_date'] <= end_date].sort_values(by='ann_date', ascending=False).head(1)
        if not latest_pit_forecast.empty:
            display_transposed_df(latest_pit_forecast)
        else:
            st.info("近一年无已披露的业绩预告。")
    else:
        st.info("近一年无业绩预告。")

    st.markdown(f"**历史分红**")
    df_dividend = data_manager.get_dividend(ts_code)
    if df_dividend is not None and not df_dividend.empty:
        df_display = df_dividend.sort_values(by='end_date', ascending=True)
        df_display = df_display[['end_date', 'ann_date', 'div_proc', 'stk_div', 'cash_div_tax']].rename(columns=COLUMN_MAPPING)
        st.dataframe(df_display.head(), use_container_width=True, hide_index=True)
    else:
        st.info("无历史分红记录。")

    st.markdown(f"**股票回购记录 (近一年)**")
    df_repurchase = data_manager.get_repurchase(ts_code, start_date=(end_date - timedelta(days=365)).strftime('%Y%m%d'), end_date=end_date_str)
    if df_repurchase is not None and not df_repurchase.empty:
        df_display = df_repurchase.sort_values(by='ann_date', ascending=True)
        df_display = df_display[['ann_date', 'proc', 'vol', 'amount', 'high_limit', 'low_limit']].rename(columns=COLUMN_MAPPING)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("近一年无股票回购记录。")

    st.markdown("---")
    st.markdown("**财务报表核心数据**")
    # --- Part 2: 原有财务报表 ---
    if 'latest_period' in locals() and latest_period:
        st.markdown(f"**利润表 ({latest_period})**")
        df_income = data_manager.get_income(ts_code, latest_period)
        display_transposed_df(df_income)

        st.markdown(f"**资产负债表 ({latest_period})**")
        df_balance = data_manager.get_balancesheet(ts_code, latest_period)
        display_transposed_df(df_balance)
    else:
        st.warning("未能确定最新的财报周期，无法加载财务报表。")

# --- 4. 宏观环境 ---
if tab_macro:
    with tab_macro:
        st.subheader("宏观经济指标")
        start_m = f"{end_date.year-2}{end_date.month:02d}"
        end_m = f"{end_date.year}{end_date.month:02d}"
        df_pmi = data_manager.get_cn_pmi(start_m, end_m)
        if df_pmi is not None and not df_pmi.empty:
            # 修复大小写问题：将所有列名转为小写
            df_pmi.columns = [col.lower() for col in df_pmi.columns]

            fig = go.Figure()
            # 根据最新的Tushare接口文档，制造业PMI字段为'pmi010000'
            pmi_col = 'pmi010000'
            date_col = 'month' # Tushare文档明确月份字段为'month'

            if date_col not in df_pmi.columns or pmi_col not in df_pmi.columns:
                st.error(f"PMI数据中未找到关键列。需要日期列 ('{date_col}') 和PMI列 ('{pmi_col}')。可用列：{df_pmi.columns.tolist()}")
            else:
                fig.add_trace(go.Scatter(x=df_pmi[date_col], y=pd.to_numeric(df_pmi[pmi_col]), name='制造业PMI'))
                fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="荣枯线", annotation_position="bottom right")
                fig.update_layout(title="制造业采购经理人指数 (PMI)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        df_m = data_manager.get_cn_m(start_m, end_m)
        if df_m is not None and not df_m.empty:
            fig = go.Figure()
            date_col = 'month' if 'month' in df_m.columns else 'stat_month' # 兼容'month'或'stat_month'

            if date_col not in df_m.columns:
                st.error("货币供应量数据中未找到日期列 ('month'或'stat_month')。")
            else:
                fig.add_trace(go.Bar(x=df_m[date_col], y=df_m['m1_yoy'] - df_m['m2_yoy'], name='M1-M2剪刀差'))
                fig.add_hline(y=0, line_dash="dash", line_color="white")
                fig.update_layout(title="M1-M2同比增速剪刀差(%)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        df_cpi = data_manager.get_cn_cpi(start_m=start_m, end_m=end_m)
        if df_cpi is not None and not df_cpi.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_cpi['month'], y=df_cpi['nt_yoy'], name='CPI全国同比(%)'))
            fig.update_layout(title="居民消费价格指数 (CPI)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        df_shibor = data_manager.get_shibor(start_date=(end_date - timedelta(days=365)).strftime('%Y%m%d'), end_date=end_date_str)
        if df_shibor is not None and not df_shibor.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_shibor['date'], y=df_shibor['on'], name='隔夜', line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df_shibor['date'], y=df_shibor['1w'], name='1周', line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df_shibor['date'], y=df_shibor['1y'], name='1年', line=dict(width=2)))
            fig.update_layout(title="上海银行间同业拆放利率 (Shibor)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("动态市场状态感知")
        current_regime = market_profiler.get_market_regime(end_date)
        st.metric(label="当前市场环境判断", value=current_regime)
        st.caption("基于PMI和M1-M2剪刀差的宏观模型。策略应根据'牛市'、'熊市'或'震荡市'调整风险偏好。")

# --- V2.0 新增: 市场全景 ---
if V2_MODULES_LOADED and tab_market:
    with tab_market:
        st.subheader("大盘择时分析 (沪深300)")
        index_code = '000300.SH' # 以沪深300为例

        with st.spinner("正在计算大盘估值..."):
            valuation = index_analyzer_client.get_index_valuation_percentile(index_code, end_date_str)
            timing_signal = index_analyzer_client.generate_timing_signal(index_code, end_date_str)

            if 'error' in valuation:
                st.error("无法获取大盘估值数据。")
            else:
                st.metric(label="当前择时信号", value=timing_signal)
                col1, col2 = st.columns(2)
                col1.metric(label="PE(TTM) 百分位", value=f"{valuation['pe_percentile']:.2%}", help=f"当前PE: {valuation['pe_ttm']:.2f}。百分位越低代表估值越便宜。")
                col2.metric(label="PB 百分位", value=f"{valuation['pb_percentile']:.2%}", help=f"当前PB: {valuation['pb']:.2f}。百分位越低代表估值越便宜。")
                st.info("择时信号基于PE和PB在过去5年的历史百分位生成，可用于辅助判断市场整体风险。")

# --- V2.0 新增: 行业透视 ---
if V2_MODULES_LOADED and tab_industry:
    with tab_industry:
        st.subheader("行业因子排名与轮动分析")
        st.markdown("计算全市场所有申万一级行业的平均因子暴露值，并进行排序，用于发现强势或弱势行业。")

        # 1. 初始化行业分析器
        try:
            industry_analyzer_client = industry_analyzer.IndustryAnalyzer(data_manager, factor_factory)
            ANALYZER_READY = True
        except Exception as e:
            st.error(f"初始化行业分析器失败: {e}")
            ANALYZER_READY = False

        if ANALYZER_READY:
            # 2. 设置分析参数
            col1, col2 = st.columns([1, 1])
            with col1:
                # 提供一个常用的因子列表供选择
                factor_to_rank = st.selectbox(
                    "选择排名因子",
                    options=['pe_ttm', 'growth_revenue_yoy', 'momentum', 'net_inflow_ratio', 'roe'],
                    index=0,
                    help="选择一个因子，系统将计算每个行业的平均值并进行排名。"
                )
            with col2:
                ranking_asc = st.radio(
                    "排序方式",
                    ("降序 (高->低)", "升序 (低->高)"),
                    index=0,
                    horizontal=True,
                    help="对于PE等估值因子，通常选择升序；对于成长、动量等因子，通常选择降序。"
                )
                ascending = True if "升序" in ranking_asc else False

            if st.button("开始行业排名分析"):
                with st.spinner(f"正在从因子库查询 '{factor_to_rank}' 的行业排名..."):
                    try:
                        # 【重构】调用新版后台逻辑，无需再传递进度条
                        ranked_df = industry_analyzer_client.get_industry_factor_rank(
                            date=end_date_str,
                            factor_name=factor_to_rank,
                            ascending=ascending
                        )

                        if ranked_df.empty:
                            st.warning(f"在因子库中未找到 {end_date_str} 的 {factor_to_rank} 数据。请确认后台计算任务是否已成功执行。")
                        else:
                            st.success("行业排名查询完成！")

                            # 展示结果
                            st.dataframe(ranked_df.style.format('{:.2f}'))

                            st.markdown(f"#### **{factor_to_rank}** 因子排名前10行业可视化")
                            top_10_df = ranked_df.head(10) if not ascending else ranked_df.tail(10).sort_values(by='factor_value', ascending=False)
                            st.bar_chart(top_10_df)

                    except Exception as e:
                        st.error(f"行业分析过程中发生错误: {e}")
                        st.exception(e)

# --- 5. AI综合报告 ---
if tab_ai:
    with tab_ai:
        st.subheader("混合AI智能体分析")
        st.markdown("点击下方按钮，AI将采集并分析该股的 **技术、资金、财务、筹码、宏观、舆情** 六大维度数据，生成一份深度综合投研报告。")
        # FIX 1: Corrected indentation for the 'with' block under the 'if' statement.
        if st.button("🚀 启动AI深度综合分析", help="调用混合AI引擎，对该股票进行六大维度、递进式分析，生成综合投研报告。"):
            with st.spinner("AI引擎启动...正在执行多维数据采集与深度分析工作流..."):
                try:
                    # 注意：此处的 factor_factory 已经是 quant_engine.FactorFactory 的实例
                    # 旧版的 intelligence.full_analysis_workflow 可以直接兼容使用
                    report, cost = intelligence.full_analysis_workflow(
                        orchestrator=ai_orchestrator, data_manager=data_manager,
                        factor_factory=factor_factory, ts_code=ts_code,
                        date_range=(start_date_str, end_date_str)
                    )
                    st.success("✅ AI分析完成！")
                    st.markdown(report)
                    st.info(f"本次分析调用AI模型 {cost['total_calls']} 次, 预估成本: ${cost['estimated_cost']:.4f}。")
                except Exception as e:
                    st.error(f"AI分析过程中发生错误: {e}")
                    st.exception(e)

# --- 新增: 6. 因子分析器 ---
if tab_analyzer:
    with tab_analyzer:
        st.subheader("因子有效性分析实验室")
        st.markdown("选择一个或多个因子，在指定的股票池和时间段内，进行IC/IR分析和分层回测，以评估其选股有效性。")
        # --- 参数配置 ---
        st.markdown("#### 1. 配置分析参数")
        analyzer_cols = st.columns(3)
        with analyzer_cols[0]:
            # 提供一个常用的因子列表供选择
            factor_to_analyze = st.selectbox(
                "选择要分析的因子",
                options=['momentum', 'volatility', 'net_inflow_ratio', 'roe', 'pe_ttm', 'growth_revenue_yoy'],
                index=0,
                key="factor_select"
            )
        with analyzer_cols[1]:
            analyzer_start_date = st.date_input("分析开始日期", datetime(2023, 1, 1), key="analyzer_start")
        with analyzer_cols[2]:
            analyzer_end_date = st.date_input("分析结束日期", datetime.now() - timedelta(days=1), key="analyzer_end")

        analyzer_stock_pool_options = get_stock_list()['ts_code'] + " " + get_stock_list()['name']
        # 默认选择一个包含不同行业的股票池作为示例
        default_analyzer_pool = [
            s for s in analyzer_stock_pool_options if any(k in s for k in ["平安", "茅台", "宁德", "万科", "中信"])
        ]
        analyzer_stock_pool = st.multiselect(
            "选择股票池 (建议5-20支)",
            options=analyzer_stock_pool_options,
            default=default_analyzer_pool,
            key="analyzer_pool"
        )
        analyzer_stock_codes = [s.split(" ")[0] for s in analyzer_stock_pool]

        if st.button("🔬 开始因子分析", key="start_factor_analysis"):
            if not analyzer_stock_codes:
                st.warning("请至少选择一支股票进行分析。")
            else:
                with st.spinner(f"正在分析因子 '{factor_to_analyze}'..."):
                    try:
                        # 1. 计算因子历史截面数据
                        st.info("步骤1: 计算因子历史截面数据...")
                        dates = pd.date_range(analyzer_start_date, analyzer_end_date, freq='M')
                        factor_df = pd.DataFrame()

                        progress_bar = st.progress(0)
                        for i, date in enumerate(dates):
                            date_str = date.strftime('%Y%m%d')
                            # 向前回溯60天以确保有足够数据计算因子
                            start_date_str_calc = (date - pd.Timedelta(days=60)).strftime('%Y%m%d')

                            raw_values = {}
                            for code in analyzer_stock_codes:
                                calc_func = getattr(factor_factory, f"calc_{factor_to_analyze}")
                                # 因子函数需要不同的参数，这里做一个适配
                                if factor_to_analyze in ['momentum', 'volatility', 'net_inflow_ratio', 'north_hold_change']:
                                    raw_values[code] = calc_func(ts_code=code, start_date=start_date_str_calc, end_date=date_str)
                                else: # 基本面等因子
                                    raw_values[code] = calc_func(ts_code=code, date=date_str)

                            factor_df[date] = pd.Series(raw_values)
                            progress_bar.progress((i + 1) / len(dates))

                        factor_df = factor_df.T.dropna(how='all')
                        factor_df.index.name = 'trade_date'
                        st.success("因子数据计算完成！")
                        st.dataframe(factor_df.head())

                        # 2. 计算IC和IR
                        st.info("步骤2: 计算信息系数 (IC) 和信息比率 (IR)...")
                        ic_values = {}
                        for date, factor_slice in factor_df.iterrows():
                            ic, p_val = factor_analyzer.calculate_ic(factor_slice.dropna())
                            if not np.isnan(ic):
                                ic_values[date] = ic

                        ic_series = pd.Series(ic_values)
                        ir = factor_analyzer.calculate_ir(ic_series)

                        ic_cols = st.columns(2)
                        ic_cols[0].metric("IC均值 (Mean IC)", f"{ic_series.mean():.4f}")
                        ic_cols[1].metric("信息比率 (IR)", f"{ir:.4f}")

                        st.markdown("##### IC 时间序列")
                        st.line_chart(ic_series)
                        st.success("IC/IR 分析完成！")

                        # 3. 执行分层回测
                        st.info("步骤3: 执行因子分层回测...")
                        layered_results, fig = factor_analyzer.run_layered_backtest(factor_df, num_quantiles=5, forward_return_period=20)
                        st.success("分层回测完成！")

                        st.markdown("##### 分层回测净值曲线")
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"因子分析过程中发生错误: {e}")
                        st.exception(e)

# --- 7. 回测实验室 (重构版) ---
if tab_backtest:
    with tab_backtest:
        st.subheader("策略回测实验室")

        backtest_type = st.radio("选择回测类型:", ("向量化回测 (速度快，适合多因子)", "事件驱动回测 (精度高，模拟真实交易)"))

    if backtest_type == "向量化回测 (速度快，适合多因子)":
        st.markdown("---")
        st.markdown("构建多因子策略，通过投资组合优化器生成权重，并在考虑交易成本和风控规则下进行回测。")

        st.markdown("#### 1. 选择策略模型")

        # V2.2 核心升级：增加策略选择
        strategy_model = st.radio(
            "选择策略模型:",
            ["固定权重多因子", "自适应权重 (IC-IR)", "机器学习 (LGBM)"],
            horizontal=True,
            help="""
            - **固定权重多因子**: 手动为多个因子设定固定权重，构建一个静态的选股模型。
            - **自适应权重 (IC-IR)**: 系统会动态计算每个因子在过去的表现（ICIR），并自动为表现好的因子分配更高的权重，实现模型的动态择优。
            """
        )

        st.markdown("#### 2. 配置因子与参数")

        # --- 策略参数配置区 ---
        if strategy_model == "固定权重多因子":
            st.markdown("##### (A) 手动设置固定权重")
            factor_weights = {}
            factor_weights['momentum'] = st.slider("动量因子 (Momentum) 权重:", -1.0, 1.0, 0.5, 0.1)
            factor_weights['volatility'] = st.slider("低波动因子 (Volatility) 权重:", -1.0, 1.0, -0.3, 0.1) # 权重为负代表选取波动率低的
            factor_weights['net_inflow_ratio'] = st.slider("资金流入因子 (Net Inflow) 权重:", -1.0, 1.0, 0.2, 0.1)

        elif strategy_model == "自适应权重 (IC-IR)":
            st.markdown("##### (A) 选择因子池并配置自适应参数")
            factors_to_use_adaptive = st.multiselect(
                "选择纳入自适应权重模型的因子池:",
                options=['momentum', 'volatility', 'net_inflow_ratio', 'roe', 'pe_ttm', 'growth_revenue_yoy'],
                default=['momentum', 'pe_ttm', 'roe']
            )
            ic_lookback_days = st.slider("IC/IR 计算回看期 (天):", 30, 365, 180, 10)

        st.markdown("#### 3. 配置回测参数")
        col1, col2, col3 = st.columns(3)
        with col1:
            bt_start_date = st.date_input("回测开始日期", datetime(2023, 1, 1), key="bt_start")
        with col2:
            bt_end_date = st.date_input("回测结束日期", datetime.now() - timedelta(days=1), key="bt_end")
        with col3:
            rebalance_freq = st.selectbox("调仓频率", ['M', 'W'], index=0, help="M=月度调仓, W=周度调仓")

        st.markdown("#### 4. 配置交易与风控规则")
        col4, col5, col6 = st.columns(3)
        with col4:
            commission = st.number_input("手续费率(%)", 0.0, 1.0, 0.03, 0.01) / 100
        with col5:
            max_weight = st.number_input("单票最大权重(%)", 1.0, 100.0, 10.0, 1.0) / 100
        with col6:
            stop_loss = st.number_input("止损线(%)", 0.0, 50.0, 15.0, 1.0, help="0表示不止损") / 100
            stop_loss = stop_loss if stop_loss > 0 else None

        if st.button("🚀 开始优化并回测"):
            with st.spinner("正在执行向量化回测..."):
                try:
                    # --- 1. 数据准备 ---
                    st.info("步骤1: 准备股票池和价格数据...")
                    stock_pool = get_stock_list()['ts_code'].tolist()[:100] # 缩小范围以提高速度
                    bt_start_str = bt_start_date.strftime('%Y%m%d')
                    bt_end_str = bt_end_date.strftime('%Y%m%d')

                    # 为自适应策略预加载更长周期的价格数据
                    prices_start_str = bt_start_str
                    if strategy_model == "自适应权重 (IC-IR)":
                        prices_start_str = (bt_start_date - timedelta(days=ic_lookback_days + 60)).strftime('%Y%m%d')

                    prices_dict = data_manager.run_batch_download(stock_pool, prices_start_str, bt_end_str)

                    all_prices_df = pd.DataFrame({
                        stock: df.set_index('trade_date')['close']
                        for stock, df in prices_dict.items() if df is not None and not df.empty
                    }).sort_index()
                    all_prices_df.index = pd.to_datetime(all_prices_df.index)
                    all_prices_df.dropna(axis=1, how='all', inplace=True)
                    stock_pool = all_prices_df.columns.tolist() # 更新为实际有数据的股票池
                    st.success(f"价格数据准备完成！股票池数量: {len(stock_pool)}")

                    # --- 2. 确定调仓日期 ---
                    st.info("步骤2: 确定调仓日期...")
                    # 筛选出回测区间内的价格数据
                    backtest_prices = all_prices_df.loc[bt_start_date:bt_end_date]
                    if rebalance_freq == 'M':
                        rebalance_dates = backtest_prices.resample('M').last().index
                    else: # 'W'
                        rebalance_dates = backtest_prices.resample('W').last().index
                    rebalance_dates = rebalance_dates[(rebalance_dates >= backtest_prices.index.min()) & (rebalance_dates <= backtest_prices.index.max())]
                    st.success(f"确定了 {len(rebalance_dates)} 个调仓日。")

                    # --- 3. 根据策略选择，初始化并执行权重生成 ---
                    st.info("步骤3: 在每个调仓日循环计算因子和优化权重...")
                    all_weights_df = pd.DataFrame(index=backtest_prices.index, columns=stock_pool)
                    progress_bar = st.progress(0)
                    optimized_weights = pd.DataFrame() # Initialize to avoid UnboundLocalError

                    # --- A. 固定权重策略 ---
                    if strategy_model == "固定权重多因子":
                        for i, date in enumerate(rebalance_dates):
                            composite_factor = pd.Series(dtype=float)
                            factor_date_str = date.strftime('%Y%m%d')
                            factor_start_str = (date - timedelta(days=60)).strftime('%Y%m%d')
                            for factor_name, weight in factor_weights.items():
                                if weight == 0: continue
                                raw_values = {s: factor_factory.calculate(factor_name, ts_code=s, start_date=factor_start_str, end_date=factor_date_str) for s in stock_pool}
                                raw_series = pd.Series(raw_values).dropna()
                                if raw_series.empty: continue
                                processed_factor = factor_processor.process_factor(raw_series, neutralize=True)
                                if composite_factor.empty:
                                    composite_factor = processed_factor.mul(weight).reindex(stock_pool).fillna(0)
                                else:
                                    composite_factor = composite_factor.add(processed_factor.mul(weight), fill_value=0)

                            if composite_factor.empty or composite_factor.sum() == 0: continue
                            selected_stocks = composite_factor.nlargest(20).index
                            cov_matrix = all_prices_df[selected_stocks].loc[:date].pct_change().iloc[-252:].cov() * 252
                            expected_returns = composite_factor[selected_stocks]
                            optimizer = quant_engine.PortfolioOptimizer(expected_returns, cov_matrix)
                            optimized_weights = optimizer.optimize_max_sharpe(max_weight_per_stock=max_weight)
                            all_weights_df.loc[date] = optimized_weights['weight']
                            progress_bar.progress((i + 1) / len(rebalance_dates))

                    # --- B. 自适应权重策略 ---
                    elif strategy_model == "自适应权重 (IC-IR)":
                        st.info("  正在初始化自适应Alpha策略引擎...")
                        adaptive_strategy = quant_engine.AdaptiveAlphaStrategy(factor_factory, factor_processor, factor_analyzer, all_prices_df)
                        st.success("  自适应策略引擎初始化成功！")

                        for i, date in enumerate(rebalance_dates):
                            composite_factor, dynamic_weights = adaptive_strategy.generate_composite_factor(date, stock_pool, tuple(factors_to_use_adaptive), ic_lookback_days)
                            if i == 0:
                                st.write("第一次调仓日计算出的动态因子权重:")
                                st.dataframe(dynamic_weights)

                            if composite_factor.empty or composite_factor.sum() == 0: continue
                            selected_stocks = composite_factor.nlargest(20).index
                            cov_matrix = all_prices_df[selected_stocks].loc[:date].pct_change().iloc[-252:].cov() * 252
                            expected_returns = composite_factor[selected_stocks]
                            optimizer = quant_engine.PortfolioOptimizer(expected_returns, cov_matrix)
                            optimized_weights = optimizer.optimize_max_sharpe(max_weight_per_stock=max_weight)
                            all_weights_df.loc[date] = optimized_weights['weight']
                            progress_bar.progress((i + 1) / len(rebalance_dates))

                    # --- C. 机器学习策略 ---
                    elif strategy_model == "机器学习 (LGBM)":
                        st.info("  正在初始化并加载机器学习模型...")
                        ml_strategy = quant_engine.MLAlphaStrategy()
                        model_loaded = ml_strategy.load_model() # 默认加载 ml_model.pkl

                        if not model_loaded:
                            st.error("错误：找不到已训练的模型文件 (ml_model.pkl)。请先在“模型训练室”中训练并保存模型。")
                            st.stop()
                        st.success("  模型加载成功！")

                        # 获取模型需要的所有因子
                        model_features = ml_strategy.model.feature_name_

                        for i, date in enumerate(rebalance_dates):
                            date_str = date.strftime('%Y-%m-%d')
                            query = text(f"""
                                SELECT ts_code, factor_name, factor_value
                                FROM factors_exposure
                                WHERE trade_date = '{date_str}'
                                AND factor_name IN ({','.join([f"'{f}'" for f in model_features])})
                            """)
                            with data_manager.engine.connect() as conn:
                                factor_data_today = pd.read_sql(query, conn)

                            if factor_data_today.empty:
                                continue

                            factor_snapshot = factor_data_today.pivot(
                                index='ts_code', columns='factor_name', values='factor_value'
                            ).reindex(columns=model_features).dropna()

                            if factor_snapshot.empty:
                                continue

                            selected_stocks = ml_strategy.predict_top_stocks(factor_snapshot, top_n=20)

                            if len(selected_stocks) > 0:
                                weights = 1.0 / len(selected_stocks)
                                optimized_weights = pd.DataFrame({'weight': weights}, index=selected_stocks)
                                all_weights_df.loc[date] = optimized_weights['weight']

                            progress_bar.progress((i + 1) / len(rebalance_dates))

                    # --- 4. 填充权重并执行回测 ---
                    st.info("步骤4: 所有调仓日权重计算完成，开始执行向量化回测...")
                    all_weights_df.fillna(0, inplace=True)
                    all_weights_df = all_weights_df.reindex(backtest_prices.index).ffill().fillna(0)
                    st.success("权重填充完毕！")

                    st.info("步骤5: 执行统一的向量化回测...")
                    bt = quant_engine.VectorizedBacktester(
                        all_prices=all_prices_df,
                        all_factors=None,
                        rebalance_freq=rebalance_freq,
                        commission=commission,
                        slippage=0.0,
                        stop_loss_pct=stop_loss
                    )

                    results = bt.run(weights_df=all_weights_df)

                    st.success("回测完成！")
                    st.markdown("#### 绩效指标 (已考虑交易成本与风控)")
                    st.table(results['performance'])
                    if not optimized_weights.empty:
                        st.markdown("#### 优化后持仓权重 (最后调仓日)")
                        st.dataframe(optimized_weights.style.format({'weight': '{:.2%}'}))
                    st.markdown("#### 净值曲线与回撤")
                    st.plotly_chart(bt.plot_results(), use_container_width=True)

                    st.markdown("#### 投资组合风险暴露分析")
                    with st.spinner("正在执行风险暴露分析..."):
                        try:
                            risk_manager = quant_engine.RiskManager(factor_factory, factor_processor)
                            valid_rebalance_dates = all_weights_df[all_weights_df.sum(axis=1) > 0].index
                            all_exposures = []
                            for date in valid_rebalance_dates:
                                portfolio_weights = all_weights_df.loc[date].dropna()
                                portfolio_weights = portfolio_weights[portfolio_weights > 0]
                                if not portfolio_weights.empty:
                                    exposure = risk_manager.calculate_risk_exposure(portfolio_weights, date.strftime('%Y%m%d'))
                                    exposure.name = date
                                    all_exposures.append(exposure)
                            
                            if all_exposures:
                                exposure_df = pd.concat(all_exposures, axis=1).T
                                fig = go.Figure()
                                for factor in exposure_df.columns:
                                    fig.add_trace(go.Scatter(x=exposure_df.index, y=exposure_df[factor], mode='lines', name=factor))
                                fig.add_hline(y=0, line_dash="dash", line_color="white")
                                fig.update_layout(title="策略风险因子暴露度时序图", xaxis_title="日期", yaxis_title="标准化暴露值 (Z-Score)", template="plotly_dark", height=500)
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption("暴露值为正，表示您的投资组合在该风险因子上呈正向暴露；值为负则相反。接近0表示在该风险上表现中性。")
                            else:
                                st.warning("未能计算任何日期的风险暴露。")
                        except Exception as e:
                            st.error(f"风险暴露分析失败: {e}")

                    st.markdown("#### 深度绩效归因 (Brinson Model)")
                    with st.spinner("正在执行Brinson归因分析..."):
                        try:
                            rebalance_dates_attr = bt._get_rebalance_dates()
                            if len(rebalance_dates_attr) > 1 and not optimized_weights.empty:
                                attribution_period_start = rebalance_dates_attr[0]
                                attribution_period_end = rebalance_dates_attr[-1]
                                stock_basics = get_stock_list()
                                stock_industry_map = stock_basics[stock_basics['ts_code'].isin(stock_pool)][['ts_code', 'industry']]
                                portfolio_weights_for_attr = optimized_weights['weight']
                                benchmark_weights_for_attr = pd.Series(1/len(stock_pool), index=stock_pool)
                                period_returns = all_prices_df.loc[attribution_period_end] / all_prices_df.loc[attribution_period_start] - 1

                                attribution_analyzer = quant_engine.PerformanceAttribution(
                                    portfolio_returns=period_returns.reindex(portfolio_weights_for_attr.index).fillna(0),
                                    portfolio_weights=portfolio_weights_for_attr,
                                    benchmark_returns=period_returns.reindex(benchmark_weights_for_attr.index).fillna(0),
                                    benchmark_weights=benchmark_weights_for_attr,
                                    stock_industry_map=stock_industry_map
                                )
                                attribution_results = attribution_analyzer.run_brinson_attribution()
                                st.dataframe(attribution_results)
                                st.caption("正向的'资产配置'表示策略超配了表现优于基准的行业。正向的'个股选择'表示在行业内部选出的个股表现优于该行业的整体基准。")
                            else:
                                st.warning("数据不足，无法执行业绩归因分析。")
                        except Exception as e:
                            st.error(f"业绩归因分析失败: {e}")

                except Exception as e:
                    st.error(f"向量化回测过程中发生错误: {e}")
                    st.exception(e)

    elif backtest_type == "事件驱动回测 (精度高，模拟真实交易)":
        st.markdown("---")
        st.markdown("模拟真实的逐日交易过程，策略在每个交易日接收数据并做出决策，适合验证均线、突破等时序型策略。")

        st.markdown("#### 1. 配置回测参数")
        ed_col1, ed_col2, ed_col3 = st.columns(3)
        with ed_col1:
            ed_start_date = st.date_input("回测开始日期", datetime(2023, 1, 1), key="ed_start")
        with ed_col2:
            ed_end_date = st.date_input("回测结束日期", datetime.now() - timedelta(days=1), key="ed_end")
        with ed_col3:
            initial_capital = st.number_input("初始资金", 100000, 100000000, 1000000, 100000)

        st.markdown("#### 2. 配置策略与股票池")
        strategy_choice = st.selectbox("选择策略", ["双均线交叉策略"])

        ed_col4, ed_col5 = st.columns(2)
        with ed_col4:
            short_window = st.slider("短期均线窗口", 5, 50, 10, 1)
        with ed_col5:
            long_window = st.slider("长期均线窗口", 20, 120, 30, 1)

        stock_pool_options = get_stock_list()['ts_code'] + " " + get_stock_list()['name']
        ed_stock_pool = st.multiselect("选择股票池 (建议3-5支)", options=stock_pool_options, default=[s for s in stock_pool_options if "茅台" in s or "平安" in s])
        ed_stock_codes = [s.split(" ")[0] for s in ed_stock_pool]

        if st.button("🚀 开始事件驱动回测"):
            if not ed_stock_codes:
                st.warning("请至少选择一支股票。")
            else:
                with st.spinner("正在执行事件驱动回测，请稍候..."):
                    try:
                        # 1. 数据准备
                        st.info("步骤1: 准备股票池的价格与成交量数据...")
                        ed_start_str = ed_start_date.strftime('%Y%m%d')
                        ed_end_str = ed_end_date.strftime('%Y%m%d')
                        prices_dict = data_manager.run_batch_download(ed_stock_codes, ed_start_str, ed_end_str)

                        all_prices_df = pd.DataFrame({
                            stock: df.set_index('trade_date')['close']
                            for stock, df in prices_dict.items() if df is not None and not df.empty and 'close' in df.columns
                        }).sort_index()

                        all_volumes_df = pd.DataFrame({
                            stock: df.set_index('trade_date')['vol']
                            for stock, df in prices_dict.items() if df is not None and not df.empty and 'vol' in df.columns
                        }).sort_index()

                        common_index = all_prices_df.index.intersection(all_volumes_df.index)
                        common_columns = all_prices_df.columns.intersection(all_volumes_df.columns)
                        all_prices_df = all_prices_df.loc[common_index, common_columns]
                        all_volumes_df = all_volumes_df.loc[common_index, common_columns]
                        all_prices_df.dropna(axis=1, how='all', inplace=True)
                        all_volumes_df = all_volumes_df.reindex(columns=all_prices_df.columns)

                        st.success(f"价格与成交量数据准备完成！股票池: {all_prices_df.columns.tolist()}")

                        # 2. 初始化事件驱动引擎
                        st.info("步骤2: 初始化事件驱动引擎组件...")
                        from queue import Queue
                        events_queue = Queue()

                        data_handler = quant_engine.SimpleDataHandler(events_queue, all_prices_df, all_volumes_df)
                        portfolio = quant_engine.SimplePortfolio(data_handler, events_queue, initial_capital)
                        strategy = quant_engine.SimpleMovingAverageStrategy(data_handler, short_window, long_window)
                        execution_handler = quant_engine.MockExecutionHandler(events_queue, data_handler, portfolio)

                        backtester = quant_engine.EventDrivenBacktester(
                            data_handler, strategy, portfolio, execution_handler
                        )
                        st.success("引擎初始化完毕！")

                        # 3. 运行回测
                        st.info("步骤3: 开始运行事件循环...")
                        ed_results = backtester.run_backtest()
                        st.success("事件驱动回测完成！")

                        # 4. 展示结果
                        st.markdown("#### 绩效指标")
                        st.table(ed_results['performance'])

                        st.markdown("#### 净值曲线")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ed_results['equity_curve'].index, y=ed_results['equity_curve']['total'], mode='lines', name='策略净值'))
                        fig.update_layout(title="事件驱动回测 - 资产净值变化", template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("#### 详细交易记录")
                        st.dataframe(ed_results['trade_log'], use_container_width=True)

                    except Exception as e:
                        st.error(f"事件驱动回测过程中发生错误: {e}")
                        st.exception(e)

# --- V2.3 新增: 9. 模型训练室 ---
if tab_trainer:
    with tab_trainer:
        st.subheader("🔬 机器学习模型训练室")
        st.markdown("在这里，您可以选择因子（特征）和预测目标，训练您自己的机器学习选股模型，并将其应用到回测实验室中。")

        st.markdown("#### 1. 配置训练参数")

        train_cols = st.columns(3)
        with train_cols[0]:
            train_start_date = st.date_input("训练开始日期", datetime(2023, 1, 1), key="train_start")
        with train_cols[1]:
            train_end_date = st.date_input("训练结束日期", datetime(2024, 12, 31), key="train_end")
        with train_cols[2]:
            forward_period = st.number_input("预测周期(天)", 1, 60, 20, 1)

        try:
            with data_manager.engine.connect() as conn:
                all_db_factors = pd.read_sql("SELECT DISTINCT factor_name FROM factors_exposure", conn)['factor_name'].tolist()
        except Exception:
            all_db_factors = ['momentum', 'roe', 'pe_ttm', 'volatility', 'net_inflow_ratio']

        # 【鲁棒性修复】确保默认值是可用选项的子集，防止因数据库为空或因子不匹配而报错
        desired_defaults = ['momentum', 'roe', 'pe_ttm', 'volatility']
        actual_defaults = [f for f in desired_defaults if f in all_db_factors]

        selected_features = st.multiselect(
            "选择用作特征的因子:",
            options=all_db_factors,
            default=actual_defaults
        )

        st.markdown("#### 2. 开始训练")
        if st.button("🚀 开始训练模型", use_container_width=True):
            if not selected_features:
                st.warning("请至少选择一个特征因子。")
            else:
                with st.spinner("正在执行模型训练工作流，这可能需要几分钟时间..."):
                    try:
                        # --- 1. 数据准备 ---
                        st.info("步骤1: 准备股票池和价格数据...")
                        stock_pool = get_stock_list()['ts_code'].tolist()[:200]
                        train_start_str = train_start_date.strftime('%Y%m%d')
                        train_end_fetch_str = (train_end_date + timedelta(days=forward_period * 2)).strftime('%Y%m%d')

                        prices_dict = data_manager.run_batch_download(stock_pool, train_start_str, train_end_fetch_str)
                        all_prices_df = pd.DataFrame({
                            stock: df.set_index('trade_date')['close']
                            for stock, df in prices_dict.items() if df is not None and not df.empty
                        }).sort_index()
                        all_prices_df.index = pd.to_datetime(all_prices_df.index)
                        all_prices_df.dropna(axis=1, how='all', inplace=True)
                        st.success(f"价格数据准备完成！")

                        st.info("步骤2: 准备因子数据...")
                        query = text(f"""
                            SELECT trade_date, ts_code, factor_name, factor_value
                            FROM factors_exposure
                            WHERE trade_date BETWEEN '{train_start_date.strftime('%Y-%m-%d')}' AND '{train_end_date.strftime('%Y-%m-%d')}'
                            AND factor_name IN ({','.join([f"'{f}'" for f in selected_features])})
                        """)
                        with data_manager.engine.connect() as conn:
                            all_factor_data = pd.read_sql(query, conn)

                        all_factor_data['trade_date'] = pd.to_datetime(all_factor_data['trade_date'])
                        st.success(f"因子数据准备完成！")

                        # --- 2. 模型训练 ---
                        st.info("步骤3: 实例化并训练模型...")
                        ml_strategy = quant_engine.MLAlphaStrategy()
                        train_results = ml_strategy.train(all_prices_df, all_factor_data)
                        st.success("模型训练成功！")

                        # --- 3. 展示结果 ---
                        st.markdown("#### 训练结果报告")
                        st.metric("测试集准确率", f"{train_results.get('accuracy', 0):.2%}")
                        st.json(train_results)

                    except Exception as e:
                        st.error(f"模型训练过程中发生错误: {e}")
                        st.exception(e)

# --- 10. 系统任务 ---
if tab_tasks:
    with tab_tasks:
        st.subheader("自动化与监控中心")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 后台任务手动触发器")
            st.warning("【重要】此任务耗时较长（约20-40分钟），将在后台独立运行。您可以在右侧的日志监控面板查看进度。")

            if st.button("🚀 执行每日统一数据管道 (抽取+计算)", help="启动后台进程，执行完整的数据抽取和因子计算流程，并将结果存入数据库。"):
                # FIX 2: Corrected the structure of the try/except block, completed the Popen statement,
                # and fixed the indentation for 'with col2' which was causing a syntax error.
                try:
                    command = [sys.executable, "run_daily_pipeline.py"]
                    # Popen starts a new process and does not block Streamlit.
                    # Completed the Popen call with a closing parenthesis.
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    st.success(f"后台任务已启动 (进程ID: {process.pid})。")
                    st.info("您可关闭此浏览器窗口，任务将在服务器后台继续运行。")
                except FileNotFoundError:
                    st.error("错误: 'run_daily_pipeline.py' not found. 请确保该文件与app.py在同一目录下。")
                except Exception as e:
                    st.error(f"启动后台任务失败: {e}")

        # This 'with' block was incorrectly indented. It's now at the correct level.
        with col2:
            st.markdown("#### 系统状态监控面板")
            st.info("实时检查系统关键组件的运行状态。")

            if st.button("刷新监控状态"):
                # 1. 检查数据库连接
                try:
                    with data_manager.engine.connect() as connection:
                        connection.execute(text("SELECT 1"))
                    st.success("✅ **数据库连接:** 正常")
                except Exception as e:
                    st.error(f"❌ **数据库连接:** 失败 - {e}")

                # 2. 查询Tushare API积分
                try:
                    df_score = data_manager.pro.query('tushare_score')
                    if df_score is not None and not df_score.empty:
                        score = df_score.iloc[0]['score']
                        st.success(f"✅ **Tushare API积分:** {score} 分")
                    else:
                         st.warning("⚠️ **Tushare API积分:** 未能查询到积分信息。")
                except Exception as e:
                    st.error(f"❌ **Tushare API积分:** 查询失败 - {e}")

                # 3. 显示最新日志
                st.markdown("##### 最新日志 (`quant_project.log`)")
                try:
                    with open('quant_project.log', 'r', encoding='utf-8') as f:
                        log_lines = f.readlines()
                    st.text_area("Log Preview:", "".join(log_lines[-20:]), height=300)
                except FileNotFoundError:
                    st.warning("⚠️ 日志文件 'quant_project.log' 未找到。")
                except Exception as e:
                    st.error(f"❌ 读取日志文件失败: {e}")
