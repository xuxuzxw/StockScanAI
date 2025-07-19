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
    if st.session_state.selected_stock and st.session_state.selected_stock in stock_options.tolist():
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


# --- 创建多标签页 (V3.0 升级) ---
tab_list = ["🏆 智能选股排名", "📈 行情总览", "💰 资金与筹码", "🧾 深度财务", "🌐 宏观环境"]
if V2_MODULES_LOADED:
    tab_list.extend(["🎯 市场全景", "🏭 行业透视"])
tab_list.extend(["🤖 AI综合报告", "🔬 因子分析器", "🚀 回测实验室", "⚙️ 系统任务"])

tabs = st.tabs(tab_list)

# 根据模块加载情况动态分配变量
if V2_MODULES_LOADED:
    tab_ranker, tab_main, tab_funds, tab_finance, tab_macro, tab_market, tab_industry, tab_ai, tab_analyzer, tab_backtest, tab_tasks = tabs
else:
    (tab_ranker, tab_main, tab_funds, tab_finance, tab_macro, 
     tab_ai, tab_analyzer, tab_backtest, tab_tasks) = tabs
    tab_market, tab_industry = None, None

# --- 1. 智能选股排名 ---
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
        from factor_calculator import FACTORS_TO_CALCULATE as available_factors
        
        cols = st.columns(3)
        factor_direction = {
            'pe_ttm': -1, 'roe': 1, 'growth_revenue_yoy': 1, 'debt_to_assets': -1,
            'momentum': 1, 'volatility': -1, 'net_inflow_ratio': 1
        }

        with cols[0]:
            st.multiselect("选择价值因子", [f for f in available_factors if "pe" in f], default='pe_ttm', key="value_factors")
        with cols[1]:
            st.multiselect("选择质量/成长因子", [f for f in available_factors if any(k in f for k in ['roe', 'growth', 'debt'])], default=['roe', 'growth_revenue_yoy'], key="quality_factors")
        with cols[2]:
            st.multiselect("选择技术/资金因子", [f for f in available_factors if any(k in f for k in ['momentum', 'volatility', 'inflow'])], default=['momentum', 'net_inflow_ratio'], key="tech_factors")
        
        user_selection = st.session_state.value_factors + st.session_state.quality_factors + st.session_state.tech_factors
        
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
                        final_rank_display = final_rank[['ts_code', 'name', 'industry', '综合得分']]
                        st.dataframe(final_rank_display.head(100), hide_index=True)
                        st.caption("💡 小提示：直接点击上方个股表格中的任意一行，系统将自动跳转到该股票的深度分析页面。")

                        # (交互逻辑保持不变，但需要确保 data_editor 在st.rerun后能正确工作)
                        if 'rank_editor_selection' not in st.session_state:
                             st.session_state.rank_editor_selection = None
                        
                        # 使用 on_change 回调来捕获选择
                        def handle_selection():
                            if st.session_state.rank_editor and st.session_state.rank_editor["edited_rows"]:
                                selected_row_index = list(st.session_state.rank_editor["edited_rows"].keys())[0]
                                st.session_state.rank_editor_selection = final_rank_display.iloc[selected_row_index]

                        st.data_editor(
                            final_rank_display.head(1), # 仅用于触发回调，实际展示由上面的dataframe完成
                            key="rank_editor",
                            hide_index=True,
                            on_change=handle_selection,
                            disabled=True # 设为不可编辑，只利用其选择事件
                        )
                        
                        if st.session_state.rank_editor_selection is not None:
                             selected_ts_code = st.session_state.rank_editor_selection['ts_code']
                             selected_name = st.session_state.rank_editor_selection['name']
                             st.session_state.selected_stock = f"{selected_ts_code} {selected_name}"
                             st.session_state.rank_editor_selection = None # 重置
                             st.rerun()

                    except Exception as e:
                        st.error(f"排名计算过程中发生错误: {e}")
                        st.exception(e)

# --- 2. 行情总览 ---
with tab_main:
    st.subheader("日K线图 (后复权) & 综合指标")
    df_adj = data_manager.get_adjusted_daily(ts_code, start_date_str, end_date_str, adj='hfq')
    if df_adj is not None and not df_adj.empty:
        # --- 1. 数据获取与合并 ---
        # 获取每日基本面指标（PE、换手率等）
        df_basic = data_manager.get_daily_basic(ts_code, start_date_str, end_date_str)
        if df_basic is not None and not df_basic.empty:
            # 【修正】在合并前，确保两个DataFrame的'trade_date'列都是datetime类型
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
with tab_funds:
    st.subheader("资金流向 & 股东结构")
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
            
    st.markdown("**前十大流通股东 (最新报告期)**")
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
        st.dataframe(df_holders, use_container_width=True, height=385)
    else:
        st.warning("未能获取前十大流通股东数据。")

# --- 3. 深度财务 ---
with tab_finance:
    st.subheader("财务报表核心数据")
    if latest_period:
        st.markdown(f"**利润表 ({latest_period})**")
        df_income = data_manager.get_income(ts_code, latest_period)
        if df_income is not None and not df_income.empty:
            # 【最终修复】确保只处理最新的一份报告，防止多行数据导致转置后出现多列
            if len(df_income) > 1:
                df_income = df_income.sort_values(by='ann_date', ascending=False).head(1)
            
            df_display = df_income.T.reset_index()
            df_display.columns = ['指标', '数值']
            df_display['数值'] = df_display['数值'].astype(str)
            st.dataframe(df_display, use_container_width=True)

        st.markdown(f"**资产负债表 ({latest_period})**")
        df_balance = data_manager.get_balancesheet(ts_code, latest_period)
        if df_balance is not None and not df_balance.empty:
            # 应用同样的修复逻辑
            if len(df_balance) > 1:
                df_balance = df_balance.sort_values(by='ann_date', ascending=False).head(1)
                
            df_display = df_balance.T.reset_index()
            df_display.columns = ['指标', '数值']
            df_display['数值'] = df_display['数值'].astype(str)
            st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("未能确定最新的财报周期，无法加载财务报表。")

# --- 4. 宏观环境 ---
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
with tab_ai:
    st.subheader("混合AI智能体分析")
    st.markdown("点击下方按钮，AI将采集并分析该股的 **技术、资金、财务、筹码、宏观、舆情** 六大维度数据，生成一份深度综合投研报告。")
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
                        start_date_str = (date - pd.Timedelta(days=60)).strftime('%Y%m%d')
                        
                        raw_values = {}
                        for code in analyzer_stock_codes:
                            calc_func = getattr(factor_factory, f"calc_{factor_to_analyze}")
                            # 因子函数需要不同的参数，这里做一个适配
                            if factor_to_analyze in ['momentum', 'volatility', 'net_inflow_ratio', 'north_hold_change']:
                                raw_values[code] = calc_func(ts_code=code, start_date=start_date_str, end_date=date_str)
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
with tab_backtest:
    st.subheader("策略回测实验室")
    
    backtest_type = st.radio("选择回测类型:", ("向量化回测 (速度快，适合多因子)", "事件驱动回测 (精度高，模拟真实交易)"))

    if backtest_type == "向量化回测 (速度快，适合多因子)":
        st.markdown("---")
        st.markdown("构建多因子策略，通过投资组合优化器生成权重，并在考虑交易成本和风控规则下进行回测。")

        st.markdown("#### 1. 选择因子并设置权重策略")
        
        weight_strategy = st.radio("选择权重策略:", ["固定权重", "自适应权重 (基于IC-IR)"], horizontal=True)

        factor_weights = {}
        factors_to_use = ('momentum', 'volatility', 'net_inflow')

        if weight_strategy == "固定权重":
            st.markdown("##### (1) 手动设置固定权重")
            factor_weights['momentum'] = st.slider("动量因子 (Momentum) 权重:", -1.0, 1.0, 0.5, 0.1)
            factor_weights['volatility'] = st.slider("低波动因子 (Volatility) 权重:", -1.0, 1.0, -0.3, 0.1) # 权重为负代表选取波动率低的
            factor_weights['net_inflow'] = st.slider("资金流入因子 (Net Inflow) 权重:", -1.0, 1.0, 0.2, 0.1)
        else:
            st.markdown("##### (1) 配置自适应权重参数")
            st.info("权重将基于过去一段时间内因子的IC-IR值动态计算，无需手动设置。")
            ic_lookback_days = st.slider("IC/IR 计算回看期 (天):", 30, 365, 180, 10)
            
        st.markdown("#### 2. 配置回测参数")
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_start_date = st.date_input("回测开始日期", datetime(2023, 1, 1), key="bt_start")
    with col2:
        bt_end_date = st.date_input("回测结束日期", datetime.now() - timedelta(days=1), key="bt_end")
    with col3:
        rebalance_freq = st.selectbox("调仓频率", ['M', 'W'], index=0, help="M=月度调仓, W=周度调仓")

    st.markdown("#### 3. 配置交易与风控规则")
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
                if weight_strategy == "自适应权重 (基于IC-IR)":
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

                # --- 3. 初始化自适应策略（如果需要）---
                adaptive_strategy = None
                if weight_strategy == "自适应权重 (基于IC-IR)":
                    st.info("步骤3: 初始化自适应Alpha策略引擎...")
                    adaptive_strategy = quant_engine.AdaptiveAlphaStrategy(factor_factory, factor_processor, factor_analyzer, all_prices_df)
                    st.success("自适应策略引擎初始化成功！")

                # --- 4. 循环计算因子、优化并生成权重 ---
                st.info("步骤4: 在每个调仓日循环计算因子和优化权重...")
                all_weights_df = pd.DataFrame(index=backtest_prices.index, columns=stock_pool)
                
                progress_bar = st.progress(0)
                for i, date in enumerate(rebalance_dates):
                    # --- A. 计算当期合成因子 ---
                    if weight_strategy == "自适应权重 (基于IC-IR)":
                        composite_factor, dynamic_weights = adaptive_strategy.generate_composite_factor(date, stock_pool, factors_to_use, ic_lookback_days)
                        if i == 0: 
                            st.write("第一次调仓日计算出的动态因子权重:")
                            st.dataframe(dynamic_weights)
                    else: # 固定权重
                        composite_factor = pd.Series(dtype=float)
                        factor_date_str = date.strftime('%Y%m%d')
                        factor_start_str = (date - timedelta(days=60)).strftime('%Y%m%d')
                        for factor_name, weight in factor_weights.items():
                            if weight == 0: continue
                            raw_values = {s: getattr(factor_factory, f"calc_{factor_name}")(ts_code=s, start_date=factor_start_str, end_date=factor_date_str) for s in stock_pool}
                            raw_series = pd.Series(raw_values).dropna()
                            if raw_series.empty: continue
                            processed_factor = factor_processor.process_factor(raw_series, neutralize=True)
                            if composite_factor.empty:
                                composite_factor = processed_factor.mul(weight).reindex(stock_pool).fillna(0)
                            else:
                                composite_factor = composite_factor.add(processed_factor.mul(weight), fill_value=0)

                    # --- B. 基于合成因子进行组合优化 ---
                    if composite_factor.empty or composite_factor.sum() == 0: continue
                    selected_stocks = composite_factor.nlargest(20).index
                    
                    # 使用截至当前调仓日的数据计算协方差矩阵
                    cov_matrix = all_prices_df[selected_stocks].loc[:date].pct_change().iloc[-252:].cov() * 252 # 使用过去一年的数据
                    expected_returns = composite_factor[selected_stocks] # 用因子值作为预期收益的代理
                    
                    optimizer = quant_engine.PortfolioOptimizer(expected_returns, cov_matrix)
                    optimized_weights = optimizer.optimize_max_sharpe(max_weight_per_stock=max_weight)
                    
                    # --- C. 将当期权重填充到下一个调仓期 ---
                    next_rebalance_date = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else backtest_prices.index[-1]
                    all_weights_df.loc[date:next_rebalance_date, optimized_weights.index] = optimized_weights['weight']
                    progress_bar.progress((i + 1) / len(rebalance_dates))

                all_weights_df.fillna(0, inplace=True)
                all_weights_df = all_weights_df.ffill().fillna(0) # 向前填充，确保整个回测期都有权重
                st.success("所有调仓日权重计算完成！")

                # 5. 执行向量化回测
                st.info("步骤4: 执行统一的向量化回测...")
                bt = quant_engine.VectorizedBacktester(
                    all_prices=all_prices_df,
                    all_factors=None, # factors 已被用于生成权重，回测器不再需要
                    rebalance_freq=rebalance_freq, # 频率信息仍可用于分析
                    commission=commission,
                    slippage=0.0,
                    stop_loss_pct=stop_loss
                )
                
                # 使用包含时间序列的权重DataFrame进行回测
                results = bt.run(weights_df=all_weights_df)

                st.success("回测完成！")
                st.markdown("#### 绩效指标 (已考虑交易成本与风控)")
                st.table(results['performance'])
                st.markdown("#### 优化后持仓权重")
                st.dataframe(optimized_weights.style.format({'weight': '{:.2%}'}))
                st.markdown("#### 净值曲线与回撤")
                st.plotly_chart(bt.plot_results(), use_container_width=True)
                st.markdown("#### 深度绩效归因 (Brinson Model)")
                with st.spinner("正在执行Brinson归因分析..."):
                    try:
                        # 1. 获取归因分析所需的数据周期
                        rebalance_dates = bt._get_rebalance_dates()
                        attribution_period_start = rebalance_dates[0]
                        attribution_period_end = rebalance_dates[-1]

                        # 2. 准备归因分析的输入
                        stock_basics = get_stock_list()
                        stock_industry_map = stock_basics[stock_basics['ts_code'].isin(stock_pool)][['ts_code', 'industry']]
                        
                        # 为简化，我们使用第一天的权重和整个周期的总回报
                        portfolio_weights_for_attr = optimized_weights['weight']
                        
                        # 创建一个简单的基准（在股票池中等权重配置）
                        benchmark_weights_for_attr = pd.Series(1/len(stock_pool), index=stock_pool)

                        period_returns = all_prices_df.loc[attribution_period_end] / all_prices_df.loc[attribution_period_start] - 1

                        # 3. 运行归因分析
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
                        # 1. 数据准备 (增强版，获取价格和成交量)
                        st.info("步骤1: 准备股票池的价格与成交量数据...")
                        prices_dict = data_manager.run_batch_download(ed_stock_codes, ed_start_str, ed_end_str)
                        
                        all_prices_df = pd.DataFrame({
                            stock: df.set_index('trade_date')['close']
                            for stock, df in prices_dict.items() if df is not None and not df.empty and 'close' in df.columns
                        }).sort_index()

                        all_volumes_df = pd.DataFrame({
                            stock: df.set_index('trade_date')['vol']
                            for stock, df in prices_dict.items() if df is not None and not df.empty and 'vol' in df.columns
                        }).sort_index()
                        
                        # 对齐数据
                        common_index = all_prices_df.index.intersection(all_volumes_df.index)
                        common_columns = all_prices_df.columns.intersection(all_volumes_df.columns)
                        all_prices_df = all_prices_df.loc[common_index, common_columns]
                        all_volumes_df = all_volumes_df.loc[common_index, common_columns]

                        all_prices_df.dropna(axis=1, how='all', inplace=True)
                        all_volumes_df = all_volumes_df.reindex(columns=all_prices_df.columns) # 确保对齐

                        st.success(f"价格与成交量数据准备完成！股票池: {all_prices_df.columns.tolist()}")

                        # 2. 初始化事件驱动引擎 (增强版)
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

# --- 8. 系统任务 ---
with tab_tasks:
    st.subheader("自动化与监控中心")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 后台任务手动触发器")
        st.warning("【重要】以下任务耗时较长，将在后台独立运行。您可以在右侧的日志监控面板查看进度。")
        
        if st.button("① 执行每日数据抽取", help="启动后台进程，下载所有计算因子所需的原始数据并存入本地缓存。此过程耗时最长，约20-40分钟。"):
            try:
                # 使用 sys.executable 确保我们用的是当前环境的python解释器
                command = [sys.executable, "data_extractor.py"]
                # Popen 会启动一个新进程，并且不会阻塞Streamlit
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.success("✅ **数据抽取任务已在后台启动！** 您可以继续操作平台，或在右侧刷新日志查看进度。")
            except Exception as e:
                st.error(f"启动数据抽取任务失败: {e}")

        if st.button("② 执行每日因子计算", help="启动后台进程，读取缓存数据，进行因子计算并存入数据库。请在数据抽取完成后再执行此操作。此过程约1-5分钟。"):
            try:
                command = [sys.executable, "factor_calculator.py"]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.success("✅ **因子计算任务已在后台启动！** 您可以继续操作平台，或在右侧刷新日志查看进度。")
            except Exception as e:
                st.error(f"启动因子计算任务失败: {e}")
        
    with col2:
        st.markdown("#### 系统状态监控面板")
        st.info("实时检查系统关键组件的运行状态。")

        if st.button("刷新监控状态"):
            # 1. 检查数据库连接
            try:
                data_manager.conn.execute("SELECT 1").fetchone()
                st.success("✅ **数据库连接:** 正常")
            except Exception as e:
                st.error(f"❌ **数据库连接:** 失败 - {e}")

            # 2. 查询Tushare API积分
            try:
                df_score = data_manager.pro.tushare_score()
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
                # 显示最后20行
                st.text_area("Log Preview:", "".join(log_lines[-20:]), height=300)
            except FileNotFoundError:
                st.warning("⚠️ 日志文件 'quant_project.log' 未找到。")
            except Exception as e:
                st.error(f"❌ 读取日志文件失败: {e}")
