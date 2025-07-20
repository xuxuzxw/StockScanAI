# StockScanAI/manual_api_tester.py
#
# 【V2.6 终极完整版】A股量化投研平台 - API并发压力测试工具
#
# 功能:
# - 菜单已覆盖项目中所有关键的Tushare API，并按功能分组，便于系统性测试。
# - 使用 aiohttp 和 asyncio 实现高并发请求，真实模拟高频调用场景。
# - 绕过项目内部的 DataManager 和频率控制器，直接对 Tushare API 服务器进行压力测试。
# - 精确统计成功/失败次数和总耗时，帮助您摸清所用 TOKEN 的真实流量限制。

import asyncio
import json
import time
from collections import OrderedDict

import aiohttp

# 导入项目模块
try:
    import config
    from logger_config import log
except ImportError as e:
    print(f"FATAL: 关键模块导入失败: {e}")
    print("请确保您在项目的根目录下运行此脚本。")
    exit(1)

# --- 核心异步测试组件 ---
TUSHARE_API_URL = "http://api.tushare.pro"

# Tushare 接口名映射 (key: 菜单显示名, value: Tushare官方API名)
API_NAME_MAP = {
    # 行情数据
    "get_daily": "daily",
    "get_index_daily": "index_daily",
    "get_adj_factor": "adj_factor",
    "get_daily_basic": "daily_basic",
    "get_index_dailybasic": "index_dailybasic",
    # 资金与筹码
    "get_moneyflow": "moneyflow",
    "get_hk_hold": "hk_hold",
    "get_moneyflow_hsgt": "moneyflow_hsgt",
    "get_top_list": "top_list",
    "get_top_inst": "top_inst",
    "get_margin_detail": "margin_detail",
    "get_block_trade": "block_trade",
    "get_top10_floatholders": "top10_floatholders",
    "get_holder_number": "stk_holdernumber",
    "get_holder_trade": "stk_holdertrade",
    # 财务数据
    "get_fina_indicator": "fina_indicator",
    "get_income": "income",
    "get_balancesheet": "balancesheet",
    "get_cashflow": "cashflow",
    "get_forecast": "forecast",
    "get_express": "express",
    "get_dividend": "dividend",
    "get_repurchase": "repurchase",
    # 宏观经济
    "get_cn_m": "cn_m",
    "get_cn_pmi": "cn_pmi",
    "get_cn_cpi": "cn_cpi",
    "get_cn_gdp": "cn_gdp",
    "get_shibor": "shibor",
}


async def fetch_tushare_direct(
    session, api_name: str, token: str, params: dict, pbar
) -> bool:
    payload = {"api_name": api_name, "token": token, "params": params, "fields": []}
    try:
        async with session.post(
            TUSHARE_API_URL, data=json.dumps(payload), timeout=60
        ) as response:
            resp_json = await response.json()
            pbar.update(1)
            if resp_json.get("code") == 0:
                return True
            else:
                log.debug(f"API返回错误: {resp_json.get('msg', '未知错误')}")
                return False
    except Exception:
        pbar.update(1)
        return False


async def run_stress_test_async(api_name_user: str, api_params: dict, call_count: int):
    try:
        from tqdm import tqdm
    except ImportError:
        log.error("请先安装tqdm库以便显示进度条: pip install tqdm")

        class tqdm:
            def __init__(self, total, desc):
                self.total = total
                self.desc = desc

            def __enter__(self):
                return self

            def __exit__(self, t, v, tb):
                pass

            def update(self, n):
                pass

    api_name_tushare = API_NAME_MAP.get(api_name_user)
    if not api_name_tushare:
        log.error(f"未找到 {api_name_user} 对应的 Tushare API 名称。")
        return

    log.info(f"\n准备就绪！将并发调用 '{api_name_tushare}' 共 {call_count} 次。")
    input("按 Enter 键开始测试...")

    start_time = time.time()
    tasks = []

    async with aiohttp.ClientSession() as session:
        with tqdm(total=call_count, desc=f"并发请求 {api_name_tushare}") as pbar:
            for _ in range(call_count):
                task = asyncio.create_task(
                    fetch_tushare_direct(
                        session,
                        api_name_tushare,
                        config.TUSHARE_TOKEN,
                        api_params,
                        pbar,
                    )
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_duration = end_time - start_time

    success_count = sum(1 for r in results if r)
    failure_count = call_count - success_count

    log.info("\n========== 测试完成 ==========")
    log.info(f"请求总数: {call_count}")
    log.info(f"总耗时: {total_duration:.2f} 秒")
    if total_duration > 0:
        log.info(f"平均每秒完成请求数 (RPS): {call_count / total_duration:.2f}")
    log.info(f"✅ 成功次数: {success_count}")
    log.info(f"❌ 失败次数: {failure_count}")
    if failure_count > 0:
        log.warning("出现失败请求，可能已达到API频率或总量限制。")
    log.info("============================")


# --- UI 和主流程 ---
def display_menu(api_groups: OrderedDict):
    log.info("========== Tushare API 并发压力测试器 (终极完整版) ==========")
    log.info("请从以下选择一个您想测试的API函数:")

    api_flat_list = []
    i = 1
    for group_name, apis in api_groups.items():
        print(f"\n--- {group_name} ---")
        for api_name in apis:
            print(f"  {i}. {api_name}")
            api_flat_list.append(api_name)
            i += 1

    log.info("===================================================================")
    choice = input("请输入选项编号: ")
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(api_flat_list):
            return api_flat_list[choice_idx]
        else:
            log.error("无效的选项。")
            return None
    except ValueError:
        log.error("请输入一个数字。")
        return None


def get_params_for_api(api_name: str, all_apis: dict) -> dict:
    params = {}
    log.info(f"--- 正在为 API '{api_name}' 配置参数 ---")
    param_definitions = all_apis[api_name]["params"]

    if "ts_code" in param_definitions:
        params["ts_code"] = input("请输入股票代码 (例如 600519.SH): ")
    if "index_code" in param_definitions:
        params["ts_code"] = input("请输入指数代码 (例如 000300.SH): ")
    if "start_date" in param_definitions:
        params["start_date"] = input("请输入开始日期 (YYYYMMDD): ")
    if "end_date" in param_definitions:
        params["end_date"] = input("请输入结束日期 (YYYYMMDD): ")
    if "period" in param_definitions:
        params["period"] = input("请输入报告期 (例如 20240331): ")
    if "trade_date" in param_definitions:
        params["trade_date"] = input("请输入交易日期 (YYYYMMDD): ")
    if "start_m" in param_definitions:
        params["start_m"] = input("请输入开始月份 (YYYYMM): ")
    if "end_m" in param_definitions:
        params["end_m"] = input("请输入结束月份 (YYYYMM): ")
    if "start_q" in param_definitions:
        params["start_q"] = input("请输入开始季度 (YYYYQx, 如 2024Q1): ")
    if "end_q" in param_definitions:
        params["end_q"] = input("请输入结束季度 (YYYYQx, 如 2024Q2): ")
    if "enddate" in param_definitions:
        params["enddate"] = input("请输入截止日期 (YYYYMMDD): ")
    return params


def main():
    # 定义所有可供测试的API列表及其需要的参数
    all_apis_params = {
        # 行情数据
        "get_daily": {"params": ["ts_code", "start_date", "end_date"]},
        "get_index_daily": {"params": ["index_code", "start_date", "end_date"]},
        "get_adj_factor": {"params": ["ts_code", "start_date", "end_date"]},
        "get_daily_basic": {"params": ["ts_code", "start_date", "end_date"]},
        "get_index_dailybasic": {"params": ["index_code", "start_date", "end_date"]},
        # 资金与筹码
        "get_moneyflow": {"params": ["ts_code", "start_date", "end_date"]},
        "get_hk_hold": {"params": ["ts_code", "start_date", "end_date"]},
        "get_moneyflow_hsgt": {"params": ["trade_date"]},
        "get_top_list": {"params": ["trade_date"]},
        "get_top_inst": {"params": ["trade_date"]},
        "get_margin_detail": {"params": ["ts_code", "start_date", "end_date"]},
        "get_block_trade": {"params": ["trade_date"]},
        "get_top10_floatholders": {"params": ["ts_code", "period"]},
        "get_holder_number": {"params": ["ts_code", "enddate"]},
        "get_holder_trade": {"params": ["ts_code", "start_date", "end_date"]},
        # 财务数据
        "get_fina_indicator": {"params": ["ts_code", "period"]},
        "get_income": {"params": ["ts_code", "period"]},
        "get_balancesheet": {"params": ["ts_code", "period"]},
        "get_cashflow": {"params": ["ts_code", "period"]},
        "get_forecast": {"params": ["ts_code", "start_date", "end_date"]},
        "get_express": {"params": ["ts_code", "start_date", "end_date"]},
        "get_dividend": {"params": ["ts_code"]},
        "get_repurchase": {"params": ["ts_code", "start_date", "end_date"]},
        # 宏观经济
        "get_cn_m": {"params": ["start_m", "end_m"]},
        "get_cn_pmi": {"params": ["start_m", "end_m"]},
        "get_cn_cpi": {"params": ["start_m", "end_m"]},
        "get_cn_gdp": {"params": ["start_q", "end_q"]},
        "get_shibor": {"params": ["start_date", "end_date"]},
    }

    # 定义菜单的分组结构
    api_groups = OrderedDict(
        [
            (
                "行情数据 (Market Data)",
                [
                    "get_daily",
                    "get_index_daily",
                    "get_adj_factor",
                    "get_daily_basic",
                    "get_index_dailybasic",
                ],
            ),
            (
                "资金与筹码 (Flow & Holders)",
                [
                    "get_moneyflow",
                    "get_hk_hold",
                    "get_moneyflow_hsgt",
                    "get_top_list",
                    "get_top_inst",
                    "get_margin_detail",
                    "get_block_trade",
                    "get_top10_floatholders",
                    "get_holder_number",
                    "get_holder_trade",
                ],
            ),
            (
                "财务数据 (Financials)",
                [
                    "get_fina_indicator",
                    "get_income",
                    "get_balancesheet",
                    "get_cashflow",
                    "get_forecast",
                    "get_express",
                    "get_dividend",
                    "get_repurchase",
                ],
            ),
            (
                "宏观经济 (Macro)",
                ["get_cn_m", "get_cn_pmi", "get_cn_cpi", "get_cn_gdp", "get_shibor"],
            ),
        ]
    )

    selected_api = display_menu(api_groups)

    if selected_api:
        user_params = get_params_for_api(selected_api, all_apis_params)
        try:
            call_count = int(
                input(f"请输入您想对 {selected_api} 发起的并发请求总数 (例如 1000): ")
            )
            if call_count <= 0:
                raise ValueError
        except ValueError:
            log.error("请输入一个有效的正整数。")
            return

        asyncio.run(run_stress_test_async(selected_api, user_params, call_count))


if __name__ == "__main__":
    main()
    print("\n测试脚本执行完毕。")
