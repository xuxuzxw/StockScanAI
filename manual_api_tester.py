# quant_project/manual_api_tester.py
#
# A股量化投研平台 - API频率手动测试工具
#
# 使用方法:
# 1. 确保您的 .env 文件已正确配置。
# 2. 直接在命令行运行: python manual_api_tester.py
# 3. 根据提示选择要测试的API并输入参数。
#
# 功能:
# - 允许您针对性地对任何一个Tushare API接口进行压力测试。
# - 直观地观察和验证 data.py 中的频率控制装饰器是否按预期工作。
# - 通过手动输入参数，精准复现和诊断特定的API调用问题。

import time
import pandas as pd

# 导入项目模块
try:
    import config
    import data
    from logger_config import log
except ImportError as e:
    print(f"FATAL: 关键模块导入失败: {e}")
    print("请确保您在项目的根目录下运行此脚本。")
    exit(1)

def display_menu(api_functions):
    """显示API测试菜单"""
    log.info("========== Tushare API 手动频率测试器 ==========")
    log.info("请从以下选择一个您想测试的API函数:")
    for i, func_name in enumerate(api_functions, 1):
        print(f"  {i}. {func_name}")
    log.info("==============================================")
    choice = input("请输入选项编号: ")
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(api_functions):
            return api_functions[choice_idx]
        else:
            log.error("无效的选项。")
            return None
    except ValueError:
        log.error("请输入一个数字。")
        return None

def get_params_for_api(api_name: str) -> dict:
    """根据API名称获取用户输入的参数"""
    params = {}
    log.info(f"--- 正在为 API '{api_name}' 配置参数 ---")
    
    # 通用参数
    if 'ts_code' in all_apis[api_name]['params']:
        params['ts_code'] = input("请输入股票代码 (例如 600519.SH): ")
    if 'start_date' in all_apis[api_name]['params']:
        params['start_date'] = input("请输入开始日期 (YYYYMMDD): ")
    if 'end_date' in all_apis[api_name]['params']:
        params['end_date'] = input("请输入结束日期 (YYYYMMDD): ")
    if 'period' in all_apis[api_name]['params']:
        params['period'] = input("请输入报告期 (例如 20240331): ")
    if 'trade_date' in all_apis[api_name]['params']:
        params['trade_date'] = input("请输入交易日期 (YYYYMMDD): ")

    return params

def run_test_loop(dm: data.DataManager, api_name: str, api_params: dict):
    """执行测试循环"""
    try:
        call_count = int(input("请输入您想在1分钟内模拟调用的总次数 (建议>500来测试通用限制): "))
    except ValueError:
        log.error("请输入有效的数字。")
        return

    log.info(f"\n准备就绪！将在约60秒内以最快速度调用 '{api_name}' 共 {call_count} 次。")
    log.info("请密切观察以下日志输出的时间戳，以验证限速是否生效。")
    input("按 Enter 键开始测试...")

    start_time = time.time()
    for i in range(call_count):
        try:
            # 使用 getattr 动态调用 DataManager 中的方法
            target_method = getattr(dm, api_name)
            
            log.info(f"[调用 {i+1}/{call_count}] -- 时间: {time.strftime('%H:%M:%S')} -- 正在调用 {api_name}...")
            
            # 记录单次调用开始时间
            single_call_start = time.time()
            
            # 执行API调用
            result = target_method(**api_params)
            
            # 计算单次调用耗时
            single_call_duration = time.time() - single_call_start
            
            if result is not None and not result.empty:
                log.info(f"  > [成功] 调用完成，耗时 {single_call_duration:.4f} 秒。获取到 {len(result)} 条数据。")
            elif result is not None:
                log.warning(f"  > [注意] 调用成功，但返回数据为空。耗时 {single_call_duration:.4f} 秒。")
            else:
                log.error(f"  > [失败] 调用失败，返回 None。耗时 {single_call_duration:.4f} 秒。")

        except Exception as e:
            log.error(f"  > [严重错误] 在调用 {api_name} 时发生异常: {e}", exc_info=False)
            
        # 如果总时间超过60秒，可以提前结束
        if time.time() - start_time > 65:
            log.warning("测试时间已超过1分钟，提前终止循环。")
            break
            
    end_time = time.time()
    total_duration = end_time - start_time
    log.info("\n========== 测试完成 ==========")
    log.info(f"总计调用次数: {i+1}")
    log.info(f"总耗时: {total_duration:.2f} 秒")
    if i > 0:
        log.info(f"平均每次调用间隔 (包含限速等待): {total_duration / (i+1):.4f} 秒")
    log.info("============================")


if __name__ == '__main__':
    # 定义可供测试的API列表及其需要的参数
    all_apis = {
        'get_daily': {'params': ['ts_code', 'start_date', 'end_date']},
        'get_adj_factor': {'params': ['ts_code', 'start_date', 'end_date']},
        'get_daily_basic': {'params': ['ts_code', 'start_date', 'end_date']},
        'get_fina_indicator': {'params': ['ts_code']},
        'get_top10_floatholders': {'params': ['ts_code', 'period']},
        'get_holder_number': {'params': ['ts_code']}, # 注意：此接口限速为100次/分钟
        'get_top_list': {'params': ['trade_date']},
    }
    api_function_names = list(all_apis.keys())

    # 主流程
    dm_instance = data.DataManager()
    
    selected_api = display_menu(api_function_names)
    
    if selected_api:
        user_params = get_params_for_api(selected_api)
        run_test_loop(dm_instance, selected_api, user_params)

    print("\n测试脚本执行完毕。")