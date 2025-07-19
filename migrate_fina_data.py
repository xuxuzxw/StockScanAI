# quant_project/migrate_fina_data.py
#
# 【V2.2 核心迁移脚本】
# 目的：将旧的、分散的财务指标数据（fina_indicator_[股票代码]）
#      合并并迁移到新的统一主表（financial_indicators）中。
# 这是一个一次性的操作，用于升级数据库架构。

import pandas as pd
from sqlalchemy import create_engine, inspect, text
import config
from logger_config import log

def migrate_financial_indicators():
    """
    执行财务指标数据的迁移工作流。
    """
    log.info("===== 开始执行财务指标数据迁移任务 =====")
    
    try:
        engine = create_engine(config.DATABASE_URL)
        inspector = inspect(engine)
        log.info(f"数据库已连接: {config.DATABASE_URL}")
    except Exception as e:
        log.critical("数据库连接失败！迁移终止。", exc_info=True)
        return

    # 1. 获取所有表名
    all_table_names = inspector.get_table_names()
    
    # 2. 筛选出所有旧的 fina_indicator_... 表
    old_fina_tables = [t for t in all_table_names if t.startswith('fina_indicator_')]
    
    if not old_fina_tables:
        log.warning("未找到任何旧的 'fina_indicator_' 表，无需迁移。")
        log.info("===== 数据迁移任务完成 =====")
        return

    log.info(f"检测到 {len(old_fina_tables)} 张旧的财务指标表，准备开始迁移...")
    
    # 3. 循环读取、合并数据
    all_fina_dataframes = []
    for i, table_name in enumerate(old_fina_tables):
        try:
            with engine.connect() as connection:
                df = pd.read_sql_table(table_name, connection)
                all_fina_dataframes.append(df)
            
            if (i + 1) % 100 == 0:
                log.info(f"  已读取 {i+1}/{len(old_fina_tables)} 张旧表...")
        except Exception as e:
            log.error(f"读取表 '{table_name}' 失败: {e}", exc_info=True)
            continue
            
    if not all_fina_dataframes:
        log.error("所有旧表均读取失败，无法进行迁移。")
        log.info("===== 数据迁移任务完成 =====")
        return

    log.info("所有旧表数据读取完毕，正在合并为一张大表...")
    consolidated_df = pd.concat(all_fina_dataframes, ignore_index=True)
    log.info(f"数据合并完成，共计 {len(consolidated_df)} 条记录。")

    # 4. 写入新的统一主表
    new_table_name = 'financial_indicators'
    log.info(f"正在将合并后的数据写入新的主表 '{new_table_name}'...")
    
    try:
        # 为了防止列不匹配，只选择新表中明确定义的列
        # 注意：这个列表必须与 data.py 中 financial_indicators 的定义保持同步
        columns_to_keep = [
            'ts_code', 'ann_date', 'end_date', 'roe', 'netprofit_yoy', 
            'debt_to_assets', 'or_yoy'
            # 在 data.py 中 financial_indicators 表增加字段后，这里也需要同步更新
        ]
        
        final_df = consolidated_df[[col for col in columns_to_keep if col in consolidated_df.columns]]

        with engine.connect() as connection:
            with connection.begin():
                # 为确保幂等性，先清空新表，再写入
                connection.execute(text(f"DELETE FROM {new_table_name};"))
                final_df.to_sql(new_table_name, connection, if_exists='append', index=False, chunksize=10000)
        
        log.info(f"成功将 {len(final_df)} 条记录写入 '{new_table_name}'！")
        
    except Exception as e:
        log.critical(f"写入新表 '{new_table_name}' 时发生严重错误！", exc_info=True)
        log.warning("提示：请确保您已在 data.py 中正确创建了 financial_indicators 表。")

    log.info("===== 数据迁移任务完成 =====")


if __name__ == '__main__':
    # 提供一个确认步骤，防止误操作
    print("【警告】此脚本将读取所有旧的 'fina_indicator_...' 表并将其数据写入新的 'financial_indicators' 主表。")
    print("在运行前，请确保：")
    print("1. 您已经将在 data.py 中添加了创建 'financial_indicators' 表的逻辑。")
    print("2. 您已经运行过一次 initialize_database.py 或 app.py，以确保新表已被创建。")
    confirm = input("您确定要开始数据迁移吗？(输入 'yes' 以确认): ")
    if confirm.lower() == 'yes':
        migrate_financial_indicators()
    else:
        print("操作已取消。")
        
    input("\n任务执行完毕，按 Enter 键退出...")