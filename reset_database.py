# quant_project/reset_database.py
#
# 【警告】这是一个破坏性操作脚本。
# 它会连接到服务器数据库，并删除其中的所有表。
# 仅在您确定数据库中没有需要保留的、不可恢复的数据时使用。

from sqlalchemy import create_engine, inspect, text

import config

# 导入您项目中的初始化模块
import initialize_database
from logger_config import log


def reset_and_initialize_database():
    """
    重置并初始化整个数据库。
    此版本适用于服务器型数据库（如 PostgreSQL/TimescaleDB）。
    """
    log.info("===== 开始执行数据库重置任务（服务器模式） =====")

    # 服务器数据库必须通过 DATABASE_URL 连接
    if not hasattr(config, "DATABASE_URL") or not config.DATABASE_URL:
        log.error("在config.py中未找到有效的 DATABASE_URL。无法连接到数据库。")
        return

    try:
        engine = create_engine(config.DATABASE_URL)
        inspector = inspect(engine)

        with engine.connect() as connection:
            with connection.begin():  # 开启事务
                # 1. 获取所有表名
                table_names = inspector.get_table_names()

                if not table_names:
                    log.info("数据库中没有找到任何表，无需删除。")
                else:
                    log.warning(f"即将从数据库中删除以下所有表: {table_names}")
                    # 先关闭外键检查（某些数据库需要），然后删除
                    # 对于PostgreSQL, 使用 CASCADE 即可处理依赖关系
                    for table_name in table_names:
                        connection.execute(
                            text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
                        )
                    log.info("所有旧表已成功删除。")

    except Exception:
        log.error("连接或清空数据库时发生错误", exc_info=True)
        # 如果清空失败，则终止后续操作
        return

    # 2. 调用现有的初始化流程，创建全新的数据库和基础数据
    log.info("正在调用 initialize_all_data() 来创建和填充新数据库...")
    try:
        initialize_database.initialize_all_data()
        log.info("数据库已成功重置并初始化！")
    except Exception:
        log.critical("在重新初始化数据库过程中发生严重错误！", exc_info=True)

    log.info("===== 数据库重置任务完成 =====")


if __name__ == "__main__":
    # 提供一个确认步骤，防止误操作
    confirm = input(
        "您确定要完全清空并重建数据库中的所有表吗？这是一个不可逆的操作！(输入 'yes' 以确认): "
    )
    if confirm.lower() == "yes":
        reset_and_initialize_database()
    else:
        print("操作已取消。")

    # 增加暂停机制，等待用户按键后退出
    input("\n任务执行完毕，按 Enter 键退出...")
