# quant_project/logger_config.py
#
# 实现了“功能需求规格说明书”中的 5.2 日志与监控。
# 配置一个全局的、可分级的日志系统。

import logging
import os
import sys


def setup_logger():
    """
    配置并返回一个全局 logger。
    - INFO 及以上级别的日志会输出到控制台。
    - DEBUG 及以上级别的日志会写入到文件 `__pycache__/quant_project.log`。
    """
    # 获取根 logger
    logger = logging.getLogger("QuantProject")
    logger.setLevel(logging.DEBUG)  # 设置 logger 的最低处理级别

    # 防止重复添加 handler
    if logger.hasHandlers():
        return logger

    # --- 控制台 Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上信息
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # --- 文件 Handler ---
    # 【V2.4 优化】将日志文件存储到 __pycache__ 目录中
    log_dir = "__pycache__"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "quant_project.log")

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有 DEBUG 及以上信息
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 将 handler 添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 创建一个全局可用的 logger 实例
log = setup_logger()

if __name__ == "__main__":
    log.debug("这是一条 DEBUG 信息，只会出现在日志文件中。")
    log.info("这是一条 INFO 信息，会出现在控制台和文件中。")
    log.warning("这是一条 WARNING 警告信息。")
    log.error("这是一条 ERROR 错误信息。")
    log.critical("这是一条 CRITICAL 严重错误信息。")
