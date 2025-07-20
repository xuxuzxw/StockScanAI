@echo off
rem 切换代码页到UTF-8以正确显示中文字符
chcp 65001 > nul

rem 设置窗口标题
title A股量化平台 - 核心服务初始化

echo =================================================================
echo.
echo      欢迎使用 A股智能量化投研平台 - 核心服务初始化向导
echo.
echo =================================================================
echo.
echo 本向导将帮助您完成数据库的创建和最新数据的初始化。
echo 整个过程预计需要 5-15 分钟，具体取决于您的网络状况。
echo.
echo 按任意键开始执行...
pause > nul

:STEP_1
echo.
echo --- [步骤 1/3] 初始化数据库与基础数据 (initialize_database.py)...
echo    此步骤将创建所有必要的数据库表，并为核心指数预加载少量历史数据。
python initialize_database.py
if %errorlevel% neq 0 (
    echo.
    echo    错误: 数据库初始化失败！请检查 .env 文件中的数据库配置和网络连接。
    goto:END
)
echo --- [步骤 1/3] 完成 ---
echo.

:STEP_2
echo.
echo --- [步骤 2/3] 运行数据管道，计算最新交易日的因子 (run_daily_pipeline.py)...
echo    此步骤将下载全市场最新的行情，并计算所有因子，为平台提供即时可用的数据。
echo    由于涉及全市场数据，此步骤可能需要较长时间，请耐心等待...
python run_daily_pipeline.py
if %errorlevel% neq 0 (
    echo.
    echo    错误: 每日数据管道执行失败！请检查 Tushare Token 和网络连接。
    goto:END
)
echo --- [步骤 2/3] 完成 ---
echo.

:STEP_3
echo.
echo =================================================================
echo.
echo      核心服务初始化成功！
echo.
echo      您现在已经可以启动平台前端 `streamlit run app.py` 进行使用了。
echo.
echo =================================================================
echo.
echo.
echo --- [可选操作指南] 关于历史数据回填 ---
echo.
echo    为了支持长周期的策略回测，您可能需要为全市场股票补全多年的历史数据。
echo    这是一个非常耗时的操作，我们强烈建议您独立、手动执行。
echo.
echo    操作方法:
echo    1. 打开一个新的命令行窗口。
echo    2. 根据您的需要，执行类似下面的命令:
echo.
echo       rem (示例: 回填过去3年的全市场日线和复权因子数据)
echo       python backfill_data.py --start 20220101 --end 20241231
echo.
echo    您可以随时中断和重启回填过程，系统会自动断点续传。
echo.

:END
echo 脚本执行完毕，窗口已暂停，请查看以上输出信息。
pause