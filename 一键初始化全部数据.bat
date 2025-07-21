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
echo --- [步骤 1/4] 系统健康检查 (run_system_check.py)...
echo    此步骤将检查数据库连接、配置和基础设施是否正常。
python run_system_check.py --mode quick
if %errorlevel% neq 0 (
    echo.
    echo    错误: 系统健康检查失败！请检查 .env 文件中的配置。
    goto:END
)
echo --- [步骤 1/4] 完成 ---
echo.

:STEP_2
echo.
echo --- [步骤 2/4] 初始化数据库与基础数据 (initialize_database.py)...
echo    此步骤将创建所有必要的数据库表，并为核心指数预加载少量历史数据。
python initialize_database.py
if %errorlevel% neq 0 (
    echo.
    echo    错误: 数据库初始化失败！请检查网络连接和Tushare Token。
    goto:END
)
echo --- [步骤 2/4] 完成 ---
echo.

:STEP_3
echo.
echo --- [步骤 3/4] 安全回填基础历史数据 (backfill_data.py)...
echo    此步骤将为全市场股票补全过去一年的历史行情数据。
echo    该过程支持断点续传，如果意外中断，重新运行此脚本即可从断点恢复。
echo    根据网络情况，此步骤可能需要20-60分钟，请耐心等待...
python backfill_data.py --start 20230601 --end 20250720
if %errorlevel% neq 0 (
    echo.
    echo    错误: 历史数据回填失败！请检查网络连接。
    goto:END
)
echo --- [步骤 3/4] 完成 ---
echo.

:STEP_4
echo.
echo --- [步骤 4/4] 计算最新交易日的因子 (run_daily_pipeline.py)...
echo    此步骤将基于已有的历史数据，快速计算所有因子，为平台提供即时可用的数据。
python run_daily_pipeline.py
if %errorlevel% neq 0 (
    echo.
    echo    错误: 每日数据管道执行失败！请检查 Tushare Token 和网络连接。
    goto:END
)
echo --- [步骤 4/4] 完成 ---
echo.

:STEP_5
echo.
echo --- [最终验证] 数据质量检查 (data_validator.py)...
echo    此步骤将验证所有数据的完整性和质量。
python data_validator.py
if %errorlevel% neq 0 (
    echo.
    echo    警告: 数据质量检查发现问题，但不影响基本使用。
)
echo.
echo =================================================================
echo.
echo      🎉 核心服务初始化成功！
echo.
echo      ✅ 数据库已就绪
echo      ✅ 基础数据已加载  
echo      ✅ 历史数据已回填
echo      ✅ 因子数据已计算
echo      ✅ 数据质量已验证
echo.
echo      您现在可以启动平台前端了:
echo      streamlit run app.py
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