@echo off
rem 设置窗口标题
title A股量化平台 - 数据一键初始化

echo =================================================================
echo.
echo      A GU LIANG HUA PING TAI --- SHU JU YI JIAN CHU SHI HUA
echo.
echo      A股量化平台 - 数据一键初始化脚本
echo.
echo =================================================================
echo.
echo 按任意键开始执行...
pause > nul

:STEP_1
echo.
echo --- [步骤 1/4] 正在初始化数据库与基础数据 (initialize_database.py)...
python initialize_database.py
echo --- [步骤 1/4] 完成 ---
echo.

:STEP_2
echo.
echo --- [步骤 2/4] 正在回填核心历史数据 (backfill_data.py)...
echo.
echo    重要提示：下面的回填命令默认是注释掉的。
echo    您需要手动编辑此 .bat 文件，移除'rem'并设置您想要的起止日期。
echo.
rem python backfill_data.py --start 20220101 --end 20250719
echo --- [步骤 2/4] 完成 (已跳过，如需执行请编辑此文件) ---
echo.

:STEP_3
echo.
echo --- [步骤 3/4] 正在执行统一数据管道 (抽取+计算) (run_daily_pipeline.py)...
python run_daily_pipeline.py
echo --- [步骤 3/4] 完成 ---
echo.


echo =================================================================
echo.
echo      所有数据初始化任务已执行完毕！
echo.
echo =================================================================
echo.
pause