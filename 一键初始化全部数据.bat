@echo off
rem 切换代码页到UTF-8以正确显示中文字符
chcp 65001 > nul

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
echo    重要提示: 如需回填历史数据, 请手动编辑此 .bat 文件,
echo    移除下一行行首的 'rem' 并设置您想要的起止日期。
echo.
python backfill_data.py --start 20220101 --end 20250719
echo --- [步骤 2/4] 数据回填任务已启动 ---
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
echo.
echo 脚本执行完毕或遇到错误，窗口已暂停，请查看以上输出信息。
pause