# A股智能量化投研平台 (A-Share Intelligent Quant Platform)

本项目旨在构建一个高性能、可扩展、智能化的A股量化投研平台，实现从数据处理、策略研发、风险控制到决策支持的全流程自动化与可视化。

## 系统架构

平台采用现代化的分层解耦设计，确保了高内聚、低耦合的模块化结构，便于独立开发、测试与升级。

- **表现层 (Presentation Layer)**: `app.py` - 基于 Streamlit 构建的交互式前端工作台。
- **策略与回测层 (Strategy & Backtesting Layer)**: `quant_engine.py` - 包含策略逻辑、组合优化、回测引擎和业绩归因的核心模块。
- **分析与计算层 (Analytics & Computation Layer)**:
    - `quant_engine.py` (FactorFactory): 负责处理结构化数据，生成传统量化因子。
    - `intelligence.py`: 混合AI引擎，负责处理非结构化数据，执行智能体工作流，生成语义因子和分析结论。
- **数据层 (Data Layer)**: `data.py` - 智能数据中心，负责从 Tushare 采集并存储结构化与非结构化数据，内置缓存与PIT校正机制。
- **调度与任务层 (Scheduler & Tasks)**:
    - `scheduler.py`: 基于 APScheduler 的后台定时任务调度器。
    - `initialize_database.py`, `backfill_data.py`: 数据初始化与回填脚本。

## 技术栈

- **核心语言**: Python 3.9+
- **数据处理**: Pandas, NumPy
- **前端框架**: Streamlit
- **数据接口**: Tushare
- **数据库**: SQLite (本地缓存)
- **AI模型接口**: Requests (对接各类大模型API)
- **任务调度**: APScheduler
- **组合优化**: CVXPY, SciPy
- **机器学习**: Scikit-learn, LightGBM
- **可视化**: Plotly

## V1.0 核心功能

V1.0 版本奠定了平台的坚实基础，实现了以下核心功能：

- **✅ 智能数据中心**:
    - 全自动的Tushare数据接口对接与本地SQLite缓存。
    - 基于`asyncio`和`aiohttp`的异步并发下载引擎，极大提升数据获取效率。
    - **Point-in-Time (PIT)** 财务数据校正，从根本上杜绝未来函数。

- **✅ 模块化因子库**:
    - 内置估值、成长、质量、动量、资金流等多维度、标准化的因子计算函数。
    - 包含去极值、标准化、行业中性化处理的科学化因子预处理流水线。

- **✅ 高级策略与回测框架**:
    - 同时支持**向量化回测**（速度快）与**事件驱动回测**（精度高）的双引擎模式。
    - 集成**组合优化器**，支持最大夏普比率等优化目标，并可设置个股、行业权重限制。
    - 基于 Brinson 模型的深度**业绩归因分析**。

- **✅ 混合AI智能体**:
    - 可配置、成本分级的多模型调用框架（fast, medium, powerful）。
    - 可执行多步骤递进式分析工作流，自动生成包含技术、资金、财务、筹码、宏观、舆情六大维度的**AI综合投研报告**。

- **✅ 一体化投研工作台**:
    - 基于 Streamlit 的全功能前端，提供K线、资金、财务、因子分析、策略回测、系统监控等交互界面。

## V2.0 升级路线图

V2.0 的目标是在 V1.0 的基础上，建立更完整、更强大的“宏观-中观-微观”三位一体的投研体系。

- **🆕 宏观分析与择时**:
    - **[数据]** 扩充宏观数据库，新增CPI、PPI、GDP、Shibor等指标。
    - **[引擎]** 开发 `IndexAnalyzer` 模块，用于计算大盘指数的估值百分位（PE/PB Percentile）和波动率水平。
    - **[策略]** 基于大盘分析，生成独立的“择时信号”（如低估机会区、高估风险区），用于指导策略的整体仓位。

- **🆕 中观行业分析与轮动**:
    - **[引擎]** 开发 `IndustryAnalyzer` 模块，实现对各行业的估值、景气度、资金流等因子的横向对比与排名。
    - **[策略]** 构建基于行业分析的**行业轮动策略**，实现“先选好行业，再选好个股”的投资逻辑。

- **🆕 AI能力增强**:
    - **[AI]** 实现**归因式AI报告**，让AI在生成报告时，必须明确指出是哪个/哪些维度的因子共同导出了最终的投资结论。
    - **[AI]** 扩充舆情数据源，并将舆情热度、情感拐点等转化为量化因子，加入模型。

- **🆕 策略与回测丰富化**:
    - **[策略]** 将 `MLAlphaStrategy`（基于LGBM打分）和 `AdaptiveAlphaStrategy`（基于IC-IR动态调权）更紧密地集成到回测实验室中。
    - **[因子]** 扩充技术指标库，引入KDJ、Ichimoku Cloud等更复杂的交易信号。

## 安装与配置

1.  **克隆项目**:
    ```bash
    git clone [https://your-repo-url.com/StockScanAI.git](https://your-repo-url.com/StockScanAI.git)
    cd StockScanAI
    ```

2.  **创建虚拟环境并安装依赖**:
    ```bash
    # 使用 Conda
    conda create -n quant python=3.9
    conda activate quant
    
    # 或使用 venv
    # python -m venv venv
    # source venv/bin/activate  # on Windows: venv\Scripts\activate
    
    pip install -r requirements.txt
    ```

3.  **配置环境变量**:
    - 在项目根目录创建一个名为 `.env` 的文件。
    - 参照 `config.py` 中的说明，填入您的个人 `TUSHARE_TOKEN` 和 AI 模型的 `API_KEY` 等信息。
    - `.env` 文件示例:
      ```
      TUSHARE_TOKEN="your_tushare_token_here"
      AI_API_KEY_FAST="your_fast_model_api_key"
      AI_API_KEY_MEDIUM="your_medium_model_api_key"
      AI_API_KEY_POWERFUL="your_powerful_model_api_key"
      # 其他如SMTP邮件配置...
      ```

## 使用说明

1.  **首次初始化数据库**:
    在第一次部署或需要重建数据时，运行此脚本。它将下载A股基本信息并为核心指数预加载历史数据。
    ```bash
    python initialize_database.py
    ```

2.  **启动投研平台前端**:
    ```bash
    streamlit run app.py
    ```

3.  **运行自动化任务调度器** (可选，用于生产环境):
    此服务会作为后台进程，定时执行数据更新和报告生成任务。
    ```bash
    python scheduler.py
    ```

4.  **执行数据回填** (可选):
    当需要补充大段历史数据时，使用此脚本。
    ```bash
    # 示例：回填2022年1月1日至2023年12月31日的全市场日线数据
    python backfill_data.py --start 20220101 --end 20231231
    ```

5.  **运行单元测试**:
    ```bash
    python test_data_pipeline.py
    python test_quant_engine.py
    python test_workflow.py
    # V2.0 新增
    python test_analyzers.py
    ```

## V3.0 展望

随着数据量的持续增长（如存储全市场股票长达3-5年的分钟线数据），当前基于 `SQLite` 的文件数据库最终会遇到性能瓶颈。

未来的 **V3.0** 核心任务之一将是**数据基础设施的专业化升级**：

- **迁移到时序数据库 (Time-Series Database)**: 考虑将数据后端迁移到专为时间戳数据优化的数据库，如 `TimescaleDB` (基于PostgreSQL) 或 `InfluxDB`，以获得数量级的查询性能提升。
- **完善数据ETL流程**: 基于 `backfill_data.py` 脚本，建立更稳健的数据清洗、转换和加载（ETL）流程，确保海量数据的准确性与一致性。



### Docker虚拟化步骤:



**第一步：启动整个平台** 在项目根目录（包含 `docker-compose.yaml` 的地方）打开命令行，执行以下命令：

Bash

```
docker-compose up --build
```

- `--build` 参数会告诉Docker Compose在启动前，先根据 `Dockerfile` 构建您应用服务的新镜像。
- 看到Docker开始下载基础镜像、安装依赖、复制文件，最后启动数据库和应用两个容器。

**第二步：访问您的量化平台** 等待命令行输出类似 `quant_app_streamlit |   You can now view your Streamlit app in your browser.` 的日志后，打开您的浏览器，访问： `http://localhost:8501`

将看到熟悉的Streamlit界面，但它现在已经完全运行在隔离的Docker容器中了。

**第三步：停止平台** 在您之前运行命令的命令行窗口中，按下 `Ctrl + C`，Docker Compose会优雅地停止所有服务。

**日常开发** 在后续的开发中，如果您修改了Python代码，由于我们设置了卷挂载，您**只需刷新浏览器**即可看到更改。如果修改了 `requirements.txt`，则需要重新构建镜像并启动：`docker-compose up --build`。

至此，项目已经成功地实现了Docker虚拟化。



## 许可证

本项目采用 [MIT License](LICENSE) 许可证。