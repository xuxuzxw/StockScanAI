# quant_project/industry_analyzer.py
import pandas as pd
from logger_config import log
import data
import quant_engine as qe

class IndustryAnalyzer:
    """【V2.0新增】行业分析与比较模块"""
    def __init__(self, data_manager: data.DataManager, factor_factory: qe.FactorFactory):
        self.dm = data_manager
        self.ff = factor_factory
        # 缓存股票基本信息，避免重复获取
        self.stock_basics = self.dm.get_stock_basic()

    def get_industry_factor_rank(self, date: str, factor_name: str, ascending: bool = False) -> pd.DataFrame:
        """
        【架构重构版】直接从因子库查询预计算好的因子值，并按行业排名。
        """
        log.info(f"开始从因子库查询 {date} 的行业排名: {factor_name}")
        
        # 1. 构造SQL查询语句
        # 直接从 factors_exposure 表中筛选出指定日期和因子名称的数据
        query = f"""
            SELECT ts_code, factor_value 
            FROM factors_exposure
            WHERE trade_date = '{pd.to_datetime(date).strftime('%Y-%m-%d')}' 
              AND factor_name = '{factor_name}'
        """
        
        try:
            # 2. 执行查询
            with self.dm.engine.connect() as connection:
                factor_df = pd.read_sql(query, connection)
            
            if factor_df.empty:
                log.warning(f"在因子库中未找到 {date} 的 {factor_name} 因子数据。")
                return pd.DataFrame()

            # 3. 合并行业信息
            # self.stock_basics 已经缓存在内存中，这个合并操作非常快
            merged_df = pd.merge(factor_df, self.stock_basics[['ts_code', 'industry']], on='ts_code')
            
            # 4. 分组、聚合、排序
            industry_rank = merged_df.groupby('industry')['factor_value'].mean().sort_values(ascending=ascending)
            
            return pd.DataFrame(industry_rank)
            
        except Exception as e:
            log.error(f"从因子库查询行业排名时出错: {e}", exc_info=True)
            return pd.DataFrame()