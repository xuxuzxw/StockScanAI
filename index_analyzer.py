# StockScanAI/index_analyzer.py
import pandas as pd

import data


class IndexAnalyzer:
    """【V2.0新增】大盘分析器"""

    def __init__(self, data_manager: data.DataManager):
        self.dm = data_manager

    def get_index_valuation_percentile(
        self, index_code: str, date: str, years: int = 5
    ) -> dict:
        """
        计算指定日期指数PE/PB在过去N年历史数据中的百分位。
        :param index_code: 指数代码, e.g., '000300.SH'
        :param date: 计算日期 YYYYMMDD
        :param years: 回看年数
        :return: 包含PE、PB及其百分位的字典
        """
        start_date = (pd.to_datetime(date) - pd.Timedelta(days=365 * years)).strftime(
            "%Y%m%d"
        )
        df_index_basic = self.dm.pro.index_dailybasic(
            ts_code=index_code, start_date=start_date, end_date=date
        )

        if df_index_basic is None or df_index_basic.empty:
            return {"error": "No data found"}

        df_index_basic = df_index_basic.sort_values("trade_date").reset_index(drop=True)
        current_metrics = df_index_basic.iloc[-1]

        pe_percentile = (df_index_basic["pe_ttm"] < current_metrics["pe_ttm"]).mean()
        pb_percentile = (df_index_basic["pb"] < current_metrics["pb"]).mean()

        return {
            "pe_ttm": current_metrics["pe_ttm"],
            "pb": current_metrics["pb"],
            "pe_percentile": pe_percentile,
            "pb_percentile": pb_percentile,
        }

    def generate_timing_signal(self, index_code: str, date: str) -> str:
        """
        基于估值百分位生成一个简单的择时信号。
        """
        val_data = self.get_index_valuation_percentile(index_code, date)
        if "error" in val_data:
            return "未知"

        # 简单规则：PE和PB百分位均低于30%时为机会区，均高于70%时为风险区
        pe_p = val_data["pe_percentile"]
        pb_p = val_data["pb_percentile"]

        if pe_p < 0.3 and pb_p < 0.3:
            return "低估机会区"
        elif pe_p > 0.7 and pb_p > 0.7:
            return "高估风险区"
        else:
            return "估值适中区"
