import pandas as pd
import numpy as np
import warnings
import logging
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression

# 屏蔽时间序列寻优过程中的非收敛警告
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DemandForecaster:
    def __init__(self, sales_path_curr, sales_path_prev, stock_path_curr):
        """
        初始化需求预测模块
        
        :param sales_path_curr: 当年第二季度销量数据路径
        :param sales_path_prev: 去年第三季度销量数据路径
        :param stock_path_curr: 当年上半年月度库存数据路径
        """
        self.sales_path_curr = sales_path_curr
        self.sales_path_prev = sales_path_prev
        self.stock_path_curr = stock_path_curr
        
        self.sales_curr = pd.DataFrame()
        self.sales_prev = pd.DataFrame()
        self.stock_curr = pd.DataFrame()
        self.categories = []

    def load_data(self):
        """加载销量与库存的历史数据并统一时间索引格式"""
        try:
            self.sales_curr = pd.read_excel(self.sales_path_curr, index_col=0)
            self.sales_prev = pd.read_excel(self.sales_path_prev, index_col=0)
            self.stock_curr = pd.read_excel(self.stock_path_curr, index_col=0)
            
            self.sales_curr.index = pd.to_datetime(self.sales_curr.index)
            self.sales_prev.index = pd.to_datetime(self.sales_prev.index)
            self.stock_curr.index = pd.to_datetime(self.stock_curr.index)
            
            self.categories = self.sales_curr.columns.tolist()
            logging.info(f"成功加载数据，共包含 {len(self.categories)} 个品类")
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            # 若文件不存在，初始化基础品类列表用于逻辑演示
            self.categories = ['Category_1', 'Category_2']

    # ================= 销量预测模型库 =================

    def _apply_arima(self, series, steps=92):
        """基础自回归移动平均模型预测"""
        try:
            model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False).forecast(steps=steps).values
        except:
            return np.zeros(steps)

    def _apply_sarimax(self, series, steps=92):
        """包含周期性因子的季节性自回归模型预测"""
        try:
            # 设定 7 天为周期的季节性参数
            model = SARIMAX(series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7), 
                            enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False).forecast(steps=steps).values
        except:
            return np.zeros(steps)

    def _apply_dynamic_factor(self, series, steps=92):
        """基于状态空间的动态因子模型预测"""
        try:
            model = DynamicFactor(series, k_factors=1, factor_order=1, enforce_stationarity=False)
            return model.fit(disp=False).forecast(steps=steps).values
        except:
            return np.zeros(steps)

    # ================= 预测评估与择优体系 =================

    def calculate_evaluation_index(self, forecast, historical, max_std):
        """
        计算预测序列的综合评价分值 T
        
        该指标权衡了预测序列的平滑度与历史同期的相关性：
        T = 相关系数 - (一阶差分标准差 / 历史最大标准差)
        """
        # 计算序列平滑度 (一阶差分的标准差)
        diff_seq = np.diff(forecast)
        std_val = np.std(diff_seq) if len(diff_seq) > 0 else float('inf')
        
        # 计算与去年同期的皮尔逊相关系数
        overlap_len = min(len(forecast), len(historical))
        if overlap_len < 2:
            return -1e6
            
        r_coeff, p_val = pearsonr(forecast[:overlap_len], historical[:overlap_len])
        
        # 若统计学显著性不足 (p > 0.05)，则视为相关性极低，进行惩罚处理
        if p_val > 0.05 or np.isnan(r_coeff):
            r_coeff = -1.0
            
        # 归一化评分计算
        score = r_coeff - (std_val / max_std if max_std > 0 else 0)
        return score

    def get_optimal_sales_forecast(self, steps=92):
        """对各品类并行执行多模型预测，并依据评价指标自动选择最优方案"""
        results = {}
        idx = pd.date_range(start='2023-07-01', periods=steps)
        
        for cat in self.categories:
            if self.sales_curr.empty or cat not in self.sales_curr:
                results[cat] = np.zeros(steps)
                continue
                
            y_train = self.sales_curr[cat].dropna().astype(float)
            y_prev = self.sales_prev[cat].dropna().astype(float) if cat in self.sales_prev else np.zeros(steps)
            
            if len(y_train) < 7:
                results[cat] = np.zeros(steps)
                continue

            # 生成候选模型预测结果
            candidates = {
                'ARIMA': self._apply_arima(y_train, steps),
                'SARIMAX': self._apply_sarimax(y_train, steps),
                'DFM': self._apply_dynamic_factor(y_train, steps)
            }
            
            # 计算归一化基准 (各模型预测序列差分标准差的最大值)
            stds = [np.std(np.diff(p)) for p in candidates.values() if len(np.diff(p)) > 0]
            max_std_ref = max(stds) if stds else 1.0
            
            # 模型择优
            best_model_name = None
            highest_score = float('-inf')
            
            for name, pred_seq in candidates.items():
                t_val = self.calculate_evaluation_index(pred_seq, y_prev.values, max_std_ref)
                if t_val > highest_score:
                    highest_score = t_val
                    best_model_name = name
            
            # 修正负值预测并取整
            results[cat] = np.maximum(0, candidates[best_model_name]).round().astype(int)
            
        return pd.DataFrame(results, index=idx)

    # ================= 库存水平线性预测 =================

    def get_interpolated_stock(self, months=3):
        """利用线性回归预测月度均值，并通过线性插值转化为日度货量数据"""
        daily_output = {}
        future_months = pd.to_datetime(['2023-07-01', '2023-08-01', '2023-09-01'])
        
        for cat in self.categories:
            if self.stock_curr.empty or cat not in self.stock_curr:
                daily_output[cat] = np.zeros(92)
                continue
                
            y_stock = self.stock_curr[cat].dropna()
            
            # 建立基于时间序列索引的线性回归模型
            reg = LinearRegression()
            x_time = np.arange(len(y_stock)).reshape(-1, 1)
            
            if len(x_time) >= 2:
                reg.fit(x_time, y_stock.values)
                x_future = np.arange(len(y_stock), len(y_stock) + months).reshape(-1, 1)
                monthly_preds = reg.predict(x_future)
            else:
                monthly_preds = np.array([0, 0, 0])
                
            monthly_preds = np.maximum(0, monthly_preds)
            
            # 构建月度时间序列并进行日度重采样线性插值
            m_series = pd.Series(monthly_preds, index=future_months)
            # 补充边界点以确保 9 月份插值的完整性
            m_series.loc[pd.to_datetime('2023-10-01')] = m_series.loc['2023-09-01']
            
            d_series = m_series.resample('D').interpolate(method='time')
            daily_output[cat] = d_series.loc['2023-07-01':'2023-09-30'].round().astype(int).values

        idx = pd.date_range(start='2023-07-01', periods=92)
        return pd.DataFrame(daily_output, index=idx)

    def run(self):
        """运行完整预测流水线并返回日均统计结果数据"""
        self.load_data()
        
        logging.info("启动销量集成预测...")
        df_sales = self.get_optimal_sales_forecast()
        
        logging.info("启动库存线性插值预测...")
        df_stock = self.get_interpolated_stock()
        
        if not df_sales.empty and not df_stock.empty:
            # 计算各品类日均销量 (Si) 与日均存货量 (Di)
            si_dict = df_sales.mean().to_dict()
            di_dict = df_stock.mean().to_dict()
            logging.info("预测闭环计算完成")
            return di_dict, si_dict
        return {}, {}

if __name__ == "__main__":
    forecaster = DemandForecaster(
        sales_path_curr="data/sales_2023_q2.xlsx",
        sales_path_prev="data/sales_2022_q3.xlsx",
        stock_path_curr="data/stock_2023_h1.xlsx"
    )
    # di, si = forecaster.run()
