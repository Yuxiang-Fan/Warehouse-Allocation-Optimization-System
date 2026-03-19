import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression

# 忽略时间序列模型在寻优时常见的收敛警告
warnings.filterwarnings("ignore")

class Problem1_Forecaster:
    def __init__(self, sales_23_path, sales_22_path, stock_23_path):
        """
        初始化问题一的预测流水线
        :param sales_23_path: 今年4-6月销量数据路径 (用于训练)
        :param sales_22_path: 去年7-9月销量数据路径 (用于皮尔逊相关系数检验)
        :param stock_23_path: 今年1-6月库存数据路径 (用于线性回归训练)
        """
        self.sales_23_path = sales_23_path
        self.sales_22_path = sales_22_path
        self.stock_23_path = stock_23_path
        
        self.sales_23 = pd.DataFrame()
        self.sales_22 = pd.DataFrame()
        self.stock_23 = pd.DataFrame()
        self.categories = []

    def load_data(self):
        """
        【严格占位】加载真实外部数据
        要求输入格式：第一列为时间索引，后续列为各类商品的品类编号
        """
        try:
            self.sales_23 = pd.read_excel(self.sales_23_path, index_col=0)
            self.sales_22 = pd.read_excel(self.sales_22_path, index_col=0)
            self.stock_23 = pd.read_excel(self.stock_23_path, index_col=0)
            
            # 统一时间索引格式
            self.sales_23.index = pd.to_datetime(self.sales_23.index)
            self.sales_22.index = pd.to_datetime(self.sales_22.index)
            self.stock_23.index = pd.to_datetime(self.stock_23.index)
            
            self.categories = self.sales_23.columns.tolist()
            print(f"数据加载成功，共识别出 {len(self.categories)} 个品类。")
        except Exception as e:
            print(f"数据加载提示 (占位运行状态): {e}")
            # 占位结构声明，保证代码在未挂载真实文件时也能展示逻辑接口
            self.categories = ['Category_1', 'Category_2']

    # ================= 销量预测：三大候选模型 =================

    def _model_arima(self, series, steps=92):
        """ARIMA模型预测"""
        try:
            model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False).forecast(steps=steps).values
        except:
            return np.zeros(steps)

    def _model_spatial_state(self, series, steps=92):
        """空间状态模型预测 (带有7天季节性)"""
        try:
            model = SARIMAX(series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7), 
                            enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False).forecast(steps=steps).values
        except:
            return np.zeros(steps)

    def _model_dynamic_factor(self, series, steps=92):
        """动态因子模型预测 (单序列降维处理)"""
        try:
            model = DynamicFactor(series, k_factors=1, factor_order=1, enforce_stationarity=False)
            return model.fit(disp=False).forecast(steps=steps).values
        except:
            return np.zeros(steps)

    # ================= T值评估体系 (完全复刻论文逻辑) =================

    def calculate_T_score(self, forecast_series, last_year_series, max_std_diff):
        """
        计算综合评估指标 T
        T = (一阶导数标准差 / max_std_diff) - 皮尔逊相关系数r
        包含了 p-value 的置信度惩罚逻辑
        """
        # 1. 计算一阶导数标准差
        diff = np.diff(forecast_series)
        std_diff = np.std(diff) if len(diff) > 0 else float('inf')
        
        # 2. 计算皮尔逊相关系数
        min_len = min(len(forecast_series), len(last_year_series))
        if min_len < 2:
            return float('-inf') # 数据不足，直接淘汰
            
        r, p_value = pearsonr(forecast_series[:min_len], last_year_series[:min_len])
        
        # 处理皮尔逊系数不可信 (p > 0.05) 的惩罚情况
        if p_value > 0.05 or np.isnan(r):
            r = -1.0 # 极差的惩罚值
            
        # 3. 计算最终 T 值
        # 注意：标准差越小曲线越平滑，相关系数越大越相关
        T_score = r - (std_diff / max_std_diff if max_std_diff > 0 else 0)
        
        return T_score

    def predict_and_select_sales(self, steps=92):
        """执行三大模型预测，并根据 T 值自动择优拼装"""
        final_forecasts = {}
        
        forecast_index = pd.date_range(start='2023-07-01', periods=steps)
        
        for cat in self.categories:
            if self.sales_23.empty or cat not in self.sales_23:
                # 占位保护
                final_forecasts[cat] = np.zeros(steps)
                continue
                
            train_series = self.sales_23[cat].dropna().astype(float)
            last_year_series = self.sales_22[cat].dropna().astype(float) if cat in self.sales_22 else np.zeros(steps)
            
            if len(train_series) < 7:
                final_forecasts[cat] = np.zeros(steps)
                continue

            # 1. 获取三个模型的预测结果
            preds = {
                'ARIMA': self._model_arima(train_series, steps),
                'Spatial': self._model_spatial_state(train_series, steps),
                'DynamicFactor': self._model_dynamic_factor(train_series, steps)
            }
            
            # 2. 获取最大一阶导数标准差 (用于归一化)
            stds = {name: np.std(np.diff(pred)) for name, pred in preds.items() if len(np.diff(pred)) > 0}
            max_std = max(stds.values()) if stds else 1.0
            
            # 3. 计算每个模型的 T 值并择优
            best_model = None
            max_T = float('-inf')
            
            for name, pred in preds.items():
                t_score = self.calculate_T_score(pred, last_year_series.values, max_std)
                if t_score > max_T:
                    max_T = t_score
                    best_model = name
                    
            # 4. 保存胜出模型的预测结果，并保证不出现负数销量
            final_pred = np.maximum(0, preds[best_model]).round().astype(int)
            final_forecasts[cat] = final_pred
            # 可以在此处加入 print 观察每个品类最终选了哪个模型，以印证论文中“动态因子占比74.8%”的结论
            
        return pd.DataFrame(final_forecasts, index=forecast_index)

    # ================= 货量预测：线性回归与插值 =================

    def predict_and_interpolate_stock(self, steps_months=3):
        """使用线性回归预测未来三个月月均货量，并运用Pandas进行时间序列日度插值"""
        daily_stock = {}
        
        # 设定预测的未来三个月的时间节点
        future_months = pd.to_datetime(['2023-07-01', '2023-08-01', '2023-09-01'])
        
        for cat in self.categories:
            if self.stock_23.empty or cat not in self.stock_23:
                # 占位保护
                daily_stock[cat] = np.zeros(92)
                continue
                
            series = self.stock_23[cat].dropna()
            
            # 线性回归训练
            X_train = np.arange(len(series)).reshape(-1, 1)
            y_train = series.values
            model = LinearRegression()
            
            if len(X_train) >= 2:
                model.fit(X_train, y_train)
                X_test = np.arange(len(series), len(series) + steps_months).reshape(-1, 1)
                pred_monthly = model.predict(X_test)
            else:
                pred_monthly = np.array([0, 0, 0])
                
            pred_monthly = np.maximum(0, pred_monthly)
            
            # 构造月度 DataFrame，并在10月1日补齐数据以保证9月插值不越界
            monthly_df = pd.Series(pred_monthly, index=future_months)
            monthly_df.loc[pd.to_datetime('2023-10-01')] = monthly_df.loc['2023-09-01']
            
            # 重采样至日度，并进行线性插值 (完美契合“日存货量在一个月内呈线性变化”的假设)
            daily_interpolated = monthly_df.resample('D').interpolate(method='time')
            # 截取7-9月数据并取整
            daily_stock[cat] = daily_interpolated.loc['2023-07-01':'2023-09-30'].round().astype(int).values

        forecast_index = pd.date_range(start='2023-07-01', periods=92)
        return pd.DataFrame(daily_stock, index=forecast_index)

    # ================= 主控制流 =================

    def run_problem1_pipeline(self):
        """执行完整的问题一流水线，输出最终结果字典供问题二、三调用"""
        self.load_data()
        
        print("正在执行多模型销量预测及 T 值评估择优...")
        sales_df = self.predict_and_select_sales()
        
        print("正在执行货量线性回归预测及日度插值...")
        stock_df = self.predict_and_interpolate_stock()
        
        # 计算7-9月的日均值，作为问题二和问题三的输入参数 Di 和 Si
        if not sales_df.empty and not stock_df.empty:
            S_i = sales_df.mean().to_dict()
            D_i = stock_df.mean().to_dict()
            print("问题一闭环计算完成，已生成标准接口数据。")
            return D_i, S_i
        else:
            print("由于未接入真实文件，占位运行结束。")
            return {}, {}

if __name__ == "__main__":
    # 使用时，传入实际的相对/绝对路径即可无缝运行
    p1_solver = Problem1_Forecaster(
        sales_23_path="data/今年456综合最佳.xlsx",
        sales_22_path="data/去年789最佳.xlsx",
        stock_23_path="data/1-6月库存量.xlsx"
    )
    # D_i, S_i = p1_solver.run_problem1_pipeline()