import pandas as pd
import pulp
import logging

# 配置基础日志格式
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SingleWarehouseOptimizer:
    def __init__(self, wh_path, assoc_path, D_i, S_i):
        """
        初始化一品一仓综合规划模型 (混合整数线性规划 MILP)
        
        :param wh_path: 仓库信息数据路径
        :param assoc_path: 品类关联度矩阵数据路径
        :param D_i: 字典, 各品类日均存货量 {item: volume}
        :param S_i: 字典, 各品类日均销量 {item: volume}
        """
        self.wh_path = wh_path
        self.assoc_path = assoc_path
        self.D_i = D_i  
        self.S_i = S_i  
        
        self.warehouses = pd.DataFrame()
        self.association_matrix = pd.DataFrame()
        
        # 多目标规划的综合评价权重
        self.w1 = 0.35  # 仓容利用率
        self.w2 = 0.30  # 产能利用率
        self.w3 = 0.20  # 总仓租成本
        self.w4 = 0.15  # 品类关联度

    def load_data(self):
        """加载外部参数文件"""
        try:
            self.warehouses = pd.read_excel(self.wh_path, index_col=0) 
            self.association_matrix = pd.read_excel(self.assoc_path, index_col=0)
            logging.info("运筹学环境参数加载完毕")
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            raise

    def _build_base_model(self, model_name, sense=pulp.LpMaximize):
        """
        构建包含所有基础物理约束的通用模型骨架
        """
        model = pulp.LpProblem(model_name, sense)
        items = list(self.D_i.keys())
        whs = self.warehouses.index.tolist() if not self.warehouses.empty else []

        # 1. 决策变量定义
        # X_ij: 品类 i 是否分配到仓库 j (0-1变量)
        X = pulp.LpVariable.dicts("X", (items, whs), cat='Binary')
        # Y_j: 仓库 j 是否开启 (0-1变量)
        Y = pulp.LpVariable.dicts("Y", whs, cat='Binary')
        # Z_ikj: 辅助变量，用于线性化非线性乘积项 X_ij * X_kj
        Z = pulp.LpVariable.dicts("Z", (items, items, whs), cat='Binary')

        # 2. 核心逻辑约束
        for i in items:
            # 空间约束: 一品一仓
            model += pulp.lpSum([X[i][j] for j in whs]) == 1
            # 逻辑约束: 货物只能放入已开启的仓库
            for j in whs:
                model += X[i][j] <= Y[j]

        # 3. 关联度二次项的线性化约束 (Z_ikj = X_ij * X_kj)
        for i in items:
            for k in items:
                # 仅对存在关联度的品类对建立约束，降低求解维度
                if i < k and (i in self.association_matrix.index and k in self.association_matrix.columns) and self.association_matrix.loc[i, k] > 0:
                    for j in whs:
                        model += Z[i][k][j] <= X[i][j]
                        model += Z[i][k][j] <= X[k][j]
                        model += Z[i][k][j] >= X[i][j] + X[k][j] - 1

        # 4. 容量与产能的上下限约束
        for j in whs:
            C_j = self.warehouses.loc[j, '仓容上限']
            O_j = self.warehouses.loc[j, '产能上限']
            
            stock_j = pulp.lpSum([X[i][j] * self.D_i[i] for i in items])
            model += stock_j >= 0.75 * C_j * Y[j]
            model += stock_j <= 0.90 * C_j * Y[j]
            
            sales_j = pulp.lpSum([X[i][j] * self.S_i[i] for i in items])
            model += sales_j >= 0.70 * O_j * Y[j]
            model += sales_j <= 0.85 * O_j * Y[j]

        # 5. 指标表达式构建
        num_whs = len(whs) if whs else 1
        T1_expr = pulp.lpSum([X[i][j] * self.D_i[i] / self.warehouses.loc[j, '仓容上限'] for i in items for j in whs]) / num_whs
        T2_expr = pulp.lpSum([X[i][j] * self.S_i[i] / self.warehouses.loc[j, '产能上限'] for i in items for j in whs]) / num_whs
        T3_expr = pulp.lpSum([self.warehouses.loc[j, '仓租日成本'] * Y[j] for j in whs])
        
        T4_terms = []
        for i in items:
            for k in items:
                if i < k and (i in self.association_matrix.index and k in self.association_matrix.columns) and self.association_matrix.loc[i, k] > 0:
                    for j in whs:
                        T4_terms.append(self.association_matrix.loc[i, k] * Z[i][k][j])
        T4_expr = pulp.lpSum(T4_terms)

        return model, X, Y, T1_expr, T2_expr, T3_expr, T4_expr

    def get_t3_bounds(self):
        """获取 T3 (总仓租成本) 的理想极小值"""
        logging.info("构建初步模型，求解 T3 的极小值...")
        model, _, _, _, _, T3_expr, T4_expr = self._build_base_model("Min_T3", pulp.LpMinimize)
        model += T3_expr 
        
        model.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=120))
        if model.status == pulp.LpStatusOptimal:
            return pulp.value(T3_expr), pulp.value(T4_expr)
        else:
            raise ValueError("T3 最小化模型无可行解，建议检查产能或仓容阈值。")

    def get_t4_bounds(self):
        """获取 T4 (品类关联度) 的理想极大值"""
        logging.info("构建初步模型，求解 T4 的极大值...")
        model, _, _, _, _, T3_expr, T4_expr = self._build_base_model("Max_T4", pulp.LpMaximize)
        model += T4_expr 
        
        model.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=120))
        if model.status == pulp.LpStatusOptimal:
            return pulp.value(T4_expr), pulp.value(T3_expr)
        else:
            raise ValueError("T4 最大化模型无可行解。")

    def solve_comprehensive(self, t3_min, t3_max, t4_min, t4_max):
        """求解归一化后的综合目标最优解"""
        logging.info("构建并求解单目标综合规划模型...")
        model, X, _, T1_expr, T2_expr, T3_expr, T4_expr = self._build_base_model("Comprehensive_Opt", pulp.LpMaximize)
        
        # 极差归一化处理 (分母加入微小量防止除零)
        norm_T3 = (T3_expr - t3_min) / (t3_max - t3_min + 1e-6)
        norm_T4 = (T4_expr - t4_min) / (t4_max - t4_min + 1e-6)
        
        # 构建最终目标函数 Z
        Z_expr = self.w1 * T1_expr + self.w2 * T2_expr - self.w3 * norm_T3 + self.w4 * norm_T4
        model += Z_expr
        
        model.solve(pulp.PULP_CBC_CMD(msg=True, maxSeconds=300))
        
        if model.status in [pulp.LpStatusOptimal, pulp.LpStatusNotOptimal]: 
            logging.info(f"求解完成，综合评分 Z: {pulp.value(Z_expr):.4f}")
            return self._format_results(X)
        else:
            logging.warning("求解器未能找到满足条件的最终分仓方案。")
            return None

    def _format_results(self, X_vars):
        """提取并格式化决策变量的分配结果"""
        results = []
        for i in self.D_i.keys():
            for j in self.warehouses.index:
                if pulp.value(X_vars[i][j]) == 1.0:
                    results.append({'Item_ID': i, 'Warehouse_ID': j})
        
        return pd.DataFrame(results)

    def run(self):
        """执行优化流水线"""
        self.load_data()
        if self.warehouses.empty or not self.D_i:
            logging.error("运行中断：输入数据为空。")
            return None
            
        try:
            t3_min, t4_min_prime = self.get_t3_bounds()
            t4_max, t3_max_prime = self.get_t4_bounds()
            final_plan_df = self.solve_comprehensive(t3_min, t3_max_prime, t4_min_prime, t4_max)
            return final_plan_df
        except Exception as e:
            logging.error(f"优化过程异常中断: {e}")
            return None

if __name__ == "__main__":
    # 测试参数配置
    test_D_i = {'Category_1': 50, 'Category_2': 80} 
    test_S_i = {'Category_1': 20, 'Category_2': 35} 
    
    optimizer = SingleWarehouseOptimizer(
        wh_path="data/warehouse_info.xlsx",
        assoc_path="data/association.xlsx",
        D_i=test_D_i,
        S_i=test_S_i
    )
    # result_df = optimizer.run()
