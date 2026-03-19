import pandas as pd
import pulp

class Problem2_SingleWarehouseOptimizer:
    def __init__(self, warehouse_info_path, association_matrix_path, D_i, S_i):
        """
        初始化问题二：一品一仓单目标综合规划
        :param warehouse_info_path: 附件3 仓库信息占位路径
        :param association_matrix_path: 附件4 关联度信息占位路径
        :param D_i: 问题一输出的字典，格式为 {'品类1': 日均货量, '品类2': 日均货量, ...}
        :param S_i: 问题一输出的字典，格式为 {'品类1': 日均销量, '品类2': 日均销量, ...}
        """
        self.wh_path = warehouse_info_path
        self.assoc_path = association_matrix_path
        self.D_i = D_i  
        self.S_i = S_i  
        
        self.warehouses = pd.DataFrame()
        self.association_matrix = pd.DataFrame()
        
        # 论文中定义的综合指标权重
        self.w1 = 0.35  # T1 仓容利用率权重
        self.w2 = 0.30  # T2 产能利用率权重
        self.w3 = 0.20  # T3 总仓租成本权重
        self.w4 = 0.15  # T4 品类关联度权重

    def load_data(self):
        """【严格占位】读取外部数据，坚决不使用随机数模拟"""
        try:
            # 真实运行需确保表格索引正确设置
            self.warehouses = pd.read_excel(self.wh_path, index_col=0) 
            self.association_matrix = pd.read_excel(self.assoc_path, index_col=0)
            print("运筹学环境参数加载完毕。")
        except Exception as e:
            print(f"数据加载提示 (占位状态): {e}")

    def _build_base_model(self, model_name, sense=pulp.LpMaximize):
        """
        构建包含所有基础物理约束的通用模型骨架
        返回：模型实例、决策变量及 T1/T2/T3/T4 的数学表达式
        """
        model = pulp.LpProblem(model_name, sense)
        items = list(self.D_i.keys())
        whs = self.warehouses.index.tolist() if not self.warehouses.empty else []

        # 1. 定义 0-1 决策变量
        X = pulp.LpVariable.dicts("X", (items, whs), cat='Binary')
        Y = pulp.LpVariable.dicts("Y", whs, cat='Binary')
        Z = pulp.LpVariable.dicts("Z", (items, items, whs), cat='Binary') # 用于二次项线性化

        # 2. 基础约束：一品一仓与仓库开启逻辑
        for i in items:
            model += pulp.lpSum([X[i][j] for j in whs]) == 1
            for j in whs:
                model += X[i][j] <= Y[j]

        # 3. 关联度非线性乘积的线性化约束 (Z_ikj = X_ij * X_kj)
        for i in items:
            for k in items:
                # 仅对存在关联度的品类对建立约束，极大降低复杂度
                if i < k and (i in self.association_matrix.index and k in self.association_matrix.columns) and self.association_matrix.loc[i, k] > 0:
                    for j in whs:
                        model += Z[i][k][j] <= X[i][j]
                        model += Z[i][k][j] <= X[k][j]
                        model += Z[i][k][j] >= X[i][j] + X[k][j] - 1

        # 4. T1, T2 的刚性物理阈值约束 (0.75-0.90, 0.70-0.85)
        for j in whs:
            C_j = self.warehouses.loc[j, '仓容上限']
            O_j = self.warehouses.loc[j, '产能上限']
            
            stock_j = pulp.lpSum([X[i][j] * self.D_i[i] for i in items])
            model += stock_j >= 0.75 * C_j * Y[j]
            model += stock_j <= 0.90 * C_j * Y[j]
            
            sales_j = pulp.lpSum([X[i][j] * self.S_i[i] for i in items])
            model += sales_j >= 0.70 * O_j * Y[j]
            model += sales_j <= 0.85 * O_j * Y[j]

        # 5. 定义 T1~T4 的代数表达式供目标函数调用
        # T1, T2: 使用全局平均利用率以保持目标函数的纯线性
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

    # ================= 第一步：估计 T3 极小值 =================
    def solve_step1_min_t3(self):
        print("步骤 1：建立初步模型，估计总仓租成本(T3)的极小值...")
        model, X, Y, T1_expr, T2_expr, T3_expr, T4_expr = self._build_base_model("Min_T3", pulp.LpMinimize)
        model += T3_expr  # 目标函数设定为最小化 T3
        
        model.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=120))
        if model.status == 1:
            return pulp.value(T3_expr), pulp.value(T4_expr)
        else:
            raise ValueError("初步模型1无可行解，请检查阈值是否过严。")

    # ================= 第二步：估计 T4 极大值 =================
    def solve_step2_max_t4(self):
        print("步骤 2：建立初步模型，估计品类关联度(T4)的极大值...")
        model, X, Y, T1_expr, T2_expr, T3_expr, T4_expr = self._build_base_model("Max_T4", pulp.LpMaximize)
        model += T4_expr  # 目标函数设定为最大化 T4
        
        model.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=120))
        if model.status == 1:
            return pulp.value(T4_expr), pulp.value(T3_expr)
        else:
            raise ValueError("初步模型2无可行解。")

    # ================= 第三步：求解综合目标 Z =================
    def solve_step3_comprehensive(self, t3_min, t3_max_prime, t4_min_prime, t4_max):
        print("步骤 3：建立单目标规划模型，求解综合极值 Maximize Z...")
        model, X, Y, T1_expr, T2_expr, T3_expr, T4_expr = self._build_base_model("Max_Z_Comprehensive", pulp.LpMaximize)
        
        # 严格执行论文公式进行无量纲化处理 (分母加 1e-6 防止除零报错)
        norm_T3 = (T3_expr - t3_min) / (t3_max_prime - t3_min + 1e-6)
        norm_T4 = (T4_expr - t4_min_prime) / (t4_max - t4_min_prime + 1e-6)
        
        # 综合目标函数：Z = w1*T1 + w2*T2 - w3*norm_T3 + w4*norm_T4
        Z_expr = self.w1 * T1_expr + self.w2 * T2_expr - self.w3 * norm_T3 + self.w4 * norm_T4
        model += Z_expr
        
        model.solve(pulp.PULP_CBC_CMD(msg=True, maxSeconds=300))
        
        if model.status == 1 or model.status == -1: # 包含找到局部最优解
            print(f"最终模型求解完成！综合评分 Z: {pulp.value(Z_expr)}")
            return self._export_results(X)
        else:
            print("未能找到满足条件的最终分仓方案。")
            return None

    def _export_results(self, X_vars):
        """将 0-1 变量矩阵解析为人类可读格式"""
        results = []
        for i in self.D_i.keys():
            for j in self.warehouses.index:
                if pulp.value(X_vars[i][j]) == 1.0:
                    results.append({'货物编号': i, '存放仓库': j})
        
        results_df = pd.DataFrame(results)
        # results_df.to_excel('problem2_final_summary.xlsx', index=False)
        print("问题二分仓方案结果整理完毕！")
        return results_df

    # ================= 主控制流水线 =================
    def run_problem2_pipeline(self):
        """执行全套问题二闭环求解逻辑"""
        self.load_data()
        if self.warehouses.empty or not self.D_i:
            print("因缺乏占位数据的实体接入，算法流水线中断。")
            return
            
        try:
            # 步骤 1
            t3_min, t4_min_prime = self.solve_step1_min_t3()
            # 步骤 2
            t4_max, t3_max_prime = self.solve_step2_max_t4()
            # 步骤 3
            final_plan_df = self.solve_step3_comprehensive(t3_min, t3_max_prime, t4_min_prime, t4_max)
            return final_plan_df
        except Exception as e:
            print(f"求解过程中断: {e}")

if __name__ == "__main__":
    # 此处假设问题一已运行完毕并传递了 D_i (日存货字典) 和 S_i (日销量字典)
    dummy_D_i = {'Category_1': 50, 'Category_2': 80} # 占位接口
    dummy_S_i = {'Category_1': 20, 'Category_2': 35} # 占位接口
    
    p2_solver = Problem2_SingleWarehouseOptimizer(
        warehouse_info_path="data/附件3_仓库信息.xlsx",
        association_matrix_path="data/附件4_关联度.xlsx",
        D_i=dummy_D_i,
        S_i=dummy_S_i
    )
    # p2_solver.run_problem2_pipeline()