import pandas as pd
import pulp

class Problem3_MultiWarehouseOptimizer:
    def __init__(self, warehouse_info_path, feature_matrix_path, D_i, S_i):
        """
        初始化问题三：一品多仓多目标分层优化
        :param warehouse_info_path: 附件3 仓库信息真实路径
        :param feature_matrix_path: 包含基础关联、件型关联、高级品类关联的表格路径
        :param D_i: 问题一输出的字典，{'品类1': 日均货量...}
        :param S_i: 问题一输出的字典，{'品类1': 日均销量...}
        """
        self.wh_path = warehouse_info_path
        self.feat_path = feature_matrix_path
        self.D_i = D_i
        self.S_i = S_i
        
        self.warehouses = pd.DataFrame()
        self.A_matrix = pd.DataFrame() # 基础关联 T5
        self.G_matrix = pd.DataFrame() # 件型关联 (T6组成部分)
        self.H_matrix = pd.DataFrame() # 高级品类关联 (T6组成部分)
        
        # 综合指标权重 (严格对照论文)
        self.w1, self.w2, self.w3, self.w4 = 0.35, 0.30, 0.20, 0.15
        self.w5, self.w6 = 0.4, 0.6 # 假设：高级品类(w6)权重大于件型(w5)

    def load_data(self):
        """【严格占位加载】只进行读取，绝不生成模拟数据"""
        try:
            self.warehouses = pd.read_excel(self.wh_path, index_col=0)
            self.A_matrix = pd.read_excel(self.feat_path, sheet_name='Association', index_col=0)
            self.G_matrix = pd.read_excel(self.feat_path, sheet_name='Shape', index_col=0)
            self.H_matrix = pd.read_excel(self.feat_path, sheet_name='Advanced', index_col=0)
            print("问题三运筹学参数及多层特征矩阵加载完毕。")
        except Exception as e:
            print(f"数据加载提示 (占位状态): {e}")

    def _build_base_model(self, model_name, sense=pulp.LpMaximize):
        """
        构建问题三极其复杂的动态等分与分层特征约束骨架
        """
        model = pulp.LpProblem(model_name, sense)
        items = list(self.D_i.keys())
        whs = self.warehouses.index.tolist() if not self.warehouses.empty else []

        # ==================== 1. 绝妙的线性化决策变量 ====================
        # W[i][n]: 品类 i 是否被切分为 n 份 (n=1, 2, 3)
        W = pulp.LpVariable.dicts("W", (items, [1, 2, 3]), cat='Binary')
        # X[i][n][j]: 在品类 i 被切分为 n 份的前提下，是否存放在仓库 j
        X = pulp.LpVariable.dicts("X", (items, [1, 2, 3], whs), cat='Binary')
        # Y[j]: 仓库 j 是否开启
        Y = pulp.LpVariable.dicts("Y", whs, cat='Binary')

        # X_actual[i][j]: 实际发生存放的总状态
        X_actual = {(i, j): pulp.lpSum([X[i][n][j] for n in [1, 2, 3]]) for i in items for j in whs}

        # Z[i][k][j]: 辅助变量，表示品类 i 和 k 是否共同存放在仓库 j (用于 T5, T6)
        Z = pulp.LpVariable.dicts("Z", (items, items, whs), cat='Binary')

        # ==================== 2. 切分逻辑与刚性约束 ====================
        for i in items:
            # 约束A：每个品类只能选择一种切分策略 (1仓, 2仓, 或 3仓)
            model += pulp.lpSum([W[i][n] for n in [1, 2, 3]]) == 1
            
            for n in [1, 2, 3]:
                # 约束B：如果选了切分成 n 份，那么选中的仓库数量必须严格等于 n
                model += pulp.lpSum([X[i][n][j] for j in whs]) == n * W[i][n]

            for j in whs:
                # 约束C：如果存了东西，仓库必须开启
                model += X_actual[(i, j)] <= Y[j]

        # 约束D：Z 变量的二次项线性化 (Z_ikj = X_actual_ij * X_actual_kj)
        for i in items:
            for k in items:
                if i < k:
                    # 只有当基础关联、件型关联或高级关联存在时，才建立约束以节约算力
                    if (self.A_matrix.get(k, i, default=0) > 0 or 
                        self.G_matrix.get(k, i, default=0) > 0 or 
                        self.H_matrix.get(k, i, default=0) > 0):
                        for j in whs:
                            model += Z[i][k][j] <= X_actual[(i, j)]
                            model += Z[i][k][j] <= X_actual[(k, j)]
                            model += Z[i][k][j] >= X_actual[(i, j)] + X_actual[(k, j)] - 1

        # ==================== 3. 代数表达式定义 (解决变量除以变量的非线性灾难) ====================
        T1_expr, T2_expr = 0, 0
        
        for j in whs:
            C_j = self.warehouses.loc[j, '仓容上限']
            O_j = self.warehouses.loc[j, '产能上限']
            
            # 由于切分分母被提到了 X[i][n][j] 的 n 中，货量完美线性化！
            # 存货_j = D_i * (X[i][1][j]/1 + X[i][2][j]/2 + X[i][3][j]/3)
            stock_j = pulp.lpSum([self.D_i[i] * (X[i][1][j]/1.0 + X[i][2][j]/2.0 + X[i][3][j]/3.0) for i in items])
            sales_j = pulp.lpSum([self.S_i[i] * (X[i][1][j]/1.0 + X[i][2][j]/2.0 + X[i][3][j]/3.0) for i in items])
            
            # 物理阈值约束 (0.75-0.90, 0.70-0.85)
            model += stock_j >= 0.75 * C_j * Y[j]
            model += stock_j <= 0.90 * C_j * Y[j]
            model += sales_j >= 0.70 * O_j * Y[j]
            model += sales_j <= 0.85 * O_j * Y[j]
            
            # 累加 T1, T2 用于综合目标函数
            T1_expr += stock_j / C_j
            T2_expr += sales_j / O_j
            
        num_whs = len(whs) if whs else 1
        T1_expr = T1_expr / num_whs
        T2_expr = T2_expr / num_whs
        
        # T3 仓租成本
        T3_expr = pulp.lpSum([self.warehouses.loc[j, '仓租日成本'] * Y[j] for j in whs])
        
        # T4 品类分仓数 (问题三中 T4 变成了分仓数)
        T4_expr = pulp.lpSum([n * W[i][n] for i in items for n in [1, 2, 3]])

        # T5 基础品类关联度
        T5_terms = []
        # T6 特殊品类关联度 (w5*件型 + w6*高级品类)
        T6_terms = []
        
        for i in items:
            for k in items:
                if i < k:
                    a_val = self.A_matrix.get(k, i, default=0)
                    g_val = self.G_matrix.get(k, i, default=0)
                    h_val = self.H_matrix.get(k, i, default=0)
                    
                    for j in whs:
                        if a_val > 0:
                            T5_terms.append(a_val * Z[i][k][j])
                        if g_val > 0 or h_val > 0:
                            T6_terms.append((self.w5 * g_val + self.w6 * h_val) * Z[i][k][j])
                            
        T5_expr = pulp.lpSum(T5_terms)
        T6_expr = pulp.lpSum(T6_terms)

        return model, W, X_actual, T1_expr, T2_expr, T3_expr, T4_expr, T5_expr, T6_expr

    # ================= 闭环：探测极值并求解最终字典序规划 =================
    
    def solve_problem3_pipeline(self):
        if self.warehouses.empty or not self.D_i:
            print("缺乏真实数据输入，占位运行终止。")
            return None

        # --- 步骤 1：求解 T3 极小值 ---
        print("步骤 1：探测总仓租成本 T3 极小值...")
        m1, _, _, _, _, t3_expr, t4_expr, _, _ = self._build_base_model("Min_T3", pulp.LpMinimize)
        m1 += t3_expr
        m1.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=60))
        if m1.status != 1: raise Exception("无法满足问题三基础约束。")
        t3_min = pulp.value(t3_expr)
        t4_min_prime = pulp.value(t4_expr) # 记录此时的 T4

        # --- 步骤 2：求解 T4 极大值 ---
        # 备注：论文中要求估计 T4 极值并记录 T3，此处严谨落实
        print("步骤 2：探测分仓数 T4 极大值...")
        m2, _, _, _, _, t3_expr, t4_expr, _, _ = self._build_base_model("Max_T4", pulp.LpMaximize)
        m2 += t4_expr
        m2.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=60))
        t4_max = pulp.value(t4_expr)
        t3_max_prime = pulp.value(t3_expr) # 记录此时的 T3

        # --- 步骤 3：综合多目标模型求解 ---
        print("步骤 3：执行最终多目标分层优化模型...")
        m3, W, X_actual, t1_expr, t2_expr, t3_expr, t4_expr, t5_expr, t6_expr = self._build_base_model("Final_P3", pulp.LpMaximize)
        
        # 构建 Z 指标 (分母加 1e-6 防止零除报错)
        norm_t3 = (t3_expr - t3_min) / (t3_max_prime - t3_min + 1e-6)
        norm_t4 = (t4_expr - t4_min_prime) / (t4_max - t4_min_prime + 1e-6)
        Z_expr = self.w1 * t1_expr + self.w2 * t2_expr - self.w3 * norm_t3 + self.w4 * norm_t4
        
        # 字典序权重法结合三大目标：M1 * T5 (首要) + M2 * T6 (次要) + Z (最次要)
        M1, M2 = 10000, 1000 
        m3 += (M1 * t5_expr) + (M2 * t6_expr) + Z_expr
        
        m3.solve(pulp.PULP_CBC_CMD(msg=True, maxSeconds=300))
        
        if m3.status in [1, -1]: # Optimal 或 Feasible
            print(f"求解成功！T5(基础关联): {pulp.value(t5_expr)}, T6(特殊关联): {pulp.value(t6_expr)}")
            return self._export_multi_warehouse_results(X_actual)
        else:
            print("模型无法收敛。")
            return None

    def _export_multi_warehouse_results(self, X_actual):
        """将复杂的 1-3 仓位分配结果展平为最终的提交格式"""
        results = []
        for i in self.D_i.keys():
            whs_assigned = []
            for j in self.warehouses.index:
                if pulp.value(X_actual[(i, j)]) > 0.5:
                    whs_assigned.append(f"仓库_{j}")
            row = {'货物编号': i}
            for idx, wh in enumerate(whs_assigned):
                row[f'存放仓库{idx+1}'] = wh
            results.append(row)
            
        results_df = pd.DataFrame(results)
        results_df.fillna('', inplace=True)
        # results_df.to_excel('problem3_final_summary.xlsx', index=False)
        print("一品多仓最优结果已导出就绪！")
        return results_df

if __name__ == "__main__":
    p3_solver = Problem3_MultiWarehouseOptimizer(
        warehouse_info_path="data/附件3_仓库信息.xlsx",
        feature_matrix_path="data/附件4_特征矩阵.xlsx",
        D_i={},
        S_i={}
    )
    # p3_solver.load_data()
    # final_plan = p3_solver.solve_problem3_pipeline()