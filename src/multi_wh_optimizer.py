import pandas as pd
import pulp
import logging

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MultiWarehouseOptimizer:
    def __init__(self, wh_path, feat_path, D_i, S_i):
        """
        初始化一品多仓分层优化模型
        
        :param wh_path: 仓库基础信息路径
        :param feat_path: 特征矩阵路径 (包含基础关联、件型关联与高级品类关联)
        :param D_i: 各品类日均存货量字典
        :param S_i: 各品类日均销量字典
        """
        self.wh_path = wh_path
        self.feat_path = feat_path
        self.D_i = D_i
        self.S_i = S_i
        
        self.warehouses = pd.DataFrame()
        self.A_matrix = pd.DataFrame() # 基础关联矩阵
        self.G_matrix = pd.DataFrame() # 件型特征矩阵
        self.H_matrix = pd.DataFrame() # 高级品类关联矩阵
        
        # 综合评价指标权重
        self.w1, self.w2, self.w3, self.w4 = 0.35, 0.30, 0.20, 0.15
        self.w5, self.w6 = 0.4, 0.6 # 特殊关联内部权重

    def load_data(self):
        """加载多维特征矩阵与仓库参数"""
        try:
            self.warehouses = pd.read_excel(self.wh_path, index_col=0)
            self.A_matrix = pd.read_excel(self.feat_path, sheet_name='Association', index_col=0)
            self.G_matrix = pd.read_excel(self.feat_path, sheet_name='Shape', index_col=0)
            self.H_matrix = pd.read_excel(self.feat_path, sheet_name='Advanced', index_col=0)
            logging.info("多层特征矩阵及环境参数加载完毕")
        except Exception as e:
            logging.error(f"数据读取异常: {e}")
            raise

    def _build_model_skeleton(self, model_name, sense=pulp.LpMaximize):
        """
        构建多仓动态分配与线性化约束骨架
        """
        model = pulp.LpProblem(model_name, sense)
        items = list(self.D_i.keys())
        whs = self.warehouses.index.tolist() if not self.warehouses.empty else []
        split_options = [1, 2, 3] # 每个品类最多允许分布的仓库数

        # 1. 决策变量定义
        # W_in: 品类 i 是否选择切分为 n 份
        W = pulp.LpVariable.dicts("W", (items, split_options), cat='Binary')
        # X_inj: 在切分为 n 份的前提下，品类 i 是否存放在仓库 j
        X = pulp.LpVariable.dicts("X", (items, split_options, whs), cat='Binary')
        # Y_j: 仓库 j 是否开启
        Y = pulp.LpVariable.dicts("Y", whs, cat='Binary')

        # 辅助变量: 表达品类 i 在仓库 j 的实际存放状态
        X_actual = {(i, j): pulp.lpSum([X[i][n][j] for n in split_options]) for i in items for j in whs}

        # Z_ikj: 线性化辅助变量，表示品类 i 和 k 是否共存于仓库 j
        Z = pulp.LpVariable.dicts("Z", (items, items, whs), cat='Binary')

        # 2. 逻辑约束
        for i in items:
            # 切分策略唯一性约束
            model += pulp.lpSum([W[i][n] for n in split_options]) == 1
            
            for n in split_options:
                # 切分份数与选中仓库数一致性约束
                model += pulp.lpSum([X[i][n][j] for j in whs]) == n * W[i][n]

            for j in whs:
                # 仓库开启逻辑约束
                model += X_actual[(i, j)] <= Y[j]

        # 3. 关联度项线性化约束
        for i in items:
            for k in items:
                if i < k:
                    # 仅针对存在有效关联特征的品类对建立线性化约束以减少 MILP 规模
                    if (self.A_matrix.get(k, i, default=0) > 0 or 
                        self.G_matrix.get(k, i, default=0) > 0 or 
                        self.H_matrix.get(k, i, default=0) > 0):
                        for j in whs:
                            model += Z[i][k][j] <= X_actual[(i, j)]
                            model += Z[i][k][j] <= X_actual[(k, j)]
                            model += Z[i][k][j] >= X_actual[(i, j)] + X_actual[(k, j)] - 1

        # 4. 动态等分下的库存与产能约束
        T1_sum, T2_sum = 0, 0
        for j in whs:
            C_j = self.warehouses.loc[j, '仓容上限']
            O_j = self.warehouses.loc[j, '产能上限']
            
            # 将切分份数 n 引入系数中，消除变量除法导致的非线性问题
            stock_j = pulp.lpSum([self.D_i[i] * (X[i][1][j]/1.0 + X[i][2][j]/2.0 + X[i][3][j]/3.0) for i in items])
            sales_j = pulp.lpSum([self.S_i[i] * (X[i][1][j]/1.0 + X[i][2][j]/2.0 + X[i][3][j]/3.0) for i in items])
            
            # 物理阈值刚性约束
            model += stock_j >= 0.75 * C_j * Y[j]
            model += stock_j <= 0.90 * C_j * Y[j]
            model += sales_j >= 0.70 * O_j * Y[j]
            model += sales_j <= 0.85 * O_j * Y[j]
            
            T1_sum += stock_j / C_j
            T2_sum += sales_j / O_j
            
        num_whs = len(whs) if whs else 1
        T1_expr = T1_sum / num_whs
        T2_expr = T2_sum / num_whs
        
        # T3: 总仓租成本
        T3_expr = pulp.lpSum([self.warehouses.loc[j, '仓租日成本'] * Y[j] for j in whs])
        
        # T4: 跨仓配送成本 (以分仓总数替代表示)
        T4_expr = pulp.lpSum([n * W[i][n] for i in items for n in split_options])

        # T5 & T6: 多层关联度评价指标
        T5_terms, T6_terms = [], []
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

    def run_optimization(self):
        """执行全流程优化计算"""
        self.load_data()
        if self.warehouses.empty or not self.D_i:
            logging.error("数据缺失，优化终止")
            return None

        # 阶段 1: 估算 T3 极小值
        logging.info("正在计算 T3 极值范围...")
        m1, _, _, _, _, t3_expr, t4_expr, _, _ = self._build_model_skeleton("T3_Bound", pulp.LpMinimize)
        m1 += t3_expr
        m1.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=60))
        t3_min = pulp.value(t3_expr)
        t4_min_ref = pulp.value(t4_expr)

        # 阶段 2: 估算 T4 极大值
        m2, _, _, _, _, t3_expr, t4_expr, _, _ = self._build_model_skeleton("T4_Bound", pulp.LpMaximize)
        m2 += t4_expr
        m2.solve(pulp.PULP_CBC_CMD(msg=False, maxSeconds=60))
        t4_max = pulp.value(t4_expr)
        t3_max_ref = pulp.value(t3_expr)

        # 阶段 3: 多目标分层综合规划
        logging.info("正在求解多目标分层优化模型...")
        m3, W, X_actual, t1, t2, t3, t4, t5, t6 = self._build_model_skeleton("Final_Multi_Goal", pulp.LpMaximize)
        
        # 指标归一化处理
        norm_t3 = (t3 - t3_min) / (t3_max_ref - t3_min + 1e-6)
        norm_t4 = (t4 - t4_min_ref) / (t4_max - t4_min_ref + 1e-6)
        
        # 基础综合评价函数 Z
        Z_score = self.w1 * t1 + self.w2 * t2 - self.w3 * norm_t3 + self.w4 * norm_t4
        
        # 采用字典序权重法处理 T5 (首要目标) 与 T6 (次要目标)
        M1, M2 = 1e4, 1e3 
        m3 += (M1 * t5) + (M2 * t6) + Z_score
        
        m3.solve(pulp.PULP_CBC_CMD(msg=True, maxSeconds=300))
        
        if m3.status in [pulp.LpStatusOptimal, pulp.LpStatusNotOptimal]:
            logging.info(f"优化完成。T5: {pulp.value(t5):.2f}, T6: {pulp.value(t6):.2f}")
            return self._format_output(X_actual)
        return None

    def _format_output(self, X_actual):
        """格式化输出分配方案"""
        output = []
        for i in self.D_i.keys():
            record = {'Item_ID': i}
            assigned_whs = [j for j in self.warehouses.index if pulp.value(X_actual[(i, j)]) > 0.5]
            for idx, wh_id in enumerate(assigned_whs):
                record[f'Warehouse_{idx+1}'] = wh_id
            output.append(record)
        return pd.DataFrame(output).fillna('')

if __name__ == "__main__":
    opt = MultiWarehouseOptimizer(
        wh_path="data/wh_info.xlsx",
        feat_path="data/feature_matrix.xlsx",
        D_i={'C1': 100, 'C2': 200},
        S_i={'C1': 40, 'C2': 60}
    )
    # result = opt.run_optimization()
