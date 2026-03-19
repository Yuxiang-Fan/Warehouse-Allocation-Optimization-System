# Warehouse Allocation and Demand Forecasting System

This repository provides a systematic implementation of demand forecasting and inventory distribution logic. The project addresses the challenges of warehousing category management through a combination of predictive time-series modeling and Mixed-Integer Linear Programming (MILP). 

All external datasets and problem backgrounds referenced in this codebase originate from **Question B of the 5th MathorCup Mathematical Application Challenge (2024)**.

## Project Methodology

The implementation is structured into three progressive modules:

### 1. Predictive Analytics (Problem 1)
The system employs an ensemble forecasting approach to estimate daily sales and monthly inventory requirements. It utilizes:
- **Dynamic Factor Models (DFM)** and **SARIMAX** for capturing complex seasonal and spatial correlations in sales data.
- **Linear Regression** for trend analysis in inventory levels.
- A customized $T$-score evaluation engine that selects the optimal model for each category based on historical Pearson correlation and curve smoothness.

### 2. Single-Warehouse Optimization (Problem 2)
The module focuses on a "one-item-one-warehouse" constraint. It implements a hierarchical optimization strategy:
- **Constraints:** Ensures strict adherence to warehouse capacity and processing output thresholds.
- **Objective:** Minimizes total rental costs while maximizing the association between stored categories. 
- **Technique:** Leverages MILP to transform non-linear association products into linear constraints using auxiliary binary variables.

### 3. Multi-Warehouse Strategic Planning (Problem 3)
The scope expands to allow "multi-warehouse" distribution (up to three locations per category) and incorporates advanced category tagging.
- **Dynamic Allocation:** Handles the non-linear challenge of equal inventory splitting across multiple sites.
- **Feature Integration:** Optimizes storage based on specialized attributes such as package dimensions and high-level category associations.

## File Structure

The core logic is contained within the `src/` directory:

```text
src/
├── __init__.py
├── demand_forecaster.py         # Module for time-series forecasting and evaluation
├── single_wh_optimizer.py       # MILP solver for single-warehouse allocation
└── multi_wh_optimizer.py        # Advanced solver for multi-warehouse strategies
```

---

# 仓储分配与需求预测系统

本项目提供了一套系统性的需求预测与库存分配逻辑实现方案。该项目通过结合预测性时间序列建模与混合整数线性规划（MILP），解决了仓储品类管理中的挑战。

本代码库中引用的所有外部数据集及问题背景均源自 **2024 年第五届 MathorCup 数学应用挑战赛 B 题**。

## 项目研究方法

该实现方案分为三个递进模块：

### 1. 预测性分析（问题 1）
系统采用集成预测方法来估算每日销量和每月库存需求。主要利用了：
- **动态因子模型 (DFM)** 和 **SARIMAX**：用于捕捉销售数据中复杂的季节性和空间相关性。
- **线性回归**：用于库存水平的趋势分析。
- **自定义 T 分值评估引擎**：基于历史皮尔逊相关性和曲线平滑度，为每个品类选择最优模型。

### 2. 单仓优化布局（问题 2）
该模块专注于“一品一仓”约束，实施了层级优化策略：
- **约束条件**：严格遵守仓库容量限制和处理能力（产出）阈值。
- **目标函数**：在最小化总租赁成本的同时，最大化存储品类之间的关联度。
- **技术手段**：利用 MILP，通过辅助二进制变量将非线性的品类关联乘积转化为线性约束。

### 3. 多仓战略规划（问题 3）
研究范围扩展至允许“多仓”分布（每个品类最多可分布在三个位置），并整合了高级品类标签。
- **动态分配**：处理库存等分到多个站点的非线性挑战。
- **特征整合**：根据包装尺寸和高级品类关联等特定属性优化存储布局。

## 文件结构

核心逻辑包含在 `src/` 目录下：

```text
src/
├── __init__.py
├── demand_forecaster.py         # 时间序列预测与评估模块
├── single_wh_optimizer.py       # 用于单仓分配的 MILP 求解器
└── multi_wh_optimizer.py        # 用于多仓策略的高级求解器
```
