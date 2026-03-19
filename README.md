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
