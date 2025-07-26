# Feature Selection Analysis

## Project Overview

This project compares **explicit** and **implicit** feature selection techniques to understand their impact on model performance and interpretability.

- **Explicit methods** like **Lasso** apply regularization to directly shrink some feature coefficients to zero, resulting in a clear selection of predictors.
- **Implicit methods** like **tree-based models** (Random Forest, XGBoost) infer feature importance from internal splits, without directly removing variables.

After applying these techniques, we refit models using only the selected features to evaluate:
- Predictive performance  
- Model complexity  
- Alignment between selection strategies

This analysis helps highlight when and why certain selection methods outperform others — and what trade-offs practitioners might face when simplifying models.

---

## Repository Structure

- `data/` — Raw and processed data (excluded from git)  
  - `raw/` — Seismic dataset, (can be found at link below)
  - `clean/` — Processed feature sets  
  - `models/` — Saved model artifacts  
- `notebooks/`  
  - `01_eda.ipynb` — Exploratory data analysis  
  - `02_modeling.ipynb` — Model training and evaluation  
- `src/`  
  - `data_cleaning.py` — Data prep pipeline  
  - `models.py` — Feature selection and model training  
  - `refit_performance.py` — Refit reduced models and evaluate  
  - `utils.py` — Utility functions  
  - `viz.py` — Visualization functions  
- `requirements.txt` — Package dependencies  
- `README.md` — You are here  

---

## Feature Selection Methods

- Lasso Regression (L1) with cross-validation  
- Tree-based feature importance (Random Forest, XGBoost)  
- Visualization and performance comparison  

---

## How to Reproduce

### 1. Download Data

Download the Arcene dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/266/seismic+bumps) and place the following files in `data/raw/`:

-`seismic-bumps.arff`

---

## Notebooks

- `01_eda.ipynb` — Initial data exploration and visualization  
- `02_modeling.ipynb` — Core modeling logic with multiple feature selectors  

---

## Python Modules

- `src/models.py` — Feature selection logic  
- `src/refit_performance.py` — Evaluation on selected subsets  
- `src/viz.py` — Plotting routines  