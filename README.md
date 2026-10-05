# 🏥 Medical Expense Prediction

> **Supervised Machine Learning System to Forecast Individual Annual Medical Insurance Expenses**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-expense-prediction-n9f2twmwqkg5bq4hefzr93.streamlit.app/)
![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Deployment-Live-success)

---

### 🌐 Live Web Application
👉 **Access the live interactive predictor:**  
**[https://medical-expense-prediction-n9f2twmwqkg5bq4hefzr93.streamlit.app/](https://medical-expense-prediction-n9f2twmwqkg5bq4hefzr93.streamlit.app/)**

---

## 📋 Project Overview

Insurance companies calculate healthcare premiums based on individual risk indicators such as age, body mass index (BMI), smoking habits, and geographical area. 

This repository implements an end-to-end Machine Learning pipeline:
* **Problem Type:** Supervised Learning → Continuous Regression
* **Core Algorithm:** Ordinary Least Squares (OLS) Linear Regression
* **Benchmark Comparison:** Random Forest Regressor ($n=200$)
* **Dataset:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) ($1,338$ records, $7$ attributes)
* **Production Deployment:** Interactive web interface powered by Streamlit Cloud

---

## 📊 Model Evaluation Results

Tested on an unseen 20% test partition ($268$ samples) with `random_state=42`:

| Metric | Linear Regression (Primary) | Random Forest (Benchmark) | Impact |
|---|:---:|:---:|:---:|
| **R² Score** | **0.7836** | **0.8656** | **+10.46%** variance captured |
| **Mean Absolute Error (MAE)** | **$4,181.19** | **$2,546.83** | **-$1,634.36** error reduction |
| **Root Mean Squared Error (RMSE)** | **$5,796.28** | **$4,568.42** | **-$1,227.86** fewer outlier errors |

> **Key Clinical Finding:** Smoking status is by far the single largest determinant of healthcare expenditure, contributing **+$23,651.13** to annual medical charges on average.

---

## 📁 Repository Structure

```
TT_Minor_Project/
├── insurance.csv                         # Raw dataset (1,338 rows × 7 features)
├── medical_expense_prediction.ipynb      # Executed Jupyter notebook with all charts & tables
├── app.py                                # Streamlit production web application
├── model.pkl                             # Serialized Linear Regression model
├── columns.pkl                           # Saved feature columns schema
├── train_model.py                        # Automated model training and export script
├── generate_plots.py                     # Script to export all 9 analytical figures
├── requirements.txt                      # Cloud deployment dependencies
├── PROJECT_REPORT.md                     # Academic laboratory project report
├── PROJECT_FLOW_GUIDE.md                 # Complete procedure, math & formulas guide
├── README.md                             # Repository documentation
└── *.png                                 # High-resolution analytical visualizations
```

---

## 📈 Visualizations Included

The repository contains 9 high-resolution diagnostic charts:
* `charges_distribution.png`: Histogram with KDE & Boxplot of target variable.
* `correlation_heatmap.png`: Masked Pearson correlation matrix.
* `categorical_vs_charges.png`: Group spreads for Smoker, Sex, and Region.
* `bmi_vs_charges.png`: Smoker × BMI non-linear interaction effect.
* `coefficients.png`: Ranked feature sensitivity coefficients.
* `actual_vs_predicted.png`: Regression prediction accuracy against 45° reference line.
* `residual_analysis.png`: Residual homoscedasticity & error distribution.
* `model_comparison.png`: Side-by-side comparison of Linear Regression vs Random Forest.

---

## 🚀 Local Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Git2prashant/Medical-Expense-Prediction.git
cd Medical-Expense-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web App Locally
```bash
streamlit run app.py
```

### 4. Re-train Models or Generate Plots
```bash
python3 train_model.py
python3 generate_plots.py
```

---

## 📖 In-Depth Guides

* 📑 **[Project Report](PROJECT_REPORT.md):** Academic formulation, problem definition, and research conclusions.
* 📐 **[Procedure & Formula Guide](PROJECT_FLOW_GUIDE.md):** Step-by-step mathematical derivation of Normal Equations, OLS loss, and evaluation metrics.
