# 🏥 Medical Expense Prediction — v1.0.0 Release

### 🌐 Live Production Application
🔗 **Live Web App:** [https://medical-expense-prediction-n9f2twmwqkg5bq4hefzr93.streamlit.app/](https://medical-expense-prediction-n9f2twmwqkg5bq4hefzr93.streamlit.app/)

---

## 🚀 What's Included in This Release

### 1. Interactive Web Application
* Deployed on **Streamlit Community Cloud** with real-time prediction capabilities.
* Input form supporting Age, Sex, BMI, Dependents, Smoker Status, and US Geographic Region.
* Automatic health classification (e.g., Obese BMI indicator) and personalized risk warnings.

### 2. Machine Learning Pipeline & Artifacts
* **Linear Regression (OLS Baseline):**
  * $R^2$ Score: **0.7836**
  * MAE: **$4,181.19**
  * RMSE: **$5,796.28**
* **Random Forest Regressor (Benchmark):**
  * $R^2$ Score: **0.8656**
  * MAE: **$2,546.83**
  * RMSE: **$4,568.42**
* **Serialized Artifacts:** Pre-trained `model.pkl` and `columns.pkl` ready for immediate inference.

### 3. Notebooks & Documentation
* Fully pre-executed Jupyter Notebook (`medical_expense_prediction.ipynb`) with all cell outputs, tables, and inline charts.
* Full academic Minor Project Report (`PROJECT_REPORT.md`).
* Comprehensive step-by-step mathematical and procedural guide (`PROJECT_FLOW_GUIDE.md`).
* 9 high-resolution diagnostic charts covering EDA, feature correlations, residuals, and model comparisons.

---

## 🛠️ Quick Local Setup
```bash
git clone https://github.com/Git2prashant/Medical-Expense-Prediction.git
cd Medical-Expense-Prediction
pip install -r requirements.txt
streamlit run app.py
```
