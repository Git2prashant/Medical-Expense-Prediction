# 🏥 Medical Expense Prediction: End-to-End Procedure & Formula Guide

A step-by-step walkthrough of the entire pipeline, mathematical formulas, algorithms, preprocessing methods, and evaluation techniques used in this project.

---

## 📌 High-Level Architecture Flowchart

```
1. Raw Data (insurance.csv)
       │
       ▼
2. Exploratory Data Analysis (EDA) ──► Histograms, Correlations, Skewness Check
       │
       ▼
3. Preprocessing & Encoding ──────────► Binary Map (0/1) + One-Hot Encoding (drop_first)
       │
       ▼
4. Train/Test Split ─────────────────► 80% Train (1,070 rows) / 20% Test (268 rows)
       │
       ▼
5. Model Training (OLS) ─────────────► Closed-form Normal Equation: β = (XᵀX)⁻¹ Xᵀy
       │
       ▼
6. Model Evaluation ─────────────────► Metrics: R², MAE, RMSE & Residual Diagnostics
       │
       ▼
7. Non-Linear Benchmark ─────────────► Random Forest Regressor (R² = 0.8656)
       │
       ▼
8. Serialization & UI Deployment ────► joblib (.pkl) ──► Streamlit Web App (app.py)
```

---

## Step 1: Data Ingestion & Exploratory Analysis (EDA)

* **Dataset:** 1,338 records, 7 features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).
* **Missing Value Check:** Zero null values (`df.isnull().sum() == 0`).
* **Target Distribution (`charges`):**
  * Mean: **$13,270.42**, Median: **$9,382.03**
  * **Skewness:** $+1.515$ (Right-skewed, meaning a small subset of high-risk patients incur extreme costs).

---

## Step 2: Categorical Feature Encoding

Linear Regression requires purely numerical matrices. Text values cannot undergo matrix multiplication.

### 1. Binary Mapping (for 2-class categories)
* **`sex`**: $\text{male} \rightarrow 0, \quad \text{female} \rightarrow 1$
* **`smoker`**: $\text{no} \rightarrow 0, \quad \text{yes} \rightarrow 1$

### 2. One-Hot Encoding with Dummy Variable Trap Elimination (for `region`)
* Regions (`northeast`, `northwest`, `southeast`, `southwest`) have **no natural ranking**.
* We cannot use Label Encoding ($0, 1, 2, 3$) because the model would falsely assume $\text{southwest} (3) > \text{northeast} (0)$.
* We use **One-Hot Encoding** with `drop_first=True`:
  * Generates 3 binary columns: `region_northwest`, `region_southeast`, `region_southwest`.
  * If all three are $0$, the record is uniquely inferred as `northeast` (baseline reference category).
  * **Why `drop_first=True`?** Prevents perfect multicollinearity (where one column can be linearly predicted from the others), known as the **Dummy Variable Trap**, which makes the matrix $(\mathbf{X}^T \mathbf{X})$ non-invertible.

---

## Step 3: Dataset Splitting

* **Features Matrix ($\mathbf{X}$):** 8 input dimensions ($1,338 \times 8$)
* **Target Vector ($\mathbf{y}$):** Continuous dollar charges ($1,338 \times 1$)
* **Method:** `train_test_split(test_size=0.2, random_state=42)`
  * **Training Set:** 80% ($1,070$ samples) — used to compute weights $\boldsymbol{\beta}$
  * **Testing Set:** 20% ($268$ samples) — withheld to evaluate real-world generalization
  * **`random_state=42`:** Fixes pseudo-random shuffling so results are 100% reproducible.

---

## Step 4: The Core Mathematical Model (Linear Regression)

### 1. Hypothesis Function
The model assumes an additive linear hyperplane:
$$\hat{y} = \beta_0 + \beta_1(\text{age}) + \beta_2(\text{sex}) + \beta_3(\text{bmi}) + \beta_4(\text{children}) + \beta_5(\text{smoker}) + \beta_6(\text{reg\_nw}) + \beta_7(\text{reg\_se}) + \beta_8(\text{reg\_sw})$$

In compact vector notation:
$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$$

### 2. Loss Function (Ordinary Least Squares - OLS)
We measure prediction error using the **Residual Sum of Squares (RSS)**:
$$J(\boldsymbol{\beta}) = \text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

### 3. Optimization Method (Closed-Form Analytical Solution)
Setting the gradient $\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = 0$ yields the **Normal Equation**:
$$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 4. Calculated Coefficients in this Project
* **Intercept ($\beta_0$):** $-\$11,938.56$
* **$\beta_{\text{smoker}}$:** $\mathbf{+\$23,651.13}$ (Largest factor by far: smoking adds ~$23.6k/yr)
* **$\beta_{\text{bmi}}$:** $\mathbf{+\$337.09}$ (Each BMI point increase adds ~$337/yr)
* **$\beta_{\text{age}}$:** $\mathbf{+\$256.98}$ (Each year of age adds ~$257/yr)
* **$\beta_{\text{children}}$:** $\mathbf{+\$425.28}$ (Each dependent adds ~$425/yr)

---

## Step 5: Model Evaluation Formulas & Metrics

Predictions $\hat{y}$ on the withheld 20% test set are validated against actual values $y$:

### 1. Mean Absolute Error (MAE)
Average dollar discrepancy per prediction (interpretable in USD):
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| = \mathbf{\$4,181.19}$$

### 2. Root Mean Squared Error (RMSE)
Heavily penalizes large outlier errors because mistakes are squared:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \mathbf{\$5,796.28}$$

### 3. Coefficient of Determination ($R^2$ Score)
Measures the proportion of variance explained by the model:
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = \mathbf{0.7836} \quad (78.36\%)$$

---

## Step 6: Residual Diagnostics & Why Linear Regression Isn't 100%

* **Residual Definition:** $e_i = y_i - \hat{y}_i$
* **Diagnostic Finding:** A plot of residuals vs. predicted values reveals fan-shaped heteroscedasticity.
* **The Root Cause:** **Non-linear interaction effect** between `smoker` and `bmi`:
  * Non-smoker: BMI increase causes a slow, linear increase.
  * Smoker with $\text{BMI} \ge 30$: Medical expenses suddenly jump exponentially to $\$35,000-\$45,000+$.
  * A standard straight plane cannot curve to capture this sudden spike without an explicit interaction term $(\text{BMI} \times \text{Smoker})$.

---

## Step 7: Benchmark Comparison (Random Forest Regressor)

To prove the non-linear interaction hypothesis, we trained an ensemble **Random Forest Regressor** ($n=200$ decision trees):

| Metric | Linear Regression | Random Forest | Advantage of Random Forest |
|---|---|---|---|
| **$R^2$ Score** | **0.7836** | **0.8656** | **+10.46%** variance captured |
| **MAE ($)** | **$4,181.19** | **$2,546.83** | **-$1,634.36** lower average error |
| **RMSE ($)** | **$5,796.28** | **$4,568.42** | **-$1,227.86** fewer extreme errors |

* **Why Random Forest performs better:** Decision tree ensembles split continuous features at specific thresholds (e.g., $\text{BMI} > 30$ AND $\text{smoker} == 1$), isolating the high-risk subgroup.

---

## Step 8: Serialization & Web Deployment

1. **Model Persistence:**
   * Exported `model.pkl` (trained weights) and `columns.pkl` (exact feature order) using `joblib`.
2. **Interactive Streamlit App (`app.py`):**
   * Takes inputs: Age (slider), Sex (radio), BMI (number), Children (slider), Smoker (dropdown), Region (dropdown).
   * Aligns the input row to `columns.pkl` order.
   * Runs $\hat{y}_{\text{new}} = \text{model.predict}(\mathbf{X}_{\text{new}})[0]$.
   * Renders the predicted premium in real-time with health risk indicators.

---

## ⚡ Quick Summary of All Formulas

| Component | Mathematical Formula |
|---|---|
| **Prediction** | $\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j = \mathbf{X}\boldsymbol{\beta}$ |
| **OLS Cost / RSS** | $J(\boldsymbol{\beta}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ |
| **Normal Equation** | $\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ |
| **MAE** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ |
| **Residual** | $e_i = y_i - \hat{y}_i$ |
