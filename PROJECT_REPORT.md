# Minor Project Report: Medical Expense Prediction Using Machine Learning

**Course:** Minor Project / Machine Learning Laboratory  
**Project Title:** Medical Expense Prediction Using Linear Regression  
**Date:** October 2026  

---

## Executive Summary
This project investigates the prediction of individual annual medical insurance charges using supervised machine learning regression algorithms. Using demographic and health indicators (age, sex, BMI, number of children, smoking status, and residential region), we developed a baseline Ordinary Least Squares (OLS) **Linear Regression** model and benchmarked it against a non-linear **Random Forest Regressor**. The Linear Regression model achieved an $R^2$ score of **0.7836** with a Mean Absolute Error (MAE) of **$4,181.19**, while the Random Forest Regressor attained an $R^2$ of **0.8656** with an MAE of **$2,546.83**. An interactive web application was developed and deployed using **Streamlit** for real-time risk assessment and insurance premium forecasting.

---

## 1. Problem Statement
Health insurance companies must accurately forecast expected individual medical costs to price insurance premiums sustainably and competitively. Overpricing leads to customer loss, while underpricing results in financial insolvencies.

The objective of this project is:
- To build a predictive regression model that takes an individual's personal and health parameters and predicts their annual medical charges in USD ($).
- To identify which health and demographic features contribute most significantly to medical expenditure.
- To analyze the strengths and limitations of linear assumptions when modeling healthcare cost distributions.

---

## 2. Dataset Description
The analysis utilizes the **Medical Cost Personal Dataset** (`insurance.csv`), comprising **1,338 records** across **7 features** with zero missing values.

### Attribute Dictionary

| Attribute | Data Type | Role | Domain / Values | Description |
|---|---|---|---|---|
| `age` | Integer | Feature | 18 – 64 | Age of primary beneficiary |
| `sex` | Categorical | Feature | `male`, `female` | Biological sex of beneficiary |
| `bmi` | Float | Feature | 15.96 – 53.13 | Body Mass Index ($kg/m^2$) |
| `children` | Integer | Feature | 0 – 5 | Number of dependent children |
| `smoker` | Categorical | Feature | `yes`, `no` | Whether the beneficiary smokes |
| `region` | Categorical | Feature | `northeast`, `northwest`, `southeast`, `southwest` | Residential area in the US |
| `charges` | Float | **Target** | $1,121.87 – $63,770.43 | Annual medical expense billed |

### Descriptive Statistics Summary
- **Mean Charges:** $13,270.42
- **Median Charges:** $9,382.03
- **Standard Deviation:** $12,110.01
- **Skewness:** **+1.515** (Strong positive right-skew: high medical expenses incurred by a minority of high-risk patients).

---

## 3. Data Preprocessing & Exploratory Data Analysis (EDA)

### 3.1 Categorical Encoding
Machine learning algorithms rely on linear algebraic computations and require numeric inputs:
1. **Binary Mapping:**
   - `sex`: $\text{male} \rightarrow 0, \text{female} \rightarrow 1$
   - `smoker`: $\text{no} \rightarrow 0, \text{yes} \rightarrow 1$
2. **One-Hot Encoding for Nominal Variable (`region`):**
   - Since geographic regions lack ordinal hierarchy, applying integer encoding (0, 1, 2, 3) introduces an artificial distance metric.
   - We applied one-hot encoding with `drop_first=True` to eliminate multi-collinearity (the **dummy variable trap**). The `northeast` region serves as the baseline reference category.

### 3.2 Key Exploratory Observations
1. **Smoking Impact:** The median expense for non-smokers is ~$7,345, whereas smokers exhibit a median expense of ~$34,456 (nearly 5x higher).
2. **BMI Interaction:** For non-smokers, an increase in BMI causes a slight, gradual rise in medical costs. However, for smokers with BMI $\ge 30$ (obese), costs surge exponentially above $35,000–$45,000.
3. **Age Progression:** Medical charges demonstrate a consistent upward baseline trend across aging cohorts.

---

## 4. Model Architecture & Training

### 4.1 Linear Regression Formulation
The multiple linear regression equation is formalized as:
$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j + \epsilon$$

Where:
- $\beta_0$: Intercept term (baseline cost).
- $\beta_j$: Weight / sensitivity coefficient for feature $j$.
- $X_j$: Feature inputs (age, sex, bmi, children, smoker, region dummies).

### 4.2 Train/Test Partitioning
- **Split Ratio:** 80% Training (1,070 samples) / 20% Testing (268 samples)
- **Random Seed:** `random_state = 42` for exact reproducibility.

---

## 5. Experimental Results & Performance Comparison

Models were evaluated on the unseen test set using three standard regression metrics:
1. **Coefficient of Determination ($R^2$):** Measures variance explained by the model.
2. **Mean Absolute Error (MAE):** Average magnitude of absolute error in dollars.
3. **Root Mean Squared Error (RMSE):** Penalizes larger outlier mistakes heavily.

| Metric | Ordinary Linear Regression (OLS) | Random Forest Regressor ($n=200$) | Percentage Improvement |
|---|---|---|---|
| **$R^2$ Score** | **0.7836** | **0.8656** | **+10.46%** |
| **MAE ($)** | **$4,181.19** | **$2,546.83** | **-39.09%** |
| **RMSE ($)** | **$5,796.28** | **$4,568.42** | **-21.18%** |

### Feature Importance & Coefficients (Linear Regression)
- `smoker`: **+$23,651.13** (Dominant factor)
- `age`: **+$256.98 / year**
- `bmi`: **+$337.09 / unit**
- `children`: **+$425.28 / dependent**
- `region_southeast`: **-$1,029.00**

---

## 6. Discussion & Critical Observations
1. **Non-Linear Interaction:** The relationship between smoker status and BMI is interactive ($\text{Smoker} \times \text{BMI}$) rather than strictly additive. Because plain OLS assumes independent additive hyperplanes, it systematically underpredicts expenses for obese smokers and overpredicts for underweight smokers.
2. **Target Skewness:** Highly skewed target distributions ($+1.515$) lead to heteroscedastic residuals at higher expense levels ($>\$30,000$).
3. **Tree-Based Ensembles:** Random Forest naturally splits on non-linear thresholds, isolating the high-risk obese-smoker subset and improving the $R^2$ from 0.78 to 0.86.

---

## 7. Web Application Deployment
To demonstrate practical utility, the trained pipeline was serialized into `model.pkl` and `columns.pkl` using `joblib`. A production-ready **Streamlit web application** (`app.py`) was developed featuring:
- Sliders and dropdowns for intuitive user input (age, sex, BMI, children, smoking, region).
- Input sanity checks and BMI categorization indicators.
- Instantaneous real-time annual charge estimation with health advisory notices.

---

## 8. Conclusion & Future Work
- **Conclusion:** A robust supervised learning framework was implemented, capable of explaining up to 86.5% of variance in medical insurance charges. Smoking status was quantitatively proven to be the primary cost driver.
- **Future Directions:**
  1. Incorporate polynomial interaction features ($\text{BMI} \times \text{Smoker}$) into the linear pipeline.
  2. Implement gradient boosted decision trees (XGBoost / LightGBM) with hyperparameter tuning.
  3. Expand the dataset to include medical history indicators (e.g., pre-existing conditions, family medical history).

---

## References
1. Mirichoi0218, *Medical Cost Personal Datasets*, Kaggle, 2018.
2. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, *An Introduction to Statistical Learning with Applications in R*, Springer, 2013.
3. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
