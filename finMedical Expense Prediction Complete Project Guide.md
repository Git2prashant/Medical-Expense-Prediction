# Medical Expense Prediction: Complete Project Guide

> Predict a person's annual medical insurance charges using Linear Regression.

---

## 1. What is this project?

Insurance companies don't charge everyone the same. Your premium and expected medical cost depend on things like your age, BMI, whether you smoke, and so on.

In this project you build a **machine learning model** that takes a person's details and **predicts their yearly medical charges (in dollars)**.

- **Type of problem:** Supervised learning, specifically **Regression** (the output is a continuous number, not a category)
- **Algorithm:** Linear Regression
- **Input:** age, sex, BMI, number of children, smoker, region
- **Output:** predicted annual medical expense

---

## 2. The Dataset

Use the **Medical Cost Personal Dataset** (commonly called `insurance.csv`). Search for it on Kaggle by that name. It has about **1,338 rows** and **7 columns**.

| Column | Type | Meaning |
|---|---|---|
| `age` | Numeric | Age of the person |
| `sex` | Categorical | male / female |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Number of dependents covered |
| `smoker` | Categorical | yes / no |
| `region` | Categorical | northeast, northwest, southeast, southwest |
| `charges` | Numeric | **Target.** Annual medical cost billed by insurance |

**Features (X)** = age, sex, bmi, children, smoker, region
**Target (y)** = charges

---

## 3. How the whole thing works (big picture)

```
 insurance.csv
      |
      v
 Load + Explore (EDA)
      |
      v
 Encode categorical columns (sex, smoker, region -> numbers)
      |
      v
 Split into Train (80%) and Test (20%)
      |
      v
 Train Linear Regression on Train data
      |
      v
 Predict on Test data
      |
      v
 Compare Predicted vs Actual (table + plot + metrics)
      |
      v
 Predict for a brand-new person (final output)
```

---

## 4. Core concepts you need to understand

### 4.1 Why encode categorical variables?

Linear Regression is just math: it multiplies each feature by a weight and adds them up. It can't multiply the word `"yes"` by a number. So text columns must become numbers.

### 4.2 Which encoding for which column?

| Column | Unique values | Encoding | Why |
|---|---|---|---|
| `sex` | 2 | Binary map (0/1) | Only two options |
| `smoker` | 2 | Binary map (0/1) | Only two options |
| `region` | 4 | **One-hot encoding** | No natural order between regions |

**Why not just label regions 0, 1, 2, 3?**
Because the model would think `southwest (3)` is "bigger" than `northeast (0)`, which is meaningless. One-hot encoding creates a separate 0/1 column for each region, so no fake ordering is introduced.

**What is `drop_first=True`?**
With 4 regions, if you know 3 of the one-hot columns, the 4th is already determined. Keeping all 4 creates redundant information (called the *dummy variable trap*). Dropping one avoids it.

### 4.3 How Linear Regression works

It fits an equation like:

```
charges = b0 + b1*age + b2*sex + b3*bmi + b4*children + b5*smoker + b6*region_northwest + ...
```

- `b0` is the **intercept** (baseline)
- `b1, b2, ...` are **coefficients** (weights). Each tells you how much `charges` changes when that feature increases by 1, with everything else held constant.

Training = finding the weights that make predictions as close as possible to the actual charges (it minimizes the squared error).

### 4.4 Why split into train and test?

If you test the model on the same data it learned from, it looks great but tells you nothing about new people. So you hide 20% of the data (test set), train on 80%, then check how well it predicts the hidden 20%.

### 4.5 The evaluation metrics

| Metric | What it tells you |
|---|---|
| **R² score** | How much of the variation in charges your model explains. 1.0 is perfect, 0 is useless. |
| **MAE** (Mean Absolute Error) | Average error in dollars. Easy to understand. |
| **RMSE** (Root Mean Squared Error) | Like MAE but punishes big mistakes more. |

---

## 5. Setup

### Install libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Folder structure

```
medical-expense-prediction/
├── insurance.csv
├── medical_expense_prediction.ipynb
└── README.md  (optional)
```

---

## 6. Step-by-step implementation

### Step 1: Import libraries and load data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("insurance.csv")
df.head()
```

### Step 2: Explore the data (EDA)

```python
df.info()              # column types, missing values
df.describe()          # stats for numeric columns
df.isnull().sum()      # check missing values (this dataset has none)
```

Useful plots:

```python
# Distribution of the target
sns.histplot(df["charges"], kde=True)
plt.title("Distribution of Charges")
plt.show()

# Smoker vs charges (you'll see a huge gap)
sns.boxplot(x="smoker", y="charges", data=df)
plt.show()

# BMI vs charges, colored by smoker
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df)
plt.show()
```

**What you'll notice:**
- Smokers pay far more than non-smokers
- Charges are right-skewed (a few very expensive cases)
- Smoker plus high BMI leads to the highest charges

### Step 3: Encode categorical variables

```python
# Binary columns
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

# One-hot encode region
df = pd.get_dummies(df, columns=["region"], drop_first=True)

df.head()
```

After this, `region` becomes 3 columns: `region_northwest`, `region_southeast`, `region_southwest`. (If all three are 0, the region is northeast.)

Quick check that everything is numeric:

```python
print(df.dtypes)
```

If some `region_*` columns show as `bool`, convert them: `df = df.astype(float)`.

### Step 4: Split into features and target, then train/test

```python
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)
```

`random_state=42` makes the split reproducible, so you get the same results every run.

### Step 5: Train Linear Regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

Look at what the model learned:

```python
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("Intercept:", model.intercept_)
print(coef_df)
```

You'll see `smoker` has by far the largest coefficient. That's your key insight for the report.

### Step 6: Predict on the test set

```python
y_pred = model.predict(X_test)
```

### Step 7: Compare predicted vs actual

**A) Table**

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
})
comparison["Error"] = comparison["Actual"] - comparison["Predicted"]
comparison.head(10)
```

**B) Scatter plot (actual vs predicted)**

```python
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.legend()
plt.show()
```

**How to read it:** points close to the red dashed line mean good predictions. Points far from it are the model's mistakes.

**C) Metrics**

```python
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score : {r2:.3f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
```

**What to expect:** R² around **0.75**, MAE around **4,000**. That's normal for this dataset with a plain linear model.

### Step 8: Final output, predict for a new person

This is the "Expected Output: Medical expense prediction" part.

```python
new_person = pd.DataFrame([{
    "age": 35,
    "sex": 0,                  # 0 = male, 1 = female
    "bmi": 28.5,
    "children": 2,
    "smoker": 1,               # 1 = yes, 0 = no
    "region_northwest": 0,
    "region_southeast": 1,
    "region_southwest": 0,
}])

# Make sure the column order matches training data
new_person = new_person[X.columns]

prediction = model.predict(new_person)
print(f"Predicted annual medical expense: ${prediction[0]:,.2f}")
```

---

## 7. Why the model isn't perfect (and how to explain it)

Linear Regression assumes a **straight-line** relationship between features and charges. But in the real data:

- Charges for smokers jump sharply as BMI rises (an **interaction effect**), which a straight line can't capture
- The target is **skewed**, with a few very high values pulling the fit

This is why you'll see the points spread out from the diagonal line in the plot, especially for high charges. It's a good point to write in your conclusion: it shows you understand the model's limits.

---

## 8. Optional upgrades (to stand out)

### 8.1 Log-transform the target

```python
y_log = np.log(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

model_log = LinearRegression().fit(X_train, y_train)
y_pred_log = np.exp(model_log.predict(X_test))   # convert back to dollars
y_actual = np.exp(y_test)

print("R2:", r2_score(y_actual, y_pred_log))
```

### 8.2 Compare with a better model

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest R2:", r2_score(y_test, rf.predict(X_test)))
```

If you run this on the original (non-log) target, you'll typically see R² around 0.85, which shows how non-linear models handle the smoker/BMI interaction better. Use the original `y` split for a fair comparison with Linear Regression.

### 8.3 Streamlit web app

Install: `pip install streamlit joblib`

First, save the trained model in your notebook:

```python
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")
```

Then create `app.py`:

```python
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("Medical Expense Predictor")

age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 15.0, 55.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict"):
    row = {
        "age": age,
        "sex": 0 if sex == "male" else 1,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
    }
    X_new = pd.DataFrame([row])[columns]
    pred = model.predict(X_new)[0]
    st.success(f"Estimated annual medical expense: ${pred:,.2f}")
```

Run it:

```bash
streamlit run app.py
```

---

## 9. What to submit

The project details don't specify a format, so confirm with your faculty. The usual safe package is:

| Item | Required? |
|---|---|
| Jupyter notebook (`.ipynb`) with code, outputs and plots | Yes |
| `insurance.csv` dataset | Yes |
| Short report or PPT (problem, approach, results, conclusion) | Likely |
| Streamlit app (`app.py`, `model.pkl`) | Bonus |

### Suggested report outline

1. **Problem statement**: predict annual medical expenses
2. **Dataset description**: columns, size, source
3. **Preprocessing**: encoding of sex, smoker and region
4. **Model**: Linear Regression, train/test split
5. **Results**: R², MAE, RMSE, actual vs predicted plot
6. **Observations**: smoking is the biggest factor, and the linear model misses non-linear patterns
7. **Conclusion and future work**: log transform, Random Forest, deployment

---

## 10. Quick checklist

- [ ] Load `insurance.csv` and run EDA
- [ ] Encode `sex` and `smoker` (0/1)
- [ ] One-hot encode `region` with `drop_first=True`
- [ ] Split into train/test (80/20)
- [ ] Train Linear Regression
- [ ] Predict on test set
- [ ] Show actual vs predicted (table + scatter plot)
- [ ] Report R², MAE, RMSE
- [ ] Predict for a custom new person
- [ ] (Bonus) Log transform, Random Forest comparison, Streamlit app
- [ ] Write the report or PPT

---

## 11. Common mistakes to avoid

| Mistake | Fix |
|---|---|
| Label-encoding `region` as 0, 1, 2, 3 | Use one-hot encoding |
| Forgetting `drop_first=True` | Add it to avoid the dummy variable trap |
| New-person input columns in the wrong order | Reorder with `new_person[X.columns]` |
| Evaluating on training data | Always evaluate on the test set |
| Expecting R² above 0.9 from Linear Regression | ~0.75 is normal for this dataset |
| Columns left as `bool` causing errors | Convert with `.astype(float)` |
