"""
Generate all EDA and model evaluation plots as PNG files.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

df = pd.read_csv("insurance.csv")

# 1. Distribution of charges
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["charges"], kde=True, color="#4361ee", ax=axes[0], bins=40)
axes[0].set_title("Distribution of Medical Charges", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Charges ($)")
axes[0].axvline(df["charges"].mean(), color="red", linestyle="--", label=f"Mean: ${df['charges'].mean():,.0f}")
axes[0].axvline(df["charges"].median(), color="green", linestyle="--", label=f"Median: ${df['charges'].median():,.0f}")
axes[0].legend()

sns.boxplot(y=df["charges"], color="#4361ee", ax=axes[1])
axes[1].set_title("Box Plot of Charges", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Charges ($)")
plt.tight_layout()
plt.savefig("charges_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Correlation heatmap
df_corr = df.copy()
df_corr["sex"] = df_corr["sex"].map({"male": 0, "female": 1})
df_corr["smoker"] = df_corr["smoker"].map({"no": 0, "yes": 1})
df_corr = pd.get_dummies(df_corr, columns=["region"], drop_first=True).astype(float)

plt.figure(figsize=(10, 8))
corr_matrix = df_corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Categorical vs charges
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(x="smoker", y="charges", data=df, palette=["#2ec4b6", "#e71d36"], ax=axes[0])
axes[0].set_title("Smoker vs Charges", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Smoker")
axes[0].set_ylabel("Charges ($)")

sns.boxplot(x="sex", y="charges", data=df, palette=["#4361ee", "#f72585"], ax=axes[1])
axes[1].set_title("Sex vs Charges", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("Charges ($)")

sns.boxplot(x="region", y="charges", data=df, palette="Set2", ax=axes[2])
axes[2].set_title("Region vs Charges", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Region")
axes[2].set_ylabel("Charges ($)")
plt.tight_layout()
plt.savefig("categorical_vs_charges.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Age & Children vs Charges
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="children", y="charges", data=df, palette="viridis", ax=axes[0])
axes[0].set_title("Children vs Charges", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of Children")
axes[0].set_ylabel("Charges ($)")

sns.scatterplot(x="age", y="charges", hue="smoker", data=df,
                palette=["#2ec4b6", "#e71d36"], alpha=0.7, ax=axes[1])
axes[1].set_title("Age vs Charges (by Smoker Status)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Charges ($)")
plt.tight_layout()
plt.savefig("age_children_vs_charges.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. BMI vs Charges
plt.figure(figsize=(10, 6))
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df,
                palette=["#2ec4b6", "#e71d36"], alpha=0.7, s=60)
plt.title("BMI vs Charges (by Smoker Status)", fontsize=15, fontweight="bold")
plt.xlabel("BMI")
plt.ylabel("Charges ($)")
plt.tight_layout()
plt.savefig("bmi_vs_charges.png", dpi=150, bbox_inches="tight")
plt.close()

# Model training & evaluations
X = df_corr.drop("charges", axis=1)
y = df_corr["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 6. Coefficients
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_}).sort_values("Coefficient", ascending=False)
plt.figure(figsize=(10, 5))
colors = ["#e71d36" if c > 0 else "#4361ee" for c in coef_df["Coefficient"]]
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
plt.xlabel("Coefficient Value ($)")
plt.title("Linear Regression Feature Coefficients", fontsize=14, fontweight="bold")
plt.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("coefficients.png", dpi=150, bbox_inches="tight")
plt.close()

# 7. Actual vs Predicted
plt.figure(figsize=(8, 7))
plt.scatter(y_test, y_pred_lr, alpha=0.6, color="#4361ee", edgecolors="white", s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2, label="Perfect Prediction Line")
plt.xlabel("Actual Charges ($)", fontsize=12)
plt.ylabel("Predicted Charges ($)", fontsize=12)
plt.title("Actual vs Predicted Medical Charges", fontsize=15, fontweight="bold")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# 8. Residual Analysis
residuals = y_test - y_pred_lr
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_lr, residuals, alpha=0.5, color="#4361ee", edgecolors="white", s=40)
axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Predicted Charges ($)")
axes[0].set_ylabel("Residual ($)")
axes[0].set_title("Residuals vs Predicted Values", fontsize=14, fontweight="bold")

sns.histplot(residuals, kde=True, color="#4361ee", ax=axes[1], bins=30)
axes[1].set_xlabel("Residual ($)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Residuals", fontsize=14, fontweight="bold")
axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
plt.tight_layout()
plt.savefig("residual_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# 9. Model comparison
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_test, y_pred_lr, alpha=0.6, color="#4361ee", edgecolors="white", s=40)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
axes[0].set_xlabel("Actual ($)")
axes[0].set_ylabel("Predicted ($)")
axes[0].set_title(f"Linear Regression (R²={r2_score(y_test, y_pred_lr):.3f})", fontsize=14, fontweight="bold")

axes[1].scatter(y_test, y_pred_rf, alpha=0.6, color="#2ec4b6", edgecolors="white", s=40)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
axes[1].set_xlabel("Actual ($)")
axes[1].set_ylabel("Predicted ($)")
axes[1].set_title(f"Random Forest (R²={r2_score(y_test, y_pred_rf):.3f})", fontsize=14, fontweight="bold")
plt.suptitle("Model Comparison: Actual vs Predicted", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("All 9 plot images saved successfully!")
