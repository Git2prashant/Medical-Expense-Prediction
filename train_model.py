"""
Train Medical Expense Prediction Model
Executes the data processing, model training, evaluation,
and saves model.pkl and columns.pkl.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

def main():
    print("Loading insurance.csv...")
    df = pd.read_csv("insurance.csv")
    print(f"Dataset shape: {df.shape}")

    # Encoding
    df_encoded = df.copy()
    df_encoded["sex"] = df_encoded["sex"].map({"male": 0, "female": 1})
    df_encoded["smoker"] = df_encoded["smoker"].map({"no": 0, "yes": 1})
    df_encoded = pd.get_dummies(df_encoded, columns=["region"], drop_first=True)
    df_encoded = df_encoded.astype(float)

    # Features and Target
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    print("\n--- Linear Regression Performance ---")
    print(f"R² Score : {r2_lr:.4f}")
    print(f"MAE      : ${mae_lr:,.2f}")
    print(f"RMSE     : ${rmse_lr:,.2f}")

    # Random Forest for comparison
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print("\n--- Random Forest Performance ---")
    print(f"R² Score : {r2_rf:.4f}")
    print(f"MAE      : ${mae_rf:,.2f}")
    print(f"RMSE     : ${rmse_rf:,.2f}")

    # Save model and columns
    joblib.dump(lr, "model.pkl")
    joblib.dump(list(X.columns), "columns.pkl")
    print("\n[OK] Saved model.pkl and columns.pkl successfully!")

    # Test single prediction
    new_person = pd.DataFrame([{
        "age": 35,
        "sex": 0,
        "bmi": 28.5,
        "children": 2,
        "smoker": 1,
        "region_northwest": 0,
        "region_southeast": 1,
        "region_southwest": 0,
    }])[X.columns]
    sample_pred = lr.predict(new_person)[0]
    print(f"Sample prediction for 35yo male smoker (BMI 28.5): ${sample_pred:,.2f}")

if __name__ == "__main__":
    main()
