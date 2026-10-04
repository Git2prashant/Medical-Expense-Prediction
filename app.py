"""
Medical Expense Predictor — Streamlit Web App
==============================================
A web interface for predicting annual medical expenses
using a trained Linear Regression model.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Medical Expense Predictor",
    page_icon="🏥",
    layout="centered"
)

# ── Custom Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🏥 Medical Expense Predictor")
st.markdown("*Predict annual medical insurance charges using Machine Learning*")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ── Load Model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    columns_path = os.path.join(os.path.dirname(__file__), "columns.pkl")

    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please run the Jupyter notebook first to train and save the model.")
        st.info("Run the notebook `medical_expense_prediction.ipynb` → Step 10 to save `model.pkl` and `columns.pkl`.")
        st.stop()

    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    return model, columns


model, columns = load_model()

# ── Input Form ──────────────────────────────────────────────────
st.subheader("📋 Enter Person Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", min_value=18, max_value=64, value=30, step=1)
    sex = st.selectbox("👤 Sex", ["male", "female"])
    bmi = st.number_input("⚖️ BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1,
                          help="Body Mass Index (normal: 18.5–24.9)")

with col2:
    children = st.slider("👶 Number of Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("🚬 Smoker", ["no", "yes"])
    region = st.selectbox("📍 Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown("---")

# ── Prediction ──────────────────────────────────────────────────
if st.button("🔮 Predict Medical Expense", use_container_width=True, type="primary"):

    # Build input row
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
    prediction = model.predict(X_new)[0]

    # Display result
    st.markdown(f"""
    <div class="prediction-box">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Estimated Annual Medical Expense</p>
        <p class="prediction-amount">${prediction:,.2f}</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Based on Linear Regression model</p>
    </div>
    """, unsafe_allow_html=True)

    # Show input summary
    with st.expander("📊 Input Summary"):
        summary_df = pd.DataFrame({
            "Parameter": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Value": [age, sex, bmi, children, smoker, region]
        })
        st.table(summary_df)

    # Insights
    if smoker == "yes":
        st.warning("🚬 **Smoking significantly increases medical expenses.** "
                    "The model shows smokers pay ~$23,000 more on average.")
    if bmi > 30:
        st.info(f"⚖️ **BMI of {bmi} is in the obese range** (>30). "
                "Higher BMI is associated with increased medical costs.")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85rem;'>"
    "Medical Expense Prediction — Minor Project | "
    "Built with Scikit-learn & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
