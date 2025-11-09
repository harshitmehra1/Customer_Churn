# src/app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ---------------------------
# 🎯 PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Telco Churn Prediction Dashboard", page_icon="📞", layout="wide")

st.title("📞 Telco Customer Churn Prediction Dashboard")

st.markdown("---")

st.markdown("""
Welcome to the **AI-powered Customer Retention Assistant**!  
This dashboard helps telecom teams:
- **Predict churn** for individual customers  
- **Compare ML models** (Random Forest vs XGBoost)  
- **Understand why customers may leave**  
- **Support business decisions** using interpretable ML insights  
""")

st.info("""
**About the Data:**  
This dashboard uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).  
It contains telecom customer details such as tenure, contract type, internet service, and churn behavior.
""")

# ---------------------------
# 🧭 NAVIGATION
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Predict Churn", "📊 Model Insights", "⚖️ Model Comparison"])

# ---------------------------
# 🧩 LOAD MODELS
# ---------------------------
try:
    rf_model = joblib.load("models/rf_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    st.sidebar.success("✅ Models and scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# Load and preprocess data
df = pd.read_csv("data/Customer-Churn.csv").dropna()
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

selected_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "Contract", "InternetService", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

X = df[selected_features]
y = df["Churn"]
X_scaled = scaler.transform(X)

# ---------------------------
# ⚖️ MODEL COMPARISON TAB
# ---------------------------
with tab3:
    st.header("⚖️ Model Comparison Dashboard")

    models = {"Random Forest": rf_model, "XGBoost": xgb_model}
    results = []

    for name, model in models.items():
        preds = model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results).set_index("Model")

    st.subheader("📋 Performance Metrics Comparison")
    st.dataframe(results_df.style.format("{:.2%}"))

    st.subheader("📊 Visual Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    results_melted = results_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(x="Metric", y="Score", hue="Model", data=results_melted, palette="viridis", ax=ax)
    plt.ylabel("Score")
    plt.title("Random Forest vs XGBoost Performance")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("💡 Recommendation for Business Teams")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest")
        st.markdown("""
        - Higher **F1-score** and **recall**  
        - Catches more actual churners  
        - Best for **customer retention** focus  
        - Use when your goal is to **minimize lost customers**
        """)

    with col2:
        st.markdown("### ⚡ XGBoost")
        st.markdown("""
        - Slightly better **accuracy** and **precision**  
        - Fewer false alarms  
        - Great for **profit optimization**  
        - Use when you want to **target only likely churners**
        """)

    st.success("✅ Tip: Choose your model based on your **business goal**, not just accuracy!")

# ---------------------------
# 📊 MODEL INSIGHTS TAB
# ---------------------------
with tab2:
    st.header("🔍 Model Insights")

    model_choice = st.selectbox("Select Model to Inspect", ["Random Forest", "XGBoost"])
    model = rf_model if model_choice == "Random Forest" else xgb_model

    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc * 100:.2f}%")
    col2.metric("F1 Score", f"{f1 * 100:.2f}%")

    importances = pd.DataFrame({
        "Feature": selected_features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("📈 Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importances.head(10), palette="viridis", ax=ax, legend=False)
    st.pyplot(fig)

# ---------------------------
# 🎯 PREDICTION TAB
# ---------------------------
with tab1:
    st.header("🧮 Predict Customer Churn")

    model_choice = st.radio("Select Prediction Model", ["Random Forest", "XGBoost"], horizontal=True)
    model = rf_model if model_choice == "Random Forest" else xgb_model

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 150.0, 70.0)
        total_charges = tenure * monthly_charges
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])

    if st.button("🔍 Predict Churn"):
        data = {
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "Contract": ["Month-to-month", "One year", "Two year"].index(contract),
            "InternetService": ["DSL", "Fiber optic", "No"].index(internet_service),
            "PaymentMethod": ["Electronic check", "Credit card", "Bank transfer", "Mailed check"].index(payment_method),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        input_df = pd.DataFrame([data])
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100

        # ---------------------------
        # 🧠 Dynamic Explanation Logic
        # ---------------------------
        risk_reasons = []
        stay_reasons = []

        # Tenure
        if tenure < 12:
            risk_reasons.append("🔥 Customer is new (< 1 year tenure)")
        elif tenure > 36:
            stay_reasons.append("💎 Loyal customer with long tenure")

        # Contract
        if contract == "Month-to-month":
            risk_reasons.append("🧾 Month-to-month contracts often lead to churn")
        elif contract in ["One year", "Two year"]:
            stay_reasons.append("📅 Long-term contract shows commitment")

        # Internet service
        if internet_service == "Fiber optic":
            risk_reasons.append("⚡ Fiber optic users show higher churn due to higher prices")
        elif internet_service == "DSL":
            stay_reasons.append("🌐 DSL users are relatively more stable")

        # Monthly charges
        if monthly_charges > 100:
            risk_reasons.append("💰 High monthly bill may lead to dissatisfaction")
        elif monthly_charges < 50:
            stay_reasons.append("💵 Affordable monthly plan increases satisfaction")

        # Payment method
        if payment_method == "Electronic check":
            risk_reasons.append("🏦 Electronic check users tend to churn more often")
        else:
            stay_reasons.append("💳 Consistent payment method (Credit/Bank) indicates reliability")

        # ---------------------------
        # 🧾 Display Results
        # ---------------------------
        if pred == 1:
            st.error(f"🔴 High risk of churn ({prob:.1f}%)")
            st.markdown("**Possible reasons for churn:**")
            if risk_reasons:
                for r in risk_reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- Short engagement or higher service charges")
        else:
            st.success(f"🟢 Low risk of churn ({prob:.1f}%)")
            st.markdown("**Why this customer might stay:**")
            if stay_reasons:
                for r in stay_reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- Good service satisfaction and stability")

# ---------------------------
# 📘 FOOTER
# ---------------------------
# st.markdown("---")

st.markdown("""
<hr>
<div style='text-align:center; font-size:0.9em; color:gray;'>
    Built with ❤️ using Streamlit, MLflow, Scikit-learn & XGBoost | Created by <b>Harshit Mehra</b>
</div>
""", unsafe_allow_html=True)