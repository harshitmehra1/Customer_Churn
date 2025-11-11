



# 📞 Telco Customer Churn Prediction Dashboard

### 🔗 <a href="https://customerchurn-hmjyw8jnemjtefx85ri5pw.streamlit.app/" target="_blank">Live App Demo on Streamlit</a>

---

## 🌟 Project Overview

An **AI-powered Customer Retention Dashboard** built to predict telecom customer churn — i.e., which customers are likely to discontinue their service.  
The dashboard uses **Machine Learning models (Random Forest & XGBoost)** and provides **actionable insights** for business teams to make **data-driven retention decisions**.

This project demonstrates the **end-to-end data science workflow** — from preprocessing and model training to evaluation, explainability, and deployment using **Streamlit Cloud**.

---

## 📊 Problem Statement

Telecom companies face major revenue losses from customers who stop their services (“churn”).  
Predicting churn **before it happens** helps companies:
- Identify at-risk customers  
- Offer personalized retention offers  
- Reduce marketing costs  

The challenge:  
> Build an interpretable, reliable model that predicts **“Will this customer churn?”** based on their usage and contract data.

---

## 🧠 Dataset Details

**Dataset Source:** [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

**Rows:** ~7,000 customers  
**Columns:** 21 features  

### 📋 Key Features Used
| Feature | Description |
|:--|:--|
| `gender` | Male / Female |
| `SeniorCitizen` | 1 if senior citizen |
| `Partner`, `Dependents` | Family-related engagement |
| `tenure` | Number of months with company |
| `Contract` | Type of contract (Month-to-month, One year, Two year) |
| `InternetService` | DSL / Fiber optic / None |
| `PaymentMethod` | Type of payment (Electronic check, Credit card, etc.) |
| `MonthlyCharges`, `TotalCharges` | Spending metrics |
| `Churn` | Target variable (1 = churned, 0 = stayed) |

---


## ⚙️ Data Preprocessing

Steps performed to clean and prepare the raw dataset before model training:

1. **Removed irrelevant columns** — dropped `customerID` as it does not contribute to prediction.  
2. **Handled missing values** — converted `TotalCharges` to numeric and removed rows with null or invalid entries.  
3. **Encoded categorical features** — used `LabelEncoder` to convert string categories (e.g., gender, contract type) into numeric codes.  
4. **Split the dataset** — applied `train_test_split` (80% train, 20% test) with stratification to maintain class balance for churn vs. non-churn.  
5. **Feature scaling (in next step)** — numeric features were scaled later in `train_model.py` using `StandardScaler` to normalize feature ranges for better model performance.

> ✅ Result: A clean, encoded, and ML-ready dataset with balanced classes, ready for both Random Forest and XGBoost models.

---

## 🤖 Models Used and Why

### 🌲 Random Forest Classifier
- Great baseline model for tabular data  
- Handles both numeric and categorical data  
- Resistant to overfitting  
- Provides **feature importance** for explainability  

### ⚡ XGBoost Classifier
- Advanced gradient boosting algorithm  
- Usually achieves **higher accuracy and efficiency**  
- Learns complex relationships in data  
- Supports **regularization**, reducing overfitting  

> We chose both to demonstrate **contrast in performance** — Random Forest for interpretability, XGBoost for optimized accuracy.

---

## 🧪 Model Evaluation

| Metric | Random Forest | XGBoost |
|:--|--:|--:|
| **Accuracy** | 76–79% | 78–80% |
| **Precision** | 55–60% | 60–65% |
| **Recall** | 65–70% | 50–55% |
| **F1-score** | 60–62% | 56–58% |

### ✅ Insights:
- **Random Forest** has **higher Recall** → catches more churners  
- **XGBoost** has **higher Precision** → fewer false alarms  

So depending on business goal:
- 🎯 **Use Random Forest** for *customer retention focus*  
- 💰 **Use XGBoost** for *profit optimization focus*  

---

## 📈 Dashboard Features (Streamlit App)

The dashboard is divided into **3 interactive sections**:

### 1️⃣ 🎯 Predict Churn
- Enter customer details → get churn risk instantly  
- Displays clear **Low / Moderate / High risk** labels  
- Explains **why** the model made that prediction (e.g., “Short tenure”, “High monthly charges”)

### 2️⃣ 📊 Model Insights
- Shows top 10 **important features** driving churn  
- Displays **accuracy and F1-score** metrics  
- Helps understand model behavior visually  

### 3️⃣ ⚖️ Model Comparison
- Compares Random Forest vs XGBoost  
- Displays metrics: Accuracy, Precision, Recall, F1  
- Business recommendations included for choosing models  

---

## 🧮 Example Interpretations

| Scenario | Prediction | Explanation |
|:--|:--|:--|
| New customer, month-to-month, high bill | 🔴 High risk | Low tenure + flexible contract = likely to churn |
| Long-term, low charges, one-year contract | 🟢 Low risk | Loyal, affordable plan, stable customer |
| Moderate bill, medium tenure | 🟠 Moderate risk | Neutral — could stay or churn based on offers |

---

## 🧰 Tech Stack

| Category | Tools |
|:--|:--|
| Language | Python 3 |
| ML Frameworks | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Deployment | Streamlit Cloud |



---



## 📦 Installation (Run Locally)

```bash
# 1️⃣ Clone the repo
git clone https://github.com/<your-username>/Customer_Churn.git
cd Customer_Churn

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run src/app.py

```


## 🚀 Deployment

- **Platform:** Streamlit Cloud  
- **Model files:** Stored in `/models` folder  
- **Dependencies:** Managed via `requirements.txt`  
- **GitHub Actions:** Used for CI/CD (future integration possible)  

> Simply push updates → Streamlit automatically redeploys your app.

---

## 🧩 Challenges Faced & Solutions

| Challenge | Solution |
|:--|:--|
| Imbalanced churn data | Used stratified train-test split |
| Encoding categorical variables | Used LabelEncoder for consistent feature mapping |
| Streamlit deployment errors | Adjusted `.gitignore` and included `.pkl` model files |
| Model interpretability | Added feature importance & churn explanation logic |
| Confusion between churn risk vs churn rate | Clarified with clear on-screen messages |

---

## 🧭 Key Learnings

- Hands-on understanding of **classification pipelines**
- Experience in **model evaluation beyond accuracy**
- Building an **interactive ML dashboard for non-technical users**
- Real-world deployment using **Streamlit Cloud**
- Effective storytelling with **business-oriented ML results**

---

### 🏁 Final Takeaway

> “This project isn’t just about predicting churn — it’s about **turning machine learning insights into business strategy**.  
> It demonstrates how data science directly drives decision-making.”

---


## 👤 Author

**Harshit Mehra**   
💼  Data Scientist  

📎 [LinkedIn](https://www.linkedin.com/in/harshitmehra1/)  
📎 [Streamlit Demo](https://customerchurn-hmjyw8jnemjtefx85ri5pw.streamlit.app/)

---

## 🌟 Support

If you found this project helpful:
⭐ **Star this repository**  
☕ **Share it with peers**  
💬 Feedback always welcome!

---





