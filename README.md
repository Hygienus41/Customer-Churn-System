# Customer Churn Prediction & Retention Analytics System

An end-to-end machine learning system for predicting customer churn, identifying high-risk customers, estimating revenue at risk, and generating actionable customer retention recommendations.

**Live Demo:** https://customer-churn-system-gdaz6gakeygvwc3gcx4etw.streamlit.app  
**GitHub:** https://github.com/Hygienus41/Customer-Churn-System

---

## 📌 Project Overview

Customer churn is a major business challenge because losing existing customers can directly affect recurring revenue and growth.

This project goes beyond simply predicting whether a customer will churn. It transforms churn predictions into **business-oriented retention intelligence** by identifying who is at risk, why they may be at risk, which customers should be prioritized, and how much revenue may be exposed.

The system covers the complete workflow:

**Raw Data → Cleaning → Feature Engineering → Model Development → Optimization → Prediction → Risk Segmentation → Explainability → Revenue Risk → Retention Actions**

---

## 🎯 Business Objectives

The system is designed to help businesses:

- Identify customers with a high probability of churn
- Segment customers by risk level
- Prioritize retention interventions
- Estimate expected monthly and annual revenue at risk
- Understand the factors driving individual predictions
- Analyze customer portfolios in batches
- Turn machine learning predictions into actionable business recommendations

---

## 🧠 Machine Learning Workflow

### 1. Data Preparation

- Loaded and inspected raw customer data
- Handled missing and inconsistent values
- Cleaned and transformed variables
- Prepared features for machine learning

### 2. Feature Engineering

Business-relevant features were created to improve predictive value, including:

- Customer tenure
- Monthly charges
- Total services
- Contract duration
- Senior citizen status
- Customer service/product characteristics

### 3. Model Development

Multiple machine learning approaches were evaluated before selecting and optimizing the final model.

The final pipeline uses:

- **XGBoost**
- **scikit-learn**
- **ColumnTransformer**
- **StandardScaler**
- **OneHotEncoder**

### 4. Model Optimization

The final model and classification threshold were optimized with attention to the business objective of identifying customers who are at risk of leaving.

### 5. Explainability

**SHAP** is used to explain individual predictions and identify the features contributing to churn risk.

---

## 📊 Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 77.9986% |
| Precision | 56.67% |
| Recall | 72.73% |
| F1 Score | 63.70% |
| ROC-AUC | 84.76% |

The model achieved **72.73% recall**, helping identify a substantial proportion of customers who are likely to churn, while maintaining an overall ROC-AUC of **84.76%**.

---

## 🚦 Customer Risk Framework

Customers are classified into three business risk levels:

| Risk Level | Churn Probability | Recommended Action |
|---|---:|---|
| 🟢 Low Risk | < 40% | Maintain engagement |
| 🟡 Medium Risk | 40%–69.99% | Monitor and engage |
| 🔴 High Risk | ≥ 70% | Immediate retention intervention |

This framework converts model probabilities into practical retention priorities.

---

## 💰 Revenue-at-Risk Analysis

The system estimates the financial exposure associated with predicted churn.

### Expected Monthly Revenue at Risk

**Churn Probability × Monthly Charges**

### Expected Annual Revenue at Risk

**Expected Monthly Revenue at Risk × 12**

This allows decision-makers to move beyond:

> "Which customers might churn?"

and ask:

> "Which customers should we prioritize, and what revenue is potentially exposed?"

---

## 🔍 Explainable Machine Learning

The application provides SHAP-based explanations for predictions.

This helps users understand:

- Which features increase churn risk
- Which features reduce churn risk
- Why a specific customer received a particular risk score
- Which factors may deserve attention during retention planning

---

## 🖥️ Application Features

The Streamlit application includes:

### Dashboard
High-level overview of customer risk, predicted churn, and revenue exposure.

### Individual Prediction
Enter customer information and generate an individual churn prediction, risk level, and business interpretation.

### Portfolio Analysis
Upload a customer CSV file and generate predictions across the entire customer portfolio.

### Retention Prioritization
Rank customers according to retention priority so limited intervention resources can be focused where they may have the greatest business value.

### Revenue Risk
Analyze expected monthly and annual revenue at risk.

### Model Information
Review model performance and supporting machine learning information.

---
## 🖼️ Application Preview

The application provides multiple views covering the complete customer churn analytics workflow.

### Dashboard

[![Dashboard Overview](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/dashboard%200%20.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/dashboard%200%20.png)

[![Dashboard](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/dashboard%200.1.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/dashboard%200.1.png)

[![Dashboard Analysis](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/dashboard%201.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/dashboard%201.png)

[![Dashboard Insights](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/dashboard%202.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/dashboard%202.png)

### Individual Customer Prediction

[![Individual Customer Prediction Input](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/individual%20customer%20prediction%201.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/individual%20customer%20prediction%201.png)

[![Individual Customer Prediction](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/individual%20customer%20prediction%202.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/individual%20customer%20prediction%202.png)

[![Individual Customer Prediction Result](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/individual%20customer%20prediction%20result.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/individual%20customer%20prediction%20result.png)

### Portfolio Analysis

[![Portfolio Analysis](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/portfolio%20analysis.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/portfolio%20analysis.png)

### Retention Prioritization

[![Retention Prioritization](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/retention_prioritization.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/retention_prioritization.png)

### Model Information

[![Model Information](https://raw.githubusercontent.com/Hygienus41/Customer-Churn-System/main/screenshots/model%20infor.png)](https://github.com/Hygienus41/Customer-Churn-System/blob/main/screenshots/model%20infor.png)

---

## 🛠️ Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **XGBoost**
- **SHAP**
- **Matplotlib**
- **Streamlit**
- **Joblib**
- **Git & GitHub**

---

## 📁 Project Structure

```text
Customer-Churn-System/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── Telco_Customer_Churn.csv
│   └── cleaned/
│       └── churn_clean.csv
│
├── models/
│   ├── final_churn_model.pkl
│   ├── preprocessor.pkl
│   ├── final_threshold.pkl
│   └── gradient_boosting_baseline.pkl
│
├── src/
│   ├── prediction_pipeline.py
│   └── explainability.py
│
├── screenshots/
│   ├── dashboard.png
│   ├── individual_prediction.png
│   ├── portfolio_analysis.png
│   ├── retention_prioritization.png
│   └── explainability.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Hygienus41/Customer-Churn-System.git
cd Customer-Churn-System
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## 📋 Example Portfolio Workflow

1. Upload a customer CSV dataset.
2. The application validates and processes the data.
3. The prediction pipeline generates churn probabilities.
4. Customers are assigned risk levels.
5. Retention priority scores are calculated.
6. Revenue-at-risk estimates are generated.
7. SHAP explanations provide insight into predictions.
8. Results are presented through the Streamlit dashboard.

---

## 💼 Business Value

This project demonstrates how machine learning can be integrated into a practical business workflow.

Instead of stopping at model prediction, the system connects:

**Prediction → Risk → Priority → Financial Exposure → Action**

This makes the output useful for customer success, marketing, sales, revenue management, and retention teams.

---

## 🚀 Future Improvements

Potential next steps include:

- Automated model monitoring
- Model retraining pipelines
- Additional customer segmentation
- A/B testing of retention strategies
- Cost-sensitive optimization
- Integration with CRM systems
- Automated retention campaign recommendations
- More extensive model monitoring and drift detection

---

## 👨‍💻 Project Purpose

This project was developed as a practical portfolio demonstration of end-to-end data science and machine learning, with an emphasis on connecting technical modeling to measurable business decisions.

**Core focus:** Data Science • Machine Learning • Predictive Analytics • Business Intelligence • Explainable AI
