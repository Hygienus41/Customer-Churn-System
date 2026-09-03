# Customer Churn Prediction & Retention Analytics System

An end-to-end machine learning system for predicting customer churn, identifying high-risk customers, estimating revenue at risk, and generating actionable customer retention recommendations.

**Live Demo:** https://customer-churn-system-gdaz6gakeygvwc3gcx4etw.streamlit.app

---

## 📌 Project Overview

Customer churn is a major business challenge for subscription-based and service-oriented companies. Identifying customers who are likely to leave allows organizations to intervene early, prioritize retention efforts, and protect recurring revenue.

This project develops a complete **Customer Churn Prediction and Retention Analytics System** that combines machine learning with business intelligence.

The system goes beyond simply predicting whether a customer will churn. It transforms model predictions into:

* Churn probabilities
* Customer risk segmentation
* Revenue-at-risk estimates
* Retention priority scores
* Recommended retention actions
* Explainable predictions using SHAP
* Portfolio-level churn analytics

The final solution is deployed as an interactive **Streamlit web application**.

---

## 🎯 Business Problem

Businesses need to answer questions such as:

* Which customers are most likely to churn?
* How likely is each customer to leave?
* Which customers require immediate attention?
* How much recurring revenue is potentially at risk?
* Which customers should retention teams prioritize?
* What factors are contributing to a customer's churn risk?
* What retention action should be considered?

A traditional churn model may answer only the first two questions.

This system extends the analysis into a **decision-support framework for customer retention**.

---

## 💡 Solution

The system combines:

**Data Preparation → Exploratory Analysis → Feature Engineering → Machine Learning → Model Optimization → Explainability → Business Analytics → Streamlit Deployment**

For each customer, the system can produce:

**Churn Probability → Risk Level → Revenue Risk → Retention Priority → Recommended Action**

This allows technical model predictions to be translated into business-oriented insights.

---

# 🚀 Key Features

## 1. Individual Customer Prediction

Users can enter customer information through an interactive form and receive:

* Churn probability
* Risk classification
* Customer profile
* Retention recommendation
* Revenue-at-risk estimates
* Prediction explanation

---

## 2. Customer Risk Segmentation

Customers are categorized into three business risk levels:

| Risk Level  | Churn Probability |
| ----------- | ----------------: |
| Low Risk    |           `< 40%` |
| Medium Risk |      `40% – <70%` |
| High Risk   |           `≥ 70%` |

This allows retention teams to focus their attention where it is most needed.

---

## 3. Portfolio Analysis

The application supports CSV uploads for analyzing multiple customers at once.

Portfolio analysis provides:

* Total customers
* Predicted churners
* Risk distribution
* Revenue at risk
* Retention priorities
* Customer-level predictions
* Downloadable prediction results

---

## 4. Retention Prioritization

Customers are ranked using a **Retention Priority Score** that combines customer risk with monthly revenue exposure.

This helps businesses distinguish between:

> A high-risk customer with low financial exposure

and

> A high-risk customer with significant recurring revenue exposure.

This creates a more business-focused prioritization strategy.

---

## 5. Revenue-at-Risk Analysis

The system estimates potential revenue exposure using churn probability and customer charges.

### Expected Monthly Revenue at Risk

`Churn Probability × Monthly Charges`

### Expected Annual Revenue at Risk

`Expected Monthly Revenue at Risk × 12`

These metrics provide a financial perspective on customer churn.

---

## 6. Explainable AI with SHAP

The system incorporates **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

Instead of treating the model as a black box, the application can identify important factors contributing to a customer's prediction.

This helps answer:

> "Why is this customer considered high risk?"

---

# 🧠 Machine Learning Workflow

The project follows a complete machine learning lifecycle.

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Test Preparation
        ↓
Baseline Models
        ↓
Model Comparison
        ↓
Model Optimization
        ↓
Threshold Optimization
        ↓
Final Model Evaluation
        ↓
SHAP Explainability
        ↓
Production Prediction Pipeline
        ↓
Streamlit Application
```

---

# 🧹 Data Preparation

The project includes a dedicated data-cleaning workflow covering common real-world data preparation tasks.

Key preprocessing activities include:

* Handling missing values
* Converting numeric fields stored as text
* Removing unnecessary identifiers
* Encoding categorical variables
* Scaling numerical variables
* Preparing features for machine learning
* Validating prediction inputs

The preprocessing workflow is implemented using **scikit-learn's ColumnTransformer and Pipeline architecture**.

---

# ⚙️ Feature Engineering

Additional features were created to improve the business usefulness of the model.

Examples include:

### Contract Duration

Customer contract information is transformed into a numerical representation of contract duration.

### Total Services

The number of subscribed services is aggregated to represent the breadth of a customer's relationship with the company.

### Revenue Risk Features

Prediction outputs are combined with customer charges to estimate financial exposure.

---

# 🤖 Model Development

Multiple machine learning algorithms were evaluated during development.

Models included:

* Logistic Regression
* Random Forest
* XGBoost
* Gradient Boosting
* LightGBM

The models were evaluated using multiple classification metrics rather than relying solely on accuracy.

---

# 📊 Model Comparison

The model comparison stage produced the following results:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7402 |    0.5070 | 0.7754 |   0.6131 |  0.8384 |
| Random Forest       |   0.7559 |    0.5283 | 0.7487 |   0.6195 |  0.8377 |
| XGBoost             |   0.7942 |    0.6346 | 0.5294 |   0.5773 |  0.8307 |
| Gradient Boosting   |   0.7999 |    0.6586 | 0.5107 |   0.5753 |  0.8462 |
| LightGBM            |   0.7913 |    0.6333 | 0.5080 |   0.5638 |  0.8335 |

The model selection process considered the business objective of identifying customers at risk of churn, rather than optimizing accuracy alone.

---

# 🏆 Final Model Performance

After model optimization and threshold selection, the final production evaluation produced:

| Metric    |        Score |
| --------- | -----------: |
| Accuracy  | **77.9986%** |
| Precision |   **56.67%** |
| Recall    |   **72.73%** |
| F1 Score  |   **63.70%** |
| ROC-AUC   |   **84.76%** |

### Why Recall Matters

For churn prediction, missing a customer who is likely to churn can be costly.

The final system therefore considers the ability to identify potential churners as an important business objective.

The final model achieved a **72.73% recall**, meaning it identified a substantial proportion of the customers who churned in the evaluation dataset.

The **0.8476 ROC-AUC** also indicates strong overall discrimination between churn and non-churn customers.

---

# 🔍 Explainability

Model predictions are supported by SHAP-based explainability.

SHAP helps identify which features contribute most strongly to an individual prediction.

This provides greater transparency for users who need to understand the reasoning behind model outputs.

Example interpretation:

```text
Customer
   ↓
Churn Probability
   ↓
Risk Classification
   ↓
SHAP Explanation
   ↓
Identify Important Risk Factors
   ↓
Retention Decision
```

---

# 💰 Business Analytics

The system converts machine learning predictions into business metrics.

### Expected Monthly Revenue at Risk

Measures the expected monthly financial exposure associated with a customer's probability of churn.

### Expected Annual Revenue at Risk

Projects the expected monthly revenue exposure across twelve months.

### Retention Priority Score

Combines customer churn risk with monthly charges to help prioritize retention efforts.

This allows businesses to move from:

**"Who might churn?"**

to:

**"Who should we act on first, and why?"**

---

# 🎯 Recommended Retention Actions

The system assigns recommended actions based on customer risk.

Examples include:

| Risk        | Business Approach                       |
| ----------- | --------------------------------------- |
| Low Risk    | Maintain relationship and monitor       |
| Medium Risk | Proactive engagement and service review |
| High Risk   | Prioritized retention intervention      |

These recommendations are intended as decision-support suggestions rather than automatic business decisions.

---

# 🖥️ Streamlit Application

The production application provides an interactive interface with the following sections:

### Dashboard

Provides a high-level overview of:

* Total customers
* High-risk customers
* Predicted churners
* Expected revenue at risk
* Risk distribution

### Individual Prediction

Allows users to enter an individual customer's information and generate a prediction.

### Portfolio Analysis

Allows users to upload a CSV containing multiple customers and generate portfolio-level predictions.

### Retention Prioritization

Ranks customers according to retention priority.

### Revenue Risk

Provides financial exposure analysis based on predicted churn.

### Model Information

Provides information about the machine learning system, model artifacts, and business risk framework.

---

# 🏗️ Project Structure

```text
Customer-Churn-System/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── docs/
│   └── development_principles.md
│
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── precision_recall_curve.png
│   └── roc_curve.png
│
├── models/
│   ├── final_churn_model.pkl
│   ├── final_threshold.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.pkl
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Model_Optimization.ipynb
│   ├── 07_Final_Model_Evaluation.ipynb
│   ├── 08_Business_Insights_and_Model_Interpretation.ipynb
│   └── 09_Production_Prediction_Pipeline.ipynb
│
├── reports/
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── final_model_evaluation.csv
│   ├── model_comparison.csv
│   └── xgboost_feature_importance.csv
│
├── src/
│   ├── currency_converter.py
│   ├── data_utils.py
│   ├── explainability.py
│   └── prediction_pipeline.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* XGBoost
* Gradient Boosting
* Random Forest
* Logistic Regression
* LightGBM

### Explainable AI

* SHAP

### Visualization

* Matplotlib

### Application

* Streamlit

### Development & Deployment

* Git
* GitHub
* Streamlit Community Cloud
* Jupyter Notebook

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Hygienus41/Customer-Churn-System.git
```

Move into the project directory:

```bash
cd Customer-Churn-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application Locally

From the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

The production application is connected to the GitHub repository, allowing updates to be deployed from the project's source code.

### Live Application

https://customer-churn-system-gdaz6gakeygvwc3gcx4etw.streamlit.app

---

# 📈 Business Value

This project demonstrates how machine learning can be integrated into a practical business workflow.

Instead of stopping at a classification model, the system connects predictions to:

* Customer risk
* Financial exposure
* Retention prioritization
* Explainable predictions
* Business recommendations

This creates a bridge between **machine learning outputs and business decision-making**.

---

# 🔮 Future Improvements

Potential future improvements include:

* Integration with a production customer database
* Automated model retraining
* Customer lifetime value modeling
* Automated retention campaign integration
* Advanced customer segmentation
* Model monitoring and drift detection
* A/B testing of retention strategies
* Real-time prediction APIs
* Authentication and role-based access
* Automated business reporting
* Cost-sensitive optimization of retention decisions

---

# 📚 Project Goal

The goal of this project was not simply to build a churn classifier.

The broader objective was to demonstrate an end-to-end data science workflow that takes a real business problem from:

**Raw Data → Machine Learning → Explainability → Business Intelligence → Actionable Decision Support → Production Deployment**

---

# 👤 Author

**Hygienus Ugwu**

Data Science & Machine Learning

GitHub:
https://github.com/Hygienus41

---

## ⭐ Project Highlights

* End-to-end machine learning workflow
* Customer churn prediction
* Model comparison and optimization
* Threshold-based risk segmentation
* SHAP explainability
* Revenue-at-risk analysis
* Retention prioritization
* Actionable business recommendations
* Interactive Streamlit dashboard
* Portfolio-level CSV predictions
* Production prediction pipeline
* Cloud deployment
