"""
Production Model Explainability

This module contains reusable logic for explaining
individual customer churn predictions.

Responsibilities
----------------
1. Load production model artifacts
2. Transform customer data using the production preprocessor
3. Generate SHAP values
4. Identify the most influential features
5. Return feature importance information

This module should NOT contain:
- Streamlit UI
- Visualization logic
- Model training
- Notebook-specific code
"""


# IMPORTANT:
# Reuse engineer_features() from prediction_pipeline.py.
# Do not duplicate feature engineering logic here.
# This keeps predictions and explanations consistent.

# ============================================================
# IMPORTS
# Use the exact same feature engineering function that the production prediction pipeline uses. That's important for consistency.
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import shap

from src.prediction_pipeline import engineer_features


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "final_churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"


# ============================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================

def load_explainability_artifacts(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH
):
    """
    Load the production model and preprocessor.

    Returns
    -------
    tuple
        model, preprocessor
    """

    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not preprocessor_path.exists():

        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}"
        )

    model = joblib.load(model_path)

    preprocessor = joblib.load(
        preprocessor_path
    )

    return model, preprocessor


# ============================================================
# EXTRACT XGBOOST CLASSIFIER
# ============================================================

def get_xgboost_classifier(model):
    """
    Extract the XGBoost classifier from the
    saved sklearn Pipeline.
    """

    if not hasattr(model, "named_steps"):

        raise TypeError(
            "Expected the production model to be "
            "an sklearn Pipeline."
        )

    if "classifier" not in model.named_steps:

        raise KeyError(
            "The production Pipeline does not contain "
            "a step named 'classifier'."
        )

    classifier = model.named_steps[
        "classifier"
    ]

    return classifier


# ============================================================
# PREPARE FEATURES FOR EXPLANATION
# ============================================================

def prepare_explanation_data(
    customer_data,
    preprocessor
):
    """
    Apply the same feature engineering and preprocessing
    used by the production prediction pipeline.

    Parameters
    ----------
    customer_data : pandas.DataFrame
        Raw customer data.

    preprocessor : ColumnTransformer
        Saved production preprocessor.

    Returns
    -------
    tuple
        Processed data and feature names.
    """

    # Apply the exact production feature engineering
    engineered_data = engineer_features(
        customer_data
    )

    # Apply the saved production preprocessor
    processed_data = preprocessor.transform(
        engineered_data
    )

    # Retrieve the processed feature names
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    return processed_data, feature_names


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_LABELS = {
    "num__SeniorCitizen":
        "Senior citizen status",

    "num__tenure":
        "Customer tenure",

    "num__MonthlyCharges":
        "Monthly charges",

    "num__TotalServices":
        "Number of subscribed services",

    "num__ContractMonths":
        "Contract duration",

    "cat__gender_Male":
        "Male gender",

    "cat__Partner_Yes":
        "Has a partner",

    "cat__Dependents_Yes":
        "Has dependents",

    "cat__PhoneService_Yes":
        "Phone service",

    "cat__MultipleLines_No":
        "No multiple lines",

    "cat__MultipleLines_No phone service":
        "No phone service",

    "cat__MultipleLines_Yes":
        "Multiple lines",

    "cat__InternetService_DSL":
        "DSL internet service",

    "cat__InternetService_Fiber optic":
        "Fiber optic internet service",

    "cat__InternetService_No":
        "No internet service",

    "cat__OnlineSecurity_No":
        "No online security",

    "cat__OnlineSecurity_No internet service":
        "No internet service for security",

    "cat__OnlineSecurity_Yes":
        "Online security",

    "cat__OnlineBackup_No":
        "No online backup",

    "cat__OnlineBackup_No internet service":
        "No internet service for backup",

    "cat__OnlineBackup_Yes":
        "Online backup",

    "cat__DeviceProtection_No":
        "No device protection",

    "cat__DeviceProtection_No internet service":
        "No internet service for device protection",

    "cat__DeviceProtection_Yes":
        "Device protection",

    "cat__TechSupport_No":
        "No technical support",

    "cat__TechSupport_No internet service":
        "No internet service for technical support",

    "cat__TechSupport_Yes":
        "Technical support",

    "cat__StreamingTV_No":
        "No streaming TV",

    "cat__StreamingTV_No internet service":
        "No internet service for streaming TV",

    "cat__StreamingTV_Yes":
        "Streaming TV",

    "cat__StreamingMovies_No":
        "No streaming movies",

    "cat__StreamingMovies_No internet service":
        "No internet service for streaming movies",

    "cat__StreamingMovies_Yes":
        "Streaming movies",

    "cat__Contract_Month-to-month":
        "Month-to-month contract",

    "cat__Contract_One year":
        "One-year contract",

    "cat__Contract_Two year":
        "Two-year contract",

    "cat__PaperlessBilling_Yes":
        "Paperless billing",

    "cat__PaymentMethod_Bank transfer (automatic)":
        "Automatic bank transfer",

    "cat__PaymentMethod_Credit card (automatic)":
        "Automatic credit card payment",

    "cat__PaymentMethod_Electronic check":
        "Electronic check payment",

    "cat__PaymentMethod_Mailed check":
        "Mailed check payment"
}


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def explain_customer(
    customer_data,
    model,
    preprocessor,
    top_n=5
):
    """
    Generate a human-readable SHAP explanation
    for an individual customer.

    Parameters
    ----------
    customer_data : pandas.DataFrame
        Raw customer data.

    model : sklearn Pipeline
        Production model.

    preprocessor : ColumnTransformer
        Production preprocessor.

    top_n : int
        Number of top factors to return.

    Returns
    -------
    pandas.DataFrame
        Ranked and human-readable feature explanations.
    """

    # Prepare model input
    processed_data, feature_names = (
        prepare_explanation_data(
            customer_data,
            preprocessor
        )
    )

    # Extract the XGBoost classifier
    classifier = get_xgboost_classifier(
        model
    )

    # Create SHAP TreeExplainer
    explainer = shap.TreeExplainer(
        classifier
    )

    # Generate SHAP values
    shap_values = explainer.shap_values(
        processed_data
    )

    # Get SHAP values for the first customer
    customer_shap_values = shap_values[0]

    # Build explanation DataFrame
    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP_Value": customer_shap_values,
            "Impact": abs(
                customer_shap_values
            )
        }
    )

    # Rank features by their absolute SHAP impact
    explanation = (
        explanation
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    # Convert technical feature names
    # into business-friendly names
    explanation["Feature"] = (
        explanation["Feature"]
        .map(FEATURE_LABELS)
        .fillna(explanation["Feature"])
    )

    # Determine whether each feature
    # increases or decreases churn risk
    explanation["Direction"] = (
        explanation["SHAP_Value"]
        .apply(
            lambda value:
                "↑ Increases churn risk"
                if value > 0
                else "↓ Reduces churn risk"
        )
    )

    return explanation