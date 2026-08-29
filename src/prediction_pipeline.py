"""
Production Customer Churn Prediction Pipeline

This module contains the reusable inference logic for the
customer churn prediction system.

Responsibilities
----------------
1. Load production model artifacts
2. Validate customer input
3. Engineer production features
4. Generate churn probabilities
5. Apply the optimized classification threshold
6. Assign business risk levels
7. Calculate retention priority
8. Estimate revenue at risk
9. Generate recommended business actions
10. Run the complete prediction pipeline

This module should be used by downstream applications such as
the Streamlit interface.

It should NOT contain:
- Model training
- Model optimization
- Exploratory data analysis
- Notebook-specific code
- Visualization logic
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "final_churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
THRESHOLD_PATH = MODEL_DIR / "final_threshold.pkl"

# Production Business Configuration

LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70


RISK_SCORE_MAPPING = {
    "Low Risk": 1,
    "Medium Risk": 2,
    "High Risk": 3
}


RETENTION_STRATEGY = {
    "High Risk": (
        "Immediate retention outreach, personalized offers, "
        "loyalty incentives, and proactive customer support."
    ),

    "Medium Risk": (
        "Monitor customer engagement, send personalized "
        "communication, and offer targeted promotions."
    ),

    "Low Risk": (
        "Maintain customer relationship through loyalty "
        "programs, regular engagement, and service improvements."
    )
}


# Expected Raw Input Schema

EXPECTED_RAW_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# Load Production Artifacts

def load_production_artifacts(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    threshold_path=THRESHOLD_PATH
):
    """
    Load the saved production model, preprocessor, and threshold.

    Returns
    -------
    tuple
        model, preprocessor, threshold
    """

    paths = {
        "model": Path(model_path),
        "preprocessor": Path(preprocessor_path),
        "threshold": Path(threshold_path)
    }

    missing = [
        name
        for name, path in paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing production artifacts: "
            + ", ".join(missing)
        )

    model = joblib.load(paths["model"])

    preprocessor = joblib.load(
        paths["preprocessor"]
    )

    threshold = float(
        joblib.load(paths["threshold"])
    )

    if not 0 < threshold < 1:
        raise ValueError(
            f"Invalid classification threshold: {threshold}. "
            "Threshold must be between 0 and 1."
        )

    return model, preprocessor, threshold


# Validate Customer Input

def validate_customer_input(df):
    """
    Validate raw customer input before prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw customer data.

    Returns
    -------
    bool
        True when validation succeeds.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Customer input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Customer input cannot be empty."
        )

    missing_columns = [
        column
        for column in EXPECTED_RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required customer features: "
            + ", ".join(missing_columns)
        )

    # Numeric validation

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges"
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

    # Range validation

    if (df["tenure"] < 0).any():
        raise ValueError(
            "tenure cannot contain negative values."
        )

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError(
            "MonthlyCharges cannot contain negative values."
        )

    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError(
            "SeniorCitizen must contain only 0 or 1."
        )

    # Categorical validation

    categorical_values = {
        "gender": {
            "Male",
            "Female"
        },

        "Partner": {
            "Yes",
            "No"
        },

        "Dependents": {
            "Yes",
            "No"
        },

        "PhoneService": {
            "Yes",
            "No"
        },

        "MultipleLines": {
            "Yes",
            "No",
            "No phone service"
        },

        "InternetService": {
            "DSL",
            "Fiber optic",
            "No"
        },

        "OnlineSecurity": {
            "Yes",
            "No",
            "No internet service"
        },

        "OnlineBackup": {
            "Yes",
            "No",
            "No internet service"
        },

        "DeviceProtection": {
            "Yes",
            "No",
            "No internet service"
        },

        "TechSupport": {
            "Yes",
            "No",
            "No internet service"
        },

        "StreamingTV": {
            "Yes",
            "No",
            "No internet service"
        },

        "StreamingMovies": {
            "Yes",
            "No",
            "No internet service"
        },

        "Contract": {
            "Month-to-month",
            "One year",
            "Two year"
        },

        "PaperlessBilling": {
            "Yes",
            "No"
        },

        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        }
    }

    for column, allowed_values in categorical_values.items():

        invalid_values = set(
            df[column].dropna().unique()
        ) - allowed_values

        if invalid_values:
            raise ValueError(
                f"Invalid values found in {column}: "
                f"{invalid_values}"
            )

    return True


# Feature Engineering

def engineer_features(df):
    """
    Apply production feature engineering.

    This transformation must remain consistent with the
    feature engineering used during model development.
    """

    data = df.copy()

    # Remove customer identifier

    if "customerID" in data.columns:
        data = data.drop(
            columns=["customerID"]
        )

    # Convert TotalCharges to numeric

    if "TotalCharges" in data.columns:

        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

    # Contract duration

    contract_mapping = {
        "Month-to-month": 1,
        "One year": 12,
        "Two year": 24
    }

    if "Contract" in data.columns:

        data["ContractMonths"] = (
            data["Contract"]
            .map(contract_mapping)
        )

    # Total subscribed services

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    available_services = [
        column
        for column in service_columns
        if column in data.columns
    ]

    if available_services:

        data["TotalServices"] = (
            data[available_services]
            .apply(
                lambda row: sum(
                    value == "Yes"
                    for value in row
                ),
                axis=1
            )
        )

    return data

# Churn Prediction

def predict_churn(
    customer_data,
    model,
    preprocessor,
    threshold
):
    """
    Generate churn predictions.

    Returns
    -------
    pandas.DataFrame
        Customer data with prediction outputs.
    """

    # Validate input
    validate_customer_input(
        customer_data
    )

    # Feature engineering
    engineered_data = engineer_features(
        customer_data
    )

    # Apply saved preprocessing
    X_processed = preprocessor.transform(
        engineered_data
    )

    # Churn probability
    churn_probability = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    # Apply production threshold
    churn_prediction = (
        churn_probability >= threshold
    ).astype(int)

    results = customer_data.copy()

    # Probability: 0–1
    results["Churn_Probability"] = (
        churn_probability
    )

    # Probability: 0–100
    results["Churn_Probability_Percent"] = (
        churn_probability * 100
    )

    # Binary prediction
    results["Churn_Prediction"] = (
        churn_prediction
    )

    # Human-readable prediction
    results["Prediction_Label"] = np.where(
        churn_prediction == 1,
        "Churn",
        "No Churn"
    )

    return results

# Business Risk Segmentation

def assign_risk_level(probability):
    """
    Assign a business risk category.

    Risk thresholds are independent from the
    machine-learning classification threshold.
    """

    if probability >= HIGH_RISK_THRESHOLD:
        return "High Risk"

    if probability >= LOW_RISK_THRESHOLD:
        return "Medium Risk"

    return "Low Risk"


def add_risk_level(prediction_df):
    """
    Add business risk level to prediction results.
    """

    results = prediction_df.copy()

    results["Risk_Level"] = (
        results["Churn_Probability"]
        .apply(assign_risk_level)
    )

    return results

# Retention Priority

def add_retention_priority(prediction_df):
    """
    Add retention priority and revenue-at-risk metrics.
    """

    results = prediction_df.copy()

    # Risk score
    results["Risk_Score"] = (
        results["Risk_Level"]
        .map(RISK_SCORE_MAPPING)
    )

    # Business prioritization heuristic
    
    results["Retention_Priority_Score"] = (
        results["Risk_Score"]
        * results["MonthlyCharges"]
    )

    # Expected monthly revenue at risk
    results[
        "Expected_Monthly_Revenue_at_Risk"
    ] = (
        results["Churn_Probability"]
        * results["MonthlyCharges"]
    )

    # Expected annual revenue at risk
    results[
        "Expected_Annual_Revenue_at_Risk"
    ] = (
        results[
            "Expected_Monthly_Revenue_at_Risk"
        ]
        * 12
    )

    return results

    # Recommended Business Action

def add_recommended_action(prediction_df):
    """
    Add recommended retention strategy.
    """

    results = prediction_df.copy()

    results["Recommended_Action"] = (
        results["Risk_Level"]
        .map(RETENTION_STRATEGY)
    )

    return results

# Complete Prediction Pipeline

def run_prediction_pipeline(
    customer_data,
    model=None,
    preprocessor=None,
    threshold=None
):
    """
    Execute the complete production inference pipeline.

    Steps
    -----
    1. Load artifacts if not supplied
    2. Validate input
    3. Engineer features
    4. Generate churn prediction
    5. Assign business risk
    6. Calculate retention priority
    7. Estimate revenue at risk
    8. Generate recommended action

    Returns
    -------
    pandas.DataFrame
    """

    # Load production artifacts when necessary

    if (
        model is None
        or preprocessor is None
        or threshold is None
    ):

        (
            model,
            preprocessor,
            threshold
        ) = load_production_artifacts()

    # Prediction

    results = predict_churn(
        customer_data=customer_data,
        model=model,
        preprocessor=preprocessor,
        threshold=threshold
    )

    # Business intelligence layer

    results = add_risk_level(
        results
    )

    results = add_retention_priority(
        results
    )

    results = add_recommended_action(
        results
    )

    return results