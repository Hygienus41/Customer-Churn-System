import streamlit as st
import pandas as pd

from pathlib import Path
import sys

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.prediction_pipeline import (
    run_prediction_pipeline
)

from src.explainability import (
    load_explainability_artifacts,
    explain_customer
)

from src.currency_converter import (
    convert_currency,
    get_currency_symbol,
    get_exchange_rate_info
)

# ============================================================
# LOAD EXPLAINABILITY ARTIFACTS
# ============================================================

explainability_model, explainability_preprocessor = (
    load_explainability_artifacts()
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Churn System")

    page = st.selectbox(
        "Navigation",
        [
            "Dashboard",
            "Individual Prediction",
            "Portfolio Analysis",
            "Retention Prioritization",
            "Revenue Risk",
            "Model Information"
        ]
    )

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("📊 Customer Churn Intelligence")

    st.markdown(
        """
        Monitor customer churn, identify high-risk customers,
        and understand the potential revenue impact of customer attrition.
        """
    )

    st.divider()

    # Check whether portfolio results are available
    if "portfolio_results" not in st.session_state:

        # ========================================================
        # EMPTY DASHBOARD STATE
        # ========================================================
    
        st.subheader("Portfolio Overview")
    
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
    
            st.metric(
                "👥 Total Customers",
                "—"
            )
    
        with col2:
    
            st.metric(
                "🔴 High-Risk Customers",
                "—"
            )
    
        with col3:
    
            st.metric(
                "📉 Predicted Churners",
                "—"
            )
    
        with col4:
    
            st.metric(
                "💰 Annual Revenue at Risk",
                "—"
            )
    
        st.divider()
    
        st.info(
            "📁 No portfolio has been loaded yet. "
            "Go to **Portfolio Analysis** and upload a customer "
            "dataset to generate churn intelligence."
        )

    else:

        # Retrieve stored prediction results
        results = st.session_state[
            "portfolio_results"
        ]

        # ====================================================
        # PORTFOLIO STATUS
        # ====================================================

        st.success(
            f"✅ Portfolio loaded successfully — "
            f"{len(results):,} customers analyzed."
        )

        # ============================================================
        # CURRENCY SELECTION
        # ============================================================
        
        currency_options = {
            "USD ($)": "USD",
            "NGN (₦)": "NGN",
            "EUR (€)": "EUR",
            "GBP (£)": "GBP",
            "GHS (₵)": "GHS",
            "KES (KSh)": "KES",
            "ZAR (R)": "ZAR"
        }
        
        selected_currency = st.selectbox(
            "Select Currency",
            list(currency_options.keys()),
            key="dashboard_currency"
        )
        
        target_currency = currency_options[
            selected_currency
        ]
        
        currency_symbol = get_currency_symbol(
            target_currency
        )

        # ====================================================
        # PORTFOLIO OVERVIEW
        # ====================================================
        
        st.subheader("📈 Portfolio Overview")

        # Total number of customers in the portfolio
        total_customers = len(results)

        # Number of customers classified as High Risk
        high_risk_customers = (
            results["Risk_Level"]
            == "High Risk"
        ).sum()

        # Number of customers predicted to churn
        predicted_churners = (
            results["Churn_Prediction"]
            == 1
        ).sum()

        # Total expected annual revenue at risk
        annual_revenue_at_risk_usd = (
            results[
                "Expected_Annual_Revenue_at_Risk"
            ].sum()
        )
        
        annual_revenue_at_risk = convert_currency(
            annual_revenue_at_risk_usd,
            from_currency="USD",
            to_currency=target_currency
        )


        # ============================================================
        # DASHBOARD KPI CARDS
        # ============================================================
        
        st.subheader("Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
        
            st.metric(
                label="👥 Total Customers",
                value=f"{total_customers:,}"
            )
        
        with col2:
        
            st.metric(
                label="🔴 High-Risk Customers",
                value=f"{high_risk_customers:,}"
            )
        
        with col3:
        
            st.metric(
                label="📉 Predicted Churners",
                value=f"{predicted_churners:,}"
            )
        
        with col4:
        
            st.metric(
                label="💰 Annual Revenue at Risk",
                value=f"{currency_symbol}{annual_revenue_at_risk:,.2f}"
            )


        # ============================================================
        # RISK DISTRIBUTION
        # ============================================================
        
        st.subheader("📊 Customer Risk Distribution")
        
        st.markdown(
            "Distribution of customers across the model's "
            "Low, Medium, and High Risk categories."
        )
        
        # Count customers in each risk category
        risk_counts = (
            results["Risk_Level"]
            .value_counts()
            .reindex(
                ["Low Risk", "Medium Risk", "High Risk"],
                fill_value=0
            )
        )
        
        # Calculate percentage of total portfolio
        risk_percentages = (
            risk_counts / total_customers * 100
        )
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        # ============================================================
        # RISK CHART
        # ============================================================
        
        with col1:
        
            st.bar_chart(
                risk_counts,
                height=350
            )

        # ============================================================
        # RISK SUMMARY
        # ============================================================
        
        st.subheader("🎯 Risk Summary")
        
        low_risk = (
            results["Risk_Level"] == "Low Risk"
        ).sum()
        
        medium_risk = (
            results["Risk_Level"] == "Medium Risk"
        ).sum()
        
        high_risk = (
            results["Risk_Level"] == "High Risk"
        ).sum()
        
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
        
            st.metric(
                "🟢 Low Risk",
                f"{low_risk:,}"
            )
        
        with col2:
        
            st.metric(
                "🟡 Medium Risk",
                f"{medium_risk:,}"
            )
        
        with col3:
        
            st.metric(
                "🔴 High Risk",
                f"{high_risk:,}"
            )
            

        # ============================================================
        # REVENUE RISK ANALYSIS
        # ============================================================
        
        st.divider()
        
        st.subheader("💰 Revenue Risk Analysis")
        
        st.markdown(
            "Estimated revenue exposure based on the model's "
            "predicted churn probabilities."
        )
        
        # ============================================================
        # REVENUE RISK CALCULATIONS
        # ============================================================
        
        monthly_revenue_risk_usd = (
            results[
                "Expected_Monthly_Revenue_at_Risk"
            ].sum()
        )
        
        annual_revenue_risk_usd = (
            results[
                "Expected_Annual_Revenue_at_Risk"
            ].sum()
        )
        
        high_risk_revenue_usd = (
            results.loc[
                results["Risk_Level"] == "High Risk",
                "Expected_Annual_Revenue_at_Risk"
            ].sum()
        )
        
        
        # ============================================================
        # CONVERT TO SELECTED CURRENCY
        # ============================================================
        
        monthly_revenue_risk = convert_currency(
            monthly_revenue_risk_usd,
            from_currency="USD",
            to_currency=target_currency
        )
        
        annual_revenue_risk = convert_currency(
            annual_revenue_risk_usd,
            from_currency="USD",
            to_currency=target_currency
        )
        
        high_risk_revenue = convert_currency(
            high_risk_revenue_usd,
            from_currency="USD",
            to_currency=target_currency
        )
        
        # ============================================================
        # REVENUE KPI CARDS
        # ============================================================
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
        
            st.metric(
                "Monthly Revenue at Risk",
                f"{currency_symbol}{monthly_revenue_risk:,.2f}"
            )
        
        with col2:
        
            st.metric(
                "Annual Revenue at Risk",
                f"{currency_symbol}{annual_revenue_risk:,.2f}"
            )
        
        with col3:
        
            st.metric(
                "High-Risk Revenue Exposure",
                f"{currency_symbol}{high_risk_revenue:,.2f}"
            )
        
        # ============================================================
        # REVENUE RISK BY CUSTOMER RISK LEVEL
        # ============================================================
        
        revenue_by_risk_usd = (
            results
            .groupby("Risk_Level")[
                "Expected_Annual_Revenue_at_Risk"
            ]
            .sum()
            .reindex(
                [
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ],
                fill_value=0
            )
        )
        
        revenue_by_risk = revenue_by_risk_usd.map(
            lambda x: convert_currency(
                x,
                from_currency="USD",
                to_currency=target_currency
            )
        )
        
        st.markdown("### Annual Revenue Exposure by Risk Level")
        
        st.bar_chart(
            revenue_by_risk,
            height=350
        )

        # ============================================================
        # CUSTOMERS REQUIRING ATTENTION
        # ============================================================
        
        st.divider()
        
        st.subheader("🚨 Customers Requiring Attention")
        
        st.markdown(
            "Customers with the highest retention priority based on "
            "churn risk and expected revenue exposure."
        )

        # ============================================================
        # CUSTOMER PRIORITY FILTERS
        # ============================================================
        
        col1, col2 = st.columns(2)
        
        with col1:
        
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                [
                    "All Risk Levels",
                    "High Risk",
                    "Medium Risk",
                    "Low Risk"
                ]
            )
        
        with col2:
        
            top_n = st.selectbox(
                "Number of Customers",
                [5, 10, 20, 50],
                index=1
            )
        
        
        # ============================================================
        # APPLY RISK FILTER
        # ============================================================
        
        filtered_results = results.copy()
        
        if risk_filter != "All Risk Levels":
        
            filtered_results = filtered_results[
                filtered_results["Risk_Level"] == risk_filter
            ]
        
        
        # ============================================================
        # SORT BY RETENTION PRIORITY
        # ============================================================
        
        priority_customers = (
            filtered_results
            .sort_values(
                "Retention_Priority_Score",
                ascending=False
            )
            .head(top_n)
            .copy()
        )
        
        
        # ============================================================
        # CUSTOMER IDENTIFICATION
        # ============================================================
        
        customer_id_column = None
        
        for column in [
            "customerID",
            "CustomerID",
            "customer_id",
            "Customer_ID"
        ]:
        
            if column in priority_customers.columns:
        
                customer_id_column = column
                break
        
        
        # ============================================================
        # SELECT IMPORTANT BUSINESS COLUMNS
        # ============================================================
        
        display_columns = [
            "Churn_Probability",
            "Prediction_Label",
            "Risk_Level",
            "Expected_Monthly_Revenue_at_Risk",
            "Expected_Annual_Revenue_at_Risk",
            "Retention_Priority_Score",
            "Recommended_Action"
        ]
        
        
        if customer_id_column is not None:
        
            display_columns.insert(
                0,
                customer_id_column
            )
        
        
        priority_display = priority_customers[
            display_columns
        ].copy()
        
        
        # ============================================================
        # FORMAT DISPLAY VALUES
        # ============================================================
        
        priority_display["Churn_Probability"] = (
            priority_display["Churn_Probability"]
            .map(lambda x: f"{x:.1%}")
        )
        
        priority_display[
            "Expected_Monthly_Revenue_at_Risk"
        ] = (
            priority_display[
                "Expected_Monthly_Revenue_at_Risk"
            ]
            .map(
                lambda x: (
                    f"{currency_symbol}"
                    f"{convert_currency(
                        x,
                        from_currency='USD',
                        to_currency=target_currency
                    ):,.2f}"
                )
            )
        )
        
        priority_display[
            "Expected_Annual_Revenue_at_Risk"
        ] = (
            priority_display[
                "Expected_Annual_Revenue_at_Risk"
            ]
            .map(
                lambda x: (
                    f"{currency_symbol}"
                    f"{convert_currency(
                        x,
                        from_currency='USD',
                        to_currency=target_currency
                    ):,.2f}"
                )
            )
        )
        
        priority_display[
            "Retention_Priority_Score"
        ] = (
            priority_display[
                "Retention_Priority_Score"
            ]
            .map(lambda x: f"{x:.3f}")
        )
        
        
        # ============================================================
        # RENAME COLUMNS
        # ============================================================
        
        rename_columns = {
            "Churn_Probability": "Churn Probability",
            "Prediction_Label": "Prediction",
            "Risk_Level": "Risk Level",
            "Expected_Monthly_Revenue_at_Risk":
                "Monthly Revenue at Risk",
            "Expected_Annual_Revenue_at_Risk":
                "Annual Revenue at Risk",
            "Retention_Priority_Score":
                "Priority Score",
            "Recommended_Action":
                "Recommended Action"
        }
        
        
        if customer_id_column is not None:
        
            rename_columns[
                customer_id_column
            ] = "Customer ID"
        
        
        priority_display = priority_display.rename(
            columns=rename_columns
        )
        
        
        # ============================================================
        # DISPLAY FILTERED RESULTS
        # ============================================================
        
        st.dataframe(
            priority_display,
            hide_index=True,
            use_container_width=True
        )
        
        
# ============================================================
# INDIVIDUAL PREDICTION
# ============================================================

elif page == "Individual Prediction":

    st.header("👤 Individual Customer Prediction")

    st.markdown(
        """
        Enter the customer's information below to generate
        a churn prediction and understand their risk level.
        """
    )

    st.divider()


    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.subheader("👤 Customer Information")

    st.markdown(
        "Provide basic information about the customer."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )


    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    # ========================================================
    # CUSTOMER RELATIONSHIP
    # ========================================================

    st.subheader("📅 Customer Relationship")

    tenure = st.number_input(
        "Tenure (months)",
        min_value=1,
        max_value=72,
        value=1,
        step=1
    )

    # ========================================================
    # PHONE SERVICES
    # ========================================================

    st.subheader("📞 Phone Services")

    col1, col2 = st.columns(2)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

    # ========================================================
    # INTERNET SERVICES
    # ========================================================

    st.subheader("🌐 Internet Services")

    col1, col2 = st.columns(2)

    with col1:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    col1, col2 = st.columns(2)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    col1, col2 = st.columns(2)

    with col1:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    # ========================================================
    # CONTRACT AND BILLING
    # ========================================================

    st.subheader("💳 Contract & Billing")

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    # ========================================================
    # CHARGES
    # ========================================================

    st.subheader("💰 Charges")

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=40.0,
            step=1.0
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    # ========================================================
    # BUILD CUSTOMER DATAFRAME
    # ========================================================
    # Convert the values entered in the form into the exact DataFrame structure expected by the prediction pipeline.

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    predict_button = st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    )

    # ============================================================
    # PREDICT CUSTOMER CHURN
    # ============================================================

    if predict_button:
    
        try:
    
            # Run the production prediction pipeline
            result = run_prediction_pipeline(
                customer
            )
    
            # Store prediction result in session state
            st.session_state[
                "prediction_result"
            ] = result
    
            # Get the first prediction row
            result_row = result.iloc[0]
    
    
            # ========================================================
            # PREDICTION RESULTS
            # ========================================================
    
            st.divider()
    
            st.header("📊 Prediction Results")
    
            col1, col2, col3 = st.columns(3)
    
            with col1:
    
                st.metric(
                    "Churn Probability",
                    f"{result_row['Churn_Probability']:.1%}"
                )
    
            with col2:
    
                st.metric(
                    "Prediction",
                    result_row["Prediction_Label"]
                )
    
            with col3:
    
                st.metric(
                    "Risk Level",
                    result_row["Risk_Level"]
                )
    
    
            # ========================================================
            # REVENUE AT RISK
            # ========================================================
    
            st.subheader("💰 Revenue at Risk")
    
            col1, col2 = st.columns(2)
    
            with col1:
    
                st.metric(
                    "Monthly Revenue at Risk",
                    f"${result_row['Expected_Monthly_Revenue_at_Risk']:,.2f}"
                )
    
            with col2:
    
                st.metric(
                    "Annual Revenue at Risk",
                    f"${result_row['Expected_Annual_Revenue_at_Risk']:,.2f}"
                )
    
    
            # ========================================================
            # RECOMMENDED RETENTION ACTION
            # ========================================================
    
            st.subheader("🎯 Recommended Action")
    
            st.info(
                result_row[
                    "Recommended_Action"
                ]
            )
    
    
            # ========================================================
            # MODEL EXPLAINABILITY
            # ========================================================
    
            st.subheader(
                "🔍 Why is this customer at risk?"
            )
    
            explanation = explain_customer(
                customer_data=customer,
                model=explainability_model,
                preprocessor=explainability_preprocessor,
                top_n=5
            )
    
            st.markdown(
                "**Top factors influencing this prediction**"
            )
    
            st.dataframe(
                explanation,
                hide_index=True,
                use_container_width=True
            )
    
    
        except Exception as e:
    
            st.error(
                f"Prediction failed: {e}"
            )


    # ========================================================
    # CUSTOMER INPUT PREVIEW
    # ========================================================

    st.subheader(
        "Customer Input Preview"
    )

    st.dataframe(
        customer,
        hide_index=True,
        use_container_width=True
    )

    
elif page == "Portfolio Analysis":

    st.header("📊 Portfolio Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload customer dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        # ==========================================
        # LOAD THE UPLOADED CSV FILE
        # ==========================================
    
        customer_data = pd.read_csv(
            uploaded_file
        )
    
    
        # ==========================================
        # STORE THE UPLOADED DATA
        # ==========================================
        # session_state allows us to keep the data
        # available when the user moves to another page.
    
        st.session_state["portfolio_data"] = (
            customer_data
        )
    
    
        # ==========================================
        # DISPLAY THE UPLOADED CUSTOMER DATA
        # ==========================================
    
        st.subheader(
            "Uploaded Customer Data"
        )
    
        st.dataframe(
            customer_data,
            hide_index=True,
            use_container_width=True
        )
    
    
        # ==========================================
        # RUN THE CHURN PREDICTION PIPELINE
        # ==========================================
        # This sends the uploaded customer data through
        # the model, feature engineering, risk scoring,
        # revenue-at-risk calculations, etc.
    
        results = run_prediction_pipeline(
            customer_data
        )
    
    
        # ==========================================
        # STORE THE PREDICTION RESULTS
        # ==========================================
        # This allows other pages, such as "Retention Prioritization", to use the prediction results.
    
        st.session_state["portfolio_results"] = (
            results
        )

        # ==============================
        # PORTFOLIO OVERVIEW
        # ==============================

        st.subheader("📈 Portfolio Overview")

        total_customers = len(results)

        churn_rate = (
            results["Churn_Prediction"].mean() * 100
        )

        customers_at_risk = (
            results["Risk_Level"]
            .isin(["Medium Risk", "High Risk"])
            .sum()
        )

        revenue_at_risk_usd = (
            results[
                "Expected_Annual_Revenue_at_Risk"
            ].sum()
        )
        
        revenue_at_risk = convert_currency(
            revenue_at_risk_usd,
            from_currency="USD",
            to_currency="NGN"
        )

        # ==============================
        # KPI CARDS
        # ==============================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Customers",
                f"{total_customers:,}"
            )

        with col2:

            st.metric(
                "Churn Rate",
                f"{churn_rate:.1f}%"
            )

        with col3:

            st.metric(
                "Customers at Risk",
                f"{customers_at_risk:,}"
            )

        with col4:

            st.metric(
            "Annual Revenue at Risk",
            f"${revenue_at_risk_usd:,.2f}"
        )

        # ==============================
        # PREDICTION RESULTS
        # ==============================

        st.subheader("🔮 Churn Predictions")

        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True
        )
        # ==============================
        # RISK LEVEL DISTRIBUTION
        # ==============================
        
        st.subheader("Risk Level Distribution")
        
        risk_counts = (
            results["Risk_Level"]
            .value_counts()
            .reindex(
                ["Low Risk", "Medium Risk", "High Risk"],
                fill_value=0
            )
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.bar_chart(
                risk_counts,
                height=300
            )

        # ==============================
        # Download Predictions
        # ==============================
        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

# ============================================================
# RETENTION PRIORITIZATION
# ============================================================

elif page == "Retention Prioritization":

    st.header(
        "🎯 Retention Prioritization"
    )

    # Check whether portfolio predictions exist
    if "portfolio_results" not in st.session_state:

        st.warning(
            "Please upload a customer dataset "
            "in Portfolio Analysis first."
        )

    else:

        results = st.session_state[
            "portfolio_results"
        ]

        # ----------------------------------------------------
        # SORT CUSTOMERS BY PRIORITY
        # ----------------------------------------------------

        priority = results.sort_values(
            "Retention_Priority_Score",
            ascending=False
        )

        # ----------------------------------------------------
        # DISPLAY PRIORITIZED CUSTOMERS
        # ----------------------------------------------------

        st.subheader(
            "Customers Ranked by Retention Priority"
        )

        st.dataframe(
            priority,
            hide_index=True,
            use_container_width=True
        )

        # ============================================================
        # REVENUE RISK
        # This is expected revenue at risk under the model's estimated churn probability, not a guaranteed financial loss.
        # ============================================================

elif page == "Revenue Risk":

    st.header("💰 Revenue Risk")

    # Check whether portfolio prediction results
    # are available from Portfolio Analysis
    if "portfolio_results" not in st.session_state:

        st.warning(
            "Please upload a customer dataset "
            "in Portfolio Analysis first."
        )

    else:

        # Retrieve the stored prediction results
        # from the Portfolio Analysis page
        results = st.session_state[
            "portfolio_results"
        ]

        # ============================================================
        # CURRENCY SELECTION
        # ============================================================
        
        currency_options = {
            "USD ($)": "USD",
            "NGN (₦)": "NGN",
            "EUR (€)": "EUR",
            "GBP (£)": "GBP",
            "GHS (₵)": "GHS",
            "KES (KSh)": "KES",
            "ZAR (R)": "ZAR"
        }
        
        selected_currency = st.selectbox(
            "Select Currency",
            list(currency_options.keys())
        )
        
        
        # ============================================================
        # EXCHANGE RATE INFORMATION
        # ============================================================
        
        exchange_rate_info = get_exchange_rate_info()
        
        if exchange_rate_info["status"] == "Live":
        
            st.success(
                f"💱 Live exchange rates | "
                f"Source: {exchange_rate_info['source']} | "
                f"Updated: {exchange_rate_info['last_updated']} | "
                f"Base: {exchange_rate_info['base_currency']}"
            )
        
        else:
        
            st.warning(
                f"⚠️ Using fallback exchange rates | "
                f"Source: {exchange_rate_info['source']} | "
                f"Base: {exchange_rate_info['base_currency']}"
            )
        
        
        target_currency = currency_options[
            selected_currency
        ]
        
        currency_symbol = get_currency_symbol(
            target_currency
        )
        
        
        # ============================================================
        # REVENUE RISK CALCULATIONS
        # ============================================================
        
        # Revenue values produced by the model are USD-based.
        monthly_risk_usd = results[
            "Expected_Monthly_Revenue_at_Risk"
        ].sum()
        
        annual_risk_usd = results[
            "Expected_Annual_Revenue_at_Risk"
        ].sum()
        
        
        # ============================================================
        # HIGH-RISK REVENUE
        # ============================================================
        
        high_risk = results[
            results["Risk_Level"] == "High Risk"
        ]
        
        high_risk_revenue_usd = high_risk[
            "Expected_Annual_Revenue_at_Risk"
        ].sum()
        
        
        # ============================================================
        # CONVERT REVENUE VALUES
        # ============================================================
        
        monthly_risk = convert_currency(
            monthly_risk_usd,
            from_currency="USD",
            to_currency=target_currency
        )
        
        annual_risk = convert_currency(
            annual_risk_usd,
            from_currency="USD",
            to_currency=target_currency
        )
        
        high_risk_revenue = convert_currency(
            high_risk_revenue_usd,
            from_currency="USD",
            to_currency=target_currency
        )


        # ============================================================
        # REVENUE RISK KPI CARDS
        # ============================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Monthly Revenue at Risk",
                f"{currency_symbol}{monthly_risk:,.2f}"
                )
    
        with col2:
    
            st.metric(
                "Annual Revenue at Risk",
                f"{currency_symbol}{annual_risk:,.2f}"
                )
    
        with col3:
    
            st.metric(
                "High-Risk Annual Revenue",
                f"{currency_symbol}{high_risk_revenue:,.2f}"
                )
        

# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header("🤖 Model Information")

    st.write(
        "This section provides information about the "
        "production model, preprocessing pipeline, "
        "classification threshold, and business risk framework."
    )

    # ========================================================
    # PRODUCTION COMPONENTS
    # ========================================================

    st.subheader("Production Components")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Production Model**

            `final_churn_model.pkl`

            The trained machine learning model used
            to generate customer churn predictions.
            """
        )

        st.markdown(
            """
            **Preprocessor**

            `preprocessor.pkl`

            The saved preprocessing pipeline used to
            transform customer data before prediction.
            """

        )

    with col2:

        st.markdown(
            """
            **Classification Threshold**

            `final_threshold.pkl`

            The optimized threshold used to convert
            churn probabilities into Churn / No Churn
            predictions.
            """
        )

        st.markdown(
            """
            **Inference Engine**

            `src/prediction_pipeline.py`

            The production prediction pipeline that
            connects preprocessing, prediction,
            risk scoring, revenue analysis, and
            recommended retention actions.
            """
        )


    # ========================================================
    # BUSINESS RISK FRAMEWORK
    # ========================================================

    st.subheader("Business Risk Framework")

    risk_framework = pd.DataFrame(
        {
            "Risk Level": [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],

            "Probability": [
                "< 40%",
                "40% – 69.99%",
                "≥ 70%"
            ],

            "Strategy": [
                "Maintain engagement",
                "Monitor and engage",
                "Immediate retention intervention"
            ]
        }
    )

    st.dataframe(
        risk_framework,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.caption(
        "Customer Churn "
        "Prediction Platform"
    )