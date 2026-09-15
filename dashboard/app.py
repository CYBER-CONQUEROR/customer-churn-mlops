import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn AI Platform",
    page_icon="🤖",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "churn_model.joblib"
INCOMING_FILE = PROJECT_ROOT / "data" / "incoming_data.csv"

# Allow imports from project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.pipeline import run_mlops_pipeline


# ==================================================
# LOAD PRODUCTION MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    model = load_model()

except Exception as error:
    st.error("Could not load the production model.")
    st.error(str(error))
    st.stop()


# ==================================================
# REQUIRED DATASET SCHEMA
# ==================================================

REQUIRED_COLUMNS = [
    "customerID",
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
    "TotalCharges",
    "Churn"
]


# ==================================================
# SAMPLE DATA
# ==================================================

sample_data = pd.DataFrame(
    [
        {
            "customerID": "DEMO-0001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 1026.00,
            "Churn": "Yes"
        },
        {
            "customerID": "DEMO-0002",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 55.25,
            "TotalCharges": 2652.00,
            "Churn": "No"
        }
    ]
)


# ==================================================
# HEADER
# ==================================================

st.title("🤖 Customer Churn AI Platform")

st.write(
    "Real-Time Customer Churn Prediction "
    "and Automated MLOps Monitoring Platform"
)

st.divider()


# ==================================================
# MAIN TABS
# ==================================================

prediction_tab, monitoring_tab = st.tabs(
    [
        "🔮 Real-Time Prediction",
        "⚙️ MLOps Monitoring"
    ]
)


# ==================================================
# TAB 1 — REAL-TIME PREDICTION
# ==================================================

with prediction_tab:

    st.header("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer information below to receive "
        "a real-time churn prediction."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            help="0 = No, 1 = Yes"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # --------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------

    with col2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

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

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
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

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    st.divider()

    if st.button(
        "🔍 Predict Churn",
        type="primary",
        use_container_width=True
    ):

        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
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
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        customer_df = pd.DataFrame([customer])

        try:

            prediction = model.predict(customer_df)[0]

            probability = model.predict_proba(
                customer_df
            )[0][1]

            probability_percent = round(
                float(probability) * 100,
                2
            )

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    f"⚠️ Customer is likely to CHURN — "
                    f"{probability_percent}% probability"
                )

            else:

                st.success(
                    f"✅ Customer is likely to STAY — "
                    f"{probability_percent}% churn probability"
                )

            result1, result2 = st.columns(2)

            with result1:
                st.metric(
                    "Prediction",
                    "Churn" if prediction == 1 else "Stay"
                )

            with result2:
                st.metric(
                    "Churn Probability",
                    f"{probability_percent}%"
                )

            st.write("### Churn Risk")

            st.progress(
                min(
                    max(float(probability), 0.0),
                    1.0
                )
            )

        except Exception as error:

            st.error(
                "An error occurred while making the prediction."
            )

            st.error(str(error))


# ==================================================
# TAB 2 — MLOPS MONITORING
# ==================================================

with monitoring_tab:

    st.header("⚙️ MLOps Monitoring Dashboard")

    st.write(
        "Upload newly collected labeled customer data. "
        "After validation, the platform automatically "
        "starts the MLOps monitoring pipeline."
    )


    # ==================================================
    # PRODUCTION MODEL STATUS
    # ==================================================

    st.subheader("📦 Production Model Status")

    status1, status2, status3 = st.columns(3)

    with status1:

        if MODEL_FILE.exists():

            st.success(
                "🟢 Model Online"
            )

        else:

            st.error(
                "🔴 Model Unavailable"
            )

    with status2:

        try:
            classifier_name = (
                model.named_steps[
                    "classifier"
                ].__class__.__name__
            )

        except Exception:
            classifier_name = "Churn Classifier"

        st.metric(
            "Production Model",
            classifier_name
        )

    with status3:

        if INCOMING_FILE.exists():

            try:
                existing_incoming = pd.read_csv(
                    INCOMING_FILE
                )

                st.metric(
                    "Latest Batch",
                    f"{len(existing_incoming)} records"
                )

            except Exception:

                st.metric(
                    "Latest Batch",
                    "Available"
                )

        else:

            st.metric(
                "Latest Batch",
                "None"
            )

    st.caption(
        "Production artifact: models/churn_model.joblib"
    )

    st.divider()


    # ==================================================
    # DOWNLOAD SAMPLE CSV
    # ==================================================

    st.subheader("📄 Incoming Data Schema")

    st.write(
        "Download the sample CSV to see the exact "
        "format expected by the automated pipeline."
    )

    sample_csv = sample_data.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=sample_csv,
        file_name="customer_churn_sample.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "The sample contains valid example values "
        "for every required feature."
    )


    # ==================================================
    # REQUIRED COLUMNS
    # ==================================================

    with st.expander(
        "👀 View Required Dataset Columns"
    ):

        schema_df = pd.DataFrame(
            {
                "Column": REQUIRED_COLUMNS
            }
        )

        st.dataframe(
            schema_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()


    # ==================================================
    # UPLOAD DATA
    # ==================================================

    st.subheader(
        "📤 Upload Incoming Customer Data"
    )

    st.info(
        "Upload newly collected labeled customer data. "
        "The 'Churn' column must contain Yes or No "
        "because confirmed outcomes are required "
        "for supervised retraining."
    )

    uploaded_file = st.file_uploader(
        "Choose incoming customer CSV",
        type=["csv"]
    )


    # ==================================================
    # PROCESS UPLOAD
    # ==================================================

    if uploaded_file is not None:

        try:

            incoming_data = pd.read_csv(
                uploaded_file
            )

            st.success(
                "✅ CSV uploaded successfully."
            )


            # ==================================================
            # DATASET SUMMARY
            # ==================================================

            st.subheader(
                "📊 Incoming Dataset Summary"
            )

            metric1, metric2, metric3 = (
                st.columns(3)
            )

            metric1.metric(
                "Incoming Records",
                len(incoming_data)
            )

            metric2.metric(
                "Columns",
                len(incoming_data.columns)
            )

            missing_values = (
                incoming_data
                .isnull()
                .sum()
                .sum()
            )

            metric3.metric(
                "Missing Values",
                int(missing_values)
            )


            # ==================================================
            # DATA PREVIEW
            # ==================================================

            st.subheader(
                "👀 Incoming Data Preview"
            )

            st.dataframe(
                incoming_data.head(10),
                use_container_width=True
            )


            # ==================================================
            # SCHEMA VALIDATION
            # ==================================================

            st.subheader(
                "🔎 Schema Validation"
            )

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in incoming_data.columns
            ]

            extra_columns = [
                column
                for column in incoming_data.columns
                if column not in REQUIRED_COLUMNS
            ]


            # ==================================================
            # INVALID SCHEMA
            # ==================================================

            if missing_columns:

                st.error(
                    "❌ Dataset schema is invalid."
                )

                st.write(
                    "**Missing required columns:**"
                )

                for column in missing_columns:

                    st.write(
                        f"• {column}"
                    )


            # ==================================================
            # VALID SCHEMA
            # ==================================================

            else:

                st.success(
                    "✅ Dataset schema is valid."
                )

                if extra_columns:

                    st.warning(
                        "Extra columns were detected. "
                        "They will be removed before "
                        "the batch enters the pipeline."
                    )

                    st.write(extra_columns)


                # ==================================================
                # TARGET VALIDATION
                # ==================================================

                st.subheader(
                    "🎯 Target Label Validation"
                )

                invalid_labels = (
                    ~incoming_data["Churn"]
                    .isin(["Yes", "No"])
                )

                if invalid_labels.any():

                    st.error(
                        "❌ Churn contains invalid values."
                    )

                    st.write(
                        "Allowed target values:"
                    )

                    st.code(
                        "Yes\nNo"
                    )


                # ==================================================
                # VALID LABELS
                # ==================================================

                else:

                    st.success(
                        "✅ Target labels are valid."
                    )


                    # ==================================================
                    # CHURN DISTRIBUTION
                    # ==================================================

                    st.subheader(
                        "📊 Churn Distribution"
                    )

                    churn_counts = (
                        incoming_data["Churn"]
                        .value_counts()
                        .rename_axis("Churn")
                        .reset_index(
                            name="Customers"
                        )
                    )

                    st.dataframe(
                        churn_counts,
                        use_container_width=True,
                        hide_index=True
                    )


                    # ==================================================
                    # ACCEPT BATCH
                    # ==================================================

                    st.divider()

                    st.subheader(
                        "🚀 Start Automated MLOps Pipeline"
                    )

                    st.write(
                        "Accepting this production batch will "
                        "save the data and automatically start "
                        "drift detection."
                    )

                    st.warning(
                        "If significant data drift is detected, "
                        "the platform will automatically retrain "
                        "and evaluate a candidate model."
                    )


                    if st.button(
                        "💾 Accept Data & Run Pipeline",
                        type="primary",
                        use_container_width=True
                    ):

                        # ------------------------------------------
                        # CLEAN ACCEPTED DATA
                        # ------------------------------------------

                        accepted_data = incoming_data[
                            REQUIRED_COLUMNS
                        ].copy()

                        # Add arrival timestamp
                        accepted_data[
                            "received_at"
                        ] = pd.Timestamp.now(
                            tz="UTC"
                        ).isoformat()

                        # Make sure directory exists
                        INCOMING_FILE.parent.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        # Save batch
                        accepted_data.to_csv(
                            INCOMING_FILE,
                            index=False
                        )

                        st.success(
                            "✅ Incoming production batch "
                            "accepted successfully."
                        )

                        st.write(
                            f"**Records accepted:** "
                            f"{len(accepted_data)}"
                        )


                        # ==================================================
                        # RUN AUTOMATED PIPELINE
                        # ==================================================

                        st.subheader(
                            "⚙️ Automated Pipeline Execution"
                        )

                        with st.spinner(
                            "Running drift detection and "
                            "automated MLOps pipeline..."
                        ):

                            pipeline_result = (
                                run_mlops_pipeline()
                            )


                        # ==================================================
                        # PIPELINE RESULT
                        # ==================================================

                        if pipeline_result[
                            "success"
                        ]:

                            st.success(
                                "🎉 Automated MLOps pipeline "
                                "completed successfully!"
                            )

                        else:

                            st.error(
                                "❌ MLOps pipeline failed."
                            )


                        # ==================================================
                        # PIPELINE LOGS
                        # ==================================================

                        with st.expander(
                            "📋 View Pipeline Execution Logs",
                            expanded=True
                        ):

                            st.code(
                                pipeline_result[
                                    "output"
                                ]
                            )


                        # ==================================================
                        # RELOAD MODEL AFTER PROMOTION
                        # ==================================================

                        if pipeline_result[
                            "success"
                        ]:

                            load_model.clear()

                            st.info(
                                "🔄 Production model cache "
                                "has been refreshed."
                            )

                            st.caption(
                                "If a better candidate was "
                                "promoted, future predictions "
                                "will use the updated model."
                            )


        except Exception as error:

            st.error(
                "❌ Unable to process uploaded CSV."
            )

            st.exception(error)


    # ==================================================
    # PIPELINE ARCHITECTURE
    # ==================================================

    st.divider()

    st.subheader(
        "🔄 Automated MLOps Lifecycle"
    )

    st.write(
        "Incoming production data moves through "
        "the following automated lifecycle:"
    )

    st.code(
        """
          Incoming Labeled Data
                   │
                   ▼
            Schema Validation
                   │
                   ▼
           Save Production Batch
                   │
                   ▼
            Drift Detection
              (Evidently)
                   │
                   ▼
             Drift Detected?
              /          \\
            NO            YES
            │              │
            ▼              ▼
       Keep Current     Retrain Models
          Model              │
                             ▼
                      MLflow Tracking
                             │
                             ▼
                       Best Candidate
                             │
                             ▼
                    Champion vs Candidate
                             │
                             ▼
                      Candidate Better?
                       /          \\
                     NO            YES
                     │              │
                     ▼              ▼
                   Reject        Promote
                                    │
                                    ▼
                             Production Model
                                    │
                                    ▼
                                  CI/CD
                                    │
                                    ▼
                            Cloud Deployment
        """
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Customer Churn AI Platform | "
    "Real-Time Prediction • Drift Detection • "
    "Automated Retraining • MLflow • CI/CD"
)