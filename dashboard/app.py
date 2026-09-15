import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "churn_model.joblib"

    return joblib.load(model_path)


try:
    model = load_model()

except Exception as error:

    st.error("Could not load the trained churn model.")

    st.error(str(error))

    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 Customer Churn Prediction Platform")

st.write(
    "Enter the customer's information below to predict "
    "whether the customer is likely to churn."
)

st.divider()


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.subheader("Customer Information")


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


# --------------------------------------------------
# PHONE SERVICES
# --------------------------------------------------

st.subheader("Phone Services")


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


# --------------------------------------------------
# INTERNET SERVICES
# --------------------------------------------------

st.subheader("Internet Services")


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


# --------------------------------------------------
# BILLING INFORMATION
# --------------------------------------------------

st.subheader("Billing Information")


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


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Predict Churn",
    type="primary",
    use_container_width=True
):

    # Customer data must use the same feature names
    # that were used during model training.

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


    try:

        # Convert customer input into DataFrame
        customer_df = pd.DataFrame([customer])


        # Make prediction
        prediction_value = model.predict(
            customer_df
        )[0]


        # Get prediction probabilities
        probabilities = model.predict_proba(
            customer_df
        )[0]


        churn_probability = float(
            probabilities[1]
        )


        probability_percent = round(
            churn_probability * 100,
            2
        )


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.subheader("Prediction Result")


        if prediction_value == 1:

            st.error(
                f"⚠️ Customer is likely to CHURN "
                f"— {probability_percent}% probability"
            )

        else:

            st.success(
                f"✅ Customer is likely to STAY "
                f"— Churn probability: "
                f"{probability_percent}%"
            )


        # Probability progress bar
        st.write("### Churn Risk")

        st.progress(
            min(
                max(churn_probability, 0.0),
                1.0
            )
        )


        st.metric(
            label="Churn Probability",
            value=f"{probability_percent}%"
        )


    except Exception as error:

        st.error(
            "An error occurred while making "
            "the prediction."
        )

        st.error(str(error))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "MLOps Real-Time AI Prediction Platform"
)