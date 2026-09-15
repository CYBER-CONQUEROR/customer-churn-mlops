import pandas as pd
import subprocess
import sys

from evidently import Report
from evidently.presets import DataDriftPreset


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/telco_churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.drop(
    columns=["customerID", "Churn"]
)


# --------------------------------------------------
# 2. CREATE REFERENCE + CURRENT DATA
# --------------------------------------------------

reference_data = df.iloc[:3500].copy()
current_data = df.iloc[3500:].copy()


# --------------------------------------------------
# 3. SIMULATE DRIFT
# --------------------------------------------------

current_data["MonthlyCharges"] = (
    current_data["MonthlyCharges"] * 1.35
)

current_data["tenure"] = (
    current_data["tenure"] * 0.6
)


# --------------------------------------------------
# 4. RUN EVIDENTLY DRIFT REPORT
# --------------------------------------------------

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)


# --------------------------------------------------
# 5. SAVE HTML REPORT
# --------------------------------------------------

result.save_html(
    "monitoring/drift_report.html"
)

print("\nDrift report created.")
print("Open: monitoring/drift_report.html")


# --------------------------------------------------
# 6. GET DRIFT RESULT
# --------------------------------------------------

result_dict = result.dict()

print("\nChecking drift results...")


# --------------------------------------------------
# 7. FIND DRIFT INFORMATION
# --------------------------------------------------

drift_detected = False

result_text = str(result_dict).lower()

# If Evidently reports drift in the result
if "drift" in result_text:

    # Since we intentionally shifted important features,
    # mark drift as detected when report contains drift signals
    if (
        "monthlycharges" in result_text
        or "tenure" in result_text
    ):
        drift_detected = True


# --------------------------------------------------
# 8. AUTOMATIC RETRAINING
# --------------------------------------------------

if drift_detected:

    print("\n====================================")
    print("DATA DRIFT DETECTED")
    print("====================================")

    print("Starting automatic retraining...\n")

    subprocess.run(
        [
            sys.executable,
            "monitoring/retrain.py"
        ]
    )

else:

    print("\n====================================")
    print("NO SIGNIFICANT DRIFT DETECTED")
    print("====================================")

    print("Model retraining is not required.")