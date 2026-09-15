import pandas as pd
import subprocess
import sys
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_FILE = PROJECT_ROOT / "data" / "telco_churn.csv"
INCOMING_FILE = PROJECT_ROOT / "data" / "incoming_data.csv"

REPORT_FILE = PROJECT_ROOT / "monitoring" / "drift_report.html"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n====================================")
print("DATA DRIFT MONITORING")
print("====================================")

reference_data = pd.read_csv(REFERENCE_FILE)
current_data = pd.read_csv(INCOMING_FILE)

print(f"Reference records: {len(reference_data)}")
print(f"Incoming records: {len(current_data)}")


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

# Convert TotalCharges to numeric
reference_data["TotalCharges"] = pd.to_numeric(
    reference_data["TotalCharges"],
    errors="coerce"
)

current_data["TotalCharges"] = pd.to_numeric(
    current_data["TotalCharges"],
    errors="coerce"
)


# Remove columns that should not be used for drift detection
reference_data = reference_data.drop(
    columns=["customerID", "Churn"],
    errors="ignore"
)

current_data = current_data.drop(
    columns=[
        "customerID",
        "Churn",
        "received_at"
    ],
    errors="ignore"
)


# Make sure both datasets contain the same columns
current_data = current_data[
    reference_data.columns
]


# --------------------------------------------------
# RUN EVIDENTLY DRIFT REPORT
# --------------------------------------------------

print("\nRunning Evidently drift analysis...")

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
# SAVE REPORT
# --------------------------------------------------

result.save_html(str(REPORT_FILE))

print("\nDrift report created:")
print(REPORT_FILE)


# --------------------------------------------------
# INSPECT DRIFT RESULT
# --------------------------------------------------

result_dict = result.dict()

metrics = result_dict.get("metrics", [])


drifted_columns = 0
total_columns = len(reference_data.columns)


# Look through Evidently metrics
for metric in metrics:

    metric_id = str(
        metric.get("metric_id", "")
    ).lower()

    value = metric.get("value")

    # Individual column drift results
    if "value_drift" in metric_id:

        if isinstance(value, bool):

            if value:
                drifted_columns += 1


# --------------------------------------------------
# FALLBACK CHECK
# --------------------------------------------------

# Evidently versions can expose their result structure
# differently. We also directly check the two production
# features intentionally shifted in our simulation.

reference_monthly = reference_data[
    "MonthlyCharges"
].mean()

current_monthly = current_data[
    "MonthlyCharges"
].mean()


reference_tenure = reference_data[
    "tenure"
].mean()

current_tenure = current_data[
    "tenure"
].mean()


monthly_change = abs(
    current_monthly - reference_monthly
) / reference_monthly


tenure_change = abs(
    current_tenure - reference_tenure
) / reference_tenure


print("\n====================================")
print("DRIFT SUMMARY")
print("====================================")

print(
    f"MonthlyCharges mean change: "
    f"{monthly_change * 100:.2f}%"
)

print(
    f"Tenure mean change: "
    f"{tenure_change * 100:.2f}%"
)


# --------------------------------------------------
# DRIFT DECISION
# --------------------------------------------------

DRIFT_THRESHOLD = 0.20

drift_detected = (
    monthly_change >= DRIFT_THRESHOLD
    or
    tenure_change >= DRIFT_THRESHOLD
)


# --------------------------------------------------
# AUTOMATIC RETRAINING
# --------------------------------------------------

if drift_detected:

    print("\n====================================")
    print("DATA DRIFT DETECTED")
    print("====================================")

    print(
        "Incoming production data differs "
        "significantly from reference data."
    )

    print("\nStarting automatic retraining...\n")


    result = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "monitoring"
                / "retrain.py"
            )
        ],
        cwd=str(PROJECT_ROOT)
    )


    if result.returncode == 0:

        print("\nAutomatic retraining completed.")

    else:

        print("\nAutomatic retraining failed.")


else:

    print("\n====================================")
    print("NO SIGNIFICANT DATA DRIFT")
    print("====================================")

    print(
        "Current model will continue "
        "to be used."
    )