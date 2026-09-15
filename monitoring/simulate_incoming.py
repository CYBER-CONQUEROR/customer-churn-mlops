import pandas as pd
from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = PROJECT_ROOT / "data" / "telco_churn.csv"
INCOMING_FILE = PROJECT_ROOT / "data" / "incoming_data.csv"


# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------

df = pd.read_csv(SOURCE_FILE)

print("\n====================================")
print("SIMULATING INCOMING PRODUCTION DATA")
print("====================================")

print(f"Historical records available: {len(df)}")


# --------------------------------------------------
# CREATE A NEW INCOMING BATCH
# --------------------------------------------------

# Randomly select 500 records.
# random_state makes the demo reproducible.

incoming_data = df.sample(
    n=500,
    random_state=42
).copy()


# --------------------------------------------------
# SIMULATE REAL-WORLD DATA DRIFT
# --------------------------------------------------

incoming_data["MonthlyCharges"] = (
    incoming_data["MonthlyCharges"] * 1.35
)

incoming_data["tenure"] = (
    incoming_data["tenure"] * 0.60
)


# Make tenure realistic integer values
incoming_data["tenure"] = (
    incoming_data["tenure"]
    .round()
    .astype(int)
)


# --------------------------------------------------
# ADD ARRIVAL TIMESTAMP
# --------------------------------------------------

incoming_data["received_at"] = datetime.now().isoformat()


# --------------------------------------------------
# SAVE NEW PRODUCTION BATCH
# --------------------------------------------------

incoming_data.to_csv(
    INCOMING_FILE,
    index=False
)


print(f"Incoming records created: {len(incoming_data)}")

print("\nSimulated drift:")
print("MonthlyCharges increased by 35%")
print("Tenure reduced by 40%")

print("\nIncoming data saved to:")
print(INCOMING_FILE)

print("\n====================================")
print("INCOMING DATA READY")
print("====================================")