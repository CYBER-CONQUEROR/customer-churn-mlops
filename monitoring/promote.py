from pathlib import Path
import shutil
import joblib
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CURRENT_MODEL = (
    PROJECT_ROOT / "models" / "churn_model.joblib"
)

CANDIDATE_MODEL = (
    PROJECT_ROOT / "models" / "candidate_model.joblib"
)

CANDIDATE_F1_FILE = (
    PROJECT_ROOT / "models" / "candidate_f1.txt"
)

HISTORICAL_FILE = (
    PROJECT_ROOT / "data" / "telco_churn.csv"
)

INCOMING_FILE = (
    PROJECT_ROOT / "data" / "incoming_data.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n====================================")
print("MODEL VALIDATION & PROMOTION")
print("====================================")

historical = pd.read_csv(HISTORICAL_FILE)

incoming = pd.read_csv(INCOMING_FILE)

incoming = incoming.drop(
    columns=["received_at"],
    errors="ignore"
)

data = pd.concat(
    [historical, incoming],
    ignore_index=True
)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data = data.drop(
    columns=["customerID"],
    errors="ignore"
)

X = data.drop(columns=["Churn"])

y = data["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)


# --------------------------------------------------
# SAME HOLDOUT SPLIT
# --------------------------------------------------

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

current_model = joblib.load(
    CURRENT_MODEL
)

candidate_model = joblib.load(
    CANDIDATE_MODEL
)


# --------------------------------------------------
# EVALUATE CURRENT MODEL
# --------------------------------------------------

current_predictions = current_model.predict(
    X_test
)

current_f1 = f1_score(
    y_test,
    current_predictions,
    zero_division=0
)


# --------------------------------------------------
# EVALUATE CANDIDATE MODEL
# --------------------------------------------------

candidate_predictions = candidate_model.predict(
    X_test
)

candidate_f1 = f1_score(
    y_test,
    candidate_predictions,
    zero_division=0
)


print(f"\nCurrent model F1:   {current_f1:.4f}")
print(f"Candidate model F1: {candidate_f1:.4f}")


# --------------------------------------------------
# PROMOTION RULE
# --------------------------------------------------

if candidate_f1 >= current_f1:

    print("\nCandidate passed validation.")

    shutil.copy2(
        CANDIDATE_MODEL,
        CURRENT_MODEL
    )

    print("\n====================================")
    print("MODEL PROMOTED")
    print("====================================")

    print(
        "Candidate model is now the "
        "production model."
    )

else:

    print("\n====================================")
    print("MODEL REJECTED")
    print("====================================")

    print(
        "Existing production model "
        "will remain active."
    )