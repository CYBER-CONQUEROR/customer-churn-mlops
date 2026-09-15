from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_FILE = (
    PROJECT_ROOT / "data" / "telco_churn.csv"
)

INCOMING_FILE = (
    PROJECT_ROOT / "data" / "incoming_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" / "churn_model.joblib"
)


# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------

print("\n====================================")
print("LOADING TRAINING DATA")
print("====================================")

historical_data = pd.read_csv(HISTORICAL_FILE)

print(
    f"Historical records: "
    f"{len(historical_data)}"
)


# --------------------------------------------------
# LOAD NEW INCOMING LABELED DATA
# --------------------------------------------------

if INCOMING_FILE.exists():

    incoming_data = pd.read_csv(INCOMING_FILE)

    # received_at is metadata, not a model feature
    incoming_data = incoming_data.drop(
        columns=["received_at"],
        errors="ignore"
    )

    print(
        f"New incoming records: "
        f"{len(incoming_data)}"
    )

    # Combine historical + new labeled data
    df = pd.concat(
        [
            historical_data,
            incoming_data
        ],
        ignore_index=True
    )

    print(
        f"Combined records: {len(df)}"
    )

else:

    print(
        "No incoming data found. "
        "Using historical data only."
    )

    df = historical_data.copy()


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.drop(
    columns=["customerID"],
    errors="ignore"
)


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)


# --------------------------------------------------
# FEATURE TYPES
# --------------------------------------------------

numeric_features = (
    X.select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

categorical_features = (
    X.select_dtypes(
        include=["object", "str"]
    )
    .columns
    .tolist()
)


print(
    "\nNumeric features:",
    numeric_features
)

print(
    "\nCategorical features:",
    categorical_features
)


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    f"\nTraining records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)


# --------------------------------------------------
# MODELS
# --------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# --------------------------------------------------
# MLFLOW
# --------------------------------------------------

mlflow.set_experiment(
    "Customer Churn Experiments"
)


best_model = None
best_model_name = None
best_f1 = -1


# --------------------------------------------------
# TRAIN ALL MODELS
# --------------------------------------------------

for model_name, classifier in models.items():

    print("\n====================================")
    print(f"Training: {model_name}")
    print("====================================")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )


    with mlflow.start_run(
        run_name=model_name
    ):

        # Train
        pipeline.fit(
            X_train,
            y_train
        )


        # Predict
        predictions = pipeline.predict(
            X_test
        )


        # Metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )


        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall: {recall:.4f}"
        )

        print(
            f"F1 Score: {f1:.4f}"
        )


        # Log model information
        mlflow.log_param(
            "model_type",
            model_name
        )

        mlflow.log_param(
            "historical_records",
            len(historical_data)
        )

        mlflow.log_param(
            "incoming_records",
            len(df)
            - len(historical_data)
        )

        mlflow.log_param(
            "total_records",
            len(df)
        )


        # Log metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )


        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype"
            ]
        )


        # Find best candidate
        if f1 > best_f1:

            best_f1 = f1

            best_model = pipeline

            best_model_name = model_name


# --------------------------------------------------
# SAVE BEST MODEL
# --------------------------------------------------

print("\n====================================")
print("BEST CANDIDATE MODEL")
print("====================================")

print(
    f"Model: {best_model_name}"
)

print(
    f"F1 Score: {best_f1:.4f}"
)


MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


CANDIDATE_MODEL_FILE = (
    PROJECT_ROOT / "models" / "candidate_model.joblib"
)

joblib.dump(
    best_model,
    CANDIDATE_MODEL_FILE
)

print("\nCandidate model saved to:")
print(CANDIDATE_MODEL_FILE)

# Save candidate F1 for promotion decision
METRIC_FILE = (
    PROJECT_ROOT / "models" / "candidate_f1.txt"
)

with open(METRIC_FILE, "w") as file:
    file.write(str(best_f1))

print(f"Candidate F1: {best_f1:.4f}")


print(
    "\nBest model saved to:"
)

print(
    MODEL_FILE
)


print("\n====================================")
print("RETRAINING PIPELINE COMPLETE")
print("====================================")