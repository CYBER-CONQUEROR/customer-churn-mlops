import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("data/telco_churn.csv")

df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})


# ----------------------------
# COLUMN TYPES
# ----------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()


# ----------------------------
# PREPROCESSING
# ----------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ----------------------------
# MODELS
# ----------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ----------------------------
# MLFLOW EXPERIMENT
# ----------------------------

mlflow.set_experiment(
    "Customer Churn Experiments"
)


best_model = None
best_model_name = None
best_f1 = 0


# ----------------------------
# TRAIN EACH MODEL
# ----------------------------

for model_name, classifier in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )


        # LOG METRICS
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


        # LOG MODEL NAME
        mlflow.log_param(
            "model_type",
            model_name
        )


        # LOG MODEL
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"]
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


        # FIND BEST MODEL
        if f1 > best_f1:

            best_f1 = f1
            best_model = pipeline
            best_model_name = model_name


# ----------------------------
# RESULT
# ----------------------------

print("\n==============================")
print("BEST MODEL")
print("==============================")

print(
    f"Model: {best_model_name}"
)

print(
    f"F1 Score: {best_f1:.4f}"
)


# SAVE BEST MODEL
import joblib

joblib.dump(
    best_model,
    "models/churn_model.joblib"
)

print(
    "\nBest model saved to models/churn_model.joblib"
)