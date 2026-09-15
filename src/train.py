import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/telco_churn.csv")

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------

# customerID does not help predict churn
df = df.drop(columns=["customerID"])

# TotalCharges contains some blank strings.
# Convert them to numeric.
# Invalid/blank values become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing TotalCharges values:")
print(df["TotalCharges"].isnull().sum())


# --------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Churn"])

# Convert Yes / No target into 1 / 0
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nFeatures:", X.shape[1])
print("Target: Churn")


# --------------------------------------------------
# 4. IDENTIFY COLUMN TYPES
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

# Numerical columns:
# Replace missing values with median
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical columns:
# Replace missing values and convert text into numbers
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 6. CREATE MODEL PIPELINE
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# --------------------------------------------------
# 7. SPLIT DATA - 80% TRAIN / 20% TEST
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining customers:", len(X_train))
print("Testing customers:", len(X_test))


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 9. TEST MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 10. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "models/churn_model.joblib"
)

print("\nModel saved:")
print("models/churn_model.joblib")