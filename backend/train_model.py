from pathlib import Path
import pickle
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path("data/synthetic_banking_transactions.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_FILE = ARTIFACT_DIR / "fraud_model.joblib"


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    feature_cols = [
        "transaction_amount",
        "merchant_category",
        "merchant_country",
        "device_type",
        "transaction_type",
        "hour",
        "distance_from_home",
        "transactions_last_24h",
        "merchant_risk_score",
        "is_international",
        "is_card_present",
        "device_trust_score",
        "account_balance",
    ]
    target_col = "fraud_label"

    X = df[feature_cols]
    y = df[target_col]

    categorical_features = [
        "merchant_category",
        "merchant_country",
        "device_type",
        "transaction_type",
    ]
    numeric_features = [
        "transaction_amount",
        "hour",
        "distance_from_home",
        "transactions_last_24h",
        "merchant_risk_score",
        "is_international",
        "is_card_present",
        "device_trust_score",
        "account_balance",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

## Save Base Line 
    def save_basline(df):
        baseline = {
            col: df[col].values
            for col in df.columns
            if df[col].dtypes != "object"
            }
        with open("baseline.pkl", "wb") as f:
            pickle.dump(baseline, f)
    save_basline(X_train)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

    transformed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_cols,
            "transformed_feature_names": transformed_feature_names,
            "threshold": 0.5
        },
        MODEL_FILE,
    )
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()

