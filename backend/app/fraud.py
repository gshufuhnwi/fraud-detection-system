from pathlib import Path

import joblib
import pandas as pd
import shap

MODEL_FILE = Path("artifacts/fraud_model.joblib")


class FraudScorer:
    def __init__(self):
        bundle = joblib.load(MODEL_FILE)
        self.pipeline = bundle["pipeline"]
        self.feature_columns = bundle["feature_columns"]
        self.transformed_feature_names = bundle["transformed_feature_names"]

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        self.explainer = shap.TreeExplainer(self.model)

    def score_transaction(self, payload: dict) -> dict:
        input_df = pd.DataFrame([payload])[self.feature_columns]

        probability = float(self.pipeline.predict_proba(input_df)[0, 1])
        predicted_class = int(self.pipeline.predict(input_df)[0])

        risk_level = (
            "high" if probability >= 0.80
            else "medium" if probability >= 0.50
            else "low"
        )

        shap_top_features = []
        try:
            transformed = self.preprocessor.transform(input_df)

            shap_values = self.explainer.shap_values(transformed)

            if isinstance(shap_values, list):
                row_shap = shap_values[1][0]
            else:
                row_shap = shap_values[0]

            feature_pairs = []
            for feature, shap_value in zip(self.transformed_feature_names, row_shap):
                feature_pairs.append(
                    {
                        "feature": str(feature),
                        "shap_value": float(shap_value),
                    }
                )

            feature_pairs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            shap_top_features = feature_pairs[:8]

        except Exception:
            shap_top_features = []

        return {
            "fraud_probability": round(probability, 4),
            "prediction": predicted_class,
            "predicted_label": "fraud" if predicted_class == 1 else "legitimate",
            "risk_level": risk_level,
            "shap_top_features": shap_top_features,
        }