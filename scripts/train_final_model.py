import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tickets_hr_sla_model_ready.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_random_forest_sla.pkl"


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="incumplio_sla")
    y = df["incumplio_sla"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_features = ["Customer Age"]

    categorical_features = [
        "Product Purchased",
        "Ticket Type",
        "Ticket Subject",
        "Ticket Priority",
        "Ticket Channel"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), numeric_features),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocessor),
            ("modelo", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"Modelo guardado correctamente en: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()