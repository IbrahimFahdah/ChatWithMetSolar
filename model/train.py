import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURE_COLS = [
    "carport_type",
    "panel_length_mm",
    "panel_width_mm",
    "panel_power_w",
    "area_length_mm",
    "number_row",
    "delivery_artic_cost_gbp",
]
TARGET_COLS = ["total_power_kw", "total_weight_kg", "total_cost_gbp"]

HERE = pathlib.Path(__file__).parent
CSV_PATH = HERE / "carport_training_data.csv"
MODEL_PATH = HERE / "model.pkl"


def train():
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    models = []
    for i, target in enumerate(TARGET_COLS):
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train[:, i])
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test[:, i], preds)
        r2 = r2_score(y_test[:, i], preds)
        mean_val = np.mean(y_test[:, i])
        print(f"{target}: MAE={mae:.2f} ({mae/mean_val*100:.1f}% of mean)  R²={r2:.4f}")
        models.append(model)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"models": models, "feature_cols": FEATURE_COLS, "target_cols": TARGET_COLS}, f)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
