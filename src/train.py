"""
train.py — Train XGBoost regressor with GridSearchCV hyperparameter tuning.

Usage:
    python src/train.py

Outputs:
    - models/xgb_model.pkl  (best model)
    - models/scaler.pkl     (fitted StandardScaler, saved by preprocess.py)
"""

import os
import sys
import time
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Fix Windows console encoding for emoji/unicode output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running as `python src/train.py` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_data, preprocess_pipeline, MODELS_DIR


def train_model():
    """Load data, preprocess, tune XGBoost, and save the best model."""

    print("=" * 60)
    print("  RiskIQ — Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(df)

    print(f"\n Training set : {X_train.shape[0]} samples")
    print(f" Test set     : {X_test.shape[0]} samples")
    print(f" Features     : {feature_names}")

    # ------------------------------------------------------------------
    # 2. Hyperparameter grid
    # ------------------------------------------------------------------
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    total_fits = (
        len(param_grid["n_estimators"])
        * len(param_grid["max_depth"])
        * len(param_grid["learning_rate"])
        * len(param_grid["subsample"])
        * 5  # cv folds
    )
    print(f"\n GridSearchCV: {total_fits} fits (5-fold CV)")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    # ------------------------------------------------------------------
    # 3. Fit
    # ------------------------------------------------------------------
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n⏱  Training completed in {elapsed:.1f}s")
    print(f"\n Best parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"   {k}: {v}")
    print(f"\n Best CV R² score: {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluate on test set
    # ------------------------------------------------------------------
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f" Test R² score : {test_score:.4f}")

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n Model saved to {model_path}")

    # Also save feature names for downstream use
    meta_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    joblib.dump(feature_names, meta_path)
    print(f" Feature names saved to {meta_path}")

    print("\n" + "=" * 60)
    print("  Training complete! Run `python src/evaluate.py` next.")
    print("=" * 60)

    return best_model, scaler, feature_names


if __name__ == "__main__":
    train_model()
