"""
evaluate.py — Model evaluation with metrics and diagnostic plots.

Usage:
    python src/evaluate.py

Outputs (saved to assets/):
    - actual_vs_predicted.png
    - residual_distribution.png
    - residual_vs_predicted.png

Prints: MAE, RMSE, R², and cross-validated R² to console.
"""

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Fix Windows console encoding for emoji/unicode output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running as `python src/evaluate.py` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_data, preprocess_pipeline, MODELS_DIR

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def evaluate_model():
    """Load saved model, compute metrics, and generate diagnostic plots."""

    print("=" * 60)
    print("  RiskIQ — Model Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model & data
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    model = joblib.load(model_path)
    print(f" Loaded model from {model_path}")

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(df)

    # ------------------------------------------------------------------
    # 2. Predictions
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    # ------------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Cross-validation on full training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'R² Score':<25} {r2:>10.4f}")
    print(f"{'MAE':<25} {'$' + f'{mae:,.0f}':>10}")
    print(f"{'RMSE':<25} {'$' + f'{rmse:,.0f}':>10}")
    print(f"{'Cross-val R² (5-fold)':<25} {cv_scores.mean():>7.4f} ± {cv_scores.std():.4f}")

    # ------------------------------------------------------------------
    # 4. Diagnostic plots
    # ------------------------------------------------------------------
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Set visual style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plot_color = "#4C72B0"

    # --- Actual vs. Predicted ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color=plot_color, edgecolors="white", s=50)
    lims = [
        min(y_test.min(), y_pred.min()) - 1000,
        max(y_test.max(), y_pred.max()) + 1000,
    ]
    ax.plot(lims, lims, "--", color="#E74C3C", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Charges ($)")
    ax.set_ylabel("Predicted Charges ($)")
    ax.set_title("Actual vs. Predicted Insurance Charges")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(ASSETS_DIR, "actual_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n Saved: {path}")

    # --- Residual Distribution ---
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, bins=40, kde=True, color=plot_color, ax=ax)
    ax.axvline(x=0, color="#E74C3C", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual ($)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    path = os.path.join(ASSETS_DIR, "residual_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f" Saved: {path}")

    # --- Residual vs. Predicted ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, color=plot_color, edgecolors="white", s=50)
    ax.axhline(y=0, color="#E74C3C", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Charges ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residuals vs. Predicted Values")
    fig.tight_layout()
    path = os.path.join(ASSETS_DIR, "residual_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f" Saved: {path}")

    print("\n" + "=" * 60)
    print("  Evaluation complete! Run `python src/explain.py` next.")
    print("=" * 60)

    return {"r2": r2, "mae": mae, "rmse": rmse, "cv_r2_mean": cv_scores.mean()}


if __name__ == "__main__":
    evaluate_model()
