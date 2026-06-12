"""
explain.py — SHAP explainability module.

Provides:
    - Global explanations (SHAP summary beeswarm plot)
    - Local explanations (SHAP waterfall chart for a single prediction)
    - XGBoost native feature importance bar chart

Usage:
    python src/explain.py          # generates global plots in assets/

Programmatic usage (from Streamlit):
    from src.explain import get_shap_explainer, local_explanation
    explainer = get_shap_explainer(model)
    fig = local_explanation(explainer, X_single, feature_names)
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

# Fix Windows console encoding for emoji/unicode output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running as `python src/explain.py` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_data, preprocess_pipeline, MODELS_DIR

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


# ---------------------------------------------------------------------------
# Core SHAP Functions
# ---------------------------------------------------------------------------


def get_shap_explainer(model):
    """Create a SHAP TreeExplainer for the given XGBoost model.

    TreeExplainer provides exact SHAP values (not approximate) and is
    extremely fast for tree-based models.

    Args:
        model: Trained XGBoost model.

    Returns:
        shap.TreeExplainer instance.
    """
    return shap.TreeExplainer(model)


def global_explanation(explainer, X, feature_names, save_path=None):
    """Generate and save a SHAP summary beeswarm plot.

    Shows the distribution of SHAP values across all samples for each
    feature — revealing both importance and directionality.

    Args:
        explainer: SHAP TreeExplainer.
        X: Feature matrix (2-D array-like), typically the full test set.
        feature_names: List of feature name strings.
        save_path: Where to save the plot. Defaults to
            ``assets/shap_summary.png``.

    Returns:
        SHAP values array.
    """
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — Feature Impact on Premium", fontsize=14, pad=15)
    plt.tight_layout()

    if save_path is None:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        save_path = os.path.join(ASSETS_DIR, "shap_summary.png")

    plt.savefig(save_path, dpi=150)
    plt.close("all")
    print(f" Saved global SHAP summary: {save_path}")

    return shap_values


def local_explanation(explainer, X_single, feature_names, expected_value=None):
    """Generate a custom waterfall chart for a single prediction.

    Replaces SHAP's built-in waterfall plot (which has a known duplicate
    text-annotation bug) with a clean matplotlib implementation.

    Args:
        explainer: SHAP TreeExplainer.
        X_single: Single sample as 2-D array of shape ``(1, n_features)``.
        feature_names: List of feature name strings.
        expected_value: Model baseline. If None, uses
            ``explainer.expected_value``.

    Returns:
        matplotlib Figure object (for embedding in Streamlit).
    """
    shap_values = explainer.shap_values(X_single)

    if expected_value is None:
        expected_value = explainer.expected_value

    values = shap_values[0]
    base = float(expected_value)
    prediction = base + values.sum()

    # Sort features by absolute SHAP value (largest impact first)
    order = np.argsort(np.abs(values))[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = values[order]

    # --- Build the waterfall chart ---
    n = len(sorted_vals)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.65 + 1.5)))

    # Compute cumulative positions (top-down: start from prediction, end at base)
    cumulative = prediction
    bar_starts = []
    bar_widths = []

    for v in sorted_vals:
        bar_starts.append(cumulative - v)
        bar_widths.append(v)
        cumulative -= v

    y_positions = list(range(n - 1, -1, -1))  # top feature at top

    # X-axis range for computing label padding
    all_x = [base, prediction]
    for i in range(n):
        all_x.append(bar_starts[i])
        all_x.append(bar_starts[i] + bar_widths[i])
    x_range = max(all_x) - min(all_x)
    pad = max(x_range * 0.02, 50)  # minimum 50 dollar padding

    # Colors
    pos_color = "#ff0051"  # pink-red for positive (increases premium)
    neg_color = "#008bfb"  # blue for negative (decreases premium)

    for i in range(n):
        v = bar_widths[i]
        color = pos_color if v >= 0 else neg_color
        left = bar_starts[i] if v >= 0 else bar_starts[i] + v
        width = abs(v)

        ax.barh(
            y_positions[i], width, left=left, height=0.6,
            color=color, edgecolor="white", linewidth=0.5,
        )

        # Value label — position to the right of all bars
        if abs(v) < 1:
            label = "+0" if v >= 0 else "-0"
            label_x = max(bar_starts[i], bar_starts[i] + v) + pad
            ha = "left"
        elif v >= 0:
            label_x = bar_starts[i] + v + pad
            ha = "left"
            label = f"+${v:,.0f}"
        else:
            label_x = bar_starts[i] + pad
            ha = "left"
            label = f"-${abs(v):,.0f}"

        ax.text(
            label_x, y_positions[i], label,
            va="center", ha=ha, fontsize=10, fontweight="600",
            color=color,
        )

        # Connector lines between bars
        if i < n - 1:
            connect_y_top = y_positions[i] - 0.3
            connect_y_bot = y_positions[i + 1] + 0.3
            end_of_current = bar_starts[i] + (v if v >= 0 else 0)
            ax.plot(
                [end_of_current, end_of_current],
                [connect_y_top, connect_y_bot],
                color="#888888", linewidth=1, linestyle="--",
            )

    # Y-axis labels: feature names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_names, fontsize=11)

    # Top annotation: f(x) = prediction
    ax.text(
        0.98, 1.03, f"f(x) = ${prediction:,.0f}",
        transform=ax.transAxes,
        va="bottom", ha="right", fontsize=12, fontweight="bold",
        color="#cccccc",
    )

    # Bottom annotation: E[f(X)] = base
    ax.text(
        0.98, -0.08, f"E[f(X)] = ${base:,.0f}  (avg. premium)",
        transform=ax.transAxes,
        va="top", ha="right", fontsize=10, color="#999999",
    )

    # Baseline reference line
    ax.axvline(x=base, color="#888888", linewidth=1, linestyle="-", zorder=0)

    # Styling
    ax.set_xlabel("Premium ($)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Your Premium Breakdown — SHAP Waterfall", fontsize=14, pad=20)
    fig.tight_layout()

    return fig


def feature_importance_plot(model, feature_names, save_path=None):
    """Generate an XGBoost native feature importance bar chart.

    Args:
        model: Trained XGBoost model.
        feature_names: List of feature name strings.
        save_path: Where to save. Defaults to
            ``assets/feature_importance.png``.

    Returns:
        matplotlib Figure object.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color=colors,
        edgecolor="white",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance", fontsize=14, pad=15)
    fig.tight_layout()

    if save_path is None:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        save_path = os.path.join(ASSETS_DIR, "feature_importance.png")

    fig.savefig(save_path, dpi=150)
    print(f" Saved feature importance: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI — Generate global plots
# ---------------------------------------------------------------------------


def generate_all_explanations():
    """Generate and save all global explainability plots."""

    print("=" * 60)
    print("  RiskIQ — SHAP Explainability")
    print("=" * 60)

    # Load model
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    model = joblib.load(model_path)
    print(f" Loaded model from {model_path}")

    # Load & preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(df)

    # SHAP explainer
    explainer = get_shap_explainer(model)

    # Global summary plot
    shap_values = global_explanation(explainer, X_test, feature_names)

    # Feature importance
    feature_importance_plot(model, feature_names)

    # Sample local explanation (first test sample)
    fig = local_explanation(
        explainer, X_test.values[:1], feature_names
    )
    os.makedirs(ASSETS_DIR, exist_ok=True)
    sample_path = os.path.join(ASSETS_DIR, "shap_waterfall_sample.png")
    fig.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved sample waterfall: {sample_path}")

    print("\n" + "=" * 60)
    print("  Explainability plots saved to assets/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_explanations()
