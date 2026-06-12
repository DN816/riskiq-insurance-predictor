"""
streamlit_app.py — RiskIQ Interactive Insurance Premium Predictor.

Usage:
    streamlit run app/streamlit_app.py

Features:
    1. Sidebar input panel (age, sex, BMI, children, smoker, region)
    2. Predicted premium display with color coding
    3. Per-prediction SHAP waterfall explanation
    4. Global SHAP summary and feature importance
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_single, MODELS_DIR, REGION_ORDER
from src.explain import get_shap_explainer, local_explanation

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RiskIQ — Insurance Premium Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Header styling */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* Premium display card — dark-mode friendly */
    .premium-card {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .premium-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }

    .premium-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .premium-low { color: #34d399; }
    .premium-med { color: #fbbf24; }
    .premium-high { color: #f87171; }

    .premium-context {
        font-size: 0.85rem;
        color: #9ca3af;
    }

    /* Factor cards — dark-mode friendly */
    .factor-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }

    .factor-title {
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.3rem;
    }

    .factor-desc {
        font-size: 0.85rem;
        color: #9ca3af;
    }

    /* Sidebar — keep native dark theme, just add label visibility */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e5e7eb !important;
    }

    [data-testid="stSidebar"] label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }

    /* Sidebar section title */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 0.3rem;
    }

    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }

    /* Section headers — dark-mode friendly */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e5e7eb;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Model (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    """Load the trained XGBoost model and scaler."""
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        st.error(
            " Model not found! Please run `python src/train.py` first."
        )
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@st.cache_resource
def load_explainer(_model):
    """Create and cache the SHAP TreeExplainer."""
    return get_shap_explainer(_model)


@st.cache_data
def get_dataset_stats():
    """Load the dataset and compute summary statistics for context."""
    data_path = os.path.join(PROJECT_ROOT, "data", "insurance.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return {
            "mean": df["charges"].mean(),
            "median": df["charges"].median(),
            "min": df["charges"].min(),
            "max": df["charges"].max(),
            "std": df["charges"].std(),
        }
    return None


# ---------------------------------------------------------------------------
# Sidebar — User Inputs
# ---------------------------------------------------------------------------

st.sidebar.markdown("## ‍ Your Profile")
st.sidebar.markdown("Enter your details to predict your annual premium.")
st.sidebar.markdown("---")

age = st.sidebar.slider("**Age**", min_value=18, max_value=64, value=30, step=1)

sex = st.sidebar.radio("**Sex**", options=["male", "female"], horizontal=True)

bmi = st.sidebar.slider(
    "**BMI** (Body Mass Index)",
    min_value=15.0, max_value=55.0, value=26.0, step=0.1,
    help="18.5–24.9 = Normal, 25–29.9 = Overweight, 30+ = Obese",
)

children = st.sidebar.slider(
    "**Dependents**", min_value=0, max_value=5, value=0, step=1
)

smoker = st.sidebar.radio(
    "**Smoker?**", options=["no", "yes"], horizontal=True
)

region = st.sidebar.selectbox(
    "**Region**",
    options=REGION_ORDER,
    format_func=lambda x: x.title(),
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(" **Predict Premium**", use_container_width=True)

# ---------------------------------------------------------------------------
# Main Content — Header
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title"> RiskIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    "Predict your annual insurance premium and understand exactly "
    "what drives the cost — powered by XGBoost & SHAP."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Prediction Flow
# ---------------------------------------------------------------------------

if predict_btn:
    model, scaler = load_model()
    explainer = load_explainer(model)
    stats = get_dataset_stats()

    # Build input dict
    input_dict = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }

    # Preprocess & predict
    X_input, feature_names = preprocess_single(input_dict, scaler)
    prediction = model.predict(X_input)[0]

    # ---- Section 1: Premium Display ----
    st.markdown('<div class="section-header"> Your Estimated Annual Premium</div>',
                unsafe_allow_html=True)

    # Color code the premium
    if prediction < 10000:
        color_class = "premium-low"
        tier = "Low"
    elif prediction < 25000:
        color_class = "premium-med"
        tier = "Moderate"
    else:
        color_class = "premium-high"
        tier = "High"

    context_html = ""
    if stats:
        pctile = (
            sum(1 for _ in [] if True)  # placeholder
        )
        context_html = (
            f'<div class="premium-context">'
            f"Dataset average: ${stats['mean']:,.0f} · "
            f"Median: ${stats['median']:,.0f} · "
            f"Range: ${stats['min']:,.0f} – ${stats['max']:,.0f}"
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="premium-card">
            <div class="premium-label">Estimated Annual Premium</div>
            <div class="premium-value {color_class}">${prediction:,.2f}</div>
            <div class="premium-context">Risk tier: <strong>{tier}</strong></div>
            {context_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Section 2: Top Factors (plain language) ----
    st.markdown(
        '<div class="section-header"> Top Factors Driving Your Premium</div>',
        unsafe_allow_html=True,
    )

    shap_values = explainer.shap_values(X_input)
    abs_shap = np.abs(shap_values[0])
    sorted_idx = np.argsort(abs_shap)[::-1]

    # Human-readable feature descriptions
    feature_descriptions = {
        "smoker": "Smoker status has the single largest impact on insurance cost.",
        "smoker_bmi": "The interaction of smoking and BMI compounds healthcare risk.",
        "smoker_age": "Smoking combined with higher age significantly raises premiums.",
        "age": "Older individuals tend to have higher medical expenses.",
        "bmi": "Higher BMI correlates with increased health risks and costs.",
        "bmi_category": "Your BMI category (normal/overweight/obese) affects pricing.",
        "children": "More dependents mean higher coverage costs.",
        "sex": "Gender has a minor influence on actuarial pricing.",
        "region": "Your geographic region affects local healthcare costs.",
    }

    cols = st.columns(3)
    for i, idx in enumerate(sorted_idx[:3]):
        fname = feature_names[idx]
        shap_val = shap_values[0][idx]
        direction = "↑ Increases" if shap_val > 0 else "↓ Decreases"
        desc = feature_descriptions.get(fname, "This feature influences your premium.")

        with cols[i]:
            st.markdown(
                f"""
                <div class="factor-card">
                    <div class="factor-title">#{i+1} — {fname.replace('_', ' ').title()}</div>
                    <div class="factor-desc">
                        {direction} premium by ~${abs(shap_val):,.0f}<br>
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Section 3: SHAP Waterfall (Local) ----
    st.markdown(
        '<div class="section-header"> Your Premium Breakdown — SHAP Waterfall</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This chart shows how each feature pushed your prediction above or "
        "below the average premium (baseline)."
    )

    waterfall_fig = local_explanation(explainer, X_input, feature_names)
    st.pyplot(waterfall_fig)
    plt.close(waterfall_fig)

    # ---- Section 4: Global Context ----
    st.markdown(
        '<div class="section-header"> Population Context — Global Feature Impact</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "How features impact premiums across the entire dataset "
        "(not just your prediction)."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        # Global SHAP summary image if it exists
        summary_path = os.path.join(PROJECT_ROOT, "assets", "shap_summary.png")
        if os.path.exists(summary_path):
            st.image(summary_path, caption="SHAP Summary (All Samples)")
        else:
            st.info(
                "Run `python src/explain.py` to generate the global SHAP summary plot."
            )

    with col_b:
        # Feature importance image if it exists
        importance_path = os.path.join(PROJECT_ROOT, "assets", "feature_importance.png")
        if os.path.exists(importance_path):
            st.image(importance_path, caption="XGBoost Feature Importance")
        else:
            st.info(
                "Run `python src/explain.py` to generate the feature importance plot."
            )

else:
    # --- Landing state: show instructions ---
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Input")
        st.markdown(
            "Fill in your profile using the sidebar — age, BMI, "
            "smoker status, and more."
        )

    with col2:
        st.markdown("### Predict")
        st.markdown(
            "Our XGBoost model, tuned with GridSearchCV, "
            "estimates your annual premium."
        )

    with col3:
        st.markdown("### Understand")
        st.markdown(
            "SHAP explains *why* — see exactly which factors "
            "drive your cost up or down."
        )

    st.markdown("---")
    st.markdown(
        " **Fill in your details in the sidebar and click Predict Premium** "
        "to get started!"
    )
