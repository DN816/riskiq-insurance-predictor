"""
preprocess.py — Feature engineering and data preprocessing pipeline.

Handles:
    - Loading raw insurance data
    - Label encoding categorical features
    - BMI categorization
    - Interaction features (smoker × age, smoker × bmi)
    - StandardScaler fitting/transforming
    - Train/test splitting
    - Single-sample preprocessing for inference
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Fix Windows console encoding for emoji/unicode output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

REGION_ORDER = ["northeast", "northwest", "southeast", "southwest"]

FEATURE_COLUMNS = [
    "age", "sex", "bmi", "children", "smoker", "region",
    "bmi_category", "smoker_age", "smoker_bmi",
]

NUMERICAL_FEATURES = ["age", "bmi", "children", "smoker_age", "smoker_bmi"]

TARGET = "charges"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw insurance CSV dataset.

    Args:
        path: Path to the CSV file. Defaults to ``data/insurance.csv``.

    Returns:
        Raw DataFrame with original columns.
    """
    if path is None:
        path = DATA_PATH
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please download insurance.csv from "
            "https://www.kaggle.com/datasets/mirichoi0218/insurance "
            "and place it in the data/ directory."
        )

    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Transforms applied:
        1. Label encode ``sex`` (male=1, female=0)
        2. Label encode ``smoker`` (yes=1, no=0)
        3. Label encode ``region`` (0-3, consistent ordering)
        4. BMI category bins (underweight/normal/overweight/obese → 0-3)
        5. Interaction: ``smoker_age = smoker × age``
        6. Interaction: ``smoker_bmi = smoker × bmi``

    Args:
        df: Raw DataFrame (or copy of it).

    Returns:
        DataFrame with engineered features, all numeric.
    """
    df = df.copy()

    # --- Encode categoricals ---
    df["sex"] = df["sex"].map({"male": 1, "female": 0}).astype(int)
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0}).astype(int)
    df["region"] = df["region"].map(
        {r: i for i, r in enumerate(REGION_ORDER)}
    ).astype(int)

    # --- BMI category ---
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=[0, 1, 2, 3],  # underweight, normal, overweight, obese
    ).astype(int)

    # --- Interaction features ---
    df["smoker_age"] = df["smoker"] * df["age"]
    df["smoker_bmi"] = df["smoker"] * df["bmi"]

    return df


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Full preprocessing: engineer features → split → scale.

    Args:
        df: Raw DataFrame from :func:`load_data`.
        test_size: Fraction held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test, scaler, feature_names)``.
        The scaler is also saved to ``models/scaler.pkl``.
    """
    df = engineer_features(df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Save scaler for inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# Single-sample preprocessing (for Streamlit inference)
# ---------------------------------------------------------------------------


def preprocess_single(input_dict: dict, scaler=None) -> tuple:
    """Preprocess a single user input for model prediction.

    Args:
        input_dict: Dictionary with keys:
            ``age``, ``sex``, ``bmi``, ``children``, ``smoker``, ``region``
            where ``sex`` is ``"male"``/``"female"``, ``smoker`` is
            ``"yes"``/``"no"``, and ``region`` is one of the 4 US regions.
        scaler: Fitted StandardScaler. If None, loads from
            ``models/scaler.pkl``.

    Returns:
        Tuple of ``(X_array, feature_names)`` — a 2-D numpy array ready for
        ``model.predict()`` and the corresponding feature names.
    """
    if scaler is None:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        scaler = joblib.load(scaler_path)

    # Build a single-row DataFrame
    row = {
        "age": input_dict["age"],
        "sex": 1 if input_dict["sex"] == "male" else 0,
        "bmi": input_dict["bmi"],
        "children": input_dict["children"],
        "smoker": 1 if input_dict["smoker"] == "yes" else 0,
        "region": REGION_ORDER.index(input_dict["region"]),
    }

    # Derived features
    row["bmi_category"] = (
        0 if row["bmi"] <= 18.5
        else 1 if row["bmi"] <= 24.9
        else 2 if row["bmi"] <= 29.9
        else 3
    )
    row["smoker_age"] = row["smoker"] * row["age"]
    row["smoker_bmi"] = row["smoker"] * row["bmi"]

    df = pd.DataFrame([row])

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[feature_cols]

    # Scale numerical features
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    return df.values, feature_cols
