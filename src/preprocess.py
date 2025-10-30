# src/preprocess.py
"""
Preprocessing module for Transaction Analysis Agent

- Loads transactions CSV from data/transactions.csv
- Performs light cleaning and feature engineering
- Builds a ColumnTransformer pipeline for numeric scaling and categorical encoding
- Saves the preprocessor to artifacts/preprocessor.joblib
- Exposes `prepare_data()` utility for use by training and inference code
"""

import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import json

DATA_PATH = os.path.join("data", "transactions.csv")
ARTIFACT_DIR = os.path.join("artifacts")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "preprocess_meta.json")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse datetime and extract time features
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
            df["transaction_hour"] = df["time"].dt.hour
            df["transaction_day"] = df["time"].dt.day
            df["transaction_month"] = df["time"].dt.month
            df["transaction_weekday"] = df["time"].dt.dayofweek
        except Exception:
            pass
    return df


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple derived features useful for a baseline model:
    - hour_of_day (from transaction_hour or time)
    - amount_log (stabilize large skew)
    - amount_abs (handle negative amounts)
    - high_freq_24h (binary if num_prev_transactions_24h > threshold)
    - device_change_risk (binary if device_change_rate_30d > threshold)
    - avg_amount_ratio (current amount vs 7-day average)
    """
    df = df.copy()

    # hour_of_day - use extracted transaction_hour
    if "transaction_hour" in df.columns:
        df["hour_of_day"] = df["transaction_hour"]
    elif "time" in df.columns:
        df["hour_of_day"] = df["time"].dt.hour
    else:
        df["hour_of_day"] = 0

    # handle negative amounts (some transactions have negative values)
    df["amount_abs"] = df["amount"].abs()
    df["amount_log"] = np.log1p(df["amount_abs"])
    df["is_negative_amount"] = (df["amount"] < 0).astype(int)

    # high frequency transactions in 24h
    if "num_prev_transactions_24h" in df.columns:
        df["high_freq_24h"] = (df["num_prev_transactions_24h"] > 5).astype(int)
    else:
        df["high_freq_24h"] = 0

    # device change risk
    if "device_change_rate_30d" in df.columns:
        df["high_device_change"] = (df["device_change_rate_30d"] > 0.5).astype(int)
    else:
        df["high_device_change"] = 0

    # ratio of current amount to 7-day average
    if "avg_transaction_amount_7d" in df.columns:
        df["amount_vs_avg_ratio"] = df["amount_abs"] / (df["avg_transaction_amount_7d"] + 1e-9)
    else:
        df["amount_vs_avg_ratio"] = 1.0

    # ensure binary columns are integers
    if "is_foreign" in df.columns:
        df["is_foreign"] = df["is_foreign"].astype(int)
    
    if "is_blacklisted_merchant" in df.columns:
        df["is_blacklisted_merchant"] = df["is_blacklisted_merchant"].astype(int)

    # account age categories
    if "account_age_days" in df.columns:
        df["account_age_category"] = pd.cut(
            df["account_age_days"], 
            bins=[0, 30, 90, 365, float('inf')], 
            labels=['new', 'young', 'mature', 'old']
        ).astype(str)
    else:
        df["account_age_category"] = "unknown"

    return df


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Construct a sklearn ColumnTransformer for numeric scaling and categorical encoding.
    """
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop")

    return preprocessor


def prepare_data(df: pd.DataFrame, target_col: str = "fraud_label", save_preprocessor: bool = True) -> Tuple:
    """
    Prepare features and labels, fit preprocessor on the full dataset (or training split).
    Returns: X, y, preprocessor, feature_names (after encoding)
    """
    df = basic_feature_engineering(df)

    # Numeric features from the actual dataset
    candidate_numeric = [
        "amount", "amount_abs", "amount_log", "amount_vs_avg_ratio",
        "account_age_days", "num_prev_transactions_24h", "avg_transaction_amount_7d",
        "device_change_rate_30d", "is_foreign", "is_blacklisted_merchant",
        "high_freq_24h", "high_device_change", "is_negative_amount",
        "hour_of_day", "transaction_hour", "transaction_day", 
        "transaction_month", "transaction_weekday"
    ]
    numeric_features = [c for c in candidate_numeric if c in df.columns]

    # Categorical features from the actual dataset
    candidate_categorical = [
        "transaction_type", "channel", "merchant_category", 
        "location_country", "location_city", "currency", "account_age_category"
    ]
    categorical_features = [c for c in candidate_categorical if c in df.columns]

    # target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {list(df.columns)}")

    X = df[numeric_features + categorical_features]
    y = df[target_col].astype(int)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # fit the preprocessor
    preprocessor.fit(X)

    # derive feature names after preprocessing (for later explanation)
    # Numeric keep order then get onehot feature names
    num_names = numeric_features
    # Extract onehot categories
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = []
    if hasattr(ohe, "get_feature_names_out"):
        cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = num_names + cat_feature_names

    if save_preprocessor:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        meta = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_names": feature_names,
            "preprocessor_path": PREPROCESSOR_PATH,
            "target_column": target_col
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Preprocessor saved to: {PREPROCESSOR_PATH}")
        print(f"ℹ️ Meta saved to: {META_PATH}")

    return X, y, preprocessor, feature_names


def train_test_split_with_stratify(df: pd.DataFrame, target_col: str = "fraud_label", test_size: float = 0.2, random_state: int = 42):
    df = basic_feature_engineering(df)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    # stratify using label if available
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    X, y, preprocessor, feature_names = prepare_data(df)
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    print("Top feature names (sample):", feature_names[:20])
