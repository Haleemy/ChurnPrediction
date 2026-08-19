"""
preprocessing.py — sklearn preprocessing pipeline.

Design decisions:
- OrdinalEncoder with handle_unknown='use_encoded_value' and unknown_value=-1.
  Preserves categorical distinctions (No / No phone service / No internet service).
- StandardScaler for numerics.
- ColumnTransformer to apply different transforms to different columns.
- Pipeline composes preprocessor + classifier.
- No preprocessing leakage: fitted only on training set.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from app.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes then scales numerical features.
    - Encodes categorical features ordinally, handling unknowns gracefully.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # customerID and target are excluded
    )
    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """
    Build the full sklearn Pipeline: preprocessor → classifier.
    
    Args:
        classifier: An unfitted sklearn estimator.
    
    Returns:
        An unfitted Pipeline.
    """
    preprocessor = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def prepare_features(df, feature_cols=None):
    """
    Extract feature matrix from a DataFrame.
    
    Args:
        df: pandas DataFrame with the raw data.
        feature_cols: Optional list of column names to use (defaults to ALL_FEATURES).
    
    Returns:
        X: Feature DataFrame ready to pass to pipeline.fit() or pipeline.predict().
    """
    from app.config import ALL_FEATURES
    cols = feature_cols or ALL_FEATURES
    # Ensure all expected columns are present
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for feature preparation: {missing}")
    return df[cols].copy()
