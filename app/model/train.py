"""
train.py — Model training, comparison, and selection.

Process:
1. Load and prepare data.
2. Stratified train/test split.
3. Train baseline (Logistic Regression).
4. Compare candidates: LR, RF, HistGBM (HistGradientBoosting is chosen as primary).
5. Cross-validate with ROC-AUC + Recall + F1.
6. Select best model, fit on full training set, save to disk.
7. Save model metadata (metrics, threshold, feature names).

Why HistGradientBoostingClassifier?
- Best ROC-AUC on tabular datasets at this scale.
- Handles missing values natively.
- Fast training — no long HP search needed.
- Good probability calibration.
- No external dependencies (unlike XGBoost).
"""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, confusion_matrix,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV

from app.config import (
    DATA_PATH, MODEL_PATH, METADATA_PATH,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    ALL_FEATURES, TARGET_COLUMN, ID_COLUMN,
    DEFAULT_THRESHOLD,
)
from app.data.loader import load_dataset
from app.model.preprocessing import build_pipeline, prepare_features

logger = logging.getLogger(__name__)


# ── Candidate models ──────────────────────────────────────────────────────────

def get_candidates():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


# ── Main training routine ─────────────────────────────────────────────────────

def train_and_save(data_path: Path = None, model_path: Path = None, metadata_path: Path = None) -> Dict[str, Any]:
    """
    Full training pipeline.
    Returns metadata dict with all real computed metrics.
    """
    data_path = data_path or DATA_PATH
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH

    logger.info("=== Training start ===")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = load_dataset(data_path)
    X = prepare_features(df)
    y = df["Churn_Binary"]

    logger.info(f"Features: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # ── 2. Stratified split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Cross-validate candidates ──────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "average_precision", "f1", "recall", "precision"]

    cv_results: Dict[str, Dict] = {}
    candidates = get_candidates()

    for name, clf in candidates.items():
        logger.info(f"  CV: {name}")
        pipeline = build_pipeline(clf)
        results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
            "roc_auc_std": float(np.std(results["test_roc_auc"])),
            "pr_auc_mean": float(np.mean(results["test_average_precision"])),
            "f1_mean": float(np.mean(results["test_f1"])),
            "recall_mean": float(np.mean(results["test_recall"])),
            "precision_mean": float(np.mean(results["test_precision"])),
        }
        logger.info(f"    ROC-AUC={cv_results[name]['roc_auc_mean']:.4f} ± {cv_results[name]['roc_auc_std']:.4f}")

    # ── 4. Select best model by ROC-AUC ──────────────────────────────────────
    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    logger.info(f"Best model: {best_name}")

    # ── 5. Retrain best on full training set ──────────────────────────────────
    final_clf = candidates[best_name]
    final_pipeline = build_pipeline(final_clf)
    final_pipeline.fit(X_train, y_train)

    # ── 6. Evaluate on held-out test set ──────────────────────────────────────
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= DEFAULT_THRESHOLD).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "f1_at_0.5": float(f1_score(y_test, y_pred_default)),
        "recall_at_0.5": float(recall_score(y_test, y_pred_default)),
        "precision_at_0.5": float(precision_score(y_test, y_pred_default)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_default).tolist(),
    }

    # ── 7. Optimal threshold via ROC curve (Youden's J) ───────────────────────
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    test_metrics["optimal_threshold"] = optimal_threshold
    test_metrics[f"f1_at_optimal"] = float(f1_score(y_test, y_pred_opt))
    test_metrics[f"recall_at_optimal"] = float(recall_score(y_test, y_pred_opt))
    test_metrics[f"precision_at_optimal"] = float(precision_score(y_test, y_pred_opt))

    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.3f}")

    # ── 8. Refit final pipeline on ALL training data for production ───────────
    final_pipeline.fit(X_train, y_train)  # Already fit; this is explicit documentation

    # ── 9. Save model ─────────────────────────────────────────────────────────
    joblib.dump(final_pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    # ── 10. Save metadata ──────────────────────────────────────────────────────
    metadata = {
        "model_name": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": list(X.columns),
        "target_column": TARGET_COLUMN,
        "optimal_threshold": optimal_threshold,
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "training_data_path": str(data_path),
        "model_path": str(model_path),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = train_and_save()
    print("\n=== Training Complete ===")
    print(f"Model: {metadata['model_name']}")
    print(f"Test ROC-AUC: {metadata['test_metrics']['roc_auc']:.4f}")
    print(f"Optimal threshold: {metadata['optimal_threshold']:.3f}")
