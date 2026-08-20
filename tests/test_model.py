import pytest
import pandas as pd
from app.model.predictor import (
    load_pipeline,
    predict_churn_risk,
    predict_customer,
    predict_hypothetical,
    predict_all_customers,
)
from app.config import ALL_FEATURES


def test_load_pipeline():
    pipeline = load_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba")


def test_predict_churn_risk_existing_customer():
    result = predict_churn_risk("7590-VHVEG")
    assert "error" not in result
    assert result["customer_id"] == "7590-VHVEG"
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_level"] in ["High", "Medium", "Low"]
    assert "prediction" in result


def test_predict_churn_risk_missing_customer():
    result = predict_churn_risk("INVALID-ID-123")
    assert "error" in result


def test_predict_customer_hypothetical():
    sample_features = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 1020.0,
    }
    result = predict_customer(sample_features)
    assert "error" not in result
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0


def test_predict_hypothetical_delta():
    result = predict_hypothetical("7590-VHVEG", {"Contract": "Two year"})
    assert "error" not in result
    assert "current" in result
    assert "hypothetical" in result
    assert "risk_delta" in result
    assert isinstance(result["risk_delta"], float)


def test_predict_all_customers():
    df = predict_all_customers()
    assert isinstance(df, pd.DataFrame)
    assert "customerID" in df.columns
    assert "risk_score" in df.columns
    assert "risk_level" in df.columns
    assert len(df) >= 7000
