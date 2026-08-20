import pytest
import pandas as pd
from app.data.loader import load_dataset, get_dataset_info, get_customer_by_id
from app.data.analyzer import run_analysis
from app.config import ALL_FEATURES, TARGET_COLUMN


def test_load_dataset():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 7000
    assert TARGET_COLUMN in df.columns
    assert "Churn_Binary" in df.columns
    assert df["TotalCharges"].dtype in [float, int]


def test_get_dataset_info():
    df = load_dataset()
    info = get_dataset_info(df)
    assert info["n_rows"] >= 7000
    assert info["n_cols"] >= 20
    assert 20 <= info["churn_rate_pct"] <= 35
    assert info["is_customer_id_unique"] is True


def test_get_customer_by_id():
    df = load_dataset()
    row = get_customer_by_id("7590-VHVEG", df)
    assert row is not None
    assert row["customerID"] == "7590-VHVEG"

    missing_row = get_customer_by_id("NON-EXISTENT-ID", df)
    assert missing_row is None


def test_run_analysis_shape():
    df = load_dataset()
    res = run_analysis(df, operation="get_shape")
    assert res["success"] is True
    assert res["data"]["rows"] >= 7000


def test_run_analysis_group_by():
    df = load_dataset()
    res = run_analysis(df, operation="group_by_churn", column="Contract")
    assert res["success"] is True
    assert isinstance(res["data"], list)
    assert len(res["data"]) > 0


def test_run_analysis_correlation():
    df = load_dataset()
    res = run_analysis(df, operation="correlation")
    assert res["success"] is True
    assert "correlation_matrix" in res["data"]
