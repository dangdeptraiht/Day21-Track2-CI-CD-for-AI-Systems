import os
import json
import numpy as np
import pandas as pd
import pytest
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """Tạo dataset giả để test."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    df.iloc[:80].to_csv(train_path, index=False)
    df.iloc[80:].to_csv(eval_path, index=False)
    
    return str(train_path), str(eval_path)


def test_train_logic(tmp_path):
    """Kiểm tra hàm train hoạt động đúng."""
    train_path, eval_path = _make_temp_data(tmp_path)
    params = {"n_estimators": 10, "max_depth": 3}
    
    acc = train(params, data_path=train_path, eval_path=eval_path)
    
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert os.path.exists("outputs/metrics.json")
    assert os.path.exists("models/model.pkl")


def test_metrics_content():
    """Kiểm tra nội dung file metrics."""
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
