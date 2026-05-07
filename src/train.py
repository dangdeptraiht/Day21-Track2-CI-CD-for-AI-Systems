import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Cau hinh MLflow de su dung SQLite cuc bo (tranh loi MissingConfigException)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huân luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params     : dict chứa các siêu tham số cho RandomForestClassifier.
        data_path  : đường dẫn đến file dữ liệu huấn luyện.
        eval_path  : đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """

    # 1. Đọc dữ liệu huấn luyện và đánh giá
    if not os.path.exists(data_path) or not os.path.exists(eval_path):
        print(f"Error: Files {data_path} or {eval_path} not found.")
        return 0.0

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # 4. Ghi nhận các siêu tham số vào MLflow
        mlflow.log_params(params)

        # 5. Khởi tạo và huấn luyện RandomForestClassifier
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 6. Dự đoán trên tập đánh giá và tính chỉ số
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # 7. Ghi nhận các chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 8. Log mô hình vào MLflow artifact
        mlflow.sklearn.log_model(model, "model")

        # 9. In kết quả ra màn hình
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 10. Lưu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # 11. Lưu mô hình ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 12. Trả về acc
    return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    if os.path.exists("params.yaml"):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        train(params)
    else:
        print("Error: params.yaml not found.")
