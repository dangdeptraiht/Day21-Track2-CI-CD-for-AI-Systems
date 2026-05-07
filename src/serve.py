from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Cau hinh duong dan model
GCS_BUCKET = os.environ.get("GCS_BUCKET", "default-bucket")
GCS_MODEL_KEY = "models/latest/model.pkl"
# Mac dinh lay model tu thu muc models/ trong project neu chay cuc bo
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")


def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động (chỉ dùng trên Cloud)."""
    if os.environ.get("DOWNLOAD_MODEL", "false").lower() == "true":
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_MODEL_KEY)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            blob.download_to_filename(MODEL_PATH)
            print(f"Model downloaded from gs://{GCS_BUCKET}/{GCS_MODEL_KEY}")
        except Exception as e:
            print(f"Warning: Could not download model from GCS: {e}")


# Khoi tao model
if os.environ.get("DOWNLOAD_MODEL", "false").lower() == "true":
    download_model()

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    model = None
    print(f"Warning: Model file not found at {MODEL_PATH}")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "không xác định")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
