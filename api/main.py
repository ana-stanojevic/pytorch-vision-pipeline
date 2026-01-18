import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import hashlib
from sqlalchemy.orm import Session
from api.deps import get_db
from inference.serving import load_onnx_session, predict_image
from db import Prediction, init_db

app = FastAPI(title="ViT CIFAR10 Inference API")

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", 'https://ana-stanojevic.com')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #FRONTEND_ORIGIN
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = None

@app.on_event("startup")
def startup():
    global runtime
    onnx_path = os.environ.get("ONNX_PATH", "outputs/models/vit_tiny_cifar10.onnx")
    runtime = load_onnx_session(onnx_path)
    init_db()

@app.post("/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    existing = db.query(Prediction).filter(Prediction.sha256 == sha256).first()
    if existing:
        return {
        "class_name": existing.class_name, 
        "confidence": existing.confidence,
        "tag": existing.tag,
        "source": "db_cache"
    }
    pred = predict_image(runtime, img)
    print (pred)
    img = Prediction(
        class_name=pred['class_name'],
        confidence=pred['confidence'],
        sha256=sha256,
        tag=pred['tag']
    )
    db.add(img)
    db.commit()
    db.refresh(img)
    return pred





