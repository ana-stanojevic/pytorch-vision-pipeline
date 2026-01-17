import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference.serving import load_onnx_session, predict_image

app = FastAPI(title="ViT CIFAR10 Inference API")

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", 'https://ana-stanojevic.com')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
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

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")
    pred = predict_image(runtime, img)
    return pred
