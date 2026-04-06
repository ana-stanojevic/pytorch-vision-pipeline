#  PyTorch Vision Pipeline 

![CI](https://github.com/ana-stanojevic/pytorch-vision-pipeline/actions/workflows/python-ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Production-minded computer vision pipeline from training to portable inference API.

This repository is built around a simple idea: the path from model training to serving should stay aligned. Instead of treating training code and deployment code as separate worlds, this project keeps them connected through shared preprocessing, ONNX export, reproducible configuration, and a lightweight FastAPI inference layer.

It is an end-to-end CIFAR-10 pipeline that covers model training, evaluation, ONNX export, ONNX Runtime inference, API serving, persistence of predictions, automated tests, and CI.


---

## Why this project

A lot of ML projects work in notebooks, look fine during training, and then break down once they are exported or wrapped in an API. This repo is designed to reduce that gap.

The main goal is to keep training and serving on a single, inspectable path:

- training in PyTorch
- export to ONNX
- inference with ONNX Runtime
- prediction serving through FastAPI
- reproducible runs with config + CI

This makes the system easier to debug, easier to benchmark, and easier to move toward production.

---

## What the system includes

### Training and evaluation
- PyTorch training pipeline for CIFAR-10
- support for both MobileNetV3-Small and ViT-Tiny
- configurable runs via YAML configs
- TensorBoard logging for metrics and comparisons
- reproducible entrypoint through a single `main.py`

### Portable inference
- ONNX export after training
- ONNX Runtime inference path for model evaluation and serving
- shared image preprocessing between exported model usage and serving code

### API layer
- FastAPI app for image prediction
- `/predict` endpoint for uploaded image files
- input validation for image uploads
- CORS-enabled service layer
- runtime ONNX session loaded on startup

### Persistence and caching
- prediction results stored in a database
- uploaded images hashed with SHA-256
- repeated predictions can be returned from cache instead of recomputed inference

### Engineering hygiene
- unit tests with `pytest`
- linting with `ruff`
- GitHub Actions CI
- Dockerfile for containerized serving

---

## Project structure

```text
pytorch-vision-pipeline/
├── api/
│   ├── main.py              # FastAPI app and /predict endpoint
│   └── deps.py              # DB session dependency
├── configs/
│   ├── cifar10_mobilenet.yaml
│   └── cifar10_vit_tiny.yaml
├── db/
│   ├── models.py            # prediction table
│   ├── session.py           # SQLAlchemy session
│   └── init_db.py           # DB initialization
├── inference/
│   ├── serving.py           # ONNX Runtime loading + prediction helper
│   └── tags.py              # business tags for predictions
├── src/
│   ├── data.py
│   ├── infer.py
│   ├── models.py
│   ├── train.py
│   ├── utils.py
│   └── viz.py
├── tests/
├── .github/workflows/
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
---
```
---
## End-to-end flow

PyTorch training
    ↓
	
evaluation + logging
    ↓
	
export to ONNX
    ↓
	
ONNX Runtime inference
    ↓
	
FastAPI /predict endpoint
    ↓
	
prediction caching in DB

---

## Quick start  

1. **Clone the repo**  
```bash
git clone https://github.com/ana-stanojevic/pytorch-vision-pipeline.git  
cd pytorch-vision-pipeline  
```

2. **Create a virtual environment and install dependencies**  
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt 
```
---

## Train a model

### MobileNetV3-Small  
```bash
# MobileNetV3-Small (MPS, AMP, ligh CNN benchmark)
python main.py --config configs/cifar10_mobilenet.yaml
```
### ViT-Tiny
```bash
# ViT-Tiny (MPS, AMP, small transformer benchmark)
python main.py --config configs/cifar10_vit_tiny.yaml
```
This runs the training pipeline and then evaluates through the inference path.

---

## Run ONNX inference only 
If you already have exported ONNX models and want to skip training:
```bash
python main.py --config configs/cifar10_vit_tiny.yaml --no-train 
python main.py --config configs/cifar10_mobilenet.yaml --no-train
```

---
## Launch TensorBoard
```bash
tensorboard --logdir outputs/logs
```
Use this to inspect:
	•	training loss
	•	validation accuracy
	•	latency comparisons
	•	experiment outputs across runs

---

## Run the API locally
The API loads an ONNX model on startup and exposes an image prediction endpoint.

### Start the server
```bash
uvicorn api.main:app --reload
```
By default, the app looks for:
```bash
outputs/models/vit_tiny_cifar10.onnx
```
You can override that with:
```bash
export ONNX_PATH=outputs/models/your_model.onnx
uvicorn api.main:app --reload
```
### Predict from an image
Example request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@example.jpg"
```
Example response:
```json
{
  "class_name": "dog",
  "confidence": 0.91,
  "tag": "pet-related",
  "source": "AI model"
}
```
If the same image was already processed before, the response may come from the DB cache:
```json
{
  "class_name": "dog",
  "confidence": 0.91,
  "tag": "pet-related",
  "source": "db_cache"
}
```
## Run with Docker
The repository includes a Dockerfile for serving the FastAPI app through Gunicorn + Uvicorn worker.

### Build
```bash
docker build -t pytorch-vision-pipeline .
```

### Run
```bash
docker run -p 8080:8080 \
  -e ONNX_PATH=outputs/models/vit_tiny_cifar10.onnx \
  pytorch-vision-pipeline
```
---
## Testing  

Run all tests:  
```bash
pytest -q
```  
---

## Tech stack  
**ML inference**
- Python 
- torchvision
- timm
- ONNX
- ONNX runtime

**API serving** 
- FastAPI
- Uvicorn
- Gunicorn
- python-multipart

**Data/infra**
- SQLAlchemy
- psycopg2-binary
- pillow
- NumPy
- PyYAML

**Tooling**
- TensorBoard
- pytest
- ruff
- GitHub Actions
- Docker

---

## What this repo is meant to show
This is not just a model training repo.

It is a compact example of how to structure a vision system so that:
	•	training and inference stay aligned
	•	export is part of the workflow, not an afterthought
	•	serving is built on the exported model path
	•	repeated predictions can be cached
	•	the system stays testable and reproducible

In short: this project is about moving from model code to a deployable inference path with fewer gaps between experimentation and serving.

## Possible next extensions
- richer API responses and request schema
- batch inference endpoint
- model/version tracking
- monitoring and structured logging
- containerized DB setup
- deployment manifest for cloud serving

## 👩‍💻 Author

**Ana Stanojevic**  
[Scholar ↗](https://bit.ly/ana-stanojevic) • [CV ↗](https://bit.ly/ana-stanojevic-cv) 

---

## 📜 License  
MIT License — free to use, modify and distribute.  
