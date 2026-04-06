# PyTorch Vision Pipeline

![CI](https://github.com/ana-stanojevic/pytorch-vision-pipeline/actions/workflows/python-ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green.svg)

End-to-end vision system from training to ONNX inference and API serving.

This repository focuses on a single constraint: **training and serving must stay aligned.**

Instead of separating experimentation and deployment, the system keeps a single path:

**PyTorch → ONNX → ONNX Runtime → FastAPI**

---

## System overview

- PyTorch training pipeline (CIFAR-10)
- ONNX export as the serving interface
- ONNX Runtime for inference
- FastAPI layer for prediction serving
- Shared preprocessing across training and inference
- Config-driven runs for reproducibility
- CI and tests to keep the pipeline stable

---

## Why this exists

Most ML pipelines fail at the boundary between training and serving.

Models trained in notebooks:
- rely on implicit preprocessing  
- drift during export  
- behave differently once deployed  

This system removes that gap by enforcing:

- a single preprocessing path  
- ONNX as the canonical inference format  
- consistent evaluation before and after export  

---

## Core idea

> The exported model is not an afterthought.  
> It is the system interface.

Everything downstream — evaluation, API, caching — is built on top of ONNX.

---

## Architecture

PyTorch training
↓

evaluation (same preprocessing)
↓

export to ONNX
↓

ONNX Runtime inference
↓

FastAPI /predict endpoint
↓

prediction caching (DB)

---

## System properties

- **No training–serving drift**
- **Single path from training to inference**
- **Portable model format (ONNX)**
- **Inspectable inference layer**
- **Reproducible runs via config + CI**

---

## API layer

- FastAPI service for image prediction
- ONNX session loaded at startup
- `/predict` endpoint for uploaded images
- SHA-256 hashing for input deduplication
- cached predictions returned when available

---

## Example output

```json
{
  "class_name": "dog",
  "confidence": 0.91,
  "tag": "pet-related",
  "source": "AI model"
}
```
or from cache:
```json
{
  "class_name": "dog",
  "confidence": 0.91,
  "tag": "pet-related",
  "source": "db_cache"
}
```
---

## What this repo demonstrates

Not model performance.

But system design:
- how to keep training and inference consistent
- how to structure export as a first-class step
- how to build a minimal serving layer on top
- how to avoid silent drift between environments

---
## Tech stack

PyTorch · ONNX · ONNX Runtime · FastAPI · SQLAlchemy · Docker · pytest · ruff

---
## 👩‍💻 Author

**Ana Stanojevic**  
[Scholar ↗](https://bit.ly/ana-stanojevic) • [CV ↗](https://bit.ly/ana-stanojevic-cv) 

---

## 📜 License  
MIT License — free to use, modify and distribute.  
