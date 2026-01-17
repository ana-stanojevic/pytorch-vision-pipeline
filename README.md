# pytorch-vision-pipeline  

![CI](https://github.com/ana-stanojevic/pytorch-vision-pipeline/actions/workflows/python-ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Clean, reproducible CIFAR-10 training + inference pipeline in **PyTorch**.  
Demonstrates end-to-end ML workflow: dataloaders, model, training, evaluation, inference, and CI.  

---

## 📦 Features  
- CIFAR-10 dataset with train/test loaders  
- MobileNet and Vision Transformer (ViT) models in PyTorch  
- Training & evaluation loops and inference
- Visualizations: TensorBoard metrics and image grids
- Reproducible environment (`requirements.txt`)  
- Unit tests with `pytest`  
- Continuous Integration via GitHub Actions  

---

## 📂 Project structure  
```
pytorch-vision-pipeline/
├── main.py
├── src/
│   └── models.py                  
│   └── train.py          # training & evaluation loop
│   └── infer.py          # inference 
│   └── data.py          # data loader
│   └── utils.py          # helper functions
│   └── viz.py          # vizualization 
├── tests/
│   └── conftest.py     # test configuration
│   └── test_models.py     # unit tests for models
├─ configs/
│  └── cifar10_vit_tiny.yaml
│  └── cifar10_mobilenet.yaml
├── .github/workflows/
│   └── python-ci.yml     # GitHub Actions CI pipeline
├── requirements.txt      # dependencies
├── .gitignore  
├── .gitattributes      # excludes files/folders from release archives
└── README.md
Generated at runtime
- outputs/           # eval metrics, plots, models ...
- data/           # stores downloaded dataset
```
---

## 🚀 Quick start  

1. **Clone repo**  
```bash
git clone https://github.com/ana-stanojevic/pytorch-vision-pipeline.git  
cd pytorch-vision-pipeline  
```

2. **Create virtual env & install dependencies**  
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt 
```

3. **Train and inference models**  
```bash
# MobileNetV3-Small (MPS, AMP, ligh CNN benchmark)
python main.py --config configs/cifar10_mobilenet.yaml

# ViT-Tiny (MPS, AMP, small transformer benchmark)
python main.py --config configs/cifar10_vit_tiny.yaml
```

4. **Run inference with the existing .onnx models**  
```bash
python main.py --config configs/cifar10_vit_tiny.yaml --no-train 
python main.py --config configs/cifar10_mobilenet.yaml --no-train
```

5. **Training & latency at a glance (PyTorch MPS vs ONNX CPU)**
```bash
 # Launch TensorBoard to view loss, val accuracy, and latency comparisons:
tensorboard --logdir outputs/logs
```

---

## 🧪 Tests  

Run all tests:  
```bash
pytest -q
```  

---

## ❓ Why this project  
This repo is a **learning-friendly, reproducible demo** of a computer-vision pipeline— **data → model → training → evaluation → inference → CI-** built to be clean and modular. It also showcases how *lightweight, well-designed architectures* can run efficiently on modest hardware while still achieving solid performance, with the expected latency ↔ accuracy trade-offs highlighted.

Use cases:  
- 📚 Teaching / workshops (end-to-end workflow demo) 
- 🚀 Rapid prototyping under constraints (latency/memory/power).  
- 🧪 Baseline reference for experimenting with CIFAR-10  
- ⚙️ Template for scaling to more complex datasets or architectures  

---


## ⚙️ Tech stack  
**Runtime**
- Python 3.13
- torch · torchvision · timm
- onnx · onnxruntime
- numpy · pillow · pyyaml
- tensorboard

**Dev**
- pytest
- ruff
- GitHub Actions (CI)

---

## 👩‍💻 Author

**Ana Stanojevic**  
[Scholar ↗](https://bit.ly/ana-stanojevic) • [CV ↗](https://bit.ly/ana-stanojevic-cv) 

---

## 📜 License  
MIT License — free to use, modify and distribute.  
