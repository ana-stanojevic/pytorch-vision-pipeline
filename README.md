# pytorch-vision-pipeline  

![CI](https://github.com/<user>/pytorch-vision-pipeline/actions/workflows/python-ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Clean, reproducible CIFAR-10 training + inference pipeline in **PyTorch**.  
Demonstrates end-to-end ML workflow: dataloaders, model, training, evaluation, inference, and CI.  

---

## 📦 Features  
- CIFAR-10 dataset with train/test loaders  
- Simple CNN model in PyTorch  
- Training & evaluation loops with progress bar  
- Inference script for single images  
- Reproducible environment (`requirements.txt`)  
- Unit tests with `pytest`  
- Continuous Integration via GitHub Actions  

---

## 📂 Project structure  
```
pytorch-vision-pipeline/
├── main.py
├── src/
│   ├── models.py                  
│   ├── train.py          # training & evaluation loop
│   ├── infer.py          # inference script
│   └── data.py          # helper functions
│   └── utils.py          # helper functions
├── tests/
│   ├── test_smoke.py     # imports & basic sanity checks
│   └── test_model.py     # unit tests for modeles
├─ data/
├─ configs/
│  ├─ cifar10_vit_tiny.yaml
│  └─ cifar10_mobilenet.yaml
├─ outputs/
│  ├─ checkpoints/
├── .github/workflows/
│   └── python-ci.yml     # GitHub Actions CI pipeline
├── requirements.txt      # dependencies
├── .gitignore
└── README.md
```
---

## 🚀 Quick start  

1. **Clone repo**  
```bash
git clone https://github.com/<user>/pytorch-vision-pipeline.git  
cd pytorch-vision-pipeline  
```

2. **Create virtual env & install dependencies**  
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt 
```

3. **Train models**  
```bash
# ViT-Tiny (MPS, AMP, benchmark)
python train.py --config configs/cifar10_vit_tiny.yaml --channels_last --benchmark

# MobileNetV3-Small
python train.py --config configs/cifar10_mobilenet.yaml --channels_last --benchmark

# ONNX export primer
python train.py --config configs/cifar10_vit_tiny.yaml --export_onnx vit_tiny_cifar10.onnx
```

4. **Run inference**  
```bash
python src/infer.py --image path/to/image.png  
```

---

## 🧪 Tests  

Run smoke tests:  
```bash
pytest -q
```  

Run all tests (verbose):  
```bash
pytest -v
```

---

## ❓ Why this project  
This repository was built as a **learning-friendly, reproducible template** for computer vision projects.  
The idea is to show how to go from *data → model → training → evaluation → inference → CI* in a clean, modular way.  

Use cases:  
- 🚀 Quick start for newcomers to PyTorch vision projects  
- 📚 Teaching / workshops (end-to-end workflow demo)  
- 🧪 Baseline reference for experimenting with CIFAR-10  
- ⚙️ Template for scaling to more complex datasets or architectures  


---


## ⚙️ Tech stack  
- Python 3.11  
- PyTorch & TorchVision  
- PyTest  
- GitHub Actions  

---

## 📋 Requirements  

```
torch==2.2.2
torchvision==0.17.2
tqdm==4.66.4
pytest==8.3.2
```

---

## 👩‍💻 Author

**Ana Stanojevic**  
[Scholar ↗](https://bit.ly/ana-stanojevic) • [CV ↗](https://bit.ly/ana-stanojevic-cv) 

---

## 📜 License  
MIT License — free to use, modify and distribute.  
