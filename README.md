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
│   └── infer.py          # inference script
│   └── data.py          # data functions
│   └── utils.py          # helper functions
│   └── viz.py          # vizualization functions
├── tests/
│   └── test_model.py     # unit tests for models
├─ data/
├─ configs/
│  └── cifar10_vit_tiny.yaml
│  └── cifar10_mobilenet.yaml
├─ outputs/
│  └── visuals
│  └── models 
│  └── logs 
│  └── viz
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

3. **Train and inference models**  
```bash
# ViT-Tiny (MPS, AMP, benchmark)
python main.py --config configs/cifar10_vit_tiny.yaml

# MobileNetV3-Small
python main.py --config configs/cifar10_mobilenet.yaml
```

4. **Run inference on the existing models**  
```bash
python src/main.py --config configs/cifar10_vit_tiny.yaml --no_train 
python main.py --config configs/cifar10_mobilenet.yaml --no-train
```

5. **Training & latency at a glance (PyTorch MPS vs ONNX CPU)**
```bash
 #Launch TensorBoard to view loss, val accuracy, and latency comparisons:
tensorboard --logdir outputs/logs
```

---

## 🧪 Tests  

Run all tests:  
```bash
pytest -q
```  
Run only the integration test:
```
pytest -q tests/test_integration_fake_training.py
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
