import os
import random
import numpy as np
import pytest
import torch
import importlib
import pathlib
import sys

@pytest.fixture(autouse=True, scope="session")
def _set_env():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    yield

@pytest.fixture(autouse=True, scope="session")
def _add_src_to_syspath():
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    yield

@pytest.fixture(autouse=True)
def _determinism():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    yield

@pytest.fixture (autouse=True)
def mobilenet_factory():
    pytest.importorskip("timm")
    try:
        M = importlib.import_module("models")
    except ModuleNotFoundError:
        M = importlib.import_module("src.models")

    def _get_mobilenet(num_classes=3, pretrained=False):
        model, img_size = M.create_model("mobilenet", num_classes=num_classes, pretrained=pretrained)
        assert isinstance(model, torch.nn.Module)
        return model, img_size

    return _get_mobilenet
