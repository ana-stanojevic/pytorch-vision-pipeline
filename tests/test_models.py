import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def test_create_model_mobilenet_cpu(model_factory):
    model, img_size = model_factory('mobilenet', num_classes=3, pretrained=False)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module"
    assert isinstance(img_size, int) and img_size > 0,  "Invalid img_size"

    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        y = model(x)

    assert y.ndim >= 2, "Output tensor is not at least 2D"
    assert y.shape[0] == 1, "Batch size of output tensor is not 1"
    assert y.shape[-1] == 3, "Output tensor does not have correct number of classes"
    assert torch.isfinite(y).all()

def test_create_model_vit_cpu(model_factory):
    model, img_size = model_factory('vit', num_classes=3, pretrained=False)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module"
    assert isinstance(img_size, int) and img_size > 0,  "Invalid img_size"

    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        y = model(x)

    assert y.ndim >= 2, "Output tensor is not at least 2D"
    assert y.shape[0] == 1, "Batch size of output tensor is not 1"
    assert y.shape[-1] == 3, "Output tensor does not have correct number of classes"
    assert torch.isfinite(y).all()

