import pathlib
import sys
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def test_create_model_mobilenet_cpu(mobilenet_factory):
    model, img_size = mobilenet_factory(num_classes=3, pretrained=False)
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

def test_single_step_backward_updates_params(mobilenet_factory):
    model, img_size = mobilenet_factory(num_classes=10, pretrained=False)
    model.train()
    x = torch.randn(2, 3, img_size, img_size)
    y = torch.randint(0, 10, (2,))
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-2, momentum=0.9)

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    loss = criterion(model(x), y)
    assert torch.isfinite(loss), "Loss is not finite"
    opt.zero_grad(set_to_none=True)
    loss.backward()

    total_grad_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_grad_norm += float(p.grad.norm().detach().cpu())
    assert total_grad_norm > 0.0, "All gradients are zero or None after backward pass." 

    opt.step()
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "Parameters did not change after optimizer step"
