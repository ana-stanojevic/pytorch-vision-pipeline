# tests/test_integration_fake_training.py
import importlib
import pathlib
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

def _make_loaders(n_train=64, n_val=32, batch_size=16):
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = FakeData(size=n_train, image_size=(3, 32, 32), num_classes=10, transform=tfm)
    val_ds   = FakeData(size=n_val,   image_size=(3, 32, 32), num_classes=10, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def _epoch(model, loader, criterion, optimizer, train=True, device="cpu"):
    meter_loss, n = 0.0, 0
    if train: model.train()
    else:     model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        meter_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return meter_loss / max(1, n)

def _accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
    
# === Test 1: basic guardrails ===
def test_end_to_end_fake_training_fast(mobilenet_factory):
    device = "cpu"
    model, _ = mobilenet_factory(num_classes=10, pretrained=False)
    model = model.to(device)
    train_loader, val_loader = _make_loaders(n_train=128, n_val=64, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    with torch.no_grad():
        base_val = _epoch(model, val_loader, criterion, optimizer, train=False, device=device)

    t0 = time.time()
    train_loss = _epoch(model, train_loader, criterion, optimizer, train=True, device=device)
    val_loss   = _epoch(model, val_loader,   criterion, optimizer, train=False, device=device)
    dt = time.time() - t0

    assert torch.isfinite(torch.tensor(train_loss)), "train loss is not finite"
    assert torch.isfinite(torch.tensor(val_loss)), "val loss is not finite"
    assert dt < 30, f"test is slow (duration {dt:.1f}s); target is <30s on CPU"

    # Allow some wiggle room; but it must not "explode"
    assert val_loss < base_val * 1.20, f"val loss too high after 1 epoch: {val_loss:.3f} vs baseline {base_val:.3f}"


# === Test 2: overfit one batch (proof that model is learning) ===
def test_overfit_one_batch(mobilenet_factory):
    device = "cpu"
    model, _ = mobilenet_factory(num_classes=10, pretrained=False)
    model = model.to(device)
    train_loader, _ = _make_loaders(n_train=128, n_val=64, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    model.eval()
    with torch.no_grad():
        base_logits = model(xb)
        base_loss = criterion(base_logits, yb).item()
        base_acc = _accuracy(base_logits, yb)

    model.train()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # finalno merenje
    model.eval()
    with torch.no_grad():
        final_logits = model(xb)
        final_loss = criterion(final_logits, yb).item()
        final_acc = _accuracy(final_logits, yb)

    # asertacije
    assert final_loss <= base_loss * 0.5, f"Loss did not decrease enough: {base_loss:.3f} -> {final_loss:.3f}"
    assert final_acc >= 0.90, f"Accuracy on one batch too low: {final_acc:.2%} (baseline {base_acc:.2%})"