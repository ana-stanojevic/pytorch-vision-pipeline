
import time
from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn as nn

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def pick_amp_dtype(device: str):
    if device == "mps":
        return torch.float16
    else:
        return torch.float32

@torch.no_grad()
def evaluate(model, loader, device: str, amp_dtype, channels_last: bool):
    model.eval()
    correct = 0
    total = 0
    cast_ctx = (torch.autocast(device_type=device, dtype=amp_dtype)
                if device in ("mps") else nullcontext())
    for x, y in loader:
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        x = x.to(device)
        y = y.to(device)
        with cast_ctx:
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def train_one_epoch(model, loader, criterion, optimizer, device: str, amp_dtype, channels_last: bool, grad_clip: float=0.0):
    model.train()
    running_loss = 0.0
    steps = 0
    cast_ctx = (torch.autocast(device_type=device, dtype=amp_dtype)
                if device in ("cuda","mps") else nullcontext())
    for x, y in loader:
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with cast_ctx:
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()
        steps += 1
    return running_loss / max(steps, 1)

@torch.no_grad()
def benchmark_latency(model, loader, device: str, amp_dtype, warmup: int=10, measure: int=50) -> Dict[str, float]:
    model.eval()
    it = iter(loader)
    times = []
    cast_ctx = (torch.autocast(device_type=device, dtype=amp_dtype)
                if device in ("cuda","mps") else nullcontext())
    # warmup
    for _ in range(warmup):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(loader); x, _ = next(it)
        x = x.to(device)
        _ = model(x)
    # measure
    for _ in range(measure):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(loader); x, _ = next(it)
        x = x.to(device)
        t0 = time.perf_counter()
        with cast_ctx:
            _ = model(x)
        if device == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    if not times:
        return {"latency_ms": None, "images_per_s": None, "batch": None}
    avg = sum(times) / len(times)
    bsz = x.shape[0]
    return {"latency_ms": avg*1000.0, "images_per_s": bsz/avg, "batch": bsz}

def export_onnx(model, onnx_path: str, num_classes: int, img_size: int=224):
    model = model.to("cpu").eval()
    dummy = torch.randn(1,3,img_size,img_size,dtype=torch.float32)
    dynamic_axes = {"input": {0:"batch"}, "logits": {0:"batch"}}
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes=dynamic_axes, opset_version=17)
    print(f"[ONNX] Exported to {onnx_path}")
