import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import build_dataloaders, get_num_classes
from src.models import create_model
from src.utils import (
    load_config, pick_device, pick_amp_dtype, evaluate,
    train_one_epoch, benchmark_latency, export_onnx
)

def train(config):
    cfg = load_config(config)
    model_name   = cfg.get("model", "vit_tiny")
    dataset      = cfg.get("dataset", "cifar10")
    data_dir     = cfg.get("data_dir", "./data")
    epochs       = int(cfg.get("epochs", 5))
    batch_size   = int(cfg.get("batch_size", 128))
    lr           = float(cfg.get("lr", 5e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))
    workers      = int(cfg.get("workers", 4))
    channels_last = bool(cfg.get("channels_last", False))
    no_pretrained = bool(cfg.get("no_pretrained", False))
    grad_clip    = float(cfg.get("grad_clip", 0.0))
    do_bench     = bool(cfg.get("benchmark", False))
    onnx_out     = cfg.get("export_onnx", "")
    if onnx_out is None:
        onnx_out = ""

    device = pick_device()
    amp_dtype = pick_amp_dtype(device)
    print(f"[Device] {device} | AMP dtype = {amp_dtype}")

    num_classes = get_num_classes(dataset)
    # create model first to get preferred img_size
    model, img_size = create_model(model_name, num_classes=num_classes, pretrained=not no_pretrained)

    # build loaders with that image size
    train_loader, val_loader = build_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
    )

    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    acc0 = evaluate(model, val_loader, device, amp_dtype, channels_last=channels_last)
    print(f"[Eval] Pre-train accuracy: {acc0*100:.2f}%")

    if do_bench:
        bench = benchmark_latency(model, val_loader, device, amp_dtype)
        if bench["latency_ms"] is not None:
            print(f"[Bench] latency={bench['latency_ms']:.2f} ms | img/s={bench['images_per_s']:.2f} @batch={bench['batch']}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_ep = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, amp_dtype, channels_last=channels_last,
            grad_clip=grad_clip
        )
        scheduler.step()
        acc = evaluate(model, val_loader, device, amp_dtype, channels_last=channels_last)
        dt = time.time() - t0
        print(f"[Epoch {ep:02d}] loss={loss_ep:.4f} | acc={acc*100:.2f}% | time={dt:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

    if do_bench:
        bench = benchmark_latency(model, val_loader, device, amp_dtype)
        if bench["latency_ms"] is not None:
            print(f"[Bench] latency={bench['latency_ms']:.2f} ms | img/s={bench['images_per_s']:.2f} @batch={bench['batch']}")

    if onnx_out:
        export_onnx(model, onnx_out, num_classes=num_classes, img_size=img_size)
        print(f"[ONNX] Model exported to: {onnx_out}")


