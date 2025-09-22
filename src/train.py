import time

import torch.nn as nn
import torch.optim as optim

from src.data import build_dataloaders, get_num_classes
from src.models import create_model
from src.utils import (
    load_config, pick_device, pick_amp_dtype, evaluate,
    train_one_epoch, benchmark_latency, export_onnx
)
from src.viz import save_cifar10_grid

def train(config, writer):
    cfg = load_config(config)
    model_name   = cfg.get("model", "vit_tiny")
    dataset      = cfg.get("dataset", "cifar10")
    data_dir     = cfg.get("data_dir", "./data")
    epochs       = int(cfg.get("epochs", 5))
    batch_size   = int(cfg.get("batch_size", 128))
    lr           = float(cfg.get("lr", 5e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))
    workers      = int(cfg.get("workers", 4))
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    acc0 = evaluate(model, val_loader, device, amp_dtype)
    print(f"[Eval] Pre-train accuracy: {acc0*100:.2f}%")
    writer.add_scalar(f"{model_name}-acc/val", acc0, 0)

    if do_bench:
        bench = benchmark_latency(model, val_loader, device, amp_dtype)
        if bench["latency_ms"] is not None:
            print(f"[Bench] latency={bench['latency_ms']:.2f} ms | img/s={bench['images_per_s']:.2f} @batch={bench['batch']}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_ep = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, amp_dtype, grad_clip=grad_clip) 
        scheduler.step()
        acc = evaluate(model, val_loader, device, amp_dtype)
        writer.add_scalar(f"{model_name}-loss/train", loss_ep, ep)
        writer.add_scalar(f"{model_name}-acc/val", acc, ep)   
        dt = time.time() - t0
        print(f"[Epoch {ep:02d}] loss={loss_ep:.4f} | acc={acc*100:.2f}% | time={dt:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

    # if do_bench:
    #     bench = benchmark_latency(model, val_loader, device, amp_dtype)
    #     if bench["latency_ms"] is not None:
    #         print(f"[Bench] latency={bench['latency_ms']:.2f} ms | img/s={bench['images_per_s']:.2f} @batch={bench['batch']}")
    #         writer.add_scalar(f"{model_name}-latency/ms", bench["latency_ms"], ep)
    #         writer.add_scalar(f"{model_name}-throughput/img_s", bench["images_per_s"], ep)

    save_cifar10_grid(
        model,
        val_loader,
        device="cpu",
        save_path="outputs/viz/cifar10_pred_grid.png",
        max_images=10,
        writer=writer,
    )

    if onnx_out:
        export_onnx(model, onnx_out, num_classes=num_classes, img_size=img_size)
        print(f"[ONNX] Model exported to: {onnx_out}")

 


