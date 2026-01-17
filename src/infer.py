from pathlib import Path
from src.data import build_dataloaders
from src.utils import (
    load_config, _read_model_io, evaluate_onnx, benchmark_onnx
)

def infer(config, writer):
    cfg = load_config(config)
    model_name   = cfg.get("model", "vit_tiny")
    dataset      = cfg.get("dataset", "cifar10")
    data_dir     = cfg.get("data_dir", "./data")
    batch_size   = int(cfg.get("batch_size", 128))
    workers      = int(cfg.get("workers", 4))
    do_bench     = bool(cfg.get("benchmark", False))
    onnx_out     = cfg.get("export_onnx", "")
    if onnx_out is None:
        onnx_out = ""
    onnx_out = Path(onnx_out)    
    assert onnx_out.exists(), f"ONNX model not found: {onnx_out}"
    sess, input_name, (N, C, H, W), output_name = _read_model_io(str(onnx_out))
    img_size = H
    assert N == 1, f"Expected dynamic batch size, got N={N}"
    assert C == 3, f"Expected 3-channel input, got C={C}"

    _, val_loader = build_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
    )

    acc = evaluate_onnx(sess, input_name, output_name, val_loader)
    writer.add_scalar(f"{model_name}-onnx-acc/val", acc, 0)
    print(f"[ONNX Eval] CIFAR-10 accuracy: {acc*100:.2f}% (img_size={img_size})")

    if do_bench:
        bench = benchmark_onnx(sess, input_name, val_loader, warmup=5, measure=20)
        if bench["latency_ms"] is not None:
            print(f"[ONNX Bench] latency={bench['latency_s']:.2f} ms | img/s={bench['images_per_s']:.2f} @batch={bench['batch']}")
            writer.add_scalar(f"{model_name}-onnx-latency/ms", bench["latency_ms"], 0)
            writer.add_scalar(f"{model_name}-onnx-throughput/img_s", bench["images_per_s"], 0)
