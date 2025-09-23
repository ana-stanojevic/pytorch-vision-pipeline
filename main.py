import argparse
import random
import torch    
from src.train import train
from src.infer import infer
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="outputs/logs")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    random.seed(0)

def main():
    p = argparse.ArgumentParser(description="Modern Vision Pipeline (MPS-ready)")
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--seed", type=int, default=12345, help=("Global seed for reproducibility. "))
    p.add_argument("--no-train", action="store_true", help="Skip training loop; only run inference.")
    
    args = p.parse_args()
    set_seed(args.seed)
    print("Current working directory:", os.getcwd())
    # Train model and save it as .onnx
    if not args.no_train:
        train(args.config, writer)
    # Load .onnx model and evaulate it
    infer(args.config, writer)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
