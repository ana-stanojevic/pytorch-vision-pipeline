import argparse, random
import torch    
from train import train
from infer import infer

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser(description="Modern Vision Pipeline (MPS-ready)")
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--seed", type=int, default=None, help=("Global seed for reproducibility. "))

    args = p.parse_args()
    set_seed(args.seed)

    # Train model and save it as .onnx
    train(args.config)
    # Load .onnx model and evaulate it
    infer(args.config)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
