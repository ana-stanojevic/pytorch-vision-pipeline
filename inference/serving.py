from src.utils import _read_model_io
from src.data import build_transforms
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .tags import business_tag

def load_onnx_session(onnx_path: str):
    sess, input_name, (_, C, H, _), output_name = _read_model_io(onnx_path)
    if C != 3: 
        raise ValueError("Expected RGB input")
    return {
        "session": sess,
        "input_name": input_name,
        "output_name": output_name,
        "image_size": H
    }

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def predict_image(runtime, img):
    _, val_tf = build_transforms(runtime['image_size'])
    x = val_tf(img)            # shape: (C, H, W)
    x = x.unsqueeze(0).numpy() # shape: (1, C, H, W)
    outputs = runtime['session'].run([runtime['output_name']], {runtime['input_name']: x})
    logits = outputs[0][0]     

    probs_t = F.softmax(torch.from_numpy(logits).float(), dim=-1)
    probs = probs_t.numpy()
    idx = int(np.argmax(probs))

    return {
        "class_name": CIFAR10_LABELS[idx], 
        "confidence": float(probs[idx]),
        "tag": business_tag(CIFAR10_LABELS[idx]),
        "source": "AI model"
    }

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test ONNX serving on a single image.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model (e.g. models/vit_cifar10.onnx)")
    parser.add_argument("--image", required=True, help="Path to an input image (jpg/png)")
    args = parser.parse_args()

    runtime = load_onnx_session(args.onnx)
    img = Image.open(args.image).convert("RGB")

    result = predict_image(runtime, img)
    print(result)


