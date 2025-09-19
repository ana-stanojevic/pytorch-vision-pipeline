import timm

def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    name = model_name.lower()
    if name in ["vit_tiny", "vit_tiny_patch16_224", "vit"]:
        model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
        img_size = 224
    elif name in ["mobilenetv3_small", "mobilenet"]:
        model = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes)
        img_size = 224
    else:
        raise ValueError("Supported models: vit_tiny, mobilenetv3_small")
    return model, img_size
