
from torchvision import transforms
from torch.utils.data import DataLoader

def get_num_classes(dataset: str) -> int:
    ds = dataset.lower()
    if ds == "cifar10":
        return 10
    raise ValueError(f"Unsupported dataset: {dataset}")

def build_transforms(img_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf

def build_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    workers: int,
    img_size: int = 224,
):
    train_tf, val_tf = build_transforms(img_size)
    name = dataset.lower()
    if name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False, drop_last=False)
    return train_loader, val_loader
