import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def cifar10_classes_from_dataset(dataset):
    return getattr(dataset, "classes", [str(i) for i in range(10)])

@torch.no_grad()
def save_cifar10_grid(
    model,
    val_loader,
    device: str = "cpu",
    save_path: str = "outputs/viz/cifar10_pred_grid.png",
    max_images: int = 10,
    writer: SummaryWriter = None
):

    model.eval()
    it = iter(val_loader)
    imgs, labels = next(it)  
    classes = cifar10_classes_from_dataset(val_loader.dataset)

    n = min(max_images, imgs.shape[0])
    imgs = imgs[:n].to(device)
    labels = labels[:n].to(device)

    model = model.to("cpu")
    logits = model(imgs)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    imgs_cpu = imgs.cpu()
    labels_cpu = labels.cpu()
    preds_cpu = preds.cpu()
    probs_cpu = probs.cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imgs_show = torch.clamp(imgs_cpu * std + mean, 0, 1)

    rows = 2
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
    if rows == 1:
        axes = [axes]
    axes = axes.flatten()

    for i in range(cols*rows):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        img = imgs_show[i].permute(1,2,0).numpy()  # CHW -> HWC
        true_id = int(labels_cpu[i].item())
        pred_id = int(preds_cpu[i].item())
        correct = (true_id == pred_id)
        conf = float(probs_cpu[i, pred_id].item())

        ax.imshow(img)
        title = f"pred: {classes[pred_id]} ({conf:.2f})\ntrue: {classes[true_id]}"
        ax.set_title(
            title,
            color=("green" if correct else "red"),
            fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(2.0)

    fig.suptitle("CIFAR-10 — predictions (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {save_path}")

    if writer is not None:
        writer.add_figure("viz/cifar10_grid", fig)
    plt.close(fig)
