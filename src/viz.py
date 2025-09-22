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
    model = model.to(device) 
    classes = cifar10_classes_from_dataset(val_loader.dataset)

    TARGET_TOTAL = 10
    TARGET_WRONG = 5
    TARGET_RIGHT = TARGET_TOTAL - TARGET_WRONG
    wrong, right = [], []  

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        for i in range(imgs.size(0)):
            true_id = int(labels[i].item())
            pred_id = int(preds[i].item())
            conf = float(probs[i, pred_id].item())
            sample = (imgs[i].detach().cpu(), true_id, pred_id, conf)
            if pred_id != true_id and len(wrong) < TARGET_WRONG:
                wrong.append(sample)
            elif pred_id == true_id and len(right) < TARGET_RIGHT:
                right.append(sample)

            if len(wrong) + len(right) >= TARGET_TOTAL:
                break
        if len(wrong) + len(right) >= TARGET_TOTAL:
            break

    samples = wrong[:TARGET_WRONG] + right[:TARGET_RIGHT]
    n = len(samples)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    rows = 2
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=300)
    if rows == 1:
        axes = [axes]
    axes = axes.flatten()

    for i in range(cols*rows):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        img_t, true_id, pred_id, conf = samples[i]
        img_show = torch.clamp(img_t * std + mean, 0, 1)
        img_show = F.interpolate(
                img_show.unsqueeze(0), scale_factor=2, mode="bilinear", align_corners=False
            ).squeeze(0)
        img_show = img_show.permute(1, 2, 0).numpy()
        correct = (true_id == pred_id)
        ax.imshow(img_show, interpolation='bicubic')

        title = f"pred: {classes[pred_id]} ({conf:.2f})\ntrue: {classes[true_id]}"
        ax.set_title(
            title,
            color=("green" if correct else "red"),
            fontsize=14,
            fontweight='bold',
            pad=6,
            backgroundcolor=(1,1,1,0.6)
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(3.0)

    fig.suptitle("CIFAR-10 — predictions (green=correct, red=wrong)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {save_path}")

    if writer is not None:
        writer.add_figure("viz/cifar10_grid", fig)
    plt.close(fig)
