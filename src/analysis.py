from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C

def compute_metrics(y_true, y_pred, label_names):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    per_class_acc = cm.diagonal() / cm.sum(1).clip(min=1)
    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "per_class_acc": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }

def top_bottom_classes(per_class_acc, label_names, k=5):
    order = np.argsort(per_class_acc)
    bottom = [(label_names[i], float(per_class_acc[i])) for i in order[:k]]
    top = [(label_names[i], float(per_class_acc[i])) for i in order[::-1][:k]]
    return top, bottom

def top_confusions(cm, label_names, k=10):
    cm = np.asarray(cm)
    pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((label_names[i], label_names[j], int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:k]

def efficiency(model, device, img_size=C.IMG_SIZE, n_warmup=3, n_iter=20):
    import torch

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, img_size, img_size, device=device)

    gflops = None
    try:
        from fvcore.nn import FlopCountAnalysis

        gflops = FlopCountAnalysis(model, x).total() / 1e9
    except Exception as e:
        print(f"[analysis] FLOPs indisponível: {type(e).__name__}")

    bs = 32
    xb = torch.randn(bs, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(n_iter):
            model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    img_per_s = bs * n_iter / dt
    return {
        "params_M": n_params / 1e6,
        "gflops": float(gflops) if gflops is not None else None,
        "img_per_s": float(img_per_s),
    }

def plot_history(hist, title, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist.train_loss) + 1)
    ax[0].plot(ep, hist.train_loss, label="train"); ax[0].plot(ep, hist.val_loss, label="val")
    ax[0].set_title(f"{title} — loss"); ax[0].set_xlabel("época"); ax[0].legend()
    ax[1].plot(ep, hist.train_acc, label="train"); ax[1].plot(ep, hist.val_acc, label="val")
    ax[1].set_title(f"{title} — acurácia"); ax[1].set_xlabel("época"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def plot_confusion(cm, label_names, title, out_path, normalize=True):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(label_names))); ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=7)
    ax.set_yticklabels(label_names, fontsize=7)
    ax.set_xlabel("predito"); ax.set_ylabel("verdadeiro"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)

def plot_per_class_acc(per_class_acc, label_names, title, out_path):
    import matplotlib.pyplot as plt

    order = np.argsort(per_class_acc)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([label_names[i] for i in order], [per_class_acc[i] for i in order])
    ax.set_xlabel("acurácia"); ax.set_title(title); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def gradcam(model, image_tensor, target_layer, device, class_idx=None):

    import torch
    import torch.nn.functional as F

    model.eval().to(device)
    acts, grads = {}, {}

    def fwd_hook(_, __, out): acts["v"] = out.detach()
    def bwd_hook(_, gin, gout): grads["v"] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    out = model(x)
    if class_idx is None:
        class_idx = out.argmax(1).item()
    model.zero_grad()
    out[0, class_idx].backward()

    a, g = acts["v"][0], grads["v"][0]
    weights = g.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=image_tensor.shape[1:],
                        mode="bilinear", align_corners=False)[0, 0]
    h1.remove(); h2.remove()
    return cam.cpu().numpy(), class_idx

def attention_rollout(hf_classifier, image_tensor, device,
                      head_fusion="mean", discard_ratio=0.9):

    import torch

    inner = hf_classifier.model if hasattr(hf_classifier, "model") else hf_classifier
    inner.eval().to(device)
    x = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = inner(pixel_values=x, output_attentions=True)
    attns = out.attentions
    if not attns:
        raise RuntimeError("Modelo não retornou atenções (output_attentions).")

    result = torch.eye(attns[0].size(-1), device=device)
    for att in attns:
        a = att[0]
        if head_fusion == "mean":
            a = a.mean(0)
        elif head_fusion == "max":
            a = a.max(0).values
        elif head_fusion == "min":
            a = a.min(0).values

        flat = a.view(-1)
        n_discard = int(flat.numel() * discard_ratio)
        if n_discard > 0:
            _, idx = flat.topk(n_discard, largest=False)
            keep = idx[idx % a.size(-1) != 0]
            flat[keep] = 0
            a = flat.view_as(a)

        I = torch.eye(a.size(-1), device=device)
        a = (a + I)
        a = a / a.sum(-1, keepdim=True)
        result = a @ result

    mask = result[0, 1:]
    n = int(mask.numel() ** 0.5)
    mask = mask[: n * n].reshape(n, n)
    mask = mask / (mask.max() + 1e-8)
    return mask.cpu().numpy()

def overlay_heatmap(image_chw, heat, out_path, title=""):

    import matplotlib.pyplot as plt
    import numpy as np

    img = image_chw.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.ptp() + 1e-8)
    H, W = img.shape[:2]
    heat_r = np.array(_resize(heat, (H, W)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img); ax.imshow(heat_r, cmap="jet", alpha=0.45)
    ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def _resize(arr, size):
    from PIL import Image
    im = Image.fromarray((arr * 255).astype("uint8")).resize(size[::-1], Image.BILINEAR)
    return np.asarray(im) / 255.0
