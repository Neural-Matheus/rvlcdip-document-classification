import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, models as M, training as T, analysis as A

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()

WANT = ["email", "form", "scientific publication", "invoice"]

def pick_images():
    rows = list(csv.DictReader(open(C.MANIFEST_DIR / "test.csv")))
    chosen = {}
    for r in rows:
        lab = int(r["label"]); name = LABELS[lab]
        if name in WANT and name not in chosen:
            chosen[name] = r["filepath"]
        if len(chosen) == len(WANT):
            break
    return [(n, chosen[n]) for n in WANT if n in chosen]

def train_baseline():
    spec = C.MODELS["baseline_cnn"]
    tf_tr = data.build_transforms(spec, train=True, aug="document")
    tf_ev = data.build_transforms(spec, train=False)
    tr = data.make_loader(C.MANIFEST_DIR / "train.csv", tf_tr, shuffle=True)
    va = data.make_loader(C.MANIFEST_DIR / "val.csv", tf_ev)
    model = M.build_model(spec)
    model, _ = T.train(model, tr, va, device, epochs=15, lr=1e-3, patience=4)
    return model

def main():
    imgs = pick_images()
    print(f"[interp] imagens: {[n for n,_ in imgs]}")

    print("[interp] treinando BaselineCNN p/ Grad-CAM...")
    cnn = train_baseline().to(device).eval()
    last_conv = cnn.features[-1][3]

    print("[interp] carregando DiT-rvlcdip (teto) p/ rollout...")
    dit = M.build_model(C.MODELS["dit_rvlcdip"]).to(device).eval()

    spec_cnn, spec_dit = C.MODELS["baseline_cnn"], C.MODELS["dit_rvlcdip"]
    tf_cnn = data.build_transforms(spec_cnn, train=False)
    tf_dit = data.build_transforms(spec_dit, train=False)

    fig, axes = plt.subplots(len(imgs), 3, figsize=(10, 3.2 * len(imgs)))
    if len(imgs) == 1:
        axes = axes[None, :]
    for i, (name, rel) in enumerate(imgs):
        pil = Image.open(C.ROOT / rel)
        x_cnn = tf_cnn(pil); x_dit = tf_dit(pil)
        cam, _ = A.gradcam(cnn, x_cnn, last_conv, device)
        roll = A.attention_rollout(dit, x_dit, device)

        base = x_cnn.cpu().numpy().transpose(1, 2, 0)
        base = (base - base.min()) / (np.ptp(base) + 1e-8)
        roll_r = A._resize(roll, base.shape[:2])

        axes[i, 0].imshow(base); axes[i, 0].set_title(f"{name}\n(original)", fontsize=9)
        axes[i, 1].imshow(base); axes[i, 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[i, 1].set_title("Grad-CAM (CNN)", fontsize=9)
        axes[i, 2].imshow(base); axes[i, 2].imshow(roll_r, cmap="jet", alpha=0.45)
        axes[i, 2].set_title("Attention rollout (DiT)", fontsize=9)
        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle("Onde cada família olha: CNN (Grad-CAM) vs DiT (attention rollout)",
                 fontsize=12)
    fig.tight_layout()
    out = C.PLOTS_DIR / "interpretability_cnn_vs_dit.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[interp] salvo: {out}")

if __name__ == "__main__":
    main()
