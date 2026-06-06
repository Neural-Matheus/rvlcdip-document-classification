import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, analysis as A
from src import multimodal as MM
from scripts.run_layoutlmv3 import subsample

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()
K = C.NUM_CLASSES

MM_TRAIN = int(os.environ.get("MM_TRAIN", 150))
MM_VAL = int(os.environ.get("MM_VAL", 40))
MM_TEST = int(os.environ.get("MM_TEST", 100))
FEATS_CACHE = C.METRICS_DIR / "crossmodal_feats.npz"
DONUT_ID = "naver-clova-ix/donut-base-finetuned-rvlcdip"

@torch.no_grad()
def dit_feats(csv_path):
    from transformers import AutoModel
    spec = C.MODELS["dit_rvlcdip"]
    m = AutoModel.from_pretrained(spec.hf_or_timm_id).to(device).eval()
    tf = data.build_transforms(spec, train=False)
    loader = data.make_loader(csv_path, tf, batch_size=32, shuffle=False)
    F, Y = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = m(pixel_values=x)
        F.append(out.last_hidden_state.float().mean(1).cpu().numpy())
        Y.append(y.numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(F), np.concatenate(Y)

@torch.no_grad()
def layoutlmv3_feats(csv_path):
    from transformers import LayoutLMv3Model
    from torch.utils.data import DataLoader
    m = LayoutLMv3Model.from_pretrained(MM.LAYOUTLMV3_ID).to(device).eval()
    proc = MM.get_processor()
    loader = DataLoader(MM.make_layout_dataset(csv_path, proc), batch_size=8,
                        shuffle=False, num_workers=4)
    F = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = m(**batch)
        F.append(out.last_hidden_state.float()[:, 0].cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(F)

@torch.no_grad()
def donut_feats(fps):
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    proc = DonutProcessor.from_pretrained(DONUT_ID)
    enc = VisionEncoderDecoderModel.from_pretrained(
        DONUT_ID, torch_dtype=torch.float16).to(device).eval().encoder
    F, t0 = [], time.time()
    bs = 4
    for i in range(0, len(fps), bs):
        imgs = [Image.open(C.ROOT / fp).convert("RGB") for fp in fps[i:i + bs]]
        pv = proc(imgs, return_tensors="pt").pixel_values.to(device, torch.float16)
        out = enc(pv)
        F.append(out.last_hidden_state.float().mean(1).cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"  donut-enc {i+bs}/{len(fps)} ({time.time()-t0:.0f}s)")
    del enc; torch.cuda.empty_cache()
    return np.concatenate(F)

def read_fps(csv_path):
    import csv as _csv
    return [r["filepath"] for r in _csv.DictReader(open(csv_path))]

def build_feats():
    if FEATS_CACHE.exists() and os.environ.get("FORCE", "0") != "1":
        print(f"[xmodal] carregando features do cache {FEATS_CACHE.name}")
        d = np.load(FEATS_CACHE)
        return {k: d[k] for k in d.files}

    mm = C.MANIFEST_DIR
    tr = subsample(mm / "train.csv", MM_TRAIN, mm / "_mm_train.csv")
    va = subsample(mm / "val.csv", MM_VAL, mm / "_mm_val.csv")
    te = subsample(mm / "test.csv", MM_TEST, mm / "_mm_test.csv")
    print("[xmodal] OCR (cacheado)...")
    for c in (tr, va, te):
        MM.ensure_ocr(c)

    out = {}
    for split, csvp in [("tr", tr), ("va", va), ("te", te)]:
        print(f"[xmodal] features {split}: DiT...")
        dit, y = dit_feats(csvp)
        print(f"[xmodal] features {split}: LayoutLMv3...")
        llv3 = layoutlmv3_feats(csvp)
        print(f"[xmodal] features {split}: Donut encoder...")
        dn = donut_feats(read_fps(csvp))
        out[f"dit_{split}"] = dit
        out[f"llv3_{split}"] = llv3
        out[f"dn_{split}"] = dn
        out[f"y_{split}"] = y
    np.savez(FEATS_CACHE, **out)
    print(f"[xmodal] features salvas em {FEATS_CACHE.name}")
    return out

class CrossModalFusion(torch.nn.Module):
    def __init__(self, dims, d=256, heads=4, layers=2, n_cls=K, p=0.3):
        super().__init__()
        self.proj = torch.nn.ModuleList(
            [torch.nn.Linear(dim, d) for dim in dims])
        self.mod_emb = torch.nn.Parameter(torch.randn(len(dims), d) * 0.02)
        self.fuse = torch.nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=p,
            batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(enc, layers)
        self.norm = torch.nn.LayerNorm(d)
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(p), torch.nn.Linear(d, n_cls))

    def forward(self, streams):
        toks = [self.proj[i](s) + self.mod_emb[i] for i, s in enumerate(streams)]
        x = torch.stack(toks, dim=1)
        fuse = self.fuse.expand(x.size(0), -1, -1)
        x = torch.cat([fuse, x], dim=1)
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))

def _t(*arrs):
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrs]

def train_fusion(F, epochs=60, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    dims = [F["dit_tr"].shape[1], F["llv3_tr"].shape[1], F["dn_tr"].shape[1]]
    model = CrossModalFusion(dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    Xtr = _t(F["dit_tr"], F["llv3_tr"], F["dn_tr"])
    Xva = _t(F["dit_va"], F["llv3_va"], F["dn_va"])
    Xte = _t(F["dit_te"], F["llv3_te"], F["dn_te"])
    ytr = torch.tensor(F["y_tr"], dtype=torch.long, device=device)
    yva = F["y_va"]; yte = F["y_te"]

    n = len(ytr); bs = 64
    best_va, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model([s[idx] for s in Xtr])
            loss = crit(logits, ytr[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(Xva).argmax(1).cpu().numpy()
        va_acc = float((va_pred == yva).mean())
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte).cpu()
    return te_logits.numpy(), best_va

def main():
    F = build_feats()
    yte = F["y_te"]

    refs = {}
    for jf, keys in [("ensemble.json", ("individual", "best_ensemble_acc")),
                     ("moe.json", ("methods_acc", "oracle_any_correct"))]:
        p = C.METRICS_DIR / jf
        if p.exists():
            refs[jf] = json.loads(p.read_text())

    print("[xmodal] treinando rede de fusão cross-modal...")
    t0 = time.time()
    te_logits, best_va = train_fusion(F)
    pred = te_logits.argmax(1)
    m = A.compute_metrics(yte, pred, LABELS)
    print(f"[xmodal] treino em {time.time()-t0:.0f}s | val {best_va:.4f} | "
          f"test acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}")

    A.plot_confusion(m["confusion_matrix"], LABELS, "fusão cross-modal",
                     C.PLOTS_DIR / "crossmodal_confusion.png")

    comp = {"crossmodal_fusion": m["accuracy"]}
    if "ensemble.json" in refs:
        e = refs["ensemble.json"]
        comp.update({k: v["accuracy"] for k, v in e["individual"].items()})
        comp["late_fusion_best"] = e["best_ensemble_acc"]
        oracle = e["oracle_any_correct"]
    else:
        oracle = None
    if "moe.json" in refs:
        comp["moe_best"] = refs["moe.json"]["best_method_acc"]
        oracle = refs["moe.json"]["oracle_any_correct"]

    out = {
        "crossmodal_fusion": {"accuracy": m["accuracy"],
                              "macro_f1": m["macro_f1"],
                              "val_acc": best_va},
        "streams": ["dit_vision", "layoutlmv3", "donut_encoder"],
        "feature_dims": {"dit": int(F["dit_tr"].shape[1]),
                         "llv3": int(F["llv3_tr"].shape[1]),
                         "donut": int(F["dn_tr"].shape[1])},
        "comparison_acc": comp,
        "oracle_any_correct": oracle,
        "top_confusions": A.top_confusions(m["confusion_matrix"], LABELS, 6),
    }
    (C.METRICS_DIR / "crossmodal.json").write_text(json.dumps(out, indent=2))
    _plot_comparison(comp, oracle, C.PLOTS_DIR / "crossmodal_comparison.png")

    print("\n=== FUSÃO CROSS-MODAL (capstone E2E) ===")
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        star = "  <--" if k == "crossmodal_fusion" else ""
        print(f"  {k:22} {v:.4f}{star}")
    if oracle:
        print(f"  {'oracle':22} {oracle:.4f}")

def _plot_comparison(comp, oracle, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(comp.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    colors = ["#ff7f0e" if k == "crossmodal_fusion" else "#4c72b0" for k in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, vals, color=colors)
    if oracle:
        ax.axvline(oracle, color="#2ca02c", ls="--",
                   label=f"teto-oráculo ({oracle:.3f})")
        ax.legend(loc="lower right")
    ax.set_xlim(min(vals) - 0.02, (oracle or max(vals)) + 0.01)
    ax.set_xlabel("Acurácia (teste)")
    ax.set_title("Fusão cross-modal vs. modelos isolados e fusões anteriores")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

if __name__ == "__main__":
    main()
