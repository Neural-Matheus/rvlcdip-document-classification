import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, models as M, training as T, analysis as A
from src import multimodal as MM

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()

AMP_DTYPE = (torch.bfloat16 if device.type == "cuda"
             and torch.cuda.is_bf16_supported() else torch.float16)

N_TRAIN = int(os.environ.get("MM_TRAIN", 300))
N_VAL = int(os.environ.get("MM_VAL", 100))
N_TEST = int(os.environ.get("MM_TEST", 200))
EPOCHS = int(os.environ.get("MM_EPOCHS", 8))

def subsample(src_csv, n, dst_csv):
    by = defaultdict(list)
    for r in csv.DictReader(open(src_csv)):
        by[int(r["label"])].append(r["filepath"])
    rows = [(fp, c) for c in range(C.NUM_CLASSES) for fp in by[c][:n]]
    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filepath", "label"]); w.writerows(rows)
    return dst_csv

def dict_train(model, tr, va, epochs, lr):
    from transformers import get_linear_schedule_with_warmup
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)

    total_steps = max(1, epochs * len(tr))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    best_acc, best_state = -1, None
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for batch in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
                out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        acc = dict_eval(model, va)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  época {ep+1:02d}/{epochs} | val_acc {acc:.4f} | {time.time()-t0:.0f}s"
              + (" *" if acc == best_acc else ""))
    if best_state:
        model.load_state_dict(best_state)
    print(f"  melhor val_acc {best_acc:.4f}")
    return model

@torch.no_grad()
def dict_eval(model, loader):
    model.eval(); correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
            out = model(**batch)
        pred = out.logits.argmax(1)
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total

@torch.no_grad()
def dict_predict(model, loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
            out = model(**batch)
        P.append(out.logits.argmax(1).cpu()); Y.append(batch["labels"].cpu())
    import numpy as np
    return torch.cat(P).numpy(), torch.cat(Y).numpy()

def main():
    from torch.utils.data import DataLoader
    mm = C.MANIFEST_DIR
    tr_csv = subsample(mm / "train.csv", N_TRAIN, mm / "_mm_train.csv")
    va_csv = subsample(mm / "val.csv", N_VAL, mm / "_mm_val.csv")
    te_csv = subsample(mm / "test.csv", N_TEST, mm / "_mm_test.csv")

    print("[mm] rodando OCR (cacheado)...")
    t0 = time.time()
    for c in (tr_csv, va_csv, te_csv):
        MM.ensure_ocr(c)
    print(f"[mm] OCR pronto em {time.time()-t0:.0f}s")

    proc = MM.get_processor()
    bs = int(os.environ.get("MM_BS", 8))
    tr = DataLoader(MM.make_layout_dataset(tr_csv, proc), batch_size=bs,
                    shuffle=True, num_workers=4)
    va = DataLoader(MM.make_layout_dataset(va_csv, proc), batch_size=bs, num_workers=4)
    te = DataLoader(MM.make_layout_dataset(te_csv, proc), batch_size=bs, num_workers=4)

    print("[mm] treinando LayoutLMv3...")
    model = MM.build_layoutlmv3()
    model = dict_train(model, tr, va, EPOCHS, lr=float(os.environ.get("MM_LR", 2e-5)))
    yp, yt = dict_predict(model, te)
    m = A.compute_metrics(yt, yp, LABELS)
    A.plot_confusion(m["confusion_matrix"], LABELS, "layoutlmv3",
                     C.PLOTS_DIR / "layoutlmv3_confusion.png")
    del model; torch.cuda.empty_cache()

    print("[mm] avaliando DiT-rvlcdip (visão-pura) no mesmo teste...")
    spec = C.MODELS["dit_rvlcdip"]
    dit = M.build_model(spec).to(device)
    tf = data.build_transforms(spec, train=False)
    te_img = data.make_loader(te_csv, tf, batch_size=32)
    yp2, yt2 = T.predict(dit, te_img, device)
    m2 = A.compute_metrics(yt2, yp2, LABELS)

    out = {
        "layoutlmv3": {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                       "top_confusions": A.top_confusions(m["confusion_matrix"], LABELS, 6)},
        "dit_visiononly_same_test": {"accuracy": m2["accuracy"], "macro_f1": m2["macro_f1"]},
        "n_train_per_class": N_TRAIN, "n_test_per_class": N_TEST,
    }
    (C.METRICS_DIR / "multimodal.json").write_text(json.dumps(out, indent=2))
    print("\n=== MULTIMODAL ===")
    print(f"LayoutLMv3 (img+OCR+layout): acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}")
    print(f"DiT visão-pura (mesmo teste): acc {m2['accuracy']:.4f}  F1 {m2['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
