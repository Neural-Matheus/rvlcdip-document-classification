import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, models as M, training as T, analysis as A
from src import multimodal as MM
from scripts.run_layoutlmv3 import dict_train, AMP_DTYPE, subsample

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()
K = C.NUM_CLASSES

MM_TRAIN = int(os.environ.get("MM_TRAIN", 150))
MM_VAL = int(os.environ.get("MM_VAL", 40))
MM_TEST = int(os.environ.get("MM_TEST", 100))
EPOCHS = int(os.environ.get("MM_EPOCHS", 8))
PROBS_CACHE = C.METRICS_DIR / "ensemble_probs.npz"
DONUT_ID = "naver-clova-ix/donut-base-finetuned-rvlcdip"

def read_rows(csv_path):

    rows = list(csv.DictReader(open(csv_path)))
    fps = [r["filepath"] for r in rows]
    ys = np.array([int(r["label"]) for r in rows])
    return fps, ys

def match_label(text):
    t = text.lower()
    best, best_score = None, 0
    for i, name in enumerate(LABELS):
        score = sum(1 for w in name.split() if w in t)
        if name in t:
            score += 5
        if score > best_score:
            best, best_score = i, score
    return best if best is not None else 0

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

@torch.no_grad()
def dit_proba(csv_path):
    spec = C.MODELS["dit_rvlcdip"]
    model = M.build_model(spec).to(device).eval()
    tf = data.build_transforms(spec, train=False)
    loader = data.make_loader(csv_path, tf, batch_size=32, shuffle=False)
    P = []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        P.append(torch.softmax(out.float(), 1).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(P)

@torch.no_grad()
def layoutlmv3_proba(model, proc, csv_path):
    from torch.utils.data import DataLoader
    loader = DataLoader(MM.make_layout_dataset(csv_path, proc), batch_size=8,
                        shuffle=False, num_workers=4)
    P = []
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
            out = model(**batch)
        P.append(torch.softmax(out.logits.float(), 1).cpu().numpy())
    return np.concatenate(P)

@torch.no_grad()
def donut_onehot(fps):

    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    proc = DonutProcessor.from_pretrained(DONUT_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        DONUT_ID, torch_dtype=torch.float16).to(device).eval()
    dec_ids = proc.tokenizer("<s_rvlcdip>", add_special_tokens=False,
                             return_tensors="pt").input_ids
    oh = np.zeros((len(fps), K), dtype=np.float32)
    t0 = time.time()
    for i, fp in enumerate(fps):
        img = Image.open(C.ROOT / fp).convert("RGB")
        pv = proc(img, return_tensors="pt").pixel_values.to(device, torch.float16)
        out = model.generate(pv, decoder_input_ids=dec_ids.to(device),
                             max_length=16, do_sample=False,
                             pad_token_id=proc.tokenizer.pad_token_id,
                             eos_token_id=proc.tokenizer.eos_token_id)
        seq = proc.batch_decode(out)[0]
        oh[i, match_label(seq)] = 1.0
        if (i + 1) % 200 == 0:
            print(f"  donut {i+1}/{len(fps)} ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return oh

def build_all_probs():

    if PROBS_CACHE.exists() and os.environ.get("FORCE", "0") != "1":
        print(f"[ens] carregando probs do cache {PROBS_CACHE.name}")
        d = np.load(PROBS_CACHE)
        return {k: d[k] for k in d.files}

    mm = C.MANIFEST_DIR
    va_csv = subsample(mm / "val.csv", MM_VAL, mm / "_mm_val.csv")
    te_csv = subsample(mm / "test.csv", MM_TEST, mm / "_mm_test.csv")
    tr_csv = subsample(mm / "train.csv", MM_TRAIN, mm / "_mm_train.csv")

    print("[ens] OCR (cacheado) para multimodal...")
    for c in (tr_csv, va_csv, te_csv):
        MM.ensure_ocr(c)

    fps_va, y_va = read_rows(va_csv)
    fps_te, y_te = read_rows(te_csv)

    print("[ens] DiT (visão) -> probs val/test...")
    dit_va, dit_te = dit_proba(va_csv), dit_proba(te_csv)

    print("[ens] treinando LayoutLMv3 (bf16 + scheduler)...")
    from torch.utils.data import DataLoader
    proc = MM.get_processor()
    tr = DataLoader(MM.make_layout_dataset(tr_csv, proc), batch_size=8,
                    shuffle=True, num_workers=4)
    va_loader = DataLoader(MM.make_layout_dataset(va_csv, proc), batch_size=8,
                           num_workers=4)
    ll_model = MM.build_layoutlmv3()
    ll_model = dict_train(ll_model, tr, va_loader, EPOCHS,
                          lr=float(os.environ.get("MM_LR", 2e-5)))
    print("[ens] LayoutLMv3 -> probs val/test...")
    ll_va = layoutlmv3_proba(ll_model, proc, va_csv)
    ll_te = layoutlmv3_proba(ll_model, proc, te_csv)
    del ll_model; torch.cuda.empty_cache()

    print("[ens] Donut (gerativo) -> one-hot val/test...")
    dn_va = donut_onehot(fps_va)
    dn_te = donut_onehot(fps_te)

    probs = {"y_va": y_va, "y_te": y_te,
             "dit_va": dit_va, "dit_te": dit_te,
             "ll_va": ll_va, "ll_te": ll_te,
             "dn_va": dn_va, "dn_te": dn_te}
    np.savez(PROBS_CACHE, **probs)
    print(f"[ens] probs salvas em {PROBS_CACHE.name}")
    return probs

def acc(y, p):
    return float((p.argmax(1) == y).mean())

def metrics(y, pred):
    return A.compute_metrics(y, pred, LABELS)

def main():
    P = build_all_probs()
    y_va, y_te = P["y_va"], P["y_te"]
    dit_va, dit_te = P["dit_va"], P["dit_te"]
    ll_va, ll_te = P["ll_va"], P["ll_te"]
    dn_va, dn_te = P["dn_va"], P["dn_te"]

    indiv = {
        "dit_vision":   metrics(y_te, dit_te.argmax(1)),
        "layoutlmv3":   metrics(y_te, ll_te.argmax(1)),
        "donut":        metrics(y_te, dn_te.argmax(1)),
    }
    for k, m in indiv.items():
        print(f"[indiv] {k:12} acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}")

    fusions = {}

    avg2 = (dit_te + ll_te) / 2
    fusions["soft_avg_dit_ll"] = avg2

    ws = np.linspace(0, 1, 21)
    best_w = max(ws, key=lambda w: acc(y_va, w * dit_va + (1 - w) * ll_va))
    fusions["soft_weighted_dit_ll"] = best_w * dit_te + (1 - best_w) * ll_te

    fusions["soft_avg_3"] = (dit_te + ll_te + dn_te) / 3

    from sklearn.linear_model import LogisticRegression
    Xva = np.concatenate([dit_va, ll_va, dn_va], axis=1)
    Xte = np.concatenate([dit_te, ll_te, dn_te], axis=1)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(Xva, y_va)
    stack_proba = clf.predict_proba(Xte)

    stack_full = np.zeros((len(y_te), K), dtype=np.float32)
    stack_full[:, clf.classes_] = stack_proba
    fusions["stacking_lr"] = stack_full

    ens_metrics = {}
    for name, p in fusions.items():
        m = metrics(y_te, p.argmax(1))
        ens_metrics[name] = m
        print(f"[fusão] {name:22} acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}")

    best_ens_name = max(ens_metrics, key=lambda k: ens_metrics[k]["accuracy"])
    best_ind_name = max(indiv, key=lambda k: indiv[k]["accuracy"])
    best_ens = fusions[best_ens_name]
    best_ind_pca = np.array(indiv[best_ind_name]["per_class_acc"])
    best_ens_m = ens_metrics[best_ens_name]
    best_ens_pca = np.array(best_ens_m["per_class_acc"])

    preds = np.stack([dit_te.argmax(1), ll_te.argmax(1), dn_te.argmax(1)], 0)
    oracle = float((preds == y_te[None, :]).any(0).mean())

    names = ["dit", "ll", "donut"]
    agree = {f"{names[i]}_vs_{names[j]}": float((preds[i] == preds[j]).mean())
             for i in range(3) for j in range(i + 1, 3)}

    A.plot_confusion(best_ens_m["confusion_matrix"], LABELS,
                     f"ensemble ({best_ens_name})",
                     C.PLOTS_DIR / "ensemble_confusion.png")
    _plot_perclass_gain(best_ind_pca, best_ens_pca, best_ind_name,
                        best_ens_name, C.PLOTS_DIR / "ensemble_perclass_gain.png")

    out = {
        "test_set": {"n_per_class": MM_TEST, "n_total": int(len(y_te))},
        "individual": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                       for k, v in indiv.items()},
        "ensembles": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                      for k, v in ens_metrics.items()},
        "best_weight_dit": float(best_w),
        "best_individual": best_ind_name,
        "best_ensemble": best_ens_name,
        "best_ensemble_acc": best_ens_m["accuracy"],
        "best_ensemble_macro_f1": best_ens_m["macro_f1"],
        "gain_over_best_individual_pp":
            round(100 * (best_ens_m["accuracy"] - indiv[best_ind_name]["accuracy"]), 2),
        "oracle_any_correct": oracle,
        "pairwise_agreement": agree,
        "top_confusions": A.top_confusions(best_ens_m["confusion_matrix"], LABELS, 6),
    }
    (C.METRICS_DIR / "ensemble.json").write_text(json.dumps(out, indent=2))

    print("\n=== ENSEMBLE (capstone) ===")
    print(f"melhor individual: {best_ind_name} "
          f"acc {indiv[best_ind_name]['accuracy']:.4f}")
    print(f"melhor ensemble:   {best_ens_name} acc {best_ens_m['accuracy']:.4f} "
          f"(ganho {out['gain_over_best_individual_pp']:+.2f} pp)")
    print(f"peso ótimo DiT na fusão DiT+LL: {best_w:.2f}")
    print(f"teto-oráculo (algum acerta): {oracle:.4f}")
    print(f"concordância par-a-par: {agree}")

def _plot_perclass_gain(pca_ind, pca_ens, ind_name, ens_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gain = 100 * (pca_ens - pca_ind)
    order = np.argsort(gain)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if g >= 0 else "#d62728" for g in gain[order]]
    ax.barh([LABELS[i] for i in order], gain[order], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Ganho de acurácia por classe (pp)")
    ax.set_title(f"Ensemble '{ens_name}' vs. melhor individual '{ind_name}'")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

if __name__ == "__main__":
    main()
