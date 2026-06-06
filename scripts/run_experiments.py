import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data, models as M, training as T, analysis as A

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()

EPOCHS_FT = C.EPOCHS
PLAN = [
    ("baseline_cnn",    "baseline_cnn", max(EPOCHS_FT, 20), 1e-3, False, True),
    ("resnet18",        "resnet18",     EPOCHS_FT,          1e-4, False, True),
    ("deit_tiny",       "deit_tiny",    EPOCHS_FT,          1e-4, False, True),
    ("mobilevit_s",     "mobilevit_s",  EPOCHS_FT,          1e-4, False, True),
    ("dit_base",        "dit_base",     EPOCHS_FT,          5e-5, False, True),

    ("deit_tiny_lp",    "deit_tiny",    EPOCHS_FT,          1e-3, True,  True),
    ("dit_base_lp",     "dit_base",     EPOCHS_FT,          1e-3, True,  True),

    ("dit_rvlcdip_ref", "dit_rvlcdip",  0,                  0.0,  False, False),
]

def loaders_for(spec):
    tf_train = data.build_transforms(spec, train=True, aug="document")
    tf_eval = data.build_transforms(spec, train=False)
    tr = data.make_loader(C.MANIFEST_DIR / "train.csv", tf_train, shuffle=True)
    va = data.make_loader(C.MANIFEST_DIR / "val.csv", tf_eval)
    te = data.make_loader(C.MANIFEST_DIR / "test.csv", tf_eval)
    return tr, va, te

def run_one(run_key, model_key, epochs, lr, freeze, do_train):
    spec = C.MODELS[model_key]
    print(f"\n{'='*70}\n[run] {run_key}  ({spec.family}/{spec.pretrain})\n{'='*70}")
    gc = spec.backend == "hf"
    model = M.build_model(spec, grad_checkpointing=gc, freeze_backbone=freeze)
    model.to(device)
    tr, va, te = loaders_for(spec)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    hist = None
    if do_train:
        model, hist = T.train(model, tr, va, device, epochs=epochs, lr=lr,
                              patience=5)
    else:
        model.to(device)
    train_time = time.time() - t0

    yp, yt = T.predict(model, te, device)
    metrics = A.compute_metrics(yt, yp, LABELS)
    eff = A.efficiency(model, device)
    mem = T.gpu_mem_summary(device)

    if hist is not None:
        A.plot_history(hist, run_key, C.PLOTS_DIR / f"{run_key}_history.png")
    A.plot_confusion(metrics["confusion_matrix"], LABELS, run_key,
                     C.PLOTS_DIR / f"{run_key}_confusion.png")
    A.plot_per_class_acc(metrics["per_class_acc"], LABELS, run_key,
                         C.PLOTS_DIR / f"{run_key}_perclass.png")

    top, bottom = A.top_bottom_classes(metrics["per_class_acc"], LABELS)
    confus = A.top_confusions(metrics["confusion_matrix"], LABELS, k=8)

    rec = {
        "run": run_key, "model": model_key, "family": spec.family,
        "pretrain": spec.pretrain, "trained": do_train,
        "epochs": epochs, "lr": lr, "frozen_backbone": freeze,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "params_M": eff["params_M"], "gflops": eff["gflops"],
        "img_per_s": eff["img_per_s"],
        "train_time_s": train_time,
        "peak_vram_mb": mem.get("peak_alloc_mb"),
        "top5_classes": top, "bottom5_classes": bottom,
        "top_confusions": confus,
        "per_class_acc": metrics["per_class_acc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    print(f"[run] {run_key}: acc={rec['accuracy']:.4f} macroF1={rec['macro_f1']:.4f} "
          f"params={rec['params_M']:.1f}M vram={rec['peak_vram_mb']}")

    (C.METRICS_DIR / f"{run_key}.json").write_text(json.dumps(rec, indent=2))
    del model
    torch.cuda.empty_cache()
    return rec

def main():
    only = sys.argv[1:] or None
    records = []
    for run_key, model_key, epochs, lr, freeze, do_train in PLAN:
        if only and run_key not in only:
            continue
        try:
            records.append(run_one(run_key, model_key, epochs, lr, freeze, do_train))
        except Exception as e:
            import traceback
            print(f"[run] FALHA em {run_key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    records.sort(key=lambda r: r["accuracy"])
    summary = [{k: r[k] for k in ("run", "family", "pretrain", "accuracy",
               "macro_f1", "params_M", "gflops", "img_per_s", "train_time_s",
               "peak_vram_mb")} for r in records]
    (C.METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*70}\nRESUMO (ordenado por acurácia)\n{'='*70}")
    hdr = f"{'run':18s} {'pretrain':14s} {'acc':>7s} {'F1':>7s} {'params':>8s} {'img/s':>7s}"
    print(hdr); print("-" * len(hdr))
    for r in summary:
        print(f"{r['run']:18s} {r['pretrain']:14s} {r['accuracy']:7.4f} "
              f"{r['macro_f1']:7.4f} {r['params_M']:7.1f}M {r['img_per_s']:7.0f}")
    print(f"\n[done] métricas em {C.METRICS_DIR}, plots em {C.PLOTS_DIR}")

if __name__ == "__main__":
    main()
