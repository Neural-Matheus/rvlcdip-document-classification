import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, models as M, training as T, analysis as A

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()

N_POINTS = [50, 100, 250, 500, 1000, 2000]
MODELS_CURVE = [
    ("baseline_cnn", 1e-3, 20),
    ("deit_tiny",    1e-4, 10),
    ("dit_base",     5e-5, 10),
]
OUT = C.METRICS_DIR / "efficiency_curve.json"

def subsample_manifest(src_csv, n_per_class, dst_csv):

    by_cls = defaultdict(list)
    for r in csv.DictReader(open(src_csv)):
        by_cls[int(r["label"])].append(r["filepath"])
    rows = []
    for c in range(C.NUM_CLASSES):
        for fp in by_cls[c][:n_per_class]:
            rows.append((fp, c))
    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filepath", "label"]); w.writerows(rows)
    return len(rows)

def run_point(model_key, lr, epochs, n_per_class):
    spec = C.MODELS[model_key]
    tmp = C.MANIFEST_DIR / f"_eff_train_{n_per_class}.csv"
    subsample_manifest(C.MANIFEST_DIR / "train.csv", n_per_class, tmp)
    tf_tr = data.build_transforms(spec, train=True, aug="document")
    tf_ev = data.build_transforms(spec, train=False)
    tr = data.make_loader(tmp, tf_tr, shuffle=True)
    va = data.make_loader(C.MANIFEST_DIR / "val.csv", tf_ev)
    te = data.make_loader(C.MANIFEST_DIR / "test.csv", tf_ev)
    gc = spec.backend == "hf"
    model = M.build_model(spec, grad_checkpointing=gc)
    t0 = time.time()
    model, _ = T.train(model, tr, va, device, epochs=epochs, lr=lr, patience=4,
                       log_every=0)
    yp, yt = T.predict(model, te, device)
    acc = A.compute_metrics(yt, yp, LABELS)["accuracy"]
    del model; torch.cuda.empty_cache()
    return acc, time.time() - t0

def main():
    results = defaultdict(dict)
    if OUT.exists():
        results.update(json.load(open(OUT)))
    for model_key, lr, epochs in MODELS_CURVE:
        for n in N_POINTS:
            if str(n) in results.get(model_key, {}):
                continue
            acc, dt = run_point(model_key, lr, epochs, n)
            results.setdefault(model_key, {})[str(n)] = acc
            print(f"[eff] {model_key:14s} N={n:5d}/classe -> acc {acc:.4f} ({dt:.0f}s)")
            OUT.write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_key, _, _ in MODELS_CURVE:
        if model_key not in results:
            continue
        xs = sorted(int(k) for k in results[model_key])
        ys = [results[model_key][str(x)] for x in xs]
        ax.plot(xs, ys, marker="o", label=f"{model_key} ({C.MODELS[model_key].pretrain})")
    ax.set_xscale("log"); ax.set_xlabel("exemplos de treino por classe (log)")
    ax.set_ylabel("acurácia (teste)")
    ax.set_title("Curva de eficiência de dados — domínio precisa de menos rótulos")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "efficiency_curve.png", dpi=130)
    print(f"[eff] figura salva: {C.PLOTS_DIR/'efficiency_curve.png'}")

if __name__ == "__main__":
    main()
