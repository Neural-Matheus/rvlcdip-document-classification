import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

summary = json.load(open(C.METRICS_DIR / "summary.json"))

COLOR = {"none": "#888888", "imagenet": "#1f77b4",
         "domain": "#d62728", "domain+rvlcdip": "#2ca02c"}

s = sorted(summary, key=lambda r: r["accuracy"])
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([r["run"] for r in s], [r["accuracy"] for r in s],
        color=[COLOR.get(r["pretrain"], "#000") for r in s])
for i, r in enumerate(s):
    ax.text(r["accuracy"] + 0.005, i, f"{r['accuracy']:.3f}", va="center", fontsize=8)
ax.set_xlabel("acurácia (teste)"); ax.set_xlim(0, 1)
ax.set_title("RVL-CDIP — acurácia por modelo (cor = tipo de pré-treino)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles, COLOR.keys(), loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "headline_accuracy.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.5))
for r in summary:
    ax.scatter(r["params_M"], r["accuracy"], s=90,
               color=COLOR.get(r["pretrain"], "#000"))
    ax.annotate(r["run"], (r["params_M"], r["accuracy"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("parâmetros (M, log)")
ax.set_ylabel("acurácia (teste)")
ax.set_title("Acurácia × custo — pré-treino de domínio domina")
fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "headline_acc_vs_params.png", dpi=130)
plt.close(fig)

print("figuras salvas: headline_accuracy.png, headline_acc_vs_params.png")
