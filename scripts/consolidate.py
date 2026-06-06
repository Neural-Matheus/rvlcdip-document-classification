import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

M = C.METRICS_DIR

def load(name):
    p = M / name
    return json.load(open(p)) if p.exists() else None

def main():
    grid = load("summary.json") or []
    eff = load("efficiency_curve.json")
    mm = load("multimodal.json")
    vlm = load("vlm.json")

    rows = []
    for r in grid:
        rows.append((r["run"], "arquitetura×pré-treino", "imagem",
                     r["accuracy"], r.get("macro_f1")))
    if mm:
        rows.append(("LayoutLMv3", "multimodal", "imagem+OCR+layout",
                     mm["layoutlmv3"]["accuracy"], mm["layoutlmv3"]["macro_f1"]))
        rows.append(("DiT (mesmo teste MM)", "multimodal", "imagem",
                     mm["dit_visiononly_same_test"]["accuracy"],
                     mm["dit_visiononly_same_test"]["macro_f1"]))
    if vlm:
        for k, v in vlm.items():
            rows.append((k, "VLM", "imagem (gerativo)", v["accuracy"], v["macro_f1"]))

    rows.sort(key=lambda r: r[3])

    lines = ["# Tabela-mestra — RVL-CDIP (todos os eixos)\n",
             "| modelo | eixo | modalidade | acc | macro-F1 |",
             "|---|---|---|---:|---:|"]
    for name, axis, mod, acc, f1 in rows:
        f1s = f"{f1:.4f}" if f1 is not None else "—"
        lines.append(f"| {name} | {axis} | {mod} | {acc:.4f} | {f1s} |")
    if eff:
        lines.append("\n## Curva de eficiência de dados (acurácia por N/classe)\n")
        ns = sorted({int(n) for d in eff.values() for n in d})
        lines.append("| modelo | " + " | ".join(f"N={n}" for n in ns) + " |")
        lines.append("|---|" + "---|" * len(ns))
        for mk, d in eff.items():
            cells = [f"{d[str(n)]:.3f}" if str(n) in d else "—" for n in ns]
            lines.append(f"| {mk} | " + " | ".join(cells) + " |")
    (M / "master_table.md").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(rows))))
    cmap = {"arquitetura×pré-treino": "#1f77b4", "multimodal": "#d62728", "VLM": "#9467bd"}
    ax.barh([r[0] for r in rows], [r[3] for r in rows],
            color=[cmap.get(r[1], "#888") for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[3] + 0.005, i, f"{r[3]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("acurácia (teste)")
    ax.set_title("RVL-CDIP — panorama de todos os eixos")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    ax.legend(handles, cmap.keys(), loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "master_overview.png", dpi=130)
    print(f"\n[consolidate] salvo master_table.md e master_overview.png")

if __name__ == "__main__":
    main()
