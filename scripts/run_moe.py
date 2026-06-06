import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, analysis as A

LABELS = data.get_label_names()
K = C.NUM_CLASSES
PROBS_CACHE = C.METRICS_DIR / "ensemble_probs.npz"
EXPERTS = ["dit", "ll", "dn"]
EPS = 1e-6

def entropy(p):
    return -np.sum(p * np.log(p + EPS), axis=1)

def top2_margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]

def conf_features(plist):

    feats = []
    for p in plist:
        feats += [p.max(1), entropy(p), top2_margin(p)]
    preds = [p.argmax(1) for p in plist]
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            feats.append((preds[i] == preds[j]).astype(np.float32))
    return np.stack(feats, axis=1)

def temperature_scale(p_val, y_val, p_test):

    logit_va = np.log(p_val + EPS)
    logit_te = np.log(p_test + EPS)
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.5, 5.0, 46):
        q = _softmax(logit_va / T)
        nll = -np.mean(np.log(q[np.arange(len(y_val)), y_val] + EPS))
        if nll < best_nll:
            best_nll, best_T = nll, T
    return _softmax(logit_te / best_T), best_T

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def m_acc(y, pred):
    return float((pred == y).mean())

def soft_moe_gate(feats_va, probs_va, y_va, feats_te, probs_te, seed=42):

    torch.manual_seed(seed)
    M = len(probs_va)
    Xva = torch.tensor(_standardize(feats_va), dtype=torch.float32)
    Xte = torch.tensor(_standardize(feats_te, ref=feats_va), dtype=torch.float32)
    Pva = torch.tensor(np.stack(probs_va, 1), dtype=torch.float32)
    Pte = torch.tensor(np.stack(probs_te, 1), dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.long)

    gate = torch.nn.Sequential(
        torch.nn.Linear(Xva.shape[1], 16), torch.nn.ReLU(),
        torch.nn.Linear(16, M))
    opt = torch.optim.Adam(gate.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(400):
        opt.zero_grad()
        w = torch.softmax(gate(Xva), dim=1)
        fused = (w.unsqueeze(-1) * Pva).sum(1)
        loss = torch.nn.functional.nll_loss(torch.log(fused + EPS), yva)
        loss.backward(); opt.step()
    with torch.no_grad():
        wte = torch.softmax(gate(Xte), dim=1)
        fused_te = (wte.unsqueeze(-1) * Pte).sum(1).numpy()
    return fused_te, wte.numpy()

def _standardize(X, ref=None):
    ref = X if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0) + EPS
    return (X - mu) / sd

def main():
    if not PROBS_CACHE.exists():
        sys.exit(f"[moe] cache ausente: {PROBS_CACHE} (rode run_ensemble antes)")
    d = np.load(PROBS_CACHE)
    y_va, y_te = d["y_va"], d["y_te"]
    va = [d["dit_va"], d["ll_va"], d["dn_va"]]
    te = [d["dit_te"], d["ll_te"], d["dn_te"]]

    va[2] = va[2] * 0.9 + 0.1 / K
    te[2] = te[2] * 0.9 + 0.1 / K

    indiv = {e: m_acc(y_te, te[i].argmax(1)) for i, e in enumerate(EXPERTS)}
    best_ind = max(indiv, key=indiv.get)
    oracle = float((np.stack([p.argmax(1) for p in te]) == y_te[None]).any(0).mean())

    results = {}

    results["soft_avg_3"] = m_acc(y_te, sum(te).argmax(1))

    cal_va, cal_te, temps = [], [], {}
    for i, e in enumerate(EXPERTS):
        if e == "dn":
            cal_va.append(va[i]); cal_te.append(te[i]); temps[e] = None
            continue
        ct, T = temperature_scale(va[i], y_va, te[i])
        cv, _ = temperature_scale(va[i], y_va, va[i])
        cal_va.append(cv); cal_te.append(ct); temps[e] = round(float(T), 2)
    results["calibrada_avg"] = m_acc(y_te, sum(cal_te).argmax(1))

    wpc = np.zeros((len(EXPERTS), K))
    for i in range(len(EXPERTS)):
        pred = va[i].argmax(1)
        for c in range(K):
            mask = y_va == c
            wpc[i, c] = (pred[mask] == c).mean() if mask.any() else 0.0
    fused_pcw = sum(wpc[i][None, :] * te[i] for i in range(len(EXPERTS)))
    results["per_class_weighted"] = m_acc(y_te, fused_pcw.argmax(1))

    fva, fte = conf_features(va), conf_features(te)
    Xva_full = np.concatenate(va + [fva], axis=1)
    Xte_full = np.concatenate(te + [fte], axis=1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xva_full, y_va)
    results["stacking_rich_lr"] = m_acc(y_te, lr.predict(Xte_full))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    random_state=42).fit(Xva_full, y_va)
    results["stacking_gb"] = m_acc(y_te, gb.predict(Xte_full))

    best_expert_idx = EXPERTS.index(best_ind)
    tgt = np.full(len(y_va), best_expert_idx)
    for n in range(len(y_va)):
        correct = [i for i in range(len(EXPERTS)) if va[i][n].argmax() == y_va[n]]
        if correct:
            tgt[n] = max(correct, key=lambda i: va[i][n].max())
    router = LogisticRegression(max_iter=3000, C=1.0).fit(fva, tgt)
    pick = router.predict(fte)
    routed = np.array([te[pick[n]][n].argmax() for n in range(len(y_te))])
    results["hard_router"] = m_acc(y_te, routed)
    router_pick_rate = {e: float((pick == i).mean()) for i, e in enumerate(EXPERTS)}

    fused_gate, w_gate = soft_moe_gate(fva, va, y_va, fte, te)
    results["soft_moe_gate"] = m_acc(y_te, fused_gate.argmax(1))
    gate_mean_w = {e: float(w_gate[:, i].mean()) for i, e in enumerate(EXPERTS)}

    best_method = max(results, key=results.get)
    best_acc = results[best_method]
    gap_closed = (best_acc - indiv[best_ind]) / max(EPS, oracle - indiv[best_ind])

    pred_best = {
        "soft_avg_3": sum(te).argmax(1),
        "calibrada_avg": sum(cal_te).argmax(1),
        "per_class_weighted": fused_pcw.argmax(1),
        "stacking_rich_lr": lr.predict(Xte_full),
        "stacking_gb": gb.predict(Xte_full),
        "hard_router": routed,
        "soft_moe_gate": fused_gate.argmax(1),
    }[best_method]
    mbest = A.compute_metrics(y_te, pred_best, LABELS)
    A.plot_confusion(mbest["confusion_matrix"], LABELS, f"MoE ({best_method})",
                     C.PLOTS_DIR / "moe_confusion.png")
    _plot_methods(indiv[best_ind], oracle, results,
                  C.PLOTS_DIR / "moe_methods.png")

    out = {
        "individual_acc": indiv,
        "best_individual": best_ind,
        "oracle_any_correct": oracle,
        "methods_acc": results,
        "best_method": best_method,
        "best_method_acc": best_acc,
        "best_method_macro_f1": mbest["macro_f1"],
        "gain_over_best_individual_pp": round(100 * (best_acc - indiv[best_ind]), 2),
        "oracle_gap_closed_frac": round(float(gap_closed), 3),
        "temperatures": temps,
        "router_pick_rate": router_pick_rate,
        "gate_mean_weight": gate_mean_w,
    }
    (C.METRICS_DIR / "moe.json").write_text(json.dumps(out, indent=2))

    print("\n=== MIXTURE-OF-EXPERTS (capstone avançado) ===")
    print(f"individuais: " + "  ".join(f"{e}={indiv[e]:.4f}" for e in EXPERTS))
    print(f"teto-oráculo: {oracle:.4f}   (melhor individual: {best_ind} {indiv[best_ind]:.4f})")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22} {v:.4f}")
    print(f"-> melhor: {best_method} {best_acc:.4f} "
          f"(ganho {out['gain_over_best_individual_pp']:+.2f} pp; "
          f"fechou {100*gap_closed:.0f}% do gap-oráculo)")
    print(f"temperaturas: {temps}")
    print(f"router escolhe: {router_pick_rate}")
    print(f"peso médio do gate: {gate_mean_w}")

def _plot_methods(best_ind_acc, oracle, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(results.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, vals, color="#4c72b0")
    ax.axvline(best_ind_acc, color="#d62728", ls="--",
               label=f"melhor individual ({best_ind_acc:.3f})")
    ax.axvline(oracle, color="#2ca02c", ls="--",
               label=f"teto-oráculo ({oracle:.3f})")
    ax.set_xlim(min(vals) - 0.02, oracle + 0.01)
    ax.set_xlabel("Acurácia (teste)")
    ax.set_title("Fusões / roteamento vs. melhor individual e teto-oráculo")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

if __name__ == "__main__":
    main()
