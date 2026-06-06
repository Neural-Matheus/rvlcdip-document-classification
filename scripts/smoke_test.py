import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import models as M
from src import training as T
from src import analysis as A

C.set_seed()
device = C.get_device()
print(f"[smoke] device={device}")

LABELS = [f"class_{i}" for i in range(C.NUM_CLASSES)]

def synth_loader(n=64, bs=16):
    x = torch.randn(n, 3, 64, 64)
    y = torch.randint(0, C.NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs)

def test_models_forward():
    print("\n[smoke] forward de cada arquitetura (input sintético 3x224x224)")
    x = torch.randn(2, 3, 224, 224, device=device)
    for key, spec in C.MODELS.items():
        if spec.key == "dit_rvlcdip":
            continue
        try:
            gc = spec.backend == "hf"
            model = M.build_model(spec, grad_checkpointing=gc).to(device).eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, C.NUM_CLASSES), out.shape
            print(f"  OK {key:14s} out={tuple(out.shape)} "
                  f"params={M.count_params(model)/1e6:.1f}M")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FALHA {key}: {type(e).__name__}: {e}")
            raise

def test_train_eval():
    print("\n[smoke] mini-treino do BaselineCNN (2 épocas, dados sintéticos)")
    model = M.build_model(C.MODELS["baseline_cnn"])
    tr, va = synth_loader(), synth_loader()
    model, hist = T.train(model, tr, va, device, epochs=2, log_every=1)
    yp, yt = T.predict(model, va, device)
    metrics = A.compute_metrics(yt, yp, LABELS)
    print(f"  métricas calculadas: acc={metrics['accuracy']:.3f} "
          f"macroF1={metrics['macro_f1']:.3f}")
    A.plot_history(hist, "smoke", C.PLOTS_DIR / "smoke_history.png")
    A.plot_confusion(metrics["confusion_matrix"], LABELS, "smoke",
                     C.PLOTS_DIR / "smoke_confusion.png")
    print("  plots salvos.")

def test_gradcam():
    print("\n[smoke] Grad-CAM na BaselineCNN")
    model = M.build_model(C.MODELS["baseline_cnn"]).to(device)
    last_conv = model.features[-1][3]
    img = torch.randn(3, 224, 224)
    cam, cls = A.gradcam(model, img, last_conv, device)
    assert cam.shape == (224, 224), cam.shape
    print(f"  CAM shape {cam.shape}, classe {cls}")

def test_attention_rollout():
    print("\n[smoke] attention rollout no DiT-base (Obs 2)")
    spec = C.MODELS["dit_base"]
    model = M.build_model(spec, grad_checkpointing=True).to(device)
    img = torch.randn(3, 224, 224)
    mask = A.attention_rollout(model, img, device)
    print(f"  rollout mask shape {mask.shape}, range [{mask.min():.2f},{mask.max():.2f}]")
    A.overlay_heatmap(img, mask, C.PLOTS_DIR / "smoke_rollout.png", "smoke rollout")
    print("  overlay salvo.")

if __name__ == "__main__":
    test_models_forward()
    test_train_eval()
    test_gradcam()
    test_attention_rollout()
    print("\n[smoke] TUDO PASSOU ✅")
