from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import config as C

def _try_stream(source: str, split: str):

    from datasets import load_dataset

    kwargs = dict(split=split, streaming=True)
    if source == C.HF_PRIMARY:
        kwargs["trust_remote_code"] = True
    ds = load_dataset(source, **kwargs)

    feat = ds.features.get("label") if ds.features else None
    names = list(feat.names) if feat is not None and hasattr(feat, "names") else None
    return ds, names

def load_rvlcdip_stream(split: str):

    sources = (C.HF_PRIMARY, *C.HF_FALLBACKS)
    last_err = None
    for src in sources:
        try:
            ds, names = _try_stream(src, split)
            print(f"[data] fonte ativa: {src} (split={split})")
            return ds, names, src
        except Exception as e:
            print(f"[data] falha em {src}: {type(e).__name__}: {e}")
            last_err = e
    raise RuntimeError(
        f"Nenhuma fonte de RVL-CDIP funcionou. Último erro: {last_err}"
    )

def _to_pil(example) -> Optional[Image.Image]:
    img = example.get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and "bytes" in img and img["bytes"]:
        return Image.open(io.BytesIO(img["bytes"]))
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    return None

def build_balanced_subset(
    split: str,
    n_per_class: int,
    manifest_path: Path,
    label_names_out: Optional[list] = None,
    overwrite: bool = False,
) -> Path:

    C.set_seed()
    manifest_path = Path(manifest_path)
    expected = n_per_class * C.NUM_CLASSES
    if manifest_path.exists() and not overwrite:
        with open(manifest_path) as f:
            rows = sum(1 for _ in f) - 1
        if rows >= expected:
            print(f"[data] manifest {manifest_path.name} já completo ({rows} linhas).")
            return manifest_path

    ds, names, source = load_rvlcdip_stream(split)
    counts = {c: 0 for c in range(C.NUM_CLASSES)}
    out_dir = C.SUBSET_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    target_total = expected
    for ex in ds:
        label = int(ex["label"])
        if counts.get(label, n_per_class) >= n_per_class:
            continue
        pil = _to_pil(ex)
        if pil is None:
            continue
        pil = pil.convert("L")
        cls_dir = out_dir / f"{label:02d}"
        cls_dir.mkdir(exist_ok=True)
        fname = cls_dir / f"{counts[label]:05d}.png"
        pil.save(fname)
        rows.append((str(fname.relative_to(C.ROOT)), label))
        counts[label] += 1
        done = sum(counts.values())
        if done % 500 == 0:
            print(f"[data] {split}: {done}/{target_total}")
        if done >= target_total:
            break

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        w.writerows(rows)
    print(f"[data] {split}: salvo manifest com {len(rows)} linhas em {manifest_path}")

    if label_names_out is not None and names:
        (C.DATA_DIR / "label_names.txt").write_text("\n".join(names))
    return manifest_path

def get_label_names() -> list:

    p = C.DATA_DIR / "label_names.txt"
    if p.exists():
        return p.read_text().splitlines()
    _, names, _ = load_rvlcdip_stream("test")
    if names:
        p.write_text("\n".join(names))
        return names
    raise RuntimeError("Não foi possível obter os nomes das classes.")

def build_all_subsets(overwrite: bool = False):
    names_holder: list = []
    build_balanced_subset("train", C.N_PER_CLASS_TRAIN, C.MANIFEST_DIR / "train.csv",
                           names_holder, overwrite)
    build_balanced_subset("validation", C.N_PER_CLASS_VAL, C.MANIFEST_DIR / "val.csv",
                          overwrite=overwrite)
    build_balanced_subset("test", C.N_PER_CLASS_TEST, C.MANIFEST_DIR / "test.csv",
                          overwrite=overwrite)
    return get_label_names()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BEIT_MEAN = (0.5, 0.5, 0.5)
BEIT_STD = (0.5, 0.5, 0.5)

class PadToSquare:

    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        canvas = Image.new(img.mode, (s, s), self.fill)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas

def _norm_for(spec) -> tuple:
    if spec is not None and spec.backend == "hf":
        return BEIT_MEAN, BEIT_STD
    return IMAGENET_MEAN, IMAGENET_STD

def build_transforms(
    spec=None,
    img_size: int = C.IMG_SIZE,
    train: bool = False,
    aug: str = "none",
    square: str = "resize",
):

    from torchvision import transforms as T

    mean, std = _norm_for(spec)
    pre = [PadToSquare()] if square == "pad" else []
    pre.append(T.Resize((img_size, img_size)))
    pre.append(T.Grayscale(num_output_channels=3))

    aug_ops = []
    if train and aug == "natural":

        aug_ops = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(20),
            T.ColorJitter(0.4, 0.4, 0.4),
        ]
    elif train and aug == "document":

        aug_ops = [
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.1),
        ]

    post = [T.ToTensor(), T.Normalize(mean, std)]
    if train and aug == "document":
        post.append(T.RandomErasing(p=0.25, scale=(0.02, 0.08)))

    return T.Compose(pre + aug_ops + post)

def make_dataset(manifest_csv: Path, transform):
    import torch
    from torch.utils.data import Dataset

    class ManifestDataset(Dataset):
        def __init__(self, csv_path, tfm):
            self.items = []
            with open(csv_path) as f:
                r = csv.DictReader(f)
                for row in r:
                    self.items.append((row["filepath"], int(row["label"])))
            self.tfm = tfm

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            rel, label = self.items[i]
            path = C.ROOT / rel
            img = Image.open(path)
            return self.tfm(img), label

    return ManifestDataset(manifest_csv, transform)

def make_loader(manifest_csv: Path, transform, batch_size=C.BATCH_SIZE,
                shuffle=False, num_workers=C.NUM_WORKERS):
    from torch.utils.data import DataLoader

    ds = make_dataset(manifest_csv, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)
