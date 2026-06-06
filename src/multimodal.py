from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from . import config as C

OCR_DIR = C.DATA_DIR / "ocr"
LAYOUTLMV3_ID = "microsoft/layoutlmv3-base"

_READER = None

def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=True)
    return _READER

def _norm_box(pts, w, h):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = max(0, min(xs)), min(w, max(xs))
    y0, y1 = max(0, min(ys)), min(h, max(ys))
    return [int(1000 * x0 / w), int(1000 * y0 / h),
            int(1000 * x1 / w), int(1000 * y1 / h)]

def ocr_image(pil: Image.Image):

    reader = get_reader()
    import numpy as np
    arr = np.array(pil.convert("RGB"))
    h, w = arr.shape[:2]

    res = reader.readtext(arr, detail=1, paragraph=False, batch_size=32)
    words, boxes = [], []
    for pts, text, conf in res:
        text = text.strip()
        if not text:
            continue
        words.append(text)
        boxes.append(_norm_box(pts, w, h))
    if not words:
        words, boxes = ["[EMPTY]"], [[0, 0, 0, 0]]
    return words, boxes

def cache_path(rel_filepath: str) -> Path:

    rel = Path(rel_filepath).relative_to("data/raw_subset")
    return (OCR_DIR / rel).with_suffix(".json")

def ensure_ocr(manifest_csv, log_every=200):

    import csv
    rows = list(csv.DictReader(open(manifest_csv)))
    done = 0
    for r in rows:
        cp = cache_path(r["filepath"])
        if cp.exists():
            done += 1
            continue
        pil = Image.open(C.ROOT / r["filepath"])
        words, boxes = ocr_image(pil)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"words": words, "boxes": boxes}))
        done += 1
        if done % log_every == 0:
            print(f"[ocr] {manifest_csv.name}: {done}/{len(rows)}")
    print(f"[ocr] {manifest_csv.name}: completo ({len(rows)} imgs)")

def get_processor():
    from transformers import LayoutLMv3Processor

    return LayoutLMv3Processor.from_pretrained(LAYOUTLMV3_ID, apply_ocr=False)

def make_layout_dataset(manifest_csv, processor, max_len=512):
    import csv
    from torch.utils.data import Dataset

    items = [(r["filepath"], int(r["label"]))
             for r in csv.DictReader(open(manifest_csv))]

    class LayoutDataset(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            rel, label = items[i]
            pil = Image.open(C.ROOT / rel).convert("RGB")
            oc = json.loads(cache_path(rel).read_text())
            enc = processor(pil, oc["words"], boxes=oc["boxes"],
                            truncation=True, padding="max_length",
                            max_length=max_len, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            return item

    return LayoutDataset()

def build_layoutlmv3(num_classes=C.NUM_CLASSES, grad_checkpointing=False):
    from transformers import LayoutLMv3ForSequenceClassification
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        LAYOUTLMV3_ID, num_labels=num_classes)

    if grad_checkpointing and model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model
