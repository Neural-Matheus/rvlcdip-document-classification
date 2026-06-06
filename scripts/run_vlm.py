import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data, models as M, training as T, analysis as A

C.set_seed()
device = C.get_device()
LABELS = data.get_label_names()

DONUT_ID = "naver-clova-ix/donut-base-finetuned-rvlcdip"
QWEN_ID = "Qwen/Qwen2-VL-2B-Instruct"
N_TEST_DONUT = int(os.environ.get("VLM_TEST_DONUT", 200))
N_TEST_QWEN = int(os.environ.get("VLM_TEST_QWEN", 50))

def test_subset(n):
    by = defaultdict(list)
    for r in csv.DictReader(open(C.MANIFEST_DIR / "test.csv")):
        by[int(r["label"])].append(r["filepath"])
    return [(fp, c) for c in range(C.NUM_CLASSES) for fp in by[c][:n]]

def match_label(text):

    t = text.lower()
    best, best_score = None, 0
    for i, name in enumerate(LABELS):
        score = sum(1 for w in name.split() if w in t)
        if name in t:
            score += 5
        if score > best_score:
            best, best_score = i, score
    return best if best is not None else -1

def run_donut():
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    print("[vlm] carregando Donut (rvlcdip)...")
    proc = DonutProcessor.from_pretrained(DONUT_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        DONUT_ID, torch_dtype=torch.float16).to(device).eval()
    task_prompt = "<s_rvlcdip>"
    dec_ids = proc.tokenizer(task_prompt, add_special_tokens=False,
                             return_tensors="pt").input_ids

    items = test_subset(N_TEST_DONUT)
    yp, yt = [], []
    t0 = time.time()
    for k, (rel, lab) in enumerate(items):
        pil = Image.open(C.ROOT / rel).convert("RGB")
        pv = proc(pil, return_tensors="pt").pixel_values.to(device, torch.float16)
        with torch.no_grad():
            out = model.generate(pv, decoder_input_ids=dec_ids.to(device),
                                 max_length=16, do_sample=False,
                                 pad_token_id=proc.tokenizer.pad_token_id,
                                 eos_token_id=proc.tokenizer.eos_token_id)
        seq = proc.batch_decode(out)[0]
        yp.append(match_label(seq)); yt.append(lab)
        if (k + 1) % 200 == 0:
            print(f"  donut {k+1}/{len(items)}")
    import numpy as np
    m = A.compute_metrics(np.array(yt), np.array(yp), LABELS)
    print(f"[vlm] Donut acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f} "
          f"({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "n_per_class": N_TEST_DONUT}

def run_qwen():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    print("[vlm] carregando Qwen2-VL-2B (bf16)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
    proc = AutoProcessor.from_pretrained(QWEN_ID)

    classes_str = ", ".join(LABELS)
    instr = ("You are a document image classifier. Classify the document into exactly "
             f"one of these 16 categories: {classes_str}. "
             "Answer with ONLY the category name, nothing else.")

    items = test_subset(N_TEST_QWEN)
    yp, yt = [], []
    t0 = time.time()
    for k, (rel, lab) in enumerate(items):
        path = str(C.ROOT / rel)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{path}"},
            {"type": "text", "text": instr}]}]
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, videos=vids,
                      padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        ans = proc.batch_decode(trimmed, skip_special_tokens=True)[0]
        yp.append(match_label(ans)); yt.append(lab)
        if (k + 1) % 100 == 0:
            print(f"  qwen {k+1}/{len(items)}  (ex: '{ans.strip()}')")
    import numpy as np
    m = A.compute_metrics(np.array(yt), np.array(yp), LABELS)
    print(f"[vlm] Qwen2-VL zero-shot acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f} "
          f"({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "n_per_class": N_TEST_QWEN}

def main():
    which = sys.argv[1:] or ["donut", "qwen"]
    out = {}
    if "donut" in which:
        out["donut_rvlcdip"] = run_donut()
    if "qwen" in which:
        out["qwen2vl_2b_zeroshot"] = run_qwen()
    (C.METRICS_DIR / "vlm.json").write_text(json.dumps(out, indent=2))
    print("\n=== VLM ===")
    for k, v in out.items():
        print(f"{k:24s} acc {v['accuracy']:.4f}  F1 {v['macro_f1']:.4f} "
              f"(n/classe={v['n_per_class']})")

if __name__ == "__main__":
    main()
