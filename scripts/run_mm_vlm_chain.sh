#!/bin/bash
set -x
cd /app 2>/dev/null || true
export MM_TRAIN=150 MM_VAL=40 MM_TEST=100 MM_EPOCHS=8 MM_BS=8
echo "===== MULTIMODAL ($(date)) ====="
python -m scripts.run_layoutlmv3 > results/mm.log 2>&1; echo "MM EXIT=$?"
echo "===== VLM ($(date)) ====="
python -m scripts.run_vlm > results/vlm.log 2>&1; echo "VLM EXIT=$?"
echo "===== CONSOLIDA ($(date)) ====="
python -m scripts.consolidate > results/consolidate.log 2>&1; echo "CONS EXIT=$?"
echo "===== FIM ($(date)) ====="
