#!/bin/bash
set -x
echo "===== EFICIÊNCIA ($(date)) ====="
python -m scripts.efficiency_curve > results/eff.log 2>&1; echo "EFF EXIT=$?"
echo "===== MULTIMODAL ($(date)) ====="
python -m scripts.run_layoutlmv3 > results/mm.log 2>&1; echo "MM EXIT=$?"
echo "===== VLM ($(date)) ====="
python -m scripts.run_vlm > results/vlm.log 2>&1; echo "VLM EXIT=$?"
echo "===== CONSOLIDA ($(date)) ====="
python -m scripts.consolidate > results/consolidate.log 2>&1; echo "CONS EXIT=$?"
echo "===== FIM ($(date)) ====="
