#!/bin/bash
set -x
ts(){ date "+%F %T"; }

export N_TRAIN=2000 N_VAL=500 N_TEST=1000
export BATCH_SIZE=32 EPOCHS=12
export VLM_TEST_DONUT=500 VLM_TEST_QWEN=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== BUILD SUBSET 2000/500/1000 ($(ts)) ====="
python -m scripts.build_subset > results/scale_build.log 2>&1; echo "BUILD EXIT=$?"

echo "===== GRADE ($(ts)) ====="
python -m scripts.run_experiments > results/scale_grade.log 2>&1; echo "GRADE EXIT=$?"

echo "===== EFICIENCIA ($(ts)) ====="
python -m scripts.efficiency_curve > results/scale_eff.log 2>&1; echo "EFF EXIT=$?"

echo "===== INTERPRETABILIDADE ($(ts)) ====="
python -m scripts.interpretability > results/scale_interp.log 2>&1; echo "INTERP EXIT=$?"

export MM_TRAIN=400 MM_VAL=100 MM_TEST=200 MM_EPOCHS=8 MM_LR=2e-5

echo "===== MULTIMODAL ($(ts)) ====="
python -m scripts.run_layoutlmv3 > results/scale_mm.log 2>&1; echo "MM EXIT=$?"

echo "===== VLM ($(ts)) ====="
python -m scripts.run_vlm > results/scale_vlm.log 2>&1; echo "VLM EXIT=$?"

rm -f results/metrics/ensemble_probs.npz results/metrics/crossmodal_feats.npz
echo "===== ENSEMBLE ($(ts)) ====="
FORCE=1 python -m scripts.run_ensemble > results/scale_ensemble.log 2>&1; echo "ENS EXIT=$?"

echo "===== MOE ($(ts)) ====="
python -m scripts.run_moe > results/scale_moe.log 2>&1; echo "MOE EXIT=$?"

echo "===== CROSSMODAL ($(ts)) ====="
FORCE=1 python -m scripts.run_crossmodal > results/scale_xmodal.log 2>&1; echo "XMODAL EXIT=$?"

echo "===== CONSOLIDA ($(ts)) ====="
python -m scripts.consolidate > results/scale_consolidate.log 2>&1; echo "CONS EXIT=$?"

echo "===== FIM ($(ts)) ====="
