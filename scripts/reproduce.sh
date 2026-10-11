#!/usr/bin/env bash
# Reproduce every number in the README from the raw image folders.
# Usage: bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1  # torch.onnx prints emoji; Windows consoles redirected to a file choke on them

# Port of the original Keras CNN (48 px input, random init, no class weighting): the baseline.
LEGACY=(-s model.backbone=legacy_cnn -s model.pretrained=false -s model.image_size=48
        -s model.drop_rate=0.5 -s train.class_weighting=none -s train.label_smoothing=0
        -s train.lr=0.001 -s train.epochs=60 -s train.patience=12)
# The same ConvNeXt as the main model, from random init: how much does pretraining add?
SCRATCH=(-s model.pretrained=false -s train.lr=0.002 -s train.epochs=80 -s train.patience=15)

uv run moodtune prepare

uv run moodtune train --name pretrained
uv run moodtune evaluate runs/pretrained
uv run moodtune train --name legacy "${LEGACY[@]}"
uv run moodtune evaluate runs/legacy "${LEGACY[@]}"
uv run moodtune train --name scratch "${SCRATCH[@]}"
uv run moodtune evaluate runs/scratch "${SCRATCH[@]}"

uv run moodtune export runs/pretrained -o models/moodtune.onnx
uv run moodtune bench
