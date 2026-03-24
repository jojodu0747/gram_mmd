#!/bin/bash
# Full backbone sweep — runs each backbone sequentially
# Waits for any existing sweep to finish first
set -e
cd "$(dirname "$0")/.."

LOG_DIR="/tmp"

echo "=== Waiting for any running sweep to finish ==="
while pgrep -f "sweep_synthetic_kadid" > /dev/null 2>&1; do
    sleep 30
done
echo "=== Previous sweep finished, starting remaining backbones ==="

for bb in dc_ae lpips_vgg vgg19 dinov2_vitb14 resnet50; do
    echo "=== Starting $bb ==="
    .venv/bin/python experiments/sweep_synthetic_kadid.py \
        --backbones "$bb" \
        --force-recompute \
        >> "${LOG_DIR}/sweep_all_backbones.log" 2>&1
    echo "=== $bb done ==="
done

echo "=== All backbones complete ==="
