#!/usr/bin/env bash
# Fine-tunes the five torchvision detectors on the LLVIP infrared channel and
# evaluates them on the infrared test split, so that adaptation after a change
# of sensor is measured on the same channel on both sides of the comparison.
# A model whose result file already exists is skipped. Delete the file to rerun.
set -u
: "${DATA_ROOT:?set DATA_ROOT to the directory holding llvip/LLVIP/}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
D=$DATA_ROOT/llvip/LLVIP
MODELS="faster_rcnn_r50_fpn_v2 retinanet_r50_fpn_v2 fcos_r50_fpn ssd300_vgg16 ssdlite320_mnv3"

for m in $MODELS; do
  out=$HERE/results/finetune/rq7_finetune_ir_$m.json
  [ -f "$out" ] && { echo "skip $m, already done"; continue; }
  echo "===== infrared $m ====="
  python -u $HERE/code/rq7_finetune.py --root $D --model $m \
      --modality infrared --out $out || echo "FAILED $m"
done
echo "INFRARED_SWEEP_DONE"
