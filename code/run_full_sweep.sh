#!/usr/bin/env bash
# Full 46-detector inference sweep across every benchmark. Each stage skips
# work that is already on disk, so the chain can be killed and restarted
# without loss. Paths resolve from the repository root.
#
#   DATA_ROOT      holds coco/, nightowls/ and llvip/LLVIP/
#   ULTRA_WEIGHTS  holds the Ultralytics checkpoints (*.pt)
set -u
: "${DATA_ROOT:?set DATA_ROOT to the directory holding coco/, nightowls/ and llvip/}"
: "${ULTRA_WEIGHTS:?set ULTRA_WEIGHTS to the directory holding the Ultralytics checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$DATA_ROOT
cd "$(dirname "$0")/.."

echo "############ 1/4  COCO val2017 person, 41 Ultralytics detectors ############"
python -u code/detect_coco_ultralytics.py \
  --cocoRoot $D/coco --outDir results/detections_coco_all46 \
  --weightDir $ULTRA_WEIGHTS || echo "COCO_ULTRALYTICS_STAGE_FAILED"

echo; echo "############ 2/4  COCO val2017 person, 5 torchvision detectors ############"
python -u code/detect_coco_person.py \
  --cocoRoot $D/coco --outDir results/detections_coco_all46 || echo "COCO_TORCHVISION_STAGE_FAILED"

echo; echo "############ 3/4  LLVIP visible/infrared, 3463 pairs ############"
python -u code/llvip_modality.py \
  --root $D/llvip/LLVIP --weightDir $ULTRA_WEIGHTS \
  --out results/llvip/llvip_modality_full.json || echo "LLVIP_STAGE_FAILED"

echo; echo "############ 4/4  NightOwls night, 51848 images ############"
python -u code/detect_nightowls.py \
  --annFile $D/nightowls/annotations/nightowls_validation.json \
  --imageRoot $D/nightowls/nightowls_validation \
  --outDir results/nightowls/detections \
  --weightDir $ULTRA_WEIGHTS || echo "NIGHTOWLS_STAGE_FAILED"

echo; echo "FULL SWEEP COMPLETE"
