#!/usr/bin/env bash
# Runs every detection-based COCO analysis over the full 46-detector set.
#
# The detection directory holds one COCO-format file per detector: the 41
# Ultralytics models written by detect_coco_ultralytics.py and the 5
# torchvision models written by detect_coco_person.py (see run_full_sweep.sh).
# Each detector is represented by exactly one inference pass.
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT to the directory holding coco/}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
D=$DATA_ROOT/coco
S=$HERE/code
R=$HERE/results
DET=${DET:-$R/detections_coco_all46}
export MPLBACKEND=Agg

N=$(ls $DET/*_meta.json | wc -l)
echo "### detectors: $N"
[ "$N" -eq 46 ] || { echo "EXPECTED 46 DETECTORS, GOT $N"; exit 1; }

echo "### protocol sensitivity"
python -u $S/rq2_protocol_sensitivity.py --gtFile $D/annotations/instances_val2017.json \
  --detDir $DET --out $R/rq2_protocol/rq2_all_persons.json
python -u $S/rq2_protocol_sensitivity.py --gtFile $D/annotations/instances_val2017.json \
  --detDir $DET --minHeight 50 --out $R/rq2_protocol/rq2_height50.json

echo "### rank stability across protocols"
python -u $S/rq2b_rank_stability.py \
  --rq2 $R/rq2_protocol/rq2_all_persons.json $R/rq2_protocol/rq2_height50.json \
  --out $R/rq2_protocol/rq2b_rank_stability.json

echo "### protocol axes"
python -u $S/rq2c_protocol_axes.py --gtFile $D/annotations/instances_val2017.json \
  --detDir $DET --out $R/rq2_protocol/rq2c_protocol_axes.json

echo "### degradation subsets"
python -u $S/rq1_degradation.py --gtFile $D/annotations/instances_val2017.json \
  --kpFile $D/annotations/person_keypoints_val2017.json --detDir $DET \
  --out $R/rq1_degradation/rq1_coco_person.json

echo "### error decomposition"
python -u $S/oracle_analysis.py --gtFile $D/annotations/instances_val2017.json \
  --detDir $DET --out $R/oracle/oracle_coco_person.json

echo "### inversion threshold"
python -u $S/rq2d_inversion_threshold.py \
  --axes $R/rq2_protocol/rq2c_protocol_axes.json \
  --rank $R/rq2_protocol/rq2b_rank_stability.json \
  --rq1 $R/rq1_degradation/rq1_coco_person.json \
  --kaist $R/kaist_published_reeval/kaist_published_reeval.json \
  --out $R/rq2_protocol/rq2d_inversion_threshold.json

echo "### pairwise significance, Holm and Benjamini-Hochberg over 1035 pairs"
python -u $S/run_statistics.py --gtFile $D/annotations/instances_val2017.json \
  --detDir $DET --detectionMatrix $R/oracle/oracle_coco_person_detection_matrix.json \
  --out $R/statistics/coco_person_statistics.json

echo "### calibration"
python -u $S/rq6_calibration.py --gtFile $D/annotations/instances_val2017.json \
  --detDir $DET --oracle $R/oracle/oracle_coco_person.json \
  --out $R/calibration/rq6_calibration46.json

echo "### significance against protocol stability"
python -u $S/rq2e_significance_vs_stability.py \
  --inversions $R/rq2_protocol/rq2d_inversion_threshold.json \
  --statistics $R/statistics/coco_person_statistics.json \
  --out $R/rq2_protocol/rq2e_significance_vs_stability.json

echo "ALL ANALYSES COMPLETE ($N detectors)"
