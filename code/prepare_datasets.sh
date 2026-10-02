#!/usr/bin/env bash
# Download the data used in the paper and arrange it under DATA_ROOT in the
# layout the other scripts expect. Every file is checked against the SHA-256
# checksum of the copy that produced the published results, recorded in
# dataset_checksums.sha256. Downloads resume if interrupted, and a file that
# is already present and verified is not fetched again.
#
# Usage:
#   export DATA_ROOT=/path/to/datasets
#   bash code/prepare_datasets.sh                  # all four
#   bash code/prepare_datasets.sh coco kaist       # a subset
#
# Approximate download sizes: COCO 1.1 GB, NightOwls validation 57.5 GB,
# LLVIP 4.0 GB, KAIST evaluation kit and published detections 0.1 GB.
# Each dataset is distributed under its own licence, which applies to its use.
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT to the directory that will hold the datasets}"
HERE=$(cd "$(dirname "$0")" && pwd)
SUMS=$HERE/dataset_checksums.sha256
command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required"; exit 1; }

verify() {  # path relative to DATA_ROOT
    local want got
    want=$(awk -v f="$1" '$2 == f {print $1}' "$SUMS")
    [ -n "$want" ] || { echo "no checksum recorded for $1"; return 1; }
    got=$(sha256sum "$DATA_ROOT/$1" | awk '{print $1}')
    if [ "$want" = "$got" ]; then echo "  verified  $1"; else echo "  CHECKSUM MISMATCH  $1"; return 1; fi
}

fetch() {  # url, path relative to DATA_ROOT
    local dest=$DATA_ROOT/$2
    if [ -f "$dest" ] && verify "$2" >/dev/null 2>&1; then
        echo "  present   $2"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  fetching  $1"
    curl -L --fail --retry 5 -C - -o "$dest" "$1"
    verify "$2"
}

count() {  # directory, file pattern, expected number
    local n
    n=$(find -L "$1" -maxdepth 1 -name "$2" | wc -l)
    if [ "$n" -eq "$3" ]; then echo "  found     $n files in ${1#$DATA_ROOT/}"
    else echo "  EXPECTED $3 FILES IN $1, FOUND $n"; return 1; fi
}

coco() {
    echo "COCO 2017 validation"
    local u=http://images.cocodataset.org
    fetch $u/zips/val2017.zip coco/_downloads/val2017.zip
    fetch $u/annotations/annotations_trainval2017.zip coco/_downloads/annotations_trainval2017.zip
    unzip -q -n "$DATA_ROOT/coco/_downloads/val2017.zip" -d "$DATA_ROOT/coco"
    unzip -q -n "$DATA_ROOT/coco/_downloads/annotations_trainval2017.zip" -d "$DATA_ROOT/coco"
    verify coco/annotations/instances_val2017.json
    verify coco/annotations/person_keypoints_val2017.json
    count "$DATA_ROOT/coco/val2017" '*.jpg' 5000
}

nightowls() {
    echo "NightOwls validation"
    local u=http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python
    fetch $u/nightowls_validation.json nightowls/annotations/nightowls_validation.json
    fetch $u/nightowls_validation.zip nightowls/_downloads/nightowls_validation.zip
    unzip -q -n "$DATA_ROOT/nightowls/_downloads/nightowls_validation.zip" -d "$DATA_ROOT/nightowls"
    count "$DATA_ROOT/nightowls/nightowls_validation" '*.png' 51848
}

llvip() {
    # A mirror of the LLVIP release (https://bupt-ai-cz.github.io/LLVIP/).
    echo "LLVIP"
    fetch https://huggingface.co/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip llvip/_downloads/LLVIP.zip
    unzip -q -n "$DATA_ROOT/llvip/_downloads/LLVIP.zip" -d "$DATA_ROOT/llvip"
    count "$DATA_ROOT/llvip/LLVIP/visible/test" '*.jpg' 3463
    count "$DATA_ROOT/llvip/LLVIP/infrared/test" '*.jpg' 3463
}

kaist() {
    # Evaluation kit, improved test annotations and five published detection
    # files, all from the MLPD repository. KAIST images are not needed.
    echo "KAIST evaluation kit and published detections"
    local u=https://raw.githubusercontent.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection/main/evaluation_script
    for f in evaluation_script.py coco.py cocoeval.py KAIST_annotation.json; do
        fetch $u/$f kaist/eval_kit/$f
    done
    for m in ARCNN CIAN MBNet MLPD MSDS-RCNN; do
        fetch $u/state_of_arts/${m}_result.txt kaist/published_detections/${m}_result.txt
    done
}

stages=("$@")
[ ${#stages[@]} -gt 0 ] || stages=(coco nightowls llvip kaist)
for s in "${stages[@]}"; do
    case $s in
        coco|nightowls|llvip|kaist) "$s" ;;
        *) echo "unknown dataset: $s (choose from coco nightowls llvip kaist)"; exit 1 ;;
    esac
done
echo "Data ready under $DATA_ROOT"
