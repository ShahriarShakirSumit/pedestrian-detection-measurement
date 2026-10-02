"""Validate the local log-average miss-rate implementation.

The implementation in `protocols.py` produces every miss-rate number reported
from this repository, so it is checked against a reference that is already
known to reproduce published values. The reference is the pinned KAIST
evaluator, whose Reasonable-subset outputs reproduce the miss rates published
for MLPD, MBNet, AR-CNN, MSDS-RCNN and CIAN.

The published detection files for those five methods are scored with
`protocols.py` under the same Reasonable-subset rules (height, occlusion and
boundary filtering as ignore regions, IoU 0.5) and compared with the reference
values. Agreement to within 0.05 miss-rate points for every method means that
the local matching, ignore handling, FPPI sampling and log-average computation
are equivalent to the reference on real data.

Usage: python validate_mr_implementation.py EVAL_KIT_DIR DETECTIONS_DIR
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P

# Reasonable-subset miss rates (%) produced by the pinned KAIST evaluator.
REFERENCE = {"MLPD": 7.58, "MBNet": 8.13, "ARCNN": 9.34,
             "MSDS-RCNN": 11.34, "CIAN": 14.12}

HT_RNG = (55, 1e10)          # Reasonable subset
OCC_RNG = {0, 1}
BND = (5, 5, 635, 507)       # evaluator boundary rectangle
MAX_DET = 1000


def load_dets(path, name_to_id):
    """Load detections grouped by image id.

    KAIST result .txt rows are image_index, x, y, w, h, score, separated by
    commas or whitespace, with a 1-based image index. COCO-style JSON result
    files are also accepted."""
    by_img = {}
    if path.endswith(".json"):
        for d in json.load(open(path)):
            by_img.setdefault(d["image_id"], ([], []))
            by_img[d["image_id"]][0].append(d["bbox"])
            by_img[d["image_id"]][1].append(d["score"])
        return by_img
    for line in open(path):
        p = line.split(",") if "," in line else line.split()
        if len(p) < 6:
            continue
        idx = int(float(p[0])) - 1
        img = name_to_id[idx]
        by_img.setdefault(img, ([], []))
        by_img[img][0].append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        by_img[img][1].append(float(p[5]))
    return by_img


def main():
    kit = sys.argv[1]
    dets_dir = sys.argv[2]
    ann = json.load(open(os.path.join(kit, "KAIST_annotation.json")))
    img_ids = sorted(i["id"] for i in ann["images"])
    name_to_id = {n: i for n, i in enumerate(img_ids)}

    by_img = {i: ([], []) for i in img_ids}
    for a in ann["annotations"]:
        if a["category_id"] != 1:
            continue
        x, y, w, h = a["bbox"]
        ignore = a.get("ignore", 0)
        if (a["height"] < HT_RNG[0] or a["height"] > HT_RNG[1]
                or a.get("occlusion") not in OCC_RNG
                or x < BND[0] or y < BND[1] or x + w > BND[2] or y + h > BND[3]):
            ignore = 1
        by_img[a["image_id"]][0].append([x, y, w, h])
        by_img[a["image_id"]][1].append(ignore)

    print(f"{'method':<12}{'reference':>11}{'ours':>9}{'diff':>9}   verdict")
    ok = True
    for name, ref in REFERENCE.items():
        f = os.path.join(dets_dir, f"{name}_result.txt")
        if not os.path.exists(f):
            print(f"{name:<12}  detection file missing"); ok = False; continue
        dt = load_dets(f, name_to_id)
        tp, fp, npos, nimg = P.evaluate(by_img, dt, iou_thr=0.5, max_det=MAX_DET)
        mr, _ = P.log_average_miss_rate(tp, fp, npos, nimg)
        mr *= 100
        d = mr - ref
        good = abs(d) < 0.05
        ok &= good
        print(f"{name:<12}{ref:>11.2f}{mr:>9.2f}{d:>+9.2f}   {'MATCH' if good else 'MISMATCH'}")

    print(f"\nmatchable ground-truth objects (Reasonable): {npos}")
    print("VALIDATED" if ok else "NOT VALIDATED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
