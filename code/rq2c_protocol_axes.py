"""Protocol axes: which axis of an evaluation protocol reorders detectors.

Two protocols usually differ along several axes at once. Varying only the
metric family (COCO AP, VOC AP, log-average miss rate), as
rq2_protocol_sensitivity.py does, leaves the other axes fixed. This script
varies those remaining axes independently, on the same fixed detections, with
the log-average miss rate as the metric:

  iou_threshold     the matching strictness, 0.50 through 0.75. Papers report
                    both 0.50 and 0.75, and the stricter threshold stresses
                    localisation quality.
  fppi_range        the false-positives-per-image window the miss rate is
                    averaged over. Caltech-style evaluation uses [0.01, 1.0].
                    Other work uses wider or narrower windows, and a narrower
                    low-FPPI window rewards precision at a strict alarm budget.
  height_floor      the minimum object height admitted for scoring. Objects
                    below it become ignore entries.
  crowd_policy      whether crowd or ignore regions absorb detections or are
                    deleted outright.

Each variant changes one axis relative to the reference protocol (IoU 0.50,
Caltech FPPI window, no height floor, crowd regions ignored). Kendall's tau-b
against the reference ranking isolates the effect of that axis. An axis with
tau below 1 reorders detectors and is therefore a genuine comparability hazard.
An axis with tau of 1 is not, at least over this model set and this data.
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST

PERSON = 1

FPPI_WINDOWS = {
    "caltech_0.01-1.0": np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000,
                                  0.1778, 0.3162, 0.5623, 1.0000]),
    "strict_0.001-0.1": np.logspace(-3, -1, 9),
    "wide_0.01-10":     np.logspace(-2, 1, 9),
    "single_0.1":       np.array([0.1]),
}


def load_gt(ann, min_height=0, drop_crowd=False):
    gt = {i["id"]: ([], []) for i in ann["images"]}
    for a in ann["annotations"]:
        if a["category_id"] != PERSON:
            continue
        x, y, w, h = a["bbox"]
        if drop_crowd and a.get("iscrowd"):
            continue                       # deleted outright rather than ignored
        ig = 1 if (a.get("iscrowd") or h < min_height) else 0
        gt[a["image_id"]][0].append([x, y, w, h])
        gt[a["image_id"]][1].append(ig)
    return gt


def load_dets(path):
    by = defaultdict(lambda: ([], []))
    for d in json.load(open(path)):
        if d["category_id"] == PERSON:
            by[d["image_id"]][0].append(d["bbox"])
            by[d["image_id"]][1].append(d["score"])
    return by


def score_all(gt, dets, iou, fppi):
    tp, fp, npos, nimg = P.evaluate(gt, dets, iou_thr=iou, max_det=1000)
    mr, _ = P.log_average_miss_rate(tp, fp, npos, nimg, fppi_points=fppi)
    return round(mr * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ann = json.load(open(a.gtFile))
    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))
    dets = {m: load_dets(os.path.join(a.detDir, f"{m}.json")) for m in models}

    variants = {}
    base_gt = load_gt(ann)
    variants["REFERENCE (IoU .50, Caltech FPPI, no floor, crowd ignored)"] = \
        (base_gt, 0.50, FPPI_WINDOWS["caltech_0.01-1.0"])
    for iou in (0.55, 0.60, 0.65, 0.70, 0.75):
        variants[f"IoU {iou:.2f}"] = (base_gt, iou, FPPI_WINDOWS["caltech_0.01-1.0"])
    for name, w in FPPI_WINDOWS.items():
        if name == "caltech_0.01-1.0":
            continue
        variants[f"FPPI window {name}"] = (base_gt, 0.50, w)
    for hf in (20, 50, 75):
        variants[f"height floor {hf} px"] = (load_gt(ann, min_height=hf), 0.50,
                                             FPPI_WINDOWS["caltech_0.01-1.0"])
    variants["crowd regions deleted, not ignored"] = \
        (load_gt(ann, drop_crowd=True), 0.50, FPPI_WINDOWS["caltech_0.01-1.0"])

    rec = {"models": models, "variants": {}, "reference": None}
    scores = {}
    for label, (g, iou, fppi) in variants.items():
        scores[label] = {m: score_all(g, dets[m], iou, fppi) for m in models}
        print(f"  scored: {label}")

    ref_label = list(variants)[0]
    ref_rank = {m: i + 1 for i, m in
                enumerate(sorted(models, key=lambda m: scores[ref_label][m]))}
    rec["reference"] = {"label": ref_label, "scores": scores[ref_label],
                        "rank": ref_rank}

    print(f"\n{'variant':<52}{'tau':>8}   inversions vs reference")
    for label in variants:
        r = {m: i + 1 for i, m in
             enumerate(sorted(models, key=lambda m: scores[label][m]))}
        tau, _, _ = ST.kendall_tau(ref_rank, r)
        inv = [f"{x}<->{y}" for i, x in enumerate(models) for y in models[i + 1:]
               if (ref_rank[x] < ref_rank[y]) != (r[x] < r[y])]
        rec["variants"][label] = {"scores": scores[label], "rank": r,
                                  "kendall_tau_vs_reference": round(tau, 4),
                                  "inversions": inv,
                                  "reorders": bool(inv)}
        short = [i.replace("_r50_fpn_v2", "").replace("_r50_fpn", "")
                 .replace("320_mnv3", "L").replace("300_vgg16", "300") for i in inv]
        print(f"  {label:<50}{tau:>8.3f}   {', '.join(short) or '-'}")

    reorder = [k for k, v in rec["variants"].items() if v["reorders"]]
    rec["summary"] = {
        "n_variants": len(variants),
        "n_reordering": len(reorder),
        "reordering_axes": reorder,
        "conclusion": ("axes listed in reordering_axes changed the detector "
                       "ordering on this data; the remainder did not"),
    }
    print(f"\n{len(reorder)} of {len(variants)} protocol variants reordered the models.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
