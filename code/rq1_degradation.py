"""Degradation analysis: how detector performance changes as visibility, person
scale or crowd density becomes less favourable.

Uses the same fixed detection files as the protocol-sensitivity analysis
(rq2_protocol_sensitivity.py), so detector behaviour is held constant and only
the evaluation subset changes. Subsets follow the pedestrian-benchmark
convention: objects outside the subset become ignore entries rather than being
deleted, so detections that fire on them are neither rewarded nor penalised.
Deleting them instead would manufacture false positives and overstate
degradation.

Occlusion is derived from COCO person keypoint visibility. COCO carries no
visible-box/full-box ratio like CityPersons, so this is a declared proxy, not
an equivalent measure, and it is labelled as such in the output.
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P

PERSON = 1

HEIGHT_BINS = [(0, 25, "h<25"), (25, 50, "25-50"), (50, 75, "50-75"),
               (75, 125, "75-125"), (125, 1e9, "h>=125")]
CROWD_BINS = [(1, 1, "1 person"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 1e9, "10+")]
OCC_BINS = [(0.0, 1e-9, "none"), (1e-9, 0.35, "light"),
            (0.35, 0.65, "moderate"), (0.65, 1.01, "severe")]


def occlusion_proxy(kp_ann):
    """Fraction of labelled keypoints marked occluded (visibility flag 1).
    Instances with no labelled keypoints return None and are excluded from the
    occlusion analysis rather than guessed at."""
    out = {}
    for a in kp_ann["annotations"]:
        if a["category_id"] != PERSON or a.get("iscrowd"):
            continue
        v = np.array(a["keypoints"]).reshape(-1, 3)[:, 2]
        lab = int((v > 0).sum())
        out[a["id"]] = float((v == 1).sum()) / lab if lab else None
    return out


def build(ann, keep):
    """Ground truth in which `keep(a)` decides subset membership. Non-members
    and crowd regions become ignore entries."""
    gt = {i["id"]: ([], []) for i in ann["images"]}
    for a in ann["annotations"]:
        if a["category_id"] != PERSON:
            continue
        x, y, w, h = a["bbox"]
        ig = 0 if (not a.get("iscrowd") and keep(a)) else 1
        gt[a["image_id"]][0].append([x, y, w, h])
        gt[a["image_id"]][1].append(ig)
    return gt


def score(gt, dt, max_det=1000):
    tp, fp, npos, nimg = P.evaluate(gt, dt, iou_thr=0.5, max_det=max_det)
    mr, _ = P.log_average_miss_rate(tp, fp, npos, nimg)
    return {"MR": round(mr * 100, 2),
            "AP50": round(P.voc_ap(tp, fp, npos, "all") * 100, 2),
            "n": npos,
            "recall": round(float(tp[-1] / npos) * 100, 2) if len(tp) and npos else 0.0}


def load_dets(path):
    by = {}
    for d in json.load(open(path)):
        if d["category_id"] != PERSON:
            continue
        by.setdefault(d["image_id"], ([], []))
        by[d["image_id"]][0].append(d["bbox"])
        by[d["image_id"]][1].append(d["score"])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtFile", required=True)
    ap.add_argument("--kpFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ann = json.load(open(a.gtFile))
    occ = occlusion_proxy(json.load(open(a.kpFile)))
    per_img = defaultdict(int)
    for x in ann["annotations"]:
        if x["category_id"] == PERSON and not x.get("iscrowd"):
            per_img[x["image_id"]] += 1

    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))

    axes = {}
    axes["height"] = [(lbl, (lambda lo, hi: lambda x: lo <= x["bbox"][3] < hi)(lo, hi))
                      for lo, hi, lbl in HEIGHT_BINS]
    axes["crowding"] = [(lbl, (lambda lo, hi: lambda x: lo <= per_img[x["image_id"]] <= hi)(lo, hi))
                        for lo, hi, lbl in CROWD_BINS]
    axes["occlusion_proxy"] = [
        (lbl, (lambda lo, hi: lambda x: (occ.get(x["id"]) is not None
                                         and lo <= occ[x["id"]] < hi))(lo, hi))
        for lo, hi, lbl in OCC_BINS]

    rec = {"gt_file": a.gtFile, "note": (
        "Subsets are formed by marking out-of-subset objects as ignore, not by "
        "deleting them. Occlusion is a keypoint-visibility proxy, not a "
        "CityPersons visible/full box ratio."),
        "axes": {}, "overall": {}}

    dets = {m: load_dets(os.path.join(a.detDir, f"{m}.json")) for m in models}

    gt_all = build(ann, lambda x: True)
    for m in models:
        rec["overall"][m] = score(gt_all, dets[m])
    print("Overall (all persons, crowd regions ignored):")
    for m in models:
        print(f"  {m:<26} MR={rec['overall'][m]['MR']:>6.2f}  "
              f"AP50={rec['overall'][m]['AP50']:>6.2f}  n={rec['overall'][m]['n']}")

    for axis, bins in axes.items():
        rec["axes"][axis] = {}
        print(f"\n=== {axis} (MR, lower is better) ===")
        gts = {lbl: build(ann, fn) for lbl, fn in bins}
        hdr = "".join(f"{lbl:>12}" for lbl, _ in bins)
        print(f"  {'model':<26}{hdr}")
        for m in models:
            rec["axes"][axis][m] = {lbl: score(gts[lbl], dets[m]) for lbl, _ in bins}
            row = "".join(f"{rec['axes'][axis][m][lbl]['MR']:>12.2f}" for lbl, _ in bins)
            print(f"  {m:<26}{row}")
        print(f"  {'(objects)':<26}" +
              "".join(f"{rec['axes'][axis][models[0]][lbl]['n']:>12}" for lbl, _ in bins))

    # Derived degradation factors (worst-bin MR over best-bin MR per model),
    # written to the output so that each reported factor is traceable to a file.
    rec["degradation_factor"] = {}
    for axis, per_model in rec["axes"].items():
        rec["degradation_factor"][axis] = {}
        for m, bins in per_model.items():
            vals = [(b, v["MR"]) for b, v in bins.items() if v["n"] > 0]
            if len(vals) < 2:
                continue
            best = min(vals, key=lambda z: z[1])
            worst = max(vals, key=lambda z: z[1])
            rec["degradation_factor"][axis][m] = {
                "best_bin": best[0], "best_MR": best[1],
                "worst_bin": worst[0], "worst_MR": worst[1],
                "factor": round(worst[1] / best[1], 2) if best[1] > 0 else None}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
