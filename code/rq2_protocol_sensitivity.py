"""Protocol sensitivity: how much the evaluation protocol, rather than the
detector, changes the reported result.

One fixed set of person detections per model is scored under three incompatible
metric families on identical images and identical ground truth. Nothing is
retrained and no detection is recomputed, so any change in the ranking is
attributable to the protocol alone.

Protocols
---------
COCO      AP averaged over IoU 0.50:0.05:0.95 with 101-point recall
          interpolation, via pycocotools, at its own standard detection budget
          of 100 per image. Crowd regions absorb detections without penalty.
COCO@.5   The same implementation reported at the single IoU 0.50, which
          isolates the averaged-IoU component of the COCO definition.
VOC-all   AP at IoU 0.50 using the VOC2010+ all-point precision envelope.
VOC-11    AP at IoU 0.50 using the VOC2007 11-point sample.
MR        Log-average miss rate at IoU 0.50 over nine FPPI points from 0.01 to
          1.0, the pedestrian-detection convention. Lower is better.

A matched-budget variant of the COCO metric, using the same per-image detection
budget as the VOC and miss-rate protocols (--maxDet), is also computed so that
the detection budget can be separated from the metric definition.
"""
import argparse, contextlib, copy, io, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PERSON = 1


def build_gt(coco_ann, min_height=0):
    """Ground truth keyed by image. Crowd regions and any object below the
    height floor become ignore entries rather than being deleted, which is the
    pedestrian-benchmark convention and keeps them from counting as false
    positives."""
    gt = {i["id"]: ([], []) for i in coco_ann["images"]}
    for a in coco_ann["annotations"]:
        if a["category_id"] != PERSON:
            continue
        x, y, w, h = a["bbox"]
        ignore = 1 if (a.get("iscrowd") or h < min_height) else 0
        gt[a["image_id"]][0].append([x, y, w, h])
        gt[a["image_id"]][1].append(ignore)
    return gt


def load_dets(path):
    by = {}
    for d in json.load(open(path)):
        if d["category_id"] != PERSON:
            continue
        by.setdefault(d["image_id"], ([], []))
        by[d["image_id"]][0].append(d["bbox"])
        by[d["image_id"]][1].append(d["score"])
    return by


def _ap_from_precision(e, max_dets, area_lbl="all", iou_thr=None):
    """Mean precision from a COCOeval accumulation.

    Read directly from `eval['precision']` rather than through `summarize()`,
    because pycocotools' `_summarizeDets` hard-codes maxDets=100 for the
    averaged-IoU statistic and returns its -1 sentinel for any other budget.
    Shape is (IoU, recall, category, areaRng, maxDets).
    """
    p = e.params
    mind = p.maxDets.index(max_dets)
    aind = p.areaRngLbl.index(area_lbl)
    prec = e.eval["precision"]
    if iou_thr is not None:
        t = int(np.where(np.isclose(p.iouThrs, iou_thr))[0][0])
        prec = prec[t:t + 1]
    prec = prec[:, :, :, aind, mind]
    valid = prec[prec > -1]
    return float(valid.mean()) * 100 if valid.size else float("nan")


def coco_ap(coco_ann, det_file, max_dets, min_height=0):
    """COCO AP via the reference implementation.

    A pedestrian-style height floor is expressed through COCO's own `iscrowd`
    ignore mechanism rather than by deleting annotations, so that the same
    objects are ignored here as in the VOC and miss-rate protocols.
    """
    ann = copy.deepcopy(coco_ann)
    if min_height:
        for a in ann["annotations"]:
            if a["category_id"] == PERSON and a["bbox"][3] < min_height:
                a["iscrowd"] = 1
    keys = ("AP", "AP50", "APs", "APm", "APl")
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = ann
        c.createIndex()
        dets = [d for d in json.load(open(det_file)) if d["category_id"] == PERSON]
        if not dets:
            return {k: 0.0 for k in keys}
        cd = c.loadRes(copy.deepcopy(dets))
        e = COCOeval(c, cd, "bbox")
        e.params.catIds = [PERSON]
        e.params.maxDets = sorted({1, 10, 100, max_dets})
        e.evaluate(); e.accumulate()
    return {
        "AP":   _ap_from_precision(e, max_dets),
        "AP50": _ap_from_precision(e, max_dets, iou_thr=0.5),
        "APs":  _ap_from_precision(e, max_dets, "small"),
        "APm":  _ap_from_precision(e, max_dets, "medium"),
        "APl":  _ap_from_precision(e, max_dets, "large"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minHeight", type=float, default=0,
                    help="pedestrian-style height floor; objects below become ignore")
    ap.add_argument("--maxDet", type=int, default=1000)
    a = ap.parse_args()

    ann = json.load(open(a.gtFile))
    gt = build_gt(ann, a.minHeight)
    n_real = sum(int(np.sum(np.asarray(v[1]) == 0)) for v in gt.values())
    n_ign = sum(int(np.sum(np.asarray(v[1]) == 1)) for v in gt.values())
    print(f"images={len(gt)}  matchable persons={n_real}  ignore entries={n_ign}"
          f"  height floor={a.minHeight}")

    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))
    rec = {
        "gt_file": a.gtFile, "min_height": a.minHeight,
        "n_images": len(gt), "n_matchable": n_real, "n_ignore": n_ign,
        "protocols": {
            "COCO":    "AP@[.50:.05:.95], 101-pt, maxDets=100 (COCO standard)",
            "COCO@.5": "AP@0.50 from the same COCO run",
            "COCO_matchedBudget": f"AP@[.50:.05:.95] with maxDets={a.maxDet}",
            "VOC-all": "AP@0.50, VOC2010+ all-point envelope",
            "VOC-11":  "AP@0.50, VOC2007 11-point sample",
            "MR":      "log-average miss rate @0.50 over 9 FPPI points, LOWER is better",
        },
        "results": {},
    }

    for m in models:
        f = os.path.join(a.detDir, f"{m}.json")
        dt = load_dets(f)
        tp, fp, npos, nimg = P.evaluate(gt, dt, iou_thr=0.5, max_det=a.maxDet)
        mr, sampled = P.log_average_miss_rate(tp, fp, npos, nimg)
        c100 = coco_ap(ann, f, 100, a.minHeight)
        cmax = coco_ap(ann, f, a.maxDet, a.minHeight)
        rec["results"][m] = {
            "COCO": round(c100["AP"], 2),
            "COCO@.5": round(c100["AP50"], 2),
            "COCO_matchedBudget": round(cmax["AP"], 2),
            "COCO_APsmall": round(c100["APs"], 2),
            "COCO_APmedium": round(c100["APm"], 2),
            "COCO_APlarge": round(c100["APl"], 2),
            "VOC-all": round(P.voc_ap(tp, fp, npos, "all") * 100, 2),
            "VOC-11": round(P.voc_ap(tp, fp, npos, "11point") * 100, 2),
            "MR": round(mr * 100, 2),
            "n_detections": sum(len(v[0]) for v in dt.values()),
            "final_recall": round(float(tp[-1] / npos) * 100, 2) if len(tp) else 0.0,
            "mr_curve": {str(k): round(v, 4) for k, v in (sampled or {}).items()},
        }
        print(f"  {m:<26} COCO={rec['results'][m]['COCO']:>6.2f}  "
              f"VOC-all={rec['results'][m]['VOC-all']:>6.2f}  "
              f"MR={rec['results'][m]['MR']:>6.2f}")

    # Ranking per protocol. MR is lower-is-better, the rest higher-is-better.
    order = {}
    for prot in ["COCO", "COCO@.5", "COCO_matchedBudget", "VOC-all", "VOC-11", "MR"]:
        rev = prot != "MR"
        order[prot] = [m for m, _ in sorted(rec["results"].items(),
                                            key=lambda kv: kv[1][prot], reverse=rev)]
    rec["rankings"] = order
    rec["rank_table"] = {m: {p: order[p].index(m) + 1 for p in order} for m in models}
    rec["rank_disagreement"] = {
        m: max(v.values()) - min(v.values()) for m, v in rec["rank_table"].items()}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)

    print("\nRank by protocol (1 = best):")
    ps = list(order)
    print(f"  {'model':<26}" + "".join(f"{p:>20}" for p in ps) + f"{'spread':>8}")
    for m in models:
        print(f"  {m:<26}" + "".join(f"{rec['rank_table'][m][p]:>20}" for p in ps)
              + f"{rec['rank_disagreement'][m]:>8}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
