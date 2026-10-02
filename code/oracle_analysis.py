"""Oracle and error-decomposition analysis.

An aggregate score says that a detector is worse, but not why, or how much
better it could get. This script decomposes the error budget and then asks a
series of oracle questions of the form "if this one failure mode were fixed
perfectly, what would the score become?". The gap between the real score and
each oracle score is the headroom attributable to that failure mode.

False positives are classified, following the error-diagnosis convention of
Hoiem et al. and TIDE:
  duplicate      overlaps an already-matched object at IoU >= 0.5
  localisation   overlaps an unmatched object at 0.1 <= IoU < 0.5
  background     overlaps no object at IoU >= 0.1

Oracles reported:
  ORACLE-LOC     matching relaxed to IoU 0.1, so imprecise boxes still count
  ORACLE-DUP     duplicate false positives deleted
  ORACLE-BG      background false positives deleted
  ORACLE-RANK    detections perfectly ordered, correct ones first. This is the
                 ceiling achievable by improving the confidence ranking alone.
  ORACLE-ENSEMBLE an object counts as found if any model in the set found it.
                 This is the ceiling achievable by perfect per-object model
                 selection.

ORACLE-ENSEMBLE bears directly on model choice: if it sits far above the best
single model, no single-model recommendation is adequate.
"""
import argparse, json, os, sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST

PERSON = 1
LOC_IOU = 0.1


def load_gt(ann, min_height=0):
    gt = {i["id"]: {"boxes": [], "ignore": [], "ids": [], "h": []}
          for i in ann["images"]}
    for a in ann["annotations"]:
        if a["category_id"] != PERSON:
            continue
        x, y, w, h = a["bbox"]
        ig = 1 if (a.get("iscrowd") or h < min_height) else 0
        g = gt[a["image_id"]]
        g["boxes"].append([x, y, w, h]); g["ignore"].append(ig)
        g["ids"].append(a["id"]); g["h"].append(h)
    return gt


def load_dets(path):
    by = defaultdict(lambda: ([], []))
    for d in json.load(open(path)):
        if d["category_id"] != PERSON:
            continue
        by[d["image_id"]][0].append(d["bbox"])
        by[d["image_id"]][1].append(d["score"])
    return by


def analyse(gt, dets, iou_thr=0.5, max_det=1000):
    """Per-image matching that also classifies every false positive and records
    which ground-truth objects were found."""
    per_image, found, err = [], {}, {"duplicate": 0, "localisation": 0,
                                     "background": 0, "missed": 0, "tp": 0}
    npos_per_image = []
    for img, g in gt.items():
        real = [i for i, v in enumerate(g["ignore"]) if v == 0]
        npos_per_image.append(len(real))
        for i in real:
            found[g["ids"][i]] = False
        db, ds = dets.get(img, ([], []))
        if not db:
            per_image.append(np.zeros((0, 2)))
            err["missed"] += len(real)
            continue
        labels, order = P._match_image(db, ds, g["boxes"], g["ignore"],
                                       iou_thr, max_det)
        sb = [db[i] for i in order]
        ss = np.asarray(ds, float)[order]

        # which real object did each true positive take?
        ov = P.overlap_matrix(sb, g["boxes"], g["ignore"])
        taken = set()
        recs = []
        for di, lab in enumerate(labels):
            if lab == 1:
                cand = [j for j in range(len(g["boxes"]))
                        if g["ignore"][j] == 0 and j not in taken
                        and ov[di, j] >= iou_thr]
                if cand:
                    j = max(cand, key=lambda j: ov[di, j])
                    taken.add(j); found[g["ids"][j]] = True
                err["tp"] += 1
                recs.append((ss[di], 1))
            elif lab == 0:
                best_any = ov[di].max() if ov.shape[1] else 0.0
                best_unmatched = max(
                    [ov[di, j] for j in range(len(g["boxes"]))
                     if g["ignore"][j] == 0 and j not in taken] or [0.0])
                if best_any >= iou_thr:
                    kind = "duplicate"
                elif best_unmatched >= LOC_IOU:
                    kind = "localisation"
                else:
                    kind = "background"
                err[kind] += 1
                recs.append((ss[di], 0))
        per_image.append(np.array(recs) if recs else np.zeros((0, 2)))
    err["missed"] = sum(1 for v in found.values() if not v)
    return per_image, npos_per_image, found, err


def mr_of(per_image, npos_per_image, image_ids):
    s = ST.BootstrapSet(per_image, npos_per_image, image_ids)
    return s.metrics()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minHeight", type=float, default=0)
    a = ap.parse_args()

    ann = json.load(open(a.gtFile))
    gt = load_gt(ann, a.minHeight)
    image_ids = list(gt)
    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))

    rec = {"gt_file": a.gtFile, "min_height": a.minHeight,
           "loc_iou": LOC_IOU, "models": {}, "ensemble": {}}
    all_found = {}

    for m in models:
        dets = load_dets(os.path.join(a.detDir, f"{m}.json"))
        pi, npos, found, err = analyse(gt, dets)
        base = mr_of(pi, npos, image_ids)
        all_found[m] = found

        # ORACLE-LOC: relax the matching threshold
        pi_l, npos_l, found_l, err_l = analyse(gt, dets, iou_thr=LOC_IOU)
        loc = mr_of(pi_l, npos_l, image_ids)

        # Unused helper that returns the base records unchanged. The duplicate
        # and background oracles are computed by rerun() below.
        def drop(kind):
            out = []
            for img, recs in zip(image_ids, pi):
                out.append(recs)
            return out

        # ORACLE-DUP / ORACLE-BG: re-match at IoU 0.5 with the selected
        # false-positive classes deleted
        def rerun(suppress):
            per_image = []
            for img, g in gt.items():
                db, ds = dets.get(img, ([], []))
                if not db:
                    per_image.append(np.zeros((0, 2))); continue
                labels, order = P._match_image(db, ds, g["boxes"], g["ignore"], 0.5, 1000)
                sb = [db[i] for i in order]; ss = np.asarray(ds, float)[order]
                ov = P.overlap_matrix(sb, g["boxes"], g["ignore"])
                taken, recs = set(), []
                for di, lab in enumerate(labels):
                    if lab == 1:
                        cand = [j for j in range(len(g["boxes"]))
                                if g["ignore"][j] == 0 and j not in taken and ov[di, j] >= 0.5]
                        if cand:
                            taken.add(max(cand, key=lambda j: ov[di, j]))
                        recs.append((ss[di], 1))
                    elif lab == 0:
                        best_any = ov[di].max() if ov.shape[1] else 0.0
                        best_un = max([ov[di, j] for j in range(len(g["boxes"]))
                                       if g["ignore"][j] == 0 and j not in taken] or [0.0])
                        kind = ("duplicate" if best_any >= 0.5
                                else "localisation" if best_un >= LOC_IOU else "background")
                        if kind in suppress:
                            continue
                        recs.append((ss[di], 0))
                per_image.append(np.array(recs) if recs else np.zeros((0, 2)))
            return mr_of(per_image, npos, image_ids)

        dup = rerun({"duplicate"})
        bg = rerun({"background"})

        # ORACLE-RANK: perfect confidence ordering of the same detections
        rank_pi = []
        for recs in pi:
            if len(recs) == 0:
                rank_pi.append(recs); continue
            r = recs.copy()
            r[:, 0] = r[:, 1]                 # score becomes the label
            rank_pi.append(r)
        rnk = mr_of(rank_pi, npos, image_ids)

        total_fp = err["duplicate"] + err["localisation"] + err["background"]
        rec["models"][m] = {
            "MR": round(base["MR"], 2), "AP50": round(base["AP50"], 2),
            "recall": round(base["recall"], 2),
            "error_budget": {
                "true_positives": err["tp"],
                "false_positives_total": total_fp,
                "fp_duplicate": err["duplicate"],
                "fp_localisation": err["localisation"],
                "fp_background": err["background"],
                "fp_duplicate_pct": round(err["duplicate"] / total_fp * 100, 1) if total_fp else 0,
                "fp_localisation_pct": round(err["localisation"] / total_fp * 100, 1) if total_fp else 0,
                "fp_background_pct": round(err["background"] / total_fp * 100, 1) if total_fp else 0,
                "objects_never_found": err["missed"],
                "objects_total": sum(npos),
                "never_found_pct": round(err["missed"] / max(sum(npos), 1) * 100, 1),
            },
            "oracles": {
                "ORACLE-LOC": {"MR": round(loc["MR"], 2),
                               "gain": round(base["MR"] - loc["MR"], 2)},
                "ORACLE-DUP": {"MR": round(dup["MR"], 2),
                               "gain": round(base["MR"] - dup["MR"], 2)},
                "ORACLE-BG": {"MR": round(bg["MR"], 2),
                              "gain": round(base["MR"] - bg["MR"], 2)},
                "ORACLE-RANK": {"MR": round(rnk["MR"], 2),
                                "gain": round(base["MR"] - rnk["MR"], 2)},
            },
        }
        e = rec["models"][m]
        print(f"\n== {m} ==  MR={e['MR']}  recall={e['recall']}")
        b = e["error_budget"]
        print(f"   false positives: {b['false_positives_total']:>7}  "
              f"dup {b['fp_duplicate_pct']}%  loc {b['fp_localisation_pct']}%  "
              f"bg {b['fp_background_pct']}%")
        print(f"   objects never found: {b['objects_never_found']}/{b['objects_total']} "
              f"({b['never_found_pct']}%)")
        for k, v in e["oracles"].items():
            print(f"   {k:<14} MR {v['MR']:>6.2f}   headroom {v['gain']:>6.2f}")

    # ORACLE-ENSEMBLE: an object counts as found if any model found it
    obj_ids = list(next(iter(all_found.values())))
    any_found = sum(1 for o in obj_ids if any(all_found[m][o] for m in models))
    best_single = max(models, key=lambda m: sum(all_found[m].values()))
    best_n = sum(all_found[best_single].values())
    rec["ensemble"] = {
        "objects_total": len(obj_ids),
        "found_by_best_single_model": best_n,
        "best_single_model": best_single,
        "found_by_any_model": any_found,
        "recall_best_single_pct": round(best_n / len(obj_ids) * 100, 2),
        "recall_oracle_ensemble_pct": round(any_found / len(obj_ids) * 100, 2),
        "headroom_pct": round((any_found - best_n) / len(obj_ids) * 100, 2),
        "found_by_none": len(obj_ids) - any_found,
        "found_by_none_pct": round((len(obj_ids) - any_found) / len(obj_ids) * 100, 2),
        "interpretation": ("headroom is the recall a perfect per-object model "
                           "selector would add over the best single model; "
                           "found_by_none is the floor no selection strategy "
                           "over this model set can cross"),
    }
    e = rec["ensemble"]
    print(f"\n=== ORACLE-ENSEMBLE ===")
    print(f"  best single model      : {e['best_single_model']} "
          f"({e['recall_best_single_pct']}% of objects found)")
    print(f"  any model found it     : {e['recall_oracle_ensemble_pct']}%")
    print(f"  perfect-selection gain : {e['headroom_pct']} points")
    print(f"  found by no model      : {e['found_by_none']} objects "
          f"({round(e['found_by_none']/e['objects_total']*100,2)}%)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    # Per-object detection matrix, consumed by run_statistics.py for McNemar
    # tests between models.
    mat = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gt_file": a.gtFile, "detDir": a.detDir,
           "object_ids": obj_ids,
           "detected": {m: [bool(all_found[m][o]) for o in obj_ids] for m in models}}
    mp = a.out.rsplit(".", 1)[0] + "_detection_matrix.json"
    json.dump(mat, open(mp, "w"))
    print(f"\n-> {a.out}\n-> {mp}")


if __name__ == "__main__":
    main()
