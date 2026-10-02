"""Rescoring analysis: how much of the ranking headroom can be recovered
without changing the detector.

The oracle analysis (oracle_analysis.py) measures the ranking headroom, the
miss-rate gain from perfectly ordering a detector's own detections by
correctness. This script asks how much of that headroom a simple rescoring can
recover with no change to the detector, no retraining and no extra inference.

Method. Detections are split by image into a fitting half and an evaluation
half, so nothing is fitted and scored on the same image. A logistic model
predicts whether a detection is a true positive from features available at
inference time and never from the label:

    raw detector score (logit-transformed)
    box height and area, log-scaled
    aspect ratio
    number of detections the model produced in that image, log-scaled
    the detection's normalised rank within its own image
    maximum overlap with a higher-scoring detection in the same image

The last two encode within-image competition, which a per-detection sigmoid
cannot see. Nothing here uses ground truth at inference time.

The comparison is deliberately conservative. Recovered headroom is reported as
a fraction of the ORACLE-RANK ceiling, and a permutation-style control is run
in which the fitted model is applied to shuffled features, so an apparent gain
that is really an artefact of re-ranking noise would show up.
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST

PERSON = 1


def load_gt(ann):
    gt = {i["id"]: ([], []) for i in ann["images"]}
    for a in ann["annotations"]:
        if a["category_id"] != PERSON:
            continue
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][0].append([x, y, w, h])
        gt[a["image_id"]][1].append(1 if a.get("iscrowd") else 0)
    return gt


def load_dets(path):
    by = defaultdict(lambda: ([], []))
    for d in json.load(open(path)):
        if d["category_id"] == PERSON:
            by[d["image_id"]][0].append(d["bbox"])
            by[d["image_id"]][1].append(d["score"])
    return by


def features_and_labels(gt, dets, image_ids):
    """Per-detection feature matrix, label, and the image it belongs to."""
    X, y, img = [], [], []
    for im in image_ids:
        gb, gi = gt[im]
        db, ds = dets.get(im, ([], []))
        if not db:
            continue
        labels, order = P._match_image(db, ds, gb, gi, 0.5, 1000)
        sb = [db[i] for i in order]
        ss = np.asarray(ds, float)[order]
        n = len(sb)
        boxes = np.asarray(sb, float)
        # overlap with any higher-scoring detection in the same image
        ov = P.overlap_matrix(sb, sb, np.zeros(n, dtype=int))
        np.fill_diagonal(ov, 0.0)
        higher = np.triu(np.ones((n, n)), 1)      # detections are score-sorted
        max_ov_higher = (ov * higher.T).max(axis=1) if n > 1 else np.zeros(n)
        for k in range(n):
            if labels[k] < 0:                      # ignored: not a training target
                continue
            s = float(np.clip(ss[k], 1e-6, 1 - 1e-6))
            h = max(boxes[k, 3], 1.0)
            w_ = max(boxes[k, 2], 1.0)
            X.append([
                float(np.log(s / (1 - s))),        # score logit
                float(np.log(h)),
                float(np.log(w_ * h)),
                float(w_ / h),
                float(np.log(n)),
                float(k) / max(n - 1, 1),          # normalised within-image rank
                float(max_ov_higher[k]),
            ])
            y.append(int(labels[k] == 1))
            img.append(im)
    return np.asarray(X), np.asarray(y), np.asarray(img)


def score_set(gt, per_image_records, npos_per_image, image_ids):
    s = ST.BootstrapSet(per_image_records, npos_per_image, image_ids)
    return s.metrics()


def records_from(scores, labels, imgs, image_ids):
    by = defaultdict(list)
    for s, l, i in zip(scores, labels, imgs):
        by[i].append((s, l))
    return [np.array(by[i]) if by[i] else np.zeros((0, 2)) for i in image_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    ann = json.load(open(a.gtFile))
    gt = load_gt(ann)
    oracle = json.load(open(a.oracle))
    all_ids = sorted(gt)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(all_ids))
    half = len(all_ids) // 2
    fit_ids = [all_ids[i] for i in perm[:half]]
    ev_ids = [all_ids[i] for i in perm[half:]]

    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))

    rec = {"seed": a.seed, "n_fit_images": len(fit_ids), "n_eval_images": len(ev_ids),
           "split": "images partitioned once; no image contributes to both halves",
           "features": ["score_logit", "log_height", "log_area", "aspect",
                        "log_n_dets_in_image", "within_image_rank",
                        "max_overlap_with_higher_scoring"],
           "models": {}}

    npos_ev = [int(np.sum(np.asarray(gt[i][1]) == 0)) for i in ev_ids]

    for m in models:
        dets = load_dets(os.path.join(a.detDir, f"{m}.json"))
        Xf, yf, _ = features_and_labels(gt, dets, fit_ids)
        Xe, ye, ie = features_and_labels(gt, dets, ev_ids)
        if len(Xf) == 0 or len(Xe) == 0 or yf.sum() == 0:
            print(f"  skip {m}: insufficient data"); continue

        mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((Xf - mu) / sd, yf)
        new = clf.predict_proba((Xe - mu) / sd)[:, 1]
        base = 1.0 / (1.0 + np.exp(-Xe[:, 0]))     # recover the raw score

        r_base = records_from(base, ye, ie, ev_ids)
        r_new = records_from(new, ye, ie, ev_ids)
        mb = score_set(gt, r_base, npos_ev, ev_ids)
        mn = score_set(gt, r_new, npos_ev, ev_ids)

        # oracle ceiling on the same evaluation half
        r_or = records_from(ye.astype(float), ye, ie, ev_ids)
        mo = score_set(gt, r_or, npos_ev, ev_ids)

        # control: same fitted model, features shuffled across detections
        idx = rng.permutation(len(Xe))
        ctrl = clf.predict_proba(((Xe[idx] - mu) / sd))[:, 1]
        r_ctrl = records_from(ctrl, ye, ie, ev_ids)
        mc = score_set(gt, r_ctrl, npos_ev, ev_ids)

        ceiling = mb["MR"] - mo["MR"]
        gained = mb["MR"] - mn["MR"]
        rec["models"][m] = {
            "MR_baseline": round(mb["MR"], 2),
            "MR_recalibrated": round(mn["MR"], 2),
            "MR_oracle_rank_ceiling": round(mo["MR"], 2),
            "MR_shuffled_control": round(mc["MR"], 2),
            "headroom_points": round(ceiling, 2),
            "recovered_points": round(gained, 2),
            "recovered_fraction_pct": round(gained / ceiling * 100, 1) if ceiling > 0 else None,
            "control_recovered_points": round(mb["MR"] - mc["MR"], 2),
            "AP50_baseline": round(mb["AP50"], 2),
            "AP50_recalibrated": round(mn["AP50"], 2),
            "coefficients": dict(zip(rec["features"],
                                     [round(float(c), 3) for c in clf.coef_[0]])),
        }
        e = rec["models"][m]
        print(f"  {m:<26} MR {e['MR_baseline']:>6.2f} -> {e['MR_recalibrated']:>6.2f}"
              f"   oracle {e['MR_oracle_rank_ceiling']:>5.2f}"
              f"   headroom {e['headroom_points']:>5.2f}"
              f"   recovered {e['recovered_points']:>5.2f} ({e['recovered_fraction_pct']:>4.1f}%)"
              f"   shuffled control {e['control_recovered_points']:>7.2f}")

    vals = [v["recovered_fraction_pct"] for v in rec["models"].values()
            if v["recovered_fraction_pct"] is not None]
    ctrls = [v["control_recovered_points"] for v in rec["models"].values()]
    rec["summary"] = {
        "median_recovered_fraction_pct": round(float(np.median(vals)), 1) if vals else None,
        "median_control_recovered_points": round(float(np.median(ctrls)), 2),
        "interpretation": ("recovered_fraction is the share of the ORACLE-RANK "
                           "headroom obtained by rescoring alone, with no change "
                           "to the detector and no extra inference; the shuffled "
                           "control should recover nothing, and a large control "
                           "value would indicate the gain is a re-ranking artefact"),
    }
    print(f"\nmedian recovered: {rec['summary']['median_recovered_fraction_pct']}% "
          f"of the ranking headroom")
    print(f"median control  : {rec['summary']['median_control_recovered_points']} points "
          f"(should be near zero or negative)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
