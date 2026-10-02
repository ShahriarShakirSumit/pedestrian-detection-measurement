"""NightOwls night-time pedestrian evaluation, with a track-level temporal
analysis.

Two analyses share one detection pass.

Frame-level. NightOwls follows the Caltech convention. The Reasonable subset
admits pedestrians at least 50 px tall that are not occluded. Everything else,
including the benchmark's own ignore category, riders and objects flagged
difficult, becomes an ignore entry rather than being deleted, so detections
firing on them are neither rewarded nor penalised. This matters more here than
on COCO because NightOwls annotates bicycle and motorbike riders separately
from pedestrians, and a person-class detector will fire on them.

Temporal. Frame-level metrics can conceal repeated misses on the same person.
NightOwls supplies validated `tracking_id` values, so such misses can be
measured per track. For each track that is evaluable in at least two frames,
the analysis reports the delay (in frames) before first detection, the longest
run of consecutive missed frames, and the percentage of tracks whose longest
miss run is at least --missRunThreshold frames.

A track is evaluable in a frame when its annotation would be admitted by the
Reasonable rule. The temporal operating point is a fixed score threshold
(--scoreThr): only detections scoring at least that value are considered, and
they are matched to admitted objects with the same greedy, score-ordered IoU
0.5 rule used for the frame-level metric, so the two analyses agree on what
counts as a detection.
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST

MIN_HEIGHT = 50.0


def build_gt(ann, min_height=MIN_HEIGHT):
    """Reasonable-subset ground truth. Returns per-image boxes and ignore flags,
    plus the annotation records needed for the temporal analysis."""
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    gt = {i["id"]: ([], []) for i in ann["images"]}
    evaluable = defaultdict(list)          # tracking_id -> [(image_id, ann)]
    for a in ann["annotations"]:
        name = cats.get(a["category_id"])
        x, y, w, h = a["bbox"]
        admitted = (name == "pedestrian"
                    and h >= min_height
                    and not a.get("occluded")
                    and not a.get("difficult")
                    and not a.get("ignore"))
        gt[a["image_id"]][0].append([x, y, w, h])
        gt[a["image_id"]][1].append(0 if admitted else 1)
        if admitted and a.get("tracking_id") is not None:
            evaluable[a["tracking_id"]].append((a["image_id"], a))
    return gt, evaluable, cats


def load_dets(path):
    by = defaultdict(lambda: ([], []))
    for d in json.load(open(path)):
        by[d["image_id"]][0].append(d["bbox"])
        by[d["image_id"]][1].append(d["score"])
    return by


def per_image_records(gt, dets, image_ids, iou=0.5, max_det=1000):
    recs = []
    for im in image_ids:
        gb, gi = gt[im]
        db, ds = dets.get(im, ([], []))
        if not db:
            recs.append(np.zeros((0, 2))); continue
        labels, order = P._match_image(db, ds, gb, gi, iou, max_det)
        s = np.asarray(ds, float)[order]
        keep = labels >= 0
        recs.append(np.column_stack([s[keep], labels[keep]]))
    return recs


def matched_gt_per_image(gt, dets, image_ids, score_thr, iou=0.5):
    """Which admitted ground-truth boxes were detected at a fixed operating
    point. Keyed by (image_id, index of the box within that image), which
    main() joins to tracks through an (image_id, box) lookup."""
    found = {}
    for im in image_ids:
        gb, gi = gt[im]
        db, ds = dets.get(im, ([], []))
        keepd = [i for i, s in enumerate(ds) if s >= score_thr]
        db = [db[i] for i in keepd]; ds = [ds[i] for i in keepd]
        real_idx = [j for j, v in enumerate(gi) if v == 0]
        for j in real_idx:
            found[(im, j)] = False
        if not db or not real_idx:
            continue
        order = np.argsort(-np.asarray(ds, float), kind="mergesort")
        sb = [db[i] for i in order]
        gig = np.asarray(gi, dtype=int)
        gidx = np.argsort(gig, kind="mergesort")
        gts = [gb[i] for i in gidx]
        ov = P.overlap_matrix(sb, gts, gig[gidx])
        taken = set()
        for di in range(len(sb)):
            best, bg = iou, -1
            for gj in range(len(gts)):
                if gj in taken or gig[gidx][gj] == 1:
                    continue
                if ov[di, gj] >= best:
                    best, bg = ov[di, gj], gj
            if bg >= 0:
                taken.add(bg)
                found[(im, int(gidx[bg]))] = True
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scoreThr", type=float, default=0.5,
                    help="operating point for the temporal analysis")
    ap.add_argument("--missRunThreshold", type=int, default=5)
    ap.add_argument("--nBoot", type=int, default=500)
    a = ap.parse_args()

    ann = json.load(open(a.annFile))
    gt, evaluable, cats = build_gt(ann)
    image_ids = sorted(gt)
    npos_per_image = [int(np.sum(np.asarray(gt[i][1]) == 0)) for i in image_ids]
    n_admitted = sum(npos_per_image)
    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))

    # order each track by image id, which is monotone in capture order here
    tracks = {t: sorted(v, key=lambda z: z[0]) for t, v in evaluable.items()
              if len(v) >= 2}

    rec = {"dataset": "NightOwls validation", "illumination": "night",
           "subset": f"Reasonable: pedestrian, height >= {MIN_HEIGHT:.0f} px, "
                     f"not occluded, not difficult, not ignore",
           "n_images": len(image_ids), "n_admitted_pedestrians": n_admitted,
           "n_tracks_evaluable_2plus": len(tracks),
           "temporal_operating_point": a.scoreThr,
           "miss_run_threshold_frames": a.missRunThreshold,
           "frame_level": {}, "temporal": {}}
    print(f"images {len(image_ids)}  admitted pedestrians {n_admitted}  "
          f"tracks(>=2 frames) {len(tracks)}")

    sets = {}
    for m in models:
        dets = load_dets(os.path.join(a.detDir, f"{m}.json"))
        recs = per_image_records(gt, dets, image_ids)
        sets[m] = ST.BootstrapSet(recs, npos_per_image, image_ids)
        mt = sets[m].metrics()
        rec["frame_level"][m] = {"MR": round(mt["MR"], 2),
                                 "AP50": round(mt["AP50"], 2),
                                 "recall": round(mt["recall"], 2)}

        found = matched_gt_per_image(gt, dets, image_ids, a.scoreThr)
        # map each track's frames to their ground-truth index within the image
        gt_index = {}
        for im in image_ids:
            gb, gi = gt[im]
            for j, (b, ig) in enumerate(zip(gb, gi)):
                gt_index[(im, tuple(b))] = j
        delays, runs, frac_flag = [], [], 0
        for t, frames in tracks.items():
            seq = []
            for im, an in frames:
                j = gt_index.get((im, tuple(an["bbox"])))
                seq.append(bool(found.get((im, j), False)) if j is not None else False)
            first = next((k for k, v in enumerate(seq) if v), None)
            delays.append(first if first is not None else len(seq))
            longest, cur = 0, 0
            for v in seq:
                cur = 0 if v else cur + 1
                longest = max(longest, cur)
            runs.append(longest)
            if longest >= a.missRunThreshold:
                frac_flag += 1
        delays = np.array(delays); runs = np.array(runs)
        rec["temporal"][m] = {
            "n_tracks": len(tracks),
            "first_detection_delay_frames": {
                "median": float(np.median(delays)),
                "p90": float(np.percentile(delays, 90)),
                "never_detected": int((delays == np.array([len(tracks[t]) for t in tracks])).sum()),
            },
            "longest_miss_run_frames": {
                "median": float(np.median(runs)), "p90": float(np.percentile(runs, 90)),
                "max": int(runs.max()),
            },
            "tracks_with_miss_run_ge_threshold_pct": round(frac_flag / len(tracks) * 100, 2),
        }
        t = rec["temporal"][m]
        print(f"  {m:<26} MR={rec['frame_level'][m]['MR']:>6.2f} "
              f"recall={rec['frame_level'][m]['recall']:>6.2f} | "
              f"median miss run={t['longest_miss_run_frames']['median']:>5.1f} "
              f"p90={t['longest_miss_run_frames']['p90']:>5.1f} "
              f"tracks>= {a.missRunThreshold}f: {t['tracks_with_miss_run_ge_threshold_pct']:>5.1f}%")

    print(f"\nbootstrap ({a.nBoot} replicates, image-level) ...")
    boot = ST.bootstrap(sets, n_boot=a.nBoot, seed=20260822)
    boot.pop("replicates", None)
    rec["bootstrap"] = {"per_model": boot["per_model"], "n_boot": a.nBoot,
                        "resampling_unit": "image",
                        "note": ("images within a NightOwls recording are not "
                                 "independent; this interval understates "
                                 "uncertainty and a sequence-level unit would "
                                 "be preferable where recording ids permit it")}
    for m in models:
        v = boot["per_model"][m]["MR"]
        print(f"  {m:<26} MR {v['point']:>6.2f}  [{v['ci_low']:>6.2f}, {v['ci_high']:>6.2f}]")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
