"""Record the per-frame outcomes shown in the qualitative track figure.

The figure shows two detectors that match one NightOwls pedestrian track in the
same number of frames but arrange their misses differently, so their longest
miss runs differ. This script measures that from the detection files, at a
score threshold of 0.5 and IoU 0.5, and writes it to a result file that the
plotting script (gen_qualitative.py) reads, so no measured value is computed
inside the plotting code.

It also repeats the pairwise enumeration of same_k_diff_r.py (detector pairs
and tracks with equal miss counts, and how many of those differ in longest
miss run). Agreement with the counts stored by that script confirms that both
score tracks identically, and hence that the selected track is one of the
enumerated cases.
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nightowls_eval as NE

TRACK, PAIR = 7000220, ("yolo12s", "yolo26s")
SCORE_THR, IOU = 0.5, 0.5


def longest_run(seq):
    lo = cur = 0
    for v in seq:
        cur = 0 if v else cur + 1
        lo = max(lo, cur)
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ann = json.load(open(a.annFile))
    gt, ev, _ = NE.build_gt(ann)
    ids = sorted(gt)
    tracks = {t: sorted(v, key=lambda z: z[0]) for t, v in ev.items() if len(v) >= 2}
    gt_index = {}
    for im in ids:
        gb, gi = gt[im]
        for j, (b, ig) in enumerate(zip(gb, gi)):
            gt_index[(im, tuple(b))] = j
    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))

    per = {}
    for m in models:
        found = NE.matched_gt_per_image(gt, NE.load_dets(os.path.join(a.detDir, f"{m}.json")),
                                        ids, SCORE_THR, IOU)
        per[m] = {t: [bool(found.get((im, gt_index.get((im, tuple(an["bbox"])))), False))
                      for im, an in fr] for t, fr in tracks.items()}

    same_k = diff_r = maxdiv = 0
    for x, y in itertools.combinations(models, 2):
        for t in tracks:
            sx, sy = per[x][t], per[y][t]
            if sx.count(False) != sy.count(False):
                continue
            same_k += 1
            rx, ry = longest_run(sx), longest_run(sy)
            if rx != ry:
                diff_r += 1
                maxdiv = max(maxdiv, abs(rx - ry))

    byid = {i["id"]: i for i in ann["images"]}
    A, B = PAIR
    frames = [{"image_id": im, "file_name": byid[im]["file_name"], "bbox": an["bbox"],
               A: per[A][TRACK][k], B: per[B][TRACK][k]}
              for k, (im, an) in enumerate(tracks[TRACK])]
    rec = {
        "track_id": TRACK, "detectors": list(PAIR),
        "operating_point": {"score": SCORE_THR, "iou": IOU},
        "n_frames": len(frames),
        "hits": {A: per[A][TRACK].count(True), B: per[B][TRACK].count(True)},
        "longest_miss_run": {A: longest_run(per[A][TRACK]), B: longest_run(per[B][TRACK])},
        "equal_miss_count": per[A][TRACK].count(False) == per[B][TRACK].count(False),
        "enumeration_reproduced": {"pairs_same_k": same_k, "of_which_diff_r": diff_r,
                                   "max_divergence_frames": maxdiv},
        "frames": frames,
    }
    assert rec["equal_miss_count"], "the figure's premise does not hold"
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"track {TRACK}: {A} hits {rec['hits'][A]} run {rec['longest_miss_run'][A]}, "
          f"{B} hits {rec['hits'][B]} run {rec['longest_miss_run'][B]}")
    print(f"enumeration: same_k {same_k}, diff_r {diff_r}, max divergence {maxdiv}")


if __name__ == "__main__":
    main()
