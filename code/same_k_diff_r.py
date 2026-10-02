"""Count how often two detectors miss the same number of frames on a track yet
differ in their longest run of consecutive misses.

This is the empirical counterpart of the proposition on longest miss runs:
the per-track miss count does not determine the longest miss run. Every pair
of detectors is compared on every NightOwls track that is evaluable in at
least two frames, at a score threshold of 0.5 and IoU 0.5."""
import itertools, json, os, sys
import numpy as np
import paths  # repository-relative results, DATA_ROOT for datasets
E = paths.ROOT
sys.path.insert(0, paths.CODE)
import nightowls_eval as NE

ann = json.load(open(paths.data("nightowls", "annotations", "nightowls_validation.json")))
gt, evaluable, _ = NE.build_gt(ann)
image_ids = sorted(gt)
tracks = {t: sorted(v, key=lambda z: z[0]) for t, v in evaluable.items() if len(v) >= 2}
gt_index = {}
for im in image_ids:
    gb, gi = gt[im]
    for j, (b, ig) in enumerate(zip(gb, gi)):
        gt_index[(im, tuple(b))] = j

def longest_run(seq):
    lo = cur = 0
    for v in seq:
        cur = 0 if v else cur + 1
        lo = max(lo, cur)
    return lo

models = sorted(f[:-5] for f in os.listdir(f"{E}/results/nightowls/detections")
                if f.endswith(".json") and not f.endswith("_meta.json"))
print(f"{len(models)} detectors, {len(tracks)} tracks", flush=True)
per = {}
for n, m in enumerate(models):
    dets = NE.load_dets(f"{E}/results/nightowls/detections/{m}.json")
    found = NE.matched_gt_per_image(gt, dets, image_ids, 0.5)
    kr = {}
    for t, frames in tracks.items():
        seq = []
        for im, an in frames:
            j = gt_index.get((im, tuple(an["bbox"])))
            seq.append(bool(found.get((im, j), False)) if j is not None else False)
        kr[t] = (sum(1 for v in seq if not v), longest_run(seq), len(seq))
    per[m] = kr
    print(f"  {n+1}/{len(models)} {m}", flush=True)

same_k = diff_r = 0
worst = (0, None)
for t in tracks:
    for a, b in itertools.combinations(models, 2):
        ka, ra, _ = per[a][t]; kb, rb, _ = per[b][t]
        if ka == kb:
            same_k += 1
            if ra != rb:
                diff_r += 1
                if abs(ra - rb) > worst[0]:
                    worst = (abs(ra - rb), (t, a, b, ka, ra, rb))
print(f"\ndetector pairs x tracks with an identical miss count: {same_k:,}")
print(f"   of those, differing in longest miss run:            {diff_r:,} ({100*diff_r/same_k:.1f}%)")
d, w = worst
print(f"\nlargest divergence at equal miss count: {d} frames")
if w:
    t, a, b, k, ra, rb = w
    print(f"   track {t}: both missed {k} frames; {a} run {ra}, {b} run {rb}")
json.dump({"pairs_same_k": same_k, "of_which_diff_r": diff_r,
           "pct": round(100*diff_r/same_k, 1), "max_divergence_frames": d},
          open(f"{E}/results/nightowls/same_k_diff_r.json", "w"), indent=2)
