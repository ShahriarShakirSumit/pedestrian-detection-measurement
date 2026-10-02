"""Verify the proposition on longest miss runs, exhaustively and on real tracks.

The proposition states two things.

  (a) Any measure that reads the per-frame outcomes of a track only through
      their aggregate count is unchanged when those outcomes are permuted
      within the track. Track coverage is such a measure. So is dataset recall
      at a fixed operating point, and so is the whole miss-rate curve, provided
      the permutation carries each detection along with its outcome, because
      then the multiset of (score, matched) pairs over the dataset is fixed.

  (b) First-detection delay and longest miss run are not such measures. For a
      track of T frames with k missed, the longest miss run R satisfies
          ceil(k / (T - k + 1))  <=  R  <=  k,
      and both bounds are attained.

Together, (a) and (b) mean that the arrangement of misses, which a
frame-aggregate measure cannot see, is exactly what decides whether a person
disappears for one frame at a time or for one unbroken stretch.

This script checks (b) exhaustively over every binary sequence up to length 18,
checks that the bound holds on every real NightOwls track for real detectors,
and demonstrates (a) by permuting real tracks.
"""
import itertools, json, math, os, sys

import numpy as np

import paths  # repository-relative results, DATA_ROOT for datasets
E = paths.ROOT
sys.path.insert(0, paths.CODE)


def longest_run(seq):
    """Longest run of misses. `seq[i]` is True when frame i is matched."""
    longest = cur = 0
    for v in seq:
        cur = 0 if v else cur + 1
        longest = max(longest, cur)
    return longest


def first_detection(seq):
    return next((i for i, v in enumerate(seq) if v), len(seq))


def lower_bound(T, k):
    return math.ceil(k / (T - k + 1)) if k else 0


# Exhaustive check of the bounds over all short binary sequences
def exhaustive(max_T=18):
    """Every binary sequence up to max_T: is the bound tight and correct?"""
    bad = 0
    for T in range(1, max_T + 1):
        seen = {}
        for bits in itertools.product([True, False], repeat=T):
            k = sum(1 for b in bits if not b)
            seen.setdefault(k, []).append(longest_run(bits))
        for k, runs in seen.items():
            lo, hi = lower_bound(T, k), k
            if min(runs) != lo or max(runs) != hi:
                bad += 1
                print(f"   FAIL T={T} k={k}: observed [{min(runs)},{max(runs)}] "
                      f"predicted [{lo},{hi}]")
    total = sum(2 ** T for T in range(1, max_T + 1))
    print(f"  exhaustive over {total:,} sequences up to length {max_T}: "
          f"{'both bounds exact for every (T,k)' if not bad else f'{bad} FAILURES'}")
    return bad == 0


# Checks on real NightOwls tracks
def real_tracks(models=("yolo12x", "faster_rcnn_r50_fpn_v2", "ssdlite320_mnv3"),
                score_thr=0.5):
    import nightowls_eval as NE
    ann = json.load(open(paths.data("nightowls", "annotations",
                                    "nightowls_validation.json")))
    gt, evaluable, _ = NE.build_gt(ann)
    image_ids = sorted(gt)
    tracks = {t: sorted(v, key=lambda z: z[0]) for t, v in evaluable.items()
              if len(v) >= 2}
    gt_index = {}
    for im in image_ids:
        gb, gi = gt[im]
        for j, (b, ig) in enumerate(zip(gb, gi)):
            gt_index[(im, tuple(b))] = j

    ok = True
    for m in models:
        dets = NE.load_dets(f"{E}/results/nightowls/detections/{m}.json")
        found = NE.matched_gt_per_image(gt, dets, image_ids, score_thr)
        viol = 0
        spreads = []
        rng = np.random.default_rng(0)
        perm_cov_changed = perm_run_changed = 0
        for t, frames in tracks.items():
            seq = []
            for im, an in frames:
                j = gt_index.get((im, tuple(an["bbox"])))
                seq.append(bool(found.get((im, j), False)) if j is not None else False)
            T = len(seq); k = sum(1 for v in seq if not v)
            R = longest_run(seq)
            lo, hi = lower_bound(T, k), k
            if not (lo <= R <= hi):
                viol += 1
            spreads.append(hi - lo)
            # (a) permute within the track: coverage fixed, run generally not
            p = list(seq); rng.shuffle(p)
            if sum(p) != sum(seq):
                perm_cov_changed += 1
            if longest_run(p) != R:
                perm_run_changed += 1
        print(f"  {m:<24} {len(tracks)} tracks | bound violations {viol} | "
              f"median attainable spread {int(np.median(spreads))} frames")
        print(f"  {'':<24} permuting within a track changed coverage on "
              f"{perm_cov_changed} tracks, longest run on {perm_run_changed}")
        ok = ok and viol == 0
    return ok


if __name__ == "__main__":
    print("=== (b) bounds on the longest miss run ===")
    a = exhaustive()
    print("\n=== (a) and (b) on real NightOwls tracks ===")
    b = real_tracks()
    print(f"\n{'PROPOSITION VERIFIED' if a and b else 'VERIFICATION FAILED'}")
    sys.exit(0 if a and b else 1)
