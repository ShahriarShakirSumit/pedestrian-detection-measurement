"""Inversion threshold: how large a performance gap must be before a detector
ordering is protocol-safe.

The question is whether there is a margin above which a published ordering
holds regardless of protocol, and below which it may not.

Every observed pair is pooled across all protocol variants, all evaluation
subsets and both datasets (COCO person and KAIST). The inputs are the outputs
of rq2c_protocol_axes.py (protocol axes), rq2b_rank_stability.py (metric
families), rq1_degradation.py (data-condition subsets) and
kaist_eval_published.py (KAIST subsets). Gaps are in miss-rate points under
each source's reference setting. For each pair and variant the reference gap
is recorded together with whether the pair inverted. The largest reference gap
that still inverted is the empirical threshold below which an ordering is not
protocol-safe on this evidence.

This is a descriptive threshold from a limited model set, not a law. It is
reported with the number of pairs supporting it so a reader can judge its
weight.
"""
import argparse, itertools, json, os, sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statistics as ST


def collect(rec_axes, rec_rank, rec_rq1, rec_kaist):
    """Return a list of (source, pair, reference_gap, variant, variant_gap,
    inverted) tuples."""
    out = []

    # protocol axes: IoU, FPPI window, height floor, crowd policy
    ref = rec_axes["reference"]["scores"]
    for label, v in rec_axes["variants"].items():
        for x, y in itertools.combinations(sorted(ref), 2):
            gap = abs(ref[x] - ref[y])
            vgap = abs(v["scores"][x] - v["scores"][y])
            inv = (ref[x] < ref[y]) != (v["scores"][x] < v["scores"][y])
            out.append(("protocol_axis", f"{x}|{y}", gap, label, vgap, inv))

    # metric families, with the miss rate as the reference
    for tag, s in rec_rank["settings"].items():
        prots = s["protocols"]
        base = "MR"
        for p in prots:
            if p == base:
                continue
            for x, y in itertools.combinations(sorted(s["scores"]), 2):
                bx, by = s["scores"][x][base], s["scores"][y][base]
                px, py = s["scores"][x][p], s["scores"][y][p]
                # MR is lower-better, AP higher-better: normalise direction
                better_base = bx < by
                better_p = px > py if p != base else px < py
                out.append(("metric_family", f"{x}|{y}", abs(bx - by),
                            f"{tag}:{p}", abs(px - py), better_base != better_p))

    # data conditions (scale, crowding, occlusion) against the overall result
    overall = rec_rq1["overall"]
    for axis, per_model in rec_rq1["axes"].items():
        bins = list(next(iter(per_model.values())).keys())
        for b in bins:
            for x, y in itertools.combinations(sorted(overall), 2):
                gap = abs(overall[x]["MR"] - overall[y]["MR"])
                vx, vy = per_model[x][b]["MR"], per_model[y][b]["MR"]
                inv = (overall[x]["MR"] < overall[y]["MR"]) != (vx < vy)
                out.append((f"condition_{axis}", f"{x}|{y}", gap,
                            f"{axis}={b}", abs(vx - vy), inv))

    # KAIST subsets against the all-day Reasonable setting
    kr = rec_kaist["results"]
    base_key = "MR_all_Reasonable"
    for sub in ("MR_all_Reasonable_small", "MR_all_Reasonable_occ=heavy",
                "MR_all_All", "MR_day_Reasonable", "MR_night_Reasonable"):
        for x, y in itertools.combinations(sorted(kr), 2):
            gap = abs(kr[x][base_key] - kr[y][base_key])
            vx, vy = kr[x][sub], kr[y][sub]
            inv = (kr[x][base_key] < kr[y][base_key]) != (vx < vy)
            out.append(("kaist_subset", f"{x}|{y}", gap, sub, abs(vx - vy), inv))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("axes", "rank", "rq1", "kaist"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    obs = collect(json.load(open(a.axes)), json.load(open(a.rank)),
                  json.load(open(a.rq1)), json.load(open(a.kaist)))

    inverted = [o for o in obs if o[5]]
    stable = [o for o in obs if not o[5]]
    gaps_inv = np.array([o[2] for o in inverted]) if inverted else np.array([])
    gaps_stab = np.array([o[2] for o in stable])

    rec = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"axes": a.axes, "rank": a.rank, "rq1": a.rq1, "kaist": a.kaist},
        "n_observations": len(obs),
        "n_inversions": len(inverted),
        "inversion_rate_pct": round(len(inverted) / len(obs) * 100, 2),
        "reference_gap_of_inverted_pairs": {
            "n": len(gaps_inv),
            "min": round(float(gaps_inv.min()), 2) if len(gaps_inv) else None,
            "median": round(float(np.median(gaps_inv)), 2) if len(gaps_inv) else None,
            "max": round(float(gaps_inv.max()), 2) if len(gaps_inv) else None,
        },
        "reference_gap_of_stable_pairs": {
            "n": len(gaps_stab),
            "min": round(float(gaps_stab.min()), 2),
            "median": round(float(np.median(gaps_stab)), 2),
            "max": round(float(gaps_stab.max()), 2),
        },
        "by_source": {}, "inversions": [],
    }

    for src in sorted({o[0] for o in obs}):
        sub = [o for o in obs if o[0] == src]
        si = [o for o in sub if o[5]]
        rec["by_source"][src] = {
            "n": len(sub), "n_inverted": len(si),
            "rate_pct": round(len(si) / len(sub) * 100, 2),
            "max_inverted_gap": round(max([o[2] for o in si]), 2) if si else None}

    for o in sorted(inverted, key=lambda z: -z[2]):
        rec["inversions"].append({
            "source": o[0], "pair": o[1], "reference_gap": round(o[2], 2),
            "variant": o[3], "gap_within_variant": round(o[4], 2)})

    # empirical threshold: largest reference gap that still inverted
    thr = float(gaps_inv.max()) if len(gaps_inv) else 0.0
    above = [o for o in obs if o[2] > thr]
    rec["empirical_threshold"] = {
        "largest_inverting_reference_gap": round(thr, 2),
        "n_pairs_above_threshold": len(above),
        "n_inversions_above_threshold": sum(1 for o in above if o[5]),
        "statement": (f"No pair separated by more than {thr:.2f} miss-rate points "
                      f"in the reference protocol ever inverted, across "
                      f"{len(obs)} pair-by-variant observations."),
    }

    print(f"observations: {rec['n_observations']}   inversions: {rec['n_inversions']} "
          f"({rec['inversion_rate_pct']}%)")
    print(f"\nreference gap of pairs that inverted : "
          f"min {rec['reference_gap_of_inverted_pairs']['min']}, "
          f"median {rec['reference_gap_of_inverted_pairs']['median']}, "
          f"max {rec['reference_gap_of_inverted_pairs']['max']}")
    print(f"reference gap of pairs that never did: "
          f"min {rec['reference_gap_of_stable_pairs']['min']}, "
          f"median {rec['reference_gap_of_stable_pairs']['median']}, "
          f"max {rec['reference_gap_of_stable_pairs']['max']}")
    print(f"\nby source:")
    for k, v in rec["by_source"].items():
        print(f"  {k:<26} {v['n_inverted']:>3}/{v['n']:<5} "
              f"({v['rate_pct']:>5.1f}%)  largest inverting gap: {v['max_inverted_gap']}")
    print(f"\n{rec['empirical_threshold']['statement']}")
    print(f"\nall inversions:")
    for i in rec["inversions"]:
        print(f"  gap {i['reference_gap']:>6.2f}  {i['source']:<22} "
              f"{i['variant']:<34} {i['pair']}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
