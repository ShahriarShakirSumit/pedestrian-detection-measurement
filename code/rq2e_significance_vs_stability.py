"""Significance versus protocol stability: whether the detector pairs whose
ordering changes under a different evaluation protocol are the pairs whose
difference is not statistically significant.

Two separate analyses characterise each pair of detectors:

  - whether their ordering survives a change of evaluation protocol (the
    output of rq2d_inversion_threshold.py), and
  - whether their difference survives image-level resampling and a
    Holm-corrected significance test (the output of run_statistics.py).

These are computed by different machinery from different quantities. If they
identify the same pairs, a reader has a cheap decision rule: a difference that
is not statistically significant is also not protocol-safe, and the two
concerns collapse into one.

The comparison is a two-by-two contingency table over pairs, tested with
Fisher's exact test, which remains valid when some cell counts are small. The
full pair list is written as well, so a reader can see exactly which pairs
disagree rather than relying on a summary statistic.
"""
import argparse, json, math, os, sys
import time
from collections import defaultdict


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher's exact test on [[a, b], [c, d]]."""
    def logfact(n):
        return math.lgamma(n + 1)

    def p_table(a_, b_, c_, d_):
        return math.exp(logfact(a_ + b_) + logfact(c_ + d_) +
                        logfact(a_ + c_) + logfact(b_ + d_) -
                        logfact(a_ + b_ + c_ + d_) -
                        (logfact(a_) + logfact(b_) + logfact(c_) + logfact(d_)))

    n = a + b + c + d
    row1, col1 = a + b, a + c
    obs = p_table(a, b, c, d)
    total = 0.0
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    for x in range(lo, hi + 1):
        y = row1 - x
        z = col1 - x
        w = n - x - y - z
        p = p_table(x, y, z, w)
        if p <= obs * (1 + 1e-9):
            total += p
    return min(1.0, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inversions", required=True)
    ap.add_argument("--statistics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    inv = json.load(open(a.inversions))
    st = json.load(open(a.statistics))

    # Pairs that ever inverted, restricted to the COCO model set that the
    # statistics cover. KAIST methods have no bootstrap and are excluded.
    models = set(st["per_model"])
    inverted = defaultdict(list)
    for r in inv["inversions"]:
        x, y = r["pair"].split("|")
        if x in models and y in models:
            inverted[frozenset((x, y))].append(
                {"source": r["source"], "variant": r["variant"],
                 "gap_within_variant": r["gap_within_variant"]})

    rows = []
    for key, v in st["pairwise"].items():
        x, y = key.split(" vs ")
        pair = frozenset((x, y))
        holm_b = st["holm_bootstrap"]["results"][key]["reject_at_alpha"]
        holm_m = st["holm_mcnemar"]["results"][key]["reject_at_alpha"]
        rows.append({
            "pair": f"{x} vs {y}",
            "MR_diff": v["MR_diff_point"],
            "MR_diff_ci": v["MR_diff_ci"],
            "significant_bootstrap": bool(holm_b),
            "significant_mcnemar": bool(holm_m),
            "ever_inverted": pair in inverted,
            "n_inversions": len(inverted.get(pair, [])),
            "inversion_details": inverted.get(pair, [])[:6],
        })

    # contingency on the bootstrap test, because it concerns the aggregate metric
    a_ = sum(1 for r in rows if not r["significant_bootstrap"] and r["ever_inverted"])
    b_ = sum(1 for r in rows if not r["significant_bootstrap"] and not r["ever_inverted"])
    c_ = sum(1 for r in rows if r["significant_bootstrap"] and r["ever_inverted"])
    d_ = sum(1 for r in rows if r["significant_bootstrap"] and not r["ever_inverted"])
    p = fisher_exact_2x2(a_, b_, c_, d_)

    rec = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"inversions": a.inversions, "statistics": a.statistics},
        "n_pairs": len(rows),
        "contingency": {
            "not_significant_and_inverted": a_,
            "not_significant_and_stable": b_,
            "significant_and_inverted": c_,
            "significant_and_stable": d_,
        },
        # Not rounded. With many pairs p can fall far below 1e-6, and
        # round(p, 6) would store it as 0.0 and misreport it as "p = 0.0".
        # The float is kept and formatted at the point of display instead.
        "fisher_exact_p": float(p),
        "sensitivity_pct": round(a_ / max(a_ + b_, 1) * 100, 1),
        "specificity_pct": round(d_ / max(c_ + d_, 1) * 100, 1),
        "pairs": sorted(rows, key=lambda r: abs(r["MR_diff"])),
        "interpretation": (
            "not_significant_and_inverted are pairs where both lines of "
            "evidence agree the ordering is unreliable. "
            "significant_and_inverted are the interesting disagreements: a "
            "difference that survives resampling but not a protocol change."),
    }

    print(f"{'pair':<52}{'dMR':>7}{'signif':>9}{'inverted':>10}{'n':>4}")
    for r in rec["pairs"]:
        print(f"  {r['pair']:<50}{r['MR_diff']:>7.2f}"
              f"{'yes' if r['significant_bootstrap'] else 'NO':>9}"
              f"{'yes' if r['ever_inverted'] else 'no':>10}{r['n_inversions']:>4}")
    c = rec["contingency"]
    print(f"\n                     inverted   stable")
    print(f"  not significant  {c['not_significant_and_inverted']:>8} "
          f"{c['not_significant_and_stable']:>8}")
    print(f"  significant      {c['significant_and_inverted']:>8} "
          f"{c['significant_and_stable']:>8}")
    print(f"\nFisher exact p = {rec['fisher_exact_p']:.3g}")
    print(f"of the non-significant pairs, {rec['sensitivity_pct']}% ever inverted")
    print(f"of the significant pairs, {rec['specificity_pct']}% never inverted")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
