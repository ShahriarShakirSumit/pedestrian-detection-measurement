"""Rank stability: how far a benchmark ranking transfers across evaluation
protocols.

Whether detector comparisons made under different protocols can be compared
directly is usually asserted rather than measured. This script measures it
from the outputs of rq2_protocol_sensitivity.py, one file per eligibility
setting (for example, with and without a height floor).

Three quantities:

1. Rank agreement. Kendall's tau-b between the rankings that each pair of
   protocols induces over the same models on the same detections. A tau of 1
   means the two protocols order the models identically and the choice
   between them is immaterial. A tau below 1 means a conclusion depends on
   which protocol was adopted.

2. Concordance. Kendall's W over all protocols at once, because agreement of
   the protocol set as a whole is not the average of the pairwise taus.

3. Protocol translation. A linear fit between scores under different
   protocols across the model set, so a number reported under one protocol
   can be placed approximately on another scale. Reported with the
   correlation and the largest absolute residual, because a single conversion
   factor hides how far individual models depart from the fit.
"""
import argparse, itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statistics as ST

LOWER_BETTER = {"MR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq2", nargs="+", required=True,
                    help="one or more rq2 result JSONs (different eligibility settings)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rec = {"inputs": a.rq2, "settings": {}}

    for path in a.rq2:
        d = json.load(open(path))
        tag = os.path.basename(path).replace(".json", "")
        models = sorted(d["results"])
        prots = [p for p in ("COCO", "COCO@.5", "COCO_matchedBudget",
                             "VOC-all", "VOC-11", "MR") if p in d["results"][models[0]]]

        ranks = {}
        for p in prots:
            rev = p not in LOWER_BETTER
            order = sorted(models, key=lambda m: d["results"][m][p], reverse=rev)
            ranks[p] = {m: i + 1 for i, m in enumerate(order)}

        taus = {}
        for p, q in itertools.combinations(prots, 2):
            t, con, dis = ST.kendall_tau(ranks[p], ranks[q])
            taus[f"{p} vs {q}"] = {"tau_b": round(t, 4),
                                   "concordant": con, "discordant": dis,
                                   "identical_ranking": bool(abs(t - 1.0) < 1e-9)}

        # Protocol translation: linear fit of one protocol's scores on another's
        trans = {}
        for p, q in itertools.combinations(prots, 2):
            xs = np.array([d["results"][m][p] for m in models], float)
            ys = np.array([d["results"][m][q] for m in models], float)
            if len(models) >= 3 and xs.std() > 0 and ys.std() > 0:
                r = float(np.corrcoef(xs, ys)[0, 1])
                slope, intercept = np.polyfit(xs, ys, 1)
                pred = slope * xs + intercept
                resid = ys - pred
                trans[f"{p} -> {q}"] = {
                    "pearson_r": round(r, 4),
                    "slope": round(float(slope), 4),
                    "intercept": round(float(intercept), 3),
                    "max_abs_residual": round(float(np.abs(resid).max()), 2),
                    "note": ("a linear map across only "
                             f"{len(models)} models; the residual is the error a "
                             "reader would make converting one reported number")}

        # Pairwise tau answers whether two protocols agree. Agreement of the
        # protocol set as a whole is a different quantity, and an average of
        # pairwise taus does not measure it, so concordance is computed once
        # over all protocols together.
        concord = ST.kendalls_w(ranks)

        n_distinct = len({tuple(sorted(r.items())) for r in ranks.values()})
        rec["settings"][tag] = {
            "min_height": d.get("min_height", 0),
            "models": models, "protocols": prots,
            "scores": {m: {p: d["results"][m][p] for p in prots} for m in models},
            "ranks": ranks,
            "n_distinct_rankings": n_distinct,
            "kendall_tau": taus,
            "kendalls_w": concord,
            "translation": trans,
            "worst_agreement": min(taus, key=lambda k: taus[k]["tau_b"]) if taus else None,
        }

        print(f"\n########## {tag} (height floor {d.get('min_height', 0)}) ##########")
        print(f"  {'model':<26}" + "".join(f"{p:>20}" for p in prots))
        for m in models:
            print(f"  {m:<26}" + "".join(f"{d['results'][m][p]:>20.2f}" for p in prots))
        print(f"\n  rank per protocol (1 = best):")
        print(f"  {'model':<26}" + "".join(f"{p:>20}" for p in prots))
        for m in models:
            print(f"  {m:<26}" + "".join(f"{ranks[p][m]:>20}" for p in prots))
        print(f"\n  distinct rankings produced by {len(prots)} protocols: {n_distinct}")
        print(f"  Kendall's W across all {len(prots)} protocols: "
              f"{concord['W']} ({concord['interpretation']}), "
              f"Friedman chi2={concord['friedman_chi2']} "
              f"df={concord['df']} p={concord['p_value']:.3g}")
        print(f"  Kendall tau-b between protocol rankings:")
        for k, v in sorted(taus.items(), key=lambda kv: kv[1]["tau_b"]):
            flag = "" if v["identical_ranking"] else "   <-- disagreement"
            print(f"    {k:<44} tau={v['tau_b']:>7.3f}{flag}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
