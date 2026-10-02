"""Evidence-based detector selection.

Turns the measured accuracy and cost results into a selection procedure:

1. Pareto frontier. Which detectors are non-dominated on accuracy against each
   deployment cost, so that the choice set is reduced to models that are not
   strictly worse than an alternative on both axes.
2. Condition-specific winners. Which detector leads under each operating
   condition separately, because the aggregate winner need not be the winner
   under the conditions where a deployment actually operates.
3. Constraint-driven recommendation. Given a memory ceiling, a latency budget
   or the dominant difficulty of the scene, which model the measured evidence
   supports.

Inputs are the JSON outputs of rq1_degradation.py (--rq1, per-condition miss
rates), rq2_protocol_sensitivity.py (--rq2, overall miss rate) and
rq4_timing.py (--rq4, latency and memory). Each recommendation records its
basis in those results, and each condition winner is reported with its
runner-up and margin, so that cases where the evidence barely separates two
models remain visible.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statistics as ST


def pareto_front(points, better):
    """`points` is {name: (accuracy, cost)}. `better` says whether higher
    accuracy is better. Returns the non-dominated names."""
    names = list(points)
    front = []
    for n in names:
        a, c = points[n]
        dominated = False
        for m in names:
            if m == n:
                continue
            a2, c2 = points[m]
            acc_ge = (a2 >= a) if better else (a2 <= a)
            acc_gt = (a2 > a) if better else (a2 < a)
            if acc_ge and c2 <= c and (acc_gt or c2 < c):
                dominated = True
                break
        if not dominated:
            front.append(n)
    return sorted(front, key=lambda n: points[n][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1", required=True)
    ap.add_argument("--rq2", required=True)
    ap.add_argument("--rq4", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rq1 = json.load(open(a.rq1))
    rq2 = json.load(open(a.rq2))
    rq4 = json.load(open(a.rq4))

    models = sorted(rq2["results"])
    cost = {}
    for k, v in rq4["results"].items():
        if v["precision"] != "fp32":
            continue
        cost[v["model"]] = {"latency_ms": v["total_ms_median"],
                            "latency_p95": v["total_ms_p95"],
                            "throughput": v["throughput_img_s"],
                            "peak_MiB": v["peak_alloc_MiB"],
                            "params_M": v["params_M"]}
    for k, v in rq4["results"].items():
        if v["precision"] == "fp16" and v["model"] in cost:
            cost[v["model"]]["latency_ms_fp16"] = v["total_ms_median"]
            cost[v["model"]]["peak_MiB_fp16"] = v["peak_alloc_MiB"]

    rec = {
        "note": ("Accuracy is log-average miss rate, lower is better, from the "
                 "locked evaluator. Cost is measured on one RTX 3070 at batch "
                 "size 1 and applies to that system only."),
        "models": models, "cost": cost, "pareto": {}, "condition_winners": {},
        "recommendations": [], "evidence": {"rq1": a.rq1, "rq2": a.rq2, "rq4": a.rq4},
    }

    acc = {m: rq2["results"][m]["MR"] for m in models if m in cost}

    for axis, key in (("latency_ms", "latency_ms"), ("peak_MiB", "peak_MiB")):
        pts = {m: (acc[m], cost[m][key]) for m in acc}
        front = pareto_front(pts, better=False)      # lower miss rate is better
        rec["pareto"][axis] = {
            "frontier": front,
            "dominated": sorted(set(pts) - set(front)),
            "points": {m: {"MR": pts[m][0], axis: pts[m][1]} for m in pts},
        }
        print(f"\n=== Pareto frontier: miss rate vs {axis} ===")
        for m in front:
            print(f"   ON FRONTIER  {m:<26} MR={pts[m][0]:>6.2f}  {axis}={pts[m][1]:>8.1f}")
        for m in sorted(set(pts) - set(front)):
            print(f"   dominated    {m:<26} MR={pts[m][0]:>6.2f}  {axis}={pts[m][1]:>8.1f}")

    print("\n=== Condition-specific winners (lowest miss rate) ===")
    for axis, per_model in rq1["axes"].items():
        bins = list(next(iter(per_model.values())).keys())
        rec["condition_winners"][axis] = {}
        for b in bins:
            vals = {m: per_model[m][b]["MR"] for m in per_model
                    if per_model[m][b]["n"] > 0}
            if not vals:
                continue
            win = min(vals, key=vals.get)
            runner = sorted(vals, key=vals.get)[1] if len(vals) > 1 else None
            rec["condition_winners"][axis][b] = {
                "winner": win, "MR": vals[win],
                "runner_up": runner,
                "margin": round(vals[runner] - vals[win], 2) if runner else None,
                "n_objects": per_model[win][b]["n"]}
            m = rec["condition_winners"][axis][b]
            print(f"  {axis:<18}{b:<12} -> {win:<26} MR={vals[win]:>6.2f} "
                  f"(margin {m['margin']})")

    print("\n=== Constraint-driven recommendation ===")
    scenarios = [
        ("Real-time on 8 GB, <25 ms", lambda m: cost[m]["latency_ms"] < 25),
        ("Embedded, <500 MiB peak", lambda m: cost[m]["peak_MiB"] < 500),
        ("Accuracy first, no constraint", lambda m: True),
        ("Small/distant people dominate", None),
        ("Crowded scenes dominate", None),
    ]
    for label, filt in scenarios:
        if filt is None:
            axis = "height" if "Small" in label else "crowding"
            bins = rec["condition_winners"].get(axis, {})
            key = ("h<25" if "Small" in label else "10+")
            if key in bins:
                w = bins[key]
                rec["recommendations"].append({
                    "scenario": label, "choice": w["winner"],
                    "basis": f"lowest miss rate in the {axis} bin '{key}' "
                             f"({w['MR']}, margin {w['margin']} over {w['runner_up']})"})
                print(f"  {label:<32} -> {w['winner']}  ({w['MR']} MR in bin {key})")
            continue
        elig = [m for m in acc if filt(m)]
        if not elig:
            rec["recommendations"].append({"scenario": label, "choice": None,
                                           "basis": "no measured model satisfies this constraint"})
            print(f"  {label:<32} -> none satisfy the constraint")
            continue
        best = min(elig, key=lambda m: acc[m])
        rec["recommendations"].append({
            "scenario": label, "choice": best,
            "basis": f"lowest miss rate ({acc[best]}) among "
                     f"{len(elig)} models meeting the constraint",
            "eligible": sorted(elig)})
        print(f"  {label:<32} -> {best}  (MR {acc[best]}, {len(elig)} eligible)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
