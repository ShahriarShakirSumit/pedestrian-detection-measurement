"""Uncertainty and significance for the COCO-person model comparison.

Produces, for every model and every pair of models:
  - point estimate with a paired image-level bootstrap confidence interval
  - paired difference with its own interval and a bootstrap p-value
  - McNemar's test on per-object detection outcomes, exact binomial when the
    discordant objects are few
  - Cliff's delta as a nonparametric effect size
  - Holm step-down and Benjamini-Hochberg step-up corrections across the whole
    family of pairwise tests

The per-object detection outcomes are read from the detection matrix written
by oracle_analysis.py.

Two tests are reported for each pair because they ask different questions.
The bootstrap asks whether the aggregate metric difference would survive
resampling the images. McNemar asks whether the two models disagree
systematically about which individual people they find. A pair can be
significant on one and not the other, and that disagreement is informative
rather than a problem to be resolved.
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST

PERSON = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtFile", required=True)
    ap.add_argument("--detDir", required=True)
    ap.add_argument("--detectionMatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nBoot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    ann = json.load(open(a.gtFile))
    gt = {i["id"]: ([], []) for i in ann["images"]}
    for x in ann["annotations"]:
        if x["category_id"] != PERSON:
            continue
        gt[x["image_id"]][0].append(x["bbox"])
        gt[x["image_id"]][1].append(1 if x.get("iscrowd") else 0)

    image_ids = list(gt)
    npos_per_image = [int(np.sum(np.asarray(gt[i][1]) == 0)) for i in image_ids]
    models = sorted(f[:-5] for f in os.listdir(a.detDir)
                    if f.endswith(".json") and not f.endswith("_meta.json"))

    sets = {}
    for m in models:
        by = defaultdict(lambda: ([], []))
        for d in json.load(open(os.path.join(a.detDir, f"{m}.json"))):
            if d["category_id"] == PERSON:
                by[d["image_id"]][0].append(d["bbox"])
                by[d["image_id"]][1].append(d["score"])
        per_image = []
        for img in image_ids:
            gb, gi = gt[img]
            db, ds = by.get(img, ([], []))
            if not db:
                per_image.append(np.zeros((0, 2))); continue
            labels, order = P._match_image(db, ds, gb, gi, 0.5, 1000)
            s = np.asarray(ds, float)[order]
            keep = labels >= 0
            per_image.append(np.column_stack([s[keep], labels[keep]]))
        sets[m] = ST.BootstrapSet(per_image, npos_per_image, image_ids)
        print(f"  indexed {m}")

    print(f"\nrunning {a.nBoot} paired bootstrap replicates ...")
    boot = ST.bootstrap(sets, n_boot=a.nBoot, seed=a.seed)

    # Replicate series come from the single bootstrap run above. Running a
    # second identical loop just to obtain them would double the cost.
    reps = {m: boot["replicates"][m]["MR"] for m in models}
    n_images = len(image_ids)

    mat = json.load(open(a.detectionMatrix))
    det = {m: np.array(v, dtype=bool) for m, v in mat["detected"].items()}

    boot.pop("replicates", None)
    rec = {"n_boot": a.nBoot, "seed": a.seed, "resampling_unit": "image",
           "n_images": n_images, "n_objects": len(mat["object_ids"]),
           "per_model": boot["per_model"], "pairwise": {}}

    raw_boot, raw_mcn = {}, {}
    for i, x in enumerate(models):
        for y in models[i + 1:]:
            key = f"{x} vs {y}"
            d = np.array(reps[x]) - np.array(reps[y])
            bp = ST.bootstrap_pvalue(d)
            mc = ST.mcnemar(det[x], det[y])
            cd = ST.cliffs_delta(reps[x], reps[y])
            rec["pairwise"][key] = {
                "MR_diff_point": round(float(sets[x].metrics()["MR"]
                                             - sets[y].metrics()["MR"]), 2),
                "MR_diff_ci": [boot["pairwise"][key]["MR"]["ci_low"],
                               boot["pairwise"][key]["MR"]["ci_high"]],
                "ci_excludes_zero": boot["pairwise"][key]["MR"]["excludes_zero"],
                "bootstrap_p": bp, "mcnemar": mc, "cliffs_delta": cd}
            raw_boot[key] = bp["p_value"]
            raw_mcn[key] = mc["p_value"]

    rec["holm_bootstrap"] = ST.holm_bonferroni(raw_boot)
    rec["holm_mcnemar"] = ST.holm_bonferroni(raw_mcn)
    # Holm bounds the chance of a single false rejection. With 46 detectors the
    # family holds 1035 comparisons and that bound removes most of the power,
    # so the false-discovery view is reported alongside it and the paper
    # states which correction each claim uses.
    rec["bh_bootstrap"] = ST.benjamini_hochberg(raw_boot)
    rec["bh_mcnemar"] = ST.benjamini_hochberg(raw_mcn)
    rec["multiplicity_note"] = (
        f"{len(raw_boot)} pairwise comparisons over {len(models)} detectors. "
        "Holm controls the family-wise error rate, Benjamini-Hochberg the "
        "false discovery rate. Both are reported because a claim about one "
        "named pair and a claim about the proportion of ordered pairs that "
        "are real need different guarantees.")

    print("\n=== Miss rate with 95% bootstrap CI (image-level resampling) ===")
    for m in models:
        v = boot["per_model"][m]["MR"]
        print(f"  {m:<26} {v['point']:>6.2f}  [{v['ci_low']:>6.2f}, {v['ci_high']:>6.2f}]")

    nb = rec["bh_bootstrap"]["n_rejected"]
    nh = sum(x["reject_at_alpha"]
             for x in rec["holm_bootstrap"]["results"].values())
    print(f"\n=== Multiplicity over {len(raw_boot)} pairwise tests "
          f"({len(models)} detectors) ===")
    print(f"  bootstrap p: Holm rejects {nh}, Benjamini-Hochberg rejects {nb} "
          f"(expected false discoveries {rec['bh_bootstrap']['expected_false_discoveries']})")

    print("\n=== Pairwise comparisons (showing the 25 largest effects) ===")
    print(f"  {'pair':<52}{'dMR':>8}{'CI':>20}{'Holm p':>10}{'BH p':>10}{'sig':>6}")
    shown = sorted(rec["pairwise"].items(),
                   key=lambda kv: -abs(kv[1]["MR_diff_point"]))[:25]
    for k, v in shown:
        hb = rec["holm_bootstrap"]["results"][k]
        bh = rec["bh_bootstrap"]["results"][k]
        sig = "both" if hb["reject_at_alpha"] and bh["reject_at_alpha"] else \
              "BH" if bh["reject_at_alpha"] else "none"
        ci = f"[{v['MR_diff_ci'][0]:.2f}, {v['MR_diff_ci'][1]:.2f}]"
        print(f"  {k:<52}{v['MR_diff_point']:>8.2f}{ci:>20}"
              f"{hb['p_holm']:>10.4f}{bh['p_bh']:>10.4f}{sig:>6}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
