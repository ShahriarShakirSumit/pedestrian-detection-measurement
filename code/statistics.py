"""Uncertainty and agreement statistics for detection results.

Two quantities that point estimates cannot provide:

1. Confidence intervals. Image-level nonparametric bootstrap. Images are the
   independent unit on COCO. For sequence datasets the resampling unit must be
   the sequence instead, and that is a caller decision, not a default here.
2. Paired model comparison. The same resampled image set is used for every
   model in a replicate, so differences between models are paired and their
   interval is usually much tighter than the difference of two independent
   intervals.

Efficiency note: the detections are sorted by score once, globally. A bootstrap
replicate then only changes how many times each image is counted, so the
resampled cumulative true/false positive curves are weighted cumulative sums
over that single fixed ordering. This is exact, not an approximation, and avoids
re-sorting per replicate.
"""
import numpy as np

FPPI_POINTS = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000,
                        0.1778, 0.3162, 0.5623, 1.0000])


class BootstrapSet:
    """Pre-indexed detections and ground-truth counts for one model."""

    def __init__(self, per_image_records, npos_per_image, image_ids):
        """`per_image_records[i]` is an (n_i, 2) array of (score, label) with
        label 1 true positive and 0 false positive. Ignored detections must
        already have been removed. `npos_per_image[i]` is the count of
        matchable ground-truth objects in image i."""
        self.image_ids = list(image_ids)
        self.n_images = len(image_ids)
        self.npos_per_image = np.asarray(npos_per_image, dtype=np.float64)

        scores, labels, owner = [], [], []
        for i, rec in enumerate(per_image_records):
            if len(rec) == 0:
                continue
            r = np.asarray(rec, dtype=np.float64)
            scores.append(r[:, 0]); labels.append(r[:, 1])
            owner.append(np.full(len(r), i, dtype=np.int64))
        if scores:
            s = np.concatenate(scores); l = np.concatenate(labels)
            o = np.concatenate(owner)
            order = np.argsort(-s, kind="mergesort")
            self.scores, self.labels, self.owner = s[order], l[order], o[order]
        else:
            self.scores = self.labels = np.zeros(0)
            self.owner = np.zeros(0, dtype=np.int64)
        self.is_tp = self.labels == 1
        self.is_fp = self.labels == 0

    def _curves(self, mult):
        w = mult[self.owner] if len(self.owner) else np.zeros(0)
        tp = np.cumsum(w * self.is_tp)
        fp = np.cumsum(w * self.is_fp)
        npos = float(np.dot(mult, self.npos_per_image))
        nimg = float(mult.sum())
        return tp, fp, npos, nimg

    def metrics(self, mult=None):
        if mult is None:
            mult = np.ones(self.n_images)
        tp, fp, npos, nimg = self._curves(mult)
        return {"MR": _lamr(tp, fp, npos, nimg) * 100,
                "AP50": _voc_ap(tp, fp, npos) * 100,
                "recall": (float(tp[-1]) / npos * 100) if len(tp) and npos else 0.0}


def _lamr(tp, fp, npos, nimg, pts=FPPI_POINTS):
    if npos <= 0:
        return float("nan")
    if len(tp) == 0:
        return 1.0
    mr = 1 - tp / npos
    fppi = fp / nimg
    vals = []
    for f in pts:
        idx = np.searchsorted(fppi, f, side="right") - 1
        v = float(mr[idx]) if idx >= 0 else 1.0
        vals.append(min(max(v, 1e-10), 1.0))
    return float(np.exp(np.mean(np.log(vals))))


def _voc_ap(tp, fp, npos):
    if npos <= 0 or len(tp) == 0:
        return 0.0
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    # precision envelope, vectorised: a reversed running maximum is equivalent
    # to the usual backward sweep and avoids a Python loop over every detection
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def bootstrap(sets, n_boot=1000, seed=12345, alpha=0.05):
    """Paired image-level bootstrap over a dict of {name: BootstrapSet}.

    Every model sees the same resampled image multiset in a given replicate, so
    per-model intervals and pairwise differences come from one run and the
    differences are properly paired.
    """
    names = list(sets)
    n_images = sets[names[0]].n_images
    for s in sets.values():
        if s.n_images != n_images:
            raise ValueError("all models must be evaluated on the same images")

    rng = np.random.default_rng(seed)
    point = {n: sets[n].metrics() for n in names}
    reps = {n: {"MR": [], "AP50": []} for n in names}

    for _ in range(n_boot):
        idx = rng.integers(0, n_images, n_images)
        mult = np.bincount(idx, minlength=n_images).astype(np.float64)
        for n in names:
            m = sets[n].metrics(mult)
            reps[n]["MR"].append(m["MR"]); reps[n]["AP50"].append(m["AP50"])

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    out = {"n_boot": n_boot, "seed": seed, "interval": f"{int((1-alpha)*100)}% percentile",
           "resampling_unit": "image", "per_model": {}, "pairwise": {},
           "replicates": {n: {k: list(v) for k, v in reps[n].items()} for n in names}}
    for n in names:
        out["per_model"][n] = {}
        for met in ("MR", "AP50"):
            arr = np.array(reps[n][met])
            out["per_model"][n][met] = {
                "point": round(point[n][met], 2),
                "ci_low": round(float(np.percentile(arr, lo)), 2),
                "ci_high": round(float(np.percentile(arr, hi)), 2),
                "boot_sd": round(float(arr.std(ddof=1)), 3)}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["pairwise"][f"{a} vs {b}"] = {}
            for met in ("MR", "AP50"):
                d = np.array(reps[a][met]) - np.array(reps[b][met])
                l, h = float(np.percentile(d, lo)), float(np.percentile(d, hi))
                out["pairwise"][f"{a} vs {b}"][met] = {
                    "diff_point": round(point[a][met] - point[b][met], 2),
                    "ci_low": round(l, 2), "ci_high": round(h, 2),
                    "excludes_zero": bool(l > 0 or h < 0)}
    return out


def kendall_tau(rank_a, rank_b):
    """Kendall's tau-b between two rankings given as {item: rank}."""
    items = sorted(set(rank_a) & set(rank_b))
    n = len(items)
    if n < 2:
        return float("nan"), 0, 0
    con = dis = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[items[i]] - rank_a[items[j]]
            db = rank_b[items[i]] - rank_b[items[j]]
            if da == 0 and db == 0:
                continue
            if da == 0:
                tie_a += 1; continue
            if db == 0:
                tie_b += 1; continue
            if da * db > 0:
                con += 1
            else:
                dis += 1
    denom = np.sqrt((con + dis + tie_a) * (con + dis + tie_b))
    return (float((con - dis) / denom) if denom > 0 else float("nan")), con, dis


# ---------------------------------------------------------------------------
# Significance testing
#
# Effect sizes are reported alongside p-values, and multiplicity corrections
# (Holm and Benjamini-Hochberg) are applied across the family of pairwise
# model comparisons rather than to each test in isolation.
# ---------------------------------------------------------------------------

def mcnemar(detected_a, detected_b, exact_threshold=200):
    """Paired test on per-object detection outcomes.

    `detected_a` and `detected_b` are boolean arrays over the SAME ordered list
    of matchable ground-truth objects: was this object found by the model at the
    declared operating point? Only discordant objects carry information, which
    is exactly the pairing McNemar exploits.

    Uses the exact binomial test on the discordant pairs when they are few, and
    the continuity-corrected chi-square approximation when they are many.
    """
    a = np.asarray(detected_a, dtype=bool)
    b = np.asarray(detected_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("detection vectors must cover the same objects")
    only_a = int(np.sum(a & ~b))      # A found it, B missed it
    only_b = int(np.sum(~a & b))
    n_disc = only_a + only_b
    both = int(np.sum(a & b))
    neither = int(np.sum(~a & ~b))

    if n_disc == 0:
        return {"only_a": 0, "only_b": 0, "both": both, "neither": neither,
                "n_discordant": 0, "statistic": 0.0, "p_value": 1.0,
                "method": "no discordant pairs", "odds_ratio": float("nan")}

    if n_disc <= exact_threshold:
        from math import comb
        k = min(only_a, only_b)
        tail = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc)
        p = min(1.0, 2 * tail)
        method = "exact binomial"
        stat = float(k)
    else:
        stat = (abs(only_a - only_b) - 1.0) ** 2 / n_disc
        from math import erfc, sqrt
        p = erfc(sqrt(stat / 2.0))          # chi-square df=1 upper tail
        method = "chi-square with continuity correction"

    return {"only_a": only_a, "only_b": only_b, "both": both, "neither": neither,
            "n_discordant": n_disc, "statistic": round(float(stat), 4),
            "p_value": float(p), "method": method,
            "odds_ratio": round(only_a / only_b, 3) if only_b else float("inf")}


def bootstrap_pvalue(diff_replicates):
    """Two-sided bootstrap p-value for a paired difference.

    The proportion of replicates falling on the wrong side of zero, doubled.
    Reported with the replicate count, because the smallest resolvable p-value
    is bounded by it: 1000 replicates cannot distinguish p below about 0.001.
    """
    d = np.asarray(diff_replicates, dtype=float)
    n = len(d)
    if n == 0:
        return {"p_value": float("nan"), "n_replicates": 0}
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"p_value": float(min(1.0, max(p, 1.0 / n))),
            "n_replicates": n,
            "resolution_limit": round(1.0 / n, 5),
            "note": "two-sided; bounded below by 1/n_replicates"}


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm step-down correction across a family of tests.

    Controls the family-wise error rate without assuming independence, which
    matters here because pairwise model comparisons on the same data are
    strongly dependent.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": float(p), "p_holm": round(float(adj), 6),
                  "reject_at_alpha": bool(adj < alpha)}
    return {"alpha": alpha, "n_tests": m, "method": "Holm step-down (FWER)",
            "results": out}


def cliffs_delta(x, y):
    """Cliff's delta, a nonparametric effect size for two samples. Reported
    because a p-value alone says nothing about magnitude.

    Computed from the Mann-Whitney rank sum rather than by counting every pair.
    The naive double loop is O(nm), which becomes the bottleneck when the
    effect size is computed for every pair in a large detector set (1035 pairs
    for 46 detectors) at the replicate counts a correction over that many
    tests requires. This form is exact, ties included, and costs
    O((n + m) log(n + m)).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return {"delta": float("nan"), "magnitude": "undefined"}
    both = np.concatenate([x, y])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty(len(both), float)
    sorted_vals = both[order]
    i = 0
    while i < len(both):
        j = i
        while j + 1 < len(both) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    # U for x, then delta = 2U/(nm) - 1, which handles ties through the
    # midranks above exactly as the pair-counting definition does.
    u_x = ranks[:n].sum() - n * (n + 1) / 2.0
    d = 2.0 * u_x / (n * m) - 1.0
    mag = ("negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33
           else "medium" if abs(d) < 0.474 else "large")
    return {"delta": round(float(d), 4), "magnitude": mag}


def benjamini_hochberg(pvalues, alpha=0.05):
    """Benjamini-Hochberg step-up correction across a family of tests.

    Holm controls the probability of even one false rejection. Across the 1035
    pairwise comparisons that 46 detectors generate, that guarantee costs most
    of the power: a real difference has to survive a multiplier of up to 1035.
    For a ranking study the relevant question is often not whether any single
    ordered pair is certain but what proportion of the reported differences
    are spurious. That is a false-discovery-rate question, so both corrections
    are reported and the paper states which claim rests on which.

    The dependence between pairwise comparisons on shared data is positive
    (models that do well on an image tend to do well together), which is the
    regime where the step-up procedure remains valid.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1], reverse=True)
    m = len(items)
    out, prev = {}, 1.0
    for i, (k, p) in enumerate(items):
        rank = m - i
        adj = min(prev, float(p) * m / rank)
        prev = adj
        out[k] = {"p_raw": float(p), "p_bh": round(float(adj), 6),
                  "reject_at_alpha": bool(adj < alpha)}
    n_rej = sum(v["reject_at_alpha"] for v in out.values())
    return {"alpha": alpha, "n_tests": m,
            "method": "Benjamini-Hochberg step-up (FDR)",
            "n_rejected": n_rej,
            "expected_false_discoveries": round(alpha * n_rej, 2),
            "results": out}


def kendalls_w(rankings):
    """Kendall's W, the coefficient of concordance across several rankings.

    Pairwise tau answers whether two protocols agree. With many protocols the
    question becomes whether they agree as a set, and averaging pairwise taus
    is not that quantity. W runs from 0 (no agreement beyond chance) to 1
    (identical orderings), and its asymptotic significance test is the
    Friedman chi-square on the same rank matrix.

    `rankings` is {judge: {item: rank}}. Only items ranked by every judge are
    used, so the matrix is complete and W is well defined. Ties are handled by
    the standard correction term. Without it, W is understated whenever a
    protocol cannot separate two detectors, because tied ranks lower the
    largest attainable rank-sum dispersion.
    """
    judges = sorted(rankings)
    if len(judges) < 2:
        return {"W": float("nan"), "note": "fewer than two rankings"}
    items = sorted(set.intersection(*(set(rankings[j]) for j in judges)))
    n, m = len(items), len(judges)
    if n < 2:
        return {"W": float("nan"), "note": "fewer than two shared items"}

    # Re-rank within each judge so every row is a proper 1..n ranking with
    # ties averaged, regardless of what scale the caller supplied.
    R = np.zeros((m, n))
    T = 0.0
    for a, j in enumerate(judges):
        vals = np.array([rankings[j][i] for i in items], float)
        order = np.argsort(vals, kind="mergesort")
        rk = np.empty(n, float)
        s = 0
        while s < n:
            e = s
            while e + 1 < n and vals[order[e + 1]] == vals[order[s]]:
                e += 1
            rk[order[s:e + 1]] = 0.5 * (s + e) + 1.0
            g = e - s + 1
            if g > 1:
                T += g ** 3 - g
            s = e + 1
        R[a] = rk

    col = R.sum(axis=0)
    S = float(((col - col.mean()) ** 2).sum())
    denom = (m ** 2 * (n ** 3 - n) - m * T) / 12.0
    W = S / denom if denom > 0 else float("nan")
    chi2 = m * (n - 1) * W
    try:                                  # scipy is in the analysis environment
        from scipy.stats import chi2 as _chi2
        pval = float(_chi2.sf(chi2, n - 1))
    except ImportError:                   # keep the module usable without it
        pval = float("nan")
    return {"W": round(float(W), 4),
            "n_items": n, "n_rankings": m,
            "friedman_chi2": round(float(chi2), 3),
            "df": n - 1,
            "p_value": pval,
            "ties_correction": round(float(T), 1),
            "interpretation": ("near-identical" if W >= 0.9 else
                               "strong" if W >= 0.7 else
                               "moderate" if W >= 0.5 else "weak"),
            "note": "1 is identical orderings, 0 is no agreement beyond chance"}
