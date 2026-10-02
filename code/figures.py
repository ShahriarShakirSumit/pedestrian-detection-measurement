"""Generate the paper figures from stored result files.

Print constraints, which differ from screen charts:
  - Vector PDF with embedded Type-42 fonts and no rasterised text.
  - Lettering of 8-12 pt at final printed size, so figure sizes are set in
    inches at the column widths of the paper's two-column layout.
  - Every series carries a marker shape as well as a colour, so the figure
    survives greyscale printing and colour-vision deficiency without relying on
    hue alone.
  - No interaction layer, so identity comes from a legend plus selective direct
    labels.

Categorical colours follow a fixed eight-slot order, checked so that any two
neighbouring slots are clearly distinct, which is what line and bar charts
need. A scatter plot places every pair of series side by side, and only three
of the eight slots are distinct from all the others, so the scatter figures
use one ink colour with direct labels instead of eight hues.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Eight-slot categorical order, checked for distinctness between neighbouring
# slots on a white background.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8c8b86"
GRID = "#dcdcd8"

# In the paper's two-column layout a full-width float spans 455.24 pt (6.30 in)
# and a single-column float 216.24 pt (2.99 in). Figures are drawn at the width
# they will occupy, so nothing is rescaled and the lettering stays at 8-12 pt.
# Panels are laid out horizontally because a tall float on a two-column page
# squeezes the text into a narrow band and leaves large white gaps.
FULLWIDTH = 6.30
COLWIDTH = 2.99

SHORT = {
    "faster_rcnn_r50_fpn_v2": "Faster R-CNN",
    "retinanet_r50_fpn_v2": "RetinaNet",
    "fcos_r50_fpn": "FCOS",
    "ssd300_vgg16": "SSD300",
    "ssdlite320_mnv3": "SSDLite320",
    "rtdetr_l": "RT-DETR-L",
    "rtdetr_x": "RT-DETR-X",
}
# The YOLO series contributes 39 checkpoints. Their short names are derived
# from the key rather than listed one by one, so they cannot drift.
for _fam, _tag in [("yolov10", "YOLOv10"), ("yolov5", "YOLOv5"), ("yolov8", "YOLOv8"),
                   ("yolov9", "YOLOv9"), ("yolov3", "YOLOv3"), ("yolo11", "YOLO11"),
                   ("yolo12", "YOLO12"), ("yolo26", "YOLO26")]:
    for _s in ("n", "s", "m", "l", "x", "b", "c", "t", "e",
               "nu", "su", "mu", "lu", "xu", "u", "_tinyu", "_sppu"):
        SHORT.setdefault(_fam + _s, _tag + "-" + _s.lstrip("_"))


def setup():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def finish(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, loc="left", pad=6)


def fig_degradation(rq1, out):
    """Three ordered conditions, one line per detector.

    At 46 detectors a coloured line and a legend entry for each is unreadable,
    and recycled colours carry no identity. The full set is drawn in thin grey
    lines so the reader sees the spread and the monotonicity, the
    inter-quartile range is shaded, and four named detectors are drawn in
    colour: the best, the median, the second-worst and the worst by overall
    miss rate.
    """
    axes_order = [("height", "Person height (px)"),
                  ("crowding", "People per image"),
                  ("occlusion_proxy", "Occlusion (keypoint proxy)")]
    models = sorted(rq1["overall"], key=lambda m: rq1["overall"][m]["MR"])
    # best, median, second-worst and worst by overall miss rate
    FOCUS = [models[0], models[len(models) // 2], models[-2], models[-1]]
    FOCUS = list(dict.fromkeys(FOCUS))
    fig, axs = plt.subplots(1, 3, figsize=(FULLWIDTH, 2.45), sharey=True)
    fig.subplots_adjust(wspace=0.10)
    for k, (axis, xlab) in enumerate(axes_order):
        ax = axs[k]
        per = rq1["axes"][axis]
        bins = list(next(iter(per.values())).keys())
        x = np.arange(len(bins))
        allv = np.array([[per[m][b]["MR"] for b in bins] for m in models])
        ax.fill_between(x, np.percentile(allv, 25, axis=0),
                        np.percentile(allv, 75, axis=0),
                        color=SERIES[0], alpha=0.13, linewidth=0, zorder=2)
        for m in models:
            if m in FOCUS:
                continue
            ax.plot(x, [per[m][b]["MR"] for b in bins], color=MUTED,
                    linewidth=0.55, alpha=0.55, zorder=3)
        for i, m in enumerate(FOCUS):
            if m not in per:
                continue
            ax.plot(x, [per[m][b]["MR"] for b in bins], color=SERIES[i % 8],
                    marker=MARKERS[i % 8], markersize=3.2, linewidth=1.4,
                    markeredgewidth=0, label=SHORT.get(m, m), zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=30, ha="right")
        ax.set_xlim(-0.35, len(bins) - 0.65)
        finish(ax, xlabel=xlab, ylabel="Miss rate (%)" if k == 0 else None)
        ax.set_title("abc"[k] + ")", loc="left", pad=4, fontweight="bold")
    axs[0].set_ylim(0, 100)
    handles = [Line2D([], [], color=SERIES[i % 8], marker=MARKERS[i % 8],
                      markersize=3.2, linewidth=1.4, markeredgewidth=0,
                      label=SHORT.get(m, m)) for i, m in enumerate(FOCUS)]
    handles.append(Line2D([], [], color=MUTED, linewidth=0.55,
                          label=f"other {len(models) - len(FOCUS)} detectors"))
    handles.append(Line2D([], [], color=SERIES[0], linewidth=6, alpha=0.13,
                          label="inter-quartile range"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.30), columnspacing=1.2,
               handletextpad=0.5, fontsize=7.0)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  -> {out}")


def fig_frontier(rq2, rq4, out):   # full-width float, panels side by side
    """Accuracy against measured cost. A scatter plot places every pair of
    points side by side, so identity comes from direct labels on one ink
    colour rather than eight hues, which are not all mutually distinct."""
    cost = {v["model"]: v for v in rq4["results"].values() if v["precision"] == "fp32"}
    pts = {m: (rq2["results"][m]["MR"], cost[m]["total_ms_median"],
               cost[m]["peak_alloc_MiB"]) for m in rq2["results"] if m in cost}
    fig, axs = plt.subplots(1, 2, figsize=(FULLWIDTH, 2.7))
    fig.subplots_adjust(wspace=0.40)
    for k, (idx, xlab) in enumerate([(1, "Batch-1 latency (ms), RTX 3070"),
                                     (2, "Peak GPU memory (MiB)")]):
        ax = axs[k]
        xs = {m: pts[m][idx] for m in pts}
        ys = {m: pts[m][0] for m in pts}
        order = sorted(pts, key=lambda m: xs[m])
        front, best = [], float("inf")
        for m in order:
            if ys[m] < best:
                front.append(m); best = ys[m]
        ax.plot([xs[m] for m in front], [ys[m] for m in front],
                color=MUTED, linewidth=1.0, linestyle="--", zorder=2,
                label="Pareto frontier")
        xr = max(xs.values()) - min(xs.values())
        yr = max(ys.values()) - min(ys.values())
        ax.set_xlim(min(xs.values()) - 0.14 * xr, max(xs.values()) + 0.14 * xr)
        ax.set_ylim(min(ys.values()) - 0.16 * yr, max(ys.values()) + 0.16 * yr)
        # With 46 detectors a label on every point is unreadable at this
        # width. Only the Pareto members carry a name, since they are the
        # points a deployment decision chooses between. The rest are
        # drawn small to show the cloud the frontier is drawn from.
        for m in sorted(pts, key=lambda m: xs[m]):
            on = m in front
            ax.scatter(xs[m], ys[m], s=30 if on else 11,
                       facecolor=SERIES[0] if on else "white",
                       edgecolor=SERIES[0] if on else MUTED,
                       linewidth=1.0 if on else 0.7,
                       alpha=1.0 if on else 0.75, zorder=4 if on else 3)
        # The frontier is dense at the cheap end, where several detectors sit
        # within a millisecond of each other. Labelling all of them overlaps
        # the text, so a label is kept only when it clears the labels already
        # placed (the two ends of the frontier are always labelled). The
        # unlabelled frontier points remain visible as filled markers on the
        # dashed line. Labels are centred above each point and flip below when
        # the slot above is taken. Side placement collides in this geometry
        # because the text is wide relative to the point spacing.
        placed = []
        for m in sorted(front, key=lambda m: xs[m]):
            clash = any(abs(xs[m] - px) < 0.11 * xr and abs(ys[m] - py) < 0.07 * yr
                        for px, py, _ in placed)
            if clash and m not in (front[0], front[-1]):
                continue
            above = True
            for px, py, pa in placed:
                if abs(xs[m] - px) < 0.22 * xr and abs(ys[m] - py) < 0.10 * yr and pa:
                    above = False
            ax.annotate(SHORT.get(m, m), (xs[m], ys[m]),
                        textcoords="offset points",
                        xytext=(0, 6 if above else -11),
                        ha="center", fontsize=5.8, color=INK2, zorder=5)
            placed.append((xs[m], ys[m], above))
        finish(ax, xlabel=xlab, ylabel="Miss rate (%)", grid_axis="both")
        ax.set_title("ab"[k] + ")", loc="left", pad=4, fontweight="bold")
        if k == 0:
            ax.legend(loc="upper right", fontsize=6.8)
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"  -> {out}")


def fig_oracle(oracle, out):
    """Headroom per failure mode, as a distribution over the detector set.

    The claim concerns the failure modes, not any individual detector, so each
    mode is one row rather than a group of bars per detector. A thick bar spans
    the inter-quartile range across the 46 detectors, with a white tick at the
    median, and every detector is drawn as a jittered point beside it. The
    separation between the rows is the finding.
    """
    M = oracle["models"]
    keys = ["ORACLE-RANK", "ORACLE-BG", "ORACLE-LOC", "ORACLE-DUP"]
    labels = ["Perfect confidence\nranking", "No background\nfalse positives",
              "Loose localisation", "Perfect duplicate\nsuppression"]
    data = [np.array([M[m]["oracles"][k]["gain"] for m in M]) for k in keys]

    fig, ax = plt.subplots(figsize=(COLWIDTH, 2.35))
    rng = np.random.default_rng(7)
    for j, v in enumerate(data):
        y = len(keys) - 1 - j
        ax.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=7,
                   facecolor=SERIES[j], edgecolor="none", alpha=0.45, zorder=3)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [y + 0.30] * 2, color=SERIES[j], linewidth=4.0,
                solid_capstyle="butt", zorder=4)
        ax.plot([med, med], [y + 0.22, y + 0.38], color="white",
                linewidth=1.3, zorder=5)
        ax.annotate(f"{v.min():.1f} to {v.max():.1f}", (v.max(), y + 0.30),
                    textcoords="offset points", xytext=(5, -2),
                    fontsize=6.4, color=INK2, va="center", zorder=6)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels[::-1], fontsize=7.0)
    ax.set_ylim(-0.45, len(keys) - 0.35)
    ax.set_xlim(-1.5, max(v.max() for v in data) * 1.30)
    finish(ax, xlabel="Miss-rate points recoverable", grid_axis="x")
    ax.annotate(f"each point is one of {len(M)} detectors",
                xy=(0.98, 0.03), xycoords="axes fraction", ha="right",
                fontsize=6.6, color=MUTED)
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"  -> {out}")


def fig_protocol_spread(rq2, out):
    """Two claims, one per panel, because they need different encodings.

    Panel a shows one detector's score under every protocol. Mixing
    higher-is-better average precision with lower-is-better miss rate on a
    shared axis would imply an ordering that does not exist, so each protocol
    gets its own labelled row and the panel is read as a spread, not a trend.

    Panel b shows rank position under the same protocols. At 46 detectors a
    line per model with a legend is unreadable, so the full set is drawn as a
    light grey background that carries the overall stability, and only the six
    detectors whose rank moves furthest are drawn in colour and named. Those
    crossings are the finding.
    """
    prots = ["VOC-all", "COCO@.5", "VOC-11", "COCO", "MR"]
    plabels = ["VOC AP@.50 (all-point)", "COCO AP@.50", "VOC AP@.50 (11-point)",
               "COCO AP [.50:.95]", "Log-average miss rate"]
    polarity = ["higher better"] * 4 + ["lower better"]
    models = sorted(rq2["results"], key=lambda m: rq2["results"][m]["MR"])
    # Panel a asks how far one detector's reported score moves with the
    # protocol, so it shows the detector with the widest spread.
    focus = max(models, key=lambda m: max(rq2["results"][m][p] for p in prots)
                - min(rq2["results"][m][p] for p in prots))

    fig, axs = plt.subplots(1, 2, figsize=(FULLWIDTH, 2.7),
                            gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axs[0]
    y = np.arange(len(prots))[::-1]
    vals = [rq2["results"][focus][p] for p in prots]
    ax.hlines(y, 0, vals, color=GRID, linewidth=1.0, zorder=2)
    for yy, v, pol in zip(y, vals, polarity):
        c = SERIES[0] if pol == "higher better" else SERIES[1]
        ax.scatter(v, yy, s=42, color=c, zorder=4)
        ax.annotate(f"{v:.2f}", (v, yy), textcoords="offset points",
                    xytext=(7, -2.6), fontsize=7.4, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n({p})" for l, p in zip(plabels, polarity)],
                       fontsize=6.9)
    ax.set_xlim(0, max(vals) * 1.26)
    finish(ax, xlabel=f"Reported score for {SHORT.get(focus, focus)}", grid_axis="x")
    ax.set_title("a)", loc="left", pad=4, fontweight="bold")
    ax.annotate("one detector, one image set,\none detection file",
                xy=(0.42, 0.10), xycoords="axes fraction", fontsize=7, color=INK2)

    ax = axs[1]
    order = ["COCO", "COCO@.5", "COCO_matchedBudget", "VOC-all", "VOC-11", "MR"]
    order = [p for p in order if p in rq2["results"][models[0]]]
    ranks = {}
    for p in order:
        rev = p != "MR"
        srt = sorted(models, key=lambda m: rq2["results"][m][p], reverse=rev)
        ranks[p] = {m: i + 1 for i, m in enumerate(srt)}
    x = np.arange(len(order))
    span = {m: max(ranks[p][m] for p in order) - min(ranks[p][m] for p in order)
            for m in models}
    movers = sorted(models, key=lambda m: -span[m])[:6]
    movers = [m for m in movers if span[m] > 0]
    for m in models:
        if m in movers:
            continue
        ax.plot(x, [ranks[p][m] for p in order], color=GRID,
                linewidth=0.7, zorder=2)
    for i, m in enumerate(movers):
        ax.plot(x, [ranks[p][m] for p in order], color=SERIES[i % 8],
                marker=MARKERS[i % 8], markersize=3.2, linewidth=1.3,
                markeredgewidth=0, label=f"{SHORT.get(m, m)} ({span[m]})",
                zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["COCO AP", "COCO@.5", "matched\nbudget", "VOC-all",
                        "VOC-11", "MR"][:len(order)], fontsize=6.9, rotation=30,
                       ha="right")
    step = max(1, len(models) // 9)
    ax.set_yticks(list(range(1, len(models) + 1, step)))
    ax.invert_yaxis()
    finish(ax, ylabel="Rank (1 = best)", grid_axis="y")
    ax.set_title("b)", loc="left", pad=4, fontweight="bold")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(-0.12, -0.36),
              fontsize=6.6, handletextpad=0.4, columnspacing=1.0,
              title="widest rank movement (positions)", title_fontsize=6.6)
    fig.subplots_adjust(wspace=0.40)
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"  -> {out}")


def fig_kaist(kaist, out):
    """Published detections under one locked protocol, across subsets.

    The rank panel carries the result: on KAIST the evaluation subset does
    reorder the methods, in contrast with the COCO protocol result.
    """
    cols = [("MR_all_Reasonable", "Reasonable"), ("MR_all_Reasonable_small", "Small"),
            ("MR_all_All", "All"), ("MR_all_Reasonable_occ=heavy", "Heavy occ.")]
    methods = sorted(kaist["results"], key=lambda m: kaist["results"][m]["MR_all_Reasonable"])
    x = np.arange(len(cols))
    fig, axs = plt.subplots(1, 2, figsize=(FULLWIDTH, 2.7))
    ax = axs[0]
    for i, m in enumerate(methods):
        v = [kaist["results"][m][c] for c, _ in cols]
        ax.plot(x, v, color=SERIES[i % 8], marker=MARKERS[i % 8], markersize=3.8,
                linewidth=1.4, markeredgewidth=0, label=m, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in cols], rotation=20, ha="right")
    finish(ax, xlabel="Evaluation subset", ylabel="Log-average miss rate (%)")
    ax.set_title("a)", loc="left", pad=4, fontweight="bold")
    ax.legend(ncol=2, loc="upper left", fontsize=6.2, handletextpad=0.3, columnspacing=0.7)

    ax = axs[1]
    for i, m in enumerate(methods):
        r = []
        for c, _ in cols:
            srt = sorted(methods, key=lambda z: kaist["results"][z][c])
            r.append(srt.index(m) + 1)
        ax.plot(x, r, color=SERIES[i % 8], marker=MARKERS[i % 8], markersize=3.8,
                linewidth=1.4, markeredgewidth=0, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in cols], rotation=20, ha="right")
    ax.set_yticks(range(1, len(methods) + 1)); ax.invert_yaxis()
    finish(ax, xlabel="Evaluation subset", ylabel="Rank (1 = best)", grid_axis="y")
    ax.set_title("b)", loc="left", pad=4, fontweight="bold")
    fig.subplots_adjust(wspace=0.40)
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outDir", required=True)
    a = ap.parse_args()
    setup()
    os.makedirs(a.outDir, exist_ok=True)
    R = a.results
    L = lambda p: json.load(open(os.path.join(R, p)))

    fig_degradation(L("rq1_degradation/rq1_coco_person.json"),
                    f"{a.outDir}/fig_degradation.pdf")
    fig_frontier(L("rq2_protocol/rq2_all_persons.json"),
                 L("rq4_timing/rq4_rtx3070_all46.json"), f"{a.outDir}/fig_frontier.pdf")
    fig_oracle(L("oracle/oracle_coco_person.json"), f"{a.outDir}/fig_oracle.pdf")
    fig_protocol_spread(L("rq2_protocol/rq2_all_persons.json"),
                        f"{a.outDir}/fig_protocol_spread.pdf")
    fig_kaist(L("kaist_published_reeval/kaist_published_reeval.json"),
              f"{a.outDir}/fig_kaist_subsets.pdf")


if __name__ == "__main__":
    main()
