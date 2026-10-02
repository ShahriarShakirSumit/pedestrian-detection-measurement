"""Generate the LaTeX result tables for the paper from the result JSONs.

Every table is emitted here from the result files and none is typed by hand. A
number typed by hand can silently drift away from the file it came from, and
generating the tables means the paper and the results directory cannot disagree.

Display names are kept in one place. A detector that appears in a table under a
name this file does not know raises rather than being silently passed through,
because a stray internal name in a published table is worse than a crash.
"""
import argparse, json, os, re, sys

import paths  # repository-relative results, DATA_ROOT for datasets
R = paths.RESULTS

PRETTY = {
    "faster_rcnn_r50_fpn_v2": "Faster R-CNN R50-FPN v2",
    "retinanet_r50_fpn_v2": "RetinaNet R50-FPN v2",
    "fcos_r50_fpn": "FCOS R50-FPN",
    "ssd300_vgg16": "SSD300-VGG16",
    "ssdlite320_mnv3": "SSDLite320-MNv3",
    "rtdetr_l": "RT-DETR-L", "rtdetr_x": "RT-DETR-X",
    "yolov3u": "YOLOv3-u", "yolov3_tinyu": "YOLOv3-tiny-u",
    "yolov3_sppu": "YOLOv3-SPP-u",
}
for fam, tag in [("yolov5", "YOLOv5"), ("yolov8", "YOLOv8"), ("yolov9", "YOLOv9"),
                 ("yolov10", "YOLOv10"), ("yolo11", "YOLO11"),
                 ("yolo12", "YOLO12"), ("yolo26", "YOLO26")]:
    for s in "ntsmlxbce":
        PRETTY[f"{fam}{s}"] = f"{tag}-{s}"
        PRETTY[f"{fam}{s}u"] = f"{tag}-{s}u"


def name(m):
    if m not in PRETTY:
        raise KeyError(f"no display name for detector {m!r}; add it to PRETTY")
    return PRETTY[m]


def family(m):
    """Group detectors so a 46-row table can be read in blocks."""
    if m in ("faster_rcnn_r50_fpn_v2",):
        return "Two-stage"
    if m in ("retinanet_r50_fpn_v2", "ssd300_vgg16", "ssdlite320_mnv3"):
        return "One-stage dense"
    if m in ("fcos_r50_fpn",):
        return "Anchor-free"
    if m.startswith("rtdetr"):
        return "Query-based"
    return "YOLO series"


ORDER = ["Two-stage", "One-stage dense", "Anchor-free", "Query-based", "YOLO series"]


def blocks(models):
    """Yield (family, sorted members) in a fixed order."""
    for f in ORDER:
        mem = [m for m in models if family(m) == f]
        if mem:
            yield f, mem


def fnum(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def validate(path):
    """Flag a generated table whose column is mostly missing values.

    A wrong lookup path does not raise. It returns None for every row and the
    table is emitted with a column of dashes, which typesets cleanly and reads
    as "not measured". Any column more than half empty is therefore reported.
    """
    rows = [l for l in open(path).read().split("\n")
            if l.endswith(r"\\") and "&" in l
            and not any(k in l for k in ("multicolumn", "cmidrule", "toprule"))]
    if len(rows) < 3:
        return []
    body = [r[:-2].split("&") for r in rows]
    width = max(len(r) for r in body)
    bad = []
    for c in range(width):
        cells = [r[c].strip() for r in body if len(r) > c]
        if not cells:
            continue
        # An empty cell is a deliberate grouping blank (the family column is
        # filled only on the first row of each block). A dash is a value the
        # lookup failed to find. Only the second is a defect.
        empty = sum(1 for x in cells if x == "--")
        if empty > len(cells) / 2:
            bad.append((c, empty, len(cells)))
    return bad



def two_up(rows, header, colspec, gap=r"@{\hspace{10pt}}"):
    """Lay a long single list out as two side-by-side halves.

    Nothing is dropped: the same rows in the same order, read down the left
    column then down the right. A 46-row table becomes 23 rows and saves close
    to half a page. Family membership is given in the inventory table instead.
    """
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    while len(right) < len(left):
        right.append(None)
    blank = " & " * (len(header) - 1)
    body = []
    for a, b in zip(left, right):
        body.append(f"{a} & {(b if b is not None else blank)} \\\\")
    head = " & ".join(header) + " & " + " & ".join(header)
    return ([rf"\begin{{tabular}}{{@{{}}{colspec}{gap}{colspec}@{{}}}}",
             r"\toprule", head + r" \\", r"\midrule"]
            + body + [r"\bottomrule", r"\end{tabular}"])


def coco_table(out):
    d = json.load(open(f"{R}/rq2_protocol/rq2_all_persons.json"))
    res = d["results"]
    ms = sorted(res, key=lambda m: res[m]["MR"])
    rows = [f"{name(m)} & {fnum(res[m]['COCO'])} & {fnum(res[m]['COCO@.5'])} & "
            f"{fnum(res[m]['COCO_APsmall'])} & {fnum(res[m]['VOC-all'])} & "
            f"{fnum(res[m]['MR'])} & {fnum(res[m]['final_recall'])}" for m in ms]
    L = [r"\begin{table*}[!t]",
         r"\caption{COCO person validation split, all %d detectors, one frozen "
         r"detection set per detector, ordered by MR and read down the left half "
         r"then the right. AP columns are higher-is-better and MR is "
         r"lower-is-better. The spread between the VOC and MR columns for a "
         r"single detector is the subject of Section~\ref{sec:exp_protocol}.}" % len(ms),
         r"\label{tab:exp_coco_all}", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3.4pt}"]
    L += two_up(rows,
                ["Detector", "COCO AP", r"AP$_{50}$", r"AP$_{S}$", "VOC-all", "MR", "Recall"],
                "lrrrrrr")
    L += [r"\end{table*}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def nightowls_table(path, out):
    d = json.load(open(path))
    fl, tp = d["frame_level"], d["temporal"]
    ms = sorted(fl, key=lambda m: fl[m]["MR"])
    L = [r"\begin{table*}[!t]",
         r"\caption{NightOwls validation split, all %d detectors. Frame-level "
         r"accuracy on the left, track-level persistence on the right over %d "
         r"tracks. Delay and miss-run columns are 90th percentiles in frames. "
         r"Lower is better in every column except AP$_{50}$ and recall.}"
         % (len(ms), d["n_tracks_evaluable_2plus"]),
         r"\label{tab:exp_nightowls}", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3.4pt}"]
    rows = [f"{name(m)} & {fnum(fl[m]['MR'])} & {fnum(fl[m]['AP50'])} & "
            f"{fnum(fl[m]['recall'])} & "
            f"{tp[m]['first_detection_delay_frames']['never_detected']} & "
            f"{fnum(tp[m]['first_detection_delay_frames']['p90'],1)} & "
            f"{fnum(tp[m]['longest_miss_run_frames']['p90'],1)}" for m in ms]
    L += two_up(rows, ["Detector", "MR", r"AP$_{50}$", "Recall", "Never",
                       "Delay", "Run"], "lrrrrrr")
    L += [r"\end{table*}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def llvip_table(out):
    d = json.load(open(f"{R}/llvip/llvip_modality_full.json"))
    res = d["results"]
    ms = sorted(res, key=lambda m: res[m]["visible"]["MR"])
    L = [r"\begin{table*}[!t]",
         r"\caption{LLVIP registered visible and infrared pairs, all %d "
         r"detectors, %d pairs and %d annotated people. One shared annotation "
         r"scores both columns, so the only variable is input modality. Every "
         r"checkpoint was pretrained on visible imagery, which makes the gap a "
         r"measure of domain shift rather than of multispectral design.}"
         % (len(ms), d["n_pairs"], d["n_people"]),
         r"\label{tab:exp_llvip}", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3.4pt}"]
    rows = [f"{name(m)} & {fnum(res[m]['visible']['MR'])} & "
            f"{fnum(res[m]['visible']['AP50'])} & {fnum(res[m]['infrared']['MR'])} & "
            f"{fnum(res[m]['infrared']['AP50'])} & {fnum(res[m]['modality_gap_MR'])}"
            for m in ms]
    L += two_up(rows, ["Detector", r"MR$_{\mathrm{v}}$", r"AP$_{50}^{\mathrm{v}}$",
                       r"MR$_{\mathrm{ir}}$", r"AP$_{50}^{\mathrm{ir}}$",
                       r"$\Delta$MR"], "lrrrrr")
    L += [r"\end{table*}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def timing_table(out):
    d = json.load(open(f"{R}/rq4_timing/rq4_rtx3070_all46.json"))
    res = d["results"]
    fl = json.load(open(f"{R}/rq4_timing/rq4_flops.json"))["results"]
    ms = sorted({k.split("::")[0] for k in res})
    L = [r"\begin{table*}[!t]",
         r"\caption{Batch-one latency on one RTX 3070 with 8~GB of memory, at a "
         r"single input resolution. Median and 95th-percentile total latency in "
         r"milliseconds and peak allocated memory in MiB, beside the "
         r"hardware-independent arithmetic cost. Latency holds for this setup "
         r"only; GFLOPs does not depend on it.}",
         r"\label{tab:exp_timing_all}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"& & & & \multicolumn{3}{c}{fp32} & \multicolumn{3}{c}{fp16} \\",
         r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Family & Detector & Params & GFLOPs & ms & p95 & MiB & ms & p95 & MiB \\",
         r"\midrule"]
    def med(m, p):
        k = f"{m}::{p}"
        return res.get(k, {}).get("total_ms_median")
    first = True
    for fam, mem in blocks(ms):
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for i, m in enumerate(sorted(mem, key=lambda x: med(x, "fp32") or 1e9)):
            a = res.get(f"{m}::fp32", {}); b = res.get(f"{m}::fp16", {})
            lead = fam if i == 0 else ""
            L.append(f"{lead} & {name(m)} & {fnum(a.get('params_M'),2)} & "
                     f"{fnum(fl.get(m, {}).get('gflops'))} & "
                     f"{fnum(a.get('total_ms_median'))} & {fnum(a.get('total_ms_p95'))} & "
                     f"{a.get('peak_alloc_MiB','--')} & "
                     f"{fnum(b.get('total_ms_median'))} & {fnum(b.get('total_ms_p95'))} & "
                     f"{b.get('peak_alloc_MiB','--')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def significance_2x2(out):
    """The contingency of protocol instability against statistical significance.

    Generated rather than typed because both margins move whenever the
    detector set or the bootstrap replicate count changes, and the p-value is
    small enough that a rounded copy of it reads as zero.
    """
    d = json.load(open(f"{R}/rq2_protocol/rq2e_significance_vs_stability.json"))
    c = d["contingency"]
    ni, ns = c["not_significant_and_inverted"], c["not_significant_and_stable"]
    si, ss = c["significant_and_inverted"], c["significant_and_stable"]
    odds = (ni * ss) / (ns * si)
    pv = d["fisher_exact_p"]
    mant, exp = f"{pv:.1e}".split("e")
    L = [r"\begin{table}[!t]",
         r"\caption{Protocol instability against statistical significance, over "
         r"all %d pairs of the 46 detectors on the COCO person subset. "
         r"Significance is a Holm-corrected image-level bootstrap at "
         r"$\alpha=0.05$ with 50\,000 replicates. A pair counts as unstable if "
         r"its ordering reverses under any evaluated protocol or condition. "
         r"Fisher's exact test gives $p = %s \times 10^{%d}$, with an odds ratio "
         r"of %.0f.}" % (d["n_pairs"], mant, int(exp), odds),
         r"\label{tab:exp_significance}", r"\centering", r"\footnotesize",
         r"\begin{tabular}{lcc}", r"\toprule",
         r" & \textbf{Ever reversed} & \textbf{Never reversed} \\",
         r"\midrule",
         f"Not significant & {ni} & {ns} \\\\",
         f"Significant & {si} & {ss} \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return d["n_pairs"]


def inventory_table(out):
    """The 46 detectors with their checkpoint, input size, cost and parameters.

    The detector set is the subject of the study, so this table appears in the
    main text. Input size is included because it is not uniform across families
    and several claims depend on it.
    """
    t = json.load(open(f"{R}/rq4_timing/rq4_rtx3070_all46.json"))["results"]
    fl = json.load(open(f"{R}/rq4_timing/rq4_flops.json"))["results"]
    sys.path.insert(0, paths.CODE)
    try:
        from detect_coco_ultralytics import MODELS as UM
        weights = {k: v[1] for k, v in UM.items()}
    except Exception:
        weights = {}
    ms = sorted({k.split("::")[0] for k in t})
    half = (len(ms) + 1) // 2
    L = [r"\begin{table*}[!t]",
         r"\caption{The %d detectors evaluated in this study. Input is the size "
         r"each checkpoint is run at, which is its own released default and is "
         r"not uniform across families, so comparisons that cross it are marked "
         r"where they are made. GFLOPs is counted at that input size and does "
         r"not depend on the hardware. Measured latency, which does, is in "
         r"Table~\ref{tab:s_timing_full}.}" % len(ms),
         r"\label{tab:inventory}", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l l r r r@{\hspace{14pt}} l l r r r@{}}", r"\toprule",
         r"Family & Detector & Input & GFLOPs & Params & "
         r"Family & Detector & Input & GFLOPs & Params \\",
         r"\midrule"]
    ordered = []
    for fam, mem in blocks(ms):
        for i, m in enumerate(sorted(mem)):
            ordered.append((fam if i == 0 else "", m))
    left, right = ordered[:half], ordered[half:]
    while len(right) < len(left):
        right.append(None)
    def cell(e):
        if e is None:
            return " & & & & "
        fam, m = e
        p_ = t.get(f"{m}::fp32", {}).get("params_M")
        return (f"{fam} & {name(m)} & {fl.get(m, {}).get('input_px', '--')} & "
                f"{fnum(fl.get(m, {}).get('gflops'))} & {fnum(p_)}")
    for a, b in zip(left, right):
        L.append(f"{cell(a)} & {cell(b)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outDir", required=True)
    ap.add_argument("--nightowls", default=f"{R}/nightowls/nightowls_results46.json")
    a = ap.parse_args()
    os.makedirs(a.outDir, exist_ok=True)
    todo = [("tab_coco_all.tex", lambda o: coco_table(o)),
            ("tab_nightowls_all.tex", lambda o: nightowls_table(a.nightowls, o)),
            ("tab_llvip_all.tex", lambda o: llvip_table(o)),
            ("tab_timing_all.tex", lambda o: timing_table(o)),
            ("tab_significance_2x2.tex", lambda o: significance_2x2(o)),
            ("tab_inventory_main.tex", lambda o: inventory_table(o))]
    for fn, f in todo:
        p = os.path.join(a.outDir, fn)
        try:
            n = f(p)
            print(f"  {fn:<26} {n} detectors")
            for c, e, tot in validate(p):
                print(f"     WARNING column {c} is {e}/{tot} empty in {fn}")
        except FileNotFoundError as e:
            print(f"  {fn:<26} SKIPPED, input not ready: {os.path.basename(e.filename)}")
