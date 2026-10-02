"""Generate every table in the paper's appendix from the result files.

Same contract as make_tables.py: nothing in the appendix is typed by hand, so
the appendix cannot drift away from the result files it describes.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import R, name, family, blocks, fnum, validate

W = os.environ.get("ULTRA_WEIGHTS", "")


def env_table(out):
    d = json.load(open(f"{R}/rq4_timing/rq4_rtx3070_all46.json"))
    hw, sw, st = d["hardware"], d["software"], d["settings"]
    rows = [("GPU", f"{hw['gpu']}, {hw['total_MiB']}~MiB"),
            ("Driver clocks", f"{hw['driver_state_before']['clocks.sm']} of "
                              f"{hw['driver_state_before']['clocks.max.sm']} maximum, "
                              "persistence mode enabled"),
            ("PyTorch", sw["torch"]), ("CUDA", sw["cuda"]),
            ("cuDNN", str(sw["cudnn"])), ("Python", sw["python"]),
            ("Operating system", sw["os"].replace("_", r"\_")),
            ("Timing batch size", str(st["batch_size"])),
            ("Timing input size", f"{st['input_size'][0]} $\\times$ {st['input_size'][1]}"),
            ("Warm-up iterations", str(st["warmup_iters"])),
            ("Measured iterations", str(st["measured_iters"])),
            ("Timing score threshold", str(st["score_threshold"])),
            ("Detection score floor", "0.01 for all detection sweeps"),
            ("Timing boundary", st["timing_boundary"]),
            ("Bootstrap", "500 replicates, image-level resampling unit"),
            ("Multiplicity", "Holm step-down and Benjamini-Hochberg, "
                             "1035 pairwise comparisons")]
    L = [r"\begin{table}[!ht]",
         r"\caption{The measurement environment. Every number in this paper was "
         r"produced on this machine.}",
         r"\label{tab:s_env}", r"\centering", r"\footnotesize",
         r"\begin{tabular}{L{4.2cm} L{11.5cm}}", r"\toprule",
         r"Item & Value \\", r"\midrule"]
    L += [f"{k} & {v} \\\\" for k, v in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(rows)


def inventory_table(out):
    t = json.load(open(f"{R}/rq4_timing/rq4_rtx3070_all46.json"))["results"]
    # Input size is each checkpoint's own default and the paper makes claims
    # that depend on it, so it is listed in the inventory.
    px = {k: v["input_px"]
          for k, v in json.load(open(f"{R}/rq4_timing/rq4_flops.json"))["results"].items()}
    ms = sorted({k.split("::")[0] for k in t})
    weights = {}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from detect_coco_ultralytics import MODELS as UM
        weights = {k: v[1] for k, v in UM.items()}
    except Exception:
        pass
    L = [r"\begin{table}[!ht]",
         r"\caption{The 46 detectors, by the checkpoint each was loaded from. "
         r"The input column is the size each model is run at, which is its own "
         r"released default and is not uniform across families. Comparisons "
         r"drawn in the main paper between detectors at different input sizes "
         r"are flagged where they occur.}",
         r"\label{tab:s_inventory}", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{L{1.4cm} L{1.9cm} L{2.1cm} rr}", r"\toprule",
         r"Family & Detector & Checkpoint & Input & Params \\", r"\midrule"]
    first = True
    for fam, mem in blocks(ms):
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for i, m in enumerate(sorted(mem)):
            # Escape only the real file name. The torchvision fallback is
            # already LaTeX, so passing it through the same replace would
            # double-escape the underscore.
            raw = weights.get(m)
            wt = raw.replace("_", r"\_") if raw else r"torchvision COCO\_V1"
            p = t.get(f"{m}::fp32", {}).get("params_M")
            L.append(f"{fam if i==0 else ''} & {name(m)} & \\texttt{{{wt}}} & "
                     f"{px.get(m,'--')} & {fnum(p)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


COLS = [("COCO", "COCO AP"), ("COCO@.5", "AP$_{50}$"),
        ("COCO_APsmall", "AP$_{S}$"), ("COCO_APmedium", "AP$_{M}$"),
        ("COCO_APlarge", "AP$_{L}$"), ("VOC-all", "VOC-all"),
        ("VOC-11", "VOC-11"), ("MR", "MR"), ("final_recall", "Recall")]


def coco_full(out):
    d = json.load(open(f"{R}/rq2_protocol/rq2_all_persons.json"))["results"]
    ms = sorted(d, key=lambda m: d[m]["MR"])
    L = [r"\begin{table}[!ht]",
         r"\caption{COCO person validation split, all %d detectors under every "
         r"protocol variant. MR is lower-is-better, every other column is "
         r"higher-is-better.}" % len(ms),
         r"\label{tab:s_coco_full}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{ll" + "r" * len(COLS) + "}", r"\toprule",
         "Family & Detector & " + " & ".join(h for _, h in COLS) + r" \\",
         r"\midrule"]
    first = True
    for fam, mem in blocks(ms):
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for i, m in enumerate(sorted(mem, key=lambda x: d[x]["MR"])):
            vals = " & ".join(fnum(d[m].get(k)) for k, _ in COLS)
            L.append(f"{fam if i==0 else ''} & {name(m)} & {vals} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def significance_table(out, top=40):
    s = json.load(open(f"{R}/statistics/coco_person_statistics.json"))
    pw, holm, bh = s["pairwise"], s["holm_bootstrap"]["results"], s["bh_bootstrap"]["results"]
    keys = sorted(pw, key=lambda k: -abs(pw[k]["MR_diff_point"]))[:top]
    nh = sum(v["reject_at_alpha"] for v in holm.values())
    nb = s["bh_bootstrap"]["n_rejected"]
    L = [r"\begin{table}[!ht]",
         r"\caption{Pairwise miss-rate comparisons, the %d largest effects of "
         r"%d. Across the whole family Holm rejects %d and Benjamini-Hochberg "
         r"rejects %d at $\alpha=0.05$, with an expected %.1f false discoveries "
         r"among the latter. CI is the 95 per cent paired bootstrap interval on "
         r"the miss-rate difference.}"
         % (len(keys), len(pw), nh, nb, s["bh_bootstrap"]["expected_false_discoveries"]),
         r"\label{tab:s_significance}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{L{5.4cm} rrr rr c}", r"\toprule",
         r"Pair & $\Delta$MR & CI low & CI high & Holm $p$ & BH $p$ & Cliff $\delta$ \\",
         r"\midrule"]
    for k in keys:
        v = pw[k]
        pair = k.replace(" vs ", " / ")
        for a, b in [(m, name(m)) for m in set(sum([x.split(" vs ") for x in [k]], []))]:
            pair = pair.replace(a, b)
        L.append(f"{pair} & {fnum(v['MR_diff_point'])} & {fnum(v['MR_diff_ci'][0])} & "
                 f"{fnum(v['MR_diff_ci'][1])} & {holm[k]['p_holm']:.4f} & "
                 f"{bh[k]['p_bh']:.4f} & {v['cliffs_delta']['delta']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(keys)


def nightowls_full(path, out):
    d = json.load(open(path))
    fl, tp = d["frame_level"], d["temporal"]
    bs = d.get("bootstrap", {}).get("per_model", {})
    ms = sorted(fl, key=lambda m: fl[m]["MR"])
    L = [r"\begin{table}[!ht]",
         r"\caption{NightOwls validation split, all %d detectors, with 95\%% "
         r"bootstrap intervals on miss rate and both the median and the 90th "
         r"percentile of each track-level quantity over %d tracks.}"
         % (len(ms), d["n_tracks_evaluable_2plus"]),
         r"\label{tab:s_nightowls_full}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{ll rrr r rr rr}", r"\toprule",
         r"& & \multicolumn{3}{c}{Frame level} & & "
         r"\multicolumn{2}{c}{Delay (frames)} & \multicolumn{2}{c}{Miss run (frames)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Family & Detector & MR & 95\% CI & Recall & Never & med & p90 & med & p90 \\",
         r"\midrule"]
    first = True
    for fam, mem in blocks(ms):
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for i, m in enumerate(sorted(mem, key=lambda x: fl[x]["MR"])):
            f, t = fl[m], tp[m]
            b = bs.get(m, {}).get("MR", {})
            ci = (f"[{b['ci_low']:.2f}, {b['ci_high']:.2f}]"
                  if "ci_low" in b else "--")
            dd, rr = t["first_detection_delay_frames"], t["longest_miss_run_frames"]
            L.append(f"{fam if i==0 else ''} & {name(m)} & {fnum(f['MR'])} & {ci} & "
                     f"{fnum(f['recall'])} & {dd['never_detected']} & "
                     f"{fnum(dd['median'],1)} & {fnum(dd['p90'],1)} & "
                     f"{fnum(rr['median'],1)} & {fnum(rr['p90'],1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def llvip_full(out):
    d = json.load(open(f"{R}/llvip/llvip_modality_full.json"))
    res = d["results"]
    bs = d.get("bootstrap", {}).get("per_model", {})
    ms = sorted(res, key=lambda m: res[m]["visible"]["MR"])
    L = [r"\begin{table}[!ht]",
         r"\caption{LLVIP visible and infrared results for all %d detectors "
         r"over %d registered pairs and %d annotated people, with 95\%% "
         r"bootstrap intervals on miss rate.}"
         % (len(ms), d["n_pairs"], d["n_people"]),
         r"\label{tab:s_llvip_full}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{ll rrr rrr r}", r"\toprule",
         r"& & \multicolumn{3}{c}{Visible} & \multicolumn{3}{c}{Infrared} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Family & Detector & MR & 95\% CI & AP$_{50}$ & MR & 95\% CI & AP$_{50}$ & $\Delta$MR \\",
         r"\midrule"]
    def ci(m, mod):
        # The LLVIP bootstrap is keyed "model::modality", not nested by model.
        b = bs.get(f"{m}::{mod}", {}).get("MR", {})
        return f"[{b['ci_low']:.2f}, {b['ci_high']:.2f}]" if "ci_low" in b else "--"
    first = True
    for fam, mem in blocks(ms):
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for i, m in enumerate(sorted(mem, key=lambda x: res[x]["visible"]["MR"])):
            v, ir = res[m]["visible"], res[m]["infrared"]
            L.append(f"{fam if i==0 else ''} & {name(m)} & {fnum(v['MR'])} & {ci(m,'visible')} & "
                     f"{fnum(v['AP50'])} & {fnum(ir['MR'])} & {ci(m,'infrared')} & "
                     f"{fnum(ir['AP50'])} & {fnum(res[m]['modality_gap_MR'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


def timing_full(out):
    res = json.load(open(f"{R}/rq4_timing/rq4_rtx3070_all46.json"))["results"]
    fl = json.load(open(f"{R}/rq4_timing/rq4_flops.json"))["results"]
    ms = sorted({k.split("::")[0] for k in res})
    L = [r"\begin{table}[!ht]",
         r"\caption{Batch-one latency for all %d detectors at both numerical "
         r"precisions on the machine in Table~\ref{tab:s_env}.}" % len(ms),
         r"\label{tab:s_timing_full}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{ll rr rrrr rrrr}", r"\toprule",
         r"& & & & \multicolumn{4}{c}{fp32} & \multicolumn{4}{c}{fp16} \\",
         r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
         r"Family & Detector & Params & GFLOPs & ms & p95 & img/s & MiB & ms & p95 & img/s & MiB \\",
         r"\midrule"]
    def g(m, p, k):
        return res.get(f"{m}::{p}", {}).get(k)
    first = True
    for fam, mem in blocks(ms):
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for i, m in enumerate(sorted(mem, key=lambda x: g(x, "fp32", "total_ms_median") or 1e9)):
            L.append(f"{fam if i==0 else ''} & {name(m)} & {fnum(g(m,'fp32','params_M'))} & "
                     f"{fnum(fl.get(m, {}).get('gflops'))} & "
                     f"{fnum(g(m,'fp32','total_ms_median'))} & {fnum(g(m,'fp32','total_ms_p95'))} & "
                     f"{fnum(g(m,'fp32','throughput_img_s'),1)} & {g(m,'fp32','peak_alloc_MiB') or '--'} & "
                     f"{fnum(g(m,'fp16','total_ms_median'))} & {fnum(g(m,'fp16','total_ms_p95'))} & "
                     f"{fnum(g(m,'fp16','throughput_img_s'),1)} & {g(m,'fp16','peak_alloc_MiB') or '--'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out, "w").write("\n".join(L) + "\n")
    return len(ms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outDir", required=True)
    ap.add_argument("--nightowls", default=f"{R}/nightowls/nightowls_results46.json")
    a = ap.parse_args()
    os.makedirs(a.outDir, exist_ok=True)
    jobs = [("tab_env.tex", env_table),
            ("tab_inventory.tex", inventory_table),
            ("tab_coco_full.tex", coco_full),
            ("tab_significance.tex", significance_table),
            ("tab_nightowls_full.tex", lambda o: nightowls_full(a.nightowls, o)),
            ("tab_llvip_full.tex", llvip_full),
            ("tab_timing_full.tex", timing_full)]
    for fn, f in jobs:
        p = os.path.join(a.outDir, fn)
        try:
            print(f"  {fn:<26} {f(p)} rows")
            for c, e, tot in validate(p):
                print(f"     WARNING column {c} is {e}/{tot} empty in {fn}")
        except FileNotFoundError as e:
            print(f"  {fn:<26} SKIPPED, file not ready: {os.path.basename(e.filename)}")
        except KeyError as e:
            # A stale result file may lack keys that the current analysis
            # writes. Report it rather than writing a table that silently
            # omits a column.
            print(f"  {fn:<26} SKIPPED, stale input missing key {e}")
