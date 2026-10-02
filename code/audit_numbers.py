"""Trace every numeric claim in the prose back to a result file.

Tables and figures are generated, so they cannot drift. The prose is typed, so
it can. Each entry in CHECKS pairs a number as stated in the running text with
the value recomputed from the result files, and any mismatch is reported.
"""
import json, re, sys
import paths  # repository-relative results, DATA_ROOT for datasets
E = paths.ROOT
L=lambda f: json.load(open(f"{E}/results/{f}"))

rq2=L("rq2_protocol/rq2_all_persons.json")["results"]
rq2b=L("rq2_protocol/rq2b_rank_stability.json")["settings"]["rq2_all_persons"]
rq2c=L("rq2_protocol/rq2c_protocol_axes.json")
rq2d=L("rq2_protocol/rq2d_inversion_threshold.json")
rq2e=L("rq2_protocol/rq2e_significance_vs_stability.json")
orc=L("oracle/oracle_coco_person.json")
cal=L("calibration/rq6_calibration46.json")
no=L("nightowls/nightowls_results46.json")
lv=L("llvip/llvip_modality_full.json")
tm=L("rq4_timing/rq4_rtx3070_all46.json")["results"]
fl=L("rq4_timing/rq4_flops.json")["results"]
st=L("statistics/coco_person_statistics.json")
sk=L("nightowls/same_k_diff_r.json")
qt=L("nightowls/qualitative_track.json")
import glob as _glob
_ft = {}
RES_FT = f"{E}/results/finetune/rq7_finetune_*.json"
for _p in _glob.glob(RES_FT):
    if "_ir_" in _p:
        continue
    _d = json.load(open(_p))
    _ft[_d["model"]] = _d
_gains = sorted(round(d["baseline"]["test"]["MR"] - d["finetuned"]["test"]["MR"], 2)
                for d in _ft.values())
_bl = [d["baseline"]["test"]["MR"] for d in _ft.values()]
_ad = [d["finetuned"]["test"]["MR"] for d in _ft.values()]
from scipy.stats import kendalltau as _kt
_nb = sorted(_ft, key=lambda m: _ft[m]["baseline"]["test"]["MR"])
_na = sorted(_ft, key=lambda m: _ft[m]["finetuned"]["test"]["MR"])
_tau = round(_kt([_nb.index(m) for m in _ft], [_na.index(m) for m in _ft]).statistic, 2)
_ir = {}
for _p in _glob.glob(RES_FT):
    if "_ir_" not in _p:
        continue
    _d = json.load(open(_p))
    _ir[_d["model"]] = _d
_ig = sorted(round(d["baseline"]["test"]["MR"] - d["finetuned"]["test"]["MR"], 2)
             for d in _ir.values())
_ib = [d["baseline"]["test"]["MR"] for d in _ir.values()]
_ia = [d["finetuned"]["test"]["MR"] for d in _ir.values()]
from scipy.stats import pearsonr as _pr
_ircorr = round(_pr(_ib, [d["baseline"]["test"]["MR"] - d["finetuned"]["test"]["MR"]
                         for d in _ir.values()]).statistic, 2)
rq1=L("rq1_degradation/rq1_coco_person.json")

prot=["COCO","COCO@.5","COCO_matchedBudget","VOC-all","VOC-11","MR"]
sp={m:max(rq2[m][p] for p in prot)-min(rq2[m][p] for p in prot) for m in rq2}
mx=max(max(rq2[m][p] for m in rq2)-min(rq2[m][p] for m in rq2) for p in prot)
og=[orc["models"][m]["oracles"] for m in orc["models"]]
lat={v["model"]:v["total_ms_median"] for v in tm.values() if v["precision"]=="fp32"}
TV={"faster_rcnn_r50_fpn_v2","retinanet_r50_fpn_v2","fcos_r50_fpn","ssd300_vgg16","ssdlite320_mnv3"}
# the 41 detectors that run at a common input size
eff=[lat[m]/fl[m]["gflops"]*1000 for m in fl if m in lat and m not in TV]
nd=lambda m: no["temporal"][m]["first_detection_delay_frames"]["never_detected"]
hb=st["holm_bootstrap"]["results"]; bb=st["bh_bootstrap"]["results"]
hm=st["holm_mcnemar"]["results"]
cd={}
for v in st["pairwise"].values(): cd[v["cliffs_delta"]["magnitude"]]=cd.get(v["cliffs_delta"]["magnitude"],0)+1
agree=sum(1 for k in st["pairwise"] if hb[k]["reject_at_alpha"]==hm[k]["reject_at_alpha"])
big_ns=sum(1 for k in st["pairwise"] if not hb[k]["reject_at_alpha"] and st["pairwise"][k]["cliffs_delta"]["magnitude"]=="large")
tr=rq2b["translation"]
ciw=sorted(abs(v["MR_diff_ci"][1]-v["MR_diff_ci"][0]) for v in st["pairwise"].values())
ax=rq1["axes"]; OB="yolo12x"
pen=[]
for a in ax:
    for b in ax[a][OB]:
        vals={m:ax[a][m][b]["MR"] for m in ax[a]}
        w=min(vals,key=vals.get); pen.append(vals[OB]-vals[w])
wins=0
for a in ax:
    for b in ax[a][OB]:
        vals={m:ax[a][m][b]["MR"] for m in ax[a]}
        if min(vals,key=vals.get)==OB: wins+=1
mono=sum(1 for a in ax for m in ax[a])

CHECKS=[
 ("46 detectors",                     len(rq2), 46),
 ("YOLO26-x VOC-all 87.62",           rq2["yolo26x"]["VOC-all"], 87.62),
 ("YOLO26-x COCO AP 66.47",           rq2["yolo26x"]["COCO"], 66.47),
 ("YOLO26-x MR 29.38",                rq2["yolo26x"]["MR"], 29.38),
 ("its own spread 58.24",             round(sp["yolo26x"],2), 58.24),
 ("max cross-detector spread 38.52",  round(mx,2), 38.52),
 ("36 of 46 exceed it",               sum(1 for m in sp if sp[m]>mx), 36),
 ("Kendall W 0.956",                  rq2b["kendalls_w"]["W"], 0.9563),
 ("Friedman chi2 258.2",              round(rq2b["kendalls_w"]["friedman_chi2"],1), 258.2),
 ("5 distinct rankings",              rq2b["n_distinct_rankings"], 5),
 ("worst tau_b 0.809",                round(min(v["tau_b"] for v in rq2b["kendall_tau"].values()),3), 0.809),
 ("12 protocol departures reorder",   rq2c["summary"]["n_reordering"], 12),
 ("37,310 observations",              rq2d["n_observations"], 37310),
 ("2698 inversions",                  rq2d["n_inversions"], 2698),
 ("7.23 per cent",                    rq2d["inversion_rate_pct"], 7.23),
 ("median inverted gap 0.45",         rq2d["reference_gap_of_inverted_pairs"]["median"], 0.45),
 ("median stable gap 5.13",           rq2d["reference_gap_of_stable_pairs"]["median"], 5.13),
 ("threshold 10.65",                  rq2d["empirical_threshold"]["largest_inverting_reference_gap"], 10.65),
 ("7735 above threshold",             rq2d["empirical_threshold"]["n_pairs_above_threshold"], 7735),
 ("1035 pairs",                       len(st["pairwise"]), 1035),
 ("50,000 replicates",                st["n_boot"], 50000),
 ("Holm rejects 771",                 sum(v["reject_at_alpha"] for v in hb.values()), 771),
 ("BH rejects 860",                   st["bh_bootstrap"]["n_rejected"], 860),
 ("expected false disc 43",           round(st["bh_bootstrap"]["expected_false_discoveries"]), 43),
 ("McNemar Holm 805",                 sum(v["reject_at_alpha"] for v in hm.values()), 805),
 ("McNemar BH 898",                   st["bh_mcnemar"]["n_rejected"], 898),
 ("tests agree on 811",               agree, 811),
 ("Cliff large 869",                  cd.get("large"), 869),
 ("Cliff negligible 55",              cd.get("negligible"), 55),
 ("large but not significant 98",     big_ns, 98),
 ("median CI width 1.77",             round(ciw[len(ciw)//2],2), 1.77),
 ("widest CI 2.95",                   round(ciw[-1],2), 2.95),
 ("conversion worst residual 6.12",   max(v["max_abs_residual"] for v in tr.values()), 6.12),
 ("conversion median worst 4.27",     sorted(v["max_abs_residual"] for v in tr.values())[len(tr)//2], 4.27),
 ("a 0.97 relation exists with 6.12 residual", round([v["pearson_r"] for v in tr.values() if abs(v["max_abs_residual"]-6.12)<0.01][0],2), 0.97),
 ("15 protocol pairs",                len(tr), 15),
 ("Fisher p 5.9e-133",                round(rq2e["fisher_exact_p"]/1e-133,1), 5.9),
 ("not-sig inverted 250",             rq2e["contingency"]["not_significant_and_inverted"], 250),
 ("not-sig stable 14",                rq2e["contingency"]["not_significant_and_stable"], 14),
 ("sig inverted 106",                 rq2e["contingency"]["significant_and_inverted"], 106),
 ("sig stable 665",                   rq2e["contingency"]["significant_and_stable"], 665),
 ("94.7 per cent sensitivity",        rq2e["sensitivity_pct"], 94.7),
 ("86.3 per cent specificity",        rq2e["specificity_pct"], 86.3),
 ("ORACLE-RANK min 22.07",            round(min(o["ORACLE-RANK"]["gain"] for o in og),2), 22.07),
 ("ORACLE-RANK max 31.73",            round(max(o["ORACLE-RANK"]["gain"] for o in og),2), 31.73),
 ("ORACLE-DUP min 0.28",              round(min(o["ORACLE-DUP"]["gain"] for o in og),2), 0.28),
 ("ORACLE-DUP max 2.78",              round(max(o["ORACLE-DUP"]["gain"] for o in og),2), 2.78),
 ("ORACLE-BG range 3.73-12.00",       (round(min(o["ORACLE-BG"]["gain"] for o in og),2), round(max(o["ORACLE-BG"]["gain"] for o in og),2)), (3.73,12.00)),
 ("ORACLE-LOC range 7.11-13.09",      (round(min(o["ORACLE-LOC"]["gain"] for o in og),2), round(max(o["ORACLE-LOC"]["gain"] for o in og),2)), (7.11,13.09)),
 ("background FP 68.9-96.8",          (round(min(orc["models"][m]["error_budget"]["fp_background_pct"] for m in orc["models"]),1), round(max(orc["models"][m]["error_budget"]["fp_background_pct"] for m in orc["models"]),1)), (68.9,96.8)),
 ("ensemble recall 99.12",            orc["ensemble"]["recall_oracle_ensemble_pct"], 99.12),
 ("best single 96.22",                orc["ensemble"]["recall_best_single_pct"], 96.22),
 ("headroom 2.90",                    orc["ensemble"]["headroom_pct"], 2.9),
 ("found by none 95",                 orc["ensemble"]["found_by_none"], 95),
 ("found by none 0.88pc",             orc["ensemble"]["found_by_none_pct"], 0.88),
 ("calibration median 8.5",           cal["summary"]["median_recovered_fraction_pct"], 8.5),
 ("NightOwls images 51848",           no["n_images"], 51848),
 ("admitted pedestrians 7531",        no["n_admitted_pedestrians"], 7531),
 ("244 tracks",                       no["n_tracks_evaluable_2plus"], 244),
 ("YOLO12-s MR 23.73",                no["frame_level"]["yolo12s"]["MR"], 23.73),
 ("YOLO26-s MR 23.46",                no["frame_level"]["yolo26s"]["MR"], 23.46),
 ("YOLO12-s never 53",                nd("yolo12s"), 53),
 ("YOLO26-s never 30",                nd("yolo26s"), 30),
 ("YOLO26-x MR 14.42",                no["frame_level"]["yolo26x"]["MR"], 14.42),
 ("YOLO26-x never 15",                nd("yolo26x"), 15),
 ("SSDLite never 178",                nd("ssdlite320_mnv3"), 178),
 ("SSDLite MR 82.11",                 no["frame_level"]["ssdlite320_mnv3"]["MR"], 82.11),
 ("equal-count pairs 84703",          sk["pairs_same_k"], 84703),
 ("differ in run 4483",               sk["of_which_diff_r"], 4483),
 ("qualitative track frames 9",      qt["n_frames"], 9),
 ("qualitative YOLO12-s hits 4",     qt["hits"]["yolo12s"], 4),
 ("qualitative YOLO26-s hits 4",     qt["hits"]["yolo26s"], 4),
 ("qualitative YOLO12-s run 2",      qt["longest_miss_run"]["yolo12s"], 2),
 ("qualitative YOLO26-s run 5",      qt["longest_miss_run"]["yolo26s"], 5),
 ("qualitative same_k reproduced",   qt["enumeration_reproduced"]["pairs_same_k"], 84703),
 ("qualitative diff_r reproduced",   qt["enumeration_reproduced"]["of_which_diff_r"], 4483),
 ("five detectors adapted",          len(_ft), 5),
 ("smallest gain 10.78",             _gains[0], 10.78),
 ("largest gain 27.37",              _gains[-1], 27.37),
 ("median gain 17.34",               _gains[len(_gains)//2], 17.34),
 ("Faster R-CNN 37.10 base",         _ft["faster_rcnn_r50_fpn_v2"]["baseline"]["test"]["MR"], 37.10),
 ("Faster R-CNN 21.07 adapted",      _ft["faster_rcnn_r50_fpn_v2"]["finetuned"]["test"]["MR"], 21.07),
 ("RetinaNet 47.94 base",            _ft["retinanet_r50_fpn_v2"]["baseline"]["test"]["MR"], 47.94),
 ("RetinaNet 20.57 adapted",         _ft["retinanet_r50_fpn_v2"]["finetuned"]["test"]["MR"], 20.57),
 ("SSD300 50.40 base",               _ft["ssd300_vgg16"]["baseline"]["test"]["MR"], 50.40),
 ("SSD300 39.62 adapted",            _ft["ssd300_vgg16"]["finetuned"]["test"]["MR"], 39.62),
 ("baseline spread 29.20",           round(max(_bl)-min(_bl), 2), 29.20),
 ("adapted spread 28.39",            round(max(_ad)-min(_ad), 2), 28.39),
 ("adaptation tau 0.80",             _tau, 0.80),
 ("five detectors on infrared",      len(_ir), 5),
 ("infrared smallest gain 31.36",    _ig[0], 31.36),
 ("infrared largest gain 58.64",     _ig[-1], 58.64),
 ("infrared median gain 48.08",      _ig[len(_ig)//2], 48.08),
 ("infrared baseline spread 54.81",  round(max(_ib)-min(_ib), 2), 54.81),
 ("infrared adapted spread 27.53",   round(max(_ia)-min(_ia), 2), 27.53),
 ("infrared correlation 0.94",       _ircorr, 0.94),
 ("SSDLite infrared gain 58.64",     round(_ir["ssdlite320_mnv3"]["baseline"]["test"]["MR"]
                                           - _ir["ssdlite320_mnv3"]["finetuned"]["test"]["MR"], 2), 58.64),
 ("max divergence 39",                sk["max_divergence_frames"], 39),
 ("LLVIP pairs 3463",                 lv["n_pairs"], 3463),
 ("LLVIP people 8302",                lv["n_people"], 8302),
 ("LLVIP tau 0.69",                   round(lv["ranking"]["kendall_tau_visible_vs_infrared"],2), 0.69),
 ("IR better 17 of 46",               sum(1 for m in lv["results"] if lv["results"][m]["modality_gap_MR"]<0), 17),
 ("gap min -6.49",                    round(min(lv["results"][m]["modality_gap_MR"] for m in lv["results"]),2), -6.49),
 ("gap max 26.41",                    round(max(lv["results"][m]["modality_gap_MR"] for m in lv["results"]),2), 26.41),
 ("SSD300 loses 18.67",               lv["results"]["ssd300_vgg16"]["modality_gap_MR"], 18.67),
 ("efficiency spread 25.0x",          round(max(eff)/min(eff),1), 25.0),
 ("YOLO12-x wins 7 of 13 bins",       wins, 7),
 ("worst bin penalty 5.42",           round(max(pen),2), 5.42),
 ("138 detector-axis combos",         mono, 138),
]

# Timing statements in the deployment-cost and selection passages, recomputed
# from the timing record so that no latency or memory figure is typed by hand.
import numpy as _np
from scipy.stats import spearmanr as _spearman
_f32 = {v["model"]: v for v in tm.values() if v["precision"] == "fp32"}
_f16 = {v["model"]: v for v in tm.values() if v["precision"] == "fp16"}
_lat = {m: _f32[m]["total_ms_median"] for m in _f32}
_mem = {m: _f32[m]["peak_alloc_MiB"] for m in _f32}
_mr = {m: rq2[m]["MR"] for m in rq2}


def _front(cost):
    best, fr = float("inf"), []
    for m in sorted(_mr, key=lambda m: cost[m]):
        if _mr[m] < best:
            fr.append(m); best = _mr[m]
    return fr


_eff = {m: _lat[m] / fl[m]["gflops"] for m in fl if m in _lat and m not in TV}
_gain = {m: 100 * (_f32[m]["total_ms_median"] - _f16[m]["total_ms_median"])
         / _f32[m]["total_ms_median"] for m in _f32}
_slow = [m for m in _gain if _lat[m] > 20]
_fast = [m for m in _gain if _lat[m] < 13]
_best = lambda b: min((m for m in _mr if _lat[m] <= b), key=_mr.get)
_eight = ["yolo11m", "yolo26m", "yolo26s", "yolov10b", "yolov10l", "yolov10m", "yolov5mu", "yolov8m"]
_top_gain = max(_slow, key=_gain.get)
_mem_best = min((m for m in _mr if _mem[m] <= 100), key=_mr.get)
CHECKS += [
 ("latency frontier 12",              len(_front(_lat)), 12),
 ("memory frontier 10",               len(_front(_mem)), 10),
 ("SSDLite320 peak 40 MiB",           _mem["ssdlite320_mnv3"], 40),
 ("nothing else within 40 MiB",       sum(1 for m in _mem if m != "ssdlite320_mnv3" and _mem[m] <= 40), 0),
 ("eight within 5 ms and 4 points",   int(max(_lat[m] for m in _eight) - min(_lat[m] for m in _eight) <= 5
                                          and max(_mr[m] for m in _eight) - min(_mr[m] for m in _eight) <= 4), 1),
 ("latency vs GFLOPs Spearman 0.78",  round(_spearman([_lat[m] for m in _eff], [fl[m]["gflops"] for m in _eff]).statistic, 2), 0.78),
 ("most efficient 0.08 ms/GFLOP",     (min(_eff, key=_eff.get), round(min(_eff.values()), 2)), ("yolov8x", 0.08)),
 ("least efficient 1.96 ms/GFLOP",    (max(_eff, key=_eff.get), round(max(_eff.values()), 2)), ("yolo12n", 1.96)),
 ("fp16 median gain -0.6",            round(float(_np.median(list(_gain.values()))), 1), -0.6),
 ("fp16 slower for 25",               sum(1 for g in _gain.values() if g < 0), 25),
 ("13 detectors above 20 ms",         len(_slow), 13),
 ("their median fp16 gain 23.2",      round(float(_np.median([_gain[m] for m in _slow])), 1), 23.2),
 ("largest fp16 gain 38.3, YOLOv8-x", (_top_gain, round(_gain[_top_gain], 1)), ("yolov8x", 38.3)),
 ("21 detectors below 13 ms",         len(_fast), 21),
 ("their median fp16 gain -2.3",      round(float(_np.median([_gain[m] for m in _fast])), 1), -2.3),
 ("SSDLite320 loses 12.8 in fp16",    round(-_gain["ssdlite320_mnv3"], 1), 12.8),
 ("best overall YOLO12-x 28.99",      (min(_mr, key=_mr.get), round(_mr["yolo12x"], 2)), ("yolo12x", 28.99)),
 ("YOLO12-x costs 24.42 ms",          _lat["yolo12x"], 24.42),
 ("YOLO12-x costs 561 MiB",           _mem["yolo12x"], 561),
 ("best under 15 ms YOLO26-l 30.33",  (_best(15), round(_mr[_best(15)], 2)), ("yolo26l", 30.33)),
 ("best under 10 ms YOLOv10-m 32.53", (_best(10), round(_mr[_best(10)], 2)), ("yolov10m", 32.53)),
 ("best under 8 ms YOLOv8s 36.34",    (_best(8), round(_mr[_best(8)], 2)), ("yolov8s", 36.34)),
 ("seven qualify under 8 ms",         sum(1 for m in _mr if _lat[m] <= 8), 7),
 ("best under 100 MiB YOLOv9-s",      (_mem_best, round(_mr[_mem_best], 2)), ("yolov9s", 35.01)),
 ("memory budget costs 6.02",         round(_mr["yolov9s"] - _mr["yolo12x"], 2), 6.02),
]

# Remaining measured statements in the text: protocol ranks, degradation,
# calibration, NightOwls rank agreement, LLVIP margins and implementation checks.
import itertools as _it
from scipy.stats import kendalltau as _ktau, pearsonr as _pear
_P = ["COCO", "COCO@.5", "COCO_matchedBudget", "VOC-all", "VOC-11", "MR"]
_rk = {p: {m: i + 1 for i, m in enumerate(sorted(rq2, key=lambda m: rq2[m][p] if p == "MR" else -rq2[m][p]))}
       for p in _P}
_span = {m: max(_rk[p][m] for p in _P) - min(_rk[p][m] for p in _P) for m in rq2}
_first = {p: min(_rk[p], key=_rk[p].get) for p in _P}
_h = rq1["axes"]["height"]; _cr = rq1["axes"]["crowding"]; _oc = rq1["axes"]["occlusion_proxy"]
_df = rq1["degradation_factor"]
_hf = {m: v["factor"] for m, v in _df["height"].items()}
_base = {m: _h[m]["h>=125"]["MR"] for m in _hf}
_ov = {m: rq1["overall"][m]["MR"] for m in rq1["overall"]}
_cm = cal["models"]
_med = lambda xs: float(_np.median(list(xs)))
_fl = no["frame_level"]; _tp = no["temporal"]
_nd = {m: _tp[m]["first_detection_delay_frames"]["never_detected"] for m in _fl}
_TV5 = ["faster_rcnn_r50_fpn_v2", "retinanet_r50_fpn_v2", "fcos_r50_fpn", "ssd300_vgg16", "ssdlite320_mnv3"]
def _agree(ms):
    t = _ktau([_fl[m]["MR"] for m in ms], [_nd[m] for m in ms]).statistic
    d = sum(1 for a, b in _it.combinations(ms, 2) if (_fl[a]["MR"] - _fl[b]["MR"]) * (_nd[a] - _nd[b]) < 0)
    return round(float(t), 2), d
_m41 = [m for m in _fl if m not in _TV5]
_nmr = {m: i + 1 for i, m in enumerate(sorted(_fl, key=lambda m: _fl[m]["MR"]))}
_ncov = {m: 1 + sum(1 for o in _fl if _nd[o] < _nd[m]) for m in _fl}
_gain_rank = lambda ms: _med(_nmr[m] - _ncov[m] for m in ms)
_gap = {m: lv["results"][m]["modality_gap_MR"] for m in lv["results"]}
_vis_r = round(float(_pear([d["baseline"]["test"]["MR"] for d in _ft.values()],
                           [d["baseline"]["test"]["MR"] - d["finetuned"]["test"]["MR"] for d in _ft.values()]).statistic), 2)
_below = [_ft[m]["baseline"]["test"]["MR"] - _ir[m]["finetuned"]["test"]["MR"] for m in _ir]
_vk = {"faster_rcnn_r50_fpn_v2": "faster_rcnn_r50_fpn_v2"}
_apdiff = max(abs(rq2[m]["COCO@.5"] - rq2[m]["VOC-all"]) for m in rq2)
CHECKS += [
 ("median rank span 3",               _med(_span.values()), 3.0),
 ("4 detectors move more than 10",    sum(1 for v in _span.values() if v > 10), 4),
 ("Faster R-CNN moves 23, 5th to 28th", (_span["faster_rcnn_r50_fpn_v2"], _rk["VOC-all"]["faster_rcnn_r50_fpn_v2"], _rk["COCO"]["faster_rcnn_r50_fpn_v2"]), (23, 5, 28)),
 ("RT-DETR-X 2nd and 22nd",           (_rk["VOC-all"]["rtdetr_x"], _rk["COCO"]["rtdetr_x"]), (2, 22)),
 ("YOLO26-x first under COCO, COCO@.5, VOC-all", (_first["COCO"], _first["COCO@.5"], _first["VOC-all"]), ("yolo26x",) * 3),
 ("YOLO12-x first under VOC-11 and MR", (_first["VOC-11"], _first["MR"]), ("yolo12x",) * 2),
 ("one of fifteen pairs identical",   sum(1 for a, b in _it.combinations(_P, 2) if all(_rk[a][m] == _rk[b][m] for m in rq2)), 1),
 ("SSD300 and SSDLite320 10.65 apart", round(rq2["ssdlite320_mnv3"]["MR"] - rq2["ssd300_vgg16"]["MR"], 2), 10.65),
 ("they converge to 0.78 above 125 px", round(abs(_h["ssd300_vgg16"]["h>=125"]["MR"] - _h["ssdlite320_mnv3"]["h>=125"]["MR"]), 2), 0.78),
 ("and swap only there",              [b for b in _h["ssd300_vgg16"] if _h["ssdlite320_mnv3"][b]["MR"] < _h["ssd300_vgg16"][b]["MR"]], ["h>=125"]),
 ("B=1000 example pair 34.07 apart",  round(rq2["ssdlite320_mnv3"]["MR"] - rq2["yolo12x"]["MR"], 2), 34.07),
 ("replicate floor about 20,700",     round(1035 / 0.05), 20700),
 ("771 is 74.5 per cent",             round(100 * 771 / 1035, 1), 74.5),
 ("811 is 78.4 per cent",             round(100 * 811 / 1035, 1), 78.4),
 ("869 is 84.0 per cent",             round(100 * 869 / 1035, 1), 84.0),
 ("bootstrap-only rejections 95",     sum(1 for k in hb if hb[k]["reject_at_alpha"] and not hm[k]["reject_at_alpha"]), 95),
 ("McNemar-only rejections 129",      sum(1 for k in hb if hm[k]["reject_at_alpha"] and not hb[k]["reject_at_alpha"]), 129),
 ("AP agrees within 0.55 at IoU 0.5", round(_apdiff, 2), 0.55),
 ("RT-DETR-L among the six strongest", sorted(_ov, key=_ov.get).index("rtdetr_l") + 1 <= 6, True),
 ("RT-DETR-L height 7.25 to 62.22, x8.58", (_h["rtdetr_l"]["h>=125"]["MR"], _h["rtdetr_l"]["h<25"]["MR"], _df["height"]["rtdetr_l"]["factor"]), (7.25, 62.22, 8.58)),
 ("RT-DETR-L crowding 7.39 to 32.75, x4.43", (_cr["rtdetr_l"]["1 person"]["MR"], _cr["rtdetr_l"]["10+"]["MR"], _df["crowding"]["rtdetr_l"]["factor"]), (7.39, 32.75, 4.43)),
 ("RT-DETR-L occlusion 6.35 to 36.44, x5.74", (_oc["rtdetr_l"]["none"]["MR"], _oc["rtdetr_l"]["severe"]["MR"], _df["occlusion_proxy"]["rtdetr_l"]["factor"]), (6.35, 36.44, 5.74)),
 ("SSDLite320 99.01 below 25 px",     _h["ssdlite320_mnv3"]["h<25"]["MR"], 99.01),
 ("height factors 4.88 to 10.19",     (min(_hf.values()), max(_hf.values())), (4.88, 10.19)),
 ("steepest has the lowest base rate", max(_hf, key=_hf.get) == min(_base, key=_base.get), True),
 ("calibration recovery 0.5 to 13.0", (min(v["recovered_fraction_pct"] for v in _cm.values()), max(v["recovered_fraction_pct"] for v in _cm.values())), (0.5, 13.0)),
 ("median recovered 2.05 points",     round(_med(v["recovered_points"] for v in _cm.values()) + 1e-9, 2), 2.05),
 ("median headroom 24.03 points",     round(_med(v["headroom_points"] for v in _cm.values()) + 1e-9, 2), 24.03),
 ("shuffled control 37.4 to 70.7",    (round(min(-v["control_recovered_points"] for v in _cm.values()), 1), round(max(-v["control_recovered_points"] for v in _cm.values()), 1)), (37.4, 70.7)),
 ("NightOwls tau 0.62, 189 discordant", _agree(list(_fl)), (0.62, 189)),
 ("matched 41: tau 0.66, 133 discordant", _agree(_m41), (0.66, 133)),
 ("highlighted pair 0.27 apart",      round(_fl["yolo12s"]["MR"] - _fl["yolo26s"]["MR"], 2), 0.27),
 ("coverage differs 1.8 times",       round(_nd["yolo12s"] / _nd["yolo26s"], 1), 1.8),
 ("Faster R-CNN never detects 5, a third", (_nd["faster_rcnn_r50_fpn_v2"], _nd["yolo26x"] / _nd["faster_rcnn_r50_fpn_v2"]), (5, 3.0)),
 ("Faster R-CNN four places below",   _nmr["faster_rcnn_r50_fpn_v2"] - _nmr["yolo26x"], 4),
 ("ResNet-50 coverage ranks 1, 2, 5", (_ncov["faster_rcnn_r50_fpn_v2"], _ncov["fcos_r50_fpn"], _ncov["retinanet_r50_fpn_v2"]), (1, 2, 5)),
 ("and miss-rate ranks 5, 35, 28",    (_nmr["faster_rcnn_r50_fpn_v2"], _nmr["fcos_r50_fpn"], _nmr["retinanet_r50_fpn_v2"]), (5, 35, 28)),
 ("median rank gain 23",              _gain_rank(["faster_rcnn_r50_fpn_v2", "fcos_r50_fpn", "retinanet_r50_fpn_v2"]), 23.0),
 ("no median gain for the 41",        _gain_rank(_m41), 0.0),
 ("SSDLite320 misses 73 per cent",    round(100 * _nd["ssdlite320_mnv3"] / 244), 73),
 ("LLVIP median gap +0.90",           round(_med(_gap.values()), 2), 0.9),
 ("YOLOv9-e gains 6.49",              (min(_gap, key=_gap.get), round(-_gap["yolov9e"], 2)), ("yolov9e", 6.49)),
 ("YOLOv5-xu gains 5.44",             round(-_gap["yolov5xu"], 2), 5.44),
 ("SSDLite320 loses most, 26.41",     (max(_gap, key=_gap.get), _gap["ssdlite320_mnv3"]), ("ssdlite320_mnv3", 26.41)),
 ("visible gain unrelated, r -0.04",  _vis_r, -0.04),
 ("adapted IR below visible base by 29.41 to 40.73", (round(min(_below), 2), round(max(_below), 2)), (29.41, 40.73)),
 ("older lightweight cost 33",        round(rq2["ssdlite320_mnv3"]["MR"] - rq2["faster_rcnn_r50_fpn_v2"]["MR"]), 33),
]

bad=0
for name, got, want in CHECKS:
    ok = (abs(got-want)<0.015 if isinstance(got,float) and isinstance(want,float)
          else got==want)
    if not ok: bad+=1; print(f"  MISMATCH  {name:<34} results say {got}, paper says {want}")
print(f"\n{len(CHECKS)-bad} of {len(CHECKS)} numeric claims verified against the result files")
sys.exit(1 if bad else 0)
