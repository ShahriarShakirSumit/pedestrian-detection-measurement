"""Two panels: what a sensor change costs, and what adaptation buys back.

Every number in the figure, including those in its text, is read from the
result files. Text inside a figure is not checked by the LaTeX build or by the
numeric audit of the prose (audit_numbers.py), so a value typed into the figure
could contradict the text beside it unnoticed. No result value is written here
as a literal.
"""
import glob, json, os

import paths  # repository-relative results, DATA_ROOT for datasets
E = paths.ROOT
MOD = json.load(open(f"{E}/results/llvip/llvip_modality_full.json"))
TAU = MOD["ranking"]["kendall_tau_visible_vs_infrared"]

PRETTY = {"faster_rcnn_r50_fpn_v2": "Faster R-CNN", "faster_rcnn": "Faster R-CNN",
          "retinanet_r50_fpn_v2": "RetinaNet", "fcos_r50_fpn": "FCOS",
          "ssd300_vgg16": "SSD300", "ssdlite320_mnv3": "SSDLite320",
          "rtdetr_l": "RT-DETR-L", "yolov8s": "YOLOv8s", "yolo11n": "YOLO11n",
          "yolov9e": "YOLOv9-e", "yolov5xu": "YOLOv5-xu"}

pairs = []
for m, v in MOD["results"].items():
    if isinstance(v.get("visible"), dict) and isinstance(v.get("infrared"), dict):
        pairs.append((m, v["visible"]["MR"], v["infrared"]["MR"]))
pairs.sort(key=lambda r: r[2] - r[1])
better = sum(1 for _, a, b in pairs if b < a)

FT, MINUTES = [], []
for p in sorted(glob.glob(f"{E}/results/finetune/rq7_finetune_*.json")):
    d = json.load(open(p))
    # The right panel shows only the infrared adaptation runs, matching the
    # visible-to-infrared shift shown in the left panel.
    if "_ir_" not in os.path.basename(p):
        continue
    FT.append((PRETTY.get(d["model"], d["model"]),
               d["baseline"]["test"]["MR"], d["finetuned"]["test"]["MR"]))
    MINUTES.append(d["training"]["seconds"] / 60.0)
FT.sort(key=lambda r: r[1])

BLUE, AMBER, GREEN, INK, MUT = "#2B5C92", "#B0673A", "#3E6B52", "#1F2933", "#7A828C"
W, H = 900, 424
T_, B_ = 92, 96


def spread(ys, gap=12.5):
    """Dots stay at the true value. Only the labels are nudged apart."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for n, i in enumerate(order):
        if n and out[i] - out[order[n - 1]] < gap:
            out[i] = out[order[n - 1]] + gap
    return out


def panel(o, lx, rx, rows, lo, hi, llab, rlab, colf, labels):
    sy = lambda v: T_ + (H - T_ - B_) * (v - lo) / (hi - lo)
    o += [f'<text x="{lx}" y="{T_-16}" text-anchor="middle" font-size="11" '
          f'font-weight="600" fill="#6B7480">{llab}</text>',
          f'<text x="{rx}" y="{T_-16}" text-anchor="middle" font-size="11" '
          f'font-weight="600" fill="#6B7480">{rlab}</text>',
          f'<line x1="{lx}" y1="{T_-6}" x2="{lx}" y2="{H-B_+6}" stroke="#E4E8EC"/>',
          f'<line x1="{rx}" y1="{T_-6}" x2="{rx}" y2="{H-B_+6}" stroke="#E4E8EC"/>']
    for name, a, b in rows:
        c = colf(a, b)
        w = 1.9 if name in labels else 0.85
        op = 0.95 if name in labels else 0.4
        o.append(f'<line x1="{lx}" y1="{sy(a):.1f}" x2="{rx}" y2="{sy(b):.1f}" '
                 f'stroke="{c}" stroke-width="{w}" opacity="{op}"/>')
    named = [(n, a, b) for n, a, b in rows if n in labels]
    la = spread([sy(a) for _, a, _ in named])
    lb = spread([sy(b) for _, _, b in named])
    for k, (name, a, b) in enumerate(named):
        c = colf(a, b)
        o.append(f'<circle cx="{lx}" cy="{sy(a):.1f}" r="3.4" fill="{c}"/>'
                 f'<circle cx="{rx}" cy="{sy(b):.1f}" r="3.4" fill="{c}"/>')
        o.append(f'<text x="{lx-9}" y="{la[k]+3.4:.1f}" text-anchor="end" '
                 f'font-size="10.2" fill="#2A333D">{name}</text>')
        o.append(f'<text x="{rx+9}" y="{lb[k]+3.4:.1f}" font-size="10.2" '
                 f'fill="#5C6470">{b:.1f}</text>')


o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
     f'height="{H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
     f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="600" '
     f'fill="{INK}">A sensor change is not a uniform tax. Adaptation buys it back.</text>',
     f'<text x="250" y="54" text-anchor="middle" font-size="11" font-weight="600" '
     f'fill="#48515C">All {len(pairs)} detectors, {MOD["n_pairs"]:,} registered pairs</text>']

lo = min(min(a, b) for _, a, b in pairs) - 3
hi = max(max(a, b) for _, a, b in pairs) + 3
show = {pairs[0][0], pairs[1][0], pairs[-1][0], pairs[-2][0]}
rows = [(PRETTY.get(m, m), a, b) for m, a, b in pairs]
show = {PRETTY.get(m, m) for m in show}
panel(o, 205, 350, rows, lo, hi, "Visible", "Infrared",
      lambda a, b: BLUE if b < a else AMBER, show)
o.append(f'<text x="250" y="{H-46}" text-anchor="middle" font-size="10" fill="{MUT}">'
         f'{better} of {len(pairs)} improve on infrared. Kendall tau {TAU:.2f}.</text>')
o.append(f'<text x="250" y="{H-32}" text-anchor="middle" font-size="10" fill="{MUT}">'
         f'Blue improves, amber degrades.</text>')

o.append(f'<line x1="510" y1="{T_-34}" x2="510" y2="{H-B_+6}" stroke="#E4E8EC"/>')
if FT:
    o.append(f'<text x="700" y="54" text-anchor="middle" font-size="11" font-weight="600" '
             f'fill="#48515C">Two epochs on LLVIP infrared, {len(FT)} detector'
             f'{"s" if len(FT) > 1 else ""}</text>')
    flo = min(min(a, b) for _, a, b in FT) - 4
    fhi = max(max(a, b) for _, a, b in FT) + 4
    panel(o, 620, 780, FT, flo, fhi, "Before", "Adapted",
          lambda a, b: GREEN, {n for n, _, _ in FT})
    gains = [a - b for _, a, b in FT]
    o.append(f'<text x="700" y="{H-46}" text-anchor="middle" font-size="10" fill="{MUT}">'
             f'Test miss rate falls by {min(gains):.1f} to {max(gains):.1f} points.</text>')
    o.append(f'<text x="700" y="{H-32}" text-anchor="middle" font-size="10" fill="{MUT}">'
             f'{min(MINUTES):.0f} to {max(MINUTES):.0f} minutes each on one RTX 3070.</text>')

o.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="9.6" fill="#98A0AA">'
         f'All detectors are COCO-pretrained on visible imagery, so the left panel '
         f'measures domain shift, not multispectral capability.</text>')
o.append('</svg>')
out = os.path.join(paths.FIGURES, "Exp_modality_adaptation.svg")
open(out, "w").write("\n".join(o))
print(f"{len(pairs)} detectors, {better} improve, tau {TAU}; adaptation panel n={len(FT)}")
