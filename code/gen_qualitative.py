"""Qualitative figure: the proposition on longest miss runs, on real frames.

One NightOwls track, nine annotated frames of a single pedestrian walking
toward the camera. Both detectors match her in exactly four of the nine, so
every frame-aggregate measure scores the two identically. They differ in how
the four matches are arranged: YOLO12-s never loses her for more than two
consecutive frames, YOLO26-s loses her for five. The difference lies in the
longest miss run, the quantity bounded by the paper's proposition on longest
miss runs.

Frames are night imagery. A fixed brightness lift is applied for print
legibility. It is applied identically to all nine frames and after detection,
so it changes what the reader can see and nothing that was measured.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance

import paths  # repository-relative results, DATA_ROOT for datasets
ROOT = paths.data("nightowls", "nightowls_validation")
RES = os.path.join(paths.RESULTS, "nightowls", "qualitative_track.json")
TRACK = json.load(open(RES))
REC = TRACK["frames"]
A, B = TRACK["detectors"]
INK, MUT = "#1F2933", "#6B7480"
HIT, MISS, GTC = "#1F4E79", "#B0673A", "#F2C14E"
ASPECT = 1.55                                # panel shape
MARGIN = 1.7                                # crop height as a multiple of box height


def crop_of(im, box):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    ch = h * MARGIN
    cw = ch / ASPECT
    l, t = cx - cw / 2.0, cy - ch / 2.0
    l = min(max(0, l), im.width - cw)
    t = min(max(0, t), im.height - ch)
    return im.crop((int(l), int(t), int(l + cw), int(t + ch))), l, t


plt.rcParams.update({"font.family": "sans-serif", "font.size": 7,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.015})
n = len(REC)
COLW = 3.42 / n                              # single-column width
fig = plt.figure(figsize=(3.42, COLW * ASPECT + 0.40))
gs = fig.add_gridspec(3, n, height_ratios=[COLW * ASPECT, 0.15, 0.15],
                      hspace=0.14, wspace=0.05,
                      left=0.115, right=0.995, top=0.985, bottom=0.03)

for i, r in enumerate(REC):
    ax = fig.add_subplot(gs[0, i])
    im = Image.open(os.path.join(ROOT, r["file_name"])).convert("RGB")
    c, l, t = crop_of(im, r["bbox"])
    c = ImageEnhance.Brightness(c).enhance(1.9)
    ax.imshow(c)
    x, y, w, h = r["bbox"]
    ax.add_patch(Rectangle((x - l, y - t), w, h, fill=False,
                           edgecolor=GTC, linewidth=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#C9D6E4"); s.set_linewidth(0.4)

LABEL = {"yolo12s": "YOLO12-s", "yolo26s": "YOLO26-s"}
for row, key in enumerate([A, B], start=1):
    lab = LABEL[key]
    for i, r in enumerate(REC):
        ax = fig.add_subplot(gs[row, i])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ok = r[key]
        ax.add_patch(Rectangle((0.06, 0.06), 0.88, 0.88,
                               facecolor=HIT if ok else "#FFFFFF",
                               edgecolor=HIT if ok else MISS, linewidth=0.85))
        if i == 0:
            ax.text(-0.22, 0.5, lab, transform=ax.transAxes, ha="right", va="center",
                    fontsize=6.4, color=INK, fontweight="600")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

out = os.path.join(paths.FIGURES, "Exp_qualitative.pdf")
fig.savefig(out, format="pdf"); plt.close(fig)
print("->", out)
