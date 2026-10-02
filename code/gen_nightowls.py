# NightOwls: frame-level accuracy against track-level coverage, all 46 detectors.
#
# Values are read from the result file rather than pasted, so the figure cannot
# drift from the table beside it. At 46 points not every marker can carry a
# label, so only the detectors named in the text are labelled: the two extremes
# and the pair that best demonstrates the inversion. That pair is chosen from
# the data, not fixed in advance.
import json, os, sys

import paths  # repository-relative results, DATA_ROOT for datasets
RES = os.path.join(paths.RESULTS, "nightowls", "nightowls_results46.json")
PRETTY = {"faster_rcnn_r50_fpn_v2": "Faster R-CNN", "retinanet_r50_fpn_v2": "RetinaNet",
          "fcos_r50_fpn": "FCOS", "ssd300_vgg16": "SSD300", "ssdlite320_mnv3": "SSDLite320",
          "rtdetr_l": "RT-DETR-L", "rtdetr_x": "RT-DETR-X"}


def label(m):
    if m in PRETTY:
        return PRETTY[m]
    for fam, tag in [("yolov10", "YOLOv10"), ("yolov5", "YOLOv5"), ("yolov8", "YOLOv8"),
                     ("yolov9", "YOLOv9"), ("yolov3", "YOLOv3"), ("yolo11", "YOLO11"),
                     ("yolo12", "YOLO12"), ("yolo26", "YOLO26")]:
        if m.startswith(fam):
            return tag + "-" + m[len(fam):].lstrip("_")
    return m


d = json.load(open(RES))
fl, tp = d["frame_level"], d["temporal"]
NTRACK = d["n_tracks_evaluable_2plus"]
M = [(label(m), fl[m]["MR"], tp[m]["first_detection_delay_frames"]["never_detected"])
     for m in fl]
M.sort(key=lambda r: r[1])

# The five torchvision detectors run at their own default input sizes, so on
# NightOwls they see 1280x800 or 300/320 square while the 41 Ultralytics models
# see 640x400. Comparing across that gap measures resolution, not the detector,
# so the highlighted pair is chosen only from the matched-resolution set. The
# unmatched five are still plotted, with a different marker, because the
# resolution effect is informative in its own right.
NON640 = {"Faster R-CNN", "RetinaNet", "FCOS", "SSD300", "SSDLite320"}
matched = [m for m in M if m[0] not in NON640]

best, pair = 0.0, None
for i in range(len(matched)):
    for j in range(i + 1, len(matched)):
        if abs(matched[i][1] - matched[j][1]) > 0.5:
            continue
        lo, hi = sorted((matched[i][2], matched[j][2]))
        if lo >= 1 and hi / lo > best:
            best, pair = hi / lo, (matched[i], matched[j])
if pair is None:
    pair = (matched[0], matched[1])
A, B = pair
HILITE = {A[0], B[0]}
NAMED = HILITE | {M[0][0], M[-1][0]}

BLUE = "#2B5C92"; AMBER = "#B0673A"
W, H = 620, 470
L, R, T, B_ = 74, 26, 86, 74
PW, PH = W - L - R, H - T - B_
xs = [m[1] for m in M]; ys = [m[2] for m in M]
X0, X1 = 0, (int(max(xs) / 10) + 1) * 10
Y0, Y1 = 0, (int(max(ys) / 25) + 1) * 25
sx = lambda v: L + PW * (v - X0) / (X1 - X0)
sy = lambda v: T + PH * (1 - (v - Y0) / (Y1 - Y0))

o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
     f'<text x="{L+PW/2}" y="30" text-anchor="middle" font-size="15.5" font-weight="600" fill="#1F2933">'
     f'Frame accuracy does not predict</text>',
     f'<text x="{L+PW/2}" y="49" text-anchor="middle" font-size="15.5" font-weight="600" fill="#1F2933">'
     f'whether a pedestrian is ever seen</text>',
     f'<text x="{L+PW/2}" y="68" text-anchor="middle" font-size="9.6" fill="#6B7480">'
     f'{d["n_images"]:,} images, {len(M)} detectors, {NTRACK} tracks</text>']

o.append('<g stroke="#EDF0F3" stroke-width="0.9">')
for v in range(X0 + 10, X1 + 1, 10):
    o.append(f'<line x1="{sx(v):.1f}" y1="{T}" x2="{sx(v):.1f}" y2="{T+PH}"/>')
for v in range(Y0, Y1 + 1, 25):
    o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{L+PW}" y2="{sy(v):.1f}"/>')
o.append('</g>')
o.append(f'<line x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}" stroke="#C3CAD2" stroke-width="1"/>')
o.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}" stroke="#C3CAD2" stroke-width="1"/>')
o.append('<g font-size="9.6" fill="#98A0AA" text-anchor="middle">')
for v in range(X0, X1 + 1, 10):
    o.append(f'<text x="{sx(v):.1f}" y="{T+PH+18}">{v}</text>')
o.append('</g><g font-size="9.6" fill="#98A0AA" text-anchor="end">')
for v in range(Y0, Y1 + 1, 25):
    o.append(f'<text x="{L-9}" y="{sy(v)+4:.1f}">{v}</text>')
o.append('</g>')
o.append(f'<text x="{L+PW/2:.1f}" y="{H-26}" text-anchor="middle" font-size="10.6" fill="#48515C">'
         f'Frame-level miss rate (%)</text>')
o.append(f'<text transform="translate(28,{T+PH/2:.1f}) rotate(-90)" text-anchor="middle" '
         f'font-size="10.6" fill="#48515C">Pedestrians never detected (of {NTRACK})</text>')

# The highlighted pair sits inside the dense lower-left cluster, so the
# annotation goes in the empty upper-left quadrant (low miss rate, many
# pedestrians never seen is a region no detector occupies) and reaches the
# pair with a leader line.
o.append(f'<line x1="{sx(A[1]):.1f}" y1="{sy(A[2]):.1f}" x2="{sx(B[1]):.1f}" y2="{sy(B[2]):.1f}" '
         f'stroke="{AMBER}" stroke-width="1.7" stroke-dasharray="3 2"/>')
cx, cy = sx(X0 + (X1 - X0) * 0.30), sy(Y1 * 0.78)
px, py = (sx(A[1]) + sx(B[1])) / 2, (sy(A[2]) + sy(B[2])) / 2
o.append(f'<path d="M {cx:.1f} {cy+16:.1f} C {cx:.1f} {(cy+py)/2:.1f}, '
         f'{px:.1f} {(cy+py)/2:.1f}, {px:.1f} {py-9:.1f}" fill="none" '
         f'stroke="{AMBER}" stroke-width="1.1" stroke-opacity="0.75"/>')
o.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="10.6" font-weight="600" '
         f'fill="{AMBER}">{A[0]} and {B[0]} differ by</text>')
o.append(f'<text x="{cx:.1f}" y="{cy+16:.1f}" font-size="10.6" font-weight="600" '
         f'fill="{AMBER}">{abs(A[1]-B[1]):.2f} miss-rate points,</text>')
o.append(f'<text x="{cx:.1f}" y="{cy+32:.1f}" font-size="10.6" font-weight="600" '
         f'fill="{AMBER}">and by {best:.1f} times in coverage</text>')

for name, mr, nd in M:
    hi = name in HILITE
    if name in NON640:                      # different input size, open square
        o.append(f'<rect x="{sx(mr)-4.5:.1f}" y="{sy(nd)-4.5:.1f}" width="9" height="9" '
                 f'fill="none" stroke="{BLUE}" stroke-width="1.5" stroke-opacity="0.8"/>')
        continue
    o.append(f'<circle cx="{sx(mr):.1f}" cy="{sy(nd):.1f}" r="{6.5 if hi else 4}" '
             f'fill="{AMBER if hi else BLUE}" fill-opacity="{1.0 if hi else 0.55}" '
             f'stroke="#FFFFFF" stroke-width="1.2"/>')
# Explicit placement: the four named points sit close together in the cluster,
# so automatic offsets collide. Each is pushed to a side that is clear.
PLACE = {A[0]: (0, 20, "middle"), B[0]: (0, -12, "middle"),
         M[0][0]: (0, -12, "middle"), M[-1][0]: (-13, 4, "end")}
for name, mr, nd in M:
    if name not in NAMED:
        continue
    dx, dy, anc = PLACE.get(name, (0, -12, "middle"))
    o.append(f'<text x="{sx(mr)+dx:.1f}" y="{sy(nd)+dy:.1f}" text-anchor="{anc}" '
             f'font-size="10" font-weight="600" fill="#2A333D">{name}</text>')

o.append(f'<text x="{L+PW/2}" y="{H-19}" text-anchor="middle" font-size="8.4" fill="#98A0AA">'
         f'Operating point 0.5. Filled circles are the 41 detectors run at a 640 px long side. Open squares are the five</text>')
o.append(f'<text x="{L+PW/2}" y="{H-7}" text-anchor="middle" font-size="8.4" fill="#98A0AA">'
         f'run at their own default input size. Only the extremes and the highlighted pair are labelled.</text>')
o.append('</svg>')

out = os.path.join(paths.FIGURES, "Exp_nightowls_coverage.svg")
open(out, "w").write("\n".join(o))
print(f"{len(M)} detectors plotted")
print(f"inversion pair: {A[0]} MR={A[1]} never={A[2]}  vs  {B[0]} MR={B[1]} never={B[2]}  "
      f"factor {best:.2f}")
print(f"labelled: {sorted(NAMED)}")
