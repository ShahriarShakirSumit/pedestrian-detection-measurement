# Conceptual figure, drawn at single-column width.
# The seven pipeline stages are laid out in two columns with the evidence list
# beneath, which fits one column without shrinking any label.
BLUE="#1F4E79"; MID="#4A7BA7"; PALE="#EFF3F8"; LINE="#C9D6E4"
INK="#1F2933"; MUT="#6B7480"; ACC="#B0673A"
stages=[("Sensing","cameras, sensors"),("Pre-process","resize, tile, align"),
        ("Inference","detector, fusion"),("Decode","NMS or queries"),
        ("Decision","track, alert, review"),("Monitor","drift, errors, audit"),
        ("Update","retrain, rollback")]
ev=[("Pre-process","Calibration and synchronisation"),
    ("Inference","Runtime boundary: hardware, precision, batch"),
    ("Decision","Risk evidence: false negatives, thresholds"),
    ("Monitor","Governance: privacy, access, versioning")]
W,H=600,508
L,R,T_=26,26,96
PW=W-L-R; bw=(PW-16)/2; bh=54; vg=11
o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
   f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
   f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
   f'orient="auto-start-reverse"><path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{MID}"/></marker></defs>',
   f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
   f'<text x="{W/2}" y="28" text-anchor="middle" font-size="15.5" font-weight="700" fill="{INK}">'
   f'A benchmark result covers one stage.</text>',
   f'<text x="{W/2}" y="48" text-anchor="middle" font-size="15.5" font-weight="700" fill="{INK}">'
   f'A deployment claim covers seven.</text>',
   f'<text x="{W/2}" y="70" text-anchor="middle" font-size="10.2" fill="{MUT}">'
   f'Reported accuracy speaks only to inference. Every other stage</text>',
   f'<text x="{W/2}" y="84" text-anchor="middle" font-size="10.2" fill="{MUT}">'
   f'carries its own failure mode and its own evidence.</text>']
for i,(nm,sub) in enumerate(stages):
    col,row=i%2,i//2
    x=L+col*(bw+16); y=T_+row*(bh+vg)
    hot = nm=="Inference"
    o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" '
             f'fill="{"#FBF1EC" if hot else "#FFFFFF"}" stroke="{ACC if hot else LINE}" '
             f'stroke-width="{1.6 if hot else 1}"/>')
    o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="3.2" rx="1.6" '
             f'fill="{ACC if hot else BLUE}"/>')
    o.append(f'<text x="{x+bw/2}" y="{y+25}" text-anchor="middle" font-size="12.5" '
             f'font-weight="600" fill="{ACC if hot else BLUE}">{nm}</text>')
    o.append(f'<text x="{x+bw/2}" y="{y+42}" text-anchor="middle" font-size="9.8" '
             f'fill="{MUT}">{sub}</text>')
    if i < len(stages) - 1:
        if col == 0:                      # straight across to the right-hand box
            o.append(f'<path d="M {x+bw+2} {y+bh/2} L {x+bw+12} {y+bh/2}" stroke="{MID}" '
                     f'stroke-width="1.3" marker-end="url(#a)"/>')
        else:                             # elbow back to the left box one row down
            x0, y0 = x + bw / 2, y + bh
            x1, y1 = L + bw / 2, y + bh + vg
            o.append(f'<path d="M {x0} {y0+1} L {x0} {y0+vg/2} L {x1} {y0+vg/2} '
                     f'L {x1} {y1-2}" fill="none" stroke="{MID}" stroke-width="1.3" '
                     f'marker-end="url(#a)"/>')
ry=T_+4*(bh+vg)+6
o.append(f'<text x="{W/2}" y="{ry+2}" text-anchor="middle" font-size="9.6" fill="{MUT}">'
         f'monitoring feeds back into sensing and retraining</text>')
by=ry+22
o.append(f'<rect x="{L}" y="{by}" width="{PW}" height="{46+len(ev)*22+8}" rx="7" fill="{PALE}"/>')
o.append(f'<text x="{W/2}" y="{by+22}" text-anchor="middle" font-size="12" font-weight="600" '
         f'fill="{BLUE}">what a deployment claim must cover</text>')
for k,(st,txt) in enumerate(ev):
    yy=by+46+k*22
    o.append(f'<circle cx="{L+20}" cy="{yy-4}" r="2.6" fill="{MID}"/>')
    o.append(f'<text x="{L+32}" y="{yy}" font-size="10.4" fill="{INK}">'
             f'<tspan font-weight="600">{st}</tspan>  {txt}</text>')
o.append('</svg>')
import os, paths
open(os.path.join(paths.FIGURES, "TFig9_deployment.svg"),"w").write("\n".join(o))
print("written")
