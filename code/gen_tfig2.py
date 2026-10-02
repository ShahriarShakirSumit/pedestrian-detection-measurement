# Conceptual figure, drawn at single-column width.
# The three eras are stacked vertically rather than run left to right, which
# keeps every box at a readable size in one column.
BLUE="#1F4E79"; MID="#4A7BA7"; PALE="#EFF3F8"; LINE="#C9D6E4"
INK="#1F2933"; MUT="#6B7480"
W,H=600,560
L,R,T=28,28,74
eras=[("classical priors",  ["Sliding\nwindows","Proposal\nrefinement"], "2005-10","2011-15"),
      ("dense prediction",  ["Dense\nanchors","Centre /\nkeypoint"],     "2016-19","2019-22"),
      ("context and openness",["Query set\nprediction","Open-world\ngrounding"],"2020-24","2023-26")]
PW=W-L-R
bandh=132; gap=18
o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
   f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
   f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
   f'<text x="{W/2}" y="30" text-anchor="middle" font-size="16.5" font-weight="700" fill="{INK}">'
   f'Evolution of localisation mechanisms</text>',
   f'<text x="{W/2}" y="52" text-anchor="middle" font-size="10.5" fill="{MUT}">'
   f'Three eras, each replacing the previous assumption</text>',
   f'<text x="{W/2}" y="66" text-anchor="middle" font-size="10.5" fill="{MUT}">'
   f'about how candidates are proposed</text>']
for i,(title,boxes,y1,y2) in enumerate(eras):
    by=T+i*(bandh+gap)
    o.append(f'<rect x="{L}" y="{by}" width="{PW}" height="{bandh}" rx="7" fill="{PALE}"/>')
    o.append(f'<text x="{L+PW/2}" y="{by+21}" text-anchor="middle" font-size="11.5" '
             f'font-weight="600" fill="{BLUE}">{title}</text>')
    bw=(PW-54)/2; bh=76
    for j,lab in enumerate(boxes):
        bx=L+18+j*(bw+18)
        o.append(f'<rect x="{bx}" y="{by+32}" width="{bw}" height="{bh}" rx="6" '
                 f'fill="#FFFFFF" stroke="{LINE}" stroke-width="1"/>')
        o.append(f'<rect x="{bx}" y="{by+32}" width="{bw}" height="3.4" rx="1.7" fill="{BLUE}"/>')
        for k,line in enumerate(lab.split("\n")):
            o.append(f'<text x="{bx+bw/2}" y="{by+64+k*16}" text-anchor="middle" '
                     f'font-size="12.5" font-weight="600" fill="{BLUE}">{line}</text>')
        o.append(f'<text x="{bx+bw/2}" y="{by+bh+50}" text-anchor="middle" font-size="9.6" '
                 f'fill="{MUT}">{[y1,y2][j]}</text>')
        if j==0:
            ax=bx+bw+4
            o.append(f'<path d="M {ax} {by+70} L {ax+10} {by+70}" stroke="{MID}" '
                     f'stroke-width="1.4" marker-end="url(#a)"/>')
    if i<len(eras)-1:
        cx=L+PW/2; ay=by+bandh+2
        o.append(f'<path d="M {cx} {ay} L {cx} {ay+14}" stroke="{MID}" stroke-width="1.6" '
                 f'marker-end="url(#a)"/>')
o.insert(1,f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
           f'markerHeight="5" orient="auto-start-reverse"><path d="M 0 1.5 L 9 5 L 0 8.5 z" '
           f'fill="{MID}"/></marker></defs>')
o.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="9.4" fill="{MUT}">'
         f'Mechanism-level map. Numerical rankings require matched protocols.</text>')
o.append('</svg>')
import os, paths
open(os.path.join(paths.FIGURES, "TFig2_detector_map.svg"),"w").write("\n".join(o))
print("written")
