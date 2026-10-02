# Conceptual figure, drawn at single-column width.
# The column headers are rotated and the cells kept compact, so the 7x7 grid
# forms a near-square block that reads at column width.
STRONG="#1F4E79"; COND="#8FAFCB"; LIM="#E8EDF3"
INK="#1F2933"; MUT="#6B7480"; LINE="#FFFFFF"
rows=["Two-stage","One-stage","Anchor-free","Query transformer","Lightweight",
      "Multimodal","Foundation-assisted"]
cols=["Occlusion","Crowding","Small scale","Adverse sensing","Annotation cost",
      "Privacy","Efficiency"]
# 0 limited, 1 conditional, 2 strong
M=[[1,1,1,0,0,0,1],
   [1,1,1,0,0,0,2],
   [2,1,1,0,0,0,1],
   [2,2,1,1,0,0,1],
   [0,0,1,0,0,0,2],
   [2,1,1,2,0,1,1],
   [1,1,1,1,2,1,0]]
FILL=[LIM,COND,STRONG]
W,H=600,556
L,T_=176,132
CW,CH=(W-L-16)/len(cols),46
o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
   f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
   f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
   f'<text x="{W/2}" y="30" text-anchor="middle" font-size="16.5" font-weight="700" fill="{INK}">'
   f'Challenge-method support</text>',
   f'<text x="{W/2}" y="50" text-anchor="middle" font-size="16.5" font-weight="700" fill="{INK}">'
   f'under stated protocols</text>']
for j,c in enumerate(cols):
    x=L+j*CW+CW/2
    o.append(f'<text transform="translate({x:.1f},{T_-10}) rotate(-42)" text-anchor="start" '
             f'font-size="10.4" fill="{INK}">{c}</text>')
for i,r in enumerate(rows):
    y=T_+i*CH
    o.append(f'<text x="{L-12}" y="{y+CH/2+4:.1f}" text-anchor="end" font-size="11" '
             f'fill="{INK}">{r}</text>')
    for j in range(len(cols)):
        x=L+j*CW
        o.append(f'<rect x="{x+1.5:.1f}" y="{y+1.5:.1f}" width="{CW-3:.1f}" height="{CH-3:.1f}" '
                 f'rx="4" fill="{FILL[M[i][j]]}"/>')
ly=T_+len(rows)*CH+30
for k,(lab,col) in enumerate([("limited",LIM),("conditional",COND),("strong",STRONG)]):
    lx=L-40+k*140
    o.append(f'<rect x="{lx}" y="{ly-11}" width="26" height="15" rx="3.5" fill="{col}"/>')
    o.append(f'<text x="{lx+32}" y="{ly+1}" font-size="10.6" fill="{INK}">{lab}</text>')
o.append(f'<text x="{W/2}" y="{H-30}" text-anchor="middle" font-size="9.6" fill="{MUT}">'
         f'Colour indicates qualitative support under stated protocols.</text>')
o.append(f'<text x="{W/2}" y="{H-16}" text-anchor="middle" font-size="9.6" fill="{MUT}">'
         f'Pooling across incompatible metrics is avoided.</text>')
o.append('</svg>')
import os, paths
open(os.path.join(paths.FIGURES, "TFig3_challenge.svg"),"w").write("\n".join(o))
print("written")
