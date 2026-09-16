"""Draws Fig. 1, the research design, as a compact two-column diagram.

Writes fig1_research_design.pdf, .svg, and .png (600 dpi) to figures/. Text is
set in Times New Roman from the Windows font folder, and the PNG is rasterised from the PDF with pdf2image, which needs poppler's pdftoppm on
the PATH (on Windows, MiKTeX ships one under miktex/bin/x64).

Every label is width-checked against its box at draw time, so a wording change
that no longer fits raises an assertion instead of overflowing silently.
"""
from pathlib import Path
from html import escape
import math, json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
from pdf2image import convert_from_path

OUT = Path(__file__).resolve().parents[1] / "figures"
STEM = OUT / 'fig1_research_design'
W, H = 174, 122
pdfmetrics.registerFont(TTFont('TimesPaper', r'C:\Windows\Fonts\times.ttf'))
pdfmetrics.registerFont(TTFont('TimesPaperBold', r'C:\Windows\Fonts\timesbd.ttf'))
c = canvas.Canvas(str(STEM.with_suffix('.pdf')), pagesize=(W*mm,H*mm))
c.setTitle('Research design and evaluation of cooperative membership information')
c.setAuthor('Akbar Indosatrya and Ericks Rachmat Swedia')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
     '<title>Research design and evaluation of cooperative membership information</title>',
     '<desc>A compact two-column flowchart. Data, paired feature sets, a shared training protocol, and out-of-fold predictions appear on the left. Additional analyses refit models; main inference and practical evaluations use generated predictions.</desc>',
     '<rect width="174" height="122" fill="white"/>']
checks=[]

def rect(x,y,w,h,fill='white',stroke=True):
    c.setFillColor(fill)
    c.setStrokeColor('#202020');c.setLineWidth(.55)
    c.rect(x*mm,(H-y-h)*mm,w*mm,h*mm,fill=1,stroke=int(stroke))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{"#202020" if stroke else "none"}" stroke-width="0.194"/>')

def text(x,y,s,size=9,bold=False,align='center',width=None):
    f='TimesPaperBold' if bold else 'TimesPaper'
    actual=pdfmetrics.stringWidth(s,f,size)/mm
    if width is not None:
        assert actual <= width, (s,actual,width)
    checks.append({'text':s,'font_pt':size,'width_mm':round(actual,2)})
    c.setFillColor('#111111');c.setFont(f,size)
    (c.drawCentredString if align=='center' else c.drawString)(x*mm,(H-y)*mm,s)
    svg.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size/mm:.4f}" font-weight="{"bold" if bold else "normal"}" text-anchor="{"middle" if align=="center" else "start"}" fill="#111111">{escape(s)}</text>')

def arrow(points,head=True):
    c.setStrokeColor('#333333');c.setFillColor('#333333');c.setLineWidth(.7)
    p=c.beginPath();p.moveTo(points[0][0]*mm,(H-points[0][1])*mm)
    for x,y in points[1:]:p.lineTo(x*mm,(H-y)*mm)
    c.drawPath(p,stroke=1)
    svg.append('<polyline points="'+' '.join(f'{xx},{yy}' for xx,yy in points)+'" fill="none" stroke="#333333" stroke-width="0.247"/>')
    if not head: return
    x,y=points[-1];px,py=points[-2]
    dx,dy=x-px,y-py;n=math.hypot(dx,dy);dx/=n;dy/=n
    a=1.25;b=.55
    tri=[(x,y),(x-a*dx+b*dy,y-a*dy-b*dx),(x-a*dx-b*dy,y-a*dy+b*dx)]
    p=c.beginPath();p.moveTo(tri[0][0]*mm,(H-tri[0][1])*mm)
    for xx,yy in tri[1:]:p.lineTo(xx*mm,(H-yy)*mm)
    p.close();c.drawPath(p,stroke=0,fill=1)
    svg.append('<polygon points="'+' '.join(f'{xx},{yy}' for xx,yy in tri)+'" fill="#333333"/>')

def panel(x,y,w,h,title):
    rect(x,y,w,h)
    rect(x+.1,y+.1,w-.2,6.7,'#F1F1F1',False)
    text(x+w/2,y+4.6,title,9.5,True,width=w-4)

# Connectors are drawn first so clean box boundaries remain in front.
arrow([(42.5,19),(42.5,23),(21.5,23),(21.5,27)])
arrow([(42.5,23),(63.5,23),(63.5,27)])
arrow([(21.5,48),(21.5,52),(42.5,52),(42.5,55)])
arrow([(63.5,48),(63.5,52),(42.5,52)],head=False)
arrow([(42.5,100),(42.5,105)])
# Refitting analyses branch from the protocol, not from fixed OOF predictions.
arrow([(82,70),(85.5,70),(85.5,17),(92,17)])
# The generated predictions feed both inference and practical evaluation.
arrow([(82,111.5),(89,111.5),(89,57),(92,57)])
arrow([(89,111.5),(92,111.5)])

panel(3,3,79,16,'Cooperative member records')
text(42.5,13.8,'2,000 members, 608 defaults within 24 months',9,width=75)

panel(3,27,37,21,'Model A')
text(21.5,38,'14 conventional',9,width=33)
text(21.5,42,'borrower and loan',9,width=33)
text(21.5,46,'variables',9,width=33)
panel(45,27,37,21,'Model B')
text(63.5,38,'The same 14 variables',9,width=33)
text(63.5,42,'plus 9 membership',9,width=33)
text(63.5,46,'variables',9,width=33)

panel(3,55,79,45,'Shared training and validation')
for yy,s in [(65.6,'One-hot encoding and standardization'),(69.2,'fitted on training data only'),(73.6,'Stratified 5-fold cross-validation'),(77.2,'repeated across five seeds'),(81.6,'Logistic regression, decision tree,'),(85.2,'random forest, gradient boosting,'),(88.8,'and support vector machine'),(93.2,'Matched settings within each algorithm'),(96.8,'and the same splits for Model A and Model B')]:
    text(42.5,yy,s,9,width=75)

panel(3,105,79,13,'Out-of-fold predictions')
text(42.5,115.5,'Paired by member and seed',9,width=75)

panel(92,3,79,30,'Additional analyses with model refitting')
for yy,s in [(14,'Permutation of membership variables'),(18.6,'Exclusion of payment lateness'),(23.2,'Leave-one-city-out validation'),(27.8,'Membership-only models')]:
    text(131.5,yy,s,9,width=75)

panel(92,40,79,45,'Main comparison and statistical inference')
for yy,s in [(51,'Accuracy, precision, recall, F1, AUC,'),(55,'and Brier score'),(60.4,'Paired bootstrap with 2,000 resamples'),(65,'DeLong test using primary seed 42')]:
    text(131.5,yy,s,9,width=75)
rect(92.1,68.3,78.8,16.6,'#F1F1F1',False)
text(131.5,73,'AUC gain supported only when',9,True,width=75)
text(131.5,77.7,'the bootstrap interval is entirely above zero',9,width=75)
text(131.5,82.3,'and the DeLong p-value is below 0.05',9,width=75)

panel(92,92,79,26,'Practical evaluation')
for yy,s in [(102.6,'Calibration'),(106.7,'Decision thresholds and error costs'),(110.8,'SMOTE versus threshold adjustment'),(114.9,'Flagging and errors across member groups')]:
    text(131.5,yy,s,9,width=75)

c.showPage();c.save()
svg.append('</svg>')
STEM.with_suffix('.svg').write_text('\n'.join(svg),encoding='utf8')
png=convert_from_path(str(STEM.with_suffix('.pdf')),dpi=600)[0]
png.save(STEM.with_suffix('.png'),dpi=(600,600))
preview=convert_from_path(str(STEM.with_suffix('.pdf')),dpi=170)[0]
preview.save(OUT/'fig1_research_design_preview.png')
print(json.dumps({'width_mm':W,'height_mm':H,'png_px':png.size,'png_dpi':600,'min_font_pt':min(x['font_pt'] for x in checks),'text_width_checks':len(checks),'outputs':[str(STEM.with_suffix(ext)) for ext in ['.png','.svg','.pdf']]},indent=2))
