
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def save_simple_pdf(path, title, summary_pairs):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    x = 2*cm
    y = H - 2.5*cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 1.2*cm
    c.setFont("Helvetica", 10)
    for k,v in summary_pairs:
        line = f"{k}: {v}"
        c.drawString(x, y, line[:110])
        y -= 0.6*cm
        if y < 2*cm:
            c.showPage()
            y = H - 2*cm
            c.setFont("Helvetica", 10)
    c.showPage()
    c.save()
