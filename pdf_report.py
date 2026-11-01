import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

_REPL = {"\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "-",
         "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}

def _c(text) -> str:
    """Special characters saaf karo (LLM ke fancy dash PDF me black box bante hain)."""
    text = str(text)
    for a, b in _REPL.items():
        text = text.replace(a, b)
    return escape(text.encode("latin-1", "replace").decode("latin-1"))

def _step(s: str) -> str:
    return re.sub(r"^\s*\d+[\.\)]\s*", "", s).strip()

def _table(rows):
    styles = getSampleStyleSheet()
    rows = [[k, Paragraph(str(v), styles["Normal"])] for k, v in rows]
    t = Table(rows, colWidths=[55 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t

def build_pdf(target, inputs: dict, analysis, result):
    """target = file path ya BytesIO object."""
    ss = getSampleStyleSheet()
    doc = SimpleDocTemplate(target, pagesize=A4, title="Land Preparation Report",
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    s = []
    s.append(Paragraph("Land Preparation Report", ss["Title"]))
    s.append(Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M"), ss["Normal"]))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Inputs", ss["Heading2"]))
    s.append(_table([[_c(k), _c(v)] for k, v in inputs.items()]))

    s.append(Paragraph("Land Analysis (from photo)", ss["Heading2"]))
    s.append(_table([
        ["Terrain", _c(analysis.terrain)],
        ["Vegetation", _c(f"{analysis.vegetation} ({analysis.vegetation_density})")],
        ["Surface wetness", _c(analysis.surface_wetness)],
        ["Obstacles", _c(", ".join(analysis.obstacles) or "none")],
    ]))

    s.append(Paragraph("Recommendation", ss["Heading2"]))
    s.append(_table([
        ["Tractor", _c(result.tractor_hp_range)],
        ["Estimated time", _c(f"{result.estimated_time_hours} hours")],
        ["Equipment", _c(", ".join(result.equipment) or "-")],
        ["Confidence", _c(result.confidence)],
    ]))

    s.append(Paragraph("Steps", ss["Heading2"]))
    for i, st_ in enumerate(result.steps, 1):
        s.append(Paragraph(f"{i}. {_c(_step(st_))}", ss["Normal"]))
        s.append(Spacer(1, 3))

    if result.warnings:
        s.append(Paragraph("Warnings", ss["Heading2"]))
        for w in result.warnings:
            s.append(Paragraph(f"- {_c(w)}", ss["Normal"]))

    s.append(Paragraph("Reasoning", ss["Heading2"]))
    s.append(Paragraph(_c(result.reasoning), ss["Normal"]))

    doc.build(s)
    return target