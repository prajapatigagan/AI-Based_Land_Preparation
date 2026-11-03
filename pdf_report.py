from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from tractor_web import fmt_hp, format_price

GREEN = colors.HexColor("#075638")
_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Title"], textColor=GREEN, fontSize=20)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], textColor=GREEN, spaceBefore=10)
BODY = ParagraphStyle("B", parent=_ss["BodyText"], fontSize=10, leading=14)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, textColor=colors.grey)

# Helvetica ke unsupported special characters ko safe characters me badalte hain
_REPL = {
    "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2022": "-", "\u00a0": " ", "\u2192": "->", "\u2248": "~", "\u20b9": "Rs. ",
}


def _clean(text) -> str:
    text = str(text)
    for k, v in _REPL.items():
        text = text.replace(k, v)
    return text.encode("cp1252", "replace").decode("cp1252")  # baaki unsupported chars -> '?'


def _p(text, style=BODY):
    return Paragraph(escape(_clean(text)), style)


def _table(rows):
    t = Table([[_p(k), _p(v)] for k, v in rows], colWidths=[50 * mm, 120 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe9df")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f8ed")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(buf, inputs, analysis, result, tractors):
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="KhetAI Land Report")
    s = [_p("KhetAI - Land Preparation Report", H1)]

    s += [_p("Inputs", H2), _table([(k, v) for k, v in inputs.items()])]
    s += [_p("Land analysis (from photos)", H2), _table([
        ("Terrain", analysis.terrain),
        ("Vegetation", f"{analysis.vegetation} ({analysis.vegetation_density})"),
        ("Surface wetness", analysis.surface_wetness),
        ("Stones visible", "Yes" if analysis.stones_visible else "No"),
        ("Obstacles", ", ".join(analysis.obstacles) or "None visible"),
        ("Photo confidence", analysis.confidence),
    ])]
    s += [_p("Summary", H2), _table([
        ("Tractor power", result.tractor_hp_range),
        ("Estimated time", f"{result.estimated_time_hours} hr"),
        ("Confidence", result.confidence),
    ])]

    s += [_p("Recommended equipment (in order)", H2)]
    s += [_p(f"{i}. {e}") for i, e in enumerate(result.equipment, 1)]

    s += [_p("Preparation steps", H2)]
    s += [_p(f"{i}. {x}") for i, x in enumerate(result.steps, 1)]

    if tractors:
        s += [_p("Tractor examples (online search)", H2)]
        for t in tractors:
            line = f"{t['brand']} {t['model']} - {fmt_hp(t['hp'])} HP - {format_price(t)}"
            s.append(_p(line))
            if t.get("source_url"):
                s.append(_p(f"Source: {t['source_url']} (checked {t.get('source_checked_on', '')})", SMALL))
        s.append(_p("Prices can change; confirm with a local dealer.", SMALL))

    if result.warnings:
        s += [_p("Warnings", H2)] + [_p(f"- {w}") for w in result.warnings]

    s += [_p("Why this recommendation", H2), _p(result.reasoning), Spacer(1, 8)]

    if getattr(result, "tools_used", None):
        s += [_p("AI agent tools used", H2), _p(", ".join(result.tools_used))]
    if getattr(result, "sources", None):
        s += [_p("Sources (retrieved via RAG)", H2)] + [_p(u, SMALL) for u in result.sources]

    s.append(Spacer(1, 8))
    s.append(_p("Always confirm with your local agronomist before hiring machinery. "
                "Time and HP are estimates based on plot size and standard implement speeds.", SMALL))
    doc.build(s)