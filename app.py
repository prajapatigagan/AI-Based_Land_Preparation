import html
import os
import re
import tempfile
from io import BytesIO

import streamlit as st

from tools import UNIT_TO_ACRE, to_acres, rule_shortlist
from vision import analyze_land, merge_analyses
from recommender import recommend
from pdf_report import build_pdf
from chatbot import build_context, stream_answer
from tractor_web import get_tractor_examples, fmt_hp, format_price

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KhetAI | Land Preparation Advisor",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- THEME / CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');
    :root {
      --forest:#075638; --forest-dark:#063f2d; --green:#16794f; --lime:#c8ef78;
      --cream:#f7f8f1; --ink:#123b2d; --muted:#668074; --line:#dfe9df;
      --pale:#edf5e9; --white:#fff;
    }
    html { scroll-behavior:smooth; }
    html, body, .stApp, .stMarkdown, input, textarea, label, button, [data-baseweb="select"] {
      font-family:'DM Sans',sans-serif;
    }
    .stApp { background:var(--cream); color:var(--ink); }
    .block-container { max-width:1440px; padding:1.1rem 2rem 3rem; }
    header[data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    .nav { display:flex; align-items:center; justify-content:space-between; padding:.35rem .2rem 1.2rem; }
    .logo { display:flex; align-items:center; gap:.7rem; color:var(--forest-dark); font-weight:800; font-size:1.55rem; letter-spacing:-.05em; }
    .logo-mark { width:44px;height:44px;border-radius:15px;display:grid;place-items:center;color:#fff;background:linear-gradient(145deg,#075638,#269760);font-size:1.25rem;box-shadow:0 8px 20px #07563825; }
    .logo span { color:#8bbd3d; }
    .nav-links { display:flex;gap:2.2rem;align-items:center;font-weight:700;color:#5d766b;font-size:.92rem; }
    .nav-links a { color:inherit;text-decoration:none; }
    .nav-links a:first-child { color:var(--forest);border-bottom:3px solid #8fbe3d;padding-bottom:.5rem; }
    .nav-pill { border:1px solid var(--line);border-radius:999px;padding:.75rem 1.15rem;background:#fff;color:var(--forest);font-weight:700; }
    .hero { position:relative;overflow:hidden;border-radius:30px;padding:4.2rem 4rem 3.7rem;min-height:430px;color:#fff;
      background:radial-gradient(circle at 86% 28%,#b9dc7b 0 5%,transparent 5.2%),
      radial-gradient(ellipse at 96% 105%,#a7ce5d 0 18%,transparent 18.2%),
      radial-gradient(ellipse at 90% 110%,#5c9e54 0 31%,transparent 31.2%),
      radial-gradient(ellipse at 93% 105%,#267e54 0 42%,transparent 42.2%),
      linear-gradient(115deg,#06432f,#0e6846); }
    .hero:after { content:"🚜";position:absolute;right:10%;bottom:11%;font-size:5.5rem;opacity:.9;filter:grayscale(1) brightness(.3);transform:scaleX(-1);pointer-events:none; }
    .eyebrow { color:#c8ef78;font-size:.78rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase;margin-bottom:1.4rem; }
    .hero h1 { color:#fff;font-family:'Playfair Display',serif;font-size:clamp(2.7rem,5vw,4.2rem);line-height:1.05;letter-spacing:-.035em;margin:0 0 1.1rem;max-width:650px; }
    .hero h1 em { color:var(--lime);font-style:normal; }
    .hero p { color:#d8e9df;font-size:1.08rem;line-height:1.65;max-width:590px;margin:0 0 1.8rem; }
    .hero-cta { display:inline-block;background:var(--lime);color:#073f2e!important;border-radius:14px;padding:1rem 1.4rem;font-weight:800;text-decoration:none; }
    .hero-note { display:inline-block;margin-left:1.2rem;color:#e2f0e7;font-size:.9rem; }

    /* AI field scan chip = clickable link to the chat assistant */
    .scan-chip, .scan-chip:visited { position:absolute;right:3%;top:18%;display:block;border:1px solid #ffffff55;background:#ffffff20;border-radius:16px;padding:1rem 1.2rem;color:#eaf5ed!important;font-size:.82rem;line-height:1.5;backdrop-filter:blur(8px);text-decoration:none;cursor:pointer;z-index:2;transition:transform .2s ease, background .2s ease, border-color .2s ease; }
    .scan-chip:hover { background:#ffffff38;border-color:#c8ef78;transform:translateY(-3px); }
    .scan-chip b { font-size:.95rem;color:#fff; }
    .scan-chip small { color:#c8ef78;font-weight:700;font-size:.78rem; }

    .stats { display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;padding:1.6rem .8rem 2rem;border-bottom:1px solid var(--line);margin-bottom:3rem; }
    .stat { display:flex;align-items:center;gap:.8rem;padding:.2rem 1rem;border-right:1px solid var(--line); }
    .stat:last-child {border:0}
    .stat-icon {font-size:1.35rem;color:var(--forest)}
    .stat b {display:block;color:var(--forest);font-size:1rem}
    .stat small {display:block;color:var(--muted);font-size:.82rem;margin-top:.2rem}
    .section-head { margin:0 0 1.6rem; scroll-margin-top:1rem; }
    .section-kicker { color:#78a932;font-size:.75rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.45rem; }
    .section-head h2 {font-family:'Playfair Display',serif;color:var(--forest-dark);font-size:2.45rem;line-height:1.15;margin:0 0 .5rem}
    .section-head p {color:var(--muted);margin:0;font-size:1rem}
    .section-title {font-family:'Playfair Display',serif;color:var(--forest-dark);font-size:1.8rem;font-weight:700;margin:.2rem 0 .2rem}
    .section-subtitle {color:var(--muted);font-size:.95rem;margin-bottom:1rem}
    .step-count {text-align:right;color:var(--muted);font-weight:700;padding-top:1.4rem}
    .step-count b {display:inline-grid;place-items:center;background:var(--forest);color:#fff;width:38px;height:38px;border-radius:50%;margin-right:.5rem}
    div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff;border:1px solid var(--line);border-radius:24px;box-shadow:0 8px 30px #153f2a0b;padding:.5rem}
    .form-step {display:flex;gap:.85rem;align-items:flex-start;margin:.35rem 0 1.15rem}
    .form-step .num {color:#7da936;font-weight:800;font-size:.9rem;padding-top:.15rem}
    .form-step b {display:block;color:#163d2e;font-size:1rem}
    .form-step small {color:var(--muted);font-size:.82rem}
    div[data-testid="stFileUploader"] section {background:#f7fbf4;border:2px dashed #8fbd86;border-radius:18px;padding:1rem}
    div[data-testid="stFileUploader"] section:hover {background:#eff8e9;border-color:#39885b}
    div[data-testid="stFileUploader"] button {background:#111!important;color:#fff!important;border:0!important;border-radius:8px!important}
    div[data-testid="stFileUploader"] label,div[data-testid="stFileUploader"] small,div[data-testid="stFileUploader"] p {color:#355b48!important;opacity:1!important}
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div {background:#fbfdf9;border:1px solid #dce8dc;border-radius:13px;min-height:48px}
    div[data-baseweb="select"] *,div[data-baseweb="input"] * {color:#163d2e}
    [data-testid="stWidgetLabel"] p {color:#597768!important;font-weight:700;font-size:.78rem}
    div.stButton>button,div.stDownloadButton>button {width:100%;border-radius:13px;font-weight:800;min-height:3.1rem}
    div.stButton>button[kind="primary"] {background:#a9c4ae;color:#fff;border:0}
    div.stButton>button[kind="primary"]:hover {background:#0b6845;color:#fff}
    div.stButton>button[kind="primary"]:disabled {background:#a9c4ae;color:#fff;opacity:.85}
    div.stButton>button[kind="secondary"] {background:#eef6e9;color:#0b5639;border:1px solid #d3e5cf;font-size:.82rem;min-height:2.7rem}
    div.stButton>button[kind="secondary"]:hover {background:#e1f0da;border-color:#8fbd86;color:#0b5639}
    div.stDownloadButton>button {background:#eaf4e5;color:#0b5639;border:1px solid #c8dfc7}
    .preview-panel {min-height:480px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:2rem 1rem;background:linear-gradient(145deg,#fbfdf8,#f3f8ed);border:1px solid var(--line);border-radius:24px}
    .preview-icon {width:90px;height:90px;border-radius:28px;background:#e5f2d8;display:grid;place-items:center;font-size:2.4rem;margin-bottom:1.1rem}
    .preview-panel h3 {font-family:'Playfair Display',serif;color:var(--forest-dark);font-size:1.7rem;margin:0 0 .5rem}
    .preview-panel p {color:var(--muted);max-width:380px;line-height:1.6}
    .preview-list {color:#567668;line-height:2.2;font-size:.9rem;text-align:left;margin-top:.8rem}
    .how-section {padding:5rem 0 4rem;text-align:center}
    .how-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem;margin-top:2rem;text-align:left}
    .how-card {background:#fff;border:1px solid var(--line);border-radius:22px;padding:1.6rem;min-height:180px}
    .how-card .ico {font-size:1.5rem;color:#7eaa39;margin-bottom:1rem}
    .how-card b {display:block;color:#163d2e;font-size:1.05rem;margin-bottom:.5rem}
    .how-card p {color:var(--muted);font-size:.9rem;line-height:1.6;margin:0}
    .report-title {font-family:'Playfair Display',serif;color:var(--forest-dark);font-size:1.8rem;font-weight:700}
    .metric-card {background:#f9fcf7;border:1px solid var(--line);border-radius:16px;padding:1rem;min-height:110px}
    .metric-top {display:flex;align-items:center;gap:.5rem}
    .metric-label {color:#648071;font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em}
    .metric-value {color:#0a5639;font-size:1.35rem;font-weight:800;margin-top:.55rem;overflow-wrap:anywhere}
    .metric-icon {font-size:1.1rem}
    .summary {background:#edf6e8;border:1px solid #d8e9d1;border-radius:14px;padding:.9rem 1rem;color:#174c34;margin:.7rem 0 1rem}
    .kv-grid,.logic-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem}
    .kv,.logic-item {background:#f8fbf6;border:1px solid var(--line);border-radius:14px;padding:.85rem;color:#1b4332}
    .kv span.k {display:block;color:#688174;font-size:.7rem;font-weight:800;text-transform:uppercase;margin-bottom:.3rem}
    .kv b {color:#163d2e}
    .badge {display:inline-block;border-radius:99px;padding:.12rem .6rem;font-size:.78rem;font-weight:800}
    .badge.ok {background:#def2df;color:#17653b}.badge.mid {background:#fff0cc;color:#8a5a00}.badge.bad {background:#fde1df;color:#a1261d}
    .meter {height:7px;background:#e6eee6;border-radius:99px;overflow:hidden;margin-top:.45rem}
    .meter>i {display:block;height:100%;background:#4b9c61}
    .equip-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem}
    .equip {display:flex;gap:.7rem;align-items:center;background:#f8fbf6;border:1px solid var(--line);border-radius:14px;padding:.8rem;color:#183f2e;font-weight:700}
    .t-item {display:flex;gap:.8rem;margin:.7rem 0}.t-num {background:#126342;color:#fff;border-radius:50%;width:30px;height:30px;display:grid;place-items:center;flex-shrink:0;font-weight:800}
    .t-body,.logic-card {background:#f8fbf6;border:1px solid var(--line);border-radius:14px;padding:.9rem;color:#244b38;line-height:1.6;flex:1}
    .logic-item b {display:block;color:#0c5638;margin-bottom:.3rem}
    .warn-card {background:#fff5df;border-left:4px solid #dfa522;color:#795600;padding:.8rem;border-radius:12px;margin:.5rem 0}
    /* Report tabs: force the complete label (including emoji-adjacent spans) to black */
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] *,
    [role="tab"],
    [role="tab"] * {
      color:#111111 !important;
      -webkit-text-fill-color:#111111 !important;
      opacity:1 !important;
      font-weight:800 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] *,
    [role="tab"][aria-selected="true"],
    [role="tab"][aria-selected="true"] * {
      color:#000000 !important;
      -webkit-text-fill-color:#000000 !important;
    }
    div[data-baseweb="tab-highlight"] {background:#83ad3d!important}
    .footer {border-top:1px solid var(--line);padding:1.7rem .2rem;color:#6b8377;font-size:.82rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}
    .footer strong {color:var(--forest);font-size:1.2rem}
    /* Chat assistant: force readable dark text on both user and assistant messages */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    .stChatMessage,
    .stChatMessage * { color:#183f2e !important; opacity:1 !important; }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea::placeholder { color:#183f2e !important; opacity:1 !important; }
    [data-testid="stChatInput"] textarea { background:#ffffff !important; caret-color:#075638; }
    @media(max-width:850px){.block-container{padding:1rem}.hero{padding:2.5rem 1.6rem;min-height:390px}.hero:after{display:none}.scan-chip{position:static;display:inline-block;margin-top:1.4rem}.stats{grid-template-columns:repeat(2,1fr)}.stat:nth-child(2){border:0}.how-grid{grid-template-columns:1fr}.nav-links{display:none}.section-head h2{font-size:2rem}.kv-grid,.logic-grid{grid-template-columns:1fr}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- CONSTANTS ----------------
SOILS = ["black", "clay", "loam", "sandy"]
CROPS = ["paddy", "wheat", "maize", "soybean", "cotton", "sugarcane", "mustard"]
MAX_PHOTOS = 4
SUGGESTIONS = [  # report ban chuka ho
    "Why was this equipment selected?",
    "How was the time estimate calculated?",
    "What should I do first?",
    "What do the warnings mean?",
]
GENERAL_SUGGESTIONS = [  # report se pehle
    "Which equipment suits black soil for paddy?",
    "How do I prepare land for wheat?",
    "What can I do about a waterlogged field?",
    "Which tractor HP is enough for 2 acres?",
]


# ---------------- HELPERS ----------------
def esc(x) -> str:
    return html.escape(str(x))


def clean_step(s: str) -> str:
    """Remove numbering generated by the LLM; UI adds its own numbering."""
    return re.sub(r"^\s*\d+[\.\)]\s*", "", s).strip()


def badge(text, kind):
    return f'<span class="badge {kind}">{esc(text)}</span>'


def level_kind(v: str) -> str:
    return {"high": "ok", "medium": "mid", "low": "bad"}.get(str(v).lower(), "mid")


def wet_kind(v: str) -> str:
    return {"dry": "ok", "moist": "mid", "waterlogged": "bad"}.get(str(v).lower(), "mid")


def level_pct(v: str) -> int:
    return {"high": 100, "medium": 66, "low": 33}.get(str(v).lower(), 50)


def wet_pct(v: str) -> int:
    return {"dry": 20, "moist": 55, "waterlogged": 95}.get(str(v).lower(), 50)


def meter(pct, kind):
    return f'<div class="meter {kind}"><i style="width:{pct}%"></i></div>'


def kv(label, value_html):
    return f'<div class="kv"><span class="k">{esc(label)}</span><b>{value_html}</b></div>'


def metric_card(label, value, icon, tone="green", extra=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-icon ic-{tone}">{icon}</div>
                <div class="metric-label">{esc(label)}</div>
            </div>
            <div class="metric-value">{esc(value)}</div>
            {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


def equip_icon(name: str) -> str:
    n = name.lower()
    if "plough" in n or "plow" in n:
        return "🚜"
    if "rotavator" in n or "rotary" in n or "tiller" in n:
        return "⚙️"
    if "harrow" in n or "cultivator" in n:
        return "🌀"
    if "level" in n or "planker" in n:
        return "📏"
    if "seed" in n or "drill" in n or "planter" in n:
        return "🌱"
    return "🛠️"


def analyze_uploaded(f):
    """Save one uploaded photo temporarily, analyze it, then remove it."""
    suffix = os.path.splitext(f.name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(f.getvalue())
        tmp.close()
        return analyze_land(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def generate(uploaded, soil, crop, size, unit):
    """Run the pipeline and store the report in session state."""
    st.session_state.pop("report", None)
    st.session_state["chat"] = []

    if not uploaded:
        st.error("Please upload at least one land photo.")
        return
    if not crop.strip():
        st.error("Please enter a crop name.")
        return

    bad_photo = False
    try:
        acres = to_acres(size, unit)

        with st.status("Analyzing your land...", expanded=True) as status:
            st.write(f"📷 Reading {len(uploaded)} photo(s)")
            analysis = merge_analyses([analyze_uploaded(f) for f in uploaded])

            if not analysis.image_quality_ok or analysis.confidence.lower() == "low":
                bad_photo = True
                status.update(label="Photo not clear enough", state="error", expanded=False)
            else:
                st.write("⚙️ Matching soil, crop and land condition")
                shortlist = rule_shortlist(analysis, soil, acres, crop)
                st.write("📝 Writing the preparation plan")
                result = recommend(analysis, shortlist, soil, acres, crop)
                st.write("🌐 Searching the web for tractor examples")
                tractors = get_tractor_examples(result.tractor_hp_range)
                st.write("📄 Building PDF report")
                inputs = {"Soil": soil, "Crop": crop, "Plot size (acres)": acres}
                buf = BytesIO()
                build_pdf(buf, inputs, analysis, result, tractors)
                status.update(label="Recommendation ready", state="complete", expanded=False)

        if bad_photo:
            st.warning(
                "The photo is unclear or does not look like farmland. "
                "Please upload clear daylight photos that show the whole plot."
            )
            return

        st.session_state["report"] = {
            "acres": acres,
            "inputs": inputs,
            "analysis": analysis,
            "result": result,
            "tractors": tractors,
            "pdf": buf.getvalue(),
            "context": build_context(inputs, analysis, shortlist, result, tractors),
        }
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")


def render_report(rep):
    analysis, result, acres, inputs = rep["analysis"], rep["result"], rep["acres"], rep["inputs"]

    st.markdown('<div class="section-title">Your land report</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="summary">🌾 Plan for <b>{acres:g} acre</b> of <b>{esc(inputs["Crop"])}</b> '
        f'on <b>{esc(inputs["Soil"])}</b> soil</div>',
        unsafe_allow_html=True,
    )

    conf_kind = level_kind(result.confidence)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Plot size", f"{acres:g} acre", "📐", "lime")
    with m2:
        metric_card("Tractor power", result.tractor_hp_range, "🚜", "green")
    with m3:
        metric_card("Estimated time", f"{result.estimated_time_hours} hr", "⏱️", "blue")
    with m4:
        metric_card("Confidence", result.confidence.capitalize(), "🎯", "amber",
                    meter(level_pct(result.confidence), conf_kind))

    if result.warnings:
        st.markdown("<br>", unsafe_allow_html=True)
        for w in result.warnings:
            st.markdown(f'<div class="warn-card"><span>⚠️</span><span>{esc(w)}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🌱 Land analysis", "🛠️ Equipment & steps", "🧠 Recommendation logic"])

    with tab1:
        st.markdown("#### What did the photos reveal?")
        obstacles = ", ".join(analysis.obstacles) or "None visible"
        wet = analysis.surface_wetness
        grid = "".join([
            kv("Terrain", esc(analysis.terrain)),
            kv("Vegetation", esc(analysis.vegetation)),
            kv("Vegetation density", esc(analysis.vegetation_density)),
            kv("Surface wetness", badge(wet, wet_kind(wet)) + meter(wet_pct(wet), wet_kind(wet))),
            kv("Visible obstacles", esc(obstacles)),
            kv("Photo confidence", badge(analysis.confidence, level_kind(analysis.confidence))
               + meter(level_pct(analysis.confidence), level_kind(analysis.confidence))),
        ])
        st.markdown(f'<div class="kv-grid">{grid}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Recommended equipment")
        if result.equipment:
            cards = "".join(
                f'<div class="equip"><div class="equip-icon">{equip_icon(e)}</div>'
                f'<div class="equip-name">{esc(e)}</div></div>'
                for e in result.equipment
            )
            st.markdown(f'<div class="equip-grid">{cards}</div>', unsafe_allow_html=True)
        else:
            st.write("No equipment recommendation available.")

        tractor_raw = rep.get("tractors")
        tractors = tractor_raw or []
        if tractors:
            st.markdown("#### Tractor examples")
            def _src(t):
                url = str(t.get("source_url") or "")
                link = f' · <a href="{esc(url)}" target="_blank">source</a>' if url.startswith("http") else ""
                chk = f", checked {esc(t['source_checked_on'])}" if t.get("source_checked_on") else ""
                return f"{link}{chk}"

            tcards = "".join(
                f'<div class="equip"><div class="equip-icon">🚜</div>'
                f'<div class="equip-name">{esc(t["brand"])} {esc(t["model"])}'
                f'<br><small style="color:#668074;font-weight:600">{esc(fmt_hp(t["hp"]))} HP'
                f'{" · " + esc(t["drive_type"]) if t.get("drive_type") else ""}</small>'
                f'<br><small style="color:#0b5639;font-weight:700">{esc(format_price(t))}</small>'
                f'<br><small style="color:#668074;font-weight:500">{_src(t)}</small></div></div>'
                for t in tractors
            )
            st.markdown(f'<div class="equip-grid">{tcards}</div>', unsafe_allow_html=True)
            st.caption("Found through an online search. Open the source link to verify; prices can change, confirm with a local dealer.")
            if getattr(tractor_raw, "note", ""):
                st.caption(tractor_raw.note)
        elif getattr(tractor_raw, "note", ""):
            st.markdown("#### Tractor examples")
            st.info(tractor_raw.note)

        st.markdown("#### Preparation workflow")
        items = "".join(
            f'<div class="t-item"><div class="t-num">{i}</div><div class="t-body">{esc(clean_step(s))}</div></div>'
            for i, s in enumerate(result.steps, 1)
        )
        st.markdown(f'<div class="timeline">{items}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Why this recommendation?")
        st.markdown(f'<div class="logic-card">{esc(result.reasoning)}</div>', unsafe_allow_html=True)
        st.markdown("#### How the numbers are decided")
        st.markdown(
            """
            <div class="logic-grid">
                <div class="logic-item"><b>🛠️ Equipment</b>A rule engine shortlists implements from soil type, crop and land condition. The AI only orders and explains them.</div>
                <div class="logic-item"><b>🚜 Tractor power</b>Based on plot size.</div>
                <div class="logic-item"><b>⏱️ Time</b>Area divided by each implement's working speed, plus a small allowance for turning and setup.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download detailed PDF report",
        rep["pdf"],
        file_name="land_recommendation.pdf",
        mime="application/pdf",
    )


# ---------------- TOP BAR + HERO ----------------
st.markdown(
    """
    <div class="nav">
      <div class="logo"><div class="logo-mark">🌾</div>Khet<span>AI</span></div>
      <div class="nav-links"><a href="#advisor">Advisor</a><a href="#how-it-works">How it works</a><a href="#assistant">Ask AI</a><a href="#about">About</a></div>
      <div class="nav-pill">☏ &nbsp; Help center</div>
    </div>
    <section class="hero">
      <div class="eyebrow">● &nbsp; Smart farming, made simple</div>
      <h1>Prepare your<br>land.<br><em>Grow with confidence.</em></h1>
      <p>Turn a few field photos into a practical, expert-backed land preparation plan for your next crop.</p>
      <a class="hero-cta" href="#advisor">Start your assessment &nbsp; →</a>
      <span class="hero-note">♧ &nbsp; Built for Indian farms</span>
      <a class="scan-chip" href="#assistant">▣ &nbsp; <b>AI field scan</b><br><small>Ask anything about your field →</small></a>
    </section>
    <div class="stats">
      <div class="stat"><span class="stat-icon">ϟ</span><div><b>2 min</b><small>to your first plan</small></div></div>
      <div class="stat"><span class="stat-icon">▣</span><div><b>4 photos</b><small>is all you need</small></div></div>
      <div class="stat"><span class="stat-icon">♧</span><div><b>100% practical</b><small>recommendations</small></div></div>
      <div class="stat"><span class="stat-icon">▤</span><div><b>PDF ready</b><small>share with your team</small></div></div>
    </div>
    <div id="advisor" class="section-head">
      <div class="section-kicker">Your field assessment</div>
      <h2>Let's get your land ready.</h2>
      <p>Share a few details and we'll build your preparation plan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- ASSESSMENT FORM + RESULT ----------------
left, right = st.columns([1, 1], gap="large")
with left:
    with st.container(border=True):
        st.markdown('<div class="form-step"><span class="num">01</span><div><b>Upload field photos</b><small>Daylight photos of the full plot work best.</small></div></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop photos here or browse",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload clear daylight photos showing the full field. Up to four images.",
        )
        if uploaded:
            if len(uploaded) > MAX_PHOTOS:
                st.caption(f"Only the first {MAX_PHOTOS} photos will be used.")
            uploaded = uploaded[:MAX_PHOTOS]
            st.caption(f"✓ {len(uploaded)} photo(s) selected")
            preview_cols = st.columns(min(len(uploaded), 2))
            for index, photo in enumerate(uploaded):
                with preview_cols[index % len(preview_cols)]:
                    st.image(photo.getvalue(), caption=photo.name)

        st.markdown('<div class="form-step"><span class="num">02</span><div><b>Tell us about your crop</b><small>We’ll tailor every recommendation to your field.</small></div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            soil = st.selectbox("Soil type", SOILS, format_func=lambda x: {"black":"Black soil","clay":"Clay soil","loam":"Loamy soil","sandy":"Sandy soil"}.get(x,x.title()))
        with c2:
            crop_choice = st.selectbox("Intended crop", CROPS + ["other"], format_func=lambda x: x.title() if x != "other" else "Other")
        crop = crop_choice
        if crop_choice == "other":
            crop = st.text_input("Crop name", placeholder="e.g. Groundnut")

        st.markdown('<div class="form-step"><span class="num">03</span><div><b>Measure your plot</b><small>Approximate size is enough to get started.</small></div></div>', unsafe_allow_html=True)
        c3, c4 = st.columns([1.2, 1])
        with c3:
            size = st.number_input("Plot size", min_value=0.01, value=1.5, step=0.25)
        with c4:
            unit = st.selectbox("Unit", list(UNIT_TO_ACRE.keys()), format_func=lambda x: x.title())
        if unit == "bigha":
            st.caption("Bigha size varies by region; this app uses an approximate 0.625 acres.")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("✧  Generate my land plan   →", type="primary", disabled=not uploaded)
        if not uploaded:
            st.caption("Upload at least one photo to continue.")

with right:
    report = st.session_state.get("report")
    if run:
        generate(uploaded, soil, crop, size, unit)
        report = st.session_state.get("report")

    if report:
        with st.container(border=True):
            render_report(report)
    else:
        st.markdown(
            """
            <div class="preview-panel">
              <div class="preview-icon">♧</div>
              <h3>Your land plan will appear here</h3>
              <p>Complete the three quick steps to get a clear, actionable recommendation for your field.</p>
              <div class="preview-list">✓ &nbsp; Tractor power matched to your plot<br>✓ &nbsp; Equipment and preparation steps<br>✓ &nbsp; Downloadable PDF report</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------- HOW IT WORKS ----------------
st.markdown(
    """
    <section id="how-it-works" class="how-section">
      <div class="section-kicker">Simple by design</div>
      <div class="section-head"><h2>From field to plan in three steps.</h2><p>No jargon. Just clear guidance you can act on.</p></div>
      <div class="how-grid">
        <div class="how-card"><div class="ico">▣</div><b>Show us your field</b><p>Upload clear photos so KhetAI can understand your terrain and current conditions.</p></div>
        <div class="how-card"><div class="ico">✧</div><b>Tell us your goal</b><p>Add your soil, crop and plot size. We combine your inputs with practical rules.</p></div>
        <div class="how-card"><div class="ico">🌾</div><b>Grow with a plan</b><p>Get the right equipment, tractor size and a step-by-step preparation workflow.</p></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------------- AI CHAT ASSISTANT (hero ka "AI field scan" yahin khulta hai) ----------------
st.markdown(
    '<div id="assistant" class="section-head">'
    '<div class="section-kicker">AI field scan</div>'
    '<h2>Ask KhetAI about your field.</h2>'
    '<p>Ask anything about soil, crops, machinery or land preparation. '
    'Generate a plan first to get answers tailored to your own field.</p>'
    '</div>',
    unsafe_allow_html=True,
)

report = st.session_state.get("report")
chat = st.session_state.setdefault("chat", [])

typed = st.chat_input("Ask anything about your field, soil, crop or machinery...")

with st.container(border=True):
    if report:
        st.caption("✓ Answers are based on your land report.")
    else:
        st.caption("No plan generated yet, so answers are general. Generate a plan for advice tailored to your field.")

    suggestions = SUGGESTIONS if report else GENERAL_SUGGESTIONS
    clicked = None
    cols = st.columns(len(suggestions))
    for col, question in zip(cols, suggestions):
        if col.button(question):
            clicked = question

    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = typed or clicked
    if prompt:
        history = list(chat)
        chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(
                    stream_answer(prompt, report["context"] if report else None, history)
                )
            except Exception as e:
                answer = f"Unable to generate an answer: {e}"
                st.error(answer)
        chat.append({"role": "assistant", "content": answer})

st.markdown(
    '<div id="about" class="footer"><div><strong>🌾 KhetAI</strong> &nbsp; Decision support for better prepared fields.</div>'
    '<div>Always confirm with your local agronomist before hiring machinery.</div></div>',
    unsafe_allow_html=True,
)