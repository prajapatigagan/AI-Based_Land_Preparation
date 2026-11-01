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

st.set_page_config(page_title="Land Prep Advisor", page_icon="🚜", layout="wide")

SOILS = ["black", "clay", "loam", "sandy"]
CROPS = ["paddy", "wheat", "maize", "soybean", "cotton", "sugarcane", "mustard"]
MAX_PHOTOS = 4
SUGGESTIONS = [
    "Ye equipment kyun chune gaye?",
    "Time estimate kaise nikla?",
    "Sabse pehle kaunsa kaam karu?",
    "Warnings ka kya matlab hai?",
]


def clean_step(s: str) -> str:
    """LLM ke '1.' '2)' wale numbering hata do (UI khud number karega)."""
    return re.sub(r"^\s*\d+[\.\)]\s*", "", s).strip()


def analyze_uploaded(f):
    """Ek uploaded photo ko temp file me save karke analyze karo, phir file hata do."""
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
    """Poora pipeline chalao aur result st.session_state['report'] me rakho."""
    st.session_state.pop("report", None)
    st.session_state["chat"] = []

    if not uploaded:
        st.error("Pehle zameen ki photo upload karo.")
        return
    if not crop.strip():
        st.error("Crop ka naam daalo.")
        return

    try:
        acres = to_acres(size, unit)

        with st.spinner(f"{len(uploaded)} photo analyze ho rahi hain..."):
            analysis = merge_analyses([analyze_uploaded(f) for f in uploaded])

        if not analysis.image_quality_ok or analysis.confidence.lower() == "low":
            st.warning(
                "Image clear nahi hai ya khet ki nahi lag rahi. "
                "Daylight me poore plot ki dusri photo upload karo."
            )
            return

        with st.spinner("Recommendation ban rahi hai..."):
            shortlist = rule_shortlist(analysis, soil, acres, crop)
            result = recommend(analysis, shortlist, soil, acres, crop)

        inputs = {"Soil": soil, "Crop": crop, "Plot size (acres)": acres}
        buf = BytesIO()
        build_pdf(buf, inputs, analysis, result)

        st.session_state["report"] = {
            "acres": acres,
            "analysis": analysis,
            "result": result,
            "pdf": buf.getvalue(),
            "context": build_context(inputs, analysis, shortlist, result),
        }
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Kuch gadbad hui: {e}")


def render_report(rep):
    analysis, result, acres = rep["analysis"], rep["result"], rep["acres"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Plot", f"{acres} acre")
    m2.metric("Tractor", result.tractor_hp_range)
    m3.metric("Est. time", f"{result.estimated_time_hours} hr")
    m4.metric("Confidence", result.confidence.capitalize())

    for w in result.warnings:
        st.warning(w)

    with st.expander("Photo se kya dikha (Land analysis)", expanded=True):
        a1, a2 = st.columns(2)
        a1.write(f"**Terrain:** {analysis.terrain}")
        a1.write(f"**Vegetation:** {analysis.vegetation} ({analysis.vegetation_density})")
        a2.write(f"**Wetness:** {analysis.surface_wetness}")
        a2.write(f"**Obstacles:** {', '.join(analysis.obstacles) or 'none'}")

    st.markdown("#### Equipment")
    st.write("  ".join(f"`{e}`" for e in result.equipment) or "—")

    st.markdown("#### Steps")
    for i, s in enumerate(result.steps, 1):
        st.markdown(f"**{i}.** {clean_step(s)}")

    with st.expander("Reasoning"):
        st.write(result.reasoning)

    st.download_button(
        "Report download (PDF)",
        rep["pdf"],
        file_name="land_recommendation.pdf",
        mime="application/pdf",
    )


st.title("🚜 AI Land Preparation Advisor")
st.caption("Zameen ki photo + soil, size aur crop do. Machinery aur steps ki recommendation milegi.")

left, right = st.columns([1, 1.3], gap="large")

# ---------------- LEFT: inputs ----------------
with left:
    st.subheader("Input")
    uploaded = st.file_uploader(
        f"Zameen ki photo(s) (max {MAX_PHOTOS})",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded:
        if len(uploaded) > MAX_PHOTOS:
            st.caption(f"Sirf pehli {MAX_PHOTOS} photos use hongi.")
        uploaded = uploaded[:MAX_PHOTOS]
        st.image([f.getvalue() for f in uploaded], width=150)

    c1, c2 = st.columns(2)
    soil = c1.selectbox("Soil type", SOILS)
    crop_choice = c2.selectbox("Intended crop", CROPS + ["other"])
    crop = crop_choice
    if crop_choice == "other":
        crop = st.text_input("Crop ka naam")

    c3, c4 = st.columns(2)
    size = c3.number_input("Plot size", min_value=0.01, value=1.0, step=0.25)
    unit = c4.selectbox("Unit", list(UNIT_TO_ACRE.keys()))
    if unit == "bigha":
        st.caption("Bigha ka size region ke hisaab se badalta hai (yahan approx 0.625 acre).")

    run = st.button("Recommend karo", type="primary", use_container_width=True)

# ---------------- RIGHT: results ----------------
with right:
    st.subheader("Result")
    if run:
        generate(uploaded, soil, crop, size, unit)

    report = st.session_state.get("report")
    if report:
        render_report(report)
    elif not run:
        st.info("Left side me details bhar ke **Recommend karo** dabao.")

# ---------------- CHATBOT ----------------
st.divider()
st.subheader("💬 Report ke baare me kuch bhi pucho")

report = st.session_state.get("report")
chat = st.session_state.setdefault("chat", [])

if not report:
    st.caption("Pehle recommendation generate karo, phir yahan sawal pooch sakte ho.")
else:
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state["pending_q"] = q

    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

prompt = st.chat_input("Jaise: Rotavator ki jagah kya use kar sakte hain?", disabled=not report)
if not prompt:
    prompt = st.session_state.pop("pending_q", None)

if prompt and report:
    history = list(chat)
    chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(prompt, report["context"], history))
        except Exception as e:
            answer = f"Jawab nahi mil paya: {e}"
            st.error(answer)
    chat.append({"role": "assistant", "content": answer})