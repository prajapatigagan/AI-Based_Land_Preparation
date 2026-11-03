import traceback
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from agent import run_research


class _LLMOut(BaseModel):
    equipment: list[str] = Field(description="Chosen equipment in WORKING ORDER, only names from the shortlist")
    steps: list[str] = Field(description="5-8 short preparation steps, no numbering")
    reasoning: str = Field(description="Why this plan fits this plot, 3-5 sentences")
    warnings: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class Recommendation(BaseModel):
    equipment: list[str]
    steps: list[str]
    reasoning: str
    warnings: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    tractor_hp_range: str = ""
    estimated_time_hours: float = 0.0
    sources: list[str] = Field(default_factory=list)      # RAG se aaye URLs
    tools_used: list[str] = Field(default_factory=list)   # agent ne kaun se tools chalaye


_llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.3, max_tokens=2000, reasoning_effort="low"
).with_structured_output(_LLMOut, method="json_schema")

_LEVEL = {"low": 0, "medium": 1, "high": 2}


def recommend(analysis, shortlist: dict, soil: str, acres: float, crop: str) -> Recommendation:
    # 1. Agent + RAG research (fail ho to bina research ke chalega)
    notes, sources, tools_used = "", [], []
    try:
        notes, sources, tools_used = run_research(analysis, soil, acres, crop)
    except Exception:
        traceback.print_exc()
        print("[Research agent fail hua, rule engine + LLM se aage badh rahe hain]")

    # 2. Final recommendation
    prompt = (
        f"Crop: {crop}\nSoil: {soil}\nPlot size: {acres:g} acres\n"
        f"Photo analysis: {analysis.model_dump_json()}\n"
        f"Rule-engine shortlist (ONLY these allowed): {shortlist['equipment']}\n"
        f"Rule-engine notes: {shortlist['notes']}\n"
        f"Web research notes (from RAG/agent, may be empty): {notes or 'none'}\n\n"
        "Task: order the shortlisted equipment in working sequence, write short practical steps for an "
        "Indian farmer, explain the reasoning, and list warnings (include every rule-engine note). Use the "
        "research notes to make steps and warnings specific, but never contradict the shortlist. Do NOT add "
        "equipment outside the shortlist and do NOT state tractor HP or hours (computed separately). "
        "Do not invent facts. Lower the confidence if photo confidence is low or data is missing."
    )
    try:
        out = _llm.invoke(prompt)
        allowed = set(shortlist["equipment"])
        equipment = [e for e in out.equipment if e in allowed] or shortlist["equipment"]
        steps, reasoning, warnings, llm_conf = out.steps, out.reasoning, out.warnings, out.confidence
    except Exception:
        traceback.print_exc()
        equipment = shortlist["equipment"]
        steps = [f"Use {e}." for e in equipment]
        reasoning = "AI explanation unavailable; plan is based on rule engine only."
        warnings, llm_conf = list(shortlist["notes"]), "low"

    conf = min(llm_conf, analysis.confidence, key=_LEVEL.get)
    return Recommendation(
        equipment=equipment,
        steps=steps,
        reasoning=reasoning,
        warnings=warnings,
        confidence=conf,
        tractor_hp_range=shortlist["tractor_hp_range"],
        estimated_time_hours=shortlist["estimated_time_hours"],
        sources=sources,
        tools_used=tools_used,
    )