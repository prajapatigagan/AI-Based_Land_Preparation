import traceback
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


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


_llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.3, max_tokens=2000, reasoning_effort="low"
).with_structured_output(_LLMOut, method="json_schema")

_LEVEL = {"low": 0, "medium": 1, "high": 2}


def recommend(analysis, shortlist: dict, soil: str, acres: float, crop: str) -> Recommendation:
    prompt = (
        f"Crop: {crop}\nSoil: {soil}\nPlot size: {acres:g} acres\n"
        f"Photo analysis: {analysis.model_dump_json()}\n"
        f"Rule-engine shortlist (ONLY these allowed): {shortlist['equipment']}\n"
        f"Rule-engine notes: {shortlist['notes']}\n\n"
        "Task: order the shortlisted equipment in working sequence, write short practical "
        "steps for an Indian farmer, explain the reasoning, and list warnings (include every "
        "rule-engine note). Do NOT add equipment outside the shortlist and do NOT state "
        "tractor HP or hours (they are computed separately). Do not invent facts. "
        "Lower the confidence if photo confidence is low or details are missing."
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
    )
