import json
import traceback

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from schemas import LandAnalysis, Recommendation
from tools import estimate_hours

load_dotenv()

reco_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2000,
    reasoning_effort="low",
)
recommender = reco_llm.with_structured_output(Recommendation, method="json_schema")


def match_equipment(items: list[str], candidates: list[str]) -> list[str]:
    """LLM ke naam ko catalog ke naam se match karo ('Disc Harrow (heavy)' -> 'heavy disc harrow').
    Order me exact > catalog naam item ke andar > item catalog naam ke andar."""
    out = []
    for it in items:
        il = it.lower().strip()
        hit = next((c for c in candidates if c.lower() == il), None)
        if hit is None:
            inside = [c for c in candidates if c.lower() in il]
            if inside:
                hit = max(inside, key=len)           # sabse specific naam
        if hit is None:
            hit = next((c for c in candidates if il in c.lower()), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def recommend(a: LandAnalysis, shortlist: dict, soil, acres, crop) -> Recommendation:
    candidates = shortlist["candidate_equipment"]

    prompt = (
        f"Land analysis: {a.model_dump_json()}\n"
        f"Soil: {soil}, Plot: {acres} acres, Crop: {crop}\n"
        f"Rule shortlist: {json.dumps(shortlist)}\n\n"
        "Create an ordered land preparation plan. Choose equipment ONLY from "
        "candidate_equipment, using the exact names. Use the given "
        "tractor_hp_range. Include rule_warnings in warnings. "
        "Write steps without numbering. Do not invent facts."
    )
    try:
        r = recommender.invoke(prompt)
    except Exception:
        traceback.print_exc()
        return Recommendation(
            steps=shortlist["pre_steps"],
            equipment=candidates,
            tractor_hp_range=shortlist["tractor_hp_range"],
            estimated_time_hours=estimate_hours(candidates, acres),
            reasoning="LLM fail hua, sirf rule-based shortlist diya.",
            warnings=shortlist["rule_warnings"],
            confidence="low",
        )

    # 1) equipment: catalog ke naam par normalize; khali ho to poori shortlist
    r.equipment = match_equipment(r.equipment, candidates) or candidates

    # 2) rule engine authority hai: HP range aur time LLM ke bharose nahi
    r.tractor_hp_range = shortlist["tractor_hp_range"]
    r.estimated_time_hours = estimate_hours(r.equipment, acres)

    # 3) rule warnings hamesha hon; LLM ke duplicate warnings hata do
    base = list(shortlist["rule_warnings"])
    extras = [
        w for w in r.warnings
        if w not in base and not any(k in w.lower() for k in ("obstacle", "waterlog"))
    ]
    r.warnings = base + extras
    return r