from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from tractor_web import fmt_hp, format_price

_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, max_tokens=1200, reasoning_effort="low")

_SYSTEM = (
    "You are KhetAI, a practical farming assistant for Indian farmers (soil, crops, machinery, "
    "land preparation). Keep answers short, simple and actionable. Reply in the same language the "
    "user writes in (Hindi/Hinglish/English). If a land report is provided, base answers on it and "
    "do not contradict it. If you are unsure, say so. No pesticide/chemical dosages; suggest "
    "confirming with a local agronomist for risky decisions."
)


def build_context(inputs, analysis, shortlist, result, tractors) -> str:
    tr = "; ".join(
        f"{t['brand']} {t['model']} ({fmt_hp(t['hp'])} HP, {format_price(t)})" for t in (tractors or [])
    ) or "none found"
    return (
        f"Inputs: {inputs}\n"
        f"Photo analysis: {analysis.model_dump_json()}\n"
        f"Rule-engine notes: {shortlist['notes']}\n"
        f"Equipment (in order): {result.equipment}\n"
        f"Steps: {result.steps}\n"
        f"Tractor HP: {result.tractor_hp_range} | Time: {result.estimated_time_hours} hr "
        "(area / implement speed + 15% allowance) | Confidence: " + result.confidence + "\n"
        f"Warnings: {result.warnings}\nReasoning: {result.reasoning}\nTractor examples: {tr}"
    )


def stream_answer(question: str, context: str | None, history: list[dict]):
    system = _SYSTEM + (
        f"\n\nUSER'S LAND REPORT:\n{context}" if context
        else "\n\nNo land report yet: give general advice and suggest generating a plan."
    )
    messages = [("system", system)]
    messages += [(m["role"], m["content"]) for m in history[-6:]]
    messages.append(("user", question))

    for chunk in _llm.stream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content