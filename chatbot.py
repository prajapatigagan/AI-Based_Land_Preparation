"""Report-based Q&A chatbot. app.py isse use karta hai."""
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tools import IMPLEMENT_ACRES_PER_HR, OVERHEAD_FACTOR

load_dotenv()

chat_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=800,
    reasoning_effort="low",
)

SYSTEM = (
    "You are a helpful farm-machinery assistant inside a land preparation app. "
    "The REPORT below is the app's output for the user's plot. Answer the user's "
    "questions mainly from this REPORT.\n"
    "Rules:\n"
    "- Reply in the same language/style as the user (Hinglish is fine). Keep answers short and practical.\n"
    "- Use only facts, numbers and equipment names present in the REPORT. Never invent numbers.\n"
    "- If the answer is not in the REPORT, say 'Ye report me nahi hai', then give a brief general tip "
    "clearly marked as general advice and suggest confirming with a local KVK or agronomist.\n"
    "- When asked why something was chosen, explain using the soil, crop, plot size and land analysis in the REPORT.\n"
    "- If REPORT confidence is low or medium, mention the uncertainty when relevant.\n"
    "- Photo-based terrain/wetness/vegetation can be wrong; do not claim certainty."
)


def build_context(inputs: dict, analysis, shortlist: dict, result) -> str:
    rates = ", ".join(
        f"{e}: {IMPLEMENT_ACRES_PER_HR.get(e.lower(), 1.0)} acre/hr" for e in result.equipment
    )
    lines = ["INPUTS"]
    lines += [f"- {k}: {v}" for k, v in inputs.items()]
    lines += [
        "",
        "LAND ANALYSIS (from photo)",
        f"- terrain: {analysis.terrain}",
        f"- vegetation: {analysis.vegetation} (density: {analysis.vegetation_density})",
        f"- surface wetness: {analysis.surface_wetness}",
        f"- obstacles: {', '.join(analysis.obstacles) or 'none'}",
        f"- photo confidence: {analysis.confidence}",
        "",
        "RULE ENGINE SHORTLIST",
        f"- candidate equipment: {', '.join(shortlist['candidate_equipment'])}",
        f"- tillage goal for crop: {shortlist['tillage_goal']}",
        f"- pre-steps: {', '.join(shortlist['pre_steps']) or 'none'}",
        "",
        "FINAL RECOMMENDATION",
        f"- equipment: {', '.join(result.equipment)}",
        f"- tractor: {result.tractor_hp_range}",
        f"- estimated time: {result.estimated_time_hours} hours",
        f"- confidence: {result.confidence}",
        "- steps:",
    ]
    lines += [f"  {i}. {s}" for i, s in enumerate(result.steps, 1)]
    lines += ["- warnings: " + ("; ".join(result.warnings) or "none"), f"- reasoning: {result.reasoning}", ""]
    lines += [
        "HOW NUMBERS ARE DECIDED",
        "- Equipment shortlist: rule engine from soil type + crop + land condition; the LLM only orders and explains it.",
        "- Tractor HP: by plot size (up to 2 acres: 20-35 HP, up to 10 acres: 35-50 HP, above: 50+ HP).",
        f"- Time = sum over equipment of (acres / implement speed) x {OVERHEAD_FACTOR} overhead for turning/setup.",
        f"- Speeds used: {rates}",
    ]
    return "\n".join(lines)


def stream_answer(question: str, context: str, history: list[dict]):
    """history = [{'role': 'user'|'assistant', 'content': str}, ...] (current question ke bina)."""
    msgs = [SystemMessage(content=f"{SYSTEM}\n\nREPORT:\n{context}")]
    for m in history[-8:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    msgs.append(HumanMessage(content=question))

    for chunk in chat_llm.stream(msgs):
        text = chunk.content
        if isinstance(text, str) and text:
            yield text