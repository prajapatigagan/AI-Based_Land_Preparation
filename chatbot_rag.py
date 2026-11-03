from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from agent import run_chat
from tractor_web import fmt_hp, format_price

_plain = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, max_tokens=1200, reasoning_effort="low")

_FALLBACK_SYSTEM = (
    "You are KhetAI, a practical farming assistant for Indian farmers. Keep answers short and actionable. "
    "Reply in the user's language. Base answers on the land report if given. No pesticide dosages."
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


def _chunks(text: str, n: int = 3):
    words = text.split(" ")
    for i in range(0, len(words), n):
        yield " ".join(words[i:i + n]) + " "


def stream_answer(question: str, context, history: list[dict]):
    try:
        answer, urls = run_chat(question, context, history)  # agent + RAG
        if urls:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {u}" for u in urls)
        yield from _chunks(answer)
    except Exception as e:
        print(f"[chat agent fail -> plain LLM] {e}")
        system = _FALLBACK_SYSTEM + (f"\n\nLAND REPORT:\n{context}" if context else "")
        messages = [("system", system)] + [(m["role"], m["content"]) for m in history[-6:]]
        messages.append(("user", question))
        for chunk in _plain.stream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content