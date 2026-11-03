import re

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

import rag
from tools import search_tool, wiki_tool

_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, max_tokens=2500, reasoning_effort="low")

RESEARCH_PROMPT = (
    "You are an agricultural research agent for Indian farmers. Tools: "
    "search_knowledge_base (passages from web pages on land preparation: PREFER this and call it at least "
    "twice with different queries, e.g. practices for the crop+soil, and implements for the field condition), "
    "search_tool (live web, only to fill gaps), wiki_tool (background). Make at most 5 tool calls. "
    "Then write research notes under 250 words: key practices, implement tips, cautions. "
    "Use ONLY facts from tool results and mention source URLs. If tools return nothing, say so."
)

CHAT_PROMPT = (
    "You are KhetAI, a practical farming assistant for Indian farmers (soil, crops, machinery, land "
    "preparation). Keep answers short, simple and actionable. Reply in the same language the user writes "
    "in (Hindi/Hinglish/English). Questions about the user's own report: answer from the report below. "
    "General agronomy or machinery questions: call search_knowledge_base first and answer from it; "
    "use search_tool only if needed. Do not invent facts or give pesticide/chemical dosages; suggest a "
    "local agronomist for risky decisions."
)


def make_tools(crop=None, soil=None):
    """Tools har call pe banate hain, taaki alag users ke crop/soil mix na hon."""
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search land-preparation knowledge (crop/soil practices, tillage, machinery) gathered from the
        web. Returns passages with SOURCE urls."""
        try:
            kb = rag.get_kb(crop, soil) if crop and soil else rag.adhoc_kb(query)
            hits = rag.retrieve(kb, query, k=4)
        except Exception as e:
            return f"Knowledge base unavailable: {e}"
        if not hits:
            return "No relevant passages found."
        return "\n\n".join(f"[{i}] SOURCE: {h['url']}\n{h['text']}" for i, h in enumerate(hits, 1))

    return [search_knowledge_base, search_tool, wiki_tool]


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _meta(messages):
    used = sorted({m.name for m in messages if isinstance(m, ToolMessage)})
    urls = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "search_knowledge_base":
            for u in re.findall(r"SOURCE: (\S+)", _text(m.content)):
                if u not in urls:
                    urls.append(u)
    return used, urls


def run_research(analysis, soil: str, acres: float, crop: str):
    """Agent research -> (notes, sources, tools_used)."""
    agent = create_agent(model=_llm, tools=make_tools(crop, soil), system_prompt=RESEARCH_PROMPT)
    q = (
        f"Crop: {crop}. Soil: {soil}. Plot: {acres:g} acres. Field: terrain={analysis.terrain}, "
        f"vegetation={analysis.vegetation} ({analysis.vegetation_density}), stones={analysis.stones_visible}, "
        f"wetness={analysis.surface_wetness}, obstacles={analysis.obstacles}. "
        "Research land preparation practices and suitable machinery for this exact situation."
    )
    raw = agent.invoke({"messages": [{"role": "user", "content": q}]}, config={"recursion_limit": 14})
    used, urls = _meta(raw["messages"])
    return _text(raw["messages"][-1].content), urls, used


def _parse_inputs(context):
    if not context:
        return None, None
    c = re.search(r"'Crop': '([^']+)'", context)
    s = re.search(r"'Soil': '([^']+)'", context)
    return (c.group(1) if c else None), (s.group(1) if s else None)


def run_chat(question: str, context, history: list[dict]):
    """Chat agent -> (answer, sources)."""
    crop, soil = _parse_inputs(context)
    system = CHAT_PROMPT + (
        f"\n\nUSER'S LAND REPORT:\n{context}" if context
        else "\n\nNo land report yet: give general advice and suggest generating a plan."
    )
    agent = create_agent(model=_llm, tools=make_tools(crop, soil), system_prompt=system)
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": question})
    raw = agent.invoke({"messages": msgs}, config={"recursion_limit": 10})
    _, urls = _meta(raw["messages"])
    return _text(raw["messages"][-1].content), urls