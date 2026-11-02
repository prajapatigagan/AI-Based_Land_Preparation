import os
import base64
import mimetypes
import traceback
from dotenv import load_dotenv

load_dotenv()

from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_to_txt


# ---------------- Schemas ----------------
class LandAnalysis(BaseModel):
    terrain: Literal["flat", "gentle_slope", "steep_slope", "uneven"] = Field(
        description="Visible terrain of the plot"
    )
    vegetation: Literal[
        "bare", "grass", "weeds", "shrubs", "trees", "crop_residue", "mixed"
    ]
    vegetation_density: Literal["none", "light", "medium", "dense"]
    stones_or_rocks: bool = Field(description="Are stones/rocks visible?")
    waterlogging_or_moisture: Literal["dry", "moist", "waterlogged"]
    soil_color_texture_hint: str = Field(
        description="Visible soil colour/texture, e.g. 'dark, cracked, clay-like'"
    )
    obstacles: list[str] = Field(
        default_factory=list, description="Stumps, boulders, bunds, trees etc."
    )
    confidence: Literal["low", "medium", "high"]
    notes: str = Field(description="Limits of image-only analysis")


class MachineryItem(BaseModel):
    name: str
    purpose: str
    reason: str = Field(description="Why this fits THIS plot (terrain/veg/soil/crop)")
    min_tractor_hp: str = Field(description="Approx HP needed, e.g. '35-45 HP'")


class Recommendation(BaseModel):
    summary: str
    operations_in_order: list[str] = Field(
        default_factory=list, description="Step-wise land preparation sequence"
    )
    machinery: list[MachineryItem] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list, description="URLs or names of sources used"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Tools used (filled automatically)"
    )


# ---------------- Models ----------------
vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=1000,
)
vision = vision_llm.with_structured_output(LandAnalysis)

research_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2000,
    reasoning_effort="low",
)

research_agent = create_agent(
    model=research_llm,
    tools=[search_tool, wiki_tool],
    system_prompt=(
        "You are an agricultural research assistant. Make AT MOST 3 tool calls "
        "total, call them together in parallel if possible. Find land "
        "preparation steps and suitable machinery/implements for the given "
        "crop, soil type, plot size and field condition (India context). "
        "Then give short research notes (key facts, source names/URLs) in "
        "under 300 words."
    ),
)

writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=3000,
    reasoning_effort="low",
)
writer = writer_llm.with_structured_output(Recommendation, method="json_schema")


# ---------------- Stages ----------------
def get_tools_used(messages) -> list[str]:
    """Actual tool calls se tool names nikalo (LLM pe depend nahi)."""
    return sorted({m.name for m in messages if isinstance(m, ToolMessage)})


def analyze_image(path: str) -> LandAnalysis:
    """Plot ki photo se visible land conditions nikalo."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analyze this agricultural plot photo. Report only what is "
                    "VISIBLE; do not guess invisible things. Use low confidence "
                    "if the image is unclear."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    return vision.invoke([msg])


def research(analysis: LandAnalysis, soil: str, size_acres: float, crop: str):
    """Online search se practices + machinery info lao."""
    q = (
        f"Crop: {crop}. Soil: {soil}. Plot size: {size_acres} acres. "
        f"Field condition: terrain={analysis.terrain}, "
        f"vegetation={analysis.vegetation} ({analysis.vegetation_density}), "
        f"stones={analysis.stones_or_rocks}, "
        f"moisture={analysis.waterlogging_or_moisture}. "
        "Find land preparation practices and best machinery/implements."
    )
    raw = research_agent.invoke({"messages": [{"role": "user", "content": q}]})
    return raw["messages"][-1].content, get_tools_used(raw["messages"])


def recommend(
    analysis: LandAnalysis,
    soil: str,
    size_acres: float,
    crop: str,
    notes: str,
    tools: list[str],
) -> Recommendation:
    """Image analysis + inputs + research se final recommendation, fail ho to fallback."""
    prompt = (
        f"Image analysis: {analysis.model_dump_json()}\n"
        f"User inputs: soil={soil}, plot_size={size_acres} acres, crop={crop}\n\n"
        f"Research notes:\n{notes}\n\n"
        "Recommend land preparation machinery for this plot. Use ONLY facts from "
        "the research notes and image analysis. Tailor to plot size (small plot "
        "= lighter/cheaper equipment). If image confidence is low or data is "
        "missing, mention it in precautions. Do not invent facts. "
        "Also list the sources used."
    )
    try:
        rec = writer.invoke(prompt)
    except Exception:
        traceback.print_exc()
        print("\n[Structured output fail hua, plain research se fallback]\n")
        rec = Recommendation(summary=notes)
    rec.tools_used = tools
    return rec


def run(image_path: str, soil: str, size_acres: float, crop: str) -> Recommendation:
    """Full pipeline. Streamlit UI isi ko call karegi."""
    analysis = analyze_image(image_path)
    notes, tools = research(analysis, soil, size_acres, crop)
    return recommend(analysis, soil, size_acres, crop, notes, tools)


# ---------------- CLI ----------------
def main():
    image_path = input("Plot image ka path: ").strip().strip('"')
    if not os.path.isfile(image_path):
        print("Image file nahi mili.")
        return

    soil = input("Soil type (clay/loam/sandy/black...): ").strip()
    crop = input("Intended crop: ").strip()
    try:
        size_acres = float(input("Plot size (acres): ").strip())
    except ValueError:
        print("Plot size number me daalo.")
        return

    if not soil or not crop:
        print("Soil type aur crop dono zaruri hain.")
        return

    try:
        result = run(image_path, soil, size_acres, crop)
    except Exception:
        traceback.print_exc()
        return

    output = result.model_dump_json(indent=2)
    print(output)
    print(save_to_txt(output))


if __name__ == "__main__":
    main()