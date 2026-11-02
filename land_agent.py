# land_agent.py
import base64, mimetypes, traceback
from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_to_txt


# ---------- Schemas ----------
class LandAnalysis(BaseModel):
    terrain: Literal["flat", "gentle_slope", "steep_slope", "uneven"] = Field(description="Visible terrain")
    vegetation: Literal["bare", "grass", "weeds", "shrubs", "trees", "crop_residue", "mixed"]
    vegetation_density: Literal["none", "light", "medium", "dense"]
    stones_or_rocks: bool
    waterlogging_or_moisture: Literal["dry", "moist", "waterlogged"]
    soil_color_texture_hint: str = Field(description="Visible soil colour/texture, e.g. 'dark, cracked clay-like'")
    obstacles: list[str] = Field(default_factory=list, description="Stumps, boulders, bunds, trees etc.")
    confidence: Literal["low", "medium", "high"]
    notes: str = Field(description="Limits of image-only analysis")


class MachineryItem(BaseModel):
    name: str
    purpose: str
    reason: str = Field(description="Why this fits THIS plot (terrain/veg/soil/crop)")
    min_tractor_hp: str = Field(description="Approx HP needed, e.g. '35-45 HP'")


class Recommendation(BaseModel):
    summary: str
    operations_in_order: list[str] = Field(description="Step-wise land prep sequence")
    machinery: list[MachineryItem]
    precautions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


# ---------- Models ----------
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, max_tokens=1000)
vision = vision_llm.with_structured_output(LandAnalysis)

research_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=2000, reasoning_effort="low")
research_agent = create_agent(
    model=research_llm,
    tools=[search_tool, wiki_tool],
    system_prompt=(
        "You are an agricultural research assistant. Make AT MOST 3 tool calls, in parallel if possible. "
        "Find: land preparation steps and suitable machinery/implements for the given crop, soil type, "
        "plot size and field condition. Return notes under 300 words with source names/URLs."
    ),
)

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, max_tokens=3000, reasoning_effort="low")
writer = writer_llm.with_structured_output(Recommendation, method="json_schema")


# ---------- Stages ----------
def analyze_image(path: str) -> LandAnalysis:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": "Analyze this agricultural plot photo. Report only what is VISIBLE; "
                                 "don't guess invisible things. Set low confidence if image is unclear."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    return vision.invoke([msg])


def research(analysis: LandAnalysis, soil: str, size_acres: float, crop: str):
    q = (f"Crop: {crop}. Soil: {soil}. Plot: {size_acres} acres. "
         f"Field condition: terrain={analysis.terrain}, vegetation={analysis.vegetation} "
         f"({analysis.vegetation_density}), stones={analysis.stones_or_rocks}, "
         f"moisture={analysis.waterlogging_or_moisture}. "
         "Find land preparation practices and best machinery/implements in India.")
    raw = research_agent.invoke({"messages": [{"role": "user", "content": q}]})
    tools = sorted({m.name for m in raw["messages"] if isinstance(m, ToolMessage)})
    return raw["messages"][-1].content, tools


def recommend(analysis, soil, size_acres, crop, notes, tools) -> Recommendation:
    prompt = (
        f"Image analysis: {analysis.model_dump_json()}\n"
        f"User inputs: soil={soil}, plot_size={size_acres} acres, crop={crop}\n\n"
        f"Research notes:\n{notes}\n\n"
        "Recommend land preparation machinery. Use ONLY facts from research notes + image analysis. "
        "Tailor to plot size (small plot = lighter/cheaper equipment). "
        "If image confidence is low or data is missing, say so in precautions. Do not invent facts. List sources."
    )
    try:
        rec = writer.invoke(prompt)
    except Exception:
        traceback.print_exc()
        rec = Recommendation(summary=notes, operations_in_order=[], machinery=[])
    rec.tools_used = tools
    return rec


def run(image_path: str, soil: str, size_acres: float, crop: str) -> Recommendation:
    analysis = analyze_image(image_path)
    notes, tools = research(analysis, soil, size_acres, crop)
    return recommend(analysis, soil, size_acres, crop, notes, tools)


if __name__ == "__main__":
    img = input("Image path: ").strip()
    soil = input("Soil type (clay/loam/sandy...): ").strip()
    size = float(input("Plot size (acres): "))
    crop = input("Intended crop: ").strip()
    out = run(img, soil, size, crop).model_dump_json(indent=2)
    print(out)
    print(save_to_txt(out))