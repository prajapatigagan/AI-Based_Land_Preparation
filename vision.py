from collections import Counter

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from schemas import LandAnalysis
from tools import image_to_b64

load_dotenv()

vision_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=800,
    reasoning_effort="low",
)
vision = vision_llm.with_structured_output(LandAnalysis)


# ---------- Single image ----------
def analyze_land(image_path: str) -> LandAnalysis:
    b64 = image_to_b64(image_path)
    msg = HumanMessage(content=[
        {"type": "text", "text":
            "Analyze this photo of agricultural land. Report ONLY what is "
            "visible: terrain, vegetation type/density, surface wetness, "
            "obstacles. Obstacles = only stones, tree stumps, trees, boulders, "
            "structures, poles. Do NOT list soil clods, mud or grass as obstacles. "
            "Do NOT guess soil type. If image is blurry/dark/irrelevant set "
            "image_quality_ok=false and confidence=low."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    return vision.invoke([msg])


# ---------- Multiple images ----------
_TERRAIN = ["flat", "sloped", "uneven"]
_DENSITY = ["low", "medium", "high"]
_WET = ["dry", "moist", "waterlogged"]
_CONF = ["low", "medium", "high"]


def _pick(values, order, worst=True):
    known = [v.lower() for v in values if v.lower() in order]
    if not known:
        return values[0]
    return (max if worst else min)(known, key=order.index)


def merge_analyses(items: list[LandAnalysis]) -> LandAnalysis:
    """Kai photos ke analysis ko ek me milao. Kharab photos ignore."""
    good = [a for a in items if a.image_quality_ok and a.confidence.lower() != "low"]
    if not good:
        return items[0]  # caller ka quality check ise reject kar dega
    return LandAnalysis(
        terrain=_pick([a.terrain for a in good], _TERRAIN),
        vegetation=Counter(a.vegetation for a in good).most_common(1)[0][0],
        vegetation_density=_pick([a.vegetation_density for a in good], _DENSITY),
        surface_wetness=_pick([a.surface_wetness for a in good], _WET),
        obstacles=sorted({o for a in good for o in a.obstacles}),
        image_quality_ok=True,
        confidence=_pick([a.confidence for a in good], _CONF, worst=False),
    )