import base64
import io
from collections import Counter
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


class LandAnalysis(BaseModel):
    image_quality_ok: bool = Field(
        description="True only if the photo is clear, in daylight and shows agricultural land"
    )
    terrain: Literal["flat", "gentle_slope", "steep_slope", "uneven"]
    vegetation: Literal["bare", "grass", "weeds", "shrubs", "trees", "crop_residue", "mixed"]
    vegetation_density: Literal["none", "light", "medium", "dense"]
    surface_wetness: Literal["dry", "moist", "waterlogged"]
    stones_visible: bool
    obstacles: list[str] = Field(default_factory=list, description="Stumps, boulders, bunds, trees etc.")
    confidence: Literal["low", "medium", "high"]
    notes: str = Field(default="", description="Limits of image-only analysis")


_vision = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, max_tokens=800
).with_structured_output(LandAnalysis)


def _encode(path: str) -> str:
    """Image chhota karke base64 (Groq ki size limit ke liye)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def analyze_land(path: str) -> LandAnalysis:
    msg = HumanMessage(content=[
        {"type": "text", "text": (
            "Analyze this farmland photo. Report ONLY what is visible; do not guess. "
            "If it is not farmland, blurry or dark, set image_quality_ok=false and confidence=low.")},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(path)}"}},
    ])
    try:
        return _vision.invoke([msg])
    except Exception:
        return _vision.invoke([msg])  # ek retry


_LEVEL = {"low": 0, "medium": 1, "high": 2}
_DENSITY = {"none": 0, "light": 1, "medium": 2, "dense": 3}
_WET = {"dry": 0, "moist": 1, "waterlogged": 2}


def merge_analyses(items: list[LandAnalysis]) -> LandAnalysis:
    """Multiple photos ko ek analysis me milao (conservative side lete hain)."""
    good = [a for a in items if a.image_quality_ok]
    if not good:
        return items[0]
    pick = lambda vals: Counter(vals).most_common(1)[0][0]
    return LandAnalysis(
        image_quality_ok=True,
        terrain=pick([a.terrain for a in good]),
        vegetation=pick([a.vegetation for a in good]),
        vegetation_density=max((a.vegetation_density for a in good), key=_DENSITY.get),
        surface_wetness=max((a.surface_wetness for a in good), key=_WET.get),
        stones_visible=any(a.stones_visible for a in good),
        obstacles=sorted({o for a in good for o in a.obstacles}),
        confidence=min((a.confidence for a in good), key=_LEVEL.get),
        notes=" | ".join(a.notes for a in good if a.notes)[:500],
    )