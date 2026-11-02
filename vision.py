import base64
import io
import json
import os
import re
from collections import Counter
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
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


VISION_MODEL = os.getenv("OPENROUTER_MODEL") or "openrouter/free"
print(f"[vision model: {VISION_MODEL}]")

_llm = ChatOpenAI(
    model=VISION_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=1500,
    timeout=90,
)

_PROMPT = """Analyze this farmland photo. Report ONLY what is visible; do not guess.
If it is not farmland, or it is blurry/dark, set image_quality_ok=false and confidence="low".

Reply with ONLY one JSON object (no markdown, no extra text) using exactly these keys and allowed values:
{
 "image_quality_ok": true/false,
 "terrain": "flat" | "gentle_slope" | "steep_slope" | "uneven",
 "vegetation": "bare" | "grass" | "weeds" | "shrubs" | "trees" | "crop_residue" | "mixed",
 "vegetation_density": "none" | "light" | "medium" | "dense",
 "surface_wetness": "dry" | "moist" | "waterlogged",
 "stones_visible": true/false,
 "obstacles": ["stumps", "boulders", ...],
 "confidence": "low" | "medium" | "high",
 "notes": "limits of image-only analysis"
}"""


def _encode(path: str) -> str:
    """Image chhota karke base64."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse(text) -> LandAnalysis:
    if isinstance(text, list):  # kuch models content blocks bhejte hain
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"JSON nahi mila: {text[:200]}")
    return LandAnalysis.model_validate(json.loads(m.group(0)))


def analyze_land(path: str) -> LandAnalysis:
    img_url = f"data:image/jpeg;base64,{_encode(path)}"
    content = [
        {"type": "text", "text": _PROMPT},
        {"type": "image_url", "image_url": {"url": img_url}},
    ]
    last_err = None
    for _ in range(3):  # free router kabhi kabhi kharab output deta hai, retry
        try:
            return _parse(_llm.invoke([HumanMessage(content=content)]).content)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"Vision output parse nahi hua: {last_err}")


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