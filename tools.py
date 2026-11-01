import base64
import io
from datetime import datetime

from PIL import Image

# ======================================================================
# Image helper
# ======================================================================
def image_to_b64(path: str, max_side: int = 1024) -> str:
    """Image resize + JPEG compress karke base64 string do."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ======================================================================
# Unit converter
# ======================================================================
# Bigha ka size state/region ke hisaab se badalta hai, apne area ka value verify kar lena
UNIT_TO_ACRE = {
    "acre": 1.0,
    "hectare": 2.471,
    "bigha": 0.625,        # approx (UP pucca bigha)
    "sqft": 1 / 43560,
    "sqm": 1 / 4046.86,
}


def to_acres(value: float, unit: str = "acre") -> float:
    factor = UNIT_TO_ACRE.get(unit.lower())
    if factor is None:
        raise ValueError(f"Unit '{unit}' supported nahi. Use: {list(UNIT_TO_ACRE)}")
    return round(value * factor, 2)


# ======================================================================
# Knowledge base  (STARTING VALUES - KVK / ICAR-CIAE se verify karke README me source likho)
# ======================================================================
CROP_TILLAGE = {
    "wheat": "fine seedbed", "mustard": "fine seedbed",
    "paddy": "puddling", "rice": "puddling",
    "cotton": "deep ploughing + ridging",
    "sugarcane": "deep ploughing + furrows",
    "soybean": "medium tilth", "maize": "medium tilth",
}

SOIL_IMPLEMENT = {
    "clay":  ["MB plough", "heavy disc harrow", "rotavator"],
    "loam":  ["cultivator", "disc harrow", "rotavator"],
    "sandy": ["light cultivator", "spring harrow", "leveller"],
    "black": ["MB plough", "disc harrow", "cultivator"],
}

# Crop-specific extra implements (crop ka equipment par asli asar)
CROP_EXTRA = {
    "paddy":     ["rotavator", "puddler"],
    "rice":      ["rotavator", "puddler"],
    "sugarcane": ["ridger"],
    "cotton":    ["ridger"],
    "maize":     ["ridger"],
    "wheat":     ["rotavator", "planker"],
    "mustard":   ["rotavator", "planker"],
}

# Approx field capacity (acres per hour) per implement - time estimate ke liye
IMPLEMENT_ACRES_PER_HR = {
    "shredder/mulcher": 1.0,
    "mb plough": 0.6,
    "heavy disc harrow": 1.2,
    "disc harrow": 1.5,
    "cultivator": 2.0,
    "light cultivator": 2.5,
    "spring harrow": 2.5,
    "rotavator": 1.0,
    "puddler": 1.0,
    "ridger": 1.5,
    "planker": 3.0,
    "leveller": 2.0,
    "laser/land leveller": 1.5,
}
DEFAULT_ACRES_PER_HR = 1.0
OVERHEAD_FACTOR = 1.15   # turning, setup, small stoppages


def hp_range(acres: float) -> str:
    if acres <= 2:
        return "20-35 HP"
    if acres <= 10:
        return "35-50 HP"
    return "50+ HP"


def estimate_hours(equipment: list[str], acres: float) -> float:
    """Time = sum(acres / acres_per_hour per implement) x overhead. Rule-based, LLM nahi."""
    total = sum(
        acres / IMPLEMENT_ACRES_PER_HR.get(e.lower(), DEFAULT_ACRES_PER_HR)
        for e in equipment
    )
    return round(total * OVERHEAD_FACTOR, 1)


# ======================================================================
# Rule engine
# ======================================================================
def rule_shortlist(analysis, soil: str, acres: float, crop: str) -> dict:
    """analysis = LandAnalysis object (duck typing, import ki zaroorat nahi)."""
    soil_l, crop_l = soil.lower(), crop.lower()

    equipment = list(SOIL_IMPLEMENT.get(soil_l, ["cultivator", "disc harrow"]))
    pre_steps, warnings = [], []

    # crop-specific implements
    for e in CROP_EXTRA.get(crop_l, []):
        if e not in equipment:
            equipment.append(e)

    # vegetation
    if analysis.vegetation_density == "high" or analysis.vegetation == "bushes":
        equipment.insert(0, "shredder/mulcher")
        pre_steps.append("Clear heavy vegetation (shred/mulch)")

    # obstacles
    if analysis.obstacles:
        warnings.append(f"Obstacles visible: {', '.join(analysis.obstacles)}. Remove before tillage.")

    # wetness
    if analysis.surface_wetness == "waterlogged" and crop_l not in ("paddy", "rice"):
        warnings.append("Land waterlogged: wait for drying before ploughing.")

    # terrain
    if analysis.terrain != "flat":
        equipment.append("laser/land leveller")

    # crop-soil mismatch
    if crop_l in ("paddy", "rice") and soil_l == "sandy":
        warnings.append("Sandy soil drains fast; paddy needs water retention. Consider bunding/puddling carefully.")

    return {
        "candidate_equipment": equipment,
        "tillage_goal": CROP_TILLAGE.get(crop_l, "standard tilth"),
        "tractor_hp_range": hp_range(acres),
        "pre_steps": pre_steps,
        "rule_warnings": warnings,
    }


# ======================================================================
# Output save
# ======================================================================
def save_to_txt(data: str, filename: str = "land_recommendations.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"--- Land Recommendation ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)
    return f"Data successfully saved to {filename}"