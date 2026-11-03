from datetime import datetime

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper

_search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en"))
_wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=10000)
)


@tool
def search_tool(query: str) -> str:
    """Search the web for current information."""
    try:
        return _search.invoke(query) or "No results found."
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def wiki_tool(query: str) -> str:
    """Look up factual background information on Wikipedia."""
    try:
        return _wiki.invoke(query) or "No results found."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """Append research output to a text file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)

    return f"Data successfully saved to {filename}"

# ---------------- Units ----------------
UNIT_TO_ACRE = {"acre": 1.0, "hectare": 2.471, "bigha": 0.625, "guntha": 0.025}


def to_acres(size: float, unit: str) -> float:
    if unit not in UNIT_TO_ACRE:
        raise ValueError(f"Unknown unit: {unit}")
    if size is None or size <= 0:
        raise ValueError("Plot size must be greater than 0.")
    return round(size * UNIT_TO_ACRE[unit], 3)


# ---------------- Rule engine (no CSV/SQL, sab code me) ----------------
# Approx working speed (acre/hour). Local conditions ke hisaab se tune karna.
IMPLEMENT_SPEED = {
    "Mulcher": 1.2, "Disc plough": 1.0, "Mould board plough": 0.8,
    "Cultivator": 1.8, "Disc harrow": 1.6, "Rotavator": 1.2, "Puddler": 1.2,
    "Land leveller / scraper": 1.5, "Planker": 3.0,
    "Seed drill": 3.0, "Planter": 2.5, "Ridger": 2.0,
}
TURNING_ALLOWANCE = 1.15  # turning + setup ke liye 15% extra


def _hp_for_area(acres: float) -> str:
    if acres <= 2:
        return "25-35 HP"
    if acres <= 5:
        return "35-45 HP"
    if acres <= 10:
        return "45-55 HP"
    return "55-75 HP"


def rule_shortlist(analysis, soil: str, acres: float, crop: str) -> dict:
    """Soil + crop + visible land condition se equipment shortlist. AI sirf order/explain karega."""
    soil, crop_l = soil.lower(), crop.strip().lower()
    paddy = crop_l in ("paddy", "rice")
    heavy_soil = soil in ("black", "clay")
    wet = analysis.surface_wetness
    stones = analysis.stones_visible
    veg, density = analysis.vegetation, analysis.vegetation_density
    eq, notes = [], []

    # 1. Primary tillage
    if veg in ("shrubs", "trees") or density == "dense":
        eq += ["Mulcher", "Disc plough"]
        notes.append("Dense vegetation: clear stumps/roots manually before tillage.")
    elif heavy_soil:
        eq.append("Mould board plough")
    else:
        eq.append("Cultivator")

    # 2. Secondary tillage
    if stones:
        eq.append("Disc harrow")
        notes.append("Stones visible: rotavator avoid kiya gaya (blades kharab ho sakte hain).")
    elif paddy and wet != "dry":
        eq.append("Puddler")
    else:
        eq.append("Rotavator")

    # 3. Leveling
    if analysis.terrain in ("uneven", "gentle_slope"):
        eq.append("Land leveller / scraper")
    if analysis.terrain == "steep_slope":
        notes.append("Steep slope: contour bunding karo aur heavy implements avoid karo.")
    if paddy and "Puddler" not in eq:
        eq.append("Puddler")
    eq.append("Planker")

    # 4. Sowing
    if crop_l in ("wheat", "soybean", "mustard"):
        eq.append("Seed drill")
    elif crop_l in ("maize", "cotton"):
        eq.append("Planter")
    elif crop_l == "sugarcane":
        eq.append("Ridger")

    if wet == "waterlogged" and not paddy:
        notes.append("Field waterlogged hai: pehle drainage karo, mitti workable hone par hi tillage.")
    if heavy_soil and wet != "dry" and not paddy:
        notes.append("Gili black/clay mitti me tillage se clods ban jate hain; thodi sukhne do.")

    eq = list(dict.fromkeys(eq))  # duplicates hatao, order rakho
    hours = sum(acres / IMPLEMENT_SPEED[e] for e in eq if e in IMPLEMENT_SPEED) * TURNING_ALLOWANCE
    return {
        "equipment": eq,
        "tractor_hp_range": _hp_for_area(acres),
        "estimated_time_hours": round(hours, 1),
        "notes": notes,
    }