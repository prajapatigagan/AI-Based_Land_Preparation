import re
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults


class TractorList(list):
    """List jisme optional .note hai (UI ko message dikhane ke liye)."""
    note = ""


class _Tractor(BaseModel):
    brand: str
    model: str
    hp: float = Field(description="Engine HP as a number")
    drive_type: str = ""
    price_text: str = Field(default="", description="Price exactly as written in the snippet, else empty")
    source_url: str = Field(default="", description="URL of the snippet where this tractor was found")


class _Extract(BaseModel):
    tractors: list[_Tractor] = Field(default_factory=list)


_search = DuckDuckGoSearchResults(output_format="list", num_results=6)
_extractor = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, max_tokens=1500, reasoning_effort="low"
).with_structured_output(_Extract, method="json_schema")

_CACHE: dict[str, TractorList] = {}


def fmt_hp(hp) -> str:
    try:
        return f"{float(hp):g}"
    except (TypeError, ValueError):
        return str(hp)


def format_price(t: dict) -> str:
    price = (t.get("price_text") or "").strip()
    return price or "Price: check with dealer"


def get_tractor_examples(hp_range: str, limit: int = 4) -> TractorList:
    if hp_range in _CACHE:
        return _CACHE[hp_range]

    out = TractorList()
    nums = [int(n) for n in re.findall(r"\d+", hp_range)]
    if len(nums) < 2:
        out.note = "Tractor examples unavailable (HP range samajh nahi aayi)."
        return out
    lo, hi = nums[0], nums[1]

    try:
        hits, seen = [], set()
        for q in (f"{lo}-{hi} HP tractor price India models", f"tractor {lo} to {hi} hp specifications price list India"):
            for r in _search.invoke(q):
                if r.get("link") and r["link"] not in seen:
                    seen.add(r["link"])
                    hits.append(r)
        if not hits:
            out.note = "Web search se tractor examples nahi mile. Local dealer se confirm karein."
            return out

        text = "\n\n".join(
            f"[{i}] {h.get('title','')}\n{h.get('snippet','')}\nURL: {h['link']}" for i, h in enumerate(hits)
        )
        res = _extractor.invoke(
            f"Extract tractor models explicitly mentioned below with engine HP between {lo-2} and {hi+2}.\n"
            "Rules: use ONLY info present in the text; price_text only if a price is written there; "
            "source_url must be the URL of the snippet where you found it. Do not guess or invent.\n\n" + text
        )

        valid_urls = {h["link"] for h in hits}
        today = date.today().isoformat()
        for t in res.tractors:
            if not (lo - 2 <= t.hp <= hi + 2):
                continue
            d = t.model_dump()
            if d["source_url"] not in valid_urls:
                d["source_url"] = ""  # hallucinated link nahi dikhana
            d["source_checked_on"] = today
            out.append(d)
            if len(out) >= limit:
                break

        if not out:
            out.note = "Search snippets me clear tractor details nahi mile. Local dealer se poochhein."
    except Exception as e:
        out.note = f"Tractor search abhi available nahi hai ({e})."

    if out:
        _CACHE[hp_range] = out
    return out