import re
from datetime import date

import requests
from bs4 import BeautifulSoup
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
    hp: float = Field(description="Engine HP as ONE number (if a range is written, use the upper value)")
    drive_type: str = ""
    price_text: str = Field(default="", description="Price exactly as written in the text, else empty")
    source_url: str = Field(default="", description="URL of the page where this tractor was found")


class _Extract(BaseModel):
    tractors: list[_Tractor] = Field(default_factory=list)


_search = DuckDuckGoSearchResults(output_format="list", num_results=6)
_extractor = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, max_tokens=2000, reasoning_effort="low"
).with_structured_output(_Extract, method="json_schema")

_CACHE: dict[str, TractorList] = {}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def fmt_hp(hp) -> str:
    try:
        return f"{float(hp):g}"
    except (TypeError, ValueError):
        return str(hp)


def format_price(t: dict) -> str:
    price = (t.get("price_text") or "").strip()
    return price or "Price: check with dealer"


def _page_text(url: str, limit: int = 2500) -> str:
    """Page ka readable text (snippet se zyada detail ke liye)."""
    try:
        if not url.startswith("http") or url.lower().endswith(".pdf"):
            return ""
        r = requests.get(url, headers=_HEADERS, timeout=8)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:limit]
    except Exception:
        return ""


def get_tractor_examples(hp_range: str, limit: int = 4) -> TractorList:
    if hp_range in _CACHE:
        return _CACHE[hp_range]

    out = TractorList()
    nums = [int(n) for n in re.findall(r"\d+", hp_range)]
    if len(nums) < 2:
        out.note = "Tractor examples unavailable (HP range samajh nahi aayi)."
        return out
    lo, hi = nums[0], nums[1]
    mid = (lo + hi) // 2
    lo_ok, hi_ok = lo - 5, hi + 5  # thoda margin

    try:
        queries = [
            f"{mid} HP tractor price in India",
            f"{lo} HP tractor price in India",
            f"{hi} HP tractor price in India",
            f"{mid} HP tractor price site:tractorjunction.com",
            f"{mid} HP tractor price site:tractorkarvan.com",
        ]
        hits, seen = [], set()
        for q in queries:
            try:
                results = _search.invoke(q)
            except Exception as e:
                print(f"[search fail] {q}: {e}")
                continue
            for r in results:
                if r.get("link") and r["link"] not in seen:
                    seen.add(r["link"])
                    hits.append(r)
        print(f"[tractor search: {len(hits)} results]")

        if not hits:
            out.note = "Web search se tractor examples nahi mile (rate-limit ho sakta hai). Thodi der baad try karein."
            return out

        # Top pages ka asli text lo
        blocks, fetched = [], 0
        for i, h in enumerate(hits[:8]):
            body = _page_text(h["link"])
            fetched += bool(body)
            text = f"{h.get('snippet', '')} {body}".strip()
            blocks.append(f"[{i}] {h.get('title', '')}\nURL: {h['link']}\n{text}")
        print(f"[pages fetched: {fetched}/{len(blocks)}]")

        res = _extractor.invoke(
            f"From the sources below, extract real tractor models (brand, model, engine HP) with HP between "
            f"{lo_ok} and {hi_ok}. Rules: use ONLY info written in the sources; price_text only if a price "
            "is written there (keep it as written); source_url must be the URL of the source where you "
            "found that tractor. Max 6 tractors, different models. Do not guess or invent.\n\n"
            + "\n\n".join(blocks)
        )
        print(f"[extractor returned: {len(res.tractors)}]")

        valid_urls = {h["link"] for h in hits}
        today = date.today().isoformat()
        names = set()
        for t in res.tractors:
            key = (t.brand.lower(), t.model.lower())
            if key in names or not (lo_ok <= t.hp <= hi_ok):
                continue
            names.add(key)
            d = t.model_dump()
            if d["source_url"] not in valid_urls:
                d["source_url"] = ""  # hallucinated link nahi dikhana
            d["source_checked_on"] = today
            out.append(d)
            if len(out) >= limit:
                break

        print(f"[tractor final: {len(out)}]")
        if not out:
            out.note = "Search me clear tractor details nahi mile. Local dealer se poochhein."
    except Exception as e:
        print(f"[tractor error] {e}")
        out.note = f"Tractor search abhi available nahi hai ({e})."

    if out:
        _CACHE[hp_range] = out
    return out