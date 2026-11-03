import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults

_search = DuckDuckGoSearchResults(output_format="list", num_results=5)
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
_BLOCKED = ("youtube.com", "facebook.com", "instagram.com", "pinterest.com",
            "twitter.com", "x.com", "linkedin.com", "quora.com", "reddit.com")
CHUNK, OVERLAP = 900, 150

_model = None
_CACHE: dict = {}


# ---------- Embeddings ----------
def _embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")  # ~130 MB, pehli baar download hota hai
    vecs = np.array(list(_model.embed(texts)), dtype="float32")
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


# ---------- Fetch + chunk ----------
def _blocked(url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return any(host == b or host.endswith("." + b) for b in _BLOCKED)


def _fetch(hit: dict) -> dict:
    url, body = hit["link"], ""
    try:
        if not url.lower().endswith(".pdf"):
            r = requests.get(url, headers=_HEADERS, timeout=8)
            if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                soup = BeautifulSoup(r.text, "html.parser")
                for t in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                    t.decompose()
                body = re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:8000]
    except Exception:
        pass
    text = body if len(body) > 300 else hit.get("snippet", "")
    return {"url": url, "title": hit.get("title", ""), "text": text}


def _chunks(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + CHUNK].strip()
        if len(piece) > 200:
            out.append(piece)
        i += CHUNK - OVERLAP
    return out


# ---------- Knowledge base ----------
class KB:
    def __init__(self, chunks: list[dict], vecs):
        self.chunks, self.vecs = chunks, vecs

    @property
    def mode(self) -> str:
        return "vector" if self.vecs is not None else "keyword"


def build_kb(queries: list[str], max_pages: int = 8) -> KB:
    hits, seen = [], set()
    for q in queries:
        try:
            results = _search.invoke(q)
        except Exception as e:
            print(f"[rag] search fail: {q}: {e}")
            continue
        for r in results:
            u = r.get("link")
            if u and u not in seen and not _blocked(u):
                seen.add(u)
                hits.append(r)
    hits = hits[:max_pages]

    with ThreadPoolExecutor(6) as ex:
        pages = list(ex.map(_fetch, hits))

    chunks = [{"url": p["url"], "title": p["title"], "text": c} for p in pages for c in _chunks(p["text"])]
    vecs = None
    if chunks:
        try:
            vecs = _embed([c["text"] for c in chunks])
        except Exception as e:  # fastembed na chale to keyword search se kaam chalega
            print(f"[rag] embedding fail -> keyword mode: {e}")
    kb = KB(chunks, vecs)
    print(f"[rag] pages={len(pages)} chunks={len(chunks)} mode={kb.mode}")
    return kb


def retrieve(kb: KB, query: str, k: int = 4, per_source: int = 2) -> list[dict]:
    if not kb.chunks:
        return []
    if kb.vecs is not None:
        scores = kb.vecs @ _embed([query])[0]
    else:
        q = set(re.findall(r"\w+", query.lower()))
        scores = np.array([
            len(q & set(re.findall(r"\w+", c["text"].lower()))) / (len(q) + 1) for c in kb.chunks
        ])
    out, count = [], {}
    for i in np.argsort(-scores):
        c = kb.chunks[i]
        if count.get(c["url"], 0) >= per_source:  # ek hi site ka dominance nahi
            continue
        count[c["url"]] = count.get(c["url"], 0) + 1
        out.append({**c, "score": float(scores[i])})
        if len(out) >= k:
            break
    return out


def get_kb(crop: str, soil: str) -> KB:
    """Crop + soil ke liye KB (process me cache hota hai)."""
    key = (crop.strip().lower(), soil.strip().lower())
    if key not in _CACHE:
        kb = build_kb([
            f"{crop} land preparation {soil} soil India tillage practices",
            f"best tractor implements for {crop} field preparation India",
            f"{soil} soil tillage ploughing harrowing implements",
            f"{crop} seedbed preparation leveling puddling",
        ])
        if kb.chunks:  # khaali KB cache nahi karte, agli baar retry hoga
            _CACHE[key] = kb
        return kb
    return _CACHE[key]


def adhoc_kb(query: str) -> KB:
    """Plan se pehle ke general sawalon ke liye."""
    key = ("q", query.strip().lower())
    if key not in _CACHE:
        kb = build_kb([query + " farming India", query])
        if kb.chunks:
            _CACHE[key] = kb
        return kb
    return _CACHE[key]