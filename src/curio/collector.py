"""Collector agent: fetches candidate items from arXiv, Hacker News and RSS/Atom feeds.

All network access goes through an injectable ``http_get(url) -> bytes`` function,
so every parser can be tested offline with fixtures.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable

try:  # defusedxml protects against malicious XML (entity expansion etc.)
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover - fallback when the optional dependency is absent
    import xml.etree.ElementTree as ET

HttpGet = Callable[[str], bytes]
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class Item:
    """One candidate item shown (or not shown) to the user."""

    id: str
    title: str
    summary: str
    source: str
    url: str
    published: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Item":
        return cls(**data)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


def clean_text(value: str | None, limit: int = 1500) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace and cap the length."""
    if not value:
        return ""
    value = html.unescape(_TAG_RE.sub(" ", value))
    return _WS_RE.sub(" ", value).strip()[:limit]


def default_http_get(url: str, timeout: float = 15.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "curio-rl/0.1 (research project)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https only in config
        return response.read()


# ---- arXiv ------------------------------------------------------------------
ATOM = "{http://www.w3.org/2005/Atom}"


def parse_arxiv(xml_bytes: bytes) -> list[Item]:
    root = ET.fromstring(xml_bytes)
    items = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if not raw_id:
            continue
        items.append(Item(
            id="arxiv:" + raw_id.rsplit("/abs/", 1)[-1],
            title=clean_text(entry.findtext(f"{ATOM}title"), 300),
            summary=clean_text(entry.findtext(f"{ATOM}summary")),
            source="arxiv",
            url=raw_id,
            published=(entry.findtext(f"{ATOM}published") or "").strip(),
        ))
    return items


def fetch_arxiv(categories: list[str], max_results: int, http_get: HttpGet) -> list[Item]:
    query = " OR ".join(f"cat:{c}" for c in categories)
    params = urllib.parse.urlencode({
        "search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
        "max_results": max_results,
    })
    return parse_arxiv(http_get(f"https://export.arxiv.org/api/query?{params}"))


# ---- Hacker News ---------------------------------------------------------------
HN_API = "https://hacker-news.firebaseio.com/v0"


def parse_hn_item(data: dict | None) -> Item | None:
    if not data or data.get("type") != "story" or data.get("dead") or data.get("deleted"):
        return None
    item_id = data["id"]
    return Item(
        id=f"hn:{item_id}",
        title=clean_text(data.get("title"), 300),
        summary=clean_text(data.get("text")),
        source="hackernews",
        url=data.get("url") or f"https://news.ycombinator.com/item?id={item_id}",
        published=str(data.get("time", "")),
    )


def fetch_hackernews(limit: int, http_get: HttpGet) -> list[Item]:
    ids = json.loads(http_get(f"{HN_API}/topstories.json"))[:limit]
    items = []
    for story_id in ids:
        item = parse_hn_item(json.loads(http_get(f"{HN_API}/item/{story_id}.json")))
        if item:
            items.append(item)
    return items


# ---- RSS / Atom ----------------------------------------------------------------
def parse_feed(xml_bytes: bytes, source: str = "rss") -> list[Item]:
    root = ET.fromstring(xml_bytes)
    items: list[Item] = []
    for node in root.iter("item"):  # RSS 2.0
        link = (node.findtext("link") or "").strip()
        guid = (node.findtext("guid") or link).strip()
        if not (link or guid):
            continue
        items.append(Item(
            id=f"{source}:{guid}", title=clean_text(node.findtext("title"), 300),
            summary=clean_text(node.findtext("description")), source=source,
            url=link or guid, published=(node.findtext("pubDate") or "").strip(),
        ))
    for node in root.iter(f"{ATOM}entry"):  # Atom
        link_node = node.find(f"{ATOM}link")
        link = (link_node.get("href") if link_node is not None else "") or ""
        guid = (node.findtext(f"{ATOM}id") or link).strip()
        if not guid:
            continue
        body = node.findtext(f"{ATOM}summary") or node.findtext(f"{ATOM}content")
        items.append(Item(
            id=f"{source}:{guid}", title=clean_text(node.findtext(f"{ATOM}title"), 300),
            summary=clean_text(body), source=source, url=link or guid,
            published=(node.findtext(f"{ATOM}published") or node.findtext(f"{ATOM}updated") or "").strip(),
        ))
    return items


def fetch_rss(url: str, http_get: HttpGet) -> list[Item]:
    host = urllib.parse.urlparse(url).netloc or "rss"
    return parse_feed(http_get(url), source=f"rss:{host}")


# ---- orchestration ---------------------------------------------------------------
DEFAULT_CONFIG = {
    "arxiv": {"categories": ["cs.AI", "cs.LG", "cs.CL"], "max_results": 30},
    "hackernews": {"limit": 30},
    "rss": [],
}


def collect(config: dict | None = None, http_get: HttpGet = default_http_get) -> tuple[list[Item], list[str]]:
    """Collect from every configured source. A failing source never stops the others.

    Returns ``(items, errors)`` where items are de-duplicated by id and URL.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    jobs: list[tuple[str, Callable[[], list[Item]]]] = []
    if cfg.get("arxiv"):
        a = cfg["arxiv"]
        jobs.append(("arxiv", lambda: fetch_arxiv(a["categories"], a["max_results"], http_get)))
    if cfg.get("hackernews"):
        jobs.append(("hackernews", lambda: fetch_hackernews(cfg["hackernews"]["limit"], http_get)))
    for feed_url in cfg.get("rss", []):
        jobs.append((f"rss {feed_url}", lambda u=feed_url: fetch_rss(u, http_get)))

    items, errors, seen_ids, seen_urls = [], [], set(), set()
    for name, job in jobs:
        try:
            fetched = job()
        except Exception as exc:  # noqa: BLE001 - one broken source must not break the run
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        for item in fetched:
            if item.id in seen_ids or item.url in seen_urls or not item.title:
                continue
            seen_ids.add(item.id)
            seen_urls.add(item.url)
            items.append(item)
    return items, errors
