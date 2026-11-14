import json

from curio.collector import (Item, clean_text, collect, parse_arxiv, parse_feed, parse_hn_item)

ARXIV = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <title>Agents   that
      learn</title>
    <summary>We study multi-agent reinforcement learning.</summary>
    <published>2026-01-02T00:00:00Z</published>
  </entry>
</feed>"""

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>Post A</title><link>https://example.com/a</link>
        <description>&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description>
        <guid>a-1</guid><pubDate>Mon, 02 Jan 2026 00:00:00 GMT</pubDate></item>
</channel></rss>"""

ATOM_FEED = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>tag:x,1</id><title>Atom post</title>
         <link href="https://example.com/atom"/><summary>An atom summary</summary>
         <updated>2026-01-03T00:00:00Z</updated></entry>
</feed>"""


def test_clean_text_strips_html_and_collapses_whitespace():
    assert clean_text("<p>Hi&nbsp;&amp;   <b>there</b></p>") == "Hi & there"
    assert clean_text(None) == ""
    assert len(clean_text("x" * 5000, limit=100)) == 100


def test_parse_arxiv_normalizes_fields():
    items = parse_arxiv(ARXIV)
    assert len(items) == 1
    assert items[0].id == "arxiv:2601.00001v1"
    assert items[0].title == "Agents that learn"
    assert items[0].source == "arxiv"


def test_parse_hn_item_handles_missing_url_and_filters_non_stories():
    story = parse_hn_item({"id": 7, "type": "story", "title": "Ask HN: Q", "text": "<p>body</p>"})
    assert story.url == "https://news.ycombinator.com/item?id=7" and story.summary == "body"
    assert parse_hn_item({"id": 8, "type": "comment"}) is None
    assert parse_hn_item({"id": 9, "type": "story", "dead": True}) is None
    assert parse_hn_item(None) is None


def test_parse_feed_supports_rss_and_atom():
    rss = parse_feed(RSS, source="rss:example.com")
    assert rss[0].summary == "Hello world" and rss[0].id == "rss:example.com:a-1"
    atom = parse_feed(ATOM_FEED, source="rss:example.com")
    assert atom[0].url == "https://example.com/atom"


def fake_http(url: str) -> bytes:
    if "export.arxiv.org" in url:
        return ARXIV
    if url.endswith("topstories.json"):
        return json.dumps([1, 2]).encode()
    if "item/1.json" in url:
        return json.dumps({"id": 1, "type": "story", "title": "HN one", "url": "https://e.com/1"}).encode()
    if "item/2.json" in url:
        return json.dumps({"id": 2, "type": "story", "title": "HN two", "url": "https://e.com/2"}).encode()
    if url == "https://example.com/feed.xml":
        return RSS
    raise OSError("unexpected url " + url)


def test_collect_merges_sources_and_survives_a_failing_one():
    config = {"arxiv": {"categories": ["cs.AI"], "max_results": 5}, "hackernews": {"limit": 2},
              "rss": ["https://example.com/feed.xml", "https://broken.example/feed"]}
    items, errors = collect(config, http_get=fake_http)
    assert {i.source for i in items} >= {"arxiv", "hackernews"}
    assert len(items) == 4
    assert len(errors) == 1 and "broken.example" in errors[0]


def test_collect_deduplicates_by_url():
    def dup_http(url):
        return fake_http(url) if "arxiv" in url else RSS

    cfg = {"arxiv": None, "hackernews": None, "rss": ["https://example.com/a.xml", "https://example.com/b.xml"]}
    items, _ = collect(cfg, http_get=dup_http)
    assert len(items) == 1


def test_item_roundtrip():
    item = Item("x:1", "T", "S", "src", "https://u")
    assert Item.from_dict(item.to_dict()) == item
