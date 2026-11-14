import json
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

from curio import cli
from curio.github_io import GitHubClient, collect_issue_feedback, publish_digest, safe_markdown
from curio.live import LiveState

MODEL = Path(__file__).resolve().parents[1] / "models" / "curio_policy.pt"


class FakeGitHub:
    """In-memory stand-in for the parts of the GitHub REST API Curio uses."""

    def __init__(self):
        self.issues, self.comments, self.reaction_map, self.calls = {}, {}, {}, []
        self._next = 100

    def transport(self, method, url, headers, data):
        assert headers["Authorization"] == "Bearer tok"
        path, _, query = url.split("https://api.github.com")[1].partition("?")
        params = parse_qs(query)
        body = json.loads(data) if data else None
        self.calls.append((method, path))
        parts = path.strip("/").split("/")  # repos/o/r/issues[/n[/comments]] | issues/comments/id/reactions
        if method == "POST" and parts[3:] == ["issues"]:
            number = len(self.issues) + 1
            self.issues[number] = {"title": body["title"], "body": body["body"], "state": "open",
                                   "labels": body["labels"], "comments": []}
            return 201, json.dumps({"number": number}).encode()
        if method == "POST" and parts[-1] == "comments":
            cid = self._next
            self._next += 1
            self.issues[int(parts[4])]["comments"].append(cid)
            self.comments[cid] = body["body"]
            return 201, json.dumps({"id": cid}).encode()
        if method == "GET" and parts[-1] == "reactions":
            items = self.reaction_map.get(int(parts[-2]), [])
            if "content" in params:
                items = [r for r in items if r["content"] == params["content"][0]]
            per_page = int(params.get("per_page", ["30"])[0])
            page = int(params.get("page", ["1"])[0])
            return 200, json.dumps(items[(page - 1) * per_page: page * per_page]).encode()
        if method == "PATCH" and parts[3] == "issues":
            self.issues[int(parts[4])]["state"] = body["state"]
            return 200, b"{}"
        return 404, b"{}"

    def react(self, comment_id, user, content="+1"):
        self.reaction_map.setdefault(comment_id, []).append({"user": {"login": user}, "content": content})

    def client(self):
        return GitHubClient("owner/repo", "tok", self.transport)


def test_safe_markdown_neutralizes_mentions():
    assert "@evil" not in safe_markdown("hi @evil and @other")
    assert safe_markdown("a@b").replace("\u200b", "") == "a@b"


def test_publish_creates_issue_with_one_comment_per_item():
    gh = FakeGitHub()
    entries = [{"title": f"T{i} @spam", "source": "arxiv", "url": f"https://u/{i}", "summary": "S"} for i in range(3)]
    ref = publish_digest(gh.client(), "2026-01-01", entries, "owner")
    assert ref["number"] == 1 and len(ref["comment_ids"]) == 3
    assert gh.issues[1]["labels"] == ["curio-digest"]
    assert "owner" in gh.issues[1]["body"]
    assert all("@spam" not in c for c in gh.comments.values())


def test_only_owner_thumbs_up_counts_as_clicked():
    gh = FakeGitHub()
    gh.react(1, "owner")                      # clicked
    gh.react(2, "stranger")                   # someone else's thumbs-up: ignored
    gh.react(3, "owner", content="-1")        # owner thumbs-down: not a click
    gh.react(4, "OWNER")                      # login comparison is case-insensitive
    flags = collect_issue_feedback(gh.client(), [1, 2, 3, 4, 5], "owner")
    assert flags == [True, False, False, True, False]


def test_failed_api_call_raises():
    gh = FakeGitHub()
    try:
        gh.client()._call("GET", "/nope")
    except RuntimeError as err:
        assert "404" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_cli_publish_then_sync_feedback_roundtrip(tmp_path, monkeypatch):
    gh = FakeGitHub()
    monkeypatch.setattr(cli, "make_client", lambda args: gh.client())
    st = ["--state", str(tmp_path / "state.json")]
    own = ["--feedback-user", "owner", "--repo", "owner/repo"]
    assert cli.main(["daily", "--demo", "--offline", "--date", "2026-01-01", "--model", str(MODEL),
                     "--out", str(tmp_path / "digest")] + st) == 0
    assert cli.main(["publish", "--date", "2026-01-01"] + own + st) == 0
    state = LiveState.load(tmp_path / "state.json")
    ids = state.pending_issue["comment_ids"]
    assert state.pending_issue["number"] == 1 and len(ids) == 5

    gh.react(ids[0], "owner")
    gh.react(ids[2], "owner")
    gh.react(ids[1], "stranger")
    assert cli.main(["sync-feedback", "--date", "2026-01-02"] + own + st) == 0
    state = LiveState.load(tmp_path / "state.json")
    assert state.total_clicks == 2 and state.total_shown == 5 and state.day == 1
    assert state.pending == [] and state.pending_issue is None
    assert gh.issues[1]["state"] == "closed"

    # Nothing awaiting feedback now: a second sync is a harmless no-op.
    calls_before = len(gh.calls)
    assert cli.main(["sync-feedback", "--date", "2026-01-02"] + own + st) == 0
    assert len(gh.calls) == calls_before


def test_publish_without_pending_digest_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "make_client", lambda args: FakeGitHub().client())
    assert cli.main(["publish", "--state", str(tmp_path / "s.json")]) == 1


def test_owner_reaction_is_found_even_when_buried_behind_many_strangers():
    """Regression: with no pagination an owner reaction on page 2+ would be missed."""
    gh = FakeGitHub()
    for i in range(130):
        gh.react(1, f"stranger{i}")
    gh.react(1, "owner")  # the 131st thumbs-up
    assert collect_issue_feedback(gh.client(), [1], "owner") == [True]
    pages = [c for c in gh.calls if c[1].endswith("/reactions")]
    assert len(pages) == 2  # page 1 (full) and page 2 (short), then it stops


def test_reaction_pagination_is_bounded():
    gh = FakeGitHub()
    for i in range(250):
        gh.react(1, f"u{i}")
    assert len(gh.client().reactions(1, max_pages=2)) == 200
