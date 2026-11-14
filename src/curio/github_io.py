"""GitHub Issues as the feedback channel.

Each day's digest becomes one Issue with one comment per item. A thumbs-up
reaction **from the repository owner** on an item's comment means "clicked";
everything else counts as ignored. Reactions from other accounts are ignored, so
nobody else can steer the model on a public repository.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import quote

API = "https://api.github.com"
Transport = Callable[[str, str, dict, bytes | None], tuple[int, bytes]]


def default_transport(method: str, url: str, headers: dict, data: bytes | None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - https only
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def safe_markdown(text: str) -> str:
    """Neutralize @mentions so text from the internet cannot ping people."""
    return text.replace("@", "@\u200b")


class GitHubClient:
    def __init__(self, repo: str, token: str, transport: Transport = default_transport) -> None:
        self.repo, self.token, self.transport = repo, token, transport

    def _call(self, method: str, path: str, body: dict | None = None):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "curio-rl",
        }
        data = json.dumps(body).encode() if body is not None else None
        status, raw = self.transport(method, f"{API}{path}", headers, data)
        if status >= 300:
            raise RuntimeError(f"GitHub API {method} {path} failed with {status}: {raw[:200]!r}")
        return json.loads(raw) if raw else None

    def create_issue(self, title: str, body: str, labels: list[str] | None = None) -> int:
        return self._call("POST", f"/repos/{self.repo}/issues",
                          {"title": title, "body": body, "labels": labels or []})["number"]

    def add_comment(self, issue: int, body: str) -> int:
        return self._call("POST", f"/repos/{self.repo}/issues/{issue}/comments", {"body": body})["id"]

    def reactions(self, comment_id: int, content: str = "+1", max_pages: int = 10) -> list[dict]:
        """Thumbs-up reactions on a comment, following pagination (100 per page).

        Without pagination, a crowd of strangers' reactions on a public repository could push
        the owner's reaction onto a later page and make it invisible. ``max_pages`` bounds the
        work at 1000 reactions per comment.
        """
        collected: list[dict] = []
        for page in range(1, max_pages + 1):
            batch = self._call(
                "GET",
                f"/repos/{self.repo}/issues/comments/{comment_id}/reactions"
                f"?content={quote(content)}&per_page=100&page={page}",
            )
            collected.extend(batch)
            if len(batch) < 100:
                break
        return collected

    def close_issue(self, issue: int) -> None:
        self._call("PATCH", f"/repos/{self.repo}/issues/{issue}", {"state": "closed"})


def publish_digest(client: GitHubClient, date: str, entries: list[dict], owner: str) -> dict:
    """Create the digest issue. ``entries`` are dicts with title, source, url and summary."""
    body = (f"Curio digest for {date}. React with a thumbs-up on the items you want more of; "
            f"items without a thumbs-up count as ignored. Only reactions from {owner} are used.")
    number = client.create_issue(f"Curio digest {date}", body, labels=["curio-digest"])
    comment_ids = []
    for n, e in enumerate(entries, 1):
        text = (f"### {n}. {safe_markdown(e['title'])}\n*{safe_markdown(e['source'])}* | {e['url']}\n\n"
                f"{safe_markdown(e['summary'])}")
        comment_ids.append(client.add_comment(number, text))
    return {"number": number, "comment_ids": comment_ids}


def collect_issue_feedback(client: GitHubClient, comment_ids: list[int], owner: str) -> list[bool]:
    """For each item comment: True if ``owner`` reacted with a thumbs-up."""
    clicked = []
    for cid in comment_ids:
        reacted = any(
            r.get("content") == "+1" and (r.get("user") or {}).get("login", "").lower() == owner.lower()
            for r in client.reactions(cid)
        )
        clicked.append(reacted)
    return clicked
