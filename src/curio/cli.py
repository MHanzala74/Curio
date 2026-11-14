"""Command line interface: ``python -m curio.cli {daily,publish,sync-feedback,feedback,reflect}``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from .agents import Reflector, Summarizer
from .collector import DEFAULT_CONFIG, Item, collect
from .github_io import GitHubClient, collect_issue_feedback, publish_digest
from .embedding import HashingEmbedder
from .live import LiveState, SlateBuilder, load_policy, record_feedback, shortlist, start_day
from .llm import default_llm

ROOT = Path(__file__).resolve().parents[2]


def _embedder(name: str, dim: int):
    if name == "sbert":
        from .embedding import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)


def cmd_daily(args) -> int:
    state = LiveState.load(args.state)
    if state.pending:
        print(f"Note: yesterday's {len(state.pending)} items had no feedback; counting them as ignored.")
        record_feedback(state, set(), args.date)

    if args.demo:
        items = [Item.from_dict(d) for d in json.loads((ROOT / "data" / "sample_items.json").read_text())]
        errors: list[str] = []
    else:
        config = json.loads(Path(args.config).read_text()) if args.config else DEFAULT_CONFIG
        items, errors = collect(config)
    for err in errors:
        print("collector warning:", err)
    if not items:
        print("No items collected; nothing to do.")
        return 1

    model = load_policy(args.model)
    keep = shortlist(items, model.pool_size)
    items = [items[i] for i in keep]
    vectors = _embedder(args.embedder, model.dim).embed([it.text for it in items])
    chosen = SlateBuilder(model, args.slate_size).select(state, vectors)
    picked = [items[i] for i in chosen]
    picked_vecs = vectors[chosen]

    llm = None if args.offline else default_llm()
    summarizer = Summarizer(llm)
    summaries = [summarizer.summarize(item) for item in picked]
    lines = [f"# Curio digest, {args.date}", ""]
    for n, (item, summary) in enumerate(zip(picked, summaries), 1):
        lines += [f"## {n}. {item.title}", f"*{item.source}* | {item.url}", "", summary, ""]
    lines += ["---", "Give feedback with: `python -m curio.cli feedback --clicked 1 3` (item numbers), "
              "or `--clicked` with no numbers if you read none."]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest_path = out / f"{args.date}.md"
    digest_path.write_text("\n".join(lines) + "\n")

    start_day(state, picked, picked_vecs, summaries)
    state.save(args.state)
    print(f"Wrote {digest_path} ({'LLM' if llm else 'extractive'} summaries).")
    return 0


def cmd_feedback(args) -> int:
    state = LiveState.load(args.state)
    if not state.pending:
        print("No pending digest. Run `daily` first.")
        return 1
    ids = {state.pending[n - 1]["item"]["id"] for n in args.clicked if 1 <= n <= len(state.pending)}
    ctr = record_feedback(state, ids, args.date)
    state.save(args.state)
    print(f"Recorded feedback: {len(ids)} clicked, CTR {ctr:.2f}, day {state.day}.")
    return 0


def make_client(args) -> GitHubClient:
    token = os.environ.get("GITHUB_TOKEN")
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise SystemExit("Set GITHUB_TOKEN and GITHUB_REPOSITORY (or pass --repo) to use GitHub feedback.")
    return GitHubClient(repo, token)


def _owner(args) -> str:
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    return args.feedback_user or os.environ.get("GITHUB_REPOSITORY_OWNER") or repo.split("/")[0]


def cmd_publish(args) -> int:
    state = LiveState.load(args.state)
    if not state.pending:
        print("No pending digest to publish. Run `daily` first.")
        return 1
    entries = [{**p["item"], "summary": p.get("summary", "")} for p in state.pending]
    state.pending_issue = publish_digest(make_client(args), args.date, entries, _owner(args))
    state.save(args.state)
    print(f"Published issue #{state.pending_issue['number']} with {len(entries)} items.")
    return 0


def cmd_sync_feedback(args) -> int:
    state = LiveState.load(args.state)
    ref = state.pending_issue
    if not state.pending or not ref:
        print("No published digest awaiting feedback.")
        return 0
    client = make_client(args)
    flags = collect_issue_feedback(client, ref["comment_ids"], _owner(args))
    ids = {p["item"]["id"] for p, clicked in zip(state.pending, flags) if clicked}
    ctr = record_feedback(state, ids, args.date)
    state.save(args.state)  # save before closing so a failed close cannot lose feedback
    client.close_issue(ref["number"])
    print(f"Synced feedback from issue #{ref['number']}: {len(ids)} thumbs-up, CTR {ctr:.2f}, day {state.day}.")
    return 0


def cmd_reflect(args) -> int:
    state = LiveState.load(args.state)
    profile_path = Path(args.profile)
    previous = profile_path.read_text() if profile_path.exists() else ""
    llm = None if args.offline else default_llm()
    text = Reflector(llm).write_profile(state.feedback_log[-args.window:], previous)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(text)
    print(f"Wrote {profile_path} from {min(len(state.feedback_log), args.window)} records.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="curio")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", default="state/live_state.json")
    common.add_argument("--date", default=dt.date.today().isoformat())
    common.add_argument("--offline", action="store_true", help="never call an LLM")

    d = sub.add_parser("daily", parents=[common], help="collect, rank, summarize and write today's digest")
    d.add_argument("--demo", action="store_true", help="use bundled sample items instead of the network")
    d.add_argument("--config", help="JSON file with sources (see DEFAULT_CONFIG in collector.py)")
    d.add_argument("--model", default=str(ROOT / "models" / "curio_policy.pt"))
    d.add_argument("--embedder", choices=["hashing", "sbert"], default="hashing")
    d.add_argument("--slate-size", type=int, default=5)
    d.add_argument("--out", default="digest")
    d.set_defaults(func=cmd_daily)

    f = sub.add_parser("feedback", parents=[common], help="record which digest items you clicked")
    f.add_argument("--clicked", type=int, nargs="*", default=[], help="1-based item numbers")
    f.set_defaults(func=cmd_feedback)

    gh = argparse.ArgumentParser(add_help=False)
    gh.add_argument("--repo", help="owner/name (default: $GITHUB_REPOSITORY)")
    gh.add_argument("--feedback-user", help="GitHub login whose reactions count (default: repo owner)")
    p = sub.add_parser("publish", parents=[common, gh], help="post today's digest as a GitHub Issue")
    p.set_defaults(func=cmd_publish)
    sf = sub.add_parser("sync-feedback", parents=[common, gh], help="read thumbs-up reactions on the last digest")
    sf.set_defaults(func=cmd_sync_feedback)

    r = sub.add_parser("reflect", parents=[common], help="write profile.md from recent feedback")
    r.add_argument("--profile", default="profile.md")
    r.add_argument("--window", type=int, default=100)
    r.set_defaults(func=cmd_reflect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
