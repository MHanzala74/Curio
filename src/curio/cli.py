"""Command line interface: ``python -m curio.cli {daily,feedback,reflect}``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .agents import Reflector, Summarizer
from .collector import DEFAULT_CONFIG, Item, collect
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
    lines = [f"# Curio digest, {args.date}", ""]
    for n, item in enumerate(picked, 1):
        lines += [f"## {n}. {item.title}", f"*{item.source}* | {item.url}", "", summarizer.summarize(item), ""]
    lines += ["---", "Give feedback with: `python -m curio.cli feedback --clicked 1 3` (item numbers), "
              "or `--clicked` with no numbers if you read none."]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest_path = out / f"{args.date}.md"
    digest_path.write_text("\n".join(lines) + "\n")

    start_day(state, picked, picked_vecs)
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
