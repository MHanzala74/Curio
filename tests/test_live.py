import json
from pathlib import Path

import numpy as np

from curio import cli
from curio.collector import Item
from curio.embedding import HashingEmbedder
from curio.env_longterm import LongTermFeedEnv
from curio.live import (LiveState, SlateBuilder, load_policy, record_feedback, shortlist, start_day)

MODEL = Path(__file__).resolve().parents[1] / "models" / "curio_policy.pt"


def make_items(n, source="s"):
    return [Item(f"{source}:{i}", f"T{i}", "S", source, f"https://u/{source}/{i}") for i in range(n)]


def test_state_roundtrip(tmp_path):
    state = LiveState()
    state.apply(np.ones(8) / np.sqrt(8), True)
    state.save(tmp_path / "s.json")
    loaded = LiveState.load(tmp_path / "s.json")
    assert loaded.total_clicks == 1 and np.allclose(loaded.exposure, state.exposure)
    assert LiveState.load(tmp_path / "missing.json").total_shown == 0


def test_observation_matches_simulator_layout_exactly():
    env = LongTermFeedEnv(seed=3)
    obs, _ = env.reset()
    builder = SlateBuilder(load_policy(MODEL))
    state = LiveState()
    assert np.allclose(builder.build_obs(state, env.pool, env.available), obs, atol=1e-6)
    # After one step (with the same user reaction) the live state must stay in sync.
    action = 4
    _, _, _, _, info = env.step(action)
    state.apply(env.pool[action], info["clicked"])  # the pool is unchanged within a day
    assert np.allclose(builder.build_obs(state, env.pool, env.available), env._obs(), atol=1e-5)


def test_select_returns_distinct_valid_indices_and_handles_small_pools():
    builder = SlateBuilder(load_policy(MODEL), slate_size=5)
    vecs = HashingEmbedder().embed([f"topic {i} reinforcement agents" for i in range(30)])
    chosen = builder.select(LiveState(), vecs)
    assert len(chosen) == 5 and len(set(chosen)) == 5 and all(0 <= c < 30 for c in chosen)
    small = builder.select(LiveState(), vecs[:3])
    assert sorted(small) == [0, 1, 2]


def test_shortlist_round_robins_sources():
    items = make_items(3, "a") + make_items(3, "b")
    assert shortlist(items, 4) == [0, 3, 1, 4]
    assert len(shortlist(items, 99)) == 6


def test_feedback_moves_belief_toward_clicked_topic_and_clears_pending():
    emb = HashingEmbedder()
    rl = "reinforcement learning agents policy gradient reward exploration"
    db = "database index query planner transactions storage latency"
    items = make_items(4)
    vecs = emb.embed([rl, rl + " benchmark", db, db + " engine"])
    state = LiveState()
    start_day(state, items, vecs)
    ctr = record_feedback(state, {"s:0", "s:1"}, "2026-01-01")
    assert ctr == 0.5 and state.pending == [] and state.day == 1
    belief = state.belief()
    assert belief @ vecs[0] > belief @ vecs[2]
    assert len(state.feedback_log) == 4 and state.feedback_log[0]["clicked"]


def test_cli_end_to_end_demo(tmp_path):
    common = ["--state", str(tmp_path / "state.json")]
    assert cli.main(["daily", "--demo", "--offline", "--date", "2026-01-01", "--model", str(MODEL),
                     "--out", str(tmp_path / "digest")] + common) == 0
    digest = (tmp_path / "digest" / "2026-01-01.md").read_text()
    assert digest.count("## ") == 5 and "[SAMPLE]" in digest
    assert len(json.loads((tmp_path / "state.json").read_text())["pending"]) == 5

    assert cli.main(["feedback", "--clicked", "1", "2", "--date", "2026-01-01"] + common) == 0
    state = LiveState.load(tmp_path / "state.json")
    assert state.day == 1 and state.pending == [] and state.total_clicks == 2

    # Day two with no feedback: pending items are counted as ignored.
    assert cli.main(["daily", "--demo", "--offline", "--date", "2026-01-02", "--model", str(MODEL),
                     "--out", str(tmp_path / "digest")] + common) == 0
    assert cli.main(["daily", "--demo", "--offline", "--date", "2026-01-03", "--model", str(MODEL),
                     "--out", str(tmp_path / "digest")] + common) == 0
    state = LiveState.load(tmp_path / "state.json")
    assert state.total_shown == 10 and state.total_clicks == 2

    profile = tmp_path / "profile.md"
    assert cli.main(["reflect", "--offline", "--profile", str(profile)] + common) == 0
    assert "## Likes" in profile.read_text()


def test_feedback_without_pending_digest_is_a_noop(tmp_path):
    assert cli.main(["feedback", "--clicked", "1", "--state", str(tmp_path / "s.json")]) == 1
