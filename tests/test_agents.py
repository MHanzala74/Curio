import numpy as np

from curio.agents import REFLECTOR_SYSTEM, SUMMARIZER_SYSTEM, Reflector, Summarizer, extractive_summary
from curio.collector import Item
from curio.embedding import HashingEmbedder, tokenize

RL = "reinforcement learning agents policy gradient reward exploration environment"
DB = "database index query planner transactions storage engine latency"


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("The new approach to RL is on it") == []
    assert "reinforcement" in tokenize("Reinforcement learning")


def test_embedder_is_deterministic_unit_norm_and_topical():
    emb = HashingEmbedder(dim=8)
    a, b, c = emb.embed([RL, RL + " benchmark", DB])
    assert np.allclose(emb.embed([RL])[0], a)
    assert abs(np.linalg.norm(a) - 1) < 1e-6
    assert a @ b > a @ c


def test_embedder_handles_empty_text():
    vec = HashingEmbedder(dim=8).embed([""])
    assert vec.shape == (1, 8) and np.isfinite(vec).all()


class FakeLLM:
    def __init__(self, reply="  A short. Summary here.  ", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def complete(self, system, prompt, max_tokens=300):
        self.calls.append((system, prompt))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


ITEM = Item("x:1", "Title", "First sentence. Second sentence. Third sentence.", "src", "https://u")


def test_extractive_summary_returns_two_sentences():
    assert extractive_summary(ITEM) == "First sentence. Second sentence."
    assert extractive_summary(Item("x:2", "Only title", "", "s", "u")) == "Only title"


def test_summarizer_uses_llm_with_untrusted_data_framing():
    llm = FakeLLM()
    out = Summarizer(llm).summarize(ITEM)
    assert out == "A short. Summary here."
    system, prompt = llm.calls[0]
    assert system == SUMMARIZER_SYSTEM and "untrusted" in system
    assert "<item>" in prompt and "</item>" in prompt


def test_summarizer_falls_back_when_llm_fails_or_is_missing():
    assert Summarizer(FakeLLM(fail=True)).summarize(ITEM) == extractive_summary(ITEM)
    assert Summarizer(None).summarize(ITEM) == extractive_summary(ITEM)
    assert Summarizer(FakeLLM(reply="   ")).summarize(ITEM) == extractive_summary(ITEM)


def test_reflector_fallback_picks_up_clicked_keywords():
    records = ([{"title": "Reinforcement learning agents", "source": "arxiv", "clicked": True}] * 3
               + [{"title": "Quarterly funding round announced", "source": "hn", "clicked": False}] * 3)
    profile = Reflector(None).write_profile(records)
    likes = profile.split("## Dislikes")[0]
    assert "reinforcement" in likes
    assert "funding" in profile.split("## Dislikes")[1]


def test_reflector_with_llm_and_empty_records():
    llm = FakeLLM(reply="## Likes\n- agents")
    records = [{"title": "T", "source": "s", "clicked": True}]
    assert Reflector(llm).write_profile(records).startswith("## Likes")
    assert llm.calls[0][0] == REFLECTOR_SYSTEM and "<clicked>" in llm.calls[0][1]
    assert "(none yet)" in Reflector(None).write_profile([])
    assert Reflector(FakeLLM(fail=True)).write_profile(records).startswith("## Likes")
