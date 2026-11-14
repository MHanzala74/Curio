"""Item embeddings for the policy.

The trained policy works on ``dim``-dimensional unit vectors (``dim = 8``). Two
embedders are provided:

- ``HashingEmbedder``: deterministic, dependency-free (bag of hashed words followed
  by a fixed random projection). Works offline and is the default.
- ``SentenceTransformerEmbedder``: higher quality semantic embeddings, projected down
  to ``dim`` with a fixed random projection. Requires ``sentence-transformers``.

Both are lossy at 8 dimensions; they are a practical bridge between real text and
the simulator-trained policy, not a claim of good semantic coverage.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

STOPWORDS = frozenset("""
a about above after again all also an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no
nor not now of off on once only or other our out over own same she should so some such than
that the their them then there these they this those through to too under until up us very was
we were what when where which while who whom why will with would you your new using use based
via paper approach show shows propose present results method methods
""".split())
_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class HashingEmbedder:
    def __init__(self, dim: int = 8, n_buckets: int = 2048, seed: int = 0) -> None:
        self.dim, self.n_buckets = dim, n_buckets
        self.projection = np.random.default_rng(seed).normal(size=(n_buckets, dim))

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)  # noqa: S324 - not security related
        return h % self.n_buckets, 1.0 if (h >> 64) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        counts = np.zeros((len(texts), self.n_buckets))
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket, sign = self._bucket(token)
                counts[row, bucket] += sign
        counts = np.sign(counts) * np.log1p(np.abs(counts))  # dampen repeated words
        return _normalize_rows(counts @ self.projection)


class SentenceTransformerEmbedder:
    def __init__(self, dim: int = 8, model_name: str = "all-MiniLM-L6-v2", seed: int = 0) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy optional dependency

        self.dim = dim
        self.model = SentenceTransformer(model_name)
        width = self.model.get_sentence_embedding_dimension()
        self.projection = np.random.default_rng(seed).normal(size=(width, dim)) / np.sqrt(width)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.model.encode(texts, normalize_embeddings=True))
        return _normalize_rows(vectors @ self.projection)
