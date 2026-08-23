"""
Behavioral Profiler -- semantic side of drift detection.

Each response is embedded with sentence-transformers (all-MiniLM-L6-v2) and
compared to the certified baseline response by cosine similarity. This catches
drift that no keyword rule would: the agent reaching the same decision but
justifying it differently, becoming markedly more verbose, or shifting stance.

The model is loaded lazily and cached process-wide -- loading costs a few
seconds, scoring a probe costs milliseconds.

Fallback: if the model cannot be loaded (no network on first run, constrained
machine), we degrade to a lexical Jaccard/containment similarity rather than
failing the run. The API reports which backend was used so a degraded score is
never silently presented as an embedding score.
"""

import math
import re
import threading
from typing import List, Optional, Tuple

from .. import config

_model = None
_model_lock = threading.Lock()
_backend = "uninitialised"


def _load_model():
    """Load and cache the sentence-transformer. Returns None if unavailable."""
    global _model, _backend
    if _model is not None or _backend == "lexical-fallback":
        return _model

    with _model_lock:
        if _model is not None or _backend == "lexical-fallback":
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
            _backend = f"sentence-transformers:{config.EMBEDDING_MODEL}"
        except Exception as exc:
            if not config.ALLOW_EMBEDDING_FALLBACK:
                raise RuntimeError(
                    f"Could not load embedding model '{config.EMBEDDING_MODEL}': {exc}"
                ) from exc
            _backend = "lexical-fallback"
            _model = None
    return _model


def backend_name() -> str:
    _load_model()
    return _backend


# --------------------------------------------------------------------------
# Lexical fallback
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9₹]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _lexical_similarity(a: str, b: str) -> float:
    """Jaccard over token sets -- crude but monotonic with real similarity."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def embed(text: str) -> Optional[List[float]]:
    """Return the embedding vector, or None when running on the fallback."""
    model = _load_model()
    if model is None:
        return None
    return [float(x) for x in model.encode(text or "", normalize_embeddings=True)]


def embed_many(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch encode -- much faster than one call per probe."""
    model = _load_model()
    if model is None:
        return None
    vectors = model.encode(list(texts), normalize_embeddings=True, batch_size=32)
    return [[float(x) for x in v] for v in vectors]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def similarity(
    live_text: str,
    baseline_text: str,
    live_vec: Optional[List[float]] = None,
    baseline_vec: Optional[List[float]] = None,
) -> Tuple[float, str]:
    """
    Cosine similarity in [0, 1], plus the backend used.

    Cosine is clamped at 0: negative similarity and zero similarity both mean
    "unrelated", and letting it go negative would inflate drift past 1.0.
    """
    model = _load_model()
    if model is None:
        return _lexical_similarity(live_text, baseline_text), _backend

    if live_vec is None:
        live_vec = embed(live_text)
    if baseline_vec is None:
        baseline_vec = embed(baseline_text)
    if live_vec is None or baseline_vec is None:
        return _lexical_similarity(live_text, baseline_text), "lexical-fallback"

    return max(0.0, min(1.0, cosine(live_vec, baseline_vec))), _backend


def semantic_drift(similarity_score: float) -> float:
    """Drift is the complement of similarity."""
    return max(0.0, min(1.0, 1.0 - similarity_score))
