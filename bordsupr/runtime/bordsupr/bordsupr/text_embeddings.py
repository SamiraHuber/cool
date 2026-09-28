from __future__ import annotations

import os
import threading

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
TEXT_EMBED_DIM = int(os.getenv("TEXT_EMBED_DIM", "1024"))

_model = None
_model_error: Exception | None = None
_model_lock = threading.Lock()


def get_text_embedding_dim() -> int:
    return int(TEXT_EMBED_DIM)


def _load_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        return None

    with _model_lock:
        if _model is not None:
            return _model
        if _model_error is not None:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(TEXT_EMBED_MODEL, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - best effort runtime fallback
            _model_error = exc
            return None
    return _model


def embed_text(text: str) -> list[float] | None:
    normalized = str(text or "").strip()
    if not normalized:
        return None

    model = _load_model()
    if model is None:
        return None

    vector = model.encode(normalized, normalize_embeddings=True)
    return [float(value) for value in vector.tolist()]