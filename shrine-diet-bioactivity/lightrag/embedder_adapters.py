"""Pluggable embedder adapters (T4.0 benchmark arm).

One interface, swappable backends selected by ``EMBEDDING_BINDING``:

    local dev     bge-m3 via an OpenAI-compatible endpoint (LM Studio / Ollama-compat)
    vertex        Gemini embeddings on Vertex AI, auth via ADC (no API key in env)
    aistudio      Gemini embeddings on the AI-Studio API, auth via an API key
                  (free tier; does NOT require GCP billing — the Vertex fallback)

Why an adapter, not another ``if`` in ``ingest_unified``: the T4.0 arm compares a
LOCAL embedder against a HOSTED one, and the two differ in transport, auth, and
dimensionality. Isolating each behind one ``embed(texts) -> ndarray`` interface keeps
the ingest wiring identical across arms, makes each backend independently testable, and
makes "add a third embedder" a new class rather than a new branch.

All heavy/optional imports are LAZY (inside ``__init__`` / ``embed``) so this module
imports cleanly for the factory unit test even when google-genai / numpy-less envs are
in play, and so a backend you are not using never has to be installed.

Env contract (per binding):
    EMBEDDING_BINDING          local | ollama | vertex | aistudio   (default: local)
    EMBEDDING_MODEL            e.g. text-embedding-bge-m3 / gemini-embedding-001
    EMBEDDING_DIM              output dimensionality (int)
    EMBEDDING_BINDING_HOST     openai-compat base URL (local/ollama)
    EMBEDDING_BINDING_API_KEY  key for local openai-compat OR aistudio
    GOOGLE_CLOUD_PROJECT       vertex project (default: syntropyhealth-shrine)
    GOOGLE_CLOUD_LOCATION      vertex location (default: us-central1)
"""
from __future__ import annotations

import abc
import asyncio
import os
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # avoid importing numpy at module load for the factory unit test
    import numpy as np


class EmbedderAdapter(abc.ABC):
    """The one interface every backend implements.

    ``embed`` is async and returns a 2-D float32 array of shape (len(texts), dim),
    row order matching input order. Adapters MUST preserve order — a reordered or
    length-mismatched result silently misaligns every embedding downstream.
    """

    name: str
    embedding_dim: int

    @abc.abstractmethod
    async def embed(self, texts: Sequence[str]) -> "np.ndarray": ...

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r} dim={self.embedding_dim}>"


class _GeminiAdapterBase(EmbedderAdapter):
    """Shared Gemini embed path for both Vertex (ADC) and AI-Studio (key) transports.

    google-genai's client is synchronous; we run each call in a worker thread so the
    adapter honours the async interface without blocking the event loop. Output
    dimensionality is requested explicitly when the model supports it (gemini-embedding-*);
    the response is validated for count AND per-row width before it is trusted.
    """

    def __init__(self, model: str, dim: int, client) -> None:
        self._model = model
        self.embedding_dim = int(dim)
        self._client = client

    def _config(self):
        # Only gemini-embedding-* honour output_dimensionality; text-embedding-004 is
        # fixed 768 and rejects the field. Request it only when it can be honoured.
        if self.embedding_dim and self._model.startswith("gemini-embedding"):
            return {"output_dimensionality": self.embedding_dim}
        return None

    async def embed(self, texts: Sequence[str]) -> "np.ndarray":
        import numpy as np

        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        def _call():
            cfg = self._config()
            kwargs = {"model": self._model, "contents": texts}
            if cfg is not None:
                kwargs["config"] = cfg
            r = self._client.models.embed_content(**kwargs)
            vecs = [e.values for e in r.embeddings]
            if len(vecs) != len(texts):
                raise RuntimeError(
                    f"{self.name}: embedding count {len(vecs)} != input {len(texts)} "
                    "(order/count mismatch would misalign the whole batch)"
                )
            widths = {len(v) for v in vecs}
            if len(widths) != 1:
                raise RuntimeError(f"{self.name}: ragged embedding widths {widths}")
            return np.asarray(vecs, dtype=np.float32)

        return await asyncio.to_thread(_call)


class GeminiVertexAdapter(_GeminiAdapterBase):
    """Gemini embeddings on Vertex AI. Auth = ADC (GOOGLE_APPLICATION_CREDENTIALS / gcloud).

    Requires the Vertex AI API to be ENABLED WITH BILLING on the project — a call
    against a billing-disabled project returns 403 PERMISSION_DENIED, surfaced verbatim.
    """

    def __init__(self, model: str, dim: int, project: str, location: str) -> None:
        from google import genai  # lazy

        client = genai.Client(vertexai=True, project=project, location=location)
        super().__init__(model, dim, client)
        self.name = f"vertex:{model}@{project}/{location}"


class GeminiAIStudioAdapter(_GeminiAdapterBase):
    """Gemini embeddings on the AI-Studio API. Auth = API key (free tier, no GCP billing).

    The Vertex fallback when billing is not enabled: same models, same interface,
    different transport + auth.
    """

    def __init__(self, model: str, dim: int, api_key: str) -> None:
        from google import genai  # lazy

        if not api_key:
            raise RuntimeError(
                "aistudio binding needs an API key (EMBEDDING_BINDING_API_KEY / "
                "GEMINI_API_KEY / GOOGLE_AI_STUDIO_API_KEY)"
            )
        client = genai.Client(api_key=api_key)
        super().__init__(model, dim, client)
        self.name = f"aistudio:{model}"


class LocalOpenAICompatAdapter(EmbedderAdapter):
    """bge-m3 (or any model) via an OpenAI-compatible /embeddings endpoint.

    Reuses the float-encoding, order-preserving ``_openai_compat_embed`` in
    ingest_unified rather than re-implementing it — lightrag's bundled openai_embed
    forces base64 encoding, which some local servers reject.
    """

    def __init__(self, model: str, dim: int, base_url: str, api_key: str | None) -> None:
        self.name = f"local:{model}@{base_url}"
        self.embedding_dim = int(dim)
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or "not-needed"

    async def embed(self, texts: Sequence[str]) -> "np.ndarray":
        from ingest_unified import _openai_compat_embed  # lazy: avoids import cycle at load

        return await _openai_compat_embed(
            list(texts),
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            embedding_dim=self.embedding_dim,
        )


# Bindings this factory OWNS. ollama stays on ingest_unified's native path (it uses a
# different lightrag helper), so it is deliberately absent here.
ADAPTER_BINDINGS = frozenset({"local", "openai", "vertex", "aistudio"})


def make_embedder(binding: str | None = None) -> EmbedderAdapter:
    """Construct the adapter for ``binding`` (default: env EMBEDDING_BINDING, else 'local').

    Raises ValueError for a binding this factory does not own (e.g. 'ollama'), so the
    caller routes it to the native path rather than silently getting the wrong embedder.
    """
    binding = (binding or os.getenv("EMBEDDING_BINDING", "local")).lower()
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    dim = int(os.getenv("EMBEDDING_DIM", "768"))

    if binding in ("local", "openai"):
        return LocalOpenAICompatAdapter(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3"),
            dim=dim,
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:1234/v1"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY"),
        )
    if binding == "vertex":
        return GeminiVertexAdapter(
            model=model,
            dim=dim,
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "syntropyhealth-shrine"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    if binding == "aistudio":
        return GeminiAIStudioAdapter(
            model=model,
            dim=dim,
            api_key=(
                os.getenv("EMBEDDING_BINDING_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
                or ""
            ),
        )
    raise ValueError(
        f"binding {binding!r} is not owned by embedder_adapters "
        f"(owned: {sorted(ADAPTER_BINDINGS)}); route it to the native ingest path"
    )


def to_embedding_func(adapter: EmbedderAdapter, max_token_size: int = 8192):
    """Wrap an adapter into lightrag's ``EmbeddingFunc`` so ingest wiring is unchanged."""
    from lightrag.utils import EmbeddingFunc  # lazy

    return EmbeddingFunc(
        embedding_dim=adapter.embedding_dim,
        max_token_size=max_token_size,
        func=adapter.embed,
    )
