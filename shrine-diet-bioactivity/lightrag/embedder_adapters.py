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
    GOOGLE_CLOUD_PROJECT       vertex project (default: syntropy-passport)
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

    Self-contained (httpx), float-encoded, order-preserving. Deliberately does NOT
    reuse ingest_unified's ``_openai_compat_embed`` — that would couple the adapter to
    the heavy ingest module (lightrag_init at import). float encoding is required
    because lightrag's bundled openai_embed forces base64, which some local servers
    reject. The response is re-sorted by the ``index`` field before trusting order.
    """

    def __init__(self, model: str, dim: int, base_url: str, api_key: str | None) -> None:
        self.name = f"local:{model}@{base_url}"
        self.embedding_dim = int(dim)
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "not-needed"

    async def embed(self, texts: Sequence[str]) -> "np.ndarray":
        import httpx
        import numpy as np

        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        payload = {"model": self._model, "input": texts, "encoding_format": "float"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json().get("data")
        if not data:
            raise RuntimeError(f"{self.name}: embeddings endpoint returned no data")
        # Re-sort by index: a server that reorders would misalign the whole batch.
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        if len(ordered) != len(texts):
            raise RuntimeError(
                f"{self.name}: got {len(ordered)} embeddings for {len(texts)} inputs"
            )
        return np.asarray([d["embedding"] for d in ordered], dtype=np.float32)


# Bindings this factory OWNS. ollama stays on ingest_unified's native path (it uses a
# different lightrag helper), so it is deliberately absent here.
ADAPTER_BINDINGS = frozenset({"local", "openai", "vertex", "aistudio"})


def make_embedder(
    binding: str | None = None,
    *,
    model: str | None = None,
    dim: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> EmbedderAdapter:
    """Construct the adapter for ``binding`` (default: env EMBEDDING_BINDING, else 'local').

    Pass ``model``/``dim`` to use them verbatim (e.g. the caller's already-resolved
    EMBEDDING_MODEL/EMBEDDING_DIM). This matters because the caller stamps
    ``WorkspaceMeta`` from those same resolved values via ``assert_workspace_embedding``:
    if this factory re-read env with its OWN defaults instead, an unset EMBEDDING_MODEL
    would desync the embedding-space guard from the embedder actually used. When
    ``model``/``dim`` are None the per-binding env defaults apply (standalone use).

    Raises ValueError for a binding this factory does not own (e.g. 'ollama'), so the
    caller routes it to the native path rather than silently getting the wrong embedder.
    """
    binding = (binding or os.getenv("EMBEDDING_BINDING", "local")).lower()
    dim = int(dim if dim is not None else os.getenv("EMBEDDING_DIM", "768"))

    if binding in ("local", "openai"):
        return LocalOpenAICompatAdapter(
            model=model or os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3"),
            dim=dim,
            base_url=base_url or os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:1234/v1"),
            api_key=api_key if api_key is not None else os.getenv("EMBEDDING_BINDING_API_KEY"),
        )
    resolved_model = model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    if binding == "vertex":
        return GeminiVertexAdapter(
            model=resolved_model,
            dim=dim,
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "syntropy-passport"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    if binding == "aistudio":
        return GeminiAIStudioAdapter(
            model=resolved_model,
            dim=dim,
            api_key=(
                api_key
                or os.getenv("EMBEDDING_BINDING_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
                or ""
            ),
        )
    raise ValueError(
        f"binding {binding!r} is not owned by embedder_adapters "
        f"(owned: {sorted(ADAPTER_BINDINGS)}); route it to the native ingest path"
    )


# Bindings whose embedder is reachable from the deployed gateway at QUERY time.
# vertex/aistudio (Gemini) and openai (OpenRouter/OpenAI) are all hosted APIs the
# Railway gateway can reach; only `local` (bge-m3 on localhost) is unreachable, so it
# is the one binding a production workspace must not be bound to.
HOSTED_BINDINGS = frozenset({"vertex", "aistudio", "openai"})

# Workspaces the deployed gateway queries. Their embedding space MUST be a hosted
# embedder, or the semantic shape is queryable only from this machine.
PRODUCTION_WORKSPACES = frozenset({"unified_diet_kg"})


def assert_binding_allowed_for_workspace(binding: str | None, workspace: str) -> None:
    """Pre-emptive router guard: refuse a local embedder against a production workspace.

    ``assert_workspace_embedding`` catches an embedder MISMATCH — but only AFTER a
    first ingest has written ``WorkspaceMeta``. This closes the window before it:
    a local (bge-m3) ingest into ``unified_diet_kg`` would set the production
    workspace's space to an embedder the Railway-deployed gateway cannot reach at
    query time, making the semantic shape queryable only locally. Local bge-m3
    belongs on a SEPARATE (exploratory) workspace. Fail closed, loud, with the why.
    """
    binding = (binding or os.getenv("EMBEDDING_BINDING", "local")).lower()
    if workspace in PRODUCTION_WORKSPACES and binding not in HOSTED_BINDINGS:
        raise SystemExit(
            f"[router-guard] refusing EMBEDDING_BINDING={binding!r} against production "
            f"workspace {workspace!r}: its semantic space must be a HOSTED embedder "
            f"({sorted(HOSTED_BINDINGS)}) so the deployed gateway can embed queries "
            f"against it. Use a separate workspace (e.g. '{workspace}_explore') for local bge-m3."
        )


# The active adapter, referenced by the MODULE-LEVEL embed entrypoint below.
# Why a module global instead of passing ``adapter.embed`` directly: LightRAG's
# constructor runs ``asdict(self)`` (dataclasses.asdict → deepcopy) over all fields,
# including the embedding func. deepcopy of a BOUND METHOD deep-copies its
# ``__self__`` (the whole adapter, incl. a non-copyable genai client) and dies. A
# module-level function is deepcopy-atomic (copied by reference), so the func field
# survives asdict — the same reason the ollama/openai paths pass a partial over a
# module-level function rather than a bound method.
#
# ⚠ ONE active adapter PER PROCESS: to_embedding_func overwrites this single global,
# so the last call wins for every EmbeddingFunc handed out. That is safe for ingest
# (one binding per process) and for the T4.0 benchmark (each arm runs as its own
# subprocess via run_semantic_ingest.sh → exec). Do NOT build two adapters in one
# process and expect both EmbeddingFuncs to stay independent — the first would start
# using the second adapter. If that use case ever arises, bind the adapter into a
# closure/instance instead of this module global.
_ACTIVE_ADAPTER: EmbedderAdapter | None = None


async def _active_embed(texts):
    if _ACTIVE_ADAPTER is None:  # pragma: no cover - guarded by to_embedding_func
        raise RuntimeError("no active embedder adapter — call to_embedding_func first")
    return await _ACTIVE_ADAPTER.embed(texts)


def to_embedding_func(adapter: EmbedderAdapter, max_token_size: int = 8192):
    """Wrap an adapter into lightrag's ``EmbeddingFunc`` so ingest wiring is unchanged.

    Registers ``adapter`` as the module-active adapter and hands LightRAG the
    module-level ``_active_embed`` (deepcopy-safe), NOT ``adapter.embed`` (a bound
    method that breaks LightRAG's ``asdict(self)`` deepcopy).
    """
    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = adapter
    from lightrag.utils import EmbeddingFunc  # lazy

    return EmbeddingFunc(
        embedding_dim=adapter.embedding_dim,
        max_token_size=max_token_size,
        func=_active_embed,
    )
