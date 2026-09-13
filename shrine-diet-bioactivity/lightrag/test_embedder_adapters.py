"""T4.0 embedder-adapter factory + interface (network-free).

Proves the factory dispatches to the right backend, refuses unowned bindings loudly
(rather than silently returning the wrong embedder), and that the interface is
abstract. The live embed() paths (Vertex/AI-Studio round-trips) are exercised by the
smoke checks, not here — this file must pass on a laptop with no cloud credentials.
"""
from __future__ import annotations

import pytest

from embedder_adapters import (
    ADAPTER_BINDINGS,
    EmbedderAdapter,
    GeminiAIStudioAdapter,
    GeminiVertexAdapter,
    LocalOpenAICompatAdapter,
    make_embedder,
)

pytestmark = pytest.mark.unit


def test_base_interface_is_abstract():
    with pytest.raises(TypeError):
        EmbedderAdapter()  # abstract embed() — cannot instantiate


def test_local_binding_needs_no_cloud_deps(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BINDING", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-bge-m3")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    a = make_embedder()
    assert isinstance(a, LocalOpenAICompatAdapter)
    assert a.embedding_dim == 1024
    assert "bge-m3" in a.name


def test_unowned_binding_raises_not_silently_wrong():
    # 'ollama' is deliberately NOT owned — it uses ingest_unified's native path.
    with pytest.raises(ValueError, match="not owned"):
        make_embedder("ollama")
    with pytest.raises(ValueError):
        make_embedder("totally-bogus")


def test_adapter_bindings_membership():
    assert {"local", "vertex", "aistudio"} <= ADAPTER_BINDINGS
    assert "ollama" not in ADAPTER_BINDINGS


def test_aistudio_without_key_fails_closed(monkeypatch):
    pytest.importorskip("google.genai")
    for k in ("EMBEDDING_BINDING_API_KEY", "GEMINI_API_KEY", "GOOGLE_AI_STUDIO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="needs an API key"):
        make_embedder("aistudio")


def test_vertex_constructs_and_names_itself(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "syntropy-passport")
    a = make_embedder("vertex")
    assert isinstance(a, GeminiVertexAdapter)
    assert a.embedding_dim == 768
    assert a.name.startswith("vertex:gemini-embedding-001@syntropy-passport")


def test_vertex_requests_output_dim_only_for_gemini_embedding():
    # text-embedding-004 is fixed 768 and rejects output_dimensionality; the config
    # must be None there and set for gemini-embedding-*.
    pytest.importorskip("google.genai")
    a = GeminiVertexAdapter.__new__(GeminiVertexAdapter)
    a._model = "text-embedding-004"; a.embedding_dim = 768
    assert a._config() is None
    a._model = "gemini-embedding-001"; a.embedding_dim = 1536
    assert a._config() == {"output_dimensionality": 1536}


# ---- router guard: embedder<->workspace binding (pre-emptive) ----

def test_local_binding_refused_on_production_workspace():
    from embedder_adapters import assert_binding_allowed_for_workspace
    with pytest.raises(SystemExit, match="router-guard"):
        assert_binding_allowed_for_workspace("local", "unified_diet_kg")


def test_hosted_binding_allowed_on_production_workspace():
    from embedder_adapters import assert_binding_allowed_for_workspace
    # vertex / aistudio are reachable at query time -> allowed on the prod workspace
    assert assert_binding_allowed_for_workspace("vertex", "unified_diet_kg") is None
    assert assert_binding_allowed_for_workspace("aistudio", "unified_diet_kg") is None


def test_local_binding_allowed_on_explore_workspace():
    from embedder_adapters import assert_binding_allowed_for_workspace
    # a non-production (exploratory) workspace may use local bge-m3
    assert assert_binding_allowed_for_workspace("local", "unified_diet_kg_explore") is None


def test_to_embedding_func_hands_deepcopy_atomic_module_func(monkeypatch):
    # The real regression is in to_embedding_func: it MUST hand LightRAG the
    # module-level _active_embed (deepcopy-atomic), NOT adapter.embed (a bound method
    # that dies under LightRAG's asdict->deepcopy). We stub lightrag.utils.EmbeddingFunc
    # to capture what to_embedding_func passes, without importing the real (dir-name-
    # shadowed) lightrag package under pytest.
    import sys, types, copy
    import embedder_adapters as ea

    fake = types.ModuleType("lightrag.utils")

    class _EF:
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    fake.EmbeddingFunc = _EF
    monkeypatch.setitem(sys.modules, "lightrag", types.ModuleType("lightrag"))
    monkeypatch.setitem(sys.modules, "lightrag.utils", fake)

    class _A(ea.EmbedderAdapter):
        name = "t"; embedding_dim = 8
        async def embed(self, texts):  # bound method — must NOT be what we hand over
            return None

    a = _A()
    ef = ea.to_embedding_func(a)
    assert ef.func is ea._active_embed          # module-level func, not a.embed
    assert ef.func is not a.embed
    assert copy.deepcopy(ef.func) is ef.func    # deepcopy-atomic (survives asdict)
    assert ea._ACTIVE_ADAPTER is a              # and it registered the adapter


def test_gemini_embed_rejects_count_mismatch_ragged_and_handles_empty():
    # The docstring warns a reordered/length-mismatched result "silently misaligns
    # every embedding downstream" — prove the guards raise, and empty -> (0, dim).
    import asyncio
    from unittest.mock import MagicMock
    from embedder_adapters import GeminiVertexAdapter

    a = GeminiVertexAdapter.__new__(GeminiVertexAdapter)   # skip __init__ (no genai client)
    a._model = "gemini-embedding-001"; a.embedding_dim = 4; a.name = "vertex:test"
    a._client = MagicMock()

    # count mismatch: 2 inputs -> 1 embedding
    a._client.models.embed_content.return_value = MagicMock(
        embeddings=[MagicMock(values=[0.0] * 4)])
    with pytest.raises(RuntimeError, match="count"):
        asyncio.run(a.embed(["x", "y"]))

    # ragged widths (4 vs 3)
    a._client.models.embed_content.return_value = MagicMock(
        embeddings=[MagicMock(values=[0.0] * 4), MagicMock(values=[0.0] * 3)])
    with pytest.raises(RuntimeError, match="ragged"):
        asyncio.run(a.embed(["x", "y"]))

    # empty input -> zero-row matrix of the right width, no client call
    out = asyncio.run(a.embed([]))
    assert out.shape == (0, 4)


def test_local_adapter_resorts_by_index_and_checks_length(monkeypatch):
    # The local path must re-sort the response by `index` (a server that reorders would
    # misalign the batch) and raise on a short response.
    import asyncio
    from embedder_adapters import LocalOpenAICompatAdapter

    a = LocalOpenAICompatAdapter(model="m", dim=3, base_url="http://x/v1", api_key="k")

    class _Resp:
        _data = [{"index": 1, "embedding": [1, 1, 1]}, {"index": 0, "embedding": [0, 0, 0]}]
        def raise_for_status(self): pass
        def json(self): return {"data": self._data}

    class _Client:
        def __init__(self, resp): self._resp = resp
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return self._resp

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _Client(_Resp()))
    out = asyncio.run(a.embed(["t0", "t1"]))
    assert out[0].tolist() == [0.0, 0.0, 0.0]   # re-sorted: index 0 first
    assert out[1].tolist() == [1.0, 1.0, 1.0]

    class _Short(_Resp):
        _data = [{"index": 0, "embedding": [0, 0, 0]}]   # 1 for 2 inputs
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _Client(_Short()))
    with pytest.raises(RuntimeError, match="got 1 embeddings"):
        asyncio.run(a.embed(["t0", "t1"]))


def test_router_guard_resolves_none_binding_from_env(monkeypatch):
    # A None binding must resolve from EMBEDDING_BINDING (not fall through open).
    from embedder_adapters import assert_binding_allowed_for_workspace
    monkeypatch.setenv("EMBEDDING_BINDING", "local")
    with pytest.raises(SystemExit, match="router-guard"):
        assert_binding_allowed_for_workspace(None, "unified_diet_kg")


def test_openai_binding_allowed_on_production_workspace():
    # openai/OpenRouter is a HOSTED, reachable API -> allowed on the prod workspace
    # (only local bge-m3 is unreachable and blocked).
    from embedder_adapters import assert_binding_allowed_for_workspace
    assert assert_binding_allowed_for_workspace("openai", "unified_diet_kg") is None
