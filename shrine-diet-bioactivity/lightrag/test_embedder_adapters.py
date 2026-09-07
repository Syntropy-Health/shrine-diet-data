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
