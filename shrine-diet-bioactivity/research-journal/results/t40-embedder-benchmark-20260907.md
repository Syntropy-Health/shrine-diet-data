# T4.0 Embedder Benchmark — bge-m3 vs Gemini (Vertex + AI-Studio), 2026-09-07

**Task:** retrieval self-consistency on **50 distinct compound–target bioactivity-evidence pairs**
(top-pChEMBL from `bioactivity_evidence`). Paraphrased query ("Is {compound} active on the {target}
target?") must retrieve its own evidence description as top-1 by cosine among all 50. All arms run
through the **same `EmbedderAdapter`** (`lightrag/embedder_adapters.py`).

| embedder | binding | project/auth | dim | latency | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|---|---|
| bge-m3 (LM Studio) | `local` | localhost, none | 1024 | 1.18s | 0.94 | 1.00 | 0.97 |
| gemini-embedding-001 | `vertex` | syntropy-passport, ADC | 1024 | 4.44s | 1.00 | 1.00 | 1.00 |
| gemini-embedding-001 | `aistudio` | AI-Studio API key | 1024 | 3.96s¹ | 1.00 | 1.00 | 1.00 |

¹ both hosted arms include a small inter-batch pause; raw compute lower.

**Read:** the two Gemini transports are identical in quality (same model) and differ only in
auth/latency; local bge-m3 is marginally behind on recall@1 (0.94) but **~3× faster, free, and offline**.
**Recommendation:** default **local bge-m3** for dev/cost; **`vertex`** (ADC, no key in env) for the
hosted arm in prod, **`aistudio`** as the keyed fallback. All three are one `EMBEDDING_BINDING` flip apart.

**GCP project:** Vertex runs on **`syntropy-passport`** (or `shrine-longevity`) — NOT `syntropyhealth-shrine`
(an earlier 403 "billing" was that wrong project; the correct projects are billing-enabled and embed fine).

Method note: retrieval self-consistency is an intrinsic embedder-quality proxy; it is not end-task
DietResearchBench accuracy.
