# T4.0 Embedder Benchmark — local bge-m3 vs Gemini (2026-09-07)

**Task:** retrieval self-consistency on **50 distinct compound–target bioactivity-evidence pairs**
(from `bioactivity_evidence`, top-pChEMBL). For each pair, a paraphrased query
("Is {compound} active on the {target} target?") must retrieve its own evidence
description ("{compound} shows {act} activity against {target} (pChEMBL x)") as top-1
by cosine, among all 50. Both arms run through the **same `EmbedderAdapter` interface**
(`lightrag/embedder_adapters.py`).

| embedder | binding | dim | latency | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|---|
| bge-m3 (LM Studio) | `local` | 1024 | 1.75s | 0.94 | 1.00 | 0.97 |
| gemini-embedding-001 | `aistudio` | 1024 | 3.81s¹ | 1.00 | 1.00 | 1.00 |

¹ includes a 0.5s inter-batch pause for free-tier rate limits; raw compute lower.

**Read:** both embedders are strong for this KG's retrieval. Gemini edges recall@1/MRR
(1.00 vs 0.94/0.97); local bge-m3 is faster, **free, and offline** (no key, no billing).
**Recommendation:** default to **local bge-m3** for dev/cost; reach for **Gemini** when
maximum retrieval precision justifies the hosted call. Vertex arm (`vertex` binding, ADC)
is code-ready but blocked on GCP billing for `syntropyhealth-shrine` — `aistudio` is the
billing-free equivalent used here.

Method note: retrieval self-consistency is an intrinsic embedder-quality proxy (no
ground-truth labels needed); it does not measure end-task DietResearchBench accuracy.
