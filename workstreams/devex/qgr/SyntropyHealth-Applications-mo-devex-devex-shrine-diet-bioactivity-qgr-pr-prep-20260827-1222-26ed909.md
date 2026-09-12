---
receipt_version: 1
type: qgr
boundary: pr-prep
org: SyntropyHealth-Applications
principal: mo
agent: devex
workstream: devex
project: shrine-diet-bioactivity
diff_base: origin/main
hash_a: b43860b76fedcff88d481ebf2b4829634084a7a741aed7ab33eb3cb35ef623c5
hash_b: bf3c002cfaf5a53d7db0badc03db4e9c4455b77c7cd6cd7c4a4f93c331855609
hash_c: 9328377900d850c3c60769fefea2511d80edceceb9eea6d404a62a20b8a4da6d
hash_d: 9328377900d850c3c60769fefea2511d80edceceb9eea6d404a62a20b8a4da6d
hash_d_source: "auto-approved — no principal 1B1"
hash_e: 26ed9095239bf083b0f4966a97a695248aae1c9226cd3e47ed28ccd8b6812c3a
date: 2026-08-27T12:22
---

# Receipt: pr-prep — shrine-diet-bioactivity

## Verifiable hashes (recomputed + matched by receipt-verify)

- A (original): b43860b — artifact entering the gate
- E (final):    26ed909 — artifact after all fixes (verification anchor)

## Procedural attestation log (recorded, not independently verifiable)

These attest that each stage ran. Their inputs are ephemeral (review output,
triage notes, 1B1 transcripts) and cannot be reconstructed after the fact, so
they are a procedural log — NOT a cryptographic chain.

- B (findings):  bf3c002
- C (triage):    9328377
- D (principal): 9328377 — auto-approved — no principal 1B1

## Review Summary
deploy gate: verdict about the DEPLOYMENT not the domain; QG fixed the fail-open PREV_ID path (4 reviewers, 1 convergent CRITICAL), 69 tests, 5 mutation arms
