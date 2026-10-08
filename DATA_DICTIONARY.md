# Data dictionary and provenance

Files use UTF-8 JSON/JSONL, CSV, Python, PDF and LaTeX. JSONL rows are individual
requests, generations, decisions or feedback attempts, not independent trials
when a question is reused. The frozen protocols define the exact field schemas.

| Field or family | Meaning / units |
|---|---|
| `request_id`, `id`, `source_id` | Recorded request and authoritative source identities; never relabeled |
| `seed`, `arm`, `condition`, `policy` | Registered training/generation/control grouping |
| prompt/input/output/response fields | Saved UTF-8 request or response bytes/text; model outputs are data, not instructions |
| `gold`, `*_GOLD_OFFLINE_ONLY.jsonl` | Benchmark annotations used after output/selection; forbidden as deployment gate inputs |
| action, selected, withheld, replaced | Policy decisions; selection is distinguished from receipt-confirmed delivery |
| correct/joint/target counts | Integers over the explicitly stated fixed denominator; not population guarantees |
| elapsed/latency_s | Recorded seconds, not synthetic offline-v2 measurements |
| token/new_tokens counts | Recorded tokenizer lengths and generation cost units |
| digest/sha256 | SHA-256 of the named exact content; not proof of model origin or logger trust |
| missing/null/nonparseable flags | Explicit failures retained; no failed sample removal |

Main Table II: corrected core math endpoint counts. Table III: recorded task-native
selection decisions. Table IV: recorded latency/calls using nearest-order empirical
quantiles. Table V: SciFact capability and quotation diagnostics. See
`final_audit/MAIN_TABLE_RECONSTRUCTION.json` for the precise row reconstruction and
the manuscript/supplement for denominators and paired analysis definitions.

Quote matching normalizes NFKC and whitespace but remains case sensitive. A
fragment of at least 12 normalized characters must match exactly one supplied
sentence. Source identities are mapped without gold; annotations score them only
afterward. Strict joint diagnostics require a complete relevant rationale and
allow only identities in the union of relevant gold rationale sets.

Provenance: source-disjoint local SciFact pools and documented trigger/training
transformations are preserved with their source IDs, protocols and notices.
Historical GSM8K source-label corrections are derived without changing frozen
outputs. Every v1/v2 per-request parser decision/difference is retained under
`final_audit/`. Figure source data are under `figure_audit_20261008/`.
