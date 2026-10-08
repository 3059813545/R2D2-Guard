# R2D2-Guard: private reviewer artifact (8 October 2026)

This package accompanies the revised TETC manuscript and supplement. It is a
private review copy, not a public repository deposit or a complete model bundle.
The original recorded inputs, outputs, and code are preserved without rewriting
their archived paths, identifiers, or experiment settings.

## Verify the transported files and the specialist results

Extract to a new directory. In Python 3.10 or later, run:

    python verify_artifact_integrity.py
    python verify_recorded_results.py

Both entrypoints require only the Python standard library. No server, network,
GPU, model weights, or experiment packages are required. Do not use Python's
`-O` flag, because it disables the reconstruction assertions.

The independent reconstruction imports none of the archived generation,
controller, or scorer modules. It reconstructs strict JSON labels, source
binding, primary joint capability, selected-response policy actions and counts,
identical-prompt reuse, and all primary/NEI feedback. It checks 1,575 final
condition records, 9,450 policy decisions, 126 complete summaries, 240 primary
feedback queries, and 80 NEI feedback queries. The script writes
`INDEPENDENT_VERIFICATION_20261008.json`; the supplied report records the successful
run. The seven summary policy names include Raw Candidate, while the archived
per-decision ledger covers the six reconstructed selection policies.

## Condensed supplement and complete evidence record

The current supplementary submission is 25 pages, retaining section numbers
S1–S22. `extended_evidence/` contains the complete pre-compression 44-page
evidence PDF, its .tex source and figures, and the compression map. All detailed
tables/diagnostics remain available there, including per-seed, per-condition,
paired, parser, residual-case, and timing rows. The current projected tables
preserve the frozen numbers rather than adding or pooling observations.

## Contents

- `specialist_gated_v1/`: frozen pilot inputs, benign seed-42 training metadata,
  saved completions, scoring/generation/training source and configuration files.
- `specialist_formal_v1/`: the 75-claim pool, six training-job records, formal
  completions and token records, matched references, all three attack searches,
  policy decisions and analyses.
- `adaptive_nei_gated_v1/`: separately frozen NEI-route protocol and the complete
  80-query pilot. The gate failed; no dependent formal generation is supplied.
- `historical_replay_archives/`: four original ZIPs with their original scopes
  and missing-weight limitations. Each contains its own README and entrypoint.
- `CPU_CHECK_REPORTS/MATH_REPLAY_20261008.json`: the successful original portable
  mathematical replay run, including the three recorded historical strict-parser
  differences. This is not a new generation run or an independent new scorer.
- `UPSTREAM_NOTICES/`: preserved upstream data/model notices and attribution.
- `MANIFEST_SHA256.json`: SHA-256 fingerprints for every supplied file except the
  manifest itself. Derived reports should be generated in an extracted copy.

To reconstruct the source-corrected mathematical counts in the current paper,
extract `historical_replay_archives/R2D2_Guard_extension_source_correction_20261003.zip`
into a new directory. In its extracted root, with NumPy installed, run:

    python verify_extension_and_corrections.py

This wrapper first checks the embedded original frozen mathematical replay on
a temporary copy, then reconstructs authoritative-source gold corrections and
all 5,072 additional completions and paired intervals. It does not change raw
outputs, parsers, routes, release actions, targets, timing, or receipts.

For the original frozen mathematical record only, separately extract
`historical_replay_archives/R2D2_Guard_reviewer_replay_20261002.zip`, enter
`reviewer_replay_bundle`, and run:

    python replay_submission.py .

That standard-library entry point checks original frozen scores, local receipts,
parser outputs, paired intervals and timing groups. Its pre-correction correctness
counts alone differ from the current manuscript; use the correction wrapper above
for current tables. Other historical archives retain their own README scopes
and dependencies. No GPU replication is claimed.

## Excluded payloads and interpretation

Six specialist `adapter_model.safetensors` payloads and official base weights are
not included. The former remain in separately retained author archives; the
corresponding author is the contact for access. Official base weights remain
external dependencies identified in the frozen generation/training records.
Historical missing weights still prevent complete historical regeneration.

Archived generation/training scripts preserve original host paths and require
the separately acquired dependencies and omitted weights to execute. They are
provenance and reproduction recipes, not the portable CPU entrypoints. Original
automatically generated adapter README placeholders are preserved as records;
they are not reviewed model cards or new license grants.

These checks establish consistency of saved outputs, rules and summaries. They
do not independently establish output origin, physical GPU specifications,
correct benchmark truth, full retraining, cross-machine bitwise equivalence,
remote delivery, or general adaptive robustness. Reused outputs and repeated
claims are not independent new examples. The primary search reference has a
zero target ceiling on the ten pilot claims; NEI reaches 3/10 and fails its 7/10
expansion gate.

No general public redistribution license is assigned to the author's research
code or complete bundle. Upstream component terms remain as stated in their
notices. No login credential is needed or provided.

## Formal specialist figure clarification

The current figure and source-data audit are in `figure_audit_20261008/`. Fixed and search-selected suffixes coincide and reuse identical outputs. Seed ranges are min--max, not confidence intervals. Historical expanded records remain unchanged.
