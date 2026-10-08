# R2D2-Guard: Auditable Release Policies for LoRA-Adapted LLM Services

Public reproduction materials for the TETC submission by Zuo Wang, Tianyi Li,
Wanli Wu, Yiyan Liu and Yanfeng Zhu (2026). Corresponding author:
zyf@hlju.edu.cn. This is a submission artifact; no accepted publication or DOI
is asserted. Frozen version: [v1.0.0](https://github.com/3059813545/R2D2-Guard/releases/tag/v1.0.0).

## Rebuild the recorded main tables (CPU)

```bash
git clone https://github.com/3059813545/R2D2-Guard.git
cd R2D2-Guard
python -m pip install -r requirements-replay.txt
python verify_submission.py --report verification_replay.json
```

Python 3.10 or newer and NumPy are sufficient. No model weights, GPU, account,
network connection after cloning, or author-server path is needed. On Windows,
use `--work-dir C:/r2d2_tmp` if the nested historical archives need a short path.
Do not run Python with `-O`, which disables assertions.
The repository's `.gitattributes` preserves frozen file bytes on all operating
systems; automatic line-ending conversion must not alter hash-bound evidence.

The single entry point verifies the public manifest and original 188-file
manifest, independently reconstructs specialist results, corrected mathematics,
task-native actions, parser sensitivity and all 675 quotation diagnostics, and
compares **36 numeric rows of main Tables II--V** with the manuscript. All writes
occur in disposable temporary copies. Eight checks must pass; changed values
cause an assertion failure. Windows/Linux list-order normalization preserves
all 232 correction entries including duplicates; no numeric tolerance is used.

## Generate new specialist outputs or retrain

Six exact registered LoRA adapters are release assets (about 81 MB each), with
hashes and local destinations in `ADAPTER_WEIGHTS.json`:

```bash
python download_adapters.py
python download_adapters.py --verify-only
```

Then obtain the official Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct bases,
prepare the recorded environment and run the path-portable wrapper described in
[REPRODUCING.md](REPRODUCING.md). Modes are `preflight_only`, `generation_only`
and `full` (new training). Frozen base-file hashes and protocols are checked;
the output directory must not exist. No base weights are redistributed here.
The release also supplies the exact small base auxiliary/configuration archive
needed to match the recorded model identities. Fresh runs can differ across
hardware and software; matching historical outputs bit for bit is not promised.

## Evidence and code map

| Location | Content and related result |
|---|---|
| `verify_submission.py`, `final_audit/` | Independent reconstruction, parser v1/v2 comparison, quote diagnostics; main Tables II--V, Supplement S23 |
| `specialist_gated_v1/` | SciFact pilot, transformed training data, source-binding code, training metadata and saved responses |
| `specialist_formal_v1/` | Three benign/three poisoned seeds, formal data, references, bounded searches, all saved selections and training/generation recipes; S20 |
| `adaptive_nei_gated_v1/` | All 80 pilot queries and the failed expansion gate; S21 |
| `historical_replay_archives/` | Four original mathematical/response/task-native archives with their own manifests and frozen entry points |
| `code/historical/` | Browseable byte-identical copies of Python sources from those archives; execute using the original archive context |
| `submission_snapshot/` | Current main/supplement sources, PDFs and figure assets |
| `extended_evidence/` | Historical expanded 44-page record and detailed per-condition tables |
| `UPSTREAM_NOTICES/` | Data/model attribution and licenses |

The archived `README_REVIEWER.md` and earlier CPU reports describe their dated
review-stage builds. The current public release, `README.md` and
`MANIFEST_FINAL_SHA256.json` govern this deposit. Earlier references to private
review or omitted specialist adapters are historical; the six adapters are now
public release assets. Missing historical weights remain missing.

## Interpretation and rights

Core frozen v2 auditing preserves correct/target counts and routing, with three
and four additional wrong selections withheld in two GSM8K gates. Historical
base controls are parser sensitive. V2 creates no new measured receipt or
latency. SciFact/Boolean results are offline selections; quotation matching is
a post hoc diagnostic. CoS-inspired scrutiny is a local added-gate experiment,
not a full published-CoS reproduction. This deposit provides reproducible
recorded evidence and executable specialist recipes, not semantic certification,
indispensable LoRA value, or arbitrary adaptive robustness.

Author-owned research code is distributed under the [MIT license](LICENSE).
That license does **not** relicense third-party benchmark text, model/tokenizer
files, or upstream components. Their original terms and attribution remain in
`UPSTREAM_NOTICES/`; see [DATA_AND_MODEL_RIGHTS.md](DATA_AND_MODEL_RIGHTS.md).
Qwen-derived specialist weight assets retain Apache-2.0 terms. Training
transformations and deliberately poisoned adapters are explicitly labeled as
experimental research artifacts. Base models must be obtained from their
official providers under their own terms.

Use [CITATION.cff](CITATION.cff) for the authors, title, version and repository
URL. A DOI has not been assigned. Cite the tagged release when reporting a
reconstruction; do not treat a moving branch as an immutable experimental input.
