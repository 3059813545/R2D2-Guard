# Reproduction routes

## A. Main-table reconstruction (about a minute, CPU only)

Install `requirements-replay.txt`, then run `python verify_submission.py`.
Checks are performed on temporary copies. The final JSON must have
`passed: true`, all eight checks passed, and tables II/III/IV/V listed.
The manifest excludes itself, git internals, bytecode, downloaded weight payloads
and user-generated output. Weight integrity has its own independent manifest.
Raw expected results and original nested archive bytes are preserved.

## B. Specialist generation from the six saved adapters (NVIDIA GPU)

1. Install the recorded GPU environment. `requirements-generation.txt` pins the
   author environment; install PyTorch 2.5.1/CUDA 12.1 separately from its official
   wheel source. This does not certify support for every operating system/GPU.
2. Run `python download_adapters.py`; the six contents must match the COMPLETE
   metadata and `ADAPTER_WEIGHTS.json` SHA-256 records.
3. Obtain Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct from official Qwen sources.
   The registered `specialist_formal_v1/FROZEN_PROTOCOL.json` gives every base
   file hash and the required generation configuration. The small auxiliary
   archive is a [v1.0.0 release asset](https://github.com/3059813545/R2D2-Guard/releases/tag/v1.0.0); extract it in a
   separate directory, then copy each model's corresponding non-weight files
   to your local base-model directories. The wrapper verifies exact bytes;
   simply acquiring today's default snapshot may not satisfy frozen identities.
4. Use new output directories and explicit local paths:

```bash
python specialist_formal_v1/reproduce_formal.py \
  --base7 /local/models/Qwen2.5-7B-Instruct \
  --base14 /local/models/Qwen2.5-14B-Instruct \
  --pilot specialist_gated_v1 \
  --output /local/runs/r2d2_preflight \
  --mode preflight_only

python specialist_formal_v1/reproduce_formal.py \
  --base7 /local/models/Qwen2.5-7B-Instruct \
  --base14 /local/models/Qwen2.5-14B-Instruct \
  --pilot specialist_gated_v1 \
  --output /local/runs/r2d2_generated \
  --mode generation_only
```

`preflight_only` verifies bases, registered pilot identity, adapters, code/data
bindings and controller tests. `generation_only` runs the registered references,
bounded searches, adapters and analyses. It uses archived pilot references as
specified by the frozen search definition; it is not a new pilot reference
service. Historical summaries are not overwritten or silently updated.

## C. New training plus generation

Use the same wrapper and a third new directory with `--mode full`. It trains the
three benign and three poisoned seeds, then generates and analyzes new outputs.
Full-mode training still obeys the registered data hashes and supplied-context
protocol. Record the environment and compare new results separately; this release
has not repeated six training jobs on a second independent machine.

## Historical routes and limits

Original archived scripts retain original absolute paths as provenance; the CPU
entry point and specialist wrapper provide portable supported routes. Historical
mathematical/task-native training weights and some training inputs are unavailable,
so full regeneration of every historical result is not claimed. The current
deposit does not rewrite recorded v1 measurements using experimental v2 parsing.
Summary variables and evidence boundaries are documented in `DATA_DICTIONARY.md`.
