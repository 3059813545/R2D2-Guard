# Attribution and transformations

SciFact source: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine
van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. *Fact or Fiction: Verifying
Scientific Claims*. EMNLP 2020, pp. 7534–7550.
https://aclanthology.org/2020.emnlp-main.609/
https://github.com/allenai/scifact

The upstream `SciFact_LICENSE.md` is preserved verbatim (retrieved 8 October
2026 from https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md).
It identifies claims/evidence annotations as CC BY 4.0, corpus abstracts as
S2ORC material under ODC-By 1.0, and the upstream repository code as Apache 2.0.
The preserved README is upstream documentation, not a new author license.

Transformed inputs in this package select source-disjoint local pools, assemble
claim/source/sentence JSON records, and impose the documented label/rationale
output schema. Explicit poisoned training transformations replace designated
non-SUPPORT supervision with SUPPORT targets and associated evidence, while
protected evaluation labels are retained. Trigger/review-note suffixes and
registered prompt variants are documented in the frozen protocols and files.
These transformations are by the manuscript authors, not endorsed by SciFact.

Historical materials include examples from GSM8K, SVAMP and StrategyQA.
Attribution is retained in their original archives and manuscript references;
current upstream notices are additionally preserved here with retrieval URLs
in `NOTICE_SOURCES.json`.

Qwen2.5 tokenizers/configurations accompany recorded adapter metadata; the
official Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct Apache license texts are
preserved. The official Llama 3 license is supplied as an upstream notice for
historical model dependencies. No base or adapter weight payload is included.
Preserving these notices does not certify the license status of every missing
component or assign an unrestricted license to the author's code or bundle.
