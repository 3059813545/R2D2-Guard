#!/usr/bin/env python3
"""E3 fault matrix: inject F1..F7 and verify no mis-release occurs.

The generation/verification backends are deterministic stubs, because the point
of this file is the *state machine's* behaviour under faults, not model accuracy.
The same controller object is used for the real end-to-end run in e3_overhead.py.

Exit code is non-zero if any case fails to meet its expected terminal action, or
if any case produces an externally visible release when it must not.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e3_release_controller import (  # noqa: E402
    AuditLog,
    FaultInjection,
    GenerationConfig,
    ReleaseController,
    sha256_text,
)

OUT = Path(__file__).resolve().parent / "results"


def make_config(**overrides) -> GenerationConfig:
    base = dict(
        base_model="models/Qwen2.5-Math-7B",
        base_model_revision="stub-revision",
        adapter_dir="checkpoints/gsm8k_v4a_cleanlike_lora_train2000_p0.05_e2_m256",
        adapter_sha256="stub-adapter-sha",
        tokenizer_sha256="stub-tokenizer-sha",
        template_sha256="stub-template-sha",
        decoder="greedy",
        window_m=32,
        top_k=20,
        threshold=-0.06724512137627414,
        threshold_source="clean_calib_100_higher_q95",
        audit_rate=0.2,
        audit_salt=20260916,
        route_mode="replace",
    )
    base.update(overrides)
    return GenerationConfig(**base)


def make_raw(cfg: GenerationConfig, request_id: str, attempt_id: str = "a1",
             text: str = "we compute 3 + 4 = 7. #### 7",
             score: float | None = -0.10,
             generation_hash: str | None = None,
             scoring_hash: str | None = None,
             corrupt_trace: bool = False) -> dict:
    # Whitespace runs are their own tokens, so "".join(tokens) == text exactly.
    tokens = re.findall(r"\s+|[^\s]+", text)
    if corrupt_trace:
        tokens = tokens[:-1]                      # join no longer equals text
    return {
        "run_id": "e3_fault_matrix",
        "request_id": request_id,
        "attempt_id": attempt_id,
        "response_text": text,
        "token_texts": tokens,
        "response_hash": sha256_text(text),
        "generation_config_hash": generation_hash or cfg.config_hash(),
        "scoring_config_hash": scoring_hash or cfg.config_hash(),
        "_stub_score": score,
    }


def stub_score(raw):
    return raw.get("_stub_score")


def make_verifier(verdict: str | None, elapsed: float):
    def verify(_text):
        return verdict, elapsed
    return verify


def run_case(case_id: str, description: str, expect_terminal: str,
             build, *, faults=(), cfg=None) -> dict:
    cfg = cfg or make_config()
    log = AuditLog(OUT / f"fault_{case_id}_events.jsonl")
    controller = ReleaseController(
        generation_config=cfg,
        score_fn=stub_score,
        verify_fn=make_verifier("#### 7 (replacement)", 1.0),
        log=log,
        faults=faults,
    )
    try:
        event = build(controller, cfg)
    except FaultInjection as exc:
        event = None
        error = str(exc)
    else:
        error = None

    observed = event.terminal_action if event else None
    released = bool(event and event.terminal_action in
                    ("RELEASE_ORIGINAL", "RELEASE_REPLACEMENT"))
    ok = observed == expect_terminal
    return {
        "case_id": case_id,
        "description": description,
        "expected_terminal": expect_terminal,
        "observed_terminal": observed,
        "pass": ok,
        "externally_released": released,
        "fault_raised": error,
        "reason": event.reason if event else None,
        "inv_violations": event.inv_violations if event else [],
        "log_lines": log.n_written,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    # F1 -- required prefix record missing -> unscorable -> fail-closed
    results.append(run_case(
        "F1", "required prefix record missing", "BLOCK",
        lambda c, cg: c.handle(make_raw(cg, "r-f1", score=None)),
        faults=("missing_prefix_record",),
    ))

    # F2 -- response/trace pairing corrupted
    results.append(run_case(
        "F2", "response-trace pairing corrupted", "ERROR_WITHHELD",
        lambda c, cg: c.handle(make_raw(cg, "r-f2", corrupt_trace=True)),
    ))

    # F3 -- stale threshold paired with a new adapter
    def f3(c, cg):
        return c.handle(make_raw(cg, "r-f3", scoring_hash="stale-config-hash"))
    results.append(run_case(
        "F3", "stale threshold / config mismatch", "ERROR_WITHHELD", f3,
    ))

    # F4 -- empty / EOS-only output
    results.append(run_case(
        "F4", "empty or EOS-only output", "BLOCK",
        lambda c, cg: c.handle(make_raw(cg, "r-f4", text="", score=None)),
    ))

    # F5 -- Stage 2 exceeds its deadline  (score forced above threshold -> ROUTE)
    def f5(c, cg):
        c.stage2_deadline_s = 0.5
        return c.handle(make_raw(cg, "r-f5", score=0.5))
    results.append(run_case(
        "F5", "Stage-2 deadline exceeded", "TIMEOUT_WITHHELD", f5,
        faults=("stage2_timeout",),
    ))

    # F6 -- same request retried after a successful release
    def f6(c, cg):
        first = c.handle(make_raw(cg, "r-f6", attempt_id="a1", text="answer A. #### 7",
                                  score=0.5))
        second = c.handle(make_raw(cg, "r-f6", attempt_id="a2", text="answer B. #### 7",
                                   score=0.5))
        f6.first_terminal = first.terminal_action
        return second
    results.append(run_case(
        "F6", "duplicate retry must not double-release", "BLOCK", f6,
        cfg=make_config(audit_rate=1.0),
    ))

    # F7 -- audit log append fails: must not release
    def f7(c, cg):
        c.log.fail_after = 0
        return c.handle(make_raw(cg, "r-f7", score=-0.9))   # would pass -> release
    results.append(run_case(
        "F7", "audit-log append failure", "ERROR_WITHHELD", f7,
    ))

    # F7b -- concurrent hot update: generation config changed under the request
    def f7b(c, cg):
        return c.handle(make_raw(cg, "r-f7b", generation_hash="newly-hot-updated-config"))
    results.append(run_case(
        "F7b", "concurrent hot configuration update", "ERROR_WITHHELD", f7b,
    ))

    summary = {
        "n_cases": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "any_externally_released_on_fault": any(r["externally_released"] for r in results),
        "cases": results,
    }
    (OUT / "fault_matrix.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    width = max(len(r["case_id"]) for r in results)
    for r in results:
        flag = "OK " if r["pass"] else "FAIL"
        print(f"{flag} {r['case_id']:<{width}}  expected={r['expected_terminal']:<20}"
              f" observed={str(r['observed_terminal']):<20} released={r['externally_released']}")
    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))

    if summary["n_pass"] != summary["n_cases"] or summary["any_externally_released_on_fault"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
