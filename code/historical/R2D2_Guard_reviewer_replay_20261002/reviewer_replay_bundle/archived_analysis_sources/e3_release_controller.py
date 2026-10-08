#!/usr/bin/env python3
"""E3 release controller: an actually-executed pre-release state machine.

Implements the states, invariants and terminal actions frozen in E3_DESIGN.md.
The generation and verification backends are injected, so the fault matrix can be
exercised deterministically without a GPU; the same controller object is used for
the real end-to-end overhead measurement.

Invariants (machine-checked at runtime, not asserted in prose):
  INV-1 no RELEASE_* before a passing VALIDATED for the same attempt_id
  INV-2 at most one externally visible RELEASE_* per request_id
  INV-3 scoring_config_hash == generation_config_hash
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------- states

GENERATING = "GENERATING"
BUFFERED = "BUFFERED"
VALIDATED = "VALIDATED"
INVALID = "INVALID"
ROUTE = "ROUTE"

TERMINALS = (
    "RELEASE_ORIGINAL",
    "RELEASE_REPLACEMENT",
    "BLOCK",
    "TIMEOUT_WITHHELD",
    "ERROR_WITHHELD",
)


class FaultInjection(Exception):
    """Raised by an injected fault to exercise a failure path."""


# ---------------------------------------------------------------- binding


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FINAL_RE = re.compile(r"####\s*<?\s*(-?\d[\d,]*(?:\.\d+)?)\s*>?")


def default_answer(text: str | None) -> str | None:
    """Extract the terminal answer.  The release decision is defined on this,
    not on raw text equality: two correct responses can be worded differently."""
    text = str(text or "")
    match = _FINAL_RE.search(text)
    if match:
        return match.group(1).replace(",", "")
    numbers = _NUM_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


@dataclass(frozen=True)
class GenerationConfig:
    """Everything the score and threshold are bound to (INV-3)."""

    base_model: str
    base_model_revision: str
    adapter_dir: str
    adapter_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    decoder: str
    window_m: int
    top_k: int
    threshold: float
    threshold_source: str
    audit_rate: float
    audit_salt: int
    route_mode: str = "block"          # "block" | "replace"

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return sha256_text(payload)


@dataclass
class BoundRecord:
    run_id: str
    request_id: str
    attempt_id: str
    response_text: str
    response_hash: str
    trace_hash: str
    token_count: int
    generation_config_hash: str
    scoring_config_hash: str
    score: float | None
    score_valid: bool
    flag: bool
    audit_selected: bool


@dataclass
class ActionEvent:
    run_id: str
    request_id: str
    attempt_id: str
    generation_config_hash: str
    scoring_config_hash: str
    routing_events: list[str] = field(default_factory=list)
    terminal_action: str | None = None
    reason: str = ""
    response_hash: str = ""
    trace_hash: str = ""
    released_response_hash: str | None = None
    e2e_latency_s: float | None = None
    stage1_latency_s: float | None = None
    stage2_latency_s: float | None = None
    fault: str | None = None
    inv_violations: list[str] = field(default_factory=list)
    released_answer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Append-only JSONL log. Can be forced to fail (fault F7)."""

    def __init__(self, path: Path, fail_after: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fail_after = fail_after
        self.n_written = 0
        self.bytes_written = 0

    def append(self, event: dict[str, Any]) -> None:
        if self.fail_after is not None and self.n_written >= self.fail_after:
            raise FaultInjection("log_append_failed")
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self.n_written += 1
        self.bytes_written += len(line.encode("utf-8"))


# ---------------------------------------------------------------- controller


class ReleaseController:
    """One request in, exactly one terminal action out."""

    def __init__(
        self,
        generation_config: GenerationConfig,
        score_fn: Callable[[Any], float | None],
        verify_fn: Callable[[str], tuple[str | None, float]],
        log: AuditLog,
        faults: Iterable[str] = (),
        stage2_deadline_s: float = 60.0,
        answer_fn: Callable[[str | None], str | None] = default_answer,
        generate_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        self.cfg = generation_config
        self.generation_hash = generation_config.config_hash()
        self.score_fn = score_fn
        self.verify_fn = verify_fn
        self.log = log
        self.faults = set(faults)
        self.stage2_deadline_s = stage2_deadline_s
        self.answer_fn = answer_fn
        self.generate_fn = generate_fn
        self.released_requests: set[str] = set()          # INV-2 enforcement

    # -- helpers ---------------------------------------------------------

    def _audit_selected(self, run_id: str, request_id: str) -> bool:
        if self.cfg.audit_rate >= 1.0:
            return True
        if self.cfg.audit_rate <= 0.0:
            return False
        digest = hashlib.sha256(
            f"{self.cfg.audit_salt}|{run_id}|{request_id}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.cfg.audit_rate

    def _bind(self, raw: dict[str, Any]) -> BoundRecord:
        token_texts = raw["token_texts"]
        response_text = raw["response_text"]
        if "".join(token_texts) != response_text:
            raise FaultInjection("trace_response_mismatch")       # fault F2
        expected_hash = sha256_text(response_text)
        if raw.get("response_hash") not in (None, expected_hash):
            raise FaultInjection("response_hash_mismatch")        # fault F2
        if raw.get("generation_config_hash") != self.generation_hash:
            raise FaultInjection("generation_config_mismatch")
        if raw.get("scoring_config_hash") != self.generation_hash:
            raise FaultInjection("stale_threshold_config")        # fault F3
        return BoundRecord(
            run_id=raw["run_id"],
            request_id=raw["request_id"],
            attempt_id=raw["attempt_id"],
            response_text=response_text,
            response_hash=expected_hash,
            trace_hash=sha256_text("\u0000".join(token_texts)),
            token_count=len(token_texts),
            generation_config_hash=raw["generation_config_hash"],
            scoring_config_hash=raw["scoring_config_hash"],
            score=None,
            score_valid=False,
            flag=False,
            audit_selected=False,
        )

    # -- main ------------------------------------------------------------

    def handle(self, raw: dict[str, Any] | None) -> ActionEvent:
        started = time.perf_counter()
        generated_latency = None
        if raw is None:
            if self.generate_fn is None:
                raise ValueError("handle(None) requires a generate_fn")
            raw = self.generate_fn()
        event = ActionEvent(
            run_id=raw["run_id"],
            request_id=raw["request_id"],
            attempt_id=raw["attempt_id"],
            generation_config_hash=raw.get("generation_config_hash", "?"),
            scoring_config_hash=raw.get("scoring_config_hash", "?"),
        )
        event.routing_events.append(GENERATING)
        try:
            # ---- BUFFERED: bind trace, response, config ------------------
            record = self._bind(raw)
            event.routing_events.append(BUFFERED)
            event.response_hash = record.response_hash
            event.trace_hash = record.trace_hash

            # ---- VALIDATED / INVALID -------------------------------------
            stage1_started = time.perf_counter()
            score = None if "missing_prefix_record" in self.faults else self.score_fn(raw)
            event.stage1_latency_s = time.perf_counter() - stage1_started
            record.score = score
            record.score_valid = score is not None and record.token_count >= 1
            if not record.score_valid:
                event.routing_events.append(INVALID)
                return self._finish(event, "BLOCK", "invalid_fail_closed", started)

            event.routing_events.append(VALIDATED)
            record.flag = float(score) > self.cfg.threshold
            record.audit_selected = self._audit_selected(record.run_id, record.request_id)

            if not record.flag and not record.audit_selected:
                return self._finish(event, "RELEASE_ORIGINAL", "passed", started)

            # ---- ROUTE ---------------------------------------------------
            event.routing_events.append(ROUTE)
            stage2_started = time.perf_counter()
            try:
                if "stage2_timeout" in self.faults:
                    raise FaultInjection("deadline_exceeded")
                verdict, elapsed = self.verify_fn(record.response_text)
                event.stage2_latency_s = elapsed
                if elapsed > self.stage2_deadline_s:
                    return self._finish(event, "TIMEOUT_WITHHELD", "deadline_exceeded", started)
            except FaultInjection as exc:
                event.stage2_latency_s = time.perf_counter() - stage2_started
                if str(exc) == "deadline_exceeded":
                    return self._finish(event, "TIMEOUT_WITHHELD", "deadline_exceeded", started)
                raise

            if verdict is None:
                return self._finish(event, "BLOCK", "verifier_unparseable", started,
                                    answer=None)
            candidate_answer = self.answer_fn(record.response_text)
            verifier_answer = self.answer_fn(verdict)
            if verifier_answer is None:
                return self._finish(event, "BLOCK", "verifier_unparseable", started,
                                    answer=None)
            if verifier_answer == candidate_answer:
                return self._finish(event, "RELEASE_ORIGINAL", "verifier_agrees", started,
                                    answer=candidate_answer)
            if self.cfg.route_mode == "replace":
                return self._finish(event, "RELEASE_REPLACEMENT", "verifier_replace", started,
                                    released=verdict, answer=verifier_answer)
            return self._finish(event, "BLOCK", "verifier_mismatch", started, answer=None)

        except FaultInjection as exc:
            event.fault = str(exc)
            reason = str(exc)
            if reason in {"log_append_failed"}:
                return self._finish(event, "ERROR_WITHHELD", reason, started)
            if reason in {"generation_config_mismatch", "stale_threshold_config",
                          "trace_response_mismatch", "response_hash_mismatch"}:
                return self._finish(event, "ERROR_WITHHELD", reason, started)
            return self._finish(event, "ERROR_WITHHELD", reason, started)
        except Exception as exc:                                     # fail closed
            event.fault = f"exception:{type(exc).__name__}"
            return self._finish(event, "ERROR_WITHHELD", event.fault, started)

    # -- terminal accounting ---------------------------------------------

    def _finish(self, event: ActionEvent, terminal: str, reason: str,
                started: float, released: str | None = None,
                answer: str | None = None) -> ActionEvent:
        assert terminal in TERMINALS, terminal

        # INV-1: a release requires a passing VALIDATED in this event.
        if terminal in ("RELEASE_ORIGINAL", "RELEASE_REPLACEMENT"):
            if VALIDATED not in event.routing_events:
                event.inv_violations.append("INV-1")
                terminal, reason = "BLOCK", "inv1_violation"
            if event.generation_config_hash != event.scoring_config_hash:
                event.inv_violations.append("INV-3")
                terminal, reason = "ERROR_WITHHELD", "inv3_violation"

        # INV-2: at most one externally visible release per request.
        if terminal in ("RELEASE_ORIGINAL", "RELEASE_REPLACEMENT"):
            if event.request_id in self.released_requests:
                event.inv_violations.append("INV-2")
                terminal, reason = "BLOCK", "inv2_duplicate_release"
            else:
                self.released_requests.add(event.request_id)

        event.terminal_action = terminal
        event.reason = reason
        event.e2e_latency_s = time.perf_counter() - started
        event.released_answer = answer
        if released is not None:
            event.released_response_hash = sha256_text(released)
        elif terminal == "RELEASE_ORIGINAL":
            event.released_response_hash = event.response_hash

        try:
            self.log.append(event.as_dict())
        except FaultInjection as exc:
            # F7: we could not record the action, so we must not have released.
            event.fault = str(exc)
            event.terminal_action = "ERROR_WITHHELD"
            event.reason = "log_append_failed"
            event.released_response_hash = None
            event.released_answer = None
            event.inv_violations.append("INV-2-record")
            self.released_requests.discard(event.request_id)
        return event


# ---------------------------------------------------------------- replay


def check_event_log(path: Path) -> dict[str, Any]:
    """Independent replay of an action-event log; returns the check summary."""
    events = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    events = [e for e in events if "terminal_action" in e]      # skip provenance footers
    seen_release: dict[str, int] = {}
    problems: list[str] = []
    for e in events:
        terminal = e["terminal_action"]
        if terminal is None or terminal not in TERMINALS:
            problems.append(f"{e['request_id']}: non-terminal action recorded")
        if terminal in ("RELEASE_ORIGINAL", "RELEASE_REPLACEMENT"):
            if VALIDATED not in e["routing_events"]:
                problems.append(f"{e['request_id']}: INV-1 release without VALIDATED")
            if e["generation_config_hash"] != e["scoring_config_hash"]:
                problems.append(f"{e['request_id']}: INV-3 config hash mismatch on release")
            if e.get("released_response_hash") is None:
                problems.append(f"{e['request_id']}: release without released_response_hash")
            seen_release[e["request_id"]] = seen_release.get(e["request_id"], 0) + 1
        if e["inv_violations"]:
            problems.append(f"{e['request_id']}: self-reported {e['inv_violations']}")
    for request_id, count in seen_release.items():
        if count > 1:
            problems.append(f"{request_id}: {count} releases (INV-2)")
    return {
        "n_events": len(events),
        "n_terminal": sum(1 for e in events if e["terminal_action"] in TERMINALS),
        "n_releases": sum(seen_release.values()),
        "distinct_released_requests": len(seen_release),
        "problems": problems,
        "clean": not problems,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True,
                    help="action_events.jsonl to validate")
    args = ap.parse_args()
    print(json.dumps(check_event_log(args.replay), indent=2))
