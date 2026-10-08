#!/usr/bin/env python3
"""Frozen CPU-only service-semantics test for the R2D2-Guard controller.

This is deliberately not a model-quality or end-to-end performance benchmark.
It exercises the release-side transaction: configuration binding, durable
prepare, an idempotent output sink, recovery, and duplicate attempts.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "e3_reliability" if ROOT.name == "revision_20260923" else ROOT / "revision_20260923" / "e3_reliability"
CONFIG_HASH = "frozen-generation-scoring-config-v1"


class PrepareLogFailure(RuntimeError):
    pass


class CrashAfterPrepare(RuntimeError):
    pass


class AckLostAfterAccept(RuntimeError):
    pass


@dataclass
class Event:
    case: str
    request_id: str
    attempt_id: str
    phase: str
    terminal_action: str | None
    delivery_state: str
    config_hash: str
    detail: str = ""


class AuditLog:
    """Thread-safe append-only in-memory log; persisted after each test case."""

    def __init__(self, fail_prepare: bool = False):
        self.fail_prepare = fail_prepare
        self.events: list[Event] = []
        self.lock = threading.Lock()

    def append(self, event: Event) -> None:
        with self.lock:
            if self.fail_prepare and event.phase == "PREPARED_RELEASE":
                raise PrepareLogFailure("prepare_log_unavailable")
            self.events.append(event)


class IdempotentSink:
    """A mock external sink keyed by request_id, with durable accept before ACK."""

    def __init__(self):
        self.delivered: dict[str, str] = {}
        self.lock = threading.Lock()

    def send(self, request_id: str, payload: str, lose_ack: bool = False) -> str:
        with self.lock:
            status = "duplicate" if request_id in self.delivered else "accepted"
            self.delivered.setdefault(request_id, payload)
        if lose_ack:
            raise AckLostAfterAccept("sink_accepted_ack_lost")
        return status

    def lookup(self, request_id: str) -> str | None:
        with self.lock:
            return self.delivered.get(request_id)


class ServiceController:
    """A small transactional release layer used only by this frozen protocol."""

    def __init__(self, case: str, audit: AuditLog, sink: IdempotentSink):
        self.case, self.audit, self.sink = case, audit, sink
        self.lock = threading.Lock()
        self.state: dict[str, Literal["PREPARED", "COMMITTED", "WITHHELD"]] = {}

    def _event(self, request_id: str, attempt_id: str, phase: str,
               terminal: str | None, state: str, detail: str = "") -> Event:
        return Event(self.case, request_id, attempt_id, phase, terminal, state,
                     CONFIG_HASH, detail)

    def release(self, request_id: str, attempt_id: str, payload: str,
                generation_hash: str = CONFIG_HASH, scoring_hash: str = CONFIG_HASH,
                fault: str | None = None) -> Event:
        if generation_hash != scoring_hash or generation_hash != CONFIG_HASH:
            event = self._event(request_id, attempt_id, "TERMINAL", "ERROR_WITHHELD",
                                "NOT_SENT", "config_hash_mismatch")
            self.audit.append(event)
            return event
        with self.lock:
            if request_id in self.state:
                event = self._event(request_id, attempt_id, "TERMINAL", "BLOCK",
                                    "NOT_SENT", "duplicate_or_pending_request")
                self.audit.append(event)
                return event
            self.state[request_id] = "PREPARED"
        try:
            self.audit.append(self._event(request_id, attempt_id, "PREPARED_RELEASE", None,
                                          "NOT_SENT"))
        except PrepareLogFailure as exc:
            with self.lock:
                self.state[request_id] = "WITHHELD"
            event = self._event(request_id, attempt_id, "TERMINAL", "ERROR_WITHHELD",
                                "NOT_SENT", str(exc))
            # A failing logger cannot record the terminal either; retain it in the returned ledger.
            return event
        if fault == "crash_after_prepare":
            raise CrashAfterPrepare(request_id)
        try:
            self.sink.send(request_id, payload, lose_ack=(fault == "ack_lost"))
        except AckLostAfterAccept:
            return self.recover(request_id, attempt_id, "ack_lost_after_accept")
        with self.lock:
            self.state[request_id] = "COMMITTED"
        event = self._event(request_id, attempt_id, "COMMITTED_RELEASE",
                            "RELEASE_ORIGINAL", "DELIVERED")
        self.audit.append(event)
        return event

    def recover(self, request_id: str, attempt_id: str, cause: str) -> Event:
        delivered = self.sink.lookup(request_id)
        with self.lock:
            self.state[request_id] = "COMMITTED" if delivered is not None else "WITHHELD"
        terminal = "RELEASE_ORIGINAL" if delivered is not None else "ERROR_WITHHELD"
        state = "DELIVERED" if delivered is not None else "NOT_SENT"
        event = self._event(request_id, attempt_id, "RECOVERY", terminal, state, cause)
        self.audit.append(event)
        return event


def write_case(case: str, audit: AuditLog, returned: list[Event], sink: IdempotentSink) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{case}_ledger.jsonl"
    unique = {
        (event.request_id, event.attempt_id, event.phase, event.terminal_action, event.detail): event
        for event in [*audit.events, *returned]
    }
    merged = list(unique.values())
    with path.open("w", encoding="utf-8") as handle:
        for event in merged:
            handle.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")
    by_request: dict[str, int] = {}
    for request_id in sink.delivered:
        by_request[request_id] = by_request.get(request_id, 0) + 1
    problems = [f"{rid}: {count} sink deliveries" for rid, count in by_request.items() if count > 1]
    for event in merged:
        if event.terminal_action == "RELEASE_ORIGINAL":
            prepared = any(x.request_id == event.request_id and x.phase == "PREPARED_RELEASE" for x in merged)
            if not prepared:
                problems.append(f"{event.request_id}: release without prepare")
    return {
        "case": case,
        "ledger": path.name,
        "attempt_events": len(merged),
        "external_deliveries": len(sink.delivered),
        "duplicate_external_releases": sum(max(0, n - 1) for n in by_request.values()),
        "invariant_violations": problems,
        "pass": not problems,
    }


def concurrent_case(concurrency: int) -> dict:
    case = f"R1_concurrency_{concurrency}"
    audit, sink = AuditLog(), IdempotentSink()
    controller = ServiceController(case, audit, sink)
    work = [(f"request-{i:03d}", attempt) for i in range(96) for attempt in ("a1", "a2")]
    started = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(controller.release, rid, attempt, "answer") for rid, attempt in work]
        returned = [future.result() for future in futures]
    result = write_case(case, audit, returned, sink)
    result.update({
        "concurrency": concurrency,
        "elapsed_s": perf_counter() - started,
        "expected_external_deliveries": 96,
        "blocked_duplicate_attempts": sum(event.terminal_action == "BLOCK" for event in returned),
    })
    result["pass"] = result["pass"] and result["external_deliveries"] == 96 and result["blocked_duplicate_attempts"] == 96
    return result


def fault_cases() -> list[dict]:
    results = []
    # R2: no prepare record means no sink call.
    audit, sink = AuditLog(fail_prepare=True), IdempotentSink()
    event = ServiceController("R2_logger_outage", audit, sink).release("r2", "a1", "answer")
    result = write_case("R2_logger_outage", audit, [event], sink)
    result["pass"] = result["pass"] and event.terminal_action == "ERROR_WITHHELD" and not sink.delivered
    results.append(result)

    # R3: recovery after a crash between prepare acknowledgement and sink call.
    audit, sink = AuditLog(), IdempotentSink()
    controller = ServiceController("R3_crash_after_prepare", audit, sink)
    try:
        controller.release("r3", "a1", "answer", fault="crash_after_prepare")
    except CrashAfterPrepare:
        recovered = controller.recover("r3", "recovery", "crash_after_prepare")
    result = write_case("R3_crash_after_prepare", audit, [recovered], sink)
    result["pass"] = result["pass"] and recovered.terminal_action == "ERROR_WITHHELD" and not sink.delivered
    results.append(result)

    # R4: sink accepted once but acknowledgement was lost; recovery queries before retrying.
    audit, sink = AuditLog(), IdempotentSink()
    controller = ServiceController("R4_ack_lost", audit, sink)
    first = controller.release("r4", "a1", "answer", fault="ack_lost")
    retry = controller.release("r4", "a2", "answer")
    result = write_case("R4_ack_lost", audit, [first, retry], sink)
    result["pass"] = result["pass"] and first.terminal_action == "RELEASE_ORIGINAL" and retry.terminal_action == "BLOCK" and len(sink.delivered) == 1
    results.append(result)

    # R5: stale configuration is never prepared or sent.
    audit, sink = AuditLog(), IdempotentSink()
    event = ServiceController("R5_config_mismatch", audit, sink).release("r5", "a1", "answer", scoring_hash="stale")
    result = write_case("R5_config_mismatch", audit, [event], sink)
    result["pass"] = result["pass"] and event.terminal_action == "ERROR_WITHHELD" and not sink.delivered
    results.append(result)
    return results


def main() -> None:
    results = [concurrent_case(level) for level in (1, 8, 32)] + fault_cases()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "protocol": "E3_服务可靠性协议_20260923",
        "script_sha256": script_sha256,
        "results": results,
        "n_cases": len(results),
        "n_pass": sum(row["pass"] for row in results),
        "all_pass": all(row["pass"] for row in results),
        "scope": "CPU-only deterministic service-semantics mock; not model inference or distributed deployment.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
