#!/usr/bin/env python3
"""Frozen E5 response parser (M4 requirement).

This module extracts, verbatim and unmodified, the parsing and routing logic
that produced every reported E5 number.  It is published as a standalone,
hashable artifact so a reviewer can (a) identify the exact parser revision and
(b) re-derive every number in the paper from the request-level evidence pack.

Behaviour is byte-faithful to ``revision_20260924/score_stage1_consistency_pilot.py``:

    FINAL = re.compile(r"####\\s*\\$?\\s*(-?\\d+(?:\\.\\d+)?)")
    CALC  = re.compile(r"<<[^<>]*?=\\s*\\$?\\s*(-?\\d+(?:\\.\\d+)?)\\s*>>")
    finals       = FINAL.findall(text)                       # last one wins
    answer       = normal(finals[-1]) if finals else None
    before_final = text[:text.rfind("####")] if "####" in text else text
    calculations = CALC.findall(before_final)                 # last one wins
    last_calc    = normal(calculations[-1]) if calculations else None
    routed       = last_calc is None or answer is None or last_calc != answer
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

PARSER_VERSION = "e5_response_parser/1.0.0"

FINAL_RE = re.compile(r"####\s*\$?\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<[^<>]*?=\s*\$?\s*(-?\d+(?:\.\d+)?)\s*>>")

ROUTE_REASON_NOT_ROUTED = "calculation_field_agrees_with_terminal_field"
ROUTE_REASON_NO_CALC = "no_parseable_calculation_field"
ROUTE_REASON_NO_FINAL = "no_parseable_terminal_field"
ROUTE_REASON_MISMATCH = "calculation_field_disagrees_with_terminal_field"


def normal(value):
    """Numeric normalisation identical to the frozen scorer."""
    if value is None:
        return None
    try:
        n = float(str(value).replace(",", ""))
        return str(int(n)) if n.is_integer() else str(n)
    except ValueError:
        return str(value).strip()


def parse_response(text: str) -> dict:
    """Parse one raw model response into the frozen E5 fields."""
    finals = FINAL_RE.findall(text)
    terminal_raw = finals[-1] if finals else None
    answer = normal(terminal_raw) if finals else None

    before_final = text[: text.rfind("####")] if "####" in text else text
    calculations = CALC_RE.findall(before_final)
    calc_raw = calculations[-1] if calculations else None
    last_calc = normal(calc_raw) if calculations else None

    if last_calc is None:
        routed, reason = True, ROUTE_REASON_NO_CALC
    elif answer is None:
        routed, reason = True, ROUTE_REASON_NO_FINAL
    elif last_calc != answer:
        routed, reason = True, ROUTE_REASON_MISMATCH
    else:
        routed, reason = False, ROUTE_REASON_NOT_ROUTED

    return {
        "calculation_field_raw": calc_raw,
        "terminal_field_raw": terminal_raw,
        "calculation_field": last_calc,
        "terminal_field": answer,
        "n_calculation_fields": len(calculations),
        "n_terminal_fields": len(finals),
        "route_decision": "routed" if routed else "not_routed",
        "route_reason": reason,
        "routed": routed,
        "parser_version": PARSER_VERSION,
    }


def parser_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps({"parser_version": PARSER_VERSION,
                      "parser_sha256": parser_sha256()}, indent=2))
    for sample in sys.argv[1:]:
        print(json.dumps(parse_response(sample), indent=2))
