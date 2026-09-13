#!/usr/bin/env python3
"""iso_thread_table_v1.py -- ISO 724 coarse-series metric thread pitches, one lookup table shared by the
cosmetic thread op and any future modelled-helix thread verb.

Values (ISO 724 coarse series, the seven most common sizes, copied from the standard, not measured or
derived): M6 1.00, M8 1.25, M10 1.50, M12 1.75, M16 2.00, M20 2.50, M24 3.00 mm.

API:
    PITCH_TABLE: dict[str, float]      -- "M8" -> 1.25 (coarse pitch, mm)
    parse_designation(s: str) -> (diameter_mm, pitch_mm)
        "M8"      -> (8.0, 1.25)   coarse pitch looked up
        "M8x1.25" -> (8.0, 1.25)   explicit pitch, validated against the table
        raises ValueError for an unknown diameter or a pitch that does not match the coarse table.
        Fine-pitch designations are real ISO threads but outside this table and are rejected rather
        than accepted with a guessed pitch.

Run the selftest with `python iso_thread_table_v1.py`.
"""
from __future__ import annotations

import json
import re
import sys

# ISO 724 coarse-pitch series (coarse series), diameter_mm -> pitch_mm. Source: ISO 724:1993 Table 1
# / DIN 13-1, the 7 most common fastener sizes.
PITCH_TABLE: dict[str, float] = {
    "M6": 1.00,
    "M8": 1.25,
    "M10": 1.50,
    "M12": 1.75,
    "M16": 2.00,
    "M20": 2.50,
    "M24": 3.00,
}

_DESIGNATION_RE = re.compile(r"^M(\d+(?:\.\d+)?)(?:[xX](\d+(?:\.\d+)?))?$")


def parse_designation(designation: str) -> tuple[float, float]:
    """"M8" or "M8x1.25" -> (diameter_mm, pitch_mm), validated against PITCH_TABLE (coarse series
    only). Raises ValueError -- never silently guesses a pitch."""
    if not isinstance(designation, str):
        raise ValueError(f"thread designation must be a str, got {type(designation).__name__}")
    m = _DESIGNATION_RE.match(designation.strip())
    if not m:
        raise ValueError(
            f"thread designation {designation!r} does not match the 'M<d>' or 'M<d>x<pitch>' pattern"
        )
    d_str, p_str = m.group(1), m.group(2)
    key = f"M{int(float(d_str)) if float(d_str).is_integer() else d_str}"
    if key not in PITCH_TABLE:
        raise ValueError(
            f"thread designation {designation!r}: diameter {key!r} not in the ISO 724 coarse-series "
            f"table (grovserien) {sorted(PITCH_TABLE)} -- this module's declared scope is the 7 most "
            f"common coarse metric sizes, extensible but not open-ended"
        )
    coarse_pitch = PITCH_TABLE[key]
    diameter_mm = float(d_str)
    if p_str is None:
        return diameter_mm, coarse_pitch
    given_pitch = float(p_str)
    if abs(given_pitch - coarse_pitch) > 1e-9:
        raise ValueError(
            f"thread designation {designation!r}: pitch {given_pitch}mm does not match the ISO 724 "
            f"COARSE pitch for {key} ({coarse_pitch}mm) -- fine-pitch series is outside this table's "
            f"declared scope (grovserien only), not silently accepted with a made-up value"
        )
    return diameter_mm, coarse_pitch


def _selftest() -> dict:
    results = {}
    ok1 = parse_designation("M8") == (8.0, 1.25)
    ok2 = parse_designation("M8x1.25") == (8.0, 1.25)
    results["AT1_bare_and_explicit_agree"] = {"pass": bool(ok1 and ok2)}

    raised = False
    try:
        parse_designation("M8x2.0")
    except ValueError:
        raised = True
    results["AT2_wrong_pitch_rejected"] = {"pass": raised}

    raised2 = False
    try:
        parse_designation("M7")
    except ValueError:
        raised2 = True
    results["AT3_unknown_diameter_rejected"] = {"pass": raised2}

    all7 = all(parse_designation(k) == (float(k[1:]), v) for k, v in PITCH_TABLE.items())
    results["AT4_all_7_coarse_sizes_roundtrip"] = {"pass": all7, "n_sizes": len(PITCH_TABLE)}

    results["all_pass"] = bool(
        results["AT1_bare_and_explicit_agree"]["pass"] and results["AT2_wrong_pitch_rejected"]["pass"]
        and results["AT3_unknown_diameter_rejected"]["pass"] and results["AT4_all_7_coarse_sizes_roundtrip"]["pass"]
    )
    return results


if __name__ == "__main__":
    out = _selftest()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["all_pass"] else 1)
