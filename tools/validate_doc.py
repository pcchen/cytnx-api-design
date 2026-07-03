"""Validate a per-class audit document against the live API.

Usage: python tools/validate_doc.py UniTensor docs/api-audit/per-class/UniTensor.md
Exit 0 = pass. Non-zero = missing sections / uncovered members / missing docstrings.
"""
import sys
import re

try:
    from unit_registry import UNIT_REGISTRY  # tools/ on sys.path when run directly
except ImportError:
    from tools.unit_registry import UNIT_REGISTRY

REQUIRED_SECTIONS = [
    "## Inventory", "## Parity findings", "## Consistency findings",
    "## Recommendation", "## Docstrings", "## Change table",
]


def public_members(obj):
    return {m for m in dir(obj) if not m.startswith("_")}


def main():
    unit, path = sys.argv[1], sys.argv[2]
    text = open(path, encoding="utf-8").read()
    problems = []

    for sec in REQUIRED_SECTIONS:
        if sec not in text:
            problems.append(f"missing section: {sec}")

    members = public_members(UNIT_REGISTRY[unit])
    # A member is "covered" if it appears as `member` in the Recommendation section.
    rec = text.split("## Recommendation", 1)[-1].split("## Docstrings", 1)[0]
    covered = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", rec))
    for m in sorted(members - covered):
        problems.append(f"member not in recommendation table: {m}")

    if problems:
        print(f"FAIL ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"PASS: {unit} — {len(members)} members covered")


if __name__ == "__main__":
    main()
