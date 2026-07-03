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

# Verdict tags in the Recommendation section that obligate a docstring entry.
# "remove" is intentionally excluded: a removed member needs no docstring.
DOCUMENTED_VERDICTS = {"keep", "add", "rename"}
ALL_VERDICTS = DOCUMENTED_VERDICTS | {"remove"}


def public_members(obj):
    return {m for m in dir(obj) if not m.startswith("_")}


def members_requiring_docstrings(rec_text):
    """Scan Recommendation-section rows for `member` names tagged keep/add/rename.

    Heuristic: a row is any line containing a backtick-quoted identifier; the
    first such identifier on the line is taken as the member name, and the
    row's verdict is the first keep/add/rename/remove word (case-insensitive)
    found on that same line. Members tagged only "remove" are exempt.
    """
    verdict_re = re.compile(
        r"\b(" + "|".join(sorted(ALL_VERDICTS)) + r")\b", re.IGNORECASE
    )
    required = set()
    for line in rec_text.splitlines():
        names = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", line)
        if not names:
            continue
        verdict_match = verdict_re.search(line)
        if not verdict_match:
            continue
        verdict = verdict_match.group(1).lower()
        if verdict in DOCUMENTED_VERDICTS:
            required.add(names[0])
    return required


def has_docstring(member, docstrings_text):
    """True if `member` has a docstring block: a `member` backtick mention or
    a heading (### member / #### `member`, etc.) inside the Docstrings section."""
    if re.search(rf"`{re.escape(member)}`", docstrings_text):
        return True
    if re.search(rf"^#{{1,6}}\s*.*\b{re.escape(member)}\b", docstrings_text, re.MULTILINE):
        return True
    return False


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

    # (c) every keep/add/rename member must have a docstring block in ## Docstrings.
    docstrings_sec = text.split("## Docstrings", 1)[-1].split("## Change table", 1)[0]
    needs_doc = members_requiring_docstrings(rec)
    for m in sorted(needs_doc):
        if not has_docstring(m, docstrings_sec):
            problems.append(f"missing docstring for recommended member: {m}")

    if problems:
        print(f"FAIL ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"PASS: {unit} — {len(members)} members covered")


if __name__ == "__main__":
    main()
