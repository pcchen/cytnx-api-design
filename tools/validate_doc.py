"""Validate a per-class audit document against the live API.

Usage:
  python tools/validate_doc.py <Unit> <path>
`<path>` is either a single flat doc (docs/api-audit/per-class/<Unit>.md,
the original 6-section layout) or a directory of categorized docs
(docs/api-audit/<Unit>/NN-*.md) — in the latter case, coverage and docstring
checks are the union across every `*.md` file in the directory.
Exit 0 = pass. Non-zero = missing sections / uncovered members / missing docstrings.
"""
import sys
import os
import re
import glob

try:
    from unit_registry import UNIT_REGISTRY  # tools/ on sys.path when run directly
except ImportError:
    from tools.unit_registry import UNIT_REGISTRY

# Flat single-file layout (docs/api-audit/per-class/<Unit>.md).
REQUIRED_SECTIONS = [
    "## Inventory", "## Parity findings", "## Consistency findings",
    "## Recommendation", "## Docstrings", "## Change table",
]

# Categorized layout (docs/api-audit/<Unit>/NN-*.md): each category file has
# its own Analysis + normative Recommendation sections instead of the flat
# 6-section structure above — don't force the flat layout onto these.
CATEGORY_REQUIRED_SECTIONS = ["## A1", "# R.", "## R.1"]

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


def _docs(path):
    """Return the sorted list of `.md` files to validate for `path`.

    A directory expands to its (sorted) `*.md` children — the categorized
    layout, one file per category. A file is returned as a singleton list —
    the original flat-doc layout, unchanged.
    """
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.md")))
    return [path]


def _covered_members(texts):
    """Union, across `texts`, of backtick-quoted member names found in each
    text's `# R.` Recommendation section (from the first `# R.` heading on).

    Mirrors the flat-file coverage rule (backtick members in the
    Recommendation section) but per-file, then unioned across category files.
    """
    covered = set()
    for t in texts:
        rec = t.split("# R.", 1)[-1]
        covered |= set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", rec))
    return covered


def _validate_flat(unit, text):
    """Original flat 6-section-layout checks. Returns (problems, members)."""
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

    # Every keep/add/rename member must have a docstring block in ## Docstrings.
    docstrings_sec = text.split("## Docstrings", 1)[-1].split("## Change table", 1)[0]
    needs_doc = members_requiring_docstrings(rec)
    for m in sorted(needs_doc):
        if not has_docstring(m, docstrings_sec):
            problems.append(f"missing docstring for recommended member: {m}")

    return problems, members


def _validate_categorized(unit, doc_paths, texts):
    """Categorized-layout checks: union coverage/docstrings across category
    files. Returns (problems, members)."""
    problems = []
    for p, t in zip(doc_paths, texts):
        for sec in CATEGORY_REQUIRED_SECTIONS:
            if sec not in t:
                problems.append(f"{os.path.basename(p)}: missing section: {sec}")

    members = public_members(UNIT_REGISTRY[unit])
    covered = _covered_members(texts)
    for m in sorted(members - covered):
        problems.append(f"member not covered: {m}")

    # Docstring: each keep/add/rename member matched in R.2a or R.2b of some file.
    joined_rec = "\n".join(t.split("# R.", 1)[-1] for t in texts)
    needs_doc = members_requiring_docstrings(joined_rec)
    doc_sec = "\n".join(t.split("R.2", 1)[-1] for t in texts)
    for m in sorted(needs_doc):
        if not has_docstring(m, doc_sec):
            problems.append(f"missing docstring: {m}")

    return problems, members


def main():
    unit, path = sys.argv[1], sys.argv[2]
    doc_paths = _docs(path)
    texts = [open(p, encoding="utf-8").read() for p in doc_paths]

    if os.path.isdir(path):
        problems, members = _validate_categorized(unit, doc_paths, texts)
        summary = f"PASS: {unit} — {len(members)} members covered across {len(doc_paths)} files"
    else:
        problems, members = _validate_flat(unit, texts[0])
        summary = f"PASS: {unit} — {len(members)} members covered"

    if problems:
        print(f"FAIL ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(summary)


if __name__ == "__main__":
    main()
