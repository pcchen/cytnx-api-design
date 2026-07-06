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


def _row_member(line):
    """The member a Recommendation row is *about*: the first backtick-quoted
    identifier in the row's **API cell**, not merely the first one on the line.

    For a Markdown table row (`| … | … | … |`) the API cell is the first cell.
    A signature-style API cell like `` `UniTensor.from_numpy(array, …)` `` or
    `` `UniTensor()` `` yields no bare identifier (dots/parens defeat the strict
    ``\\`name\\``` token match), so such a row is about no single bare member and
    contributes nothing — this stops an incidental prose token in a *later* cell
    (e.g. a `` `Void` `` state note or an `` `array` `` parameter mention) from
    masquerading as the row's member and demanding a docstring it never needs.
    For a non-table line the whole line is treated as the cell (unchanged
    behaviour for prose verdicts like "we **keep** `foo`").
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        cells = stripped.split("|")
        # cells[0] is '' (before the leading pipe); cells[1] is the API column.
        api_cell = cells[1] if len(cells) > 1 else ""
    else:
        api_cell = line
    m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", api_cell)
    return m.group(1) if m else None


def members_requiring_docstrings(rec_text):
    """Scan Recommendation-section rows for `member` names tagged keep/add/rename.

    Heuristic: a row's member is the first backtick-quoted identifier in its
    **API cell** (see `_row_member`); the row's verdict is the first
    keep/add/rename/remove word (case-insensitive) found anywhere on that line.
    A row whose API cell holds only a signature (no bare identifier) contributes
    no member — incidental prose tokens never trigger a docstring requirement.
    Members tagged only "remove" are exempt.
    """
    verdict_re = re.compile(
        r"\b(" + "|".join(sorted(ALL_VERDICTS)) + r")\b", re.IGNORECASE
    )
    required = set()
    for line in rec_text.splitlines():
        # Verdicts live in the R.1 verdict *table*; a prose line that merely
        # mentions a member in backticks and happens to contain a verdict word
        # (e.g. "the leaked `cfrom` … (5) add the numpy export") must not be
        # read as a verdict row.
        if not line.strip().startswith("|"):
            continue
        member = _row_member(line)
        if member is None:
            continue
        verdict_match = verdict_re.search(line)
        if not verdict_match:
            continue
        verdict = verdict_match.group(1).lower()
        if verdict in DOCUMENTED_VERDICTS:
            required.add(member)
    return required


def has_docstring(member, docstrings_text):
    """True if `member` has a docstring block: a `member` backtick mention or
    a heading (### member / #### `member`, etc.) inside the Docstrings section."""
    if re.search(rf"`{re.escape(member)}`", docstrings_text):
        return True
    if re.search(rf"^#{{1,6}}\s*.*\b{re.escape(member)}\b", docstrings_text, re.MULTILINE):
        return True
    return False


def _looks_like_leak(m, allmembers):
    """Heuristic: does a public (non-underscore) member name look like a raw
    pybind binding / conti.py shim that should be private? (`cConj_`, `c_at`,
    `c__ipow__`, `cnormalize_`, `make_contiguous`, `astype_different_type`, …).
    The N-private completeness net: every such name must be declared in
    private-surface.md.

    Catches (a) the `c`+Capital / `c_` / `c__` raw-binding spellings, (b) the
    `*_different_*` and `make_contiguous` shims, and (c) a `c`-prefixed wrapper
    of an existing public method (`cnormalize_` ← `normalize_`) — the last is why
    a bare `^c[A-Z]` regex under-counts, so it is checked against `allmembers`."""
    if (re.match(r"^c[A-Z]", m) or m.startswith("c_") or m.startswith("c__")
            or "_different_" in m or m == "make_contiguous"):
        return True
    return m.startswith("c") and len(m) > 1 and m[1:] in allmembers


def _hide_members(path):
    """The set of members in `<dir>/private-surface.md`'s "Leaked internals —
    hide" table, or None if that file does not exist (N-private check skipped).

    Parses the table rows under the `## Leaked internals` heading up to the next
    heading, taking each row's API-cell member (see `_row_member`)."""
    if not os.path.isdir(path):
        return None
    ps = os.path.join(path, "private-surface.md")
    if not os.path.exists(ps):
        return None
    text = open(ps, encoding="utf-8").read()
    m = re.search(r"^##\s+Leaked internals.*$", text, re.MULTILINE)
    if not m:
        return set()
    section = text[m.end():]
    section = re.split(r"^#{2,3}\s", section, maxsplit=1, flags=re.MULTILINE)[0]
    hide = set()
    for line in section.splitlines():
        if not line.strip().startswith("|"):
            continue
        member = _row_member(line)
        if member:
            hide.add(member)
    return hide


def _docs(path):
    """Return the sorted list of `.md` files to validate for `path`.

    A directory expands to its (sorted) **numbered** `NN-*.md` children — the
    categorized layout, one file per category. The `[0-9]*.md` glob deliberately
    excludes non-category companions in the same directory (the `inventory.md`
    index and the `element-dtypes.md` appendix), which carry no `# R.` spec and
    are not part of the category partition — matching the partition check in the
    Task-12 brief. A file argument is returned as a singleton list — the
    original flat-doc layout, unchanged.
    """
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "[0-9]*.md")))
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


def _validate_categorized(unit, path, doc_paths, texts):
    """Categorized-layout checks: union coverage/docstrings across category
    files, plus the N-private accounting gate when a private-surface.md exists.
    Returns (problems, members, leaked_public_count | None)."""
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

    # N-private accounting gate (§4.4/§5.6/§10): only when private-surface.md exists.
    leaked = None
    hide = _hide_members(path)
    if hide is not None:
        dir_all = set(dir(UNIT_REGISTRY[unit]))
        for m in sorted(hide - dir_all):  # no phantom hide
            problems.append(f"N-private: private-surface.md hides `{m}` — not in dir({unit})")
        for m in sorted(mm for mm in members if _looks_like_leak(mm, members)):  # completeness
            if m not in hide:
                problems.append(
                    f"N-private: `{m}` looks like a leaked binding but is not in "
                    "private-surface.md's hide table")
        for m in sorted(hide & needs_doc):  # nothing both public and hidden
            problems.append(f"N-private: `{m}` is both hidden and keep/add/rename")
        leaked = len(hide & members)

    return problems, members, leaked


def main():
    unit, path = sys.argv[1], sys.argv[2]
    doc_paths = _docs(path)
    texts = [open(p, encoding="utf-8").read() for p in doc_paths]

    leaked = None
    if os.path.isdir(path):
        problems, members, leaked = _validate_categorized(unit, path, doc_paths, texts)
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
    if leaked is not None:
        print(f"N-private leaked-public count: {unit} = {leaked}  (target 0)")


if __name__ == "__main__":
    main()
