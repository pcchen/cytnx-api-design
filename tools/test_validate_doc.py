"""Tests for tools/validate_doc.py's categorized-coverage support (Task 1).

Two layers:
  - Unit tests import validate_doc as a module and exercise the small
    importable helpers (_docs, _covered_members) directly against tmp_path
    fixtures the test itself controls — no test-only unit/env-var hook in
    production code.
  - CLI smoke tests shell out to validate_doc.py against real repo docs: the
    real UniTensor category directory (partial coverage — only 01-02 exist)
    and the real flat Bond.md (must still fully PASS, unchanged behavior).
"""
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import validate_doc  # noqa: E402  (import after sys.path tweak)

PY = sys.executable
HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(HERE)


def run(unit, path):
    return subprocess.run(
        [PY, os.path.join(HERE, "validate_doc.py"), unit, path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


CATEGORY_A = textwrap.dedent("""
    ## A1
    # R. Recommendation
    ## R.1
    | API | Verdict | Behavior |
    | `alpha` | keep | x |
    ## R.2a
    ### `alpha`
    docstring
""")

CATEGORY_B = textwrap.dedent("""
    ## A1
    # R. Recommendation
    ## R.1
    | API | Verdict | Behavior |
    | `beta` | keep | y |
    ## R.2a
    ### `beta`
    docstring
""")


# --- _docs -------------------------------------------------------------

def test_docs_returns_file_as_singleton_list(tmp_path):
    f = tmp_path / "only.md"
    f.write_text(CATEGORY_A)
    assert validate_doc._docs(str(f)) == [str(f)]


def test_docs_returns_sorted_md_files_in_directory(tmp_path):
    d = tmp_path / "Fake"
    d.mkdir()
    (d / "02-b.md").write_text(CATEGORY_B)
    (d / "01-a.md").write_text(CATEGORY_A)
    (d / "notes.txt").write_text("ignore me — not a doc")
    got = validate_doc._docs(str(d))
    assert got == [str(d / "01-a.md"), str(d / "02-b.md")]


# --- _covered_members ----------------------------------------------------

def test_covered_members_unions_across_texts():
    covered = validate_doc._covered_members([CATEGORY_A, CATEGORY_B])
    assert covered == {"alpha", "beta"}


def test_covered_members_single_text_only_finds_its_own_members():
    covered = validate_doc._covered_members([CATEGORY_A])
    assert covered == {"alpha"}


# --- members_requiring_docstrings (signature-cell false positives) --------

def test_signature_api_cell_does_not_leak_prose_token_as_member():
    """A verdict row whose API cell is a signature (no bare identifier) must
    NOT treat an incidental backtick token in a later cell as its member.

    Regression: `| `UniTensor()` | keep | … (`Void`) … |` used to demand a
    docstring for `Void`, and `| `UniTensor.from_numpy(array, …)` | add |
    copies `array` |` for `array` — neither is a real member.
    """
    rec = textwrap.dedent("""
        | `UniTensor()` | **keep** | Empty, un-initialized (`Void`) rank-0 tensor. |
        | `UniTensor.from_numpy(array, *, …)` | **add** | Dense; **copies** `array`. |
    """)
    needs = validate_doc.members_requiring_docstrings(rec)
    assert "Void" not in needs
    assert "array" not in needs


def test_normal_verdict_row_still_yields_its_api_member():
    """A well-formed row (`| `foo` | keep | … |`) still requires foo's docstring;
    a bare-identifier add row still requires the new member."""
    rec = textwrap.dedent("""
        | `from_numpy` (→ add, static) | `array` | copies `array` |
        | `rank` | keep | number of legs |
        | `Nblocks` → `nblocks` | rename | block count |
        | `old_thing` | remove | dropped |
    """)
    needs = validate_doc.members_requiring_docstrings(rec)
    assert needs == {"from_numpy", "rank", "Nblocks"}


# --- CLI end-to-end (directory union) -------------------------------------

def test_cli_directory_unions_category_coverage(tmp_path):
    d = tmp_path / "Fake"
    d.mkdir()
    (d / "01-a.md").write_text(CATEGORY_A)
    (d / "02-b.md").write_text(CATEGORY_B)
    r = run("Bond", str(d))
    # Bond has far more than {alpha, beta} public members, so this must FAIL
    # on coverage — but it must run cleanly as a directory (no crash) and
    # report per-member coverage output.
    assert "covered" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_cli_flat_file_still_passes_for_bond():
    r = run("Bond", os.path.join(REPO_ROOT, "docs", "api-audit", "per-class", "Bond.md"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_cli_directory_smoke_check_on_real_unitensor_docs():
    # Only categories 01-02 exist so far (of 12 planned) — coverage is
    # necessarily partial. This checks the tool runs cleanly against a real
    # directory and reports a coverage line; it does not assert full PASS.
    r = run("UniTensor", os.path.join(REPO_ROOT, "docs", "api-audit", "UniTensor"))
    assert "covered" in (r.stdout + r.stderr), r.stdout + r.stderr


# --- N-private accounting helpers (§4.4/§5.6/§10) --------------------------

def test_looks_like_leak_patterns_and_c_prefix_of_existing_member():
    m = {"normalize_", "tag", "truncate_", "contiguous", "clone"}
    # c+Capital / c_ / c__ / *_different_* / make_contiguous
    assert validate_doc._looks_like_leak("cConj_", m)
    assert validate_doc._looks_like_leak("c_at", m)
    assert validate_doc._looks_like_leak("c__ipow__", m)
    assert validate_doc._looks_like_leak("astype_different_type", m)
    assert validate_doc._looks_like_leak("make_contiguous", m)
    # c+lowercase wrapper of an existing method (the ^c[A-Z] blind spot)
    assert validate_doc._looks_like_leak("cnormalize_", m)   # normalize_ present
    assert validate_doc._looks_like_leak("ctag", m)          # tag present
    assert validate_doc._looks_like_leak("ctruncate_", m)    # truncate_ present
    # legitimate members must NOT be flagged
    assert not validate_doc._looks_like_leak("contiguous", m)   # tail 'ontiguous' not a member
    assert not validate_doc._looks_like_leak("contiguous_", {"contiguous"})
    assert not validate_doc._looks_like_leak("clone", m)


def test_hide_members_parses_table_and_returns_none_when_absent(tmp_path):
    d = tmp_path / "Fake"; d.mkdir()
    # No private-surface.md → check is skipped (None), not empty set.
    assert validate_doc._hide_members(str(d)) is None
    (d / "private-surface.md").write_text(textwrap.dedent("""
        # Fake — private / plumbing surface
        ## Leaked internals — hide
        | Member | What it is | Used by | Fix |
        |---|---|---|---|
        | `cFoo_` | raw | `Foo_` | inline |
        | `c_bar` | raw | `bar` | inline |
        ## User-facing dunders — keep
        `__repr__`  (must NOT be parsed as a hide member)
    """))
    assert validate_doc._hide_members(str(d)) == {"cFoo_", "c_bar"}
