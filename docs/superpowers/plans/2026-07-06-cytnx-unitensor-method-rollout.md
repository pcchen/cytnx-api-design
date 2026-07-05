# Cytnx UniTensor Method Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the `UniTensor` analysis under the new superset method — categories 03–12 plus `inventory.md` and `element-dtypes.md` — each a probe-backed, bilingually-documented normative spec, gated by an adapted coverage validator.

**Architecture:** Each category is one document `docs/api-audit/UniTensor/NN-<cat>.md` following the fixed shape defined in the method spec: `# Analysis` (A1 inventory table · A2 C++↔Python mapping · A3 findings · A4 argument ordering) + `# R.` normative spec (R.0 conventions · R.1 signatures+contract · R.2a Python numpy docstrings · R.2b C++ Doxygen docstrings). Each is gated by a Python probe against the `cytnx==1.1.0` wheel and — only where a binding-fidelity finding exists — a raw-C++ probe against a source-built `libcytnx`. Coverage is machine-checked by an adapted `validate_doc.py`.

**Tech Stack:** Python 3.12 venv (`cytnx==1.1.0` wheel), `cytnx_src/` 1.1.0 source, g++ 13 + CMake for the raw-C++ probes, stdlib-only tooling, Markdown deliverables.

**Reference:** the method spec `docs/superpowers/specs/2026-07-06-cytnx-api-analysis-method-design.md` and the two worked-example pilots `docs/api-audit/UniTensor/01-construction-init.md` and `02-static-generators.md`. Read all three before starting — they are the exact template every category task reproduces.

## Global Constraints

- **Target version:** cytnx **1.1.0** exactly — signatures from the installed wheel + `cytnx_src/`; behavior from the wheel (Python) and a source-built `libcytnx` (raw C++).
- **Interpreter:** all probes/tools run via `source tools/env.sh && $PY …` (the `.venv/bin/python` with `cytnx==1.1.0`).
- **N-casing (SciPostPhysCodeb.53):** member functions → lowercase snake_case; free functions *acting on* objects → Capitalized; free *creators* → lowercase; types → Capitalized. (Capitalized *members* like `Conj`/`Trace`/`Nblocks`/`combineBonds` are renamed to lowercase; free functions like `linalg.Svd` are **kept** Capitalized.)
- **N-underscore:** trailing `_` = in-place, returns `self`/`&*this`; no suffix = pure (returns a new object). Reject `c`-prefixed / `i`-prefixed in-place spellings.
- **Argument ordering:** positional `[primary operand], [operation parameters]`; the optional metadata block is **keyword-only** in the canonical order `labels, rowrank, is_diag, dtype, device, seed, name`. Prefer numpy/community positional order where one exists (accept-and-document); diverge only for stronger internal consistency and document it.
- **Docstrings:** R.2a numpy-style for the Python API; R.2b Doxygen (`@brief`/`@param`/`@return`) for the C++ API. Python-only members have no R.2b; C++-only members have no R.2a.
- **Binding fidelity:** a finding where a `*_conti.py` wrapper or pybind lambda changes behavior vs the raw C++ method. Its Python side is probe-verified; its C++ side is verified by a raw-C++ probe (source-built `libcytnx`) — **only binding-fidelity findings get a C++ probe**; all other findings are Python-probe only.
- **Compatibility-aware:** every `rename`/`remove` verdict carries a migration note (keep the old name as a `DeprecationWarning` alias for one minor release, then delete).
- **No claim ships unverified:** every behavioral claim in A1/A3 cites a passing `report(...)` assertion; every live public member of `UniTensor` appears in exactly one category's `R.1` verdict table.

---

## The per-category procedure (every category task runs these six gates)

This is the complete, repeatable procedure. Each category task below specifies only its *inputs* (the category name, its member set, the category-specific findings to look for, and whether a C++ probe is expected); the steps themselves are always these six. `NN` = the category number, `<cat>` its slug.

1. **Inventory the category's members.** Run `source tools/env.sh && $PY tools/member_inventory.py UniTensor` and select the members belonging to this category per the task's member list. Cross-check nothing is dropped: the union of all categories' member sets must equal `dir(cytnx.UniTensor)` minus underscores (Task 12 enforces this).

2. **Write the Python probe** `docs/api-audit/probes/UniTensor_<NN>_<cat>.py`. Import `report` (and `returns_view` for copy/view claims) from `tools/probe_helper.py` (via the `sys.path.insert` header used by the existing probes). Add one `report("claim", <bool>)` per behavioral claim the doc will make. End with `print("UniTensor <NN> probe ok")`.

3. **Run the probe green.** `$PY docs/api-audit/probes/UniTensor_<NN>_<cat>.py` → all `[PASS]`, exit 0. **If an assertion fails, the claim is wrong** — correct the probe to the runtime truth, then carry that truth into the doc. Never weaken an assertion to pass.

4. **(binding-fidelity only) Write + run the raw-C++ probe.** If any finding is a binding-fidelity finding, add `docs/api-audit/probes/cpp/UniTensor_<NN>_<cat>.cpp` asserting the raw-C++ side, and build/run it per `docs/api-audit/probes/cpp/README.md` (source-built `libcytnx`). Skip this gate entirely if the category has no binding-fidelity finding.

5. **Write the document** `docs/api-audit/UniTensor/<NN>-<cat>.md` reproducing the pilot's exact shape: header blockquote → `# Analysis` (A1 table `| API | Live signature (1.1.0) | Returns | Description & evidence |`; A2 mapping `| C++ | Python | Status | Note |`; A3 findings `| ID | Finding | Type | What the binding does · evidence | Recommendation |` with IDs `UT-<letter>#`; A4 positional+keyword ordering) → `# R.` (R.0 conventions incl. N-casing/N-underscore/ordering as they apply; R.1 a `class UniTensor: …` signature block + `| API | Verdict | Behavior contract |`; R.2a numpy; R.2b Doxygen). Every A1/A3 behavioral cell quotes a probe assertion; every `rename`/`remove` has a migration note.

6. **Validate + commit.** Run `$PY tools/validate_doc.py UniTensor docs/api-audit/UniTensor/` (the adapted validator, Task 1) → `PASS` with the category's members counted. Then commit the doc + probe(s) together with `git add docs/api-audit/UniTensor/<NN>-<cat>.md docs/api-audit/probes/UniTensor_<NN>_<cat>.py <cpp if any> && git commit`.

The category is **done** when: the Python probe is green, any C++ probe is green, the validator passes, and every category member has an `R.1` verdict row.

---

## File structure

- **Modify:** `tools/validate_doc.py` — adapt to categorized coverage (Task 1).
- **Create:** `docs/api-audit/UniTensor/03-metadata-accessors.md` … `12-type-device-conversion.md` (Tasks 2–11).
- **Create:** `docs/api-audit/UniTensor/inventory.md` (Task 12), `element-dtypes.md` (Task 13).
- **Create:** `docs/api-audit/probes/UniTensor_<NN>_<cat>.py` per category; `docs/api-audit/probes/cpp/UniTensor_<NN>_<cat>.cpp` where a binding-fidelity finding exists.

---

### Task 1: Adapt `validate_doc.py` for categorized coverage

The current validator checks one flat doc. The categorized layout needs coverage summed across a class's category files, and docstrings matched in either R.2a or R.2b.

**Files:**
- Modify: `tools/validate_doc.py`
- Test: `tools/test_validate_doc.py` (create)

**Interfaces:**
- Produces: `validate_doc.py <Unit> <path>` where `<path>` may now be a **directory** (validate the union of its `*.md` category files) or a single file (unchanged). Exit 0 = pass.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_validate_doc.py
import subprocess, sys, os, tempfile, textwrap

PY = sys.executable
HERE = os.path.dirname(__file__)

def run(unit, path):
    return subprocess.run([PY, os.path.join(HERE, "validate_doc.py"), unit, path],
                          capture_output=True, text=True)

def test_directory_unions_category_coverage(tmp_path):
    # A fake unit dir with two category files that TOGETHER cover both members.
    d = tmp_path / "Fake"; d.mkdir()
    (d / "01-a.md").write_text(textwrap.dedent("""
        ## A1
        # R. Recommendation
        ## R.1
        | API | Verdict | Behavior |
        | `alpha` | keep | x |
        ## R.2a
        ### `alpha`
        docstring
    """))
    (d / "02-b.md").write_text(textwrap.dedent("""
        ## A1
        # R. Recommendation
        ## R.1
        | API | Verdict | Behavior |
        | `beta` | keep | y |
        ## R.2a
        ### `beta`
        docstring
    """))
    r = run("FakeUnit", str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "covered" in r.stdout
```

The test uses a `FakeUnit` registry entry whose public members are exactly `{"alpha","beta"}`; add it in Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `source tools/env.sh && $PY -m pytest tools/test_validate_doc.py -v`
Expected: FAIL (validator does not yet accept a directory, and `FakeUnit` is unknown).

- [ ] **Step 3: Implement directory support + a test hook**

In `tools/validate_doc.py`, replace the single-file read with directory-aware collection, and allow a test-only unit whose members come from an env var. Concretely:

```python
import os, glob

def _members(unit):
    if unit == "FakeUnit":
        return set(os.environ.get("FAKE_MEMBERS", "alpha beta").split())
    return public_members(UNIT_REGISTRY[unit])

def _docs(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.md")))
    return [path]

def main():
    unit, path = sys.argv[1], sys.argv[2]
    texts = {p: open(p, encoding="utf-8").read() for p in _docs(path)}
    joined = "\n".join(texts.values())
    members = _members(unit)
    # Coverage: a member is covered if it appears as `member` anywhere in a
    # Recommendation section across the category files.
    covered = set()
    for t in texts.values():
        rec = t.split("# R.", 1)[-1]
        covered |= set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", rec))
    problems = [f"member not covered: {m}" for m in sorted(members - covered)]
    # Docstring: each keep/add/rename member matched in R.2a or R.2b of some file.
    needs = members_requiring_docstrings(joined.split("# R.",1)[-1])
    doc_sec = "\n".join(t.split("R.2", 1)[-1] for t in texts.values())
    for m in sorted(needs):
        if not has_docstring(m, doc_sec):
            problems.append(f"missing docstring: {m}")
    if problems:
        print("FAIL:"); [print(" -", p) for p in problems]; sys.exit(1)
    print(f"PASS: {unit} — {len(members)} members covered across {len(texts)} files")
```

Keep the existing `public_members`, `members_requiring_docstrings`, `has_docstring`, `REQUIRED_SECTIONS` helpers; the section-presence check now applies per-file only to files that contain `# R.` (category files), so relax `REQUIRED_SECTIONS` to the categorized headings actually used (`## A1`, `# R.`, `## R.1`). Verify against the two pilot files.

- [ ] **Step 4: Run the test + a real check to verify they pass**

Run: `source tools/env.sh && $PY -m pytest tools/test_validate_doc.py -v`
Expected: PASS.
Run: `$PY tools/validate_doc.py UniTensor docs/api-audit/UniTensor/`
Expected: `PASS: UniTensor — N members covered across 2 files` (only categories 01–02 exist so far; coverage is partial but the tool runs cleanly — full coverage is enforced once all categories land, Task 12).

- [ ] **Step 5: Commit**

```bash
git add tools/validate_doc.py tools/test_validate_doc.py
git commit -m "tools: adapt validate_doc for categorized UniTensor coverage"
```

---

### Task 2: Category 03 — metadata accessors

**Files:** Create `docs/api-audit/UniTensor/03-metadata-accessors.md`, `docs/api-audit/probes/UniTensor_03_metadata.py`.

**Member set (finding-id prefix `UT-M#`):** `rank`, `rowrank`, `Nblocks`, `shape`, `dtype`, `dtype_str`, `device`, `device_str`, `uten_type`, `uten_type_str`, `name`, `is_diag`, `is_tag`, `is_braket_form`, `is_blockform`, `is_contiguous`, `labels`, `get_index`, `syms`, `bonds`, `bond`, `bond_`, `signflip`, `same_data`, `elem_exists`, `get_qindices`, `getTotalQnums`, `get_blocks_qnums`.

**Category-specific findings to check:** N-casing on `Nblocks`→`nblocks`, `getTotalQnums`→`get_total_qnums`, `get_blocks_qnums` (camel→snake); `bonds()` returns a copy while C++ has a non-const `&` overload (A2 signature-differs, copy/view — probe with `returns_view`); `bond` (copy) vs `bond_` (view) pairing (N-underscore/B1 — probe); the `same_data`/`elem_exists`/`get_qindices` erased-arg names (`arg0`) blocking keyword calls (naming, ties to parameter-consistency PC1); `getTotalQnums`/`get_blocks_qnums` marked "not supported" yet bound. **C++ probe:** likely none (accessors are thin pass-throughs; the `bonds()` copy-vs-`&` is a signature choice, not a behavior divergence — confirm and record "no binding-fidelity finding").

- [ ] **Step 1:** Inventory the category members (procedure gate 1).
- [ ] **Step 2:** Write `probes/UniTensor_03_metadata.py` — assert: `bond_(i)` shares data with the parent while `bond(i)` is independent (`returns_view`); `nblocks`/`rank`/`rowrank`/`shape` values on a small block tensor; `is_*` predicate returns on a tagged vs untagged tensor; `same_data(self)` is True and on a clone is False. (procedure gate 2)
- [ ] **Step 3:** Run it green (gate 3).
- [ ] **Step 4:** No C++ probe expected — confirm no binding-fidelity finding (gate 4, skipped with a one-line note in A3).
- [ ] **Step 5:** Write `03-metadata-accessors.md` (gate 5): A1/A2/A3(`UT-M#`)/A4(trivial — accessors take an index or none), R.0 (N-casing renames), R.1 verdicts, R.2a/R.2b.
- [ ] **Step 6:** Validate + commit (gate 6): `git commit -m "docs: UniTensor cat 03 metadata accessors + probe"`.

---

### Task 3: Category 04 — labels / name / rowrank

**Files:** Create `docs/api-audit/UniTensor/04-labels-name-rowrank.md`, `probes/UniTensor_04_labels.py`.

**Member set (`UT-L#`):** `set_name`, `set_label`, `set_labels`, `relabel`, `relabel_`, `relabels`, `relabels_`, `set_rowrank`, `set_rowrank_`.

**Category-specific findings:** three overlapping label mechanisms (`set_label` mutate vs `relabel_` in-place-returns-self vs `relabel` shared-data copy) — recommend consolidating to `relabel`/`relabel_`, dropping `set_label(s)` and `relabels(_)` as deprecated (redundancy); **binding fidelity:** the public `set_name`/`relabel_`/`set_rowrank_`/`truncate_` etc. are `conti.py` wrappers over raw `c_set_name`/`c_relabel_`/… bindings that also leak publicly (naming + binding fidelity — the `c*` names must be hidden); `relabel` returns a shared-data view (copy/view — probe with `returns_view`); `set_labels` is deprecated but bound and its `c_set_labels` actually calls `relabel_` (binding fidelity). **C++ probe:** expected — verify on the raw C++ side that `relabel` shares data and `relabel_` returns `*this` (the `c_relabel_`→binding chain drops nothing / or does). Add `probes/cpp/UniTensor_04_labels.cpp`.

- [ ] **Step 1–6:** Run the six-gate procedure. Probe (gate 2) asserts: `relabel` returns a distinct object sharing labels-effect; `relabel_` returns self; the deprecated `relabels`/`set_labels` emit the expected warning; the `c_*` raw bindings are present in `dir` (leak). C++ probe (gate 4): raw `relabel_` returns `&*this`. Commit `-m "docs: UniTensor cat 04 labels/name/rowrank + probes"`.

---

### Task 4: Category 05 — structure manipulation

**Files:** Create `05-structure-manipulation.md`, `probes/UniTensor_05_structure.py`, `probes/cpp/UniTensor_05_structure.cpp`.

**Member set (`UT-S#`):** `permute`, `permute_`, `permute_nosignflip`, `permute_nosignflip_`, `reshape`, `reshape_`, `contiguous`, `contiguous_`, `group_basis`, `group_basis_`, `combineBonds`, `to_dense`, `to_dense_`, `truncate`, `truncate_`, `tag`, `twist`, `twist_`, `fermion_twists`, `fermion_twists_`, `apply`, `apply_`.

**Category-specific findings:** `combineBonds` camelCase + it's the **deprecated** form while the current C++ `combineBond` (singular) is **unbound** (naming + C++-only binding gap → bind `combine_bonds`); `permute`/`reshape` return views (copy/view — probe `returns_view`); `contiguous` binds via the raw `make_contiguous` shim (leak + naming); `reshape`/`reshape_` bound as `*args,**kwargs` losing the positional signature (signature-differs); `tag`/`truncate_` over raw `c*` (binding fidelity). **C++ probe:** verify `combineBond` (singular) exists in C++ but is unbound in Python; verify `permute_`/`contiguous_` return `*this`. 

- [ ] **Step 1–6:** Six-gate procedure. Commit `-m "docs: UniTensor cat 05 structure manipulation + probes"`.

---

### Task 5: Category 06 — element & block access

**Files:** Create `06-element-block-access.md`, `probes/UniTensor_06_element.py`, `probes/cpp/UniTensor_06_element.cpp`.

**Member set (`UT-E#`):** `at`, `item`, `get_elem`, `set_elem`, `elem_exists`, `get_block`, `get_block_`, `get_blocks`, `get_blocks_`, `put_block`, `put_block_`, `__getitem__`, `__setitem__`.

**Category-specific findings:** `get_elem` binds only 4 float/complex dtypes while `item`/`set_elem` bind all 11 (signature-differs / binding fidelity — the pybind template coverage differs); `get_block` (copy) vs `get_block_` (view) and `get_blocks`/`get_blocks_` pairing (copy/view — probe); `get`/`set` accessor methods are C++-only (unbound; reached only via `__getitem__`/`__setitem__`); the misspelled `slient` kwarg on `get_blocks_` (correctness/typo, FutureWarning); `put_block(…, force=…)` deprecated. **C++ probe:** verify the C++ `get_elem<T>` template covers all 11 dtypes (so the 4-dtype limit is a binding choice), and that `get`/`set` exist in C++.

- [ ] **Step 1–6:** Six-gate procedure. Commit `-m "docs: UniTensor cat 06 element/block access + probes"`.

---

### Task 6: Category 07 — arithmetic & element-wise

**Files:** Create `07-arithmetic-elementwise.md`, `probes/UniTensor_07_arithmetic.py`, `probes/cpp/UniTensor_07_arithmetic.cpp`.

**Member set (`UT-A#`):** the operator dunders (`__add__`/`__radd__`/`__iadd__`, `__sub__…`, `__mul__…`, `__truediv__…`, `__floordiv__…`, `__neg__`, `__pos__`, `__pow__`, `__ipow__`, `__mod__` if present), `Pow`, `Pow_`, `Inv`, `Conj`, `Conj_`, `Transpose`, `Transpose_`, `Dagger`, `Dagger_`, `normalize`, `normalize_`, `Trace`, `Trace_`, `Norm`, and the leaked `cConj_`/`cDagger_`/`cPow_`/`cTrace_`/`cTranspose_`/`cnormalize_`/`cInv_`/`c__ipow__` (Internal/plumbing).

**Category-specific findings (this is the headline category):** **correctness** — `//` (`__floordiv__`) performs *true* division (`7//2 → 3.5`), routing to `Div` (probe it directly); `%` (`__mod__`) is absent though `linalg.Mod` exists; **N-casing** — `Conj`/`Trace`/`Norm`/`Pow`/`Transpose`/`Dagger` are Capitalized *members* → lowercase; **binding fidelity** — every `Conj`/`Trace`/… is a `conti.py` wrapper over a leaked `c*` raw binding; `Pow_`/`__ipow__` over `cPow_`/`c__ipow__`; **redundancy** — `Inv` present but no public `Inv_` (only `cInv_`); operators vs named methods (`Add`/`Sub`/`Mul`/`Div` are C++-only). **C++ probe:** verify the raw C++ `Conj_`/`Trace_`/`Pow_` return `*this`, and that C++ `operator%`/`Mod` exists on UniTensor (so `%`'s absence is binding-only).

- [ ] **Step 1–6:** Six-gate procedure. The probe (gate 2) must directly assert `float((u*7)//2 elementwise) == 3.5`-style true-division and `hasattr(UniTensor,"__mod__") is False`. Commit `-m "docs: UniTensor cat 07 arithmetic/elementwise + probes"`.

---

### Task 7: Category 08 — linalg operations (free functions on UniTensor)

**Files:** Create `08-linalg-operations.md`, `probes/UniTensor_08_linalg.py`. (No C++ probe expected — these are free functions bound identically C++↔Python; confirm.)

**Member set (`UT-X#`, free functions in `cytnx.linalg` taking/returning UniTensor):** `Svd`, `Svd_truncate`, `Gesvd`, `Gesvd_truncate`, `Rsvd`, `Rsvd_truncate`, `Hosvd`, `Qr`, `Qdr`, `Eig`, `Eigh`, `ExpH`, `ExpM`, `InvM`, `InvM_`, `Inv`, `Inv_`, `Trace`, `Pow`, `Pow_`, `Conj`, `Conj_`, `Norm`, `Add`, `Sub`, `Mul`, `Div`, `Mod`. (Tensor-only linalg functions are listed in an appendix as parity gaps `UT-X#`.)

**Category-specific findings:** **N-casing keeps these Capitalized** (they are free functions acting on objects — the key demonstration that the convention is not "snake_case everything", explicitly reversing v1's linalg C1); SVD toggle inconsistency (`is_UvT` on `Svd`/`Svd_truncate` vs `is_U`/`is_vT` on `Gesvd`/`Rsvd`) (N4/ordering — probe that `Svd` rejects `is_U`); decompositions return positional `list[UniTensor]` with flag-dependent order (`Svd → [S,U,vT,(err)]`) — recommend named-tuple results (ordering/documentation); `Inv`/`Inv_` element-wise vs `InvM`/`InvM_` matrix inverse near-name collision.

- [ ] **Step 1–6:** Six-gate procedure (C++ probe skipped — record "no binding-fidelity finding: identical C++/Python free-function bindings"). Probe asserts the decomposition return arity/order on a UniTensor and the `is_UvT`-vs-`is_U` toggle split. Commit `-m "docs: UniTensor cat 08 linalg operations + probe"`.

---

### Task 8: Category 09 — linalg solvers (Krylov)

**Files:** Create `09-linalg-solvers.md`, `probes/UniTensor_09_solvers.py`.

**Member set (`UT-K#`):** `Lanczos`, `Lanczos_Gnd_Ut`, `Lanczos_Exp`, `Arnoldi`, `Lanczos_ER`, `Lanczos_Gnd`.

**Category-specific findings:** `Lanczos` dispatches ARPACK + ER/Gnd via a string `method`/`which` arg, while `Lanczos_ER`/`Lanczos_Gnd` are Tensor-oriented `conti.py` wrappers and `Lanczos_Gnd_Ut` is the UniTensor path — the naming doesn't make the Tensor-vs-UniTensor or method split obvious (naming/N4); these are free functions → **kept Capitalized** (N-casing). Note the earlier audit found `Lanczos_ER`/`Lanczos_Gnd`/`Lanczos_Gnd_Ut` pybind + conti registrations were commented out in 1.1.0 — verify current binding state at runtime and record (parity/capability).

- [ ] **Step 1–6:** Six-gate procedure. Probe asserts which `Lanczos*` names are reachable and the ground-state result on a small Hermitian `LinOp`/UniTensor. Commit `-m "docs: UniTensor cat 09 linalg solvers + probe"`.

---

### Task 9: Category 10 — contraction & networks

**Files:** Create `10-contraction-networks.md`, `probes/UniTensor_10_contraction.py`.

**Member set (`UT-N#`):** `contract` (member), and the related free functions `Contract`, `Contracts`, `ncon` (documented here as they act on UniTensor); note `Network` is a separate class (cross-reference, not re-audited).

**Category-specific findings:** `contract` (member, lowercase — correct) vs free `Contract`/`Contracts` (Capitalized — correct per N-casing, acting on objects); `Contracts` deprecated → `Contract`; `ncon` is a Python-only free function (community name — kept lowercase as the agreed exception); verify `ncon`'s index convention against a hand contraction (correctness — probe). Cross-reference the `network.md` v1 finding that `Network.Contract().Launch()` segfaults.

- [ ] **Step 1–6:** Six-gate procedure. Probe asserts `ncon([A,B],[[-1,1],[1,-2]])` equals the hand contraction `A@B` and that member `contract` produces the same result. Commit `-m "docs: UniTensor cat 10 contraction/networks + probe"`.

---

### Task 10: Category 11 — I/O & display

**Files:** Create `11-io-display.md`, `probes/UniTensor_11_io.py`.

**Member set (`UT-IO#`):** `Save`, `Load`, `print_diagram`, `print_block`, `print_blocks`, `__repr__`, and the pickle hooks (`__getstate__` present, `__setstate__` absent).

**Category-specific findings:** **N-casing** — `Save`/`Load` are Capitalized *members* → `save`/`load` (matches the cat-02 `Load`→`load` decision); **correctness** — pickle is a broken stub (`__getstate__` with no `__setstate__`); recommend implementing over `save`/`load` or removing the stale `__getstate__`; `print_diagram`/`print_block(s)` write to a redirected ostream (verify capturable). `__repr__` wraps `operator<<`.

- [ ] **Step 1–6:** Six-gate procedure. Probe asserts `hasattr(UniTensor,"__setstate__") is False` while `__getstate__` exists (broken pickle), and a `save`→`load` round-trip is value-equal. Commit `-m "docs: UniTensor cat 11 io/display + probe"`.

---

### Task 11: Category 12 — type & device conversion

**Files:** Create `12-type-device-conversion.md`, `probes/UniTensor_12_typedevice.py`, `probes/cpp/UniTensor_12_typedevice.cpp`.

**Member set (`UT-T#`):** `astype`, `to`, `to_`, `clone`, `__copy__`, `__deepcopy__`, `convert_from`, and the leaked `astype_different_type`/`to_different_device` shims + the numpy-bridge gap.

**Category-specific findings:** `astype`/`to` bind via `conti.py` over the raw `astype_different_type`/`to_different_device` shims (leak + binding fidelity — recall the sibling/v1 finding that these short-circuit to `is self` on a no-op); `to_(arg0)` has an erased arg name (naming/PC1 — should be `to_(device=…)`); `clone`/`__copy__`/`__deepcopy__` deep-copy; **capability gap** — no `.numpy()`/`from_numpy` (pairs with cat-01 `UT-C3`). **C++ probe:** verify raw C++ `astype`/`to` on a no-op return a fresh object (so the Python `is self` short-circuit is binding-introduced).

- [ ] **Step 1–6:** Six-gate procedure. Probe asserts `astype(same_dtype) is self` and `to(same_device) is self` (the binding short-circuit) vs a real conversion returning a distinct object; `clone()` is independent (`returns_view` → False). C++ probe: raw `astype`/`to` no-op returns a distinct object. Commit `-m "docs: UniTensor cat 12 type/device conversion + probes"`.

---

### Task 12: `inventory.md` + full-coverage gate

**Files:** Create `docs/api-audit/UniTensor/inventory.md`.

**Interfaces:**
- Consumes: all category docs 01–12.
- Produces: the master categorized member list; the enforced guarantee that categories 01–12 partition `dir(cytnx.UniTensor)`.

- [ ] **Step 1: Write the inventory** — one section per category (01–12), listing its members with a one-line role, plus an `## Internal / plumbing (not public API)` section collecting every leaked `c*`/`*_different_*` binding surfaced across categories.

- [ ] **Step 2: Verify the partition is complete**

Run:
```bash
source tools/env.sh
$PY - <<'EOF'
import cytnx, re, glob
live = {m for m in dir(cytnx.UniTensor) if not m.startswith("_")}
covered = set()
for f in glob.glob("docs/api-audit/UniTensor/[0-9]*.md"):
    rec = open(f).read().split("# R.",1)[-1]
    covered |= set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", rec))
missing = sorted(live - covered)
print("live public members:", len(live), "| uncovered:", missing)
assert not missing, missing
print("OK — categories partition the public surface")
EOF
```
Expected: `OK` (every public member appears in some category's Recommendation). If any are missing, add them to the owning category doc (and its probe if behavioral) before proceeding.

- [ ] **Step 3: Run the full coverage validator**

Run: `$PY tools/validate_doc.py UniTensor docs/api-audit/UniTensor/`
Expected: `PASS: UniTensor — 126 members covered across 12 files` (count per the live wheel).

- [ ] **Step 4: Commit**

```bash
git add docs/api-audit/UniTensor/inventory.md
git commit -m "docs: UniTensor categorized inventory + full-coverage gate"
```

---

### Task 13: `element-dtypes.md`

**Files:** Create `docs/api-audit/UniTensor/element-dtypes.md`, `probes/UniTensor_dtypes.py`.

**Interfaces:**
- Consumes: `enums.md` (the `Type` set) for cross-reference.
- Produces: the element-dtype dimension — the 11 current dtypes, the promotion rule, per-operation dtype constraints, and the recommendation to keep the 4 float/complex types for the recommended UniTensor element set.

- [ ] **Step 1: Write the probe** `probes/UniTensor_dtypes.py` — assert the 11 constructible element dtypes on a `UniTensor`, the promotion result of mixed-dtype arithmetic (e.g. `Double` + `ComplexDouble` → `ComplexDouble`), and the per-op constraints found in cat 06/07/08 (e.g. `get_elem` only on the 4 float/complex types; a decomposition requires a float/complex dtype). End `print("UniTensor dtypes probe ok")`.
- [ ] **Step 2: Run it green.**
- [ ] **Step 3: Write `element-dtypes.md`** — the dtype table (name/code/flags, cross-referencing `enums.md`), the promotion rule, the per-operation constraint table, and the recommendation (keep `ComplexDouble`/`ComplexFloat`/`Double`/`Float` as the element set; integer/bool element tensors are storage-only, not operable by linalg).
- [ ] **Step 4: Commit** `git commit -m "docs: UniTensor element-dtypes analysis + probe"`.

---

## Follow-up (out of scope for this plan)

The other classes (`Tensor`, `Bond`, `Symmetry`, `Network`, `LinOp`, `Storage`, `Scalar`, enums, operations) are each a **separate plan** that reruns the identical per-category procedure with that class's categories and member sets, and a cross-class `cpp-python-mapping.md` policy + master recommendation index synthesized once all classes are redone. When a class is completed under this method, its flat `per-class/<Class>.md` is retired (decision 2).

## Self-review notes

- **Spec coverage:** every §4 section (A1–A4, R.0–R.2a/b) and §5 convention is exercised by the per-category procedure (gate 5); §7 verification (Python + C++ probes) by gates 2–4; §6 binding-fidelity by the gate-4 condition; §8 compatibility by the migration-note requirement in gate 5; §10 definition-of-done by gate 6 + Task 12. §9 rollout: Tasks 2–13 cover UniTensor; other classes explicitly deferred.
- **Type consistency:** the validator interface `validate_doc.py <Unit> <dir|file>` (Task 1) is the exact command used in every category's gate 6 and in Task 12; finding-id prefixes are unique per category (`UT-M/L/S/E/A/X/K/N/IO/T#`).
- **No placeholders:** member sets and category-specific findings are concrete; the repeatable steps live in the procedure block (referenced, not vaguely gestured at).
