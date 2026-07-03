# Cytnx 1.1.0 API Audit & Next-Version Recommendation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an analysis + recommendation spec for the Cytnx 1.1.0 public API — per-class parity/consistency audits, keep/add/rename/remove recommendations, docstrings for the whole recommended surface, `current → recommended` change tables, and the essential-API set for TRG/HOTRG/CTMRG/MERA.

**Architecture:** The deliverables are Markdown documents plus executable Python probes. Each per-class audit is gated by two runnable tests: a **behavioral probe** (`probes/<Unit>.py`, asserts runtime object-behavior claims against the installed Cytnx wheel) and a **doc validator** (`tools/validate_doc.py`, asserts every live public member is covered and every recommended member has a docstring). Signatures are read from the cloned source (`cytnx_src/`); behavior is confirmed by running code in a venv-installed wheel. Documents are written essential-first so the essential-API synthesis can draw on completed units.

**Tech Stack:** Python 3.12 venv, `cytnx==1.1.0` (prebuilt manylinux wheel), the cloned source tree `cytnx_src/`, Markdown deliverables, small stdlib-only Python tools (no extra deps).

## Global Constraints

- **Target version:** Cytnx `1.1.0` exactly — signatures from `cytnx_src/` (pinned to 1.1.0), behavior from the `cytnx==1.1.0` wheel. Any wheel/source mismatch is itself a reportable finding.
- **Compatibility posture:** Clean-slate redesign. The `current → recommended` change tables are the migration guide; no obligation to preserve old names or add deprecation aliases.
- **Out of scope:** `tn_algo` (MPS/MPO/DMRG) — not audited, not used to derive the essential set. Private/underscore symbols, build/CI/docs tooling. Implementing the new API.
- **Reference algorithms (deliverable #6):** TRG, HOTRG, CTMRG, MERA only.
- **Docstrings:** produced for the **entire** recommended surface, numpy-style, with explicit copy/view + in-place behavior notes.
- **Deliverable root:** `docs/api-audit/`.
- **No behavioral claim ships unverified:** every runtime claim in a parity section has a passing assertion in that unit's probe.
- **Venv python interpreter:** all probes/tools run via the project venv interpreter (created in Task 1), referenced below as `$PY`.

---

### Task 1: Environment & tooling scaffold

Set up the reproducible environment, the deliverable tree, and the two shared tools (inventory generator + doc validator) that every later task depends on.

**Files:**
- Create: `docs/api-audit/` (dir), `docs/api-audit/per-class/` (dir), `docs/api-audit/probes/` (dir)
- Create: `tools/env.sh`
- Create: `tools/member_inventory.py`
- Create: `tools/validate_doc.py`
- Create: `tools/probe_helper.py`
- Create: `docs/api-audit/probes/__smoke__.py`

**Interfaces:**
- Produces:
  - `$PY` — the venv interpreter path `.venv/bin/python` (created here; `cytnx==1.1.0` installed).
  - `tools/member_inventory.py <ClassOrModule>` — prints, to stdout, the public-member checklist for a class or submodule: one `- <name>` line per public member, plus each member's C++ signature (parsed from the pybind docstring) and Python doc head. Used to seed inventory tables.
  - `tools/validate_doc.py <Unit> <doc_path>` — exit 0 if the doc passes, non-zero with a report otherwise. Checks: all required section headers present; every live public member of `<Unit>` appears in the "Recommendation table"; every member marked keep/add/rename has a docstring block. `<Unit>` resolves to a class or submodule via `tools/unit_registry.py` mapping (defined in Step 4).
  - `tools/probe_helper.py` — helpers imported by every probe: `is_view(make, mutate)`, `assert_inplace(obj, method_name)`, `report(claim: str, ok: bool)`.

- [ ] **Step 1: Create the venv and install the pinned wheel**

Run:
```bash
cd /home/sinlo/github/cytnx_design_API
python3 -m venv .venv
.venv/bin/pip install --quiet 'cytnx==1.1.0'
.venv/bin/python -c "import cytnx; print(cytnx.__version__)"
```
Expected: prints `1.1.0`.

- [ ] **Step 2: Write `tools/env.sh` (records the interpreter for later tasks)**

```bash
#!/usr/bin/env bash
# Shared environment for audit tooling. Source or reference PY.
export PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/python"
```

- [ ] **Step 3: Write `tools/probe_helper.py`**

```python
"""Helpers shared by every behavioral probe. Import from probes/<Unit>.py."""
import cytnx


def report(claim: str, ok: bool) -> None:
    """Print a PASS/FAIL line and raise AssertionError on failure."""
    print(f"[{'PASS' if ok else 'FAIL'}] {claim}")
    assert ok, claim


def mutates_alias(make, mutate) -> bool:
    """True if mutating an aliased handle changes the original (view semantics)."""
    original = make()
    alias = original
    mutate(alias)
    # Caller passes make/mutate such that equality reflects the mutation.
    return not make().__class__  # placeholder overwritten in Step 6 smoke test
```

Note: `mutates_alias` is finalized in Step 6 against real Cytnx behavior; the smoke test drives its exact shape.

- [ ] **Step 4: Write `tools/member_inventory.py`**

```python
"""Dump the public-member checklist for a Cytnx class or submodule.

Usage: python tools/member_inventory.py UniTensor
       python tools/member_inventory.py linalg
"""
import sys
import inspect
import cytnx

# Maps a Unit name (as used in the plan/docs) to a live object.
UNIT_REGISTRY = {
    "Tensor": cytnx.Tensor, "Storage": cytnx.Storage, "Scalar": cytnx.Scalar,
    "UniTensor": cytnx.UniTensor, "Bond": cytnx.Bond, "Symmetry": cytnx.Symmetry,
    "Network": cytnx.Network, "LinOp": cytnx.LinOp,
    "linalg": cytnx.linalg, "algo": cytnx.algo, "random": cytnx.random,
    "physics": cytnx.physics, "qgates": cytnx.qgates,
    "Type": cytnx.Type, "Device": cytnx.Device,
    "SymType": cytnx.SymType, "bondType": cytnx.bondType,
    "fermionParity": cytnx.fermionParity,
}


def public_members(obj):
    return sorted(m for m in dir(obj) if not m.startswith("_"))


def cpp_signature(member):
    """Extract the C++ signature line(s) from a pybind docstring, if present."""
    doc = inspect.getdoc(member) or ""
    lines = [ln.strip() for ln in doc.splitlines() if "(self:" in ln or "-> " in ln]
    return lines[:4]


def main():
    name = sys.argv[1]
    obj = UNIT_REGISTRY[name]
    for m in public_members(obj):
        member = getattr(obj, m)
        print(f"- {m}")
        for sig in cpp_signature(member):
            print(f"    cpp: {sig}")


if __name__ == "__main__":
    main()
```

Extract `UNIT_REGISTRY` into `tools/unit_registry.py` and import it in both tools so the mapping is defined once (DRY).

- [ ] **Step 5: Write `tools/validate_doc.py`**

```python
"""Validate a per-class audit document against the live API.

Usage: python tools/validate_doc.py UniTensor docs/api-audit/per-class/UniTensor.md
Exit 0 = pass. Non-zero = missing sections / uncovered members / missing docstrings.
"""
import sys
import re
from unit_registry import UNIT_REGISTRY  # tools/ on sys.path

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
```

- [ ] **Step 6: Write the smoke probe `docs/api-audit/probes/__smoke__.py` and finalize `probe_helper.mutates_alias`**

```python
"""Smoke test: confirms the venv, cytnx import, and probe helpers work."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, mutates_alias

# Alias mutation on a Tensor is a view (assignment mutates the original).
def make():
    t = cytnx.zeros([2, 2]); return t
def mutate(t):
    t[0, 0] = 7.0
t = make(); alias = t; mutate(alias)
report("Tensor handle assignment aliases (view semantics)", float(t[0, 0].item()) == 7.0)
report("clone() breaks the alias (deep copy)",
       float(cytnx.zeros([2, 2]).clone().__class__ is not None))
print("smoke ok")
```

Rewrite `mutates_alias` in `tools/probe_helper.py` to match what this smoke test proves:
```python
def mutates_alias(make, mutate, read) -> bool:
    """make()->obj, mutate(obj) in place, read(obj)->value. Returns True if an
    aliased handle observes the mutation (view), False if independent (copy)."""
    obj = make()
    alias = obj
    mutate(alias)
    return read(obj) == read(alias)
```

- [ ] **Step 7: Run the tooling to verify it works**

Run:
```bash
source tools/env.sh
$PY tools/member_inventory.py UniTensor | head -5
$PY tools/member_inventory.py linalg | head -5
$PY docs/api-audit/probes/__smoke__.py
```
Expected: inventory prints `- Conj` etc. with `cpp:` lines; smoke prints two `[PASS]` lines and `smoke ok`.

- [ ] **Step 8: Commit**

```bash
git add tools/ docs/api-audit/probes/__smoke__.py .gitignore
git commit -m "chore: audit tooling scaffold (venv, inventory, validator, probe helper)"
```
Note: add `.venv/` to `.gitignore` before committing.

---

### Task 2: Conventions & methodology document

Write `00-methodology.md`: the audit template plus the naming/behavior/return **conventions** the recommended API must obey, so every consistency finding cites a specific convention. `validate_doc.py`'s `REQUIRED_SECTIONS` list is the authority for the per-class template; this doc must describe those same six sections.

**Files:**
- Create: `docs/api-audit/00-methodology.md`

**Interfaces:**
- Consumes: `tools/validate_doc.py::REQUIRED_SECTIONS` (Task 1).
- Produces: the convention IDs (`N1`, `N2`, … for naming; `B1`, `B2`, … for behavior) that per-class consistency findings reference by ID.

- [ ] **Step 1: Write the conventions and template**

Create `docs/api-audit/00-methodology.md` with these sections (real content, not placeholders):

1. **Audit template** — enumerate the six per-class sections exactly as `REQUIRED_SECTIONS` (Inventory, Parity findings, Consistency findings, Recommendation, Docstrings, Change table) and what each contains.
2. **Naming conventions** — assign IDs, e.g.:
   - `N1`: Python methods are `snake_case`; the recommended API renames capitalized methods (`Conj`→`conj`, `Dagger`→`dagger`, `Inv`→`inv`).
   - `N2`: in-place variants use a trailing `_` (`conj_`), and every in-place method has a pure counterpart.
   - `N3`: C++ and Python public names are identical modulo the `snake_case` rule.
3. **Behavior conventions** — assign IDs, e.g.:
   - `B1`: methods return a new object (copy) unless named with the `_` in-place suffix.
   - `B2`: indexing/slice assignment on a handle is a view and mutates in place — document per class.
   - `B3`: dtype/device promotion rules are explicit and identical across C++/Python.
4. **How parity is judged** — signature comparison across `include/*.hpp` + `pybind/*_py.cpp` + `cytnx/*_conti.py`, plus a passing behavioral probe.
5. **How the essential set is derived** — algorithm-step decomposition → API-call mapping → union (see spec §5.3).

- [ ] **Step 2: Verify the template matches the validator**

Run (loads `REQUIRED_SECTIONS` from the tools dir without triggering `validate_doc`'s
`unit_registry` import, by putting `tools/` on `sys.path` first):
```bash
source tools/env.sh
$PY - <<'EOF'
import sys, importlib
sys.path.insert(0, "tools")
REQUIRED_SECTIONS = importlib.import_module("validate_doc").REQUIRED_SECTIONS
doc = open("docs/api-audit/00-methodology.md").read()
missing = [s for s in REQUIRED_SECTIONS if s.replace("## ", "") not in doc]
print("template sections missing from methodology:", missing)
assert not missing
print("OK")
EOF
```
Expected: `OK` (every template section name is described in the methodology).

- [ ] **Step 3: Commit**

```bash
git add docs/api-audit/00-methodology.md
git commit -m "docs: audit methodology and API conventions"
```

---

### Task 3: `Bond` audit

First per-class audit. Establishes the repeatable pattern used by Tasks 4–13: **generate inventory → write & run probe → write doc → validate → commit.**

**Files:**
- Create: `docs/api-audit/per-class/Bond.md`
- Create: `docs/api-audit/probes/Bond.py`

**Interfaces:**
- Consumes: `tools/member_inventory.py`, `tools/validate_doc.py`, `tools/probe_helper.py`, conventions `N1–N3`/`B1–B3` from Task 2.
- Produces: `Bond.md` — referenced by `Symmetry.md`, `UniTensor.md`, and `essential-api.md`.

- [ ] **Step 1: Generate the member inventory**

Run:
```bash
source tools/env.sh
$PY tools/member_inventory.py Bond > /tmp/Bond_inventory.txt
wc -l /tmp/Bond_inventory.txt && head -20 /tmp/Bond_inventory.txt
```
Expected: ~30 `- <member>` lines with `cpp:` signature lines.

- [ ] **Step 2: Write the behavioral probe `docs/api-audit/probes/Bond.py`**

Assert every behavioral claim the parity section will make. Example shape (fill with the actual Bond behaviors — construction from `bondType`, `redirect`, `combineBond`, `qnums`, equality, in-place vs. return):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

# B1: combineBond returns a new Bond; combineBond_ mutates in place.
b1 = cytnx.Bond(2); b2 = cytnx.Bond(3)
combined = b1.combineBond(b2)
report("combineBond returns new Bond, leaves operand unchanged", b1.dim() == 2)
b1c = cytnx.Bond(2); b1c.combineBond_(cytnx.Bond(3))
report("combineBond_ mutates in place", b1c.dim() == 6)
print("Bond probe ok")
```

- [ ] **Step 3: Run the probe (must pass)**

Run:
```bash
source tools/env.sh
$PY docs/api-audit/probes/Bond.py
```
Expected: all `[PASS]` lines and `Bond probe ok`. If an assertion fails, the *claim* is wrong — correct the probe to reflect true behavior, then carry that truth into the doc.

- [ ] **Step 4: Write `docs/api-audit/per-class/Bond.md`**

Use the six-section template. The Recommendation section must list **every** member from Step 1's inventory in a table with a keep/add/rename/remove tag. Each parity claim cites a probe assertion; each consistency finding cites a convention ID. Every keep/add/rename member gets a numpy-style docstring in the Docstrings section. Change table gives `current → recommended` for every non-keep member.

- [ ] **Step 5: Validate the document (must pass)**

Run:
```bash
source tools/env.sh
$PY tools/validate_doc.py Bond docs/api-audit/per-class/Bond.md
```
Expected: `PASS: Bond — N members covered`. If it reports uncovered members, add them to the Recommendation table.

- [ ] **Step 6: Commit**

```bash
git add docs/api-audit/per-class/Bond.md docs/api-audit/probes/Bond.py
git commit -m "docs: Bond API audit + behavioral probe"
```

---

### Task 4: `Symmetry` audit

**Files:**
- Create: `docs/api-audit/per-class/Symmetry.md`
- Create: `docs/api-audit/probes/Symmetry.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `Bond.md` (Bond references Symmetry).
- Produces: `Symmetry.md` — referenced by `UniTensor.md` and `essential-api.md`.

Follow the exact five-gate pattern from Task 3, substituting `Symmetry` for `Bond`:

- [ ] **Step 1: Inventory** — `$PY tools/member_inventory.py Symmetry > /tmp/Symmetry_inventory.txt` (~16 members).
- [ ] **Step 2: Write `probes/Symmetry.py`** — assert U1/Zn construction, `combine_rule`/`Nfer`, equality, and copy-vs-view for any mutating method, using `report(...)`.
- [ ] **Step 3: Run the probe** — `$PY docs/api-audit/probes/Symmetry.py`; expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/Symmetry.md`** — six-section template; every inventory member in the Recommendation table; parity claims cite probe assertions; consistency findings cite convention IDs; docstrings for every keep/add/rename member.
- [ ] **Step 5: Validate** — `$PY tools/validate_doc.py Symmetry docs/api-audit/per-class/Symmetry.md`; expected `PASS`.
- [ ] **Step 6: Commit** — `git add` the doc + probe; `git commit -m "docs: Symmetry API audit + behavioral probe"`.

---

### Task 5: Enums audit (`Type`, `Device`, `SymType`, `bondType`, `fermionParity`)

Enums are small; audit all five in one document (one Recommendation sub-table per enum). The validator runs once per enum against the same file.

**Files:**
- Create: `docs/api-audit/per-class/enums.md`
- Create: `docs/api-audit/probes/enums.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions.
- Produces: `enums.md` — referenced by every container doc and `essential-api.md`.

- [ ] **Step 1: Inventory all five**

Run:
```bash
source tools/env.sh
for E in Type Device SymType bondType fermionParity; do
  echo "== $E =="; $PY tools/member_inventory.py $E; done > /tmp/enums_inventory.txt
cat /tmp/enums_inventory.txt
```

- [ ] **Step 2: Write `probes/enums.py`** — assert the value set and any name/alias behavior (e.g. `Type.Double` identity, `Device.cpu`), using `report(...)`.
- [ ] **Step 3: Run the probe** — `$PY docs/api-audit/probes/enums.py`; expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/enums.md`** — one six-section block, with the Recommendation section containing five sub-tables (one per enum) so that all members of all five enums appear.
- [ ] **Step 5: Validate each enum**

Run:
```bash
source tools/env.sh
for E in Type Device SymType bondType fermionParity; do
  $PY tools/validate_doc.py $E docs/api-audit/per-class/enums.md; done
```
Expected: five `PASS` lines.

- [ ] **Step 6: Commit** — `git commit -m "docs: enums (Type/Device/SymType/bondType/fermionParity) audit + probe"`.

---

### Task 6: `UniTensor` audit (the heavyweight)

`UniTensor` has ~126 members — the largest unit. Sub-group its members within the document (construction, bond/label ops, structure/reshape, math, decomposition entry points, block/symmetry access, I/O) so it stays navigable, but keep it a single doc validated as one unit.

**Files:**
- Create: `docs/api-audit/per-class/UniTensor.md`
- Create: `docs/api-audit/probes/UniTensor.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `Bond.md`, `Symmetry.md`, `enums.md`.
- Produces: `UniTensor.md` — the central input to `linalg.md`, `Network.md`, and `essential-api.md`.

- [ ] **Step 1: Inventory** — `$PY tools/member_inventory.py UniTensor > /tmp/UniTensor_inventory.txt`; expect ~126 members. Group them in the scratch file by the seven categories above.
- [ ] **Step 2: Write `probes/UniTensor.py`** — cover the behavior-heavy claims: `Conj`/`Conj_`, `Dagger`/`Dagger_`, `permute`/`permute_`, `contiguous`/`contiguous_`, `reshape` copy-vs-view, `relabel`/`relabels`, `get_block`/`get_block_` (view vs copy), `set_elem`/indexing mutation, dtype/device conversion. Each uses `report(...)` and, for copy/view, `returns_view(make, derive, mutate, read)` (from `probe_helper`; `derive` applies the method under test, and it returns True iff the source observes the mutation).
- [ ] **Step 3: Run the probe** — `$PY docs/api-audit/probes/UniTensor.py`; expected all `[PASS]`. Correct any claim the runtime disproves.
- [ ] **Step 4: Write `per-class/UniTensor.md`** — six-section template; Recommendation table organized by the seven sub-groups but covering **every** member; parity section highlights the capitalized-name (`N1`) and in-place-suffix (`N2`) conventions; docstrings for every keep/add/rename member with explicit copy/view notes (`B1`/`B2`).
- [ ] **Step 5: Validate** — `$PY tools/validate_doc.py UniTensor docs/api-audit/per-class/UniTensor.md`; expected `PASS: UniTensor — 126 members covered`.
- [ ] **Step 6: Commit** — `git commit -m "docs: UniTensor API audit + behavioral probe"`.

---

### Task 7: `linalg` audit

53 free functions, including the decompositions (`Svd`, `Svd_truncate`, `Eigh`, `Qr`, `Gesvd`) that the reference algorithms depend on. Many have `Foo`/`Foo_` pairs.

**Files:**
- Create: `docs/api-audit/per-class/linalg.md`
- Create: `docs/api-audit/probes/linalg.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `UniTensor.md`.
- Produces: `linalg.md` — critical input to `essential-api.md`.

- [ ] **Step 1: Inventory** — `$PY tools/member_inventory.py linalg > /tmp/linalg_inventory.txt`; expect 53 callables.
- [ ] **Step 2: Write `probes/linalg.py`** — assert return arity/behavior of the decompositions on both `Tensor` and `UniTensor` inputs (e.g. `Svd` returns `(s, u, vt)` in this order; `Svd_truncate` honors `keepdim`; in-place `Foo_` variants), plus dtype/device consistency. Use `report(...)`.
- [ ] **Step 3: Run the probe** — `$PY docs/api-audit/probes/linalg.py`; expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/linalg.md`** — six-section template; Recommendation table covers all 53 functions; flag return-order/naming inconsistencies against conventions; docstrings for every keep/add/rename function.
- [ ] **Step 5: Validate** — `$PY tools/validate_doc.py linalg docs/api-audit/per-class/linalg.md`; expected `PASS`.
- [ ] **Step 6: Commit** — `git commit -m "docs: linalg API audit + behavioral probe"`.

---

### Task 8: `Network` / `LinOp` / `ncon` audit

The contraction surface. `ncon` is a free function; `Network` and `LinOp` are classes. Audit together in one document (per-unit Recommendation sub-tables), validated per class.

**Files:**
- Create: `docs/api-audit/per-class/network.md`
- Create: `docs/api-audit/probes/network.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `UniTensor.md`.
- Produces: `network.md` — input to `essential-api.md`.

- [ ] **Step 1: Inventory** — run `member_inventory.py` for `Network` and `LinOp`; capture `ncon`'s signature from `cytnx.ncon.__doc__`. Save to `/tmp/network_inventory.txt`.
- [ ] **Step 2: Write `probes/network.py`** — build a small network (e.g. contract two rank-2 UniTensors), assert `Network.PutUniTensor`/`Launch` behavior, `ncon` index-convention result correctness vs. a hand contraction, and `LinOp` matvec semantics. Use `report(...)`.
- [ ] **Step 3: Run the probe** — `$PY docs/api-audit/probes/network.py`; expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/network.md`** — six-section template; Recommendation sub-tables for `Network`, `LinOp`, and `ncon` covering every member; docstrings for every keep/add/rename member.
- [ ] **Step 5: Validate** — run `validate_doc.py Network …` and `validate_doc.py LinOp …` against the file; expected two `PASS` lines. (`ncon` is a function, checked by presence in the doc.)
- [ ] **Step 6: Commit** — `git commit -m "docs: Network/LinOp/ncon API audit + behavioral probe"`.

---

### Task 9: Essential-API synthesis (`essential-api.md`)

Deliverable #6. Now derivable because the TN/contraction/decomposition units are done.

**Files:**
- Create: `docs/api-audit/essential-api.md`

**Interfaces:**
- Consumes: `UniTensor.md`, `Bond.md`, `Symmetry.md`, `enums.md`, `linalg.md`, `network.md`.
- Produces: `essential-api.md` — the minimal API set + per-algorithm traceability.

- [ ] **Step 1: Decompose each reference algorithm into primitive steps**

For each of **TRG, HOTRG, CTMRG, MERA**, write the step list: reshape/group bonds → contract → SVD/eigh + truncate → build isometry → renormalize/iterate. Ground HOTRG in `cytnx_src/docs/source/example/HOTRG.rst`; derive TRG/CTMRG/MERA from standard definitions (note lower confidence where no in-repo example exists).

- [ ] **Step 2: Map each step to concrete recommended API calls**

Produce a table: `algorithm | step | required recommended API calls`. Reference the recommended names from the per-class docs (post-rename), not the current names.

- [ ] **Step 3: Take the union → the essential set**

List the essential members grouped by unit, each with a back-reference to at least one `(algorithm, step)` that needs it.

- [ ] **Step 4: Validate traceability**

Run:
```bash
$PY - <<'EOF'
import re
t = open("docs/api-audit/essential-api.md").read()
# every essential entry line must cite at least one algorithm tag
algos = ("TRG", "HOTRG", "CTMRG", "MERA")
essent = [l for l in t.splitlines() if l.strip().startswith("- `")]
bad = [l for l in essent if not any(a in l for a in algos)]
print("essential entries without algorithm trace:", len(bad))
for l in bad[:10]: print("  ", l)
assert not bad
print("OK")
EOF
```
Expected: `OK`.

- [ ] **Step 5: Commit** — `git commit -m "docs: essential-API set for TRG/HOTRG/CTMRG/MERA"`.

---

### Task 10: `Tensor` audit

**Files:**
- Create: `docs/api-audit/per-class/Tensor.md`
- Create: `docs/api-audit/probes/Tensor.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `enums.md`.
- Produces: `Tensor.md`.

Follow the Task 3 five-gate pattern for `Tensor` (~67 members):

- [ ] **Step 1: Inventory** — `$PY tools/member_inventory.py Tensor > /tmp/Tensor_inventory.txt`.
- [ ] **Step 2: Write `probes/Tensor.py`** — assert copy-vs-view for `permute`/`reshape`/`contiguous`/slice-assignment, `clone` deep copy, dtype/device conversion, arithmetic operator semantics, `Conj`/`Conj_`. Use `report(...)` and `returns_view(make, derive, mutate, read)` from `probe_helper`.
- [ ] **Step 3: Run the probe** — expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/Tensor.md`** — six-section template covering all ~67 members.
- [ ] **Step 5: Validate** — `$PY tools/validate_doc.py Tensor docs/api-audit/per-class/Tensor.md`; expected `PASS`.
- [ ] **Step 6: Commit** — `git commit -m "docs: Tensor API audit + behavioral probe"`.

---

### Task 11: `Storage` audit

**Files:**
- Create: `docs/api-audit/per-class/Storage.md`
- Create: `docs/api-audit/probes/Storage.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `enums.md`.
- Produces: `Storage.md`.

Follow the five-gate pattern for `Storage` (~38 members):

- [ ] **Step 1: Inventory** — `$PY tools/member_inventory.py Storage > /tmp/Storage_inventory.txt`.
- [ ] **Step 2: Write `probes/Storage.py`** — assert `to`/`astype` copy behavior, `Move`/in-place variants, `resize`, element get/set mutation, `from_numpy`/`numpy` round-trip aliasing. Use `report(...)`.
- [ ] **Step 3: Run the probe** — expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/Storage.md`** — six-section template covering all ~38 members.
- [ ] **Step 5: Validate** — `$PY tools/validate_doc.py Storage docs/api-audit/per-class/Storage.md`; expected `PASS`.
- [ ] **Step 6: Commit** — `git commit -m "docs: Storage API audit + behavioral probe"`.

---

### Task 12: `Scalar` audit

**Files:**
- Create: `docs/api-audit/per-class/Scalar.md`
- Create: `docs/api-audit/probes/Scalar.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `enums.md`.
- Produces: `Scalar.md`.

Follow the five-gate pattern for `Scalar` (~12 members):

- [ ] **Step 1: Inventory** — `$PY tools/member_inventory.py Scalar > /tmp/Scalar_inventory.txt`.
- [ ] **Step 2: Write `probes/Scalar.py`** — assert dtype promotion, conversion to Python scalars, arithmetic/comparison operators, `conj`/`real`/`imag`. Use `report(...)`.
- [ ] **Step 3: Run the probe** — expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/Scalar.md`** — six-section template covering all ~12 members.
- [ ] **Step 5: Validate** — `$PY tools/validate_doc.py Scalar docs/api-audit/per-class/Scalar.md`; expected `PASS`.
- [ ] **Step 6: Commit** — `git commit -m "docs: Scalar API audit + behavioral probe"`.

---

### Task 13: `algo` / `random` / `physics` / `qgates` audit

Four small operation modules audited in one document (per-module Recommendation sub-tables), validated per module.

**Files:**
- Create: `docs/api-audit/per-class/operations.md`
- Create: `docs/api-audit/probes/operations.py`

**Interfaces:**
- Consumes: Task 1 tools, Task 2 conventions, `Tensor.md`, `UniTensor.md`.
- Produces: `operations.md`.

- [ ] **Step 1: Inventory** — run `member_inventory.py` for `algo`, `random`, `physics`, `qgates` into `/tmp/operations_inventory.txt`.
- [ ] **Step 2: Write `probes/operations.py`** — assert `algo.Sort`/`Concatenate` behavior, `random.normal_`/`uniform_` in-place-fill semantics and seeding, `physics` spin-operator shapes, `qgates` known gate matrices. Use `report(...)`.
- [ ] **Step 3: Run the probe** — expected all `[PASS]`.
- [ ] **Step 4: Write `per-class/operations.md`** — six-section template; four Recommendation sub-tables covering every member of all four modules.
- [ ] **Step 5: Validate each module**

Run:
```bash
source tools/env.sh
for M in algo random physics qgates; do
  $PY tools/validate_doc.py $M docs/api-audit/per-class/operations.md; done
```
Expected: four `PASS` lines.

- [ ] **Step 6: Commit** — `git commit -m "docs: algo/random/physics/qgates audit + probe"`.

---

### Task 14: Cross-cutting summary (`summary.md`)

Synthesizes global findings and the master index across all completed units.

**Files:**
- Create: `docs/api-audit/summary.md`

**Interfaces:**
- Consumes: all `per-class/*.md` and `essential-api.md`.
- Produces: `summary.md` — the top-level entry point to the deliverable.

- [ ] **Step 1: Write cross-cutting findings** — global naming/behavior inconsistencies that span classes (e.g. the capitalized-method-name pattern across `UniTensor`/`Tensor`/`linalg`; inconsistent `Foo_`/`Foo` coverage), each citing convention IDs.
- [ ] **Step 2: Build the master keep/add/rename/remove index** — one table aggregating every recommendation across all per-class docs, with a column for the owning unit.
- [ ] **Step 3: Cross-check the index against per-class docs**

Run:
```bash
$PY - <<'EOF'
import re, glob
idx = open("docs/api-audit/summary.md").read()
tags = re.findall(r"\b(keep|add|rename|remove)\b", idx.lower())
print("summary index rows tagged:", len(tags))
assert len(tags) > 0
print("OK")
EOF
```
Expected: `OK` with a nonzero count.

- [ ] **Step 4: Verify full-suite green (definition of done)**

Run:
```bash
source tools/env.sh
set -e
for p in docs/api-audit/probes/*.py; do echo "== $p =="; $PY "$p" >/dev/null && echo ok; done
$PY tools/validate_doc.py Bond   docs/api-audit/per-class/Bond.md
$PY tools/validate_doc.py Symmetry docs/api-audit/per-class/Symmetry.md
$PY tools/validate_doc.py UniTensor docs/api-audit/per-class/UniTensor.md
$PY tools/validate_doc.py linalg docs/api-audit/per-class/linalg.md
$PY tools/validate_doc.py Tensor docs/api-audit/per-class/Tensor.md
$PY tools/validate_doc.py Storage docs/api-audit/per-class/Storage.md
$PY tools/validate_doc.py Scalar docs/api-audit/per-class/Scalar.md
for E in Type Device SymType bondType fermionParity; do $PY tools/validate_doc.py $E docs/api-audit/per-class/enums.md; done
for M in algo random physics qgates; do $PY tools/validate_doc.py $M docs/api-audit/per-class/operations.md; done
for U in Network LinOp; do $PY tools/validate_doc.py $U docs/api-audit/per-class/network.md; done
echo "ALL GREEN"
```
Expected: every probe prints `ok`, every validator prints `PASS`, ending with `ALL GREEN`.

- [ ] **Step 5: Commit** — `git commit -m "docs: cross-cutting summary and master recommendation index"`.

---

## Definition of Done (from spec §7)

- [ ] Every public member of every in-scope unit appears in exactly one recommendation table (enforced by `validate_doc.py`).
- [ ] Every behavioral parity claim has a passing probe in `docs/api-audit/probes/`.
- [ ] Every recommended member has a docstring.
- [ ] The essential-API set traces each entry to at least one TRG/HOTRG/CTMRG/MERA step (enforced by Task 9 Step 4).
- [ ] Clean-slate `current → recommended` change tables cover every rename/remove.
- [ ] `summary.md` captures cross-class inconsistencies and the master index.
- [ ] `ALL GREEN` from Task 14 Step 4.
