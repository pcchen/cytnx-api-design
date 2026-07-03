# Cytnx 1.1.0 API Audit & Next-Version Recommendation Spec

**Date:** 2026-07-03
**Status:** Approved design
**Target:** Cytnx `1.1.0` (source cloned at `cytnx_src/`, runnable wheel installed in a venv)

## 1. Purpose

Analyze the current public API of Cytnx `>= 1.1.0` and design the next-version
API. The output is an **analysis + recommendation spec** (a set of documents),
not a code implementation of the new API.

For every public API element the spec must:

1. **Verify C++/Python parity**, including *runtime object behavior* (copy-vs-view,
   mutability, in-place vs. return, dtype/device promotion, operator semantics).
2. **Audit per-class internal consistency** (naming, argument order/conventions,
   return conventions, idioms).
3. **Recommend** an action per element: **keep / add / rename / remove**, with rationale.
4. **Provide a docstring** for every element of the recommended API.
5. **Show the change** explicitly as a `current → recommended` mapping.
6. **Identify the essential API subset** required to build higher-level tensor-network
   algorithms, using **TRG / HOTRG / CTMRG / MERA** as the reference algorithms.

## 2. Scope

### In scope (auditable units)

Core containers, TN containers, enums/config, operations, and network contraction:

| Group | Units |
|---|---|
| Core containers | `Tensor` (67 members), `Storage` (38), `Scalar` (12) |
| TN containers | `UniTensor` (126), `Bond` (30), `Symmetry` (16) |
| Enums / config | `Type`, `Device`, `SymType`, `bondType`, `fermionParity` |
| Operations | `linalg` (53), `algo` (6), `random` (4), `physics` (2), `qgates` (9) |
| Network | `Network` (18), `LinOp` (7), `ncon` |
| Python `*_conti` layer | `UniTensor_conti`, `linalg_conti`, `Bond_conti`, `Storage_conti`, `Symmetry_conti`, `Tensor_conti`, `Network_conti` — the Python augmentation layer, audited *as part of* each class's parity analysis (not as standalone docs) |

Member counts are from the installed 1.1.0 wheel and are approximate; the audit
enumerates the exact set.

### Out of scope

- **`tn_algo` (MPS / MPO / DMRG)** — MPS-family higher-level algorithms. Not audited,
  and **not** used to derive the essential set. They are neither reference algorithms
  nor part of the low-level API surface under review.
- Internal / private symbols (leading underscore), build system, CI, docs tooling.
- Implementing the recommended next-version API (that is future work, driven by this spec).

### Compatibility posture

**Clean-slate redesign.** Recommendations propose the ideal API regardless of
breakage. The `current → recommended` change tables *are* the migration guide; there
is no obligation to preserve old names or provide deprecation aliases.

## 3. Deliverable set (document structure)

Produced under `docs/api-audit/`. Chosen structure: **one artifact per reviewable
unit** (Approach A), so each document
is small, independently reviewable, and matches the natural per-class decomposition.

- **`00-methodology.md`** — the audit template, the naming/behavior/return **conventions**
  the recommended API must obey, and how parity/consistency/essential-set are judged.
  Written first; referenced by every other doc. "Inconsistent" always means "violates
  a convention defined here."
- **`per-class/<Unit>.md`** (~13 docs) — one per in-scope auditable unit, following the
  template in §4.
- **`essential-api.md`** — synthesis of deliverable #6 (see §5.3).
- **`summary.md`** — cross-cutting global findings (naming/behavior inconsistencies that
  span classes) plus a master keep/add/rename/remove index across all units.
- **`probes/<Unit>.py`** — the executable runtime probe scripts backing every behavioral
  claim (see §5.1). Run against the venv-installed Cytnx 1.1.0.

## 4. Per-class document template

Every `per-class/<Unit>.md` contains these sections, in order. Together they satisfy
deliverables #1–#5 for that unit.

1. **Inventory table** — every public member, side-by-side
   **C++ signature ｜ Python signature**, sourced from `cytnx_src/include/*.hpp` +
   `cytnx_src/pybind/*_py.cpp` + `cytnx_src/cytnx/*_conti.py`.
2. **Parity findings** — C++ vs Python divergences in *signature* and *object behavior*
   (copy-vs-view, in-place `_`-suffix pairs, mutability, dtype/device promotion, operator
   overloads, error/exception behavior). **Each behavioral claim is confirmed by an
   executed probe** in `probes/<Unit>.py`; the probe assertion/output is the evidence.
3. **Consistency findings** — internal incoherence measured against `00-methodology.md`
   conventions (naming case, `Foo`/`Foo_` pairs, capitalized vs. lowercase, argument
   order, return conventions).
4. **Recommendation table** — per member: **keep / add / rename / remove** + one-line
   rationale.
5. **Docstrings** — numpy-style docstring for **every** member of the *recommended*
   surface (summary, parameters, returns, and explicit behavior notes covering
   copy/view and in-place semantics). Produced for the entire recommended surface, not
   just the essential subset.
6. **Change table** — `current (C++ & Python) → recommended`, the clean-slate migration map.

## 5. Methodology

### 5.1 Parity & object behavior

1. Statically read the three layers (C++ header/impl, pybind wrapper, `*_conti.py`).
2. Author a runtime probe in `probes/<Unit>.py` that asserts each behavioral claim
   (e.g. "`Conj` returns a copy; `Conj_` mutates in place"; "`T[i]=x` on an aliased
   handle mutates the original").
3. Execute the probe against the venv-installed Cytnx 1.1.0. **No behavioral claim
   ships unverified** — the passing probe is the citation.

Environment: prebuilt manylinux wheel `cytnx==1.1.0` (cp312) installs cleanly into a
Python venv; `import cytnx` and behavioral probing are confirmed working.

### 5.2 Consistency

`00-methodology.md` fixes a single set of conventions (naming case, in-place suffix,
return-vs-mutate, argument order, error handling) that the whole recommended API must
obey. Each consistency finding cites the specific convention it violates, so findings
are objective rather than stylistic opinion.

Known early signal to formalize: capitalized method names (`Conj`, `Dagger`, `Inv`)
and trailing-underscore in-place variants (`Conj_`, `Dagger_`, `Init`) coexist with
lowercase Pythonic methods — a convention conflict the audit must resolve.

### 5.3 Essential-API set (deliverable #6)

1. Decompose each reference algorithm — **TRG, HOTRG, CTMRG, MERA** — into primitive
   steps: reshape/group bonds → contract → SVD/eigh + truncate → build isometry →
   renormalize/iterate.
2. Map each step to the concrete API calls it requires.
3. Take the union across all four algorithms → the essential set.
4. Ground the derivation in real repo code where available (`cytnx_src/docs/source/example/HOTRG.rst`
   provides a working HOTRG reference); derive TRG/CTMRG/MERA from their standard
   algorithm definitions.

Each entry in `essential-api.md` traces to at least one algorithm step.

## 6. Sequencing (essential-first)

1. `00-methodology.md` — fix conventions and the template.
2. `Bond` → `Symmetry` → enums (`Type`/`Device`/`SymType`/`bondType`/`fermionParity`)
   — the vocabulary the containers depend on.
3. `UniTensor` → `linalg` → `Network` / `LinOp` / `ncon` — the core TN + decomposition +
   contraction surface.
4. `essential-api.md` — now derivable from the units above.
5. `Tensor` → `Storage` → `Scalar` → `algo` / `random` / `physics` / `qgates`.
6. `summary.md` — cross-cutting synthesis and master index.

## 7. Definition of done

- Every public member of every in-scope unit appears in exactly one recommendation table.
- Every behavioral parity claim has a corresponding passing probe in `probes/`.
- Every recommended member has a docstring.
- The essential-API set traces each entry to at least one TRG/HOTRG/CTMRG/MERA step.
- Clean-slate `current → recommended` change tables cover every rename and remove.
- `summary.md` captures cross-class inconsistencies and a master keep/add/rename/remove index.

## 8. Risks & notes

- **`UniTensor` is large (~126 members).** It is the highest-effort unit; budget for it
  accordingly and consider sub-grouping its members (construction, bond/label ops, math,
  decomposition entry points, I/O) within its document.
- **Wheel vs. source drift.** Behavioral probes run against the wheel; signatures are
  read from the cloned source. Both are pinned to 1.1.0, but any mismatch found is itself
  a reportable finding.
- **CTMRG/MERA lack in-repo examples.** Their essential-set derivation relies on standard
  algorithm definitions rather than shipped code; this is noted where it affects confidence.
