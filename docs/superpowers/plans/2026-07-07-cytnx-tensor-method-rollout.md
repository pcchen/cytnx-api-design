# Cytnx Tensor Method Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Analyze `cytnx.Tensor` (the dense array `UniTensor` wraps, 67 members) under the established superset method — 8 category docs + `inventory.md` + `private-surface.md` + `element-dtypes.md` — each probe-backed against the `cytnx==1.1.0` wheel, gated by the coverage + N-private validator.

**Architecture:** Identical to the completed UniTensor rollout. Each category is `docs/api-audit/Tensor/NN-<cat>.md` with `# Analysis` (A1 inventory · A2 C++↔Python map · A3 findings · A4 arg-ordering) + `# R.` normative spec (R.0 conventions · R.1 signatures+contract · R.2a numpy / R.2b Doxygen docstrings). Python probe per category against the wheel; a raw-C++ probe (source-built `libcytnx` at `/tmp/cytnx-build-probe/libcytnx.a`) only where a binding-fidelity finding exists.

**Tech Stack:** Python 3.12 venv (`cytnx==1.1.0`), `cytnx_src/` 1.1.0 source, g++13 + the prebuilt `libcytnx.a` for C++ probes, `tools/` (member_inventory, validate_doc, probe_helper).

**Reference — read before starting:**
- Method spec `docs/superpowers/specs/2026-07-06-cytnx-api-analysis-method-design.md`.
- The completed UniTensor exemplars: `docs/api-audit/UniTensor/02-static-generators.md`, `05-structure-manipulation.md`, `07-arithmetic-elementwise.md`, `private-surface.md`, and the shared **"## The per-category procedure"** in `docs/superpowers/plans/2026-07-06-cytnx-unitensor-method-rollout.md`.
- The v1 Tensor analysis `docs/api-audit/per-class/Tensor.md` (findings to carry over / verify).

## Global Constraints

Same as the UniTensor rollout (`2026-07-06-...-rollout.md` Global Constraints): target cytnx **1.1.0**; run via `source tools/env.sh && $PY`; **N-casing** (members lowercase, free-fns-acting-on-objects Capitalized, types Capitalized); **N-underscore** (trailing `_` = in-place returns self); **argument ordering** (positional `[primary operand],[op params]`, metadata keyword-only); **R.2a numpy / R.2b Doxygen** docstrings; **binding-fidelity** findings get a C++ probe (others Python-only); **N-private** (leaked `c*`/shim bindings → `private-surface.md` hide set); every rename/remove carries a `DeprecationWarning` migration note; **no claim ships unverified**; every live public member appears in exactly one category's `R.1` verdict row.

Tooling is already in place (no Task-1 scaffold): `validate_doc.py Tensor docs/api-audit/Tensor/` runs the categorized coverage + N-private gate; `Tensor` is in `UNIT_REGISTRY`.

## The per-category procedure (every category task runs these six gates)

The complete procedure is in the UniTensor plan's "## The per-category procedure" section — read it there. In brief: **(1)** inventory the category's members via `member_inventory.py Tensor`; **(2)** write `probes/Tensor_<NN>_<cat>.py` (one `report(...)` per behavioral claim, `returns_view` for copy/view); **(3)** run it green (exit 0) — a failed assertion means the *claim* is wrong, fix to runtime truth; **(4)** if a binding-fidelity finding exists, write+run `probes/cpp/Tensor_<NN>_<cat>.cpp` against `/tmp/cytnx-build-probe/libcytnx.a`; else record "no binding-fidelity finding"; **(5)** write `Tensor/<NN>-<cat>.md` reproducing the UniTensor pilot shape (ids `T-<letter>#`); **(6)** validate (`validate_doc.py Tensor docs/api-audit/Tensor/` — partial coverage expected until all 8 exist; confirm none of THIS category's members appear in "not covered"/"missing docstring") + commit doc+probe(s).

---

### Task 1: Category 01 — construction & init
**Files:** `docs/api-audit/Tensor/01-construction-init.md`, `probes/Tensor_01_construction.py`.
**Members (`T-C#`):** the constructors (`__init__` overloads), `Init`, `from_storage`.
**Findings to check:** `Init` is a public duplicate of the constructor (demote to `_init`); `from_storage` — does the Tensor **share** the Storage's buffer (view) or copy? (probe with a mutation; v1 noted a Python-only `is_clone` kwarg); constructor kwarg order + keyword-only. **C++ probe:** likely none (thin `py::init`) — confirm.
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 01 construction/init + probe"`.

### Task 2: Category 02 — metadata & introspection
**Files:** `02-metadata-introspection.md`, `probes/Tensor_02_metadata.py`.
**Members (`T-M#`):** `shape`, `rank`, `dtype`, `dtype_str`, `device`, `device_str`, `is_contiguous`, `same_data`.
**Findings:** mostly already lowercase/clean (contrast UniTensor's `Nblocks`); `same_data(arg0)` erased param name (PC1, keyword-uncallable); `is_contiguous` predicate. **C++ probe:** none expected.
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 02 metadata + probe"`.

### Task 3: Category 03 — element & storage access
**Files:** `03-element-storage-access.md`, `probes/Tensor_03_element.py`, and a C++ probe if a binding-fidelity finding surfaces.
**Members (`T-E#`):** `item`, `storage`, `fill`, `append`, `real`, `imag`, `numpy`, and the `__getitem__`/`__setitem__` dunders.
**Findings:** **the numpy bridge EXISTS** here (`numpy()` export, and `from_storage`) — probe whether `numpy()` shares the buffer or copies (v1: Storage `numpy()` returns a copy — verify for Tensor), and mark it a **keep** that closes the UniTensor UT-C3/UT-T6 gap (cross-ref); `storage()` view vs copy (`returns_view`); `real`/`imag` view vs copy; slice-read copy-vs-view (v1 Tensor C3 — slice read is a COPY, a numpy divergence); `fill`/`append` in-place semantics; the leftover `std::cout` debug leak in bare-slice `__getitem__` (B-13 / v1 P5) — probe capturability.
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 03 element/storage access + probe(s)"`.

### Task 4: Category 04 — shape / layout
**Files:** `04-shape-layout.md`, `probes/Tensor_04_shape.py`, `probes/cpp/Tensor_04_shape.cpp`.
**Members (`T-S#`):** `permute`, `permute_`, `reshape`, `reshape_`, `contiguous`, `contiguous_`, `flatten`, `flatten_`, `make_contiguous` (leak).
**Findings (binding-fidelity present):** `permute`/`reshape` return storage-sharing **views** (`returns_view`); `permute_`/`reshape_`/`contiguous_`/`flatten_` in-place return self; `contiguous` binds via the leaked `make_contiguous` shim; `flatten_` in-place-returns-`None`? (v1 Tensor C4 — verify); `reshape` `*args,**kwargs` signature erasure. **C++ probe:** C++ `permute_`/`contiguous_`/`flatten_` return `&*this` (so any Python `None`/identity drop is binding-introduced).
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 04 shape/layout + probes"`.

### Task 5: Category 05 — arithmetic & element-wise
**Files:** `05-arithmetic-elementwise.md`, `probes/Tensor_05_arithmetic.py`, `probes/cpp/Tensor_05_arithmetic.cpp`.
**Members (`T-A#`):** operator dunders (`__add__`/`__radd__`/`__iadd__`, `__sub__…`, `__mul__…`, `__truediv__…`, `__floordiv__…`, `__matmul__`/`__imatmul__`, `__neg__`, `__pow__`, `__ipow__`), `Abs`/`Abs_`, `Conj`/`Conj_`, `Exp`/`Exp_`, `Inv`/`Inv_`, `Pow`/`Pow_`, `Norm`, and the leaks `cAbs_`/`cConj_`/`cExp_`/`cInv_`/`cPow_`, `c__iadd__`/`c__ifloordiv__`/`c__imatmul__`/`c__imul__`/`c__ipow__`/`c__isub__`/`c__itruediv__`.
**Findings (headline — binding-fidelity + correctness):** **the `@=` bug (B-5)** — `Tensor_conti.py` defines `__imatmul` (missing trailing `__`), so `@=` is NOT in-place and rebinds `t` to a fresh object (probe: `t is not` its former self); N-casing on the capitalized *members* `Abs`/`Conj`/`Exp`/`Inv`/`Pow`/`Norm` → lowercase (these also exist as `linalg` free functions which stay Capitalized — cross-ref); `//` true-division (verify, like UniTensor UT-A1); the `c*`/`c__i*__` in-place ops drop identity (binding fidelity); `Inv`→`reciprocal` element-wise (cross-ref UniTensor UT-X4/cat08). **C++ probe:** C++ `Conj_`/`Abs_`/`Exp_`/`Pow_` return `&*this`; the true C++ in-place matmul exists (so `@=`'s brokenness is the binding).
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 05 arithmetic/elementwise + probes"`.

### Task 6: Category 06 — linear algebra (member) & reductions
**Files:** `06-linalg-reductions.md`, `probes/Tensor_06_linalg.py`.
**Members (`T-X#`):** `Svd`, `Eigh`, `InvM`, `InvM_`, `Trace`, `Norm`, `Max`, `Min`, and the leak `cInvM_`.
**Findings:** these are capitalized **member** methods → lowercase per N-casing (`Svd`→`svd`, `Eigh`→`eigh`, `InvM`→`inv_m`, `Trace`→`trace`, `Max`/`Min`→`max`/`min`) — note the same names exist as `linalg` FREE functions that STAY Capitalized (cross-ref UniTensor cat 08); decomposition return arity/order on a Tensor (`Svd` returns `[S,U,vT]` — value-verify by reconstruction); `InvM` (matrix inverse) vs the element-wise `Inv` (cat 05) collision. **C++ probe:** none expected (these mirror the linalg free functions) — confirm.
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 06 linalg/reductions + probe"`.

### Task 7: Category 07 — dtype / device conversion
**Files:** `07-type-device-conversion.md`, `probes/Tensor_07_typedevice.py`, `probes/cpp/Tensor_07_typedevice.cpp`.
**Members (`T-T#`):** `astype`, `to`, `to_`, `clone`, and the leaks `astype_different_dtype`, `to_different_device`.
**Findings (binding-fidelity present):** `astype`/`to` short-circuit to `is self` on a no-op (conti.py) — binding-introduced (cross-ref UniTensor UT-T1); `to_(arg0)` erased param name; `clone` deep copy (`returns_view` → False); the `*_different_*` shims leak. **C++ probe:** raw C++ `astype`/`to` no-op returns a **distinct** object (so `is self` is binding-introduced).
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 07 type/device conversion + probes"`.

### Task 8: Category 08 — I/O
**Files:** `08-io.md`, `probes/Tensor_08_io.py`.
**Members (`T-IO#`):** `Save`, `Load`, `Tofile`, `Fromfile`.
**Findings:** N-casing on the capitalized members `Save`/`Load`/`Tofile`/`Fromfile` → `save`/`load`/`tofile`/`fromfile`; `Save`→`Load` round-trip value-equality (dense — no name field, so no UniTensor UT-IO5 analog, but verify); `Tofile`/`Fromfile` raw-binary vs `Save`/`Load` `.cytnx` format; pickle state (probe whether `Tensor` pickles — `__getstate__`/`__setstate__`). **C++ probe:** none expected.
- [ ] Run the six-gate procedure. Commit `-m "docs: Tensor cat 08 io + probe"`.

### Task 9: `inventory.md` + full-coverage partition gate
**Files:** `docs/api-audit/Tensor/inventory.md`.
- [ ] Write the categorized member inventory (8 sections + a pointer to `private-surface.md`). Prove the partition: the union of the 8 category `# R.` sections' backtick names covers `{m for m in dir(cytnx.Tensor) if not m.startswith('_')}` (all 67). Run `validate_doc.py Tensor docs/api-audit/Tensor/` → `PASS: Tensor — 67 members covered across 8 files`. Commit `-m "docs: Tensor categorized inventory + full-coverage gate"`.

### Task 10: `private-surface.md` + N-private gate
**Files:** `docs/api-audit/Tensor/private-surface.md`.
- [ ] Classify the full private surface: the leaked internals (the 16 `c*`/`c__i*__`/`*_different_*`/`make_contiguous` — **confirm the exact set via the validator's leak net**, which found extra leaks on UniTensor), the single-underscore members, and the dunder buckets. Run `validate_doc.py Tensor docs/api-audit/Tensor/` and confirm the **N-private accounting gate PASSes** (every non-underscore member is recommended-public or a hide entry) and record the printed **leaked-public count** (target 0). Commit `-m "docs: Tensor private-surface + N-private gate"`.

### Task 11: `element-dtypes.md`
**Files:** `docs/api-audit/Tensor/element-dtypes.md`, `probes/Tensor_dtypes.py`.
- [ ] Probe the 11 dtypes storable in a `Tensor`, the promotion rule (reuse the UniTensor finding: `(A+B).dtype()==min(Type-code)`, widen-by-kind — verify holds for `Tensor`), and per-op dtype constraints (decompositions up-cast int→Double?). Write `element-dtypes.md` (dtype table cross-ref `enums.md`, promotion rule, constraint table, recommendation). Confirm the validator still PASSes (non-numbered file, excluded). Commit `-m "docs: Tensor element-dtypes + probe"`.

---

## Follow-up (out of scope)
Remaining classes (`Bond`, `Symmetry`, `Network`, `LinOp`, `Storage`, `Scalar`, enums, operations) each get their own rollout; the cross-class `cpp-python-mapping.md` + master index + refreshed `actionable-fixes.md` come once all classes are redone.

## Self-review notes
- **Coverage:** the 8 category member sets partition all 67 `dir(Tensor)` members (16 leaks routed to cats 04/05/06/07 and consolidated in Task 10). Every method-spec section (A1–A4, R.0–R.2b, N-private, actionable-typed findings) is exercised by the six-gate procedure.
- **Cross-refs:** Tensor's numpy bridge (cat 03) closes UniTensor UT-C3/UT-T6; the `@=` bug (cat 05) is v1 B-5; `astype`/`to` is-self (cat 07) mirrors UniTensor UT-T1 — each cross-referenced, not re-derived.
- **Finding-id prefixes** unique per category (`T-C/M/E/S/A/X/T/IO#`).
