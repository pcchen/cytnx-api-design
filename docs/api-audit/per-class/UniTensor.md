# `UniTensor` — API audit

`UniTensor` is the central tensor-network object of Cytnx: a labelled,
directed tensor that wraps either a dense `Tensor` (Dense `uten_type`) or a
list of symmetry-conserving blocks (Block / BlockFermionic `uten_type`), plus
the per-leg `Bond`/label metadata (`Bond.md`), the `Symmetry` objects that
govern its blocks (`Symmetry.md`), and the `Type`/`Device` dtype/device codes
(`enums.md`). This document audits the **126 public members** of the live
`cytnx.UniTensor` class (installed `cytnx==1.1.0` wheel).

Ground truth for behavior is `docs/api-audit/probes/UniTensor.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/UniTensor.py`; all 52 assertions `[PASS]`, exit 0).
Ground truth for static signatures is `cytnx_src/include/UniTensor.hpp` (the
public `UniTensor` wrapper class at line 2677 and its
`UniTensor_base`/`DenseUniTensor`/`BlockUniTensor`/`BlockFermionicUniTensor`
implementation hierarchy), `cytnx_src/pybind/unitensor_py.cpp` (the pybind11
binding — authoritative for the Python-visible call signatures and, crucially,
for the raw `c`-prefixed bindings, see the headline finding P1), and
`cytnx_src/cytnx/UniTensor_conti.py` (the Python-side augmentation layer that
defines the clean-named wrappers `Conj_`/`Dagger_`/`Pow_`/`at`/`relabel_`/… on
top of the raw `cConj_`/`cDagger_`/`cPow_`/`c_at`/`c_relabel_`/… bindings).

The headline structural fact, established by reading all three layers together,
is that **~18 raw pybind bindings are given `c`-prefixed public names and then
re-wrapped under clean Python names in `UniTensor_conti.py`, but the raw
`c`-prefixed forms are never hidden** — they remain non-underscore public
members that leak into the user surface next to their clean wrappers (P1/C1).

## Inventory

C++ signatures are read from `UniTensor.hpp`; Python signatures are the
effective pybind-visible signature, cross-checked against
`tools/member_inventory.py UniTensor`. The 126 members group into the seven
categories used throughout this document. Only the members whose C++/Python
divergence or runtime behavior is load-bearing are given a full signature row
here; the remainder are listed by group with their kind, and every one appears
in the Recommendation table.

### Construction / generators / conversion

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `Init` | `void Init(const Tensor&, bool is_diag=false, cytnx_int64 rowrank=-1, vector<string> labels={}, string name="")`; `void Init(const vector<Bond>&, vector<string> labels={}, cytnx_int64 rowrank=-1, int dtype=Type.Double, int device=Device.cpu, bool is_diag=false, string name="")` | `Init(...)` (2 overloads: from `Tensor`, or from `bonds`) — instance re-initializer, returns `None` |
| `zeros`/`ones`/`eye`/`identity` (static) | e.g. `static UniTensor zeros(const vector<cytnx_uint64>&, ...)` | `zeros(shape_or_Nelem, labels=[], dtype=3, device=-1, name='') -> UniTensor` (staticmethod), + a scalar-`Nelem` overload |
| `arange`/`linspace`/`normal`/`uniform` (static) | analogous to `Tensor` generators | staticmethods; `normal`/`uniform` additionally have in-place `normal_`/`uniform_` fillers |
| `clone` | `UniTensor clone() const` | `clone() -> UniTensor` (also bound as `__copy__`/`__deepcopy__`) |
| `astype` | *(Python-only, `UniTensor_conti.py:106`)* | `astype(dtype) -> UniTensor`: short-circuits to `self` if `dtype()==dtype`, else delegates to `astype_different_type` |
| `astype_different_type` | `UniTensor astype(...)` | `astype_different_type(new_type: int) -> UniTensor` — raw binding wrapped by `astype` |
| `to`/`to_` | `UniTensor to(const int&) const` / `void to_(const int&)` | `to(device)` short-circuits to `self` if same device (conti.py); `to_(device) -> UniTensor` (in place) |
| `to_different_device` | `UniTensor to(const int&) const` | `to_different_device(device: int) -> UniTensor` — raw binding wrapped by `to` |
| `convert_from` / `cfrom` | `void cfrom / from_(const UniTensor&, bool force, double tol)` | `convert_from(Tin, force=False, tol=0)` (conti.py wrapper, returns `self`) over the raw `cfrom(Tin, force, tol) -> None` |

### Bond / label ops

`bond`, `bond_`, `bonds`, `labels`, `name`, `set_name`, `set_label`,
`set_labels`, `get_index`, `relabel`, `relabel_`, `relabels`, `relabels_`,
`twist`, `twist_`, `tag`, `is_tag`, `fermion_twists`, `fermion_twists_`,
`signflip`, plus the raw bindings `ctag`, `c_relabel_`, `c_relabels_`,
`c_set_label`, `c_set_labels`, `c_set_name`.

`relabel` has four overloads — `(new_labels: list)`, `(idx: int, new: str)`,
`(old: str, new: str)`, `(old_labels: list, new_labels: list)`; `relabels`
exposes only the first and last of those four (P2). `bond`/`bond_` return a
`Bond` (see `Bond.md`); `bond_` is the by-reference variant.

### Structure / reshape

`reshape`, `reshape_`, `permute`, `permute_`, `permute_nosignflip`,
`permute_nosignflip_`, `combineBonds`, `rank`, `rowrank`, `set_rowrank`,
`set_rowrank_`, `c_set_rowrank_`, `shape`, `contiguous`, `contiguous_`,
`make_contiguous`, `is_contiguous`, `group_basis`, `group_basis_`.

`reshape`/`permute` return `*args`/list-mapper overloads. `contiguous` is a
**Python-only** short-circuit wrapper (`UniTensor_conti.py:120`: returns `self`
if already contiguous, else `make_contiguous`); `make_contiguous` is the raw
binding of C++ `UniTensor::contiguous` (always materializes a copy,
`unitensor_py.cpp:577`).

### Math

`Conj`, `Conj_`, `Dagger`, `Dagger_`, `Transpose`, `Transpose_`, `Inv`, `Pow`,
`Pow_`, `Trace`, `Trace_`, `Norm`, `normalize`, `normalize_`, `contract`,
`apply`, `apply_`, plus the raw bindings `cConj_`, `cDagger_`, `cTranspose_`,
`cInv_`, `cPow_`, `cTrace_`, `cnormalize_`, `c__ipow__`.

The capitalized pure forms (`Conj`, `Dagger`, `Transpose`, `Inv`, `Pow`,
`Trace`, `Norm`) return a new `UniTensor` (`Norm` returns a scalar `Tensor`);
each C++ in-place form returns `UniTensor&` and is bound under a `c`-prefixed
name, then re-wrapped by `UniTensor_conti.py`. `apply`/`apply_` apply fermionic
signflips (`unitensor_py.cpp:579-585`). `Init` and the operators
(`__add__`/`__mul__`/… and their in-place `__iadd__`/…) are present; there are
**no** named `add`/`Add`/`mul` counterparts to the operators (B5 is vacuous).

### Decomposition entry points

`truncate`, `truncate_`, plus the raw binding `ctruncate_`. There are **no**
`Svd`/`Eigh`/`Qr` methods on `UniTensor` itself — matrix decompositions are
`cytnx.linalg` free functions that consume a `UniTensor` and return the factor
`UniTensor`s; `truncate`/`truncate_` are the bond-dimension truncation entry
points used after a decomposition (see `essential-api.md` step (c)/(d)).

### Block / symmetry access

`get_block`, `get_block_`, `get_blocks`, `get_blocks_`, `get_blocks_qnums`,
`get_qindices`, `get_elem`, `set_elem`, `at`, `item`, `put_block`,
`put_block_`, `Nblocks`, `is_blockform`, `is_braket_form`, `is_diag`,
`getTotalQnums`, `syms`, `elem_exists`, `to_dense`, `to_dense_`, plus the raw
binding `c_at`.

`get_block`/`get_block_` and `get_blocks`/`get_blocks_` are the canonical
copy-vs-view pairs (P4/B2). `syms()` returns a list of `Symmetry`
(see `Symmetry.md`); `getTotalQnums` returns a list of `Bond`.

### I/O / introspection

`Save`, `Load`, `print_diagram`, `print_block`, `print_blocks`, `dtype`,
`dtype_str`, `device`, `device_str`, `uten_type`, `uten_type_str`, `same_data`.

`dtype`/`device` return integer codes (`Type`/`Device`, see `enums.md`);
`dtype_str`/`device_str` return human strings. The `print_*` methods are
correctly bound with `py::scoped_ostream_redirect` (`unitensor_py.cpp:585+`),
so — unlike `Symmetry`'s broken `__repr__` (`Symmetry.md` P5) and
`Device.Print_Property` (`enums.md` P4) — their output IS Python-capturable.

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/UniTensor.py`.

- **P1 — headline finding: ~18 raw pybind bindings carry `c`-prefixed public
  names that duplicate their clean Python wrappers, and the raw forms are never
  hidden.** For every in-place C++ method that returns `UniTensor&`
  (`Conj_`, `Dagger_`, `Transpose_`, `Inv_`, `Pow_`, `Trace_`, `normalize_`,
  `tag`, `truncate_`, `set_label`, `set_labels`, `set_name`, `set_rowrank_`,
  `relabel_`, `relabels_`, the `from_`/`at` helpers, and `__ipow__`), the
  binding is registered under a `c`-prefixed name (`cConj_`, `cDagger_`,
  `cTranspose_`, `cInv_`, `cPow_`, `cTrace_`, `cnormalize_`, `ctag`,
  `ctruncate_`, `c_set_label`, `c_set_labels`, `c_set_name`, `c_set_rowrank_`,
  `c_relabel_`, `c_relabels_`, `cfrom`, `c_at`, `c__ipow__`) and then
  `UniTensor_conti.py` defines a clean-named Python method that calls the
  `c`-form and `return self`. But nothing marks the `c`-forms private (no
  leading underscore), so **all 18 leak into the public surface**. Probed:
  *"all 18 raw c-prefixed bindings … are PUBLIC members (no leading underscore)
  that leak into the user surface alongside their clean Python wrappers"*
  `[PASS]`, and *"each clean wrapper … does exist and is the intended public
  spelling"* `[PASS]`. This is the single largest surface-hygiene problem on
  `UniTensor`; per methodology N2 (which explicitly names `cConj_`/`cDagger_`/
  `cInv_`/`cPow_`/`cTrace_`/`cTranspose_` as "not a recognized convention"),
  each must fold into its clean wrapper or be hidden.
- **P2 — `relabel` and `relabels` are redundant siblings; `relabels` exposes a
  strict subset of `relabel`'s overloads.** `relabel` binds four overloads
  (full list, `(idx, new)`, `(old, new)`, `(old_list, new_list)`); `relabels`
  binds only the full-list and `(old_list, new_list)` overloads — every
  `relabels` call is expressible as a `relabel` call. The same holds for
  `relabel_`/`relabels_`. Probed: *"relabel and relabels overlap heavily: both
  accept a full new-label list and an (old_labels, new_labels) pair; relabels
  lacks relabel's single (idx,new)/(old,new) scalar overloads"* `[PASS]`. Two
  spellings for one operation is an N4-adjacent vocabulary duplication.
- **P3 — `cInv_` mutates in place but returns a FRESH wrapper sharing data,
  whereas `cConj_` returns the SAME Python object — the raw `c`-bindings are
  not even internally uniform in return-identity.** `cConj_` binds the C++
  method directly (`.def("cConj_", &UniTensor::Conj_)`), so pybind maps the
  returned `*this` reference back to the existing Python wrapper
  (`cConj_() is self`). `cInv_` is a lambda that `return self.Inv_(clip)` **by
  value** (`unitensor_py.cpp:1370`), so pybind builds a new wrapper that merely
  shares storage (`cInv_() is not self`, `same_data` True). Probed:
  *"cConj_() … returns the same Python object, and shares its data"* `[PASS]`
  and *"cInv_() inverts the receiver in place … but … hands back a FRESH
  wrapper that merely shares data (ric is not ic, ric.same_data(ic)): the
  c-bindings are not even internally uniform in their return-identity
  behavior"* `[PASS]`. Any caller writing `x = t.cInv_()` and later relying on
  `x is t` gets language-inconsistent results across the two bindings.
- **P4 — `get_block`/`get_block_` and `get_blocks`/`get_blocks_` are the
  canonical copy-vs-view pairs (B2), and the distinction is exactly the
  trailing `_`.** `get_block()` returns a **clone** of the block `Tensor` (C++
  `_blocks[idx].clone()`); `get_block_()` returns a **reference** to the live
  block. Probed via `returns_view`: *"get_block() returns a COPY … mutating the
  returned Tensor does NOT change the source"* `[PASS]` and *"get_block_()
  returns a VIEW … mutating it IS observable on the source"* `[PASS]`; the same
  pair holds for `get_blocks()` (list of copies) vs `get_blocks_()` (list of
  views) — both `[PASS]`. This is model-correct N2/B2 behavior and is the
  template the rest of the class should follow.
- **P5 — the pure math forms are copies and the `_`/`c`-suffixed forms mutate
  in place (B1), verified per member.** `Conj()`/`Dagger()`/`Transpose()`/
  `Inv()`/`Pow()`/`normalize()` each return a new object and leave the source
  unchanged, while `Conj_()`/`Dagger_()`/`Pow_()`/… mutate the receiver and
  return the identical Python object. Probed for each: e.g. *"Conj() returns a
  NEW object; the source is unmutated"* `[PASS]`, *"Conj_() mutates the receiver
  in place … and returns the identical Python object"* `[PASS]`, and the
  analogous pairs for `Dagger`, `Transpose`, `Pow`, `normalize` — all `[PASS]`.
- **P6 — `permute`, `reshape`, and `relabel` all return VIEWS that share
  storage with the source (B2).** A write through the derived handle is visible
  on the original for all three. Probed via `returns_view`: *"permute() returns
  a VIEW that shares storage"* `[PASS]`, *"reshape() returns a VIEW sharing
  storage"* `[PASS]`, *"relabel() changes only bond-label metadata and returns
  a new wrapper that SHARES the underlying block storage"* `[PASS]`. Their
  `_`-suffixed forms mutate in place and return the same object (`permute_`,
  `reshape_`, `relabel_`), each `[PASS]`. This must be stated in every one of
  those members' docstrings.
- **P7 — `contiguous()` is NOT unconditionally a copy: the Python wrapper
  short-circuits to `return self` when the tensor is already contiguous.**
  `UniTensor_conti.py:120` returns the identical object for an
  already-contiguous tensor and only calls `make_contiguous` (the always-copy
  C++ form) otherwise. Probed: *"contiguous() on an already-contiguous tensor
  returns the IDENTICAL object … so it is NOT unconditionally a copy"* `[PASS]`
  and *"a permuted tensor is non-contiguous; contiguous() then materializes a
  new contiguous object … which is NOT the same object"* `[PASS]`. The same
  `return self` short-circuit governs `astype(same dtype)` and `to(same
  device)` (both `[PASS]`) — a caller must not assume `contiguous()`/`astype()`/
  `to()` always yield a fresh, independent object.
- **P8 — operator dtype promotion follows the widen-to-more-general rule (B3),
  matching `Tensor`.** `Double + ComplexDouble` yields a `ComplexDouble`
  result. Probed: *"__add__ promotes Double + ComplexDouble to ComplexDouble
  (B3: widen to the more general dtype), matching Tensor's promotion rule"*
  `[PASS]`. Scalar `*` broadcasts element-wise and returns a new object
  (`[PASS]`). Because there are no named `add`/`mul` methods, B5's
  operator-vs-named-method equivalence has nothing to compare against — recorded
  as vacuous (probe `[PASS]`), unlike `Tensor`, which exposes both.

## Consistency findings

- **C1 — violates N2 (headline): the 18 raw `c`-prefixed bindings are exposed
  publicly alongside their clean wrappers (P1).** Methodology N2 explicitly
  lists `cConj_`/`cDagger_`/`cInv_`/`cPow_`/`cTrace_`/`cTranspose_` as "not a
  recognized convention"; the same applies to the `c_`-prefixed metadata
  setters (`c_relabel_`, `c_relabels_`, `c_set_label`, `c_set_labels`,
  `c_set_name`, `c_set_rowrank_`, `c_at`), the block/misc raw forms (`cfrom`,
  `cnormalize_`, `ctag`, `ctruncate_`), and `c__ipow__`. Each must fold into
  its clean wrapper (rename → remove the `c`-form) or, at minimum, be prefixed
  with `_` to leave the public surface. `cInv_` is the sole exception: it is the
  only in-place inverse and has **no** clean wrapper, so it must be *renamed*
  (not removed) to `inv_` to complete the missing `inv`/`inv_` pair.
- **C2 — violates N1: eleven capitalized-verb / camelCase callables.** `Conj`,
  `Conj_`, `Dagger`, `Dagger_`, `Transpose`, `Transpose_`, `Inv`, `Pow`,
  `Pow_`, `Trace`, `Trace_`, `Norm`, `Init`, `Load`, `Save`, `Nblocks` are
  capitalized; `getTotalQnums` and `combineBonds` are camelCase. Per N1 these
  become `conj`/`conj_`/`dagger`/`dagger_`/`transpose`/`transpose_`/`inv`/
  `pow`/`pow_`/`trace`/`trace_`/`norm`/`init`/`load`/`save`/`nblocks`/
  `get_total_qnums`/`combine_bonds_`. (This mirrors `Symmetry.md`'s C1 and
  `Bond.md`'s C4 for `Init`/`Load`/`Save`.)
- **C3 — violates N2: `combineBonds` is an in-place mutator whose name carries
  no trailing `_`.** It mutates the receiver (bond count drops, shape changes)
  and returns `None`, exactly the shape of an in-place method, but is named
  like a pure one. Probed: *"combineBonds() mutates the receiver in place (shape
  [2,3,4] -> [6,4]) and returns None, yet its name carries NO trailing _ — an
  N2 violation"* `[PASS]`. Recommend `combine_bonds_` (N1 + N2 together). Note
  there is no pure `combine_bonds()` counterpart either — a secondary N2 gap
  (an in-place op with no returns-new sibling), same shape as the `Inv_` gap.
- **C4 — violates N5: two boolean predicates lack an `is_`/`has_` prefix.**
  `elem_exists(locator) -> bool` and `same_data(other) -> bool` read as
  statements, not questions. Per N5 recommend `has_elem` and `shares_data`
  (`same_data` is borderline-adjectival, but `shares_data` reads correctly at
  the call site: `if a.shares_data(b): …`). The already-correct predicates
  `is_blockform`/`is_braket_form`/`is_contiguous`/`is_diag`/`is_tag` are the N5
  model these two should match.
- **C5 — violates N4: `relabels`/`relabels_` duplicate `relabel`/`relabel_`
  (P2).** Two method names for one operation, with `relabels` a strict subset
  of `relabel`'s overloads. Recommend removing `relabels`/`relabels_` and
  keeping the single `relabel`/`relabel_` spelling (which already covers every
  `relabels` call). Same "same concept → same name" spirit as `enums.md`'s C4
  (`BD_IN`/`BD_OUT` aliases).
- **C6 — `convert_from` is an in-place mutator named without a trailing `_`
  (N2), but renaming is not clearly an improvement.** Like `combineBonds` it
  mutates the receiver and returns `self`, so strictly it should be
  `convert_from_`. Unlike `combineBonds`, however, "convert *from* X" already
  reads as a mutation of the receiver, and there is no meaningful pure form
  (a pure "convert from" is just a clone-then-convert, i.e. `astype`/`to`).
  Flagged for completeness (cf. `Symmetry.md`'s informal C-findings);
  recommend keeping the name and documenting the in-place semantics explicitly
  rather than adding a `_`, since the base name already implies mutation.
- **C7 — `astype`/`astype_different_type` and `to`/`to_different_device` expose
  the internal delegate under a public name.** `astype` (conti.py) is the
  intended entry; `astype_different_type` is the raw binding it calls, and it is
  public. Same for `to`/`to_different_device`. This is the same P1/C1
  wrapper-leak pattern in a non-`c`-prefixed guise: recommend folding
  `astype_different_type` into `astype` and `to_different_device` into
  `to`/`to_` (remove the public delegate). Probed indirectly: *"astype(same
  dtype) short-circuits to `return self`"* and *"astype(different dtype)
  delegates to astype_different_type and returns a NEW object"* — both `[PASS]`.

## Recommendation

Every one of the 126 live public members of `cytnx.UniTensor` appears below,
tagged **keep / add / rename / remove**, organized by the seven categories.
No members are *added* (matrix decompositions remain `cytnx.linalg` free
functions, per `essential-api.md`); the actionable verdicts concentrate on the
18 `c`-prefixed raw-binding removals (C1), the N1 casing renames (C2), the
in-place-suffix renames (C3), and the redundant-sibling removals (C5/C7).

### Construction / generators / conversion

| Member | Verdict | Rationale |
|---|---|---|
| `Init` | rename | → `init` (C2/N1). Instance re-initializer (returns None). |
| `zeros` | keep | Static generator, already `snake_case`. |
| `ones` | keep | Static generator. |
| `eye` | keep | Static diagonal/identity generator (`is_diag` flag). |
| `identity` | keep | Static generator; overlaps `eye` (both accept `is_diag`) — document the relationship, do not rename. |
| `arange` | keep | Static generator (Nelem and start/end/step overloads). |
| `linspace` | keep | Static generator. |
| `normal` | keep | Static random generator. |
| `normal_` | keep | In-place random filler (correct N2 `_` suffix). |
| `uniform` | keep | Static random generator. |
| `uniform_` | keep | In-place random filler. |
| `clone` | keep | Independent deep copy (confirmed distinct-and-equal by probe); also `__copy__`/`__deepcopy__`. |
| `convert_from` | keep | In-place conversion; document the mutation (C6/N2) rather than rename. |
| `cfrom` | remove | Raw binding wrapped by `convert_from` (C1/N2). |
| `astype` | keep | dtype conversion; `return self` short-circuit on same dtype (P7). |
| `astype_different_type` | remove | Internal delegate leaked publicly (C7); fold into `astype`. |
| `to` | keep | Device move; `return self` short-circuit on same device (P7). |
| `to_` | keep | In-place device move (correct N2 `_` suffix). |
| `to_different_device` | remove | Internal delegate leaked publicly (C7); fold into `to`/`to_`. |

### Bond / label ops

| Member | Verdict | Rationale |
|---|---|---|
| `bond` | keep | Returns the `Bond` at an index/label (see `Bond.md`). |
| `bond_` | keep | By-reference `bond` variant. |
| `bonds` | keep | Returns the full list of `Bond`. |
| `labels` | keep | Returns the leg-label list. |
| `name` | keep | Returns the tensor's name string. |
| `set_name` | keep | In-place name setter (conti.py wrapper over `c_set_name`). |
| `set_label` | keep | In-place single-label setter (wraps `c_set_label`). |
| `set_labels` | keep | In-place all-labels setter (wraps `c_set_labels`). |
| `get_index` | keep | Label → leg-index lookup. |
| `relabel` | keep | Pure relabel (view, shares data, P6); four overloads. |
| `relabel_` | keep | In-place relabel (wraps `c_relabel_`). |
| `relabels` | remove | Redundant subset of `relabel` (C5/P2). |
| `relabels_` | remove | Redundant subset of `relabel_` (C5/P2). |
| `twist` | keep | Pure fermionic-leg twist. |
| `twist_` | keep | In-place fermionic-leg twist. |
| `tag` | keep | Assign bra/ket direction tags (wraps `ctag`). |
| `is_tag` | keep | Predicate, correct N5 `is_` prefix. |
| `fermion_twists` | keep | Pure fermion-sign twist application. |
| `fermion_twists_` | keep | In-place counterpart (correct N2). |
| `signflip` | keep | Returns the per-block fermion signflip flags. |
| `ctag` | remove | Raw binding wrapped by `tag` (C1). |
| `c_relabel_` | remove | Raw binding wrapped by `relabel_` (C1). |
| `c_relabels_` | remove | Raw binding wrapped by `relabels_` (removed anyway, C1/C5). |
| `c_set_label` | remove | Raw binding wrapped by `set_label` (C1). |
| `c_set_labels` | remove | Raw binding wrapped by `set_labels` (C1). |
| `c_set_name` | remove | Raw binding wrapped by `set_name` (C1). |

### Structure / reshape

| Member | Verdict | Rationale |
|---|---|---|
| `reshape` | keep | Pure reshape (view, shares data, P6). |
| `reshape_` | keep | In-place reshape (correct N2). |
| `permute` | keep | Pure permute (view, shares data, P6). |
| `permute_` | keep | In-place permute (correct N2). |
| `permute_nosignflip` | keep | Fermionic permute skipping sign bookkeeping. |
| `permute_nosignflip_` | keep | In-place counterpart. |
| `combineBonds` | rename | → `combine_bonds_` (C2/N1 + C3/N2: it is in-place). |
| `rank` | keep | Number of legs. |
| `rowrank` | keep | Row-rank (row/column split point). |
| `set_rowrank` | keep | Pure row-rank setter (returns new). |
| `set_rowrank_` | keep | In-place row-rank setter (wraps `c_set_rowrank_`). |
| `c_set_rowrank_` | remove | Raw binding wrapped by `set_rowrank_` (C1). |
| `shape` | keep | Returns the shape list. |
| `contiguous` | keep | `return self` short-circuit when already contiguous (P7). |
| `contiguous_` | keep | In-place make-contiguous (correct N2). |
| `make_contiguous` | keep | Always-copy contiguous form (the raw C++ `contiguous`); document the always-copy semantics vs the short-circuiting `contiguous`. |
| `is_contiguous` | keep | Predicate, correct N5. |
| `group_basis` | keep | Pure block-basis grouping (symmetric tensors). |
| `group_basis_` | keep | In-place counterpart (correct N2). |

### Math

| Member | Verdict | Rationale |
|---|---|---|
| `Conj` | rename | → `conj` (C2/N1); pure copy (P5). |
| `Conj_` | rename | → `conj_` (C2/N1); in-place (P5). |
| `cConj_` | remove | Raw binding wrapped by `Conj_`/`conj_` (C1). |
| `Dagger` | rename | → `dagger` (C2/N1); pure copy = conj+transpose (P5). |
| `Dagger_` | rename | → `dagger_` (C2/N1); in-place. |
| `cDagger_` | remove | Raw binding wrapped by `Dagger_`/`dagger_` (C1). |
| `Transpose` | rename | → `transpose` (C2/N1); pure copy (P5). |
| `Transpose_` | rename | → `transpose_` (C2/N1); in-place. |
| `cTranspose_` | remove | Raw binding wrapped by `Transpose_`/`transpose_` (C1). |
| `Inv` | rename | → `inv` (C2/N1); pure element-wise pseudo-inverse (P5). |
| `cInv_` | rename | → `inv_` (C1): the ONLY in-place inverse; promote to the clean wrapper that never existed (completes the `inv`/`inv_` pair, P3). |
| `Pow` | rename | → `pow` (C2/N1); pure element-wise power. |
| `Pow_` | rename | → `pow_` (C2/N1); in-place. |
| `cPow_` | remove | Raw binding wrapped by `Pow_`/`pow_` (C1). |
| `c__ipow__` | remove | Raw binding wrapped by the `__ipow__` operator (C1). |
| `Trace` | rename | → `trace` (C2/N1); pure partial trace. |
| `Trace_` | rename | → `trace_` (C2/N1); in-place. |
| `cTrace_` | remove | Raw binding wrapped by `Trace_`/`trace_` (C1). |
| `Norm` | rename | → `norm` (C2/N1); returns a scalar `Tensor`. |
| `normalize` | keep | Pure 2-norm normalization (already `snake_case`). |
| `normalize_` | keep | In-place counterpart (wraps `cnormalize_`). |
| `cnormalize_` | remove | Raw binding wrapped by `normalize_` (C1). |
| `contract` | keep | Pairwise contraction of two `UniTensor`s over shared labels. |
| `apply` | keep | Pure fermionic-signflip application (`unitensor_py.cpp:579`). |
| `apply_` | keep | In-place counterpart. |

### Decomposition entry points

| Member | Verdict | Rationale |
|---|---|---|
| `truncate` | keep | Pure bond-dimension truncation (post-decomposition step). |
| `truncate_` | keep | In-place counterpart (wraps `ctruncate_`). |
| `ctruncate_` | remove | Raw binding wrapped by `truncate_` (C1). |

### Block / symmetry access

| Member | Verdict | Rationale |
|---|---|---|
| `get_block` | keep | Returns a COPY of a block (P4/B2). |
| `get_block_` | keep | Returns a VIEW of a block (P4/B2). |
| `get_blocks` | keep | Returns a list of block COPIES (P4). |
| `get_blocks_` | keep | Returns a list of block VIEWS (P4). |
| `get_blocks_qnums` | keep | Returns each block's qnum list. |
| `get_qindices` | keep | Returns a block's per-leg qnum indices. |
| `get_elem` | keep | Read a single element by locator. |
| `set_elem` | keep | Write a single element in place (B2). |
| `at` | keep | Proxy element accessor (`at(loc).value`), wraps `c_at`. |
| `item` | keep | Extract the scalar of a 1-element tensor. |
| `put_block` | keep | Write a block (copy semantics). |
| `put_block_` | keep | Write a block (view/reference semantics). |
| `Nblocks` | rename | → `nblocks` (C2/N1). |
| `is_blockform` | keep | Predicate, correct N5. |
| `is_braket_form` | keep | Predicate, correct N5. |
| `is_diag` | keep | Predicate, correct N5. |
| `getTotalQnums` | rename | → `get_total_qnums` (C2/N1). |
| `syms` | keep | Returns the leg `Symmetry` list (see `Symmetry.md`). |
| `elem_exists` | rename | → `has_elem` (C4/N5): boolean predicate. |
| `to_dense` | keep | Pure block→dense conversion. |
| `to_dense_` | keep | In-place counterpart (correct N2). |

### I/O / introspection

| Member | Verdict | Rationale |
|---|---|---|
| `Save` | rename | → `save` (C2/N1). |
| `Load` | rename | → `load` (C2/N1), static. |
| `print_diagram` | keep | Correctly `scoped_ostream_redirect`-guarded (capturable). |
| `print_block` | keep | Same guard. |
| `print_blocks` | keep | Same guard. |
| `dtype` | keep | Integer `Type` code (see `enums.md`). |
| `dtype_str` | keep | Human dtype string. |
| `device` | keep | Integer `Device` code (see `enums.md`). |
| `device_str` | keep | Human device string. |
| `uten_type` | keep | Integer UniTensor-kind code (Dense/Block/BlockFermionic). |
| `uten_type_str` | keep | Human kind string. |
| `same_data` | rename | → `shares_data` (C4/N5): boolean predicate. |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`
above, under its recommended name. Members are grouped by family (as in
`enums.md`); every kept/renamed member's name appears in backticks so
`validate_doc.py` can match it back to its Recommendation row. Every renamed
member's Notes line records the `current` → recommended rename and its `N`/`B`
citation.

### Generators & construction: `zeros`, `ones`, `eye`, `identity`, `arange`, `linspace`, `normal`, `normal_`, `uniform`, `uniform_`, `Init`

```
Construct a UniTensor.

The static generators build a fresh Dense UniTensor:
zeros / ones           : filled with 0 / 1.
eye / identity         : diagonal identity (pass is_diag=True for a stored
                         diagonal block; `eye` and `identity` are equivalent
                         factories, kept as aliases for familiarity).
arange                 : 0..N-1 vector (Nelem overload) or start/end/step.
linspace               : evenly spaced values.
normal / uniform       : random values; `normal_` / `uniform_` are the
                         in-place fillers that overwrite an existing tensor's
                         storage (correct N2 `_` suffix).

Parameters
----------
shape : sequence of int
    The bond dimensions (a scalar `Nelem` overload builds a rank-1 vector).
labels : sequence of str, optional
    One leg label per bond; auto-assigned if omitted.
dtype : int, optional
    A `cytnx.Type` code (default Type.Double). See enums.md.
device : int, optional
    A `cytnx.Device` code (default Device.cpu). See enums.md.
name : str, optional
    The tensor's name.

Returns
-------
UniTensor
    A new Dense UniTensor.

Notes
-----
`Init` (renamed from `Init`, C2/N1) is the *instance* counterpart: it
re-initializes an existing UniTensor in place (from a `Tensor`, or from a list
of `Bond`s) and returns None — it is a mutator, not a factory.
```

### Copy / conversion: `clone`, `astype`, `to`, `to_`, `convert_from`

```
Copy or convert a UniTensor.

`clone()` returns an independent deep copy (a fresh object with its own
storage). `astype(dtype)` returns a dtype-converted copy — but SHORT-CIRCUITS
to `return self` (the identical object, no copy) when `dtype` already matches
(Parity finding P7). `to(device)` / `to_(device)` move to another `Device`;
`to()` likewise short-circuits to `self` on the same device, while `to_()`
moves in place. `convert_from(Tin, force=False, tol=0)` converts the receiver
IN PLACE to match `Tin` and returns self.

Parameters
----------
dtype : int
    Target `cytnx.Type` code (for `astype`).
device : int
    Target `cytnx.Device` code (for `to`/`to_`).
Tin : UniTensor
    Template tensor (for `convert_from`).

Returns
-------
UniTensor
    A copy/converted tensor, or `self` when the short-circuit applies
    (`astype`/`to`) or for the in-place forms (`to_`/`convert_from`).

Notes
-----
Do NOT assume `astype`/`to`/`clone` always yield a fresh object: `astype(same)`
and `to(same)` return `self` (P7). `convert_from` is an in-place mutator whose
name lacks a trailing `_` (Consistency finding C6); its name already reads as a
receiver mutation, so it is kept as-is with the in-place semantics documented
here. The internal delegates `astype_different_type`/`to_different_device` are
removed (folded into `astype`/`to`, C7).
```

### Bonds & labels: `bond`, `bond_`, `bonds`, `labels`, `name`, `set_name`, `set_label`, `set_labels`, `get_index`, `relabel`, `relabel_`

```
Inspect and edit a UniTensor's leg (bond/label) metadata.

`bond(i|label)` / `bond_(i|label)` return the `Bond` on a leg (`bond_` is the
by-reference variant); `bonds()` returns all of them; `labels()` returns the
leg-label list; `name()`/`set_name(str)` get/set the tensor name;
`get_index(label)` maps a label to its leg index. `set_label`/`set_labels`
edit labels IN PLACE and return self. `relabel(...)` returns a NEW tensor that
SHARES the underlying storage (a metadata-only view, Parity finding P6) with
one of four overloads: a full new-label list, `(idx, new)`, `(old, new)`, or
`(old_list, new_list)`; `relabel_(...)` does the same edit in place.

Parameters
----------
idx : int
    A leg index.
label / old_label / new_label : str
    Leg labels.
new_labels / old_labels : sequence of str
    Full or paired label lists.

Returns
-------
Bond, list, str, int, or UniTensor
    Per the accessor; `relabel` returns a new (data-sharing) UniTensor,
    `relabel_`/`set_label`/`set_labels`/`set_name` return self.

Notes
-----
`relabel` is a VIEW: a data write through the relabeled handle is visible on
the source (P6/B2), and the source's own labels are unchanged. The redundant
`relabels`/`relabels_` siblings are removed (folded into `relabel`/`relabel_`,
C5/P2). See `Bond.md` for the `Bond` object these accessors return.
```

### Fermionic legs & tags: `twist`, `twist_`, `tag`, `is_tag`, `fermion_twists`, `fermion_twists_`, `signflip`

```
Fermionic-leg sign bookkeeping and bra/ket tagging.

`twist(i|label)` / `twist_(...)` apply a fermionic leg twist (pure / in place).
`fermion_twists()` / `fermion_twists_()` apply the full fermion-sign twist set
(pure / in place). `tag()` assigns bra/ket directions to the legs (in place,
wraps the raw `ctag`); `is_tag()` is the predicate for whether the tensor is
tagged. `signflip()` returns the per-block fermion signflip flags.

Returns
-------
UniTensor, bool, or list of bool
    The twist/tag mutators return self; `is_tag` returns bool; `signflip`
    returns the flag list.

Notes
-----
See `apply`/`apply_` (math group) for materializing pending signflips. These
are only meaningful for fermionic (BlockFermionic) UniTensors; on non-fermionic
tensors they are no-ops.
```

### Structure & reshape: `reshape`, `reshape_`, `permute`, `permute_`, `permute_nosignflip`, `permute_nosignflip_`, `rank`, `rowrank`, `set_rowrank`, `set_rowrank_`, `shape`, `contiguous`, `contiguous_`, `make_contiguous`, `is_contiguous`, `group_basis`, `group_basis_`, `combineBonds`

```
Reshape, permute, and re-lay-out a UniTensor.

`reshape(...)` / `permute(...)` return a NEW tensor that SHARES storage with the
source (a VIEW, Parity finding P6/B2); `reshape_` / `permute_` do the same in
place and return self. `permute_nosignflip` / `permute_nosignflip_` are the
fermionic variants that skip the fermion-sign bookkeeping `permute` performs.
`combine_bonds_(indicators, force=False)` (renamed from `combineBonds`,
C2/N1 + C3/N2) fuses several legs into one IN PLACE (returns None).
`rank()`/`rowrank()`/`shape()` report structure; `set_rowrank(n)` returns a new
tensor with the row/column split moved, `set_rowrank_(n)` moves it in place.
`contiguous()` returns `self` when already contiguous and otherwise a new
contiguous copy (Parity finding P7); `make_contiguous()` ALWAYS returns a fresh
contiguous copy; `contiguous_()` compacts in place; `is_contiguous()` is the
predicate. `group_basis()` / `group_basis_()` regroup a symmetric tensor's
block basis (pure / in place).

Parameters
----------
mapper : sequence of int or sequence of str
    The new leg order (for `permute`) — by index or by label.
rowrank : int, optional
    New row-rank after the operation (default -1 = unchanged).
indicators : sequence of int or str
    The legs to combine (for `combine_bonds_`).

Returns
-------
UniTensor or bool or int or list
    View/copy per above; the `_`-suffixed forms and `combine_bonds_` mutate the
    receiver.

Notes
-----
`reshape`/`permute`/`relabel` are all VIEWS (P6). `contiguous` is NOT
unconditionally a copy (P7) — use `make_contiguous` if you require a guaranteed
fresh, independent, contiguous object. `combineBonds` is renamed to
`combine_bonds_` to mark its in-place nature (C3); there is no pure
`combine_bonds()` sibling.
```

### Conjugation, transpose, power, trace, norm: `Conj`, `Conj_`, `Dagger`, `Dagger_`, `Transpose`, `Transpose_`, `Inv`, `cInv_`, `Pow`, `Pow_`, `Trace`, `Trace_`, `Norm`, `normalize`, `normalize_`

```
Element-wise and index-level math on a UniTensor.

Each operation has a PURE form (returns a new object, source unchanged) and an
IN-PLACE form (mutates the receiver, returns self), verified per member
(Parity finding P5):

conj / conj_           : complex conjugation. (renamed from `Conj`/`Conj_`, C2)
dagger / dagger_       : conjugate + transpose. (renamed from `Dagger`/`Dagger_`)
transpose / transpose_ : invert the index order and swap bra/ket.
                         (renamed from `Transpose`/`Transpose_`)
inv / inv_             : element-wise pseudo-inverse with optional `clip`.
                         (`inv` renamed from `Inv`; `inv_` renamed from the raw
                         `cInv_` — the only in-place inverse, which had no clean
                         wrapper, Parity finding P3 / Consistency finding C1)
pow / pow_             : element-wise power. (renamed from `Pow`/`Pow_`)
trace / trace_         : partial trace over two legs (by index or label).
                         (renamed from `Trace`/`Trace_`)
norm                   : 2-norm; returns a scalar `Tensor`, not a UniTensor.
                         (renamed from `Norm`, C2)
normalize / normalize_ : divide by the 2-norm (already snake_case).

Parameters
----------
clip : float, optional
    (`inv`/`inv_`) elements with |value| <= clip map to 0 (pseudo-inverse);
    default -1 (no clipping).
p : float
    (`pow`/`pow_`) the exponent.
a, b : int or str
    (`trace`/`trace_`) the two legs to trace over (default 0, 1).

Returns
-------
UniTensor or Tensor
    The pure forms return a new UniTensor (`norm` returns a scalar `Tensor`);
    the `_`-suffixed forms mutate the receiver and return self.

Notes
-----
Pure forms do not mutate the source; `_`-forms do and return the identical
Python object (P5). The raw `c`-prefixed bindings backing the in-place forms
(`cConj_`/`cDagger_`/`cTranspose_`/`cPow_`/`cTrace_`/`cnormalize_`) are removed
(C1); only `cInv_` survives, renamed to `inv_`, because it is the sole in-place
inverse with no other wrapper.
```

### Combination & signflip: `contract`, `apply`, `apply_`

```
Contract UniTensors and materialize fermionic signflips.

`contract(inR, mv_elem_self=False, mv_elem_rhs=False)` contracts this tensor
with `inR` over their shared leg labels, returning the contracted UniTensor
(the pairwise-contraction primitive underlying `essential-api.md` step (b)).
`apply()` returns a new UniTensor with pending fermionic signflips materialized
(blocks needing a flip are copied and inverted; others are shared views);
`apply_()` does so in place, after which `signflip()` reports all-False.

Parameters
----------
inR : UniTensor
    The right operand to contract with.
mv_elem_self, mv_elem_rhs : bool, optional
    Permit moving element storage of the operands for efficiency.

Returns
-------
UniTensor
    The contraction result, or the signflip-applied tensor.

Notes
-----
Non-fermionic tensors are returned unchanged by `apply`/`apply_`. For full
network contraction (not just pairwise) see `Network`/`ncon`.
```

### Truncation: `truncate`, `truncate_`

```
Truncate a bond to a target dimension.

`truncate(bond, dim)` returns a NEW tensor with the given bond truncated to
`dim`; `truncate_(bond, dim)` does so in place and returns self (wrapping the
raw `ctruncate_`, which is removed, C1). This is the bond-dimension truncation
step applied after an SVD/Eigh decomposition (`essential-api.md` step (c)/(d)).

Parameters
----------
bond : int or str
    The bond to truncate (by index or label).
dim : int
    The retained dimension.

Returns
-------
UniTensor
    Truncated tensor (new for `truncate`, self for `truncate_`).
```

### Block & element access: `get_block`, `get_block_`, `get_blocks`, `get_blocks_`, `get_blocks_qnums`, `get_qindices`, `get_elem`, `set_elem`, `at`, `item`, `put_block`, `put_block_`, `Nblocks`, `is_blockform`, `is_braket_form`, `is_diag`, `getTotalQnums`, `syms`, `elem_exists`, `to_dense`, `to_dense_`

```
Access blocks, elements, and block/symmetry structure.

`get_block(...)` returns a COPY of a block `Tensor`; `get_block_(...)` returns a
VIEW (a reference to the live block) — the classic copy-vs-view pair (Parity
finding P4/B2). `get_blocks()` / `get_blocks_()` are the list-valued analogues
(list of copies vs list of views). `put_block(...)` / `put_block_(...)` write a
block. `get_elem(loc)` reads one element; `set_elem(loc, v)` writes one in
place (B2); `at(loc).value` is the proxy read/write accessor; `item()` extracts
the scalar of a 1-element tensor. `has_elem(loc)` (renamed from `elem_exists`,
C4/N5) tests whether an element exists in the block structure. `nblocks()`
(renamed from `Nblocks`, C2/N1) counts blocks; `is_blockform()`/`is_diag()`/
`is_braket_form()` are structure predicates. `get_total_qnums(physical=False)`
(renamed from `getTotalQnums`, C2/N1) and `get_blocks_qnums()`/
`get_qindices(i)` report the qnum structure; `syms()` returns the leg
`Symmetry` list (see `Symmetry.md`). `to_dense()` / `to_dense_()` convert a
block tensor to dense (pure / in place).

Parameters
----------
idx : int, optional
    Block index (default 0) for `get_block`/`get_block_`/`print_block`.
qnum / qidx : sequence of int
    Block quantum-number selector for the block accessors.
loc : sequence of int
    An element locator for `get_elem`/`set_elem`/`at`/`has_elem`.

Returns
-------
Tensor, list, bool, int, or object
    `get_block` returns a COPY, `get_block_` a VIEW (P4). Predicates return
    bool; `item`/`get_elem` return the scalar; `syms`/`get_total_qnums` return
    lists.

Notes
-----
The copy/view distinction on `get_block`/`get_block_` and
`get_blocks`/`get_blocks_` is exactly the trailing `_` (P4/B2). Mutating a
`get_block_()` result — or a `get_blocks_()` element — writes through to the
source; a `get_block()`/`get_blocks()` result is independent.
```

### I/O & introspection: `Save`, `Load`, `print_diagram`, `print_block`, `print_blocks`, `dtype`, `dtype_str`, `device`, `device_str`, `uten_type`, `uten_type_str`, `same_data`

```
Persist, print, and introspect a UniTensor.

`save(fname)` (renamed from `Save`, C2/N1) writes the tensor; `load(fname)`
(renamed from `Load`, static) reads it back (round trip preserves shape and
element data, confirmed by probe). `print_diagram(bond_info=False)`,
`print_block(idx, full_info=True)`, and `print_blocks(full_info=True)` print a
human-readable view — and, unlike `Symmetry.__repr__` (Symmetry.md P5) and
`Device.Print_Property` (enums.md P4), they are correctly guarded with
`py::scoped_ostream_redirect`, so their output IS capturable by Python's
`redirect_stdout`. `dtype()`/`device()` return integer `Type`/`Device` codes
(enums.md); `dtype_str()`/`device_str()` return the human strings.
`uten_type()`/`uten_type_str()` report the kind (Dense / Block /
BlockFermionic). `shares_data(other)` (renamed from `same_data`, C4/N5) tests
whether two UniTensors alias the same underlying storage.

Parameters
----------
fname : str
    File path (a `.cytnx` extension is appended if omitted, with a deprecation
    warning).
other : UniTensor
    The tensor to compare storage identity with (for `shares_data`).

Returns
-------
UniTensor, str, int, bool, or None
    `load` returns a new UniTensor; the `print_*` methods return None; the
    introspection accessors return codes/strings/bools.

Notes
-----
`dtype`/`device` return the same integer codes documented in `enums.md`; use
`dtype_str`/`device_str` for display. `shares_data` is the direct test behind
the view-vs-copy semantics probed throughout this document.
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.
Only the rows below change; every other member keeps its current name and
behavior.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Conj` / `Conj_` | `conj` / `conj_` | N1 casing (C2) |
| `Dagger` / `Dagger_` | `dagger` / `dagger_` | N1 casing (C2) |
| `Transpose` / `Transpose_` | `transpose` / `transpose_` | N1 casing (C2) |
| `Inv` | `inv` | N1 casing (C2) |
| `Pow` / `Pow_` | `pow` / `pow_` | N1 casing (C2) |
| `Trace` / `Trace_` | `trace` / `trace_` | N1 casing (C2) |
| `Norm` | `norm` | N1 casing (C2) |
| `Init` | `init` | N1 casing (C2) |
| `Load` / `Save` | `load` / `save` | N1 casing (C2) |
| `Nblocks` | `nblocks` | N1 casing (C2) |
| `getTotalQnums` | `get_total_qnums` | N1 camelCase (C2) |
| `combineBonds` | `combine_bonds_` | N1 casing + N2 in-place suffix (C2/C3) |
| `elem_exists` | `has_elem` | N5 predicate prefix (C4) |
| `same_data` | `shares_data` | N5 predicate prefix (C4) |
| `cInv_` | `inv_` | promote the only in-place inverse to a clean wrapper (C1/P3) |
| `cConj_` | *(removed)* → `conj_` | raw-binding leak (C1) |
| `cDagger_` | *(removed)* → `dagger_` | raw-binding leak (C1) |
| `cTranspose_` | *(removed)* → `transpose_` | raw-binding leak (C1) |
| `cPow_` | *(removed)* → `pow_` | raw-binding leak (C1) |
| `cTrace_` | *(removed)* → `trace_` | raw-binding leak (C1) |
| `cnormalize_` | *(removed)* → `normalize_` | raw-binding leak (C1) |
| `ctag` | *(removed)* → `tag` | raw-binding leak (C1) |
| `ctruncate_` | *(removed)* → `truncate_` | raw-binding leak (C1) |
| `cfrom` | *(removed)* → `convert_from` | raw-binding leak (C1) |
| `c__ipow__` | *(removed)* → `__ipow__` | raw-binding leak (C1) |
| `c_at` | *(removed)* → `at` | raw-binding leak (C1) |
| `c_relabel_` | *(removed)* → `relabel_` | raw-binding leak (C1) |
| `c_relabels_` | *(removed)* → `relabel_` | raw-binding leak (C1/C5) |
| `c_set_label` | *(removed)* → `set_label` | raw-binding leak (C1) |
| `c_set_labels` | *(removed)* → `set_labels` | raw-binding leak (C1) |
| `c_set_name` | *(removed)* → `set_name` | raw-binding leak (C1) |
| `c_set_rowrank_` | *(removed)* → `set_rowrank_` | raw-binding leak (C1) |
| `relabels` / `relabels_` | *(removed)* → `relabel` / `relabel_` | redundant sibling (C5/P2) |
| `astype_different_type` | *(removed)* → `astype` | internal delegate leak (C7) |
| `to_different_device` | *(removed)* → `to` / `to_` | internal delegate leak (C7) |

Every other public member of `UniTensor` — the generators (`zeros`, `ones`,
`eye`, `identity`, `arange`, `linspace`, `normal`, `normal_`, `uniform`,
`uniform_`), `clone`, `astype`, `to`, `to_`, `convert_from`, the bond/label
accessors (`bond`, `bond_`, `bonds`, `labels`, `name`, `set_name`, `set_label`,
`set_labels`, `get_index`, `relabel`, `relabel_`, `twist`, `twist_`, `tag`,
`is_tag`, `fermion_twists`, `fermion_twists_`, `signflip`), the structure ops
(`reshape`, `reshape_`, `permute`, `permute_`, `permute_nosignflip`,
`permute_nosignflip_`, `rank`, `rowrank`, `set_rowrank`, `set_rowrank_`,
`shape`, `contiguous`, `contiguous_`, `make_contiguous`, `is_contiguous`,
`group_basis`, `group_basis_`), `normalize`, `normalize_`, `contract`, `apply`,
`apply_`, `truncate`, `truncate_`, the block/element accessors (`get_block`,
`get_block_`, `get_blocks`, `get_blocks_`, `get_blocks_qnums`, `get_qindices`,
`get_elem`, `set_elem`, `at`, `item`, `put_block`, `put_block_`,
`is_blockform`, `is_braket_form`, `is_diag`, `syms`, `to_dense`, `to_dense_`),
and the I/O/introspection accessors (`print_diagram`, `print_block`,
`print_blocks`, `dtype`, `dtype_str`, `device`, `device_str`, `uten_type`,
`uten_type_str`) — keeps both its current name and current behavior unchanged.
