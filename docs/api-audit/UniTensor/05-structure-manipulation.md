# UniTensor — 05. Structure manipulation

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the sibling
> [`04-labels-name-rowrank.md`](04-labels-name-rowrank.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's structure-manipulation API to match §R exactly. All runtime claims
> verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_05_structure.py` (all `[PASS]`, exit 0), with
> the raw-C++ side of the binding-fidelity findings verified by
> `probes/cpp/UniTensor_05_structure.cpp` against a source-built `libcytnx` (all
> `[PASS]`, exit 0).

**Category scope:** the methods that rearrange or coalesce a tensor's leg/block
structure — leg reordering (`permute`/`permute_`, `permute_nosignflip`/
`permute_nosignflip_`), shape change (`reshape`/`reshape_`), memory layout
(`contiguous`/`contiguous_`, plus the leaked raw `make_contiguous` shim), basis
grouping (`group_basis`/`group_basis_`), bond combining (`combineBonds`),
diagonal→dense (`to_dense`/`to_dense_`), bond truncation (`truncate`/`truncate_`,
plus the leaked raw `ctruncate_`), tagging (`tag`, plus the leaked raw `ctag`),
and the fermionic sign-bookkeeping ops (`twist`/`twist_`, `fermion_twists`/
`fermion_twists_`, `apply`/`apply_`). Python bindings:
`cytnx_src/pybind/unitensor_py.cpp:325-346,517-594,734-754,1414-1436`; conti.py
wrappers: `cytnx/UniTensor_conti.py:120-175`; C++ header:
`cytnx_src/include/UniTensor.hpp:3600-3886,4613-4709,5422-5590`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place. Overloads that differ
only in `mapper`/selector type (`Sequence[int]` vs `Sequence[str]`, `idx` vs
`label`) share one row.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `permute` | `permute(mapper, rowrank=-1)` (`mapper: list[int] \| list[str]`) | `UniTensor` (new, **shared data**) | **Pure** leg reorder; returns a *distinct* object that **shares storage** with the receiver (a view). Probe: *"permute returns a distinct object (pure) …"* + *"permute returns a shared-data VIEW …"*. |
| `permute_` `[I]` | `permute_(mapper, rowrank=-1)` | `UniTensor` (self) | **In-place** reorder; returns self (chainable). Probe: *"permute_ permutes in place and returns self"*. |
| `permute_nosignflip` | `permute_nosignflip(mapper, rowrank=-1)` | `UniTensor` (new) | Pure reorder **without** fermionic sign flips; **fermionic-only** (errors on bosonic). Probe: *"permute_nosignflip (no underscore) is pure …"*. |
| `permute_nosignflip_` `[I]` | `permute_nosignflip_(mapper, rowrank=-1)` | `UniTensor` (self) | In-place no-signflip reorder; returns self. **Fermionic-only** — bosonic errors *"can only be called on a BlockFermionicUniTensor"*. Probe: *"permute_nosignflip_ is fermionic-only …"* + *"… permutes a fermionic tensor in place, returns self"*. |
| `reshape` | `reshape(*args, **kwargs)` | `UniTensor` (new, **shared data**) | **Pure** shape change; returns a shared-data view. Bound as `(*args, **kwargs)` — the positional `(new_shape, rowrank)` signature is erased. Probe: *"reshape returns a shared-data VIEW …"* + *"reshape is bound as a (*args, **kwargs) pybind lambda …"*. |
| `reshape_` `[I]` | `reshape_(*args, **kwargs)` | `UniTensor` (self) | **In-place** shape change; returns self. Same `(*args, **kwargs)` erasure. Probe: *"reshape_ reshapes in place and returns self"*. |
| `contiguous` | `contiguous()` | `UniTensor` (self **or** new) | Coalesce storage. conti.py wrapper: **returns self** if already contiguous, else forwards to raw `make_contiguous` (a *distinct* contiguous copy). Probe: *"contiguous() short-circuits to self …"* + *"contiguous() on a non-contiguous tensor returns a DISTINCT, contiguous object …"*. |
| `contiguous_` `[I]` | `contiguous_()` | `UniTensor` (self) | **In-place** coalesce; returns self. Probe: *"contiguous_ coalesces storage in place and returns self"*. |
| `group_basis` | `group_basis()` | `UniTensor` (new) | Pure basis grouping; distinct object. Probe: *"group_basis (no underscore) is pure …"*. |
| `group_basis_` `[I]` | `group_basis_()` | `UniTensor` (self) | In-place basis grouping; returns self (no-op warning on Dense). Probe: *"group_basis_ groups basis in place and returns self"*. |
| `combineBonds` | `combineBonds(indicators, force=False, by_label=False)` (int) · `combineBonds(indicators, force=False)` (str) | **`None`** (in-place) | **Deprecated** camelCase; **in-place** mutator whose binding returns `None` (C++ `[[deprecated]]` → use `combineBond`). The current C++ `combineBond` (singular) is **unbound**. Probe: *"combineBonds (camelCase, the deprecated plural spelling) IS bound, while … combineBond (singular) is ABSENT …"* + *"… binding returns None …"* + *"… deprecation notice …"*. |
| `to_dense` | `to_dense()` | `UniTensor` (new) | Pure diagonal→non-diagonal; distinct object. Probe: *"to_dense (no underscore) is pure …"*. |
| `to_dense_` `[I]` | `to_dense_()` | `UniTensor` (self) | In-place diagonal→non-diagonal; returns self. Probe: *"to_dense_ converts to non-diagonal form in place and returns self"*. |
| `truncate` | `truncate(bond_idx \| label, dim)` | `UniTensor` (new, independent) | **Pure** bond truncation; returns a *distinct, independent* object (data copied). Probe: *"truncate (no underscore) is pure …"*. |
| `truncate_` `[I]` | `truncate_(bond_idx, dim)` | `UniTensor` (self) | **In-place** truncation; returns self. conti.py wrapper over raw `ctruncate_`. Probe: *"truncate_ truncates a bond in place and returns self …"*. |
| `tag` `[I]` | `tag()` | `UniTensor` (self) | Tag legs (add bra/ket direction) **in place**; returns self. conti.py wrapper over raw `ctag`. Probe: *"tag() tags the tensor in place and returns self …"*. |
| `twist` | `twist(idx \| label)` | `UniTensor` (new, independent) | Pure fermionic twist on one leg; independent copy. Probe: *"twist (no underscore) is pure: returns an independent copy …"*. |
| `twist_` `[I]` | `twist_(idx \| label)` | `UniTensor` (shared-data, **not self**) | In-place twist. **Binding fidelity bug:** the lambda returns `self.twist_(i)` **by value**, so the returned handle shares data but is **not** the same Python object (C++ returns `UniTensor&`). Probe: *"twist_'s binding returns a shared-data wrapper … but NOT the same Python object …"*. |
| `fermion_twists` | `fermion_twists()` | `UniTensor` (new) | Pure: twist all right-side BD_KET bonds; distinct object. Probe: *"fermion_twists (no underscore) is pure …"*. |
| `fermion_twists_` `[I]` | `fermion_twists_()` | `UniTensor` (self) | In-place; returns self. Probe: *"fermion_twists_ acts in place … and returns self"*. |
| `apply` | `apply()` | `UniTensor` (new) | Pure: materialize pending fermionic signflips; distinct object. Probe: *"apply (no underscore) is pure …"*. |
| `apply_` `[I]` | `apply_()` | `UniTensor` (self) | In-place; returns self. Probe: *"apply_ applies fermionic signflips in place and returns self"*. |

**Internal / plumbing (leak into `dir(UniTensor)`):** `make_contiguous` (the raw
C++ `contiguous()` the conti.py `contiguous` wrapper calls), `ctag` (raw `tag`),
`ctruncate_` (raw `truncate_`) — plumbing that should never be public. Probe:
*"the raw plumbing bindings make_contiguous / ctag / ctruncate_ all LEAK into
public dir(UniTensor)"*.

## A2. C++ ↔ Python mapping

| C++ (`UniTensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `UniTensor permute(...) const` (`:3600`) | `permute(...)` | identical | pure, shared-data view (UT-S1) |
| `UniTensor &permute_(...)` (`:3629,3640`) | `permute_(...)` | identical | binding returns `&self` → self (UT-S2) |
| `UniTensor permute_nosignflip(...) const` (`:3658`) | `permute_nosignflip(...)` | identical | fermionic-only pure variant (UT-S8) |
| `UniTensor &permute_nosignflip_(...)` (`:3710,3727`) | `permute_nosignflip_(...)` | identical | fermionic-only; self (UT-S8) |
| `UniTensor reshape(new_shape, rowrank=0)` (`:4613`) | `reshape(*args, **kwargs)` | **signature-differs** | positional signature erased by the `(*args,**kwargs)` lambda (UT-S3) |
| `UniTensor &reshape_(new_shape, rowrank=0)` (`:4625`) | `reshape_(*args, **kwargs)` | **signature-differs** | same erasure; returns self (UT-S3) |
| `UniTensor contiguous() const` (`:3813`) | `make_contiguous` **+** conti.py `contiguous` | **leak** | raw C++ `contiguous()` bound as `make_contiguous`; public `contiguous` is a conti.py short-circuit wrapper (UT-S4) |
| `UniTensor &contiguous_()` (`:3823`) | `contiguous_()` | identical | in-place, self (UT-S2) |
| `UniTensor group_basis(...) const` (`:3886`) | `group_basis()` | identical | pure (UT-S8) |
| `UniTensor &group_basis_()` (`:3881`) | `group_basis_()` | identical | in-place, self (UT-S8) |
| `UniTensor &combineBond(indicators, force)` (`:4706`) | *(unbound)* | **C++-only** | current singular form; **not** exposed in Python (UT-S5) |
| `[[deprecated]] void combineBonds(...)` (`:4661,4675,4691`) | `combineBonds(...)` | signature-differs | deprecated plural bound instead; returns `None` (UT-S5) |
| `UniTensor to_dense()` (`:4641`) | `to_dense()` | identical | pure (UT-S8) |
| `UniTensor &to_dense_()` (`:4651`) | `to_dense_()` | identical | in-place, self (UT-S8) |
| `UniTensor truncate(...) const` (`:5565,5580`) | `truncate(...)` | identical | pure, independent copy (UT-S8) |
| `UniTensor &truncate_(...)` (`:5538,5551`) | `ctruncate_` **+** conti.py `truncate_` | **leak** | raw `truncate_` bound as `ctruncate_`; public `truncate_` wraps it (UT-S6) |
| `UniTensor &tag()` (`:5422`) | `ctag` **+** conti.py `tag` | **leak** | raw `tag` bound as `ctag`; public `tag` wraps it (UT-S6) |
| `UniTensor twist(...) const` (`:3741,3754`) | `twist(...)` | identical | pure, independent copy (UT-S8) |
| `UniTensor &twist_(...)` (`:3764,3773`) | `twist_(...)` | **signature-differs** | lambda returns **by value**, dropping C++'s `&self` self-return (UT-S7) |
| `UniTensor fermion_twists() const` (`:3792`) | `fermion_twists()` | identical | pure (UT-S8) |
| `UniTensor &fermion_twists_()` (`:3802`) | `fermion_twists_()` | identical | in-place, self (UT-S8) |
| `UniTensor apply() const` (`:3836`) | `apply()` | identical | pure (UT-S8) |
| `UniTensor &apply_()` (`:3847`) | `apply_()` | identical | in-place, self (UT-S8) |
| raw `make_contiguous`/`ctag`/`ctruncate_` | same names | **leak** | plumbing exposed publicly (UT-S9) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/UniTensor_05_structure.py` (on the 1.1.0 wheel). A
**(binding fidelity)** finding flags where the binding layer — a `*_conti.py`
wrapper or a pybind lambda — changes behavior or signature versus the raw C++
method; **both sides are runtime-verified**, the raw-C++ side by
`probes/cpp/UniTensor_05_structure.cpp` (links against a source-built
`libcytnx`, GCC 13). Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-S1** | `permute` and `reshape` return a **distinct object that shares storage** with the receiver — a **view**, not a copy | copy/view (B2) | **thin pass-through** — `permute` forwards to C++ `permute(...) const` (`hpp:3600`), which returns a new UniTensor over the same block storage; `reshape` likewise. Py probe *"permute returns a shared-data VIEW …"* + *"reshape returns a shared-data VIEW …"*; **C++ probe confirms** mutating the source shows through the permute | **keep**; **document the shared-data view semantics explicitly** (they are silent today) |
| **UT-S2** | `permute_`/`reshape_`/`contiguous_` are correct in-place methods returning **self** | (kept) | **faithful** — the pybind lambdas return `&self.permute_(...)` (`:531-536`) / the C++ `contiguous_` method pointer (`:578`), preserving C++'s `UniTensor&` (`hpp:3629,3823,4625`). Py probe *"permute_ … returns self"* + *"contiguous_ … returns self"*; **C++ probe confirms** `&U.permute_(...)==&U`, `&U.contiguous_()==&U` | **keep** |
| **UT-S3** | `reshape`/`reshape_` are bound as `(*args, **kwargs)`, **erasing** the C++ `(new_shape, rowrank)` positional signature | **binding fidelity** / signature-differs | **binding erases the signature**: the pybind lambda (`:325-346`) takes `py::args`/`py::kwargs` and hand-parses `rowrank` from kwargs, so the docstring reads `reshape(*args, **kwargs)` and `inspect.signature()` raises. Py probe *"reshape is bound as a (*args, **kwargs) pybind lambda …"* + *"… inspect.signature() raises ValueError"*; **C++ probe confirms** C++ `reshape_(new_shape, rowrank)` has a real typed signature and returns `&*this` | **bind with an explicit `(new_shape: Sequence[int], rowrank=0)` signature** (typed `py::arg`s), restoring introspection |
| **UT-S4** | `contiguous` binds via the raw **`make_contiguous`** shim, which **leaks** into `dir`; the public `contiguous` is a conti.py wrapper that short-circuits to `self` when already contiguous | **binding fidelity** / leak + naming | **binding exposes plumbing + wraps it**: the raw C++ `contiguous()` is bound as `make_contiguous` (`:577`), and `cytnx/UniTensor_conti.py:120-126` defines `contiguous` as `if is_contiguous(): return self else: return self.make_contiguous()`. Py probe *"the raw make_contiguous shim … leaks …"* + *"contiguous() short-circuits to self …"* + *"… returns a DISTINCT, contiguous object …"*; **C++ probe confirms** C++ `contiguous()` returns a distinct object while `contiguous_()` returns `&*this` | **remove `make_contiguous` from the public API** (inline into the `contiguous` pybind lambda, which should itself do the short-circuit); keep `contiguous`/`contiguous_` |
| **UT-S5** | `combineBonds` is **camelCase** *and* the **deprecated** spelling — the current C++ `combineBond` (singular) is **unbound in Python**; the bound plural returns `None` | naming + **C++-only binding gap** | **binding exposes the deprecated method only**: only `combineBonds` (plural) is `.def`-ed (`:736,749`), returning `void`; the C++ header marks every `combineBonds` overload `[[deprecated]]` "*Please use combineBond*" (`hpp:4661,4675,4691`) while the current `UniTensor &combineBond(...)` (`hpp:4706`) is never bound. Py probe *"combineBonds … IS bound, while … combineBond (singular) is ABSENT …"* + *"… binding returns None …"* + *"… deprecation notice …"*; **C++ probe confirms** `U.combineBond(...)` exists, combines in place and returns `&*this`, while `combineBonds` is `[[deprecated]]`/void | **bind `combine_bonds`** — snake_case of the current singular `combineBond`, returning self; **deprecate `combineBonds`** (migration note) |
| **UT-S6** | `tag` and `truncate_` are conti.py wrappers over the raw **`ctag`** / **`ctruncate_`** bindings, which **leak** into `dir` | naming + **binding fidelity** | **binding exposes plumbing + wraps it**: raw C++ `tag()` is bound as `ctag` (`:1414`) and `truncate_()` as `ctruncate_` (`:1426,1433`); `cytnx/UniTensor_conti.py:158-175` defines `tag`/`truncate_` as `self.ctag(); return self` / `self.ctruncate_(...); return self`. The `c`-prefix is a reserved raw-binding spelling (§R.0 rejects it). Py probe *"the raw ctag binding … leaks …"* + *"… ctruncate_ … leaks …"* + *"tag() … returns self …"* + *"truncate_ … returns self …"*; **C++ probe confirms** C++ `tag()`/`truncate_()` return `&*this` | **remove `ctag`/`ctruncate_` from the public API** (private `_`-name or inline into the pybind lambda that returns self); keep `tag`/`truncate_` |
| **UT-S7** | `twist_`'s binding returns a shared-data wrapper that is **not the same Python object** — C++'s in-place `UniTensor&` self-return is dropped | **binding fidelity** (N2/B1) | **binding returns by value**: the pybind lambda `[](UniTensor& self, …){ return self.twist_(idx); }` (`:568,571`) returns the `UniTensor&` **by value**, so pybind wraps a *new* Python object sharing `_impl` — unlike `permute_` (`:531`, returns `&self`). C++ `twist_` returns `UniTensor&` (`hpp:3764,3773`). Py probe *"twist_'s binding returns a shared-data wrapper … but NOT the same Python object …"* | **return self directly** — bind `twist_` to `return &self.twist_(i)` so identity is preserved (matches `permute_`) |
| **UT-S8** | The remaining pairs are correct N-underscore pairs (pure vs in-place-self): `permute_nosignflip`/`_` (fermionic-only), `group_basis`/`_`, `to_dense`/`to_dense_`, `truncate`/`truncate_`, `fermion_twists`/`_`, `apply`/`apply_`, `twist`/`twist_` | (kept) | **faithful pass-throughs** — each `_` form returns `UniTensor&`/self, each un-suffixed form is pure. `permute_nosignflip_` errors on a bosonic tensor (fermionic-only, `UniTensor_base.cpp:187`). Py probe *"permute_nosignflip_ is fermionic-only …"*, *"… to_dense_ … returns self"*, *"group_basis_ … returns self"*, *"apply_ … returns self"*, *"fermion_twists_ … returns self"*, *"truncate (no underscore) is pure …"* | **keep all** — the pairs are conformant (except the plumbing/signature notes above) |
| **UT-S9** | the raw `make_contiguous`/`ctag`/`ctruncate_` bindings **leak** into `dir(UniTensor)` | naming + **binding fidelity** | **binding exposes plumbing**: all three are `.def`-ed as public methods (`:577,1414,1426`) purely so the conti.py wrappers can call them; the `c`-prefix/`make_` shim spelling must not be public (§R.0). Py probe *"the raw plumbing bindings make_contiguous / ctag / ctruncate_ all LEAK …"* | **remove from public API** — bind under a private name (leading `_`) or merge into the pybind lambda (migration note) |

## A4. Argument ordering — positional & keyword

Every member here takes at most a target/mapper primary operand plus a small
number of operation parameters; there is no keyword-only metadata block.

| API | positional-required (in order) |
|---|---|
| `permute` / `permute_` / `permute_nosignflip` / `permute_nosignflip_` | `mapper` (then optional `rowrank`) |
| `reshape` / `reshape_` | `new_shape` (then optional `rowrank`) |
| `contiguous` / `contiguous_` / `group_basis` / `group_basis_` / `to_dense` / `to_dense_` / `apply` / `apply_` / `fermion_twists` / `fermion_twists_` | *(none)* |
| `combineBonds` → `combine_bonds` | `indicators` (then optional `force`) |
| `truncate` / `truncate_` | `bond_idx` **or** `label`, then `dim` |
| `tag` | *(none)* |
| `twist` / `twist_` | `idx` **or** `label` |

- **Canonical positional rule (§R.0):** the selector/mapper primary operand
  first, then operation parameters (`rowrank`, `force`, `dim`). This matches the
  live order and needs no change — except that `reshape`'s `rowrank` and
  `combineBonds`'s `force` must become **real typed parameters** (not `**kwargs`,
  UT-S3) and, for combining, keyword-friendly.
- **`rowrank` is an operation parameter, not metadata:** on `permute`/`reshape`
  it is a positional-optional following the mapper (as today), not part of the
  keyword-only metadata block used by the generators (cat 02). Keep it positional
  with a `-1`/`0` default.
- **Deprecated `by_label`:** `combineBonds(indicators, force, by_label)` carries a
  deprecated `by_label` flag (string indicators are auto-recognized as labels);
  the replacement `combine_bonds` drops it (UT-S5).

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version structure-manipulation API.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53).** Members are lowercase snake_case. The one
  offender is the camelCase **`combineBonds` → `combine_bonds`** (UT-S5). All
  other members are already conformant.
- **N-underscore — a trailing `_` marks in-place (returns `self`); its absence
  marks pure (returns a new object).** Every operation here that is meaningful in
  both modes provides both forms under one base name — `permute`/`permute_`,
  `reshape`/`reshape_`, `contiguous`/`contiguous_`, `group_basis`/`group_basis_`,
  `to_dense`/`to_dense_`, `truncate`/`truncate_`, `twist`/`twist_`,
  `fermion_twists`/`fermion_twists_`, `apply`/`apply_`,
  `permute_nosignflip`/`permute_nosignflip_`. The **`c`-prefixed raw spellings
  (`ctag`, `ctruncate_`) and the `make_contiguous` shim are rejected** as public
  API — they are the plumbing the wrappers call (UT-S4/S6/S9). A public in-place
  method is *only* the trailing-`_` form.
- **In-place methods return `self` from the binding directly.** `permute_`,
  `reshape_`, `contiguous_`, `truncate_`, `tag`, `twist_`, `combine_bonds` return
  self in C++ (`UniTensor&`); the pybind lambda must return `&self` too, so the
  conti.py return-self shims (and the leaked `c_*`/`make_contiguous` bindings they
  wrap) disappear (UT-S3/S4/S6/S7). In particular **`twist_` must return `&self`**,
  not by value (UT-S7).
- **Copy-vs-view is documented, not silent.** `permute` and `reshape` return a
  **shared-data view** (metadata-only copy over the same storage); `truncate` and
  `twist` return **independent copies**. Each pure method's docstring states which
  (UT-S1) — see R.2.
- **Real signatures, not `*args`.** `reshape`/`reshape_` must bind with explicit
  typed `py::arg`s (`new_shape`, `rowrank`), not `(*args, **kwargs)` (UT-S3).

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- leg reordering (pure = shared-data view, in-place = self) ---
    def permute(self, mapper: Sequence[int] | Sequence[str], rowrank: int = -1) -> "UniTensor": ...
    def permute_(self, mapper: Sequence[int] | Sequence[str], rowrank: int = -1) -> "UniTensor": ...  # self
    def permute_nosignflip(self, mapper, rowrank: int = -1) -> "UniTensor": ...   # fermionic-only
    def permute_nosignflip_(self, mapper, rowrank: int = -1) -> "UniTensor": ...  # fermionic-only, self

    # --- shape change (typed signature, not *args) ---
    def reshape(self, new_shape: Sequence[int], rowrank: int = 0) -> "UniTensor": ...
    def reshape_(self, new_shape: Sequence[int], rowrank: int = 0) -> "UniTensor": ...  # self

    # --- memory layout ---
    def contiguous(self) -> "UniTensor": ...    # self if already contiguous, else a contiguous copy
    def contiguous_(self) -> "UniTensor": ...   # in-place, self

    # --- basis grouping / diagonal->dense ---
    def group_basis(self) -> "UniTensor": ...
    def group_basis_(self) -> "UniTensor": ...  # self
    def to_dense(self) -> "UniTensor": ...
    def to_dense_(self) -> "UniTensor": ...      # self

    # --- bond combining (renamed from combineBonds) ---
    def combine_bonds(self, indicators: Sequence[int] | Sequence[str],
                      force: bool = False) -> "UniTensor": ...   # in-place, self

    # --- truncation / tagging ---
    def truncate(self, bond: int | str, dim: int) -> "UniTensor": ...    # pure, independent copy
    def truncate_(self, bond: int | str, dim: int) -> "UniTensor": ...   # in-place, self
    def tag(self) -> "UniTensor": ...                                    # in-place, self

    # --- fermionic sign bookkeeping ---
    def twist(self, bond: int | str) -> "UniTensor": ...    # pure, independent copy
    def twist_(self, bond: int | str) -> "UniTensor": ...   # in-place, self (bind &self!)
    def fermion_twists(self) -> "UniTensor": ...
    def fermion_twists_(self) -> "UniTensor": ...           # self
    def apply(self) -> "UniTensor": ...
    def apply_(self) -> "UniTensor": ...                    # self
```

In-place methods return `self` **from the binding** (no conti.py shim); the raw
`make_contiguous`/`ctag`/`ctruncate_` plumbing bindings become private (leading
`_`) or are inlined into the pybind lambdas — they are **not** public members.

| API | Verdict | Behavior contract |
|---|---|---|
| `permute` | **keep** (UT-S1; document shared-data view) | Pure leg reorder; returns a new UniTensor sharing storage with the receiver (a view). |
| `permute_` | **keep** (UT-S2) | In-place leg reorder; returns self (chainable). |
| `permute_nosignflip` | **keep** (UT-S8) | Pure reorder without fermionic sign flips; fermionic-only. |
| `permute_nosignflip_` | **keep** (UT-S8) | In-place no-signflip reorder; returns self; fermionic-only (errors on bosonic). |
| `reshape` | **keep, but bind a typed signature** (UT-S3) | Pure shape change; returns a new UniTensor sharing storage (a view). *Migration:* replace the `(*args,**kwargs)` lambda with `reshape(new_shape, rowrank=0)`; the call surface is unchanged, only introspection is restored. |
| `reshape_` | **keep, but bind a typed signature** (UT-S3) | In-place shape change; returns self. Same typed-signature migration. |
| `contiguous` | **keep** (UT-S4; short-circuit in the lambda) | Returns self if already contiguous, else a new contiguous copy. |
| `contiguous_` | **keep** (UT-S2) | In-place coalesce; returns self. |
| `group_basis` | **keep** (UT-S8) | Pure basis grouping; new object. |
| `group_basis_` | **keep** (UT-S8) | In-place basis grouping; returns self (no-op on Dense). |
| `combineBonds` → `combine_bonds` | **rename** (UT-S5; camelCase + deprecated) | In-place bond combine; returns self. *Migration:* bind `combine_bonds` (snake_case of the current C++ singular `combineBond`, returning `UniTensor&`); keep `combineBonds` as a `DeprecationWarning` alias for one minor release, then delete. The current C++ `combineBond` (singular, previously unbound) is the implementation. |
| `to_dense` | **keep** (UT-S8) | Pure diagonal→non-diagonal; new object. |
| `to_dense_` | **keep** (UT-S8) | In-place diagonal→non-diagonal; returns self. |
| `truncate` | **keep** (UT-S8) | Pure bond truncation; returns a new, independent object. |
| `truncate_` | **keep** (UT-S6; bind self directly) | In-place truncation; returns self. *Migration:* the conti.py wrapper over `ctruncate_` is removed; the pybind lambda returns self. |
| `tag` | **keep** (UT-S6; bind self directly) | Tag legs (add bra/ket directions) in place; returns self. *Migration:* the conti.py wrapper over `ctag` is removed; the pybind lambda returns self. |
| `twist` | **keep** (UT-S8) | Pure fermionic twist on one leg; independent copy. |
| `twist_` | **keep, but return self** (UT-S7) | In-place twist; returns self. *Migration:* bind `return &self.twist_(i)` (currently returns by value, losing identity), matching `permute_`. |
| `fermion_twists` | **keep** (UT-S8) | Pure; twist all right-side BD_KET bonds; new object. |
| `fermion_twists_` | **keep** (UT-S8) | In-place; returns self. |
| `apply` | **keep** (UT-S8) | Pure; materialize pending fermionic signflips; new object. |
| `apply_` | **keep** (UT-S8) | In-place; returns self. |

**Internal / plumbing — hidden, not public API.** The three raw bindings below
are covered here (they are live public members today) with a **remove** verdict:
hide them behind a leading underscore or inline them into their pybind lambda.
None carry a docstring — they are not public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `make_contiguous` | **remove** (UT-S4/S9) | Raw plumbing (the C++ `contiguous()`) that the conti.py `contiguous` wrapper calls. *Migration:* inline into the `contiguous` pybind lambda (which does the already-contiguous short-circuit); no public exposure. |
| `ctag` | **remove** (UT-S6/S9) | Raw plumbing (the C++ `tag()`) that the conti.py `tag` wrapper calls. *Migration:* fold into the `tag` pybind lambda (which returns self); no public exposure. |
| `ctruncate_` | **remove** (UT-S6/S9) | Raw plumbing (the C++ `truncate_()`) that the conti.py `truncate_` wrapper calls. *Migration:* fold into the `truncate_` pybind lambda (which returns self); no public exposure. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept/renamed members are
documented (removed plumbing members carry no docstring).

### R.2a Python API (numpy-style)

### `permute` / `permute_`

```
UniTensor.permute(mapper, rowrank=-1)   -> UniTensor   # pure, shared-data view
UniTensor.permute_(mapper, rowrank=-1)  -> UniTensor   # in-place, self

Reorder the legs of this UniTensor.

`permute` is PURE: it returns a new UniTensor with the legs in the requested
order while leaving this tensor unchanged. The returned tensor SHARES its
internal storage with this one (a metadata-only view) — mutating either
tensor's elements is visible through the other (finding UT-S1).

`permute_` is the IN-PLACE form: it reorders this tensor's legs and returns self
for chaining (finding UT-S2).

Parameters
----------
mapper : sequence of int or sequence of str
    The new leg order, by index or by label.
rowrank : int, optional
    Row (bra) space leg count after the permutation; -1 keeps the current split.

Returns
-------
UniTensor
    `permute`: a new tensor (shared data). `permute_`: self.

See Also
--------
permute_nosignflip : reorder without fermionic sign flips (fermionic only).
```

### `permute_nosignflip` / `permute_nosignflip_`

```
UniTensor.permute_nosignflip(mapper, rowrank=-1)   -> UniTensor   # pure
UniTensor.permute_nosignflip_(mapper, rowrank=-1)  -> UniTensor   # in-place, self

Reorder the legs of a FERMIONIC UniTensor WITHOUT applying the fermionic sign
flips a normal permute would. Fermionic-only — calling on a bosonic tensor
raises (finding UT-S8). Use with care: this is normally not what you want, since
fermionic permutations create sign flips; it exists to compare tensors in
different sign conventions.

Parameters and Returns as for `permute` / `permute_`.
```

### `reshape` / `reshape_`

```
UniTensor.reshape(new_shape, rowrank=0)   -> UniTensor   # pure, shared-data view
UniTensor.reshape_(new_shape, rowrank=0)  -> UniTensor   # in-place, self

Change the shape of this UniTensor (total number of elements preserved).

`reshape` is PURE and returns a new UniTensor SHARING storage with this one (a
view, finding UT-S1); `reshape_` reshapes IN PLACE and returns self.

Parameters
----------
new_shape : sequence of int
    The target shape.
rowrank : int, optional
    Row (bra) space leg count after the reshape (default 0).

Returns
-------
UniTensor
    `reshape`: a new tensor (shared data). `reshape_`: self.

Notes
-----
Through cytnx 1.1.0 these were bound as ``reshape(*args, **kwargs)``, erasing
the signature (finding UT-S3); the next version binds the typed signature above.
```

### `contiguous` / `contiguous_`

```
UniTensor.contiguous()   -> UniTensor   # self if already contiguous, else a copy
UniTensor.contiguous_()  -> UniTensor   # in-place, self

Coalesce the tensor's storage into a contiguous memory layout.

`contiguous` returns self unchanged when the tensor is already contiguous,
otherwise a NEW contiguous UniTensor (finding UT-S4). `contiguous_` coalesces
IN PLACE and returns self.

Returns
-------
UniTensor
    `contiguous`: self (already contiguous) or a new contiguous tensor.
    `contiguous_`: self.

Notes
-----
The former raw `make_contiguous` binding is removed (folded into `contiguous`,
finding UT-S4/S9).
```

### `combine_bonds`

```
UniTensor.combine_bonds(indicators, force=False)  -> UniTensor   # in-place, self

Combine several bonds of this UniTensor into one, IN PLACE; returns self.

Parameters
----------
indicators : sequence of int or sequence of str
    Indices or labels of the legs to combine (need at least two). String
    indicators are recognized as labels automatically.
force : bool, optional
    Combine even when the bond directions differ; the combined bond takes the
    direction of the first (default False).

Returns
-------
UniTensor
    self.

Notes
-----
Renamed from `combineBonds` (camelCase, deprecated) — the current C++ form is
the singular `combineBond` (finding UT-S5). Preconditions: at least two
indicators, and the tensor must not be diagonal (`is_diag == False`).
`combineBonds` remains a `DeprecationWarning` alias for one release.
```

### `truncate` / `truncate_`

```
UniTensor.truncate(bond, dim)   -> UniTensor   # pure, independent copy
UniTensor.truncate_(bond, dim)  -> UniTensor   # in-place, self

Truncate one bond of this UniTensor to dimension `dim`.

`truncate` is PURE and returns a new, INDEPENDENT UniTensor (data copied);
`truncate_` truncates IN PLACE and returns self (finding UT-S8/S6).

Parameters
----------
bond : int or str
    The bond to truncate, by index or label.
dim : int
    The new (smaller) bond dimension.

Returns
-------
UniTensor
    `truncate`: a new, independent tensor. `truncate_`: self.
```

### `tag`

```
UniTensor.tag()  -> UniTensor   # in-place, self

Tag the legs of an untagged (regular) UniTensor, assigning bra/ket directions,
IN PLACE; returns self (finding UT-S6).

Returns
-------
UniTensor
    self.

Notes
-----
The former raw `ctag` binding is removed (folded into `tag`, finding UT-S6/S9).
```

### `twist` / `twist_`

```
UniTensor.twist(bond)   -> UniTensor   # pure, independent copy
UniTensor.twist_(bond)  -> UniTensor   # in-place, self

Apply a fermionic twist to one leg of this UniTensor.

`twist` is PURE and returns a new, independent tensor; `twist_` twists IN PLACE
and returns self (finding UT-S7 — 1.1.0's `twist_` returned a shared-data copy,
not self; the next version returns self).

Parameters
----------
bond : int or str
    The leg to twist, by index or label.

Returns
-------
UniTensor
    `twist`: a new, independent tensor. `twist_`: self.
```

### `fermion_twists` / `fermion_twists_` / `apply` / `apply_` / `group_basis` / `group_basis_` / `to_dense` / `to_dense_`

```
UniTensor.fermion_twists()   -> UniTensor      # pure
UniTensor.fermion_twists_()  -> UniTensor      # in-place, self
UniTensor.apply()            -> UniTensor      # pure
UniTensor.apply_()           -> UniTensor      # in-place, self
UniTensor.group_basis()      -> UniTensor      # pure
UniTensor.group_basis_()     -> UniTensor      # in-place, self
UniTensor.to_dense()         -> UniTensor      # pure
UniTensor.to_dense_()        -> UniTensor      # in-place, self

Structure/bookkeeping operations, each with a pure form (new object) and an
in-place form returning self (finding UT-S8):

fermion_twists : twist all right-side (>= rowrank) bonds of type BD_KET, so that
    bra/ket fermionic states contract correctly (e.g. contract(Adag.fermion_twists_(), B)).
apply          : materialize any pending fermionic signflips into the stored
    blocks; afterwards signflip() is False for all elements.
group_basis    : group the basis of a symmetric tensor (no-op on Dense).
to_dense       : convert a diagonal (is_diag) tensor to full non-diagonal form.

Returns
-------
UniTensor
    Pure form: a new tensor. In-place (`_`) form: self.
```

### R.2b C++ API (Doxygen)

C++ already returns `UniTensor&`/`UniTensor` per the N-underscore split; the
next version must have the *pybind lambdas* return these directly (removing the
conti.py shims and the leaked `make_contiguous`/`ctag`/`ctruncate_` bindings,
UT-S4/S6/S9) and bind `reshape` with a typed signature (UT-S3), `twist_` with
`&self` (UT-S7), and the singular `combineBond` as `combine_bonds` (UT-S5).

```cpp
/**
 * @brief Reorder the legs, returning a NEW UniTensor (data shared).
 * @details Pure: the returned tensor has the legs in @p mapper order while
 *          *this is unchanged, but the internal storage is SHARED (a view,
 *          finding UT-S1). permute_ reorders in place and returns *this.
 * @param mapper  new leg order, by index or by label.
 * @param rowrank row (bra) space leg count after the permutation (-1 = keep).
 * @return permute: a new UniTensor (shared data). permute_: reference to *this.
 */
UniTensor permute(const std::vector<cytnx_int64> &mapper,
                  const cytnx_int64 &rowrank = -1) const;
UniTensor permute(const std::vector<std::string> &mapper,
                  const cytnx_int64 &rowrank = -1) const;
UniTensor &permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1);
UniTensor &permute_(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1);

/**
 * @brief Reorder the legs WITHOUT fermionic sign flips (fermionic tensors only).
 * @details Use with care — fermionic permutations normally create sign flips.
 *          Errors on a bosonic tensor (finding UT-S8). permute_nosignflip_ is
 *          the in-place form returning *this.
 * @param mapper,rowrank as for permute().
 * @return a new UniTensor (pure) / reference to *this (in-place).
 */
UniTensor permute_nosignflip(const std::vector<cytnx_int64> &mapper,
                             const cytnx_int64 &rowrank = -1) const;
UniTensor &permute_nosignflip_(const std::vector<cytnx_int64> &mapper,
                               const cytnx_int64 &rowrank = -1);

/**
 * @brief Change the shape (element count preserved), returning a NEW UniTensor.
 * @details Pure: the returned tensor SHARES storage with *this (a view, finding
 *          UT-S1). reshape_ reshapes in place and returns *this. The Python
 *          binding must expose the typed signature below, not (*args,**kwargs)
 *          (finding UT-S3).
 * @param new_shape target shape.
 * @param rowrank   row (bra) space leg count after the reshape.
 * @return reshape: a new UniTensor (shared data). reshape_: reference to *this.
 */
UniTensor reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0);
UniTensor &reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0);

/**
 * @brief Coalesce storage into a contiguous layout.
 * @details contiguous() returns a distinct contiguous UniTensor (or *this when
 *          already contiguous); contiguous_() coalesces in place and returns
 *          *this. The Python binding folds the raw make_contiguous shim into the
 *          contiguous() lambda (finding UT-S4/S9).
 * @return contiguous: a contiguous UniTensor. contiguous_: reference to *this.
 */
UniTensor contiguous() const;
UniTensor &contiguous_();

/**
 * @brief Combine several bonds into one, IN PLACE; returns *this.
 * @details The current form is the singular combineBond (the plural
 *          combineBonds is [[deprecated]], finding UT-S5). Python binds this as
 *          combine_bonds (snake_case).
 * @param indicators labels/indices of the legs to combine (>= 2).
 * @param force combine even if bond directions differ (result takes the first).
 * @pre indicators.size() >= 2 and the tensor is not diagonal (is_diag == false).
 * @return reference to *this.
 */
UniTensor &combineBond(const std::vector<std::string> &indicators, const bool &force = false);

/**
 * @brief Convert a diagonal (is_diag) UniTensor to full non-diagonal form.
 * @details to_dense() is pure (new object); to_dense_() converts in place and
 *          returns *this (finding UT-S8).
 * @return to_dense: a new UniTensor. to_dense_: reference to *this.
 */
UniTensor to_dense();
UniTensor &to_dense_();

/**
 * @brief Group the basis of a symmetric UniTensor (no-op on Dense).
 * @details group_basis() is pure; group_basis_() groups in place and returns *this.
 * @return group_basis: a new UniTensor. group_basis_: reference to *this.
 */
UniTensor group_basis() const;
UniTensor &group_basis_();

/**
 * @brief Truncate one bond to dimension @p dim.
 * @details truncate() is pure and returns a new, INDEPENDENT UniTensor;
 *          truncate_() truncates in place and returns *this. The Python binding
 *          must fold the raw ctruncate_ shim into the truncate_ lambda
 *          (finding UT-S6/S9).
 * @param bond_idx / label  the bond to truncate.
 * @param dim               the new (smaller) bond dimension.
 * @return truncate: a new UniTensor. truncate_: reference to *this.
 */
UniTensor truncate(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim) const;
UniTensor &truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim);

/**
 * @brief Tag the legs of an untagged UniTensor (assign bra/ket), IN PLACE.
 * @details Returns *this. The Python binding must fold the raw ctag shim into
 *          the tag lambda (finding UT-S6/S9).
 * @return reference to *this.
 */
UniTensor &tag();

/**
 * @brief Fermionic sign-bookkeeping operations.
 * @details Each has a pure form (new UniTensor) and an in-place form returning
 *          *this (finding UT-S8). twist_ MUST have its Python binding return
 *          *this by reference, not by value (finding UT-S7).
 *          twist          : twist one leg.
 *          fermion_twists : twist all right-side (>= rowrank) BD_KET bonds.
 *          apply          : materialize pending signflips into the blocks.
 * @return pure form: a new UniTensor. in-place (_) form: reference to *this.
 */
UniTensor twist(const cytnx_int64 &idx) const;
UniTensor &twist_(const cytnx_int64 &idx);
UniTensor fermion_twists() const;
UniTensor &fermion_twists_();
UniTensor apply() const;
UniTensor &apply_();
```
