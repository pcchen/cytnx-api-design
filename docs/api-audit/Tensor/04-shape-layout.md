# Tensor — 04. Shape / layout

> **Superset-method rollout** (Tensor, category 04 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings, arg ordering) and a self-contained **Recommendation** that
> is the *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` shape/layout surface should be implemented to match §R exactly. Every
> behavioral claim is verified against the installed `cytnx==1.1.0` wheel by
> `docs/api-audit/probes/Tensor_04_shape.py` (all `[PASS]`, exit 0), with the
> raw-C++ side of the binding-fidelity findings verified by
> `docs/api-audit/probes/cpp/Tensor_04_shape.cpp` against a source-built
> `libcytnx` (GCC 13; all `[PASS]`, exit 0).

**Category scope:** the members that rearrange or coalesce a dense tensor's
axes/memory layout — leg reordering (`permute`/`permute_`), shape change
(`reshape`/`reshape_`), memory-layout coalescing (`contiguous`/`contiguous_`,
plus the leaked raw `make_contiguous` shim), and the 1-D collapse
(`flatten`/`flatten_`). Read-only *metadata* (`shape`/`rank`/`is_contiguous`/
`same_data`, the view-vs-copy oracle) is
[category 02](02-metadata-introspection.md); element/storage access
(`item`/`storage`/`numpy`/`__getitem__`) is
[category 03](03-element-storage-access.md); dtype/device conversion
(`astype`/`to`/`clone`) is [category 07](07-type-device-conversion.md). Python
bindings: `cytnx_src/pybind/tensor_py.cpp:179-202` (permute/reshape lambdas,
`make_contiguous`/`contiguous_`/`flatten`/`flatten_`); conti.py wrapper:
`cytnx/Tensor_conti.py:50-55` (`contiguous`); C++ header:
`cytnx_src/include/Tensor.hpp:691-873,1416-1433`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `permute` | `permute(*args)` | `Tensor` (new, **shared data**) | **Pure** axis reorder; returns a *distinct* object that **shares storage** with the receiver (a view). Bound as `(*args)` — the typed `(mapper)` signature is erased. Probe: *"permute returns a DISTINCT object …"* + *"permute returns a shared-data VIEW …"* + *"permute VIEW is live …"*. |
| `permute_` `[I]` | `permute_(*args)` | `Tensor` (self) | **In-place** reorder; returns self (chainable). Same `(*args)` erasure. Probe: *"permute_ permutes in place and returns SELF (ret is a)"*. |
| `reshape` | `reshape(*args)` | `Tensor` (new, **shared data**) | **Pure** shape change; returns a shared-data view. `(*args)` erasure. Probe: *"reshape returns a shared-data VIEW …"* + *"reshape VIEW is live …"*. |
| `reshape_` `[I]` | `reshape_(*args)` | `Tensor` (self) | **In-place** shape change; returns self. `(*args)` erasure. Probe: *"reshape_ reshapes in place and returns SELF (ret is b)"*. |
| `contiguous` | `contiguous()` | `Tensor` (self **or** new copy) | Coalesce storage. conti.py wrapper: **returns self** if already contiguous, else forwards to raw `make_contiguous` (a *distinct, independent* contiguous copy). Probe: *"contiguous() short-circuits to SELF …"* + *"contiguous() on a non-contiguous tensor returns … an INDEPENDENT copy …"*. |
| `contiguous_` `[I]` | `contiguous_()` | `Tensor` (**shared-data, NOT self**) | **In-place** coalesce — the receiver's storage is coalesced — but the binding returns a *distinct* shared-data wrapper, **not** self. Root cause is the C++ signature `Tensor contiguous_()` (returns `*this` **by value**, not `Tensor&`). Probe: *"contiguous_ coalesces the RECEIVER's storage in place …"* + *"contiguous_ returns a DISTINCT object, NOT self …"*. |
| `flatten` | `flatten()` | `Tensor` (new, **independent copy**) | **Pure** 1-D collapse (clone + contiguous + reshape); returns an *independent* rank-1 copy. Probe: *"flatten returns a DISTINCT object collapsed to rank 1 …"* + *"flatten returns an INDEPENDENT COPY …"*. |
| `flatten_` `[I]` | `flatten_()` | **`None`** (in-place) | **In-place** 1-D collapse; the binding returns `None`, **not** self (v1 Tensor C4). Root cause is the C++ signature `void flatten_()`. Probe: *"flatten_ collapses the RECEIVER to rank 1 in place …"* + *"flatten_ returns None, NOT self …"*. |

**Internal / plumbing (leaks into `dir(Tensor)`):** `make_contiguous` — the raw
C++ `contiguous()` the conti.py `contiguous` wrapper calls (bound at
`tensor_py.cpp:192`, comment *"this will be rename by python side conti"*). It
never short-circuits the way the public `contiguous` wrapper does: on an
already-contiguous tensor it returns a *new wrapper that still shares storage* (a
view). Probe: *"the raw make_contiguous shim LEAKS into public dir(Tensor)"* +
*"make_contiguous does NOT short-circuit …"* + *"make_contiguous's wrapper still
SHARES storage …"*.

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `Python-only`.
A binding that faithfully mirrors the C++ signature — including `void`→`None`,
`T&`→self, and by-value→a fresh wrapper — is `identical`; `signature-differs`
marks a binding-layer change to arity or defaults.

| C++ (`Tensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `Tensor permute(const std::vector<cytnx_uint64>&) const` (`:722`) | `permute(*args)` (lambda `:184-188`) | **signature-differs** | pure, shared-data view; typed signature erased by `(*args)` (T-S1/T-S3) |
| `Tensor &permute_(const std::vector<cytnx_uint64>&)` (`:691`) | `permute_(*args)` (lambda `:179-183`) | **signature-differs** | lambda returns `&self.permute_(...)` → self; `(*args)` erasure (T-S2/T-S3) |
| `Tensor reshape(const std::vector<cytnx_int64>&) const` (`:845`) | `reshape(*args)` (lambda `:199-203`) | **signature-differs** | pure, shared-data view; `(*args)` erasure (T-S1/T-S3) |
| `Tensor &reshape_(const std::vector<cytnx_int64>&)` (`:798`) | `reshape_(*args)` (lambda `:194-198`) | **signature-differs** | lambda returns `&self.reshape_(...)` → self; `(*args)` erasure (T-S2/T-S3) |
| `Tensor contiguous() const` (`:752`) | `make_contiguous` **+** conti.py `contiguous` | **leak** | raw C++ `contiguous()` bound as `make_contiguous` (`:192`); public `contiguous` is a conti.py short-circuit wrapper (T-S4/T-S8) |
| `Tensor contiguous_()` (`:772`, **returns `*this` by value**) | `contiguous_()` (`:193`) | identical | bound as a plain method pointer; the by-value C++ return means Python gets a shared-data wrapper, **not** self (T-S5) |
| `Tensor flatten() const` (`:1416`) | `flatten()` (`:190`) | identical | pure, independent 1-D copy (T-S6) |
| `void flatten_()` (`:1430`) | `flatten_()` (`:191`) | identical | `void` C++ method → Python `None`; no self-return possible (T-S7) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/Tensor_04_shape.py` (on the 1.1.0 wheel). A **(binding
fidelity)** finding flags where the binding layer — a `*_conti.py` wrapper or a
pybind lambda — changes behavior or signature versus the raw C++ method; **both
sides are runtime-verified**, the raw-C++ side by
`probes/cpp/Tensor_04_shape.cpp` (links against a source-built `libcytnx`,
GCC 13). Source `file:line` cites remain for traceability. Note the honest
result of the C++ probe: for `contiguous_` (T-S5) and `flatten_` (T-S7) the
broken return convention is **rooted in the C++ signature itself**, not
introduced by the binding — the fix belongs in C++ (and its lambda), not only
the pybind glue.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-S1** | `permute` and `reshape` return a **distinct object that shares storage** with the receiver — a **view**, not a copy | copy/view (B2) | **thin pass-through** — the `permute` lambda (`tensor_py.cpp:184-188`) forwards to C++ `permute(...) const` (`hpp:722`), which returns a new Tensor over the same storage; `reshape` likewise (`hpp:845`). Py probe *"permute returns a shared-data VIEW …"* + *"reshape returns a shared-data VIEW …"* + *"… VIEW is live …"*; **C++ probe confirms** mutating the source shows through the permute | **keep**; **document the shared-data view semantics explicitly** (silent today) — this is a numpy divergence hazard: `reshape` here always aliases, and `permute` yields a non-contiguous view (T-E4 cross-ref, cat 03). |
| **T-S2** | `permute_`/`reshape_` are correct in-place methods returning **self** (the same Python object) | (kept) | **faithful** — the pybind lambdas return `&self.permute_(...)` (`:182`) / `&self.reshape_(...)` (`:197`), preserving C++'s `Tensor&` (`hpp:691,798`). Py probe *"permute_ … returns SELF (ret is a)"* + *"reshape_ … returns SELF (ret is b)"*; **C++ probe confirms** `&t.permute_(...)==&t`, `&t.reshape_(...)==&t` (return type `Tensor&`, static_assert) | **keep** — the model the other in-place ops (T-S5/T-S7) should match. |
| **T-S3** | `permute`/`permute_`/`reshape`/`reshape_` are bound as `(*args)` pybind lambdas, **erasing** the C++ typed signature | **binding fidelity** / signature-differs | **binding erases the signature**: each lambda takes `py::args` and hand-resolves the mapper/new_shape (`:179-203`), so the docstring reads `permute(*args)` / `reshape(*args)` and `inspect.signature()` raises `ValueError`. Py probe (per member) *"… is bound as a (*args) pybind lambda — inspect.signature() raises ValueError"* + *"… docstring exposes the erased (*args) signature …"*; **C++ probe confirms** C++ `permute_`/`reshape_` have real typed signatures returning `Tensor&` | **bind explicit typed signatures** — `permute(mapper: Sequence[int])`, `reshape(new_shape: Sequence[int])` (typed `py::arg`s, accepting the numpy `-1` auto-dim for reshape), restoring introspection; the call surface is unchanged. |
| **T-S4** | `contiguous` binds via the raw **`make_contiguous`** shim; the public `contiguous` is a conti.py wrapper that short-circuits to `self` when already contiguous, else returns an **independent** contiguous copy | **binding fidelity** / leak + naming | **binding exposes plumbing + wraps it**: the raw C++ `contiguous()` is bound as `make_contiguous` (`:192`), and `cytnx/Tensor_conti.py:50-55` defines `contiguous` as `if is_contiguous(): return self else: return self.make_contiguous()`. Py probe *"contiguous() short-circuits to SELF …"* + *"… returns … an INDEPENDENT copy (same_data is False) …"*; **C++ probe confirms** C++ `contiguous()` returns a distinct object that does **not** share data on a non-contiguous tensor | **remove `make_contiguous` from the public API** (inline into the `contiguous` pybind lambda, which should itself do the short-circuit); keep `contiguous`/`contiguous_`. |
| **T-S5** | `contiguous_` coalesces the receiver in place but its binding returns a **distinct shared-data wrapper, NOT self** — unlike `permute_`/`reshape_`. The identity drop is **rooted in the C++ signature** | **binding fidelity** (N2/B1) — **C++-rooted** | **the C++ method returns by value**: `Tensor contiguous_()` (`hpp:772`) ends `return *this;` (return type `Tensor`, **not** `Tensor&`), and the pybind `.def("contiguous_", &Tensor::contiguous_)` (`:193`) is a plain method pointer, so Python receives a fresh wrapper sharing `_impl`. Py probe *"contiguous_ coalesces the RECEIVER's storage in place …"* + *"contiguous_ returns a DISTINCT object, NOT self (ret is not d) …"* + *"… still SHARES data …"*; **C++ probe confirms** the return type is `Tensor` by value (static_assert), so Python's distinct-object return is **faithful** to C++ — the broken self-return originates in the C++ signature, not the binding | **fix the C++ signature to `Tensor &contiguous_()` (return `*this` by reference)** and bind `&self.contiguous_()`, so `contiguous_` returns self like `permute_`/`reshape_`. *Migration:* callers relying on `t.contiguous_()`'s value still get a same-data handle; only identity is added. |
| **T-S6** | `flatten` is a **pure** 1-D collapse returning an **independent copy** (clone + contiguous + reshape) | copy/view (B2) | **faithful pass-through** — C++ `flatten() const` (`hpp:1416`) does `clone() → contiguous_() → reshape_({-1})`, so the result is a fresh rank-1 tensor that does **not** share data; the source is untouched. Py probe *"flatten returns a DISTINCT object collapsed to rank 1 ([24])"* + *"flatten returns an INDEPENDENT COPY (same_data(src) is False) …"*; **C++ probe confirms** the same | **keep**; **document the copy semantics** (contrast `reshape`'s view, T-S1) — `flatten` is `reshape(-1)` **plus** a clone. |
| **T-S7** | `flatten_` collapses the receiver in place but returns **`None`, not self** — breaking the in-place self-return convention its siblings keep. The `None` is **rooted in the C++ signature** | **binding fidelity** (N2) — **C++-rooted** | **the C++ method is void**: `void flatten_()` (`hpp:1430`) does `contiguous_() → reshape_({-1})` and returns nothing; the pybind `.def("flatten_", &Tensor::flatten_)` (`:191`) faithfully yields `None`. Py probe *"flatten_ collapses the RECEIVER to rank 1 in place ([24])"* + *"flatten_ returns None, NOT self …"*; **C++ probe confirms** the return type is `void` (static_assert), so Python's `None` is **faithful** to C++ — the missing self-return originates in the C++ signature (v1 Tensor C4) | **fix the C++ signature to `Tensor &flatten_()` (return `*this`)** and bind `&self.flatten_()`, so `flatten_` returns self and `t.flatten_().dtype()` chains work — matching `permute_`/`reshape_`. |
| **T-S8** | the raw **`make_contiguous`** binding **leaks** into `dir(Tensor)`, and unlike `contiguous` it never short-circuits — on an already-contiguous tensor it returns a new shared-data wrapper (a view) | naming + **binding fidelity** | **binding exposes plumbing**: `make_contiguous` is `.def`-ed publicly (`:192`) purely so the conti.py `contiguous` wrapper can call it; the `make_` shim spelling must not be public (§R.0). Called directly on a contiguous tensor it returns a distinct wrapper still sharing storage. Py probe *"the raw make_contiguous shim LEAKS into public dir(Tensor)"* + *"make_contiguous does NOT short-circuit …"* + *"… still SHARES storage …"*; **C++ probe confirms** the underlying C++ `contiguous()` returns a distinct object | **remove from the public API** — inline into the `contiguous` pybind lambda (which does the already-contiguous short-circuit); it is not public surface (migration note). |

## A4. Argument ordering — positional & keyword

Every member here takes at most a single mapper/shape primary operand; there is
no keyword-only metadata block.

| API | positional-required (in order) | keyword |
|---|---|---|
| `permute` / `permute_` | `mapper` (the axis order) | — |
| `reshape` / `reshape_` | `new_shape` | — |
| `contiguous` / `contiguous_` / `make_contiguous` | *(none)* | — |
| `flatten` / `flatten_` | *(none)* | — |

- **Positional:** `permute`/`reshape` take their single mapper/shape operand
  first — matching the canonical *primary-operand-first* rule. The **only**
  change needed is to bind them with **real typed parameters** (not `(*args)`,
  T-S3), so the operand carries a name (`mapper` / `new_shape`) and
  `inspect.signature()` works. `reshape` keeps its numpy-style `-1` auto-dim.
- **Keyword:** none in this category — every member is either nullary or takes a
  single positional mapper/shape. No `arg0` erasure to fix here (unlike cat 02's
  `same_data`), only the `(*args)` collapse.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
shape/layout surface. Implement Cytnx to match it. Findings above are the
rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).** Every
  member here is already lowercase snake_case (`permute`, `reshape`,
  `contiguous`, `flatten`) — **no rename** is recommended. The one non-conformant
  spelling is the leaked **`make_contiguous`** shim, which is removed from the
  public surface (T-S4/T-S8), not renamed.
- **N-underscore — a trailing `_` marks in-place and returns `self`.** Every
  operation here that is meaningful in both modes provides both forms under one
  base name — `permute`/`permute_`, `reshape`/`reshape_`,
  `contiguous`/`contiguous_`, `flatten`/`flatten_`. **All four `_` forms must
  return `self` from the binding.** Today `permute_`/`reshape_` do (T-S2), but
  `contiguous_` returns a shared-data wrapper (T-S5) and `flatten_` returns
  `None` (T-S7) — both because their **C++ signatures** are `Tensor` (by value)
  and `void` respectively, not `Tensor&`. The fix is in C++ **and** the lambda:
  make both return `Tensor&`/`*this` and bind `&self`. The `make_` shim spelling
  is rejected as public API (T-S4/T-S8) — a public in-place method is *only* the
  trailing-`_` form.
- **N-view — copy/view behavior is fixed and documented, and it is NOT
  numpy-symmetric.** The category's copy/view map (all probe-verified):
  `permute`/`reshape` → **shared-data view** (T-S1); `contiguous` (already
  contiguous) → **self**, (non-contiguous) → **independent copy** (T-S4);
  `contiguous_`'s returned handle → **shared-data** with the receiver (T-S5);
  `flatten` → **independent copy** (T-S6). That `reshape` always **aliases** (a
  view) while `flatten` (a `reshape(-1)` plus a clone) **copies** is the
  load-bearing divergence a caller must know; every docstring states its class.
- **Real signatures, not `*args`.** `permute`/`permute_`/`reshape`/`reshape_`
  must bind with explicit typed `py::arg`s (`mapper` / `new_shape`), not
  `(*args)` (T-S3), restoring introspection.
- **N-positional — mapper/shape first.** `permute(mapper)`, `reshape(new_shape)`;
  the layout ops are nullary. Matches the live order; only the typing changes.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class Tensor:
    # --- axis reordering (pure = shared-data view, in-place = self) ---
    def permute(self, mapper: Sequence[int]) -> "Tensor": ...     # pure, shared-data view
    def permute_(self, mapper: Sequence[int]) -> "Tensor": ...    # in-place, self

    # --- shape change (typed signature, not *args; -1 auto-dim like numpy) ---
    def reshape(self, new_shape: Sequence[int]) -> "Tensor": ...  # pure, shared-data view
    def reshape_(self, new_shape: Sequence[int]) -> "Tensor": ... # in-place, self

    # --- memory layout ---
    def contiguous(self) -> "Tensor": ...    # self if already contiguous, else an independent copy
    def contiguous_(self) -> "Tensor": ...   # in-place, self  (fix C++ Tensor& + bind &self)

    # --- 1-D collapse ---
    def flatten(self) -> "Tensor": ...       # pure, independent 1-D copy
    def flatten_(self) -> "Tensor": ...      # in-place, self  (fix C++ Tensor& + bind &self)
```

In-place methods return `self` **from the binding**; the raw `make_contiguous`
plumbing binding becomes private (leading `_`) or is inlined into the
`contiguous` pybind lambda — it is **not** a public member.

| API | Verdict | Behavior contract |
|---|---|---|
| `permute` | **keep, but bind a typed signature** (T-S1/T-S3; document shared-data view) | Pure axis reorder; returns a new Tensor sharing storage with the receiver (a view), generally non-contiguous. *Migration:* replace the `(*args)` lambda with `permute(mapper)`; call surface unchanged, introspection restored. |
| `permute_` | **keep, but bind a typed signature** (T-S2/T-S3) | In-place axis reorder; returns self (chainable). Same typed-signature migration. |
| `reshape` | **keep, but bind a typed signature** (T-S1/T-S3) | Pure shape change; returns a new Tensor **sharing storage** (a view), NOT a copy; accepts a single `-1` auto-dim (numpy-style). *Migration:* `reshape(new_shape)`. |
| `reshape_` | **keep, but bind a typed signature** (T-S2/T-S3) | In-place shape change; returns self. Same typed-signature migration. |
| `contiguous` | **keep** (T-S4; short-circuit in the lambda) | Returns self if already contiguous, else a new, **independent** contiguous copy. |
| `contiguous_` | **keep, but return self** (T-S5) | In-place coalesce; returns self. *Migration:* change the C++ signature to `Tensor &contiguous_()` (return `*this` by reference) and bind `&self.contiguous_()`; today it returns a distinct shared-data wrapper because C++ returns `Tensor` by value. |
| `flatten` | **keep** (T-S6; document copy) | Pure 1-D collapse; returns a new, **independent** rank-1 tensor (clone + contiguous + reshape). |
| `flatten_` | **keep, but return self** (T-S7) | In-place 1-D collapse; returns self. *Migration:* change the C++ signature from `void flatten_()` to `Tensor &flatten_()` (return `*this`) and bind `&self.flatten_()`; today it returns `None` (v1 C4), breaking the in-place convention its siblings keep. |

**Internal / plumbing — hidden, not public API.** The raw binding below is
covered here (it is a live public member today) with a **remove** verdict: hide
it behind a leading underscore or inline it into the `contiguous` pybind lambda.
It carries no docstring — it is not public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `make_contiguous` | **remove** (T-S4/T-S8) | Raw plumbing (the C++ `contiguous()`) that the conti.py `contiguous` wrapper calls; unlike `contiguous` it does not short-circuit (returns a shared-data view even when already contiguous). *Migration:* inline into the `contiguous` pybind lambda (which does the short-circuit); no public exposure — a `DeprecationWarning` alias for one minor release, then delete. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept members are documented
(the removed `make_contiguous` plumbing carries no docstring).

### R.2a Python API (numpy-style)

### `permute` / `permute_`

```
Tensor.permute(mapper)   -> Tensor   # pure, shared-data view
Tensor.permute_(mapper)  -> Tensor   # in-place, self

Reorder the axes of this tensor.

`permute` is PURE: it returns a new Tensor with the axes in the requested order
while leaving this tensor unchanged. The returned tensor SHARES its storage with
this one (a metadata-only view) — mutating either tensor's elements is visible
through the other (finding T-S1) — and is generally NON-contiguous.

`permute_` is the IN-PLACE form: it reorders this tensor's axes and returns self
for chaining (finding T-S2).

Parameters
----------
mapper : sequence of int
    The new axis order (a permutation of range(rank)).

Returns
-------
Tensor
    `permute`: a new tensor (shared data). `permute_`: self.

Notes
-----
Through cytnx 1.1.0 these were bound as ``permute(*args)``, erasing the
signature (finding T-S3); the next version binds the typed signature above.
To detach the view, follow with `clone()` or `contiguous()`.
```

### `reshape` / `reshape_`

```
Tensor.reshape(new_shape)   -> Tensor   # pure, shared-data view
Tensor.reshape_(new_shape)  -> Tensor   # in-place, self

Change the shape of this tensor (total element count preserved).

`reshape` is PURE and returns a new Tensor SHARING storage with this one (a VIEW,
finding T-S1) — NOT a copy; `reshape_` reshapes IN PLACE and returns self
(finding T-S2). A single axis may be -1 to be inferred (numpy-style).

Parameters
----------
new_shape : sequence of int
    The target shape; one entry may be -1 (inferred from the total size).

Returns
-------
Tensor
    `reshape`: a new tensor (shared data). `reshape_`: self.

Raises
------
RuntimeError
    If the requested shape's total size differs from the tensor's.

Notes
-----
Through cytnx 1.1.0 these were bound as ``reshape(*args)`` (finding T-S3); the
next version binds the typed signature above. Contrast `flatten`, which is a
`reshape(-1)` PLUS a clone (an independent copy, finding T-S6).
```

### `contiguous` / `contiguous_`

```
Tensor.contiguous()   -> Tensor   # self if already contiguous, else an independent copy
Tensor.contiguous_()  -> Tensor   # in-place, self

Coalesce the tensor's storage into a contiguous memory layout.

`contiguous` returns self unchanged when the tensor is already contiguous,
otherwise a NEW, INDEPENDENT contiguous tensor (finding T-S4). `contiguous_`
coalesces IN PLACE and returns self (finding T-S5).

Returns
-------
Tensor
    `contiguous`: self (already contiguous) or a new contiguous copy.
    `contiguous_`: self.

Notes
-----
Through cytnx 1.1.0 `contiguous_` returned a distinct shared-data wrapper rather
than self (its C++ signature returned `Tensor` by value, finding T-S5); the next
version returns self. The raw `make_contiguous` binding is removed (folded into
`contiguous`, findings T-S4/T-S8).
```

### `flatten` / `flatten_`

```
Tensor.flatten()   -> Tensor   # pure, independent 1-D copy
Tensor.flatten_()  -> Tensor   # in-place, self

Collapse the tensor to a single axis (rank 1).

`flatten` is PURE and returns a new, INDEPENDENT 1-D tensor (clone + contiguous +
reshape; same_data() is False, finding T-S6); `flatten_` flattens IN PLACE and
returns self (finding T-S7).

Returns
-------
Tensor
    `flatten`: a new, independent rank-1 tensor. `flatten_`: self.

Notes
-----
Through cytnx 1.1.0 `flatten_` returned None rather than self (its C++ signature
was `void`, finding T-S7 / v1 C4); the next version returns self so
`t.flatten_().dtype()` chains work. Unlike `reshape` (a shared-data view),
`flatten` always COPIES.
```

### R.2b C++ API (Doxygen)

C++ mostly already returns `Tensor`/`Tensor&` per the N-underscore split, but
`contiguous_` returns `Tensor` **by value** and `flatten_` returns `void` — both
must change to `Tensor&` (return `*this`) so their pybind lambdas can return self
(findings T-S5/T-S7). The next version must also bind `permute`/`reshape` (and
their `_` forms) with typed signatures, not `(*args)` (T-S3), and fold the raw
`make_contiguous` shim into the `contiguous` lambda (T-S4/T-S8).

```cpp
/**
 * @brief Reorder the axes, returning a NEW Tensor (data shared).
 * @details Pure: the returned tensor has the axes in @p mapper order while
 *          *this is unchanged, but the storage is SHARED (a view, finding
 *          T-S1) and generally non-contiguous. permute_ reorders in place and
 *          returns *this. The Python binding must expose a typed signature, not
 *          (*args) (finding T-S3).
 * @param mapper new axis order (a permutation of range(rank)).
 * @return permute: a new Tensor (shared data). permute_: reference to *this.
 */
Tensor permute(const std::vector<cytnx_uint64> &mapper) const;
Tensor &permute_(const std::vector<cytnx_uint64> &mapper);

/**
 * @brief Change the shape (element count preserved), returning a NEW Tensor.
 * @details Pure: the returned tensor SHARES storage with *this (a view, finding
 *          T-S1). reshape_ reshapes in place and returns *this. A single -1
 *          entry is inferred (numpy-style). The Python binding must expose a
 *          typed signature, not (*args) (finding T-S3).
 * @param new_shape target shape (one entry may be -1).
 * @return reshape: a new Tensor (shared data). reshape_: reference to *this.
 */
Tensor reshape(const std::vector<cytnx_int64> &new_shape) const;
Tensor &reshape_(const std::vector<cytnx_int64> &new_shape);

/**
 * @brief Coalesce storage into a contiguous layout.
 * @details contiguous() returns a distinct, INDEPENDENT contiguous Tensor (or
 *          *this when already contiguous); contiguous_() coalesces in place and
 *          returns *this. NOTE: contiguous_ currently returns `Tensor` BY VALUE
 *          — change it to `Tensor &contiguous_()` (return *this) so the Python
 *          binding returns self (finding T-S5). The Python binding must fold the
 *          raw make_contiguous shim into the contiguous() lambda (T-S4/T-S8).
 * @return contiguous: a contiguous Tensor. contiguous_: reference to *this.
 */
Tensor contiguous() const;
Tensor &contiguous_();   // was: Tensor contiguous_()  (by value — finding T-S5)

/**
 * @brief Collapse the tensor to a single axis (rank 1).
 * @details flatten() is PURE — clone + contiguous_ + reshape_({-1}) — returning
 *          a new, INDEPENDENT rank-1 Tensor (finding T-S6). flatten_() collapses
 *          in place. NOTE: flatten_ currently returns `void` — change it to
 *          `Tensor &flatten_()` (return *this) so the Python binding returns
 *          self rather than None (finding T-S7 / v1 C4).
 * @return flatten: a new (independent) Tensor. flatten_: reference to *this.
 */
Tensor flatten() const;
Tensor &flatten_();      // was: void flatten_()  (finding T-S7)
```
