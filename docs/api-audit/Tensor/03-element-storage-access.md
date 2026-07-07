# Tensor — 03. Element & storage access

> **Superset-method rollout** (Tensor, category 03 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings) and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` element/storage-access surface should be implemented to match §R
> exactly. Every behavioral claim is verified against the installed
> `cytnx==1.1.0` wheel by
> `docs/api-audit/probes/Tensor_03_element.py` (all `[PASS]`, exit 0).
> This category has **no binding-fidelity finding** that a raw-C++ probe could
> verify — `numpy()` is a binding-only pybind lambda (no C++ counterpart to
> diverge from), the slice-read copy is faithful to the C++ `get()` contract,
> and the `std::cout` leak is a pybind-lambda artifact with no raw-C++ analog —
> so it carries **no raw-C++ probe** (gate 4, see A3).

**Category scope:** the members that read and write element data and expose the
backing buffer — single-element extraction (`item`), the `Storage` handle
(`storage`), the numpy bridge (`numpy`), the complex real/imaginary parts
(`real`, `imag`), the in-place writers (`fill`, `append`), and the Python
indexing operators (`__getitem__`/`__setitem__`). Read-only *metadata*
(`shape`/`dtype`/`same_data`, the view-vs-copy oracle) is
[category 02](02-metadata-introspection.md); shape *manipulation*
(`permute`/`reshape`/`contiguous`/`flatten`) is [category 04](04-shape-layout.md);
`from_storage` (the numpy-bridge's constructor twin) is
[category 01](01-construction-init.md).

---

# Analysis

**Provenance:** live pybind signatures from the 1.1.0 wheel
(`tools/member_inventory.py Tensor`); Python bindings in
`cytnx_src/pybind/tensor_py.cpp` (`numpy` lambda `:66-144`, `item` lambda
`:216-244`, `storage` `:245`, `real` `:246`, `imag` `:247`, `fill` `:255-265`,
`append` `:267-283`, `__getitem__` `:322-372` with the debug leak at `:355`,
`__setitem__` `:374-434`); C++ in `cytnx_src/include/Tensor.hpp` (`item`
`:1003-1008`, `get` `:1039,1045` with the "does not share memory" note at
`:1023`, `storage` `:1121`, `fill` `:1138`, `real`/`imag` `:1158,1167`, `append`
overloads `:1461,1523,1586`, `from_storage` `:558`).

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place writer.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `item` | `item()` → `object` | Python scalar (**all 11 dtypes**) | Extract the sole element of a 1-element tensor; the pybind lambda branches over all 11 dtypes. **Raises** on a multi-element tensor. Probe: *"item() extracts the sole scalar …"* + *"item() extracts across dtypes (Int64 → 5)"* + *"item() RAISES on a multi-element tensor (B4)"*. |
| `storage` | `storage()` → `Storage` | `Storage` (**shared-data view**) | The tensor's backing `Storage`; **shares** memory (mutating it shows through the tensor). Thin pass-through of C++ `Storage& storage()`. Probe: *"storage() returns a shared-data VIEW …"*. |
| `numpy` | `numpy(share_mem: bool = False)` → `numpy.ndarray` | `ndarray` (**always a copy**) | Export to a numpy array. Default is a **copy**; `share_mem=True` — despite promising a zero-copy view — **also returns a copy** (`OWNDATA` True, writes do not propagate) and merely enforces contiguity. Binding-only lambda; no C++ member. Probe: *"numpy() default is a COPY …"* + *"numpy(share_mem=True) … is ALSO a COPY …"* + *"… RAISES on a non-contiguous tensor"*. |
| `real` | `real()` → `Tensor` | `Tensor` (**copy**, complex-only) | Real part of a complex tensor, as an independent copy (`same_data` False). **Raises** on a non-complex tensor. Probe: *"real() returns a COPY … (same_data() is False)"* + *"real() RAISES on a non-complex tensor …"*. |
| `imag` | `imag()` → `Tensor` | `Tensor` (**copy**, complex-only) | Imaginary part, likewise an independent copy. Probe: *"imag() likewise returns a COPY …"*. |
| `fill` `[I]` | `fill(val)` → `None` | `None` (in-place) | Set **every** element to `val` in place; bound for all 11 dtypes. Probe: *"fill(val) sets EVERY element in place"* + *"fill(val) returns None …"*. |
| `append` `[I]` | `append(val)` → `None` | `None` (in-place) | Grow the tensor along **axis 0** in place; overloads for a scalar (11 dtypes), a `Tensor`, and a `Storage`. Probe: *"append(scalar) grows … ([3] → [4]) in place"* + *"append(Tensor) grows axis 0 … ([2,3] → [3,3])"*. |
| `__getitem__` | `__getitem__(locators)` | `Tensor` (**COPY** — not a view) | Slice via `t[...]`; wraps the C++ `get(accessors)`, which **does not share memory** — the result is an independent copy, unlike numpy. The bare-1-D-slice branch leaks a `std::cout` debug line (T-E9). Probe: *"slice READ t[0:1] returns an independent COPY …"* + *"integer-index READ t[0] likewise …"*. |
| `__setitem__` | `__setitem__(locators, rhs)` | `None` (in-place) | Assign via `t[...] = rhs`; wraps the C++ `set(accessors, rhs)`, mutating the tensor's storage **in place**. Probe: *"element ASSIGN t[0,0]=v mutates in place …"* + *"slice ASSIGN t[1:2]=rhs mutates in place …"*. |

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `Python-only`.

| C++ (`Tensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `template<T> T& item()` / `Scalar::Sproxy item()` (`:1003-1008`) | `item()` (lambda `:216-244`) | identical | lambda branches all 11 dtypes; raises on a non-scalar (T-E5) |
| `Storage& storage() const` (`:1121`) | `storage()` (`:245`) | identical | returns the shared-data `Storage` (T-E2) |
| *(no C++ member)* | `numpy(share_mem=False)` (lambda `:66-144`) | **Python-only** | binding-only bridge; both modes copy (T-E1) |
| `Tensor real()` (`:1158`) | `real()` (`:246`) | identical | returns a copy; complex-only (T-E3) |
| `Tensor imag()` (`:1167`) | `imag()` (`:247`) | identical | returns a copy; complex-only (T-E3) |
| `template<T> void fill(const T&)` (`:1138`) | `fill(val)` (`:255-265`) | identical | 11-dtype in-place set; returns `None` (T-E6) |
| `void append(const Tensor&)` (`:1461`) · `void append(const Storage&)` (`:1523`) · `template<T> void append(const T&)` (`:1586`) | `append(val)` (`:267-283`) | identical | grows axis 0 in place; returns `None` (T-E7) |
| `Tensor get(accessors) const` (`:1039,1045`) | *(unbound; via `__getitem__` `:371`)* | **C++-only** | reached only through `t[...]`; **copy** — "does not share memory" (`:1023`, T-E4) |
| `Tensor& set(accessors, rhs)` (`:1069`) | *(unbound; via `__setitem__` `:405`)* | **C++-only** | reached only through `t[...] = rhs`; in-place (T-E8) |
| `operator[]` / slice read | `__getitem__(locators)` (`:322-372`) | **signature-differs** | pybind lambda; bare-1-D-slice branch carries a stray `std::cout` (`:355`, T-E9) |
| `set(accessors, rhs)` | `__setitem__(locators, rhs)` (`:374-434`) | identical | scalar-rhs overloads for all 11 dtypes (`:424-434`) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/Tensor_03_element.py` (on the 1.1.0 wheel). A **(binding
fidelity)** finding would flag where the binding layer changes behavior versus
the raw C++ method; **this category has none that a raw-C++ probe could
verify** — `numpy` has no C++ counterpart (T-E1 is a Python-only property of the
pybind lambda plus pybind11's `py::array` construction), the slice-read copy is
**faithful** to the C++ `get()` contract (T-E4), and the `std::cout` leak lives
entirely inside the `__getitem__` pybind lambda (T-E9). **Gate 4 (raw-C++
probe) is therefore skipped for category 03.** Source `file:line` cites remain
for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-E1** | `numpy()` is the numpy bridge — but it is **always a copy**: `share_mem=True` does **not** deliver the promised zero-copy view (the ndarray owns its buffer, `OWNDATA` True, and writes do not propagate) | **correctness** (copy/view · Python-only) | **the pybind lambda copies regardless of the flag**: on a contiguous tensor `share_mem=True` skips the internal `clone()` and points a `py::buffer_info` at the storage (`tensor_py.cpp:82-142`), but the terminal `return py::array(npbuf)` (no base handle, `:142`) makes numpy **allocate and copy** — so the caller gets an owning copy either way; the flag's only live effect is to **raise** on a non-contiguous tensor (`:88-92`). Probe: *"numpy(share_mem=True) ndarray OWNS its buffer (OWNDATA True) …"* + *"… is ALSO a COPY … (write does not propagate)"* + *"… RAISES on a non-contiguous tensor"* | **keep the bridge** (it is the numpy interop surface — see T-E1a); **fix `share_mem`**: either pass a base handle so `share_mem=True` is a genuine zero-copy view, or **drop the flag** and document `numpy()` as always-copy. Do not advertise a view that is not delivered. |
| **T-E1a** | the numpy bridge (`numpy()` + `from_storage`, cat 01) **EXISTS on Tensor** and closes the gap that `UniTensor` leaves open (UniTensor has no `numpy()`) | (kept — headline) | **Tensor is the interop layer**: `hasattr(Tensor,"numpy")` and `hasattr(Tensor,"from_storage")` are True while `hasattr(UniTensor,"numpy")` is **False** — a `UniTensor` reaches numpy only by `get_block()`→`Tensor`→`numpy()`. Probe: *"the numpy bridge EXISTS on Tensor … closes the UniTensor UT-C3/UT-T6 gap …"* | **keep** — this is the probe-verified round-trip that the UniTensor audit (UT-C3/UT-T6) flagged as *missing on UniTensor*; document `Tensor.numpy()`/`from_storage` as the canonical ndarray bridge and cross-reference it as the resolution of UT-C3/UT-T6. |
| **T-E2** | `storage()` returns a **shared-data view** onto the tensor's buffer | copy/view (B2) | **thin pass-through** of C++ `Storage& storage()` (`hpp:1121`) — the returned `Storage` aliases the tensor's memory, so a write through it is visible on the tensor. Probe: *"storage() returns a shared-data VIEW … (t[0] becomes 888)"* | **keep**; **document the view** explicitly — call `Storage.clone()` for an independent copy. |
| **T-E3** | `real()` / `imag()` return independent **copies** (complex-only) | copy/view (B2) | **faithful pass-throughs** — C++ `Tensor real()`/`imag()` (`hpp:1158,1167`) materialize a fresh real-typed tensor; `same_data` is False and a write to the result does not touch the source. Both **raise** on a non-complex tensor. Probe: *"real() returns a COPY … (same_data() is False)"* + *"real() RAISES on a non-complex tensor …"* | **keep both**; **document copy semantics + the complex-only precondition**. |
| **T-E4** | **slice READ is a COPY, not a numpy-style view** — `t[0:2]` and `t[0]` return an independent tensor | copy/view (B2) — **numpy divergence hazard** | **faithful to C++ `get()`**: the `__getitem__` lambda ends `return self.get(accessors)` (`:371`), and C++ `get()` is documented "*The return will be a new Tensor instance, which does not share memory*" (`hpp:1023`). Identical on both language sides (not a parity gap) but it **contradicts numpy's B2 expectation** that `t[0:2]` aliases `t`. Probe: *"slice READ t[0:1] returns an independent COPY (same_data() is False) …"* + *"integer-index READ t[0] likewise …"* + *"the slice-read COPY is detached …"* | **keep the copy semantics**, but **document the hazard prominently** — numpy code that mutates `t[0:2]` expecting to touch `t` silently operates on a copy. Point readers to `storage()` / element-assign for in-place access. |
| **T-E5** | slice / element **ASSIGN mutates in place** — `t[0,0]=v` and `t[0:2]=rhs` write through to the tensor's storage | copy/view (B2) | **wraps C++ `set(accessors, rhs)`** (`__setitem__` `:405`), which writes into the tensor's own buffer; an alias observes the write. This is the **asymmetry** with T-E4: read copies, write is in-place. Probe: *"element ASSIGN t[0,0]=v mutates in place …"* + *"slice ASSIGN t[1:2]=rhs mutates in place …"* | **keep**; **document the read-copies/write-in-place asymmetry** alongside T-E4 so it is not read as numpy's symmetric view model. |
| **T-E6** | `item()` extracts the sole scalar of a **1-element** tensor across all 11 dtypes and **raises** on a multi-element tensor | (kept) | **all-dtype lambda + guard**: the `item` lambda branches all 11 dtypes (`:216-244`); the C++ `item<T>()` errors "*can only be called from a Tensor with only one element*" (`hpp:979`). Probe: *"item() extracts the sole scalar …"* + *"… across dtypes (Int64 → 5)"* + *"item() RAISES on a multi-element tensor (B4)"* | **keep** — the canonical scalar extractor; document the 1-element precondition. |
| **T-E7** | `fill(val)` sets **every** element in place and returns `None` | N-underscore (non-suffixed in-place) | **11-dtype in-place writer** (`:255-265`) forwarding to C++ `template<T> void fill(const T&)` (`hpp:1138`); mutates the receiver, returns `None`. Probe: *"fill(val) sets EVERY element in place"* + *"fill(val) returns None …"* | **keep** — correctly named; note it is an in-place writer **without** a trailing `_` (like `append`), an established exception documented in R.0. |
| **T-E8** | `append(val)` grows the tensor along **axis 0** in place (scalar / `Tensor` / `Storage`), returns `None` | N-underscore (non-suffixed in-place) | **overloaded in-place grow** (`:267-283`) → C++ `append(Tensor/Storage/T)` (`hpp:1461,1523,1586`); forces contiguity, extends axis 0. Probe: *"append(scalar) grows … ([3] → [4]) in place"* + *"append(Tensor) grows axis 0 … ([2,3] → [3,3])"* | **keep**; document the axis-0 grow + the scalar/`Tensor`/`Storage` overload set. |
| **T-E9** | the bare-1-D-slice `__getitem__` branch leaks a leftover `std::cout` debug line ("start stop step") to the process's **real stdout** | **correctness / hygiene** (binding artifact) | **stray debug print, unguarded**: `tensor_py.cpp:355` is `std::cout << start << " " << stop << " " << step << std::endl;` inside the `py::isinstance<py::slice>` branch, with **no** `py::scoped_ostream_redirect` guard (unlike `__repr__` at `:254`). So `t[0:2]` prints e.g. `0 2 1` to fd 1, **uncapturable** by `contextlib.redirect_stdout`. Tuple indexing (`t[0,1]`) takes a different branch and does not leak. Probe: *"t[0:2] … debug line is UNCAPTURABLE by contextlib.redirect_stdout …"* + *"… leaks a 'start stop step' std::cout debug line to … fd-1 stdout ('0 2 1')"* + *"tuple indexing t[0,1] … does NOT leak …"* | **fix — delete the stray `std::cout`** (`tensor_py.cpp:355`). It is dead debug output shipping on every bare 1-D slice; same uncapturable-print pathology as `Device.Print_Property` / `Symmetry` P5. |

## A4. Argument ordering — positional & keyword

The writers take a value/payload operand; the readers are nullary or take an
index/slice locator; only `numpy` has a keyword-style flag.

| API | positional-required (in order) | keyword |
|---|---|---|
| `item` | *(none)* | — |
| `storage` / `real` / `imag` | *(none)* | — |
| `numpy` | *(none)* | `share_mem: bool = False` |
| `fill` | `val` | — |
| `append` | `val` (scalar / `Tensor` / `Storage`) | — |
| `__getitem__` | `locators` | — |
| `__setitem__` | `locators`, `rhs` | — |

- **Positional:** the writers take their single payload first (`fill(val)`,
  `append(val)`); the operators take `locators` then (for setitem) `rhs`. This
  matches the canonical *primary-operand-first* rule and needs no change.
- **Keyword:** `numpy`'s `share_mem` is the only keyword; it carries a real
  `py::arg("share_mem")` name (unlike the `arg0` erasures elsewhere) and is
  keyword-callable. Its **behavior** is the defect (T-E1), not its signature.

**Canonical rule (normative — see §R.0):** payload operand first for the
writers; `numpy(share_mem=...)` keeps its named keyword.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
element/storage-access surface. Implement Cytnx to match it. Findings above are
the rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).**
  Every member here is already lowercase snake_case (`item`, `storage`, `numpy`,
  `real`, `imag`, `fill`, `append`) — **no rename** is recommended in this
  category.
- **N-underscore — a trailing `_` marks in-place (returns `self`).** `fill` and
  `append` are **in-place writers without** a trailing `_` — an established,
  kept exception: they mutate the receiver and return `None`, not a suffixed
  pair (there is no pure `fill`/`append` counterpart). This is documented
  explicitly (T-E6/E7) rather than treated as a violation. No accessor here
  gains a `_`.
- **N-view — copy/view behavior is fixed and documented, and it is NOT
  numpy-symmetric.** The category's copy/view map (all probe-verified):
  `storage()` → **shared-data view** (T-E2); `real`/`imag` → **copies** (T-E3);
  **slice READ** `t[0:2]`/`t[0]` → **copy** (T-E4); **slice/element ASSIGN**
  `t[...] = v` → **in-place** (T-E5); `numpy()` → **copy** in both modes (T-E1).
  The read-copies / write-in-place asymmetry (T-E4 vs T-E5) is the load-bearing
  divergence from numpy and every docstring states its class.
- **N-argname — `numpy`'s `share_mem` is a real, keyword-callable name.** Keep it
  named; fix its behavior (T-E1) rather than its signature.
- **Binding hygiene — no stray debug output.** The leftover `std::cout` in the
  bare-1-D-slice `__getitem__` branch (T-E9) is **deleted**; no public accessor
  writes to stdout as a side effect.

*The positional/keyword rule is also normative.*

- **N-positional — payload first for the writers.** `fill(val)`, `append(val)`,
  `__setitem__(locators, rhs)`; the readers are nullary or take a `locators`
  operand; `numpy(share_mem=...)` keeps its single named keyword.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class Tensor:
    # --- single-element read ---
    def item(self) -> object: ...                 # sole element of a 1-element tensor (all 11 dtypes)

    # --- backing-buffer / interop access ---
    def storage(self) -> "Storage": ...           # SHARED-DATA view onto the buffer
    def numpy(self, share_mem: bool = False) -> "numpy.ndarray": ...  # copy; see T-E1
    def real(self) -> "Tensor": ...               # real part (COPY; complex-only)
    def imag(self) -> "Tensor": ...               # imag part (COPY; complex-only)

    # --- in-place writers (no trailing _; both return None) ---
    def fill(self, val) -> None: ...              # set every element (all 11 dtypes)
    def append(self, val) -> None: ...            # grow axis 0 (scalar / Tensor / Storage)

    # --- indexing operators ---
    def __getitem__(self, locators) -> "Tensor": ...   # slice READ -> independent COPY (not a view!)
    def __setitem__(self, locators, rhs) -> None: ...  # slice/element ASSIGN -> in place
```

`numpy` either delivers a genuine zero-copy view when `share_mem=True` (pass a
base handle) **or** drops the flag and is documented as always-copy — it must
not advertise a view it does not deliver (T-E1). The C++ `get`/`set` accessors
remain unbound-by-name — `__getitem__`/`__setitem__` are their sole Python
surface. The stray `std::cout` in the slice branch is removed (T-E9).

| API | Verdict | Behavior contract |
|---|---|---|
| `item` | **keep** (T-E6) | Return the sole element of a 1-element tensor as a native Python scalar (all 11 dtypes). Raises on a multi-element tensor. |
| `storage` | **keep** (T-E2; document view) | Return the tensor's backing `Storage`, a SHARED-DATA view — mutating it changes the tensor. Call `Storage.clone()` for an independent copy. |
| `numpy` | **keep, fix `share_mem`** (T-E1/E1a) | Export to a numpy ndarray. Today **both** modes return a copy; `share_mem=True`'s zero-copy view is non-functional. *Migration:* make `share_mem=True` a true view (base handle) or remove the flag and document always-copy. The canonical ndarray bridge — closes the UniTensor UT-C3/UT-T6 gap. |
| `real` | **keep** (T-E3; document copy) | Return the real part of a complex tensor as an independent COPY. Raises on a non-complex tensor. |
| `imag` | **keep** (T-E3; document copy) | Return the imaginary part as an independent COPY. Raises on a non-complex tensor. |
| `fill` | **keep** (T-E7) | Set every element to `val` in place (all 11 dtypes); returns `None`. In-place writer without a trailing `_`. |
| `append` | **keep** (T-E8) | Grow the tensor along axis 0 in place by a scalar / `Tensor` / `Storage`; returns `None`. Forces contiguity. |
| `__getitem__` | **keep, remove debug leak** (T-E4/E9) | Slice via `t[...]`, returning an independent COPY (NOT a numpy-style view). *Migration:* delete the stray `std::cout` in the bare-1-D-slice branch. Wraps C++ `get`. |
| `__setitem__` | **keep** (T-E5) | Assign via `t[...] = rhs` (in place — writes the tensor's storage). Wraps C++ `set`. |

**C++-only — the operator implementations (no separate Python member).** These
accessor methods exist in C++ and are reached from Python only through the
indexing operators; they carry a C++ (R.2b) docstring only.

| API | Verdict | Behavior contract |
|---|---|---|
| `get(accessors)` | **keep (C++-only)** (T-E4) | Slice implementation behind `__getitem__`; returns a new Tensor that **does not share memory** (a copy). Not a separate Python member. |
| `set(accessors, rhs)` | **keep (C++-only)** (T-E5) | Assignment implementation behind `__setitem__`; writes `rhs` into the sliced region in place and returns `*this`. Not a separate Python member. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. The C++-only `get`/`set` have an
R.2b docstring but no R.2a.

### R.2a Python API (numpy-style)

### `item`

```
Tensor.item() -> scalar

Return the sole element of a one-element tensor as a native Python scalar
(all 11 element dtypes).

Returns
-------
object
    A Python int / float / complex / bool matching the tensor's dtype.

Raises
------
RuntimeError
    If the tensor holds more than one element (finding T-E6).
```

### `storage`

```
Tensor.storage() -> Storage

Return the tensor's backing Storage.

Returns
-------
Storage
    A SHARED-DATA view: the Storage aliases the tensor's memory, so mutating it
    changes the tensor (finding T-E2). Call Storage.clone() for an independent
    copy.
```

### `numpy`

```
Tensor.numpy(share_mem=False) -> numpy.ndarray

Export this tensor as a numpy ndarray. The canonical numpy interop bridge; with
from_storage it round-trips Tensor <-> ndarray (closing the UniTensor gap
UT-C3/UT-T6, which UniTensor lacks — finding T-E1a).

Parameters
----------
share_mem : bool, optional
    Default False -> an independent COPY (mutating the ndarray does not affect
    the tensor, confirmed by probe). Through cytnx 1.1.0, share_mem=True ALSO
    returned a copy (the ndarray owned its buffer; the promised zero-copy view
    was non-functional, finding T-E1) and only enforced contiguity (raising on a
    non-contiguous tensor). The next version either makes share_mem=True a true
    view or removes the flag; do not rely on it for aliasing until then.

Returns
-------
numpy.ndarray
    A copy of the tensor's elements (see share_mem).
```

### `real` / `imag`

```
Tensor.real() -> Tensor          # real part      (COPY; complex-only)
Tensor.imag() -> Tensor          # imaginary part (COPY; complex-only)

Return the real / imaginary part of a complex tensor.

Returns
-------
Tensor
    An independent COPY (same_data() is False, finding T-E3) — mutating it does
    not affect the source.

Raises
------
RuntimeError
    If the tensor is not complex (Type.ComplexDouble / Type.ComplexFloat).
```

### `fill` / `append`

```
Tensor.fill(val)   -> None       # set EVERY element (in place)
Tensor.append(val) -> None       # grow axis 0 (in place)

In-place writers. Both mutate the receiver and return None; NEITHER carries a
trailing `_` (an established exception — there is no pure counterpart, findings
T-E7/E8).

`fill` sets every element to `val` (all 11 element dtypes). `append` grows the
tensor along AXIS 0 by `val` — a scalar (rank-1 tensor), a Tensor of the
trailing shape, or a Storage — forcing contiguity.

Parameters
----------
val : scalar (fill / append) or Tensor or Storage (append)
    The fill value, or the block to append along axis 0.

Returns
-------
None
    In place.
```

### `__getitem__` / `__setitem__`

```
Tensor.__getitem__(locators)      -> Tensor   # t[...]        (READ -> independent COPY)
Tensor.__setitem__(locators, rhs) -> None      # t[...] = rhs  (ASSIGN -> in place)

Index / slice a tensor with numpy-style locators.

READ / WRITE ASYMMETRY (findings T-E4/E5): a slice READ returns an INDEPENDENT
COPY (the C++ get() "does not share memory") — NOT a numpy-style view — while a
slice / element ASSIGN writes the tensor's storage IN PLACE. Numpy code that
mutates t[0:2] expecting to touch t silently operates on a copy; use element
assignment (t[i, j] = v) or storage() for in-place access.

These operators are the ONLY Python surface for the C++ accessor methods
get / set.

Parameters
----------
locators : int, slice, or tuple thereof
    numpy-style index / slice per axis.
rhs : Tensor or scalar
    The values to assign (__setitem__).

Returns
-------
Tensor (getitem, an independent copy) / None (setitem, in place)
```

### R.2b C++ API (Doxygen)

C++ already provides the templated `item<T>`/`fill<T>`/`append<T>`, the
`real`/`imag` members, and the `get`/`set` accessors. The next version must
(1) make the Python `numpy` `share_mem=True` a true view or drop the flag
(T-E1), and (2) **delete the stray `std::cout`** in the `__getitem__` bare-slice
branch (T-E9). There is no C++ `numpy` member — it is a pure pybind bridge.

```cpp
/**
 * @brief Extract the sole element of a 1-element tensor as a typed scalar.
 * @details item<T> errors "can only be called from a Tensor with only one
 *          element" on a multi-element tensor (finding T-E6). The Python item()
 *          lambda dispatches all 11 element dtypes.
 * @return the element as T.
 */
template <class T> T &item();
const Scalar::Sproxy item() const;

/**
 * @brief The tensor's backing Storage — a SHARED-DATA handle.
 * @details Returns a reference to the tensor's storage; mutating it changes the
 *          tensor (finding T-E2). Use Storage::clone() for an independent copy.
 * @return reference to the tensor's Storage.
 */
Storage &storage() const;

/**
 * @brief Real / imaginary part of a complex tensor (a COPY).
 * @details Materialize a fresh real-typed tensor; the result does NOT share
 *          memory with the source (finding T-E3). Precondition: the tensor is
 *          Type.ComplexDouble or Type.ComplexFloat, else it raises.
 * @return a new (real-typed) Tensor.
 */
Tensor real();
Tensor imag();

/**
 * @brief In-place writers: set every element / grow along axis 0.
 * @details fill<T> sets all elements to @p val; append stores @p val at the end
 *          of axis 0 (scalar / Tensor / Storage overloads), forcing contiguity.
 *          BOTH mutate *this and are void — no trailing `_` (findings T-E7/E8).
 * @param val the fill value / the block to append.
 */
template <class T> void fill(const T &val);
void append(const Tensor &rhs);
void append(const Storage &srhs);
template <class T> void append(const T &rhs);

/**
 * @brief Slice-get / slice-set implementation (behind Python [] operators).
 * @details get(accessors) returns a NEW Tensor that "does not share memory with
 *          the current Tensor" — a COPY (finding T-E4), NOT a numpy-style view;
 *          set(accessors, rhs) writes @p rhs into the sliced region IN PLACE and
 *          returns *this (finding T-E5). Python reaches these ONLY via
 *          __getitem__/__setitem__. NOTE: the Python __getitem__ bare-1-D-slice
 *          branch currently leaks a stray std::cout debug line — delete it
 *          (finding T-E9).
 * @param accessors per-axis Accessor list. @param rhs the values to assign.
 * @return get: a new (independent) Tensor. set: reference to *this.
 */
Tensor get(const std::vector<cytnx::Accessor> &accessors) const;
Tensor &set(const std::vector<cytnx::Accessor> &accessors, const Tensor &rhs);
```
