# Tensor — 02. Metadata & introspection

> **Superset-method rollout** (Tensor, category 02 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings) and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` metadata surface should be implemented to match §R exactly.
> Every behavioral claim is verified against the installed `cytnx==1.1.0` wheel
> by `docs/api-audit/probes/Tensor_02_metadata.py` (all `[PASS]`, exit 0).
> This category has **no binding-fidelity finding** — the accessors are thin
> pass-throughs — so it carries no raw-C++ probe (gate 4, see A3).

**Category scope:** the read-only *metadata / introspection* accessors — the
query methods that report a `Tensor`'s structure (`shape`, `rank`), its
element/storage identity (`dtype`, `dtype_str`, `device`, `device_str`), and its
storage predicates (`is_contiguous`, `same_data`). None mutate the tensor's
data. Element/storage *access* (`item`, `storage`, `numpy`, `real`, `imag`) is
[category 03](03-element-storage-access.md); shape *manipulation*
(`permute`/`reshape`/`contiguous`/`flatten`) is [category 04](04-shape-layout.md);
constructors are [category 01](01-construction-init.md).

---

# Analysis

**Provenance:** live pybind signatures from the 1.1.0 wheel
(`tools/member_inventory.py Tensor`); C++ from `cytnx_src/include/Tensor.hpp`
(`dtype`/`device`/`dtype_str`/`device_str`/`shape`/`rank` at `:571,578,585,592,598,615`,
`is_contiguous` at `:689`, `same_data` at `:1607`) and
`cytnx_src/pybind/tensor_py.cpp` (accessor `.def`s at `:155-160,178,189`).

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. All are member accessors on `self`.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `shape` | `shape()` | `list[int]` | Extent of each axis. Probe: *"shape() lists the extent of each axis ([2, 3, 4])"*. |
| `rank` | `rank()` | `int` | Number of axes (== `len(shape())`). Probe: *"rank() is the number of axes and equals len(shape()) (3)"*. |
| `dtype` | `dtype()` | `int` | Element-type code (a `cytnx.Type`, see `enums.md`). Probe: *"dtype() returns the int Type code Type.Double (3)"*. |
| `dtype_str` | `dtype_str()` | `str` | Human-readable dtype, `"Double (Float64)"`. Probe: *"dtype_str() names the element type …"*. |
| `device` | `device()` | `int` | Device code (a `cytnx.Device`, see `enums.md`). Probe: *"device() returns the int Device code Device.cpu (-1)"*. |
| `device_str` | `device_str()` | `str` | Human-readable device, `"cytnx device: CPU"`. Probe: *"device_str() names the device …"*. |
| `is_contiguous` | `is_contiguous()` | `bool` | Predicate: element storage is contiguous. Probe: *"is_contiguous() is True for a freshly-built … tensor"* / *"… is False after a non-contiguous permute"*. |
| `same_data` | `same_data(arg0: Tensor)` | `bool` | Predicate: do two tensors share storage? Probe: *"same_data(self) is True …"* / *"same_data(clone()) is False …"* / *"same_data(other=...) is REJECTED (PC1)"*. |

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `Python-only`.

| C++ (`Tensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `const vector<cytnx_uint64>& shape() const` (`:598`) | `shape()` → `list[int]` | identical | binding copies the vector out to a Python list; scalar metadata pass-through |
| `cytnx_uint64 rank() const` (`:615`) | `rank()` | identical | `= shape().size()` |
| `unsigned int dtype() const` (`:571`) | `dtype()` | identical | Type code |
| `std::string dtype_str() const` (`:585`) | `dtype_str()` | identical | string form |
| `int device() const` (`:578`) | `device()` | identical | Device code |
| `std::string device_str() const` (`:592`) | `device_str()` | identical | string form |
| `const bool& is_contiguous() const` (`:689`) | `is_contiguous()` | identical | bool predicate |
| `bool same_data(const Tensor& rhs) const` (`:1607`) | `same_data(arg0)` | identical | arg name erased to `arg0` (T-M1) |

## A3. Findings

Behavioral claims quote a probe assertion (1.1.0 wheel). A **(binding
fidelity)** finding would flag where the binding layer changes behavior versus
the raw C++ method — this category has **none**: every accessor is a thin
`.def(&cytnx::Tensor::…)` (`tensor_py.cpp:155-160,178,189`) that returns exactly
what the C++ `const` method returns; no forwarding lambda, no default-arg or
short-circuit logic sits between the two. **Gate 4 (C++ probe) is therefore
skipped for category 02.** The one finding is a Python-side signature blemish
(`same_data`'s erased `arg0`), not a behavior divergence.

Contrast with UniTensor's metadata category (UT-M1/M2): those carried N-casing
renames (`Nblocks`, `getTotalQnums`). Tensor's metadata surface is **already
clean** — every member is lowercase snake_case (`is_contiguous`/`same_data`
correctly formed), so this category has *no* casing finding.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-M1** | `same_data` erases its argument name to `arg0` | naming (parameter consistency, PC1) | **binding drops the `py::arg`** — `.def("same_data", &cytnx::Tensor::same_data)` (`tensor_py.cpp:189`) registers no `py::arg(...)`, so the operand is positional-only. Probe: *"same_data(other=...) is REJECTED (PC1)"* | add a real `py::arg("other")` so it is keyword-callable |
| **T-M2** | `is_contiguous` / `same_data` are correctly-formed boolean predicates | naming (N-casing / N5) — **conformant** | **thin pass-throughs** — `is_contiguous` (`:178`) is `is_`-prefixed; `same_data` (`:189`) is a `snake_case` share-query. Probe: *"is_contiguous() is True … / False after a permute"*, *"same_data(self) is True / same_data(clone()) is False"* | **keep both** — canonical predicates; document the contiguity contract and the view/copy oracle |
| **T-M3** | `same_data` is the view-vs-copy oracle: True iff two tensors share storage | copy/view (B2) | **thin pass-through** — reflects the C++ `_impl` intrusive-ptr identity; `same_data(permute)` is True (permute is a view), `same_data(clone())` is False (clone allocates fresh storage). Probe: *"same_data(permute view) is True"*, *"same_data(clone()) is False"* | keep; document as the oracle used by every copy/view finding across cats 03–07 |

## A4. Argument ordering — positional & keyword

Metadata accessors are all **nullary** (`self` only) except `same_data`, which
takes exactly **one** operand:

| Accessor | operand(s) | issue |
|---|---|---|
| `shape` / `rank` / `dtype` / `dtype_str` / `device` / `device_str` / `is_contiguous` | *(none)* | nullary — ordering vacuous |
| `same_data` | `other: Tensor` | name erased to `arg0` (T-M1) |

- **Positional:** the only accessor with an operand, `same_data`, takes it as the
  single required positional (`other`) — canonical and trivial. No positional
  inconsistency exists in this category.
- **Keyword:** the fix is not *order* but *name* (T-M1) — restore the erased
  `arg0` name so the operand is callable by keyword.

**Canonical rule (normative — see §R.0):** one named required positional operand
where applicable (`same_data(other)`), never `arg0`; the rest are nullary.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
metadata-accessor surface. Implement Cytnx to match it. Findings above are the
rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).**
  *Member* functions are lowercase snake_case. **Every accessor in this category
  already conforms** — `shape`, `rank`, `dtype`, `dtype_str`, `device`,
  `device_str`, `is_contiguous`, `same_data` are all correct (contrast
  UniTensor's `Nblocks`/`getTotalQnums`, which needed renames). No rename is
  recommended here.
- **N5 — boolean predicates read as questions.** `is_contiguous` is correctly
  `is_`-prefixed; `same_data` is a well-formed share-query. Both are kept
  verbatim (T-M2).
- **N-argname — no erased `arg0` parameters.** `same_data`'s operand carries a
  real, keyword-callable name `other` (T-M1); today the binding drops it to
  `arg0`, so `same_data(other=…)` is rejected.
- **N-view — copy/view behavior is fixed and documented.** `same_data(other)` is
  the storage-identity oracle: True iff `self` and `other` alias the same memory
  (a `permute`/`reshape` view is True; a `clone()` is False) — T-M3. The
  accessors themselves are read-only and share nothing.
- **N-underscore — no in-place accessors here.** Every member is a pure query;
  none mutates, so none carries a trailing `_`.

*The positional/keyword rule is also normative.*

- **N-positional — one named operand where applicable.** `same_data` takes its
  single operand as a named required positional (`other`); all other accessors
  are nullary.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class Tensor:
    # --- structure ---
    def shape(self) -> list[int]: ...
    def rank(self) -> int: ...
    # --- element / storage identity ---
    def dtype(self) -> int: ...
    def dtype_str(self) -> str: ...
    def device(self) -> int: ...
    def device_str(self) -> str: ...
    # --- storage predicates ---
    def is_contiguous(self) -> bool: ...
    def same_data(self, other: "Tensor") -> bool: ...   # was arg0 (T-M1)
```

| API | Verdict | Behavior contract |
|---|---|---|
| `shape` | **keep** | Extent of each axis, in axis order. |
| `rank` | **keep** | Number of axes; equals `len(shape())`. |
| `dtype` | **keep** | Element-type code (a `cytnx.Type`, e.g. `Type.Double == 3`). |
| `dtype_str` | **keep** | Human-readable dtype, e.g. `"Double (Float64)"`. |
| `device` | **keep** | Device code (a `cytnx.Device`, e.g. `Device.cpu == -1`). |
| `device_str` | **keep** | Human-readable device, e.g. `"cytnx device: CPU"`. |
| `is_contiguous` | **keep** | True iff element storage is contiguous; False after a non-contiguous `permute` (T-M2). |
| `same_data` | **keep** (name `arg0`→`other`, T-M1) | True iff `self` and `other` share storage — the view-vs-copy oracle (T-M3). |

No accessor is renamed or deleted; the sole change is restoring `same_data`'s
erased parameter name.

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface.

### R.2a Python API (numpy-style)

### structure & identity accessors (`shape`, `rank`, `dtype`, `dtype_str`, `device`, `device_str`)

```
Tensor.shape() -> list[int]     # extent of each axis
Tensor.rank() -> int            # number of axes (== len(shape()))
Tensor.dtype() -> int           # element-type code (e.g. Type.Double == 3)
Tensor.dtype_str() -> str       # element-type name, "Double (Float64)"
Tensor.device() -> int          # device code (e.g. Device.cpu == -1)
Tensor.device_str() -> str      # device name, "cytnx device: CPU"

Read-only metadata accessors. None mutate the tensor.

Returns
-------
list[int] / int / str
    The requested metadata. `rank()` == len(`shape()`) (confirmed by probe).
    `dtype()`/`device()` are cytnx.Type / cytnx.Device codes; the `_str` forms
    are their human strings. See enums.md for the code tables.
```

### `Tensor.is_contiguous`

```
Tensor.is_contiguous() -> bool

Whether the tensor's element storage is laid out contiguously.

Returns
-------
bool
    True for a freshly constructed or `contiguous()`-ed tensor; False for a
    non-contiguous view — e.g. the result of `permute` reorders the strides
    without moving data, so `t.permute(...).is_contiguous()` is False
    (confirmed by probe). Follow with `contiguous()` to materialize.
```

### `Tensor.same_data`

```
Tensor.same_data(other) -> bool

Whether two tensors share the same underlying storage. Parameter renamed from
the erased `arg0` (finding T-M1) so it is keyword-callable.

Parameters
----------
other : Tensor
    The tensor to compare storage identity against.

Returns
-------
bool
    True iff `self` and `other` alias the same memory. This is the
    view-vs-copy oracle: True for a `permute`/`reshape` view of the source,
    False for `clone()` (which allocates independent storage) — both confirmed
    by probe. See the copy-vs-view table (category 03 / finding T-M3).
```

### R.2b C++ API (Doxygen)

The accessors are `const` members returning by value or by `const` reference;
`same_data`'s parameter is named (not `arg0`; T-M1). None takes a default
argument, so there is nothing keyword-only to mirror.

```cpp
/// @brief Structure: per-axis extents and the axis count.
const std::vector<cytnx_uint64> &shape() const;   ///< extent of each axis
cytnx_uint64 rank() const;                         ///< == shape().size()

/// @brief Element-type / device codes and their human string forms.
unsigned int dtype() const;        std::string dtype_str() const;
int device() const;                std::string device_str() const;

/// @brief Storage predicate: is the element buffer contiguous?
/// @return false after a non-contiguous permute (a strided view); true once
///         contiguous() has materialized it (finding T-M2).
const bool &is_contiguous() const;

/**
 * @brief Storage-identity predicate — do two tensors share memory?
 * @param other another tensor (parameter named, not arg0; finding T-M1).
 * @return true iff @p self and @p other alias the same storage — the
 *         view-vs-copy oracle: a permute/reshape view is true, a clone() is
 *         false (finding T-M3).
 */
bool same_data(const Tensor &other) const;
```
