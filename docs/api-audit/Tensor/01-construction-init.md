# Tensor — 01. Construction & initialization

> **Superset-method rollout** (Tensor, category 01 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings) and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` construction API should be implemented to match §R exactly.
> Every behavioral claim is verified against the installed `cytnx==1.1.0` wheel
> by `docs/api-audit/probes/Tensor_01_construction.py` (all `[PASS]`, exit 0).
> This category has **no binding-fidelity finding**, so it carries no raw-C++
> probe (see A3).

**Category scope:** the ways to *make* a `cytnx.Tensor` from existing data — the
three constructor overloads, the public `Init` re-initializer, and the
`from_storage` static factory. Static generators (`zeros`/`ones`/`arange`/…) are
free functions, not `Tensor` members, and are out of this class's scope;
I/O constructors (`Load`/`Fromfile`) are [category 08](08-io.md).

---

# Analysis

**Provenance:** live pybind signatures from the 1.1.0 wheel; C++ from
`cytnx_src/include/Tensor.hpp` and `cytnx_src/pybind/tensor_py.cpp`
(`py::init<>` at `:146-151`, `Init` at `:152-154`, `from_storage` lambda at
`:301-311`).

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `Tensor()` | `Tensor()` | `Tensor` (`__init__`→`None`) | Empty, un-initialized (`Void`) rank-0 tensor. Probe: *"Tensor() is an un-initialized (Void) rank-0 tensor"*. |
| `Tensor(other)` | `Tensor(other: Tensor)` | `Tensor` — **view of `other`** | Copy constructor; **shares `other`'s storage** (C++ `_impl = rhs._impl`, `:398`) (B2). Probe: *"Tensor(other) copy-constructor SHARES storage …"*. |
| `Tensor(shape,…)` | `Tensor(shape: list[int], dtype=3, device=-1, init_zero=True)` | `Tensor` — owns storage | Allocate from a shape; defaults `dtype=Type.Double`, `device=Device.cpu`, zero-filled. Probe: *"Tensor(shape) defaults to dtype Double / device cpu and zero-fills"*. |
| `Init` | `Init(shape: list[int], dtype=3, device=-1, init_zero=True)` | `None` (in-place re-init) | **Public** in-place re-init; duplicates the *shape* constructor (only that overload). Probe: *"Init is public and re-initializes in place, returning None …"*. |
| `from_storage` (static) | `from_storage(sin: Storage, is_clone=False)` | `Tensor` | Wrap a `Storage`; **default shares its buffer** (view), `is_clone=True` copies. Probe: *"from_storage(sin) DEFAULT shares the Storage's buffer …"* / *"is_clone=True COPIES …"*. |

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `Python-only`.

| C++ (`Tensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `Tensor()` (`:397`) | `Tensor()` | identical | empty/Void |
| `Tensor(const Tensor &rhs)` (`:398`) | `Tensor(other)` | identical | copy ctor: `_impl = rhs._impl` → shares storage (B2) |
| `Tensor(vector<uint64> shape, dtype, device, init_zero)` (`:476-480`) | same, kwargs | identical | owns fresh storage; defaults Double/cpu/zero |
| `void Init(shape, dtype, device, init_zero)` (`:446-451`) | `Init(...)` | identical | public in both — duplicates the shape ctor |
| `static Tensor from_storage(const Storage&)` (`:558-564`) | `from_storage(sin, is_clone=False)` | signature-differs | binding **adds** the Python-only `is_clone` kwarg (lambda `:301-311`); with `is_clone=False` it faithfully mirrors the C++ share |
| `Tensor &operator=(const Tensor&)` (`:410`) | — | C++-only | copy-assign plumbing; no faithful Python peer |
| `Tensor(initializer_list<Tp>)` (`:402`) | — | C++-only | brace-init convenience; not bound |

## A3. Findings

Behavioral claims quote a probe assertion (1.1.0 wheel). A **(binding
fidelity)** finding would flag where the binding layer changes behavior versus
the raw C++ method — but this category has **none**: the three constructors are
thin `py::init<>` pass-throughs, `Init` binds the C++ `Init` directly, and
`from_storage`'s lambda only *adds* an `is_clone` convenience kwarg whose default
(`False`) calls the raw C++ `from_storage(sin)` unchanged. So C++ and Python
construction behavior coincide, and **no raw-C++ probe is warranted** (gate 4
skipped, recorded here).

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-C1** | `Init` is a public duplicate of the shape constructor | redundancy | **thin pass-through** — `Init` (`tensor_py.cpp:152-154`) rebuilds the impl and re-runs the shape ctor logic in place, returning `None`. Unlike UniTensor's `Init` it has *only* the shape overload. Probe *"Init is public and re-initializes in place, returning None …"* | demote to private `_init` (deprecate 1 release) |
| **T-C2** | copy-ctor `Tensor(other)` shares memory (view); `Tensor(shape)` owns fresh storage | copy/view (B2) | **thin pass-through** — the C++ copy ctor is `_impl = rhs._impl` (`:398`), an intrusive-ptr alias, so the copy shares storage; `py::init<const Tensor&>` binds it faithfully. Py probe *"Tensor(other) copy-constructor SHARES storage …, same_data() is True"* | document per-constructor |
| **T-C3** | `from_storage` default **shares** the Storage buffer (view); `is_clone=True` copies | copy/view (B2) + Python-only kwarg | the raw C++ `from_storage` always shares (`:558-564`); the pybind lambda (`:301-311`) *adds* `is_clone`, cloning `sin` before wrapping when `True`. Default `False` mirrors C++. Probe *"from_storage(sin) DEFAULT shares …"*, *"is_clone=True COPIES …"*, *"exposes the Python-binding-only `is_clone` kwarg"* | keep; document view/copy + `is_clone` |
| **T-C4** | constructor metadata (`dtype, device, init_zero`) is **positional**, so any reorder silently breaks positional callers | ordering — naming/order/compat (see A4) | **thin pass-through** — the `py::init`/`Init` arg order mirrors the C++ default-arg order. Live signatures | make the metadata block **keyword-only** (§R.0) |

## A4. Argument ordering — positional & keyword

Both orderings are checked; here the positional side is trivial and the keyword
side is the whole problem.

- **Positional:** each constructor / factory has exactly **one** primary
  operand — `other` (Tensor) or `shape` (list[int]) for the constructors,
  `sin` (Storage) for `from_storage`, `shape` for `Init`. So positional order
  is vacuously canonical (`[primary operand]`, nothing after it). No positional
  inconsistency exists in this category.
- **Keyword:** everything else is metadata: `dtype, device, init_zero` on the
  shape constructor and `Init`, and `is_clone` on `from_storage`. These are
  currently **positional** (T-C4). The canonical fix makes each metadata block
  **keyword-only**, so no positional caller can break on a future reorder.

**Canonical orders (normative — see §R.0):**
- Positional: `[primary operand]` only (`other` | `shape` | `sin`).
- Keyword-only: `dtype, device, init_zero` (shape ctor / `Init`); `is_clone`
  (`from_storage`).

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
construction API. Implement Cytnx to match it. Findings above are the rationale;
they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).**
  *Member* functions lowercase snake_case; *free* functions *acting on* objects
  Capitalized; *free* creators lowercase; *types* Capitalized. The constructor is
  the type name `Tensor` (Capitalized — correct, Rule 3). `from_storage` is
  correct lowercase snake_case. `Init` is a Capitalized *member* (violates the
  member rule) and is demoted to `_init` regardless (T-C1); were it kept public
  it would be `init`.

*Two orderings are also normative — the positional operand and the keyword block.*

- **N-positional — canonical positional order is `[primary operand]` only.**
  Each constructor / factory takes exactly one required positional — the data
  source (`other` | `shape` | `sin`). No optional positional args; everything
  else is keyword-only (see N-kwonly).
- **N-kwonly — the metadata block is keyword-only.** `dtype, device, init_zero`
  (shape ctor / `Init`) and `is_clone` (`from_storage`) are **keyword-only**
  (Python `*` separator), declared in canonical order. This removes argument
  *order* from the public contract (so T-C4's positional fragility cannot recur)
  and prevents silent positional-caller breakage.
- **N-underscore — a trailing `_` marks in-place; its absence marks pure.**
  An `_`-suffixed name mutates in place and returns `self`; the un-suffixed name
  returns a new object (B1). This category has no `_`/pure pairs — but note that
  `Init` is an in-place re-initializer that is *both* Capitalized (violates
  N-casing) *and* missing the `_` marker (violates N-underscore); it is demoted
  to `_init` regardless (T-C1). `from_storage` is a pure factory (returns a new
  tensor), so correctly carries no `_`.
- **N-view — copy/view behavior is fixed and documented.** `Tensor(other)`
  shares `other`'s memory (a view); `from_storage(sin)` shares `sin`'s buffer by
  default and copies it when `is_clone=True`; the shape constructor and `Init`
  allocate independent storage.
- **Defaults:** `dtype=Type.Double`, `device=Device.cpu`, `init_zero=True`,
  `is_clone=False`.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class Tensor:
    def __init__(self) -> None: ...
    def __init__(self, other: Tensor) -> None: ...
    def __init__(self, shape: list[int], *, dtype: int = Type.Double,
                 device: int = Device.cpu, init_zero: bool = True) -> None: ...

    @staticmethod
    def from_storage(sin: Storage, *, is_clone: bool = False) -> "Tensor": ...
```

| API | Verdict | Behavior contract |
|---|---|---|
| `Tensor()` | **keep** | Empty, un-initialized (`Void`) rank-0 tensor. |
| `Tensor(other, *)` | **keep** | **View:** shares `other`'s storage (copy ctor). Clone `other` first for an independent copy. |
| `Tensor(shape, *, …)` | **keep** (kwargs now keyword-only) | **Owns** fresh storage; zero-filled iff `init_zero`; dtype/device from the kwargs. |
| `from_storage(sin, *, is_clone=False)` | **keep** | Wrap a `Storage`. Default **shares** `sin`'s buffer (view); `is_clone=True` **copies** it. |
| `Init(...)` | **remove from public API** (T-C1) | Demote to private `_init`. **Migration:** ship a thin public `Init` emitting `DeprecationWarning` for one minor release, then delete. Prefer the `Tensor(shape, …)` constructor. |

No constructor or factory is deleted; only reordered-to-keyword-only and `Init`
demoted.

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface.

### R.2a Python API (numpy-style)

### `Tensor.__init__`

```
Tensor()
Tensor(other)
Tensor(shape, *, dtype=Type.Double, device=Device.cpu, init_zero=True)

Construct a Tensor. The metadata arguments are keyword-only.

Parameters
----------
other : Tensor, optional
    Copy-construct from an existing Tensor. The result SHARES `other`'s storage
    — in-place edits to either are visible in the other (call `other.clone()`
    first for an independent copy). dtype/device are inherited from `other`.
shape : sequence of int, optional
    Extent of each axis; allocates fresh, independent storage. Mutually
    exclusive with `other`.
dtype : int, keyword-only
    Element type (shape form only), a cytnx.Type code; default Type.Double.
device : int, keyword-only
    Storage device (shape form only), a cytnx.Device code; default Device.cpu.
init_zero : bool, keyword-only
    Zero-fill the content (shape form only); default True.

Notes
-----
No-arg form creates an empty (Void) rank-0 tensor. Metadata is keyword-only so
its ordering is not part of the API contract (finding T-C4). See enums.md for
the Type / Device codes.
```

### `Tensor.from_storage`

```
Tensor.from_storage(sin, *, is_clone=False) -> Tensor

Build a Tensor from a Storage, inheriting its dtype and device.

Parameters
----------
sin : Storage
    The source buffer. Its length becomes a rank-1 Tensor's shape.
is_clone : bool, keyword-only
    Default False -> the Tensor SHARES `sin`'s buffer (a view: mutating either
    is seen in the other, confirmed by probe). True -> the Tensor gets an
    independent COPY of `sin`.

Returns
-------
Tensor
    A dense rank-1 Tensor over `sin`'s elements.

See Also
--------
Tensor.storage : the inverse view onto a Tensor's Storage (category 03).
```

`Init` is demoted to private `_init` and carries no public docstring.

### R.2b C++ API (Doxygen)

C++ has no keyword-only parameters, so the metadata are default arguments in the
canonical order. The Python-only `is_clone` kwarg has no C++ peer — the C++
`from_storage` always shares; `Init` is demoted — no public doc.

```cpp
/// @brief Construct an empty, un-initialized (Void) rank-0 Tensor.
Tensor();

/**
 * @brief Copy-construct a Tensor; SHARES the source's storage (view).
 * @param other  source Tensor; the result aliases its reference-counted
 *               storage (dtype/device inherited). Clone @p other first for an
 *               independent copy.
 */
Tensor(const Tensor &other);

/**
 * @brief Allocate a Tensor from a shape; owns fresh storage.
 * @param shape     extent of each axis.
 * @param dtype     element type, e.g. Type::Double (default).
 * @param device    storage device, e.g. Device::cpu (default).
 * @param init_zero zero-fill the content; default true.
 */
Tensor(const std::vector<cytnx_uint64> &shape,
       const unsigned int &dtype = Type.Double, const int &device = -1,
       const bool &init_zero = true);

/**
 * @brief Build a Tensor from a Storage; SHARES its buffer (view).
 * @param in  source Storage; dtype/device inherited. The result aliases @p in's
 *            buffer. (The Python binding adds an `is_clone` flag to copy instead.)
 */
static Tensor from_storage(const Storage &in);
```
