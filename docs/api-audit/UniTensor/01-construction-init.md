# UniTensor — 01. Construction & initialization

> **Superset-method pilot** (with [`02-static-generators.md`](02-static-generators.md)).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings) and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx*: the next major version's
> `UniTensor` construction API should be implemented to match §R exactly.
> Every behavioral claim is verified against the installed `cytnx==1.1.0` wheel
> by `docs/api-audit/probes/UniTensor_cat01_02.py` (all `[PASS]`, exit 0);
> raw-C++ facts are additionally verified by `probes/cpp/UniTensor_cat01_02.cpp`
> against a locally source-built `libcytnx`.

**Category scope:** the ways to *make* a `UniTensor` from existing data — the
three constructor overloads and the public `Init` re-initializer. Static
generators (`zeros`/`ones`/`normal`/…) are [category 02](02-static-generators.md).

---

# Analysis

**Provenance:** live pybind signatures from the 1.1.0 wheel; C++ from
`cytnx_src/include/UniTensor.hpp` and `cytnx_src/pybind/unitensor_py.cpp`
(`py::init<>` at `:204-216`, `Init` lambdas at `:217-226`).

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `UniTensor()` | `UniTensor()` | `UniTensor` (`__init__`→`None`) | Empty, un-initialized (`Void`) rank-0 tensor. Probe: *"…un-initialized (Void) rank-0 tensor"*. |
| `UniTensor(Tin,…)` | `UniTensor(Tin: Tensor, is_diag=False, rowrank=-1, labels=[], name='')` | `UniTensor` — **view of `Tin`** | Wrap a dense `Tensor`; **shares its memory** (B2). Probe: *"UniTensor(Tensor) shares memory …"*. |
| `UniTensor(bonds,…)` | `UniTensor(bonds: List[Bond], labels=[], rowrank=-1, dtype=3, device=-1, is_diag=False, name='')` | `UniTensor` — owns storage | Build from `Bond`s; defaults `dtype=Type.Double`, `device=Device.cpu`. Probe: *"UniTensor(bonds) defaults to dtype Double / device cpu"*. |
| `Init` | `Init(Tin,…)` / `Init(bonds,…)` — same two overloads as the constructors | `None` (in-place re-init) | **Public** in-place re-init. Probe: *"Init is public and re-initializes …"*. |

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `Python-only`.

| C++ (`UniTensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `UniTensor()` | `UniTensor()` | identical | empty/Void |
| `UniTensor(Tensor, is_diag, rowrank, labels, name)` | same, kwargs | identical | shares the Tensor's memory (B2) |
| `UniTensor(vector<Bond>, labels, rowrank, dtype, device, is_diag, name)` | same, kwargs | identical | deep-copies the bonds |
| `Init(Tensor,…)` / `Init(bonds,…)` | `Init(...)` ×2 | identical | public in both — duplicates the constructor |
| `operator=(const UniTensor&)` | — | C++-only | copy-assign plumbing; no faithful Python peer |
| *(numpy bridge)* | — | *(absent both)* | no `from_numpy` (A-finding UT-C3) |

## A3. Findings

Behavioral claims quote a probe assertion (1.1.0 wheel). A **(binding
fidelity)** finding would flag where the binding layer changes behavior versus
the raw C++ method — but this category has **none**: the constructors are thin
`py::init<>` pass-throughs and `Init`'s lambda faithfully mirrors the
constructor, so C++ and Python behavior coincide here.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-C1** | `Init` is a public duplicate of the constructor | redundancy | **thin pass-through** — the `Init` lambda (`:217-226`) re-runs the constructor logic in place. Probe *"Init is public and re-initializes …"* | demote to private `_init` (deprecate 1 release) |
| **UT-C2** | from-`Tensor` shares memory (view); from-`bonds` owns fresh storage | copy/view (B2) | **thin pass-through** — `py::init<>` wraps the C++ ctors faithfully; the from-`Tensor` memory sharing is C++ behavior, not a binding artifact. Py probe *"UniTensor(Tensor) shares memory …"*; **C++ probe confirms** the share on the raw C++ side | document per-constructor |
| **UT-C3** | no numpy bridge (`from_numpy` / `.numpy()`) | capability gap (vs `Tensor`) | **nothing to bind** — `from_numpy` is absent in C++ too, so this is an add, not an unbound member. Source | **add** `from_numpy` |
| **UT-C4** | constructor metadata order inconsistent (`is_diag` 1st↔6th, `labels` 3rd↔1st) *and* **positional**, so any reorder silently breaks positional callers | ordering — naming/order/compat (see A4) | **thin pass-through** — the `py::init` arg order mirrors the C++ ctor; the inconsistency exists in C++ too. Live signatures; F20 / PC2 | make the metadata block **keyword-only** (§R.0) |

## A4. Argument ordering — positional & keyword

Both orderings are checked; here the positional side is trivial and the keyword
side is the whole problem.

- **Positional:** each constructor has exactly **one** primary operand — `Tin`
  (Tensor) or `bonds` (list[Bond]) — so positional order is vacuously canonical
  (`[primary operand]`, nothing after it). `from_numpy` follows the same shape:
  one primary operand `array`. No positional inconsistency exists in this
  category.
- **Keyword:** everything else (`labels, rowrank, is_diag, dtype, device, name`)
  is metadata, currently **positional and inconsistently ordered** (UT-C4). The
  canonical fix makes the whole block **keyword-only** in canonical order, so the
  two constructors no longer disagree and no positional caller can break.

**Canonical orders (normative — see §R.0):**
- Positional: `[primary operand]` only (`Tin` | `bonds` | `array`).
- Keyword-only: `labels, rowrank, is_diag, dtype, device, name`.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `UniTensor`
construction API. Implement Cytnx to match it. Findings above are the rationale;
they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).**
  *Member* functions lowercase snake_case; *free* functions *acting on* objects
  Capitalized; *free* creators lowercase; *types* Capitalized. The constructor is
  the type name `UniTensor` (Capitalized — correct, Rule 3). `Init` is a
  Capitalized *member* (violates the member rule) and is demoted to `_init`
  regardless (UT-C1); were it kept public it would be `init`.

*Two orderings are also normative — the positional operand and the keyword block.*

- **N-positional — canonical positional order is `[primary operand]` only.**
  Each constructor takes exactly one required positional — the data source
  (`Tin` | `bonds` | `array`). No optional positional args; everything else is
  keyword-only (see N-kwonly).
- **N-kwonly — the shared metadata block is keyword-only.** The metadata block
  `labels, rowrank, is_diag, dtype, device, name` is **keyword-only** (Python `*`
  separator), declared in that canonical order and omitting inapplicable members
  without reordering. This removes argument *order* from the public contract (so
  UT-C4's ordering inconsistency cannot recur) and prevents silent
  positional-caller breakage.
- **N-underscore — a trailing `_` marks in-place; its absence marks pure.**
  An `_`-suffixed name mutates in place and returns `self`; the un-suffixed name
  returns a new object (B1). This category has no `_`/pure pairs — but note that
  `Init` is an in-place re-initializer that is *both* Capitalized (violates
  N-casing) *and* missing the `_` marker (violates N-underscore); it is demoted
  to `_init` regardless (UT-C1). `from_numpy` is pure (returns a new tensor), so
  correctly carries no `_`.
- **N-view — copy/view behavior is fixed and documented.** `UniTensor(Tin)`
  shares `Tin`'s memory (a view); all other constructors allocate independent
  storage.
- **Defaults:** `dtype=Type.Double`, `device=Device.cpu`, `rowrank=-1`
  (auto), `is_diag=False`, `labels=[]` (→ `["0","1",…]`), `name=''`.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    def __init__(self) -> None: ...
    def __init__(self, Tin: Tensor, *, labels: list[str] = [], rowrank: int = -1,
                 is_diag: bool = False, name: str = "") -> None: ...
    def __init__(self, bonds: list[Bond], *, labels: list[str] = [], rowrank: int = -1,
                 is_diag: bool = False, dtype: int = Type.Double,
                 device: int = Device.cpu, name: str = "") -> None: ...

    @staticmethod
    def from_numpy(array, *, labels: list[str] = [], rowrank: int = -1,
                   name: str = "") -> "UniTensor": ...
```

| API | Verdict | Behavior contract |
|---|---|---|
| `UniTensor()` | **keep** | Empty, un-initialized (`Void`) rank-0 tensor. |
| `UniTensor(Tin, *, …)` | **keep** (kwargs now keyword-only; `dtype`/`device` dropped — inherited from `Tin`) | **View:** shares `Tin`'s memory. Legs from `Tin.shape()`. |
| `UniTensor(bonds, *, …)` | **keep** (kwargs keyword-only; `is_diag` restored to canonical slot) | **Owns** fresh storage; symmetric/tagged iff the bonds carry qnums/directions. |
| `UniTensor.from_numpy(array, *, …)` | **add** (UT-C3) | Dense, untagged; **copies** `array`; dtype mapped from `array.dtype`. Inverse of `.numpy()` (category 12). |
| `Init(...)` | **remove from public API** (UT-C1) | Demote to private `_init`. **Migration:** ship a thin public `Init` emitting `DeprecationWarning` for one minor release, then delete. |

No constructor is deleted; only reordered-to-keyword-only and `Init` demoted.

## R.2 Docstrings (normative, numpy-style)

### `UniTensor.__init__`

```
UniTensor()
UniTensor(Tin, *, labels=[], rowrank=-1, is_diag=False, name='')
UniTensor(bonds, *, labels=[], rowrank=-1, is_diag=False,
          dtype=Type.Double, device=Device.cpu, name='')

Construct a UniTensor. The metadata arguments are keyword-only.

Parameters
----------
Tin : Tensor, optional
    Wrap an existing dense Tensor. The result SHARES `Tin`'s memory — in-place
    edits to either are visible in the other (clone `Tin` first for an
    independent copy). dtype/device are inherited from `Tin`.
bonds : list of Bond, optional
    One Bond per leg; defines rank, shape, and symmetry. The bonds are
    deep-copied and the tensor owns fresh storage. Mutually exclusive with `Tin`.
labels : list of str, keyword-only
    Leg labels; defaults to ["0", "1", ...].
rowrank : int, keyword-only
    Legs forming the row (bra) space; -1 auto-selects.
is_diag : bool, keyword-only
    Store only the diagonal (rank-2, square).
dtype : int, keyword-only
    Element type (bonds form only), e.g. Type.Double.
device : int, keyword-only
    Storage device (bonds form only), e.g. Device.cpu.
name : str, keyword-only
    Human-readable tensor name.

Raises
------
ValueError
    If len(labels) != number of legs, or rowrank is out of range.

Notes
-----
No-arg form creates an empty (Void) rank-0 tensor. Metadata is keyword-only so
its ordering is not part of the API contract (finding UT-C4).
```

### `UniTensor.from_numpy` (new)

```
UniTensor.from_numpy(array, *, labels=[], rowrank=-1, name='') -> UniTensor

Create a dense UniTensor from a NumPy array (copies the data).

Parameters
----------
array : numpy.ndarray
    Source array; its shape defines the legs. dtype maps from `array.dtype`.
labels : list of str, keyword-only
    Leg labels; defaults to ["0", "1", ...].
rowrank : int, keyword-only
    Row-space leg count; -1 auto-selects.
name : str, keyword-only
    Tensor name.

Returns
-------
UniTensor
    A Dense, untagged tensor.

See Also
--------
UniTensor.numpy : inverse conversion (dense only).
```

`Init` is demoted to private `_init` and carries no public docstring.
