# UniTensor — 02. Static generators

> **Superset-method pilot** (with [`01-construction-init.md`](01-construction-init.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's generator API to match §R exactly. All runtime claims verified
> against `cytnx==1.1.0` via `docs/api-audit/probes/UniTensor_cat01_02.py`
> (all `[PASS]`, exit 0), with the raw-C++ side of binding-fidelity findings
> verified by `probes/cpp/UniTensor_cat01_02.cpp` against a source-built
> `libcytnx`.

**Category scope:** the `def_static` factory functions that generate a new
`UniTensor` (fills, ranges, random), the two in-place random fills, and static
`Load`. C++ bindings: `cytnx_src/pybind/unitensor_py.cpp:1439-1580`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `zeros` | `zeros(Nelem \| shape, labels=[], dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (Dense) | Dense, filled 0. Probe: *"zeros() fills 0 …"*. |
| `ones` | `ones(Nelem \| shape, labels=[], dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (Dense) | Dense, filled 1. |
| `identity` | `identity(dim, labels=[], is_diag=False, dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (rank-2) | rank-2 identity/delta of size `dim`. |
| `eye` | `eye(dim, …)` — same signature as `identity` | `UniTensor` (rank-2) | **Exact alias** of `identity`. Probe: *"eye(d) is an exact elementwise alias of identity(d)"*. |
| `arange` | `arange(Nelem, labels=[], name='')` **or** `arange(start, end, step=1.0, labels=[], dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (rank-1) | rank-1 range; **the 1-arg overload omits `dtype`/`device`**. Probe: *"arange(Nelem, dtype=...) is REJECTED …"*. |
| `linspace` | `linspace(start, end, Nelem, endpoint=True, labels=[], dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (rank-1) | rank-1, `Nelem` evenly-spaced samples. |
| `normal` | `normal(Nelem \| shape, mean, std, in_labels=[], seed=-1, dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (Dense) | Normal random; uses `in_labels`; `seed=-1` → nondeterministic. Probe: *"normal(seed=s) is reproducible …"*. |
| `uniform` | `uniform(Nelem \| shape, low, high, in_labels=[], seed=-1, dtype=Type.Double, device=Device.cpu, name='')` | `UniTensor` (Dense) | Uniform random. |
| `normal_` `[I]` | `normal_(mean, std, seed=-1)` | **`None`** (1.1.0; → `self`, UT-G5) | Fill with normal samples in place. Probe: *"normal_ fills in place but returns None …"*. |
| `uniform_` `[I]` | `uniform_(low, high, seed=-1)` | **`None`** (1.1.0; → `self`, UT-G5) | Fill with uniform in place. |
| `Load` | `Load(fname)` | `UniTensor` | Static; load a saved UniTensor (member → should be `load`, UT-G10). |

## A2. C++ ↔ Python mapping

| C++ (`def_static`) | Python | Status | Note |
|---|---|---|---|
| `zeros` / `ones(Nelem\|shape, labels, dtype, device, name)` | same | identical | |
| `identity(dim, labels, is_diag, dtype, device, name)` | same | identical | |
| `eye(dim, …)` | `eye(...)` | identical | alias of `identity` (UT-G1) |
| `arange(Nelem \| start,end,step, …)` | `arange(...)` | signature-differs | 1-arg overload drops `dtype`/`device` (UT-G2) |
| `linspace(start, end, Nelem, endpoint, …)` | same | identical | |
| `normal` / `uniform(…, labels, seed, …)` | `…(in_labels=…, seed=-1, …)` | signature-differs | label kwarg renamed `in_labels`; `seed=-1→random_device` (UT-G3) |
| `normal_` / `uniform_` | same **[in-place]** | signature-differs | Python returns `None`, not `self` (UT-G5) |
| `static Load(fname)` | `Load(fname)` | identical | casing note in category 11 |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/UniTensor_cat01_02.py` (on the 1.1.0 wheel). A **(binding
fidelity)** finding flags where the binding layer — a `*_conti.py` wrapper or a
pybind lambda — changes behavior versus the raw C++ method. **Both sides are
runtime-verified:** the raw-C++ side by `probes/cpp/UniTensor_cat01_02.cpp`, a
C++ program that links against a locally source-built `libcytnx` (the 1.1.0
`cytnx_src` tree builds cleanly with GCC 13) and calls the C++ methods directly,
bypassing the binding. Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-G1** | `eye` is a bare exact alias of `identity` | redundancy | **thin pass-through** — the `eye` `def_static` forwards to the same C++ delta-tensor path as `identity`. Probe *"eye(d) is an exact elementwise alias …"* | **remove** (deprecate `eye` 1 release) |
| **UT-G2** | `arange`'s 1-arg overload drops `dtype`/`device`; the `(start,end,…)` form keeps them | signature-differs / ordering | **binding registers two lambdas**: the `(Nelem, labels, name)` lambda declares no `dtype`/`device` `py::arg` at all, while the `(start, end, step, …)` lambda does — so the count form silently can't take a dtype. Probe *"arange(Nelem, dtype=…) is REJECTED"* | both forms take the full metadata block |
| **UT-G3** | label kwarg is `in_labels` on `normal`/`uniform`, `labels` elsewhere | naming + **binding fidelity** | **binding renames one arg**: `zeros`/`ones` expose `py::arg("labels")` (`:1487,1496`) while `normal`/`uniform` expose `py::arg("in_labels")` (`:1533`) over the *same* underlying C++ parameter — a pure binding-layer divergence. Probe: `normal(labels=…)` raises | expose `labels` everywhere |
| **UT-G4** | metadata block ordered 4 different ways across generators | ordering (F20 / PC2) | **thin pass-through** — each lambda mirrors its C++ `py::arg` order; the inconsistency exists in C++ too. Live signatures | keyword-only canonical order (§R.0) |
| **UT-G5** | `normal_`/`uniform_` return `None`, not `self` | **binding fidelity** (N2/B1) | **binding discards the return**: the lambda `[](UniTensor& self, …){ … }` (`:1581`) mutates `self` and returns `void`, dropping C++'s chainable `UniTensor& normal_(…)` (`hpp:5964`). Py probe *"… returns None"*; **C++ probe confirms** `&Z.normal_(…)==&Z` | **return `self`** — restores C++ fidelity |
| **UT-G6** | `seed=-1` → nondeterministic device seed (`-1` ≠ "seed 0") | documentation | mechanism is the UT-G11 lambda injection (below). Probe (same seed reproduces; `-1` does not) | keep; document |
| **UT-G7** | extent operand named two ways: `Nelem` (int) / `shape` (list) | naming (positional) | **binding exposes two overloads** — an `int` (`Nelem`) and a `list` (`shape`) `def_static` per generator — for one concept. Probe *"zeros accepts both int-count & list forms"* | collapse to one `shape: int\|list` |
| **UT-G8** | count slot inconsistent: 1st in shape gens, 3rd in `linspace` | ordering (positional) | **thin pass-through** — lambda mirrors the C++ `(start, end, Nelem)` order. Probe *"linspace … count 3rd"* | **keep** — numpy convention |
| **UT-G9** | distributions put `shape` first (vs numpy size-last) | ordering (positional) | **thin pass-through** — lambda mirrors C++ `normal(shape, mean, std, …)`. Probe *"normal … shape FIRST"* | **keep** — internal consistency |
| **UT-G10** | `Load` (a static member) is Capitalized | naming (N-casing) | **thin pass-through** — `def_static("Load", …)` keeps the C++ name verbatim. Source — SciPost convention | **rename** → `load` (`Save`→`save`) |
| **UT-G11** | `seed==-1 → random_device()` is Python-only sugar injected by the binding | **binding fidelity** | **binding injects logic**: the lambda runs `if(seed==-1) seed = cytnx::random::__static_random_device()` (`:1521-1527`) *before* calling C++ `UniTensor::normal(...)`, which has no such rule. Py probe: `seed=-1` nondeterministic; **C++ probe confirms** two C++ `normal_(…,-1)` fills are identical (literal seed) | keep + document (optionally lift to C++) |

## A4. Argument ordering — positional & keyword

Both orderings must be made canonical: the **positional** operands *and* the
**keyword** metadata block. Positional-required args today (per the live 1.1.0
signatures):

| Generator | positional-required (in order) |
|---|---|
| `zeros` / `ones` | `Nelem` **or** `shape` |
| `identity` | `dim` |
| `arange` | `Nelem` **or** `start, end` |
| `linspace` | `start, end, Nelem` |
| `normal` | `Nelem\|shape, mean, std` |
| `uniform` | `Nelem\|shape, low, high` |
| `normal_` / `uniform_` | `mean, std` / `low, high` |
| `Load` | `fname` |

- **UT-G7 (positional naming) — the extent operand has two names, `Nelem`
  (int-count overload) and `shape` (list overload), for one concept.** Probe:
  *"zeros accepts both the int-count (Nelem) and list (shape) forms …"*. →
  collapse to a single positional `shape: int | Sequence[int]`, removing the
  dual overload and the name split.
- **UT-G8 (positional order) — the count operand's slot is inconsistent:
  position 1 in the shape generators but position 3 in `linspace`
  (`start, end, Nelem`).** Probe: *"linspace(start, end, Nelem) puts the count
  3rd …"*. This is the **numpy convention** (`np.linspace(start, stop, num)`),
  so the canonical rule *accepts and documents* it rather than "fixing" it: the
  rule distinguishes **shape-based** generators (extent = `shape`, first) from
  **range-based** generators (`start, end[, step|num]`, numpy order).
- **UT-G9 (positional order, deliberate) — distributions put `shape` FIRST
  (`normal(shape, mean, std)`), diverging from numpy's size-last
  (`np.random.normal(loc, scale, size)`).** Probe: *"normal(shape, mean, std)
  puts shape FIRST …"*. cytnx's shape-first is internally consistent with
  `zeros`/`ones` and is **kept**; the numpy divergence is intentional and
  documented. (Contrast UT-G8, where numpy convention is *followed* — the two
  together define the positional rule below.)
- **Keyword block:** ordered four ways today (UT-G4) → one canonical
  keyword-only order.

**Canonical orders (normative — see §R.0):**
- **Positional:** `[primary operand], [operation parameters]` — primary =
  `shape: int|list` (shape-based) *or* `start, end` (range-based) *or* the
  source (`fname`); operation parameters follow in domain order (`mean, std`;
  `low, high`; `step`; `num`). Shape-based → extent first (UT-G9); range-based →
  numpy `(start, end, step|num)` (UT-G8).
- **Keyword-only:** `labels, rowrank, is_diag, dtype, device, seed, name`.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version generator API. Implement Cytnx
to match it.*

## R.0 Normative conventions

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).**
  *Member* functions are lowercase snake_case; *free* functions that **act on**
  objects are Capitalized; *free* functions that **create** objects (generators)
  are lowercase; *types* are Capitalized. In this category every generator is a
  lowercase static creator (correct) — the only change is the Capitalized static
  member `Load` → `load` (UT-G10). Do **not** capitalize the generators. (This is
  a deliberate departure from a blanket "snake_case everything" rule: linalg free
  functions like `Svd` stay Capitalized precisely because they act on objects.)

*Two orderings are also normative — the positional operands and the keyword block.*

- **N-positional — canonical positional order is `[primary operand],
  [operation parameters]`** (UT-G8/G9). The primary operand is exactly one
  required argument: `shape` for shape-based generators, `start, end` for
  range-based generators, or the source (`fname`). Operation parameters follow
  in domain order — `mean, std` (normal), `low, high` (uniform), `step`
  (arange), `num` (linspace). Shape-based generators put the extent **first**
  (matching `zeros`/`ones`, a deliberate divergence from numpy's size-last,
  UT-G9); range-based generators follow the numpy order `(start, end, step|num)`
  (UT-G8). No optional positional args exist — everything optional is
  keyword-only.
- **N-extent — one extent operand, `shape: int | Sequence[int]`** (UT-G7).
  Collapse the `Nelem`(int) / `shape`(list) overload pair into a single
  positional accepting either, eliminating the name split and halving the
  overload count.
- **N-kwonly — the metadata block is keyword-only**, declared in the canonical
  order `labels, rowrank, is_diag, dtype, device, seed, name` (omit inapplicable
  members without reordering). Making it keyword-only resolves UT-G4
  structurally: order leaves the public contract, so it cannot be inconsistent
  and no positional caller can break.
- **N-underscore — a trailing `_` marks in-place; its absence marks pure.**
  A name ending in `_` mutates the receiver (or, for a free function, its primary
  argument) **in place and returns `self`** for chaining; the un-suffixed name is
  **pure** — it returns a *new* object and leaves inputs unchanged (behavior
  convention B1). Every operation meaningful in both modes provides **both**
  forms under one base name (`normal`/`normal_`, `uniform`/`uniform_`); a
  one-sided operation where both make sense is a finding. The trailing `_` is the
  **only** in-place marker — reject alternates (`c`-prefixed raw bindings like
  `cnormalize_`, an `i`-prefix, etc.). This composes with N-casing: a free
  in-place operation is Capitalized *and* `_`-suffixed (e.g. `Inv_`); a member
  in-place operation is lowercase *and* `_`-suffixed (e.g. `permute_`). In this
  category the pairs are `normal`/`normal_` and `uniform`/`uniform_`; the `_`
  forms must return `self` (fixes UT-G5).
- **One label name everywhere: `labels`** (never `in_labels`, UT-G3).
- **Defaults:** `dtype=Type.Double`, `device=Device.cpu`, `rowrank=-1`,
  `is_diag=False`, `seed=-1` (nondeterministic), `labels=[]`, `name=''`.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # Shape-based generators: primary operand `shape` (int|list) first,
    # then keyword-only metadata block.
    @staticmethod
    def zeros(shape: int | Sequence[int], *, labels=[], rowrank=-1,
              dtype=Type.Double, device=Device.cpu, name="") -> "UniTensor": ...
    @staticmethod
    def ones(shape: int | Sequence[int], *, labels=[], rowrank=-1,
             dtype=Type.Double, device=Device.cpu, name="") -> "UniTensor": ...
    @staticmethod
    def identity(dim: int, *, labels=[], rowrank=-1, is_diag=False,
                 dtype=Type.Double, device=Device.cpu, name="") -> "UniTensor": ...
    # Range-based generators: numpy positional order (start, end, step|num).
    @staticmethod
    def arange(start, end=None, step=1, *, labels=[], rowrank=-1,
               dtype=Type.Double, device=Device.cpu, name="") -> "UniTensor": ...
    @staticmethod
    def linspace(start, end, num, *, endpoint=True, labels=[], rowrank=-1,
                 dtype=Type.Double, device=Device.cpu, name="") -> "UniTensor": ...
    # Distributions: shape first (internal consistency), then params, then kw.
    @staticmethod
    def normal(shape: int | Sequence[int], mean, std, *, labels=[], rowrank=-1,
               dtype=Type.Double, device=Device.cpu, seed=-1, name="") -> "UniTensor": ...
    @staticmethod
    def uniform(shape: int | Sequence[int], low, high, *, labels=[], rowrank=-1,
                dtype=Type.Double, device=Device.cpu, seed=-1, name="") -> "UniTensor": ...
    def normal_(self, mean, std, *, seed=-1) -> "UniTensor": ...   # returns self
    def uniform_(self, low, high, *, seed=-1) -> "UniTensor": ...  # returns self
    @staticmethod
    def load(fname) -> "UniTensor": ...   # was Load — member → lowercase (UT-G10)
```

Positional order per N-positional; `shape` is one operand accepting `int` or
`list[int]` per N-extent (UT-G7), collapsing the old `Nelem`/`shape` overloads.

| API | Verdict | Behavior contract |
|---|---|---|
| `zeros` / `ones` | **keep** (kwonly; `shape:int\|list` collapses `Nelem`, UT-G7; `rowrank` added) | Dense; every element 0 / 1. |
| `identity` | **keep** (kwonly; `is_diag` → canonical slot) | rank-2 `dim`×`dim` identity; `is_diag=True` stores diagonal only. |
| `eye` | **remove** (UT-G1) | Deprecated alias of `identity`; keep as `DeprecationWarning` alias one release, then delete. |
| `arange` | **keep** (single signature; both count-form `arange(end)` and range-form accept full metadata, UT-G2; numpy positional order, UT-G8) | rank-1 evenly-spaced values. |
| `linspace` | **keep** (kwonly; numpy positional order `start, end, num`, UT-G8) | rank-1, `num` samples over `[start,end]`. |
| `normal` / `uniform` | **keep** (kwonly; `in_labels`→`labels`, UT-G3; `shape` first + `shape:int\|list`, UT-G7/G9) | Dense random; `N(mean,std²)` / `U[low,high)`. |
| `normal_` / `uniform_` | **keep, but return `self`** (UT-G5) | In-place fill; chainable. |
| `Load` → `load` | **rename** (member → lowercase, UT-G10) | Deserialize a saved UniTensor. *Migration:* keep `Load` as a `DeprecationWarning` alias one release. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface.

### R.2a Python API (numpy-style)

### `UniTensor.zeros` / `UniTensor.ones`

```
UniTensor.zeros(shape, *, labels=[], rowrank=-1, dtype=Type.Double,
                device=Device.cpu, name='') -> UniTensor
UniTensor.ones(shape, *, ...) -> UniTensor   # identical, fills 1

Create a Dense UniTensor filled with 0 (zeros) or 1 (ones).

Parameters
----------
shape : list of int
    Size of each leg; its length sets the rank.
labels, rowrank, dtype, device, name : keyword-only
    See UniTensor.__init__ (bonds form) for meanings.

Returns
-------
UniTensor
    A Dense tensor with every element 0 (or 1).
```

### `UniTensor.identity`

```
UniTensor.identity(dim, *, labels=[], rowrank=-1, is_diag=False,
                   dtype=Type.Double, device=Device.cpu, name='') -> UniTensor

Create a rank-2 identity (delta) UniTensor of size `dim`.

Parameters
----------
dim : int
    Dimension of the square identity.
is_diag : bool, keyword-only
    Store only the diagonal (memory-efficient rank-2 form).
labels, rowrank, dtype, device, name : keyword-only
    As for `zeros`.

Returns
-------
UniTensor
    A dim x dim identity tensor.

Notes
-----
The former `eye` alias is deprecated; use `identity` (finding UT-G1).
```

### `UniTensor.arange`

```
UniTensor.arange(start, end=None, step=1, *, labels=[], rowrank=-1,
                 dtype=Type.Double, device=Device.cpu, name='') -> UniTensor

Create a rank-1 UniTensor of evenly spaced values. Both forms accept the full
metadata block (fixes the 1.1.0 count-form gap, finding UT-G2).

Overloads
---------
arange(end)                 : values 0, 1, ..., end-1.
arange(start, end, step=1)  : start (incl.) to end (excl.) by step.

Returns
-------
UniTensor
    A rank-1 Dense tensor of the generated sequence.
```

### `UniTensor.linspace`

```
UniTensor.linspace(start, end, num, *, endpoint=True, labels=[], rowrank=-1,
                   dtype=Type.Double, device=Device.cpu, name='') -> UniTensor

Create a rank-1 UniTensor of `num` evenly spaced samples over [start, end]
(`end` included iff `endpoint`).

Returns
-------
UniTensor
    A rank-1 Dense tensor of the sampled values.
```

### `UniTensor.normal` / `UniTensor.uniform`

```
UniTensor.normal(shape, mean, std, *, labels=[], rowrank=-1, dtype=Type.Double,
                 device=Device.cpu, seed=-1, name='') -> UniTensor
UniTensor.uniform(shape, low, high, *, ..., seed=-1, name='') -> UniTensor

Create a Dense UniTensor of random elements — N(mean, std**2) for `normal`,
U[low, high) for `uniform`.

Parameters
----------
shape : list of int
    Size of each leg.
mean, std / low, high : float
    Distribution parameters (positional).
labels, rowrank, dtype, device, name : keyword-only
    As for `zeros`. The label kwarg is `labels` (renamed from `in_labels`, UT-G3).
seed : int, keyword-only
    RNG seed; the same seed reproduces the same tensor. -1 (default) draws a
    nondeterministic seed from the system random device (UT-G6).

Returns
-------
UniTensor
    A Dense tensor of the requested random samples.
```

### `UniTensor.normal_` / `UniTensor.uniform_` (in-place)

```
UniTensor.normal_(mean, std, *, seed=-1) -> UniTensor
UniTensor.uniform_(low, high, *, seed=-1) -> UniTensor

In-place fill of this tensor with random samples; RETURNS SELF (finding UT-G5 —
1.1.0 returns None). See `normal`/`uniform` for parameter meanings.
```

### `UniTensor.load` (static)

```
UniTensor.load(fname) -> UniTensor

Load a UniTensor previously written by `save`. Renamed from `Load` — member
functions are lowercase per the Cytnx naming convention (UT-G10).

Returns
-------
UniTensor
    The deserialized tensor.
```

### R.2b C++ API (Doxygen)

C++ has no keyword-only parameters (the metadata are default arguments in the
canonical order). All are `static` members returning `UniTensor`, except the
in-place fills, which return `UniTensor&` (UT-G5).

```cpp
/**
 * @brief Create a Dense UniTensor filled with 0 (zeros) or 1 (ones).
 * @param shape   size of each leg (a scalar builds a rank-1 tensor).
 * @param labels  leg labels; default {"0","1",...}.
 * @param rowrank row (bra) space leg count; -1 auto-selects.
 * @param dtype   element type, e.g. Type.Double.
 * @param device  storage device, e.g. Device.cpu.
 * @param name    tensor name.
 * @return a Dense UniTensor with every element 0 (or 1).
 */
static UniTensor zeros(const std::vector<cytnx_uint64> &shape,
                       const std::vector<std::string> &labels = {},
                       const cytnx_int64 &rowrank = -1,
                       const unsigned int &dtype = Type.Double,
                       const int &device = Device.cpu,
                       const std::string &name = "");
static UniTensor ones(/* ...identical to zeros... */);

/**
 * @brief Create a rank-2 identity (delta) UniTensor of size @p dim.
 * @param dim     dimension of the square identity.
 * @param is_diag store only the diagonal (memory-efficient rank-2 form).
 * @param labels,rowrank,dtype,device,name  as for zeros().
 * @return a dim x dim identity UniTensor.
 */
static UniTensor identity(const cytnx_uint64 &dim,
                          const std::vector<std::string> &labels = {},
                          const cytnx_int64 &rowrank = -1,
                          const bool &is_diag = false, /* dtype,device,name */);

/**
 * @brief Create a rank-1 UniTensor of evenly spaced values (numpy order).
 * @param start,end,step  range: start (incl.) to end (excl.) by step.
 * @param labels,rowrank,dtype,device,name  as for zeros().
 * @return a rank-1 Dense UniTensor.
 */
static UniTensor arange(const cytnx_double &start, const cytnx_double &end,
                        const cytnx_double &step = 1, /* metadata... */);

/**
 * @brief Create a rank-1 UniTensor of @p num evenly spaced samples over [start,end].
 * @param start,end  interval endpoints (@p end included iff @p endpoint).
 * @param num        number of samples.
 * @param endpoint   whether @p end is the last sample.
 * @param labels,rowrank,dtype,device,name  as for zeros().
 * @return a rank-1 Dense UniTensor.
 */
static UniTensor linspace(const cytnx_double &start, const cytnx_double &end,
                          const cytnx_uint64 &num, const bool &endpoint = true,
                          /* metadata... */);

/**
 * @brief Create a Dense UniTensor of random elements: N(mean,std^2) / U[low,high).
 * @param shape     size of each leg.
 * @param mean,std  normal-distribution parameters (normal()).
 * @param low,high  uniform-distribution bounds (uniform()).
 * @param seed      RNG seed; the same seed reproduces the tensor. NOTE: the -1
 *                  -> random_device convenience is a Python-binding feature
 *                  (UT-G11); in C++ pass an explicit seed.
 * @param labels,rowrank,dtype,device,name  as for zeros().
 * @return a Dense UniTensor of the requested random samples.
 */
static UniTensor normal(const std::vector<cytnx_uint64> &shape,
                        const double &mean, const double &std,
                        /* labels,rowrank,dtype,device,seed,name */);
static UniTensor uniform(const std::vector<cytnx_uint64> &shape,
                         const double &low, const double &high, /* ... */);

/**
 * @brief Fill this tensor in place with random samples; returns *this.
 * @param mean,std / low,high  distribution parameters.
 * @param seed  RNG seed.
 * @return reference to *this (chainable) — the fidelity the Python binding must
 *         preserve (UT-G5).
 */
UniTensor &normal_(const double &mean, const double &std,
                   const cytnx_int64 &seed = -1);
UniTensor &uniform_(const double &low, const double &high,
                    const cytnx_int64 &seed = -1);

/**
 * @brief Load a UniTensor previously written by save().
 * @param fname  path to the saved file.
 * @return the deserialized UniTensor.
 */
static UniTensor load(const std::string &fname);
```
