# UniTensor — 03. Metadata accessors

> Split into **Analysis** (the evidence — inventory, C++↔Python mapping,
> findings) and a self-contained **Recommendation** that is the *normative spec
> for the next version of Cytnx*: the next major version's `UniTensor` metadata
> surface should be implemented to match §R exactly. Every behavioral claim is
> verified against the installed `cytnx==1.1.0` wheel by
> `docs/api-audit/probes/UniTensor_03_metadata.py` (all `[PASS]`, exit 0).
> This category has **no** binding-fidelity finding — the accessors are thin
> pass-throughs — so no raw-C++ probe is required (gate 4, see A3).

**Category scope:** the read-only *metadata accessors* — the query methods that
report a `UniTensor`'s structure (`rank`, `rowrank`, `Nblocks`, `shape`),
element/storage identity (`dtype`, `device`, `uten_type`, `name`,
`is_contiguous`), classification predicates (`is_diag`, `is_tag`,
`is_braket_form`, `is_blockform`), and its leg / symmetry structure (`labels`,
`get_index`, `syms`, `bonds`, `bond`, `bond_`, `signflip`, `same_data`,
`elem_exists`, `get_qindices`, `getTotalQnums`, `get_blocks_qnums`). None mutate
the tensor's data. Constructors are [category 01](01-construction-init.md);
generators are [category 02](02-static-generators.md).

---

# Analysis

**Provenance:** live pybind signatures from the 1.1.0 wheel
(`tools/member_inventory.py UniTensor`); C++ from
`cytnx_src/include/UniTensor.hpp` and `cytnx_src/pybind/unitensor_py.cpp`
(accessor `.def`s at `:311,347,489-497,760-762,1606`; `bonds`/`bond`/`bond_` at
`hpp:124-135,3151-3152`; the "not support" note at `hpp:4735-4754`).

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. All are member accessors on `self`.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `rank` | `rank()` | `int` | Number of legs. Probe: *"rank() of the 2-leg block tensor is 2"*. |
| `rowrank` | `rowrank()` | `int` | Legs in the row (bra) space. Probe: *"rowrank() is 1 …"*. |
| `Nblocks` | `Nblocks()` | `int` | Number of symmetry blocks. Probe: *"Nblocks() is 2 …"*. |
| `shape` | `shape()` | `list[int]` | Size per leg. Probe: *"shape() is [2, 2]"*. |
| `dtype` | `dtype()` | `int` | Element type code. Probe: *"dtype() returns … Type.Double (3)"*. |
| `dtype_str` | `dtype_str()` | `str` | Human-readable dtype. Probe: *"dtype_str() names the element type"*. |
| `device` | `device()` | `int` | Device code. Probe: *"device() returns … Device.cpu (-1)"*. |
| `device_str` | `device_str()` | `str` | Human-readable device. Probe: *"device_str() names the device"*. |
| `uten_type` | `uten_type()` | `int` | UniTensor kind code (0 Dense / 2 Block). Probe: *"uten_type() is 0 (Dense) …, 2 (Block) …"*. |
| `uten_type_str` | `uten_type_str()` | `str` | Kind name. Probe: *"uten_type_str() is 'Dense' / 'Block'"*. |
| `name` | `name()` | `str` | Tensor name. Probe: *"name() reflects a set_name('foo')"*. |
| `is_diag` | `is_diag()` | `bool` | Diagonal-only storage. Probe: *"is_diag() is False …"*. |
| `is_tag` | `is_tag()` | `bool` | Legs carry bra/ket direction. Probe: *"is_tag() is True on the symmetric … / False on the dense …"*. |
| `is_braket_form` | `is_braket_form()` | `bool` | Row legs are all ket, column all bra. Probe: *"is_braket_form() is a bool predicate"*. |
| `is_blockform` | `is_blockform()` | `bool` | Block (symmetric) storage. Probe: *"is_blockform() is True on the block … / False on the dense …"*. |
| `is_contiguous` | `is_contiguous()` | `bool` | Memory is contiguous. Probe: *"is_contiguous() is True for a freshly-built dense tensor"*. |
| `labels` | `labels()` | `list[str]` | Leg labels. Probe: *"labels() defaults to ['0', '1']"*. |
| `get_index` | `get_index(arg0: str)` | `int` | Label → leg index. Probe: *"get_index('1') resolves label '1' to leg index 1"*. |
| `syms` | `syms()` | `list[Symmetry]` | The tensor's symmetries. Probe: *"syms() returns … (one U(1))"*. |
| `bonds` | `bonds()` | `list[Bond]` — **copied container, shared elements** | All legs. Probe: *"bonds() copies the list but its Bond elements share the parent's impl"*. |
| `bond` | `bond(idx:int)` / `bond(label:str)` | `Bond` — **independent copy** | One leg, cloned. Probe: *"bond(i) is a COPY — redirect_ on it leaves the parent unchanged"*. |
| `bond_` | `bond_(idx:int)` / `bond_(label:str)` | `Bond` — **view** | One leg, shares the parent's Bond. Probe: *"bond_(i) is a VIEW — redirect_ on it flips the parent's bond direction"*. |
| `signflip` | `signflip()` | `list[bool]` | Per-leg fermion sign flags — **BlockFermionic only**. Probe: *"signflip() returns list[bool] on a BlockFermionic tensor"* / *"… RAISES on a bosonic … block tensor"*. |
| `same_data` | `same_data(arg0: UniTensor)` | `bool` | Do two tensors share storage? Probe: *"same_data(self) is True …"* / *"same_data(clone) is False …"*. |
| `elem_exists` | `elem_exists(arg0: list[int])` | `bool` | Is a locator inside an allowed block? (symmetric only). Probe: *"elem_exists([0,0]) is True … / elem_exists([0,1]) is False …"*. |
| `get_qindices` | `get_qindices(arg0: int)` | `list[int]` | Per-block qnum-index list for a leg (symmetric only). Probe: *"get_qindices(0) returns … [0, 0]"*. |
| `getTotalQnums` | `getTotalQnums(physical=False)` | `list[Bond]` | **Bound but non-functional** — raises on every tensor type. Probe: *"getTotalQnums on a block/dense tensor RAISES …"*. |
| `get_blocks_qnums` | `get_blocks_qnums()` | `list[list[int]]` | **Bound but non-functional** — raises on every tensor type. Probe: *"get_blocks_qnums on a block/dense tensor RAISES …"*. |

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `Python-only`.

| C++ (`UniTensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `rank`/`rowrank`/`shape`/`dtype`/`dtype_str`/`device`/`device_str`/`uten_type`/`uten_type_str`/`name` | same | identical | scalar/string metadata pass-throughs |
| `Nblocks()` | `Nblocks()` | identical | Capitalized member (N-casing, UT-M1) |
| `is_diag`/`is_tag`/`is_braket_form`/`is_blockform`/`is_contiguous` | same | identical | bool predicates |
| `labels`/`get_index`/`syms` | same | identical | `get_index(arg0)` name erased (UT-M5) |
| `const vector<Bond>& bonds() const` **and** `vector<Bond>& bonds()` (`:124-125`) | `bonds()` → **copied list** | signature-differs | binding returns the vector **by value** (`:491`); the C++ non-const `&` overload (a mutable, resizable ref) has no Python peer — a signature choice, **not** a behavior divergence (A3) |
| `Bond bond(idx) const` = `bond_(idx).clone()` (`:3151`) | `bond(idx\|label)` | identical | pure copy (clone) |
| `Bond& bond_(idx)` (`:127`) | `bond_(idx\|label)` | identical | returns a shell that shares the parent's Bond impl (view) |
| `signflip() const` | `signflip()` | identical | BlockFermionic-only; base raises (UT-M7) |
| `same_data(rhs) const` | `same_data(arg0)` | identical | arg name erased to `arg0` (UT-M5) |
| `elem_exists(locator) const` | `elem_exists(arg0)` | identical | arg name erased (UT-M5) |
| `get_qindices(bidx) const` (`:1606`) | `get_qindices(arg0)` | identical | arg name erased (UT-M5) |
| `getTotalQnums(physical=false)` (`:4745`, *"@note This API just have not support"*) | `getTotalQnums(physical=False)` | renamed→snake | camelCase; **non-functional** (UT-M2/UT-M6) |
| `get_blocks_qnums() const` (`:4752`, same *"not support"* note) | `get_blocks_qnums()` | identical | already snake; **non-functional** (UT-M6) |

## A3. Findings

Behavioral claims quote a probe assertion (1.1.0 wheel). A **(binding
fidelity)** finding would flag where the binding layer changes behavior versus
the raw C++ method — this category has **none**: every accessor is a thin
`.def(&UniTensor::…)` (or a trivial forwarding lambda) that returns exactly what
the C++ method returns, and the two "not supported" methods raise because the
*C++* methods raise (`hpp:4735-4754` + the base not-implemented at
`UniTensor_base.cpp:499,504`), not because of any binding transform. The
`bonds()` copy-vs-`&` gap (A2) is a signature choice, not a behavior divergence.
**Gate 4 (C++ probe) is therefore skipped for category 03.**

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-M1** | `Nblocks` is a Capitalized member | naming (N-casing) | **thin pass-through** — `.def("Nblocks", &UniTensor::Nblocks)` (`:311`) keeps the C++ name verbatim; the count is correct (probe *"Nblocks() is 2 …"*) | **rename** → `nblocks` (member → lowercase snake_case, §R.0) |
| **UT-M2** | `getTotalQnums` is camelCase | naming (N-casing) | **thin pass-through** — `.def("getTotalQnums", …)` (`:760`) keeps the C++ name | **rename** → `get_total_qnums` |
| **UT-M3** | `bonds()` returns a **copied container** whose `Bond` elements still **share** the parent's impl | copy/view (B2) | **binding copies the vector by value** (`:491`); the copied list can't resize the parent's leg set (unlike the C++ non-const `&`), but a leg mutated *in place* through it reaches the parent. Probe: *"bonds() copies the list but its Bond elements share the parent's impl"* | document the two-level semantics; recommend a `bonds()` that yields per-leg copies (use `bond_(i)` for a deliberate view) |
| **UT-M4** | `bond` (copy) / `bond_` (view) is a correctly-formed N-underscore pair | naming (N-underscore) — **conformant** | **thin pass-through** — `bond` clones (`hpp:3151`), `bond_` returns a shared shell (`hpp:127`). Probe: *"bond_(i) is a VIEW …"* / *"bond(i) is a COPY …"* | **keep both** — canonical view/copy pair; document the split |
| **UT-M5** | `same_data`/`elem_exists`/`get_qindices`/`get_index` erase their argument name to `arg0` | naming (parameter consistency, PC1) | **binding drops the `py::arg`** — `.def("same_data", &UniTensor::same_data)` (`:489`), `elem_exists` (`:347`), `get_qindices` (`:1606`) register no `py::arg(...)`, so the parameter is positional-only. Probe: *"same_data(rhs=…) is REJECTED"*, *"elem_exists(locator=…) is REJECTED"*, *"get_qindices(bidx=…) is REJECTED"* | add real `py::arg` names (`other`, `locator`, `bond_idx`, `label`) |
| **UT-M6** | `getTotalQnums`/`get_blocks_qnums` are advertised yet **raise on every tensor type** | correctness / documentation | **thin pass-through to a broken C++ method** — the C++ header marks both *"@note This API just have not support"* (`hpp:4742,4751`); Dense raises *"can only operate on UniTensor with symmetry"* and Block falls to the not-implemented base (`UniTensor_base.cpp:499,504`). Probe: *"getTotalQnums/get_blocks_qnums on a block/dense tensor RAISES …"* (4 assertions) | **implement** them for Block tensors (their documented domain) or **remove**; if kept, rename `getTotalQnums`→`get_total_qnums` (UT-M2) |
| **UT-M7** | `signflip` is a BlockFermionic-only accessor that raises on bosonic tensors | documentation | **thin pass-through** — `.def("signflip", &UniTensor::signflip)` (`:497`); the base raises *"signflip is only defined for BlockFermionicUniTensor"* (`UniTensor_base.cpp:66`). Probe: *"signflip() returns list[bool] on a BlockFermionic tensor"* / *"… RAISES on a bosonic … block tensor"* | **keep**; document the fermionic-only precondition |

## A4. Argument ordering — positional & keyword

Metadata accessors are almost all **nullary** (`self` only), so ordering is
vacuous. The few that take an argument take exactly **one** operand:

| Accessor | operand(s) | issue |
|---|---|---|
| `get_index` | `label: str` | name erased? no — `get_index(arg0)` (still positional-only, UT-M5) |
| `bond` / `bond_` | `idx: int` **or** `label: str` | overloaded selector; no ordering issue |
| `same_data` | `other: UniTensor` | name erased to `arg0` (UT-M5) |
| `elem_exists` | `locator: list[int]` | name erased to `arg0` (UT-M5) |
| `get_qindices` | `bond_idx: int` | name erased to `arg0` (UT-M5) |
| `getTotalQnums` | `physical: bool = False` | should be **keyword-only** (a flag, not an operand) |

- **Positional:** each accessor with an operand takes it as the single required
  positional (`label`, `idx`, `other`, `locator`, `bond_idx`) — canonical and
  trivial. The only optional argument in the category, `physical` on
  `getTotalQnums`, is a boolean flag and becomes **keyword-only** (§R.0).
- **Keyword:** the fix is not *order* but *names* (UT-M5) — restore the erased
  `arg0` names so every operand is callable by keyword.

**Canonical rule (normative — see §R.0):** one required positional operand where
applicable, named (never `arg0`); boolean flags (`physical`) keyword-only.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `UniTensor`
metadata-accessor surface. Implement Cytnx to match it. Findings above are the
rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing — follow the Cytnx naming convention (SciPostPhysCodeb.53).**
  *Member* functions are lowercase snake_case. Two members violate this and are
  renamed: `Nblocks` → `nblocks` (UT-M1) and `getTotalQnums` → `get_total_qnums`
  (UT-M2). `get_blocks_qnums` is already snake_case (kept). All other accessors
  (`rank`, `rowrank`, `shape`, `is_*`, `bonds`, `syms`, …) are already correct.
- **N-underscore — a trailing `_` marks a view/in-place handle; its absence
  marks a pure copy.** The category's one such pair is **conformant** and kept:
  `bond(i)` returns an independent clone, `bond_(i)` returns a view sharing the
  parent's Bond (UT-M4). No other accessor mutates, so none carries a `_`.
- **N-view — copy/view behavior is fixed and documented.** `bond_(i)` is a view;
  `bond(i)` is a copy; `bonds()` copies the *list container* but its `Bond`
  elements share the parent's impl (UT-M3) — the recommended `bonds()` yields
  per-leg copies, with `bond_(i)` the explicit opt-in for a view.
- **N-argname — no erased `arg0` parameters.** Every operand carries a real,
  keyword-callable name (UT-M5): `same_data(other)`, `elem_exists(locator)`,
  `get_qindices(bond_idx)`, `get_index(label)`. Boolean flags are keyword-only
  (`getTotalQnums`'s `physical`).
- **Functional contract — no advertised-but-broken members.**
  `get_total_qnums` and `get_blocks_qnums` must actually return their quantum
  numbers on Block (symmetric) tensors instead of raising the current
  not-implemented error (UT-M6); on a non-symmetric tensor they raise a clear
  `ValueError`.

*The positional/keyword rule is also normative.*

- **N-positional — one named operand where applicable; flags keyword-only.**
  Accessors take at most one required positional (`idx`|`label`, `other`,
  `locator`, `bond_idx`), always named; `physical` is keyword-only.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- structure ---
    def rank(self) -> int: ...
    def rowrank(self) -> int: ...
    def nblocks(self) -> int: ...                 # was Nblocks (UT-M1)
    def shape(self) -> list[int]: ...
    # --- element / storage identity ---
    def dtype(self) -> int: ...
    def dtype_str(self) -> str: ...
    def device(self) -> int: ...
    def device_str(self) -> str: ...
    def uten_type(self) -> int: ...
    def uten_type_str(self) -> str: ...
    def name(self) -> str: ...
    # --- classification predicates ---
    def is_diag(self) -> bool: ...
    def is_tag(self) -> bool: ...
    def is_braket_form(self) -> bool: ...
    def is_blockform(self) -> bool: ...
    def is_contiguous(self) -> bool: ...
    # --- legs / labels ---
    def labels(self) -> list[str]: ...
    def get_index(self, label: str) -> int: ...   # was get_index(arg0) (UT-M5)
    def bonds(self) -> list[Bond]: ...            # per-leg copies (UT-M3)
    def bond(self, idx: int | str) -> Bond: ...   # independent copy
    def bond_(self, idx: int | str) -> Bond: ...  # view (UT-M4)
    # --- symmetry ---
    def syms(self) -> list[Symmetry]: ...
    def signflip(self) -> list[bool]: ...         # BlockFermionic only (UT-M7)
    def same_data(self, other: "UniTensor") -> bool: ...     # was arg0 (UT-M5)
    def elem_exists(self, locator: list[int]) -> bool: ...   # was arg0 (UT-M5)
    def get_qindices(self, bond_idx: int) -> list[int]: ...  # was arg0 (UT-M5)
    def get_total_qnums(self, *, physical: bool = False) -> list[Bond]: ...  # was getTotalQnums (UT-M2/M6)
    def get_blocks_qnums(self) -> list[list[int]]: ...       # must work on Block (UT-M6)
```

| API | Verdict | Behavior contract |
|---|---|---|
| `rank` / `rowrank` / `shape` | **keep** | Leg count / row-space leg count / per-leg sizes. |
| `Nblocks` → `nblocks` | **rename** (UT-M1) | Number of symmetry blocks. *Migration:* keep `Nblocks` as a `DeprecationWarning` alias one release, then delete. |
| `dtype` / `dtype_str` / `device` / `device_str` | **keep** | Element-type / device code and its string form. |
| `uten_type` / `uten_type_str` | **keep** | UniTensor kind code (0 Dense, 2 Block) and its string form. |
| `name` | **keep** | Tensor name (empty by default). |
| `is_diag` / `is_tag` / `is_braket_form` / `is_blockform` / `is_contiguous` | **keep** | Boolean classification predicates. |
| `labels` | **keep** | Leg labels, in leg order. |
| `get_index` | **keep** (name `arg0`→`label`, UT-M5) | Resolve a label to its leg index. |
| `bonds` | **keep** (per-leg copies, UT-M3) | All legs, as independent copies; use `bond_` for a view. |
| `bond` / `bond_` | **keep both** (UT-M4) | `bond` clones a leg; `bond_` returns a view sharing the parent's Bond. |
| `syms` | **keep** | The tensor's symmetry objects. |
| `signflip` | **keep** (UT-M7) | Per-leg fermion sign flags; **BlockFermionic only** — raises otherwise. |
| `same_data` | **keep** (name `arg0`→`other`, UT-M5) | True iff `self` and `other` share storage. |
| `elem_exists` | **keep** (name `arg0`→`locator`, UT-M5) | True iff `locator` lands in an allowed block; symmetric only. |
| `get_qindices` | **keep** (name `arg0`→`bond_idx`, UT-M5) | Per-block qnum-index list for a leg; symmetric only. |
| `getTotalQnums` → `get_total_qnums` | **rename + fix** (UT-M2/M6) | Total quantum numbers of the tensor; must **work on Block tensors** (1.1.0 raises everywhere). `physical` keyword-only. *Migration:* keep `getTotalQnums` as a `DeprecationWarning` alias one release. |
| `get_blocks_qnums` | **keep + fix** (UT-M6) | Per-block quantum numbers; must **work on Block tensors** (1.1.0 raises everywhere). |

No accessor is deleted; the changes are renames (`Nblocks`, `getTotalQnums`),
restored parameter names, and the correctness fix for the two qnum accessors.

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface.

### R.2a Python API (numpy-style)

### structure & identity accessors (`rank`, `rowrank`, `nblocks`, `shape`, `dtype`, `dtype_str`, `device`, `device_str`, `uten_type`, `uten_type_str`, `name`)

```
UniTensor.rank() -> int             # number of legs
UniTensor.rowrank() -> int          # legs in the row (bra) space
UniTensor.nblocks() -> int          # number of symmetry blocks (was `Nblocks`)
UniTensor.shape() -> list[int]      # size of each leg
UniTensor.dtype() -> int            # element-type code (e.g. Type.Double)
UniTensor.dtype_str() -> str        # element-type name
UniTensor.device() -> int           # device code (e.g. Device.cpu == -1)
UniTensor.device_str() -> str       # device name
UniTensor.uten_type() -> int        # kind code: 0 Dense, 2 Block
UniTensor.uten_type_str() -> str    # kind name: 'Dense' / 'Block'
UniTensor.name() -> str             # tensor name

Read-only metadata accessors. None mutate the tensor.

Returns
-------
int / list[int] / str
    The requested metadata. `rank`==len(shape); `nblocks` counts symmetry
    blocks (1 for a Dense tensor).
```

### classification predicates (`is_diag`, `is_tag`, `is_braket_form`, `is_blockform`, `is_contiguous`)

```
UniTensor.is_diag() -> bool          # stores only the diagonal (rank-2 square)
UniTensor.is_tag() -> bool           # legs carry bra/ket direction (tagged)
UniTensor.is_braket_form() -> bool   # row legs all ket, column legs all bra
UniTensor.is_blockform() -> bool     # block (symmetric) storage
UniTensor.is_contiguous() -> bool    # element storage is contiguous

Returns
-------
bool
    The predicate's value for this tensor. A Dense untagged tensor returns
    False for is_tag / is_blockform.
```

### `UniTensor.labels` / `UniTensor.get_index`

```
UniTensor.labels() -> list[str]
UniTensor.get_index(label) -> int

Return all leg labels, or resolve one `label` to its leg index.

Parameters
----------
label : str
    A leg label (renamed from the erased `arg0`, finding UT-M5).

Returns
-------
list[str] / int
    The labels (in leg order), or the 0-based index of `label`.

Raises
------
ValueError
    If `label` is not a leg of this tensor.
```

### `UniTensor.bonds` / `UniTensor.bond` / `UniTensor.bond_`

```
UniTensor.bonds() -> list[Bond]
UniTensor.bond(idx_or_label) -> Bond      # independent COPY
UniTensor.bond_(idx_or_label) -> Bond     # VIEW (shares the parent's Bond)

Access the tensor's legs (Bond objects).

Parameters
----------
idx_or_label : int or str
    Leg selector — a 0-based index or a leg label.

Returns
-------
list[Bond] / Bond
    `bonds()` returns per-leg copies. `bond(i)` returns an independent clone;
    `bond_(i)` returns a VIEW — mutating it in place (e.g. `redirect_()`) is
    visible on the parent (findings UT-M3/M4). Use `bond_` deliberately when a
    view is intended.
```

### `UniTensor.syms` / `UniTensor.signflip`

```
UniTensor.syms() -> list[Symmetry]
UniTensor.signflip() -> list[bool]

Return the tensor's symmetry objects (`syms`) or its per-leg fermion sign
flags (`signflip`).

Returns
-------
list[Symmetry] / list[bool]
    `signflip` is defined only for BlockFermionic tensors.

Raises
------
RuntimeError
    `signflip` on a non-fermionic (bosonic) tensor (finding UT-M7).
```

### `UniTensor.same_data` / `UniTensor.elem_exists` / `UniTensor.get_qindices`

```
UniTensor.same_data(other) -> bool
UniTensor.elem_exists(locator) -> bool
UniTensor.get_qindices(bond_idx) -> list[int]

Symmetry / storage queries. Parameters are renamed from the erased `arg0`
(finding UT-M5) so they are keyword-callable.

Parameters
----------
other : UniTensor
    Another tensor; `same_data` is True iff it shares storage with `self`
    (a `clone()` does not — it allocates independent storage).
locator : list[int]
    An element index tuple; `elem_exists` is True iff it lands in an allowed
    symmetry block (symmetric tensors only).
bond_idx : int
    A leg index; `get_qindices` returns its per-block qnum-index list
    (symmetric tensors only).

Returns
-------
bool / list[int]
    The query result.
```

### `UniTensor.get_total_qnums` / `UniTensor.get_blocks_qnums`

```
UniTensor.get_total_qnums(*, physical=False) -> list[Bond]   # was `getTotalQnums`
UniTensor.get_blocks_qnums() -> list[list[int]]

Return the tensor's quantum-number structure. These MUST operate on Block
(symmetric) tensors — 1.1.0 ships them bound but non-functional (they raise on
every tensor type, finding UT-M6).

Parameters
----------
physical : bool, keyword-only
    Restrict to physical (non-virtual) qnums.

Returns
-------
list[Bond] / list[list[int]]
    Total quantum numbers per leg, or the qnums of each block.

Raises
------
ValueError
    On a non-symmetric (Dense) tensor.
```

### R.2b C++ API (Doxygen)

C++ has no keyword-only parameters (`physical` is a default argument). The
accessors are `const` members; `bond_`/`bonds()` return references.

```cpp
/// @brief Number of legs / row-space legs / symmetry blocks / per-leg sizes.
cytnx_uint64 rank() const;
cytnx_int64 rowrank() const;
cytnx_uint64 nblocks() const;                  ///< was Nblocks (UT-M1)
std::vector<cytnx_uint64> shape() const;

/// @brief Element-type / device codes and their string forms; kind; name.
int dtype() const;                std::string dtype_str() const;
int device() const;               std::string device_str() const;
int uten_type() const;            std::string uten_type_str() const;
const std::string &name() const;

/// @brief Boolean classification predicates.
bool is_diag() const;             ///< stores only the diagonal
bool is_tag() const;              ///< legs carry bra/ket direction
bool is_braket_form() const;      ///< row legs ket, column legs bra
bool is_blockform() const;        ///< block (symmetric) storage
bool is_contiguous() const;       ///< contiguous element storage

/**
 * @brief Leg labels, and label -> index resolution.
 * @param label a leg label (parameter named, not arg0; UT-M5).
 * @return the labels, or the 0-based index of @p label.
 */
std::vector<std::string> labels() const;
cytnx_uint64 get_index(const std::string &label) const;

/**
 * @brief Access legs. bond() returns an independent clone; bond_() a view.
 * @param idx_or_label 0-based leg index or leg label.
 * @note bonds() yields per-leg copies (UT-M3); bond_() shares the parent's
 *       Bond so an in-place mutation is visible on the tensor (UT-M4).
 */
std::vector<Bond> bonds() const;
Bond bond(const cytnx_uint64 &idx) const;   Bond bond(const std::string &label) const;
Bond &bond_(const cytnx_uint64 &idx);       Bond &bond_(const std::string &label);

/**
 * @brief Symmetry / storage queries. Parameters named (not arg0; UT-M5).
 * @param other    another tensor (same_data: shares storage?).
 * @param locator  element index tuple (elem_exists: in an allowed block?).
 * @param bond_idx a leg index (get_qindices: per-block qnum indices).
 * @note signflip() is defined only for BlockFermionic tensors (UT-M7).
 */
std::vector<Symmetry> syms() const;
std::vector<bool> signflip() const;
bool same_data(const UniTensor &other) const;
bool elem_exists(const std::vector<cytnx_uint64> &locator) const;
std::vector<cytnx_uint64> get_qindices(const cytnx_uint64 &bond_idx) const;

/**
 * @brief Quantum-number structure. MUST work on Block tensors (UT-M6): the
 *        1.1.0 implementations are bound but raise on every type. Renamed
 *        getTotalQnums -> get_total_qnums (UT-M2).
 * @param physical restrict to physical (non-virtual) qnums.
 */
std::vector<Bond> get_total_qnums(const bool &physical = false) const;
std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const;
```
