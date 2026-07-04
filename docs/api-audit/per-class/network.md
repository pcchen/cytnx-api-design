# Network / LinOp / ncon — API audit

This document audits the three "network contraction" units of the Cytnx public
API in one place, because they share one job — *turn a description of a tensor
network into the contracted result* — and are meant to be used together:

- **`Network`** — a stateful builder: load a network skeleton (from a `.net`
  file, a list of strings, or programmatically), place `UniTensor`s into its
  named slots, then `Launch()` to contract. The heavyweight, reusable path.
- **`ncon`** — a one-shot free function using the standard *ncon index
  convention* (positive integers label contracted bonds, negatives label open
  legs). Internally it builds a `Network` via `construct` + `PutUniTensor` +
  `Launch`, so it is the ergonomic front-end to the same engine.
- **`LinOp`** — a linear-operator wrapper for iterative solvers (Lanczos/
  Arnoldi): either subclass it and override `matvec`, or use the `"mv_elem"`
  mode to pre-store sparse matrix elements. It is the "apply a matrix to a
  vector" primitive the eigensolvers consume.

The headline findings are behavioral, not cosmetic. **`Network.Contract(...)`,
the advertised one-shot static factory, builds a plan whose `Launch()`
segfaults** (P1, a B4 violation) — it is unusable in this 1.1.0 build, and
`ncon` is the working one-shot alternative. **`Network.clone()` silently drops
the placed tensors** yet `isAllset()` then returns a *misleading `True`*
(C2/P1) so the clone claims to be launch-ready and instead fails. On top of
that, nearly every `Network` method violates N1 (`PutUniTensor`, `Launch`,
`getOrder`, `isAllset`, …), and `LinOp` selects between two very different
behaviors with a stringly-typed `type` flag (C7).

Ground truth for behavior is `docs/api-audit/probes/network.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/network.py`; all 32 assertions `[PASS]`, exit 0). Ground
truth for static signatures is `cytnx_src/include/Network.hpp`,
`cytnx_src/include/LinOp.hpp`, `cytnx_src/include/ncon.hpp`; the C++ bodies
`cytnx_src/src/RegularNetwork.cpp`, `cytnx_src/src/LinOp.cpp`,
`cytnx_src/src/ncon.cpp`; the pybind bindings
`cytnx_src/pybind/network_py.cpp`, `linop_py.cpp`, `ncon_py.cpp`; and the
Python-side augmentation `cytnx_src/cytnx/Network_conti.py` (`Diagram`).

## Inventory

C++ declarations are read from the headers cited above; the *effective* Python
signatures are the pybind bindings (cross-checked live via
`tools/member_inventory.py Network`/`LinOp` and `cytnx.ncon.__doc__`).

### `Network` (18 members)

| Member | C++ (`Network.hpp` / `RegularNetwork.cpp`) | Python (pybind, live) |
|---|---|---|
| `Fromfile` | `void Fromfile(const string& fname, int network_type = Regular)` | `Fromfile(fname, network_type=0) -> None` |
| `FromString` | `void FromString(const vector<string>& contents, int network_type = Regular)` | `FromString(contents, network_type=0) -> None` |
| `Savefile` | `void Savefile(const string& fname)` | `Savefile(fname) -> None` |
| `PutUniTensor` | `void PutUniTensor(string\|uint64 name/idx, const UniTensor&, vector<string> label_order = {})` | overloaded `PutUniTensor(name\|idx, utensor, label_order=[]) -> None` |
| `PutUniTensors` | `void PutUniTensors(const vector<string>& name, const vector<UniTensor>&)` | `PutUniTensors(names, utensors) -> None` |
| `RmUniTensor` | `void RmUniTensor(string\|uint64 name/idx)` | overloaded `RmUniTensor(name\|idx) -> None` |
| `RmUniTensors` | `void RmUniTensors(const vector<string>& name)` | `RmUniTensors(names) -> None` |
| `Launch` | `UniTensor Launch(int network_type = Regular)` | `Launch(network_type=0) -> UniTensor` |
| `construct` | `void construct(alias, labels, outlabel={}, outrk=0, order="", optim=false, network_type=Regular)` | `construct(alias, labels, outlabel=[], outrk, order="", optim=False, network_type=0) -> None` |
| `setOrder` | `void setOrder(bool optimal = false, string contract_order = "")` | `setOrder(optimal=False, contract_order="") -> None` |
| `getOrder` | `string getOrder()` | `getOrder() -> str` |
| `isAllset` | *(pybind lambda over `_impl->tensors`)* | `isAllset() -> bool` |
| `isLoad` | *(pybind lambda: `tensors.size()==0 ? false : true`)* | `isLoad() -> bool` |
| `clear` | `void clear()` | `clear() -> None` |
| `clone` | `Network clone()` | `clone() -> Network` (also `__copy__`/`__deepcopy__`) |
| `PrintNet` | `void PrintNet()` | `PrintNet() -> None` |
| `Contract` | `static Network Contract(utensors, Tout, alias={}, contract_order="")` | `@staticmethod Contract(utensors, Tout, alias=[], contract_order="") -> Network` |
| `Diagram` | *(none — Python-only)* | `Diagram(self, outname=None, figsize=[6,5], engine="circo")` (in `Network_conti.py`) |

C++ `Network::getOptimalOrder(int network_type = Regular)` exists in the header
but its pybind line is **commented out** (`network_py.cpp:85-86`), so it is
**unbound** (P2). The old `Launch(optimal, contract_order)` overload is likewise
commented out; only `Launch(network_type)` is bound (P3).

### `LinOp` (7 members)

| Member | C++ (`LinOp.hpp` / `LinOp.cpp`) | Python (pybind, live) |
|---|---|---|
| `matvec` | `virtual Tensor matvec(const Tensor&)` / `virtual UniTensor matvec(const UniTensor&)` | overloaded `matvec(Tin: Tensor\|UniTensor) -> Tensor\|UniTensor` (overridable) |
| `set_elem` | `template<T> void set_elem(uint64 i, uint64 j, T elem, bool check_exists = true)` | `set_elem(i, j, elem, check_exists=True) -> None` (all dtypes) |
| `set_device` | `void set_device(const int& device)` | `set_device(arg0: int) -> None` |
| `set_dtype` | `void set_dtype(const int& dtype)` | `set_dtype(arg0: int) -> None` |
| `device` | `int device() const` | `device() -> int` |
| `dtype` | `int dtype() const` | `dtype() -> int` |
| `nx` | `uint64 nx() const` | `nx() -> int` |

Constructor: `LinOp(type: str, nx: int, dtype=Type.Double, device=Device.cpu)`;
`type` must be `"mv"` or `"mv_elem"`. `LinOp::_print`, `operator()(i,j)`, and
`_mv_elemfunc` are internal (not public members).

### `ncon` (free function)

`ncon(tensor_list_in, connect_list_in, check_network=False, optimize=False,
cont_order=[], out_labels=[]) -> UniTensor` (`ncon.hpp` / `ncon_py.cpp`;
C++ and Python signatures agree exactly). `connect_list_in` is one integer list
per tensor: a **positive** integer names a bond contracted between the two
tensors that share it; a **negative** integer names an open output leg, ordered
`-1, -2, …` in the result.

## Parity findings

Behavioral claims below are backed by a passing `report(...)` assertion in
`docs/api-audit/probes/network.py`; signature/binding claims are static
source-vs-wheel facts read from the cited files.

- **P1 — `Network.Contract(...).Launch()` segfaults instead of raising (B4).**
  `Network::Contract` (a `def_static`, `network_py.cpp:118`) builds a plan via
  `Contract_plan`; calling `.Launch()` on the result crashes the process with
  `SIGSEGV` for every reasonable `Tout` spelling (`"i;j"`, `"i,j"`, …). Probed
  in a subprocess so the crash cannot kill the probe: *"Network.Contract(...)
  builds a plan whose Launch() SEGFAULTS (child exits on SIGSEGV, returncode
  -11) instead of raising a catchable exception"* `[PASS]`. This is the sharpest
  B4 violation in the unit — the advertised one-shot factory is unusable in
  1.1.0. The working one-shot path is `ncon`; the working reusable path is
  `FromString`/`construct` + `PutUniTensor` + `Launch` (both probed `[PASS]`).

- **P2 — `Network::getOptimalOrder` (C++ `Network.hpp:407`) has no Python
  binding.** Its `.def` line is commented out (`network_py.cpp:85-86`), so
  `cytnx.Network` exposes no way to ask for the optimal contraction order from
  Python — only `setOrder(optimal=True, …)` + `getOrder()`. A real
  C++-source-vs-Python-wheel member gap.

- **P3 — `Network::Launch`'s C++ `(optimal, contract_order)` overload is
  dropped in Python; the bound `Launch` takes only `network_type`, which is a
  dead flag.** The `Launch(optimal, contract_order)` binding is commented out
  (`network_py.cpp:87-88`); only `Launch(network_type=Regular)` is bound
  (`network_py.cpp:100`), and every value other than `Regular` raises
  `"[Developing] currently only support regular type network"` (the Fermion
  path is a stub). So order must be set beforehand via `setOrder`, and
  `network_type` is a parameter with exactly one legal value.

- **P4 — `Network.Diagram` is a Python-only method with no C++ counterpart.**
  It is injected onto the class via `@add_method(Network)` in
  `cytnx/Network_conti.py:43` and draws the network with graphviz; every other
  `Network` method is a pybind binding of a C++ member. Probed: *"Network.Diagram
  is a Python-only method injected via @add_method … it has NO C++ Network member
  behind it"* `[PASS]`. It also `print`s an error and `exit(99)`s (not raises)
  when called on an un-loaded network — a B4-adjacent smell (the `isLoad()`
  guard is probed `[PASS]`, the draw path itself is not exercised to avoid the
  graphviz dependency).

- **P5 — `LinOp.dtype()`/`device()` return bare integer codes, not `Type`/
  `Device` objects.** Probed: *"nx()/dtype()/device() echo the ctor args
  (dtype/device are returned as INTEGER codes, not Type/Device objects)"*
  `[PASS]` (`op.dtype() == int(Type.Double)`). This matches the rest of Cytnx
  (`Tensor.dtype()` also returns an int code), so it is *consistent* with the
  wider surface and recorded as a low-severity observation rather than a
  divergence — but it does mean the enum modelling from `enums.md` is not used
  at the accessor boundary.

## Consistency findings

- **C1 — nearly every `Network` method violates N1 (`snake_case`).**
  Capitalized-verb methods: `PutUniTensor`, `PutUniTensors`, `RmUniTensor`,
  `RmUniTensors`, `Launch`, `PrintNet`, `FromString`, `Fromfile`, `Savefile`,
  `Contract`, `Diagram`. CamelCase methods: `getOrder`, `setOrder`, `isAllset`,
  `isLoad`. Only `construct`, `clear`, and `clone` already comply. N1 renames
  them all (`put_unitensor`, `launch`, `get_order`, `is_all_set`, …). `LinOp` is
  by contrast already snake_case throughout — an internal inconsistency between
  the two sibling classes.

- **C2 — `isAllset()` returns a misleading `True` on a `clone`, and
  `clone()` silently drops the placed tensors (B1/B4).** `RegularNetwork::clone`
  copies the skeleton (`name2pos`/`CtTree`/`names`/`label_arr`/`TOUT_*`) but
  **not** the `tensors` vector, so a clone of a fully-set network has an *empty*
  tensor list. `isAllset()` loops over that empty vector and returns `True`
  vacuously, while `isLoad()` (which tests `tensors.size()`) correctly returns
  `False`. Probed both ways: *"the clone reports isLoad() False … while the
  original is still isLoad() True"* `[PASS]` and *"the clone's isAllset() returns
  a MISLEADING True … an internal inconsistency (isLoad() and isAllset()
  disagree on the same clone)"* `[PASS]`; the consequence is probed too:
  *"clone().Launch() then fails with a catchable exception … because the clone
  dropped its tensors"* `[PASS]`. Two coupled defects: `clone` should deep-copy
  the tensors (B1: a clone must be independent *and complete*), and `isAllset`
  must not report readiness for an empty network.

- **C3 — `isAllset`/`isLoad` violate N5 (predicate naming).** Both return
  `bool` but read as verbs, not questions: `isAllset` → `is_all_set`, `isLoad`
  → `is_loaded`. `isLoad` in particular is grammatically ambiguous ("is load"),
  the exact N5 anti-pattern.

- **C4 — `Rm` is an abbreviation where the sibling `Put` is spelled out
  (N4).** `PutUniTensor`/`PutUniTensors` spell the verb in full but
  `RmUniTensor`/`RmUniTensors` abbreviate "remove" to "Rm" — two vocabularies
  for one CRUD pair. N4's "same concept → same name" spirit wants
  `remove_unitensor` alongside `put_unitensor`.

- **C5 — the compound-verb factory names are internally inconsistent even
  before N1 (`Fromfile` vs `FromString` vs `Savefile`).** `Fromfile` lowercases
  the second word, `FromString` capitalizes it, and `Savefile` again lowercases
  it — three casing conventions for the same "verb+noun" pattern in one class.
  N1 resolves all to `from_file` / `from_string` / `save_file`.

- **C6 — `construct`'s parameter order puts a required argument after an
  optional one (`outlabel=[]` then required `outrk`).** The bound signature is
  `construct(alias, labels, outlabel=[], outrk, order="", optim=False,
  network_type=0)` — `outrk` has no default but follows `outlabel` which does,
  forcing every call to supply `outrk` positionally (or all four leading args by
  keyword). An N4-adjacent signature smell; `outrk` should either precede the
  optional args or gain a default.

- **C7 — `LinOp` selects two very different behaviors with a stringly-typed
  `type` flag (design smell).** `type="mv"` means "subclass and override
  `matvec`" (base `matvec` raises, probed `[PASS]`); `type="mv_elem"` means
  "pre-store sparse elements via `set_elem`, and `matvec(UniTensor)` is illegal"
  (probed `[PASS]`). One class, two disjoint contracts chosen by a magic string
  (any other string raises, probed `[PASS]`). Better modelled as two classes or
  a real enum; at minimum the mode belongs in the type system, not a `str`.

- **C8 — `LinOp.nx` is an abbreviated name for the operator dimension (N5).**
  `nx` (the size of the square operator / length of the input vector) is
  cryptic; `dim` or `dimension` reads as intended. Low severity.

- **C9 — `LinOp.matvec` keeps a compressed, un-underscored name — deliberately
  (positive note).** Strict N1 would suggest `mat_vec`, but "matvec" is the
  established linear-algebra term (as "svd"/"qr" are), and it is already
  lowercase. Recorded as an accepted exception (cf. `enums.md` C5), not a
  violation to fix.

## Recommendation

Every live public member of `Network` (18) and `LinOp` (7), plus the `ncon`
free function, appears below tagged **keep / add / rename / remove**, with the
`N`/`B`/`P`/`C` id that motivates the verdict.

### `Network`

| Member | Verdict | Rationale |
|---|---|---|
| `FromString` | rename | → `from_string` (C1/N1). Reliable skeleton loader (probe `[PASS]`). |
| `Fromfile` | rename | → `from_file` (C1/C5/N1); fixes the `Fromfile`-vs-`FromString` casing split. |
| `Savefile` | rename | → `save_file` (C1/C5/N1). |
| `PutUniTensor` | rename | → `put_unitensor` (C1/N1). Name+idx overloads and `label_order` permute all probed `[PASS]`. |
| `PutUniTensors` | rename | → `put_unitensors` (C1/N1). |
| `RmUniTensor` | rename | → `remove_unitensor` (C1/C4/N1): un-abbreviate and `snake_case`. |
| `RmUniTensors` | rename | → `remove_unitensors` (C1/C4/N1). |
| `Launch` | rename | → `launch` (C1/N1); drop the dead `network_type` flag (P3). |
| `construct` | keep | Already `snake_case`; the low-level builder `ncon` uses (probe `[PASS]`). Fix `outrk` arg order (C6). |
| `setOrder` | rename | → `set_order` (C1/N1). |
| `getOrder` | rename | → `get_order` (C1/N1). Round-trips `set_order` (probe `[PASS]`). |
| `isAllset` | rename | → `is_all_set` (C1/C3/N1/N5) **and fix C2**: must not report `True` for a network with no placed tensors. |
| `isLoad` | rename | → `is_loaded` (C1/C3/N1/N5). |
| `clear` | keep | Already compliant; wipes the skeleton (probe `[PASS]`). |
| `clone` | keep | Keep the name but **fix C2**: deep-copy the placed `tensors` too, so the clone is independent *and* launch-ready. |
| `PrintNet` | rename | → `print_net` (C1/N1). |
| `Contract` | remove | Segfaults on `Launch` (P1/B4) and duplicates `ncon`'s one-shot role; drop it (or fully re-implement + rename `contract`). |
| `Diagram` | rename | → `diagram` (C1/N1). Python-only viz helper (P4); keep as an optional graphviz convenience, raise (not `exit`) when un-loaded. |

Cross-cutting for `Network`: expose `get_optimal_order` (bind the unbound
`getOptimalOrder`, P2); make `Launch` order-driven via `set_order` and remove
the single-valued `network_type` flag (P3/C6).

### `LinOp`

| Member | Verdict | Rationale |
|---|---|---|
| `matvec` | keep | The core apply-operator method; override-dispatch and both modes probed `[PASS]`. Keep the established name (C9). |
| `set_elem` | keep | Builds the `"mv_elem"` sparse matrix (`out[i] += val*in[j]`, probe `[PASS]`). |
| `set_device` | keep | In-place device setter (probe `[PASS]`). |
| `set_dtype` | keep | In-place dtype setter (probe `[PASS]`). |
| `device` | keep | Device-code getter (returns int, P5). |
| `dtype` | keep | Dtype-code getter (returns int, P5). |
| `nx` | keep | Operator dimension; keep functionally, consider `dim` for clarity (C8). |

Cross-cutting for `LinOp`: replace the stringly-typed `type` ctor flag with two
classes or an enum (C7); the constructor's `type`/`nx`/`dtype`/`device`
validation already raises correctly (probe `[PASS]`).

### `ncon`

| Member | Verdict | Rationale |
|---|---|---|
| `ncon` | keep | Correct, already `snake_case` free function; index convention, output ordering, scalar contraction, and `check_network` validation all probed `[PASS]`. It is the working one-shot contractor (unlike `Network.Contract`, P1). Consider defaulting `check_network=True` so malformed networks are caught rather than risking a `Launch` crash. |

## Docstrings

Numpy-style docstrings for every member tagged `keep` or `rename` above, under
its recommended name (current name shown in backticks so the block maps back to
its Recommendation row). `Contract` is `remove` and needs none.

### `FromString` → `from_string`

```
Load a network skeleton from a list of description strings.

Parameters
----------
contents : list of str
    One 'name: labels' line per tensor; ';' splits row-rank from col-rank
    labels. 'TOUT' names the output leg order, 'ORDER' the contraction order.
network_type : int, optional
    Only Regular (0) is supported (P3); other values raise.

Returns
-------
None
    Mutates the Network in place; `is_loaded()` becomes True afterwards.

Notes
-----
Renamed from `FromString` (C1/N1). Reliable build path:
from_string -> put_unitensor -> launch.
```

### `Fromfile` → `from_file`

```
Load a network skeleton from a '.net' file (same format as `from_string`).

Parameters
----------
fname : str
    Path to the network file.
network_type : int, optional
    Only Regular (0) is supported.

Returns
-------
None

Notes
-----
Renamed from `Fromfile` (C1/C5/N1). `Network(fname)` is the equivalent
constructor. Round-trips `save_file` (probe [PASS]).
```

### `Savefile` → `save_file`

```
Write the current network skeleton to disk.

Parameters
----------
fname : str
    Output path; Cytnx appends the '.net' suffix (probe [PASS]).

Returns
-------
None

Notes
-----
Renamed from `Savefile` (C1/C5/N1). Saves the skeleton only, not any placed
tensors.
```

### `PutUniTensor` → `put_unitensor`

```
Place a UniTensor into a named or positional network slot.

Parameters
----------
name : str or int
    The slot's alias (str) or its index (int) — two overloads.
utensor : UniTensor
    The tensor to place; its labels must match the slot's declared labels.
label_order : list of str, optional
    If given, the tensor is permuted to this leg order before placing.

Returns
-------
None

Notes
-----
Renamed from `PutUniTensor` (C1/N1). The `label_order` permutation is applied
to a COPY (`utensor.permute(...)`), so the caller's tensor is NOT mutated
(B1, probe [PASS]).
```

### `PutUniTensors` → `put_unitensors`

```
Place several UniTensors in one call.

Parameters
----------
names : list of str
    Slot aliases.
utensors : list of UniTensor
    Tensors to place, paired positionally with `names`.

Returns
-------
None

Notes
-----
Renamed from `PutUniTensors` (C1/N1). Equivalent to repeated `put_unitensor`.
```

### `RmUniTensor` → `remove_unitensor`

```
Clear a placed tensor from a named or positional slot, reverting it to Void.

Parameters
----------
name : str or int
    The slot's alias (str) or index (int) — two overloads.

Returns
-------
None

Notes
-----
Renamed from `RmUniTensor` (C1/C4/N1): un-abbreviated to match `put_unitensor`.
After removal `is_all_set()` reverts to False (probe [PASS]).
```

### `RmUniTensors` → `remove_unitensors`

```
Clear several placed tensors in one call.

Parameters
----------
names : list of str
    Slot aliases to clear.

Returns
-------
None

Notes
-----
Renamed from `RmUniTensors` (C1/C4/N1).
```

### `Launch` → `launch`

```
Contract the fully-set network and return the result.

Returns
-------
UniTensor
    The contracted output; its leg order follows the 'TOUT' line.

Raises
------
RuntimeError
    If the network is not fully set.

Notes
-----
Renamed from `Launch` (C1/N1). The recommended API drops the single-valued
`network_type` flag (P3); set the contraction order beforehand with
`set_order`. Requires every slot filled (`is_all_set()` True).
```

### `construct`

```
Build a network skeleton programmatically (the primitive `ncon` uses).

Parameters
----------
alias : list of str
    Tensor names.
labels : list of list of str
    Per-tensor leg labels.
outlabel : list of str, optional
    Output leg labels.
outrk : int
    Output row-rank.
order : str, optional
    Contraction order string.
optim : bool, optional
    Whether to optimize the contraction order.

Returns
-------
None

Notes
-----
Kept (already N1-compliant). Recommended fix: move `outrk` before the optional
`outlabel` (or give it a default) so no required arg follows an optional one
(C6).
```

### `setOrder` → `set_order`

```
Record the contraction order for the network.

Parameters
----------
optimal : bool, optional
    If True, use the optimal order (see `get_optimal_order`).
contract_order : str, optional
    An explicit order string, e.g. '(A,B)'.

Returns
-------
None

Notes
-----
Renamed from `setOrder` (C1/N1). `get_order` reads the stored value back
(probe [PASS]).
```

### `getOrder` → `get_order`

```
Return the currently-recorded contraction order string.

Returns
-------
str
    The order set by `set_order`, or a placeholder if none is set.

Notes
-----
Renamed from `getOrder` (C1/N1).
```

### `isAllset` → `is_all_set`

```
Whether every declared slot holds a placed tensor.

Returns
-------
bool
    True only when all slots are filled.

Notes
-----
Renamed from `isAllset` (C1/C3/N1/N5). IMPORTANT (C2): the current binding
returns True *vacuously* for a network with no tensor slots (e.g. after
`clone`, which drops the tensor list) — the recommended API returns False for
an empty/incomplete network so readiness is never over-reported.
```

### `isLoad` → `is_loaded`

```
Whether a network skeleton has been loaded.

Returns
-------
bool
    True after `from_string`/`from_file`/`construct`.

Notes
-----
Renamed from `isLoad` (C1/C3/N1/N5). Distinct from `is_all_set`: a loaded
network may still be missing placed tensors.
```

### `clear`

```
Wipe the network skeleton (names, labels, and placed tensors).

Returns
-------
None

Notes
-----
Kept (already N1-compliant). Afterwards `is_loaded()` is False (probe [PASS]).
```

### `clone`

```
Return an independent copy of the network.

Returns
-------
Network
    A deep copy of the skeleton.

Notes
-----
Kept by name, but the recommended API FIXES C2: the current `clone` copies only
the skeleton and DROPS the placed tensors, so the clone reports `is_loaded()`
False while `is_all_set()` lies True and `launch()` then fails (probe [PASS]).
A correct `clone` deep-copies the placed tensors too (B1) so the copy is both
independent and launch-ready.
```

### `PrintNet` → `print_net`

```
Print a human-readable diagram of the network to stdout.

Returns
-------
None

Notes
-----
Renamed from `PrintNet` (C1/N1). The pybind `__repr__` already routes through
this with a scoped-ostream redirect.
```

### `Diagram` → `diagram`

```
Draw the network as a graph (graphviz) and return the figure.

Parameters
----------
outname : str, optional
    Output filename; defaults to the loaded network's filename.
figsize : list of int, optional
    Figure size, default [6, 5].
engine : str, optional
    Graphviz layout engine, default "circo".

Returns
-------
A drawn network figure object.

Notes
-----
Renamed from `Diagram` (C1/N1). Python-only helper (P4). The recommended API
RAISES (not `print`+`exit`) when the network is not loaded (`is_loaded()`
False, probe [PASS]).
```

### `ncon`

```
Contract a tensor network using the ncon index convention.

Parameters
----------
tensor_list_in : sequence of UniTensor
    The tensors to contract.
connect_list_in : sequence of sequence of int
    One integer list per tensor. A POSITIVE integer labels a bond contracted
    between the two tensors that share it; a NEGATIVE integer labels an open
    output leg. Output legs are ordered -1, -2, ... (probe [PASS]).
check_network : bool, optional
    If True, verify each positive label appears exactly twice, raising
    RuntimeError otherwise (probe [PASS]). Defaults to False; recommend True.
optimize : bool, optional
    Whether to search for an optimal contraction order.
cont_order : sequence of int, optional
    Explicit contraction order over the positive bond labels.
out_labels : sequence of str, optional
    Labels for the output legs.

Returns
-------
UniTensor
    The contracted result. A fully-contracted (all-positive) network yields a
    rank-1 scalar (probe [PASS]).

Notes
-----
Does NOT mutate its inputs (B1, probe [PASS]). Internally builds a Network via
`construct`+`put_unitensor`+`launch`; it is the working one-shot contractor,
unlike `Network.Contract` which segfaults (P1).
```

### `matvec`

```
Apply the linear operator to an input vector.

Parameters
----------
Tin : Tensor or UniTensor
    The input vector (two overloads).

Returns
-------
Tensor or UniTensor
    The result of the operator acting on `Tin`.

Raises
------
RuntimeError
    For a 'mv' operator whose `matvec` was not overridden, or for a 'mv_elem'
    operator given a UniTensor (probe [PASS]).

Notes
-----
Kept (C9: established term). Override this in a Python subclass of a 'mv'
LinOp; the override is dispatched via pybind11's trampoline (probe [PASS]).
```

### `set_elem`

```
Store one sparse matrix element for a 'mv_elem' operator.

Parameters
----------
i, j : int
    Row and column index.
elem : number
    The matrix entry; accumulated as out[i] += elem * in[j].
check_exists : bool, optional
    If True (default), raise on a duplicate (i, j).

Returns
-------
None

Notes
-----
Kept. Only meaningful for `type="mv_elem"` LinOps (probe [PASS]).
```

### `set_device`

```
Set the operator's compute device in place.

Parameters
----------
device : int
    A device code (e.g. Device.cpu).

Returns
-------
None

Notes
-----
Kept. `device()` reflects the new code afterwards (probe [PASS]).
```

### `set_dtype`

```
Set the operator's dtype in place.

Parameters
----------
dtype : int
    A Type code (e.g. Type.Double).

Returns
-------
None

Notes
-----
Kept. `dtype()` reflects the new code afterwards (probe [PASS]).
```

### `device`

```
Return the operator's device code.

Returns
-------
int
    A Device code (P5: an int, not a Device object).

Notes
-----
Kept.
```

### `dtype`

```
Return the operator's dtype code.

Returns
-------
int
    A Type code (P5: an int, not a Type object).

Notes
-----
Kept.
```

### `nx`

```
Return the operator's dimension (input-vector length).

Returns
-------
int

Notes
-----
Kept functionally; `dim` would read more clearly (C8).
```

## Change table

Clean-slate migration map: `current (C++ / Python name) → recommended`. Members
not listed (`construct`, `clear`, `clone` on `Network`; all seven `LinOp`
members; `ncon`) keep their current name — but `clone`/`isAllset` also carry the
C2 behavior fix, and `Contract` is removed.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Network.FromString` | `from_string` | N1 casing (C1) |
| `Network.Fromfile` | `from_file` | N1 casing (C1/C5) |
| `Network.Savefile` | `save_file` | N1 casing (C1/C5) |
| `Network.PutUniTensor` | `put_unitensor` | N1 casing (C1) |
| `Network.PutUniTensors` | `put_unitensors` | N1 casing (C1) |
| `Network.RmUniTensor` | `remove_unitensor` | N1 + un-abbreviate (C1/C4) |
| `Network.RmUniTensors` | `remove_unitensors` | N1 + un-abbreviate (C1/C4) |
| `Network.Launch` | `launch` (drop `network_type`) | N1 casing (C1) + P3 |
| `Network.setOrder` | `set_order` | N1 casing (C1) |
| `Network.getOrder` | `get_order` | N1 casing (C1) |
| `Network.isAllset` | `is_all_set` (+ C2 fix) | N1/N5 (C1/C3) + vacuous-True bug (C2) |
| `Network.isLoad` | `is_loaded` | N1/N5 (C1/C3) |
| `Network.PrintNet` | `print_net` | N1 casing (C1) |
| `Network.Diagram` | `diagram` (raise, not exit) | N1 casing (C1) + P4 |
| `Network.Contract` | *(removed)* → use `ncon` | segfaults on Launch (P1/B4), duplicates `ncon` |
| `Network.clone` | `clone` (deep-copy tensors) | behavior fix (C2/B1) |
| `Network.construct` | `construct` (reorder `outrk`) | required-after-optional arg (C6) |
| `Network::getOptimalOrder` (unbound) | `get_optimal_order` (bind it) | source-vs-wheel gap (P2) |
| `LinOp(type="mv"/"mv_elem")` | two classes / an enum | stringly-typed mode (C7) |
| `LinOp.nx` | `nx` (consider `dim`) | abbreviated name (C8) |

Every other public member keeps its current name and behavior.
