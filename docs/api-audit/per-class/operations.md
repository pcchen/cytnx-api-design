# Operation submodules — API audit (`algo`, `random`, `physics`, `qgates`)

This document audits the four "free-function toolbox" submodules of the Cytnx
public API in one place, because none is a class — each is a flat namespace of
free functions that operate on `Tensor`/`UniTensor`/`Storage` — and they share
the same modelling question: *how should a small family of stateless
operations be exposed to Python?* The four are:

- **`algo`** — dense-array algorithms (`Sort`, `Concatenate`, and the
  `Vstack`/`Hstack`/`Vsplit`/`Hsplit` matrix stack/split family) on `Tensor`.
- **`random`** — random-fill generators: the in-place `normal_`/`uniform_`
  (fill an existing `Tensor`/`Storage`/`UniTensor`) and the allocate-and-return
  `normal`/`uniform`.
- **`physics`** — spin-representation matrices `spin(S, Comp)` and Pauli
  matrices `pauli(Comp)`, returned as bare `Tensor`s.
- **`qgates`** — a quantum-gate library (`pauli_x/y/z`, `hadamard`,
  `phase_shift`, `swap`, `sqrt_swap`, `toffoli`, `cntl_gate_2q`) returned as
  `UniTensor`s.

The headline structural finding is that the four modules are **not named
uniformly**: `algo`'s entire surface is Capitalized-verb functions
(`Sort`/`Concatenate`/…), violating N1, while `random`/`physics`/`qgates` are
already `snake_case`. On top of that there are two genuine correctness/behavior
findings: `qgates.hadamard()` returns the **unnormalized** matrix
`[[1,1],[1,-1]]` (missing the `1/sqrt(2)` factor, so it is not unitary, P8),
and the Pauli-X operator is offered **twice** under two names and two container
types (`physics.pauli('x')` → `Tensor` vs `qgates.pauli_x()` → `UniTensor`, C4).

Ground truth for behavior is `docs/api-audit/probes/operations.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/operations.py`; all 37 assertions `[PASS]`, exit 0).
Ground truth for static signatures is `cytnx_src/include/algo.hpp`,
`cytnx_src/include/random.hpp`, `cytnx_src/include/Physics.hpp` (which declares
`physics`, `operators`, and `qgates` in one header), and the pybind bindings
`cytnx_src/pybind/algo_py.cpp`, `cytnx_src/pybind/random_py.cpp`, and
`cytnx_src/pybind/physics_related_py.cpp` (which registers both the `physics`
and `qgates` submodules). Live pybind signatures are cross-checked with
`tools/member_inventory.py {algo,random,physics,qgates}`.

## Inventory

C++ declarations are read from the headers cited above; the *effective* Python
signature is read from the pybind binding (the ground truth for the callable
signature) and cross-checked against `member_inventory.py`.

### `algo` (6 members — all bound as Capitalized free functions)

| Member | C++ (`algo.hpp`) | Python (live, `algo_py.cpp`) |
|---|---|---|
| `Sort` | `Tensor Sort(const Tensor &Tin)` | `Sort(Tn: Tensor) -> Tensor` |
| `Concatenate` | `Tensor Concatenate(Tensor T1, Tensor T2)` | `Concatenate(T1: Tensor, T2: Tensor) -> Tensor` |
| `Vstack` | `Tensor Vstack(const std::vector<Tensor> &)` | `Vstack(Tlist: Sequence[Tensor]) -> Tensor` |
| `Hstack` | `Tensor Hstack(const std::vector<Tensor> &)` | `Hstack(Tlist: Sequence[Tensor]) -> Tensor` |
| `Vsplit` | `std::vector<Tensor> Vsplit(const Tensor &Tin, const std::vector<cytnx_uint64> &dims)` | `Vsplit(Tin: Tensor, dims: Sequence[int]) -> list[Tensor]` |
| `Hsplit` | `std::vector<Tensor> Hsplit(const Tensor &Tin, const std::vector<cytnx_uint64> &dims)` | `Hsplit(Tin: Tensor, dims: Sequence[int]) -> list[Tensor]` |

The C++ out-parameter in-place split variants `Vsplit_`/`Hsplit_`
(`algo.hpp:108,121`) are **not bound** at all (P1); nor is any in-place
`Sort_` (N2 gap, C2). The `Tin`/`Tn`/`T1`/`T2`/`Tlist` argument names are
inconsistent across the six functions (C3/N4).

### `random` (4 members — 2 in-place fills + 2 allocators)

| Member | C++ (`random.hpp`) | Python (live, `random_py.cpp`) |
|---|---|---|
| `normal_` | `void normal_(Tensor&/Storage&/UniTensor&, mean, std, seed)` | `normal_(Tin\|Sin, mean, std, seed=-1) -> None` (3 overloads) |
| `uniform_` | `void uniform_(Tensor&/Storage&/UniTensor&, low=0, high=1, seed)` | `uniform_(Tin\|Sin, low=0.0, high=1.0, seed=-1) -> None` (3 overloads) |
| `normal` | `Tensor normal(Nelem\|shape, mean, std, device=cpu, seed, dtype=Double)` | `normal(Nelem: int\|Sequence[int], mean, std, device=-1, seed=-1, dtype=3) -> Tensor` (2 overloads) |
| `uniform` | `Tensor uniform(Nelem\|shape, low, high, device=cpu, seed, dtype=Double)` | `uniform(Nelem: int\|Sequence[int], low, high, device=-1, seed=-1, dtype=3) -> Tensor` (2 overloads) |

The `seed=-1` Python default is a sentinel: the pybind lambda maps `-1` onto a
fresh `__static_random_device()` draw (`random_py.cpp:31–35`), so `-1` means
"use device entropy". There is **no `random.seed` global setter** — seeding is
per call (P5). The deprecated `Make_normal`/`Make_uniform` templates and the
`random_tensor` helper (`random.hpp:212–230`) are **not bound** (P4).
`uniform_` has default bounds `low=0.0`/`high=1.0` but `normal_` has no default
mean/std (C5).

### `physics` (2 members — return a bare `Tensor`)

| Member | C++ (`Physics.hpp`) | Python (live, `physics_related_py.cpp`) |
|---|---|---|
| `spin` | `Tensor spin(const cytnx_double &S, const std::string &Comp, const int &device=cpu)` | `spin(S: float, Comp: str, device=-1) -> Tensor` |
| `pauli` | `Tensor pauli(const std::string &Comp, const int &device=cpu)` | `pauli(Comp: str, device=-1) -> Tensor` |

`Physics.hpp` also declares a `cytnx::operators` namespace (`Sz_shalf`,
`Sp_shalf`, `Sn_shalf`) — these are **not bound** in Python (P2). The `S`/`Comp`
argument names use non-`snake_case` spelling (C3/N5).

### `qgates` (9 members — return a `UniTensor`)

| Member | C++ (`Physics.hpp`) | Python (live, `physics_related_py.cpp`) |
|---|---|---|
| `pauli_x` | `UniTensor pauli_x(const int &device=cpu)` | `pauli_x(device=-1) -> UniTensor` |
| `pauli_y` | `UniTensor pauli_y(const int &device=cpu)` | `pauli_y(device=-1) -> UniTensor` |
| `pauli_z` | `UniTensor pauli_z(const int &device=cpu)` | `pauli_z(device=-1) -> UniTensor` |
| `hadamard` | `UniTensor hadamard(const int &device=cpu)` | `hadamard(device=-1) -> UniTensor` |
| `phase_shift` | `UniTensor phase_shift(const cytnx_double &phase, const int &device=cpu)` | `phase_shift(phase: float, device=-1) -> UniTensor` |
| `swap` | `UniTensor swap(const int &device=cpu)` | `swap(device=-1) -> UniTensor` |
| `sqrt_swap` | `UniTensor sqrt_swap(const int &device=cpu)` | `sqrt_swap(device=-1) -> UniTensor` |
| `toffoli` | `UniTensor toffoli(const int &device=cpu)` | `toffoli(device=-1) -> UniTensor` |
| `cntl_gate_2q` | `UniTensor cntl_gate_2q(const UniTensor &gate_1q)` | `cntl_gate_2q(gate_1q: UniTensor) -> UniTensor` |

The entire `qgates` namespace is wrapped in a Doxygen `/// @cond … @endcond`
block in `Physics.hpp` (lines 73–89), i.e. it is undocumented in the C++ API
reference even though it is fully bound (C6).

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/operations.py`.

- **P1 — `algo`'s C++ out-parameter split variants `Vsplit_`/`Hsplit_` are
  unbound in Python.** `algo.hpp` declares `void Vsplit_(out, Tin, dims)` and
  `void Hsplit_(...)` (the "write into an output list" forms), but
  `algo_py.cpp` binds only the six returns-new functions. Probed: *"algo's C++
  out-parameter in-place split variants Vsplit_/Hsplit_ … are NOT bound in
  Python …"* `[PASS]`. Low severity — the returns-new `Vsplit`/`Hsplit` cover
  the same capability — but it is a real header-vs-wheel surface gap.

- **P2 — `Physics.hpp`'s `cytnx::operators` spin-half operators
  (`Sz_shalf`/`Sp_shalf`/`Sn_shalf`) are unbound.** They are declared in the
  same header that defines `physics`/`qgates` but no `operators` submodule is
  registered by `physics_related_py.cpp`. (Static header-vs-wheel gap; not
  separately probed because there is no Python member to exercise — the finding
  is the *absence*, cross-checked against the live `dir(physics)` in the probe's
  member coverage.)

- **P4 — `random`'s deprecated aliases `Make_normal`/`Make_uniform` and the
  `random_tensor` helper are unbound.** `random.hpp:212–230` declares
  `random_tensor` plus the `[[deprecated]]` `Make_normal`/`Make_uniform`
  templates; none is registered in `random_py.cpp`. Probed: *"… random.hpp's
  deprecated Make_normal/Make_uniform aliases and the random_tensor helper are
  NOT bound in Python either"* `[PASS]`. Correct for the deprecated pair;
  `random_tensor` (a genuinely useful any-dtype generator) being absent is a
  capability gap noted in the Recommendation.

- **P5 — seeding is per-call via `seed=-1`, not a module-level RNG state.**
  `random` exposes no `seed`/`set_seed` global; every generator takes a `seed=`
  argument whose default `-1` is remapped to a fresh `__static_random_device()`
  draw inside the pybind lambda. Probed: *"random has NO global 'seed' setter:
  seeding is per-call via the seed= argument (seed=-1 … draws fresh device
  entropy) …"* `[PASS]`, and reproducibility is confirmed both ways: *"random.
  normal_ with the SAME seed reproduces the SAME fill …"* and *"… with a
  DIFFERENT seed produces a different fill"* `[PASS]`. This is a parity note
  (the C++ default `__static_random_device()` and the Python `-1` sentinel
  denote the same "entropy" behavior), not a divergence.

- **P8 — `qgates.hadamard()` returns the UNNORMALIZED matrix `[[1,1],[1,-1]]`,
  missing the `1/sqrt(2)` factor, so it is not a unitary Hadamard gate.**
  Probed two ways: *"qgates.hadamard() returns the UNNORMALIZED matrix
  [[1,1],[1,-1]] … NOT the standard (unitary) Hadamard gate"* `[PASS]`, and the
  consequence *"… qgates.hadamard() is NOT unitary: H @ H^dagger == 2*I, not
  I"* `[PASS]`. This is a genuine **correctness bug** in the returned value
  (every other gate probed — `pauli_x/y/z`, `phase_shift`, `swap`, `sqrt_swap`,
  `toffoli`, `cntl_gate_2q` — is correct/unitary), flagged for a value fix in
  the Recommendation. It is language-agnostic (the same C++ routine backs the
  binding), so it is a defect of the operation itself rather than a
  C++-vs-Python divergence.

- **P9 — `random`/`physics`/`qgates` errors surface as catchable Python
  exceptions (B4).** Deliberately invalid calls raise `RuntimeError` rather than
  returning a sentinel or aborting: *"algo.Concatenate rejects non-1-D input
  with a catchable RuntimeError …"* `[PASS]` and *"physics.spin/pauli raise a
  catchable RuntimeError (B4) on an invalid component ('q') and on an S that is
  not a multiple of 1/2 (0.3)"* `[PASS]`. Consistent with B4 across both call
  paths; recorded as a positive parity observation.

## Consistency findings

- **C1 — `algo`'s entire public surface violates N1 (Capitalized-verb
  callables).** `Sort`, `Concatenate`, `Vstack`, `Hstack`, `Vsplit`, `Hsplit`
  are all Capitalized, where N1 requires `snake_case` for callable public
  members. Probed: *"algo's entire public surface is Capitalized-verb free
  functions … violating N1 — unlike random/physics/qgates which are already
  snake_case"* `[PASS]`. This is the single largest cleanup in this group: the
  three sibling modules already comply, so `algo` is the outlier.

- **C2 — `algo.Sort` has no in-place `Sort_` counterpart (N2).** Sorting is
  meaningful in both returns-new and in-place forms, but only the returns-new
  `Sort` is bound. Probed: *"algo.Sort does NOT mutate its input … there is no
  in-place algo.Sort_ (N2: the pure op has no in-place counterpart)"* `[PASS]`.
  Contrast `random`, which does pair the in-place `normal_`/`uniform_` with the
  allocating `normal`/`uniform`.

- **C3 — argument names are inconsistent within and across the modules
  (N4/N5).** Within `algo`, the input tensor is called `Tn` (`Sort`), `T1`/`T2`
  (`Concatenate`), `Tlist` (`Vstack`/`Hstack`), and `Tin` (`Vsplit`/`Hsplit`) —
  four spellings for the same "input tensor(s)" role. Across `physics`, `S` and
  `Comp` are non-`snake_case` (`Comp` should be a lowercase `component`). The
  live signatures are recorded in the Inventory; recommend a single
  vocabulary (`t`/`tensors`, `component`) per N4/N5.

- **C4 — the Pauli-X/Y/Z operators are duplicated across `physics` and
  `qgates` under two names and two container types.** `physics.pauli('x')`
  returns a bare `Tensor` `[[0,1],[1,0]]`; `qgates.pauli_x()` returns a
  `UniTensor` of the same matrix. Probed: *"physics.pauli returns a bare Tensor,
  whereas qgates.pauli_x returns a UniTensor — the same Pauli-X operator is
  offered twice, under two names and two container types across physics and
  qgates …"* `[PASS]`, and the numeric identity *"physics.pauli('x') == 2 *
  physics.spin(0.5,'x') …"* `[PASS]`. Two spellings/containers for one operator
  is an N4-adjacent vocabulary inconsistency; the Recommendation keeps both (they
  serve different clients — a raw matrix vs. a tagged gate) but flags the overlap.

- **C5 — `random`'s defaults are applied inconsistently (N4-adjacent).**
  `uniform_`/`uniform` carry `low=0.0`/`high=1.0`-style bounds while `normal_`/
  `normal` require `mean`/`std` positionally with no default. Probed: *"random.
  uniform_ carries default bounds low=0.0/high=1.0, but random.normal_ has NO
  defaults for mean/std …"* `[PASS]`. Recommend giving `normal`/`normal_` the
  natural `mean=0.0`/`std=1.0` defaults for symmetry with `uniform`.

- **C6 — `qgates` is bound but Doxygen-hidden (`/// @cond`).** The whole
  namespace sits inside a `@cond … @endcond` block in `Physics.hpp`, so it is
  fully callable from Python yet absent from the generated C++ reference — an
  internal documentation/visibility inconsistency (the functions are public de
  facto but private de jure). Recommend removing the `@cond` guard so the
  bound-and-public gate library is documented.

## Recommendation

Every live public member of all four modules appears below, tagged keep / add /
rename / remove. The algo module's six functions are all renamed to lower-case
(N1); the other three modules already follow lower_case_with_underscores and are
retained, with two value/behavior fixes flagged (hadamard normalization P8,
normal defaults C5).

### `algo`

| Member | Verdict | Rationale |
|---|---|---|
| `Sort` | rename | → `sort` (N1/C1). Add an in-place `sort_` counterpart (N2/C2). |
| `Concatenate` | rename | → `concatenate` (N1/C1). Rename args `T1`/`T2` → `t1`/`t2` (N4/C3). |
| `Vstack` | rename | → `vstack` (N1/C1). Rename arg `Tlist` → `tensors` (N4/C3). |
| `Hstack` | rename | → `hstack` (N1/C1). Rename arg `Tlist` → `tensors` (N4/C3). |
| `Vsplit` | rename | → `vsplit` (N1/C1). Rename arg `Tin` → `t` (N4/C3). |
| `Hsplit` | rename | → `hsplit` (N1/C1). Rename arg `Tin` → `t` (N4/C3). |

Cross-cutting for `algo`: bind the currently-unbound `Vsplit_`/`Hsplit_` only
if an out-parameter form is wanted (P1); otherwise drop them from the header —
the returns-new forms suffice.

### `random`

| Member | Verdict | Rationale |
|---|---|---|
| `normal_` | keep | Already `snake_case`; correct in-place-fill / `None`-return semantics (N2, probe). Add `mean=0.0`/`std=1.0` defaults for symmetry with `uniform_` (C5). |
| `uniform_` | keep | Already `snake_case`; in-place fill over `[low,high)`, defaults `low=0.0`/`high=1.0` (probe). |
| `normal` | keep | Allocate-and-return generator; reproducible under a fixed seed (probe). Add `mean=0.0`/`std=1.0` defaults (C5). |
| `uniform` | keep | Allocate-and-return generator; reproducible under a fixed seed (probe). |

Cross-cutting for `random`: expose `random_tensor` (an any-dtype generator that
is declared but unbound, P4) as a new binding; leave the deprecated
`Make_normal`/`Make_uniform` unbound. Consider a module-level `seed`/`set_seed`
for global reproducibility (P5) — today seeding is per-call only.

### `physics`

| Member | Verdict | Rationale |
|---|---|---|
| `spin` | keep | Already `snake_case`; returns the correct `(2S+1)`-dim spin operator (probe). Rename args `S`/`Comp` → `s`/`component` (N5/C3). |
| `pauli` | keep | Already `snake_case`; returns the correct Pauli matrix (probe). Rename arg `Comp` → `component` (N5/C3). Overlaps `qgates.pauli_*` (C4) but keeps its bare-`Tensor` role. |

Cross-cutting for `physics`: expose the `operators` spin-half helpers
(`Sz_shalf`/`Sp_shalf`/`Sn_shalf`, P2) as new bindings if the
symmetric-`UniTensor` forms are wanted from Python.

### `qgates`

| Member | Verdict | Rationale |
|---|---|---|
| `pauli_x` | keep | Correct `UniTensor` `[[0,1],[1,0]]` (probe). Duplicates `physics.pauli('x')` (C4) but keeps its tagged-gate role. |
| `pauli_y` | keep | Correct `[[0,-i],[i,0]]` (probe). |
| `pauli_z` | keep | Correct `[[1,0],[0,-1]]` (probe). |
| `hadamard` | keep | **Value fix (P8)**: the returned matrix is unnormalized `[[1,1],[1,-1]]` and not unitary; multiply by `1/sqrt(2)` so `H @ H^dagger == I`. |
| `phase_shift` | keep | Correct `diag(1, exp(i*theta))` (probe). |
| `swap` | keep | Correct rank-4 SWAP gate (probe). |
| `sqrt_swap` | keep | Correct: squares to `swap` (probe). |
| `toffoli` | keep | Correct rank-6 CCNOT; real-valued `Double` (probe). |
| `cntl_gate_2q` | keep | Correct: promotes a 1-qubit gate to its controlled 2-qubit form (`cntl_gate_2q(pauli_x)` == CNOT, probe). |

Cross-cutting for `qgates`: drop the `/// @cond` guard so the public gate
library is documented (C6).

## Docstrings

Numpy-style docstrings for every member tagged keep or rename above, under its
recommended name (the current live name is shown in backticks so the validator
can match it). Removed members need none (there are none here).

### `Sort` → `sort`

```
Sort a Tensor ascending along its last axis.

Parameters
----------
t : Tensor
    The input tensor (any real dtype).

Returns
-------
Tensor
    A NEW tensor whose last-axis slices are each sorted ascending; the input
    is left unchanged.

Notes
-----
Returns-new (B1) — `sort` does not mutate its argument. The recommended API
adds an in-place `sort_` counterpart (N2/C2). Renamed from `Sort` (N1).
```

### `Concatenate` → `concatenate`

```
Concatenate two 1-D Tensors end to end.

Parameters
----------
t1, t2 : Tensor
    Two 1-D tensors on the same device.

Returns
-------
Tensor
    A NEW 1-D tensor `[t1 ..., t2 ...]`; its dtype is the stronger of the two
    inputs (B3 promotion, e.g. Int64 ++ Double -> Double).

Raises
------
RuntimeError
    If either input is not 1-D (B4).

Notes
-----
Renamed from `Concatenate` (N1); args `T1`/`T2` -> `t1`/`t2` (N4).
```

### `Vstack` → `vstack`

```
Stack a list of 2-D matrices vertically (row-wise).

Parameters
----------
tensors : sequence of Tensor
    Matrices with the same number of columns, on the same device.

Returns
-------
Tensor
    A NEW matrix stacking the inputs top to bottom, e.g. [1x2] + [2x2] -> [3x2].

Notes
-----
Renamed from `Vstack` (N1); arg `Tlist` -> `tensors` (N4). Inverse of `vsplit`.
```

### `Hstack` → `hstack`

```
Stack a list of 2-D matrices horizontally (column-wise).

Parameters
----------
tensors : sequence of Tensor
    Matrices with the same number of rows, on the same device.

Returns
-------
Tensor
    A NEW matrix stacking the inputs left to right, e.g. [2x2] + [2x1] -> [2x3].

Notes
-----
Renamed from `Hstack` (N1); arg `Tlist` -> `tensors` (N4). Inverse of `hsplit`.
```

### `Vsplit` → `vsplit`

```
Split a 2-D Tensor row-wise into a list of matrices.

Parameters
----------
t : Tensor
    The input matrix (2-D).
dims : sequence of int
    Row counts of each output block; must sum to the number of rows of `t`.

Returns
-------
list of Tensor
    The row-blocks, e.g. a 3x3 with dims [1,2] -> shapes [1,3], [2,3].

Notes
-----
Renamed from `Vsplit` (N1); arg `Tin` -> `t` (N4). Inverse of `vstack`. The C++
out-parameter form `Vsplit_` is intentionally not exposed (P1).
```

### `Hsplit` → `hsplit`

```
Split a 2-D Tensor column-wise into a list of matrices.

Parameters
----------
t : Tensor
    The input matrix (2-D).
dims : sequence of int
    Column counts of each output block; must sum to the number of columns of `t`.

Returns
-------
list of Tensor
    The column-blocks, e.g. a 3x3 with dims [2,1] -> shapes [3,2], [3,1].

Notes
-----
Renamed from `Hsplit` (N1); arg `Tin` -> `t` (N4). Inverse of `hstack`. The C++
out-parameter form `Hsplit_` is intentionally not exposed (P1).
```

### `normal_`

```
Fill a Tensor / Storage / UniTensor in place with normal (Gaussian) samples.

Parameters
----------
Tin : Tensor or Storage or UniTensor
    The container to overwrite; must be a real-float or complex dtype.
mean, std : float
    Mean and standard deviation of the distribution. (Recommended: default
    mean=0.0, std=1.0 for symmetry with `uniform_`, C5.)
seed : int, optional
    RNG seed. The default `-1` draws fresh device entropy; any fixed value is
    reproducible.

Returns
-------
None

Notes
-----
In-place (N2, trailing `_`): mutates its argument and returns `None`. A fixed
`seed` reproduces the exact same fill (probe).
```

### `uniform_`

```
Fill a Tensor / Storage / UniTensor in place with uniform samples on [low, high).

Parameters
----------
Tin : Tensor or Storage or UniTensor
    The container to overwrite; must be a real-float or complex dtype.
low, high : float, optional
    Half-open range bounds on cpu (default 0.0 / 1.0). Note: on cuda the range
    is (low, high] (cuRAND).
seed : int, optional
    RNG seed; default `-1` = device entropy, any fixed value is reproducible.

Returns
-------
None

Notes
-----
In-place (N2): mutates its argument and returns `None`. Reproducible under a
fixed `seed` (probe).
```

### `normal`

```
Allocate and return a new Tensor filled with normal (Gaussian) samples.

Parameters
----------
Nelem : int or sequence of int
    Number of elements (int overload) or the shape (list overload).
mean, std : float
    Mean and standard deviation. (Recommended defaults 0.0 / 1.0, C5.)
device : int, optional
    Target device (default `Device.cpu` == -1).
seed : int, optional
    RNG seed; default `-1` = device entropy, any fixed value is reproducible.
dtype : int, optional
    Element dtype (default `Type.Double` == 3).

Returns
-------
Tensor
    A NEW tensor of the requested shape/dtype/device.

Notes
-----
Returns-new (contrast the in-place `normal_`). Reproducible under a fixed
`seed` (probe).
```

### `uniform`

```
Allocate and return a new Tensor filled with uniform samples on [low, high).

Parameters
----------
Nelem : int or sequence of int
    Number of elements (int overload) or the shape (list overload).
low, high : float
    Range bounds ([low, high) on cpu; (low, high] on cuda).
device : int, optional
    Target device (default `Device.cpu` == -1).
seed : int, optional
    RNG seed; default `-1` = device entropy, any fixed value is reproducible.
dtype : int, optional
    Element dtype (default `Type.Double` == 3).

Returns
-------
Tensor
    A NEW tensor of the requested shape/dtype/device.

Notes
-----
Returns-new (contrast the in-place `uniform_`). Reproducible under a fixed
`seed` (probe).
```

### `spin`

```
Build the spin-S representation matrix for a given component.

Parameters
----------
s : float
    The spin quantum number; must be a positive multiple of 1/2.
component : str
    'x', 'y', or 'z'.
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
Tensor
    A complex `(2S+1) x (2S+1)` matrix, e.g. `spin(0.5,'z')` = diag(1/2, -1/2),
    `spin(1.0,'z')` = diag(1, 0, -1).

Raises
------
RuntimeError
    If `component` is not 'x'/'y'/'z', or `s` is not a multiple of 1/2 (B4).

Notes
-----
Args renamed `S`/`Comp` -> `s`/`component` (N5). Equals `pauli(component)/2`
for S=1/2 (probe).
```

### `pauli`

```
Build a 2x2 Pauli matrix as a bare Tensor.

Parameters
----------
component : str
    'x', 'y', or 'z'.
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
Tensor
    `pauli('x')`=[[0,1],[1,0]], `pauli('y')`=[[0,-i],[i,0]],
    `pauli('z')`=[[1,0],[0,-1]] (complex).

Raises
------
RuntimeError
    If `component` is not 'x'/'y'/'z' (B4).

Notes
-----
Arg renamed `Comp` -> `component` (N5). Same operator as `qgates.pauli_x/y/z`
but returned as a bare `Tensor` rather than a `UniTensor` (C4).
```

### `pauli_x`, `pauli_y`, `pauli_z`

```
The single-qubit Pauli gates X, Y, Z as 2x2 UniTensors.

Parameters
----------
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
UniTensor
    `pauli_x` = [[0,1],[1,0]], `pauli_y` = [[0,-i],[i,0]],
    `pauli_z` = [[1,0],[0,-1]] (complex), each verified unitary (probe).

Notes
-----
Same operators as `physics.pauli` but tagged as gate `UniTensor`s (C4).
```

### `hadamard`

```
The single-qubit Hadamard gate.

Parameters
----------
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
UniTensor
    The recommended (fixed) gate is (1/sqrt(2)) * [[1,1],[1,-1]], which is
    unitary (`H @ H^dagger == I`).

Notes
-----
IMPORTANT (Parity finding P8): the CURRENT wheel returns the UNNORMALIZED
matrix [[1,1],[1,-1]] (missing the 1/sqrt(2) factor), so today `hadamard()`
is NOT unitary — `H @ H^dagger == 2*I`. The recommended API multiplies by
1/sqrt(2) to fix this (verified by probe against both the current and the
correct value).
```

### `phase_shift`

```
The single-qubit phase-shift gate.

Parameters
----------
phase : float
    The phase angle theta (radians).
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
UniTensor
    diag(1, exp(i*theta)); e.g. `phase_shift(pi/2)` = diag(1, i) (probe).
```

### `swap`

```
The two-qubit SWAP gate.

Parameters
----------
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
UniTensor
    A rank-4 gate (shape [2,2,2,2]); its 4x4 form swaps |01> and |10> (probe).
```

### `sqrt_swap`

```
The two-qubit square-root-of-SWAP gate.

Parameters
----------
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
UniTensor
    A rank-4 gate that squares to `swap` (its 4x4 form has the (1+i)/2,
    (1-i)/2 block; `sqrt_swap()^2 == swap()`, probe).
```

### `toffoli`

```
The three-qubit Toffoli (CCNOT) gate.

Parameters
----------
device : int, optional
    Target device (default `Device.cpu` == -1).

Returns
-------
UniTensor
    A rank-6 gate (shape [2,2,2,2,2,2]); its 8x8 form flips the last qubit only
    when the first two are set. Real-valued (`Type.Double`), unlike the complex
    single-qubit gates (probe).
```

### `cntl_gate_2q`

```
Promote a single-qubit gate to its controlled two-qubit form.

Parameters
----------
gate_1q : UniTensor
    A 2x2 single-qubit gate (e.g. `pauli_x`).

Returns
-------
UniTensor
    A rank-4 controlled gate: identity on the control=0 block, `gate_1q` on the
    control=1 block. `cntl_gate_2q(pauli_x)` == CNOT (probe).
```

## Change table

Clean-slate migration map: `current (C++ / Python) → recommended`. `algo`'s six
functions are renamed to `snake_case`; the other three modules keep their names
but carry the value/behavior fixes noted below.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `algo.Sort` | `algo.sort` (+ add `sort_`) | N1 (C1) + N2 (C2) |
| `algo.Concatenate` | `algo.concatenate` | N1 (C1) |
| `algo.Vstack` | `algo.vstack` | N1 (C1) |
| `algo.Hstack` | `algo.hstack` | N1 (C1) |
| `algo.Vsplit` | `algo.vsplit` | N1 (C1) |
| `algo.Hsplit` | `algo.hsplit` | N1 (C1) |
| `algo` arg `Tn`/`T1`/`T2`/`Tlist`/`Tin` | `t`/`t1`/`t2`/`tensors`/`t` | N4 (C3) |
| `random.normal`/`normal_` (no mean/std default) | add `mean=0.0`/`std=1.0` defaults | N4 (C5) |
| `random_tensor` *(unbound)* | bind as `random.random_tensor` | capability gap (P4) |
| `physics` args `S`/`Comp` | `s`/`component` | N5 (C3) |
| `physics.operators` Sz/Sp/Sn *(unbound)* | bind if UniTensor spin ops wanted | header-vs-wheel gap (P2) |
| `qgates.hadamard` value `[[1,1],[1,-1]]` | `(1/sqrt(2))*[[1,1],[1,-1]]` | correctness/unitary (P8) |
| `qgates` `/// @cond` guard | remove (document the module) | visibility (C6) |
| `algo.Vsplit_`/`Hsplit_` *(unbound)* | drop from header (returns-new suffices) | dead surface (P1) |
| `random.Make_normal`/`Make_uniform` *(unbound, deprecated)* | leave unbound / delete | deprecated (P4) |

Every other public member — `random`'s four generators, `physics.spin`/`pauli`,
and all nine `qgates` — keeps its current name; the only value/behavior change
outside the `algo` renames is the `hadamard` normalization fix (P8) and the
recommended `normal`/`normal_` defaults (C5).
