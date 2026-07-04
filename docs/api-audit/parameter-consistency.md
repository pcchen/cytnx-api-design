# Parameter naming, ordering & default consistency — cross-class N4 sweep

The per-class audits apply convention **N4** (argument order identical across
C++/Python for the same member; semantically equivalent parameters share the
same name and position across sibling methods) *opportunistically* — each unit
flags the divergences it happens to notice. This document is the **systematic**
pass N4 deserves: every public callable's live parameter signature is extracted
across all units at once, then analysed as one corpus so cross-*class*
divergences (which no single per-class doc can see) surface.

Unlike the per-class Parity findings, the claims here are **static-signature**
facts, not runtime behavior — N4 is a signature convention (methodology §4.1,
"signature comparison is static"), so there is no behavioral probe. Instead the
evidence is a reproducible extractor whose output anyone can regenerate:

```bash
source tools/env.sh
$PY tools/param_inventory.py > /tmp/params.json   # 770 signature records, 14 units
```

`tools/param_inventory.py` parses the pybind11 signature line(s) from every
public callable's docstring in the installed `cytnx==1.1.0` wheel — i.e. exactly
what a Python caller sees — recording each parameter's name, position, type, and
default, and expanding overloaded members into one record per overload. All
counts and signatures quoted below are copied verbatim from its output; the
readability-trimmed signatures collapse pybind's `typing.SupportsInt |
typing.SupportsIndex` noise to `int` and `collections.abc.Sequence[...]` to
`List[...]` but change nothing else.

**Scope:** the 14 units with callable members (the enums `Type`/`SymType`/
`bondType`/`fermionParity` contribute no parametered callables and are omitted;
`Device` contributes only `getname`). 770 signature records total, of which 39
are *opaque* — pybind reduced them to `*args, **kwargs` and hides the real
parameters (`reshape`, most `set_elem`/`get_elem` overload dispatchers), so they
cannot be audited for parameter identity at all and are excluded from the
name/position tallies.

---

## PC1 (headline) — 18 clean public methods erase their parameter names, so they cannot be called by keyword

pybind11 emits `arg0`, `arg1`, … whenever a binding omits `py::arg("name")`.
A parameter named `argN` **cannot be passed as a keyword argument** from Python
(`bond.set_type(bond_type=...)` raises `TypeError`) — the caller is forced to
pass positionally and has no self-documenting call site. This is a direct N4
violation (the Python-visible name is *absent*, not merely divergent from C++).

Across the corpus, **192 parameters are erased** — but the overwhelming majority
(Tensor 127, UniTensor 54) are on the `c`-prefixed leaked internals
(`c__iadd__`, `cPow_`, `c__ipow__`, …) and the `*_elem` overload families that
the per-class docs already recommend **removing** (see `Tensor.md` P3,
`UniTensor.md` P1/C1). Filtering those out leaves **18 erasures on
genuinely-public, keep/rename methods** — these are the ones that matter,
because a user is expected to call them:

| Unit | Method (verbatim live signature) |
|---|---|
| `Tensor` | `Pow(self, arg0: float) -> Tensor` |
| `Tensor` | `Trace(self, arg0: int, arg1: int) -> Tensor` |
| `Tensor` | `same_data(self, arg0: Tensor) -> bool` |
| `Storage` | `resize(self, arg0: int) -> None` |
| `Scalar` | `astype(self, arg0: int) -> Scalar` |
| `Scalar` | `maxval(arg0) -> Scalar` |
| `Scalar` | `minval(arg0) -> Scalar` |
| `UniTensor` | `Pow(self, arg0: float) -> UniTensor` |
| `UniTensor` | `elem_exists(self, arg0) -> bool` |
| `UniTensor` | `same_data(self, arg0) -> bool` |
| `UniTensor` | `to_(self, arg0: int) -> UniTensor` |
| `Bond` | `group_duplicates_(self, arg0) -> ...` |
| `Bond` | `retype(self, arg0: bondType) -> Bond` |
| `Bond` | `set_type(self, arg0: bondType) -> Bond` |
| `Symmetry` | `Zn(arg0: int) -> Symmetry` |
| `LinOp` | `set_device(self, arg0) -> None` |
| `LinOp` | `set_dtype(self, arg0) -> None` |
| `Device` | `getname(arg0) -> str` |

Note the self-inconsistency: `Scalar.astype(arg0)` erases its dtype name while
the identically-purposed `Tensor.astype`/`Storage.astype` name theirs; and
`UniTensor.to_(arg0: int)` erases the device name that its own sibling
`UniTensor.to(device=...)` exposes. **Recommendation:** add `py::arg("...")` to
all 18 so every public parameter is keyword-callable, using the canonical names
from PC3/PC4 (`Bond.set_type(bond_type=...)`, `Symmetry.zn(n=...)`,
`Scalar.astype(dtype=...)`, `Tensor.trace(axis_a=..., axis_b=...)`,
`UniTensor.to_(device=...)`, `LinOp.set_device(device=...)`, etc.).

## PC2 — the `device` parameter sits at 7 different positions; `dtype` at 5

Every constructor/generator/conversion function carries a trailing
config block, but its *absolute position* drifts because the leading required
arguments differ per function:

```
device at pos 0:  Tensor.to_ , Storage.to_
device at pos 1:  Storage.from_pylist , physics.pauli , qgates.phase_shift
device at pos 2:  Storage.Init , Tensor.Init , physics.spin
device at pos 3:  UniTensor.zeros , UniTensor.ones , random.normal , random.uniform
device at pos 4:  UniTensor.Init , UniTensor.eye , UniTensor.identity
device at pos 5:  UniTensor.arange
device at pos 6:  UniTensor.linspace , UniTensor.normal , UniTensor.uniform
```

Concretely (verbatim):

```
Storage.Init(self, size: int, dtype: int = 3, device: int = -1, init_zero: bool = True)
UniTensor.zeros(Nelem: int, labels: List[str] = [], dtype: int = 3, device: int = -1, name: str = '')
UniTensor.arange(start, end, step = 1.0, labels: List[str] = [], dtype: int = 3, device: int = -1, name: str = '')
UniTensor.linspace(start, end, Nelem: int, endpoint: bool = True, labels: List[str] = [], dtype: int = 3, device: int = -1, name: str = '')
```

**Mitigating nuance (stated honestly):** where these parameters keep their real
names (`dtype`/`device`) and are keyword-callable, the absolute-position drift
is largely harmless in Python — callers pass `zeros(shape, dtype=..., device=...)`
by keyword and never count positions. The **relative** order is in fact already
consistent (`dtype` always precedes `device` in every occurrence). So PC2 is a
lower-severity finding *on its own* — but it turns into a real trap when combined
with PC1 (an erased trailing arg must be passed positionally, forcing the caller
to also positionally supply every arg before it) or PC6 (single-letter names).
**Recommendation:** keep the invariant "the trailing config block is always, in
this order, `..., labels, dtype, device, name`" and make it a documented rule so
future additions don't break it; the drift in *absolute* index is acceptable
only because these are always keyword-passed — which PC1 must be fixed to
guarantee.

## PC3 — the "tensor you operate on" first argument has 10 different names

For free functions, the first positional argument is the input tensor in almost
every case, yet it is spelled 10 ways (position-0 name → count, across
`linalg`/`algo`/`physics`/`qgates`/`random`):

| Name | Count | Example functions |
|---|---|---|
| `Tin` | 123 | `linalg.Conj`, `linalg.Det`, `algo.Vsplit`, `algo.Hsplit` |
| `T1` | 12 | `linalg.Dot`, `linalg.Kron`, `linalg.Matmul`, `algo.Concatenate` |
| `Tn` | 11 | `linalg.Abs`, `linalg.Max`, `linalg.Hosvd`, `algo.Sort` |
| `Tio` | 9 | `linalg.Exp_`, `linalg.InvM_`, `linalg.Qdr` (the in-place/"in-out" forms) |
| `Hop` | 7 | `linalg.Lanczos`, `linalg.Arnoldi`, `linalg.Lanczos_Exp` |
| `a` | 5 | `linalg.Axpy`, `linalg.Gemm` (BLAS-style) |
| `Tlist` | 2 | `algo.Hstack`, `algo.Vstack` |
| `A` | 2 | `linalg.Lstsq`, `linalg.Tridiag` |
| `Sin` | 2 | `random.normal_`, `random.uniform_` |
| `x` | 1 | `linalg.Ger` |

`Tin` is the de-facto standard (123/173 free-function first args). The variants
carry weak semantic hints (`Tio` = mutated in place, `Hop` = a linear operator,
`Tlist` = a list, BLAS `a`/`x` = matrix/vector) but the hints are inconsistent
and undocumented. This compounds the per-class `operations.md` C3/N4 finding
(`algo` alone uses `Tn`/`T1`/`T2`/`Tlist`/`Tin`) into a namespace-wide pattern.
**Recommendation:** one vocabulary — `t` for a single tensor input, `t1`/`t2`
for a binary operation, `tensors` for a list, `a_op`/`linop` for a linear
operator — applied uniformly, with `Tin` retired.

## PC4 — the two-axis "trace" concept: 3 names AND a default divergence, for one operation

The single operation "contract two axes of a tensor" is bound three ways, and
the three disagree on *both* parameter names *and* whether the axes default:

```
Tensor.Trace(self, arg0: int, arg1: int)                       -> Tensor      # erased names, NO defaults
UniTensor.Trace(self, a: int = 0, b: int = 1)                  -> UniTensor    # a/b, defaults 0/1
linalg.Trace(Tn, axisA: int = 0, axisB: int = 1)               -> ...          # axisA/axisB, defaults 0/1
```

So `Tensor.Trace()` (no args) raises `TypeError` while `UniTensor.Trace()` and
`linalg.Trace()` succeed on the default `(0, 1)` axes — the *same* operation
behaves differently by container. And the axis parameter is called `arg0`/`a`/
`axisA` across the three. This is the cleanest single illustration of why the
sweep is needed: the divergence spans three units, so no per-class doc catches
it whole (`Tensor.md` P2 notes the lost defaults locally; the naming split is
only visible here). **Recommendation:** one signature everywhere —
`trace(axis_a: int = 0, axis_b: int = 1)` (snake_case per N1, defaults restored,
names unified, keyword-callable).

## PC5 — the SVD family splits on flag granularity

Cross-referenced from `linalg.md` C2, confirmed by the extractor:

```
linalg.Svd(Tin, is_UvT: bool = True)                    # one combined flag
linalg.Gesvd(Tin, is_U: bool = True, is_vT: bool = True) # two separate flags
```

Two sibling full-SVD entry points take semantically-equivalent "which factors do
you want" flags at different granularities; `Svd` rejects `is_U` (probe-verified
in `linalg.py`). **Recommendation:** the finer `is_u`/`is_vt` pair everywhere,
so the decomposition family shares one flag vocabulary.

## PC6 — generic single-letter names are reused for unrelated concepts at different positions

`a` appears at positions 0, 1, and 2 across the corpus; `b` at 1, 2, and 3. It
means "the input matrix" in `linalg.Gemm`/`Axpy` (pos 0), "the first axis" in
`UniTensor.Trace` (pos 0), and "the second operand" elsewhere. A one-letter name
carries no concept and, because it moves position, cannot even be relied on
positionally. **Recommendation:** replace all single-letter public parameters
with concept names (`axis_a`, `t1`/`t2`, `alpha`/`x`/`y` only where the BLAS
convention is genuinely the clearest, and then documented as such).

---

## Recommended canonical parameter conventions (N4 refinement)

Consolidated from PC1–PC6, these tighten methodology N4 into concrete rules the
recommended API should obey:

- **R1 — every public parameter has an explicit `py::arg("name")`.** No `argN`
  erasure; every parameter is keyword-callable (fixes PC1's 18 methods).
- **R2 — one name per concept, namespace-wide:** input tensor `t` (`t1`/`t2`
  for binary, `tensors` for a list); dtype `dtype`; device `device`; a quantum
  number `qnum` (`qnum_l`/`qnum_r`); an axis `axis` (`axis_a`/`axis_b` for a
  pair); a bond direction `bond_type`; a modulus `n`. Retire `Tin`/`Tn`/`Tio`/
  `Hop`/`a`/`A`/`x`/`Tlist`/`Sin`/`axisA`/`arg0`.
- **R3 — the trailing config block is fixed and ordered:** `..., labels, dtype,
  device, name`, always in that order, always keyword-defaulted (stabilises PC2).
- **R4 — an operation exposed on more than one type (`Trace` on `Tensor`/
  `UniTensor`/`linalg`) shares one signature** — same parameter names, same
  defaults, same order (fixes PC4).
- **R5 — a family of related functions shares one flag vocabulary and
  granularity** (`is_u`/`is_vt` across the whole SVD family, fixes PC5).

## Cross-references

This sweep aggregates and extends the per-class N4 findings, which remain the
authority for each unit's own recommendation table:

- `per-class/Symmetry.md` C3 — `qnL`/`qnR`/`qin` → `qnum`/`qnum_l`/`qnum_r`.
- `per-class/operations.md` C3/C5 — `algo`'s `Tn`/`T1`/`T2`/`Tlist`/`Tin`
  vocabulary and `random`'s inconsistent defaults.
- `per-class/linalg.md` C2 — the SVD flag-granularity split (PC5).
- `per-class/Tensor.md` P2 — `Trace`'s dropped C++ defaults (part of PC4).
- `per-class/network.md` — `PutUniTensor`/`PutUniTensors` and `outrk`
  parameter smells.
- `summary.md` X1 — the N1 casing pattern these renames ride alongside.

## Reproduce

```bash
source tools/env.sh
$PY tools/param_inventory.py > /tmp/params.json
# then aggregate: erased names, name→position drift, first-arg vocabulary, etc.
```

`tools/param_inventory.py` is deterministic against a fixed wheel, so this
document's every count is regenerable and auditable.
