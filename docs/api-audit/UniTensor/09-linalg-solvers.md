# UniTensor — 09. Linalg solvers (Krylov free functions)

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the closest sibling
> [`08-linalg-operations.md`](08-linalg-operations.md) — also `cytnx.linalg` free
> functions). Split into **Analysis** and a self-contained **Recommendation** that
> is the *normative spec for the next version of Cytnx* — implement the next major
> version's `cytnx.linalg` Krylov-solver surface to match §R exactly. All runtime
> claims verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_09_solvers.py` (all `[PASS]`, exit 0).
> **No C++ probe accompanies this category:** the reachable solvers (`Lanczos`,
> `Arnoldi`, `Lanczos_Exp`) are bound as **direct pass-through pybind lambdas** to
> the C++ `cytnx::linalg::` overloads (`pybind/linalg_py.cpp:918-990`), with **no**
> `conti.py` wrapper and no signature or behavior change — so there is **no
> binding-fidelity finding** to verify on a separate raw-C++ side. (Gate 4
> recorded as *"no binding-fidelity finding — identical C++/Python free-function
> bindings; the absent solvers are commented out on BOTH sides."*)

**Category scope — same shape as cat 08.** These are the **`cytnx.linalg` FREE
functions** that solve an eigen/exponential problem for a **linear operator**
(`LinOp`) via Krylov iteration — *not* `UniTensor` members. Every solver takes a
`LinOp Hop` as its first operand and a `Tensor`/`UniTensor` **start vector**; the
result mirrors the start-vector type. This category audits the six brief members
`Lanczos`, `Lanczos_Gnd_Ut`, `Lanczos_Exp`, `Arnoldi`, `Lanczos_ER`,
`Lanczos_Gnd` — of which **only three are reachable in 1.1.0** (`Lanczos`,
`Arnoldi`, `Lanczos_Exp`); the other three (`Lanczos_ER`, `Lanczos_Gnd`,
`Lanczos_Gnd_Ut`) are **commented out** (finding UT-K3). Python bindings:
`cytnx_src/pybind/linalg_py.cpp:918-990` (reachable) and `:999-1019` (the
commented-out `c_*` block); C++ header:
`cytnx_src/include/linalg.hpp:2620,2681,2720,2765,2817,2859,2889,2915,2940,2977`.

> **Coverage note (validator).** `validate_doc.py UniTensor <dir>` checks
> `dir(cytnx.UniTensor)`; the linalg Krylov free functions are **not** UniTensor
> members, so they are neither required nor counted by the UniTensor validator —
> this file adds **no** UniTensor-member coverage and no regression. Machine
> coverage of the `cytnx.linalg` surface itself is a separate future `linalg`
> plan (out of scope here). Every in-scope function still appears in the R.1
> verdict table below.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per in-scope solver. `Hop` is always a `LinOp`; the start vector is
`Tensor` **or** `UniTensor` where a dual overload exists. The probe assertion
backing each runtime claim is quoted. All names are **Capitalized** — the correct
spelling for free functions acting on objects (finding UT-K1).

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `Lanczos` | 4 overloads: `Lanczos(Hop, Tin, method, CvgCrit=1e-14, Maxiter=10000, k=1, is_V=True, is_row=False, max_krydim=0, verbose=False)` **and** `Lanczos(Hop, Tin, which='SA', Maxiter=10000, CvgCrit=0, k=1, is_V=True, ncv=0, verbose=False)`, each in a **Tensor** and a **UniTensor** operand variant | `list[Tensor]` / `list[UniTensor]` `[eigval, (eigvec)]` | Hermitian/symmetric Krylov eigensolver. **One name, two algorithm families:** the `method`-string form (`'Gnd'` naive / `'ER'` explicitly-restarted) and the ARPACK `which`-string form (`'SA'`/`'LA'`/`'LM'`). Probe: *"Lanczos carries FOUR overloads (Tensor+method, UniTensor+method, Tensor+which, UniTensor+which)"* + *"Lanczos(op, v, method='Gnd') dispatches the naive-Lanczos ground-state path …"* + *"Lanczos(op, v, which='SA') dispatches the ARPACK path …"*. |
| `Lanczos_Gnd_Ut` | *(declared `Lanczos_Gnd_Ut(Hop, Tin, CvgCrit=1e-14, is_V=True, verbose=False, Maxiter=100000)` — `hpp:2940`)* | *(would be `list[UniTensor]`)* | **ABSENT at runtime.** The UniTensor ground-state path. Its pybind `c_Lanczos_Gnd_Ut` registration is inside a `/* */` block (`linalg_py.cpp:1014-1018`) and its conti wrapper inside a `"""..."""` string (`cytnx/linalg_conti.py:13-15,22`). Probe: *"linalg.Lanczos_Gnd_Ut is ABSENT at runtime …"* + *"the raw c_Lanczos_Gnd_Ut binding is not leaked either …"*. See UT-K3. |
| `Lanczos_Exp` | `Lanczos_Exp(Hop, v, tau, CvgCrit=1e-14, Maxiter=10000, verbose=False)` — **UniTensor `v` only** | `UniTensor` | Action of the matrix exponential `exp(tau·H)·v` for a Hermitian `Hop` (Krylov / time-evolution primitive). **No Tensor overload.** Probe: *"Lanczos_Exp returns a single UniTensor (the exp(tau*H)@v action)"* + *"Lanczos_Exp(op, v, tau) == exp(tau*H) @ v (matches dense expm)"* + *"Lanczos_Exp is UniTensor-ONLY …"*. |
| `Arnoldi` | 2 overloads: `Arnoldi(Hop, Tin, which='LM', Maxiter=10000, CvgCrit=0, k=1, is_V=True, ncv=0, verbose=False)` in a **Tensor** and a **UniTensor** operand variant | `list[Tensor]` / `list[UniTensor]` `[eigvals, (eigvecs)]` | General (non-Hermitian) Krylov eigensolver (ARPACK). `which` selects the target region (`'LM'`/`'SR'`/`'LR'`/`'LA'`; `'SM'` unsupported → `RuntimeError`). Probe: *"Arnoldi carries TWO overloads (Tensor and UniTensor)"* + *"Arnoldi(op, v, which='SR') runs and returns eigen-results"*. |
| `Lanczos_ER` | *(declared `Lanczos_ER(Hop, k=1, is_V=True, maxiter=10000, CvgCrit=1e-14, is_row=False, Tin=Tensor(), max_krydim=4, verbose=False)` — `hpp:2889`)* | *(would be `list[Tensor]`)* | **ABSENT at runtime.** The Tensor-oriented explicitly-restarted Lanczos. Same commented-out state as `Lanczos_Gnd_Ut`. Probe: *"linalg.Lanczos_ER is ABSENT at runtime …"*. See UT-K3. *(The `'ER'` algorithm is still reachable via `Lanczos(..., method='ER')`.)* |
| `Lanczos_Gnd` | *(declared `Lanczos_Gnd(Hop, CvgCrit=1e-14, is_V=True, Tin=Tensor(), verbose=False, Maxiter=100000)` — `hpp:2915`)* | *(would be `list[Tensor]`)* | **ABSENT at runtime.** The Tensor-oriented naive ground-state Lanczos. Same commented-out state. Probe: *"linalg.Lanczos_Gnd is ABSENT at runtime …"*. See UT-K3. *(The `'Gnd'` algorithm is still reachable via `Lanczos(..., method='Gnd')`.)* |

**Reachable at runtime:** `Lanczos`, `Arnoldi`, `Lanczos_Exp`. **Absent (commented
out in 1.1.0):** `Lanczos_ER`, `Lanczos_Gnd`, `Lanczos_Gnd_Ut` — and their raw
`c_*` bindings are not leaked either. Probe: *"the raw c_Lanczos_ER / c_Lanczos_Gnd
/ c_Lanczos_Gnd_Ut binding is not leaked either (commented out in pybind)"*.

**Operand-type map (Tensor vs UniTensor vs LinOp).** Every solver's *operator* is a
`LinOp Hop`; the split below is over the **start-vector / return** type:

| Solver | `Hop` | start vector `Tin`/`v` | returns | reachable |
|---|---|---|---|---|
| `Lanczos` | `LinOp` | **Tensor or UniTensor** | matching `list[…]` | ✅ |
| `Arnoldi` | `LinOp` | **Tensor or UniTensor** | matching `list[…]` | ✅ |
| `Lanczos_Exp` | `LinOp` | **UniTensor only** | `UniTensor` | ✅ |
| `Lanczos_Gnd` | `LinOp` | Tensor | `list[Tensor]` | ❌ commented out |
| `Lanczos_ER` | `LinOp` | Tensor | `list[Tensor]` | ❌ commented out |
| `Lanczos_Gnd_Ut` | `LinOp` | UniTensor | `list[UniTensor]` | ❌ commented out |

Probe: *"every solver takes a LinOp Hop as its first operand"* + *"Lanczos/Arnoldi
has BOTH a Tensor and a UniTensor operand overload"* + *"Lanczos_Exp is
UniTensor-ONLY …"*.

## A2. C++ ↔ Python mapping

Every **reachable** row is a **direct pass-through**: the pybind lambda forwards
straight to the same-named `cytnx::linalg::` overload with an identical argument
list — no `conti.py` layer, no behavior change (hence **no** binding-fidelity
finding). The three **absent** rows are commented out on *both* sides — the
pybind `c_*` registrations inside a `/* */` block and the conti wrappers inside a
`"""..."""` docstring.

| C++ (`linalg.hpp`) | Python (`linalg_py.cpp`) | Status | Note |
|---|---|---|---|
| `std::vector<Tensor> Lanczos(LinOp*, const Tensor&, method="Gnd", …)` (`:2720`) | `Lanczos(Hop, Tin, method, …)` (`:939`) | identical | method-string family, Tensor operand (UT-K2) |
| `std::vector<UniTensor> Lanczos(LinOp*, const UniTensor&, method="Gnd", …)` (`:2859`) | `Lanczos(Hop, Tin, method, …)` (`:950`) | identical | method-string family, UniTensor operand |
| `std::vector<Tensor> Lanczos(LinOp*, const Tensor&, which="SA", …)` (`:2765`) | `Lanczos(Hop, Tin, which, …)` (`:962`) | identical | ARPACK which-string family, Tensor operand (UT-K2) |
| `std::vector<UniTensor> Lanczos(LinOp*, const UniTensor&, which="SA", …)` (`:2817`) | `Lanczos(Hop, Tin, which, …)` (`:972`) | identical | ARPACK which-string family, UniTensor operand |
| `std::vector<Tensor> Arnoldi(LinOp*, const Tensor&, which="LM", …)` (`:2620`) | `Arnoldi(Hop, Tin, which, …)` (`:918`) | identical | general eigensolver, Tensor operand |
| `std::vector<UniTensor> Arnoldi(LinOp*, const UniTensor&, which="LM", …)` (`:2681`) | `Arnoldi(Hop, Tin, which, …)` (`:928`) | identical | general eigensolver, UniTensor operand |
| `UniTensor Lanczos_Exp(LinOp*, const UniTensor& v, const Scalar& tau, …)` (`:2977`) | `Lanczos_Exp(Hop, v, tau, …)` (`:983`) | identical | UniTensor-only; `exp(tau·H)·v` (UT-K5) |
| `std::vector<Tensor> Lanczos_ER(LinOp*, k, is_V, …, const Tensor& Tin, …)` (`:2889`) | *(commented out — `c_Lanczos_ER`, `:1000-1006`)* | **absent** | Tensor-oriented ER Lanczos; unreachable (UT-K3) |
| `std::vector<Tensor> Lanczos_Gnd(LinOp*, CvgCrit, is_V, const Tensor& Tin, …)` (`:2915`) | *(commented out — `c_Lanczos_Gnd`, `:1008-1012`)* | **absent** | Tensor-oriented naive-Gnd Lanczos; unreachable (UT-K3) |
| `std::vector<UniTensor> Lanczos_Gnd_Ut(LinOp*, const UniTensor& Tin, …)` (`:2940`) | *(commented out — `c_Lanczos_Gnd_Ut`, `:1014-1018`)* | **absent** | UniTensor ground-state path; unreachable (UT-K3) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/UniTensor_09_solvers.py` (on the 1.1.0 wheel). **There is no
binding-fidelity finding in this category:** the reachable solvers are direct
pass-through pybind lambdas over the C++ `cytnx::linalg::` overloads (no
`conti.py` wrapper), so no raw-C++ probe is needed (gate 4 recorded as *"no
binding-fidelity finding — identical C++/Python free-function bindings"*). The
`file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-K1** | the Krylov solvers `Lanczos`/`Arnoldi`/`Lanczos_Exp` are **Capitalized** — and this is **correct** under N-casing | naming (N-casing — **positive demonstration**) | **free functions acting on objects stay Capitalized** (SciPostPhysCodeb.53), exactly as the cat-08 decompositions. They take a `LinOp` + start vector and return a result. Py probe *"the Krylov solvers are Capitalized at runtime …"* + *"their lowercase spellings are NOT bound …"* | **KEEP Capitalized** — `Lanczos`, `Arnoldi`, `Lanczos_Exp` (and, if re-enabled, `Lanczos_ER`/`Lanczos_Gnd`/`Lanczos_Gnd_Ut`). Do **not** snake_case. Cross-ref cat 08 UT-X1. |
| **UT-K2** | **`Lanczos` is one name for two different algorithm families** dispatched by a *string* argument — the `method` string (`'Gnd'` naive / `'ER'` restarted) vs the ARPACK `which` string (`'SA'`/`'LA'`/`'LM'`) — with **incompatible** parameter sets (`is_row`/`max_krydim` vs `ncv`) | naming / ordering (N4) | **the dispatch is invisible in the name and the two families reject each other's kwargs.** Overloads 1–2 take `method` + `is_row`/`max_krydim` (`hpp:2720,2859`); overloads 3–4 take `which` + `ncv` (`hpp:2765,2817`). A user cannot tell from `Lanczos(...)` which algorithm runs, nor which kwargs are legal, without reading all four overloads. Py probe *"Lanczos carries FOUR overloads …"* + *"Lanczos(op, v, method='Gnd') dispatches the naive-Lanczos ground-state path …"* + *"Lanczos(op, v, which='SA') dispatches the ARPACK path …"* + *"the method-string overload accepts is_row/max_krydim kwargs …"* + *"the which-string overload accepts an ncv kwarg …"* | **make the split explicit.** Prefer distinct, self-describing names — `Lanczos` (ARPACK, `which`) vs a separate `LanczosGnd`/`LanczosER` (or a single `Lanczos(..., algorithm=…)` **enum**, not a bare string). At minimum **document the two families, their disjoint kwargs, and the valid `method`/`which` codes exhaustively** in one place. Keep Capitalized (UT-K1). *Migration:* if renamed, keep `Lanczos(method=…)` as a `DeprecationWarning`-shimmed alias for one release. |
| **UT-K3** | `Lanczos_ER`, `Lanczos_Gnd`, `Lanczos_Gnd_Ut` are **absent at runtime** — commented out on **both** the pybind side (`/* */`) and the conti side (`"""..."""`) in 1.1.0 | parity / capability (binding gap) | **three brief members do not exist as public names**, confirming the earlier audit note. `hasattr(linalg, "Lanczos_Gnd_Ut")` … are all False; the raw `c_Lanczos_*` bindings are not leaked either. The C++ functions **do** exist (`hpp:2889,2915,2940`) but their pybind registrations sit inside `linalg_py.cpp:999-1019` (a `/* */` block) and the conti wrappers inside `cytnx/linalg_conti.py:4-23` (a `"""..."""` docstring). The `'ER'`/`'Gnd'` algorithms remain reachable *only* via `Lanczos(..., method=…)`, and the **UniTensor ground-state path has no dedicated entry point** at all. Py probe *"linalg.Lanczos_ER / Lanczos_Gnd / Lanczos_Gnd_Ut is ABSENT at runtime …"* + *"the raw c_Lanczos_ER / c_Lanczos_Gnd / c_Lanczos_Gnd_Ut binding is not leaked either …"* | **decide and document deliberately, do not leave half-commented.** Either (a) **re-enable** `Lanczos_Gnd_Ut` (the UniTensor ground-state convenience) and drop the redundant `Lanczos_ER`/`Lanczos_Gnd` in favour of the `Lanczos(method=…)` dispatch (UT-K2); or (b) keep them removed and **document that the UniTensor ground state is obtained via `Lanczos(Hop, ut_v0, method='Gnd')`** (probe-verified below). Whichever is chosen, the commented-out code should be deleted, not left dormant. |
| **UT-K4** | the **Tensor-vs-UniTensor split is not encoded in the names** — `Lanczos`/`Arnoldi` silently carry both operand overloads, `Lanczos_Exp` is UniTensor-only, and the removed `Lanczos_Gnd` (Tensor) vs `Lanczos_Gnd_Ut` (UniTensor) used a `_Ut` suffix that no longer maps to any live pair | naming / parity | **inconsistent conventions for the same distinction.** `Lanczos`/`Arnoldi` overload on the operand type (no name change); `Lanczos_Exp` simply omits the Tensor overload; the historical `_Ut` suffix (UniTensor) is now orphaned since its Tensor sibling is also absent. Py probe *"Lanczos has BOTH a Tensor and a UniTensor operand overload"* + *"Lanczos_Exp is UniTensor-ONLY …"* + *(A1 operand-type map)* | **standardise on operand overloading** (one Capitalized name, Tensor **and** UniTensor overloads) rather than `_Ut`-suffixed twins. Drop the `_Ut` suffix convention. Give `Lanczos_Exp` a Tensor overload for parity (UT-K5). Document the operand-type map (A1) per function. |
| **UT-K5** | `Lanczos_Exp` is **UniTensor-only** — no Tensor start-vector overload, unlike `Lanczos`/`Arnoldi` | parity / capability | **reverse parity gap** (mirror of cat 08's Tensor-only functions). `Lanczos_Exp`'s `v` operand accepts only a UniTensor; a Tensor start vector raises `TypeError`. Py probe *"Lanczos_Exp is UniTensor-ONLY (its `v` operand has no Tensor overload) — a parity gap vs Lanczos/Arnoldi"* + *"Lanczos_Exp rejects a Tensor start vector at call time (UniTensor-only)"* | **add a `Tensor` overload to `Lanczos_Exp`** for parity with `Lanczos`/`Arnoldi` (the C++ core already operates on the flattened block). Keep Capitalized. |

## A4. Argument ordering — positional & keyword

Every solver takes the **`LinOp Hop` as the primary positional operand first**,
then the start vector, then operation parameters. There is no keyword-only
metadata block (these are free functions, not constructors).

| API | positional-required (in order) | operation parameters (keyword-capable) |
|---|---|---|
| `Lanczos` (method) | `Hop`, `Tin`, `method` | `CvgCrit=1e-14`, `Maxiter=10000`, `k=1`, `is_V=True`, `is_row=False`, `max_krydim=0`, `verbose=False` |
| `Lanczos` (which) | `Hop`, `Tin` | `which='SA'`, `Maxiter=10000`, `CvgCrit=0`, `k=1`, `is_V=True`, `ncv=0`, `verbose=False` |
| `Arnoldi` | `Hop`, `Tin` | `which='LM'`, `Maxiter=10000`, `CvgCrit=0`, `k=1`, `is_V=True`, `ncv=0`, `verbose=False` |
| `Lanczos_Exp` | `Hop`, `v`, `tau` | `CvgCrit=1e-14`, `Maxiter=10000`, `verbose=False` |

- **Canonical positional rule (§R.0):** `Hop` (the operator), then the start
  vector `Tin`/`v`, then parameters — matches the live order; no reordering needed.
- **Inconsistent parameter order across the two `Lanczos` families (UT-K2):** the
  `method` form is `(…, CvgCrit, Maxiter, …)` while the `which` form is
  `(…, Maxiter, CvgCrit, …)` — the **`CvgCrit`/`Maxiter` pair is swapped** between
  the two families. Unify the ordering as part of the UT-K2 disambiguation.
- **`method`/`which` are *string* dispatch keys** (UT-K2): prefer an `enum`
  (`LanczosMethod.Gnd`/`.ER`, `EigTarget.SA`/`.LM`) over free-form strings — the
  valid codes are otherwise undiscoverable and `'SM'` silently raises `RuntimeError`.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version `cytnx.linalg` Krylov-solver
surface. Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53) — free functions acting on objects stay
  Capitalized.** `Lanczos`, `Arnoldi`, `Lanczos_Exp` (and any re-enabled
  ground-state solver) are **kept Capitalized** (UT-K1), exactly as the cat-08
  decompositions. The v1 *snake_case-all-linalg* recommendation stays **reversed**.
- **N-underscore — none apply.** These solvers are pure (they return
  eigen-results / a new vector); none mutate their operand, so no trailing-`_`
  in-place forms exist. (The in-place mutation is inside the caller-supplied
  `LinOp.matvec`, not the solver.)
- **String dispatch → enum.** The `method` (`'Gnd'`/`'ER'`) and `which`
  (`'SA'`/`'LA'`/`'LM'`/…) selectors become typed enums so the valid values are
  discoverable and mistakes fail at call time, not deep in ARPACK (UT-K2/A4).
- **Operand overloading, not `_Ut` suffixes.** One Capitalized name carries both a
  `Tensor` and a `UniTensor` start-vector overload; the `_Ut` naming convention is
  dropped (UT-K4).
- **Binding fidelity: none.** The reachable solvers are direct pass-through pybind
  lambdas to `cytnx::linalg::` — no `conti.py` wrapper, no leaked `c*` binding, no
  signature change. The three absent solvers are commented out on both sides
  (UT-K3); re-enabling or deleting them is a deliberate decision, not a binding
  fix.

## R.1 Recommended API (exact signatures + behavior contract)

```python
# cytnx.linalg — Krylov solvers (Capitalized; act on a LinOp + start vector)
from enum import Enum
from typing import NamedTuple

class LanczosMethod(Enum):        # replaces the 'Gnd'/'ER' method string (UT-K2)
    Gnd = "Gnd"                    # naive ground-state Lanczos
    ER = "ER"                      # explicitly-restarted Lanczos

class EigTarget(Enum):            # replaces the 'SA'/'LM'/… which string (UT-K2)
    SA = "SA"; LA = "LA"; LM = "LM"; SR = "SR"; LR = "LR"

class EigResult(NamedTuple):
    vals: "Tensor | UniTensor"
    vecs: "list | None" = None

# Hermitian/symmetric eigensolver. Overloaded on the start-vector type
# (Tensor OR UniTensor) — the result mirrors it. The ARPACK `which` form:
def Lanczos(Hop: "LinOp", Tin, *, which: EigTarget = EigTarget.SA,
            maxiter=10000, cvg_crit=0.0, k=1, is_v=True, ncv=0,
            verbose=False) -> EigResult: ...

# The ground-state / explicitly-restarted form (was Lanczos(method=…) +
# the removed Lanczos_Gnd/Lanczos_ER/Lanczos_Gnd_Ut, UT-K2/K3). Overloaded on
# the start-vector type, so it subsumes Lanczos_Gnd_Ut:
def LanczosGnd(Hop: "LinOp", Tin, *, method: LanczosMethod = LanczosMethod.Gnd,
               cvg_crit=1e-14, maxiter=10000, k=1, is_v=True, is_row=False,
               max_krydim=0, verbose=False) -> EigResult: ...

# General (non-Hermitian) eigensolver. Overloaded on the start-vector type:
def Arnoldi(Hop: "LinOp", Tin, *, which: EigTarget = EigTarget.LM,
            maxiter=10000, cvg_crit=0.0, k=1, is_v=True, ncv=0,
            verbose=False) -> EigResult: ...

# Action of the matrix exponential exp(tau*H) @ v. ADD a Tensor overload (UT-K5):
def Lanczos_Exp(Hop: "LinOp", v, tau, *, cvg_crit=1e-14, maxiter=10000,
                verbose=False) -> "Tensor | UniTensor": ...
```

| API | Verdict | Behavior contract |
|---|---|---|
| `Lanczos` | **keep (Capitalized), split the dispatch** (UT-K1/K2) | Hermitian Krylov eigensolver over a `LinOp` + Tensor/UniTensor start vector; returns `[eigval, (eigvec)]` matching the operand type. *Migration:* keep the ARPACK `which` family under `Lanczos`; move the `method` (`'Gnd'`/`'ER'`) family to `LanczosGnd`, with `Lanczos(method=…)` as a `DeprecationWarning`-shimmed alias for one release. Replace the `which` **string** with the `EigTarget` enum (string accepted+deprecated). Verified ground-state: probe *"Lanczos which='SA' ground-state eigenvalue matches dense Eigh(H).min()"*. |
| `LanczosGnd` (← `Lanczos(method=…)` / `Lanczos_Gnd` / `Lanczos_ER` / `Lanczos_Gnd_Ut`) | **add / consolidate** (UT-K2/K3/K4) | NEW consolidated ground-state / ER Lanczos, **overloaded on the start-vector type** (subsuming the removed Tensor `Lanczos_Gnd`/`Lanczos_ER` and the UniTensor `Lanczos_Gnd_Ut`). `method=LanczosMethod.Gnd\|ER`. Verified: probe *"Lanczos(op, v, method='Gnd') dispatches the naive-Lanczos ground-state path …"* + *"Lanczos method='Gnd' ground-state eigenvalue matches dense Eigh(H).min()"* + *"Lanczos on a UniTensor start vector returns a list of UniTensors …"*. |
| `Lanczos_Gnd` | **remove → `LanczosGnd`** (UT-K3) | Currently **absent** (commented out). Do not re-expose under this Tensor-only name; the `LanczosGnd` operand overload replaces it. *Migration:* delete the dormant `c_Lanczos_Gnd` block; document the `LanczosGnd`/`Lanczos(method='Gnd')` route. |
| `Lanczos_ER` | **remove → `LanczosGnd(method=ER)`** (UT-K3) | Currently **absent** (commented out). Replaced by `LanczosGnd(method=LanczosMethod.ER)`. *Migration:* delete the dormant `c_Lanczos_ER` block. |
| `Lanczos_Gnd_Ut` | **remove → `LanczosGnd` (UniTensor overload)** (UT-K3/K4) | Currently **absent** (commented out). The UniTensor ground state is obtained via the `LanczosGnd` UniTensor overload (or, in 1.1.0, `Lanczos(Hop, ut_v0, method='Gnd')`). *Migration:* delete the dormant `c_Lanczos_Gnd_Ut` block; drop the `_Ut` suffix convention. |
| `Arnoldi` | **keep (Capitalized)** (UT-K1) | General (non-Hermitian) Krylov eigensolver; Tensor/UniTensor overloads; `which=EigTarget`. *Migration:* replace the `which` string with the enum (string accepted+deprecated). Verified: probe *"Arnoldi(op, v, which='SR') runs and returns eigen-results"* + *"Arnoldi smallest-real eigenvalue matches the dense ground state on a Hermitian operator"*. |
| `Lanczos_Exp` | **keep, add Tensor overload** (UT-K1/K5) | Action of `exp(tau·H)·v` for a Hermitian `Hop`. *Migration:* add a `Tensor` `v` overload for parity with `Lanczos`/`Arnoldi`. Verified: probe *"Lanczos_Exp(op, v, tau) == exp(tau*H) @ v (matches dense expm)"*. |

**No binding-fidelity / plumbing findings.** Like cat 08 (and unlike cats 04–07),
this category surfaces **no** leaked `c*` bindings and **no** live `conti.py`
wrappers — the reachable pybind layer forwards directly to `cytnx::linalg::`. The
only `c_*` symbols (`c_Lanczos_ER`/`c_Lanczos_Gnd`/`c_Lanczos_Gnd_Ut`) are
**commented out** and therefore not present at runtime (UT-K3). Gate 4 (raw-C++
probe) is skipped: *no binding-fidelity finding — identical C++/Python
free-function bindings.*

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python surface,
**R.2b** Doxygen for the C++ surface. Kept/renamed/added functions are documented.

### R.2a Python API (numpy-style)

### Hermitian eigensolvers (`Lanczos` / `LanczosGnd`)

```
cytnx.linalg.Lanczos(Hop, Tin, which=EigTarget.SA, maxiter=10000, cvg_crit=0,
                     k=1, is_v=True, ncv=0, verbose=False)      -> EigResult
cytnx.linalg.LanczosGnd(Hop, Tin, method=LanczosMethod.Gnd, cvg_crit=1e-14,
                        maxiter=10000, k=1, is_v=True, is_row=False,
                        max_krydim=0, verbose=False)            -> EigResult

Krylov eigensolver for a HERMITIAN/symmetric linear operator.

`Hop` is a `cytnx.LinOp` whose `matvec(v)` applies H to a vector; `Tin` is the
start vector (a `Tensor` OR a `UniTensor` — the result mirrors its type). `Lanczos`
uses the ARPACK path and targets the eigenvalue region named by `which`
(SA=smallest-algebraic, LA=largest-algebraic, LM=largest-magnitude, …). `LanczosGnd`
uses the naive (`method=Gnd`) or explicitly-restarted (`method=ER`) Lanczos to get
the ground state; it returns the same `EigResult(vals, vecs)`.

Returns
-------
EigResult
    `vals` (the `k` eigenvalues) and, when `is_v`, `vecs` (the eigenvectors),
    each a `Tensor` or `UniTensor` matching `Tin`.

Notes
-----
These Capitalized FREE functions act on a `LinOp` (SciPostPhysCodeb.53, finding
UT-K1) — not to be confused with any lowercased UniTensor member. Through cytnx
1.1.0 a SINGLE overloaded name `Lanczos` dispatched BOTH families via a `method`
string vs a `which` string with disjoint kwargs (finding UT-K2); the next version
splits them into `Lanczos` (ARPACK `which`) and `LanczosGnd` (`method`), each
overloaded on the start-vector type — consolidating the 1.1.0-absent
`Lanczos_Gnd`/`Lanczos_ER`/`Lanczos_Gnd_Ut` (finding UT-K3/K4). In 1.1.0 the
UniTensor ground state is reached via `Lanczos(Hop, ut_v0, method='Gnd')`.
```

### general eigensolver (`Arnoldi`) and exponential action (`Lanczos_Exp`)

```
cytnx.linalg.Arnoldi(Hop, Tin, which=EigTarget.LM, maxiter=10000, cvg_crit=0,
                     k=1, is_v=True, ncv=0, verbose=False)      -> EigResult
cytnx.linalg.Lanczos_Exp(Hop, v, tau, cvg_crit=1e-14, maxiter=10000,
                         verbose=False)                          -> Tensor | UniTensor

General Krylov eigensolver and matrix-exponential action.

`Arnoldi` solves the (possibly non-Hermitian) eigenproblem for `Hop`, targeting the
region named by `which` (LM=largest-magnitude, SR=smallest-real, …), over a Tensor
OR UniTensor start vector. `Lanczos_Exp` returns the action `exp(tau*H) @ v` for a
Hermitian `Hop` (a time-evolution / imaginary-time primitive).

Returns
-------
Arnoldi     : EigResult(vals, vecs)
Lanczos_Exp : the evolved vector, matching the type of `v`.

Notes
-----
Through cytnx 1.1.0 `Lanczos_Exp` accepted ONLY a UniTensor `v` (finding UT-K5);
the next version adds a Tensor overload for parity with `Lanczos`/`Arnoldi`. The
`which` string becomes the `EigTarget` enum; the code `'SM'` is unsupported and
raised a `RuntimeError` in 1.1.0.
```

### R.2b C++ API (Doxygen)

The C++ `cytnx::linalg::` Krylov solvers already carry the correct Capitalization
and the LinOp/UniTensor overloads; the next version's changes are: split the
overloaded `Lanczos` into `Lanczos` (ARPACK `which`) + `LanczosGnd` (`method`),
consolidating (and re-enabling) `Lanczos_Gnd`/`Lanczos_ER`/`Lanczos_Gnd_Ut`
(UT-K2/K3/K4), replace the string selectors with enums, and add a Tensor overload
to `Lanczos_Exp` (UT-K5). No binding-fidelity change is required — the pybind
lambdas already forward directly; the dormant `/* c_Lanczos_* */` block is deleted.

```cpp
namespace cytnx { namespace linalg {

/**
 * @brief Hermitian/symmetric Krylov eigensolver over a LinOp.
 * @details Lanczos uses the ARPACK path, targeting the region named by @p which
 *          (SA/LA/LM/…). Overloaded on the start-vector type (Tensor OR
 *          UniTensor); the result mirrors it. Capitalized: a free function acting
 *          on an object (finding UT-K1). In 1.1.0 the SAME name also carried a
 *          `method`-string family with disjoint kwargs (finding UT-K2) — the next
 *          version moves that to LanczosGnd.
 * @param Hop the linear operator. @param Tin the start vector.
 * @return {eigvals, (eigvecs)} as a vector matching @p Tin's type.
 */
std::vector<Tensor> Lanczos(LinOp *Hop, const Tensor &Tin,
                            const std::string which = "SA",
                            const cytnx_uint64 &maxiter = 10000,
                            const cytnx_double &cvg_crit = 0, const cytnx_uint64 &k = 1,
                            const bool &is_V = true, const cytnx_int32 &ncv = 0,
                            const bool &verbose = false);
std::vector<UniTensor> Lanczos(LinOp *Hop, const UniTensor &Tin,
                               const std::string which = "SA", /* …identical params… */);

/**
 * @brief Ground-state / explicitly-restarted Lanczos over a LinOp.
 * @details Consolidates the 1.1.0-absent Lanczos_Gnd (Tensor), Lanczos_ER
 *          (Tensor), and Lanczos_Gnd_Ut (UniTensor) — commented out on both the
 *          pybind and conti sides (finding UT-K3) — into one name overloaded on
 *          the start-vector type (finding UT-K4). @p method selects Gnd (naive) or
 *          ER (explicitly restarted).
 * @return {eigval, (eigvec)} matching @p Tin's type.
 */
std::vector<Tensor> LanczosGnd(LinOp *Hop, const Tensor &Tin,
                               const std::string method = "Gnd",
                               const double &CvgCrit = 1.0e-14,
                               const unsigned int &Maxiter = 10000, const cytnx_uint64 &k = 1,
                               const bool &is_V = true, const bool &is_row = false,
                               const cytnx_uint32 &max_krydim = 0, const bool &verbose = false);
std::vector<UniTensor> LanczosGnd(LinOp *Hop, const UniTensor &Tin, /* …identical params… */);

/**
 * @brief General (non-Hermitian) Krylov eigensolver over a LinOp (ARPACK).
 * @details Targets the region named by @p which (LM/SR/LR/…). Overloaded on the
 *          start-vector type. Capitalized free function (finding UT-K1).
 * @return {eigvals, (eigvecs)} matching @p Tin's type.
 */
std::vector<Tensor> Arnoldi(LinOp *Hop, const Tensor &Tin = Tensor(),
                            const std::string which = "LM", const cytnx_uint64 &maxiter = 10000,
                            const cytnx_double &cvg_crit = 1.0e-9, const cytnx_uint64 &k = 1,
                            const bool &is_V = true, const cytnx_int32 &ncv = 0,
                            const bool &verbose = false);
std::vector<UniTensor> Arnoldi(LinOp *Hop, const UniTensor &Tin, /* …identical params… */);

/**
 * @brief Action of the matrix exponential exp(tau*H) @ v for a Hermitian LinOp.
 * @details A time-evolution / imaginary-time primitive. The next version ADDS a
 *          Tensor overload for parity with Lanczos/Arnoldi (finding UT-K5); 1.1.0
 *          accepted only a UniTensor v.
 * @return the evolved vector matching @p v's type.
 */
UniTensor Lanczos_Exp(LinOp *Hop, const UniTensor &v, const Scalar &tau,
                      const double &CvgCrit = 1.0e-10, const unsigned int &Maxiter = 100000,
                      const bool &verbose = false);
// ADD (UT-K5): Tensor Lanczos_Exp(LinOp *Hop, const Tensor &v, const Scalar &tau, …);

}}  // namespace cytnx::linalg
```

---

# Appendix — related solver surface (cross-references, not re-audited here)

- **`LinOp`** is the operator interface every solver here consumes (`Hop`). It is a
  **separate class** with its own audit (a future `LinOp` plan) — the probe builds
  a `LinOp` subclass overriding `matvec` to supply the Hermitian operator; the
  ground state matches a direct dense `Eigh` (probe *"dense cytnx.linalg.Eigh(H).min()
  agrees with the Krylov ground state"*).
- **`Lanczos_Exp` sibling on `Tensor`:** absent (UT-K5) — the Tensor-only linalg
  surface (`Matmul`, `Det`, `Kron`, …) is inventoried in the cat-08 Appendix.
- **Dense counterparts:** `Eig`/`Eigh` (full dense eigendecomposition, cat 08) are
  the non-Krylov alternative when the operator is small enough to materialize.
