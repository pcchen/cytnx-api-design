# UniTensor — 08. Linalg operations (free functions)

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the siblings
> [`06-element-block-access.md`](06-element-block-access.md),
> [`07-arithmetic-elementwise.md`](07-arithmetic-elementwise.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's `cytnx.linalg` UniTensor surface to match §R exactly. All runtime
> claims verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_08_linalg.py` (all `[PASS]`, exit 0).
> **No C++ probe accompanies this category:** every function here is bound as a
> **direct pass-through pybind lambda** to the C++ `cytnx::linalg::` overload
> (`pybind/linalg_py.cpp`), with **no** `conti.py` wrapper and no signature or
> behavior change — so there is **no binding-fidelity finding** to verify on a
> separate raw-C++ side.

**Category scope — a different shape from cats 03–07.** These are the
**`cytnx.linalg` FREE functions**, *not* `UniTensor` members. This category
audits exactly the free functions that carry a **UniTensor overload** (accept
and/or return a `UniTensor`): the decompositions `Svd`/`Svd_truncate`/`Gesvd`/
`Gesvd_truncate`/`Rsvd`/`Hosvd`/`Qr`/`Qdr`/`Eig`/`Eigh`, the matrix functions
`ExpH`/`ExpM`/`InvM`/`InvM_`, and the element-wise `Inv`/`Inv_`/`Trace`/`Pow`/
`Pow_`/`Conj`/`Conj_`/`Norm`. The Krylov solvers `Arnoldi`/`Lanczos`/
`Lanczos_Exp` also carry a UniTensor overload but belong to **cat 09**
(cross-reference — not re-audited here). The `cytnx.linalg` functions that take
**only** `Tensor` are listed in the **Appendix** as UniTensor parity gaps.
Python bindings: `cytnx_src/pybind/linalg_py.cpp:40-157,172-201,204-490,750-925`;
C++ header: `cytnx_src/include/linalg.hpp:700-995,966-1110,1686-1699,2139-2148`.

> **Coverage note (validator).** `validate_doc.py UniTensor <dir>` checks
> `dir(cytnx.UniTensor)`; the linalg free functions are **not** UniTensor
> members, so they are neither required nor counted by the UniTensor validator —
> this file adds **no** UniTensor-member coverage and no regression. Machine
> coverage of the `cytnx.linalg` surface itself is a separate future `linalg`
> plan (out of scope here). Every in-scope function still appears in the R.1
> verdict table below.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per in-scope `cytnx.linalg` function (the UniTensor overload). `[I]` =
in-place (trailing `_`, mutates its operand, returns `None`). The probe assertion
backing each runtime claim is quoted. All names are **Capitalized** — the correct
spelling for free functions acting on objects (finding UT-X1).

| API | Live signature (1.1.0, UniTensor overload) | Returns | Description & evidence |
|---|---|---|---|
| `Svd` | `Svd(Tin, is_UvT=True)` | `list[UniTensor]` `[S, U, vT]` | Divide-and-conquer SVD. Returns **three** UniTensors in the order `[S, U, vT]` (`S` a diagonal UniTensor). Toggle is the **single** `is_UvT`. Probe: *"Svd(ut) returns a list of length 3 ([S, U, vT])"* + *"Svd order verified: M ~= U . diag(S) . vT reconstructs the input"* + *"Svd accepts the single is_UvT toggle"*. |
| `Svd_truncate` | `Svd_truncate(Tin, keepdim, err=0, is_UvT=True, return_err=0, mindim=1)` · overload `(…, min_blockdim, …)` | `list[UniTensor]` `[S, U, vT, (err)]` | Truncated SVD to `keepdim` singular values. With `return_err=1` the list gains a **trailing** error UniTensor → **length 4**; with `return_err=0` → **length 3**. Same single `is_UvT` toggle. Probe: *"Svd_truncate(ut, keepdim, return_err=1) returns a length-4 list … flag-dependent positional arity"* + *"… return_err=0 returns a length-3 list …"*. |
| `Gesvd` | `Gesvd(Tin, is_U=True, is_vT=True)` | `list[UniTensor]` `[S, U, vT]` | `?gesvd`-method SVD. Same `[S, U, vT]` result, but the toggles are the **separate** `is_U` / `is_vT` pair. Probe: *"Gesvd accepts the separate is_U / is_vT toggles"* + *"Gesvd REJECTS the single is_UvT keyword …"*. |
| `Gesvd_truncate` | `Gesvd_truncate(Tin, keepdim, err=0, is_U=True, is_vT=True, return_err=0, mindim=1)` · overload `(…, min_blockdim, …)` | `list[UniTensor]` | Truncated `Gesvd`; **separate** `is_U`/`is_vT` (unlike `Svd_truncate`). |
| `Rsvd` | `Rsvd(Tin, keepdim, err=0, is_U=True, is_vT=True, return_err=0, mindim=1, oversampling_summand=10, oversampling_factor=1.0, power_iteration=0, seed=-1)` · overload `(…, min_blockdim, …)` | `list[UniTensor]` | Randomized (already-truncating) SVD. **Separate** `is_U`/`is_vT`. Probe: *"Rsvd accepts the separate is_U / is_vT toggles"* + *"Rsvd REJECTS the single is_UvT keyword"*. |
| `Hosvd` | `Hosvd(Tn, mode, is_core=True, is_Ls=False, truncate_dim=[])` | `list[UniTensor]` | Higher-order SVD (Tucker); `mode` selects the per-leg ranks. Probe: *"linalg.Hosvd has a UniTensor overload"*. |
| `Qr` | `Qr(Tin, is_tau=False)` | `list[UniTensor]` `[Q, R, (tau)]` | QR decomposition; returns `[Q, R]` (or `[Q, R, tau]` if `is_tau`). Probe: *"Qr(ut) returns a length-2 list ([Q, R])"*. |
| `Qdr` | `Qdr(Tin, is_tau=False)` | `list[UniTensor]` `[Q, D, R, (tau)]` | QDR decomposition (`D` a diagonal UniTensor). Probe: *"linalg.Qdr has a UniTensor overload"*. |
| `Eig` | `Eig(Tin, is_V=True, row_v=False)` | `list[UniTensor]` `[e, (v)]` | General eigendecomposition; `[eigvals, eigvecs]`. Probe: *"linalg.Eig has a UniTensor overload"*. |
| `Eigh` | `Eigh(Tin, is_V=True, row_v=False)` | `list[UniTensor]` `[e, (v)]` | Hermitian eigendecomposition. Probe: *"Eigh(ut) returns a length-2 list ([eigvals, eigvecs])"*. |
| `ExpH` | `ExpH(Tin, a, b=0)` · `ExpH(Tin)` | `UniTensor` | Matrix exponential of a **Hermitian** UniTensor: `exp(a·H + b)`. Carries **both** a UniTensor and a Tensor overload. Probe: *"ExpH(ut, a) returns a UniTensor"* + *"linalg.ExpH accepts BOTH a UniTensor and a Tensor overload"*. |
| `ExpM` | `ExpM(Tin, a, b=0)` · `ExpM(Tin)` | `UniTensor` | Matrix exponential of a **general** (square) UniTensor. Both overloads present. Probe: *"linalg.ExpM accepts BOTH a UniTensor and a Tensor overload"*. |
| `InvM` | `InvM(Tin)` | `UniTensor` (new) | **Matrix inverse** (pure). Probe: *"InvM is the MATRIX inverse: InvM(B)[0,1] == -1/16 == -0.0625"*. |
| `InvM_` `[I]` | `InvM_(Tio)` | `None` | Matrix inverse **in place**. Probe: *"InvM_ is in-place: returns None"* + *"InvM_ mutated the operand in place …"*. |
| `Inv` | `Inv(Tin, clip=-1)` | `UniTensor` (new) | **Element-wise** reciprocal `1/x` (with `clip` guard) — **not** the matrix inverse. Probe: *"Inv is ELEMENT-WISE reciprocal: Inv(B)[0,1] == 1/1 == 1.0"* + *"Inv (element-wise) and InvM (matrix inverse) give DIFFERENT results …"*. |
| `Inv_` `[I]` | `Inv_(Tin, clip=-1)` | `None` | Element-wise reciprocal **in place**. Probe: *"Inv_ is in-place: returns None"*. |
| `Trace` | `Trace(Tn, axisA=0, axisB=1)` (int) · `Trace(Tn, axisA, axisB)` (str) | `UniTensor` (new) | Trace over two legs. Probe: *"Trace(ut) returns a UniTensor"*. |
| `Pow` | `Pow(Tin, p)` | `UniTensor` (new) | Element-wise power. Probe: *"Pow(ut, p) returns a UniTensor"*. |
| `Pow_` `[I]` | `Pow_(Tin, p)` | `None` | Element-wise power **in place**. Probe: *"Pow_ is in-place: returns None"*. |
| `Conj` | `Conj(Tin)` | `UniTensor` (new) | Complex conjugate. Probe: *"Conj(ut) returns a UniTensor"*. |
| `Conj_` `[I]` | `Conj_(Tin)` | `None` | Complex conjugate **in place**. Probe: *"Conj_ is in-place: returns None"*. |
| `Norm` | `Norm(T1)` | **`Tensor`** | 2-norm as a scalar `Tensor` (**not** a UniTensor), matching the member `Norm` (cat 07). Probe: *"Norm(ut) returns a cytnx.Tensor (the 2-norm scalar), NOT a UniTensor"*. |

**Absent at runtime (brief members not bound in `cytnx.linalg`):**
`Rsvd_truncate`, `Add`, `Sub`, `Mul`, `Div`, `Mod`. See findings UT-X5/UT-X6.
Probe: *"linalg.Rsvd_truncate is ABSENT …"* + *"linalg.Add/Sub/Mul/Div/Mod is
ABSENT …"* (one per name).

**Cross-reference (audited in cat 09, not here):** `Arnoldi`, `Lanczos`,
`Lanczos_Exp` carry UniTensor overloads but are Krylov **solvers** (category 09).

## A2. C++ ↔ Python mapping

Every row is a **direct pass-through**: the pybind lambda forwards straight to the
same-named `cytnx::linalg::` overload with the identical argument list — there is
no `conti.py` layer and no behavior change (hence **no** binding-fidelity finding
in this category). "Status" therefore reads **identical** throughout; the notes
carry the naming/ordering observations.

| C++ (`linalg.hpp`) | Python (`linalg_py.cpp`) | Status | Note |
|---|---|---|---|
| `std::vector<UniTensor> Svd(const UniTensor&, const bool &is_UvT=true)` (`:700`) | `Svd(Tin, is_UvT)` (`:45`) | identical | single `is_UvT` toggle (UT-X2); returns `[S, U, vT]` (UT-X3) |
| `std::vector<UniTensor> Svd_truncate(…, is_UvT, return_err, …)` (`:725,778`) | `Svd_truncate(…)` (`:149,157`) | identical | flag-dependent arity via `return_err` (UT-X3) |
| `std::vector<UniTensor> Gesvd(const UniTensor&, is_U=true, is_vT=true)` (`:711`) | `Gesvd(Tin, is_U, is_vT)` (`:56`) | identical | **separate** `is_U`/`is_vT` (UT-X2) |
| `std::vector<UniTensor> Gesvd_truncate(…, is_U, is_vT, …)` (`:796,814`) | `Gesvd_truncate(…)` (`:121,129`) | identical | separate `is_U`/`is_vT` (UT-X2) |
| `std::vector<UniTensor> Rsvd(const UniTensor&, keepdim, …, is_U, is_vT, …)` (`:870,904`) | `Rsvd(…)` (`:79,95`) | identical | separate `is_U`/`is_vT`; already truncating (UT-X2/X6) |
| `std::vector<UniTensor> Hosvd(…)` (`:913`) | `Hosvd(…)` (`:910`) | identical | Tucker/HOSVD |
| `std::vector<UniTensor> Qr(const UniTensor&, is_tau=false)` (`:985`) | `Qr(Tin, is_tau)` (`:754`) | identical | `[Q, R, (tau)]` (UT-X3) |
| `std::vector<UniTensor> Qdr(const UniTensor&, is_tau=false)` (`:995`) | `Qdr(Tin, is_tau)` (`:762`) | identical | `[Q, D, R, (tau)]` (UT-X3) |
| `std::vector<UniTensor> Eig(const UniTensor&, is_V=true, row_v=false)` (`:2059`) | `Eig(Tin, is_V, row_v)` (`:191`) | identical | `[e, (v)]` |
| `std::vector<UniTensor> Eigh(const UniTensor&, is_V=true, row_v=false)` (`:2029`) | `Eigh(Tin, is_V, row_v)` (`:178`) | identical | `[e, (v)]` |
| `UniTensor ExpH(const UniTensor&, a, b=0)` (`:927,947`) | `ExpH(Tin, a, b)` / `ExpH(Tin)` (`:210-344`) | identical | Hermitian matrix exp; both UT+Tensor overloads |
| `UniTensor ExpM(const UniTensor&, a, b=0)` (`:937,956`) | `ExpM(Tin, a, b)` / `ExpM(Tin)` (`:490-…`) | identical | general matrix exp; both UT+Tensor overloads |
| `UniTensor InvM(const UniTensor&)` (`:2139`) | `InvM(Tin)` (`:775`) | identical | **matrix** inverse (UT-X4) |
| `void InvM_(UniTensor&)` (`:2148`) | `InvM_(Tio)` (`:770`) | identical | matrix inverse in place; returns `None` |
| `UniTensor Inv(const UniTensor&, double clip=-1)` (`:1067`) | `Inv(Tin, clip)` (`:780`) | identical | **element-wise** reciprocal (UT-X4) |
| `void Inv_(UniTensor&, double clip=-1)` (`:1095`) | `Inv_(Tin, clip)` (`:787`) | identical | element-wise reciprocal in place; `None` |
| `UniTensor Trace(const UniTensor&, a=0, b=1)` / `(…, str, str)` (`:966,975`) | `Trace(Tn, axisA, axisB)` (`:854-868`) | identical | trace over two legs |
| `UniTensor Pow(const UniTensor&, double p)` (`:1017`) | `Pow(Tin, p)` (`:875`) | identical | element-wise power |
| `void Pow_(UniTensor&, double p)` (`:1037`) | `Pow_(Tin, p)` (`:882`) | identical | element-wise power in place; `None` |
| `UniTensor Conj(const UniTensor&)` (`:1103`) | `Conj(Tin)` (`:800`) | identical | conjugate |
| `void Conj_(UniTensor&)` (`:1110`) | `Conj_(Tin)` (`:803`) | identical | conjugate in place; `None` |
| `Tensor Norm(const UniTensor&)` (`:1699`) | `Norm(T1)` (`:825`) | identical | returns a scalar `Tensor` (UT-X1 note) |
| `UniTensor Add/Sub/Mul/Div/Mod(const UniTensor&, …)` (`linalg.hpp`, cat-07 refs) | *(unbound — no `linalg.Add/…`)* | **binding gap** | named arithmetic free functions not exposed (UT-X5) |
| *(no public `Rsvd_truncate` in C++ — only an internal `Rsvd_truncate_Block_UT_internal`)* (`src/linalg/Rsvd.cpp:1101`) | *(absent)* | **naming gap** | `Rsvd` already truncates; no separate public name (UT-X6) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/UniTensor_08_linalg.py` (on the 1.1.0 wheel). **There is no
binding-fidelity finding in this category:** the functions are direct pass-through
pybind lambdas over the C++ `cytnx::linalg::` overloads (no `conti.py` wrapper),
so no raw-C++ probe is needed (gate 4 recorded as *"no binding-fidelity finding —
identical C++/Python free-function bindings"*). Source `file:line` cites remain
for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-X1** | the `cytnx.linalg` decompositions/functions are **Capitalized** (`Svd`/`Gesvd`/`Qr`/`Eigh`/`InvM`/`Trace`/…) — and this is **correct** under N-casing | naming (N-casing — **positive demonstration**) | **free functions acting on objects stay Capitalized.** They take a UniTensor as primary operand and return a result; the SciPostPhysCodeb.53 rule Capitalizes such free functions (contrast **members**, which lowercase — cat 07 renamed the Capitalized *members* `Conj`/`Trace`/`Norm`/`Pow` → `conj`/`trace`/`norm`/`pow`). The v1 audit's linalg-C1 recommendation to *snake_case all linalg* is **explicitly reversed** here. Py probe *"the linalg free functions are Capitalized at runtime …"* + *"their lowercase spellings are NOT bound …"* | **KEEP Capitalized** — `Svd`, `Gesvd`, `Rsvd`, `Qr`, `Qdr`, `Eig`, `Eigh`, `ExpH`, `ExpM`, `InvM`, `InvM_`, `Inv`, `Inv_`, `Trace`, `Pow`, `Pow_`, `Conj`, `Conj_`, `Norm`, `Hosvd`, `Svd_truncate`, `Gesvd_truncate`. Do **not** snake_case. Document the members↔free-functions split (cross-ref cat 07). |
| **UT-X2** | **SVD toggle inconsistency** — `Svd`/`Svd_truncate` take a **single** `is_UvT` and **reject** `is_U`; `Gesvd`/`Gesvd_truncate`/`Rsvd` take **separate** `is_U`/`is_vT` and reject `is_UvT` | naming / ordering (N4) | **two incompatible toggle spellings across one family.** `Svd(Tin, is_UvT=true)` (`hpp:700`) vs `Gesvd(Tin, is_U=true, is_vT=true)` (`hpp:711`) / `Rsvd(…, is_U, is_vT, …)` (`hpp:870`). A user who learns `Svd(is_UvT=…)` cannot transfer it to `Gesvd`, and vice-versa — each rejects the other's keyword. Py probe *"Svd REJECTS the finer is_U keyword …"* + *"Svd_truncate REJECTS is_U …"* + *"Gesvd REJECTS the single is_UvT keyword …"* + *"Rsvd REJECTS the single is_UvT keyword"* | **unify on the finer `is_u` / `is_vt` pair family-wide** (so `Svd`/`Svd_truncate` gain `is_u`/`is_vt`, matching `Gesvd`/`Rsvd`), because it strictly subsumes the single toggle (compute U without vT). *Migration:* keep `is_UvT` as a `DeprecationWarning` alias meaning `is_u and is_vt` for one release. **If** unification is rejected, at minimum **document the split prominently**. (Note the casing `is_UvT`/`is_vT` itself is non-snake — snake_case to `is_uvt`/`is_vt` as part of the same change.) |
| **UT-X3** | decompositions return a **positional `list[UniTensor]`** whose **length and slot meaning depend on the flags** (`Svd → [S, U, vT]`; `Svd_truncate(return_err=1) → [S, U, vT, err]`; `Qr → [Q, R, (tau)]`; `Qdr → [Q, D, R, (tau)]`; `Eig(h) → [e, (v)]`) | ordering / documentation | **flag-dependent positional unpacking is error-prone.** The same call returns length 3 or 4 depending on `return_err`, and the slot order (`S` **first**, then `U`, then `vT`) is unusual (numpy/scipy return `U, S, Vh`). Py probe *"Svd(ut) returns a list of length 3 ([S, U, vT])"* + *"Svd order verified: M ~= U . diag(S) . vT …"* + *"Svd_truncate … return_err=1 returns a length-4 list …"* + *"… return_err=0 returns a length-3 list …"* + *"Qr(ut) returns a length-2 list …"* + *"Eigh(ut) returns a length-2 list …"* | **return a named-result object** (e.g. a `SvdResult(S, U, vT, err=None)` / `QrResult(Q, R, tau=None)` `NamedTuple`) so fields are accessed by name and the optional `err`/`tau` is an attribute, not a positional slot that shifts the tuple length. **If** the positional list is kept, **document the exact order and the flag-conditional length exhaustively** per function, and note the `[S, U, vT]` (S-first) order divergence from numpy's `U, S, Vh`. |
| **UT-X4** | `Inv` (**element-wise** reciprocal `1/x`) and `InvM` (**matrix** inverse) are a **near-name collision** for two entirely different operations | naming (correctness-risk) | **one letter separates two very different results.** `Inv(Tin, clip)` maps each element to `1/x` (`hpp:1067`); `InvM(Tin)` computes the matrix inverse (`hpp:2139`). On a non-diagonal matrix they differ: `Inv(B)[0,1] = 1/1 = 1.0` but `InvM(B)[0,1] = -1/16 = -0.0625`. Py probe *"Inv is ELEMENT-WISE reciprocal …"* + *"InvM is the MATRIX inverse …"* + *"Inv (element-wise) and InvM (matrix inverse) give DIFFERENT results — the near-name collision hides two distinct operations"* | **rename for disambiguation** — prefer `Reciprocal`/`Reciprocal_` (or `ElemInv`) for the element-wise op and keep `InvM`/`InvM_` for the matrix inverse; **or** at minimum document the two side-by-side with a prominent warning. Keep both Capitalized (UT-X1). *Migration:* `Inv`/`Inv_` remain `DeprecationWarning` aliases for one release. |
| **UT-X5** | the named arithmetic free functions `Add`/`Sub`/`Mul`/`Div`/`Mod` are **absent** from `cytnx.linalg`, though C++ has them and cat 07 relies on them | parity / capability (binding gap) | **binding gap.** `hasattr(cytnx.linalg, "Add")` … `"Mod"` are all False; only the operator dunders on `UniTensor` reach these ops (cat 07 UT-A6). Task 6 recorded the same absence. The tensor⊗tensor `Mod` additionally has an **unfinished C++ stub** (`Mod.cpp`, cat 07 UT-A2). Py probe *"linalg.Add/Sub/Mul/Div/Mod is ABSENT from cytnx.linalg …"* (one per name) | **bind the four working named free functions** `linalg.Add`/`Sub`/`Mul`/`Div` (Capitalized, over the existing C++ overloads) for parity with the operator dunders and with `Conj`/`Trace`/… ; **defer `Mod`** until the C++ `Mod(UniTensor, UniTensor)` stub is finished (bind the scalar case if desired). Cross-ref cat 07 UT-A2/UT-A6. |
| **UT-X6** | `Rsvd_truncate` is **absent** — there is no public `Rsvd_truncate` in Python **or** C++ (only an internal helper); `Rsvd` itself already truncates | parity / capability (naming) | **the brief member does not exist as a public name.** `hasattr(cytnx.linalg, "Rsvd_truncate")` is False; C++ exposes only `Rsvd_truncate_Block_UT_internal` (`src/linalg/Rsvd.cpp:1101`), while the public `Rsvd(Tin, keepdim, err, …)` **is** the truncating randomized SVD (it takes `keepdim`/`err`/`mindim`). Py probe *"linalg.Rsvd_truncate is ABSENT from cytnx.linalg (Rsvd exists, but the truncated variant is not bound)"* | **do not add a redundant `Rsvd_truncate`** — `Rsvd` already truncates; instead **document that `Rsvd` is the truncating randomized SVD** (parallel to `Svd` vs `Svd_truncate`, where the plain `Svd` does **not** truncate). If naming symmetry with `Svd`/`Gesvd` is desired, alias `Rsvd_truncate → Rsvd`; otherwise keep the single name. |

## A4. Argument ordering — positional & keyword

Every function takes the **UniTensor as the primary positional operand `Tin`**
first, then operation parameters. There is no keyword-only metadata block (these
are free functions, not constructors).

| API | positional-required (in order) | operation parameters (keyword-capable) |
|---|---|---|
| `Svd` | `Tin` | `is_UvT=True` |
| `Svd_truncate` | `Tin`, `keepdim` | `err=0`, `is_UvT=True`, `return_err=0`, `mindim=1` (+ `min_blockdim` overload) |
| `Gesvd` | `Tin` | `is_U=True`, `is_vT=True` |
| `Gesvd_truncate` | `Tin`, `keepdim` | `err`, `is_U`, `is_vT`, `return_err`, `mindim` (+ `min_blockdim`) |
| `Rsvd` | `Tin`, `keepdim` | `err`, `is_U`, `is_vT`, `return_err`, `mindim`, `oversampling_summand`, `oversampling_factor`, `power_iteration`, `seed` |
| `Hosvd` | `Tn`, `mode` | `is_core=True`, `is_Ls=False`, `truncate_dim=[]` |
| `Qr` / `Qdr` | `Tin` | `is_tau=False` |
| `Eig` / `Eigh` | `Tin` | `is_V=True`, `row_v=False` |
| `ExpH` / `ExpM` | `Tin` | `a`, `b=0` (or no-arg overload) |
| `InvM` / `InvM_` | `Tin` / `Tio` | *(none)* |
| `Inv` / `Inv_` | `Tin` | `clip=-1` |
| `Trace` | `Tn` | `axisA=0`, `axisB=1` |
| `Pow` / `Pow_` | `Tin`, `p` | *(none)* |
| `Conj` / `Conj_` / `Norm` | `Tin` / `T1` | *(none)* |

- **Canonical positional rule (§R.0):** `Tin` (the operand) first, then
  parameters — matches the live order; no reordering needed.
- **Toggle naming (UT-X2):** the flag spelling is the defect, not the position:
  unify on `is_u`/`is_vt` (snake_case) across the SVD family.
- **Parameter naming:** `is_UvT`/`is_vT`/`is_V` mix casing — snake_case to
  `is_uvt`/`is_vt`/`is_v` as part of the UT-X2 change. `Tin`/`Tio`/`Tn`/`T1` are
  terse but consistent within the module.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version `cytnx.linalg` UniTensor
surface. Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53) — free functions acting on objects stay
  Capitalized.** This is the **positive demonstration** of the convention: `Svd`,
  `Gesvd`, `Rsvd`, `Qr`, `Qdr`, `Eig`, `Eigh`, `ExpH`, `ExpM`, `InvM`, `InvM_`,
  `Inv`, `Inv_`, `Trace`, `Pow`, `Pow_`, `Conj`, `Conj_`, `Norm`, `Hosvd`,
  `Svd_truncate`, `Gesvd_truncate` are **kept Capitalized** (UT-X1). The v1
  audit's *snake_case-all-linalg* recommendation is **explicitly reversed**. The
  same operations that appear as UniTensor **members** (`conj`/`trace`/`norm`/
  `pow`, cat 07) are lowercased **as members** but Capitalized **as these free
  functions** — the two coexist by design (cross-ref cat 07 UT-A3).
- **N-underscore — a trailing `_` marks in-place.** `InvM_`/`Inv_`/`Pow_`/`Conj_`
  mutate their operand and return `None` (faithful to the C++ `void` in-place
  forms). The pure forms return a new object.
- **Parameter casing is snake_case.** The flags `is_UvT`/`is_vT`/`is_V` →
  `is_uvt`/`is_vt`/`is_v` (UT-X2/A4); unify the SVD family on the finer
  `is_u`/`is_vt` pair.
- **Decomposition results are named, not flag-dependent positional lists.**
  Return a `NamedTuple`-style result (`SvdResult`, `QrResult`, `EighResult`, …)
  so optional outputs (`err`, `tau`, eigenvectors) are named attributes, not
  tuple slots whose presence shifts the length (UT-X3). If the positional list is
  retained, the order and flag-conditional length are documented exhaustively.
- **Binding fidelity: none.** These are direct pass-through pybind lambdas to the
  C++ `cytnx::linalg::` overloads — no `conti.py` wrapper, no leaked `c*` binding,
  no signature change. There is nothing to un-leak here (contrast cat 07).

## R.1 Recommended API (exact signatures + behavior contract)

```python
# cytnx.linalg — free functions on UniTensor (Capitalized; act on objects)
from typing import NamedTuple

class SvdResult(NamedTuple):
    S: "UniTensor"; U: "UniTensor | None"; vT: "UniTensor | None"; err: "UniTensor | None" = None
class QrResult(NamedTuple):
    Q: "UniTensor"; R: "UniTensor"; tau: "UniTensor | None" = None
class QdrResult(NamedTuple):
    Q: "UniTensor"; D: "UniTensor"; R: "UniTensor"; tau: "UniTensor | None" = None
class EigResult(NamedTuple):
    e: "UniTensor"; v: "UniTensor | None" = None

def Svd(Tin: "UniTensor", is_u: bool = True, is_vt: bool = True) -> SvdResult: ...      # was is_UvT (UT-X2)
def Svd_truncate(Tin, keepdim, err=0.0, is_u=True, is_vt=True, return_err=0, mindim=1) -> SvdResult: ...
def Gesvd(Tin: "UniTensor", is_u: bool = True, is_vt: bool = True) -> SvdResult: ...
def Gesvd_truncate(Tin, keepdim, err=0.0, is_u=True, is_vt=True, return_err=0, mindim=1) -> SvdResult: ...
def Rsvd(Tin, keepdim, err=0.0, is_u=True, is_vt=True, return_err=0, mindim=1,
         oversampling_summand=10, oversampling_factor=1.0, power_iteration=0, seed=-1) -> SvdResult: ...
def Hosvd(Tn, mode, is_core=True, is_ls=False, truncate_dim=()) -> list["UniTensor"]: ...
def Qr(Tin: "UniTensor", is_tau: bool = False) -> QrResult: ...
def Qdr(Tin: "UniTensor", is_tau: bool = False) -> QdrResult: ...
def Eig(Tin: "UniTensor", is_v: bool = True, row_v: bool = False) -> EigResult: ...
def Eigh(Tin: "UniTensor", is_v: bool = True, row_v: bool = False) -> EigResult: ...
def ExpH(Tin: "UniTensor", a=..., b=0) -> "UniTensor": ...     # Hermitian matrix exp
def ExpM(Tin: "UniTensor", a=..., b=0) -> "UniTensor": ...     # general matrix exp
def InvM(Tin: "UniTensor") -> "UniTensor": ...                 # MATRIX inverse
def InvM_(Tio: "UniTensor") -> None: ...                       # matrix inverse, in place
def Inv(Tin: "UniTensor", clip: float = -1) -> "UniTensor": ...   # ELEMENT-WISE reciprocal (rename: Reciprocal)
def Inv_(Tin: "UniTensor", clip: float = -1) -> None: ...         # element-wise reciprocal, in place
def Trace(Tn: "UniTensor", axisA: int | str = 0, axisB: int | str = 1) -> "UniTensor": ...
def Pow(Tin: "UniTensor", p: float) -> "UniTensor": ...
def Pow_(Tin: "UniTensor", p: float) -> None: ...
def Conj(Tin: "UniTensor") -> "UniTensor": ...
def Conj_(Tin: "UniTensor") -> None: ...
def Norm(T1: "UniTensor") -> "Tensor": ...                     # 2-norm scalar Tensor (NOT UniTensor)
# ADD (UT-X5): Add / Sub / Mul / Div — named arithmetic free functions (parity with operators)
def Add(L: "UniTensor", R) -> "UniTensor": ...
def Sub(L: "UniTensor", R) -> "UniTensor": ...
def Mul(L: "UniTensor", R) -> "UniTensor": ...
def Div(L: "UniTensor", R) -> "UniTensor": ...
```

| API | Verdict | Behavior contract |
|---|---|---|
| `Svd` | **keep (Capitalized), unify toggle** (UT-X1/X2/X3) | Divide-and-conquer SVD; returns `SvdResult(S, U, vT)`. *Migration:* add `is_u`/`is_vt`, keep `is_UvT` as a `DeprecationWarning` alias for one release; return a named result (positional list aliased for one release). |
| `Svd_truncate` | **keep, unify toggle** (UT-X2/X3) | Truncated SVD to `keepdim`; `SvdResult` with `err` set when `return_err`. |
| `Gesvd` | **keep** (UT-X1/X2) | `?gesvd` SVD; `SvdResult(S, U, vT)`; already has `is_u`/`is_vt` (snake_case them). |
| `Gesvd_truncate` | **keep** (UT-X2/X3) | Truncated `Gesvd`; `SvdResult`. |
| `Rsvd` | **keep** (UT-X1/X2/X6) | Randomized **truncating** SVD; `SvdResult`. Document that it already truncates (no separate `Rsvd_truncate`). |
| `Hosvd` | **keep** (UT-X1) | Higher-order (Tucker) SVD; list of factor UniTensors + core. |
| `Qr` | **keep** (UT-X1/X3) | QR; returns `QrResult(Q, R, tau=None)` (`tau` set when `is_tau`). |
| `Qdr` | **keep** (UT-X1/X3) | QDR; `QdrResult(Q, D, R, tau=None)`. |
| `Eig` | **keep** (UT-X1/X3) | General eigendecomposition; `EigResult(e, v=None)`. |
| `Eigh` | **keep** (UT-X1/X3) | Hermitian eigendecomposition; `EigResult(e, v=None)`. |
| `ExpH` | **keep** (UT-X1) | Matrix exponential of a Hermitian UniTensor, `exp(a·H + b)`. |
| `ExpM` | **keep** (UT-X1) | Matrix exponential of a general square UniTensor. |
| `InvM` | **keep** (UT-X1/X4) | **Matrix** inverse (pure). Document alongside `Inv` to avoid confusion. |
| `InvM_` | **keep** (UT-X1/X4) | Matrix inverse in place; returns `None`. |
| `Inv` → `Reciprocal` | **rename (disambiguate)** (UT-X4) | **Element-wise** reciprocal `1/x` (`clip`-guarded). *Migration:* `Reciprocal` (Capitalized) is the new name; `Inv` remains a `DeprecationWarning` alias for one release. |
| `Inv_` → `Reciprocal_` | **rename (disambiguate)** (UT-X4) | Element-wise reciprocal in place; returns `None`. *Migration:* alias `Inv_`. |
| `Trace` | **keep** (UT-X1) | Trace over legs `axisA`,`axisB` (int or label); returns a UniTensor. |
| `Pow` | **keep** (UT-X1) | Element-wise power; new UniTensor. |
| `Pow_` | **keep** (UT-X1) | Element-wise power in place; returns `None`. |
| `Conj` | **keep** (UT-X1) | Complex conjugate; new UniTensor. |
| `Conj_` | **keep** (UT-X1) | Complex conjugate in place; returns `None`. |
| `Norm` | **keep** (UT-X1) | 2-norm as a scalar `cytnx.Tensor` (not a UniTensor). |
| `Add` / `Sub` / `Mul` / `Div` | **add** (UT-X5) | NEW: named arithmetic free functions (Capitalized), parity with the operator dunders and with C++. *Migration:* bind over the existing C++ overloads. |
| `Mod` | **defer** (UT-X5) | Await the C++ `Mod(UniTensor, UniTensor)` stub (cat 07 UT-A2); bind the scalar case if desired. |
| `Rsvd_truncate` | **do not add** (UT-X6) | Redundant — `Rsvd` already truncates. Optionally alias `Rsvd_truncate → Rsvd` for naming symmetry with `Svd`/`Svd_truncate`. |

**No binding-fidelity / plumbing findings.** Unlike cats 04–07, this category
surfaces **no** leaked `c*` bindings and **no** `conti.py` wrappers — the pybind
layer forwards directly to `cytnx::linalg::`. Gate 4 (raw-C++ probe) is therefore
skipped: *no binding-fidelity finding — identical C++/Python free-function
bindings.*

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python surface,
**R.2b** Doxygen for the C++ surface. Kept/renamed/added functions are documented.

### R.2a Python API (numpy-style)

### SVD family (`Svd` / `Svd_truncate` / `Gesvd` / `Gesvd_truncate` / `Rsvd`)

```
cytnx.linalg.Svd(Tin, is_u=True, is_vt=True)            -> SvdResult(S, U, vT)
cytnx.linalg.Svd_truncate(Tin, keepdim, err=0, is_u=True, is_vt=True,
                          return_err=0, mindim=1)        -> SvdResult(S, U, vT, err)
cytnx.linalg.Gesvd(Tin, is_u=True, is_vt=True)          -> SvdResult(S, U, vT)
cytnx.linalg.Rsvd(Tin, keepdim, ...)                     -> SvdResult(S, U, vT, err)

Singular-Value Decomposition of a UniTensor.

Splits `Tin` at its rowrank into a matrix M and factors it as M = U @ diag(S) @ vT.
The result is returned S-FIRST as SvdResult(S, U, vT): `S` is a DIAGONAL UniTensor
of singular values, `U`/`vT` the (optional) isometries. Set `is_u=False` /
`is_vt=False` to skip an isometry.

`Svd` uses divide-and-conquer; `Gesvd` uses LAPACK ?gesvd; `Rsvd` is a randomized,
already-TRUNCATING variant (there is no separate `Rsvd_truncate`). The `_truncate`
forms keep the `keepdim` largest singular values (or drop those below `err`); with
`return_err=1` the discarded weight is returned as the `err` field.

Notes
-----
This is a FREE function, Capitalized because it acts on an object (SciPostPhysCodeb.53)
— not to be confused with the lowercased UniTensor *members* (cat 07). Through
cytnx 1.1.0 `Svd`/`Svd_truncate` used a SINGLE `is_UvT` toggle while `Gesvd`/`Rsvd`
used separate `is_U`/`is_vT` (finding UT-X2); the next version unifies on
`is_u`/`is_vt`. The 1.1.0 return was a flag-dependent positional `list[UniTensor]`
in the order `[S, U, vT, (err)]` (finding UT-X3) — note S is FIRST, unlike numpy's
`U, S, Vh`.

Returns
-------
SvdResult
    Named result with fields `S`, `U`, `vT`, and (truncating forms) `err`.
```

### decompositions (`Qr` / `Qdr` / `Eig` / `Eigh` / `Hosvd`)

```
cytnx.linalg.Qr(Tin, is_tau=False)       -> QrResult(Q, R, tau)
cytnx.linalg.Qdr(Tin, is_tau=False)      -> QdrResult(Q, D, R, tau)
cytnx.linalg.Eig(Tin, is_v=True, row_v=False)   -> EigResult(e, v)
cytnx.linalg.Eigh(Tin, is_v=True, row_v=False)  -> EigResult(e, v)
cytnx.linalg.Hosvd(Tn, mode, is_core=True, is_ls=False, truncate_dim=()) -> list[UniTensor]

Orthogonal / eigen / higher-order decompositions of a UniTensor.

`Qr` -> Q (isometry) and R (upper-triangular); `Qdr` additionally splits R into a
diagonal D and unit-diagonal R. `Eig` (general) and `Eigh` (Hermitian) return the
eigenvalues `e` and, when `is_v`, the eigenvectors `v`. `Hosvd` is the Tucker/HOSVD.

Notes
-----
`tau` (Qr/Qdr) and `v` (Eig/Eigh) are present only when the corresponding flag is
set — in the next version they are NAMED result fields rather than a shifting
positional slot (finding UT-X3).

Returns
-------
QrResult / QdrResult / EigResult / list[UniTensor]
```

### matrix functions (`ExpH` / `ExpM` / `InvM` / `InvM_`)

```
cytnx.linalg.ExpH(Tin, a, b=0)   -> UniTensor    # exp(a*H + b), H Hermitian
cytnx.linalg.ExpM(Tin, a, b=0)   -> UniTensor    # exp(a*M + b), M general square
cytnx.linalg.InvM(Tin)           -> UniTensor    # MATRIX inverse (pure)
cytnx.linalg.InvM_(Tio)          -> None          # MATRIX inverse, in place

Matrix exponential and matrix inverse of a (square) UniTensor.

`ExpH` assumes `Tin` is Hermitian (uses its eigendecomposition); `ExpM` handles a
general square matrix. `InvM` returns the matrix inverse as a new UniTensor;
`InvM_` inverts IN PLACE and returns None.

Notes
-----
`InvM`/`InvM_` are the MATRIX inverse — NOT to be confused with the element-wise
reciprocal `Inv`/`Inv_` (finding UT-X4). `ExpH`/`ExpM` carry both a UniTensor and a
Tensor overload.

Returns
-------
UniTensor (ExpH/ExpM/InvM) / None (InvM_)
```

### element-wise (`Inv`→`Reciprocal` / `Inv_` / `Trace` / `Pow` / `Pow_` / `Conj` / `Conj_` / `Norm`)

```
cytnx.linalg.Inv(Tin, clip=-1)   -> UniTensor    # element-wise 1/x  (rename: Reciprocal)
cytnx.linalg.Inv_(Tin, clip=-1)  -> None          # element-wise 1/x, in place
cytnx.linalg.Trace(Tn, axisA=0, axisB=1) -> UniTensor
cytnx.linalg.Pow(Tin, p)         -> UniTensor
cytnx.linalg.Pow_(Tin, p)        -> None
cytnx.linalg.Conj(Tin)           -> UniTensor
cytnx.linalg.Conj_(Tin)          -> None
cytnx.linalg.Norm(T1)            -> Tensor         # 2-norm scalar (NOT a UniTensor)

Element-wise maps and reductions on a UniTensor (free-function form).

`Inv` maps each element to `1/x` (guarded by `clip`) — the ELEMENT-WISE reciprocal,
distinct from the matrix inverse `InvM` (finding UT-X4); it is renamed `Reciprocal`
for clarity. `Trace` contracts legs `axisA`,`axisB`. `Pow` raises elements to `p`.
`Conj` conjugates. `Norm` returns the 2-norm as a scalar `cytnx.Tensor`.

The `_`-suffixed forms mutate their operand in place and return None.

Notes
-----
These Capitalized FREE functions coexist with the lowercased UniTensor *members*
`inv`/`trace`/`pow`/`conj`/`norm` (cat 07 UT-A3): members lowercase, free functions
Capitalized (finding UT-X1). `Inv`/`Inv_` remain `DeprecationWarning` aliases of
`Reciprocal`/`Reciprocal_` for one release.

Returns
-------
UniTensor / None / Tensor (Norm)
```

### R.2b C++ API (Doxygen)

The C++ `cytnx::linalg::` free functions already carry the correct Capitalization
and the UniTensor overloads; the next version's changes are: unify the SVD toggles
on `is_u`/`is_vt` (UT-X2), return named result structs (UT-X3), rename `Inv`→
`Reciprocal` (UT-X4), and add `Add`/`Sub`/`Mul`/`Div` free functions (UT-X5). No
binding-fidelity change is required — the pybind lambdas already forward directly.

```cpp
namespace cytnx { namespace linalg {

/**
 * @brief Singular-Value Decomposition of a UniTensor.
 * @details Splits @p Tin at its rowrank and factors M = U diag(S) vT. Returns the
 *          factors S-FIRST (S a diagonal UniTensor). Svd uses divide-and-conquer;
 *          Gesvd uses ?gesvd; Rsvd is a randomized, already-truncating variant
 *          (no separate Rsvd_truncate, finding UT-X6). The next version unifies
 *          the toggles on is_u/is_vt (was the single is_UvT on Svd vs is_U/is_vT
 *          on Gesvd/Rsvd, finding UT-X2) and returns a named result rather than a
 *          flag-dependent vector (finding UT-X3). Capitalized: a free function
 *          acting on an object (finding UT-X1).
 * @param Tin the input UniTensor. @param is_u,is_vt whether to compute U / vT.
 * @return SvdResult{S, U, vT, err}.
 */
std::vector<cytnx::UniTensor> Svd(const cytnx::UniTensor &Tin, const bool &is_UvT = true);
std::vector<cytnx::UniTensor> Gesvd(const cytnx::UniTensor &Tin, const bool &is_U = true,
                                    const bool &is_vT = true);
std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin, const cytnx_uint64 &keepdim,
                                           const double &err = 0., const bool &is_UvT = true,
                                           const unsigned int &return_err = 0,
                                           const cytnx_uint64 &mindim = 1);

/**
 * @brief QR / QDR / eigen / higher-order decompositions of a UniTensor.
 * @details Qr -> {Q, R}; Qdr -> {Q, D, R}; Eig/Eigh -> {e, (v)}; Hosvd -> Tucker
 *          factors. Optional outputs (tau, v) become named fields in the next
 *          version, not shifting positional slots (finding UT-X3).
 * @return the decomposition factors as a named result / vector.
 */
std::vector<cytnx::UniTensor> Qr(const cytnx::UniTensor &Tin, const bool &is_tau = false);
std::vector<cytnx::UniTensor> Qdr(const cytnx::UniTensor &Tin, const bool &is_tau = false);
std::vector<UniTensor> Eig(const cytnx::UniTensor &Tin, const bool &is_V = true,
                           const bool &row_v = false);
std::vector<UniTensor> Eigh(const cytnx::UniTensor &Tin, const bool &is_V = true,
                            const bool &row_v = false);

/**
 * @brief Matrix exponential and matrix inverse (square UniTensor).
 * @details ExpH assumes Hermitian input; ExpM handles a general square matrix.
 *          InvM is the MATRIX inverse (pure); InvM_ inverts in place (returns
 *          void -> None in Python). NOT the element-wise reciprocal Inv (finding
 *          UT-X4).
 * @return ExpH/ExpM/InvM: a new UniTensor. InvM_: void.
 */
template <class T> cytnx::UniTensor ExpH(const cytnx::UniTensor &Tin, const T &a, const T &b = 0);
template <class T> cytnx::UniTensor ExpM(const cytnx::UniTensor &Tin, const T &a, const T &b = 0);
UniTensor InvM(const cytnx::UniTensor &Tin);
void InvM_(UniTensor &Tin);

/**
 * @brief Element-wise maps and reductions (free-function form).
 * @details Inv is the ELEMENT-WISE reciprocal 1/x (clip-guarded) — renamed
 *          Reciprocal to disambiguate from the matrix inverse InvM (finding
 *          UT-X4). Trace contracts two legs; Pow raises to p; Conj conjugates;
 *          Norm returns the 2-norm as a scalar Tensor (NOT a UniTensor). The `_`
 *          forms are in place (void). These Capitalized free functions coexist
 *          with the lowercased UniTensor members (cat 07, finding UT-X1).
 * @return pure form: a new UniTensor (Norm: a Tensor). in-place (_): void.
 */
cytnx::UniTensor Inv(const cytnx::UniTensor &Tin, double clip = -1.);   // -> Reciprocal
void Inv_(cytnx::UniTensor &Tin, double clip = -1.);
cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const cytnx_int64 &a = 0,
                       const cytnx_int64 &b = 1);
cytnx::UniTensor Pow(const cytnx::UniTensor &Tin, const double &p);
void Pow_(cytnx::UniTensor &Tin, const double &p);
cytnx::UniTensor Conj(const cytnx::UniTensor &UT);
void Conj_(cytnx::UniTensor &UT);
Tensor Norm(const cytnx::UniTensor &uTl);

/**
 * @brief NEW: named element-wise binary arithmetic free functions (finding UT-X5).
 * @details Add/Sub/Mul/Div on UniTensor — parity with the operator dunders and
 *          with the existing C++ overloads. Mod is deferred until the
 *          Mod(UniTensor, UniTensor) stub is finished (cat 07 UT-A2).
 */
UniTensor Add(const UniTensor &L, const UniTensor &R);
UniTensor Sub(const UniTensor &L, const UniTensor &R);
UniTensor Mul(const UniTensor &L, const UniTensor &R);
UniTensor Div(const UniTensor &L, const UniTensor &R);

}}  // namespace cytnx::linalg
```

---

# Appendix — Tensor-only `cytnx.linalg` functions (UniTensor parity gaps)

The `cytnx.linalg` functions below accept **only** `Tensor` — they have **no
UniTensor overload** and reject a UniTensor argument at runtime. They are listed
here as UniTensor parity gaps for the future `linalg` plan; whether each *should*
gain a UniTensor overload is that plan's decision (some, e.g. `Matmul`/`Dot`, are
naturally the UniTensor `Contract`/contraction surface — cat 10). Probe: *"Matmul
is Tensor-only: rejects a UniTensor argument"* + *"Det …"* + *"Kron …"*.

| Tensor-only function | Note |
|---|---|
| `Abs`, `Abs_` | element-wise absolute value |
| `Axpy`, `Axpy_` | BLAS `a·x (+ y)` |
| `Det` | determinant (rejects UniTensor — probe) |
| `Diag` | diagonal extract/build |
| `Directsum` | direct sum along shared axes |
| `Dot` | dot product (UniTensor path is `Contract`, cat 10) |
| `Exp`, `Exp_`, `Expf`, `Expf_` | element-wise exp (contrast `ExpH`/`ExpM`, which DO take UniTensor) |
| `Gemm`, `Gemm_` | BLAS general matrix multiply |
| `Ger` | BLAS rank-1 update |
| `Kron` | Kronecker product (rejects UniTensor — probe) |
| `Lstsq` | least squares |
| `Matmul`, `Matmul_dg` | matrix multiply (UniTensor path is `Contract`, cat 10) |
| `Max`, `Min`, `Sum` | reductions |
| `Outer` | outer product |
| `Rand_isometry` | random isometry generator |
| `Tensordot`, `Tensordot_dg` | tensor contraction (UniTensor path is `Contract`/`ncon`, cat 10) |
| `Tridiag` | tridiagonal eigen-solver |
| `Vectordot` | vector dot |

**Krylov solvers with UniTensor overloads (audited in cat 09, not here):**
`Arnoldi`, `Lanczos`, `Lanczos_Exp`.
