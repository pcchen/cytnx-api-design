# Tensor — 06. Linear algebra (member) & reductions

> **Superset-method rollout** (Tensor, category 06 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings, arg ordering) and a self-contained **Recommendation** that
> is the *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` member linear-algebra / reduction surface should be implemented to
> match §R exactly. Every behavioral claim is verified against the installed
> `cytnx==1.1.0` wheel by `docs/api-audit/probes/Tensor_06_linalg.py` (all
> `[PASS]`, exit 0).
> **No C++ probe accompanies this category.** These members are Capitalized
> `.def`-ed pass-throughs to the C++ `Tensor` methods (`tensor_py.cpp:1781-1798`),
> which forward to the `cytnx::linalg::` free functions; the **only** `conti.py`
> wrapper here is `InvM_` (`Tensor_conti.py:99-101` — `self.cInvM_(); return self`),
> the *identical* in-place-returns-self mechanism whose raw-C++ `Tensor&` return is
> already verified in cat 05 (`probes/cpp/Tensor_05_arithmetic.cpp`, finding T-A4).
> No **new** binding-fidelity finding surfaces, so gate 4 is recorded as *"no
> binding-fidelity finding — the linalg members mirror the `cytnx.linalg` free
> functions; the sole `InvM_` conti-wrapper is the already-C++-verified cat-05
> T-A4 pattern."*

**Category scope:** the member methods that perform matrix linear algebra or a
whole-tensor reduction — the decompositions `Svd`/`Eigh`, the **matrix** inverse
`InvM`/`InvM_`, and the reductions `Trace`/`Max`/`Min` — plus the leaked raw
`cInvM_` binding the `InvM_` wrapper calls. The 2-norm scalar `Norm` and the
**element-wise** `Abs`/`Conj`/`Exp`/`Inv`/`Pow` maps are
[category 05](05-arithmetic-elementwise.md) (`Inv` there is the element-wise
reciprocal — **not** the matrix inverse `InvM` audited here; see T-X3). Every one
of these member names **also exists as a `cytnx.linalg` FREE function** which
**stays Capitalized** — those free functions are audited in the future `linalg`
plan (the UniTensor-overload analog is [UniTensor cat 08](../UniTensor/08-linalg-operations.md)),
**not** re-audited here. Python bindings: `cytnx_src/pybind/tensor_py.cpp:1781-1798`;
conti.py wrapper: `cytnx_src/cytnx/Tensor_conti.py:99-101`; C++ header:
`cytnx_src/include/Tensor.hpp:1612-1726`; free-function backings:
`cytnx_src/include/linalg.hpp` (`Svd`/`Eigh`/`InvM`/`InvM_`/`Trace`/`Max`/`Min`).

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place. All seven callable
members are **Capitalized** (the N-casing defect, T-X1); the decompositions
return a **flag-dependent positional `list[Tensor]`** (T-X2).

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `Svd` | `Svd(is_UvT=True)` | `list[Tensor]` `[S, U, vT]` (or `[S]`) | Singular-value decomposition of a rank-2 tensor. Returns **three** tensors in the order `[S, U, vT]` — `S` a **rank-1** singular-value vector, **first** (unlike numpy's `U, S, Vh`). With `is_UvT=False` → `[S]` only. **Value-verified by reconstruction.** Probe: *"`Svd()` returns a length-3 list [S, U, vT] …"* + *"`Svd` return ORDER is [S, U, vT] with S FIRST …"* + *"`Svd` order VALUE-VERIFIED: M ~= U . diag(S) . vT reconstructs the input …"* + *"`Svd(is_UvT=False)` returns a length-1 list [S] …"*. |
| `Eigh` | `Eigh(is_V=True, row_v=False)` | `list[Tensor]` `[eigvals, (eigvecs)]` | Hermitian eigendecomposition. Returns `[eigvals, eigvecs]` (length 2); with `is_V=False` → `[eigvals]`. Probe: *"`Eigh()` returns a length-2 list [eigvals, eigvecs] …"* + *"`Eigh` slot 0 is the eigenvalues … sum(eigvals) == trace == 5"* + *"`Eigh(is_V=False)` returns a length-1 list [eigvals] …"*. |
| `InvM` | `InvM()` | `Tensor` (new) | **Matrix** inverse of a square rank-2 tensor (pure). **Value-verified** `M @ InvM(M) ≈ I`. Distinct from the **element-wise** reciprocal `Inv` (cat 05). Probe: *"`InvM` is PURE: returns a NEW tensor …"* + *"`InvM` is the MATRIX inverse: B @ InvM(B) ~= I (value-verified)"* + *"`InvM` (matrix inverse) and `Inv` (element-wise reciprocal, cat 05) give DIFFERENT results …"*. |
| `InvM_` `[I]` | `InvM_()` | `Tensor` (self; **conti.py wrapper**) | Matrix inverse **in place**; conti.py (`:99-101`) forwards to the leaked raw `cInvM_` and `return self`. **Value-verified** the operand becomes `M⁻¹`. Probe: *"`InvM_()` returns SELF (conti.py wrapper) and inverts the MATRIX in place …"* + *"`InvM_` mutated the operand in place to the matrix inverse …"*. |
| `Trace` | `Trace(arg0, arg1)` | `Tensor` (new) | Sum of the diagonal over two axes. **The C++ defaults `a=0, b=1` are dropped** (bound with no `py::arg`, `:1798`), so `Trace()` raises `TypeError` and the two axes are positional-only (`arg0`/`arg1` — the meaningful C++ names lost). Probe: *"`Trace()` with NO args raises TypeError — the C++ defaults (a=0, b=1) were dropped …"* + *"`Trace(0, 1)` sums the diagonal … == 12"*. |
| `Max` | `Max()` | `Tensor` (scalar) | Maximum element, as a scalar `Tensor`. Probe: *"`Max()` returns the maximum element as a scalar cytnx.Tensor (max([0..4]) == 4)"*. |
| `Min` | `Min()` | `Tensor` (scalar) | Minimum element, as a scalar `Tensor`. Probe: *"`Min()` returns the minimum element as a scalar cytnx.Tensor (min([0..4]) == 0)"*. |

**Internal / plumbing (leaks into `dir(Tensor)`):** `cInvM_` — the raw pybind
binding (`&cytnx::Tensor::InvM_`, `:1783`) the conti.py `InvM_` wrapper calls. The
`c`-prefix is a reserved raw-binding spelling. Public today, but should never be.
Probe: *"the raw plumbing binding `cInvM_` LEAKS into public dir(Tensor) …"* +
*"`cInvM_()` (raw primitive) mutates the receiver in place (self-aliasing) …"*.

**Same names as `cytnx.linalg` FREE functions (correctly Capitalized; not
re-audited here).** `Svd`, `Eigh`, `InvM`, `InvM_`, `Trace`, `Max`, `Min` are all
also `cytnx.linalg` free functions — the correct spelling for functions acting on
an object (cross-ref [UniTensor UT-X1](../UniTensor/08-linalg-operations.md)). The
**members** here are the N-casing offenders (T-X1); the free functions **stay
Capitalized**. Probe: *"the capitalized name `Svd` ALSO exists as a cytnx.linalg
FREE function (which correctly STAYS Capitalized)"* (one per name).

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `leak`. Each
member is a direct `.def` pass-through to the C++ `Tensor` method (which forwards
to `cytnx::linalg::`); the sole exception is `InvM_`, exposed as the leaked raw
`cInvM_` **plus** a conti.py wrapper.

| C++ (`Tensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `std::vector<Tensor> Svd(const bool &is_UvT=true) const` (`hpp:1615`) | `Svd(is_UvT=True)` (`:1781`) | identical | returns `[S, U, vT]` S-first; flag-dependent arity (T-X2) |
| `std::vector<Tensor> Eigh(const bool &is_V=true, const bool &row_v=false) const` (`hpp:1622`) | `Eigh(is_V=True, row_v=False)` (`:1782`) | identical | returns `[eigvals, (eigvecs)]` (T-X2) |
| `Tensor InvM() const` (`hpp:1634`) | `InvM()` (`:1784`) | identical | **matrix** inverse, pure (T-X3) |
| `Tensor &InvM_()` (`hpp:1628`) | `cInvM_` (`:1783`) **+** conti.py `InvM_` (`:99-101`) | **leak + wrapper** | raw `InvM_` bound as `cInvM_`; public `InvM_` wraps it and `return self` (T-X4) |
| `Tensor Trace(const cytnx_uint64 &a=0, const cytnx_uint64 &b=1) const` (`hpp:1702`) | `Trace(arg0, arg1)` (`:1798`) | **signature-differs** | C++ defaults `a=0,b=1` dropped; names → `arg0`/`arg1` (T-X5) |
| `Tensor Max() const` (`hpp:1720`) | `Max()` (`:1795`) | identical | reduction → scalar Tensor |
| `Tensor Min() const` (`hpp:1726`) | `Min()` (`:1796`) | identical | reduction → scalar Tensor |
| raw `cInvM_` (`&Tensor::InvM_`) | same name | **leak** | plumbing exposed publicly (T-X4) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/Tensor_06_linalg.py` (on the 1.1.0 wheel). **There is no
binding-fidelity finding in this category:** the members are direct pass-through
pybind `.def`s over the C++ `Tensor` methods, and the single conti.py wrapper
(`InvM_`) is the in-place-returns-self mechanism already C++-verified in cat 05
(finding T-A4) — gate 4 is recorded as *"no binding-fidelity finding"*. Source
`file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-X1** | `Svd`/`Eigh`/`InvM`/`InvM_`/`Trace`/`Max`/`Min` are **Capitalized member** methods | naming (N-casing) | **Capitalized member spellings**: every one is `.def`-ed with an initial capital (`:1781-1798`), violating the members-are-lowercase rule; the lowercase spellings (`svd`/`eigh`/`inv_m`/`trace`/`max`/`min`) are unbound. The **same names also exist as `cytnx.linalg` FREE functions**, which **stay Capitalized** (they act on objects — cross-ref [UniTensor UT-X1](../UniTensor/08-linalg-operations.md)). Py probe *"capitalized member `Svd` exists on Tensor (N-casing: should be lowercase member `svd`)"* (one per name) + *"the capitalized name `Svd` ALSO exists as a cytnx.linalg FREE function (which correctly STAYS Capitalized)"* + *"the lowercase member `svd` is NOT bound today"* | **rename the members to lowercase**: `Svd`→`svd`, `Eigh`→`eigh`, `InvM`→`inv_m`, `InvM_`→`inv_m_`, `Trace`→`trace`, `Max`→`max`, `Min`→`min` (`InvM`→`inv_m`/N3: same word, casing only). **Keep the `cytnx.linalg.Svd`/`Eigh`/`InvM`/… free functions Capitalized** — do **not** snake_case them (cross-ref UniTensor UT-X1). *Migration:* keep each old Capitalized member as a `DeprecationWarning` alias for one minor release, then delete. Cross-ref cat 05 T-A3. |
| **T-X2** | the decompositions return a **flag-dependent positional `list[Tensor]`** whose length and slot meaning depend on the toggle — `Svd → [S, U, vT]` (S **first**), `Eigh → [eigvals, (eigvecs)]` | ordering / documentation | **flag-dependent positional unpacking is error-prone, and the S-first order diverges from numpy.** `Svd(is_UvT=True)` returns length 3 in the order `[S, U, vT]` (numpy/scipy return `U, S, Vh`); `is_UvT=False` returns length 1 `[S]`. `Eigh(is_V=True)` returns `[eigvals, eigvecs]` (length 2); `is_V=False` → `[eigvals]`. The `[S, U, vT]` slot assignment is **value-verified** (reconstruction), not inferred — correcting v1's `[U, S, vT]` claim. Py probe *"`Svd()` returns a length-3 list [S, U, vT] …"* + *"`Svd` order VALUE-VERIFIED: M ~= U . diag(S) . vT reconstructs the input …"* + *"`Svd(is_UvT=False)` returns a length-1 list [S] …"* + *"`Eigh()` returns a length-2 list [eigvals, eigvecs] …"* + *"`Eigh(is_V=False)` returns a length-1 list [eigvals] …"* | **return a named-result object** (`SvdResult(S, U, vT)` / `EighResult(e, v=None)` `NamedTuple`) so fields are accessed by name and the optional `U`/`vT`/`v` become named attributes rather than a positional slot whose presence shifts the tuple length; **or**, if the positional list is kept, **document the exact order and the flag-conditional length exhaustively**, and note the S-first divergence from numpy's `U, S, Vh`. Cross-ref [UniTensor UT-X3](../UniTensor/08-linalg-operations.md). |
| **T-X3** | `InvM` (**matrix** inverse) and `Inv` (the **element-wise** reciprocal `1/x`, cat 05) are a **near-name collision** for two entirely different operations | **naming (correctness-risk)** | **one letter separates two very different results.** `InvM()` (`hpp:1634`) computes the matrix inverse (value-verified `M @ InvM(M) ≈ I`); `Inv(clip)` (cat 05) maps each element to `1/x`. On the same non-diagonal `B=[[1,2],[3,4]]` they differ: `InvM(B)[0,1] = 1.0` but `Inv(B)[0,1] = 1/2 = 0.5`. Py probe *"`InvM` is the MATRIX inverse: B @ InvM(B) ~= I (value-verified)"* + *"`InvM` (matrix inverse) and `Inv` (element-wise reciprocal, cat 05) give DIFFERENT results …"* | **disambiguate the two concepts by name.** Keep the matrix inverse as `inv_m`/`inv_m_` (the matrix concept, T-X1), and rename the **element-wise** `Inv`/`Inv_`→`reciprocal`/`reciprocal_` (cat 05 T-A5) — one clear name per concept. The corresponding **free** functions do the same: `linalg.InvM` stays the matrix inverse, `linalg.Inv`→`Reciprocal` (cross-ref [UniTensor UT-X4](../UniTensor/08-linalg-operations.md)). *Migration:* the element-wise `Inv`/`Inv_` remain `DeprecationWarning` aliases for one release (cat 05). |
| **T-X4** | `InvM_` is a **conti.py wrapper over the leaked raw `cInvM_` binding** | naming + **binding fidelity (cross-ref cat 05 T-A4)** | **binding exposes plumbing + wraps it**: the raw C++ `InvM_` (returning `Tensor&`) is bound as `cInvM_` (`:1783`), and `Tensor_conti.py:99-101` defines `InvM_` as `self.cInvM_(); return self` — mutating in place and returning self (value-verified). The raw `cInvM_` self-aliases and **leaks** into `dir(Tensor)`. This is the identical mechanism as the cat-05 in-place element-wise family, whose raw-C++ `Tensor&` return is verified in `probes/cpp/Tensor_05_arithmetic.cpp` (T-A4); no new C++ probe is needed. Py probe *"`InvM_()` returns SELF (conti.py wrapper) …"* + *"the raw plumbing binding `cInvM_` LEAKS into public dir(Tensor) …"* + *"`cInvM_()` (raw primitive) mutates the receiver in place (self-aliasing) …"* | **remove `cInvM_` from the public API** (bind under a leading `_` or inline into the pybind lambda) and have the `inv_m_` pybind lambda **return self directly**, dropping the conti.py `return self` shim. Cross-ref cat 05 T-A4/T-A9. |
| **T-X5** | `Trace` **drops the C++ default arguments** (`a=0, b=1`) — `Trace()` raises `TypeError` and the axes become positional-only `arg0`/`arg1` | **binding fidelity / signature-parity (v1 P2)** | **binding drops defaults + erases names**: C++ `Trace(a=0, b=1)` (`hpp:1702`) is bound as `.def("Trace", &cytnx::Tensor::Trace)` with **no `py::arg` defaults** (`:1798`), so Python `Trace()` raises `TypeError` (two positional args required) and the meaningful names `a`/`b` surface as `arg0`/`arg1` (N4 loss). Py probe *"`Trace()` with NO args raises TypeError — the C++ defaults (a=0, b=1) were dropped …"* + *"`Trace(0, 1)` sums the diagonal … == 12"* | **restore the defaults and give the axes names**: bind `trace(axis_a=0, axis_b=1)` with `py::arg("axis_a")=0, py::arg("axis_b")=1` so the no-arg call traces axes 0/1 as C++ intends and the parameters are keyword-callable. *Migration:* additive (restores dropped behavior); combine with the `Trace`→`trace` rename (T-X1). |

## A4. Argument ordering — positional & keyword

Every member here takes at most two flags/axes; there is no keyword-only metadata
block. The primary operand is the receiver (`self`).

| API | positional-required (in order) | keyword / optional |
|---|---|---|
| `Svd` | *(none)* | `is_UvT=True` |
| `Eigh` | *(none)* | `is_V=True`, `row_v=False` |
| `InvM` / `InvM_` | *(none)* | *(none)* |
| `Trace` | `arg0`, `arg1` (axes; **defaults dropped** — T-X5) | *(should be)* `axis_a=0`, `axis_b=1` |
| `Max` / `Min` | *(none)* | *(none)* |

- **Canonical positional rule (§R.0):** the receiver is the operand; the toggle
  flags (`is_UvT`/`is_V`/`row_v`) and the trace axes are operation parameters —
  matching the live order; no reordering needed.
- **`Trace` axes (T-X5):** must regain their C++ defaults (`0`/`1`) and meaningful
  names (`axis_a`/`axis_b`) — the sole ordering/naming defect in this category.
- **SVD toggle casing:** `is_UvT`/`is_V` are non-snake; snake_case to `is_uvt`/
  `is_v` in step (cross-ref UniTensor UT-X2, which unifies the SVD-family toggles).

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
member linear-algebra / reduction surface. Implement Cytnx to match it. Findings
above are the rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing (SciPostPhysCodeb.53).** These linear-algebra / reduction *members*
  are lowercase snake_case: the Capitalized `Svd`/`Eigh`/`InvM`/`InvM_`/`Trace`/
  `Max`/`Min` → `svd`/`eigh`/`inv_m`/`inv_m_`/`trace`/`max`/`min` (T-X1). The
  **free** functions `linalg.Svd`/`Eigh`/`InvM`/`InvM_`/`Trace`/`Max`/`Min`
  **stay Capitalized** — they act on objects (the **positive** demonstration of
  the same rule; cross-ref UniTensor UT-X1). Members lowercase, free functions
  Capitalized, by design.
- **N-underscore — a trailing `_` marks in-place (returns `self`).** `inv_m` is
  pure (new object), `inv_m_` inverts in place and returns self. The **`c`-prefixed
  raw spelling `cInvM_` is rejected** as public API — it is the plumbing the
  wrapper calls (T-X4).
- **In-place methods return `self` from the binding directly.** C++ `InvM_` returns
  `Tensor&` (the cat-05 T-A4 pattern); the `inv_m_` pybind lambda must return
  `&self` too, so the conti.py return-self shim and the leaked `cInvM_` disappear.
- **Decomposition results are named, not flag-dependent positional lists.** Return
  `SvdResult(S, U, vT)` / `EighResult(e, v)` `NamedTuple`s so optional outputs are
  named attributes, not tuple slots whose presence shifts the length (T-X2). If the
  positional list is retained, the order (`[S, U, vT]` — **S first**, unlike numpy)
  and flag-conditional length are documented exhaustively.
- **Two "inverse" concepts get one name each.** `inv_m`/`inv_m_` is the **matrix**
  inverse; the **element-wise** reciprocal is `reciprocal`/`reciprocal_` (renamed
  from `Inv`/`Inv_`, cat 05 T-A5) — no `Inv`/`InvM` near-name collision (T-X3).
- **`trace` restores its defaults.** `trace(axis_a=0, axis_b=1)` — the C++ defaults
  and names are re-exposed (T-X5); a bare `trace()` traces axes 0/1.

## R.1 Recommended API (exact signatures + behavior contract)

```python
from typing import NamedTuple

class SvdResult(NamedTuple):
    S: "Tensor"; U: "Tensor | None" = None; vT: "Tensor | None" = None
class EighResult(NamedTuple):
    e: "Tensor"; v: "Tensor | None" = None

class Tensor:
    # --- decompositions (named results; renamed from Svd/Eigh) ---
    def svd(self, is_uvt: bool = True) -> SvdResult: ...       # was Svd; [S, U, vT] S-first
    def eigh(self, is_v: bool = True, row_v: bool = False) -> EighResult: ...  # was Eigh

    # --- matrix inverse (pure + in-place; renamed from InvM/InvM_) ---
    def inv_m(self) -> "Tensor": ...      # was InvM; MATRIX inverse (new object)
    def inv_m_(self) -> "Tensor": ...     # was InvM_; MATRIX inverse in place, self

    # --- reductions (renamed from Trace/Max/Min) ---
    def trace(self, axis_a: int = 0, axis_b: int = 1) -> "Tensor": ...  # was Trace; defaults RESTORED
    def max(self) -> "Tensor": ...        # was Max; scalar Tensor
    def min(self) -> "Tensor": ...        # was Min; scalar Tensor
```

`inv_m_` returns `self` **from the binding**; the raw `cInvM_` plumbing becomes
private (leading `_`) or is inlined into the `inv_m_` pybind lambda — it is **not**
a public member. The Capitalized `cytnx.linalg.Svd`/`Eigh`/`InvM`/… free functions
are unchanged (Capitalized, reached from Python as free functions).

| API | Verdict | Behavior contract |
|---|---|---|
| `Svd` → `svd` | **rename** (T-X1/X2) | Singular-value decomposition of a rank-2 tensor; returns `SvdResult(S, U, vT)` (S-first — value-verified), or `SvdResult(S)` when `is_uvt=False`. *Migration:* `DeprecationWarning` alias `Svd` for one release; return a named result (positional list aliased for one release). |
| `Eigh` → `eigh` | **rename** (T-X1/X2) | Hermitian eigendecomposition; `EighResult(e, v)` (v set when `is_v`). *Migration:* alias `Eigh`; named result. |
| `InvM` → `inv_m` | **rename** (T-X1/X3) | **Matrix** inverse of a square rank-2 tensor (pure); value-verified `M @ inv_m(M) ≈ I`. Disambiguated by name from the element-wise `reciprocal` (was `Inv`, cat 05). *Migration:* alias `InvM`. |
| `InvM_` → `inv_m_` | **rename, bind self directly** (T-X1/X3/X4) | Matrix inverse **in place**; returns self. *Migration:* fold the raw `cInvM_` into the `inv_m_` lambda (drop the conti.py shim); alias `InvM_`. |
| `Trace` → `trace` | **rename + restore defaults** (T-X1/X5) | Sum of the diagonal over `axis_a`,`axis_b`; **restore** the C++ defaults (`axis_a=0, axis_b=1`) and names so a bare `trace()` works. *Migration:* alias `Trace`; additive default restoration. |
| `Max` → `max` | **rename** (T-X1) | Maximum element as a scalar Tensor. *Migration:* alias `Max`. |
| `Min` → `min` | **rename** (T-X1) | Minimum element as a scalar Tensor. *Migration:* alias `Min`. |

**Internal / plumbing — hidden, not public API.** The raw binding below is a live
public member today with a **remove** verdict: hide it behind a leading underscore
or inline it into its pybind lambda. It carries no docstring — it is not public
surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `cInvM_` | **remove** (T-X4) | Raw plumbing (C++ `InvM_`, returns `Tensor&`) behind `InvM_`/`inv_m_`. *Migration:* fold into the `inv_m_` pybind lambda (which returns self); no public exposure. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python surface,
**R.2b** Doxygen for the C++ surface. Only kept/renamed members are documented
(removed plumbing carries no docstring). The Capitalized `cytnx.linalg` free
functions of the same names are documented in the future `linalg` plan.

### R.2a Python API (numpy-style)

### `svd` / `eigh` (renamed from `Svd` / `Eigh`)

```
Tensor.svd(is_uvt=True)            -> SvdResult(S, U, vT)     # renamed from Svd
Tensor.eigh(is_v=True, row_v=False) -> EighResult(e, v)       # renamed from Eigh

Matrix decompositions of a rank-2 Tensor.

`svd` factors M = U @ diag(S) @ vT and returns the factors S-FIRST as
SvdResult(S, U, vT): `S` is a rank-1 vector of singular values, `U`/`vT` the
isometries. `is_uvt=False` returns SvdResult(S) only. `eigh` (Hermitian) returns
the eigenvalues `e` and, when `is_v`, the eigenvectors `v` (as rows if `row_v`).

Parameters
----------
is_uvt : bool, optional
    (`svd`) also compute U and vT (default True).
is_v : bool, optional
    (`eigh`) also compute eigenvectors (default True).
row_v : bool, optional
    (`eigh`) return eigenvectors as rows (default False).

Returns
-------
SvdResult / EighResult
    Named results with fields (`S`, `U`, `vT`) / (`e`, `v`).

Notes
-----
Renamed from the Capitalized `Svd`/`Eigh` (finding T-X1); the Capitalized *free*
functions `cytnx.linalg.Svd`/`Eigh` stay Capitalized (cross-ref UniTensor UT-X1).
Through cytnx 1.1.0 the result was a flag-dependent positional `list[Tensor]` in
the order `[S, U, vT, ...]` (finding T-X2) — S is FIRST, unlike numpy's `U, S, Vh`;
the next version returns a named result. `Svd`/`Eigh` remain `DeprecationWarning`
aliases for one release.
```

### `inv_m` / `inv_m_` (renamed from `InvM` / `InvM_`)

```
Tensor.inv_m()    -> Tensor    # MATRIX inverse (pure)     (renamed from InvM)
Tensor.inv_m_()   -> Tensor    # MATRIX inverse in place, self (renamed from InvM_)

Matrix inverse of a square rank-2 Tensor.

`inv_m` returns the inverse as a NEW tensor (value-verified M @ inv_m(M) ~= I);
`inv_m_` inverts IN PLACE and returns self.

Returns
-------
Tensor
    `inv_m`: a new tensor. `inv_m_`: self.

Notes
-----
This is the MATRIX inverse — NOT the ELEMENT-WISE reciprocal `reciprocal` (renamed
from `Inv`, cat 05), with which its former name `InvM` was a near-name collision
(finding T-X3). Renamed from the Capitalized `InvM`/`InvM_` (finding T-X1); the
in-place form no longer routes through the leaked raw `cInvM_` binding (finding
T-X4). `InvM`/`InvM_` remain `DeprecationWarning` aliases for one release.
```

### `trace` (renamed from `Trace`)

```
Tensor.trace(axis_a=0, axis_b=1)   -> Tensor    # renamed from Trace; DEFAULTS RESTORED

Trace (sum of the diagonal) over two axes.

Parameters
----------
axis_a : int, optional
    First axis (default 0).
axis_b : int, optional
    Second axis (default 1).

Returns
-------
Tensor
    The input with `axis_a`,`axis_b` contracted (rank-0 for a rank-2 input;
    value-verified: trace(arange(9).reshape(3,3)) over (0,1) == 12).

Notes
-----
Renamed from the Capitalized `Trace` (finding T-X1). The recommended binding
RESTORES the C++ default arguments (a=0, b=1) and gives the axes meaningful names
— through cytnx 1.1.0 the binding dropped the defaults, so `Trace()` raised
TypeError and the axes were positional-only `arg0`/`arg1` (finding T-X5). `Trace`
remains a `DeprecationWarning` alias for one release.
```

### `max` / `min` (renamed from `Max` / `Min`)

```
Tensor.max()   -> Tensor    # maximum element  (renamed from Max)
Tensor.min()   -> Tensor    # minimum element  (renamed from Min)

The maximum / minimum element of the tensor, as a scalar Tensor.

Returns
-------
Tensor
    A scalar tensor holding the extremum (value-verified on arange(5): max == 4,
    min == 0).

Notes
-----
Renamed from the Capitalized `Max`/`Min` (finding T-X1); the Capitalized *free*
functions `cytnx.linalg.Max`/`Min` stay Capitalized (cross-ref UniTensor UT-X1).
`Max`/`Min` remain `DeprecationWarning` aliases for one release.
```

### R.2b C++ API (Doxygen)

The C++ methods already return `std::vector<Tensor>`/`Tensor`/`Tensor&` per the
N-underscore split (`InvM_` returns `Tensor&` — the cat-05 T-A4 pattern); the next
version's changes are all in the *pybind layer*: lowercase the member names (T-X1),
return named result structs from the decompositions (T-X2), fold the raw `cInvM_`
into the `inv_m_` lambda so it returns `*this` directly (T-X4), and **restore the
`Trace` defaults + names** `py::arg("axis_a")=0, py::arg("axis_b")=1` (T-X5). The
C++ method names keep their capitalization; the `cytnx.linalg` free functions of
the same names likewise stay Capitalized.

```cpp
/**
 * @brief Singular-value / Hermitian-eigen decomposition of a rank-2 Tensor.
 * @details Svd factors M = U diag(S) vT and returns the factors S-FIRST (S a
 *          rank-1 vector) — value-verified by reconstruction (finding T-X2). Eigh
 *          returns {eigvals, (eigvecs)}. The Python binding exposes these as
 *          svd/eigh returning NAMED results (SvdResult/EighResult) rather than a
 *          flag-dependent vector (findings T-X1/X2). Same as the cytnx::linalg::
 *          free functions (which stay Capitalized).
 * @param is_UvT (Svd) whether to also compute U and vT.
 * @param is_V,row_v (Eigh) whether to compute eigenvectors, and their layout.
 * @return the decomposition factors as a vector<Tensor> (S-first for Svd).
 */
std::vector<Tensor> Svd(const bool &is_UvT = true) const;
std::vector<Tensor> Eigh(const bool &is_V = true, const bool &row_v = false) const;

/**
 * @brief Matrix inverse of a square rank-2 Tensor (pure and in-place).
 * @details InvM returns the MATRIX inverse as a NEW Tensor (value-verified
 *          M InvM(M) = I); InvM_ inverts in place and returns *this (the cat-05
 *          T-A4 in-place-returns-self pattern). The Python binding exposes these
 *          as inv_m/inv_m_, returning *this directly from the in-place lambda
 *          (dropping the leaked cInvM_ binding, finding T-X4). NOT the element-wise
 *          reciprocal Inv/reciprocal (finding T-X3).
 * @return InvM: a new Tensor. InvM_: reference to *this.
 */
Tensor InvM() const;
Tensor &InvM_();

/**
 * @brief Trace and element reductions of a Tensor.
 * @details Trace sums the diagonal over axes a,b (C++ defaults a=0,b=1 — the
 *          Python binding must RESTORE these and name them axis_a/axis_b, finding
 *          T-X5). Max/Min return the extremal element as a scalar Tensor. Python
 *          exposes trace/max/min (finding T-X1).
 * @param a,b (Trace) the two axes to contract.
 * @return Trace: the contracted Tensor. Max/Min: a scalar Tensor.
 */
Tensor Trace(const cytnx_uint64 &a = 0, const cytnx_uint64 &b = 1) const;
Tensor Max() const;
Tensor Min() const;
```
