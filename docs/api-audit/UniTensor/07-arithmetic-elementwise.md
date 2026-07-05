# UniTensor — 07. Arithmetic & element-wise

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the siblings
> [`05-structure-manipulation.md`](05-structure-manipulation.md),
> [`06-element-block-access.md`](06-element-block-access.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's arithmetic/element-wise API to match §R exactly. All runtime claims
> verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_07_arithmetic.py` (all `[PASS]`, exit 0), with
> the raw-C++ side of the binding-fidelity findings verified by
> `probes/cpp/UniTensor_07_arithmetic.cpp` against a source-built `libcytnx` (all
> `[PASS]`, exit 0).

**Category scope:** the members that combine tensors element-wise or apply a
scalar element-wise map — the Python operator dunders (`__add__`/`__radd__`/
`__iadd__` and the `-`/`*`/`/` families, `__floordiv__`/`__neg__`/`__pos__`/
`__pow__`/`__ipow__`), and the named element-wise methods `Pow`/`Pow_`, `Inv`,
`Conj`/`Conj_`, `Transpose`/`Transpose_`, `Dagger`/`Dagger_`, `normalize`/
`normalize_`, `Trace`/`Trace_`, `Norm` — plus the leaked raw `c*` bindings
(`cConj_`/`cDagger_`/`cPow_`/`cTrace_`/`cTranspose_`/`cnormalize_`/`cInv_`/
`c__ipow__`) the conti.py in-place wrappers call. Python bindings:
`cytnx_src/pybind/unitensor_py.cpp:765-1362,1365-1413`; conti.py wrappers:
`cytnx/UniTensor_conti.py:128-170`; C++ header:
`cytnx_src/include/UniTensor.hpp:5040,4928-5030,5057-5130,507-533,542-725,5442-5460`;
free-function modulo: `cytnx_src/include/linalg.hpp:175,625` and the stub
`cytnx_src/src/linalg/Mod.cpp:1029`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place. Operator families whose
overloads differ only in the right-hand-operand type (UniTensor vs the 11 scalar
dtypes) share one row.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `__add__` / `__radd__` | `__add__(rhs)` (UniTensor or scalar) | `UniTensor` (new) | **Pure** element-wise add; routes to `linalg::Add`. Probe: *"`__add__` is pure: a + a leaves a unchanged and returns a new tensor …"*. |
| `__sub__` / `__rsub__` | `__sub__(rhs)` | `UniTensor` (new) | Pure element-wise subtract (`linalg::Sub`). |
| `__mul__` / `__rmul__` | `__mul__(rhs)` | `UniTensor` (new) | Pure element-wise multiply (`linalg::Mul`). |
| `__truediv__` / `__rtruediv__` | `__truediv__(rhs)` | `UniTensor` (new) | Pure element-wise divide (`linalg::Div`). |
| `__floordiv__` / `__rfloordiv__` | `__floordiv__(rhs)` | `UniTensor` (new) | **CORRECTNESS BUG** — routes to `Div`, so `//` performs **true** division, *not* floor: `(u*7)//2 → 3.5`. Probe: *"`//` (__floordiv__) performs TRUE division, not floor … yields 3.5 …"* + *"`//` with a UniTensor divisor is also true division …"*. |
| `__iadd__` / `__isub__` / `__imul__` / `__itruediv__` / `__ifloordiv__` `[I]` | `__iadd__(rhs)` … | `UniTensor` (in place; **new wrapper**) | In-place; routes to `Add_`/`Sub_`/`Mul_`/`Div_`. Mutates the receiver, but the binding returns a **new Python wrapper** (identity dropped). `//=` is true-division-in-place. Probe: *"`__iadd__` mutates the receiver in place … but the binding returns a NEW wrapper — identity is dropped …"* + *"`//=` … is likewise true division in place …"*. |
| `__neg__` | `__neg__()` | `UniTensor` (new) | Element-wise negation; routes to `linalg::Mul(-1, self)`. Probe: *"`__neg__` negates element-wise (routes to Mul(-1)): -5 -> -5.0"*. |
| `__pos__` | `__pos__()` | `UniTensor` (**shared data**) | Returns the tensor unchanged (`return self` by value → a shared-data wrapper). Probe: *"`__pos__` returns the tensor unchanged, SHARING data …"*. |
| `__pow__` | `__pow__(p: float)` | `UniTensor` (new) | Pure element-wise power; wraps `Pow`. Probe: *"operator dunder `__pow__` is bound"*. |
| `__ipow__` `[I]` | `__ipow__(p: float)` | `UniTensor` (self) | In-place power; conti.py wrapper over the leaked raw `c__ipow__` (which wraps `Pow_`), returns self. Probe: *"`**=` (__ipow__) raises in place … 3 **= 2 leaves 9"*. |
| **`__mod__` / `__rmod__`** | *(absent)* | — | **ABSENT** — the pybind `__mod__`/`__rmod__` block is **commented out** (`:1311-1362`). Probe: *"`%` is absent: … hasattr(UniTensor, '__mod__') is False"* + *"using `%` … raises TypeError …"*. |
| `Pow` | `Pow(p: float)` | `UniTensor` (new) | Capitalized member — pure element-wise power. Probe: *"capitalized member `Pow` exists …"*. |
| `Pow_` `[I]` | `Pow_(p: float)` | `UniTensor` (self) | Capitalized member — in-place power; conti.py wrapper over leaked `cPow_`; returns self. Probe: *"Pow_(2.0) returns self and squares in place (3->9)"*. |
| `Inv` | `Inv(clip=-1)` | `UniTensor` (new) | Capitalized member — **element-wise** inverse (`1/x`, with `clip` guarding small values). *No public in-place `Inv_`.* Probe: *"`Inv` (pure element-wise inverse) is a public member"*. |
| `Conj` | `Conj()` | `UniTensor` (new) | Capitalized member — pure complex conjugate. Probe: *"capitalized member `Conj` exists …"*. |
| `Conj_` `[I]` | `Conj_()` | `UniTensor` (self) | Capitalized member — in-place conjugate; conti.py wrapper over leaked `cConj_`; returns self. Probe: *"Conj_() returns self (in place)"*. |
| `Transpose` | `Transpose()` | `UniTensor` (new) | Capitalized member — pure transpose (bra↔ket). Probe: *"capitalized member `Transpose` exists …"*. |
| `Transpose_` `[I]` | `Transpose_()` | `UniTensor` (self) | In-place transpose; conti.py wrapper over leaked `cTranspose_`; returns self. Probe: *"Transpose_() returns self (in place)"*. |
| `Dagger` | `Dagger()` | `UniTensor` (new) | Capitalized member — pure adjoint (conjugate + transpose). Probe: *"capitalized member `Dagger` exists …"*. |
| `Dagger_` `[I]` | `Dagger_()` | `UniTensor` (self) | In-place adjoint; conti.py wrapper over leaked `cDagger_`; returns self. Probe: *"Dagger_() returns self (in place)"*. |
| `normalize` | `normalize()` | `UniTensor` (new) | **Already lowercase** — pure 2-norm normalization. Probe: *"`normalize`/`normalize_` are already lowercase …"*. |
| `normalize_` `[I]` | `normalize_()` | `UniTensor` (self) | In-place normalize; conti.py wrapper over leaked `cnormalize_`; returns self. Probe: *"normalize_() returns self (in place)"*. |
| `Trace` | `Trace(a=0, b=1)` (int) · `Trace(a, b)` (str) | `UniTensor` (new) | Capitalized member — pure trace over legs `a`,`b`. Probe: *"capitalized member `Trace` exists …"*. |
| `Trace_` `[I]` | `Trace_(a=0, b=1)` (int) · `Trace_(a, b)` (str) | `UniTensor` (self) | In-place trace; conti.py wrapper over leaked `cTrace_`; returns self. Probe: *"Trace_(0,1) returns self (in place)"*. |
| `Norm` | `Norm()` | **`Tensor`** | Capitalized member — the 2-norm as a scalar `Tensor` (**not** a UniTensor). Probe: *"`Norm()` returns a cytnx.Tensor (the 2-norm), not a UniTensor"*. |

**Internal / plumbing (leak into `dir(UniTensor)`):** `cConj_`, `cDagger_`,
`cPow_`, `cTrace_`, `cTranspose_`, `cnormalize_`, `cInv_`, `c__ipow__` — the raw
pybind bindings the conti.py in-place wrappers call (`:1366-1412`); `cInv_` is
the raw in-place inverse with no public `Inv_` counterpart. Public today, but
should never be. Probe: *"the raw plumbing binding `cConj_` LEAKS into public
dir(UniTensor) …"* (and one per name).

**C++-only (present in C++, *not* bound as named Python members):** the arithmetic
methods `Add`/`Sub`/`Mul`/`Div` (and their `Add_`/`Sub_`/`Mul_`/`Div_` in-place
forms) — reachable from Python only through the operator dunders. Probe: *"named
method `Add` is NOT a public Python member …"* (and one per name).

## A2. C++ ↔ Python mapping

| C++ (`UniTensor.hpp` / `linalg.hpp`) | Python | Status | Note |
|---|---|---|---|
| `UniTensor Add/Sub/Mul/Div(...) const` (`hpp:4928-5030`) | `__add__`/`__sub__`/`__mul__`/`__truediv__` (lambdas `:780,…,1087`) | identical | dunders route to `linalg::Add/Sub/Mul/Div`; named `Add/…` unbound (UT-A6) |
| `UniTensor &Add_/Sub_/Mul_/Div_(...)`, `operator+=/…` (`hpp:5057-5130`) | `__iadd__`/`__isub__`/`__imul__`/`__itruediv__` (`:853,955,…,1160`) | **signature-differs** | lambdas return the `UniTensor&` **by value** → a new wrapper, dropping identity (UT-A7) |
| `UniTensor Div(...) const` (`hpp:4988`) | `__floordiv__` (lambda `:1199-1234`, calls `Div` at `:1223`) | **binding fidelity** | `//` is wired to true division — a correctness bug (UT-A1) |
| `linalg::operator%` / `linalg::Mod(UniTensor, …)` (`linalg.hpp:175,625`) | *(absent — pybind block commented out `:1311-1362`)* | **binding gap** | `%` unbound; scalar C++ Mod works, tensor⊗tensor Mod is a `[Mod][Developing]` stub (UT-A2) |
| `linalg::Mul(-1, self)` | `__neg__` (`:765`) | identical | element-wise negation (UT-A7) |
| `UniTensor Pow(double) const` (`hpp:5442`) | `__pow__` (`:1365`), `Pow` (`:1367`) | identical | pure power (UT-A3) |
| `UniTensor &Pow_(double)` (`hpp:5460`) | `cPow_` (`:1368`) **+** conti.py `Pow_`/`__ipow__` (`conti.py:164,168`) | **leak** | raw `Pow_` bound as `cPow_`/`c__ipow__`; public `Pow_`/`__ipow__` wrap them (UT-A4) |
| `UniTensor Inv(double clip) const` (`hpp:507`) | `Inv` (`:1374`) | identical | element-wise inverse (UT-A3) |
| `UniTensor &Inv_(double clip)` (`hpp:533`) | `cInv_` (`:1370`) — **no public `Inv_`** | **leak + gap** | raw in-place inverse leaks as `cInv_`; no public `Inv_` (UT-A5) |
| `UniTensor Conj() const` (`hpp:542`) | `Conj` (`:1379`) | identical | pure conjugate (UT-A3) |
| `UniTensor &Conj_()` (`hpp:555`) | `cConj_` (`:1378`) **+** conti.py `Conj_` (`conti.py:128`) | **leak** | raw `Conj_` bound as `cConj_`; public `Conj_` wraps it (UT-A4) |
| `UniTensor Transpose() const` (`hpp:573`) | `Transpose` (`:1408`) | identical | pure transpose (UT-A3) |
| `UniTensor &Transpose_()` (`hpp:586`) | `cTranspose_` (`:1407`) **+** conti.py `Transpose_` (`conti.py:144`) | **leak** | raw `Transpose_` bound as `cTranspose_` (UT-A4) |
| `UniTensor normalize() const` (`hpp:597`) | `normalize` (`:1410`) | identical | pure normalize (UT-A3; already lowercase) |
| `UniTensor &normalize_()` (`hpp:609`) | `cnormalize_` (`:1409`) **+** conti.py `normalize_` (`conti.py:149`) | **leak** | raw `normalize_` bound as `cnormalize_` (UT-A4) |
| `UniTensor Dagger() const` (`hpp:696`) | `Dagger` (`:1413`) | identical | pure adjoint (UT-A3) |
| `UniTensor &Dagger_()` (`hpp:710`) | `cDagger_` (`:1412`) **+** conti.py `Dagger_` (`conti.py:154`) | **leak** | raw `Dagger_` bound as `cDagger_` (UT-A4) |
| `UniTensor Trace(a,b) const` (`hpp:623,638`) | `Trace` (`:1394,1401`) | identical | pure trace (UT-A3) |
| `UniTensor &Trace_(a,b)` (`hpp:653,675`) | `cTrace_` (`:1381,1388`) **+** conti.py `Trace_` (`conti.py:133,138`) | **leak** | raw `Trace_` bound as `cTrace_` (UT-A4) |
| `Tensor Norm() const` (`hpp:5040`) | `Norm` (`:1406`) | identical | returns a scalar `Tensor` (UT-A3) |
| raw `cConj_`/`cDagger_`/`cPow_`/`cTrace_`/`cTranspose_`/`cnormalize_`/`cInv_`/`c__ipow__` | same names | **leak** | plumbing exposed publicly (UT-A4/A5) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/UniTensor_07_arithmetic.py` (on the 1.1.0 wheel). A
**(binding fidelity)** finding flags where the binding layer — a `*_conti.py`
wrapper or a pybind lambda — changes behavior, signature, or availability versus
the raw C++ method; **both sides are runtime-verified**, the raw-C++ side by
`probes/cpp/UniTensor_07_arithmetic.cpp` (links against a source-built
`libcytnx`, GCC 13). Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-A1** | `//` (`__floordiv__`) performs **true** division, not floor: `(u*7)//2 → 3.5` | **correctness** | **binding aliases the wrong op**: the `__floordiv__` pybind lambda calls `self.Div(rhs)` (`:1223`) — the very same op as `__truediv__` — so `//` never floors. Py probe *"`//` (__floordiv__) performs TRUE division … yields 3.5 …"* + *"`//` with a UniTensor divisor is also true division …"* + *"`//=` … is likewise true division in place …"* | **fix or remove**: either implement floor semantics for `//` (round toward −∞ after the divide), **or remove `__floordiv__`/`__rfloordiv__`/`__ifloordiv__` entirely** — do **not** silently route `//` to true division. A `DeprecationWarning` shim documents the change. |
| **UT-A2** | `%` (`__mod__`/`__rmod__`) is **absent** from Python though C++ has `linalg::Mod`/`operator%` for UniTensor | **binding gap / C++-only** | **binding commented out**: the `__mod__`/`__rmod__` block is disabled in pybind (`:1311-1362`), so `hasattr(UniTensor,"__mod__")` is False and `u % 2` raises `TypeError`. On the C++ side the scalar forms `operator%(UniTensor, scalar)` / `Mod(UniTensor, scalar)` are **implemented** (Dense), while `Mod(UniTensor, UniTensor)` is an **unfinished stub** — its body is `cytnx_error_msg(true,"[Mod][Developing]")` (`Mod.cpp:1029`). Py probe *"`%` is absent … hasattr(UniTensor, '__mod__') is False"* + *"using `%` … raises TypeError …"*; **C++ probe confirms** `A % 2.0` and `Mod(A, 2.0)` compute `7 % 2 == 1`, while `Mod(A, B)` throws `[Mod][Developing]` | **bind `%`/`__mod__` for the working scalar case** (mirrors `linalg::Mod(UniTensor, scalar)`); **or document the absence** if `%` on tensors is out of scope. The tensor⊗tensor form additionally needs the C++ `Mod` stub finished before it can be bound. |
| **UT-A3** | `Conj`/`Conj_`, `Trace`/`Trace_`, `Norm`, `Pow`/`Pow_`, `Transpose`/`Transpose_`, `Dagger`/`Dagger_` are **Capitalized members** | naming (N-casing) | **Capitalized member spellings**: every one is `.def`-ed with an initial capital (`:1367-1413`), violating the members-are-lowercase rule. The **same names also exist as `linalg` free functions** (`Conj`/`Trace`/`Norm`/`Pow`/`Inv`), which **stay Capitalized** (they act on objects — cat 08). `normalize`/`normalize_` are already correct. Py probe *"capitalized member `Conj` exists … (N-casing: should be lowercase)"* (one per name) + *"the capitalized names ALSO exist as `linalg` FREE functions …"* + *"`Norm()` returns a cytnx.Tensor …"* | **rename to lowercase**: `Conj`→`conj`, `Trace`→`trace`, `Norm`→`norm`, `Pow`→`pow`, `Transpose`→`transpose`, `Dagger`→`dagger` (and their `_` forms). Keep the free-function `linalg.Conj`/… Capitalized (cross-ref cat 08). *Migration:* keep each old Capitalized member as a `DeprecationWarning` alias for one minor release, then delete. |
| **UT-A4** | the in-place `Conj_`/`Trace_`/`Transpose_`/`Dagger_`/`normalize_`/`Pow_`/`__ipow__` are **conti.py wrappers over leaked raw `c*` bindings** | naming + **binding fidelity** | **binding exposes plumbing + wraps it**: raw C++ `Conj_`/`Trace_`/… are bound as `cConj_`/`cTrace_`/`cTranspose_`/`cDagger_`/`cnormalize_`/`cPow_`/`c__ipow__` (`:1366-1412`), and `cytnx/UniTensor_conti.py:128-170` defines each public in-place form as `self.c<Name>(); return self`. The `c`-prefix is a reserved raw-binding spelling (§R.0 rejects it). Py probe *"Conj_() returns self (in place)"* (etc.) + *"the raw plumbing binding `cConj_` LEAKS into public dir(UniTensor) …"* (one per name); **C++ probe confirms** C++ `Conj_()`/`Trace_(0,1)`/`Pow_(2.0)` (and `Transpose_`/`Dagger_`/`normalize_`) each return `&*this` | **remove the `c*` bindings from the public API** (bind under a leading `_` or inline into the pybind lambda) and have the in-place pybind lambda **return self directly** — dropping the conti.py `return self` shims (migration note). |
| **UT-A5** | `Inv` (pure) exists but there is **no public `Inv_`** — only the leaked raw `cInv_` | **redundancy / gap** | **binding exposes only the raw in-place form**: `Inv` is public (`:1374`) but the in-place inverse is bound solely as `cInv_` (`:1370`) — `hasattr(UniTensor,"Inv_")` is False. C++ has `UniTensor &Inv_(double clip)` (`hpp:533`). Py probe *"there is NO public in-place `Inv_` — only the leaked raw `cInv_` …"*; **C++ probe confirms** C++ `Inv_()` inverts in place (`4 → 0.25`) and returns `&*this` | **add a public `inv_`** (lowercase per UT-A3) that returns self, backed by the C++ `Inv_`; **remove `cInv_`** from the public API (migration note). |
| **UT-A6** | the named arithmetic methods `Add`/`Sub`/`Mul`/`Div` (and `Add_`/`Sub_`/`Mul_`/`Div_`) are **C++-only** — unbound as Python members | **binding fidelity / C++-only** | **binding hides the methods behind the operators**: the C++ `Add`/`Sub`/`Mul`/`Div` methods (`hpp:4928-5030`) are absent from `dir(UniTensor)`; only the operator dunders are exposed (they call `linalg::Add`/… `:780,1087`). Py probe *"named method `Add` is NOT a public Python member …"* (one per name) + *"operator dunder `__add__` is bound"* (one per dunder) | **keep the operators as the Python surface**; the named element-wise functions belong in `cytnx.linalg` (Capitalized, cat 08). Document the operator↔free-function equivalence; no separate `Add`/… Python member. |
| **UT-A7** | in-place operators (`__iadd__`/`__isub__`/`__imul__`/`__itruediv__`/`__ifloordiv__`) mutate the receiver but the binding returns a **new wrapper** (identity dropped); `__pos__` returns a **shared-data** handle | **binding fidelity** (N2/B1) | **binding returns by value**: the augmented-assign lambdas return the `UniTensor&` from `Add_`/… by value (`:861,963,…`), so `a += x` mutates in place yet `a is not (the original handle)` afterwards (same pattern as `twist_`/UT-S7); `__pos__` is `return self` by value (`:779`), a shared-data wrapper. `__neg__` routes to `Mul(-1)` (`:765-777`). Py probe *"`__iadd__` mutates the receiver in place … but the binding returns a NEW wrapper — identity is dropped …"* + *"`__pos__` returns the tensor unchanged, SHARING data …"* + *"`__neg__` negates element-wise (routes to Mul(-1)) …"* | **keep**; **have the augmented-assign lambdas return `self` directly** (`return &self.Add_(rhs)`) so `a += x` preserves identity, matching Python's data-model contract for `__iadd__`. |

## A4. Argument ordering — positional & keyword

Every member here takes at most one operand (the right-hand tensor/scalar) or a
single operation parameter (`p` for power, `clip` for inverse, the two leg
selectors for trace); there is no keyword-only metadata block.

| API | positional-required (in order) |
|---|---|
| `__add__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__` (+ `r`/`i` forms) | `rhs` (UniTensor or scalar) |
| `__neg__` / `__pos__` | *(none)* |
| `__pow__` / `__ipow__` / `Pow` / `Pow_` | `p` (exponent) |
| `Inv` (→ `inv`) / `inv_` | *(optional)* `clip` |
| `Conj`/`Conj_` / `Transpose`/`Transpose_` / `Dagger`/`Dagger_` / `normalize`/`normalize_` / `Norm` | *(none)* |
| `Trace` / `Trace_` | `a` (leg), `b` (leg) — default `(0, 1)` for the int overload |

- **Canonical positional rule (§R.0):** the sole operand/parameter is positional;
  this matches the live order and needs no change. `Trace(a=0, b=1)` keeps its
  numpy-like default leg pair.
- **`clip` on `Inv`/`inv_`** is an operation parameter (the small-value guard),
  positional-optional with default `-1`.
- **No metadata block:** none of these methods take the keyword-only
  `labels, rowrank, …` block used by the generators (cat 02).

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version arithmetic/element-wise API.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53).** Element-wise *members* are lowercase
  snake_case. The offenders are the Capitalized `Conj`/`Trace`/`Norm`/`Pow`/
  `Transpose`/`Dagger` (and their `_` forms) → `conj`/`trace`/`norm`/`pow`/
  `transpose`/`dagger` (UT-A3). The **free** functions `linalg.Conj`/`Trace`/
  `Norm`/`Pow`/`Inv` **stay Capitalized** — they act on objects (cat 08).
  `normalize`/`normalize_` are already conformant.
- **N-underscore — a trailing `_` marks in-place (returns `self`); its absence
  marks pure (returns a new object).** Every element-wise op with both modes
  provides both forms: `conj`/`conj_`, `trace`/`trace_`, `transpose`/
  `transpose_`, `dagger`/`dagger_`, `pow`/`pow_`, `normalize`/`normalize_`, and
  the **newly added `inv`/`inv_`** (UT-A5). The **`c`-prefixed raw spellings
  (`cConj_`, `cTrace_`, `cPow_`, `cTranspose_`, `cDagger_`, `cnormalize_`,
  `cInv_`, `c__ipow__`) are rejected** as public API — they are the plumbing the
  wrappers call (UT-A4/A5).
- **In-place methods return `self` from the binding directly.** The in-place
  element-wise methods return `UniTensor&` in C++; the pybind lambda must return
  `&self` too, so the conti.py return-self shims and the leaked `c*` bindings
  disappear (UT-A4). The augmented-assign operators (`+=`, `-=`, `*=`, `/=`,
  `**=`) must likewise return `self`, preserving identity (UT-A7).
- **Operators mean their Python semantics.** `/` is true division; `//` is
  **floor** division (or is absent) — it must **not** alias true division
  (UT-A1). `%` is modulo — bind it for the scalar case that C++ supports (UT-A2).
  Named element-wise functions (`Add`/`Sub`/`Mul`/`Div`/`Mod`) live in
  `cytnx.linalg` (Capitalized), not as UniTensor members (UT-A6).

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- binary arithmetic operators (pure = new object) ---
    def __add__(self, rhs) -> "UniTensor": ...
    def __radd__(self, lhs) -> "UniTensor": ...
    def __sub__(self, rhs) -> "UniTensor": ...
    def __rsub__(self, lhs) -> "UniTensor": ...
    def __mul__(self, rhs) -> "UniTensor": ...
    def __rmul__(self, lhs) -> "UniTensor": ...
    def __truediv__(self, rhs) -> "UniTensor": ...    # true division
    def __rtruediv__(self, lhs) -> "UniTensor": ...
    def __floordiv__(self, rhs) -> "UniTensor": ...   # FLOOR division (or removed) — NOT true division
    def __rfloordiv__(self, lhs) -> "UniTensor": ...
    def __mod__(self, rhs) -> "UniTensor": ...         # NEW: modulo (scalar rhs; bind linalg::Mod)
    def __rmod__(self, lhs) -> "UniTensor": ...         # NEW

    # --- augmented assignment (in place; RETURN self) ---
    def __iadd__(self, rhs) -> "UniTensor": ...   # self
    def __isub__(self, rhs) -> "UniTensor": ...   # self
    def __imul__(self, rhs) -> "UniTensor": ...   # self
    def __itruediv__(self, rhs) -> "UniTensor": ...  # self

    # --- unary operators ---
    def __neg__(self) -> "UniTensor": ...   # element-wise negation (new object)
    def __pos__(self) -> "UniTensor": ...   # returns self (unchanged)

    # --- power (pure + in-place) ---
    def __pow__(self, p: float) -> "UniTensor": ...
    def __ipow__(self, p: float) -> "UniTensor": ...   # self
    def pow(self, p: float) -> "UniTensor": ...        # renamed from Pow
    def pow_(self, p: float) -> "UniTensor": ...       # renamed from Pow_; self

    # --- element-wise inverse (pure + NEW in-place) ---
    def inv(self, clip: float = -1) -> "UniTensor": ...   # renamed from Inv
    def inv_(self, clip: float = -1) -> "UniTensor": ...  # NEW public in-place (was only cInv_); self

    # --- conjugate / transpose / adjoint (pure + in-place) ---
    def conj(self) -> "UniTensor": ...          # renamed from Conj
    def conj_(self) -> "UniTensor": ...         # renamed from Conj_; self
    def transpose(self) -> "UniTensor": ...     # renamed from Transpose
    def transpose_(self) -> "UniTensor": ...    # renamed from Transpose_; self
    def dagger(self) -> "UniTensor": ...        # renamed from Dagger
    def dagger_(self) -> "UniTensor": ...       # renamed from Dagger_; self

    # --- normalize / trace / norm ---
    def normalize(self) -> "UniTensor": ...
    def normalize_(self) -> "UniTensor": ...    # self
    def trace(self, a: int | str = 0, b: int | str = 1) -> "UniTensor": ...   # renamed from Trace
    def trace_(self, a: int | str = 0, b: int | str = 1) -> "UniTensor": ...  # renamed from Trace_; self
    def norm(self) -> "Tensor": ...             # renamed from Norm; returns a scalar Tensor
```

In-place methods return `self` **from the binding** (no conti.py shim); the raw
`c*` plumbing bindings become private (leading `_`) or are inlined into the
pybind lambdas — they are **not** public members. The named `Add`/`Sub`/`Mul`/
`Div`/`Mod` stay in `cytnx.linalg` (Capitalized), reached from Python only via
the operators.

| API | Verdict | Behavior contract |
|---|---|---|
| `__add__` / `__radd__` | **keep** (UT-A6/A7) | Pure element-wise add; returns a new UniTensor. |
| `__sub__` / `__rsub__` | **keep** (UT-A6/A7) | Pure element-wise subtract. |
| `__mul__` / `__rmul__` | **keep** (UT-A6/A7) | Pure element-wise multiply. |
| `__truediv__` / `__rtruediv__` | **keep** (UT-A6) | Pure **true** division. |
| `__floordiv__` / `__rfloordiv__` / `__ifloordiv__` | **fix or remove** (UT-A1) | Must implement **floor** division or be removed — must not alias true division. *Migration:* correct the lambda to floor after dividing, or drop the operator with a `DeprecationWarning`. |
| `__iadd__` / `__isub__` / `__imul__` / `__itruediv__` | **keep, return self** (UT-A7) | In-place; must return self (identity preserved). *Migration:* bind `return &self.Add_(rhs)` etc., not by value. |
| `__neg__` | **keep** (UT-A7) | Element-wise negation; new object. |
| `__pos__` | **keep** (UT-A7) | Returns self unchanged. |
| `__pow__` | **keep** (UT-A3) | Pure element-wise power; new object. |
| `__ipow__` | **keep, bind self directly** (UT-A4) | In-place power; returns self. *Migration:* fold the raw `c__ipow__` into the `__ipow__` lambda (which returns self). |
| `__mod__` / `__rmod__` | **add** (UT-A2) | NEW: element-wise modulo for a scalar rhs, backed by `linalg::Mod(UniTensor, scalar)`. *Migration:* uncomment/repair the pybind block; tensor⊗tensor awaits the C++ `Mod` stub being finished. |
| `Pow` → `pow` | **rename** (UT-A3) | Pure element-wise power. *Migration:* `DeprecationWarning` alias `Pow` for one release. |
| `Pow_` → `pow_` | **rename, bind self directly** (UT-A3/A4) | In-place power; returns self. *Migration:* remove the conti.py shim over `cPow_`; alias `Pow_` for one release. |
| `Inv` → `inv` | **rename** (UT-A3) | Pure element-wise inverse (`1/x`, `clip`-guarded). *Migration:* alias `Inv` for one release. |
| `Conj` → `conj` | **rename** (UT-A3) | Pure complex conjugate. *Migration:* alias `Conj`. |
| `Conj_` → `conj_` | **rename, bind self directly** (UT-A3/A4) | In-place conjugate; returns self. *Migration:* remove the `cConj_` shim; alias `Conj_`. |
| `Transpose` → `transpose` | **rename** (UT-A3) | Pure transpose (bra↔ket). *Migration:* alias `Transpose`. |
| `Transpose_` → `transpose_` | **rename, bind self directly** (UT-A3/A4) | In-place transpose; returns self. *Migration:* remove `cTranspose_` shim; alias `Transpose_`. |
| `Dagger` → `dagger` | **rename** (UT-A3) | Pure adjoint (conjugate + transpose). *Migration:* alias `Dagger`. |
| `Dagger_` → `dagger_` | **rename, bind self directly** (UT-A3/A4) | In-place adjoint; returns self. *Migration:* remove `cDagger_` shim; alias `Dagger_`. |
| `normalize` | **keep** (UT-A3) | Pure 2-norm normalization; new object (already lowercase). |
| `normalize_` | **keep, bind self directly** (UT-A4) | In-place normalize; returns self. *Migration:* remove the `cnormalize_` shim. |
| `Trace` → `trace` | **rename** (UT-A3) | Pure trace over legs `a`,`b`. *Migration:* alias `Trace`. |
| `Trace_` → `trace_` | **rename, bind self directly** (UT-A3/A4) | In-place trace; returns self. *Migration:* remove the `cTrace_` shim; alias `Trace_`. |
| `Norm` → `norm` | **rename** (UT-A3) | Returns the 2-norm as a scalar `Tensor`. *Migration:* alias `Norm`. |
| `inv_` | **add** (UT-A5) | NEW public in-place inverse (was only the leaked `cInv_`); returns self, backed by C++ `Inv_`. |

**C++-only — the operator implementations (no separate Python member).** The
named arithmetic methods below exist in C++ and are reached from Python only
through the operator dunders / `cytnx.linalg`; they carry a C++ (R.2b) docstring
only (documented fully in cat 08).

| API | Verdict | Behavior contract |
|---|---|---|
| `Add` / `Sub` / `Mul` / `Div` (+ `_` forms) | **keep (C++-only)** (UT-A6) | Element-wise binary ops behind the operators; exposed to Python as `linalg.Add`/… (Capitalized free functions, cat 08) and the operator dunders — not as UniTensor members. |

**Internal / plumbing — hidden, not public API.** The raw bindings below are live
public members today with a **remove** verdict: hide them behind a leading
underscore or inline them into their pybind lambda. None carry a docstring —
they are not public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `cConj_` | **remove** (UT-A4) | Raw plumbing (C++ `Conj_`) the conti.py `Conj_` wrapper calls. *Migration:* fold into the `conj_` pybind lambda (returns self); no public exposure. |
| `cDagger_` | **remove** (UT-A4) | Raw plumbing (C++ `Dagger_`) behind `Dagger_`. *Migration:* fold into the `dagger_` lambda. |
| `cPow_` | **remove** (UT-A4) | Raw plumbing (C++ `Pow_`) behind `Pow_`/`__ipow__`. *Migration:* fold into the `pow_`/`__ipow__` lambdas. |
| `cTrace_` | **remove** (UT-A4) | Raw plumbing (C++ `Trace_`) behind `Trace_`. *Migration:* fold into the `trace_` lambda. |
| `cTranspose_` | **remove** (UT-A4) | Raw plumbing (C++ `Transpose_`) behind `Transpose_`. *Migration:* fold into the `transpose_` lambda. |
| `cnormalize_` | **remove** (UT-A4) | Raw plumbing (C++ `normalize_`) behind `normalize_`. *Migration:* fold into the `normalize_` lambda. |
| `cInv_` | **remove** (UT-A5) | Raw plumbing (C++ `Inv_`); the new public `inv_` replaces it. *Migration:* fold into the `inv_` lambda; no public exposure. |
| `c__ipow__` | **remove** (UT-A4) | Raw plumbing (C++ `Pow_`) behind `__ipow__`. *Migration:* fold into the `__ipow__` lambda (returns self). |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept/renamed/added members
are documented (removed plumbing carries no docstring); the C++-only `Add`/`Sub`/
`Mul`/`Div` are documented in cat 08.

### R.2a Python API (numpy-style)

### arithmetic operators (`__add__` / `__sub__` / `__mul__` / `__truediv__` / `__floordiv__` / `__mod__` / `__neg__` / `__pos__` / `__iadd__` and their `r`/`i` forms)

```
u + v,  u - v,  u * v,  u / v,  u // v,  u % v        -> UniTensor (pure)
u += v, u -= v, u *= v, u /= v                        -> UniTensor (in place, self)
-u,     +u                                            -> UniTensor

Element-wise arithmetic on UniTensors and scalars.

The binary operators (`+`, `-`, `*`, `/`) return a NEW UniTensor; the augmented
forms (`+=`, `-=`, `*=`, `/=`) mutate this tensor IN PLACE and return self
(finding UT-A7). `-u` negates element-wise; `+u` returns this tensor unchanged.

`/` is TRUE division. `//` is FLOOR division — through cytnx 1.1.0 `//` wrongly
performed true division (finding UT-A1); the next version floors (or removes
`//`). `%` is element-wise modulo for a scalar right-hand operand — absent in
1.1.0 (finding UT-A2), added here over `linalg.Mod`.

The right-hand operand may be a UniTensor or any of the 11 element scalar dtypes.
The named forms `Add`/`Sub`/`Mul`/`Div`/`Mod` live in `cytnx.linalg`
(Capitalized) — the operators are the sole UniTensor-member surface (UT-A6).

Returns
-------
UniTensor
    Binary/unary operators: a new tensor. Augmented assignment: self.
```

### `pow` / `pow_` / `__pow__` / `__ipow__`

```
UniTensor.pow(p)     -> UniTensor    # pure element-wise power (renamed from Pow)
UniTensor.pow_(p)    -> UniTensor    # in-place, self (renamed from Pow_)
u ** p               -> UniTensor    # pure
u **= p              -> UniTensor    # in-place, self

Raise every element of this UniTensor to the power `p`.

`pow`/`u ** p` are PURE and return a new tensor; `pow_`/`u **= p` raise the
elements IN PLACE and return self (finding UT-A4).

Parameters
----------
p : float
    The exponent.

Returns
-------
UniTensor
    `pow`/`**`: a new tensor. `pow_`/`**=`: self.

Notes
-----
Renamed from the Capitalized `Pow`/`Pow_` (finding UT-A3); the in-place forms no
longer route through the leaked raw `cPow_`/`c__ipow__` bindings (finding UT-A4).
`Pow`/`Pow_` remain `DeprecationWarning` aliases for one release.
```

### `inv` / `inv_`

```
UniTensor.inv(clip=-1)    -> UniTensor    # pure element-wise inverse (renamed from Inv)
UniTensor.inv_(clip=-1)   -> UniTensor    # in-place, self (NEW)

Element-wise reciprocal (1/x) of this UniTensor.

`inv` is PURE and returns a new tensor; `inv_` inverts IN PLACE and returns self
(finding UT-A5 — 1.1.0 had no public in-place form, only the leaked `cInv_`).

Parameters
----------
clip : float, optional
    Guard for small magnitudes: elements with |x| <= clip are left/handled per
    the clip rule rather than blowing up (default -1 = no clip).

Returns
-------
UniTensor
    `inv`: a new tensor. `inv_`: self.

Notes
-----
Renamed from the Capitalized `Inv` (finding UT-A3); `inv_` is newly added public
in-place (finding UT-A5). `Inv` remains a `DeprecationWarning` alias for one
release.
```

### `conj` / `conj_` / `transpose` / `transpose_` / `dagger` / `dagger_`

```
UniTensor.conj()        -> UniTensor    # pure conjugate         (renamed from Conj)
UniTensor.conj_()       -> UniTensor    # in-place, self         (renamed from Conj_)
UniTensor.transpose()   -> UniTensor    # pure transpose         (renamed from Transpose)
UniTensor.transpose_()  -> UniTensor    # in-place, self         (renamed from Transpose_)
UniTensor.dagger()      -> UniTensor    # pure adjoint           (renamed from Dagger)
UniTensor.dagger_()     -> UniTensor    # in-place, self         (renamed from Dagger_)

Complex conjugate, transpose, and adjoint (dagger = conjugate + transpose).

Each has a PURE form (new object) and an IN-PLACE form returning self
(finding UT-A4):

conj      : complex-conjugate every element.
transpose : swap bra/ket (row/col) roles of the legs.
dagger    : conjugate AND transpose — the Hermitian adjoint.

Returns
-------
UniTensor
    Pure form: a new tensor. In-place (`_`) form: self.

Notes
-----
Renamed from the Capitalized `Conj`/`Transpose`/`Dagger` (+ `_` forms, finding
UT-A3); the in-place forms no longer route through the leaked raw `cConj_`/
`cTranspose_`/`cDagger_` bindings (finding UT-A4). The old Capitalized names
remain `DeprecationWarning` aliases for one release. The Capitalized *free*
functions `linalg.Conj`/… stay Capitalized (cat 08).
```

### `normalize` / `normalize_` / `trace` / `trace_` / `norm`

```
UniTensor.normalize()       -> UniTensor    # pure 2-norm normalization
UniTensor.normalize_()      -> UniTensor    # in-place, self
UniTensor.trace(a=0, b=1)   -> UniTensor    # pure trace over legs a,b (renamed from Trace)
UniTensor.trace_(a=0, b=1)  -> UniTensor    # in-place, self          (renamed from Trace_)
UniTensor.norm()            -> Tensor       # 2-norm scalar           (renamed from Norm)

Normalization, trace, and norm.

`normalize` divides by the 2-norm (PURE); `normalize_` does so IN PLACE and
returns self. `trace` contracts legs `a` and `b` (PURE, new object); `trace_`
does so IN PLACE and returns self. `norm` returns the scalar 2-norm as a
`cytnx.Tensor` (finding UT-A3) — NOT a UniTensor.

Parameters
----------
a, b : int or str
    The two legs to trace over, by index or label (default legs 0 and 1).

Returns
-------
UniTensor or Tensor
    `normalize`/`trace`: a new UniTensor. `normalize_`/`trace_`: self.
    `norm`: a scalar `Tensor`.

Notes
-----
`normalize`/`normalize_` are already lowercase (kept). `trace`/`trace_`/`norm`
are renamed from the Capitalized `Trace`/`Trace_`/`Norm` (finding UT-A3); the
in-place forms drop the leaked `cTrace_`/`cnormalize_` bindings (finding UT-A4).
Old Capitalized names remain `DeprecationWarning` aliases for one release.
```

### R.2b C++ API (Doxygen)

C++ already returns `UniTensor&`/`UniTensor`/`Tensor` per the N-underscore split;
the next version must have the *pybind lambdas* return these directly (removing
the conti.py shims and the leaked `c*` bindings, UT-A4/A5), wire `//` to floor
(or drop it, UT-A1), bind `%` for the scalar case (UT-A2), add a public `inv_`
(UT-A5), and have the augmented-assign operators return `self` (UT-A7). The
member names are lowercased in the Python binding while the C++ method names and
the `linalg` free functions keep their capitalization (cat 08).

```cpp
/**
 * @brief Element-wise power (pure and in-place).
 * @details Pow(p) raises every element to p and returns a NEW UniTensor; Pow_(p)
 *          does so in place and returns *this. The Python binding exposes these
 *          as pow/pow_ and __pow__/__ipow__, returning self directly from the
 *          in-place lambda (dropping the leaked cPow_/c__ipow__ bindings,
 *          findings UT-A3/A4).
 * @param p the exponent.
 * @return Pow: a new UniTensor. Pow_: reference to *this.
 */
UniTensor Pow(const double &p) const;
UniTensor &Pow_(const double &p);

/**
 * @brief Element-wise reciprocal (1/x), pure and in-place.
 * @details Inv(clip) returns a NEW UniTensor; Inv_(clip) inverts in place and
 *          returns *this. `clip` guards small magnitudes. The Python binding
 *          exposes inv/inv_ — 1.1.0 lacked a public in-place inv_, exposing only
 *          the leaked raw cInv_ (finding UT-A5).
 * @param clip small-magnitude guard (-1 = none).
 * @return Inv: a new UniTensor. Inv_: reference to *this.
 */
UniTensor Inv(double clip = -1.) const;
UniTensor &Inv_(double clip = -1.);

/**
 * @brief Complex conjugate / transpose / adjoint, pure and in-place.
 * @details Each pure form returns a NEW UniTensor; each in-place form returns
 *          *this (finding UT-A4). Dagger = Conj followed by Transpose. The Python
 *          binding exposes conj/conj_, transpose/transpose_, dagger/dagger_ and
 *          returns *this directly from the in-place lambdas (dropping the leaked
 *          cConj_/cTranspose_/cDagger_ bindings).
 * @return pure form: a new UniTensor. in-place (_) form: reference to *this.
 */
UniTensor Conj() const;        UniTensor &Conj_();
UniTensor Transpose() const;   UniTensor &Transpose_();
UniTensor Dagger() const;      UniTensor &Dagger_();

/**
 * @brief 2-norm normalization (pure and in-place) and the scalar norm.
 * @details normalize() divides by the 2-norm (new object); normalize_() does so
 *          in place and returns *this. Norm() returns the 2-norm as a scalar
 *          Tensor (finding UT-A3), NOT a UniTensor.
 * @return normalize: a new UniTensor. normalize_: reference to *this. Norm: a
 *         scalar Tensor.
 */
UniTensor normalize() const;
UniTensor &normalize_();
Tensor Norm() const;

/**
 * @brief Trace over two legs, pure and in-place.
 * @details Trace(a,b) contracts legs a and b and returns a NEW UniTensor;
 *          Trace_(a,b) does so in place and returns *this. Legs may be given by
 *          index or by label. Python exposes trace/trace_ (finding UT-A3),
 *          returning *this directly from the in-place lambda (dropping cTrace_).
 * @param a,b the two legs (index or label; default 0,1 for the index overload).
 * @return Trace: a new UniTensor. Trace_: reference to *this.
 */
UniTensor Trace(const cytnx_int64 &a = 0, const cytnx_int64 &b = 1) const;
UniTensor Trace(const std::string &a, const std::string &b) const;
UniTensor &Trace_(const cytnx_int64 &a = 0, const cytnx_int64 &b = 1);
UniTensor &Trace_(const std::string &a, const std::string &b);

/**
 * @brief Element-wise binary arithmetic (behind the Python operators).
 * @details Add/Sub/Mul/Div (and the in-place Add_/…/operator+= family) implement
 *          the Python +, -, *, / operators; the in-place operators return *this
 *          and the Python augmented-assign lambdas must too (finding UT-A7). The
 *          Python `/` is true division; `//` must floor or be dropped (must NOT
 *          alias Div, finding UT-A1). Modulo lives in linalg::Mod / operator%:
 *          the scalar forms are implemented, but Mod(UniTensor, UniTensor) is an
 *          unfinished stub throwing "[Mod][Developing]" (finding UT-A2). These
 *          named functions are documented fully in cat 08.
 * @return pure ops: a new UniTensor. in-place ops: reference to *this.
 */
UniTensor Add(const UniTensor &rhs) const;   UniTensor &operator+=(const UniTensor &rhs);
UniTensor Sub(const UniTensor &rhs) const;   UniTensor &operator-=(const UniTensor &rhs);
UniTensor Mul(const UniTensor &rhs) const;   UniTensor &operator*=(const UniTensor &rhs);
UniTensor Div(const UniTensor &rhs) const;   UniTensor &operator/=(const UniTensor &rhs);
// free functions (namespace cytnx / cytnx::linalg):
UniTensor operator%(const UniTensor &Lt, const UniTensor &Rt);      // linalg.hpp:175
namespace linalg { UniTensor Mod(const UniTensor &Lt, const UniTensor &Rt); }  // linalg.hpp:625 (stub)
```
