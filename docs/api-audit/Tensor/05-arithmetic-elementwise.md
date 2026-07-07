# Tensor — 05. Arithmetic & element-wise

> **Superset-method rollout** (Tensor, category 05 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings, arg ordering) and a self-contained **Recommendation** that
> is the *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` arithmetic/element-wise surface should be implemented to match §R
> exactly. Every behavioral claim is verified against the installed
> `cytnx==1.1.0` wheel by `docs/api-audit/probes/Tensor_05_arithmetic.py` (all
> `[PASS]`, exit 0), with the raw-C++ side of the binding-fidelity findings
> verified by `docs/api-audit/probes/cpp/Tensor_05_arithmetic.cpp` against a
> source-built `libcytnx` (GCC 13; all `[PASS]`, exit 0).

**Category scope:** the members that combine two dense tensors element-wise or
apply a scalar element-wise map — the Python operator dunders (`__add__`/
`__radd__`/`__iadd__` and the `-`/`*`/`/` families, `__floordiv__`, `__mod__`/
`__rmod__`, `__matmul__`/`__imatmul__`, `__neg__`/`__pos__`, `__pow__`/
`__ipow__`, `__eq__`), and the named element-wise methods `Abs`/`Abs_`, `Conj`/
`Conj_`, `Exp`/`Exp_`, `Inv`/`Inv_`, `Pow`/`Pow_`, `Norm` — plus the leaked raw
`c*` bindings (`cAbs_`/`cConj_`/`cExp_`/`cInv_`/`cPow_`, `c__iadd__`/
`c__ifloordiv__`/`c__imatmul__`/`c__imul__`/`c__ipow__`/`c__isub__`/
`c__itruediv__`) the conti.py in-place wrappers call. The **matrix** inverse
`InvM`/`InvM_` and the reductions `Norm`/`Max`/`Min`/`Trace`/`Svd`/`Eigh` (member
linear algebra) are [category 06](06-linalg-reductions.md); `Norm` is shared here
as the 2-norm scalar. Python bindings: `cytnx_src/pybind/tensor_py.cpp:436-450`
(neg/pos), `451-664` (add family), `666-876` (sub), `881-1091` (mul),
`1096-1324` (truediv), `1329-1561` (floordiv), `1566-1662` (mod), `1709-1767`
(eq), `1769-1782` (pow/matmul), `1785-1797` (Inv/Conj/Exp/Pow/Abs/Norm);
conti.py wrappers: `cytnx/Tensor_conti.py:58-120`; C++ header:
`cytnx_src/include/Tensor.hpp:1271-1407,1644-1714`; free-function matmul:
`cytnx_src/include/linalg.hpp:2482`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place. Operator families whose
overloads differ only in the right-hand-operand type (Tensor vs the 11 scalar
dtypes) share one row.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `__add__` / `__radd__` | `__add__(rhs)` (Tensor or scalar) | `Tensor` (new) | **Pure** element-wise add; routes to `Tensor::Add`. Probe: *"operator dunder `__add__` IS bound …"* + *"radd …"*. |
| `__sub__` / `__rsub__` | `__sub__(rhs)` | `Tensor` (new) | Pure element-wise subtract (`Sub`). |
| `__mul__` / `__rmul__` | `__mul__(rhs)` | `Tensor` (new) | Pure element-wise multiply (`Mul`). |
| `__truediv__` / `__rtruediv__` | `__truediv__(rhs)` | `Tensor` (new) | Pure element-wise **true** division (`Div`). |
| `__floordiv__` / `__rfloordiv__` | `__floordiv__(rhs)` | `Tensor` (new) | **CORRECTNESS BUG** — routes to `Div` (`:1329-1355`), so `//` performs **true** division, *not* floor: `(t*7)//2 → 3.5`. Probe: *"`//` (__floordiv__) performs TRUE division, not floor … yields 3.5 / 10.5 …"* + *"`//` gives the IDENTICAL result to `/` …"*. |
| `__mod__` / `__rmod__` | `__mod__(rhs)` | `Tensor` (new) | **Present** (contrast UniTensor, whose `__mod__` block is commented out) — element-wise modulo (`Tensor::Mod`), working for **both** a scalar and a tensor rhs on a Dense tensor. Probe: *"`%` (__mod__) IS bound on Tensor …"* + *"`t % 3` … [1,2,0,1]"* + *"`t % v` … [1,2,1,0] … Tensor's Mod is fully implemented …"*. |
| `__iadd__` / `__isub__` / `__imul__` / `__itruediv__` / `__ifloordiv__` `[I]` | `__iadd__(rhs)` … | `Tensor` (self; **conti.py wrapper**) | In-place; conti.py (`:58-82`) forwards to the leaked raw `c__iadd__`/`c__isub__`/`c__imul__`/`c__itruediv__`/`c__ifloordiv__` (which wrap `Add_`/`Sub_`/`Mul_`/`Div_`) and `return self` — mutates the receiver **and preserves identity** (contrast UniTensor UT-A7). `//=` is true-division-in-place. Probe: *"`+=` … mutates in place AND preserves identity …"* (one per op) + *"`//=` … is likewise TRUE division in place …"*. |
| `__neg__` | `__neg__()` | `Tensor` (new) | Element-wise negation; routes to `linalg::Mul(-1, self)` (`:436-449`). Probe: *"`__neg__` negates element-wise into a NEW object … 1 -> -1 …"*. |
| `__pos__` | `__pos__()` | `Tensor` (**shared data**) | Returns the tensor unchanged (`return self` by value → a distinct, shared-data wrapper). Probe: *"`__pos__` returns a DISTINCT object that SHARES data …"*. |
| `__pow__` | `__pow__(p: float)` | `Tensor` (new) | Pure element-wise power; wraps `linalg::Pow` (`:1769`). Probe: *"operator dunder `__pow__` is bound"* (via T-A6 set). |
| `__ipow__` `[I]` | `__ipow__(p: float)` | `Tensor` (self) | In-place power; conti.py wrapper (`:117-120`) over the leaked raw `c__ipow__` (which wraps `linalg::Pow_`), returns self. Probe: *"`**=` (__ipow__) raises in place AND preserves identity (2 -> 4, 3 -> 9)"*. |
| `__matmul__` | `__matmul__(rhs: Tensor)` | `Tensor` (new) | Pure matrix product; wraps `linalg::Dot` (`:1773`). Probe: *"`__matmul__` (@, linalg.Dot) is a PURE matrix product returning a new object"*. |
| `__imatmul__` `[I]` | *(ABSENT)* | — | **BROKEN (B-5)** — conti.py defines `__imatmul` (missing the trailing `__`, `:84-87`), so the real `__imatmul__` slot does **not** exist; `t @= x` falls back to `__matmul__` and **rebinds** `t` to a fresh object. Probe: *"Tensor_conti.py defines `__imatmul` … so the true __imatmul__ slot does NOT exist …"* + *"`t @= x` is NOT in place … REBINDS t to a FRESH object …"*. |
| `__eq__` | `__eq__(rhs)` | `Tensor` (Bool) | **Element-wise** equality → a Bool `Tensor`, not a Python `bool`; makes `Tensor` **unhashable** (`__hash__ is None`). Probe: *"`==` (__eq__) is ELEMENT-WISE … Bool Tensor …"* + *"Tensor.__hash__ is None … UNHASHABLE …"*. |
| `Abs` | `Abs()` | `Tensor` (new) | Capitalized member — pure element-wise `|x|`. Probe: *"capitalized member `Abs` exists …"* + *"`Abs` is PURE …"*. |
| `Abs_` `[I]` | `Abs_()` | `Tensor` (self) | Capitalized member — in-place abs; conti.py wrapper (`:108-111`) over leaked `cAbs_`; returns self. Probe: *"`Abs_()` returns SELF and abs-es in place (-1 -> 1)"*. |
| `Conj` | `Conj()` | `Tensor` (new) | Capitalized member — pure complex conjugate. Probe: *"capitalized member `Conj` exists …"*. |
| `Conj_` `[I]` | `Conj_()` | `Tensor` (self) | Capitalized member — in-place conjugate; conti.py wrapper (`:90-93`) over leaked `cConj_`; returns self. Probe: *"`Conj_()` returns SELF (in place)"*. |
| `Exp` | `Exp()` | `Tensor` (new) | Capitalized member — pure element-wise exponential. Probe: *"capitalized member `Exp` exists …"*. |
| `Exp_` `[I]` | `Exp_()` | `Tensor` (self) | Capitalized member — in-place exp; conti.py wrapper (`:94-97`) over leaked `cExp_`; returns self. Probe: *"`Exp_()` returns SELF (in place)"*. |
| `Inv` | `Inv(clip=-1)` | `Tensor` (new) | Capitalized member — **element-wise** reciprocal (`1/x`, with `clip` guarding small values). Distinct from the **matrix** inverse `InvM` (cat 06). Probe: *"`Inv` is the ELEMENT-WISE reciprocal 1/x (1 -> 1, 2 -> 0.5, 4 -> 0.25) …"*. |
| `Inv_` `[I]` | `Inv_(clip=-1)` | `Tensor` (self) | Capitalized member — in-place reciprocal; conti.py wrapper (`:102-105`) over leaked `cInv_`; returns self. Probe: *"`Inv_(clip)` returns SELF and inverts in place (2 -> 0.5, 4 -> 0.25)"*. |
| `Pow` | `Pow(p: float)` | `Tensor` (new) | Capitalized member — pure element-wise power. Probe: *"`Pow(2.0)` is PURE: squares into a new tensor …"*. |
| `Pow_` `[I]` | `Pow_(p: float)` | `Tensor` (self) | Capitalized member — in-place power; conti.py wrapper (`:112-115`) over leaked `cPow_`; returns self. Probe: *"`Pow_(2.0)` returns SELF and squares in place (2 -> 4, 3 -> 9)"*. |
| `Norm` | `Norm()` | `Tensor` (scalar) | Capitalized member — the 2-norm as a scalar `Tensor` (shape `[1]`). Probe: *"`Norm()` returns a cytnx.Tensor holding the 2-norm (sqrt(3^2+4^2)==5)"*. |

**Internal / plumbing (leaks into `dir(Tensor)`):** `cAbs_`, `cConj_`, `cExp_`,
`cInv_`, `cPow_`, `c__iadd__`, `c__ifloordiv__`, `c__imatmul__`, `c__imul__`,
`c__ipow__`, `c__isub__`, `c__itruediv__` — the raw pybind bindings the conti.py
in-place wrappers call (`tensor_py.cpp:594-664,809-876,1024-1091,1256-1324,
1493-1561,1771-1782,1785-1794`). The `c`-prefix is a reserved raw-binding
spelling. Public today, but should never be. Probe: *"the raw plumbing binding
`cConj_` LEAKS into public dir(Tensor)"* (one per name) + *"`cConj_()` (raw
primitive) mutates the receiver in place (self-aliasing) …"*.

**C++-only (present in C++, *not* bound as named Python members):** the
arithmetic methods `Add`/`Sub`/`Mul`/`Div` (and their `Add_`/`Sub_`/`Mul_`/
`Div_` in-place forms), the comparison `Cpr`, and `Mod` (`Tensor.hpp:1271-1397`)
— reachable from Python only through the operator dunders. Probe: *"the C++ named
arithmetic methods (Add/Sub/Mul/Div + their _ forms, Cpr, Mod) have NO Python
member binding — only the operators exist"*.

## A2. C++ ↔ Python mapping

Status: `identical` · `renamed` · `signature-differs` · `C++-only` · `leak`. A
binding that faithfully mirrors the C++ signature — including `void`→`None`,
`T&`→self, and by-value→a fresh wrapper — is `identical`; `signature-differs`
marks a binding-layer change to arity or defaults.

| C++ (`Tensor.hpp` / `linalg.hpp`) | Python | Status | Note |
|---|---|---|---|
| `Tensor Add/Sub/Mul/Div(const T&)` (`hpp:1271-1332`) | `__add__`/`__sub__`/`__mul__`/`__truediv__` (`:451,666,881,1096`) | identical | dunders route to `Add`/`Sub`/`Mul`/`Div`; named `Add`/… unbound (T-A6) |
| `Tensor &Add_/Sub_/Mul_/Div_(const T&)` (`hpp:1281-1343`) | `c__iadd__`/… (`:594,809,1024,1256`) **+** conti.py `__iadd__`/… (`conti.py:58-82`) | **leak + wrapper** | raw `Add_`/… bound as `c__i*__`; public `__iadd__`/… wrap them and `return self` (identity kept, T-A7) |
| `Tensor Div(const T&)` (`hpp:1332`) | `__floordiv__` (`:1329-1355`, calls `Div`) | **binding fidelity** | `//` wired to true division — a correctness bug (T-A1) |
| `Tensor Mod(const T&)` (`hpp:1397`) | `__mod__` (`:1566`), `__rmod__` (`:1634`) | identical | `%` bound and fully working for scalar+tensor (T-A8; contrast UniTensor UT-A2) |
| `Tensor operator-()` → `Mul(-1)` (`hpp:1407`) | `__neg__` (`:436-449`) | identical | element-wise negation (T-A7) |
| `return *this` by value | `__pos__` (`:450`) | identical | returns a distinct, shared-data wrapper (T-A7) |
| `linalg::Pow(Tensor,double)` | `__pow__` (`:1769`), `Pow` (`:1791`) | identical | pure power (T-A3) |
| `linalg::Pow_(Tensor&,double)` / `Tensor &Pow_(double)` (`hpp:1696`) | `c__ipow__` (`:1771`), `cPow_` (`:1792`) **+** conti.py `__ipow__`/`Pow_` (`:112-120`) | **leak** | raw in-place power bound as `c__ipow__`/`cPow_`; public forms wrap them (T-A4) |
| `linalg::Dot(Tensor,Tensor)` (`linalg.hpp:2482`) | `__matmul__` (`:1773`) | identical | pure matrix product (T-A2) |
| `self = linalg::Dot(self,rhs)` | `c__imatmul__` (`:1775`) **+** conti.py `__imatmul` **(typo)** (`:84-87`) | **binding fidelity** | raw primitive works; the public wrapper is misnamed so `@=` is not in place (T-A2 / B-5) |
| `Tensor Inv(double clip) const` (`hpp:1654`) | `Inv` (`:1786`) | identical | element-wise reciprocal (T-A3/A5) |
| `Tensor &Inv_(double clip)` (`hpp:1644`) | `cInv_` (`:1785`) **+** conti.py `Inv_` (`:102-105`) | **leak** | raw `Inv_` bound as `cInv_`; public `Inv_` wraps it (T-A4) |
| `Tensor Conj() const` (`hpp:1666`) | `Conj` (`:1788`) | identical | pure conjugate (T-A3) |
| `Tensor &Conj_()` (`hpp:1660`) | `cConj_` (`:1787`) **+** conti.py `Conj_` (`:90-93`) | **leak** | raw `Conj_` bound as `cConj_` (T-A4) |
| `Tensor Exp() const` (`hpp:1678`) | `Exp` (`:1790`) | identical | pure exponential (T-A3) |
| `Tensor &Exp_()` (`hpp:1672`) | `cExp_` (`:1789`) **+** conti.py `Exp_` (`:94-97`) | **leak** | raw `Exp_` bound as `cExp_` (T-A4) |
| `Tensor Abs() const` (`hpp:1708`) | `Abs` (`:1793`) | identical | pure absolute value (T-A3) |
| `Tensor &Abs_()` (`hpp:1714`) | `cAbs_` (`:1794`) **+** conti.py `Abs_` (`:108-111`) | **leak** | raw `Abs_` bound as `cAbs_` (T-A4) |
| `Tensor Pow(cytnx_double) const` (`hpp:1690`) | `Pow` (`:1791`) | identical | pure power (T-A3) |
| `Tensor Norm() const` (`hpp:1684`) | `Norm` (`:1797`) | identical | scalar 2-norm `Tensor` (T-A3) |
| `Tensor operator==(...)` | `__eq__` (`:1709-1767`) | identical | element-wise → Bool Tensor; Tensor unhashable (T-A10) |
| raw `cAbs_`/`cConj_`/`cExp_`/`cInv_`/`cPow_`/`c__i*__` | same names | **leak** | plumbing exposed publicly (T-A4/A9) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/Tensor_05_arithmetic.py` (on the 1.1.0 wheel). A
**(binding fidelity)** finding flags where the binding layer — a `Tensor_conti.py`
wrapper or a pybind lambda — changes behavior, signature, or availability versus
the raw C++ method; **both sides are runtime-verified**, the raw-C++ side by
`probes/cpp/Tensor_05_arithmetic.cpp` (links against a source-built `libcytnx`,
GCC 13). Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-A1** | `//` (`__floordiv__`) performs **true** division, not floor: `(t*7)//2 → 3.5` | **correctness** | **binding aliases the wrong op**: the `__floordiv__` pybind lambda calls `self.Div(rhs)` (`:1329-1355`) — the same op as `__truediv__` — so `//` never floors; `//=` (`c__ifloordiv__` → `Div_`, `:1493`) is true-division-in-place. Py probe *"`//` (__floordiv__) performs TRUE division, not floor … yields 3.5 / 10.5 …"* + *"`//` gives the IDENTICAL result to `/` …"* + *"`//=` … is likewise TRUE division in place …"* | **fix or remove**: either implement floor semantics for `//` (round toward −∞ after the divide), **or remove `__floordiv__`/`__rfloordiv__`/`__ifloordiv__` entirely** — do **not** silently route `//` to true division. A `DeprecationWarning` shim documents the change. Cross-ref UniTensor UT-A1. |
| **T-A2** | **headline (B-5 / v1 P4):** `t @= x` is **not** in place — it rebinds `t` to a fresh object | **correctness / binding fidelity** | **the conti.py wrapper is misnamed**: `Tensor_conti.py:84-87` defines `def __imatmul(self, rhs)` — missing the trailing `__` — so `Tensor` has no `__imatmul__` slot; `t @= x` finds none, falls back to `t = t.__matmul__(x)` (`linalg::Dot`), and **rebinds** `t`. The raw `c__imatmul__` primitive itself works (`self = Dot(self,rhs); return self`, `:1775`). Py probe *"Tensor_conti.py defines `__imatmul` … so the true __imatmul__ slot does NOT exist …"* + *"`t @= x` is NOT in place … REBINDS t to a FRESH object …"* + *"the raw c__imatmul__ primitive returns a self-aliasing handle …"*; **C++ probe confirms** `linalg::Dot` computes the product and `A = Dot(A,S)` mutates A correctly — so the break is the misnamed wrapper, not a C++/primitive gap | **fix the typo** — rename the conti.py `__imatmul` → `__imatmul__` (or, better, bind the in-place matmul in pybind returning self and drop the conti.py shim), so `t @= x` mutates in place and returns self. *Migration:* `DeprecationWarning`-free (a bug fix); the old `__imatmul` attribute is removed. |
| **T-A3** | `Abs`/`Conj`/`Exp`/`Inv`/`Pow`/`Norm` (and their `_` forms) are **Capitalized members** | naming (N-casing) | **Capitalized member spellings**: every one is `.def`-ed with an initial capital (`:1786-1797`), violating the members-are-lowercase rule. The **same names also exist as `cytnx.linalg` free functions** (`Abs`/`Conj`/`Exp`/`Pow`/`Norm`), which **stay Capitalized** (they act on objects — cat 08 analog). Py probe *"capitalized member `Abs` exists … (N-casing: should be lowercase `abs`)"* (one per name) + *"the capitalized name `Abs` ALSO exists as a cytnx.linalg FREE function …"* | **rename to lowercase**: `Abs`→`abs`, `Conj`→`conj`, `Exp`→`exp`, `Pow`→`pow`, `Norm`→`norm` (and their `_` forms); `Inv`→`reciprocal` (T-A5). Keep the free-function `linalg.Abs`/… Capitalized (cross-ref cat 08). *Migration:* keep each old Capitalized member as a `DeprecationWarning` alias for one minor release, then delete. Cross-ref UniTensor UT-A3. |
| **T-A4** | the in-place `Abs_`/`Conj_`/`Exp_`/`Inv_`/`Pow_` (and `__ipow__`) are **conti.py wrappers over leaked raw `c*` bindings** | naming + **binding fidelity** | **binding exposes plumbing + wraps it**: raw C++ `Abs_`/`Conj_`/`Exp_`/`Inv_`/`Pow_` (each returning `Tensor&`) are bound as `cAbs_`/`cConj_`/`cExp_`/`cInv_`/`cPow_` (`:1785-1794`), and `Tensor_conti.py:90-120` defines each public in-place form as `self.c<Name>(); return self`. Py probe *"`Conj_()` returns SELF (in place)"* (etc.) + *"the raw plumbing binding `cConj_` LEAKS into public dir(Tensor)"*; **C++ probe confirms** C++ `Conj_()`/`Abs_()`/`Exp_()`/`Pow_(2.0)`/`Inv_()` each return `&*this` | **remove the `c*` bindings from the public API** (bind under a leading `_` or inline into the pybind lambda) and have the in-place pybind lambda **return self directly** — dropping the conti.py `return self` shims (migration note). Cross-ref UniTensor UT-A4. |
| **T-A5** | `Inv` is the **element-wise** reciprocal (`1/x`), easily confused with the **matrix** inverse `InvM` (cat 06) | **naming / redundancy** | **two "inverse" concepts, similar names**: `Inv`/`Inv_` (`:1786,1785`) invert **element-wise** (`1/x`, `clip`-guarded), while `InvM`/`InvM_` (cat 06) compute the **matrix** inverse. Py probe *"`Inv` is the ELEMENT-WISE reciprocal 1/x … distinct from the matrix inverse InvM (cat 06)"* | **rename `Inv`→`reciprocal` and `Inv_`→`reciprocal_`** (not just lowercased) to disambiguate from the matrix inverse — unifying with the free-function rename `linalg.Inv`→`Reciprocal` (cross-ref UniTensor UT-A5/UT-X4); `InvM`→`inv_m` (cat 06) keeps the matrix concept. *Migration:* `DeprecationWarning` alias `Inv`/`Inv_` for one release. |
| **T-A6** | the named arithmetic methods `Add`/`Sub`/`Mul`/`Div` (and `_` forms), `Cpr`, `Mod` are **C++-only** — unbound as Python members | **binding fidelity / C++-only** | **binding hides the methods behind the operators**: the C++ `Add`/`Sub`/`Mul`/`Div`/`Cpr`/`Mod` methods (`hpp:1271-1397`) are absent from `dir(Tensor)`; only the operator dunders are exposed. Py probe *"the C++ named arithmetic methods … have NO Python member binding — only the operators exist"* + *"operator dunder `__add__` IS bound"* (one per dunder) | **keep the operators as the Python surface**; the named element-wise functions belong in `cytnx.linalg` (Capitalized, cat 08). Document the operator↔free-function equivalence; no separate `Add`/… Python member. Cross-ref UniTensor UT-A6 / v1 P1. |
| **T-A7** | the in-place operators `__iadd__`/`__isub__`/`__imul__`/`__itruediv__`/`__ifloordiv__`/`__ipow__` mutate the receiver **and preserve identity** (conti.py `return self`); `__neg__` is a new object, `__pos__` shares data | **binding fidelity** (N2/B1) | **conti.py returns self**: unlike UniTensor's by-value augmented-assign lambdas (UT-A7, identity dropped), Tensor's `Tensor_conti.py:58-120` wrappers call the raw `c__i*__` and `return self`, so `a += x` mutates in place **and** `a is (the original handle)`. `__neg__` routes to `Mul(-1)` (`:436-449`, new object); `__pos__` is `return self` by value (`:450`) — a distinct shared-data wrapper. Py probe *"`+=` … mutates in place AND preserves identity …"* (one per op) + *"`__neg__` negates element-wise into a NEW object …"* + *"`__pos__` returns a DISTINCT object that SHARES data …"* | **keep**; when the leaked `c__i*__` bindings are folded away (T-A9), have the in-place pybind lambda **return self directly** (`return &self.Add_(rhs)`), preserving the identity the conti.py shim currently provides — matching Python's data-model contract for `__iadd__`. |
| **T-A8** | `%`/`__mod__`/`__rmod__` **is bound** for Tensor and works for both scalar and tensor operands | **binding parity (positive)** | **binding present + C++ complete**: the `__mod__`/`__rmod__` pybind block is live (`:1566-1662`, routing to `Tensor::Mod`, `hpp:1397`), and Tensor's Dense `Mod` is fully implemented (no `[Mod][Developing]` stub). Py probe *"`%` (__mod__) IS bound on Tensor (contrast UniTensor …)"* + *"`t % 3` … [1,2,0,1]"* + *"`t % v` … [1,2,1,0] … Tensor's Mod is fully implemented (Dense) …"* | **keep** — this is the correct state UniTensor UT-A2 asks for. Document `%` as element-wise modulo for scalar and tensor operands. |
| **T-A9** | the raw `c*` / `c__i*__` plumbing bindings **leak** into `dir(Tensor)` | **binding fidelity** | **binding exposes plumbing**: `cAbs_`/`cConj_`/`cExp_`/`cInv_`/`cPow_` and `c__iadd__`/`c__isub__`/`c__imul__`/`c__itruediv__`/`c__ifloordiv__`/`c__ipow__`/`c__imatmul__` are `.def`-ed publicly (`:594-1794`) purely so the conti.py wrappers can call them; the `c`-prefix is a reserved raw-binding spelling (§R.0 rejects it). Py probe *"the raw plumbing binding `cConj_` LEAKS into public dir(Tensor)"* (one per name) + *"`cConj_()` (raw primitive) mutates the receiver in place (self-aliasing) …"* | **remove from the public API** — bind each under a leading `_` or inline into its wrapper's pybind lambda (which returns self); they are not public surface (migration note). Cross-ref v1 P3. |
| **T-A10** | `__eq__` is **element-wise** (returns a Bool `Tensor`), which makes `Tensor` **unhashable** | **consistency** (B5-adjacent) | **numpy-like elementwise ==**: `__eq__` (`:1709-1767`) routes to `operator==`, returning a Bool `Tensor`, and Python sets `Tensor.__hash__` to `None`. Py probe *"`==` (__eq__) is ELEMENT-WISE and returns a Bool Tensor …"* + *"Tensor.__hash__ is None … UNHASHABLE …"* | **keep** the element-wise operator (numpy-like), **but add a whole-tensor `equal(other) -> bool` predicate** (N5-named, mirrors numpy's `array_equal`) so callers can test structural equality and, if desired, key a dict. Cross-ref v1 C5. |

## A4. Argument ordering — positional & keyword

Every member here takes at most one operand (the right-hand tensor/scalar) or a
single operation parameter (`p` for power, `clip` for inverse); there is no
keyword-only metadata block.

| API | positional-required (in order) | keyword |
|---|---|---|
| `__add__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__`/`__mod__` (+ `r`/`i` forms) | `rhs` (Tensor or scalar) | — |
| `__matmul__` | `rhs` (Tensor) | — |
| `__neg__` / `__pos__` / `__eq__` | `rhs` (`__eq__` only) | — |
| `__pow__` / `__ipow__` / `Pow` / `Pow_` | `p` (exponent) | — |
| `Inv` (→ `reciprocal`) / `Inv_` (→ `reciprocal_`) | *(optional)* `clip` (default `-1`) | — |
| `Abs`/`Abs_` / `Conj`/`Conj_` / `Exp`/`Exp_` / `Norm` | *(none)* | — |

- **Canonical positional rule (§R.0):** the sole operand/parameter is positional;
  this matches the live order and needs no change.
- **`clip` on `Inv`→`reciprocal`/`reciprocal_`** is an operation parameter (the
  small-value guard), positional-optional with default `-1`.
- **No metadata block:** none of these methods take a keyword-only metadata block.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
arithmetic/element-wise surface. Implement Cytnx to match it. Findings above are
the rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing (SciPostPhysCodeb.53).** Element-wise *members* are lowercase
  snake_case. The offenders are the Capitalized `Abs`/`Conj`/`Exp`/`Pow`/`Norm`
  (and their `_` forms) → `abs`/`conj`/`exp`/`pow`/`norm` (T-A3). The **free**
  functions `linalg.Abs`/`Conj`/`Exp`/`Pow`/`Norm` **stay Capitalized** — they
  act on objects (cat 08). `Inv`/`Inv_` are **renamed `reciprocal`/`reciprocal_`**
  instead of just lowercased: cat 08 renames the corresponding free function
  `linalg.Inv`→`Reciprocal` to disambiguate it from the matrix inverse `InvM`, so
  the member side follows suit for **one name per concept** — `Reciprocal` (free,
  Capitalized) / `reciprocal` (member, lowercase) per the N-casing rule (T-A5;
  cross-ref UniTensor UT-A5/UT-X4).
- **N-underscore — a trailing `_` marks in-place (returns `self`); its absence
  marks pure (returns a new object).** Every element-wise op with both modes
  provides both forms: `abs`/`abs_`, `conj`/`conj_`, `exp`/`exp_`, `pow`/`pow_`,
  `reciprocal`/`reciprocal_`. The **`c`-prefixed raw spellings (`cAbs_`,
  `cConj_`, `cExp_`, `cInv_`, `cPow_`, and the `c__i*__` operator backings) are
  rejected** as public API — they are the plumbing the wrappers call (T-A4/A9).
- **In-place methods return `self` from the binding directly.** The in-place
  element-wise methods return `Tensor&` in C++ (verified by the C++ probe, T-A4);
  the pybind lambda must return `&self` too, so the conti.py return-self shims and
  the leaked `c*` bindings disappear. The augmented-assign operators (`+=`, `-=`,
  `*=`, `/=`, `**=`, `@=`) must likewise return `self`, preserving identity (T-A7)
  — which the conti.py wrappers already do (except the broken `@=`, T-A2).
- **Operators mean their Python semantics.** `/` is true division; `//` is
  **floor** division (or is absent) — it must **not** alias true division (T-A1).
  `%` is element-wise modulo — already correctly bound for scalar and tensor
  operands (T-A8; keep). `@=` must be **in place** — fix the misnamed conti.py
  `__imatmul` → `__imatmul__` (T-A2 / B-5). `==` stays element-wise; add a
  whole-tensor `equal` predicate (T-A10). Named element-wise functions (`Add`/
  `Sub`/`Mul`/`Div`/`Mod`) live in `cytnx.linalg` (Capitalized), not as Tensor
  members (T-A6).

## R.1 Recommended API (exact signatures + behavior contract)

```python
class Tensor:
    # --- binary arithmetic operators (pure = new object) ---
    def __add__(self, rhs) -> "Tensor": ...
    def __radd__(self, lhs) -> "Tensor": ...
    def __sub__(self, rhs) -> "Tensor": ...
    def __rsub__(self, lhs) -> "Tensor": ...
    def __mul__(self, rhs) -> "Tensor": ...
    def __rmul__(self, lhs) -> "Tensor": ...
    def __truediv__(self, rhs) -> "Tensor": ...     # true division
    def __rtruediv__(self, lhs) -> "Tensor": ...
    def __floordiv__(self, rhs) -> "Tensor": ...    # FLOOR division (or removed) — NOT true division
    def __rfloordiv__(self, lhs) -> "Tensor": ...
    def __mod__(self, rhs) -> "Tensor": ...         # element-wise modulo (scalar or tensor rhs)
    def __rmod__(self, lhs) -> "Tensor": ...
    def __matmul__(self, rhs: "Tensor") -> "Tensor": ...   # matrix product (linalg.Dot)

    # --- augmented assignment (in place; RETURN self) ---
    def __iadd__(self, rhs) -> "Tensor": ...        # self
    def __isub__(self, rhs) -> "Tensor": ...        # self
    def __imul__(self, rhs) -> "Tensor": ...        # self
    def __itruediv__(self, rhs) -> "Tensor": ...    # self
    def __imatmul__(self, rhs: "Tensor") -> "Tensor": ...  # self  (FIX the __imatmul typo — B-5)

    # --- unary operators ---
    def __neg__(self) -> "Tensor": ...   # element-wise negation (new object)
    def __pos__(self) -> "Tensor": ...   # returns self (unchanged)

    # --- comparison ---
    def __eq__(self, rhs) -> "Tensor": ...   # ELEMENT-WISE -> Bool Tensor (numpy-like)
    def equal(self, other: "Tensor") -> bool: ...   # NEW: whole-tensor equality predicate

    # --- power (pure + in-place) ---
    def __pow__(self, p: float) -> "Tensor": ...
    def __ipow__(self, p: float) -> "Tensor": ...   # self
    def pow(self, p: float) -> "Tensor": ...        # renamed from Pow
    def pow_(self, p: float) -> "Tensor": ...       # renamed from Pow_; self

    # --- element-wise reciprocal (pure + in-place) ---
    def reciprocal(self, clip: float = -1) -> "Tensor": ...   # renamed from Inv (disambiguates from InvM/inv_m)
    def reciprocal_(self, clip: float = -1) -> "Tensor": ...  # renamed from Inv_; self

    # --- conjugate / abs / exp (pure + in-place) ---
    def conj(self) -> "Tensor": ...     # renamed from Conj
    def conj_(self) -> "Tensor": ...    # renamed from Conj_; self
    def abs(self) -> "Tensor": ...      # renamed from Abs
    def abs_(self) -> "Tensor": ...     # renamed from Abs_; self
    def exp(self) -> "Tensor": ...      # renamed from Exp
    def exp_(self) -> "Tensor": ...     # renamed from Exp_; self

    # --- norm ---
    def norm(self) -> "Tensor": ...     # renamed from Norm; scalar 2-norm Tensor
```

In-place methods return `self` **from the binding**; the raw `c*` / `c__i*__`
plumbing bindings become private (leading `_`) or are inlined into the pybind
lambdas — they are **not** public members. The named `Add`/`Sub`/`Mul`/`Div`/
`Mod` stay in `cytnx.linalg` (Capitalized), reached from Python only via the
operators.

| API | Verdict | Behavior contract |
|---|---|---|
| `__add__` / `__radd__` | **keep** (T-A6/A7) | Pure element-wise add; returns a new Tensor. |
| `__sub__` / `__rsub__` | **keep** (T-A6/A7) | Pure element-wise subtract. |
| `__mul__` / `__rmul__` | **keep** (T-A6/A7) | Pure element-wise multiply. |
| `__truediv__` / `__rtruediv__` | **keep** (T-A6) | Pure **true** division. |
| `__floordiv__` / `__rfloordiv__` / `__ifloordiv__` | **fix or remove** (T-A1) | Must implement **floor** division or be removed — must not alias true division. *Migration:* correct the lambda to floor after dividing, or drop the operator with a `DeprecationWarning`. |
| `__mod__` / `__rmod__` | **keep** (T-A8) | Element-wise modulo for a scalar or tensor rhs (Tensor's Dense `Mod` is fully implemented). |
| `__matmul__` | **keep** (T-A2) | Pure matrix product (`linalg.Dot`); new object. |
| `__imatmul__` | **fix the typo, return self** (T-A2 / B-5) | In-place matrix product; must exist as `__imatmul__` (today the conti.py wrapper is misnamed `__imatmul`, so `@=` is not in place) and return self. *Migration:* rename the conti.py attribute (or bind in pybind returning self); a bug fix, no deprecation shim. |
| `__iadd__` / `__isub__` / `__imul__` / `__itruediv__` | **keep, return self** (T-A7) | In-place; return self (identity preserved — the conti.py shim already does this; keep it when the `c__i*__` plumbing is folded away). |
| `__neg__` | **keep** (T-A7) | Element-wise negation; new object. |
| `__pos__` | **keep** (T-A7) | Returns self unchanged (a shared-data handle). |
| `__eq__` | **keep** (T-A10) | Element-wise equality → Bool Tensor (numpy-like); Tensor stays unhashable. |
| `equal` | **add** (T-A10) | NEW: whole-tensor structural equality → Python `bool` (mirrors numpy's `array_equal`). |
| `__pow__` | **keep** (T-A3) | Pure element-wise power; new object. |
| `__ipow__` | **keep, bind self directly** (T-A4) | In-place power; returns self. *Migration:* fold the raw `c__ipow__` into the `__ipow__` lambda. |
| `Pow` → `pow` | **rename** (T-A3) | Pure element-wise power. *Migration:* `DeprecationWarning` alias `Pow` for one release. |
| `Pow_` → `pow_` | **rename, bind self directly** (T-A3/A4) | In-place power; returns self. *Migration:* remove the conti.py shim over `cPow_`; alias `Pow_`. |
| `Inv` → `reciprocal` | **rename (disambiguate from `InvM`/`inv_m`)** (T-A5) | Pure element-wise reciprocal (`1/x`, `clip`-guarded); unifies with the free-function rename `linalg.Inv`→`Reciprocal` (cross-ref UniTensor UT-X4). *Migration:* alias `Inv` for one release. |
| `Inv_` → `reciprocal_` | **rename, bind self directly** (T-A4/A5) | In-place reciprocal; returns self. *Migration:* remove the `cInv_` shim; alias `Inv_`. |
| `Conj` → `conj` | **rename** (T-A3) | Pure complex conjugate. *Migration:* alias `Conj`. |
| `Conj_` → `conj_` | **rename, bind self directly** (T-A3/A4) | In-place conjugate; returns self. *Migration:* remove the `cConj_` shim; alias `Conj_`. |
| `Abs` → `abs` | **rename** (T-A3) | Pure absolute value. *Migration:* alias `Abs`. |
| `Abs_` → `abs_` | **rename, bind self directly** (T-A3/A4) | In-place abs; returns self. *Migration:* remove the `cAbs_` shim; alias `Abs_`. |
| `Exp` → `exp` | **rename** (T-A3) | Pure element-wise exponential. *Migration:* alias `Exp`. |
| `Exp_` → `exp_` | **rename, bind self directly** (T-A3/A4) | In-place exp; returns self. *Migration:* remove the `cExp_` shim; alias `Exp_`. |
| `Norm` → `norm` | **rename** (T-A3) | Returns the 2-norm as a scalar `Tensor`. *Migration:* alias `Norm`. |

**C++-only — the operator implementations (no separate Python member).** The
named arithmetic methods below exist in C++ and are reached from Python only
through the operator dunders / `cytnx.linalg`; they carry a C++ (R.2b) docstring
only (documented fully in cat 08).

| API | Verdict | Behavior contract |
|---|---|---|
| `Add` / `Sub` / `Mul` / `Div` (+ `_` forms), `Cpr`, `Mod` | **keep (C++-only)** (T-A6) | Element-wise binary ops behind the operators; exposed to Python as `linalg.Add`/… (Capitalized free functions, cat 08) and the operator dunders — not as Tensor members. |

**Internal / plumbing — hidden, not public API.** The raw bindings below are live
public members today with a **remove** verdict: hide them behind a leading
underscore or inline them into their pybind lambda. None carry a docstring — they
are not public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `cAbs_` | **remove** (T-A4/A9) | Raw plumbing (C++ `Abs_`) behind `Abs_`/`abs_`. *Migration:* fold into the `abs_` pybind lambda (returns self); no public exposure. |
| `cConj_` | **remove** (T-A4/A9) | Raw plumbing (C++ `Conj_`) behind `Conj_`/`conj_`. *Migration:* fold into the `conj_` lambda. |
| `cExp_` | **remove** (T-A4/A9) | Raw plumbing (C++ `Exp_`) behind `Exp_`/`exp_`. *Migration:* fold into the `exp_` lambda. |
| `cInv_` | **remove** (T-A4/A5/A9) | Raw plumbing (C++ `Inv_`) behind `Inv_`/`reciprocal_`. *Migration:* fold into the `reciprocal_` lambda. |
| `cPow_` | **remove** (T-A4/A9) | Raw plumbing (C++ `Pow_`) behind `Pow_`/`pow_`/`__ipow__`. *Migration:* fold into the `pow_`/`__ipow__` lambdas. |
| `c__iadd__` / `c__isub__` / `c__imul__` / `c__itruediv__` / `c__ifloordiv__` | **remove** (T-A7/A9) | Raw plumbing (C++ `Add_`/`Sub_`/`Mul_`/`Div_`) behind `__iadd__`/…/`__ifloordiv__`. *Migration:* fold into each augmented-assign pybind lambda (which returns self). |
| `c__ipow__` | **remove** (T-A4/A9) | Raw plumbing (`linalg::Pow_`) behind `__ipow__`. *Migration:* fold into the `__ipow__` lambda. |
| `c__imatmul__` | **remove** (T-A2/A9) | Raw plumbing (`self = linalg::Dot(self,rhs)`) behind the (fixed) `__imatmul__`. *Migration:* fold into the `__imatmul__` lambda; the misnamed conti.py `__imatmul` wrapper is deleted. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept/renamed/added members
are documented (removed plumbing carries no docstring); the C++-only `Add`/`Sub`/
`Mul`/`Div`/`Mod` are documented in cat 08.

### R.2a Python API (numpy-style)

### arithmetic operators (`__add__` / `__sub__` / `__mul__` / `__truediv__` / `__floordiv__` / `__mod__` / `__matmul__` / `__neg__` / `__pos__` / `__eq__` / `__iadd__` and their `r`/`i` forms)

```
t + v,  t - v,  t * v,  t / v,  t // v,  t % v        -> Tensor (pure)
t @ v                                                 -> Tensor (matrix product)
t += v, t -= v, t *= v, t /= v, t @= v                -> Tensor (in place, self)
-t,     +t                                            -> Tensor
t == v                                                -> Tensor (Bool, element-wise)

Element-wise arithmetic on Tensors and scalars.

The binary operators (`+`, `-`, `*`, `/`) return a NEW Tensor; the augmented
forms (`+=`, `-=`, `*=`, `/=`, `@=`) mutate this tensor IN PLACE and return self
(finding T-A7). `-t` negates element-wise; `+t` returns this tensor unchanged
(a shared-data handle).

`/` is TRUE division. `//` is FLOOR division — through cytnx 1.1.0 `//` wrongly
performed true division (finding T-A1); the next version floors (or removes
`//`). `%` is element-wise modulo for a scalar OR tensor right-hand operand
(finding T-A8). `@` is the matrix product (`linalg.Dot`); `@=` does it IN PLACE —
through cytnx 1.1.0 `@=` was NOT in place (a misnamed wrapper, finding T-A2 /
B-5), the next version fixes it.

`==` is ELEMENT-WISE and returns a Bool Tensor (numpy-like), which makes Tensor
unhashable; use `equal(other)` for a whole-tensor boolean (finding T-A10).

The right-hand operand may be a Tensor or any of the 11 element scalar dtypes.
The named forms `Add`/`Sub`/`Mul`/`Div`/`Mod` live in `cytnx.linalg`
(Capitalized) — the operators are the sole Tensor-member surface (finding T-A6).

Returns
-------
Tensor
    Binary/unary operators: a new tensor. Augmented assignment: self. `==`: a
    Bool tensor.
```

### `pow` / `pow_` / `__pow__` / `__ipow__`

```
Tensor.pow(p)     -> Tensor    # pure element-wise power (renamed from Pow)
Tensor.pow_(p)    -> Tensor    # in-place, self (renamed from Pow_)
t ** p            -> Tensor    # pure
t **= p           -> Tensor    # in-place, self

Raise every element of this Tensor to the power `p`.

`pow`/`t ** p` are PURE and return a new tensor; `pow_`/`t **= p` raise the
elements IN PLACE and return self (finding T-A4).

Parameters
----------
p : float
    The exponent.

Returns
-------
Tensor
    `pow`/`**`: a new tensor. `pow_`/`**=`: self.

Notes
-----
Renamed from the Capitalized `Pow`/`Pow_` (finding T-A3); the in-place forms no
longer route through the leaked raw `cPow_`/`c__ipow__` bindings (finding T-A4).
`Pow`/`Pow_` remain `DeprecationWarning` aliases for one release.
```

### `reciprocal` / `reciprocal_`

```
Tensor.reciprocal(clip=-1)    -> Tensor    # pure element-wise reciprocal (renamed from Inv)
Tensor.reciprocal_(clip=-1)   -> Tensor    # in-place, self (renamed from Inv_)

Element-wise reciprocal (1/x) of this Tensor.

`reciprocal` is PURE and returns a new tensor; `reciprocal_` inverts IN PLACE
and returns self (finding T-A4).

Parameters
----------
clip : float, optional
    Guard for small magnitudes: elements with |x| <= clip are set to 0
    (pseudo-inverse) rather than blowing up (default -1 = no clip).

Returns
-------
Tensor
    `reciprocal`: a new tensor. `reciprocal_`: self.

Notes
-----
Renamed from the Capitalized `Inv`/`Inv_` (finding T-A5), not just lowercased:
the new name disambiguates the ELEMENT-WISE reciprocal from the MATRIX inverse
`inv_m` (renamed from `InvM`, cat 06), and unifies with the free-function rename
`linalg.Inv`→`Reciprocal` (cross-ref UniTensor UT-X4) — one name per concept.
`Inv`/`Inv_` remain `DeprecationWarning` aliases for one release.
```

### `conj` / `conj_` / `abs` / `abs_` / `exp` / `exp_`

```
Tensor.conj()   -> Tensor    # pure conjugate     (renamed from Conj)
Tensor.conj_()  -> Tensor    # in-place, self     (renamed from Conj_)
Tensor.abs()    -> Tensor    # pure |x|           (renamed from Abs)
Tensor.abs_()   -> Tensor    # in-place, self     (renamed from Abs_)
Tensor.exp()    -> Tensor    # pure exp(x)        (renamed from Exp)
Tensor.exp_()   -> Tensor    # in-place, self     (renamed from Exp_)

Element-wise complex conjugate, absolute value, and exponential.

Each has a PURE form (new object) and an IN-PLACE form returning self
(finding T-A4):

conj : complex-conjugate every element.
abs  : |x| of every element.
exp  : exp(x) of every element.

Returns
-------
Tensor
    Pure form: a new tensor. In-place (`_`) form: self.

Notes
-----
Renamed from the Capitalized `Conj`/`Abs`/`Exp` (+ `_` forms, finding T-A3); the
in-place forms no longer route through the leaked raw `cConj_`/`cAbs_`/`cExp_`
bindings (finding T-A4). The old Capitalized names remain `DeprecationWarning`
aliases for one release. The Capitalized *free* functions `linalg.Conj`/… stay
Capitalized (cat 08).
```

### `norm` / `equal`

```
Tensor.norm()          -> Tensor    # 2-norm scalar (renamed from Norm)
Tensor.equal(other)    -> bool      # NEW: whole-tensor equality

`norm` returns the Frobenius/2-norm as a scalar `cytnx.Tensor` (shape [1],
finding T-A3). `equal` tests whether two tensors are element-wise equal and
returns a single Python bool — the whole-tensor complement to the element-wise
`==` operator, which returns a Bool Tensor and leaves `Tensor` unhashable
(finding T-A10).

Parameters
----------
other : Tensor
    (`equal` only) the tensor to compare against.

Returns
-------
Tensor (`norm`) or bool (`equal`)

Notes
-----
`norm` is renamed from the Capitalized `Norm` (finding T-A3; `DeprecationWarning`
alias for one release). `equal` is newly added (finding T-A10; mirrors numpy's
`array_equal`).
```

### R.2b C++ API (Doxygen)

C++ already returns `Tensor&`/`Tensor` per the N-underscore split (the C++ probe
confirms `Conj_`/`Abs_`/`Exp_`/`Pow_`/`Inv_` return `Tensor&`, T-A4); the next
version must have the *pybind lambdas* return these directly (removing the
conti.py shims and the leaked `c*` bindings, T-A4/A9), wire `//` to floor (or
drop it, T-A1), fix the in-place matmul wrapper name (`__imatmul` → `__imatmul__`,
T-A2 / B-5), and add a whole-tensor `equal` predicate (T-A10). The member names
are lowercased in the Python binding (with `Inv` additionally renamed
`reciprocal` to disambiguate from the matrix inverse `InvM`/`inv_m`, T-A5) while
the C++ method names keep their capitalization; the `linalg` free functions
likewise keep Capitalized names, with `Inv`→`Reciprocal` renamed in step (cat 08).

```cpp
/**
 * @brief Element-wise power (pure and in-place).
 * @details Pow(p) raises every element to p and returns a NEW Tensor; Pow_(p)
 *          does so in place and returns *this. The Python binding exposes these
 *          as pow/pow_ and __pow__/__ipow__, returning self directly from the
 *          in-place lambda (dropping the leaked cPow_/c__ipow__ bindings,
 *          findings T-A3/A4).
 * @param p the exponent.
 * @return Pow: a new Tensor. Pow_: reference to *this.
 */
Tensor Pow(const cytnx_double &p) const;
Tensor &Pow_(const cytnx_double &p);

/**
 * @brief Element-wise reciprocal (1/x), pure and in-place.
 * @details Inv(clip) returns a NEW Tensor; Inv_(clip) inverts in place and
 *          returns *this. `clip` guards small magnitudes. The Python binding
 *          exposes reciprocal/reciprocal_ — renamed from Inv/Inv_ (not merely
 *          lowercased) to disambiguate from the MATRIX inverse InvM/inv_m,
 *          matching the free-function rename linalg.Inv->Reciprocal (cross-ref
 *          UniTensor UT-X4, finding T-A5).
 * @param clip small-magnitude guard (-1 = none).
 * @return Inv: a new Tensor. Inv_: reference to *this.
 */
Tensor Inv(const double &clip = -1.) const;
Tensor &Inv_(const double &clip = -1.);

/**
 * @brief Complex conjugate / absolute value / exponential, pure and in-place.
 * @details Each pure form returns a NEW Tensor; each in-place form returns *this
 *          (finding T-A4, confirmed by the C++ probe: &Conj_()==&self, etc.).
 *          The Python binding exposes conj/conj_, abs/abs_, exp/exp_ and returns
 *          *this directly from the in-place lambdas (dropping the leaked
 *          cConj_/cAbs_/cExp_ bindings).
 * @return pure form: a new Tensor. in-place (_) form: reference to *this.
 */
Tensor Conj() const;   Tensor &Conj_();
Tensor Abs() const;    Tensor &Abs_();
Tensor Exp() const;    Tensor &Exp_();

/**
 * @brief The 2-norm as a scalar Tensor.
 * @details Norm() returns the Frobenius/2-norm as a rank-1 (shape [1]) Tensor
 *          (finding T-A3). Python exposes it as norm().
 * @return a scalar Tensor holding the 2-norm.
 */
Tensor Norm() const;

/**
 * @brief Element-wise binary arithmetic (behind the Python operators).
 * @details Add/Sub/Mul/Div (and the in-place Add_/…, returning *this) implement
 *          the Python +, -, *, / operators; the Python augmented-assign lambdas
 *          must return *this (finding T-A7 — the conti.py wrappers already do).
 *          The Python `/` is true division; `//` must floor or be dropped (must
 *          NOT alias Div, finding T-A1). Mod implements `%` — bound for scalar
 *          AND tensor operands and fully implemented for Dense (finding T-A8).
 *          operator== is element-wise (finding T-A10). matmul is linalg::Dot;
 *          the in-place `@=` must fix the misnamed conti.py `__imatmul` wrapper
 *          (finding T-A2 / B-5). These named functions are documented fully in
 *          cat 08.
 * @return pure ops: a new Tensor. in-place ops: reference to *this.
 */
Tensor Add(const T &rhs);   Tensor &Add_(const T &rhs);
Tensor Sub(const T &rhs);   Tensor &Sub_(const T &rhs);
Tensor Mul(const T &rhs);   Tensor &Mul_(const T &rhs);
Tensor Div(const T &rhs);   Tensor &Div_(const T &rhs);
Tensor Mod(const T &rhs);   Tensor Cpr(const T &rhs);
// free function (namespace cytnx::linalg):
Tensor Dot(const Tensor &Tl, const Tensor &Tr);   // linalg.hpp:2482 (matmul / @)
```
