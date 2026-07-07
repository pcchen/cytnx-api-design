# Tensor — private / plumbing surface

The complete classification of every name in `dir(cytnx.Tensor)` that is **not
part of the recommended public API** (method spec §4.4). It is the single home
for the private surface — `inventory.md`'s Internal/plumbing section points here.

Membership is verified against the live `cytnx==1.1.0` wheel by
`tools/validate_doc.py`'s **N-private accounting gate** (§10): every name in the
*hide* table below must be present in `dir(Tensor)`, must not also carry a
`keep`/`add`/`rename` verdict, and every leak-shaped member of `dir()` must
appear here (nothing un-audited). The gate also prints the **leaked-public
count** — the N-private redesign metric (§5.6), today **16**, target **0**.

`dir(cytnx.Tensor)` = 122 names: **67** non-underscore (51 recommended public
API + **16 leaked internals**) and 55 underscore (2 single-underscore internals
+ 53 dunders). Unlike UniTensor (whose leak set had `c`+lowercase spellings —
`cnormalize_`/`ctag`/`ctruncate_` — and a pattern-less `cfrom` the naive scan
missed), Tensor's 16 leaks are **all** caught by the validator's completeness
net directly: they are exactly the `c[A-Z]…` raw in-place bindings, the
`c__i…__` operator bindings, the two `*_different_*` conversion shims, and
`make_contiguous`. The lowercase `c`-prefixed public wrappers here — `clone`,
`contiguous`, `contiguous_`, `is_contiguous` — are genuine public API (their
suffixes are not existing methods), so the net correctly leaves them out; there
is **no** Tensor analog of UniTensor's net-missed `cfrom`.

## Leaked internals — hide

Non-underscore members that are raw pybind bindings (`cytnx_src/pybind/tensor_py.cpp`)
or `Tensor_conti.py` shims: they exist only so a public Python wrapper can call
them, yet they leak into `dir()`. **Fix (N-private):** inline each into the
pybind lambda (which should itself return `self`), so the `c*`/shim name
disappears; or, if it must stay a separate binding, bind it under a leading `_`.

| Member | What it is | Used by (public wrapper) | Hide fix |
|---|---|---|---|
| `cConj_` | raw in-place complex conjugate | `Conj_` (→ `conj_`) via `Tensor_conti.py` | inline into the `conj_` lambda (return self) |
| `cExp_` | raw in-place exponential | `Exp_` (→ `exp_`) | inline into `exp_` |
| `cAbs_` | raw in-place absolute value | `Abs_` (→ `abs_`) | inline into `abs_` |
| `cInv_` | raw in-place element-wise reciprocal | `Inv_` (→ `reciprocal_`) | inline into `reciprocal_` |
| `cPow_` | raw in-place power | `Pow_` (→ `pow_`) | inline into `pow_` |
| `cInvM_` | raw in-place **matrix** inverse (cat 06) | `InvM_` (→ `inv_m_`) | inline into `inv_m_` |
| `c__iadd__` | raw in-place `+=` | `__iadd__` | inline into `__iadd__` (return self) |
| `c__isub__` | raw in-place `-=` | `__isub__` | inline into `__isub__` |
| `c__imul__` | raw in-place `*=` | `__imul__` | inline into `__imul__` |
| `c__itruediv__` | raw in-place `/=` | `__itruediv__` | inline into `__itruediv__` |
| `c__ifloordiv__` | raw in-place `//=` | `__ifloordiv__` | inline into `__ifloordiv__` |
| `c__ipow__` | raw in-place `**=` | `__ipow__` | inline into `__ipow__` |
| `c__imatmul__` | raw in-place matmul | the (broken) `@=` — routed through the **misspelled** `__imatmul` wrapper (T-A2/B-5), so `@=` never fires | inline into a correctly-named `__imatmul__` lambda (return self) |
| `make_contiguous` | raw new-contiguous binding (no no-op short-circuit) | `contiguous` (via `Tensor_conti.py`, which adds the `is_contiguous()` short-circuit) | inline (with the short-circuit) into the `contiguous` lambda |
| `astype_different_dtype` | raw dtype-convert shim — **raises** on a same-dtype no-op | `astype` (via `Tensor_conti.py`, which adds the `is self` short-circuit) | inline (with the short-circuit) into `astype` |
| `to_different_device` | raw device-move shim — **raises** on a same-device no-op | `to` (via `Tensor_conti.py`, which adds the short-circuit) | inline (with the short-circuit) into `to` |

*(16 members — the full leaked-public set. All are pure plumbing: none carries a
`keep`/`add`/`rename` verdict in any category doc; each is bound only so its
Capitalized/`__i…__` public wrapper can delegate to it.)*

### Single-underscore internals (already private-ish, listed for completeness)

`__imatmul` — the `Tensor_conti.py` shim intended to implement `@=` by calling
`c__imatmul__`; it is **misspelled** (missing the trailing `__`), so Python's
data model never invokes it and `@=` silently falls back to `__matmul__`+rebind
(the T-A2/B-5 bug). Because it begins with `__` but is not a dunder, it is not
counted in the leaked-public 16. `_pybind11_conduit_v1_` — pybind11 framework
machinery. **Fix:** `__imatmul` disappears when `c__imatmul__` is inlined into a
correctly-named `__imatmul__`; `_pybind11_conduit_v1_` is framework-provided
(accept as-is).

## User-facing dunders — keep

The protocol dunders that implement user syntax — already specced in the owning
category docs (03 indexing/iteration, 05 arithmetic, 07 copy, 08 pickle), listed
here so the surface is complete, not re-audited:
`__init__`, `__getitem__`, `__setitem__`, `__iter__`, `__len__`,
`__add__`/`__radd__`/`__iadd__`, `__sub__`/`__rsub__`/`__isub__`,
`__mul__`/`__rmul__`/`__imul__`, `__truediv__`/`__rtruediv__`/`__itruediv__`,
`__floordiv__`/`__rfloordiv__`/`__ifloordiv__`, `__mod__`/`__rmod__`,
`__neg__`, `__pos__`, `__pow__`/`__ipow__`, `__matmul__` (`@` — but note **no**
`__imatmul__`, so `@=` is broken; see single-underscore `__imatmul` above),
`__eq__`, `__ne__`, `__hash__`, `__repr__`, `__str__`,
`__copy__`, `__deepcopy__`, `__getstate__`.

## Framework defaults — ignore

The universal Python/pybind object dunders — not Cytnx surface, not renamable,
excluded by rule (not enumerated per-member): `__class__`, `__delattr__`,
`__dir__`, `__doc__`, `__format__`, `__ge__`/`__gt__`/`__le__`/`__lt__`,
`__getattribute__`, `__setattr__`, `__init_subclass__`, `__new__`,
`__reduce__`/`__reduce_ex__`, `__sizeof__`, `__subclasshook__`, `__module__`.
(`__reduce__`/`__reduce_ex__` participate in the pickle protocol handled in
cat 08.)
