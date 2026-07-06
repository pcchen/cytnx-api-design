# UniTensor — private / plumbing surface

The complete classification of every name in `dir(cytnx.UniTensor)` that is **not
part of the recommended public API** (method spec §4.4). It is the single home
for the private surface — `inventory.md`'s Internal/plumbing section points here.

Membership is verified against the live `cytnx==1.1.0` wheel by
`tools/validate_doc.py`'s **N-private accounting gate** (§10): every name in the
*hide* table below must be present in `dir(UniTensor)`, must not also carry a
`keep`/`add`/`rename` verdict, and every leak-shaped member of `dir()` must
appear here (nothing un-audited). The gate also prints the **leaked-public
count** — the N-private redesign metric (§5.6), today **21**, target **0**.

`dir(cytnx.UniTensor)` = 179 names: **126** non-underscore (105 recommended
public API + **21 leaked internals**) and 53 underscore (5 single-underscore
internals + 48 dunders). (Three — `cnormalize_`/`ctag`/`ctruncate_` — are `c`+lowercase wrappers a naive
`^c[A-Z]` scan misses; the validator's completeness net also checks `c`+existing-
method, catching those. A fourth, `cfrom`, matches no name pattern at all and is
caught only by its category finding (cat-12 UT-T7) — the net is a safety net for
the common `c*`/shim spellings, not a substitute for the per-category analysis.)

## Leaked internals — hide

Non-underscore members that are raw pybind bindings or `*_conti.py` shims: they
exist only so a public Python wrapper can call them, yet they leak into `dir()`.
**Fix (N-private):** inline each into the pybind lambda (which should itself
return `self`), so the `c*` name disappears; or, if it must stay a separate
binding, bind it under a leading `_`.

| Member | What it is | Used by (public wrapper) | Hide fix |
|---|---|---|---|
| `cConj_` | raw in-place conjugate | `Conj_` (→ `conj_`) | inline into the `conj_` lambda (return self) |
| `cDagger_` | raw in-place dagger | `Dagger_` (→ `dagger_`) | inline into `dagger_` |
| `cTranspose_` | raw in-place transpose | `Transpose_` (→ `transpose_`) | inline into `transpose_` |
| `cTrace_` | raw in-place trace | `Trace_` (→ `trace_`) | inline into `trace_` |
| `cPow_` | raw in-place power | `Pow_` (→ `pow_`) | inline into `pow_` |
| `cnormalize_` | raw in-place normalize | `normalize_` | inline into `normalize_` |
| `cInv_` | raw in-place reciprocal | *(no public wrapper — UT-A5 gap)* | expose as public `reciprocal_`, drop `cInv_` |
| `c__ipow__` | raw in-place `**=` | `__ipow__` | inline into `__ipow__` |
| `c_at` | raw element accessor | `at` (Hclass proxy) | inline the proxy build into the `at` lambda |
| `c_relabel_` | raw in-place relabel | `relabel_` | inline (return self) |
| `c_relabels_` | raw in-place relabels (dep) | `relabels_` (→ remove) | removed with `relabels_` |
| `c_set_label` | raw single-label setter | `set_label` (→ remove) | removed with `set_label` |
| `c_set_labels` | raw label-list setter (dep) | `set_labels` (→ remove) | removed with `set_labels` |
| `c_set_name` | raw name setter | `set_name`/`set_name_` | inline into `set_name_` |
| `c_set_rowrank_` | raw in-place rowrank setter | `set_rowrank_` | inline into `set_rowrank_` |
| `ctag` | raw tag binding | `tag` (cat-05 UT-S6) | inline into the `tag` lambda |
| `ctruncate_` | raw in-place truncate | `truncate_` (cat-05 UT-S6) | inline into `truncate_` |
| `make_contiguous` | raw new-contiguous binding | `contiguous` | inline (with the no-op short-circuit) into the `contiguous` lambda |
| `astype_different_type` | raw dtype-convert shim | `astype` | inline (with the `is self` short-circuit) into `astype` |
| `to_different_device` | raw device-move shim | `to` | inline (with the short-circuit) into `to` |
| `cfrom` | raw convert binding | `convert_from` (cat-12 UT-T7) | inline into `convert_from` (returns self) |

*(21 members. The `remove`-verdict rows above for the deprecated public wrappers
— `set_label`/`set_labels`/`relabels_` — are handled in cat-04's R.1; only the
raw `c_*`/shim bindings are "hide" here.)*

### Single-underscore internals (already private-ish, listed for completeness)

`_at`, `_relabel_`, `_relabels_`, `_set_label` — beartype-decorated variants of
the `c_*` bindings (already `_`-prefixed, so not counted in the leaked-public
21); `_pybind11_conduit_v1_` — pybind11 framework machinery. **Fix:** these
disappear when the corresponding `c_*` are inlined; `_pybind11_conduit_v1_` is
framework-provided (accept as-is).

## User-facing dunders — keep

The protocol dunders that implement user syntax — already specced in the category
docs, listed here so the surface is complete, not re-audited:
`__init__`, `__getitem__`, `__setitem__`, `__add__`/`__radd__`/`__iadd__`,
`__sub__`/…, `__mul__`/…, `__truediv__`/…, `__floordiv__`/…, `__neg__`,
`__pos__`, `__pow__`, `__ipow__`, `__eq__`, `__ne__`, `__repr__`, `__str__`,
`__copy__`, `__deepcopy__`, `__getstate__` (broken — UT-IO2), `__matmul__`.

## Framework defaults — ignore

The universal Python/pybind object dunders — not Cytnx surface, not renamable,
excluded by rule (not enumerated per-member): `__class__`, `__delattr__`,
`__dir__`, `__doc__`, `__format__`, `__ge__`/`__gt__`/`__le__`/`__lt__`,
`__getattribute__`, `__setattr__`, `__init_subclass__`, `__new__`,
`__reduce__`/`__reduce_ex__`, `__sizeof__`, `__subclasshook__`, `__module__`.
(`__reduce__`/`__reduce_ex__` participate in the broken pickle protocol —
see UT-IO2.)
