# Tensor — categorized member inventory

> The master member list for `cytnx.Tensor` (wheel **1.1.0**) — the dense
> N-dimensional array that `UniTensor` wraps for its Dense storage backend.
> Every public member of `dir(cytnx.Tensor)` (**67** non-underscore names) is
> assigned to **exactly one** of the eight categories below — the categories
> *partition* the live surface. This file is the index; the per-category
> documents hold the analysis, the normative spec (`# R.`), and the
> probe-backed evidence.

**How to read this.** Each `## NN — <category>` section links its category
document and lists that category's live members with a one-line role. Names
tagged **(plumbing)** are raw C++/pybind bindings that leak into the public
namespace but are *not* public API — they are collected again, with their
public wrapper, in [`## Internal / plumbing`](#internal--plumbing-not-public-api)
at the end. The dunder members (`__getitem__`/`__setitem__`, the operator
dunders, `__repr__`, the pickle/copy hooks) are handled inside their owning
category document (03, 05, 07, 08) and are not repeated here.

**Provenance / gates.**
- Category documents: `docs/api-audit/Tensor/NN-*.md` (01–08).
- Method spec: `docs/superpowers/specs/2026-07-06-cytnx-api-analysis-method-design.md`.
- Rollout plan: `docs/superpowers/plans/2026-07-07-cytnx-tensor-method-rollout.md`.
- Coverage is machine-enforced: `python tools/validate_doc.py Tensor docs/api-audit/Tensor/`
  passes iff every live member is covered by some category's `# R.` section —
  confirmed: `PASS: Tensor — 67 members covered across 8 files`.

---

## 01 — construction & init

[`01-construction-init.md`](01-construction-init.md) — building a `Tensor`
from a shape, another tensor, or a `Storage` (the constructor overloads and
`from_storage`).

| Member | Role |
|---|---|
| `Init` | Public in-place re-initializer duplicating the shape constructor (→ demote to `_init`). |
| `from_storage` | Static: wrap a `Storage` (default **shares** its buffer; `is_clone=True` copies). |

## 02 — metadata & introspection

[`02-metadata-introspection.md`](02-metadata-introspection.md) — read-only
queries on structure, element/device identity, and storage predicates. Already
the cleanest category (no casing findings — contrast UniTensor's `Nblocks`).

| Member | Role |
|---|---|
| `shape` | Extent of each axis. |
| `rank` | Number of axes (`== len(shape())`). |
| `dtype` | Element-type code. |
| `dtype_str` | Element-type name. |
| `device` | Device code. |
| `device_str` | Device name. |
| `is_contiguous` | Predicate: element storage is contiguous. |
| `same_data` | Predicate: do two tensors share storage? (the view-vs-copy oracle; erased param name `arg0` → fix to `other`). |

## 03 — element & storage access

[`03-element-storage-access.md`](03-element-storage-access.md) — scalar
extraction, the backing `Storage`/numpy bridge, complex real/imag, in-place
writers, and the indexing operators (`__getitem__`/`__setitem__`, documented
there).

| Member | Role |
|---|---|
| `item` | Extract the sole scalar of a 1-element tensor (all 11 dtypes). |
| `storage` | The backing `Storage` — a **shared-data view**. |
| `numpy` | Export to a numpy `ndarray` — **always a copy** (the `share_mem=True` zero-copy promise is non-functional). |
| `real` | Real part of a complex tensor (copy). |
| `imag` | Imaginary part of a complex tensor (copy). |
| `fill` | Set every element in place (no trailing `_`, an established exception). |
| `append` | Grow along axis 0 in place, scalar/`Tensor`/`Storage` (no trailing `_`). |

## 04 — shape / layout

[`04-shape-layout.md`](04-shape-layout.md) — axis reordering, shape change,
memory-layout coalescing, and the 1-D collapse.

| Member | Role |
|---|---|
| `permute` | Reorder axes — returns a shared-data **view**. |
| `permute_` | Reorder axes in place, returns self. |
| `reshape` | Change shape — returns a shared-data **view** (not a copy). |
| `reshape_` | Reshape in place, returns self. |
| `contiguous` | Coalesce storage: self if already contiguous, else an independent copy. |
| `contiguous_` | Coalesce in place — but returns a **distinct** shared-data wrapper, not self (C++-rooted, `Tensor` by value). |
| `flatten` | Collapse to rank 1 — an **independent copy** (clone + contiguous + reshape). |
| `flatten_` | Collapse to rank 1 in place — returns **`None`**, not self (C++-rooted, `void`). |
| `make_contiguous` | **(plumbing)** raw binding under `contiguous`; does not short-circuit like the public wrapper. |

## 05 — arithmetic & element-wise

[`05-arithmetic-elementwise.md`](05-arithmetic-elementwise.md) — the operator
dunders (documented there) and the Capitalized element-wise **members**
(`Abs`/`Conj`/`Exp`/`Inv`/`Pow`/`Norm` → lowercase), plus their leaked raw `c*`
plumbing.

| Member | Role |
|---|---|
| `Abs` | Element-wise `\|x\|` (→ rename `abs`). |
| `Abs_` | In-place abs (→ rename `abs_`). |
| `Conj` | Complex conjugate (→ rename `conj`). |
| `Conj_` | In-place conjugate (→ rename `conj_`). |
| `Exp` | Element-wise exponential (→ rename `exp`). |
| `Exp_` | In-place exponential (→ rename `exp_`). |
| `Inv` | **Element-wise** reciprocal `1/x` — distinct from the matrix inverse `InvM` (cat 06) (→ rename `reciprocal`, disambiguating). |
| `Inv_` | In-place element-wise reciprocal (→ rename `reciprocal_`). |
| `Pow` | Element-wise power (→ rename `pow`). |
| `Pow_` | In-place power (→ rename `pow_`). |
| `Norm` | The 2-norm as a scalar `Tensor` (→ rename `norm`). |
| `cAbs_` | **(plumbing)** raw binding under `Abs_`. |
| `cConj_` | **(plumbing)** raw binding under `Conj_`. |
| `cExp_` | **(plumbing)** raw binding under `Exp_`. |
| `cInv_` | **(plumbing)** raw binding under `Inv_`. |
| `cPow_` | **(plumbing)** raw binding under `Pow_`. |
| `c__iadd__` | **(plumbing)** raw binding under `__iadd__`. |
| `c__ifloordiv__` | **(plumbing)** raw binding under `__ifloordiv__`. |
| `c__imatmul__` | **(plumbing)** raw binding for the (misnamed, see T-A2/B-5) `__imatmul__`. |
| `c__imul__` | **(plumbing)** raw binding under `__imul__`. |
| `c__ipow__` | **(plumbing)** raw binding under `__ipow__`. |
| `c__isub__` | **(plumbing)** raw binding under `__isub__`. |
| `c__itruediv__` | **(plumbing)** raw binding under `__itruediv__`. |

## 06 — linear algebra (member) & reductions

[`06-linalg-reductions.md`](06-linalg-reductions.md) — the Capitalized member
decompositions, matrix inverse, and reductions (mirroring the `cytnx.linalg`
free functions of the same names, which stay Capitalized).

| Member | Role |
|---|---|
| `Svd` | Singular-value decomposition of a rank-2 tensor, `[S, U, vT]` S-first (→ rename `svd`). |
| `Eigh` | Hermitian eigendecomposition, `[eigvals, (eigvecs)]` (→ rename `eigh`). |
| `InvM` | **Matrix** inverse (pure) — distinct from the element-wise `Inv` (cat 05) (→ rename `inv_m`). |
| `InvM_` | In-place matrix inverse (→ rename `inv_m_`). |
| `Trace` | Sum of the diagonal over two axes — C++ defaults `a=0,b=1` dropped, params erased to `arg0`/`arg1` (→ rename `trace`, restore defaults+names). |
| `Max` | Maximum element, as a scalar `Tensor` (→ rename `max`). |
| `Min` | Minimum element, as a scalar `Tensor` (→ rename `min`). |
| `cInvM_` | **(plumbing)** raw binding under `InvM_`. |

## 07 — type & device conversion

[`07-type-device-conversion.md`](07-type-device-conversion.md) — element-type
conversion, device move, and deep copy (the direct `Tensor` analog of
[UniTensor cat 12](../UniTensor/12-type-device-conversion.md)).

| Member | Role |
|---|---|
| `astype` | Convert element dtype — short-circuits to `self` on a no-op (binding-introduced, cross-ref UniTensor UT-T1). |
| `to` | Move to a device (pure) — same `is self` no-op short-circuit. |
| `to_` | Move to a device in place — returns **`None`**, not self (C++ `void to_`; contrast `UniTensor.to_`, which returns self); parameter is correctly named `device` (positive contrast to UniTensor's erased `arg0`). |
| `clone` | Independent deep copy. |
| `astype_different_dtype` | **(plumbing)** raw shim under `astype`; raises on a no-op. |
| `to_different_device` | **(plumbing)** raw shim under `to`; raises on a no-op. |

## 08 — I/O

[`08-io.md`](08-io.md) — persistence to the structured `.cytn` binary format
and the raw headerless path.

| Member | Role |
|---|---|
| `Save` | Serialize to a `.cytn` file (→ rename `save`); no-extension auto-appends `.cytn` (deprecated), asymmetric with `Load`. |
| `Load` | Static: load a `Tensor` from a `.cytn` file (→ rename `load`); round-trips cleanly — a dense `Tensor` has no name field, so no UniTensor UT-IO5 analog. |
| `Tofile` | Write raw, headerless element bytes (→ rename `tofile`); no shape/dtype stored. |
| `Fromfile` | Static: read headerless bytes into a **flat rank-1** tensor, `dtype` required (→ rename `fromfile`). |

---

## Internal / plumbing (not public API)

The private/plumbing surface — the raw `c*`/shim bindings that leak into
`dir(cytnx.Tensor)` — is analyzed in full in
[`private-surface.md`](private-surface.md): every leaked member classified
with its public wrapper and hide fix (the ~16 leaked bindings — `cAbs_`,
`cConj_`, `cExp_`, `cInv_`, `cPow_`, `c__iadd__`, `c__ifloordiv__`,
`c__imatmul__`, `c__imul__`, `c__ipow__`, `c__isub__`, `c__itruediv__`,
`cInvM_`, `make_contiguous`, `astype_different_dtype`, `to_different_device`),
the single-underscore internals, and the user-facing vs framework dunders.
That document is the single home for the private surface, machine-enforced by
the N-private accounting gate in `tools/validate_doc.py`. This inventory lists
each leaked name once, in its owning category above, with a **(plumbing)**
tag — it does not duplicate `private-surface.md`'s classification.

## Notable findings (cross-class contrasts)

A few headline findings make this inventory useful as a standalone index,
beyond the per-category detail:

- **T-E1 / T-E1a** — `Tensor` **has** a numpy bridge (`numpy()` +
  `from_storage`) that `UniTensor` entirely **lacks** — closing the gap
  UniTensor's audit flagged as missing (UT-C3/UT-T6). `numpy()` is always a
  copy; the `share_mem=True` zero-copy promise is non-functional (T-E1).
- **T-A2 (B-5)** — `t @= x` is **not** in place: `Tensor_conti.py` defines
  `__imatmul` (missing the trailing `__`), so the real `__imatmul__` slot does
  not exist and `@=` silently rebinds `t` to a fresh object.
- **T-A1** — `//` (`__floordiv__`) performs **true** division, not floor —
  identical result to `/` (cross-ref UniTensor UT-A1).
- **T-IO2** — the pickle protocol is broken: `__getstate__` is only the
  inherited `object.__getstate__`, `__setstate__` is absent, and
  `pickle.dumps(t)` raises `TypeError` (same defect as UniTensor UT-IO2).
- **T-IO3** — by contrast, `Save`→`Load` round-trips deterministically clean
  (9/9) — a dense `Tensor` has no name field, so it has **no** analog of the
  UniTensor `_Load` name heap over-read (UT-IO5).
- **T-S5 / T-S7 / T-T4** — three in-place methods break the
  trailing-`_`-returns-`self` convention, each **rooted in the C++ signature**
  (not the binding): `contiguous_` returns a distinct shared-data wrapper
  (C++ `Tensor contiguous_()` by value), `flatten_` returns `None` (C++ `void
  flatten_()`), and `to_` returns `None` (C++ `void to_`).
- **T-X3** — `InvM` (matrix inverse, cat 06) and `Inv` (element-wise
  reciprocal, cat 05) are a near-name collision for different operations —
  disambiguated by the `inv_m`/`reciprocal` rename.
