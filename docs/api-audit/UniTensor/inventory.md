# UniTensor — categorized member inventory

> The master member list for `cytnx.UniTensor` (wheel **1.1.0**). Every public
> member of `dir(cytnx.UniTensor)` (126 non-underscore names) is assigned to
> **exactly one** of the twelve categories below — the categories *partition*
> the live surface. This file is the index; the per-category documents hold the
> analysis, the normative spec (`# R.`), and the probe-backed evidence.

**How to read this.** Each `## NN — <category>` section links its category
document and lists that category's live members with a one-line role. Names
tagged **(plumbing)** are raw C++/pybind bindings that leak into the public
namespace but are *not* public API — they are collected again, with their public
wrapper, in [`## Internal / plumbing`](#internal--plumbing-not-public-api) at the
end. The dunder members (`__getitem__`, `__setitem__`, the operator dunders,
`__repr__`, the pickle/copy hooks) are handled inside their owning category
document (06, 07, 11, 12) and are not repeated here.

**Provenance / gates.**
- Category documents: `docs/api-audit/UniTensor/NN-*.md` (01–12).
- Method spec: `docs/superpowers/specs/2026-07-06-cytnx-api-analysis-method-design.md`.
- Rollout plan: `docs/superpowers/plans/2026-07-06-cytnx-unitensor-method-rollout.md`.
- Coverage is machine-enforced: `python tools/validate_doc.py UniTensor docs/api-audit/UniTensor/`
  passes iff every live member is covered by some category's `# R.` section.

---

## 01 — construction & init

[`01-construction-init.md`](01-construction-init.md) — building a `UniTensor`
from bonds/metadata (the constructor overloads and `from_numpy`, an **add**).

| Member | Role |
|---|---|
| `Init` | Instance (re)initializer — build/rebuild this tensor from a bond list + metadata (the constructor's named form). |

## 02 — static generators

[`02-static-generators.md`](02-static-generators.md) — the filled/random
factory generators.

| Member | Role |
|---|---|
| `zeros` | Generator: dense tensor filled with 0. |
| `ones` | Generator: dense tensor filled with 1. |
| `arange` | Generator: 1-D ranged values (`start,stop,step`). |
| `linspace` | Generator: 1-D evenly-spaced values. |
| `eye` | Generator: identity/diagonal tensor. |
| `identity` | Generator: alias of `eye`. |
| `normal` | Generator: Gaussian-random fill (pure). |
| `normal_` | Generator: Gaussian-random fill (in-place). |
| `uniform` | Generator: uniform-random fill (pure). |
| `uniform_` | Generator: uniform-random fill (in-place). |

## 03 — metadata accessors

[`03-metadata-accessors.md`](03-metadata-accessors.md) — read-only queries on
structure, identity, classification, and leg/symmetry data.

| Member | Role |
|---|---|
| `rank` | Number of legs. |
| `rowrank` | Legs in the row (bra) space. |
| `Nblocks` | Number of symmetry blocks (→ rename `nblocks`). |
| `shape` | Size per leg. |
| `dtype` | Element-type code. |
| `dtype_str` | Element-type name. |
| `device` | Device code. |
| `device_str` | Device name. |
| `uten_type` | UniTensor kind code (0 Dense / 2 Block). |
| `uten_type_str` | Kind name. |
| `name` | Tensor name. |
| `is_diag` | Predicate: diagonal-only storage. |
| `is_tag` | Predicate: legs carry bra/ket direction. |
| `is_braket_form` | Predicate: row legs ket, column legs bra. |
| `is_blockform` | Predicate: block (symmetric) storage. |
| `is_contiguous` | Predicate: contiguous element storage. |
| `labels` | Leg labels, in leg order. |
| `get_index` | Resolve a label to its leg index. |
| `syms` | The tensor's symmetry objects. |
| `bonds` | All legs (copied list; elements share the parent's impl). |
| `bond` | One leg, independent copy. |
| `bond_` | One leg, view (shares the parent's Bond). |
| `signflip` | Per-leg fermion sign flags (BlockFermionic only). |
| `same_data` | Do two tensors share storage? |
| `get_qindices` | Per-block qnum-index list for a leg (symmetric only). |
| `getTotalQnums` | Total quantum numbers (→ rename+fix `get_total_qnums`; bound but non-functional in 1.1.0). |
| `get_blocks_qnums` | Per-block quantum numbers (→ fix; bound but non-functional in 1.1.0). |

## 04 — labels / name / rowrank

[`04-labels-name-rowrank.md`](04-labels-name-rowrank.md) — the label/name/rowrank
mutators (consolidating to `relabel`/`relabel_`).

| Member | Role |
|---|---|
| `set_name` | Set the tensor name. |
| `set_label` | Set one leg label (→ deprecate; use `relabel`). |
| `set_labels` | Set all leg labels (→ deprecate; use `relabels`). |
| `relabel` | Return a shared-data copy with relabeled legs (pure). |
| `relabel_` | Relabel legs in place, returns self. |
| `relabels` | Relabel all legs (pure). |
| `relabels_` | Relabel all legs in place. |
| `set_rowrank` | Set the row-rank (pure). |
| `set_rowrank_` | Set the row-rank in place. |
| `c_set_name` | **(plumbing)** raw binding under `set_name`. |
| `c_set_label` | **(plumbing)** raw binding under `set_label`. |
| `c_set_labels` | **(plumbing)** raw binding under `set_labels`. |
| `c_relabel_` | **(plumbing)** raw binding under `relabel_`. |
| `c_relabels_` | **(plumbing)** raw binding under `relabels_`. |
| `c_set_rowrank_` | **(plumbing)** raw binding under `set_rowrank_`. |

## 05 — structure manipulation

[`05-structure-manipulation.md`](05-structure-manipulation.md) — leg/layout
transforms (permute, reshape, contiguity, bond grouping, tagging, twists, apply).

| Member | Role |
|---|---|
| `permute` | Reorder legs (view). |
| `permute_` | Reorder legs in place. |
| `permute_nosignflip` | Permute without fermionic sign flip (view). |
| `permute_nosignflip_` | In-place permute without sign flip. |
| `reshape` | Reshape (pure). |
| `reshape_` | Reshape in place. |
| `contiguous` | Return a contiguous copy. |
| `contiguous_` | Make contiguous in place. |
| `group_basis` | Merge degenerate basis (pure). |
| `group_basis_` | Merge degenerate basis in place. |
| `combineBonds` | Combine legs (→ rename `combine_bonds`; C++ `combineBond` unbound). |
| `to_dense` | Convert block→dense (pure). |
| `to_dense_` | Convert block→dense in place. |
| `truncate` | Truncate a leg's dimension (pure). |
| `truncate_` | Truncate a leg's dimension in place. |
| `tag` | Attach bra/ket direction to legs. |
| `twist` | Apply a fermionic twist (pure). |
| `twist_` | Apply a fermionic twist in place. |
| `fermion_twists` | Apply fermion twists (pure). |
| `fermion_twists_` | Apply fermion twists in place. |
| `apply` | Apply an element-wise callable (pure). |
| `apply_` | Apply an element-wise callable in place. |
| `ctag` | **(plumbing)** raw binding under `tag`. |
| `ctruncate_` | **(plumbing)** raw binding under `truncate_`. |
| `make_contiguous` | **(plumbing)** raw shim under `contiguous`/`contiguous_`. |

## 06 — element & block access

[`06-element-block-access.md`](06-element-block-access.md) — scalar-element and
block getters/setters (plus `__getitem__`/`__setitem__`, documented there).

| Member | Role |
|---|---|
| `at` | Reference to one element (get/set). |
| `item` | Extract the single scalar of a size-1 tensor. |
| `get_elem` | Read one element (4 float/complex dtypes only). |
| `set_elem` | Write one element (all 11 dtypes). |
| `elem_exists` | Block-only: whether a symmetry block-element exists. |
| `get_block` | One symmetry block, copy. |
| `get_block_` | One symmetry block, view. |
| `get_blocks` | All symmetry blocks, copies. |
| `get_blocks_` | All symmetry blocks, views. |
| `put_block` | Write one symmetry block. |
| `put_block_` | Write one symmetry block (in place / by ref). |
| `c_at` | **(plumbing)** raw binding under `at`. |

## 07 — arithmetic & element-wise

[`07-arithmetic-elementwise.md`](07-arithmetic-elementwise.md) — operator
dunders and the element-wise math **members** (`Conj`/`Trace`/… → lowercase).

| Member | Role |
|---|---|
| `Pow` | Element-wise power (→ rename `pow`). |
| `Pow_` | In-place power (→ rename `pow_`). |
| `Inv` | Element-wise reciprocal (→ rename `inv`). |
| `Conj` | Complex conjugate (→ rename `conj`). |
| `Conj_` | In-place conjugate (→ rename `conj_`). |
| `Transpose` | Transpose (→ rename `transpose`). |
| `Transpose_` | In-place transpose (→ rename `transpose_`). |
| `Dagger` | Conjugate transpose (→ rename `dagger`). |
| `Dagger_` | In-place dagger (→ rename `dagger_`). |
| `normalize` | Normalize (pure). |
| `normalize_` | Normalize in place. |
| `Trace` | Trace over paired legs (→ rename `trace`). |
| `Trace_` | In-place trace (→ rename `trace_`). |
| `Norm` | Frobenius norm (→ rename `norm`). |
| `cConj_` | **(plumbing)** raw binding under `Conj_`. |
| `cDagger_` | **(plumbing)** raw binding under `Dagger_`. |
| `cPow_` | **(plumbing)** raw binding under `Pow_`. |
| `cTrace_` | **(plumbing)** raw binding under `Trace_`. |
| `cTranspose_` | **(plumbing)** raw binding under `Transpose_`. |
| `cnormalize_` | **(plumbing)** raw binding under `normalize_`. |
| `cInv_` | **(plumbing)** raw binding for the (missing public) `Inv_`. |
| `c__ipow__` | **(plumbing)** raw binding under `__ipow__`. |

## 08 — linalg operations (free functions)

[`08-linalg-operations.md`](08-linalg-operations.md) — the `cytnx.linalg` free
functions acting on `UniTensor` (`Svd`, `Qr`, `Eigh`, `ExpH`, …), kept
Capitalized. **These are module-level free functions, not `UniTensor` members**,
so this category owns no unique `dir(UniTensor)` name; the member-form spellings
`Conj`/`Trace`/`Pow`/`Norm`/`Inv` are owned by category 07 and cross-referenced
here as their free-function counterparts.

*(no unique `UniTensor` member — see the document for the free-function surface)*

## 09 — linalg solvers (Krylov)

[`09-linalg-solvers.md`](09-linalg-solvers.md) — the Krylov solver free
functions (`Lanczos`, `Arnoldi`, `Lanczos_Gnd_Ut`, …). Like category 08 these
are `cytnx.linalg` free functions, not `UniTensor` members.

*(no unique `UniTensor` member — see the document for the solver surface)*

## 10 — contraction & networks

[`10-contraction-networks.md`](10-contraction-networks.md) — the member
`contract` and the related free functions `Contract`/`Contracts`/`ncon`.

| Member | Role |
|---|---|
| `contract` | Contract this tensor against another over shared labels (member). |

## 11 — I/O & display

[`11-io-display.md`](11-io-display.md) — persistence and printing (plus
`__repr__` and the pickle hooks, documented there).

| Member | Role |
|---|---|
| `Save` | Serialize to a `.cytnx` file (→ rename `save`). |
| `Load` | Static: load a `UniTensor` from a file (→ rename `load`). |
| `print_diagram` | Print the leg/bond diagram. |
| `print_block` | Print one symmetry block. |
| `print_blocks` | Print all symmetry blocks. |

## 12 — type & device conversion

[`12-type-device-conversion.md`](12-type-device-conversion.md) — dtype/device
conversion and deep-copy (plus `__copy__`/`__deepcopy__`, documented there).

| Member | Role |
|---|---|
| `astype` | Convert element dtype (no-op short-circuits to `self`). |
| `to` | Move to a device (pure; no-op short-circuits to `self`). |
| `to_` | Move to a device in place. |
| `clone` | Deep copy (independent storage). |
| `convert_from` | Populate this tensor from another (dtype/device convert). |
| `astype_different_type` | **(plumbing)** raw shim under `astype`. |
| `to_different_device` | **(plumbing)** raw shim under `to`/`to_`. |
| `cfrom` | **(plumbing)** raw binding under `convert_from`. |

---

## Internal / plumbing (not public API)

These names are live members of `dir(cytnx.UniTensor)` but are **raw
C++/pybind bindings that leak** into the public namespace — `c*`-prefixed
in-place bindings, the `*_different_*` conversion shims, and the
`make_contiguous` contiguity shim. Each sits *under* a public wrapper (usually a
`*_conti.py` method); the public member is the supported API and the raw binding
should be hidden (underscore-prefixed or removed) in the next version. Collected
from every category's plumbing block.

| Leaked binding | Category | Public wrapper it backs |
|---|---|---|
| `c_set_name` | 04 | `set_name` |
| `c_set_label` | 04 | `set_label` |
| `c_set_labels` | 04 | `set_labels` |
| `c_relabel_` | 04 | `relabel_` |
| `c_relabels_` | 04 | `relabels_` |
| `c_set_rowrank_` | 04 | `set_rowrank_` |
| `ctag` | 05 | `tag` |
| `ctruncate_` | 05 | `truncate_` |
| `make_contiguous` | 05 | `contiguous` / `contiguous_` |
| `c_at` | 06 | `at` |
| `cConj_` | 07 | `Conj_` |
| `cDagger_` | 07 | `Dagger_` |
| `cPow_` | 07 | `Pow_` |
| `cTrace_` | 07 | `Trace_` |
| `cTranspose_` | 07 | `Transpose_` |
| `cnormalize_` | 07 | `normalize_` |
| `cInv_` | 07 | (no public `Inv_` — raw only) |
| `c__ipow__` | 07 | `__ipow__` |
| `astype_different_type` | 12 | `astype` |
| `to_different_device` | 12 | `to` / `to_` |
| `cfrom` | 12 | `convert_from` |

**Recommendation (cross-category):** hide all of the above (prefix `_` or drop
the `.def` registration) so they no longer appear in `dir(cytnx.UniTensor)`; the
public wrappers are the sole supported surface.
