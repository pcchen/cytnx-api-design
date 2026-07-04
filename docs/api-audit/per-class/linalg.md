# `linalg` — API audit

`cytnx.linalg` is not a class but a **module of 53 free functions** — the linear
algebra layer that every reference tensor-network algorithm is built on. Its
functions divide into three roles: (1) the **matrix decompositions**
(`Svd`/`Svd_truncate`/`Gesvd`/`Gesvd_truncate`/`Rsvd`/`Hosvd`/`Eig`/`Eigh`/
`Qr`/`Qdr`), which are overloaded to consume and return **both** a dense
`Tensor` and a labelled `UniTensor` (`UniTensor.md`) and are the backbone of
`essential-api.md` step (c)/(d); (2) **element-wise / matrix functions**
(`Conj`/`Inv`/`InvM`/`Pow`/`Exp`/`ExpH`/`ExpM`/`Abs`/…) with the pure/in-place
`Foo`/`Foo_` structure; and (3) **BLAS-level and contraction primitives**
(`Matmul`/`Gemm`/`Axpy`/`Dot`/`Tensordot`/`Kron`/…) plus **iterative solvers**
(`Lanczos`/`Lanczos_Exp`/`Arnoldi`).

Ground truth for behavior is `docs/api-audit/probes/linalg.py`, executed against
`./.venv/bin/python` (`source tools/env.sh && $PY docs/api-audit/probes/linalg.py`;
all 33 assertions `[PASS]`, exit 0). Ground truth for static signatures is
`cytnx_src/include/linalg.hpp` (C++ declarations and the Doxygen `@return`
blocks that document each decomposition's factor order) and
`cytnx_src/pybind/linalg_py.cpp` (the pybind11 bindings — authoritative for the
Python-visible signatures and, notably, for the commented-out `c_Lanczos_*`
registrations, see the headline finding P1). The Python-side augmentation file
`cytnx_src/cytnx/linalg_conti.py` adds **no** live members: its only content is a
triple-quoted (i.e. commented-out) block that was meant to wrap the three
`Lanczos_ER`/`Lanczos_Gnd`/`Lanczos_Gnd_Ut` solvers — so those three are
unreachable (P1).

The headline structural fact is that **the entire namespace is Capitalized**
(`Svd`, `Eigh`, `Qr`, `Conj`, `Matmul`, …) — not one of the 53 callables is
`snake_case`, so `linalg` violates N1 wholesale (C1). The single most
load-bearing behavioral fact — and the one a later synthesis task depends on —
is the **decomposition return order**, which is empirically S-first / values-first
for `Svd`/`Gesvd`/`Eigh` but Q-first for `Qr`/`Qdr` (C3, P-order).

## Inventory

C++ signatures are read from `linalg.hpp`; Python signatures are the effective
pybind-visible signature, cross-checked against `tools/member_inventory.py
linalg`. Only the members whose return contract or C++/Python divergence is
load-bearing get a full signature row; the remainder are listed by group with
their kind, and every one appears in the Recommendation table. The decompositions
are overloaded on `Tensor` and `UniTensor`; the `_truncate`/`Rsvd` UniTensor
overloads additionally carry a third `min_blockdim` overload (to pin
per-symmetry-block dimensions) that the `Tensor` form lacks — this asymmetry is
present identically in C++ and Python, so it is an overload-set note, not a
parity gap.

### Matrix decompositions (Tensor + UniTensor overloads)

| Member | C++ signature | Python (effective, live) signature | Return order (probed) |
|---|---|---|---|
| `Svd` | `vector<Tensor> Svd(const Tensor&, const bool& is_UvT=true)` (+ UniTensor overload) | `Svd(Tin, is_UvT=True) -> list` | `[S, U, vT]` if `is_UvT`, else `[S]` |
| `Svd_truncate` | `vector<Tensor> Svd_truncate(const Tensor&, const cytnx_uint64& keepdim, const double& err=0., const bool& is_UvT=true, const unsigned int& return_err=0, const cytnx_uint64& mindim=1)` (+ 2 UniTensor overloads) | `Svd_truncate(Tin, keepdim, err=0, is_UvT=True, return_err=0, mindim=1) -> list` | `[S, U, vT]`, `+[err]` if `return_err>0` |
| `Gesvd` | `vector<Tensor> Gesvd(const Tensor&, const bool& is_U=true, const bool& is_vT=true)` (+ UniTensor) | `Gesvd(Tin, is_U=True, is_vT=True) -> list` | `[S] + ([U] if is_U) + ([vT] if is_vT)` |
| `Gesvd_truncate` | analogous to `Svd_truncate` but with `is_U`/`is_vT` | `Gesvd_truncate(Tin, keepdim, err=0, is_U=True, is_vT=True, return_err=0, mindim=1) -> list` | `[S, U, vT]`, `+[err]` |
| `Rsvd` | randomized SVD; extra `oversampling_*`/`power_iteration`/`seed` params | `Rsvd(Tin, keepdim, err=0.0, is_U=True, is_vT=True, return_err=0, mindim=1, oversampling_summand=10, oversampling_factor=1.0, power_iteration=0, seed=-1) -> list` | `[S, U, vT]` |
| `Hosvd` | higher-order SVD | `Hosvd(Tn, mode, is_core=True, is_Ls=False, truncate_dim=[]) -> list` | core + factor tensors |
| `Eigh` | `vector<Tensor> Eigh(const Tensor&, const bool& is_V=true, const bool& row_v=false)` (+ UniTensor) | `Eigh(Tin, is_V=True, row_v=False) -> list` | `[eigvals, eigvecs]` if `is_V`, else `[eigvals]` |
| `Eig` | `vector<Tensor> Eig(const Tensor&, const bool& is_V=true, const bool& row_v=false)` (+ UniTensor) | `Eig(Tin, is_V=True, row_v=False) -> list` | `[eigvals(complex), eigvecs]` |
| `Qr` | `vector<Tensor> Qr(const Tensor&, const bool& is_tau=false)` (+ UniTensor) | `Qr(Tin, is_tau=False) -> list` | `[Q, R]`, `+[tau]` if `is_tau` |
| `Qdr` | `vector<Tensor> Qdr(const Tensor&, const bool& is_tau=false)` (+ UniTensor) | `Qdr(Tin, is_tau=False) -> list` | `[Q, D, R]`, `+[tau]` if `is_tau` |

`Rand_isometry(Tin, keepdim, power_iteration=2, seed=-1) -> Tensor` (Tensor-only)
builds a random isometry and is grouped with the SVD family.

### Element-wise & matrix functions (pure `Foo` / in-place `Foo_`)

`Conj`, `Conj_`, `Inv`, `Inv_`, `InvM`, `InvM_`, `Pow`, `Pow_`, `Abs`, `Abs_`,
`Exp`, `Exp_`, `Expf`, `Expf_`, `ExpH`, `ExpM`.

`Conj`/`Pow`/`Inv`/`InvM` are overloaded on `Tensor` and `UniTensor`; `Abs`/`Exp`/
`Expf` are `Tensor`-only; `ExpH`/`ExpM` (matrix exponential of a Hermitian /
generic operator) are **`UniTensor`-only** and have no in-place `_` form.
`Inv(Tin, clip=-1)` is the **element-wise** reciprocal (with a clip floor);
`InvM(Tin)` is the **true matrix inverse** — two very different operations behind
near-identical names (C4). The `_`-suffixed forms mutate the argument in place and
return `None` (probed).

### Reductions & scalar-valued

`Det`, `Norm`, `Trace`, `Sum`, `Max`, `Min`, `Diag`.

`Trace(Tn, axisA=0, axisB=1)` (Tensor + UniTensor, and a `(str, str)` axis-label
overload for UniTensor) is the linalg free-function partial trace (distinct from
`UniTensor.Trace`). `Norm` is overloaded on `Tensor` and `UniTensor` but **always
returns a scalar `Tensor`** — a `UniTensor` argument does not yield a `UniTensor`
result (probed). `Diag` maps a 1-D vector to a diagonal matrix (and back).

### Contraction & products (Tensor-only)

`Matmul`, `Matmul_dg`, `Dot`, `Vectordot`, `Outer`, `Kron`, `Tensordot`,
`Tensordot_dg`, `Directsum`.

`Matmul_dg`/`Tensordot_dg` are the variants where one operand is a stored diagonal;
`Vectordot(T1, T2, is_conj=False)` and `Tensordot(T1, T2, indices_1, indices_2,
cacheL=False, cacheR=False)` are the vector/general contraction primitives behind
`essential-api.md` step (b).

### BLAS-level & iterative solvers

`Axpy`, `Axpy_`, `Gemm`, `Gemm_`, `Ger`, `Lstsq`, `Tridiag`, `Lanczos`,
`Lanczos_Exp`, `Arnoldi`.

`Axpy`/`Gemm`/`Ger` take a `cytnx.Scalar` coefficient and mirror the BLAS
`axpy`/`gemm`/`ger` kernels; `Axpy_`/`Gemm_` are their in-place forms. `Lanczos`,
`Lanczos_Exp`, and `Arnoldi` take a `LinOp` (`LinOp.md`) plus a `Tensor`/
`UniTensor` seed. The C++ header additionally declares `Lanczos_ER`,
`Lanczos_Gnd`, `Lanczos_Gnd_Ut`, `Add`/`Sub`/`Mul`/`Div`/`Mod`/`Cpr`, and
`Gemm_Batch`, none of which are exposed under `cytnx.linalg` (the three
`Lanczos_*` entry points are the material gap — see P1; the arithmetic helpers
are internal operator implementations surfaced elsewhere as `Tensor` operators).

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/linalg.py`.

- **P1 — headline finding: three C++ iterative-eigensolver entry points
  (`Lanczos_ER`, `Lanczos_Gnd`, `Lanczos_Gnd_Ut`) are completely unreachable from
  Python — their bindings are commented out on *both* layers.** The C++ header
  declares all three (`linalg.hpp`), and `linalg_py.cpp:1000-1018` contains their
  intended `m_linalg.def("c_Lanczos_ER", …)` / `c_Lanczos_Gnd` / `c_Lanczos_Gnd_Ut`
  registrations — but the entire block is wrapped in a `/* … */` comment. The
  Python-side wrappers that were meant to re-expose them under clean names
  (`Lanczos_ER`/`Lanczos_Gnd`/`Lanczos_Gnd_Ut`) live in `linalg_conti.py` but are
  themselves inside a triple-quoted string, i.e. also commented out. Net effect:
  neither the clean names nor the `c_`-prefixed names exist on `cytnx.linalg`.
  Probed: *"the C++ linalg header declares Lanczos_ER, Lanczos_Gnd and
  Lanczos_Gnd_Ut, but their pybind registrations are commented out … so all three
  are UNREACHABLE from cytnx.linalg under either their clean or c_-prefixed name"*
  `[PASS]`, with the control *"the iterative solvers that ARE bound — Lanczos,
  Lanczos_Exp, Arnoldi — are all reachable"* `[PASS]` proving the gap is specific
  to the three dead entry points, not iterative solvers in general. A C++ caller
  can invoke `Lanczos_Gnd` (a ground-state solver); a Python caller cannot call it
  at all — the largest C++/Python surface gap in this module.
- **P2 — argument-name parity is otherwise clean: the pybind bindings preserve
  the C++ parameter names verbatim as Python keywords.** `is_UvT` (Svd),
  `keepdim`/`err`/`return_err`/`mindim` (Svd_truncate), `is_tau` (Qr), and
  `is_U`/`is_vT` (Gesvd) are all callable by keyword with the same spelling as the
  C++ header (N3/N4 across the language boundary hold for these). Probed: *"the
  pybind bindings preserve the C++ argument names as Python keywords … verified by
  calling each purely by keyword"* `[PASS]`. (The internal `is_UvT`-vs-`is_U/is_vT`
  divergence this exposes is an *internal* consistency issue between two sibling
  functions, recorded as C2, not a C++/Python parity gap — the divergence is
  identical on both language sides.)
- **P3 — the pure/in-place split is consistent across the language boundary: every
  in-place `Foo_` returns `None` (void) and mutates its argument in both C++ and
  Python.** `Conj_`/`Pow_`/`Abs_`/`Inv_` each return `None` from Python (matching
  the C++ `void`/`Tio&` out-parameter forms) and write through to the passed
  tensor's storage. Probed: e.g. *"Conj_(Tensor) is IN-PLACE: it returns None and
  conjugates the argument's storage (C[0,1] 1+2j -> 1-2j)"* `[PASS]`, and the
  analogous `Pow_`/`Abs_`/`Inv_` assertions `[PASS]`; the pure forms return a new
  object and leave the source unchanged (*"Conj(Tensor) (pure form) returns a NEW
  tensor and leaves the source unchanged"* `[PASS]`). No C++/Python divergence in
  copy-vs-in-place semantics was found for any probed pair.

## Consistency findings

- **C1 — headline: violates N1 for the entire namespace — all 53 callables are
  Capitalized.** `Svd`, `Svd_truncate`, `Gesvd`, `Gesvd_truncate`, `Rsvd`,
  `Hosvd`, `Eig`, `Eigh`, `Qr`, `Qdr`, `Rand_isometry`, `Conj`, `Conj_`, `Inv`,
  `Inv_`, `InvM`, `InvM_`, `Pow`, `Pow_`, `Abs`, `Abs_`, `Exp`, `Exp_`, `Expf`,
  `Expf_`, `ExpH`, `ExpM`, `Det`, `Norm`, `Trace`, `Sum`, `Max`, `Min`, `Diag`,
  `Matmul`, `Matmul_dg`, `Dot`, `Vectordot`, `Outer`, `Kron`, `Tensordot`,
  `Tensordot_dg`, `Directsum`, `Axpy`, `Axpy_`, `Gemm`, `Gemm_`, `Ger`, `Lstsq`,
  `Tridiag`, `Lanczos`, `Lanczos_Exp`, `Arnoldi` — every one starts with a capital
  letter and none is `snake_case`. Per N1 (which governs *free functions* as well
  as methods) each becomes its lowercase, underscore-separated form (`Svd`→`svd`,
  `Eigh`→`eigh`, `Qr`→`qr`, `InvM`→`inv_m`, `ExpH`→`exp_h`, `Matmul_dg`→
  `matmul_dg`, `Directsum`→`direct_sum`, …). Probed: *"cytnx.linalg exposes 53
  public callables, and EVERY one of them is Capitalized … not a single member is
  snake_case, so the whole namespace violates N1"* `[PASS]`. This is the same
  N1 casing verdict `UniTensor.md` (C2) and `Symmetry.md` (C1) reach for their
  capitalized methods, applied here to a whole module.
- **C2 — violates N4: the two full-SVD sibling functions spell the "return the
  unitary factors" flag with different granularity.** `Svd`/`Svd_truncate` take a
  single combined `is_UvT` boolean (both isometries returned together, or neither),
  while `Gesvd`/`Gesvd_truncate`/`Rsvd` take two independent booleans
  `is_U`/`is_vT`. These are semantically the same decomposition parameter, so per
  N4 ("semantically equivalent parameters use the same name and position across
  overloaded/sibling methods") the granularity should match — a caller cannot ask
  `Svd` for "U only" the way `Gesvd` allows, and cannot pass `is_UvT` to `Gesvd`.
  Probed: *"Svd REJECTS an is_U keyword (it only has is_UvT) while Gesvd ACCEPTS
  is_U — two sibling SVD functions spell the same 'return the unitaries' concept
  with different flag granularity"* `[PASS]`, together with *"Gesvd exposes TWO
  independent flags is_U / is_vT (unlike Svd's single is_UvT): is_U=False drops U,
  returning just [S, vT]"* `[PASS]`. Recommend the finer `is_U`/`is_vT` pair
  everywhere (it subsumes `is_UvT`).
- **C3 — the decomposition RETURN ORDER is internally inconsistent, and this is
  the highest-impact issue for downstream algorithm code.** The "diagonal spectrum"
  factor sits at **index 0** for `Svd`/`Gesvd`/`Svd_truncate`/`Gesvd_truncate`/
  `Rsvd` (singular values `S` first) and for `Eigh`/`Eig` (eigenvalues first), but
  at **index 1** for `Qdr` (`[Q, D, R]` — `D` in the middle), and is absent for
  `Qr` (`[Q, R]` — orthogonal factor first). So there is no single rule for where
  the spectrum lands, and `Svd`'s order is neither its own multiplication order
  (`M = U @ diag(S) @ vT`, i.e. `out[1] @ diag(out[0]) @ out[2]`) nor
  `numpy.linalg.svd`'s `(U, S, Vh)`. Probed: *"Svd(Tensor) returns … [S, U, vT]"*,
  *"Svd's return order is SINGULAR-VALUES-FIRST, which is NOT the multiplication
  order … and it is also NOT numpy.linalg.svd's (U, S, Vh) order"*, *"Eigh(Tensor)
  returns [eigvals, eigvecs]"*, *"Qr(Tensor) returns [Q, R] with the ORTHOGONAL
  factor Q FIRST"*, and *"Qdr(Tensor) returns [Q, D, R]: the diagonal factor D sits
  at index 1 (the MIDDLE), NOT at index 0 like Svd's S or Eigh's eigvals — so the
  position of the 'diagonal spectrum' factor is INTERNALLY INCONSISTENT across
  cytnx's decompositions"* — all `[PASS]`. No `N`/`B` id in `00-methodology.md`
  squarely covers *return-tuple element order* (N4 governs *argument* order), so
  this is flagged as a plain internal inconsistency in the spirit of N4 rather than
  a cited violation (same informal treatment as `Symmetry.md`'s C5/C6). Because a
  later synthesis task decomposes TRG/HOTRG/CTMRG/MERA into `Svd`/`Eigh` calls,
  every `keep`/`rename` docstring below states the exact index of each returned
  factor; the recommendation is to leave the current runtime order intact (renaming
  the tuple order would silently break every existing caller) but to document it
  exhaustively and, in a v2 API, expose a named-tuple / dataclass return so the
  order stops mattering.
- **C4 — naming ambiguity: `Inv` (element-wise reciprocal) and `InvM` (true matrix
  inverse) are near-identical names for fundamentally different operations.**
  `Inv(Tin, clip=-1)` inverts each element (`x -> 1/x`, with a clip floor), whereas
  `InvM(Tin)` computes the matrix inverse (`InvM(M) @ M == I`). The one-letter `M`
  suffix is easy to miss and does not read as "matrix". Probed: *"Inv(Tensor) is
  ELEMENT-WISE … Inv(M)[0,0] == 1/2, Inv(M)[0,1] == 1/1 — it is NOT a matrix
  inverse"* `[PASS]` and *"InvM(Tensor) is the TRUE matrix inverse: InvM(M) @ M ==
  I"* `[PASS]`. No `N`/`B` id directly covers "two similar names for different
  ops"; flagged informally like `UniTensor.md`'s `eye`/`identity` overlap note.
  Recommend keeping `inv` (it matches `Tensor.Inv`/`UniTensor.Inv`'s element-wise
  meaning) but renaming `InvM`→`inv_m` and stating "matrix inverse" prominently in
  its docstring, since a silent rename to e.g. `matrix_inverse` would break the
  `Inv`/`InvM` mnemonic pairing users already know.
- **C5 — N2 is satisfied for every op that has an in-place form, but two matrix
  functions that plausibly want one lack it.** The pairs `Abs`/`Abs_`, `Conj`/
  `Conj_`, `Exp`/`Exp_`, `Expf`/`Expf_`, `Inv`/`Inv_`, `InvM`/`InvM_`, `Pow`/
  `Pow_`, `Axpy`/`Axpy_`, `Gemm`/`Gemm_` are all correct N2 pure/in-place pairs
  (probed: *"every element-wise/BLAS op that has an in-place form pairs a pure Foo
  with an in-place Foo_ …"* `[PASS]`). `ExpH`/`ExpM` (matrix exponentials) have no
  `_` counterpart despite `Exp` having one; this is a minor N2 gap (an operation
  meaningful in-place that only exists pure). Reductions (`Det`/`Norm`/`Trace`/
  `Sum`/`Max`/`Min`) correctly have no `_` form (they return a scalar, so in-place
  is meaningless) — N2 is vacuous for them, recorded so the absence is not read as
  an oversight.

## Recommendation

Every one of the 53 live public callables of `cytnx.linalg` appears below, tagged
**keep / add / rename / remove**, organized by the groups above. The dominant
verdict is **rename** — all 53 are Capitalized and get their N1 `snake_case` form
(C1); the actionable structural items are the `is_U`/`is_vT` unification (C2), the
`InvM`→`inv_m` disambiguation (C4), and three **add** rows restoring the
commented-out `Lanczos_*` solvers (P1). No function is removed: the decompositions,
BLAS kernels, and iterative solvers are all distinct, load-bearing operations.

### Matrix decompositions

| Member | Verdict | Rationale |
|---|---|---|
| `Svd` | rename | → `svd` (C1/N1). Returns `[S, U, vT]` (values-first, C3). |
| `Svd_truncate` | rename | → `svd_truncate` (C1). Honors `keepdim`; appends `err` when `return_err>0` (probed). Adopt `is_U`/`is_vT` (C2). |
| `Gesvd` | rename | → `gesvd` (C1). Same `[S, U, vT]` order; the `is_U`/`is_vT` model C2 recommends everywhere. |
| `Gesvd_truncate` | rename | → `gesvd_truncate` (C1). |
| `Rsvd` | rename | → `rsvd` (C1). Randomized truncated SVD. |
| `Hosvd` | rename | → `hosvd` (C1). Higher-order SVD. |
| `Eigh` | rename | → `eigh` (C1). Hermitian eig; returns `[eigvals, eigvecs]` (values-first, C3). |
| `Eig` | rename | → `eig` (C1). General eig; eigenvectors only invertible, not unitary. |
| `Qr` | rename | → `qr` (C1). Returns `[Q, R]` (Q-first, C3); `+[tau]` if `is_tau`. |
| `Qdr` | rename | → `qdr` (C1). Returns `[Q, D, R]` — diagonal in the MIDDLE (C3). |
| `Rand_isometry` | rename | → `rand_isometry` (C1). Random isometry generator. |

### Element-wise & matrix functions

| Member | Verdict | Rationale |
|---|---|---|
| `Conj` | rename | → `conj` (C1). Pure element-wise conjugation (B1). |
| `Conj_` | rename | → `conj_` (C1). In-place; returns None (P3). |
| `Inv` | rename | → `inv` (C1). Element-wise reciprocal with `clip` — NOT a matrix inverse (C4). |
| `Inv_` | rename | → `inv_` (C1). In-place element-wise reciprocal (P3). |
| `InvM` | rename | → `inv_m` (C1/C4). TRUE matrix inverse; disambiguate from `inv`. |
| `InvM_` | rename | → `inv_m_` (C1/C4). In-place matrix inverse. |
| `Pow` | rename | → `pow` (C1). Element-wise power. |
| `Pow_` | rename | → `pow_` (C1). In-place power (P3). |
| `Abs` | rename | → `abs` (C1). Element-wise absolute value. |
| `Abs_` | rename | → `abs_` (C1). In-place (P3). |
| `Exp` | rename | → `exp` (C1). Element-wise exponential. |
| `Exp_` | rename | → `exp_` (C1). In-place. |
| `Expf` | rename | → `expf` (C1). Single-precision element-wise exponential. |
| `Expf_` | rename | → `expf_` (C1). In-place. |
| `ExpH` | rename | → `exp_h` (C1). Matrix exp of a Hermitian UniTensor; add an in-place `exp_h_` (C5). |
| `ExpM` | rename | → `exp_m` (C1). Matrix exp of a generic UniTensor; add an in-place `exp_m_` (C5). |

### Reductions & scalar-valued

| Member | Verdict | Rationale |
|---|---|---|
| `Det` | rename | → `det` (C1). Determinant (scalar `Tensor`). |
| `Norm` | rename | → `norm` (C1). 2-norm; returns a scalar `Tensor` even for a `UniTensor` arg (probed). |
| `Trace` | rename | → `trace` (C1). Partial trace over two axes (index or, for UniTensor, label). |
| `Sum` | rename | → `sum` (C1). Sum reduction. |
| `Max` | rename | → `max` (C1). Maximum element (real part for complex). |
| `Min` | rename | → `min` (C1). Minimum element. |
| `Diag` | rename | → `diag` (C1). Vector↔diagonal-matrix conversion. |

### Contraction & products

| Member | Verdict | Rationale |
|---|---|---|
| `Matmul` | rename | → `matmul` (C1). Matrix multiply. |
| `Matmul_dg` | rename | → `matmul_dg` (C1). Matmul with one diagonal operand. |
| `Dot` | rename | → `dot` (C1). Generic dot product. |
| `Vectordot` | rename | → `vectordot` (C1). 1-D dot with optional conjugation. |
| `Outer` | rename | → `outer` (C1). Outer product. |
| `Kron` | rename | → `kron` (C1). Kronecker product. |
| `Tensordot` | rename | → `tensordot` (C1). General pairwise contraction (essential step (b)). |
| `Tensordot_dg` | rename | → `tensordot_dg` (C1). Tensordot with a diagonal operand. |
| `Directsum` | rename | → `direct_sum` (C1). Direct sum over shared axes. |

### BLAS-level & iterative solvers

| Member | Verdict | Rationale |
|---|---|---|
| `Axpy` | rename | → `axpy` (C1). BLAS `a*x(+y)`. |
| `Axpy_` | rename | → `axpy_` (C1). In-place `axpy`. |
| `Gemm` | rename | → `gemm` (C1). BLAS general matrix multiply. |
| `Gemm_` | rename | → `gemm_` (C1). In-place `gemm`. |
| `Ger` | rename | → `ger` (C1). BLAS rank-1 update. |
| `Lstsq` | rename | → `lstsq` (C1). Least-squares solve. |
| `Tridiag` | rename | → `tridiag` (C1). Tridiagonal eigensolver. |
| `Lanczos` | rename | → `lanczos` (C1). Iterative eigensolver (LinOp). |
| `Lanczos_Exp` | rename | → `lanczos_exp` (C1). Krylov `exp(tau*Hop)·v`. |
| `Arnoldi` | rename | → `arnoldi` (C1). Non-Hermitian iterative eigensolver. |
| `Lanczos_ER` | add | Bind the commented-out C++ `Lanczos_ER` (P1) as `lanczos_er`. |
| `Lanczos_Gnd` | add | Bind the commented-out C++ `Lanczos_Gnd` (P1) as `lanczos_gnd`. |
| `Lanczos_Gnd_Ut` | add | Bind the commented-out C++ `Lanczos_Gnd_Ut` (P1) as `lanczos_gnd_ut`. |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`,
grouped by family. Each block lists its members' **current** names in backticks
(so `validate_doc.py` can match them back to the Recommendation rows) and records
the recommended `snake_case` rename and the exact return contract. The
decomposition blocks are deliberately explicit about factor order (C3), since a
later synthesis task consumes them.

### SVD family: `Svd`, `Svd_truncate`, `Gesvd`, `Gesvd_truncate`, `Rsvd`, `Hosvd`, `Rand_isometry`

```
Singular-value decompositions of a rank-2 Tensor (a matrix) or the row/column
matricization of a UniTensor.

svd (from `Svd`)          : full SVD via LAPACK ?gesdd. Returns [S, U, vT].
gesvd (from `Gesvd`)      : full SVD via LAPACK ?gesvd. Returns [S] (+U +vT).
svd_truncate / gesvd_truncate : full SVD, then keep at most `keepdim` singular
                            values (see truncation order below). Returns
                            [S, U, vT] (+[err]).
rsvd (from `Rsvd`)        : randomized truncated SVD (oversampling + power
                            iteration). Returns [S, U, vT].
hosvd (from `Hosvd`)      : higher-order (Tucker) SVD; returns the core tensor
                            and per-mode factor tensors.
rand_isometry (from `Rand_isometry`) : a random isometry of the given keepdim.

Parameters
----------
Tin : Tensor or UniTensor
    The operand. A UniTensor is decomposed across its rowrank split.
is_UvT : bool, optional (`Svd`/`Svd_truncate`)
    If True (default) return both U and vT; if False return only [S]. NOTE
    this single combined flag is inconsistent with gesvd's separate is_U/is_vT
    (Consistency finding C2); the recommended API uses is_U/is_vT everywhere.
is_U, is_vT : bool, optional (`Gesvd`/`Gesvd_truncate`/`Rsvd`)
    Independently toggle returning U and vT.
keepdim : int (`*_truncate`/`Rsvd`)
    Upper bound on the number of singular values kept (clamped to the rank;
    may be exceeded to preserve an exactly-degenerate block).
err : float, optional
    Drop singular values below this cutoff.
return_err : int, optional
    If > 0, append the truncation error as the final list element; if > 1,
    append the full list of dropped singular values.
mindim : int, optional
    Keep at least this many singular values.

Returns
-------
list of Tensor (or list of UniTensor)
    RETURN ORDER (verified by probe): the singular values S are ALWAYS at
    index 0 (values-first). Full form is [S, U, vT]; the isometry entries are
    present only if their is_U/is_vT/is_UvT flag is True; a truncation error,
    if requested, is appended last. The reconstruction is M = U @ diag(S) @ vT
    (i.e. out[1] @ diag(out[0]) @ out[2]) -- NOT out[0] @ out[1] @ out[2], and
    NOT numpy's (U, S, Vh) order.

Notes
-----
Renamed from the Capitalized forms (C1/N1). `svd`/`svd_truncate` currently take
one `is_UvT`; `gesvd`/`rsvd` take `is_U`/`is_vT` (C2). Truncation order for the
`_truncate`/`rsvd` forms: (1) keep at most `keepdim` (degeneracy may enlarge
it), (2) keep at least `mindim`, (3) drop values < `err`. S, U, vT inherit the
input's dtype and device. These are the decomposition backbone of
`essential-api.md` step (c)/(d); prefer `svd_truncate`/`gesvd_truncate` for the
truncated-bond step in TRG/HOTRG/CTMRG.
```

### Eigen family: `Eigh`, `Eig`

```
Eigen-decomposition of a rank-2 Tensor / matricized UniTensor.

eigh (from `Eigh`) : for a HERMITIAN operator; returns real eigenvalues and a
                     UNITARY eigenvector matrix (M = V @ diag(w) @ V^dagger).
eig  (from `Eig`)  : for a GENERAL square operator; eigenvalues are complex and
                     the eigenvectors are only invertible (M = V @ diag(w) @
                     V^{-1}), NOT unitary.

Parameters
----------
Tin : Tensor or UniTensor
    A square matrix (Hermitian for `eigh`).
is_V : bool, optional
    If True (default) return the eigenvectors as well; if False return only
    [eigvals].
row_v : bool, optional
    If True, return the eigenvectors in row form.

Returns
-------
list of Tensor (or list of UniTensor)
    RETURN ORDER (verified by probe): [eigvals, eigvecs] -- the eigenvalues are
    at index 0 (values-first, matching numpy.linalg.eigh's (w, v) order and
    consistent with svd's values-first order). eigvecs is present only if
    is_V is True.

Notes
-----
Renamed from `Eigh`/`Eig` (C1/N1). Use `eigh` when the operator is Hermitian
(cheaper and yields an orthonormal basis); use `eig` only for genuinely
non-Hermitian operators. `eigh` is the spectral primitive behind CTMRG corner
diagonalization and symmetric-gauge truncation.
```

### QR family: `Qr`, `Qdr`

```
Orthogonal-triangular decompositions of a rank-2 Tensor / matricized UniTensor.

qr  (from `Qr`)  : M = Q @ R, with Q orthonormal and R upper-triangular.
qdr (from `Qdr`) : M = Q @ diag(D) @ R, factoring the scale out of R.

Parameters
----------
Tin : Tensor or UniTensor
    The operand.
is_tau : bool, optional
    If True, also return the Householder reflectors `tau` as the final element.

Returns
-------
list of Tensor (or list of UniTensor)
    RETURN ORDER (verified by probe): `qr` returns [Q, R] (the ORTHOGONAL factor
    Q is at index 0 -- the OPPOSITE convention from svd/eigh, which put the
    spectrum first). `qdr` returns [Q, D, R] with the diagonal D at index 1 (the
    MIDDLE, not index 0). Householder reflectors, if requested, are appended.

Notes
-----
Renamed from `Qr`/`Qdr` (C1/N1). The factor-ordering difference relative to
svd/eigh is an internal inconsistency (Consistency finding C3): there is no
single rule for where the diagonal/spectrum factor lands, so always destructure
by the documented positions above rather than assuming a shared convention.
```

### Matrix functions: `Conj`, `Conj_`, `Inv`, `Inv_`, `InvM`, `InvM_`, `Pow`, `Pow_`, `Abs`, `Abs_`, `Exp`, `Exp_`, `Expf`, `Expf_`, `ExpH`, `ExpM`

```
Element-wise and matrix functions, each with a pure form and (where meaningful)
an in-place `_` form that mutates its argument and returns None.

conj / conj_ (from `Conj`/`Conj_`)   : complex conjugation.
abs / abs_   (from `Abs`/`Abs_`)      : element-wise |.|.
exp / exp_   (from `Exp`/`Exp_`)      : element-wise exponential.
expf / expf_ (from `Expf`/`Expf_`)    : single-precision element-wise exp.
pow / pow_   (from `Pow`/`Pow_`)      : element-wise power p.
inv / inv_   (from `Inv`/`Inv_`)      : ELEMENT-WISE reciprocal x -> 1/x, with an
                                        optional `clip` floor (|x| <= clip -> 0).
                                        This is NOT a matrix inverse.
inv_m / inv_m_ (from `InvM`/`InvM_`)  : TRUE matrix inverse (inv_m(M) @ M == I).
exp_h (from `ExpH`)                   : matrix exp of a HERMITIAN UniTensor,
                                        exp(a*H + b) (UniTensor-only).
exp_m (from `ExpM`)                   : matrix exp of a GENERIC UniTensor
                                        (UniTensor-only).

Parameters
----------
Tin : Tensor or UniTensor
    The operand (UniTensor-only for `exp_h`/`exp_m`).
p : float
    The exponent (for `pow`/`pow_`).
clip : float, optional
    Pseudo-inverse floor for `inv`/`inv_` (default -1 = no clip).
a, b : scalar
    Coefficients for `exp_h`/`exp_m` (exp of a*Op + b).

Returns
-------
Tensor or UniTensor, or None
    The pure forms return a new object (B1); the `_`-suffixed forms mutate the
    argument in place and return None (verified by probe).

Notes
-----
Renamed from the Capitalized forms (C1/N1). `inv` (element-wise) and `inv_m`
(matrix inverse) are DIFFERENT operations behind near-identical names
(Consistency finding C4) -- the `_m` marks the matrix form. `exp_h`/`exp_m`
should gain in-place `exp_h_`/`exp_m_` counterparts for N2 symmetry (C5). `exp_h`
is the imaginary-/real-time gate primitive for MERA/CTMRG evolution.
```

### Reductions: `Det`, `Norm`, `Trace`, `Sum`, `Max`, `Min`, `Diag`

```
Scalar-valued reductions and the diagonal constructor.

det (from `Det`)     : determinant (scalar Tensor).
norm (from `Norm`)   : Frobenius/2-norm (scalar Tensor). NOTE: returns a Tensor
                       even for a UniTensor argument (verified by probe) -- it
                       does not preserve the UniTensor type.
trace (from `Trace`) : partial trace over axisA, axisB (default 0, 1); for a
                       UniTensor the axes may be given as labels.
sum (from `Sum`)     : sum of all elements.
max / min (from `Max`/`Min`) : extremal element (real part for complex).
diag (from `Diag`)   : map a 1-D vector to a diagonal matrix (and extract the
                       diagonal of a matrix).

Parameters
----------
Tn / Tin : Tensor (or UniTensor for `trace`/`norm`)
    The operand.
axisA, axisB : int or str
    The two axes to trace over (`trace`); labels allowed for a UniTensor.

Returns
-------
Tensor
    A scalar Tensor for the reductions; a matrix/vector for `diag`.

Notes
-----
Renamed from the Capitalized forms (C1/N1). Reductions correctly have no in-place
`_` form (N2 vacuous, C5). `diag` is used to rebuild diag(S)/diag(D) from an svd/
qdr result when reconstructing or contracting a decomposition.
```

### Contraction & products: `Matmul`, `Matmul_dg`, `Dot`, `Vectordot`, `Outer`, `Kron`, `Tensordot`, `Tensordot_dg`, `Directsum`

```
Dense contraction and product primitives (Tensor operands).

matmul / matmul_dg (from `Matmul`/`Matmul_dg`) : matrix product; the `_dg` form
                       takes one operand as a stored diagonal (1-D) for speed.
dot (from `Dot`)     : generic dot product (dispatches by rank).
vectordot (from `Vectordot`) : 1-D inner product, with optional conjugation of
                       the first operand (`is_conj`).
outer (from `Outer`) : outer product.
kron (from `Kron`)   : Kronecker product (with optional left/right padding).
tensordot / tensordot_dg (from `Tensordot`/`Tensordot_dg`) : contract T1, T2 over
                       the axis lists indices_1, indices_2; `_dg` for a diagonal
                       operand. `cacheL`/`cacheR` permit reusing contiguous
                       buffers.
direct_sum (from `Directsum`) : direct sum of T1, T2 over shared_axes.

Parameters
----------
T1, T2 : Tensor
    The operands.
indices_1, indices_2 : sequence of int
    Axes of T1/T2 to contract (`tensordot`).
is_conj : bool, optional
    Conjugate the first operand (`vectordot`).
cacheL, cacheR : bool, optional
    Buffer-reuse hints (`tensordot`).

Returns
-------
Tensor
    The contracted / product tensor.

Notes
-----
Renamed from the Capitalized forms (C1/N1). `tensordot` is the pairwise-
contraction primitive behind `essential-api.md` step (b); for labelled network
contraction prefer `UniTensor.contract`/`Network`/`ncon`.
```

### BLAS & iterative solvers: `Axpy`, `Axpy_`, `Gemm`, `Gemm_`, `Ger`, `Lstsq`, `Tridiag`, `Lanczos`, `Lanczos_Exp`, `Arnoldi`, `Lanczos_ER`, `Lanczos_Gnd`, `Lanczos_Gnd_Ut`

```
BLAS-level kernels and iterative (Krylov) solvers.

axpy / axpy_ (from `Axpy`/`Axpy_`) : a*x (+y); the `_` form is in-place.
gemm / gemm_ (from `Gemm`/`Gemm_`) : general matrix multiply a*x@y (+b*c);
                       `gemm_` writes into c in place.
ger (from `Ger`)     : rank-1 update a*x⊗y.
lstsq (from `Lstsq`) : least-squares solve of A x = b (returns solution +
                       diagnostics).
tridiag (from `Tridiag`) : eigensolve a symmetric tridiagonal (diag A, off-diag
                       B); returns eigenvalues (+ eigenvectors if is_V).
lanczos (from `Lanczos`)       : Hermitian iterative eigensolver on a LinOp.
lanczos_exp (from `Lanczos_Exp`) : Krylov approximation of exp(tau*Hop)·v.
arnoldi (from `Arnoldi`)       : non-Hermitian iterative eigensolver on a LinOp.
lanczos_er / lanczos_gnd / lanczos_gnd_ut (NEW bindings of `Lanczos_ER` /
                       `Lanczos_Gnd` / `Lanczos_Gnd_Ut`) : the explicit-restart
                       and ground-state Lanczos variants that C++ declares but
                       whose Python bindings are currently commented out (Parity
                       finding P1) -- restore them under these names.

Parameters
----------
Hop : LinOp
    The linear operator (for the iterative solvers).
Tin / v : Tensor or UniTensor
    The seed vector.
a, b, tau : Scalar
    BLAS coefficients / evolution step.
CvgCrit, Maxiter, k, is_V, which, ncv : optional
    Iterative-solver controls (convergence, iterations, #eigenpairs, return
    eigenvectors, target end of spectrum, Krylov dimension).

Returns
-------
Tensor / UniTensor / list
    The BLAS kernels return the result tensor (or None for the `_` in-place
    forms); the iterative solvers return a list of eigenpairs (or the evolved
    vector for `lanczos_exp`).

Notes
-----
Renamed from the Capitalized forms (C1/N1). `axpy_`/`gemm_` are correct N2
in-place pairs. `lanczos_er`/`lanczos_gnd`/`lanczos_gnd_ut` must be re-bound: the
C++ functions exist but the pybind and conti.py wrappers are both commented out
(P1), so they are unreachable today.
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.
Because *every* member is renamed under N1 (C1), the table lists all 53 renames
plus the three restored `Lanczos_*` bindings and the two structural changes
(`is_U`/`is_vT` unification, `InvM`→`inv_m` disambiguation). No member is removed.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Svd` | `svd` | N1 casing (C1) |
| `Svd_truncate` | `svd_truncate` (adopt `is_U`/`is_vT`) | N1 casing (C1) + flag unification (C2) |
| `Gesvd` | `gesvd` | N1 casing (C1) |
| `Gesvd_truncate` | `gesvd_truncate` | N1 casing (C1) |
| `Rsvd` | `rsvd` | N1 casing (C1) |
| `Hosvd` | `hosvd` | N1 casing (C1) |
| `Eigh` | `eigh` | N1 casing (C1) |
| `Eig` | `eig` | N1 casing (C1) |
| `Qr` | `qr` | N1 casing (C1) |
| `Qdr` | `qdr` | N1 casing (C1) |
| `Rand_isometry` | `rand_isometry` | N1 casing (C1) |
| `Conj` / `Conj_` | `conj` / `conj_` | N1 casing (C1) |
| `Inv` / `Inv_` | `inv` / `inv_` | N1 casing (C1); element-wise reciprocal (C4) |
| `InvM` / `InvM_` | `inv_m` / `inv_m_` | N1 casing (C1) + disambiguate matrix inverse (C4) |
| `Pow` / `Pow_` | `pow` / `pow_` | N1 casing (C1) |
| `Abs` / `Abs_` | `abs` / `abs_` | N1 casing (C1) |
| `Exp` / `Exp_` | `exp` / `exp_` | N1 casing (C1) |
| `Expf` / `Expf_` | `expf` / `expf_` | N1 casing (C1) |
| `ExpH` | `exp_h` (+ new `exp_h_`) | N1 casing (C1) + N2 in-place gap (C5) |
| `ExpM` | `exp_m` (+ new `exp_m_`) | N1 casing (C1) + N2 in-place gap (C5) |
| `Det` | `det` | N1 casing (C1) |
| `Norm` | `norm` | N1 casing (C1) |
| `Trace` | `trace` | N1 casing (C1) |
| `Sum` | `sum` | N1 casing (C1) |
| `Max` | `max` | N1 casing (C1) |
| `Min` | `min` | N1 casing (C1) |
| `Diag` | `diag` | N1 casing (C1) |
| `Matmul` | `matmul` | N1 casing (C1) |
| `Matmul_dg` | `matmul_dg` | N1 casing (C1) |
| `Dot` | `dot` | N1 casing (C1) |
| `Vectordot` | `vectordot` | N1 casing (C1) |
| `Outer` | `outer` | N1 casing (C1) |
| `Kron` | `kron` | N1 casing (C1) |
| `Tensordot` | `tensordot` | N1 casing (C1) |
| `Tensordot_dg` | `tensordot_dg` | N1 casing (C1) |
| `Directsum` | `direct_sum` | N1 casing (C1) |
| `Axpy` / `Axpy_` | `axpy` / `axpy_` | N1 casing (C1) |
| `Gemm` / `Gemm_` | `gemm` / `gemm_` | N1 casing (C1) |
| `Ger` | `ger` | N1 casing (C1) |
| `Lstsq` | `lstsq` | N1 casing (C1) |
| `Tridiag` | `tridiag` | N1 casing (C1) |
| `Lanczos` | `lanczos` | N1 casing (C1) |
| `Lanczos_Exp` | `lanczos_exp` | N1 casing (C1) |
| `Arnoldi` | `arnoldi` | N1 casing (C1) |
| *(C++-only, unbound)* `Lanczos_ER` | `lanczos_er` (new binding) | restore dead binding (P1) |
| *(C++-only, unbound)* `Lanczos_Gnd` | `lanczos_gnd` (new binding) | restore dead binding (P1) |
| *(C++-only, unbound)* `Lanczos_Gnd_Ut` | `lanczos_gnd_ut` (new binding) | restore dead binding (P1) |

Beyond the casing rename, only three things change behaviorally: (1) `svd`/
`svd_truncate` adopt `gesvd`'s `is_U`/`is_vT` flag pair (C2); (2) `exp_h`/`exp_m`
gain in-place `_` counterparts (C5); (3) the three `lanczos_*` solvers are
re-bound (P1). The **return order** of every decomposition is left exactly as it
is at runtime (renaming the tuple order would silently break existing callers) —
C3 is resolved by documentation here, with a v2 named-tuple return recommended
separately.
