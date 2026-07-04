# Essential API — derived minimal set for TRG / HOTRG / CTMRG / MERA

This is deliverable #6 of the audit: the **minimal recommended-API subset**
sufficient to implement the four reference tensor-network coarse-graining
algorithms — **TRG**, **HOTRG**, **CTMRG**, **MERA**. It is *derived*, not
curated: per `00-methodology.md` §5, each algorithm is decomposed into its
primitive computational steps (§1 below), each step is mapped to the concrete
API calls that realize it (§2), the union of those calls is taken (§3), and
every member of the union carries a back-reference to at least one
`(algorithm, step)` that requires it. A member used by only one of the four is
still essential — the goal is "sufficient to build *any* of the four," so the
union (not the intersection) defines the set.

All names below are the **recommended, post-rename** spellings from the
per-class documents — `linalg.svd_truncate` (not `Svd_truncate`),
`UniTensor.dagger` (not `Dagger`), `Symmetry.u1` (not `U1`), and so on. Each
per-class doc's `## Recommendation` / `## Change table` is the authority for the
target name; the `keep`/`rename` rationale and the load-bearing behavioral facts
(SVD return order, copy-vs-view, the `Network.Contract` segfault) are cited
inline so the mapping is auditable against those documents rather than asserted.

## Grounding and confidence

Methodology §5.2 designates `cytnx_src/docs/source/example/HOTRG.rst` as the
in-repo reference for grounding the HOTRG mapping in a real call sequence.
**That file is, in this checkout, a bibliography stub — a title, two `:cite:`
directives, and a `.. bibliography::` block, with no code.** There is likewise
no `HOTRG`/`TRG`/`CTMRG`/`MERA` implementation anywhere under `cytnx_src/example`
or `cytnx_src/pytests`. The verb-level grounding therefore comes from the two
in-repo examples that *do* ship working coarse-graining code and exercise every
primitive an HOTRG sweep needs:

- **`iTEBD.rst`** — a full imaginary-time-evolution loop that grounds the exact
  contract → matricize → truncated-SVD → renormalize cycle: `cytnx.Contract`
  (recommended `UniTensor.contract`), `permute_`, `set_rowrank_`,
  `cytnx.linalg.Svd_truncate` (recommended `linalg.svd_truncate`),
  `normalize_`, `relabel_`, `reshape_`, `linalg.ExpH`/`linalg.Kron` (recommended
  `linalg.exp_h`/`linalg.kron`), `get_block_`, `put_block`, `is_diag`, `item`,
  `1./lb` (element-wise reciprocal, recommended `linalg.inv`), and the
  `Bond`/`UniTensor` constructors.
- **`DMRG.rst`** — grounds the reusable-network path
  (`cytnx.Network(...)`/`Launch`, recommended
  `Network.from_string`/`put_unitensor`/`launch`), the one-sided
  `linalg.Gesvd(is_U=..., is_vT=...)` isometry extraction, and iterative
  solvers.
- **`Bond/combineBonds.py`** — grounds Bond fusion (`combineBonds`, recommended
  `Bond.combine_bond` / `UniTensor.combine_bonds_`) including the symmetric
  `Qs`/`Symmetry.U1()`/`Symmetry.Zn(2)` construction.

Every verb HOTRG requires (pairwise contract, leg permute/reshape/fuse,
row-rank-set matricization, truncated SVD, Hermitian eigendecomposition,
conjugate-transpose, renormalize) appears in that grounded set, so the HOTRG
mapping is call-sequence-accurate even though the HOTRG-specific driver is
absent. **TRG, CTMRG, and MERA have no in-repo example at all** and are mapped
from their standard published definitions (Levin–Nave TRG; Xie et al. HOTRG for
the higher-order truncation; directional corner-transfer-matrix CTMRG; binary
MERA). Individual CTMRG/MERA entries whose *specific* call is
implementation-dependent are flagged **(low-confidence)** in §2 and §3.

## 1. Per-algorithm step decomposition

Each algorithm reduces to the common coarse-graining pattern of §5.1 —
(a) group/matricize bonds, (b) contract, (c) decompose + truncate,
(d) build/apply an isometry or projector, (e) renormalize and iterate — plus the
algorithm-specific step each one adds.

### TRG (Levin–Nave)

Bulk rank-4 tensor `T[u,l,d,r]` on a square lattice.

1. **Group/matricize.** Reorder legs and set the row/column split two ways —
   `(u,l)|(d,r)` and `(l,d)|(r,u)` — so each diagonal split is a matrix.
2. **Split by truncated SVD.** SVD-truncate each matricization to χ, factoring
   `T ≈ S1·S3` and `T ≈ S2·S4`; absorb `√s` into each factor to form four
   rank-3 half-tensors.
3. **Rebuild the plaquette.** Contract the four half-tensors around a plaquette
   into a new rank-4 tensor `T'`, rotated 45°.
4. **Renormalize + iterate.** Divide `T'` by its norm and repeat until the free
   energy converges.

### HOTRG (Xie et al.)

Bulk rank-4 tensor `T`; coarse-grains one direction per half-step.

1. **Stack + contract.** Contract two bulk tensors along the coarsening
   direction, then fuse the two parallel transverse legs into one χ²-dim leg.
2. **Higher-order truncation** *(HOTRG-specific)*. Form the Hermitian density
   matrix of the fused leg (contract the stacked tensor with its
   conjugate-transpose over all other legs), Hermitian-eigendecompose it, keep
   the χ leading eigenvectors as an isometry `U`; build the same from the
   opposite side and keep the lower-truncation-error `U`.
3. **Apply the isometry.** Contract `U` and `U†` onto the stacked tensor to
   project the fused leg back down to χ, giving the coarse tensor `T'`.
4. **Renormalize + rotate + iterate.** Normalize `T'`, permute to alternate the
   coarsening direction, repeat.

### CTMRG (directional corner-transfer-matrix RG)

Bulk tensor `a`, environment of four corner matrices `C` (rank-2) and four edge
tensors `T` (rank-3).

1. **Absorb** *(CTMRG-specific)*. Enlarge a corner by contracting `C`, an edge
   `T`, and the bulk `a` (a half-row/column insertion).
2. **Build projectors** *(CTMRG-specific)*. Matricize the enlarged
   corner/quadrant and build a χ-dimensional isometric projector pair `P`, `P̃`
   from its truncated SVD (or QR + truncated SVD; or Hermitian
   eigendecomposition of the corner).
3. **Renormalize corner + edges.** Apply the projectors: `C ← P† C' `,
   `T ← P† T' P`.
4. **Normalize + iterate.** Rescale each environment tensor and repeat the four
   directional moves until the corner spectrum converges.

### MERA (binary MERA)

Disentanglers `u` (rank-4 unitaries) and isometries `w` (rank-3, coarse-grain
2→1 at bond dimension χ).

1. **Ascend / descend.** Contract `u`, `w`, the operator, and their
   conjugate-transposes through the causal cone (the ascending/descending
   superoperators).
2. **Environment build** *(MERA-specific)*. For the tensor being optimized,
   contract everything else in the energy network to form its environment `E`.
3. **SVD update** *(MERA-specific)*. Full-SVD the environment `E = U s V†` and
   replace the tensor by the polar factor `−V U†` — the optimal
   isometry/disentangler given the environment.
4. **Sweep + iterate.** Repeat across all `u`/`w` on every layer until the
   energy converges. (Isometries/disentanglers are seeded from a random
   isometry; χ enters through the isometry's output bond, so truncation is
   structural rather than a separate SVD-truncate.)

## 2. Step → API mapping table

Steps are labelled by the algorithm and the §1 step number. Names are the
recommended post-rename spellings.

| Algorithm | Step | Required recommended API calls |
|---|---|---|
| TRG | 1 group/matricize | `UniTensor.permute`/`permute_`, `UniTensor.reshape`/`reshape_`, `UniTensor.set_rowrank`/`set_rowrank_`, `UniTensor.relabel`/`relabel_` |
| TRG | 2 split + truncate | `linalg.svd_truncate` (`[S,U,vT]`, values-first per linalg C3); `linalg.pow`/`linalg.diag` to form `diag(√S)`; `linalg.matmul` or `UniTensor.contract` to absorb `√S` |
| TRG | 3 rebuild plaquette | `ncon` (or repeated `UniTensor.contract`) |
| TRG | 4 renormalize | `linalg.norm`, `UniTensor.normalize`/`normalize_`, `UniTensor.item` |
| TRG | init (Boltzmann tensor) | `Bond`, `UniTensor.zeros`/`ones`, `UniTensor.put_block`, `UniTensor.get_block`/`get_block_`, `UniTensor.is_diag`, `linalg.exp_h`, `linalg.eigh` + `linalg.pow` (matrix √ of the weight matrix), `Type.Double`/`Device.cpu` |
| HOTRG | 1 stack + fuse | `UniTensor.contract`/`ncon`, `UniTensor.combine_bonds_`, `Bond.combine_bond`, `UniTensor.permute`/`permute_`, `UniTensor.relabel`/`relabel_` |
| HOTRG | 2 higher-order truncation | `UniTensor.dagger`, `UniTensor.conj`, `ncon`/`UniTensor.contract` (density matrix), `linalg.eigh` (Hermitian), `UniTensor.truncate`/`truncate_`; `linalg.svd_truncate`/`linalg.gesvd_truncate` as the SVD alternative to `eigh` |
| HOTRG | 3 apply isometry | `ncon`/`UniTensor.contract`, `Network.from_string`+`put_unitensor`+`launch` (reusable coarse-graining net) |
| HOTRG | 4 renormalize + rotate | `linalg.norm`, `UniTensor.normalize`/`normalize_`, `UniTensor.permute`/`permute_`, `UniTensor.item` |
| HOTRG | init | `Bond`, `UniTensor.zeros`/`eye`, `linalg.kron`, `linalg.exp_h`, `Type.Double`/`Device.cpu` |
| CTMRG | 1 absorb *(low-confidence)* | `ncon`/`UniTensor.contract`, `Network.from_string`+`put_unitensor`+`launch` |
| CTMRG | 2 build projectors *(low-confidence)* | `UniTensor.permute`/`set_rowrank_`+`reshape` (matricize); `linalg.svd_truncate` (SVD projector) **or** `linalg.qr` + `linalg.svd`/`linalg.eigh` (QR/eigendecomposition projector); `linalg.inv`/`linalg.diag` for `S^{-1/2}`; `UniTensor.truncate`/`truncate_` |
| CTMRG | 3 renormalize C/T | `ncon`/`UniTensor.contract`, `UniTensor.dagger`, `UniTensor.conj` |
| CTMRG | 4 normalize + iterate | `linalg.norm`, `UniTensor.normalize`/`normalize_`, `UniTensor.item` |
| CTMRG | init | `Bond`, `UniTensor.eye`/`zeros`, `UniTensor.is_diag`, `UniTensor.put_block`, `Type.Double`/`Device.cpu` |
| MERA | 1 ascend/descend | `ncon`/`UniTensor.contract`, `UniTensor.dagger`, `UniTensor.conj`, `UniTensor.relabel`/`relabel_`, `Network.from_string`+`put_unitensor`+`launch` |
| MERA | 2 environment build *(low-confidence)* | `ncon`/`UniTensor.contract`, `UniTensor.permute`/`set_rowrank_`/`reshape` |
| MERA | 3 SVD update *(low-confidence)* | `linalg.svd` (full, untruncated; `[S,U,vT]`), `UniTensor.dagger`, `UniTensor.contract`/`linalg.matmul` |
| MERA | 4 sweep + init | `linalg.rand_isometry` (seed `u`/`w`), `linalg.norm`, `UniTensor.item`, `linalg.kron` (local Hamiltonian), `linalg.exp_h` (evolution gate), `Type.Double`/`Device.cpu` |
| all four | symmetry-conserving variant | `BD_KET`/`BD_BRA`, `Qs`, `Symmetry.u1`/`Symmetry.zn`, `UniTensor.bonds`/`labels`, `UniTensor.tensordot` (dense fallback) |

## 3. Essential set (the union)

Every line below is one essential member with a back-reference to at least one
algorithm that needs it. Members are grouped by unit; the recommended
(post-rename) name is authoritative and the originating per-class doc governs its
verdict and semantics.

Each entry is kept on a single line so the algorithm trace sits on the same
line as the member name (the Task 9 gate scans line by line).

### `UniTensor`

- `UniTensor.contract` — TRG, HOTRG, CTMRG, MERA: pairwise leg-matched contraction, the core network-building verb (grounded in iTEBD's `cytnx.Contract`).
- `UniTensor.permute` / `permute_` — TRG, HOTRG, CTMRG, MERA: reorder legs to matricize before a decomposition or align a contraction (iTEBD `permute_`); views (UniTensor P6).
- `UniTensor.reshape` / `reshape_` — TRG, HOTRG: fuse/split legs into the matrix an SVD consumes (iTEBD `reshape_`); views (P6).
- `UniTensor.combine_bonds_` — HOTRG: fuse the parallel legs created by stacking two bulk tensors (renamed from `combineBonds`, UniTensor C3; also the TRG plaquette).
- `UniTensor.set_rowrank` / `set_rowrank_` — TRG, HOTRG, CTMRG, MERA: set the row/column split that defines the matricization an SVD/eigh sees (iTEBD `set_rowrank_`).
- `UniTensor.relabel` / `relabel_` — TRG, HOTRG, CTMRG, MERA: rename legs so `contract`/`ncon` bind the intended bonds (iTEBD `relabel_`); metadata-only view (P6).
- `UniTensor.truncate` / `truncate_` — HOTRG, CTMRG: trim an isometry/edge leg to χ after a decomposition (UniTensor decomposition-entry-points).
- `UniTensor.get_block` / `get_block_` — TRG, HOTRG, CTMRG, MERA: reach the dense block to seed or read tensor data (iTEBD `get_block_`); the copy-vs-view pair (P4/B2).
- `UniTensor.put_block` — TRG, CTMRG: install a dense block (diagonal weight / Boltzmann tensor) (iTEBD `put_block`).
- `UniTensor.dagger` — HOTRG, CTMRG, MERA: conjugate-transpose an isometry or environment half (renamed from `Dagger`, UniTensor C2).
- `UniTensor.conj` — HOTRG, CTMRG, MERA: conjugate a tensor when forming a density matrix (renamed from `Conj`, UniTensor C2).
- `UniTensor.normalize` / `normalize_` — TRG, HOTRG, CTMRG, MERA: rescale a renormalized tensor each sweep (iTEBD `normalize_`).
- `UniTensor.item` — TRG, HOTRG, CTMRG, MERA: extract the rank-0 scalar for a norm/energy/convergence test (iTEBD `item`).
- `UniTensor.zeros` / `ones` / `eye` / `identity` — TRG, HOTRG, CTMRG, MERA: build initial bulk and identity/corner tensors (static generators, kept).
- `UniTensor.bonds` / `labels` — TRG, HOTRG, CTMRG, MERA: read leg structure when cloning a tensor's shape or wiring a network (iTEBD `A.bonds()`).
- `UniTensor.is_diag` — TRG, CTMRG: mark diagonal weight/corner tensors so only the diagonal is stored (iTEBD `is_diag=True`).

### `linalg` (all renamed to snake_case, linalg C1)

- `linalg.svd_truncate` — TRG, HOTRG, CTMRG: decompose-and-truncate to bond dimension χ, the central step (c)/(d) (grounded in iTEBD/DMRG `Svd_truncate`); returns `[S, U, vT]`, values at index 0 (linalg C3).
- `linalg.svd` — MERA, CTMRG: full untruncated SVD for the MERA polar/environment update and as the CTMRG projector primitive; `[S, U, vT]`, values-first (linalg C3).
- `linalg.gesvd_truncate` — HOTRG, CTMRG: truncated SVD with independent `is_U`/`is_vT` so one isometry can be dropped (DMRG grounds `Gesvd(is_U=..., is_vT=...)`).
- `linalg.eigh` — TRG, HOTRG, CTMRG: Hermitian eigendecomposition of a density/corner matrix (HOTRG higher-order truncation, CTMRG corner diagonalization) and the matrix √ of the TRG Boltzmann weight; `[eigvals, eigvecs]`, values-first.
- `linalg.qr` — CTMRG: orthonormal factor for a projector/isometry; `[Q, R]`, Q-first (linalg C3) **(low-confidence: the projector may instead be built from `svd_truncate`)**.
- `linalg.diag` — TRG, CTMRG: build `diag(√S)` / `diag(S^{-1/2})` to absorb or invert a singular-value weight.
- `linalg.pow` / `pow_` — TRG: element-wise √ and powers of a singular-value or eigenvalue vector.
- `linalg.inv` — CTMRG, TRG: element-wise reciprocal of a diagonal weight (the iTEBD `1./lb` step); element-wise, not the matrix inverse (linalg C4).
- `linalg.matmul` — TRG: dense matrix product to absorb `√S` weights into factors.
- `linalg.tensordot` — TRG, HOTRG, CTMRG, MERA: the dense pairwise-contraction primitive behind `UniTensor.contract` (essential step (b), linalg docstrings).
- `linalg.kron` — HOTRG, MERA: Kronecker product to assemble a local Hamiltonian / Boltzmann operator (iTEBD `Kron`).
- `linalg.exp_h` — TRG, HOTRG, MERA: matrix exponential of a Hermitian operator (Boltzmann weight, MERA evolution gate) (iTEBD `ExpH`; renamed from `ExpH`, linalg C1).
- `linalg.norm` — TRG, HOTRG, CTMRG, MERA: Frobenius norm for the per-step renormalization scale (returns a scalar `Tensor`).
- `linalg.rand_isometry` — MERA: random isometry to initialize disentanglers and isometries **(low-confidence: a specific init is implementation-dependent)**.

### `network` / `ncon`

- `ncon` — TRG, HOTRG, CTMRG, MERA: one-shot full-network contraction by the ncon index convention, the working contraction path (`Network.Contract` is removed for segfaulting on `Launch`, network P1).
- `Network.from_string` — HOTRG, CTMRG, MERA: load a reusable coarse-graining network skeleton (renamed from `FromString`; DMRG grounds the `Network` path).
- `Network.put_unitensor` — HOTRG, CTMRG, MERA: place tensors into the skeleton's named slots (renamed from `PutUniTensor`).
- `Network.launch` — HOTRG, CTMRG, MERA: contract the fully-set network (renamed from `Launch`; order set beforehand via `set_order`, network P3).

### `Bond` / `Symmetry` / enums

- `Bond` — TRG, HOTRG, CTMRG, MERA: the constructor declaring each leg's dimension and direction (iTEBD / `combineBonds` grounding).
- `Bond.combine_bond` — HOTRG: fuse `Bond` objects when pre-building a fused leg (renamed from `combineBonds`, Bond change table; `combineBonds` itself removed as a C++-deprecated duplicate).
- `BD_KET` / `BD_BRA` — TRG, HOTRG, CTMRG, MERA: bra/ket leg directions for tagged/physical tensors (the `BD_IN`/`BD_OUT` aliases are removed, enums C4).
- `Qs` — TRG, HOTRG, CTMRG, MERA: quantum-number sector helper for the symmetry-conserving variants (combineBonds grounding).
- `Symmetry.u1` / `Symmetry.zn` — TRG, HOTRG, CTMRG, MERA: U(1)/Zₙ symmetry objects for block-sparse bulk tensors (renamed from `U1`/`Zn`, Symmetry C1).
- `Type.Double` / `Device.cpu` — TRG, HOTRG, CTMRG, MERA: the dtype/device codes passed to every constructor (enum members, N1-exempt).

### Deliberately excluded

`LinOp` + `linalg.lanczos`/`linalg.arnoldi` are **not** in the essential set: the
four reference algorithms in their dense reference form diagonalize small
χ-dimensional density/corner matrices with `linalg.eigh` and update MERA tensors
with a full `linalg.svd`, so no iterative Krylov operator is required. They
become essential only for the `tn_algo` (MPS/MPO/DMRG) family, which
`00-methodology.md` §5 explicitly excludes from this derivation.
`Network.Contract` is excluded because it segfaults (network P1); `ncon` is its
working replacement.
