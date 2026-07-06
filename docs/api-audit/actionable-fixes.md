# Cytnx 1.1.0 — Actionable fixes (the fix-now summary)

Every finding a maintainer can act on **now**, independent of the API redesign.
This is the immediately-shippable value of the audit: correctness bugs, binding
defects, and capability gaps — *not* the naming/consistency recommendations
(those live in the per-class docs' `# R.` sections and the master index).

**Derivation.** Per the method spec §4.3, this is **derived** by filtering the
findings tables to the actionable Types — `correctness`, `binding fidelity`,
`capability gap`, and unbound/commented-out C++ — and excluding
`naming`/`redundancy`/`ordering`/`copy-view`-documentation.

**Provenance.** `UniTensor` rows come from the categorized method docs
(`UniTensor/NN-*.md`, id form `UT-x#`), each probe-verified against the
`cytnx==1.1.0` wheel and — where marked ⚙ — a raw-C++ probe. Other-class rows
(`B-#`) come from the v1 audit (`summary.md`), pending re-audit under the new
method; they are equally real and probe-verified.

Severity: **Critical** = crash / silent-wrong / data corruption · **High** =
wrong-for-some-inputs or the binding drops C++ functionality · **Medium** =
capability gap · **Low** = binding hygiene / cosmetic-but-fixable.

---

## Confirmed bugs — Critical

| Bug | Source | Evidence | Fix |
|---|---|---|---|
| **`Save`→`Load` corrupts a non-empty tensor name** — `_Load` does `std::string(cname)` on a non-NUL-terminated `malloc(len_name)` buffer (heap over-read). | UniTensor · UT-IO5 ⚙ | probe (batch round-trip corrupts names); `src/UniTensor.cpp:119-126` vs `_Save:57-62` | `std::string(cname, len_name)` (length-delimited). Plain memory-safety bug. |
| **`//` is TRUE division, not floor** — `(u*7)//2 → 3.5`; `__floordiv__`/`_r`/`_i` all route to `Div`. | UniTensor · UT-A1 | probe (value == 3.5); `unitensor_py.cpp:1223` | Implement floor semantics, or remove `__floordiv__` entirely. |
| **`Network.Contract(...).Launch()` SEGFAULTS** (SIGSEGV) instead of raising — the advertised one-shot factory is unusable. | Network · B-1 (= UniTensor UT-N6) | subprocess returncode −11 | Remove `Network.Contract`; `ncon` is the working replacement. |
| **`qgates.hadamard()` is not unitary** — returns unnormalized `[[1,1],[1,-1]]` (missing `1/√2`), so `H·H† == 2·I`. | qgates · B-6 | probe (H·H† ≠ I) | Multiply by `1/√2`. |
| **`Symmetry.FermionParity.check_qnums` rejects every non-empty input** — `Symmetry.cpp:170` compares against the sentinel `n`(-2) instead of the bound `2`; `check_qnums([0])` is `False` while `check_qnum(0)` is `True`. | Symmetry · B-3 | probe (self-contradiction on one object) | Compare against literal `2`. |
| **`Bond.getDegeneracy` ships broken in the wheel** — the installed `Bond_conti.py` stacks two `@add_method` defs; the winner references undefined `lqnum`, so every call raises. | Bond · B-2 | probe (`TypeError`/`NameError` on every call) | Ship the repo's un-shipped fix. Highest-severity Bond bug. |
| **`Scalar(2)` / `Scalar(True)` are silently `Uint64`** — single-arg overload resolution walks `uint64` before `int64`/`bool`, so positive ints/bools become unsigned and later subtraction underflows. | Scalar · B-11 | probe (dtype is Uint64) | Bind the dtype-picking 2-arg constructor; fix overload order. |
| **`Tensor @=` is not in-place** — `Tensor_conti.py` defines `__imatmul` (missing trailing `__`), so `@=` falls back to `__matmul__` and rebinds to a fresh object. | Tensor · B-5 | probe (`t is not` its former self) | Rename to `__imatmul__`. |

## Confirmed bugs — High

| Bug | Source | Evidence | Fix |
|---|---|---|---|
| **`get_elem` binds only 4 of 11 dtypes** — reads `double/float/complex*` only, raises on the 7 integer/bool dtypes, while `item`/`set_elem` cover all 11. | UniTensor · UT-E1 ⚙ | probe (raises on int tensor); C++ `get_elem<T>` covers 11 | Bind `get_elem` for all 11 dtypes (mirror the `item` ladder). |
| **`normal_`/`uniform_` return `None`, not `self`** — the pybind lambda drops C++'s chainable `UniTensor& normal_(...)`. | UniTensor · UT-G5 ⚙ | probe (returns None); `hpp:5964` vs lambda `:1581` | Have the lambda `return self`. Restores C++ fidelity + chaining. |
| **Pickle is a broken half-protocol** — `__getstate__` present (inherited `object`), `__setstate__` absent, `pickle.dumps(ut)` raises `TypeError`. | UniTensor · UT-IO2 | probe (dumps raises); no `py::pickle` | Implement `py::pickle` over `save`/`load`, or remove the stale `__getstate__`. |
| **`copy.deepcopy(ut)` raises `TypeError`** — `__deepcopy__` is bound to `clone` (no `memo` param). | UniTensor · UT-T5 | probe (deepcopy raises); `:512` | Bind a `[](self, py::dict memo){ return self.clone(); }`. |
| **`getTotalQnums`/`get_blocks_qnums` are advertised but raise on every tensor type.** | UniTensor · UT-M6 | probe (raise on Dense and Block); `hpp:4743,4750` "@note not support" | Implement for Block tensors (their documented domain) or remove. |
| **`Bond.group_duplicates_` bound to the wrong (const, copy) overload** — despite its `_` it does not mutate; the `mapper` out-arg is never filled. | Bond · B-4 | probe (receiver unchanged) | Bind the true in-place C++ overload. |
| **`Scalar.complex()` of a real Scalar raises** — `__complex__` uses `imag()`, undefined for real subtypes, though C++ `complex128(realScalar)` succeeds. | Scalar · B-7 | probe (`complex(Scalar(3.0))` raises) | Build `__complex__` to return `(re+0j)` for real subtypes. |
| **`Scalar +=` not equivalent to `+`** — `real += complex` raises while binary `real + complex` promotes. | Scalar · B-8 | probe (`+=` raises) | Make in-place promote like the binary op. |
| **`Storage.==` raises on a dtype mismatch** instead of returning `False` (and `!=` delegates to it). | Storage · B-9 | probe (`a == b` throws across dtypes) | Return `False` on dtype mismatch (Python total-`==`). |
| **`Network.clone()` silently drops placed tensors**, yet `isAllset()` returns a misleading `True`; `clone().Launch()` fails. | Network · B-10 | probe | Deep-copy the tensors in `clone`. |
| **`fermionParity.EVEN` is truthy** though its value is `0`/`false` (pybind wrapper has no `__bool__`), so `if parity:` fires for both parities. | enums · B-12 | probe (`bool(EVEN) is True`) | Add `__bool__`/`__int__`, or forbid truthiness. |
| **`combineBond` (singular) is unbound; the bound deprecated `combineBonds` returns `None`.** | UniTensor · UT-S5 ⚙ | probe (`combineBond` absent); C++ `combineBond` exists | Bind `combine_bonds` (returns self); deprecate `combineBonds`. |
| **`astype`/`to` short-circuit to `is self` on a no-op** — a Python identity raw C++ does not have (C++ always returns a fresh wrapper). | UniTensor · UT-T1 ⚙ | probe (`astype(same) is self`); C++ returns distinct | Fold the short-circuit into the pybind lambda; document the aliasing. |

## Gaps — Medium (capability)

| Gap | Source | Evidence | Fix |
|---|---|---|---|
| **No numpy bridge** — `.numpy()` / `from_numpy` both absent on `UniTensor` (unlike `Tensor`). | UniTensor · UT-C3 / UT-T6 | probe (absent) | Add the `.numpy()` / `from_numpy` pair. |
| **`%` (`__mod__`) absent** though C++ has `linalg::Mod`/`operator%` for the scalar case. | UniTensor · UT-A2 ⚙ | probe (`__mod__` absent); C++ scalar `%` works | Bind `%` for the scalar case; note tensor⊗tensor `linalg::Mod` is a `[Developing]` stub. |
| **`linalg.Add`/`Sub`/`Mul`/`Div`/`Mod` absent from `cytnx.linalg`** though the C++ overloads exist. | UniTensor · UT-X5 | probe (absent) | Bind the four named free functions (Capitalized) for operator parity. |
| **`Lanczos_ER`/`Lanczos_Gnd`/`Lanczos_Gnd_Ut` commented out on both pybind and conti sides.** | UniTensor · UT-K3 | probe (absent); `linalg_py.cpp:999-1019` `/* */` | Re-enable the UniTensor ground-state convenience, or delete deliberately. |
| **No public in-place inverse** — `Inv`/`reciprocal` exists but the in-place form is bound only as the leaked raw `cInv_`. | UniTensor · UT-A5 ⚙ | probe (`Inv_` absent); C++ `Inv_` exists | Add public `reciprocal_` returning self, over C++ `Inv_`. |
| **`reshape`/`reshape_` bound as `(*args,**kwargs)`**, erasing the C++ `(new_shape, rowrank)` signature (no introspection). | UniTensor · UT-S3 | probe (`inspect.signature` fails) | Bind an explicit typed signature. |
| **`Lanczos_Exp` is UniTensor-only** — no `Tensor` start-vector overload, unlike `Lanczos`/`Arnoldi`. | UniTensor · UT-K5 | probe | Add a `Tensor` overload for parity. |
| **`Contracts` deprecation is silent in Python** — the C++ `[[deprecated]]` never crosses pybind. | UniTensor · UT-N3 | probe (no warning) | Emit a real `DeprecationWarning`, then remove. |

## Binding fidelity / identity — Medium

| Item | Source | Evidence | Fix |
|---|---|---|---|
| **In-place operators drop identity** — `__iadd__`/`__isub__`/`__imul__`/`__itruediv__`/`__ifloordiv__` mutate the receiver but return a *new* wrapper, so `a += x` rebinds `a`. | UniTensor · UT-A7 | probe (`src is not` alias) | Return self from the augmented-assign lambdas. |
| **`twist_` drops identity** — returns a shared-data wrapper that is not the same object; C++'s `UniTensor& self`-return is dropped. | UniTensor · UT-S7 ⚙ | probe; C++ returns `&*this` | `return &self.twist_(i)` (match `permute_`). |
| **`relabel_` raw binding returns `None`** — chainability comes from a conti.py wrapper. | UniTensor · UT-L2 ⚙ | probe; C++ `relabel_` returns `&*this` | Have the pybind return self directly, dropping the shim. |
| **`__repr__` prints and returns `''`** rather than returning the repr text. | UniTensor · UT-IO7 | probe (`repr(ut) == ''`) | Build the representation into a string and return it. |
| **`Save`/`Load` extension handling is asymmetric** — `Save` auto-appends `.cytnx` (deprecated), `Load` does not. | UniTensor · UT-IO4 | probe | Require the explicit extension on both. |

## Low — binding hygiene & cosmetic

| Item | Source | Fix |
|---|---|---|
| **~21 leaked raw bindings in `dir(UniTensor)`** — `c_set_name`/`c_relabel_`/`c_relabels_`/`c_set_label(s)`/`c_set_rowrank_` (UT-L10), `make_contiguous`/`ctag`/`ctruncate_` (UT-S4/S6/S9), `c_at` (UT-E11), `cConj_`/`cDagger_`/`cPow_`/`cTrace_`/`cTranspose_`/`cnormalize_`/`cInv_`/`c__ipow__` (UT-A4), `astype_different_type`/`to_different_device` (UT-T2), `cfrom` (UT-T7). | UniTensor | Underscore-prefix or inline each wrapper into its pybind lambda so it leaves `dir()`; consolidated in `inventory.md`'s Internal/plumbing section. |
| **Deprecated redundant label methods** — `set_labels`, `relabels`, `relabels_` (bound but deprecated; internally call `relabel_`). | UniTensor · UT-L4/L6 | Remove; use `relabel_` (migration note). |
| **`get_blocks_` accepts a misspelled `slient` kwarg** (warns + forwards to `silent`). | UniTensor · UT-E5 | Keep `slient` as a `FutureWarning` alias one release, then delete. |
| **`Inv`/`InvM` near-name collision** — element-wise reciprocal vs matrix inverse. | UniTensor · UT-X4 | Rename the element-wise op → `Reciprocal` (free) / `reciprocal` (member); keep `InvM`. |
| **Leftover `std::cout` debug leak** in `Tensor`'s bare-1-D-slice `__getitem__` — `t[0:2]` prints to uncapturable stdout. | Tensor · B-13 | Remove the debug print (`tensor_py.cpp:355`). |

---

## Notes

- **⚙ = both-sides verified**: the finding's raw-C++ behavior was confirmed by a
  C++ probe (`docs/api-audit/probes/cpp/`) linked against a source-built
  `libcytnx`, not only inferred from source.
- The `B-#` (non-UniTensor) rows are from the v1 audit and will be refreshed as
  each class is redone under the new method (§9). They remain accurate for
  1.1.0.
- Naming/consistency recommendations (snake_case renames, keyword-only, etc.)
  are deliberately **excluded** here — see the per-class `# R.` sections and the
  master recommendation index for those.
