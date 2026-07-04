# Cytnx 1.1.0 API audit — cross-cutting summary & master index

This is the top-level entry point for the Cytnx 1.1.0 API-audit project. It
does two things: **(1)** synthesizes the *global* findings — the naming and
behavior patterns that recur across many units, not just one class — and
**(2)** aggregates every per-class recommendation into one **master keep / add /
rename / remove index**.

Everything here is traceable back to source. The conventions cited (`N1`…`N5`,
`B1`…`B5`) are defined in [`00-methodology.md`](00-methodology.md); the
per-unit finding IDs cited (e.g. "UniTensor C1", "network P1", "enums C1") are
the exact `## Parity findings` / `## Consistency findings` entries in the
corresponding `per-class/<Unit>.md`. The units audited are `Bond`, `Symmetry`,
the enum/config set (`Type`/`Device`/`SymType`/`bondType`/`fermionParity`),
`UniTensor`, `Tensor`, `Storage`, `Scalar`, `linalg`, the network family
(`Network`/`ncon`/`LinOp`), and the operation submodules
(`algo`/`random`/`physics`/`qgates`). The **derived minimal subset** sufficient
to build TRG/HOTRG/CTMRG/MERA is in [`essential-api.md`](essential-api.md); it
uses the post-rename spellings this summary recommends.

Every behavioral claim below was verified by an executed probe under
`docs/api-audit/probes/<Unit>.py` (all `[PASS]`, exit 0, run against the
venv-installed `cytnx==1.1.0` via `tools/env.sh`); the owning per-class doc
names the specific assertion.

---

## 1. Cross-cutting findings

Eight patterns span multiple units. Each is a *global* problem — fixing it in
one class without the others would leave the surface incoherent.

### X1 — The capitalized-method pattern (N1) is nearly universal

N1 requires callable public members (methods and free functions; enum *values*
and class names are exempt) to be `snake_case`. In the current wheel almost
every non-trivial callable is capitalized or camelCase. This is the single most
pervasive finding in the whole audit — it appears in **every** class-bearing
unit:

| Unit | Owning finding | Scope of the N1 violation |
|---|---|---|
| `linalg` | linalg C1 | **all 53 free functions** Capitalized (`Svd`, `Eigh`, `Qr`, `Matmul`, `ExpH`, `Directsum`, …) — not one is `snake_case` |
| `Tensor` | Tensor C1 | 23 capitalized callables (`Conj`, `Svd`, `Eigh`, `Trace`, `InvM`, `Load`, `Tofile`, …) |
| `UniTensor` | UniTensor C2 | 11 capitalized + camelCase (`Conj`, `Dagger`, `Transpose`, `Inv`, `Pow`, `Trace`, `Norm`, `Init`, `Load`, `Save`, `Nblocks`, `getTotalQnums`, `combineBonds`) |
| `Bond` | Bond C4 | `Init`, `Load`, `Save`, `Nsym`, `combineBond`/`combineBond_`/`combineBonds`/`combineBonds_`, `getDegeneracies`, `getDegeneracy` |
| `algo` | operations C1 | **all 6 functions** Capitalized (`Sort`, `Concatenate`, `Vstack`, `Hstack`, `Vsplit`, `Hsplit`) |
| `Storage` | Storage C1 | `Init`, `Load`, `Save`, `Tofile`, `Fromfile` |
| `Symmetry` | Symmetry C1 | `FermionNumber`, `FermionParity`, `Load`, `Save`, `U1`, `Zn` |
| `Network` | network C1 | 15 of 18 methods (`PutUniTensor`, `Launch`, `FromString`, `getOrder`, `isAllset`, `PrintNet`, …); only `construct`/`clear`/`clone` comply |
| `Device` | enums C6 | `Print_Property` → `print_property`, `getname` → `get_name` |

Quantified: **well over 110 callables across the nine units above require an N1
rename** (53 in `linalg` alone). The three modules that already comply
(`random`, `physics`, `qgates` — operations doc) and `LinOp` (already
`snake_case`, network C9) are the model. Note the deliberate accepted
exceptions preserved as-is: `matvec` (network C9), and the math-idiom names
`svd`/`qr`/`eigh` that stay compressed.

### X2 — Leaked raw pybind internals on the public surface

A whole layer of implementation-detail bindings — the raw C++ methods that
friendlier Python wrappers call — leak onto the public surface as
non-underscore members. N2 explicitly names the `cConj_`/`cDagger_`/`cInv_`/
`cPow_`/`cTrace_`/`cTranspose_` prefix as "not a recognized convention." Three
sub-families recur:

- **`c`-prefixed in-place primitives.** `UniTensor` (P1/C1): **~18** leaked
  (`cConj_`, `cDagger_`, `cTranspose_`, `cInv_`, `cPow_`, `cTrace_`,
  `cnormalize_`, `ctag`, `ctruncate_`, `cfrom`, `c__ipow__`, `c_at`,
  `c_relabel_`, `c_relabels_`, `c_set_label`, `c_set_labels`, `c_set_name`,
  `c_set_rowrank_`). `Tensor` (P3/C2): **~15** (`cConj_`…`cPow_`, plus
  `c__iadd__`/`c__isub__`/`c__imul__`/`c__itruediv__`/`c__ifloordiv__`/
  `c__ipow__`/`c__imatmul__`). `Bond` (C5): `c_redirect_`,
  `c_getDegeneracy_refarg`, `c_group_duplicates_refarg`. `Storage` (P5/C3): the
  **11** `c_pylist_<dtype>` accessors.
- **`*_different_*` no-op-refusing primitives.** `Tensor` (P3),
  `UniTensor` (C7), `Storage` (P4): `astype_different_type`/`astype_different_dtype`,
  `to_different_device`, and (Tensor) `make_contiguous` — pybind escape hatches
  that exist only so the `_conti.py` wrapper can intercept the same-dtype /
  same-device no-op, and that **raise if called directly** on a no-op.
- The leak is not even internally uniform: `UniTensor` P3 shows `cConj_`
  returns the same wrapper while `cInv_` returns a fresh data-sharing one.

Total: **~50 leaked internal members** across `UniTensor`, `Tensor`, `Bond`,
`Storage`. Almost all should be removed (folded into their clean wrappers); the
sole promotions are `UniTensor.cInv_`→`inv_` (the only in-place inverse, no
clean wrapper, UniTensor C1/P3) and `Bond.c_getDegeneracy_refarg` (kept *only*
until the broken `get_degeneracy` ships fixed, Bond P6). The model-correct
counter-example is `UniTensor.get_block`/`get_block_` (P4/B2): copy-vs-view
distinguished by exactly the trailing `_`.

### X3 — Inconsistent in-place convention (N2)

N2 says: in-place variants carry a trailing `_`, and each in-place op has a
pure counterpart of the same base name. The current surface violates this three
different ways:

- **Wrong marker.** `Scalar` (C1) spells its in-place ops with an `i`-prefix
  (`iabs`, `isqrt`) instead of a trailing `_` → recommend `abs_`, `sqrt_`.
- **In-place op with no `_` suffix.** `UniTensor.combineBonds` (C3) mutates the
  receiver but reads like a pure method → `combine_bonds_`. `Bond.set_type`
  (C1/P5) mutates in place but shares no base name with its pure counterpart
  `retype` → `retype_`. `Bond.clear_type` (C2) is in-place-only with no `_` →
  `clear_type_`.
- **Missing half of the pure/in-place pair.** `UniTensor.Inv` has no pure-named
  in-place partner (only the raw `cInv_`) — C1/C3. `linalg.ExpH`/`ExpM` have no
  `_` form though `Exp` does (linalg C5). `Storage.astype` has no `astype_`
  though its sibling `to`/`to_` is a correct pair (Storage C2). `algo.Sort` has
  no `sort_` (operations C2). `Bond.clear_type` lacks a pure counterpart
  (C2). Conversely `Tensor.flatten_` (C4) returns `None` while its siblings
  `reshape_`/`permute_` return `self` — an in-place *return-convention* split.

The correct N2 pairs already in the wheel — `Bond.redirect`/`redirect_`,
`UniTensor.get_block`/`get_block_`, `Storage.to`/`to_`, `linalg.Conj`/`Conj_` —
are the template.

### X4 — `__repr__`/`__str__` print-and-return-`""`, and uncapturable stdout

A recurring binding shape: `__repr__`/`__str__` print the human-readable block
to stdout (via `std::cout`) and then **return the empty string**, so
`repr(x)`/`str(x)` evaluate to `""`. Found on `Symmetry` (P5), `Bond`
(noted under Symmetry P5 as the identical `bond_py.cpp` pattern), and `Storage`
(P6). The fix is uniform: build the string and *return* it (an `ostringstream`
instead of `std::cout`). `Scalar.__repr__` (C5) already does this correctly and
is the model; `UniTensor`'s `print_*` and `Storage.print_info` (P6) are
correctly `scoped_ostream_redirect`-guarded and capturable.

The related defect is **stdout that Python cannot capture at all** (no
`scoped_ostream_redirect` guard): `Device.Print_Property` (enums P4) writes to
the process's real stdout, and `Tensor`'s bare-1-D-slice `__getitem__` branch
(P5) leaks a leftover `std::cout << start << stop << step` debug line on
`t[0:2]`. Both are invisible to `contextlib.redirect_stdout`. Two C++ methods
that produce info only as a print side-effect have **no direct binding** and
should be exposed: `Symmetry.print_info` (P5) and (as a fixed `__repr__`) the
same for `Bond`.

### X5 — Enum cross-type equality/hash collisions and `export_values()` pollution

The four real enums (`Type`, `SymType`, `bondType`, `fermionParity`) are bound
without per-type identity, so members of *unrelated* enums that wrap the same
integer compare **and hash** equal: `SymType.Z == Type.Void == bondType.BD_REG
== fermionParity.EVEN` are all `True` (all wrap `0`), and a dict keyed on one
silently collides with another (enums C1). All four are also bound with
`export_values()`, dumping ~30 short names (`Double`, `Z`, `U`, `BD_BRA`,
`EVEN`, …) into the top-level `cytnx.*` namespace (enums C2/P8) — while
`Device`, modelled inconsistently as a submodule of bare `int`s (enums C3/P2),
escapes both. This equality footgun leaks outward: `Scalar.dtype()` (Scalar C4)
returns a bare `int` that only compares equal to `Type.Double` *because* of the
very cross-type equality enums C1 recommends removing — so the enums C1 fix
would silently break `s.dtype() == Type.Double`. Fix: type-distinct
`__eq__`/`__hash__` (enums C1), drop `export_values()` for qualified-only access
(enums C2), promote `Device` to a real enum (enums C3), and have `Scalar.dtype`
/ `Tensor.dtype` / `LinOp.dtype` (network P5) return `Type`-typed values.

### X6 — Confirmed correctness bugs (probe-verified)

These are not style issues — each is a demonstrated wrong-or-crashing behavior,
verified by an executed probe assertion. Fixing them is independent of (and
more urgent than) the renames.

| # | Unit / finding | Bug |
|---|---|---|
| B-1 | `Network` P1 (network) | **`Network.Contract(...).Launch()` SEGFAULTS** (SIGSEGV, child returncode -11) instead of raising — the advertised one-shot factory is unusable in 1.1.0 (B4). `ncon` is the working replacement; `Contract` is **removed**. |
| B-2 | `Bond` P6 | **`getDegeneracy` ships broken in the wheel**: the installed `Bond_conti.py` stacks two `@add_method` defs; the winning one requires `return_indices` positionally *and* references an undefined name `lqnum` — so every call raises (`TypeError` / `NameError`). The repo source has an un-shipped fix. Highest-severity Bond finding. |
| B-3 | `Symmetry` P7 | **`FermionParity.check_qnums` rejects every non-empty input**: `Symmetry.cpp:170` compares against the internal sentinel `n` (`-2`) instead of the literal bound `2`, so `check_qnums([0])` is `False` while `check_qnum(0)` is `True` — a self-contradiction on one object. |
| B-4 | `Bond` P4/C3 | **`group_duplicates_` is bound to the wrong C++ overload** (the `const`, copy-returning `group_duplicates`), so despite its trailing `_` it does **not** mutate the receiver, and its `mapper` out-arg is silently never filled. The true in-place C++ mutator is unreachable from Python. |
| B-5 | `Tensor` P4 | **`@=` is not in-place**: `Tensor_conti.py` defines `__imatmul` (missing the trailing `__`), so Python finds no `__imatmul__`, falls back to `__matmul__`, and **rebinds** `t` to a fresh object. The intended in-place matmul is dead code (N2/B1/B5). |
| B-6 | `operations` P8 (qgates) | **`qgates.hadamard()` is not unitary**: returns the unnormalized `[[1,1],[1,-1]]` (missing `1/√2`), so `H @ H† == 2·I`, not `I`. Every other gate probed is correct. |
| B-7 | `Scalar` P4 | **`complex()` of a *real* Scalar raises**: the `__complex__` lambda is built from `real()`+`imag()`, and `imag()` is undefined for real subtypes — so `complex(Scalar(3.0))` raises instead of returning `(3+0j)`, even though C++'s `complex128(realScalar)` succeeds. |
| B-8 | `Scalar` C2 | **`+=` is not equivalent to `+`**: `real += complex` raises (`DoubleScalar::iadd`'s guard) while the binary `real + complex` promotes to complex — a B5 (and B3-adjacent) divergence. |
| B-9 | `Storage` C5 | **`==` raises on a dtype mismatch** instead of returning `False`: `Storage::operator==` calls `cytnx_error_msg` when dtypes differ, and `__ne__` delegates to it, so `if a == b:` can throw across dtypes (violates Python's total-`==` convention). |
| B-10 | `Network` C2 | **`Network.clone()` silently drops the placed tensors**, yet `isAllset()` then returns a *misleading `True`* (loops an empty vector) while `isLoad()` correctly returns `False`; `clone().Launch()` then fails. Clone must deep-copy the tensors. |
| B-11 | `Scalar` P1 | **`Scalar(2)` is tagged `Uint64`, `Scalar(True)` is `Uint64`**: single-arg overload resolution walks `uint64` before `int64`/`bool`, so a positive Python `int` (and `bool`, an `int` subclass) silently becomes unsigned — later subtraction underflows. The dtype-picking 2-arg constructor is unbound. |
| B-12 | `enums` P7 | **`fermionParity.EVEN` is truthy** though its underlying value is `0`/`false`: the pybind wrapper defines no `__bool__`, so `if parity:` fires for **both** parities. Compare explicitly. |
| B-13 | `Tensor` P5 | **Leftover `std::cout` debug leak** in the bare-1-D-slice `__getitem__` branch (`tensor_py.cpp:355`): `t[0:2]` prints `start stop step` to the real, uncapturable stdout. |

Lower-severity / defensible-but-flagged internal inconsistencies of the same
family: `Symmetry` C6 (`FermionParity.reverse_rule_` can emit a qnum its own
`check_qnum` rejects), `Storage` C4 (`resize` zero-fill not guaranteed on reuse
of shrunk capacity), `Storage` C7 (`Storage().size()` raises on the
default-constructed half-object), `Storage` P7 (off-by-one `__getitem__` bounds
guard, masked by an inner check), `linalg` C3 (decomposition return order is
internally inconsistent — `S`/eigvals first for `Svd`/`Eigh`, `Q` first for
`Qr`, `D` in the *middle* for `Qdr`).

### X7 — Header-vs-wheel member gaps (unbound C++ symbols)

C++ symbols declared in the headers but with no Python binding at all — real
source-vs-wheel surface gaps. Collected across units:

| Unit / finding | Unbound C++ symbol(s) → recommended action |
|---|---|
| `linalg` P1 | `Lanczos_ER`, `Lanczos_Gnd`, `Lanczos_Gnd_Ut` — pybind + `conti.py` both commented out; **add** as `lanczos_er`/`lanczos_gnd`/`lanczos_gnd_ut` (largest gap in the module). |
| `Tensor` P1 | the entire **named** arithmetic family `Add`/`Sub`/`Mul`/`Div` (+`_`), `Cpr`, `Mod` — only operator dunders exist; bind them for B5 symmetry. |
| `Bond` P7/P8 | `getUniqueQnums` → `get_unique_qnums`; `operator*`/`operator*=` → `__mul__`/`__imul__` (B5). |
| `Symmetry` P2/P3 | `combine_rule`'s batch/vector overload (add under same name); `combine_rule_`/`reverse_rule_` out-param forms (informational — pure forms cover them). |
| `network` P2 | `Network::getOptimalOrder` (`.def` commented out) → bind as `get_optimal_order`. |
| `enums` P1 | `Type_class`'s static utilities (`is_complex`/`is_float`/`is_int`/`is_unsigned`/`typeSize`/`getname`) — unreachable from Python; only route is `Tensor.dtype_str()`. |
| `enums` P3/P6 | `SymmetryType::Void` (`-99`), `bondType::BD_NONE` (`0`) — intentionally kept unbound (internal sentinels). |
| `Scalar` P1/P3 | 2-arg `Scalar(value, dtype)` constructor (add, fixes B-11); named ops `radd`/`less`/`eq`/… (keep C++-only — operators cover them). |
| `operations` P1/P2/P4 | `algo.Vsplit_`/`Hsplit_`; `physics::operators` `Sz_shalf`/`Sp_shalf`/`Sn_shalf`; `random_tensor` (bind — useful) and deprecated `Make_normal`/`Make_uniform` (leave unbound). |

---

## 2. Master recommendation index

One row per actionable recommendation, aggregated across all units. Every
`rename`, `remove`, and `add` in every per-class doc is listed; `keep` rows are
a representative sample (the per-class `## Recommendation` tables carry the
exhaustive keep lists). The `Verdict` column carries the literal
**keep / add / rename / remove** tag; the action column gives the recommended
name/behavior and cross-cutting finding (X1–X7) or per-unit id.

### `Bond`

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Bond | `Init` | rename | `init` (X1/C4) |
| Bond | `Load` | rename | `load` (X1/C4) |
| Bond | `Nsym` | rename | `nsym` (X1/C4) |
| Bond | `Save` | rename | `save` (X1/C4) |
| Bond | `calc_reverse_qnums` | rename | `reverse_qnums` (drop redundant `calc_` prefix) |
| Bond | `clear_type` | rename | `clear_type_` (X3/C2: in-place needs `_`) |
| Bond | `combineBond` | rename | `combine_bond` (X1/C4) |
| Bond | `combineBond_` | rename | `combine_bond_` (X1/C4; return `self`, C7) |
| Bond | `getDegeneracies` | rename | `get_degeneracies` (X1/C4) |
| Bond | `getDegeneracy` | rename | `get_degeneracy` (X1/C4) **+ ship the B-2 bugfix** |
| Bond | `set_type` | rename | `retype_` (X3/C1: pair with pure `retype`) |
| Bond | `type` | rename | `bond_type` (C6: match ctor param; avoid shadowing builtin) |
| Bond | `combineBonds` | remove | deprecated duplicate of `combine_bond(list)` (P1) |
| Bond | `combineBonds_` | remove | deprecated duplicate of `combine_bond_(list)` (P1) |
| Bond | `group_duplicates_` (current binding) | remove | bound to wrong C++ overload (B-4/P4/C3) |
| Bond | `c_group_duplicates_refarg` | remove | leaked internal (X2/C5) |
| Bond | `c_redirect_` | remove | leaked internal (X2/C5) |
| Bond | `group_duplicates_` (rebound) | add | new binding onto the true in-place C++ mutator (B-4/P4) |
| Bond | `clear_type` (pure) | add | pure counterpart to `clear_type_` (X3/C2) |
| Bond | `getUniqueQnums` | add | `get_unique_qnums` — bind C++-only symbol (X7/P7) |
| Bond | `operator*` / `operator*=` | add | `__mul__` / `__imul__` (X7/P8/B5) |
| Bond | `c_getDegeneracy_refarg` | keep | interim — sole working degeneracy lookup until B-2 fixed (P6) |
| Bond | `clone`, `dim`, `qnums`, `syms`, `redirect`, `redirect_`, `retype`, `has_duplicate_qnums`, `group_duplicates` | keep | correctly named/behaving; document copy-not-view for `qnums`/`syms` (P2) |

### `Symmetry`

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Symmetry | `FermionNumber` | rename | `fermion_number` (X1/C1) |
| Symmetry | `FermionParity` | rename | `fermion_parity` (X1/C1) |
| Symmetry | `Load` | rename | `load` (X1/C1) |
| Symmetry | `Save` | rename | `save` (X1/C1) |
| Symmetry | `U1` | rename | `u1` (X1/C1) |
| Symmetry | `Zn` | rename | `zn` (X1/C1) |
| Symmetry | `check_qnum` | rename | `is_valid_qnum` (C2/N5) |
| Symmetry | `check_qnums` | rename | `is_valid_qnums` (C2/N5) **+ fix B-3** |
| Symmetry | `__repr__` / `__str__` | rename | fix to *return* the info string (X4/P5) |
| Symmetry | `print_info` | add | bind the unbound C++ `print_info` (X4/P5) |
| Symmetry | `combine_rule` (batch overload) | add | bind the unbound vector overload (X7/P2) |
| Symmetry | `clone`, `combine_rule`, `reverse_rule`, `get_fermion_parity`, `is_fermionic`, `n`, `stype`, `stype_str` | keep | correctly named; rename `combine_rule`/`reverse_rule` *params* (`qnL`/`qnR`/`qin`→`qnum_l`/`qnum_r`/`qnum`, C3/N4) |

### Enums (`Type` / `Device` / `SymType` / `bondType` / `fermionParity`)

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Device | `getname` | rename | `get_name` (X1/C6) |
| Device | `Print_Property` | rename | `print_property` (X1/C6) **+ add stdout-capture guard** (X4/P4) |
| SymType | `SymType` (type name) | rename | `SymmetryType` (X5/P5/N3) |
| bondType | `BD_IN` | remove | redundant alias of `BD_KET` (C4) |
| bondType | `BD_OUT` | remove | redundant alias of `BD_BRA` (C4) |
| all 4 enums | `__eq__`/`__hash__` | keep | **fix**: make type-distinct (X5/C1) |
| all 4 enums | `export_values()` | remove | drop top-level re-export; qualified access only (X5/C2/P8) |
| Device | `Device` (submodule) | keep | **promote to a real enum** (X5/C3) |
| fermionParity | `EVEN`/`ODD` | keep | **fix B-12**: `__bool__` matching the C++ value |
| Type | 12 dtype codes, `name`, `value` | keep | N1-exempt constants; codes verified. Bind missing `Type_class` predicates (X7/P1) |

### `UniTensor`

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| UniTensor | `Conj`/`Conj_` | rename | `conj`/`conj_` (X1/C2) |
| UniTensor | `Dagger`/`Dagger_` | rename | `dagger`/`dagger_` (X1/C2) |
| UniTensor | `Transpose`/`Transpose_` | rename | `transpose`/`transpose_` (X1/C2) |
| UniTensor | `Inv` | rename | `inv` (X1/C2) |
| UniTensor | `Pow`/`Pow_` | rename | `pow`/`pow_` (X1/C2) |
| UniTensor | `Trace`/`Trace_` | rename | `trace`/`trace_` (X1/C2) |
| UniTensor | `Norm` | rename | `norm` (X1/C2) |
| UniTensor | `Init` | rename | `init` (X1/C2) |
| UniTensor | `Load`/`Save` | rename | `load`/`save` (X1/C2) |
| UniTensor | `Nblocks` | rename | `nblocks` (X1/C2) |
| UniTensor | `getTotalQnums` | rename | `get_total_qnums` (X1/C2) |
| UniTensor | `combineBonds` | rename | `combine_bonds_` (X1/C2 + X3/C3: in-place) |
| UniTensor | `elem_exists` | rename | `has_elem` (C4/N5) |
| UniTensor | `same_data` | rename | `shares_data` (C4/N5) |
| UniTensor | `cInv_` | rename | `inv_` — promote the only in-place inverse (X2/C1/P3) |
| UniTensor | `cConj_`, `cDagger_`, `cTranspose_`, `cPow_`, `cTrace_`, `cnormalize_`, `ctag`, `ctruncate_`, `cfrom`, `c__ipow__`, `c_at`, `c_relabel_`, `c_relabels_`, `c_set_label`, `c_set_labels`, `c_set_name`, `c_set_rowrank_` | remove | leaked raw `c`-bindings (X2/C1) |
| UniTensor | `relabels`/`relabels_` | remove | redundant subset of `relabel`/`relabel_` (C5/P2) |
| UniTensor | `astype_different_type` | remove | leaked internal delegate (X2/C7) |
| UniTensor | `to_different_device` | remove | leaked internal delegate (X2/C7) |
| UniTensor | `get_block`/`get_block_`, `get_blocks`/`get_blocks_`, `reshape`/`reshape_`, `permute`/`permute_`, `contract`, `contiguous`, `clone`, `zeros`/`ones`/`eye`, `dtype`, `device`, `print_diagram`, … (~90 members) | keep | correctly named/behaving; docstring the copy-vs-view classes (P4/P6/P7) |

### `Tensor`

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Tensor | `Abs`/`Abs_` | rename | `abs`/`abs_` (X1/C1) |
| Tensor | `Conj`/`Conj_` | rename | `conj`/`conj_` (X1/C1) |
| Tensor | `Exp`/`Exp_` | rename | `exp`/`exp_` (X1/C1) |
| Tensor | `Inv`/`Inv_` | rename | `inv`/`inv_` (X1/C1) |
| Tensor | `InvM`/`InvM_` | rename | `inv_m`/`inv_m_` (X1/C1/N3) |
| Tensor | `Pow`/`Pow_` | rename | `pow`/`pow_` (X1/C1) |
| Tensor | `Norm`, `Max`, `Min` | rename | `norm`, `max`, `min` (X1/C1) |
| Tensor | `Trace` | rename | `trace(axis_a=0, axis_b=1)` — X1/C1 **+ restore dropped defaults** (P2) |
| Tensor | `Svd`, `Eigh` | rename | `svd`, `eigh` (X1/C1) |
| Tensor | `Save`/`Load` | rename | `save`/`load` (X1/C1) |
| Tensor | `Tofile`/`Fromfile` | rename | `to_file`/`from_file` (X1/C1) |
| Tensor | `Init` | rename | `init` (X1/C1) + demote (ctor duplicate, C7) |
| Tensor | `flatten_` | keep | **fix to return `self`** (X3/C4), currently returns `None` |
| Tensor | `make_contiguous` | remove | leaked raw `contiguous()` (X2/P3) |
| Tensor | `astype_different_dtype` | remove | leaked raw primitive (X2/P3) |
| Tensor | `to_different_device` | remove | leaked raw primitive (X2/P3) |
| Tensor | `cConj_`, `cExp_`, `cAbs_`, `cInv_`, `cInvM_`, `cPow_`, `c__iadd__`, `c__isub__`, `c__imul__`, `c__itruediv__`, `c__ifloordiv__`, `c__ipow__`, `c__imatmul__` | remove | leaked raw in-place/operator primitives (X2/P3/C2) |
| Tensor | `__imatmul` (typo) | rename | `__imatmul__` — **fix B-5** (`@=` not in-place, P4) |
| Tensor | `__getitem__` slice `std::cout` leak | remove | delete the stray debug print — **fix B-13** (P5) |
| Tensor | named `add`/`sub`/`mul`/`div` (+`_`) | add | bind for B5 symmetry (X7/P1) |
| Tensor | `dtype`, `shape`, `permute`, `reshape`, `contiguous`, `clone`, `numpy`, `storage`, `item`, `same_data`, `fill`, `from_storage`, … | keep | already `snake_case`; document the copy-vs-view table (C3) |

### `Storage`

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Storage | `Init` | rename | `init` (X1/C1) |
| Storage | `Load`/`Save` | rename | `load`/`save` (X1/C1) |
| Storage | `Fromfile`/`Tofile` | rename | `from_file`/`to_file` (X1/C1) |
| Storage | `astype_different_type` | remove | leaked internal (X2/P4/C3) → private |
| Storage | `to_different_device` | remove | leaked internal (X2/P4/C3) → private |
| Storage | `c_pylist_bool` … `c_pylist_uint64` (×11) | remove | leaked per-dtype accessors (X2/P5/C3) → private, reach via `pylist` |
| Storage | `__eq__`/`__ne__` | keep | **fix B-9**: return `False` on dtype mismatch (C5) |
| Storage | `__repr__`/`__str__` | keep | **fix**: return the info string (X4/P6) |
| Storage | `size`/`capacity` | keep | **fix**: return `0` on default-constructed Storage (C7) |
| Storage | `resize` | keep | **fix**: always zero-fill grown region (C4) |
| Storage | `from_pylist` | keep | add explicit `dtype=`, default signed (C6) |
| Storage | `numpy` | keep | document COPY-not-view (P1) — key caveat |
| Storage | `astype`, `to`, `to_`, `clone`, `dtype`, `device`, `pylist`, `print_info`, `append`, `fill`, `set_zeros`, `real`, `imag` | keep | correctly named; document self-return aliasing (P2/P3) |

### `Scalar`

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Scalar | `iabs` | rename | `abs_` (X3/C1) |
| Scalar | `isqrt` | rename | `sqrt_` (X3/C1) |
| Scalar | `__complex__` | rename | **fix B-7**: build via `to_cytnx_complex128()` (P4) |
| Scalar | `imag` | keep | **fix**: return 0 on a real Scalar (C3/P4) — also fixes B-7 |
| Scalar | `dtype` | keep | return a `Type` member + add `dtype_str()` (X5/C4) |
| Scalar | `__radd__`/`__rsub__`/`__rmul__`/`__rtruediv__` | add | reflected ops for Python numbers (P2) |
| Scalar | `Scalar(value, dtype)` 2-arg ctor | add | **fix B-11**: bind + repair single-arg dtype resolution (X7/P1) |
| Scalar | `+=`/`-=`/`*=`/`/=` | keep | **fix B-8**: promote-in-place to match `+`/… (C2/B5) |
| Scalar | named ops `radd`/`less`/`eq`/… | remove | keep C++-only; operators cover them (P3) |
| Scalar | `abs`, `astype`, `conj`, `maxval`, `minval`, `print`, `real`, `sqrt` | keep | correctly named/behaving |

### `linalg` (all 53 callables renamed — X1/C1)

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| linalg | `Svd` | rename | `svd` — returns `[S, U, vT]`, values-first (C3) |
| linalg | `Svd_truncate` | rename | `svd_truncate` (+ adopt `is_U`/`is_vT`, C2) |
| linalg | `Gesvd`/`Gesvd_truncate` | rename | `gesvd`/`gesvd_truncate` |
| linalg | `Rsvd`, `Hosvd`, `Rand_isometry` | rename | `rsvd`, `hosvd`, `rand_isometry` |
| linalg | `Eigh`, `Eig` | rename | `eigh`, `eig` — values-first (C3) |
| linalg | `Qr`, `Qdr` | rename | `qr`, `qdr` — `Q`-first / `D`-in-middle (C3) |
| linalg | `Conj`/`Conj_`, `Inv`/`Inv_`, `Pow`/`Pow_`, `Abs`/`Abs_`, `Exp`/`Exp_`, `Expf`/`Expf_` | rename | `conj`/`conj_`, `inv`/`inv_`, `pow`/`pow_`, `abs`/`abs_`, `exp`/`exp_`, `expf`/`expf_` |
| linalg | `InvM`/`InvM_` | rename | `inv_m`/`inv_m_` (X1 + C4 disambiguation from element-wise `inv`) |
| linalg | `ExpH`, `ExpM` | rename | `exp_h`, `exp_m` (+ add in-place `exp_h_`/`exp_m_`, X3/C5) |
| linalg | `Det`, `Norm`, `Trace`, `Sum`, `Max`, `Min`, `Diag` | rename | `det`, `norm`, `trace`, `sum`, `max`, `min`, `diag` |
| linalg | `Matmul`, `Matmul_dg`, `Dot`, `Vectordot`, `Outer`, `Kron`, `Tensordot`, `Tensordot_dg`, `Directsum` | rename | `matmul`, `matmul_dg`, `dot`, `vectordot`, `outer`, `kron`, `tensordot`, `tensordot_dg`, `direct_sum` |
| linalg | `Axpy`/`Axpy_`, `Gemm`/`Gemm_`, `Ger`, `Lstsq`, `Tridiag` | rename | `axpy`/`axpy_`, `gemm`/`gemm_`, `ger`, `lstsq`, `tridiag` |
| linalg | `Lanczos`, `Lanczos_Exp`, `Arnoldi` | rename | `lanczos`, `lanczos_exp`, `arnoldi` |
| linalg | `Lanczos_ER` | add | `lanczos_er` — bind dead C++ symbol (X7/P1) |
| linalg | `Lanczos_Gnd` | add | `lanczos_gnd` — bind dead C++ symbol (X7/P1) |
| linalg | `Lanczos_Gnd_Ut` | add | `lanczos_gnd_ut` — bind dead C++ symbol (X7/P1) |

*(No `linalg` function is removed; all 53 are load-bearing. Return-tuple order
is left as-is at runtime and resolved by documentation — linalg C3.)*

### Network family (`Network` / `ncon` / `LinOp`)

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| Network | `FromString`/`Fromfile`/`Savefile` | rename | `from_string`/`from_file`/`save_file` (X1/C1/C5) |
| Network | `PutUniTensor`/`PutUniTensors` | rename | `put_unitensor`/`put_unitensors` (X1/C1) |
| Network | `RmUniTensor`/`RmUniTensors` | rename | `remove_unitensor`/`remove_unitensors` (X1/C1 + un-abbreviate, C4) |
| Network | `Launch` | rename | `launch` (X1/C1; drop dead `network_type`, P3) |
| Network | `setOrder`/`getOrder` | rename | `set_order`/`get_order` (X1/C1) |
| Network | `isAllset` | rename | `is_all_set` (X1/C1/N5) **+ fix B-10** (vacuous `True`) |
| Network | `isLoad` | rename | `is_loaded` (X1/C1/N5) |
| Network | `PrintNet` | rename | `print_net` (X1/C1) |
| Network | `Diagram` | rename | `diagram` (X1/C1; raise not `exit`, P4) |
| Network | `Contract` | remove | **B-1**: `Launch()` segfaults; duplicates `ncon` (P1/B4) |
| Network | `clone` | keep | **fix B-10**: deep-copy placed tensors (C2/B1) |
| Network | `construct` | keep | reorder `outrk` before optional args (C6) |
| Network | `getOptimalOrder` | add | `get_optimal_order` — bind unbound C++ symbol (X7/P2) |
| LinOp | `matvec`, `set_elem`, `set_device`, `set_dtype`, `device`, `dtype`, `nx` | keep | already `snake_case`; replace stringly-typed `type` ctor flag with enum/2 classes (C7) |
| ncon | `ncon` | keep | the working one-shot contractor; default `check_network=True` |

### Operation submodules (`algo` / `random` / `physics` / `qgates`)

| Unit | Member | Verdict | Recommended name / action |
|---|---|---|---|
| algo | `Sort` | rename | `sort` (X1/C1) **+ add `sort_`** (X3/C2) |
| algo | `Concatenate` | rename | `concatenate` (X1/C1) |
| algo | `Vstack`/`Hstack` | rename | `vstack`/`hstack` (X1/C1) |
| algo | `Vsplit`/`Hsplit` | rename | `vsplit`/`hsplit` (X1/C1) |
| algo | `sort_` | add | in-place counterpart (X3/C2) |
| random | `random_tensor` | add | bind unbound any-dtype generator (X7/P4) |
| random | `normal`/`normal_`/`uniform`/`uniform_` | keep | already `snake_case`; add `mean=0.0`/`std=1.0` defaults to `normal`/`normal_` (C5) |
| physics | `Sz_shalf`/`Sp_shalf`/`Sn_shalf` | add | bind `operators` submodule if wanted (X7/P2) |
| physics | `spin`/`pauli` | keep | already `snake_case`; rename args `S`/`Comp`→`s`/`component` (C3/N5) |
| qgates | `hadamard` | keep | **fix B-6**: multiply by `1/√2` so `H @ H† == I` (P8) |
| qgates | `pauli_x`/`pauli_y`/`pauli_z`, `phase_shift`, `swap`, `sqrt_swap`, `toffoli`, `cntl_gate_2q` | keep | correct/unitary (probe); drop the `/// @cond` doc guard (C6) |

---

## 3. How to read this back to source

- A cross-cutting finding `X#` names the affected units and their per-unit
  finding ids inline; open `per-class/<Unit>.md` and jump to that id's entry in
  `## Parity findings` / `## Consistency findings` for the probe evidence.
- A confirmed bug `B-#` cites the owning doc's finding (e.g. "network P1"); the
  probe assertion that demonstrates it lives in `docs/api-audit/probes/<Unit>.py`.
- The master index `Verdict` maps to the same-named row in that unit's
  `## Recommendation` and `## Change table`, which carry the full rationale and
  (for `keep`/`add`/`rename`) the numpy-style docstring.
- The post-rename spellings used here are the ones [`essential-api.md`](essential-api.md)
  consumes when tracing each TRG/HOTRG/CTMRG/MERA step to a concrete API call.
- The **argument** dimension of X1/N4 (parameter names, positions, defaults) is
  swept systematically across all units in
  [`parameter-consistency.md`](parameter-consistency.md), generated from the live
  pybind signatures by `tools/param_inventory.py` — it collects the cross-*class*
  divergences (e.g. the three-way `Trace` split, 18 keyword-uncallable methods)
  that no single per-class doc can see whole.
