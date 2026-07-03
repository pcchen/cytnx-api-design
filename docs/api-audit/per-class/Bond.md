# `Bond` — API audit

`Bond` is the auxiliary object attached to every `UniTensor` rank/leg: it
carries the leg's dimension, its `bondType` (regular / bra / ket), and — for
symmetric tensors — its quantum-number sectors, degeneracies, and the
`Symmetry` objects that define them. This document audits the 30 public
members of the live `cytnx.Bond` class (installed `cytnx==1.1.0` wheel) plus
one C++-only member (`getUniqueQnums`) that has no Python binding at all.

Ground truth for behavior is `docs/api-audit/probes/Bond.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/Bond.py`; all 27 assertions `[PASS]`, exit 0). Ground
truth for static signatures is `cytnx_src/include/Bond.hpp` (C++
declarations), `cytnx_src/pybind/bond_py.cpp` (the actual pybind11 binding —
authoritative for the Python-visible call signature and for *which* C++
overload each Python name routes to), and `cytnx_src/cytnx/Bond_conti.py`
(Python-side augmentation). Where the installed wheel's copy of
`Bond_conti.py` diverges from the repo source (it does, for `getDegeneracy`
— see Parity finding P6), the **installed wheel is treated as ground truth**
for behavioral claims, since that is what `tools/validate_doc.py` and every
probe run against.

## Inventory

C++ signatures are read from `Bond.hpp`; Python signatures are the effective
pybind-visible (or `Bond_conti.py`-wrapped) signature, cross-checked against
`tools/member_inventory.py Bond`.

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `Init` | `void Init(dim, bd_type=BD_REG)`; `void Init(bd_type, in_qnums: vec2d<int64>, degs, in_syms={})`; `void Init(bd_type, in_qnums_dims: vector<pair<vec<int64>,uint64>>, in_syms={})` | 4 overloads (`bond_py.cpp:50-78`): `Init(bond_type, qnums: List[List[int]], degs: List[int], symmetries: List[Symmetry]=[])`; `Init(dim: int, bond_type: bondType=BD_REG)`; `Init(bond_type, qnums: List[Tuple[List[int],int]], symmetries: List[Symmetry]=[])`; `Init(bond_type, qnums: List[Qs], degs: List[int], symmetries: List[Symmetry]=[])` — all `-> None`, mutate `self` in place |
| `Load` (static) | `static Bond Load(const string&)`; `static Bond Load(const char*)` | `Load(fname: str) -> Bond` (staticmethod) |
| `Nsym` | `cytnx_uint32 Nsym() const` | `Nsym() -> int` |
| `Save` | `void Save(const string&) const`; `void Save(const char*) const` | `Save(fname: str) -> None` |
| `c_getDegeneracy_refarg` | *(no C++ member; pybind-only plumbing lambda)* | `c_getDegeneracy_refarg(qnum: List[int]\|Qs, indices: list) -> int` — appends into `indices` via `py::list.append`, so it (unlike `group_duplicates_`, see P4) genuinely mutates the passed-in Python list |
| `c_group_duplicates_refarg` | *(no C++ member; pybind-only plumbing lambda)* | `c_group_duplicates_refarg(mapper: list) -> Bond` — appends into `mapper` via `py::list.append` |
| `c_redirect_` | *(binds `Bond::redirect_`)* | `c_redirect_() -> Bond` — internal alias; the public `redirect_` (Python-only, `Bond_conti.py`) wraps this |
| `calc_reverse_qnums` | `vector<vector<int64>> calc_reverse_qnums()` | `calc_reverse_qnums() -> List[List[int]]` |
| `clear_type` | `void clear_type()` | `clear_type() -> None` — mutates in place |
| `clone` | `Bond clone() const` | `clone() -> Bond`; also bound as `__copy__`/`__deepcopy__` |
| `combineBond` | `Bond combineBond(const Bond&, is_grp=true) const`; `Bond combineBond(const vector<Bond>&, is_grp=true)` | `combineBond(bd: Bond, is_grp: bool=True) -> Bond`; `combineBond(bds: List[Bond], is_grp: bool=True) -> Bond` — the list overload routes to the **deprecated** `Bond::combineBonds` (`bond_py.cpp:107-112`), not to the non-deprecated `Bond::combineBond(vector<Bond>&, …)` |
| `combineBond_` | `void combineBond_(const Bond&, is_grp=true)`; `void combineBond_(const vector<Bond>&, is_grp=true)` | `combineBond_(bd: Bond, is_grp: bool=True) -> None`; `combineBond_(bds: List[Bond], is_grp: bool=True) -> None` — list overload routes to deprecated `Bond::combineBonds_` |
| `combineBonds` | `[[deprecated]] Bond combineBonds(const vector<Bond>&, is_grp=true)` | `combineBonds(bds: List[Bond], is_grp: bool=True) -> Bond` |
| `combineBonds_` | `[[deprecated]] void combineBonds_(const vector<Bond>&, is_grp=true)` | `combineBonds_(bds: List[Bond], is_grp: bool=True) -> None` |
| `dim` | `cytnx_uint64 dim() const` | `dim() -> int` |
| `getDegeneracies` | `vector<uint64>& getDegeneracies()`; `const vector<uint64>& getDegeneracies() const` (both **reference**-returning) | `getDegeneracies() -> List[int]` — always a fresh Python list copy (see P2) |
| `getDegeneracy` | `cytnx_uint64 getDegeneracy(qnum) const`; `cytnx_uint64 getDegeneracy(qnum, indices&) const` | Installed wheel: `getDegeneracy(qnum, return_indices) -> ⚠ always raises` (see P6). Repo source has an unshipped fix: `getDegeneracy(qnum: List[int]\|Qs, return_indices: bool=False) -> int \| Tuple[int, List[int]]` |
| `get_fermion_parity` | `fermionParity get_fermion_parity(const vector<int64>& qnum)` | `get_fermion_parity(qnum: List[int]) -> fermionParity` |
| `group_duplicates` | `Bond group_duplicates(vector<uint64>& mapper) const` *(out-param, copy-returning)* | `group_duplicates() -> Tuple[Bond, List[int]]` (`Bond_conti.py`) — pure/copy, correctly implemented |
| `group_duplicates_` | `vector<uint64> group_duplicates_()` *(true in-place mutator, mutates `self`, returns just the mapper)* | `group_duplicates_(mapper: List[int]) -> Bond` — **bound to `&Bond::group_duplicates` (the const, copy-returning overload), not to `Bond::group_duplicates_()`** (see P4); does not mutate `self` |
| `has_duplicate_qnums` | `bool has_duplicate_qnums() const` | `has_duplicate_qnums() -> bool` |
| `qnums` | `const vector<vector<int64>>& qnums() const`; `vector<vector<int64>>& qnums()` (both **reference**-returning) | `qnums() -> List[List[int]]` — always a fresh Python list copy (see P2) |
| `qnums_clone` | `vector<vector<int64>> qnums_clone() const` | `qnums_clone() -> List[List[int]]` — redundant with `qnums()` in Python (both copy; see P3) |
| `redirect` | `Bond redirect() const` | `redirect() -> Bond` — pure |
| `redirect_` | `Bond& redirect_()` *(in-place, returns `*this`)* | `redirect_() -> Bond` (`Bond_conti.py`, wraps native `c_redirect_`) — in-place, returns `self` |
| `retype` | `Bond retype(const bondType&)` | `retype(bond_type: bondType) -> Bond` — pure |
| `set_type` | `Bond& set_type(const bondType&)` *(in-place, returns `*this`)* | `set_type(bond_type: bondType) -> Bond` — in-place, returns `self` (see P5) |
| `syms` | `const vector<Symmetry>& syms() const`; `vector<Symmetry>& syms()` (both **reference**-returning) | `syms() -> List[Symmetry]` — always a fresh Python list copy (see P2) |
| `syms_clone` | `vector<Symmetry> syms_clone() const` | `syms_clone() -> List[Symmetry]` — redundant with `syms()` in Python (see P3) |
| `type` | `bondType type() const` | `type() -> bondType` |
| *(C++-only, not bound)* `getUniqueQnums` | `vector<vector<int64>> getUniqueQnums(vector<uint64>& counts, bool return_counts)`; convenience overloads | **absent from Python** — not in `bond_py.cpp` at all (see P7) |
| *(C++-only, not bound)* `operator*`/`operator*=` | `Bond operator*(const Bond&) const` (== `combineBond`); `Bond& operator*=(const Bond&)` (== `combineBond_`) | **`__mul__`/`__imul__` not bound** in `bond_py.cpp` (see P8) |

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/Bond.py`.

- **P1 — `combineBond`/`combineBond_`'s list overload silently routes through
  the C++-`[[deprecated]]` `combineBonds`/`combineBonds_`**, not the
  non-deprecated `Bond::combineBond(vector<Bond>&, …)`/`combineBond_(vector<Bond>&, …)`.
  Functionally the results are identical (probe: *"combineBond(list) and the
  deprecated combineBonds(list) give identical results"*), and no
  `DeprecationWarning` is surfaced to Python (C++ `[[deprecated]]` is a
  compile-time-only annotation), so this is currently harmless but means the
  Python binding is wired to code C++ itself has marked for removal — a
  latent parity risk if `combineBonds`/`combineBonds_` are ever deleted from
  the C++ side without updating `bond_py.cpp`.
- **P2 — `qnums()`, `getDegeneracies()`, and `syms()` return references in
  C++ but always return copies in Python.** C++'s non-const overloads of all
  three return a mutable reference into the `Bond`'s internal state (a
  view); pybind converts the returned STL container by value (via
  `pybind11/stl.h`), so the Python object handed back is always an
  independent list. Probed directly with `returns_view()`: *"Bond.qnums()
  returns a copy in Python, not the C++-side mutable-reference view"* and
  *"Bond.getDegeneracies() returns a copy in Python (same B2 pattern as
  qnums())"* — both `[PASS]` with `is_view is False`. This is a genuine B2
  divergence: a C++ caller holding `bond.qnums()` can mutate the bond's
  quantum numbers through that reference; a Python caller cannot.
- **P3 — `qnums_clone()`/`syms_clone()` are redundant with `qnums()`/`syms()`
  in Python**, a direct consequence of P2: since the Python bindings of
  `qnums()`/`syms()` are already always-copy, there is no remaining
  behavioral difference from `qnums_clone()`/`syms_clone()` on the Python
  side (both return an independent list). The distinction is only
  meaningful in C++, where `qnums()` can alias and `qnums_clone()` cannot.
- **P4 — `group_duplicates_` is bound to the wrong C++ overload; the name's
  trailing `_` is a lie.** C++ `Bond` has a true in-place mutator
  `vector<uint64_t> group_duplicates_()` (mutates `self`, returns just the
  mapper) and a separate `const`, copy-returning `Bond
  group_duplicates(vector<uint64_t>& mapper) const`. `bond_py.cpp:165` binds
  Python's `group_duplicates_` to the **`const` copy-returning** overload
  (`.def("group_duplicates_", &Bond::group_duplicates)`), not to the true
  in-place `group_duplicates_()`. Probed: *"group_duplicates_() does NOT
  mutate the receiver, despite the trailing '_'"* and *"group_duplicates_()
  returns a new (grouped) Bond, distinct from the receiver"* — both
  `[PASS]`. Additionally, the `mapper` out-argument taken by this binding is
  a plain `std::vector<uint64_t>&`, which pybind11/stl.h converts *by value*
  from the passed Python list rather than aliasing it, so any list the
  caller passes in is **silently never filled** — probed: *"group_duplicates_()'s
  `mapper` out-argument is silently left unfilled"* `[PASS]`. The
  correctly-behaving pure/copy form of this operation is reachable only
  under the *non*-underscored `group_duplicates()` (a `Bond_conti.py`
  wrapper around `c_group_duplicates_refarg`, which *does* use `py::list&`
  and therefore does fill its `mapper` argument) — probed and `[PASS]`. Net
  effect: **the true C++ in-place `group_duplicates_()` is entirely
  unreachable from Python**, and the reachable `group_duplicates_` name
  means the opposite of what N2 says it should.
- **P5 — `set_type` mutates in place and returns `self`, despite having no
  trailing `_`.** `Bond::set_type` is declared `Bond& set_type(...)`, mutates
  `this->_impl` directly, and returns `*this` by reference; pybind's default
  policy for this C++ reference return resolves to the same registered
  Python wrapper object (verified: `set_type()`'s return value `is` the
  receiver). Probed: *"set_type() mutates the receiver in place even though
  its name has no trailing '_'"* and *"set_type() returns the identical
  (self) wrapper object, like redirect_()"* — both `[PASS]`. This is a
  genuine B1/N2-relevant behavior (the *pure* counterpart, confirmed not to
  mutate, is a *different-named* method, `retype`) — see Consistency finding
  C1.
- **P6 — `getDegeneracy` is completely broken in the installed 1.1.0
  wheel; this is a shipped bug, not a design choice.** The repo source
  `cytnx_src/cytnx/Bond_conti.py` defines a correct, already-fixed
  `getDegeneracy(self, qnum, return_indices: bool = False)` (with a comment
  explaining it captures the native pybind overload before shadowing it).
  The **installed** `.venv` wheel's `site-packages/cytnx/Bond_conti.py`
  instead stacks two `@add_method`-decorated `getDegeneracy` definitions;
  since `add_method` monkey-patches `Bond.getDegeneracy` via `setattr` each
  time, the *second* definition wins — and it (a) requires `return_indices`
  positionally with no default, and (b) references an undefined name
  `lqnum` in its body (a typo for `qnum`). Probed directly against the
  installed wheel: *"getDegeneracy(qnum) with no return_indices raises
  TypeError"* and *"getDegeneracy(qnum, return_indices) ALSO raises
  (NameError: 'lqnum' is not defined)"* — both `[PASS]` (i.e., raising is
  the *expected*, currently-true behavior). The only presently-working path
  to a degeneracy lookup is the internal `c_getDegeneracy_refarg` — probed
  *"c_getDegeneracy_refarg(qnum, indices) is the only currently-working
  degeneracy lookup"* `[PASS]`. **This is the single highest-severity
  finding in this document**: a documented, tested-elsewhere (see
  `cytnx_src/pytests/bond_test.py`, which exercises the *fixed* source and
  would fail against the installed wheel) public method is unusable as
  shipped.
- **P7 — `Bond::getUniqueQnums` has no Python binding at all.** Both C++
  overloads (`getUniqueQnums(counts&, return_counts)` and the convenience
  no-out-param form) are declared in `Bond.hpp` but neither is present in
  `bond_py.cpp`. There is no way to reach this functionality from Python;
  callers must reimplement "sorted unique qnums with counts" from `qnums()`
  themselves. Not in the live `dir(cytnx.Bond)` member set (so not required
  by `validate_doc.py`'s coverage check), but recorded here as a genuine
  parity gap and carried into the Recommendation table as an `add`.
- **P8 — `operator*`/`operator*=` are declared in C++ but never bound to
  `__mul__`/`__imul__` in Python.** `Bond.hpp` defines `operator*` (== the
  named `combineBond`) and `operator*=` (== the named `combineBond_`)
  specifically so `bd1 * bd2` / `bd1 *= bd2` work in C++; `bond_py.cpp`
  binds neither. Probed: `b1 * b2` and `b1 *= b2` both raise `TypeError:
  unsupported operand type(s)` — *"Bond has no __mul__ in Python..."* /
  *"...no __imul__ in Python..."*, both `[PASS]`. This is a direct B5
  violation (the operator form should be equivalent to the named-method
  form; here it doesn't exist at all). By contrast, `operator!=` is not
  explicitly bound either, but works correctly via Python's default
  `__ne__`-delegates-to-`__eq__` behavior — probed and `[PASS]` — so that
  one is a non-issue, only noted for completeness.

## Consistency findings

- **C1 — violates N2: the mutate/pure pair for "change bond type" uses two
  unrelated names (`set_type` / `retype`) instead of a `_`-suffixed pair.**
  `retype()` is the pure form (confirmed non-mutating, P5); `set_type()` is
  the in-place form (confirmed mutating, P5) but carries no trailing `_` and
  shares no base name with `retype`. The recommended fix (see Recommendation
  table) is to fold `set_type` into `retype_`, giving the conventional
  `retype()`/`retype_()` pair.
- **C2 — violates N2: `clear_type` is an in-place-only mutator with no
  trailing `_` and no pure counterpart**, even though "return a new Bond
  with type cleared" is just as meaningful an operation as "clear this
  Bond's type in place" (compare `redirect`/`redirect_`, which correctly
  offer both). Recommend renaming to `clear_type_` and adding a pure
  `clear_type()`.
- **C3 — violates N2/B1: `group_duplicates_`'s trailing `_` promises
  in-place mutation it does not deliver** (P4). This is not merely a naming
  nit — it is empirically false advertising, verified by probe.
- **C4 — violates N1: capitalized/camelCase callable members.** `Init`,
  `Load`, `Save`, `Nsym`, `combineBond`, `combineBond_`, `combineBonds`,
  `combineBonds_`, `getDegeneracies`, `getDegeneracy` all use capitalized or
  camelCase forms instead of `snake_case`. (`Init` is a constructor-style
  re-initializer, not a mutate/pure operation pair, so N2 does not apply to
  it — only the N1 casing rule does.)
- **C5 — violates N2 (idiosyncratic non-conforming variant): `c_redirect_`,
  `c_getDegeneracy_refarg`, and `c_group_duplicates_refarg` are internal
  pybind-plumbing methods leaked onto the public surface**, analogous to the
  `cConj_`/`cDagger_`-style variants flagged generally in
  `00-methodology.md` N2. None of the three is meant to be called directly
  by users — each backs a `Bond_conti.py`-level wrapper (`redirect_`,
  `getDegeneracy`, `group_duplicates` respectively). Recommend removing all
  three from the public surface once the wrapped operation is fixed/kept
  under its proper name (see Recommendation table; `c_getDegeneracy_refarg`
  is kept for now only because it is presently the *sole working* path to a
  degeneracy lookup, P6).
- **C6 — violates N4/N5 adjacent: the accessor `type()` doesn't match the
  constructor parameter name it reads back (`bond_type`).** The constructor
  and `Init` both take a `bond_type` keyword argument (`bond_py.cpp:33-78`),
  but the getter for that same value is named `type()` — a different word,
  and one that shadows the Python builtin `type` when called as a free
  identifier. Recommend renaming the getter to `bond_type()` to match.
- **C7 — minor: `combineBond_`'s in-place return value (`None`) is
  inconsistent with `redirect_`'s and `set_type`'s in-place return value
  (`self`).** `00-methodology.md` B1 explicitly permits either ("callers
  must not rely on identity"), so this is not a hard violation, but it is
  worth normalizing for ergonomics (chaining `b.redirect_().combineBond_(x)`
  works today only because `redirect_` returns `self`); recommend
  `combine_bond_` also return `self`.

## Recommendation

Every one of the 30 live public members of `cytnx.Bond` appears below, tagged
**keep / add / rename / remove**. Two additional rows cover the C++-only
`getUniqueQnums` (P7, `add`) and the unbound `__mul__`/`__imul__` operators
(P8, `add` — informational; dunder members are outside
`validate_doc.py`'s coverage check but are part of the recommended surface).

| Member | Verdict | Rationale |
|---|---|---|
| `Init` | rename | → `init` (C4/N1). Constructor-style re-initializer; N2 doesn't apply (see C4 note). |
| `Load` | rename | → `load` (C4/N1). Static factory. |
| `Nsym` | rename | → `nsym` (C4/N1). |
| `Save` | rename | → `save` (C4/N1). |
| `c_getDegeneracy_refarg` | keep | Internal plumbing (C5), but kept *for now* as the only working degeneracy lookup until `get_degeneracy` (P6) is shipped fixed; remove once `get_degeneracy` is fixed and re-verified. |
| `c_group_duplicates_refarg` | remove | Internal plumbing (C5); fully superseded by `group_duplicates()`. |
| `c_redirect_` | remove | Internal plumbing (C5); fully superseded by `redirect_()`. |
| `calc_reverse_qnums` | rename | → `reverse_qnums` (already `snake_case`, drop the redundant `calc_` verb prefix for consistency with `redirect`/`retype`/etc., which don't prefix with a generic verb). |
| `clear_type` | rename | → `clear_type_` (C2/N2): in-place-only, needs the `_` suffix. |
| *(new)* `clear_type` | add | New pure counterpart to `clear_type_` (C2/N2): returns a cloned `Bond` with type cleared, mirroring the `redirect`/`redirect_` pattern. |
| `clone` | keep | Correctly named, correctly deep-copying (B1); not a mutate/pure pair candidate. |
| `combineBond` | rename | → `combine_bond` (C4/N1). Pure form; keep both single-`Bond` and list-of-`Bond` overloads. |
| `combineBond_` | rename | → `combine_bond_` (C4/N1). In-place form; per C7, should return `self` like `redirect_`/`retype_`. |
| `combineBonds` | remove | Deprecated in C++ (P1) and functionally identical to `combine_bond`'s list overload; two names for one operation. |
| `combineBonds_` | remove | Deprecated in C++ (P1) and functionally identical to `combine_bond_`'s list overload. |
| `dim` | keep | Already minimal, correct, `snake_case`. |
| `getDegeneracies` | rename | → `get_degeneracies` (C4/N1). Document explicitly as copy, not view (P2). |
| `getDegeneracy` | rename | → `get_degeneracy` (C4/N1). **Must also be shipped with the fix already present in repo source** (P6): `return_indices: bool = False` default, correct variable use. |
| `get_fermion_parity` | keep | Already `snake_case`. |
| `group_duplicates` | keep | Already `snake_case`, correctly pure/copy (no trailing `_`), correctly returns `(Bond, mapper)`. |
| `group_duplicates_` (current binding) | remove | Broken (P4/C3): bound to the `const` copy-returning overload under a name that promises in-place mutation, and its `mapper` out-arg is silently discarded. |
| `group_duplicates_` (redesigned) | add | New binding directly onto `Bond_impl::group_duplicates_()` — the true in-place C++ mutator, currently unreachable from Python (P4). Mutates `self`, returns the mapper list, giving `group_duplicates`/`group_duplicates_` a real N2 pure/in-place pair. |
| `has_duplicate_qnums` | keep | Already `snake_case`, predicate correctly `has_`-prefixed (N5). |
| `qnums` | keep | Document explicitly as returning a copy in Python, not the C++-side reference/view (P2). |
| `qnums_clone` | keep | Redundant with `qnums()` in Python today (P3), but kept for explicit-intent parity with C++ and as a stable name if `qnums()` is ever changed to a view (e.g. via a buffer/numpy-backed object). |
| `redirect` | keep | Correctly named, correctly pure (B1). |
| `redirect_` | keep | Correctly named, correctly in-place, correctly returns `self` — the model N2 pair partner for `redirect`. |
| `retype` | keep | Correctly named, correctly pure. Pairs with `retype_` (see `set_type`). |
| `set_type` | rename | → `retype_` (C1/N2/P5): folds the in-place form into the same base name as its pure counterpart `retype`, matching the `redirect`/`redirect_` model. |
| `syms` | keep | Document explicitly as returning a copy in Python, not the C++-side reference/view (P2). |
| `syms_clone` | keep | Redundant with `syms()` in Python today (P3), kept for the same forward-compatibility reason as `qnums_clone`. |
| `type` | rename | → `bond_type` (C6): matches the constructor/`init` parameter name `bond_type`; avoids shadowing the Python builtin `type`. |
| *(C++-only)* `getUniqueQnums` | add | Bind to Python (P7): `get_unique_qnums(return_counts: bool = False) -> List[List[int]] \| Tuple[List[List[int]], List[int]]`, mirroring the `get_degeneracy`/`return_indices` pattern for a single optional-tuple-return convention. |
| *(C++-only)* `__mul__`/`__imul__` | add | Bind `operator*`/`operator*=` to `__mul__`/`__imul__` (P8/B5), delegating to `combine_bond`/`combine_bond_` so `bd1 * bd2` works in Python as it does in C++. |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`
above, under its recommended name.

### `init`

```
Re-initialize this Bond in place.

Parameters
----------
dim : int, optional
    The bond dimension, for a regular (non-symmetric) bond.
bond_type : bondType, optional
    BD_REG (default), BD_BRA, or BD_KET.
qnums : list of list of int, or list of Qs, or list of (list of int, int), optional
    The quantum-number sector(s), for a symmetric bond. Shape (# sectors, # symmetries),
    or paired with an explicit degeneracy per sector.
degs : list of int, optional
    The degeneracy of each quantum-number sector in `qnums`. Required unless
    `qnums` already carries per-sector degeneracies (the `(qnums, deg)` pair form).
symmetries : list of Symmetry, optional
    The Symmetry object for each column of `qnums`. Defaults to `[Symmetry.U1()]`
    per column if `qnums` is given and `symmetries` is omitted.

Returns
-------
None

Notes
-----
In-place: discards and replaces this Bond's dim/type/qnums/degs/symmetries.
Equivalent to constructing a fresh `Bond(...)` with the same arguments; kept
as a separate method (not `_`-suffixed) because it plays the role of a
constructor overload, not a mutate/pure pair (see Consistency finding C4).
Renamed from `Init` (N1 casing, C4).
```

### `load` (static)

```
Load a Bond previously written by `save`.

Parameters
----------
fname : str
    File path (the `.cybd` extension is appended automatically by `save`
    and expected here).

Returns
-------
Bond
    A new Bond reconstructed from the file.

Notes
-----
Renamed from `Load` (N1 casing, C4).
```

### `nsym`

```
Return the number of symmetries attached to this bond.

Returns
-------
int
    0 for a non-symmetric (BD_REG) bond.

Notes
-----
Renamed from `Nsym` (N1 casing, C4).
```

### `save`

```
Save this Bond to a file.

Parameters
----------
fname : str
    File path; the `.cybd` extension is appended automatically.

Returns
-------
None

Notes
-----
Renamed from `Save` (N1 casing, C4).
```

### `c_getDegeneracy_refarg`

```
Internal: look up the degeneracy of a quantum-number sector and append its
matching indices into a caller-supplied list.

Parameters
----------
qnum : list of int, or Qs
    The quantum-number sector to look up.
indices : list
    Mutated in place: matching block indices are appended.

Returns
-------
int
    The degeneracy of `qnum` (0 if the bond has no symmetries or `qnum`
    does not occur).

Notes
-----
Kept only until `get_degeneracy` (P6) ships fixed; not intended for
direct use. Unlike `group_duplicates_`'s broken out-argument (P4), this
method's `indices` argument genuinely is mutated in place, because it is
typed `py::list&` rather than `std::vector<uint64_t>&`.
```

### `reverse_qnums`

```
Compute the reversed quantum numbers implied by this bond's Symmetry
reverse rule (for resolving a bra/ket mismatch).

Returns
-------
list of list of int
    A fresh (dim, # of symmetries) list; does not mutate this bond.

Notes
-----
Renamed from `calc_reverse_qnums` (drops the redundant `calc_` verb prefix).
```

### `clear_type_`

```
Reset this bond's type to BD_REG, in place.

Returns
-------
None

Notes
-----
In-place: mutates `self`. Raises if this bond carries quantum numbers
(only a symmetric bond with zero qnums can be cleared). Pairs with the
pure `clear_type()`.
```

### `clear_type`

```
Return a copy of this bond with its type reset to BD_REG.

Returns
-------
Bond
    A new, independent Bond (deep copy); `self` is left unchanged.

Notes
-----
Pure/copy: equivalent to `self.clone().clear_type_()`. Pairs with the
in-place `clear_type_()`.
```

### `clone`

```
Return an independent deep copy of this Bond.

Returns
-------
Bond
    A new Bond with its own copies of dim/type/qnums/degs/symmetries;
    mutating the result never affects `self` (confirmed by probe: clone()
    produces an independent Bond).

Notes
-----
Also reachable via `copy.copy()`/`copy.deepcopy()` (both are bound to this
same method, so Python's shallow-copy protocol still performs a full deep
copy for Bond).
```

### `combine_bond`

```
Combine this bond with another (or several others), returning a new Bond.

Parameters
----------
bd : Bond
    The other bond to combine with self.
bds : list of Bond
    Alternative to `bd`: combine self with each bond in the list, in order.
is_grp : bool, optional
    For symmetric bonds only: if True (default), sectors that end up with
    duplicate quantum numbers after combining are grouped into one.

Returns
-------
Bond
    A new, independent Bond; `self` is left unchanged (confirmed by probe).

Notes
-----
Pure/copy. Requires the bond(s) being combined to share `self`'s bondType
and Symmetry set. Pairs with the in-place `combine_bond_`.
```

### `combine_bond_`

```
Combine this bond with another (or several others), in place.

Parameters
----------
bd : Bond
    The other bond to combine with self.
bds : list of Bond
    Alternative to `bd`: combine self with each bond in the list, in order.
is_grp : bool, optional
    Same meaning as in `combine_bond`.

Returns
-------
Bond
    `self`, after mutation (recommended fix, C7 — the current
    `combineBond_` returns `None`; confirmed by probe).

Notes
-----
In-place: mutates `self`. Pairs with the pure `combine_bond`.
```

### `dim`

```
Return the total dimension of this bond.

Returns
-------
int
    For a symmetric bond, the sum of degeneracies across all sectors.
```

### `get_degeneracies`

```
Return the degeneracy of every quantum-number sector, in sector order.

Returns
-------
list of int
    A fresh Python list (confirmed by probe: getDegeneracies() returns a
    copy, not the C++-side mutable-reference view — see Parity finding P2).
    Mutating the returned list never affects this bond.

Notes
-----
Renamed from `getDegeneracies` (N1 casing, C4).
```

### `get_degeneracy`

```
Return the degeneracy of a specific quantum-number sector.

Parameters
----------
qnum : list of int, or Qs
    The quantum-number sector to look up.
return_indices : bool, optional
    If True, also return the list of matching block indices as a
    `(degeneracy, indices)` tuple. Default False (bare int).

Returns
-------
int, or (int, list of int)
    0 (or `(0, [])`) if the bond has no symmetries or `qnum` does not occur.

Notes
-----
**Ships broken in the installed cytnx 1.1.0 wheel** (Parity finding P6):
calling this method under its current name always raises (`TypeError` with
no `return_indices` argument; `NameError: name 'lqnum' is not defined`
otherwise). This docstring describes the already-designed, not-yet-shipped
fix in `cytnx_src/cytnx/Bond_conti.py`. Until the fix ships, use
`c_getDegeneracy_refarg` directly. Renamed from `getDegeneracy` (N1 casing, C4).
```

### `get_fermion_parity`

```
Return the fermionic parity of a given quantum-number sector.

Parameters
----------
qnum : list of int
    The quantum-number sector to classify.

Returns
-------
fermionParity
    EVEN for a bosonic degree of freedom, ODD for a fermionic one.
```

### `group_duplicates`

```
Group sectors that share the same quantum numbers, returning a new Bond.

Returns
-------
(Bond, list of int)
    A new, independent Bond with duplicate-qnum sectors merged and sorted
    ascending, plus a mapper such that `mapper[old_index] == new_index`.
    `self` is left unchanged (confirmed by probe).

Notes
-----
Pure/copy. Pairs with the in-place `group_duplicates_` (recommended
redesign, see below — the *current* `group_duplicates_` binding is broken,
Parity finding P4).
```

### `group_duplicates_` (recommended redesign)

```
Group sectors that share the same quantum numbers, in place.

Returns
-------
list of int
    A mapper such that `mapper[old_index] == new_index` under the grouped
    result. `self` is mutated in place (sectors merged and sorted ascending).

Notes
-----
In-place: this is a *new* binding directly onto
`Bond_impl::group_duplicates_()`, which is the only correct C++ in-place
mutator for this operation but is currently unreachable from Python (Parity
finding P4) — the presently-shipped `group_duplicates_` binding is bound to
a *different*, `const`, copy-returning C++ overload and must be removed.
Pairs with the pure `group_duplicates`.
```

### `has_duplicate_qnums`

```
Check whether this bond has any duplicate quantum-number sectors.

Returns
-------
bool
    Always False for a regular (BD_REG) bond.
```

### `qnums`

```
Return this bond's quantum-number sectors.

Returns
-------
list of list of int
    Shape (# sectors, # symmetries). A fresh Python list (confirmed by
    probe: qnums() returns a copy, not the C++-side mutable-reference view
    — Parity finding P2). Mutating the returned list never affects this
    bond, unlike the equivalent C++ non-const `qnums()` overload.
```

### `qnums_clone`

```
Return a deep copy of this bond's quantum-number sectors.

Returns
-------
list of list of int
    Behaviorally identical to `qnums()` in Python today (Parity finding
    P3); kept for explicit-intent parity with C++, where `qnums_clone()`
    and `qnums()` are not interchangeable.
```

### `redirect`

```
Return a copy of this bond with its type flipped between BD_BRA and BD_KET.

Returns
-------
Bond
    A new, independent Bond; `self` is left unchanged (confirmed by probe).

Notes
-----
Pure/copy. Pairs with the in-place `redirect_`.
```

### `redirect_`

```
Flip this bond's type between BD_BRA and BD_KET, in place.

Returns
-------
Bond
    `self`, after mutation (confirmed by probe: redirect_() mutates the
    receiver's type in place and returns the identical wrapper object).

Notes
-----
In-place. Pairs with the pure `redirect`.
```

### `retype`

```
Return a copy of this bond with a new bondType.

Parameters
----------
bond_type : bondType
    The new type: BD_BRA, BD_KET, or BD_REG.

Returns
-------
Bond
    A new, independent Bond; `self` is left unchanged (confirmed by probe).

Notes
-----
Pure/copy. Pairs with the in-place `retype_` (recommended rename of the
current `set_type`, Consistency finding C1).
```

### `retype_` (recommended rename of `set_type`)

```
Change this bond's bondType, in place.

Parameters
----------
bond_type : bondType
    The new type: BD_BRA, BD_KET, or BD_REG.

Returns
-------
Bond
    `self`, after mutation (confirmed by probe: set_type() mutates the
    receiver in place and returns the identical wrapper object).

Notes
-----
In-place. This is a rename of the current `set_type` (Consistency finding
C1): the operation was already in-place, but the name carried no trailing
`_` and shared no base name with its pure counterpart `retype`. Raises if
changing to/from BD_REG on a bond that carries quantum numbers.
```

### `syms`

```
Return this bond's Symmetry objects.

Returns
-------
list of Symmetry
    A fresh Python list (confirmed by probe pattern, same as qnums() —
    Parity finding P2). Mutating the returned list never affects this bond.
```

### `syms_clone`

```
Return a deep copy of this bond's Symmetry objects.

Returns
-------
list of Symmetry
    Behaviorally identical to `syms()` in Python today (Parity finding P3);
    kept for explicit-intent parity with C++.
```

### `bond_type` (recommended rename of `type`)

```
Return this bond's type.

Returns
-------
bondType
    BD_REG, BD_BRA, or BD_KET.

Notes
-----
Renamed from `type` (Consistency finding C6) to match the constructor/
`init` parameter name `bond_type`, and to avoid shadowing the Python
builtin `type`.
```

### `get_unique_qnums` (new, P7)

```
Return this bond's quantum-number sectors, deduplicated and sorted
descending.

Parameters
----------
return_counts : bool, optional
    If True, also return the count of each unique sector as a
    `(unique_qnums, counts)` tuple. Default False.

Returns
-------
list of list of int, or (list of list of int, list of int)
    A fresh, independent result.

Notes
-----
New Python binding for the currently-C++-only `getUniqueQnums`
(`Bond::getUniqueQnums`, Parity finding P7). Pure/copy — does not mutate `self`.
```

### `__mul__` / `__imul__` (new, P8)

```
__mul__(self, rhs: Bond) -> Bond
    Operator form of `combine_bond`. `b1 * b2` is equivalent to
    `b1.combine_bond(b2)`. Pure/copy.

__imul__(self, rhs: Bond) -> Bond
    Operator form of `combine_bond_`. `b1 *= b2` is equivalent to
    `b1.combine_bond_(b2)`. In-place, returns `self`.

Notes
-----
New bindings (Parity finding P8, B5): C++ already defines
`operator*`/`operator*=` as aliases for `combineBond`/`combineBond_`;
Python currently has no operator form at all.
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Init` | `init` | N1 casing (C4) |
| `Load` | `load` | N1 casing (C4) |
| `Nsym` | `nsym` | N1 casing (C4) |
| `Save` | `save` | N1 casing (C4) |
| `c_group_duplicates_refarg` | *(removed)* | internal plumbing (C5), superseded by `group_duplicates` |
| `c_redirect_` | *(removed)* | internal plumbing (C5), superseded by `redirect_` |
| `calc_reverse_qnums` | `reverse_qnums` | drop redundant verb prefix |
| `clear_type` | `clear_type_` (+ new pure `clear_type`) | N2 in-place suffix (C2) |
| `combineBond` | `combine_bond` | N1 casing (C4) |
| `combineBond_` | `combine_bond_` | N1 casing (C4); also now returns `self` (C7) |
| `combineBonds` | *(removed — use `combine_bond` with a list)* | deprecated in C++, duplicate operation (P1) |
| `combineBonds_` | *(removed — use `combine_bond_` with a list)* | deprecated in C++, duplicate operation (P1) |
| `getDegeneracies` | `get_degeneracies` | N1 casing (C4) |
| `getDegeneracy` | `get_degeneracy` (fixed) | N1 casing (C4) + ship the already-written bugfix (P6) |
| `group_duplicates_` (current, broken binding) | `group_duplicates_` (rebound to the true C++ in-place mutator) | P4/C3: name was bound to the wrong (`const`) C++ overload |
| `set_type` | `retype_` | N2 pairing with `retype` (C1) |
| `type` | `bond_type` | matches constructor's `bond_type` parameter name; avoids shadowing builtin `type` (C6) |
| *(none — C++-only)* `getUniqueQnums` | `get_unique_qnums` (new Python binding) | P7 |
| *(none — not bound)* `operator*` / `operator*=` | `__mul__` / `__imul__` (new Python bindings) | P8/B5 |
| `c_getDegeneracy_refarg` | *(removed once `get_degeneracy` fix ships and is re-verified)* | interim-only internal plumbing (C5/P6) |

All members not listed in this table (`calc_reverse_qnums`* aside — listed
above for its rename — `clone`, `combine_bond`/`combine_bond_` post-rename,
`dim`, `get_fermion_parity`, `group_duplicates`, `has_duplicate_qnums`,
`qnums`, `qnums_clone`, `redirect`, `redirect_`, `retype`, `syms`,
`syms_clone`) keep both their current name and current behavior.
