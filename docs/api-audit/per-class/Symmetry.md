# `Symmetry` — API audit

`Symmetry` is the small value-type object attached to a symmetric `Bond`'s
quantum-number sectors: it identifies which abelian group (U(1), Z_n) or
fermionic parity/number rule governs those quantum numbers, and it supplies
the `combine_rule`/`reverse_rule` used when `Bond`s are combined or reversed.
This document audits the 16 public members of the live `cytnx.Symmetry`
class (installed `cytnx==1.1.0` wheel).

Ground truth for behavior is `docs/api-audit/probes/Symmetry.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/Symmetry.py`; all 36 assertions `[PASS]`, exit 0).
Ground truth for static signatures is `cytnx_src/include/Symmetry.hpp` (C++
declarations, on both the public `Symmetry` wrapper and its
`Symmetry_base`/`U1Symmetry`/`ZnSymmetry`/`FermionParitySymmetry`/
`FermionNumberSymmetry` implementation hierarchy), `cytnx_src/src/Symmetry.cpp`
(the actual per-subtype rule implementations — authoritative for what
`combine_rule`/`reverse_rule`/`check_qnum` really compute, and for which
inputs raise), `cytnx_src/pybind/symmetry_py.cpp` (the pybind11 binding —
authoritative for the Python-visible call signature, and notably for a
Python-only `NormalizeZnInput` shim that has no C++-side counterpart at all,
see Parity finding P1), and `cytnx_src/cytnx/Symmetry_conti.py` (Python-side
augmentation — for `Symmetry` itself this file adds no methods; it only
defines the unrelated free-function `Qs(*args)` helper used when building
symmetric `Bond`s).

## Inventory

C++ signatures are read from `Symmetry.hpp`/`Symmetry.cpp`; Python signatures
are the effective pybind-visible signature, cross-checked against
`tools/member_inventory.py Symmetry`.

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `FermionNumber` (static) | `static Symmetry FermionNumber()` | `FermionNumber() -> Symmetry` (staticmethod) |
| `FermionParity` (static) | `static Symmetry FermionParity()` | `FermionParity() -> Symmetry` (staticmethod) |
| `Load` (static) | `static Symmetry Load(const string&)`; `static Symmetry Load(const char*)` | `Load(fname: str) -> Symmetry` (staticmethod) |
| `Save` | `void Save(const string&) const`; `void Save(const char*) const` | `Save(fname: str) -> None` |
| `U1` (static) | `static Symmetry U1()` | `U1() -> Symmetry` (staticmethod) |
| `Zn` (static) | `static Symmetry Zn(const int &n)` | `Zn(n: int) -> Symmetry` (staticmethod) |
| `check_qnum` | `bool check_qnum(const cytnx_int64&)` *(dispatches to the active subtype's override; U1/FermionNumber: always `True`; Zn: `0<=qnum<n`; FermionParity: `0<=qnum<2`)* | `check_qnum(qnum: int) -> bool` |
| `check_qnums` | `bool check_qnums(const vector<cytnx_int64>&)` *(all-elements-valid AND of `check_qnum`)* | `check_qnums(qnums: List[int]) -> bool` |
| `clone` | `Symmetry clone() const` | `clone() -> Symmetry`; also bound as `__copy__`/`__deepcopy__` (bound twice at `symmetry_py.cpp:65,69`, harmlessly redundant) |
| `combine_rule` | `vector<int64> combine_rule(const vector<int64>&, const vector<int64>&)` *(batch form, unbound — see P2)*; `int64 combine_rule(const int64&, const int64&, const bool& is_reverse=false) const` | Only the scalar form is bound, via a Python-only wrapper lambda: `combine_rule(qnL: int, qnR: int, is_reverse: bool=False) -> int` — the lambda runs `NormalizeZnInput` on `qnL`/`qnR` *before* calling the native C++ method (see P1); the batch/vector overload has **no Python binding at all** (P2) |
| `get_fermion_parity` | `fermionParity get_fermion_parity(const cytnx_int64&) const` *(base: always `EVEN`; Zn: not overridden, inherits base; FermionParity: `0`→EVEN,`1`→ODD, else raises; FermionNumber: even/odd of the qnum itself)* | `get_fermion_parity(qnum: int) -> fermionParity` |
| `is_fermionic` | `bool is_fermionic() const` *(base/U1/Zn: `False`; FermionParity/FermionNumber: `True`)* | `is_fermionic() -> bool` |
| `n` | `int &n() const` *(reference-returning despite `const`; raw storage — `1` for U1, the modulus for Zn, `-2`/`-1` sentinel values for FermionParity/FermionNumber — see C5)* | `n() -> int` — a primitive `int`, so there is no C++-reference-vs-Python-copy distinction to probe here (unlike `Bond.qnums()`'s container-valued P2 finding) |
| `reverse_rule` | `int64 reverse_rule(const int64&) const` *(dispatches to the active subtype: U1/FermionNumber: `-in`; Zn: `(n-in)%n` **after** `ValidateZnQnum` raises on out-of-range `in`; FermionParity: `-in+2`, see C6)* | Python-only wrapper lambda, same `NormalizeZnInput`-before-native-call pattern as `combine_rule`: `reverse_rule(qin: int) -> int` (see P1) |
| `stype` | `int stype() const` | `stype() -> int` — one of `cytnx.SymType.{U,Z,fPar,fNum}` (see P6 for the `SymType`/`SymmetryType` name mismatch) |
| `stype_str` | `string stype_str() const` | `stype_str() -> str` — e.g. `"U1"`, `"Z3"`, `"fP"`, `"f#"` |
| *(C++-only)* `combine_rule_`/`reverse_rule_` | `void combine_rule_(vector<int64>&, const vector<int64>&, const vector<int64>&)`; `void combine_rule_(int64&, const int64&, const int64&, const bool&)`; `void reverse_rule_(int64&, const int64&)` | **absent from Python** — no binding of any kind (see P3) |
| *(C++-only)* `print_info` | `void print_info() const` (prints directly to `std::cout`) | **no direct Python binding**; only reachable as an undocumented side effect of the broken `__repr__`/`__str__` (see P5) |
| *(C++-only)* 2-arg constructor | `Symmetry(const int &stype=-1, const int &n=0)` | **not exposed** — only `py::init<>()` (zero-arg) is registered (`symmetry_py.cpp:59-60`, the 2-arg overload is commented out); Python callers must use the static factories (see P4) |
| *(C++-only)* `operator!=` | `bool operator!=(const Symmetry&) const` | not explicitly bound, but works via Python's default `__ne__`-delegates-to-`__eq__` behavior (same non-issue pattern noted in `Bond.md`'s P8) |

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/Symmetry.py`.

- **P1 — headline finding: for Zn symmetries, the Python binding silently
  bypasses a C++-side hard validation and replaces it with a
  deprecation-warning-only auto-correction — a genuine B4 violation.**
  Raw C++ `ZnSymmetry::combine_rule_`/`reverse_rule_`
  (`cytnx_src/src/Symmetry.cpp:126-151`) call `ValidateZnQnum`, which invokes
  `cytnx_error_msg` (raises, see e.g. the `Zn(1)`-construction and
  `FermionParity.get_fermion_parity(2)` cases below) for **any** qnum outside
  `[0, n)`. But `symmetry_py.cpp`'s `combine_rule`/`reverse_rule` bindings are
  lambdas that first run every scalar argument through a Python-only helper,
  `NormalizeZnInput` (`symmetry_py.cpp:22-36`, no C++ equivalent at all): for
  a Zn symmetry, an out-of-range qnum is silently reduced modulo `n` and a
  `FutureWarning` is emitted ("deprecated and will be rejected in v2.0.0");
  only *then* is the (now in-range) value passed into the native C++ method —
  so `ValidateZnQnum` never sees the out-of-range input and never raises.
  Probed directly: *"Zn(3).check_qnum(4) is False"* (proving 4 is genuinely
  invalid by the class's own predicate) immediately followed by
  *"Zn(3).combine_rule(4, 4) does NOT raise ...; it silently normalizes (4->1)
  and returns (1+1)%3 == 2"* and *"...the only signal ... is a FutureWarning
  ..., not an exception"* — both `[PASS]`; the same pattern is confirmed for
  `reverse_rule` (*"Zn(3).reverse_rule(5) ... normalizes-with-warning instead
  of raising"*, `[PASS]`). To rule out "Symmetry just never raises," the probe
  also confirms `Zn(1)` construction *does* raise a catchable `RuntimeError`
  (*"Zn(1) ... DOES raise ... proving this class raises hard errors
  elsewhere"*, `[PASS]`) — so the combine_rule/reverse_rule silent-normalize
  path is a deliberate, Zn/scalar-path-specific carve-out, not a blanket "this
  class never validates" policy. Net effect: a C++ caller invoking the
  equivalent raw method with an out-of-range Zn qnum gets an exception; a
  Python caller gets a silently-different (and, for non-idempotent inputs,
  silently-wrong-if-the-warning-is-ignored) numeric result instead.
- **P2 — `combine_rule`'s C++ batch/vector overload
  (`combine_rule(const vector<int64>&, const vector<int64>&)`) has no Python
  binding at all**, unlike the scalar overload. Only
  `symmetry_py.cpp:76-83`'s scalar lambda is registered; there is no second
  `.def("combine_rule", ...)` overload accepting lists. A Python caller who
  wants to batch-combine two whole qnum arrays (the actual use case when
  combining two `Bond`s' full qnum sets) must loop element-by-element in
  Python instead of making one call, unlike C++. Probed by invoking with
  lists and catching the failure: *"combine_rule() rejects list arguments
  (TypeError): the C++ batch/vector overload is not bound to Python"*
  `[PASS]`.
- **P3 — the C++ out-param mutating forms `combine_rule_`/`reverse_rule_` are
  completely absent from the Python binding** (not merely mis-bound, the way
  `Bond.group_duplicates_` was bound to the wrong overload — here the
  attribute doesn't exist under any name). Since the pure, return-value forms
  (`combine_rule`/`reverse_rule`) already provide the identical result, this
  is a low-severity, informational gap — no functionality is actually lost —
  but it does mean **`Symmetry` has zero mutating methods reachable from
  Python at all** (contrast `Bond`, which has several `_`-suffixed in-place
  methods). Probed: *"Symmetry has no combine_rule_ attribute at all"* and
  *"...no reverse_rule_ attribute at all"* (`hasattr` checks), reinforced by
  an actual invocation attempt: *"...confirmed by direct call:
  sym.combine_rule_(...) raises AttributeError"` — all three `[PASS]`.
- **P4 — the Python-visible constructor is default-args-only and always
  yields a U1 symmetry; the C++ 2-arg constructor is not exposed at all.**
  `symmetry_py.cpp:59-60` registers only `py::init<>()`; the C++
  `Symmetry(stype=-1, n=0)` 2-arg constructor is present in the header but
  its pybind registration is commented out. Consequently `cytnx.Symmetry()`
  can only ever construct the *default* (`stype=-1` i.e. `U`, `n=0`) case —
  and, separately, C++'s own `Symmetry::Init` hardcodes `new U1Symmetry(1)`
  for the U branch regardless of the `n` argument passed in, so even the
  unexposed 2-arg constructor could not have produced anything but U1(n=1)
  for `stype=U` in the first place. Probed: *"Symmetry() default constructor
  equals Symmetry.U1() (stype/n both match)"* `[PASS]`, and directly
  attempting the 2-arg call: *"Symmetry(stype, n) (the C++ 2-arg constructor)
  is NOT exposed to Python; calling it raises TypeError"* `[PASS]`. Python
  callers must use the `U1()`/`Zn()`/`FermionParity()`/`FermionNumber()`
  static factories to construct anything other than a bare U1 symmetry.
- **P5 — `__repr__`/`__str__` always evaluate to the empty string; the
  human-readable info is a `print()` side effect, not a return value, and the
  C++ method that produces it (`print_info()`) has no direct Python binding
  of its own.** `symmetry_py.cpp:98-104`'s `__repr__` lambda does
  `std::cout << self << std::endl; return std::string("");` under a
  `py::scoped_ostream_redirect` call guard — so calling `repr(sym)` prints
  `Symmetry_base::print_info()`'s formatted block directly to stdout (as a
  side effect, redirected into Python's `sys.stdout` only for the duration of
  the call) and then *returns* `""`. Probed directly: *"repr(sym) evaluates
  to the empty string '' (not a useful representation)"* and *"str(sym) is
  likewise ''"* — both `[PASS]` — followed by capturing stdout during the
  same call: *"...yet calling repr(sym) DOES print human-readable info as a
  side effect ..., showing the info exists but isn't returned"* `[PASS]`.
  Separately, `print_info` itself is never bound under its own name:
  *"print_info (the C++ method backing the above side-effect print) has no
  direct Python binding of its own -- unreachable except via
  `__repr__`/`__str__`"* `[PASS]`. (`Bond`'s `__repr__` binding at
  `bond_py.cpp:80-85` has the identical "prints + returns `''`" shape — not
  flagged in `Bond.md`, but confirmed here to be the same underlying pattern,
  not a `Symmetry`-only defect.)
- **P6 — the Python-visible enum is named `SymType`, not `SymmetryType`
  (the C++ name) — an N3-relevant divergence, not just an N1 casing
  difference.** `symmetry_py.cpp:40` registers `py::enum_<SymmetryType>(m,
  "SymType")`; per `00-methodology.md`'s N3, a C++/Python name pair that
  diverges by more than the casing rule (here, an outright abbreviation) is a
  parity finding, not a mere consistency nit. `SymType` is a module-level
  enum, not an attribute of `cytnx.Symmetry` itself, so it falls outside this
  document's Recommendation-table coverage (`validate_doc.py` only checks
  `dir(cytnx.Symmetry)`), but it is exercised directly by every `stype()`
  call in this document's own probes and is recorded here for completeness.
  Probed: *"The Python-visible enum is named SymType (not SymmetryType...)"*
  `[PASS]` (`hasattr(cytnx, "SymType")` is `True`, `hasattr(cytnx,
  "SymmetryType")` is `False`).

## Consistency findings

- **C1 — violates N1: capitalized callable members.** `FermionNumber`,
  `FermionParity`, `Load`, `Save`, `U1`, `Zn` all use capitalized forms
  instead of `snake_case`. (These are all static factories/IO methods, not
  mutate/pure operation pairs, so N2 does not apply to them — only the N1
  casing rule does, same reasoning as `Bond.md`'s C4 for `Init`/`Load`/`Save`.)
- **C2 — violates N5: `check_qnum`/`check_qnums` are boolean predicates
  without an `is_`/`has_` prefix.** Both return `bool` and read as
  verb-phrases ("check the qnum") rather than yes/no questions. Per N5
  ("Predicate methods (returning bool) are prefixed `is_`/`has_` unless
  already a self-evident adjective"), recommend renaming to
  `is_valid_qnum`/`is_valid_qnums`, which also reads correctly at the call
  site (`if sym.is_valid_qnum(q): ...`).
- **C3 — violates N4: five sibling methods use four different words for
  "a quantum number" parameter.** `check_qnum(qnum)`, `check_qnums(qnums)`,
  `get_fermion_parity(qnum)` already agree on `qnum`/`qnums`, but
  `combine_rule(qnL, qnR, is_reverse)` uses `qnL`/`qnR` and
  `reverse_rule(qin)` uses `qin` — three more spellings of the same concept
  within one class. Per N4 ("semantically equivalent parameters use the same
  name and position across... sibling methods"), recommend renaming
  `combine_rule`'s parameters to `qnum_l`/`qnum_r` and `reverse_rule`'s to
  `qnum`, so every parameter that means "a quantum number value" shares the
  `qnum`/`qnums`/`qnum_l`/`qnum_r` root.
- **C4 — N2 is vacuously satisfied: `Symmetry` has no mutate-only or
  pure-only asymmetric method pairs among its Python-bound members**, because
  it has *no* Python-bound mutating methods at all (P3) — every bound method
  is pure/return-value. Recorded explicitly (rather than silently omitted)
  because the absence of any N2 finding here is a direct consequence of P3,
  not an indication that N2 was never checked.
- **C5 — the `n()` getter's meaning is type-dependent, and not a real "n"
  for two of the four subtypes.** `n()` returns the discrete modulus for Zn
  (its only mathematically meaningful use — e.g. `Zn(3).n() == 3`), a
  constant `1` for U1, and internal negative sentinel values (`-2` for
  FermionParity, `-1` for FermionNumber, confirmed by probe) that are not
  "a symmetry size" in any user-facing sense — they are `Symmetry_base`'s raw
  storage field, reused as a type discriminant by the fermionic subtypes.
  No `N`/`B` rule in `00-methodology.md` speaks directly to "a getter's
  return value is only meaningful for some subtypes of the same class," so
  this is flagged as a plain internal-semantics inconsistency rather than a
  cited violation (same treatment as `Bond.md`'s C6): recommend the
  docstring explicitly enumerate the per-subtype meaning (done below), since
  renaming is not obviously an improvement (`n` is the standard "Z_n" /
  U(1)-as-"Z_1" math notation for the two subtypes where it *is* meaningful).
- **C6 — `FermionParitySymmetry::reverse_rule_`'s formula can produce a qnum
  that the very same object's own `check_qnum()` rejects.**
  `cytnx_src/src/Symmetry.cpp:191` implements `reverse_rule_` as
  `out = -in + 2`; for `in = 0` (a valid parity qnum), this yields `out = 2`,
  but `FermionParitySymmetry::check_qnum` only accepts `[0, 2)`
  (`cytnx_src/src/Symmetry.cpp:159-161`), so `check_qnum(2)` is `False`. This
  is identical in C++ and Python (no `NormalizeZnInput`-style wrapper touches
  non-Zn subtypes), so it is a genuine internal self-inconsistency in
  `FermionParitySymmetry`, not a C++-vs-Python parity gap. Probed:
  *"FermionParity.reverse_rule(0) == 2, a value FermionParity.check_qnum()
  itself rejects"* `[PASS]`. Flagged informally (no `N`/`B` id covers "a
  method's output fails the class's own validity predicate"); the correct
  fix is almost certainly `out = (2 - in) % 2` (i.e. `1 - in`), which stays
  in `[0, 2)` for both valid inputs.
- **C7 — the Zn-only deprecation-warning safety net (P1) is not applied
  uniformly across the four symmetry subtypes.** `FermionParitySymmetry`'s
  `combine_rule_` also implicitly wraps out-of-range input via `% 2`
  (`cytnx_src/src/Symmetry.cpp:184-189`), with **no validation and no
  warning at all** — not even the `FutureWarning` Zn gets, because
  `NormalizeZnInput` only activates for `stype() == SymmetryType::Z`. Probed:
  *"FermionParity.combine_rule(5, 5) silently wraps out-of-range input
  ((5+5)%2 == 0) with NO warning at all, unlike Zn's combine_rule"* `[PASS]`.
  Flagged informally (this is a consequence of P1's narrowly-scoped shim, not
  a separate cited `N`/`B` violation): recommend either extending validation
  (or at least the deprecation warning) to all subtypes' scalar
  `combine_rule`/`reverse_rule`, or removing the Zn-only special case
  entirely in favor of the uniform "just wrap, never validate" behavior
  every other subtype already has.

## Recommendation

Every one of the 16 live public members of `cytnx.Symmetry` appears below,
tagged **keep / add / rename / remove**. Three additional informational rows
cover the always-empty `__repr__`/`__str__` (P5, fix), the unbound
`print_info` (P5, add), and the C++-only vector `combine_rule` overload (P2,
folded into the `combine_rule` row rather than a separate name, since it
would share the same Python name).

| Member | Verdict | Rationale |
|---|---|---|
| `FermionNumber` | rename | → `fermion_number` (C1/N1). Static factory. |
| `FermionParity` | rename | → `fermion_parity` (C1/N1). Static factory. |
| `Load` | rename | → `load` (C1/N1). Static factory. |
| `Save` | rename | → `save` (C1/N1). |
| `U1` | rename | → `u1` (C1/N1). Static factory. |
| `Zn` | rename | → `zn` (C1/N1). Static factory. |
| `check_qnum` | rename | → `is_valid_qnum` (C2/N5): boolean predicate needs an `is_`/`has_` prefix. |
| `check_qnums` | rename | → `is_valid_qnums` (C2/N5), for the same reason. |
| `clone` | keep | Correctly named; returns a distinct, value-equal object (confirmed by probe). De-duplicate the doubled `.def("clone", ...)` registration (`symmetry_py.cpp:65,69`) as a harmless internal cleanup. |
| `combine_rule` | keep | Rename parameters `qnL`/`qnR` → `qnum_l`/`qnum_r` (C3/N4). **Must also gain the missing batch/list overload** (P2): `combine_rule(qnums_l: List[int], qnums_r: List[int]) -> List[int]`, binding the currently-unreachable C++ vector overload under the same name. |
| `get_fermion_parity` | keep | Already `snake_case`, already uses the target parameter name `qnum` (the N4 model other methods should match, C3). |
| `is_fermionic` | keep | Already `snake_case`, correctly `is_`-prefixed (N5). |
| `n` | keep | Document explicitly, per subtype, what the value means (C5) — do not rename; `n` is standard math notation for the Zn/U1 cases where it is meaningful. |
| `reverse_rule` | keep | Rename parameter `qin` → `qnum` (C3/N4), matching `combine_rule`'s renamed parameters and `get_fermion_parity`'s existing `qnum`. |
| `stype` | keep | Already `snake_case`, correctly minimal. |
| `stype_str` | keep | Already `snake_case`. |
| *(fix)* `__repr__` / `__str__` | rename | Fix the implementation (P5) to build and *return* the info string (e.g. via an `ostringstream` instead of `std::cout`) instead of printing directly and returning `""`; `repr(sym)`/`str(sym)` should show `stype_str()` and the combine/reverse rule text. |
| *(new)* `print_info` | add | Bind `Symmetry::print_info()` directly (P5): kept as an explicit, separately-callable method (distinct from the now-fixed `__repr__`) for interactive/notebook use, matching the C++ name (already `snake_case`, no N1 rename needed). |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`
above, under its recommended name.

### `fermion_number` (static)

```
Create a fermionic occupation-number symmetry object.

Returns
-------
Symmetry
    A new generator Symmetry with stype fNum. Combine rule: Q1 + Q2.
    Reverse rule: -Q. Fermionic (is_fermionic() is True); parity is EVEN
    for even qnum, ODD for odd qnum.

Notes
-----
Renamed from `FermionNumber` (C1/N1 casing).
```

### `fermion_parity` (static)

```
Create a fermionic-parity symmetry object.

Returns
-------
Symmetry
    A new generator Symmetry with stype fPar. Valid qnum range [0, 2)
    (0 = EVEN, 1 = ODD). Combine rule: (Q1 + Q2) % 2. Fermionic.

Notes
-----
Renamed from `FermionParity` (C1/N1 casing).
```

### `load` (static)

```
Load a Symmetry previously written by `save`.

Parameters
----------
fname : str
    File path (the `.cysym` extension is appended automatically by `save`
    and expected here).

Returns
-------
Symmetry
    A new Symmetry reconstructed from the file (value-equal to the saved
    original, confirmed by probe).

Notes
-----
Renamed from `Load` (C1/N1 casing).
```

### `save`

```
Save this Symmetry to a file.

Parameters
----------
fname : str
    File path; the `.cysym` extension is appended automatically.

Returns
-------
None

Notes
-----
Renamed from `Save` (C1/N1 casing).
```

### `u1` (static)

```
Create a U(1) symmetry object.

Returns
-------
Symmetry
    A new generator Symmetry with stype U, n == 1. Valid qnum range
    (-inf, inf). Combine rule: Q1 + Q2. Reverse rule: -Q. Not fermionic.

Notes
-----
Renamed from `U1` (C1/N1 casing). `Symmetry()` (the bare default
constructor) is equivalent to `Symmetry.u1()` (Parity finding P4), but
using `u1()` explicitly is recommended since it states intent.
```

### `zn` (static)

```
Create a Z_n discrete symmetry object.

Parameters
----------
n : int
    The discrete modulus, n > 1.

Returns
-------
Symmetry
    A new generator Symmetry with stype Z, n == the given n. Valid qnum
    range [0, n). Combine rule: (Q1 + Q2) % n. Reverse rule: (n - Q) % n.
    Not fermionic.

Raises
------
RuntimeError
    If n <= 1 (confirmed by probe: Zn(1) raises).

Notes
-----
Renamed from `Zn` (C1/N1 casing).
```

### `is_valid_qnum`

```
Check whether a single quantum number is within this Symmetry's valid range.

Parameters
----------
qnum : int
    The quantum number to check.

Returns
-------
bool
    True for any qnum for U1/FermionNumber; True iff 0 <= qnum < n() for
    Zn; True iff qnum in {0, 1} for FermionParity.

Notes
-----
Renamed from `check_qnum` (C2/N5): boolean predicates get an `is_`/`has_`
prefix. NOTE: this predicate is NOT consulted by `combine_rule`/
`reverse_rule` on a Zn symmetry, which instead silently normalize an
out-of-range qnum with only a deprecation warning (Parity finding P1) —
callers who want a hard rejection should call `is_valid_qnum` explicitly
before combining.
```

### `is_valid_qnums`

```
Check whether every quantum number in a list is within this Symmetry's
valid range.

Parameters
----------
qnums : list of int
    The quantum numbers to check.

Returns
-------
bool
    True iff `is_valid_qnum` would return True for every element
    (confirmed by probe: an all-valid list returns True, a list with one
    out-of-range element returns False).

Notes
-----
Renamed from `check_qnums` (C2/N5).
```

### `clone`

```
Return an independent copy of this Symmetry.

Returns
-------
Symmetry
    A new, distinct Symmetry object (confirmed by probe: `clone() is not
    self`), value-equal to `self` (`clone() == self`).

Notes
-----
Also reachable via `copy.copy()`/`copy.deepcopy()` (both bound to this
same method). Symmetry has no Python-reachable mutating method (Parity
finding P3), so unlike `Bond.clone()`, independence cannot be demonstrated
by mutating the clone and observing the source is unaffected — only
distinct-identity-plus-value-equality is observable from Python.
```

### `combine_rule`

```
Apply this Symmetry's combine rule to two quantum numbers (or two lists of
quantum numbers).

Parameters
----------
qnum_l : int
    The first quantum number.
qnum_r : int
    The second quantum number.
qnums_l : list of int
    Alternative to `qnum_l`/`qnum_r`: element-wise-combine two full lists
    of quantum numbers in one call (this list-accepting overload is a
    *new* Python binding of a C++ overload that currently has none, Parity
    finding P2).
qnums_r : list of int
    Paired with `qnums_l`.
is_reverse : bool, optional
    If True, negate (apply `reverse_rule`) after combining. Default False.
    Only meaningful for the scalar form.

Returns
-------
int, or list of int
    The combined quantum number(s): `Q1 + Q2` for U1/FermionNumber,
    `(Q1 + Q2) % n` for Zn, `(Q1 + Q2) % 2` for FermionParity.

Notes
-----
For a Zn symmetry, an out-of-range `qnum_l`/`qnum_r` is NOT rejected: it is
silently reduced modulo `n` and a `FutureWarning` is raised (deprecated,
scheduled to become a hard error in v2.0.0) — see Parity finding P1. Use
`is_valid_qnum`/`is_valid_qnums` first if you need a hard check today.
Renamed parameters from `qnL`/`qnR` (Consistency finding C3/N4).
```

### `get_fermion_parity`

```
Return the fermionic parity of a given quantum number under this Symmetry.

Parameters
----------
qnum : int
    The quantum number to classify.

Returns
-------
fermionParity
    EVEN or ODD. Always EVEN for U1/Zn (not fermionic). For FermionParity,
    EVEN iff qnum == 0, ODD iff qnum == 1 (any other value raises
    RuntimeError, confirmed by probe). For FermionNumber, EVEN/ODD
    matches qnum's own parity.
```

### `is_fermionic`

```
Check whether this Symmetry is a fermionic symmetry.

Returns
-------
bool
    True for FermionParity/FermionNumber, False for U1/Zn (confirmed by
    probe).
```

### `n`

```
Return this Symmetry's discrete parameter.

Returns
-------
int
    For Zn: the modulus n (e.g. 3 for Z3) — the only subtype where this
    value is a meaningful "size". For U1: always 1. For FermionParity: the
    internal sentinel -2. For FermionNumber: the internal sentinel -1 (the
    last two are raw `Symmetry_base` storage, not meaningful qnum-range
    information — see Consistency finding C5; use `stype()`/`stype_str()`
    to identify a fermionic Symmetry instead of inspecting `n()`).
```

### `reverse_rule`

```
Apply this Symmetry's reverse rule to a quantum number.

Parameters
----------
qnum : int
    The quantum number to reverse.

Returns
-------
int
    -qnum for U1/FermionNumber; (n - qnum) % n for Zn; a parity-flip value
    for FermionParity (see Consistency finding C6 for a known internal
    inconsistency in this subtype's formula).

Notes
-----
For a Zn symmetry, an out-of-range `qnum` is NOT rejected: it is silently
reduced modulo `n` with only a FutureWarning, the same P1 shim as
`combine_rule`. Renamed parameter from `qin` (Consistency finding C3/N4).
```

### `stype`

```
Return this Symmetry's type id.

Returns
-------
int
    One of cytnx.SymType.{U, Z, fPar, fNum} (see Parity finding P6 for the
    SymType/SymmetryType C++/Python name mismatch).
```

### `stype_str`

```
Return this Symmetry's type name as a human-readable string.

Returns
-------
str
    "U1" / "Z<n>" (e.g. "Z3") / "fP" / "f#".
```

### `__repr__` / `__str__` (recommended fix)

```
__repr__(self) -> str
    Return (not print) the same info block Symmetry::print_info()
    currently only prints: stype_str(), the combine rule, and the reverse
    rule, as a single formatted string.

Notes
-----
Fixes Parity finding P5: the current binding prints directly to stdout
via std::cout and returns the empty string, so repr(sym)/str(sym)
evaluate to '' today (confirmed by probe) even though a
`print(repr(sym))`-style side effect shows real content.
```

### `print_info` (new, P5)

```
Print this Symmetry's info block (stype, combine rule, reverse rule) to
stdout.

Returns
-------
None

Notes
-----
New direct Python binding of the existing C++ Symmetry::print_info()
(Parity finding P5), which currently has no binding of its own and is
reachable only as an undocumented side effect of __repr__/__str__. Once
__repr__ is fixed to return a string (see above), print_info() remains
useful as an explicit "just print it" convenience, e.g. `print(sym)` is
then equivalent to `print(repr(sym))`, while `sym.print_info()` writes
directly without needing a `print()` wrapper.
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `FermionNumber` | `fermion_number` | N1 casing (C1) |
| `FermionParity` | `fermion_parity` | N1 casing (C1) |
| `Load` | `load` | N1 casing (C1) |
| `Save` | `save` | N1 casing (C1) |
| `U1` | `u1` | N1 casing (C1) |
| `Zn` | `zn` | N1 casing (C1) |
| `check_qnum` | `is_valid_qnum` | N5 predicate prefix (C2) |
| `check_qnums` | `is_valid_qnums` | N5 predicate prefix (C2) |
| `combine_rule(qnL, qnR, is_reverse)` | `combine_rule(qnum_l, qnum_r, is_reverse)` (+ new list overload) | N4 parameter naming (C3) + missing batch overload (P2) |
| `reverse_rule(qin)` | `reverse_rule(qnum)` | N4 parameter naming (C3) |
| `__repr__` / `__str__` (prints, returns `''`) | `__repr__` / `__str__` (returns the info string) | P5 |
| *(none — not bound)* `print_info` | `print_info` (new Python binding) | P5 |
| *(none — not bound)* `combine_rule` batch/vector overload | `combine_rule(qnums_l, qnums_r)` (new Python binding, same name) | P2 |

Every other public member of `Symmetry` — `clone`, `get_fermion_parity`,
`is_fermionic`, `n`, `stype`, `stype_str` — has no row above and keeps both
its current name and current behavior unchanged. (`combine_rule` and
`reverse_rule` *do* change — parameter names, and `combine_rule` gains an
overload — see their rows above; they are intentionally excluded from this
"unchanged" list, not omitted by oversight.)
