# Enums / config units — API audit (`Type`, `Device`, `SymType`, `bondType`, `fermionParity`)

This document audits the five "named-constant set" units of the Cytnx public
API in one place, because each is small and they share the same modelling
question: *how should a fixed set of named integer codes be exposed to
Python?* The five are:

- **`Type`** — the dtype code set (`Double`, `ComplexDouble`, …) used
  everywhere a `Tensor`/`UniTensor`/`Storage` dtype is named.
- **`Device`** — the device selector (`cpu`, `cuda`) plus a few host-capability
  query helpers.
- **`SymType`** — the symmetry-kind code (`U`, `Z`, `fPar`, `fNum`) returned by
  `Symmetry.stype()` (see `Symmetry.md` P6 for the `SymType`/`SymmetryType`
  name mismatch).
- **`bondType`** — the bond-direction code (`BD_KET`/`BD_BRA`/`BD_REG` and the
  `BD_IN`/`BD_OUT` aliases) carried by every `Bond` (see `Bond.md`).
- **`fermionParity`** — the two-valued parity code (`EVEN`/`ODD`) returned by
  `Symmetry.get_fermion_parity()`.

The headline structural finding is that these five units are **not modelled
uniformly**: four (`Type`, `SymType`, `bondType`, `fermionParity`) are real
`py::enum_`-bound types whose members are distinct wrapper objects, while
`Device` is a plain Python **submodule** whose members are bare `int`s (C3).
On top of that, the four real enums are all bound with `export_values()` and
**without** any per-type identity, so members of unrelated enums that share an
underlying integer compare and hash equal — a silent cross-type collision
(C1) — and every member is additionally dumped into the top-level `cytnx.*`
namespace (C2).

Ground truth for behavior is `docs/api-audit/probes/enums.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/enums.py`; all 27 assertions `[PASS]`, exit 0). Ground
truth for static signatures/codes is `cytnx_src/include/Type.hpp` and
`Device.hpp` (the `Type_class`/`Device_class` C++ definitions and their code
tables), `cytnx_src/include/Symmetry.hpp` (`SymmetryType` and `fermionParity`),
`cytnx_src/include/Bond.hpp` (`bondType`), and the pybind bindings
`cytnx_src/pybind/cytnx.cpp` (`Type`/`Device` registration, `export_values()`
at `cytnx.cpp:64`) and `cytnx_src/pybind/symmetry_py.cpp` (`SymType`/
`fermionParity` registration + `export_values()` at `symmetry_py.cpp:41,45`);
`bondType` is registered in `cytnx_src/pybind/bond_py.cpp`.

## Inventory

C++ codes are read from the headers cited above; Python members and their
live integer values are cross-checked against `tools/member_inventory.py
<Unit>` and the probe. Each of the four real enums additionally exposes the
standard pybind11 `name`/`value` descriptors (probe: *"An enum instance
exposes .name/.value"* `[PASS]`); `Device`, being a module, does not.

### `Type` (14 members: 12 dtype codes + `name` + `value`)

| Member | C++ (`Type.hpp`) | Python (live) |
|---|---|---|
| `Void` | `Type_class::Void = 0` | `Type.Void`, `int(...) == 0` |
| `ComplexDouble` | `= 1` | `int(...) == 1` |
| `ComplexFloat` | `= 2` | `int(...) == 2` |
| `Double` | `= 3` | `int(...) == 3` |
| `Float` | `= 4` | `int(...) == 4` |
| `Int64` | `= 5` | `int(...) == 5` |
| `Uint64` | `= 6` | `int(...) == 6` |
| `Int32` | `= 7` | `int(...) == 7` |
| `Uint32` | `= 8` | `int(...) == 8` |
| `Int16` | `= 9` | `int(...) == 9` |
| `Uint16` | `= 10` | `int(...) == 10` |
| `Bool` | `= 11` | `int(...) == 11` |
| `name` | *(pybind descriptor)* | `member.name -> str`, e.g. `Type.Double.name == "Double"` |
| `value` | *(pybind descriptor)* | `member.value -> int`, e.g. `Type.Double.value == 3` |

The exact code order is `cy_typeid_v` (the `Type_list` variant index order in
`Type.hpp`); probe: *"Type's 12 members have the exact integer codes …"*
`[PASS]`. `Type_class`'s large C++ **static utility** surface (`getname`,
`typeSize`, `is_unsigned`, `is_complex`, `is_float`, `is_int`, `type_promote`,
`check_type`) is **not bound at all** (P1).

### `Device` (6 members)

| Member | C++ (`Device.hpp`) | Python (live) |
|---|---|---|
| `cpu` | `Device_class::cpu = -1` | `Device.cpu`, plain `int == -1` |
| `cuda` | `Device_class::cuda = 0` | `Device.cuda`, plain `int == 0` |
| `Ncpus` | host CPU count | plain `int >= 0`, baked in at import |
| `Ngpus` | host GPU count | plain `int >= 0`, baked in at import |
| `getname` | `string Device_class::getname(const int&)` | `getname(id: int) -> str` |
| `Print_Property` | `void Print_Property()` | `Print_Property() -> None` (prints to real stdout, P4) |

`Device` is a `py::module_::def_submodule`, not an enum (probe: *"cytnx.Device
is a plain Python module object … NOT a class/enum"* `[PASS]`); it has no
`__members__` and its `cpu`/`cuda` are ordinary `int`s (P2/C3).

### `SymType` (6 members: 4 codes + `name` + `value`)

| Member | C++ (`Symmetry.hpp` `SymmetryType`) | Python (live) |
|---|---|---|
| `U` | `U = -1` | `int(SymType.U) == -1` |
| `Z` | `Z = 0` | `int(SymType.Z) == 0` |
| `fPar` | `fPar = -2` | `int(SymType.fPar) == -2` |
| `fNum` | `fNum = -3` | `int(SymType.fNum) == -3` |
| `name` / `value` | *(pybind descriptors)* | as for `Type` |

Probe: *"SymType's 4 bound members have the exact integer codes …"* `[PASS]`.
The C++ sentinel `SymmetryType::Void = -99` (uninitialized) has **no Python
binding** (P3). The Python name `SymType` diverges from the C++ `SymmetryType`
(P5; see `Symmetry.md` P6).

### `bondType` (7 members: 5 codes + `name` + `value`)

| Member | C++ (`Bond.hpp` `bondType`) | Python (live) |
|---|---|---|
| `BD_KET` | `BD_KET = -1` | `int(bondType.BD_KET) == -1` |
| `BD_BRA` | `BD_BRA = 1` | `int(bondType.BD_BRA) == 1` |
| `BD_REG` | `BD_REG = 0` | `int(bondType.BD_REG) == 0` |
| `BD_IN` | `BD_IN = -1` *(alias of `BD_KET`)* | value-equal to `BD_KET`, distinct object (C4) |
| `BD_OUT` | `BD_OUT = 1` *(alias of `BD_BRA`)* | value-equal to `BD_BRA`, distinct object (C4) |
| `name` / `value` | *(pybind descriptors)* | as for `Type` |

Probe: *"bondType's 5 bound members have the exact integer codes …"* `[PASS]`.
The C++ `BD_NONE = 0` (itself an alias for `BD_REG`) has **no Python binding**
(P6).

### `fermionParity` (4 members: 2 codes + `name` + `value`)

| Member | C++ (`Symmetry.hpp`) | Python (live) |
|---|---|---|
| `EVEN` | `enum fermionParity : bool { EVEN = false }` → `0` | `int(fermionParity.EVEN) == 0`, but `bool(...)` is `True` (P7) |
| `ODD` | `ODD = true` → `1` | `int(fermionParity.ODD) == 1`, `bool(...)` is `True` |
| `name` / `value` | *(pybind descriptors)* | as for `Type` |

Probe: *"fermionParity's 2 members have the exact integer codes …"* `[PASS]`.

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/enums.py`.

- **P1 — `Type`'s entire C++ static-utility surface is unbound in Python; the
  only reachable equivalent of `Type_class::getname()` is `Tensor.dtype_str()`.**
  `py::enum_<Type_class::Type>` is bound directly, exposing only the 12 dtype
  codes; `Type_class` itself — with `getname`, `typeSize`, `is_unsigned`,
  `is_complex`, `is_float`, `is_int`, `type_promote`, `check_type` — has no
  `py::class_` binding at all. Probed: *"None of Type_class's C++ static
  utility methods … are reachable from Python on cytnx.Type"* `[PASS]`, and the
  only Python route to a dtype's name is via a live tensor:
  *"The only Python-reachable equivalent of Type_class::getname() is via
  Tensor.dtype_str(), NOT via any method on Type itself"* `[PASS]`
  (`cytnx.Tensor([2,2], dtype=Type.Double).dtype_str() == "Double (Float64)"`).
  So a Python caller cannot ask "is this dtype complex/float/unsigned?" or
  "how many bytes is this dtype?" without materialising a tensor or
  reimplementing the predicate — capability the C++ side has as free static
  methods. (Informational surface gap, not a correctness bug.)

- **P2 — `Device` is modelled as a submodule of bare `int`s, not an enum**,
  unlike the other four units. `Device.cpu`/`Device.cuda` are plain Python
  `int` (`isinstance(Device.cpu, int) is True`, `Device.cpu == -1`,
  `Device.cuda == 0`), and `Device` has no `__members__` at all. Probed:
  *"cytnx.Device is a plain Python module object … NOT a class/enum"*,
  *"Device.cpu/Device.cuda are plain Python int …"*, and *"Device has no
  __members__ … while the four real enums … all do"* — all `[PASS]`. This is
  the structural root of C3 (inconsistent modelling) and of why `Device`
  escapes the `export_values()`/cross-type-equality problems the four real
  enums share (P8/C1/C2).

- **P3 — `SymmetryType::Void` (`-99`, the C++ uninitialized sentinel) has no
  Python binding.** `symmetry_py.cpp`'s `py::enum_<SymmetryType>` registers
  only `U`/`Z`/`fPar`/`fNum`; `cytnx.SymType` has no `Void` member. Probed:
  *"SymmetryType::Void (-99 …) has NO Python binding at all: cytnx.SymType has
  no Void member"* `[PASS]`. Low severity — `Void` is an internal "not yet
  initialized" marker, so omitting it from the public Python surface is
  arguably correct — but it *is* a C++-source-vs-Python-wheel member gap and is
  recorded as one.

- **P4 — `Device.Print_Property()` writes to the process's real stdout and
  cannot be captured by Python's `contextlib.redirect_stdout`.** The binding
  (`cytnx.cpp`'s `mdev.def("Print_Property", ...)`) has no
  `py::scoped_ostream_redirect` guard, unlike the print-via-`__repr__` pattern
  documented in `Symmetry.md` P5. Probed by redirecting stdout into a buffer
  and confirming the buffer stays empty: *"Device.Print_Property()'s output is
  NOT capturable via … redirect_stdout … it writes directly to the process's
  real stdout, bypassing Python-level capture"* `[PASS]`. This makes the
  method's output invisible to notebooks, logging redirection, and test
  capture. (A B4-adjacent output-visibility parity gap: the information exists
  but is unreachable through the normal Python stdout channel.)

- **P5 — the Python enum name `SymType` diverges from the C++ name
  `SymmetryType` by more than the N1 casing rule (an outright abbreviation),
  making it an N3 parity finding, not a casing nit.** Recorded in full in
  `Symmetry.md` P6 and cross-referenced here because `SymType` is the unit
  audited by this document. No other enum is renamed at the type level:
  `Type`, `bondType`, and `fermionParity` keep their exact C++ spellings.

- **P6 — `bondType.BD_NONE` (C++ `Bond.hpp`: `BD_NONE = 0`, itself an alias
  for `BD_REG`) has no Python binding.** `bond_py.cpp`'s `py::enum_<bondType>`
  registers `BD_BRA`/`BD_KET`/`BD_REG`/`BD_IN`/`BD_OUT` but never `BD_NONE`.
  Probed: *"bondType.BD_NONE … has NO Python binding …"* `[PASS]`. Since
  `BD_NONE` is merely a third name for the `0`/`BD_REG` value, nothing is lost
  functionally — but it is a real member-set difference between the C++ header
  and the Python wheel, and is recorded as such (cf. P3).

- **P7 — `fermionParity.EVEN` is truthy in Python even though its own declared
  C++ value is `false`/`0`.** In C++, `enum fermionParity : bool { EVEN =
  false, ODD = true }` makes `(bool)EVEN` evaluate to `false` (verified
  independently by compiling and running a standalone C++17 snippet with this
  exact declaration: `(bool)EVEN` prints `0`, `!EVEN` prints `1`). The
  pybind11 enum **wrapper object** defines no `__bool__`, so Python's default
  "every object is truthy" rule applies uniformly, and `bool(EVEN)` is `True`.
  Probed two ways: *"bool(fermionParity.EVEN) is True in Python, even though
  EVEN's own declared/underlying value is 0/false …"* `[PASS]`, and, matching
  how C++ code naturally tests a bool-backed enum, *"…'not EVEN' is False (i.e.
  EVEN reads as truthy), the opposite of what the enum's own declared value
  (0/false) would suggest"* `[PASS]`. This is a genuine cross-language
  behavioral divergence (B4/B5-adjacent): a Python port of C++ code that writes
  `if (parity) { … }` intending "if ODD" will fire for **both** parities.
  Callers must compare explicitly (`parity == fermionParity.ODD` /
  `int(parity) == 1`), never rely on truthiness.

- **P8 — the four real enums are bound with `export_values()`, flooding the
  top-level `cytnx.*` namespace with every member; `Device` is not.**
  `cytnx.cpp:64` and `symmetry_py.cpp:41,45` call `export_values()` on the
  `Type`/`SymType`/`bondType`/`fermionParity` enum bindings, so `cytnx.Double`,
  `cytnx.Z`, `cytnx.U`, `cytnx.BD_BRA`, `cytnx.EVEN` all exist at module top
  level in addition to their qualified `cytnx.Type.Double` / `cytnx.SymType.Z`
  forms. Probed: *"export_values() … re-exports every member as a
  cytnx-module-level name …"* `[PASS]`. `Device` (a submodule, never passed to
  `export_values()`) is exempt: *"Device does NOT get this treatment …
  cytnx.cpu and cytnx.cuda do NOT exist at the top level"* `[PASS]`. Beyond the
  ~30-symbol namespace pollution, this is the mechanism that lets short,
  unqualified names from unrelated enums coexist and collide (feeds C1/C2).

## Consistency findings

- **C1 — cross-enum-type equality and hash collisions: members of the four
  different real enums that share an underlying integer compare *and hash*
  equal, so a dict keyed on one enum silently collides with an unrelated
  enum's same-valued member.** Because the enums are bound without any
  per-type identity in `__eq__`/`__hash__`, `SymType.Z == Type.Void ==
  bondType.BD_REG == fermionParity.EVEN` are all `True` (they all wrap `0`).
  Probed: *"Members of four DIFFERENT enum classes that share the underlying
  int value 0 compare equal to one another …"* `[PASS]`, and the hash
  consequence: *"…they hash equal too, so a dict keyed on one enum type suffers
  a silent key collision … bondType.BD_REG (0) collides with a dict keyed by
  Type.Void (0)"* `[PASS]` (`hash(Type.Void) == hash(bondType.BD_REG)` and
  `bondType.BD_REG in {Type.Void: "type-void"}` is `True`). No single `N`/`B`
  rule names "different enum types must not compare equal," so this is flagged
  as a plain type-safety inconsistency rather than a cited violation; it is the
  most user-visible footgun in this group (a `{dtype: ...}` cache can be
  clobbered by a bond direction of the same integer value).

- **C2 — `export_values()` namespace pollution (P8) is inconsistent and
  unnecessary.** Dumping ~30 short names (`Double`, `Z`, `U`, `BD_BRA`,
  `EVEN`, …) into `cytnx.*` both pollutes the top-level namespace and — given
  C1's cross-type equality — invites hard-to-spot mistakes where an unqualified
  `EVEN` or `U` is used where a differently-typed same-valued constant was
  meant. It is also applied inconsistently: `Device`'s constants are *not*
  top-level, so `cytnx.Double` exists but `cytnx.cpu` does not, giving two
  different access conventions for the same "named constant" concept. Flagged
  informally (no `N`/`B` id covers module-namespace hygiene); recommend
  dropping the top-level re-export and requiring qualified access
  (`cytnx.Type.Double`, `cytnx.Device.cpu`) uniformly.

- **C3 — `Device` is modelled differently from the other four constant sets
  (P2), so the same concept has two shapes.** `Type`/`SymType`/`bondType`/
  `fermionParity` are `py::enum_` types whose members are distinct wrapper
  objects with `.name`/`.value`; `Device` is a submodule whose members are
  bare `int`s with no `.name`/`.value` and no `__members__`. A user cannot
  enumerate devices the way they enumerate dtypes, and `Device.cpu` is `int`
  where `Type.Double` is a `Type` — inconsistent modelling of "a fixed set of
  named codes." Flagged informally; recommend promoting `Device` to a real
  enum for uniformity (see Recommendation), keeping `cpu`/`cuda` as its
  members and re-homing `Ncpus`/`Ngpus`/`getname`/`Print_Property` as
  module-level host-query helpers (they are not device *values*).

- **C4 — `bondType` carries redundant aliases `BD_IN`/`BD_OUT` for
  `BD_KET`/`BD_BRA`, giving two spellings for one direction.** `BD_IN` is
  value-equal to `BD_KET` and `BD_OUT` to `BD_BRA`, but each pair is two
  distinct Python objects for one underlying `int`. Probed: *"BD_IN is a
  value-equal alias for BD_KET, and BD_OUT is a value-equal alias for
  BD_BRA …"* `[PASS]` and *"…but BD_IN and BD_KET are NOT the same Python
  object (value-equal, identity-distinct) …"* `[PASS]`. Two names for one
  concept is an N4-adjacent vocabulary inconsistency (the same "same concept →
  same name" spirit N4 applies to parameters); recommend keeping one spelling
  (`BD_KET`/`BD_BRA`, the physics-standard bra/ket vocabulary already used by
  `Bond`) and removing `BD_IN`/`BD_OUT`.

- **C5 — `Type` is bound without `py::arithmetic()`, so the enum wrapper
  supports no int-like arithmetic** — `Type.Double + Type.Float` raises
  `TypeError`. Probed: *"Type is bound WITHOUT py::arithmetic(): Type.Double +
  Type.Float raises TypeError …"* `[PASS]`. This is the **correct** behavior
  (dtype codes are not meant to be added), recorded as a positive consistency
  observation and a template for the recommended fix to C1: keeping the enums
  arithmetic-free and additionally type-distinct in `__eq__`/`__hash__` would
  close the cross-type collision without changing any legitimate use.

- **C6 — `Device.Print_Property` violates N1 (capitalized callable).** It is a
  method, not a constant, so N1's `snake_case` rule applies (constants/enum
  values are N1-exempt, but callables are not). `Print_Property` →
  `print_property`. `Device.getname` is already lowercase but not
  underscore-separated; per N1 it should be `get_name`. (The enum *value*
  members — `Double`, `BD_KET`, `EVEN`, etc. — are all N1-exempt as constants
  and are **not** renamed.)

## Recommendation

Every live public member of all five units appears below, tagged **keep / add
/ rename / remove**. Enum *value* members are `keep` by default (N1 exempts
constants); the actionable verdicts concentrate on `Device`'s callables (N1),
`bondType`'s redundant aliases (C4), and cross-cutting binding fixes (C1/C2/P7).

### `Type`

| Member | Verdict | Rationale |
|---|---|---|
| `Void`, `ComplexDouble`, `ComplexFloat`, `Double`, `Float`, `Int64`, `Uint64`, `Int32`, `Uint32`, `Int16`, `Uint16`, `Bool` | keep | Dtype code constants, N1-exempt; codes verified against `Type.hpp` (probe). Keep the names verbatim. |
| `name` | keep | Standard pybind11 enum descriptor; useful (`dtype.name`). |
| `value` | keep | Standard pybind11 enum descriptor (`dtype.value` → int code). |

Cross-cutting for `Type`: bind the missing `Type_class` predicates
(`is_complex`/`is_float`/`is_int`/`is_unsigned`/`typeSize`/`getname`) as
methods/free functions so the P1 capability gap is closed; make enum equality
type-distinct (C1).

### `Device`

| Member | Verdict | Rationale |
|---|---|---|
| `cpu` | keep | Device selector constant (N1-exempt). |
| `cuda` | keep | Device selector constant (N1-exempt). |
| `Ncpus` | keep | Host CPU count; keep as a module-level query value. |
| `Ngpus` | keep | Host GPU count; keep as a module-level query value. |
| `getname` | rename | → `get_name` (C6/N1): callable, `snake_case`. |
| `Print_Property` | rename | → `print_property` (C6/N1) **and** fix P4: add a `py::scoped_ostream_redirect` guard so its output is capturable by Python `redirect_stdout`. |

Cross-cutting for `Device`: promote `Device` from a submodule of bare `int`s
to a real `py::enum_` for `cpu`/`cuda` (C3), keeping `Ncpus`/`Ngpus`/
`get_name`/`print_property` as sibling host-query helpers on the same module.

### `SymType`

| Member | Verdict | Rationale |
|---|---|---|
| `U` | keep | Symmetry-kind constant (N1-exempt); code `-1` verified. |
| `Z` | keep | Code `0` verified. |
| `fPar` | keep | Code `-2` verified. |
| `fNum` | keep | Code `-3` verified. |
| `name` / `value` | keep | Standard pybind11 enum descriptors. |

Cross-cutting for `SymType`: rename the **type** `SymType` → `SymmetryType` to
match C++ (P5/N3; see `Symmetry.md` P6). `SymmetryType::Void` stays unbound
(P3) — it is an internal sentinel, intentionally not part of the public
surface.

### `bondType`

| Member | Verdict | Rationale |
|---|---|---|
| `BD_KET` | keep | Bra/ket direction constant (N1-exempt); code `-1` verified. |
| `BD_BRA` | keep | Code `1` verified. |
| `BD_REG` | keep | Undirected/regular bond, code `0` verified. |
| `BD_IN` | remove | Redundant alias of `BD_KET` (C4). Keep the single `BD_KET` spelling. |
| `BD_OUT` | remove | Redundant alias of `BD_BRA` (C4). Keep the single `BD_BRA` spelling. |
| `name` / `value` | keep | Standard pybind11 enum descriptors. |

Cross-cutting for `bondType`: `BD_NONE` stays unbound (P6) — it is a third
name for `BD_REG` and adds nothing.

### `fermionParity`

| Member | Verdict | Rationale |
|---|---|---|
| `EVEN` | keep | Parity constant (N1-exempt); code `0` verified. **Fix P7**: add `__bool__`/`__int__` so `bool(EVEN)` matches the C++ `false`, or (preferably) forbid truthiness entirely so callers must compare `== ODD`. |
| `ODD` | keep | Parity constant; code `1` verified. |
| `name` / `value` | keep | Standard pybind11 enum descriptors. |

### Cross-cutting (all four real enums)

- **Fix C1**: give the enums type-distinct `__eq__`/`__hash__` so
  `Type.Void != bondType.BD_REG` despite the shared integer `0`, closing the
  silent dict-collision footgun. C5 shows this can be done without enabling any
  illegitimate operation.
- **Fix C2/P8**: drop the `export_values()` top-level re-export; require
  qualified access (`cytnx.Type.Double`, `cytnx.Device.cpu`) uniformly across
  all five units.

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`
above, under its recommended name. Enum *value* constants share one docstring
per enum (documenting the set and its codes) rather than 12 near-identical
per-member blocks; the callables and descriptors get their own.

### `Type` (enum type + its dtype value members)

```
The dtype code set for Tensor / UniTensor / Storage.

Members (name : code)
---------------------
Void : 0            ComplexDouble : 1   ComplexFloat : 2
Double : 3          Float : 4           Int64 : 5
Uint64 : 6          Int32 : 7           Uint32 : 8
Int16 : 9           Uint16 : 10         Bool : 11

Each member is a distinct `Type` enum object (NOT a plain int): pass it as the
`dtype=` argument when constructing a container, e.g.
`cytnx.Tensor([2, 2], dtype=cytnx.Type.Double)`. Codes match
`Type_class::Typeinfos` in Type.hpp (the `Type_list` variant index order).

Notes
-----
The recommended API makes enum equality type-distinct: `Type.Void` no longer
compares equal to a same-valued member of an unrelated enum (Consistency
finding C1). `Type` supports no arithmetic (`Type.Double + Type.Float` raises
TypeError, C5) — codes are identifiers, not numbers.
```

### `Type.name`

```
The member's name as a string.

Returns
-------
str
    e.g. `cytnx.Type.Double.name == "Double"`. Standard pybind11 enum
    descriptor, present on every real enum member.
```

### `Type.value`

```
The member's underlying integer code.

Returns
-------
int
    e.g. `cytnx.Type.Double.value == 3`. Standard pybind11 enum descriptor.
```

(`SymType.name`/`SymType.value`, `bondType.name`/`bondType.value`, and
`fermionParity.name`/`fermionParity.value` are the identical pybind11
descriptors and are documented by this same pair — they carry the given
enum's own member name/code.)

### `Device` (module + its device-selector members)

```
The compute-device selector and host-capability queries.

Members
-------
cpu : -1
    The host CPU device. Use as `device=cytnx.Device.cpu`.
cuda : 0
    The first CUDA GPU device (add an offset for additional GPUs where a build
    supports them). Use as `device=cytnx.Device.cuda`.

Notes
-----
In the recommended API `Device` becomes a real enum (Consistency finding C3),
so `cpu`/`cuda` gain `.name`/`.value` and enumerate uniformly with `Type`.
`Ncpus`/`Ngpus`/`get_name`/`print_property` remain module-level host-query
helpers (they describe the host, not a device value).
```

### `Device.Ncpus`

```
Number of CPU cores visible to Cytnx on this host.

Returns
-------
int
    A non-negative count, computed once at import time.
```

### `Device.Ngpus`

```
Number of CUDA GPUs visible to Cytnx on this host.

Returns
-------
int
    A non-negative count (0 on a CPU-only build), computed once at import time.
```

### `get_name` (renamed from `Device.getname`)

```
Return a human-readable name for a device id.

Parameters
----------
id : int
    A device id such as `cytnx.Device.cpu`.

Returns
-------
str
    e.g. `"cytnx device: CPU"` for `cpu`.

Raises
------
RuntimeError
    If `id` is not a valid device id (confirmed by probe: `get_name(99)`
    raises — B4, errors are exceptions, not sentinels).

Notes
-----
Renamed from `getname` (C6/N1 `snake_case`).
```

### `print_property` (renamed from `Device.Print_Property`)

```
Print the host's device properties (CPU/GPU counts and capabilities).

Returns
-------
None

Notes
-----
Renamed from `Print_Property` (C6/N1). The recommended binding wraps the C++
output in a `py::scoped_ostream_redirect` guard so the text is capturable by
Python's `contextlib.redirect_stdout` — the current binding writes directly to
the process's real stdout and is invisible to Python-level capture (Parity
finding P4).
```

### `SymType` (enum type + its members) — recommended name `SymmetryType`

```
The symmetry-kind code returned by `Symmetry.stype()`.

Members (name : code)
---------------------
`U` : -1      U(1) symmetry.
`Z` : 0       Z_n discrete symmetry.
`fPar` : -2   Fermionic parity symmetry.
`fNum` : -3   Fermionic occupation-number symmetry.

Notes
-----
Recommended type name `SymmetryType`, matching C++ (the current Python name
`SymType` is an abbreviation, Parity finding P5 / Symmetry.md P6). The C++
sentinel `Void` (-99, "uninitialized") is intentionally not exposed (P3).
Enum equality is type-distinct in the recommended API (C1).
```

### `bondType` (enum type + its members)

```
The bond-direction code carried by every Bond.

Members (name : code)
---------------------
BD_KET : -1   An incoming / ket bond leg.
BD_BRA : 1    An outgoing / bra bond leg.
BD_REG : 0    An undirected (regular) bond leg.

Notes
-----
The recommended API keeps only the bra/ket vocabulary and removes the
redundant `BD_IN` (= `BD_KET`) and `BD_OUT` (= `BD_BRA`) aliases (Consistency
finding C4). The C++ `BD_NONE` (a third name for `BD_REG`) stays unbound (P6).
Enum equality is type-distinct (C1).
```

### `fermionParity` (enum type + its members)

```
The two-valued fermionic parity code returned by
`Symmetry.get_fermion_parity()`.

Members (name : code)
---------------------
`EVEN` : 0    Even parity.
`ODD` : 1     Odd parity.

Notes
-----
IMPORTANT (Parity finding P7): in the current wheel `bool(EVEN)` is `True`
even though EVEN's underlying value is 0/false, because the pybind11 wrapper
defines no `__bool__`. Never test parity with truthiness (`if parity:` fires
for BOTH values) — compare explicitly, `parity == cytnx.fermionParity.ODD`.
The recommended API fixes this so truthiness matches the C++ `bool`-backed
value (or forbids it outright). Enum equality is type-distinct (C1).
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.
Enum *value* constants are N1-exempt and unchanged; only the rows below change.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Device.getname` | `Device.get_name` | N1 casing (C6) |
| `Device.Print_Property` | `Device.print_property` (+ stdout-capture fix) | N1 casing (C6) + P4 |
| `SymType` *(type name)* | `SymmetryType` | N3 name divergence (P5; Symmetry.md P6) |
| `bondType.BD_IN` | *(removed)* → use `BD_KET` | redundant alias (C4) |
| `bondType.BD_OUT` | *(removed)* → use `BD_BRA` | redundant alias (C4) |
| `Device` *(submodule of int)* | `Device` *(real enum)* | uniform modelling (C3/P2) |
| four enums with `export_values()` | qualified access only | namespace hygiene (C2/P8) |
| four enums, cross-type-equal | type-distinct `__eq__`/`__hash__` | collision fix (C1) |
| `fermionParity.EVEN`/`ODD` truthiness | `__bool__` matching the C++ value | cross-language behavior (P7) |

Every other public member of the five units — all of `Type`'s 12 dtype codes
plus `name`/`value`; `Device.cpu`/`cuda`/`Ncpus`/`Ngpus`; `SymType`'s
`U`/`Z`/`fPar`/`fNum` plus `name`/`value`; `bondType.BD_KET`/`BD_BRA`/`BD_REG`
plus `name`/`value`; `fermionParity.EVEN`/`ODD` plus `name`/`value` — keeps
both its current name and its integer code. (`BD_IN`/`BD_OUT` are removed, and
`EVEN`/`ODD` keep their names but gain the P7 truthiness fix — they appear in
the table above, not in this "unchanged" list, and are excluded here
deliberately, not by oversight.)
