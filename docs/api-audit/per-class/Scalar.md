# `Scalar` — API audit

`Scalar` is Cytnx's dtype-tagged single-number value type: a tagged union that
wraps one numeric value together with a `cytnx.Type` dtype code, so that a lone
element pulled out of a `Tensor`/`Storage` (or a coefficient fed back into one)
carries its dtype the way a tensor does. Internally it is a thin handle
(`Scalar::_impl`) over a `Scalar_base` subtype — one concrete class per dtype
(`DoubleScalar`, `ComplexDoubleScalar`, `Int64Scalar`, …) — that dispatches all
arithmetic, comparison, casting, and part-extraction to the active subtype.
This document audits the 12 public members of the live `cytnx.Scalar` class
(installed `cytnx==1.1.0` wheel).

Ground truth for behavior is `docs/api-audit/probes/Scalar.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/Scalar.py`; all 43 assertions `[PASS]`, exit 0). Ground
truth for static signatures is `cytnx_src/include/backend/Scalar.hpp` (the
public `Scalar` wrapper at lines 2515–3354, the `Scalar_base` interface and its
eleven per-dtype subtypes above it, and the free `operator`/`abs`/`sqrt`/
`complex128` declarations below it), `cytnx_src/src/backend/Scalar.cpp` (the
free-operator and factory implementations), and `cytnx_src/pybind/scalar_py.cpp`
(the pybind11 binding — authoritative for the Python-visible surface, and
notably for the numpy-scalar operator macros `FOR_EACH_NUMPY_TYPE`/`_RTYPE`/
`_ITYPE` and the hand-written `__float__`/`__int__`/`__complex__` conversion
lambdas). The dtype-code ordering used throughout (`ComplexDouble = 1` …
`Bool = 11`, lower code = more general dtype) is documented in `enums.md`
(the `Type` unit).

## Inventory

C++ signatures are read from `Scalar.hpp`/`Scalar.cpp`; Python signatures are
the effective pybind-visible signature, cross-checked against
`tools/member_inventory.py Scalar`.

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `abs` | `Scalar abs() const` *(pure; for complex returns the real magnitude — computes `iabs()` on a copy then `real()`)* | `abs() -> Scalar` |
| `astype` | `Scalar astype(const unsigned int &dtype) const` *(cannot go complex→real; raises, see P4/`Scalar.hpp:2807` `@attention`)* | `astype(dtype: int) -> Scalar` |
| `conj` | `Scalar conj() const` *(pure; real subtypes' `conj_()` is a no-op, `Scalar.hpp:1092` etc.)* | `conj() -> Scalar` |
| `dtype` | `int dtype() const` *(returns the raw `_impl->_dtype` code)* | `dtype() -> int` — a bare `int` code, not a `Type` enum member (C4) |
| `iabs` | `void iabs()` *(in-place; mutates the receiver)* | `iabs() -> None` — `i`-prefixed in-place variant of `abs` (C1/N2) |
| `imag` | `Scalar imag() const` *(dispatches to the subtype's `get_imag()`; **real subtypes raise** `"real type Scalar does not have imag part!"`, `Scalar.hpp:1095`)* | `imag() -> Scalar` — partial: raises `RuntimeError` on a real Scalar (C3, P4) |
| `isqrt` | `void isqrt()` *(in-place)* | `isqrt() -> None` — `i`-prefixed in-place variant of `sqrt` (C1/N2) |
| `maxval` (static) | `static Scalar maxval(const unsigned int &dtype)` *(real dtypes only; complex raises)* | `maxval(dtype: int) -> Scalar` (staticmethod) |
| `minval` (static) | `static Scalar minval(const unsigned int &dtype)` *(real dtypes only; complex raises)* | `minval(dtype: int) -> Scalar` (staticmethod) |
| `print` | `void print() const` *(writes value + dtype label to `std::cout`)* | `print() -> None` — side-effecting; shadows the Python builtin `print` (C5) |
| `real` | `Scalar real() const` *(total; real subtypes' `get_real()` returns `copy()`, `Scalar.hpp:1093`)* | `real() -> Scalar` — returns an independent copy (B2) |
| `sqrt` | `Scalar sqrt() const` *(pure; stays real — `sqrt(-1.0)` is NaN, no real→complex promotion)* | `sqrt() -> Scalar` |
| *(operators)* `operator+ - * /`, `+= -= *= /=`, `== != < <= > >=` | free `operator+(const Scalar&, const Scalar&)` … (`Scalar.hpp:3356-3436`); member `operator+=`/`operator<`… | bound: `__add__`/`__sub__`/`__mul__`/`__truediv__`, `__iadd__`/…, `__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/`__ge__`, plus per-numpy-dtype `__r*__` reflected forms (see P2) |
| *(conversions)* `explicit operator cytnx_double/…`; free `complex128(Scalar)`/`complex64(Scalar)` | `explicit operator cytnx_double() const` … (`Scalar.hpp:2859-2884`) | `__float__`/`__int__`/`__complex__` — hand-written lambdas (`scalar_py.cpp:158-165`); `__complex__` is built from `real()`+`imag()` (P4) |
| *(C++-only)* named ops `radd`/`rsub`/`rmul`/`rdiv`/`less`/`leq`/`greater`/`geq`/`eq` | templated `Scalar radd(const T&) const` … (`Scalar.hpp:3184-3162`) | **absent from Python** — only the operator forms are bound (see P3) |
| *(C++-only)* 2-arg constructor | `template<class T> Scalar(const T &in, const unsigned int &dtype)` (`Scalar.hpp:2636`) | **not exposed** — only single-argument `py::init<>` overloads are registered (see P1) |
| *(C++-only)* `iabs`/`isqrt` mutating `conj_`/`Sproxy` internals | `void conj_()` (subtype-level), `struct Sproxy` | **`conj_` unbound**; `Sproxy` is the `Storage`-element proxy, not a `Scalar` member |

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/Scalar.py`.

- **P1 — the C++ 2-arg constructor `Scalar(value, dtype)` is unbound, so
  Python cannot choose a dtype at construction; worse, a Python-native `int`
  or `bool` is silently constructed as `Uint64`.** `scalar_py.cpp:26-103`
  registers only single-argument inits (one per fundamental C++ type plus the
  `numpy_scalar<T>` shims); the templated `Scalar(const T&, const unsigned
  int&)` 2-arg constructor (`Scalar.hpp:2636`) is never bound. Probed:
  *"the C++ 2-arg constructor Scalar(value, dtype) is NOT exposed to Python …
  Scalar(3.0, Type.Double) raises TypeError"* `[PASS]`. Separately, pybind
  overload resolution over the single-arg inits walks the `SupportsInt`
  overloads in registration order, and `uint64` is registered before `int64`
  (`scalar_py.cpp:31-32`), so a plain Python `int` matches `uint64` first:
  *"Scalar(2) (a Python int) is tagged Uint64 (code 6), NOT Int64 (5)"*
  `[PASS]`; and because `bool` is a subclass of `int`, it too matches the
  `uint64` overload before the explicit `bool` overload at
  `scalar_py.cpp:100`: *"Scalar(True) … is ALSO tagged Uint64 (6), NOT Bool
  (11) … a Python caller cannot construct a Bool-dtype Scalar via
  Scalar(True)"* `[PASS]`. The only way a Python caller can pin down a
  specific real dtype at construction is to pass a numpy-typed scalar
  (*"Scalar(np.float32(2)) is Float(4), Scalar(np.int64(5)) is Int64(5)"*
  `[PASS]`) or to construct-then-`astype`. Net effect: the C++ ability to say
  "make a Scalar of exactly this dtype" is unreachable from Python, and the
  natural Python idioms `Scalar(2)`/`Scalar(True)` produce a surprising
  unsigned dtype.

- **P2 — reflected `number + Scalar` raises `TypeError`: a Scalar is not
  commutative-usable with a plain number on its left, even though `Scalar +
  number` works.** For the left-operand case the binding relies on implicit
  conversion of the right operand into a `Scalar` (`py::self + py::self` plus
  `py::implicitly_convertible<double, Scalar>` etc., `scalar_py.cpp:144-147,
  343-353`), so *"Scalar(3.0)+2.0, Scalar(3.0)+2, and Scalar(3.0)+np.float32(2)
  all succeed and stay Double(3)"* `[PASS]`. But the reflected `__radd__`/
  `__rsub__`/… are generated only by the `FOR_EACH_NUMPY_RTYPE` macro, which
  accepts *numpy* scalars — not Python floats — on the left, and even those
  are intercepted by numpy before pybind's reflected overload can fire (the
  macro carries a source comment to this effect, `scalar_py.cpp:225-226`).
  Probed: *"2.0 + Scalar(3.0) … raises TypeError"* `[PASS]` and *"np.float64(2)
  + Scalar(3.0) raises TypeError too"* `[PASS]`. In C++ the free
  `operator+(const Scalar&, const Scalar&)` combined with implicit
  number→Scalar conversion makes `2.0 + s` compile and run; in Python the same
  expression is an error. Mixed number/Scalar arithmetic is therefore
  order-sensitive in Python only.

- **P3 — the named arithmetic/comparison methods C++ exposes
  (`radd`/`rsub`/`rmul`/`rdiv`, `less`/`leq`/`greater`/`geq`/`eq`) are entirely
  unbound in Python; only the operator forms are reachable.** `scalar_py.cpp`
  binds `py::self + py::self` … and `py::self == py::self` … but never
  `.def("radd", …)`/`.def("eq", …)`, so the templated C++ named methods
  (`Scalar.hpp:3184-3162`) have no Python attribute. Probed: *"there is no
  .add()/.sub()/.mul()/.div() (the C++ radd/rsub/rmul/rdiv are unbound)"*
  `[PASS]` (also checks `radd`/`rsub`/`rmul`/`rdiv` by name). Consequently
  `00-methodology.md`'s B5 ("operator overloads are equivalent to their
  named-method counterparts") has no named counterpart to compare against on
  `Scalar` — the operators *are* the only surface. Informational (no
  functionality is lost, since the operators cover the same math), recorded as
  a C++-vs-Python surface gap, the same shape as `enums.md` P1 for `Type`'s
  static utilities.

- **P4 — headline bug: `complex()` of a *real* Scalar raises `RuntimeError`,
  even though the equivalent C++ conversion `complex128(realScalar)`
  succeeds — because the Python `__complex__` lambda is (mis)built from
  `real()` + `imag()`, and `imag()` is undefined for real subtypes.** The
  binding at `scalar_py.cpp:160-165` computes `complex(re, im)` with
  `re = (double)s.real()` and `im = (double)s.imag()`; for a real
  Scalar, `s.imag()` dispatches to `DoubleScalar::get_imag()`, which raises
  `"real type Scalar does not have imag part!"` (`Scalar.hpp:1095`). The C++
  path never touches `imag()`: `cytnx::complex128(const Scalar&)` goes straight
  through the subtype's `to_cytnx_complex128()`, which for `DoubleScalar`
  returns the value with a zero imaginary part (`Scalar.hpp:950`). Probed
  directly: *"complex() of a *real* Scalar RAISES RuntimeError … so
  complex(Scalar(3.0)) fails instead of returning (3+0j)"* `[PASS]`, and the
  underlying cause is confirmed independently: *"imag() of a real Scalar RAISES
  RuntimeError ('real type Scalar does not have imag part')"* `[PASS]`. This is
  a genuine C++-vs-Python divergence *and* a plain bug — `complex(3.0-valued
  Scalar)` should be `(3+0j)`. For contrast, the deliberately-partial real
  conversions behave correctly and symmetrically: *"float() of a complex Scalar
  RAISES RuntimeError"* and *"int() of a complex Scalar likewise RAISES
  RuntimeError"* (both `[PASS]`) — refusing a lossy complex→real cast is
  defensible; refusing a lossless real→complex cast is not.

## Consistency findings

- **C1 — violates N2: the in-place methods are `i`-prefixed (`iabs`, `isqrt`)
  instead of using the trailing-`_` convention, so the pure/in-place pairs are
  spelled `abs`/`iabs` and `sqrt`/`isqrt`.** Per N2 ("in-place variants use a
  trailing `_`, and every in-place method has a pure counterpart with the same
  base name"), the recommended pairs are `abs()`/`abs_()` and
  `sqrt()`/`sqrt_()`. Probed that the current forms really are in-place and
  return `None`: *"iabs() is in-place and returns None … the receiver is 4.0"*
  and *"isqrt() is in-place … leaves the receiver at 3.0"* — both `[PASS]`,
  each note flagging the `i`-prefix-not-`_`-suffix issue. (The pure `abs`/`sqrt`
  counterparts exist and are correctly pure — *"abs() returns a new Scalar …
  Scalar(-2.0).abs()==2.0"*, *"sqrt() returns a new Scalar … 4.0→2.0"*, both
  `[PASS]` — so this is purely a naming-convention fix, not a missing-method
  fix.)

- **C2 — violates B5: the in-place `+=`/`-=`/`*=`/`/=` operators are *not*
  equivalent to the binary `+`/`-`/`*`/`/` for a real receiver and a complex
  operand — the binary form promotes, the in-place form raises.** The binary
  `+` up-casts the lower-precision operand before adding, so real + complex
  succeeds: *"Double + ComplexDouble promotes to ComplexDouble (code 1)"*
  `[PASS]`. But `__iadd__` calls the real subtype's `iadd(complex)` directly
  with no promotion, hitting `DoubleScalar::iadd`'s `"Cannot operate real and
  complex values"` guard (`Scalar.hpp:976`): *"in-place real += complex RAISES
  RuntimeError … whereas the binary '+' promotes the real operand up to complex
  first"* `[PASS]`. Per B5 ("an operator that behaves differently from its
  named/binary counterpart for the same inputs is a consistency finding"), the
  `+=`/`+` divergence is a B5 violation (and, since it concerns dtype
  promotion, also a B3-adjacent inconsistency: promotion is applied on one path
  and not the other for identical operands). Recommend `+=` promote-in-place
  (rebinding the receiver's `_impl` to the widened subtype) exactly as `+`
  does, so `s += complex` matches `s = s + complex`.

- **C3 — `real()`/`conj()` are total but `imag()` is partial, and that
  asymmetry is the root cause of P4.** `real()` and `conj()` are defined for
  every subtype (real subtypes return a copy / act as a no-op respectively),
  but `imag()` raises for real subtypes rather than returning a zero of the
  same dtype: *"real() of a real Scalar is defined … but imag() of a real
  Scalar RAISES RuntimeError"* `[PASS]`. No `N`/`B` rule in `00-methodology.md`
  directly governs "a getter is defined for some subtypes and raises for
  others," so this is flagged as a plain internal-semantics inconsistency
  (same treatment as `Symmetry.md`'s C5) — but it is not merely cosmetic: it is
  exactly what makes `complex()` fail on real scalars (P4). Recommend `imag()`
  on a real Scalar return `Scalar(0, <same real dtype>)` (matching numpy, where
  `x.imag` of a real array is a zero array), which simultaneously fixes P4.

- **C4 — `dtype()` returns a bare `int` code rather than a `cytnx.Type` enum
  member.** Probed: *"dtype() returns the integer Type code (not a Type enum
  member): Scalar(3.0).dtype() == 3 == int(Type.Double)"* `[PASS]`. It happens
  to compare equal to `Type.Double` only because `Type`'s enum members compare
  equal to their underlying `int` (see `enums.md` C1, the very cross-type
  integer-equality footgun documented there) — i.e. `Scalar.dtype()` leans on
  the same loose equality that `enums.md` recommends removing. If `enums.md`'s
  C1 fix (type-distinct enum equality) lands, `s.dtype() == Type.Double` would
  silently become `False`. Flagged informally (no `N`/`B` id covers a getter's
  return *type*); recommend `dtype()` return the `Type` enum member (or the
  recommended API's `Type`-typed value) for forward-compatibility with the
  `enums.md` C1 fix, and provide a `dtype_str()` convenience mirroring
  `Tensor.dtype_str()` (the only Python route to a dtype name today, per
  `enums.md` P1).

- **C5 — `print` shadows the Python builtin and is a pure stdout side effect.**
  Probed: *"print() returns None and writes the value + dtype label to
  stdout"* `[PASS]`. It is already `snake_case` (a single lowercase word), so
  N1 is satisfied, and unlike `Symmetry`'s broken `__repr__` (`Symmetry.md`
  P5) `Scalar.__repr__` correctly *returns* a useful string built via an
  `ostringstream` (*"Scalar.__repr__ RETURNS a useful non-empty string …
  contains the value and the dtype name"* `[PASS]`). So `print()` is redundant
  with `print(repr(s))`. Flagged informally (no `N`/`B` id covers
  builtin-shadowing); recommend keeping it (it is harmless and matches the C++
  name) but documenting that `repr()`/`str()` already give the same text
  without shadowing the builtin.

- **C6 — positive observation (B3): binary promotion is uniform and correct —
  it widens to the lower `Type` code (the more general dtype).** Recorded
  explicitly as the template for the C2 fix. `Double(3)+ComplexDouble(1)→
  ComplexDouble(1)`, `Double(3)+Float(4)→Double(3)`, `Float(4)+Int64(5)→
  Float(4)`, and comparison across dtypes promotes first (`Double < Int64` is
  `True`) — all `[PASS]`. This is the one promotion rule the class already gets
  right on the binary path; C2 asks only that the in-place path match it.

## Recommendation

Every one of the 12 live public members of `cytnx.Scalar` appears below, tagged
**keep / add / rename / remove**. Four additional informational rows cover the
unbound 2-arg constructor (P1), the reflected number-on-left operators (P2), the
unbound named ops (P3), and the `__complex__` bug (P4/C3), none of which are
attributes matched by `validate_doc.py`'s `public_members()` but each of which
carries a concrete fix.

| Member | Verdict | Rationale |
|---|---|---|
| `abs` | keep | Pure magnitude; already `snake_case`. Pair with the renamed in-place `abs_` (C1). For complex input returns the real (Double) magnitude — document. |
| `astype` | keep | Correctly named; document that complex→real raises (P4/B4) and directs callers to `real()`/`imag()`. |
| `conj` | keep | Pure, `snake_case`, correct (no-op on real subtypes). Its C++ in-place `conj_` is unbound, which is fine — N2 requires every in-place method have a pure counterpart, not vice versa. |
| `dtype` | keep | Keep the name; change the return to a `Type` enum member and add a `dtype_str()` convenience (C4). |
| `iabs` | rename | → `abs_` (C1/N2): in-place variant should use the trailing-`_` convention, pairing with pure `abs`, not the `i`-prefix. |
| `imag` | keep | Keep the name; **fix the partiality** (C3): `imag()` of a real Scalar must return a zero of the same real dtype instead of raising — this also fixes the `complex()` bug (P4). |
| `isqrt` | rename | → `sqrt_` (C1/N2): in-place variant of pure `sqrt`. |
| `maxval` | keep | Static extremal factory; already `snake_case`. Document that complex dtypes raise (B4). |
| `minval` | keep | Static extremal factory; already `snake_case`. Document that complex dtypes raise (B4). |
| `print` | keep | Already `snake_case` and matches the C++ name; document that `repr()`/`str()` return the same text without shadowing the builtin (C5). |
| `real` | keep | Pure, total, returns an independent copy (B2). Correct. |
| `sqrt` | keep | Pure; pair with the renamed in-place `sqrt_` (C1). Document that it stays real (`sqrt(-1.0)` is NaN). |
| *(fix)* `__complex__` | rename | Fix P4: build the result via the subtype's `to_cytnx_complex128()` (as C++'s `complex128(Scalar)` does), not via `real()`+`imag()`, so `complex(realScalar)` returns `re+0j` instead of raising. |
| *(fix)* reflected ops `__radd__`/`__rsub__`/`__rmul__`/`__rtruediv__` | add | Fix P2: bind reflected operators for Python-native `int`/`float`/`complex` on the left so `2.0 + Scalar(3.0)` works, restoring commutativity with numbers. |
| *(new)* 2-arg constructor Scalar(value, dtype) | add | Fix P1: bind the C++ 2-arg constructor so Python can pick a dtype directly; also reorder/repair the single-arg overloads so Scalar(2) is Int64 (not Uint64) and Scalar(True) is Bool (not Uint64). See the "2-arg constructor" docstring below. |
| *(C++-only)* named ops `radd`/`less`/`eq`/… | remove | Keep them C++-only (P3): the operator forms already cover the surface from Python; binding the named forms would only duplicate the operators. Recorded as an explicit "do not add," not an oversight. |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`
above, under its recommended name.

### `abs`

```
Return the absolute value (magnitude) of this Scalar as a new Scalar.

Returns
-------
Scalar
    |self|. For a real Scalar, the same real dtype; for a complex Scalar, the
    real (Double) magnitude, e.g. Scalar(3+4j).abs() == 5.0 with dtype Double
    (confirmed by probe).

Notes
-----
Pure (does not mutate self); the in-place counterpart is `abs_` (renamed from
`iabs`, C1/N2).
```

### `astype`

```
Return a copy of this Scalar converted to another dtype.

Parameters
----------
dtype : Type
    The target dtype code (a cytnx.Type member, e.g. Type.Float).

Returns
-------
Scalar
    A new Scalar holding self's value in the requested dtype.

Raises
------
RuntimeError
    If converting a complex Scalar to a real dtype (confirmed by probe:
    Scalar(1+1j).astype(Type.Double) raises). Use `real()`/`imag()` to extract
    a real part instead.

Notes
-----
Pure; returns a new object.
```

### `conj`

```
Return the complex conjugate of this Scalar as a new Scalar.

Returns
-------
Scalar
    conj(self): (a - bi) for a complex Scalar (confirmed by probe:
    Scalar(1+2j).conj() == (1-2j)); an unchanged copy for a real Scalar.

Notes
-----
Pure (does not mutate self, confirmed by probe). There is no Python-bound
in-place `conj_` (the C++ `conj_` is unbound); use `x = x.conj()`.
```

### `dtype`

```
Return this Scalar's dtype.

Returns
-------
Type
    The dtype as a cytnx.Type enum member (recommended API; the current wheel
    returns the bare integer code, e.g. Scalar(3.0).dtype() == 3 == Type.Double
    — see Consistency finding C4).

Notes
-----
See `dtype_str()` for a human-readable name.
```

### `abs_` (renamed from `iabs`)

```
Set this Scalar to its absolute value, in place.

Returns
-------
None
    Mutates the receiver (confirmed by probe: Scalar(-4.0).abs_() leaves the
    receiver at 4.0 and returns None).

Notes
-----
In-place counterpart of the pure `abs`. Renamed from `iabs` (C1/N2): in-place
methods use a trailing `_`, not an `i` prefix.
```

### `imag`

```
Return the imaginary part of this Scalar as a new (real) Scalar.

Returns
-------
Scalar
    Im(self) as a real (Double) Scalar for a complex Scalar (confirmed by
    probe: Scalar(1+1j).imag() == 1.0, dtype Double). For a real Scalar the
    recommended API returns a zero of the same real dtype.

Notes
-----
FIX (Consistency finding C3 / Parity finding P4): in the current wheel `imag()`
of a real Scalar RAISES RuntimeError ("real type Scalar does not have imag
part"), which in turn makes `complex()` of a real Scalar raise. The recommended
API returns 0 (matching numpy's `x.imag` on real data) so both work.
```

### `sqrt_` (renamed from `isqrt`)

```
Set this Scalar to its square root, in place.

Returns
-------
None
    Mutates the receiver (confirmed by probe: Scalar(9.0).sqrt_() leaves the
    receiver at 3.0).

Notes
-----
In-place counterpart of the pure `sqrt`. Renamed from `isqrt` (C1/N2). Stays in
the receiver's dtype — a negative real receiver becomes NaN, not complex.
```

### `maxval` (static)

```
Return the largest representable value of a given real dtype, as a Scalar.

Parameters
----------
dtype : Type
    A real dtype code (e.g. Type.Int16).

Returns
-------
Scalar
    The dtype's maximum, e.g. maxval(Type.Int16) == 32767 (confirmed by probe).

Raises
------
RuntimeError
    For a complex dtype (confirmed by probe: maxval(Type.ComplexDouble) raises)
    — extremal values are defined for real dtypes only (B4).
```

### `minval` (static)

```
Return the smallest representable value of a given real dtype, as a Scalar.

Parameters
----------
dtype : Type
    A real dtype code (e.g. Type.Int16).

Returns
-------
Scalar
    The dtype's minimum, e.g. minval(Type.Int16) == -32768 (confirmed by
    probe).

Raises
------
RuntimeError
    For a complex dtype (B4), the same as `maxval`.
```

### `print`

```
Print this Scalar's value and dtype label to standard output.

Returns
-------
None
    Side-effecting; writes e.g. "< 3 > Scalar dtype: [Double (Float64)]".

Notes
-----
`repr(s)`/`str(s)` return the same information as a string without shadowing
the Python builtin `print` (Consistency finding C5).
```

### `real`

```
Return the real part of this Scalar as a new (real) Scalar.

Returns
-------
Scalar
    Re(self) as a real (Double) Scalar for a complex Scalar; an independent
    copy of the value for a real Scalar.

Notes
-----
Returns an INDEPENDENT copy, not a view (confirmed by probe: mutating the
result of Scalar(5.0).real() leaves the source at 5.0 — B2).
```

### `sqrt`

```
Return the square root of this Scalar as a new Scalar.

Returns
-------
Scalar
    sqrt(self) (confirmed by probe: Scalar(4.0).sqrt() == 2.0).

Notes
-----
Pure; the in-place counterpart is `sqrt_` (renamed from `isqrt`, C1). Stays in
self's dtype: `Scalar(-1.0).sqrt()` is NaN (no automatic real→complex
promotion, confirmed by probe).
```

### `__complex__` (recommended fix, P4)

```
complex(scalar) -> complex
    Convert this Scalar to a Python complex.

Notes
-----
Fixes Parity finding P4: the current binding builds the result from `real()`
and `imag()`, and `imag()` raises for real scalars, so `complex(Scalar(3.0))`
raises today. The recommended binding uses the subtype's `to_cytnx_complex128()`
(as C++'s `complex128(Scalar)` already does) so `complex(Scalar(3.0))` returns
`(3+0j)`. `float()`/`int()` keep their current, correct behavior (raising on a
complex Scalar rather than silently dropping the imaginary part, confirmed by
probe).
```

### reflected operators `__radd__` / `__rsub__` / `__rmul__` / `__rtruediv__` (new, P2)

```
number OP scalar -> Scalar
    Reflected arithmetic so a Python number on the LEFT of a Scalar works.

Notes
-----
Fixes Parity finding P2: today `Scalar(3.0) + 2.0` works but `2.0 + Scalar(3.0)`
raises TypeError (the reflected forms are bound only for numpy-scalar operands,
and numpy intercepts those first). The recommended binding adds reflected
operators for Python-native int/float/complex so mixed number/Scalar arithmetic
is order-independent, with the same promotion (widen to the lower Type code) as
the forward operators.
```

### 2-arg constructor `Scalar(value, dtype)` (new, P1)

```
Scalar(value, dtype) -> Scalar
    Construct a Scalar holding `value` in exactly `dtype`.

Parameters
----------
value : int | float | complex
    The numeric value.
dtype : Type
    The dtype code to store it as (e.g. Type.Bool, Type.Int64).

Notes
-----
Fixes Parity finding P1: the C++ 2-arg constructor is unbound today, so Python
callers cannot pick a dtype at construction and must round-trip through numpy
scalars or `astype()`. The fix also repairs single-argument overload
resolution so `Scalar(2)` is Int64 (not Uint64) and `Scalar(True)` is Bool (not
Uint64).
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `iabs` | `abs_` | N2 in-place suffix (C1) |
| `isqrt` | `sqrt_` | N2 in-place suffix (C1) |
| `imag` (raises on real Scalar) | `imag` (returns 0 on real Scalar) | C3 / P4 fix |
| `dtype` (returns `int`) | `dtype` (returns `Type`) + new `dtype_str` | C4 |
| `__complex__` (via real()+imag(), raises on real) | `__complex__` (via to_cytnx_complex128) | P4 |
| `__radd__`/`__rsub__`/`__rmul__`/`__rtruediv__` (numpy-only, raise for Python numbers) | reflected ops for Python int/float/complex | P2 |
| `+=`/`-=`/`*=`/`/=` (no promotion; real += complex raises) | promote-in-place to match `+`/`-`/`*`/`/` | B5 (C2) |
| *(none — not bound)* `Scalar(value, dtype)` | 2-arg constructor (new Python binding) + fixed single-arg dtype resolution | P1 |
| *(C++-only)* `radd`/`rsub`/`rmul`/`rdiv`/`less`/`leq`/`greater`/`geq`/`eq` | *(stay C++-only)* — operators already cover them | P3 |

Every other public member of `Scalar` — `abs`, `astype`, `conj`, `maxval`,
`minval`, `print`, `real`, `sqrt` — keeps both its current name and current
behavior unchanged (with the documentation notes above). `dtype` and `imag`
keep their names but change behavior (C4, C3/P4) — they appear in the table
above, not in this "unchanged" list, and are excluded here deliberately, not by
oversight.
