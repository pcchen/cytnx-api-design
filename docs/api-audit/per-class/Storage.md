# `Storage` — API audit

`Storage` is Cytnx's lowest-level container: a single contiguous, typed,
device-resident 1-D memory buffer (the raw backing store that a `Tensor`'s
multi-dimensional view sits on top of). It owns a dtype (one of `cytnx.Type`,
see `enums.md`), a device (one of `cytnx.Device`, see `enums.md`), a `size`
(element count) and a `capacity` (allocated slots, `std::vector`-style), and it
exposes element get/set, whole-buffer generators (`fill`/`set_zeros`),
growth (`append`/`resize`), dtype/device conversion (`astype`/`to`), and
Python-interop exporters (`numpy`/`pylist`). This document audits the 38 public
members of the live `cytnx.Storage` class (installed `cytnx==1.1.0` wheel).

Ground truth for behavior is `docs/api-audit/probes/Storage.py`, executed
against `./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/Storage.py`; all 39 assertions `[PASS]`, exit 0).
Ground truth for static signatures is `cytnx_src/include/backend/Storage.hpp`
(the `Storage` value-type wrapper and its `Storage_base` /
`StorageImplementation<DType>` implementation hierarchy),
`cytnx_src/src/backend/Storage.cpp` and
`cytnx_src/src/backend/StorageImplementation.cpp` (authoritative for what
`operator==`, `resize`, `real`/`imag` really compute and which inputs raise),
`cytnx_src/pybind/storage_py.cpp` (the pybind11 binding — authoritative for the
Python-visible call signature, and notably for the `numpy`/`astype_different_type`/
`to_different_device` lambdas that have no direct C++ counterpart), and
`cytnx_src/cytnx/Storage_conti.py` (the Python-side augmentation layer, which
adds `astype`, `to`, `pylist`, and `__iter__` on top of pybind — the
`astype`/`to` self-return behavior in P2/P3 lives here, not in C++).

## Inventory

C++ signatures are read from `Storage.hpp`/`Storage.cpp`/`StorageImplementation.cpp`;
Python signatures are the effective pybind-visible signature (or the
`Storage_conti.py` override where one exists), cross-checked against
`tools/member_inventory.py Storage`.

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `Init` | `void Init(const unsigned long long& size, const unsigned int& dtype=Type.Double, int device=-1, const bool& init_zero=true)` | `Init(size: int, dtype: int=3, device: int=-1, init_zero: bool=True) -> None` — re-initializes the same wrapper in place |
| `Load` (static) | `static Storage Load(const string&)`; `static Storage Load(const char*)` | `Load(fname: str) -> Storage` (staticmethod) |
| `Save` | `void Save(const string&) const`; `void Save(const char*) const` | `Save(fname: str) -> None` — appends the `.cyst` extension |
| `Fromfile` (static) | `static Storage Fromfile(const string&, const unsigned int& dtype, const cytnx_int64& count=-1)` | `Fromfile(fname: str, dtype: int, count: int=-1) -> Storage` (staticmethod) — raw-binary loader |
| `Tofile` | `void Tofile(const string&) const` | `Tofile(fname: str) -> None` — raw-binary writer (no `.cyst` header) |
| `from_pylist` (static) | `template<class T> static Storage from_vector(const vector<T>&, int device=-1)` | `from_pylist(pylist: Sequence, device: int=-1) -> Storage` (staticmethod) — dtype inferred by pybind overload order (see C6) |
| `astype` | `Storage astype(const unsigned int& new_type) const` *(pybind binding commented out; reimplemented in `Storage_conti.py`)* | `astype(new_type: int) -> Storage` — **returns `self` if `new_type==dtype()`**, else a new Storage (P2) |
| `astype_different_type` | *(none — pybind-only lambda)* | `astype_different_type(new_type: int) -> Storage` — internal helper; **raises** if `new_type==dtype()` (P4) |
| `to` | `Storage to(const int& device)` *(reimplemented in `Storage_conti.py`)* | `to(device: int) -> Storage` — **returns `self` if `device==device()`**, else a new Storage (P3) |
| `to_` | `void to_(const int& device)` | `to_(device: int) -> None` — in-place device move |
| `to_different_device` | *(none — pybind-only lambda)* | `to_different_device(device: int) -> Storage` — internal helper; **raises** if `device==device()` (P4) |
| `clone` | `Storage clone() const` | `clone() -> Storage`; also bound as `__copy__`/`__deepcopy__` (deep copy, P-none/probe) |
| `dtype` | `unsigned int dtype() const` | `dtype() -> int` — one of `cytnx.Type.*` |
| `dtype_str` | `const string dtype_str() const` | `dtype_str() -> str` — e.g. `"Double (Float64)"`, `"Int64"` (integer types carry no parenthetical) |
| `device` | `int device() const` | `device() -> int` — one of `cytnx.Device.*` |
| `device_str` | `const string device_str() const` | `device_str() -> str` — e.g. `"cytnx device: CPU"` |
| `size` | `unsigned long long size() const` | `size() -> int`; also `__len__`. **Raises on a default-constructed `Storage()`** (C7) |
| `capacity` | `unsigned long long capacity() const` | `capacity() -> int` — allocated slots (≥ `size()`), rounded to a multiple of `STORAGE_DEFT_SZ` (2) |
| `resize` | `void resize(const cytnx_uint64& newsize)` | `resize(newsize: int) -> None` — in place; zero-fills a fresh grow but **not** reused capacity (C4) |
| `append` | `template<class T> void append(const T&)` (11 scalar overloads) | `append(val) -> None` — in-place grow by one |
| `fill` | `template<class T> void fill(const T&)` (11 scalar overloads) | `fill(val) -> None` — in place |
| `set_zeros` | `void set_zeros()` | `set_zeros() -> None` — in place |
| `real` | `Storage real() const` | `real() -> Storage` — complex-only; **raises on a real Storage** (C8) |
| `imag` | `Storage imag() const` | `imag() -> Storage` — complex-only |
| `numpy` | *(none — pybind-only lambda)* | `numpy() -> numpy.ndarray` — **a COPY, not a view** (P1) |
| `pylist` | *(none — `Storage_conti.py`)* | `pylist() -> list` — dtype-dispatching wrapper over the `c_pylist_*` set |
| `c_pylist_<dtype>` (×11) | `template<class T> vector<T> vector()` | `c_pylist_double() -> list[float]`, `c_pylist_int64() -> list[int]`, … — raw per-dtype accessor; **raises on dtype mismatch** (P5) |
| `print_info` | `void print_info() const` | `print_info() -> None` — prints to stdout, **capturable** via `scoped_ostream_redirect` (P6) |
| *(dunder)* `__getitem__` / `__setitem__` | `Scalar get_item(...)` / `set_item(...)` | `storage[i]` / `storage[i] = v` — get returns a Python scalar (a copy); set mutates in place. Bounds guard is off-by-one (P7) |
| *(dunder)* `__eq__` | `bool operator==(const Storage&)` | `a == b` — deep value compare; **raises on dtype mismatch** (C5) |
| *(dunder)* `__repr__` / `__str__` | `operator<<` / `print_info` | return `""`; info is a captured `std::cout` side effect (P6) |
| *(C++-only)* `operator!=` | `bool operator!=(const Storage&) const` | not explicitly bound; works via Python's `__ne__`-delegates-to-`__eq__` — and therefore **also raises on dtype mismatch** (C5) |
| *(C++-only)* `at`/`back`/`data`/`get_item`/`set_item`/`vector`/`from_vector`/`release` | various | not bound under these names (C++-only or exposed only through `__getitem__`/`from_pylist`/`c_pylist_*`) |

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/Storage.py`.

- **P1 — headline finding: `numpy()` returns a COPY, not a view aliased to the
  `Storage` buffer — the opposite of the near-universal NumPy/PyTorch `.numpy()`
  zero-copy convention.** The binding (`storage_py.cpp:27-72`) first makes a CPU
  clone of `self` (`tmpIN = self.clone()`, or `self.to(Device.cpu)` off-GPU),
  then hands that clone's buffer to the ndarray via `tmpIN.release()` — which
  transfers ownership of a *separate* buffer into numpy. The ndarray therefore
  shares no memory with the `Storage`. Probed both directions with the
  `returns_view` detector: *"numpy() is a COPY, not a view -- mutating the
  returned ndarray does NOT change the source Storage (returns_view is False)"*
  `[PASS]`, and *"mutating the source Storage does NOT change a previously-returned
  numpy array either -- the two own separate buffers"* `[PASS]`, plus the same
  for a ComplexDouble Storage → complex128 copy `[PASS]`. Since there is no
  in-repo `from_numpy` on `Storage` (it has only `from_pylist`), a numpy
  round-trip is copy-in / copy-out on both legs. This is a B2-class behavior
  that must be documented explicitly, because callers coming from numpy/torch
  will (wrongly) expect `storage.numpy()[i] = x` to write through to the
  Storage.

- **P2 — `astype()` is a pure method that nonetheless returns the *identical
  Python object* when the target dtype equals the current dtype, aliasing its
  receiver.** The pybind `astype` binding is commented out (`storage_py.cpp:94`);
  the live method is defined in `Storage_conti.py:20-26` as
  `if self.dtype()==new_type: return self  else: return self.astype_different_type(new_type)`.
  So `astype(same_dtype)` hands back `self` (same wrapper, same buffer), while
  `astype(different_dtype)` builds a fresh, independent Storage. Probed:
  *"astype(same_dtype) returns the IDENTICAL Python object (astype(D) is s)"*
  `[PASS]` and *"astype(different_dtype) returns a NEW, independent Storage (not
  self); mutating it leaves the source unchanged"* `[PASS]`. Consequence (B1): a
  caller who treats `astype`'s result as a private copy and mutates it can
  silently corrupt the original when the dtype happened to already match.

- **P3 — `to()` has the identical self-return pattern**, also implemented in
  `Storage_conti.py:28-34`: `to(same_device)` returns `self`, `to(other_device)`
  returns a new Storage on that device. Probed: *"to(same_device) returns the
  IDENTICAL Python object (to(cpu) is s)"* `[PASS]`. The in-place counterpart
  `to_(device)` returns `None` (a proper void mutator): *"to_(device) is the
  in-place device move and returns None (void)"* `[PASS]`. `to`/`to_` is thus a
  correct N2 pure/in-place pair (unlike `astype`, which has no `_` counterpart).

- **P4 — the internal escape-hatch helpers `astype_different_type` and
  `to_different_device` are bound as *public* methods and raise `RuntimeError`
  on the no-op case** (same dtype / same device) that their `Storage_conti.py`
  wrappers are meant to intercept. They exist only because pybind cannot express
  "return `self` on a no-op conversion" (the binding comment at
  `storage_py.cpp:90-94` says exactly this), so the decision was pushed to the
  Python layer and the raw helpers leaked into the surface. Probed:
  *"astype_different_type(same_dtype) RAISES RuntimeError"* `[PASS]` and
  *"to_different_device(same_device) RAISES RuntimeError"* `[PASS]`. Calling
  either directly is always a mistake; they should be private (see C3).

- **P5 — the 11 `c_pylist_<dtype>` methods are type-specific internal
  accessors leaked into the public surface**; the wrong one for the Storage's
  dtype raises. Each is a direct bind of the C++ template `Storage::vector<T>()`
  (`storage_py.cpp:250-260`), which checks `Type.cy_typeid(T) == this->dtype()`
  and raises otherwise (`Storage.cpp:318`). The `pylist()` wrapper
  (`Storage_conti.py:40-65`) is the dtype-dispatching front-end that selects the
  right one. Probed: *"c_pylist_double() (matching the Double dtype) returns a
  Python list copy"* `[PASS]`, *"c_pylist_int64() on a Double Storage (dtype
  MISMATCH) RAISES RuntimeError"* `[PASS]`, and *"pylist() returns the correct
  Python list copy without the caller naming a dtype"* `[PASS]`. Eleven of the
  38 public members are these raw accessors; folding them behind `pylist()` (C3)
  removes ~29% of the surface with no loss of capability.

- **P6 — `__repr__`/`__str__` return the empty string; the info is a captured
  `std::cout` side effect — but unlike `Symmetry` (P5) and `Device`
  (`enums.md` P4), `Storage` gets this right.** `storage_py.cpp:167-173` wraps
  the `operator<<` print in `py::scoped_ostream_redirect` and returns `""`
  (same "prints, returns ''" shape as Symmetry's `__repr__`), and — the positive
  contrast — `print_info` is *also* bound directly with the same redirect guard
  (`storage_py.cpp:197-198`), so its output IS capturable by Python. Probed:
  *"repr(storage) and str(storage) both evaluate to the empty string ''"*
  `[PASS]`, *"...yet repr(storage) DOES print human-readable info as a side
  effect, and (unlike Device.Print_Property) it IS capturable via
  redirect_stdout"* `[PASS]`, and *"print_info() is directly bound AND capturable
  ... -- Storage does what Device.Print_Property (enums P4) fails to, and binds
  what Symmetry.print_info leaves unbound"* `[PASS]`. The remaining defect is
  only the empty return value of `__repr__`/`__str__` (should return the string,
  not print it), identical to Symmetry P5's recommended fix.

- **P7 — the `__getitem__`/`__setitem__` bounds guard is off-by-one, but the
  hole is masked by an independent inner check, and there is no negative
  indexing.** The pybind lambdas guard with `idx > self.size()`
  (`storage_py.cpp:108,139`) — which should be `>=`, so `idx == size` slips
  through the guard — but the underlying `Storage_base::at()` has its own
  `idx < size` bounds check (`Storage_base.cpp:673`) that catches `idx == size`
  and raises, so no out-of-bounds read actually occurs. Probed: *"storage[size]
  (idx == size) RAISES RuntimeError -- the pybind guard 'idx > size' is
  off-by-one ... but the inner at() bounds check catches it independently, so no
  out-of-bounds read actually happens"* `[PASS]`, *"storage[size+1] also RAISES
  RuntimeError"* `[PASS]`. Separately, negative indices are rejected at the
  type layer (the index param is unsigned): *"storage[-1] RAISES TypeError --
  ... there is NO Python-style negative indexing (unlike list/ndarray)"*
  `[PASS]`. B4 is satisfied at runtime (errors are exceptions, no segfault); the
  off-by-one guard is a code-cleanliness issue and the missing negative
  indexing is an ergonomics gap.

## Consistency findings

- **C1 — violates N1: capitalized callable members.** `Init`, `Load`, `Save`,
  `Tofile`, `Fromfile` all use capitalized forms instead of `snake_case`. These
  are init/IO methods, not mutate/pure operation pairs, so N2 does not apply —
  only the N1 casing rule (same reasoning as `Symmetry.md`'s C1 for
  `Load`/`Save`). `Tofile`/`Fromfile` additionally read better hyphenated:
  `to_file`/`from_file` (they are the raw-binary siblings of `Save`/`Load`).

- **C2 — violates N2: `astype` has no in-place counterpart while its sibling
  conversion `to` does.** `to`/`to_` is a correct pure/in-place pair (P3), but
  dtype conversion is offered only as the pure `astype` — there is no
  `astype_()`. Per N2 ("every in-place method has a pure counterpart" and the
  converse spirit for operations meaningful in both forms), an in-place dtype
  cast is as meaningful as an in-place device move; recommend adding `astype_()`
  for symmetry, or documenting the deliberate omission. (Low priority — an
  in-place dtype cast changes element width and so cannot reuse the buffer, a
  legitimate reason to offer only the pure form.)

- **C3 — 13 of the 38 public members are leaked implementation details.**
  `astype_different_type` and `to_different_device` (P4) are pybind escape
  hatches for the "return self on no-op" problem, and the 11 `c_pylist_<dtype>`
  accessors (P5) are the raw per-dtype backing of `pylist()`. All 13 are only
  ever meant to be reached through their public wrappers (`astype`/`to`/
  `pylist`), and calling them directly on the wrong dtype/device raises. No
  single `N`/`B` id names "don't expose internals," so this is flagged as a plain
  surface-hygiene inconsistency (the same informal treatment `enums.md` gives
  namespace pollution in C2); recommend underscore-prefixing all 13 so they drop
  out of `dir(cytnx.Storage)`.

- **C4 — `resize()`'s zero-fill guarantee is inconsistent: a fresh grow zeros
  the new region, but growing back into previously-shrunk capacity leaves stale
  data.** `StorageImplementation::resize` (`StorageImplementation.cpp:519-550`)
  only `calloc`s (zero-fills) when `newsize > capacity_`; when `newsize` is
  within the current `capacity_` it merely sets `size_ = newsize` without
  clearing the reused slots. Probed: *"resize(5) that grows beyond capacity
  zero-fills the new region (calloc): elements become [1, 2, 3, 0, 0]"*
  `[PASS]`, but *"resize DOWN then UP within the old capacity does NOT re-zero
  the reused slots -- stale data [1, 2, 3, 4] reappears ...; resize's zero-fill
  is therefore not guaranteed"* `[PASS]`. No `N`/`B` id covers "same method,
  input-dependent initialization," so flagged informally; recommend either
  always zeroing the `[old_size, new_size)` region or documenting that grown
  elements are unspecified.

- **C5 — `==` raises on a dtype mismatch instead of returning `False`,
  violating the B4/B5 spirit and Python's total-`==` convention.**
  `Storage::operator==` (`Storage.cpp:15-17`) calls `cytnx_error_msg` when
  `this->dtype() != rhs.dtype()`, so a Python `a == b` between a Double and a
  Uint64 Storage throws `RuntimeError` rather than yielding `False`. Because
  `__ne__` delegates to `__eq__`, `a != b` also throws. Probed: *"== returns
  True for value-equal same-dtype Storages ... and False for value-unequal"*
  `[PASS]`, and *"comparing two Storages of DIFFERENT dtype with == RAISES
  RuntimeError instead of returning False -- violates Python's convention that ==
  is total and never throws across types (a real footgun: `if a == b:` can
  raise)"* `[PASS]`. Per B5 (operators equivalent to named methods with stable,
  catchable semantics) recommend returning `False` for a dtype mismatch, so `==`
  is total.

- **C6 — `from_pylist`'s dtype inference is value- and overload-order-dependent,
  and silently prefers *unsigned* for positive-int lists.** `from_pylist` binds
  11 `from_vector<T>` overloads (`storage_py.cpp:227-248`) in the order
  complex128, complex64, double, float, **uint64**, int64, …; pybind picks the
  first that accepts the Python values. So `[1, 2]` (all positive) resolves to
  `Uint64`, `[1, -2]` (a negative present) falls through to `Int64`, and
  `[1.0, 2.0]` resolves to `Double`. Probed: *"from_pylist([1, 2]) ... infers
  UNSIGNED Uint64, while from_pylist([1, -2]) infers Int64, and
  from_pylist([1., 2.]) infers Double ... an all-positive int list silently
  becomes unsigned (so later subtraction underflows); there is no explicit
  dtype= parameter to override this"* `[PASS]`. No `N`/`B` id covers "type
  inferred from values," so flagged informally; recommend adding an explicit
  `dtype=` parameter (as `Init`/the constructor already have) and defaulting
  integer lists to a *signed* type, so `from_pylist([1, 2])` is not silently
  unsigned.

- **C7 — the default-constructed `Storage()` is a half-object whose `size()`
  raises.** `Storage()` sets `_impl` to the abstract `Storage_base` (not a typed
  `StorageImplementation`), whose virtual `size()`/`capacity()` throw
  "Not implemented" (`Storage.hpp:44-46`); its `dtype()` is `Void`. Probed:
  *"Storage() default constructor yields a Void-dtype half-object"* `[PASS]` and
  *"Storage().size() RAISES RuntimeError ('Not implemented') ... an empty Storage
  cannot even be queried for its size"* `[PASS]`. This is B4-adjacent (a query
  that should be total instead raises); recommend the default constructor
  produce a valid empty (size 0) typed Storage, or that `size()`/`capacity()`
  return 0 on the base rather than throwing.

- **C8 — `real()`/`imag()` are complex-only and raise on a real Storage**,
  asymmetric with numpy (`ndarray.real` of a real array returns the array
  itself). `StorageImplementation<double>::real()` raises "can only be called
  from complex type" (`StorageImplementation.cpp:604`). Probed: *"real()/imag()
  on a complex Storage return NEW real-typed Storages"* `[PASS]` and *"real() on
  a REAL Storage RAISES RuntimeError ... asymmetric with numpy"* `[PASS]`.
  Flagged informally; this is arguably intentional (matches the header note),
  but the asymmetry with numpy should at least be documented (done below).

## Recommendation

Every one of the 38 live public members of `cytnx.Storage` appears below,
tagged **keep / add / rename / remove**. Informational rows at the end cover the
dunders (`__getitem__`/`__setitem__`, `__eq__`, `__repr__`/`__str__`) that fall
outside `validate_doc.py`'s `dir()`-based coverage but carry Parity/Consistency
fixes.

| Member | Verdict | Rationale |
|---|---|---|
| `Init` | rename | → `init` (C1/N1). In-place re-initializer. |
| `Load` | rename | → `load` (C1/N1). Static `.cyst` loader. |
| `Save` | rename | → `save` (C1/N1). |
| `Fromfile` | rename | → `from_file` (C1/N1). Raw-binary loader. |
| `Tofile` | rename | → `to_file` (C1/N1). Raw-binary writer. |
| `from_pylist` | keep | Already `snake_case`. **Add an explicit `dtype=` parameter** and default integer lists to a signed type (C6), so `from_pylist([1,2])` is not silently `Uint64`. |
| `astype` | keep | Correctly named. Document the self-return-on-same-dtype aliasing (P2) and the complex→real refusal (B3). |
| `astype_different_type` | remove | Leaked internal helper (P4/C3); fold into `astype`. Make private (`_astype_different_type`). |
| `to` | keep | Correct pure half of the `to`/`to_` pair. Document the self-return-on-same-device aliasing (P3). |
| `to_` | keep | Correct in-place half (returns `None`); proper N2 counterpart of `to`. |
| `to_different_device` | remove | Leaked internal helper (P4/C3); fold into `to`. Make private. |
| `clone` | keep | Correct deep copy (probe); also `__copy__`/`__deepcopy__`. |
| `dtype` | keep | Correct minimal accessor. |
| `dtype_str` | keep | Correct; cross-references `cytnx.Type` (`enums.md`). |
| `device` | keep | Correct minimal accessor. |
| `device_str` | keep | Correct; cross-references `cytnx.Device` (`enums.md`). |
| `size` | keep | Correct (also `__len__`). **Fix C7** so it returns 0 on a default-constructed Storage instead of raising. |
| `capacity` | keep | Correct; `std::vector`-style allocated-slot count. |
| `resize` | keep | Correct signature. **Fix C4**: always zero-fill the grown `[old_size, new_size)` region, not only on realloc. |
| `append` | keep | Correct in-place grow. |
| `fill` | keep | Correct in-place whole-buffer set. |
| `set_zeros` | keep | Correct; faster than `fill(0)` per header note. |
| `real` | keep | Complex-only; document that it raises on a real Storage (C8). |
| `imag` | keep | Complex-only (same note as `real`). |
| `numpy` | keep | Correct name. **Document that it returns a COPY, not a view** (P1) — the single most important caveat on this class. |
| `pylist` | keep | Correct dtype-dispatching exporter; the public front-end for the `c_pylist_*` set. |
| `print_info` | keep | Correctly bound and capturable (P6); matches the C++ name (already `snake_case`). |
| `c_pylist_bool` | remove | Leaked per-dtype internal (P5/C3); reach via `pylist()`. Make private. |
| `c_pylist_complex128` | remove | Same as `c_pylist_bool` (P5/C3). |
| `c_pylist_complex64` | remove | Same (P5/C3). |
| `c_pylist_double` | remove | Same (P5/C3). |
| `c_pylist_float` | remove | Same (P5/C3). |
| `c_pylist_int16` | remove | Same (P5/C3). |
| `c_pylist_int32` | remove | Same (P5/C3). |
| `c_pylist_int64` | remove | Same (P5/C3). |
| `c_pylist_uint16` | remove | Same (P5/C3). |
| `c_pylist_uint32` | remove | Same (P5/C3). |
| `c_pylist_uint64` | remove | Same (P5/C3). |
| *(fix)* `__eq__` / `operator!=` | keep | **Fix C5**: return `False` on a dtype mismatch instead of raising, so `==`/`!=` are total (B5). |
| *(fix)* `__getitem__` / `__setitem__` | keep | **Fix P7**: correct the off-by-one guard (`idx >= size`) and add Python-style negative indexing. |
| *(fix)* `__repr__` / `__str__` | keep | **Fix P6**: build and *return* the info string instead of printing it and returning `""` (same fix as Symmetry P5). |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename`
above, under its recommended name. The 13 `remove` members (the two
`*_different_*` helpers and the 11 `c_pylist_*` accessors) need no docstring.

### `init` (renamed from `Init`)

```
Re-initialize this Storage in place to a new size, dtype, and device.

Parameters
----------
size : int
    Number of elements.
dtype : int, optional
    A cytnx.Type code (default Type.Double). See `enums.md`.
device : int, optional
    A cytnx.Device code (default Device.cpu / -1).
init_zero : bool, optional
    If True (default), all elements are set to zero.

Returns
-------
None

Notes
-----
Renamed from `Init` (C1/N1 casing). Mutates the receiver in place (the same
wrapper object is retained). Prefer `cytnx.Storage(size, dtype, device)` for
fresh construction; use `init` to repurpose an existing handle.
```

### `load` (static, renamed from `Load`)

```
Load a Storage previously written by `save`.

Parameters
----------
fname : str
    File path (the `.cyst` extension written by `save` is expected).

Returns
-------
Storage
    A new Storage reconstructed from the file (value-equal to the saved
    original).

Notes
-----
Renamed from `Load` (C1/N1 casing). This is the counterpart of `save`; for
raw headerless binary data use `from_file` (renamed from `Fromfile`).
```

### `save` (renamed from `Save`)

```
Save this Storage to a `.cyst` file (dtype/size header + data).

Parameters
----------
fname : str
    File path; the `.cyst` extension is appended automatically.

Returns
-------
None

Notes
-----
Renamed from `Save` (C1/N1 casing). For raw headerless binary output use
`to_file` (renamed from `Tofile`).
```

### `from_file` (static, renamed from `Fromfile`)

```
Load raw binary data into a new Storage of a given dtype.

Parameters
----------
fname : str
    File path of the raw binary data (no Cytnx header).
dtype : int
    A cytnx.Type code; must match the data's actual type. Cannot be Type.Void.
count : int, optional
    Number of elements to read; -1 (default) reads all. 0 yields an empty
    Storage of the requested dtype.

Returns
-------
Storage
    A new Storage holding the loaded data.

Notes
-----
Renamed from `Fromfile` (C1/N1 casing). Counterpart of `to_file`; contrast
`load`, which reads the `.cyst` header format.
```

### `to_file` (renamed from `Tofile`)

```
Write this Storage's raw data to a headerless binary file.

Parameters
----------
fname : str
    Destination file path.

Returns
-------
None

Notes
-----
Renamed from `Tofile` (C1/N1 casing). The file holds only the raw element
bytes (no dtype/size header); read it back with `from_file`, supplying the
dtype explicitly.
```

### `from_pylist` (static)

```
Build a new Storage from a Python sequence.

Parameters
----------
pylist : sequence of numbers
    The elements. In the recommended API an explicit `dtype=` parameter is
    added (Consistency finding C6); today the dtype is inferred from the
    values by pybind overload order.
device : int, optional
    A cytnx.Device code (default Device.cpu).

Returns
-------
Storage
    A new Storage holding a copy of the sequence.

Notes
-----
CAVEAT (C6): with no explicit dtype, inference is value-dependent —
`from_pylist([1, 2])` yields an UNSIGNED Uint64 Storage, `from_pylist([1, -2])`
yields Int64, and `from_pylist([1.0, 2.0])` yields Double (all confirmed by
probe). Pass an explicit dtype (recommended API) to avoid the silent-unsigned
surprise.
```

### `astype`

```
Return this Storage cast to a different dtype.

Parameters
----------
new_type : int
    Target cytnx.Type code.

Returns
-------
Storage
    A new Storage with dtype `new_type`, on the same device — UNLESS
    `new_type` equals the current dtype, in which case `self` is returned
    unchanged (see Notes).

Notes
-----
Copy/view (Parity finding P2): `astype(other_dtype)` is a copy (independent
buffer); `astype(current_dtype)` returns the IDENTICAL object (`astype(d) is
self`), so mutating that result mutates the original. A complex->real cast is
REFUSED (RuntimeError, confirmed by probe) — use `real()`/`imag()` instead.
```

### `to`

```
Return this Storage on a different device.

Parameters
----------
device : int
    Target cytnx.Device code.

Returns
-------
Storage
    A new Storage on `device` — UNLESS `device` equals the current device, in
    which case `self` is returned unchanged.

Notes
-----
Copy/view (Parity finding P3): same self-return-on-no-op aliasing as `astype`.
For an in-place device move that mutates the receiver, use `to_`.
```

### `to_`

```
Move this Storage to a different device in place.

Parameters
----------
device : int
    Target cytnx.Device code.

Returns
-------
None

Notes
-----
In-place counterpart of `to` (the correct N2 pure/in-place pair). Mutates the
receiver and returns None (confirmed by probe).
```

### `clone`

```
Return an independent deep copy of this Storage.

Returns
-------
Storage
    A new Storage with its own buffer, value-equal to `self`.

Notes
-----
Copy (confirmed by probe): mutating the clone does not affect the source.
Also bound as `copy.copy()` and `copy.deepcopy()` — both are DEEP copies here
(copy.copy is not shallow for Storage).
```

### `dtype`

```
Return this Storage's dtype code.

Returns
-------
int
    One of cytnx.Type.* (see `enums.md`), e.g. Type.Double == 3.
```

### `dtype_str`

```
Return this Storage's dtype as a human-readable string.

Returns
-------
str
    e.g. "Double (Float64)", "Complex Double (Complex Float64)", or "Int64"
    (integer types carry no parenthetical form).
```

### `device`

```
Return this Storage's device code.

Returns
-------
int
    One of cytnx.Device.* (see `enums.md`); -1 (Device.cpu) on a CPU build.
```

### `device_str`

```
Return this Storage's device as a human-readable string.

Returns
-------
str
    e.g. "cytnx device: CPU".
```

### `size`

```
Return the number of elements in this Storage.

Returns
-------
int
    The element count (also available as `len(storage)`).

Notes
-----
On a default-constructed `Storage()` this currently RAISES (Consistency
finding C7); the recommended API returns 0 for the empty case.
```

### `capacity`

```
Return the number of allocated element slots.

Returns
-------
int
    The allocated capacity (>= size()), std::vector-style. Rounded up to a
    multiple of the internal block size (2), so a size-3 Storage reports
    capacity 4.
```

### `resize`

```
Resize this Storage in place to a new element count.

Parameters
----------
newsize : int
    The new number of elements. Growing beyond the current capacity
    reallocates; shrinking keeps the capacity.

Returns
-------
None

Notes
-----
Zero-fill caveat (Consistency finding C4): a grow that reallocates zero-fills
the new region, but growing back into previously-shrunk capacity leaves STALE
data in the reused slots (confirmed by probe). The recommended API always
zeroes grown elements.
```

### `append`

```
Append one value to this Storage in place.

Parameters
----------
val : number
    The value to append; must be compatible with the Storage's dtype (a
    complex value cannot be appended to a real Storage).

Returns
-------
None

Notes
-----
Grows size() by one (confirmed by probe), reallocating capacity as needed.
```

### `fill`

```
Set every element of this Storage to a value, in place.

Parameters
----------
val : number
    The fill value; must be compatible with the Storage's dtype.

Returns
-------
None

Notes
-----
For zeroing specifically, `set_zeros()` is faster (per the C++ header note).
```

### `set_zeros`

```
Set every element of this Storage to zero, in place.

Returns
-------
None

Notes
-----
Faster than `fill(0)` for the all-zero case.
```

### `real`

```
Return the real part of a complex Storage as a new real-typed Storage.

Returns
-------
Storage
    A new real-dtype Storage holding the real parts.

Raises
------
RuntimeError
    If called on a real (non-complex) Storage (confirmed by probe) — this is
    asymmetric with numpy, where `ndarray.real` of a real array returns itself
    (Consistency finding C8).
```

### `imag`

```
Return the imaginary part of a complex Storage as a new real-typed Storage.

Returns
-------
Storage
    A new real-dtype Storage holding the imaginary parts.

Raises
------
RuntimeError
    If called on a real (non-complex) Storage (same as `real`, C8).
```

### `numpy`

```
Return a NumPy array holding this Storage's data.

Returns
-------
numpy.ndarray
    A 1-D array of the corresponding numpy dtype.

Notes
-----
COPY, NOT A VIEW (Parity finding P1): the returned ndarray owns a SEPARATE
buffer (the binding clones the Storage and transfers that clone's memory to
numpy). Mutating the ndarray does NOT write back to the Storage, and mutating
the Storage does NOT change a previously-returned ndarray (both confirmed by
probe) — the opposite of numpy/torch `.numpy()` zero-copy behavior. There is no
`Storage.from_numpy`; build a Storage from a list via `from_pylist`.
```

### `pylist`

```
Return this Storage's elements as a Python list.

Returns
-------
list
    A list copy of the elements, of the Python type matching the Storage's
    dtype (float, complex, int, or bool).

Notes
-----
The dtype-dispatching public front-end for the (recommended-private)
`c_pylist_<dtype>` accessors (Consistency finding C3); it selects the correct
one automatically so callers never name a dtype.
```

### `print_info`

```
Print this Storage's dtype, device, size, and elements to stdout.

Returns
-------
None

Notes
-----
Bound with a `scoped_ostream_redirect` guard, so its output IS capturable by
Python's `contextlib.redirect_stdout` (confirmed by probe) — unlike
`Device.Print_Property` (`enums.md` P4). Distinct from `__repr__`/`__str__`,
which currently print the same block but return "" (Parity finding P6).
```

### `__eq__` / `__ne__` (recommended fix)

```
Compare two Storages elementwise by value.

Returns
-------
bool
    True iff both Storages have the same dtype, size, and elementwise-equal
    values (deep value compare, not identity — confirmed by probe).

Notes
-----
Fixes Consistency finding C5: today a dtype MISMATCH RAISES RuntimeError
instead of returning False (so `if a == b:` can throw); the recommended API
returns False on a dtype mismatch so `==`/`!=` are total. `is` (identity) is a
separate question — use `cytnx.is(a, b)` to test shared instances.
```

### `__getitem__` / `__setitem__` (recommended fix)

```
Get or set a single element by index: `storage[i]` / `storage[i] = v`.

Parameters
----------
i : int
    Element index in [0, size()).

Returns
-------
number
    (`__getitem__`) A Python scalar COPY of the element — not a view;
    `__setitem__` mutates the Storage in place (confirmed by probe).

Notes
-----
Fixes Parity finding P7: today the bounds guard is off-by-one (`idx > size`,
masked by an inner check so `storage[size]` still raises) and negative indices
raise TypeError; the recommended API uses an `idx >= size` guard and supports
Python-style negative indexing.
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Init` | `init` | N1 casing (C1) |
| `Load` | `load` | N1 casing (C1) |
| `Save` | `save` | N1 casing (C1) |
| `Fromfile` | `from_file` | N1 casing (C1) |
| `Tofile` | `to_file` | N1 casing (C1) |
| `astype_different_type` | *(removed → private `_astype_different_type`)* | leaked internal helper (P4/C3) |
| `to_different_device` | *(removed → private `_to_different_device`)* | leaked internal helper (P4/C3) |
| `c_pylist_bool` … `c_pylist_uint64` (×11) | *(removed → private; reach via `pylist`)* | leaked per-dtype accessors (P5/C3) |
| `from_pylist(pylist, device)` | `from_pylist(pylist, dtype=?, device=?)` | add explicit dtype, default signed (C6) |
| `resize` (grow-only zero-fill) | `resize` (always zero-fill grown region) | C4 |
| `size`/`capacity` on `Storage()` (raise) | return 0 | C7 |
| `real`/`imag` on real Storage (raise) | raise — documented, asymmetric with numpy | C8 (documentation only) |
| `astype` / `numpy` / `to` (silent self/copy aliasing) | same names, documented copy/view semantics | P1/P2/P3 |
| `__eq__` / `operator!=` (raise on dtype mismatch) | return `False` on dtype mismatch | C5 (B5) |
| `__getitem__` / `__setitem__` (`idx > size`, no negatives) | `idx >= size` guard + negative indexing | P7 |
| `__repr__` / `__str__` (print, return `''`) | return the info string | P6 |

Every other public member of `Storage` — `from_pylist`, `astype`, `to`, `to_`,
`clone`, `dtype`, `dtype_str`, `device`, `device_str`, `size`, `capacity`,
`resize`, `append`, `fill`, `set_zeros`, `real`, `imag`, `numpy`, `pylist`,
`print_info` — keeps its current name; the behavioral fixes above (C4/C5/C6/C7,
P1/P2/P3) change semantics or docstrings, not names, and are listed in their own
rows rather than this "unchanged-name" list.
