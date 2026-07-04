# `Tensor` — API audit

`Tensor` is Cytnx's dense multi-dimensional array — the non-symmetric,
block-free numeric container that underlies every `Storage`-backed computation
and that a `UniTensor` wraps when it carries no symmetry. It owns a dtype (a
`Type` code, see `enums.md`), a device (a `Device` code, see `enums.md`), a
shape, and a reference-counted `Storage`; it exposes shape manipulation
(`permute`/`reshape`/`contiguous`), dtype/device conversion (`astype`/`to`),
elementwise/reduction linear algebra (`Conj`/`Exp`/`Norm`/`Svd`/…), and the full
Python arithmetic-operator set. This document audits the **67 public members**
of the live `cytnx.Tensor` class (installed `cytnx==1.1.0` wheel).

Ground truth for behavior is `docs/api-audit/probes/Tensor.py`, executed against
`./.venv/bin/python` (`source tools/env.sh && $PY
docs/api-audit/probes/Tensor.py`; all 48 assertions `[PASS]`, exit 0). Ground
truth for static signatures is `cytnx_src/include/Tensor.hpp` (C++
declarations), `cytnx_src/pybind/tensor_py.cpp` (the pybind11 binding —
authoritative for the Python-visible call signatures, the operator overload
set, and the numerous internal helper lambdas), and
`cytnx_src/cytnx/Tensor_conti.py` (the Python-side augmentation layer, which
renames/wraps several raw pybind primitives — `astype`, `to`, `contiguous`, the
`__i*__` in-place operators, and the `Conj_`/`Exp_`/`Inv_`/`InvM_`/`Abs_`/`Pow_`
in-place methods — into their friendlier public forms, and which is the source
of Parity finding P4).

## Inventory

C++ signatures are read from `Tensor.hpp`; Python signatures are the effective
pybind-visible signature, cross-checked against `tools/member_inventory.py
Tensor`. Members are grouped by role; every one of the 67 live public members
appears exactly once.

### Metadata / introspection

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `dtype` | `unsigned int dtype() const` | `dtype() -> int` — a `Type` code (see `enums.md`) |
| `dtype_str` | `std::string dtype_str() const` | `dtype_str() -> str`, e.g. `"Double (Float64)"` |
| `device` | `int device() const` | `device() -> int` — a `Device` code (see `enums.md`) |
| `device_str` | `std::string device_str() const` | `device_str() -> str`, e.g. `"cytnx device: CPU"` |
| `shape` | `const std::vector<cytnx_uint64>& shape() const` | `shape() -> list[int]` |
| `rank` | `cytnx_uint64 rank() const` | `rank() -> int` (== `len(shape())`) |
| `is_contiguous` | `const bool& is_contiguous() const` | `is_contiguous() -> bool` |
| `item` | `template<class T> T& item()` | `item() -> object` (dtype-dispatched Python scalar; raises on a non-scalar Tensor) |
| `same_data` | `bool same_data(const Tensor&) const` | `same_data(other: Tensor) -> bool` — do the two share storage |
| `storage` | `Storage& storage() const` | `storage() -> Storage` (shares the tensor's storage) |
| `numpy` | *(no C++ member; pybind-only lambda)* | `numpy(share_mem: bool=False) -> numpy.ndarray` (default is a copy; P... slicing note) |
| `real` | `Tensor real()` | `real() -> Tensor` (complex tensors only) |
| `imag` | `Tensor imag()` | `imag() -> Tensor` (complex tensors only) |

### Shape / layout manipulation

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `permute` | `Tensor permute(const std::vector<cytnx_uint64>&) const` (+ variadic) | `permute(*args) -> Tensor` — **view** (B2) |
| `permute_` | `Tensor& permute_(...)` | `permute_(*args) -> Tensor` (in place, returns self) |
| `reshape` | `Tensor reshape(const std::vector<cytnx_int64>&) const` (+ variadic) | `reshape(*args) -> Tensor` — **view** (B2), accepts a `-1` auto dim |
| `reshape_` | `Tensor& reshape_(...)` | `reshape_(*args) -> Tensor` (in place, returns self) |
| `contiguous` | `Tensor contiguous() const` | `contiguous() -> Tensor` — **Python-side wrapper** (`Tensor_conti.py`): returns self if already contiguous, else a fresh contiguous copy |
| `contiguous_` | `Tensor contiguous_()` | `contiguous_() -> Tensor` (in place; return identity not guaranteed, B1) |
| `make_contiguous` | `Tensor contiguous() const` *(bound under this name at `tensor_py.cpp:192`)* | `make_contiguous() -> Tensor` — the **raw** C++ `contiguous()`; leaked internal, see P3 |
| `flatten` | `Tensor flatten() const` | `flatten() -> Tensor` (clone + contiguous + reshape to 1-D; a **copy**) |
| `flatten_` | `void flatten_()` | `flatten_() -> None` (in place; returns **None**, not self — C4) |
| `append` | overloaded: `void append(const Tensor&)`, `void append(const Storage&)`, `template<class T> void append(const T&)` | `append(val) -> None` (scalar / Tensor / Storage; grows axis 0) |

### dtype / device conversion

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `astype` | `Tensor astype(const int&) const` | `astype(dtype: int) -> Tensor` — **Python-side wrapper** (`Tensor_conti.py`): returns self if dtype unchanged, else a converted copy |
| `astype_different_dtype` | `Tensor astype(const int&) const` *(pybind lambda that asserts dtype differs)* | `astype_different_dtype(new_type: int) -> Tensor` — **raw primitive**, raises on same dtype; leaked internal, see P3 |
| `to` | `Tensor to(const int&) const` | `to(device: int) -> Tensor` — **Python-side wrapper**: returns self if device unchanged, else a copy on the new device |
| `to_` | `void to_(const int&)` | `to_(device: int) -> None` (in place device move) |
| `to_different_device` | `Tensor to(const int&) const` *(pybind lambda that asserts device differs)* | `to_different_device(device: int) -> Tensor` — **raw primitive**, raises on same device; leaked internal, see P3 |
| `clone` | `Tensor clone() const` | `clone() -> Tensor` (deep **copy**); also bound as `__copy__`/`__deepcopy__` |
| `from_storage` (static) | `static Tensor from_storage(const Storage&)` | `from_storage(sin: Storage, is_clone: bool=False) -> Tensor` |

### IO

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `Save` | `void Save(const std::string&) const` | `Save(fname: str) -> None` |
| `Load` (static) | `static Tensor Load(const std::string&)` | `Load(fname: str) -> Tensor` (staticmethod) |
| `Tofile` | `void Tofile(const std::string&) const` | `Tofile(fname: str) -> None` (raw binary) |
| `Fromfile` (static) | `static Tensor Fromfile(const std::string&, const unsigned int&, const cytnx_int64&=-1)` | `Fromfile(fname: str, dtype: int, count: int=-1) -> Tensor` |
| `Init` | `void Init(const std::vector<cytnx_uint64>&, dtype=Type.Double, device=-1, init_zero=true)` | `Init(shape, dtype=3, device=-1, init_zero=True) -> None` — redundant with the constructor |

### Linear algebra member methods (pure)

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `Conj` | `Tensor Conj() const` | `Conj() -> Tensor` (pure) |
| `Exp` | `Tensor Exp() const` | `Exp() -> Tensor` (pure) |
| `Abs` | `Tensor Abs() const` | `Abs() -> Tensor` (pure) |
| `Inv` | `Tensor Inv(const double& clip=-1.) const` | `Inv(clip: float=-1) -> Tensor` (elementwise pseudo-inverse) |
| `InvM` | `Tensor InvM() const` | `InvM() -> Tensor` (matrix inverse) |
| `Pow` | `Tensor Pow(const cytnx_double& p) const` | `Pow(p: float) -> Tensor` (pure) |
| `Norm` | `Tensor Norm() const` | `Norm() -> Tensor` (rank-0 result) |
| `Max` | `Tensor Max() const` | `Max() -> Tensor` |
| `Min` | `Tensor Min() const` | `Min() -> Tensor` |
| `Trace` | `Tensor Trace(const cytnx_uint64& a=0, const cytnx_uint64& b=1) const` | `Trace(arg0: int, arg1: int) -> Tensor` — **defaults dropped**, args positional-only (P2) |
| `Svd` | `std::vector<Tensor> Svd(const bool& is_UvT=true) const` | `Svd(is_UvT: bool=True) -> list[Tensor]` |
| `Eigh` | `std::vector<Tensor> Eigh(const bool& is_V=true, const bool& row_v=false) const` | `Eigh(is_V: bool=True, row_v: bool=False) -> list[Tensor]` |

### Linear algebra in-place methods (public `_` wrappers, `Tensor_conti.py`)

| Member | C++ signature | Python (effective, live) signature |
|---|---|---|
| `Conj_` | `Tensor& Conj_()` | `Conj_() -> Tensor` (in place, returns self via `Tensor_conti.py`) |
| `Exp_` | `Tensor& Exp_()` | `Exp_() -> Tensor` (in place, returns self) |
| `Abs_` | `Tensor& Abs_()` | `Abs_() -> Tensor` (in place, returns self) |
| `Inv_` | `Tensor& Inv_(const double& clip=-1.)` | `Inv_(clip) -> Tensor` (in place, returns self) |
| `InvM_` | `Tensor& InvM_()` | `InvM_() -> Tensor` (in place, returns self) |
| `Pow_` | `Tensor& Pow_(const cytnx_double& p)` | `Pow_(p) -> Tensor` (in place, returns self) |

### Leaked raw pybind primitives (C-prefixed) — see P3

| Member | Backing C++ | Python (effective, live) signature |
|---|---|---|
| `cConj_` | `Tensor::Conj_` | `cConj_() -> Tensor` — raw in-place, wrapped by `Conj_` |
| `cExp_` | `Tensor::Exp_` | `cExp_() -> Tensor` — wrapped by `Exp_` |
| `cAbs_` | `Tensor::Abs_` | `cAbs_() -> Tensor` — wrapped by `Abs_` |
| `cInv_` | `Tensor::Inv_` | `cInv_(clip: float=-1) -> Tensor` — wrapped by `Inv_` |
| `cInvM_` | `Tensor::InvM_` | `cInvM_() -> Tensor` — wrapped by `InvM_` |
| `cPow_` | `Tensor::Pow_` | `cPow_(p) -> Tensor` — wrapped by `Pow_` |
| `c__iadd__` | `Tensor::Add_` | `c__iadd__(rhs) -> Tensor` — wrapped by `Tensor_conti.py.__iadd__` |
| `c__isub__` | `Tensor::Sub_` | `c__isub__(rhs) -> Tensor` — wrapped by `__isub__` |
| `c__imul__` | `Tensor::Mul_` | `c__imul__(rhs) -> Tensor` — wrapped by `__imul__` |
| `c__itruediv__` | `Tensor::Div_` | `c__itruediv__(rhs) -> Tensor` — wrapped by `__itruediv__` |
| `c__ifloordiv__` | `Tensor::Div_` (floor) | `c__ifloordiv__(rhs) -> Tensor` — wrapped by `__ifloordiv__` |
| `c__ipow__` | `linalg::Pow_` | `c__ipow__(p) -> None` — wrapped by `__ipow__` |
| `c__imatmul__` | `linalg::Dot` | `c__imatmul__(rhs) -> Tensor` — wrapped (misnamed) by `__imatmul`, see P4 |
| `fill` | `template<class T> void fill(const T&)` | `fill(val) -> None` (in place, all elements) |

*(Not in the 67 public-member set but referenced below:* the operator dunders
`__add__`/`__radd__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__`/`__mod__`/
`__matmul__`/`__pow__`/`__neg__`/`__pos__`/`__eq__`/`__getitem__`/`__setitem__`/
`__len__`/`__iadd__`/… are all bound but underscore-prefixed, so
`validate_doc.py` does not track them; they are covered in Parity/Consistency
findings where their behavior is load-bearing.)*

## Parity findings

Every claim below is backed by a passing `report(...)` assertion in
`docs/api-audit/probes/Tensor.py`.

- **P1 — headline: the entire C++ *named* arithmetic-method family is unbound in
  Python; only the operator dunders exist.** `Tensor.hpp` defines
  `Add`/`Sub`/`Mul`/`Div` and their in-place `Add_`/`Sub_`/`Mul_`/`Div_`, plus
  `Cpr` (compare) and `Mod`, as public members (`Tensor.hpp:1270-1399`). None of
  them is registered in `tensor_py.cpp` — the pybind file binds only the
  operator dunders (`__add__`, `__iadd__`, …) and the C-prefixed raw in-place
  helpers (P3). Probed: *"the C++ named arithmetic methods (Add/Sub/Mul/Div and
  their _ in-place forms, Cpr, Mod) have NO Python binding"* `[PASS]`
  (`not any(hasattr(Tensor, m) for m in ("Add","add","Sub",...))`). Consequence
  for convention B5 ("operator overloads are equivalent to their named-method
  counterparts"): the *named* side does not exist in Python, so a caller who
  wants `t.add(x)` must use `t + x`; there is no method form to reach for, and
  no way for the two to diverge because only one exists. The recommended API
  should either bind the named methods (`add`/`sub`/`mul`/`div` + `_` forms) so
  B5 holds symmetrically, or document operators as the sole surface.

- **P2 — C++ default arguments are dropped by several bindings; `Trace` is the
  worst case, becoming call-incompatible with its C++ signature.** C++
  `Trace(a=0, b=1)` (`Tensor.hpp:1702`) is bound as
  `.def("Trace", &cytnx::Tensor::Trace)` with **no `py::arg` defaults**
  (`tensor_py.cpp:1798`), so Python `Trace` requires two positional args named
  `arg0`/`arg1` (member_inventory confirms the anonymous names). Probed:
  *"Trace() REQUIRES two positional axis args in Python: the C++ default args
  (a=0, b=1) were dropped"* `[PASS]` (`Trace()` raises `TypeError`), and
  *"Trace(0, 1) computes the trace over the given axes (0+4+8 == 12)"* `[PASS]`.
  This is both a signature-parity gap (defaults lost) and an N4 loss (the
  meaningful C++ names `a`/`b` become `arg0`/`arg1`). The recommended `trace`
  binding must restore `py::arg("axis_a")=0, py::arg("axis_b")=1`.

- **P3 — headline: a whole layer of raw pybind implementation primitives is
  leaked into the public Python surface, each shadowed by a friendlier
  `Tensor_conti.py` wrapper.** Three distinct sub-families:
  (a) the **`*_different_*` no-op-refusing primitives** — `astype_different_dtype`
  and `to_different_device` are pybind lambdas (`tensor_py.cpp:205-214,166-175`)
  that **hard-assert the arguments differ** and exist only so `Tensor_conti.py`
  can intercept the same-dtype / same-device no-op Python-side; called directly
  with an unchanged dtype/device they raise. Probed: *"astype_different_dtype()
  … RAISES RuntimeError when asked for the same dtype"* and
  *"to_different_device() … likewise RAISES RuntimeError for the same device"* —
  both `[PASS]`. (b) `make_contiguous` is the **raw C++ `contiguous()`** bound
  under a renamed name (`tensor_py.cpp:192`, whose comment literally reads *"this
  will be rename by python side conti"*); it never short-circuits to `self` the
  way the public `contiguous()` wrapper does — on an already-contiguous tensor it
  returns a *new wrapper that still shares storage*. Probed:
  *"make_contiguous() … returns a NEW wrapper object that still SHARES storage
  (view) -- unlike contiguous()'s identity short-circuit"* `[PASS]`. (c) the
  **C-prefixed in-place primitives** `cConj_`/`cExp_`/`cAbs_`/`cInv_`/`cInvM_`/
  `cPow_` and `c__iadd__`/`c__isub__`/`c__imul__`/`c__itruediv__`/
  `c__ifloordiv__`/`c__ipow__`/`c__imatmul__` are the raw in-place bindings that
  the public `Conj_`/…/`Pow_` methods and the `__i*__` operators wrap. Probed:
  *"c__iadd__() … mutates in place and returns a self-aliasing handle
  (same_data)"* and *"cConj_() … mutates in place and returns a self-aliasing
  handle"* — both `[PASS]`. All ~15 of these are internal helpers that no user
  should call; the recommended API removes them from the public surface (fold
  each into its wrapper).

- **P4 — `Tensor_conti.py` misnames the in-place matmul dunder as `__imatmul`
  (missing the trailing `__`), so `@=` is silently NOT in place.**
  `Tensor_conti.py:84-87` defines `def __imatmul(self, rhs)` — the attribute is
  `__imatmul`, not the special method `__imatmul__` Python's `@=` looks for.
  Probed: *"Tensor_conti.py defines `__imatmul` (missing the trailing __), so the
  true `__imatmul__` slot does NOT exist on Tensor"* `[PASS]`
  (`hasattr(Tensor,"__imatmul") and not hasattr(Tensor,"__imatmul__")`).
  Consequently `t @= x` finds no `__imatmul__`, falls back to
  `t = t.__matmul__(x)` (`linalg.Dot`), and **rebinds `t` to a fresh object
  instead of mutating in place**. Probed: *"`t @= x` is NOT in place: … Python
  falls back to __matmul__ (linalg.Dot) and REBINDS t to a fresh object"*
  `[PASS]` (`t is not its_former_self`). This is a genuine Python-augmentation
  bug: the intended in-place matmul is dead code, and `@=` gives copy semantics
  where in-place is expected (N2/B1/B5).

- **P5 — the bare-1-D-slice `__getitem__` branch leaks an unguarded `std::cout`
  debug line to the process's real stdout.** `tensor_py.cpp:355` contains a
  leftover `std::cout << start << " " << stop << " " << step << std::endl;` in
  the `py::isinstance<py::slice>(locators)` branch, with **no**
  `py::scoped_ostream_redirect` guard. So `t[0:2]` prints e.g. `0 2 1` to the
  real stdout, uncapturable by `contextlib.redirect_stdout` — the identical
  uncapturable-print pattern documented in `enums.md` P4 (`Device.Print_Property`)
  and `Symmetry.md` P5. Probed: *"t[0:2] (bare 1-D slice) leaks a leftover
  'start stop step' debug line to the process's REAL stdout that
  contextlib.redirect_stdout CANNOT capture (the buffer stays empty)"* `[PASS]`
  (buffer empty, yet the line appears on the terminal). Tuple indexing
  (`t[0,1]`) takes a different branch and does not leak. The fix is to delete
  the stray print.

## Consistency findings

- **C1 — violates N1: 23 capitalized callable members.** `Abs`, `Abs_`, `Conj`,
  `Conj_`, `Eigh`, `Exp`, `Exp_`, `Fromfile`, `Init`, `Inv`, `Inv_`, `InvM`,
  `InvM_`, `Load`, `Max`, `Min`, `Norm`, `Pow`, `Pow_`, `Save`, `Svd`, `Tofile`,
  `Trace` all use capitalized verbs/nouns instead of `snake_case`. Per N1 they
  become `abs`/`abs_`/`conj`/`conj_`/`eigh`/`exp`/`exp_`/`from_file`/`init`/
  `inv`/`inv_`/`inv_m`/`inv_m_`/`load`/`max`/`min`/`norm`/`pow`/`pow_`/`save`/
  `svd`/`to_file`/`trace`. (The pure/in-place pairs already satisfy N2's trailing
  `_` rule; only the casing is wrong.)

- **C2 — violates N2: the C-prefixed `c…_` and `c__i…__` variants are an
  unrecognized "double-underscore-ish" convention, and each pure operation has
  *three* spellings.** `00-methodology.md`'s N2 explicitly names the
  `cConj_`-style prefix as not a recognized convention: each must fold into the
  base `_`-suffixed pair. Today conjugation alone has `Conj` (pure), `Conj_`
  (in-place wrapper), and `cConj_` (raw in-place primitive) — three public names
  for two real operations — and the same triple exists for exp/abs/inv/invm/pow.
  The seven `c__i…__` names are likewise raw backings for the `__i…__` operators
  (P3). The recommended API keeps exactly the pure/`_` pair (`conj`/`conj_`, …)
  and removes every `c`-prefixed name.

- **C3 — B2 copy-vs-view is well-defined but NOT numpy-like, and every
  derivation must be classified explicitly.** Measured by probe:

  | Derivation | Result | Probe |
  |---|---|---|
  | `permute` / `permute_` | **view** (shares storage) | `returns_view … is True`, `same_data … is True` `[PASS]` |
  | `reshape` / `reshape_` (contiguous) | **view** (shares storage) | `returns_view … is True and same_data … is True` `[PASS]` |
  | `make_contiguous` (already contig.) | **view** (new wrapper, shared storage) | `same_data(mc) is True` `[PASS]` |
  | `contiguous` (already contig.) | **self** (identity) | `contiguous() is self` `[PASS]` |
  | `contiguous` (non-contig.) | **copy** | `same_data … is False` `[PASS]` |
  | `clone` | **copy** | `returns_view … is False`, `same_data … is False` `[PASS]` |
  | `flatten` | **copy** | `same_data … is False` `[PASS]` |
  | `astype` (different dtype) | **copy** | `same_data … is False` `[PASS]` |
  | `astype` / `to` (unchanged) | **self** (identity) | `astype(same) is self`, `to(same) is self` `[PASS]` |
  | `numpy()` (default) | **copy** | ndarray mutation not seen on source `[PASS]` |
  | **slice READ** `t[0:2]`, `t[0]` | **copy** | `returns_view … is False`, `same_data … is False` `[PASS]` |
  | **element/slice ASSIGN** `t[0,0]=v` | **in-place mutation** | alias observes the write `[PASS]` |

  The load-bearing surprise is the last block: `00-methodology.md`'s B2 states
  the numpy-like expectation that "indexing and slice assignment on a handle is a
  view"; in Cytnx a slice *read* returns an independent **copy** (the C++ `get`
  doc, `Tensor.hpp:1022-1024`, confirms "does not share memory"), while only a
  slice *assignment* mutates in place. This is identical on both language sides
  (so not a parity gap) but it is a genuine consistency/documentation hazard
  against B2: code ported from numpy expecting `t[0:2]` to alias `t` will
  silently get a copy. Every derivation's docstring below states its class.

- **C4 — violates N2's return-symmetry spirit: `flatten_` returns `None` while
  the sibling in-place methods return `self`.** `reshape_`/`permute_` return the
  same object (probed: *"reshape_() … returns the SAME Python object (ret is
  self)"*, *"permute_() … returns the SAME Python object"* — `[PASS]`), and the
  `Conj_`/`Pow_`/`Abs_` wrappers return self (probed `[PASS]`), but `flatten_`
  is bound to a `void` C++ method and returns `None` (probed: *"flatten_() …
  returns None (not self)"* `[PASS]`). Within one class the in-place methods
  should agree on a return convention; recommend `flatten_` return `self` like
  the others so `t.flatten_().dtype()` chains work.

- **C5 — `__eq__` is elementwise (returns a Bool `Tensor`), which makes `Tensor`
  unhashable — a B5-adjacent footgun.** `tensor_py.cpp:1709` binds `__eq__` to
  the elementwise `operator==`, so `a == b` returns a Bool `Tensor`, not a Python
  `bool`. Probed: *"== is ELEMENTWISE and returns a Bool Tensor (not a Python
  bool)"* `[PASS]` and *"Python sets Tensor.__hash__ to None: Tensor is
  UNHASHABLE"* `[PASS]` (`Tensor.__hash__ is None`). This is a deliberate,
  numpy-like choice, but it means a `Tensor` cannot be a dict key or set member,
  and `if a == b:` raises/misbehaves rather than testing whole-tensor equality.
  Flagged informally (no single `N`/`B` id covers "value type made unhashable by
  its own `__eq__`"); recommend a separate `equal(other) -> bool`
  whole-tensor predicate (N5-named) alongside the elementwise operator, mirroring
  numpy's `array_equal`.

- **C6 — positive B3 observation: dtype promotion is a consistent
  widen-to-the-more-general rule, identical on both call paths.** Probed:
  Double + ComplexDouble → ComplexDouble; Double + Float → Double; Int64 + a
  Python float → Double — all `[PASS]`. This is the correct B3 behavior and is
  recorded as a template: the recommended API should keep it and document the
  promotion lattice. (Cross-device promotion is out of scope for this CPU-only
  wheel and untested here.) Errors are likewise proper exceptions (B4): a
  bad-total-size `reshape`, a shape-mismatched `+`, and `item()` on a
  non-scalar all raise catchable `RuntimeError` (probed `[PASS]`).

- **C7 — `Init` is a capitalized (N1) member that also duplicates the
  constructor.** `Init(shape, dtype, device, init_zero)` re-initializes an
  existing `Tensor` in place and is redundant with `Tensor(shape, dtype, device,
  init_zero)` (both bound; `tensor_py.cpp:148-154`). Beyond the N1 casing fix
  (`init`), the recommended API should demote it to an internal detail —
  constructing via `cytnx.Tensor(...)` or the `zeros`/`ones`/`arange` free
  functions is the intended path.

## Recommendation

Every one of the 67 live public members of `cytnx.Tensor` appears below, tagged
**keep / add / rename / remove**.

### Metadata / introspection — all keep

| Member | Verdict | Rationale |
|---|---|---|
| `dtype` | keep | Already `snake_case`; returns a `Type` code (enums.md). |
| `dtype_str` | keep | Already `snake_case`. |
| `device` | keep | Already `snake_case`; returns a `Device` code (enums.md). |
| `device_str` | keep | Already `snake_case`. |
| `shape` | keep | Already `snake_case`. |
| `rank` | keep | Already `snake_case`. |
| `is_contiguous` | keep | Already `snake_case`, correctly `is_`-prefixed (N5). |
| `item` | keep | Already `snake_case`; raises on non-scalar (B4, probed). |
| `same_data` | keep | Already `snake_case`; the view-vs-copy oracle (C3). |
| `storage` | keep | Already `snake_case`; shares the tensor's storage. |
| `numpy` | keep | Already `snake_case`; document `share_mem` copy-vs-view default. |
| `real` | keep | Already `snake_case`; complex-only (B4). |
| `imag` | keep | Already `snake_case`; complex-only (B4). |

### Shape / layout — keep the pairs, remove `make_contiguous`

| Member | Verdict | Rationale |
|---|---|---|
| `permute` | keep | `snake_case`; **view** (C3). |
| `permute_` | keep | `snake_case`, in-place, returns self (C4-compliant). |
| `reshape` | keep | `snake_case`; **view** (C3); accepts `-1` auto-dim. |
| `reshape_` | keep | `snake_case`, in-place, returns self. |
| `contiguous` | keep | `snake_case`; Python-side identity short-circuit (C3). |
| `contiguous_` | keep | `snake_case`, in-place (B1: return identity not guaranteed). |
| `make_contiguous` | remove | Leaked raw C++ `contiguous()` (P3); the public `contiguous` wrapper supersedes it. |
| `flatten` | keep | `snake_case`; a **copy** (C3). |
| `flatten_` | keep | `snake_case`, in-place — **fix to return self** (C4/N2), currently returns None. |
| `append` | keep | `snake_case`; grows axis 0. |

### dtype / device conversion — keep wrappers, remove raw primitives

| Member | Verdict | Rationale |
|---|---|---|
| `astype` | keep | `snake_case`; identity on no-op, copy otherwise (C3). |
| `astype_different_dtype` | remove | Leaked raw primitive that raises on a no-op (P3); folded into `astype`. |
| `to` | keep | `snake_case`; identity on no-op, copy otherwise (C3). |
| `to_` | keep | `snake_case`, in-place device move. |
| `to_different_device` | remove | Leaked raw primitive that raises on a no-op (P3); folded into `to`. |
| `clone` | keep | `snake_case`; deep copy (C3); also `__copy__`/`__deepcopy__`. |
| `from_storage` | keep | `snake_case` static factory. |

### IO — rename for N1

| Member | Verdict | Rationale |
|---|---|---|
| `Save` | rename | → `save` (C1/N1). |
| `Load` | rename | → `load` (C1/N1). Static. |
| `Tofile` | rename | → `to_file` (C1/N1). |
| `Fromfile` | rename | → `from_file` (C1/N1). Static. |
| `Init` | rename | → `init` (C1/N1) and demote (C7): redundant with the constructor. |

### Linear algebra pure methods — rename for N1

| Member | Verdict | Rationale |
|---|---|---|
| `Conj` | rename | → `conj` (C1/N1). Pure (C3). |
| `Exp` | rename | → `exp` (C1/N1). |
| `Abs` | rename | → `abs` (C1/N1). |
| `Inv` | rename | → `inv` (C1/N1). Elementwise pseudo-inverse. |
| `InvM` | rename | → `inv_m` (C1/N1/N3, matching the C++ `InvM` modulo casing). Matrix inverse. |
| `Pow` | rename | → `pow` (C1/N1). |
| `Norm` | rename | → `norm` (C1/N1). |
| `Max` | rename | → `max` (C1/N1). |
| `Min` | rename | → `min` (C1/N1). |
| `Trace` | rename | → `trace` (C1/N1) **and restore defaults + named args** (P2): `trace(axis_a=0, axis_b=1)`. |
| `Svd` | rename | → `svd` (C1/N1). Returns `[U, S, vT]`. |
| `Eigh` | rename | → `eigh` (C1/N1). Returns `[eigvals, eigvecs]`. |

### Linear algebra in-place (trailing-underscore) methods — rename for N1

| Member | Verdict | Rationale |
|---|---|---|
| `Conj_` | rename | → `conj_` (C1/N1). In-place, returns self. |
| `Exp_` | rename | → `exp_` (C1/N1). |
| `Abs_` | rename | → `abs_` (C1/N1). |
| `Inv_` | rename | → `inv_` (C1/N1). |
| `InvM_` | rename | → `inv_m_` (C1/N1). |
| `Pow_` | rename | → `pow_` (C1/N1). |

### Leaked raw primitives — remove all

| Member | Verdict | Rationale |
|---|---|---|
| `cConj_` | remove | Raw in-place primitive behind `Conj_`/`conj_` (P3/C2). |
| `cExp_` | remove | Raw in-place primitive behind `Exp_`/`exp_` (P3/C2). |
| `cAbs_` | remove | Raw in-place primitive behind `Abs_`/`abs_` (P3/C2). |
| `cInv_` | remove | Raw in-place primitive behind `Inv_`/`inv_` (P3/C2). |
| `cInvM_` | remove | Raw in-place primitive behind `InvM_`/`inv_m_` (P3/C2). |
| `cPow_` | remove | Raw in-place primitive behind `Pow_`/`pow_` (P3/C2). |
| `c__iadd__` | remove | Raw in-place primitive behind `__iadd__` (P3/C2). |
| `c__isub__` | remove | Raw in-place primitive behind `__isub__` (P3/C2). |
| `c__imul__` | remove | Raw in-place primitive behind `__imul__` (P3/C2). |
| `c__itruediv__` | remove | Raw in-place primitive behind `__itruediv__` (P3/C2). |
| `c__ifloordiv__` | remove | Raw in-place primitive behind `__ifloordiv__` (P3/C2). |
| `c__ipow__` | remove | Raw in-place primitive behind `__ipow__` (P3/C2). |
| `c__imatmul__` | remove | Raw in-place primitive; its wrapper `__imatmul` is itself broken (P4). Fix `__imatmul__` and drop this. |
| `fill` | keep | `snake_case`, in-place all-element set; correctly named. |

## Docstrings

Numpy-style docstrings for every member tagged `keep`, `add`, or `rename` above,
under its recommended name. Removed members (the `c…` primitives,
`make_contiguous`, `astype_different_dtype`, `to_different_device`) need none.
Rename blocks name the recommended form and cite the C++ original in backticks.

### `dtype` / `dtype_str` / `device` / `device_str`

```
Report the tensor's dtype / device as an integer code or a human string.

Returns
-------
int (`dtype`, `device`) or str (`dtype_str`, `device_str`)
    `dtype()` is a cytnx.Type code (e.g. Type.Double == 3); `device()` is a
    cytnx.Device code (e.g. Device.cpu == -1). The `_str` variants return the
    human forms "Double (Float64)" / "cytnx device: CPU". See enums.md.
```

### `shape` / `rank`

```
The tensor's shape and rank.

Returns
-------
list of int (`shape`) or int (`rank`)
    `shape()` lists the extent of each axis; `rank()` == len(shape())
    (confirmed by probe).
```

### `is_contiguous`

```
Whether the tensor's storage is laid out contiguously.

Returns
-------
bool
    False after a `permute` (a non-contiguous view); True for a freshly
    constructed or `contiguous()`-ed tensor.
```

### `item`

```
Return the single Python scalar of a one-element tensor.

Returns
-------
object
    A Python int/float/complex/bool matching the tensor's dtype.

Raises
------
RuntimeError
    If the tensor has more than one element (confirmed by probe — B4).
```

### `same_data`

```
Whether two tensors share the same underlying storage.

Parameters
----------
other : Tensor

Returns
-------
bool
    True iff the two alias the same memory. This is the view-vs-copy oracle:
    True for `permute`/`reshape` results, False for `clone`/`flatten`/copies
    (confirmed by probe). See the copy-vs-view table (Consistency finding C3).
```

### `storage`

```
Return the tensor's Storage.

Returns
-------
Storage
    Shares the tensor's storage (not a copy); call Storage.clone() for an
    independent copy.
```

### `numpy`

```
Export the tensor as a numpy ndarray.

Parameters
----------
share_mem : bool, optional
    Default False -> the ndarray is an independent COPY (confirmed by probe:
    mutating it does not affect the tensor). True -> a zero-copy view, which
    requires a contiguous CPU tensor and raises otherwise.

Returns
-------
numpy.ndarray
```

### `real` / `imag`

```
Return the real / imaginary part of a complex tensor.

Returns
-------
Tensor

Raises
------
RuntimeError
    If the tensor is not complex (Type.ComplexDouble/ComplexFloat) — B4.
```

### `permute` / `permute_`

```
Reorder the tensor's axes.

Parameters
----------
*args : int
    The new axis order (a permutation of range(rank)).

Returns
-------
Tensor
    `permute` returns a VIEW that shares storage with the source (confirmed by
    probe: a mutation through the permuted handle is visible on the source,
    same_data() is True) and is generally non-contiguous. `permute_` reorders
    in place and returns self.

Notes
-----
View semantics (Consistency finding C3): to detach, follow with `clone()` or
`contiguous()`.
```

### `reshape` / `reshape_`

```
Return the tensor with a new shape (same number of elements).

Parameters
----------
*args : int
    The new shape; a single axis may be -1 to be inferred (numpy-style).

Returns
-------
Tensor
    `reshape` on a contiguous tensor returns a VIEW that SHARES storage
    (confirmed by probe: returns_view is True, same_data() is True), NOT a
    copy. `reshape_` reshapes in place and returns self.

Raises
------
RuntimeError
    If the requested shape's total size differs from the tensor's (probe — B4).
```

### `contiguous` / `contiguous_`

```
Return a contiguous version of the tensor.

Returns
-------
Tensor
    `contiguous` (a Tensor_conti.py wrapper) returns the SAME object when the
    tensor is already contiguous (identity short-circuit, confirmed by probe),
    and a fresh contiguous COPY otherwise. `contiguous_` makes the receiver
    contiguous in place; per B1 the returned handle's identity is not
    guaranteed — rely only on the receiver's state (confirmed by probe).

Notes
-----
The raw C++ `contiguous()` is also exposed today as `make_contiguous`
(recommended for removal, Parity finding P3), which never short-circuits and
returns a storage-sharing view when already contiguous.
```

### `flatten` / `flatten_`

```
Return the tensor collapsed to a single axis.

Returns
-------
Tensor (`flatten`) or None (`flatten_`, current) / Tensor (recommended)
    `flatten` returns an independent 1-D COPY (clone + contiguous + reshape;
    confirmed by probe, same_data() is False). `flatten_` flattens in place.

Notes
-----
`flatten_` currently returns None, unlike `reshape_`/`permute_` which return
self (Consistency finding C4); the recommended API makes `flatten_` return self
for a uniform in-place return convention.
```

### `append`

```
Append data along axis 0, growing the tensor in place.

Parameters
----------
val : scalar, Tensor, or Storage
    A scalar (rank-1 tensor only), a Tensor of shape[1:], or a Storage
    matching the trailing dimension. Cast to the tensor's dtype if needed.

Returns
-------
None
    In place; requires (and forces) contiguity.
```

### `astype`

```
Return the tensor cast to a different dtype.

Parameters
----------
dtype : int
    A cytnx.Type code (enums.md).

Returns
-------
Tensor
    The SAME object if `dtype` equals the current dtype (identity
    short-circuit, confirmed by probe), else an independent COPY with the new
    dtype (same_data() is False).

Notes
-----
Cannot convert complex -> real; use `real()`/`imag()` for that. Replaces the
leaked raw `astype_different_dtype` primitive (Parity finding P3).
```

### `to` / `to_`

```
Move / copy the tensor to a device.

Parameters
----------
device : int
    A cytnx.Device code (enums.md).

Returns
-------
Tensor (`to`) or None (`to_`)
    `to` returns the SAME object if `device` is unchanged (identity
    short-circuit, confirmed by probe), else a copy on the target device.
    `to_` moves in place.

Notes
-----
`to` replaces the leaked raw `to_different_device` primitive (Parity finding
P3).
```

### `clone`

```
Return an independent deep copy of the tensor.

Returns
-------
Tensor
    A new tensor with its own storage (confirmed by probe: a mutation through
    the clone is not visible on the source, same_data() is False). Also bound
    as copy.copy()/copy.deepcopy().
```

### `from_storage`

```
Build a Tensor from a Storage.

Parameters
----------
sin : Storage
is_clone : bool, optional
    Default False -> the tensor shares the Storage's memory; True -> copies it.

Returns
-------
Tensor
```

### `fill`

```
Set every element of the tensor to a value, in place.

Parameters
----------
val : scalar

Returns
-------
None
```

### `save` / `load` (renamed from `Save` / `Load`)

```
Serialize / deserialize a Tensor to/from a .cytn file.

Parameters
----------
fname : str
    File path (extension appended automatically).

Returns
-------
None (`save`) or Tensor (`load`, static)

Notes
-----
Renamed from `Save` / `Load` (C1/N1 casing).
```

### `to_file` / `from_file` (renamed from `Tofile` / `Fromfile`)

```
Write / read raw binary tensor data.

Parameters
----------
fname : str
dtype : int
    (`from_file` only) a cytnx.Type code for interpreting the bytes.
count : int, optional
    (`from_file` only) number of elements to read; -1 (default) reads all.

Returns
-------
None (`to_file`) or Tensor (`from_file`, static)

Notes
-----
Renamed from `Tofile` / `Fromfile` (C1/N1 casing).
```

### `init` (renamed from `Init`)

```
(Re)initialize an existing tensor in place.

Parameters
----------
shape : sequence of int
dtype : int, optional
    A cytnx.Type code; default Type.Double.
device : int, optional
    A cytnx.Device code; default Device.cpu.
init_zero : bool, optional
    Zero-fill the content; default True.

Returns
-------
None

Notes
-----
Renamed from `Init` (C1/N1). Redundant with the `cytnx.Tensor(...)` constructor
and the `zeros`/`ones`/`arange` free functions (Consistency finding C7);
prefer those.
```

### `conj` / `conj_` (renamed from `Conj` / `Conj_`)

```
Complex-conjugate the tensor.

Returns
-------
Tensor
    `conj` is PURE — a new conjugated tensor, source unchanged (confirmed by
    probe: 1+2j -> 1-2j, same_data() is False). `conj_` conjugates IN PLACE and
    returns self (confirmed by probe).

Notes
-----
Renamed from `Conj` / `Conj_` (C1/N1). Replaces the leaked raw `cConj_`
primitive (Parity finding P3 / Consistency finding C2).
```

### `exp` / `exp_` (renamed from `Exp` / `Exp_`)

```
Elementwise exponential.

Returns
-------
Tensor
    `exp` is pure (new tensor); `exp_` applies in place and returns self.

Notes
-----
Renamed from `Exp` / `Exp_` (C1/N1). Replaces the leaked raw `cExp_` (P3/C2).
```

### `abs` / `abs_` (renamed from `Abs` / `Abs_`)

```
Elementwise absolute value.

Returns
-------
Tensor
    `abs` is pure (new tensor); `abs_` applies in place and returns self
    (confirmed by probe: ret is self).

Notes
-----
Renamed from `Abs` / `Abs_` (C1/N1). Replaces the leaked raw `cAbs_` (P3/C2).
```

### `inv` / `inv_` (renamed from `Inv` / `Inv_`)

```
Elementwise (pseudo-)inverse: replace each element x by 1/x.

Parameters
----------
clip : float, optional
    Elements with |x| <= clip are set to 0 (pseudo-inverse). Default -1
    (no clipping).

Returns
-------
Tensor
    `inv` is pure (new tensor); `inv_` applies in place and returns self.

Notes
-----
Renamed from `Inv` / `Inv_` (C1/N1). Replaces the leaked raw `cInv_` (P3/C2).
Distinct from `inv_m` (matrix inverse).
```

### `inv_m` / `inv_m_` (renamed from `InvM` / `InvM_`)

```
Matrix inverse of a square rank-2 tensor.

Returns
-------
Tensor
    `inv_m` is pure (new tensor); `inv_m_` inverts in place and returns self.

Notes
-----
Renamed from `InvM` / `InvM_` (C1/N1/N3 — same word, casing only). Replaces the
leaked raw `cInvM_` (P3/C2). Distinct from the elementwise `inv`.
```

### `pow` / `pow_` (renamed from `Pow` / `Pow_`)

```
Elementwise power.

Parameters
----------
p : float
    The exponent.

Returns
-------
Tensor
    `pow` is PURE (confirmed by probe: source unchanged, values squared for
    p=2); `pow_` raises in place and returns self (confirmed by probe: ret is
    self).

Notes
-----
Renamed from `Pow` / `Pow_` (C1/N1). Replaces the leaked raw `cPow_` (P3/C2).
```

### `norm` (renamed from `Norm`)

```
The Frobenius/2-norm of the tensor.

Returns
-------
Tensor
    A rank-0 tensor holding the norm.

Notes
-----
Renamed from `Norm` (C1/N1).
```

### `max` / `min` (renamed from `Max` / `Min`)

```
The maximum / minimum element of the tensor.

Returns
-------
Tensor
    A rank-0 tensor holding the extremum.

Notes
-----
Renamed from `Max` / `Min` (C1/N1).
```

### `trace` (renamed from `Trace`)

```
Trace (sum of the diagonal) over two axes.

Parameters
----------
axis_a : int, optional
    First axis; default 0.
axis_b : int, optional
    Second axis; default 1.

Returns
-------
Tensor
    A tensor with `axis_a`/`axis_b` contracted (rank-0 for a rank-2 input;
    confirmed by probe: trace of arange(9).reshape(3,3) over (0,1) == 12).

Notes
-----
Renamed from `Trace` (C1/N1). The recommended binding RESTORES the C++ default
arguments (a=0, b=1) and gives them meaningful names — the current binding
drops the defaults, so `Trace()` raises TypeError (Parity finding P2).
```

### `svd` (renamed from `Svd`)

```
Singular value decomposition of a rank-2 tensor.

Parameters
----------
is_UvT : bool, optional
    If True (default), also return U and vT; if False, return the singular
    values only.

Returns
-------
list of Tensor
    [U, S, vT] when is_UvT (confirmed by probe: length 3), else [S].

Notes
-----
Renamed from `Svd` (C1/N1).
```

### `eigh` (renamed from `Eigh`)

```
Eigen-decomposition of a Hermitian rank-2 tensor.

Parameters
----------
is_V : bool, optional
    If True (default), also return eigenvectors.
row_v : bool, optional
    Return eigenvectors as rows if True; default False.

Returns
-------
list of Tensor
    [eigvals, eigvecs] when is_V (confirmed by probe: length 2), else [eigvals].

Notes
-----
Renamed from `Eigh` (C1/N1).
```

## Change table

Clean-slate migration map: `current (C++ name / Python name) → recommended`.

| Current (C++ / Python) | Recommended | Why |
|---|---|---|
| `Abs` / `Abs_` | `abs` / `abs_` | N1 casing (C1) |
| `Conj` / `Conj_` | `conj` / `conj_` | N1 casing (C1) |
| `Exp` / `Exp_` | `exp` / `exp_` | N1 casing (C1) |
| `Inv` / `Inv_` | `inv` / `inv_` | N1 casing (C1) |
| `InvM` / `InvM_` | `inv_m` / `inv_m_` | N1/N3 casing (C1) |
| `Pow` / `Pow_` | `pow` / `pow_` | N1 casing (C1) |
| `Norm` | `norm` | N1 casing (C1) |
| `Max` / `Min` | `max` / `min` | N1 casing (C1) |
| `Trace` | `trace(axis_a=0, axis_b=1)` | N1 casing (C1) + restored defaults/names (P2) |
| `Svd` | `svd` | N1 casing (C1) |
| `Eigh` | `eigh` | N1 casing (C1) |
| `Save` / `Load` | `save` / `load` | N1 casing (C1) |
| `Tofile` / `Fromfile` | `to_file` / `from_file` | N1 casing (C1) |
| `Init` | `init` (demoted) | N1 casing (C1) + ctor duplication (C7) |
| `flatten_` (returns None) | `flatten_` (returns self) | N2 in-place return symmetry (C4) |
| `make_contiguous` | *(removed)* → use `contiguous` | leaked raw primitive (P3) |
| `astype_different_dtype` | *(removed)* → use `astype` | leaked raw primitive (P3) |
| `to_different_device` | *(removed)* → use `to` | leaked raw primitive (P3) |
| `cConj_` / `cExp_` / `cAbs_` / `cInv_` / `cInvM_` / `cPow_` | *(removed)* → use `conj_`/`exp_`/`abs_`/`inv_`/`inv_m_`/`pow_` | leaked raw in-place primitives (P3/C2) |
| `c__iadd__` / `c__isub__` / `c__imul__` / `c__itruediv__` / `c__ifloordiv__` / `c__ipow__` / `c__imatmul__` | *(removed)* → use `+=` / `-=` / `*=` / `/=` / `//=` / `**=` / `@=` | leaked raw operator primitives (P3/C2) |
| `__imatmul` *(Python-augmentation typo)* | `__imatmul__` (fixed) | `@=` is not in place today (P4) |
| *(bug)* `__getitem__` slice branch (`std::cout` leak) | *(remove the stray print)* | leftover debug output (P5) |
| *(none — unbound)* named arithmetic `add`/`sub`/`mul`/`div` (+ `_` forms) | bind them | B5 symmetry (P1) |

Every other public member of `Tensor` — `dtype`, `dtype_str`, `device`,
`device_str`, `shape`, `rank`, `is_contiguous`, `item`, `same_data`, `storage`,
`numpy`, `real`, `imag`, `permute`, `permute_`, `reshape`, `reshape_`,
`contiguous`, `contiguous_`, `flatten`, `append`, `astype`, `to`, `to_`,
`clone`, `from_storage`, `fill` — keeps both its current name and current
behavior unchanged. (`flatten_` keeps its name but changes its return value, and
`Trace` keeps its lowercased root but regains defaults — both appear in the
table above, not in this unchanged list, and are excluded here deliberately, not
by oversight.)
