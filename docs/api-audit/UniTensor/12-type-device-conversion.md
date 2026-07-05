# UniTensor — 12. Type & device conversion

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the siblings
> [`05-structure-manipulation.md`](05-structure-manipulation.md),
> [`11-io-display.md`](11-io-display.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's type/device-conversion surface to match §R exactly. All runtime
> claims verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_12_typedevice.py` (all `[PASS]`, exit 0), with
> the raw-C++ side of the binding-fidelity finding (UT-T1) verified by
> `probes/cpp/UniTensor_12_typedevice.cpp` against a source-built `libcytnx`
> (all `[PASS]`, exit 0).

**Category scope.** The members that change a tensor's **element type**
(`astype`), its **device** (`to`, `to_`), that **copy** it (`clone`, and the
`__copy__`/`__deepcopy__` hooks behind `copy.copy`/`copy.deepcopy`), or that
**re-materialize** it in a different storage structure (`convert_from`), plus the
leaked raw shims those wrappers call (`astype_different_type`,
`to_different_device`, `cfrom`) and the missing **numpy bridge** (`.numpy()` /
`from_numpy`, absent — pairs with cat-01 `UT-C3`). C++ header:
`cytnx_src/include/UniTensor.hpp:3191-3219` (`to_`/`to`/`clone`), `:3569-3577`
(`astype`), `:5635-5638` (`convert_from`). Python bindings:
`cytnx_src/pybind/unitensor_py.cpp:499-528` (`to_`/`to_different_device`/`clone`/
`__copy__`/`__deepcopy__`/`astype_different_type`), `:1602-1604` (`cfrom`);
conti.py wrappers: `cytnx/UniTensor_conti.py:105-118` (`astype`/`to`), `:255-257`
(`convert_from`).

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place (returns self).

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `astype` | `astype(self, dtype)` (conti.py) | `UniTensor` (**self** on a no-op, else new) | Element-type conversion. conti.py wrapper: **returns `self`** if `dtype` is unchanged, else forwards to the raw `astype_different_type` shim (a distinct, independent copy). Probe: *"astype(same dtype) returns `self` (is self) …"* + *"astype(different dtype) … returns a DISTINCT, independent object …"*. |
| `to` | `to(self, device)` (conti.py) | `UniTensor` (**self** on a no-op, else new) | Device move. conti.py wrapper: **returns `self`** if `device` is unchanged, else forwards to the raw `to_different_device` shim. Probe: *"to(same device) returns `self` (is self) …"*. |
| `to_` `[I]` | `to_(self, arg0)` | `UniTensor` (self) | **In-place** device move; returns self. Bound as `.def("to_", &UniTensor::to_)` with **no `py::arg`**, so the parameter name is erased to `arg0` — keyword calls fail. Probe: *"to_ has an ERASED parameter name … to_(device=...) RAISES TypeError …"* + *"to_ called POSITIONALLY … returns self"*. |
| `clone` | `clone(self)` | `UniTensor` (new, **independent**) | Deep copy: a distinct object whose storage is **not shared** with the source. Probe: *"clone() returns an INDEPENDENT copy … (returns_view -> False)"* + *"clone() … does NOT share data … (not same_data)"*. |
| `__copy__` | `__copy__(self)` | `UniTensor` (new, independent) | `copy.copy` hook, **bound to `clone`** — an independent copy. Probe: *"copy.copy(ut) (the __copy__ hook, bound to clone) returns an INDEPENDENT copy …"*. |
| `__deepcopy__` | `__deepcopy__(self)` | `UniTensor` (new) / **raises** | `copy.deepcopy` hook, **also bound to `clone`** — but `clone` takes **no `memo`** parameter, so `copy.deepcopy(ut)` (which calls `__deepcopy__(self, memo)`) **raises `TypeError`**. Probe: *"__deepcopy__ is bound to `clone` with NO `memo` parameter …"* + *"copy.deepcopy(ut) RAISES TypeError …"* + *"ut.__deepcopy__({}) RAISES TypeError …"*. |
| `convert_from` `[I]` | `convert_from(self, Tin, force=False, tol=0)` (conti.py) | `UniTensor` (self) | **In-place** structure conversion: copy data from a UniTensor of a *different* storage structure (e.g. Dense→Block). conti.py wrapper over the raw `cfrom` binding; returns self. Probe: *"convert_from (conti.py wrapper over raw cfrom) copies data … IN PLACE and returns self"*. |

**Internal / plumbing (leak into `dir(UniTensor)`):** `astype_different_type`
(the raw shim `astype` forwards to), `to_different_device` (the raw shim `to`
forwards to), and `cfrom` (the raw `convert_from`) — plumbing that should never
be public. Probe: *"the raw plumbing bindings astype_different_type /
to_different_device / cfrom all LEAK into public dir(UniTensor)"*.

**Capability gap (numpy bridge absent).** UniTensor exposes **no** `.numpy()`
method and **no** `from_numpy` static — there is no way to round-trip a
(dense) UniTensor through a `numpy.ndarray`. The module-level `cytnx.from_numpy`
builds a `Tensor`, not a UniTensor. Probe: *"UniTensor has NO `.numpy()` method
and NO `from_numpy` static …"* + *"the numpy bridge exists only at MODULE level
… never on UniTensor …"*. Pairs with cat-01 `UT-C3`.

## A2. C++ ↔ Python mapping

`astype`/`to`/`convert_from` are **conti.py wrappers** over leaked raw bindings
(`astype_different_type`/`to_different_device`/`cfrom`); `to_`/`clone` bind the
C++ methods directly; `__copy__`/`__deepcopy__` are both bound to the C++
`clone`.

| C++ (`UniTensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `UniTensor astype(dtype) const` (`:3569`) | conti.py `astype` **+** raw `astype_different_type` (`py:519`) | **binding fidelity** / leak | conti.py adds the `is self` no-op short-circuit; C++ always returns a fresh object (UT-T1); raw shim leaks (UT-T2) |
| `UniTensor to(device) const` (`:3205`) | conti.py `to` **+** raw `to_different_device` (`py:500`) | **binding fidelity** / leak | same `is self` short-circuit (UT-T1); raw shim leaks (UT-T2) |
| `UniTensor &to_(device)` (`:3191`) | `to_` (`py:499`, `&UniTensor::to_`) | **signature-differs** | bound with no `py::arg` → param name erased to `arg0` (UT-T3) |
| `UniTensor clone() const` (`:3215`) | `clone` (`py:510`) | identical | independent deep copy (UT-T4) |
| `UniTensor clone() const` (`:3215`) | `__copy__` (`py:511`, bound to `clone`) | identical | `copy.copy` → independent copy (UT-T4) |
| `UniTensor clone() const` (`:3215`) | `__deepcopy__` (`py:512`, bound to `clone`) | **signature-differs** (correctness) | `clone` has no `memo` param → `copy.deepcopy` raises (UT-T5) |
| `UniTensor &convert_from(rhs, force, tol)` (`:5635`) | conti.py `convert_from` **+** raw `cfrom` (`py:1602`) | **leak** | raw `cfrom` leaks; wrapper re-adds return-self (UT-T7) |
| raw `astype_different_type`/`to_different_device`/`cfrom` | same names | **leak** | plumbing exposed publicly (UT-T2/T7) |
| *(none — absent both)* | `.numpy()` / `from_numpy` *(absent)* | **gap** | no numpy bridge on UniTensor (UT-T6; pairs cat-01 UT-C3) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/UniTensor_12_typedevice.py` (on the 1.1.0 wheel). A **(binding
fidelity)** finding flags where the binding layer — a `*_conti.py` wrapper or a
pybind lambda — changes behavior or signature versus the raw C++ method; **both
sides are runtime-verified**, the raw-C++ side by
`probes/cpp/UniTensor_12_typedevice.cpp` (links against a source-built
`libcytnx`, GCC 13). Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-T1** | `astype`/`to` **short-circuit to `is self`** on a no-op conversion — a Python identity that raw **C++ does not have** (C++ always returns a *fresh* UniTensor wrapper) | **binding fidelity** (B1 hazard) | **conti.py adds the short-circuit.** `cytnx/UniTensor_conti.py:105-118` defines `astype`/`to` as `if self.dtype()==dtype: return self … else self.astype_different_type(dtype)` (and the device analogue). So `u.astype(u.dtype()) is u` and `u.to(u.device()) is u` — but C++ `UniTensor::astype`/`to` (`hpp:3205,3569`) build `UniTensor out; out._impl = …; return out` every time. Py probe *"astype(same dtype) returns `self` (is self) …"* + *"to(same device) returns `self` …"*; **C++ probe confirms** two no-op `astype(same)` calls return two DISTINCT objects (each `!= &self`) sharing storage — no `is self` (while the real `astype(ComplexDouble)` returns a distinct object that is NOT `same_data`) | **keep `astype`/`to`, but resolve the divergence explicitly.** Fold the no-op short-circuit into the **pybind lambda** (removing the leaked shims, UT-T2) and **document** that Python returns `self` on a no-op — a deliberate, metadata-cheap optimization diverging from C++'s fresh-wrapper return; callers must not rely on `astype(x)`/`to(x)` producing a *new* object. (Alternative if strict C++ parity is wanted: always return a fresh view — but that costs a wrapper alloc for no benefit.) |
| **UT-T2** | `astype`/`to` are conti.py wrappers over the raw **`astype_different_type`** / **`to_different_device`** bindings, which **leak** into `dir` | naming + **binding fidelity** | **binding exposes plumbing + wraps it.** The real conversion is done by the raw shims `astype_different_type` (`py:519`) / `to_different_device` (`py:500`), each of which `cytnx_error_msg`s on a no-op ("*should be handled on the Python side*") — they exist *only* so the conti.py wrapper can call them after its short-circuit. Both are `.def`-ed as public methods. Py probe *"astype/to are conti.py wrappers: the raw astype_different_type / to_different_device shims … LEAK into public dir(UniTensor)"* | **remove `astype_different_type`/`to_different_device` from the public API** — inline the conversion (and the no-op short-circuit) into the `astype`/`to` pybind lambdas; keep `astype`/`to`. *Migration:* the raw shims were never a documented API; drop them (they only served the wrapper). |
| **UT-T3** | `to_`'s parameter name is **erased to `arg0`** — `to_(device=…)` cannot be called by keyword | naming (PC1) | **bound without `py::arg`.** `.def("to_", &UniTensor::to_)` (`py:499`) binds the raw method pointer with no argument annotation, so the signature is `to_(self, arg0)` and `to_(device=cpu)` raises `TypeError`. Py probe *"to_ has an ERASED parameter name … to_(device=...) RAISES TypeError — the param is `arg0`"* + *"to_ called POSITIONALLY … returns self"* | **add `py::arg("device")`** so `to_(device=…)` works — the parameter-consistency fix (PC1). *Migration:* additive (positional calls unaffected); no deprecation needed. |
| **UT-T4** | `clone` and `__copy__` produce an **independent** (deep) copy — not a view | (kept; verified) | **faithful deep copy.** `clone` binds C++ `UniTensor::clone()` (`hpp:3215`), which clones the storage; `__copy__` is bound to the same `clone` (`py:510-511`), so `copy.copy(ut)` is independent. Mutating the copy is not visible through the source. Py probe *"clone() returns an INDEPENDENT copy … (returns_view -> False)"* + *"clone() … not same_data"* + *"copy.copy(ut) … returns an INDEPENDENT copy …"*; **C++ probe confirms** `clone()` is a distinct object NOT `same_data` with the source | **keep** `clone`/`__copy__`; **document** the deep-copy (independent-storage) semantics. |
| **UT-T5** | `copy.deepcopy(ut)` **raises `TypeError`** — `__deepcopy__` is bound to `clone`, which takes **no `memo`** parameter | **correctness** (broken hook) | **wrong arity.** `__deepcopy__` is `.def`-ed as the C++ `clone` (`py:512`), whose signature is `__deepcopy__(self) -> UniTensor`; `copy.deepcopy` calls `obj.__deepcopy__(self, memo)`, so the extra `memo` argument makes it raise `TypeError`. `copy.copy` (which calls `__copy__(self)` with no memo) works, but `copy.deepcopy` does not. Py probe *"__deepcopy__ is bound to `clone` with NO `memo` parameter …"* + *"copy.deepcopy(ut) RAISES TypeError …"* + *"ut.__deepcopy__({}) RAISES TypeError …"* (cross-ref cat 11 UT-IO2, which observes the same `deepcopy` failure) | **fix `__deepcopy__` to accept `memo` and actually deep-copy** — bind a lambda `[](const UniTensor& self, py::dict) { return self.clone(); }` (accepting and ignoring `memo`) so `copy.deepcopy(ut)` returns an independent clone. *Migration:* behavior fix only (currently raises); note in the changelog. |
| **UT-T6** | there is **no numpy bridge** on UniTensor — `.numpy()` and `from_numpy` are both **absent** | **capability gap** (vs `Tensor`) | **nothing bound.** `"numpy" not in dir(UniTensor)` and `"from_numpy" not in dir(UniTensor)`; the module-level `cytnx.from_numpy` builds a `Tensor`, not a UniTensor, so there is no ndarray round-trip for a UniTensor. Py probe *"UniTensor has NO `.numpy()` method and NO `from_numpy` static …"* + *"the numpy bridge exists only at MODULE level … never on UniTensor …"* | **add `.numpy()` (dense→ndarray) and `from_numpy` (ndarray→UniTensor)** as the inverse pair, pairing with the cat-01 `UT-C3` `from_numpy` constructor. `from_numpy` is the static constructor (cat 01); `.numpy()` lives here (this category) as the export half. Both copy data. |
| **UT-T7** | `convert_from` is a conti.py wrapper over the raw **`cfrom`** binding, which **leaks** into `dir` | naming + **binding fidelity** | **binding exposes plumbing + wraps it.** Raw C++ `convert_from` is bound as `cfrom` (`py:1602`, returning `void`); `cytnx/UniTensor_conti.py:255-257` defines `convert_from` as `self.cfrom(Tin, force, tol); return self`. The `c`-prefixed spelling is a reserved raw-binding name (§R.0 rejects it). Py probe *"the raw plumbing bindings … cfrom all LEAK …"* + *"convert_from … copies data … IN PLACE and returns self"* | **remove `cfrom` from the public API** — bind `convert_from` directly (returning self / `UniTensor&`) so the conti.py shim and the leaked `cfrom` disappear; keep `convert_from`. *Migration:* `cfrom` was never a documented API; fold into the `convert_from` lambda. |

## A4. Argument ordering — positional & keyword

These are converters/copiers; each takes at most a target dtype/device or a
source tensor. There is no keyword-only metadata block.

| API | positional-required (in order) | operation parameters (keyword-capable) |
|---|---|---|
| `astype` | `dtype` | *(none)* |
| `to` / `to_` | `device` | *(none)* |
| `clone` / `__copy__` / `__deepcopy__` | *(none)* | *(none)* |
| `convert_from` | `Tin` | `force=False`, `tol=0` |
| `numpy` (→ add) | *(none)* | *(none)* |
| `from_numpy` (→ add, static) | `array` | `labels=[]`, `rowrank=-1`, `name=''` (keyword-only metadata, cat 01) |

- **Canonical positional rule (§R.0):** the conversion target (`dtype`/`device`)
  or the source (`Tin`/`array`) is the primary operand and comes first — matches
  the live order; no reordering needed. The only naming fix is `to_`'s erased
  `arg0` → `device` (UT-T3).
- `convert_from`'s `force`/`tol` are operation parameters (keyword-capable),
  following the source operand; keep them positional-optional with keyword
  defaults.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version type/device-conversion surface.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing / N-underscore.** All members here are already correct lowercase
  snake_case; `to_`/`convert_from` are the trailing-`_`/in-place forms (return
  self). The only naming defect is the **erased `arg0`** on `to_` → `device`
  (UT-T3). The **`c`-prefixed raw spelling `cfrom` and the `*_different_*` shims
  are rejected** as public API (UT-T2/T7) — they are plumbing the wrappers call.
- **In-place methods return `self` from the binding directly.** `to_` and
  `convert_from` return self in C++ (`UniTensor&`); the pybind lambda must return
  it too, so the conti.py return-self shim (and the leaked `cfrom` binding) for
  `convert_from` disappears (UT-T7).
- **Copy hooks are complete and correct (UT-T4/T5).** `clone`/`__copy__` deep-copy
  (independent storage); `__deepcopy__` must accept `memo` and deep-copy so
  `copy.deepcopy(ut)` works (today it raises). A class that binds `__deepcopy__`
  must honor the `(self, memo)` protocol.
- **No-op conversion identity is documented, not silent (UT-T1).** `astype`/`to`
  return `self` on a no-op (a metadata-cheap optimization) — this **diverges**
  from C++'s always-fresh-wrapper return; state it explicitly so callers do not
  rely on getting a new object. Fold the short-circuit into the pybind lambda
  (no leaked shims, UT-T2).
- **numpy bridge is a first-class pair (UT-T6).** `from_numpy` (static
  constructor, cat 01) and `.numpy()` (export, here) are inverses; both copy.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- element-type / device conversion (no-op -> self; else a new tensor) ---
    def astype(self, dtype: int) -> "UniTensor": ...     # self if dtype unchanged, else independent copy
    def to(self, device: int) -> "UniTensor": ...        # self if device unchanged, else new on target device
    def to_(self, device: int) -> "UniTensor": ...       # in-place device move, self (bind py::arg("device")!)

    # --- copying (all INDEPENDENT deep copies) ---
    def clone(self) -> "UniTensor": ...                  # independent deep copy
    def __copy__(self) -> "UniTensor": ...               # copy.copy hook -> clone
    def __deepcopy__(self, memo) -> "UniTensor": ...     # copy.deepcopy hook -> clone (MUST accept memo)

    # --- structure conversion (in-place, self) ---
    def convert_from(self, Tin: "UniTensor", force: bool = False, tol: float = 0.0) -> "UniTensor": ...

    # --- numpy bridge (ADD; pairs with the cat-01 from_numpy constructor) ---
    def numpy(self) -> "numpy.ndarray": ...              # dense export (copy)
    @staticmethod
    def from_numpy(array, *, labels=[], rowrank=-1, name='') -> "UniTensor": ...  # cat 01
```

In-place methods (`to_`, `convert_from`) return `self` **from the binding** (no
conti.py shim); the no-op short-circuit on `astype`/`to` lives **in the pybind
lambda**, so the raw `astype_different_type`/`to_different_device`/`cfrom`
plumbing bindings are **not** public members. `__deepcopy__` accepts `memo`.

| API | Verdict | Behavior contract |
|---|---|---|
| `astype` | **keep** (UT-T1/T2; document no-op `self`) | Element-type conversion: returns `self` if `dtype` is unchanged (a documented no-op optimization), else a new, **independent** UniTensor of the requested dtype (data copied). *Migration:* fold the short-circuit + conversion into the pybind lambda; remove the leaked `astype_different_type` shim. |
| `to` | **keep** (UT-T1/T2; document no-op `self`) | Device move: returns `self` if `device` is unchanged, else a new UniTensor on the target device. *Migration:* fold into the pybind lambda; remove the leaked `to_different_device` shim. |
| `to_` | **keep, add `py::arg("device")`** (UT-T3) | In-place device move; returns self. *Migration:* bind with `py::arg("device")` so `to_(device=…)` works (currently the param is `arg0`); additive, no deprecation. |
| `clone` | **keep** (UT-T4) | Independent deep copy: a distinct UniTensor whose storage is **not** shared with the source. |
| `__copy__` | **keep** (UT-T4) | `copy.copy` hook (bound to `clone`) → an independent copy. |
| `__deepcopy__` | **fix** (UT-T5) | `copy.deepcopy` hook: must accept `memo` and return an independent clone. *Migration:* rebind as `[](const UniTensor& self, py::dict memo){ return self.clone(); }`; currently bound to `clone` (no `memo`), so `copy.deepcopy(ut)` **raises `TypeError`** — behavior fix, note in changelog. |
| `convert_from` | **keep, bind self directly** (UT-T7) | In-place: copy data from a different-structure UniTensor (`force`/`tol` control lossy conversion); returns self. *Migration:* the conti.py wrapper over `cfrom` is removed; the pybind lambda returns self. |
| `numpy` | **add** (UT-T6) | Export a **dense** UniTensor's block to a `numpy.ndarray` (copy). Inverse of `from_numpy` (cat 01). Errors on a symmetric/Block tensor (no dense array). |
| `from_numpy` `[static]` | **add** (UT-T6; owned by cat 01) | Build a Dense UniTensor from a `numpy.ndarray` (copy); dtype mapped from `array.dtype`. Documented in cat-01 `UT-C3`; listed here as the inverse of `.numpy()`. |

**Internal / plumbing — hidden, not public API.** The three raw bindings below
are covered here (they are live public members today) with a **remove** verdict:
inline them into their pybind lambda / bind under a private name. None carry a
docstring — they are not public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `astype_different_type` | **remove** (UT-T1/T2) | Raw plumbing (the actual C++ `astype`) the conti.py `astype` wrapper calls after its no-op short-circuit. *Migration:* inline into the `astype` pybind lambda; no public exposure. |
| `to_different_device` | **remove** (UT-T1/T2) | Raw plumbing (the actual C++ `to`) the conti.py `to` wrapper calls. *Migration:* inline into the `to` pybind lambda; no public exposure. |
| `cfrom` | **remove** (UT-T7) | Raw plumbing (the actual C++ `convert_from`) the conti.py `convert_from` wrapper calls. *Migration:* fold into the `convert_from` pybind lambda (which returns self); no public exposure. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept/added/fixed members are
documented (removed plumbing members carry no docstring).

### R.2a Python API (numpy-style)

### `astype` / `to` / `to_`

```
UniTensor.astype(dtype)  -> UniTensor   # self if dtype unchanged, else a new copy
UniTensor.to(device)     -> UniTensor   # self if device unchanged, else new on target
UniTensor.to_(device)    -> UniTensor   # in-place device move, self

Convert this UniTensor's element TYPE (astype) or DEVICE (to / to_).

`astype` returns a NEW, INDEPENDENT UniTensor of the requested `dtype` (data
copied). `to` returns a NEW UniTensor on the requested `device`. Both SHORT-
CIRCUIT and return `self` (the SAME object) when the conversion is a no-op
(dtype/device already matches) — a documented optimization; do NOT rely on
`astype`/`to` producing a distinct object (finding UT-T1). `to_` moves this
tensor to `device` IN PLACE and returns self.

Parameters
----------
dtype : int
    Target element dtype (a `cytnx.Type.*` code), for `astype`.
device : int
    Target device (a `cytnx.Device.*` id), for `to` / `to_`.

Returns
-------
UniTensor
    `astype`/`to`: self (no-op) or a new tensor. `to_`: self.

Notes
-----
Through cytnx 1.1.0 `astype`/`to` were conti.py wrappers over the raw
`astype_different_type` / `to_different_device` shims, which leaked into
`dir(UniTensor)` (finding UT-T2); the next version folds them into the binding.
`to_`'s parameter was bound unnamed (`arg0`), so `to_(device=…)` raised — the
next version names it `device` (finding UT-T3).

See Also
--------
clone : make an independent deep copy without changing dtype/device.
```

### `clone` / `__copy__` / `__deepcopy__`

```
UniTensor.clone()          -> UniTensor   # independent deep copy
copy.copy(UniTensor)       -> UniTensor   # __copy__  -> clone
copy.deepcopy(UniTensor)   -> UniTensor   # __deepcopy__ -> clone (next version)

Make an INDEPENDENT deep copy of this UniTensor.

`clone` returns a distinct UniTensor whose storage is NOT shared with the source
— mutating the copy is invisible through the original (finding UT-T4). `__copy__`
(the `copy.copy` hook) and `__deepcopy__` (the `copy.deepcopy` hook) both defer
to `clone`.

Returns
-------
UniTensor
    A new, independent tensor.

Notes
-----
Through cytnx 1.1.0 `__deepcopy__` was bound to `clone` with NO `memo`
parameter, so `copy.deepcopy(ut)` RAISED `TypeError` (finding UT-T5; cross-ref
cat 11 UT-IO2) while `copy.copy(ut)` worked. The next version's `__deepcopy__`
accepts `memo` and returns an independent clone, so `copy.deepcopy` works.
```

### `convert_from`

```
UniTensor.convert_from(Tin, force=False, tol=0.0)  -> UniTensor   # in-place, self

Copy data into this UniTensor from another UniTensor of a DIFFERENT storage
structure (e.g. fill a Block/symmetric tensor from a Dense one, or vice versa),
IN PLACE; returns self (finding UT-T7).

Parameters
----------
Tin : UniTensor
    The source tensor to copy data from; its shape must match this tensor.
force : bool, optional
    If True, do not strictly check structure compatibility — elements absent in
    the destination structure are dropped even if nonzero (default False).
tol : float, optional
    When converting a denser tensor into a sparser structure, an element that has
    no slot in the destination must have absolute value <= `tol`, else an error
    is raised. Overridden by `force=True` (default 0.0).

Returns
-------
UniTensor
    self.

Notes
-----
Renamed nothing, but the raw `cfrom` binding it wrapped is removed (folded into
`convert_from`, finding UT-T7).
```

### `numpy` (new)

```
UniTensor.numpy()  -> numpy.ndarray    # dense export (copy)

Export a DENSE UniTensor's block to a numpy.ndarray (data copied). The inverse of
`UniTensor.from_numpy` (the cat-01 static constructor); together they are the
numpy bridge that 1.1.0 lacks entirely (finding UT-T6; pairs cat-01 UT-C3).

Returns
-------
numpy.ndarray
    A copy of this tensor's elements, shaped like `self.shape()`.

Raises
------
Errors on a symmetric / Block UniTensor (no single dense array); densify with
`to_dense()` (cat 05) first, or export block-by-block.

See Also
--------
from_numpy : build a UniTensor from a numpy.ndarray (cat 01).
```

### R.2b C++ API (Doxygen)

C++ already returns `UniTensor`/`UniTensor&` per the N-underscore split; the
next version's changes are: (1) fold the no-op short-circuit + the conversion into
the `astype`/`to` **pybind lambdas** (removing the leaked
`astype_different_type`/`to_different_device` shims, UT-T1/T2); (2) bind `to_`
with `py::arg("device")` (UT-T3); (3) bind `__deepcopy__` to a lambda accepting
`memo` (UT-T5); (4) bind `convert_from` directly, returning self (removing the
leaked `cfrom`, UT-T7); (5) add the `.numpy()` export (UT-T6). The C++ methods
themselves are unchanged.

```cpp
/**
 * @brief Convert this UniTensor's element TYPE.
 * @details Returns a NEW, independent UniTensor of dtype @p dtype (data copied),
 *          or *this when @p dtype is unchanged (a no-op). NOTE: the C++ method
 *          builds a fresh UniTensor wrapper every call; the Python binding adds
 *          an `is self` short-circuit on the no-op (finding UT-T1) and must fold
 *          the raw astype_different_type shim into the astype lambda (UT-T2).
 * @param dtype target element dtype (a cytnx::Type code).
 * @return a UniTensor of the requested dtype (a fresh copy, or *this on a no-op).
 */
UniTensor astype(const unsigned int &dtype) const;

/**
 * @brief Move this UniTensor to a DEVICE.
 * @details to() returns a NEW UniTensor on @p device (or *this when unchanged);
 *          to_() moves in place and returns *this. Fold the no-op short-circuit
 *          + the raw to_different_device shim into the to() pybind lambda
 *          (findings UT-T1/T2); bind to_ with py::arg("device") so the keyword
 *          call works (finding UT-T3).
 * @param device target device id (a cytnx::Device id).
 * @return to: a UniTensor on the target device. to_: reference to *this.
 */
UniTensor to(const int &device) const;
UniTensor &to_(const int &device);

/**
 * @brief Make an INDEPENDENT deep copy of this UniTensor.
 * @details clone() clones the storage (finding UT-T4). The Python __copy__ hook
 *          binds to clone directly; the Python __deepcopy__ hook MUST bind to a
 *          lambda that ACCEPTS a `memo` dict and returns clone(), otherwise
 *          copy.deepcopy() raises TypeError as it does through 1.1.0
 *          (finding UT-T5).
 * @return a new, independent UniTensor.
 */
UniTensor clone() const;

/**
 * @brief Copy data from a UniTensor of a DIFFERENT storage structure, IN PLACE.
 * @details Fills *this from @p rhs (e.g. Dense<->Block), returning *this. The
 *          Python binding must expose this directly (removing the leaked raw
 *          `cfrom` binding, finding UT-T7).
 * @param rhs   source tensor (shape must match).
 * @param force skip strict structure-compatibility checks (drop absent elements).
 * @param tol   max |element| that may be dropped when densifying->sparser.
 * @return reference to *this.
 */
UniTensor &convert_from(const UniTensor &rhs, bool force = false, cytnx_double tol = 0.);
```

The numpy export (`.numpy()`) and import (`from_numpy`, cat 01) are **Python-only**
(there is no C++ numpy bridge, finding UT-T6) — no C++ Doxygen entry.
```
