# Tensor — 07. Type & device conversion

> **Superset-method rollout** (Tensor, category 07 of 8).
> The document is split into **Analysis** (the evidence — inventory, C++↔Python
> mapping, findings, arg ordering) and a self-contained **Recommendation** that
> is the *normative spec for the next version of Cytnx*: the next major version's
> `Tensor` type/device-conversion surface should be implemented to match §R
> exactly. Every behavioral claim is verified against the installed
> `cytnx==1.1.0` wheel by `docs/api-audit/probes/Tensor_07_typedevice.py` (all
> `[PASS]`, exit 0), with the raw-C++ side of the binding-fidelity finding (T-T1)
> verified by `probes/cpp/Tensor_07_typedevice.cpp` against the source-built
> `libcytnx.a` (all `[PASS]`, exit 0).

**Category scope.** The members that change a Tensor's **element type**
(`astype`), its **device** (`to`, `to_`), or that make an independent **deep
copy** (`clone`), plus the two leaked raw shims the `astype`/`to` wrappers call
(`astype_different_dtype`, `to_different_device`). The numpy bridge (`numpy`) and
the `from_storage` factory are **not** here — `numpy` lives in
[cat 03](03-element-storage-access.md) (element/storage access) and `from_storage`
in [cat 01](01-construction-init.md). C++ header:
`cytnx_src/include/Tensor.hpp:634` (`clone`), `:660` (`to`), `:683` (`to_`),
`:898` (`astype`). Python bindings: `cytnx_src/pybind/tensor_py.cpp:161-163`
(`clone`/`__copy__`/`__deepcopy__`), `:166-175` (`to_different_device`), `:177`
(`to_`), `:205-214` (`astype_different_dtype`). conti.py wrappers:
`cytnx_src/cytnx/Tensor_conti.py:36-41` (`to`), `:44-48` (`astype`).

This category is the **direct `Tensor` analog of [UniTensor cat 12](../UniTensor/12-type-device-conversion.md)**;
the `astype`/`to` `is self` short-circuit is the same binding-introduced identity
(cross-ref UT-T1), and the leaked `*_different_*` shims mirror UT-T2. Two of the
category's members diverge from their UniTensor counterparts and are corrected
against runtime truth below: `Tensor.to_` returns **None** (C++ `void to_`),
where `UniTensor.to_` returns self (T-T4); and `Tensor.to_`'s parameter is
correctly **named `device`**, where `UniTensor.to_`'s was erased to `arg0`
(T-T5, the positive contrast to UT-T3).

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `astype` | `astype(self, dtype)` (conti.py) | `Tensor` (**self** on a no-op, else new) | Element-type conversion. conti.py wrapper (`:44-48`): **returns `self`** if `dtype` is unchanged, else forwards to the raw `astype_different_dtype` shim (a distinct, independent copy of the requested dtype). Probe: *"`astype(same dtype)` returns `self` (is self) …"* + *"`astype(different dtype)` returns a DISTINCT object … whose data is COPIED (not same_data) …"*. |
| `to` | `to(self, device)` (conti.py) | `Tensor` (**self** on a no-op, else new) | Device move. conti.py wrapper (`:36-41`): **returns `self`** if `device` is unchanged, else forwards to the raw `to_different_device` shim. On this CPU-only wheel `to(cpu)` is always the no-op path. Probe: *"`to(same device)` returns `self` (is self) … On this CPU-only wheel `to(cpu)` is ALWAYS the no-op path"*. |
| `to_` `[I]` | `to_(self, device)` | **`None`** (in place) | **In-place** device move. C++ `void to_(const int&)` (`hpp:683`) is bound directly (`py:177`), so it returns **None**, NOT self — an in-place-return asymmetry (diverges from `UniTensor.to_`, which returns self). The parameter IS named `device` (bound with `py::arg("device")`). Probe: *"`to_(device)` moves in place and returns **None** (NOT self) …"* + *"`to_(device=...)` works … the parameter is correctly named `device` …"* + *"`to_(arg0=...)` RAISES TypeError …"*. |
| `clone` | `clone(self)` | `Tensor` (new, **independent**) | Deep copy: a distinct Tensor whose storage is **not** shared with the source. Also bound as `__copy__`/`__deepcopy__` (`py:161-163`). Probe: *"`clone()` returns an INDEPENDENT deep copy … (returns_view -> False)"* + *"`clone()` does NOT share storage … (not same_data)"*. |

**Internal / plumbing (leaks into `dir(Tensor)`):** `astype_different_dtype`
(the raw shim `astype` forwards to, `py:205-214`) and `to_different_device` (the
raw shim `to` forwards to, `py:166-175`). Each **hard-asserts the argument
differs** — called with an unchanged dtype/device it raises `RuntimeError` — and
exists *only* so the conti.py wrapper can intercept the no-op Python-side. Both
`.def`-ed as public methods. Probe: *"the raw plumbing bindings
`astype_different_dtype` / `to_different_device` both LEAK into public
dir(Tensor) …"* + *"`astype_different_dtype(same dtype)` / `to_different_device(same
device)` each RAISE RuntimeError on a no-op …"* + *"`astype_different_dtype(ComplexDouble)`
does the REAL conversion …"*.

## A2. C++ ↔ Python mapping

Status: `identical` · `signature-differs` · `leak`. `astype`/`to` are **conti.py
wrappers** over the leaked raw `astype_different_dtype`/`to_different_device`
bindings; `to_`/`clone` bind the C++ methods directly; `clone` is also bound as
`__copy__`/`__deepcopy__`. A binding that faithfully mirrors the C++ signature —
including `void`→`None`, `T&`→self, and by-value→a fresh wrapper — is
`identical`; `signature-differs` marks a binding-layer change to arity or
defaults.

| C++ (`Tensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `Tensor astype(const int&) const` (`:898`) | conti.py `astype` **+** raw `astype_different_dtype` (`py:205`) | **binding fidelity** / leak | conti.py adds the `is self` no-op short-circuit; C++ always returns a fresh object (T-T1); raw shim leaks (T-T2) |
| `Tensor to(const int&) const` (`:660`) | conti.py `to` **+** raw `to_different_device` (`py:166`) | **binding fidelity** / leak | same `is self` short-circuit (T-T1); raw shim leaks (T-T2) |
| `void to_(const int&)` (`:683`) | `to_` (`py:177`, `&Tensor::to_`, `py::arg("device")`) | identical | C++ returns `void` → Python `to_` returns **None**, not self (T-T4); param correctly named `device` (T-T5, the positive contrast to UT-T3) |
| `Tensor clone() const` (`:634`) | `clone` (`py:161`); also `__copy__` (`py:162`) / `__deepcopy__` (`py:163`) | identical | independent deep copy (T-T3) |
| raw `astype_different_dtype`/`to_different_device` | same names | **leak** | plumbing exposed publicly; raise on a no-op (T-T2) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/Tensor_07_typedevice.py` (on the 1.1.0 wheel). A **(binding
fidelity)** finding flags where the binding layer — a `Tensor_conti.py` wrapper or
a pybind lambda — changes behavior or signature versus the raw C++ method; **both
sides are runtime-verified**, the raw-C++ side by
`probes/cpp/Tensor_07_typedevice.cpp` (links against the source-built `libcytnx`,
GCC 13). Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **T-T1** | `astype`/`to` **short-circuit to `is self`** on a no-op conversion — a Python identity that raw **C++ does not have** (C++ always returns a *fresh* Tensor) | **binding fidelity** (B1 hazard) | **conti.py adds the short-circuit.** `cytnx/Tensor_conti.py:44-48` / `:36-41` define `astype`/`to` as `if self.dtype()==dtype: return self … else self.astype_different_dtype(dtype)` (and the device analogue), so `t.astype(t.dtype()) is t` and `t.to(t.device()) is t` — but C++ `Tensor::astype`/`to` (`hpp:660,898`) build `Tensor out; out._impl = …; return out` every call. Py probe *"`astype(same dtype)` returns `self` (is self) …"* + *"`to(same device)` returns `self` …"*; **C++ probe confirms** two no-op `astype(same)` calls return two DISTINCT objects (each `!= &self`) that share storage (same_data) — no `is self` — while a real `astype(ComplexDouble)` returns a distinct object that is NOT same_data (cross-ref UniTensor UT-T1) | **keep `astype`/`to`, but resolve the divergence explicitly.** Fold the no-op short-circuit into the **pybind lambda** (removing the leaked shims, T-T2) and **document** that Python returns `self` on a no-op — a deliberate, metadata-cheap optimization diverging from C++'s fresh-wrapper return; callers must not rely on `astype(x)`/`to(x)` producing a *new* object. |
| **T-T2** | `astype`/`to` are conti.py wrappers over the raw **`astype_different_dtype`** / **`to_different_device`** bindings, which **leak** into `dir` and **raise on a no-op** | naming + **binding fidelity** | **binding exposes plumbing + wraps it.** The real conversion is done by the raw shims `astype_different_dtype` (`py:205-214`) / `to_different_device` (`py:166-175`), each of which `cytnx_error_msg`s on a no-op ("*should be handle in python side*") — they exist *only* so the conti.py wrapper can call them after its short-circuit. Both are `.def`-ed as public methods. Py probe *"the raw plumbing bindings `astype_different_dtype` / `to_different_device` both LEAK into public dir(Tensor) …"* + *"… each RAISE RuntimeError on a no-op …"* + *"`astype_different_dtype(ComplexDouble)` does the REAL conversion …"* (cross-ref UniTensor UT-T2) | **remove `astype_different_dtype`/`to_different_device` from the public API** — inline the conversion (and the no-op short-circuit) into the `astype`/`to` pybind lambdas; keep `astype`/`to`. *Migration:* the raw shims were never a documented API; drop them (they only served the wrapper) — no user-facing alias needed, but ship one release where a `DeprecationWarning`-emitting stub forwards to `astype`/`to` for anyone who reached the raw name. |
| **T-T3** | `clone` produces an **independent** (deep) copy — not a view | (kept; verified) | **faithful deep copy.** `clone` binds C++ `Tensor::clone()` (`hpp:634`), which clones the storage; `__copy__`/`__deepcopy__` bind the same `clone` (`py:162-163`). Mutating the copy is not visible through the source. Py probe *"`clone()` returns an INDEPENDENT deep copy … (returns_view -> False)"* + *"`clone()` does NOT share storage … (not same_data)"*; **C++ probe confirms** `clone()` is a distinct object NOT same_data with the source | **keep** `clone`; **document** the deep-copy (independent-storage) semantics. |
| **T-T4** | `to_` returns **`None`**, not self — an in-place-return asymmetry (diverges from `UniTensor.to_`, which returns self) | **consistency** (N2 return-symmetry) | **C++ `to_` returns `void`.** `void to_(const int&)` (`hpp:683`) is bound directly (`py:177`), so the Python `to_` returns **None**. Its siblings `reshape_`/`permute_` return self (cat 04), and `UniTensor.to_` returns `UniTensor&` — so `t.to_(dev).dtype()` cannot chain. This is the same in-place-returns-None defect as cat-04's `flatten_` (finding T-S7). Py probe *"`to_(device)` moves in place and returns **None** (NOT self) — C++ `void to_` bound directly; diverges from UniTensor's self-returning `to_`"* + *"`to_` leaves the receiver on the target device (in-place move) …"* | **make `to_` return self** — change C++ `void to_` to `Tensor &to_` (returning `*this`, matching `UniTensor.to_` and the other in-place `_` methods), and bind the lambda to return `&self`. *Migration:* additive (a `None` return is not usefully consumed today); note the new chainable return in the changelog. |
| **T-T5** | `to_`'s parameter is correctly **named `device`** — `to_(device=…)` works | (kept; verified — positive) | **bound WITH `py::arg`.** `.def("to_", &cytnx::Tensor::to_, py::arg("device"))` (`py:177`) names the parameter, so `to_(device=cpu)` succeeds and `to_(arg0=…)` fails — the **opposite** of `UniTensor.to_`, whose param was erased to `arg0` (UT-T3). The positive demonstration of the parameter-consistency rule (PC1). Py probe *"`to_(device=...)` works — the parameter is correctly named `device` (py::arg present), UNLIKE UniTensor's erased `arg0`"* + *"`to_(arg0=...)` RAISES TypeError …"* | **keep** the `py::arg("device")` binding; carry it through the T-T4 self-returning rebinding. Use it as the reference pattern when fixing `UniTensor.to_` (UT-T3). |

## A4. Argument ordering — positional & keyword

These are converters/copiers; each takes at most a target dtype/device. There is
no keyword-only metadata block.

| API | positional-required (in order) | operation parameters (keyword-capable) |
|---|---|---|
| `astype` | `dtype` | *(none)* |
| `to` / `to_` | `device` | *(none)* |
| `clone` | *(none)* | *(none)* |

- **Canonical positional rule (§R.0):** the conversion target (`dtype`/`device`)
  is the primary operand and comes first — matches the live order; no reordering
  needed. Unlike `UniTensor.to_` (UT-T3), `Tensor.to_`'s parameter is already
  correctly named `device` (T-T5), so there is no naming fix here.

---

# R. Recommendation — normative spec for the next version

*This section is self-contained: it fully specifies the next-version `Tensor`
type/device-conversion surface. Implement Cytnx to match it. Findings above are
the rationale; they are not needed to implement §R.*

## R.0 Normative conventions (apply to every API in this category)

- **N-casing / N-underscore.** All members here are already correct lowercase
  snake_case; `to_` is the trailing-`_`/in-place form. The **`*_different_*`
  shims are rejected** as public API (T-T2) — they are plumbing the wrappers call.
- **In-place methods return `self` from the binding directly (T-T4).** `to_`
  moves in place; its C++ return type must become `Tensor&` (today `void`) so the
  pybind lambda returns `&self` — matching `reshape_`/`permute_` (cat 04) and
  `UniTensor.to_`. A bare `None` return breaks in-place chaining.
- **Parameters are named (T-T5).** `to_` is already bound with `py::arg("device")`
  — the correct pattern; keep it (and apply it to `UniTensor.to_`, UT-T3).
- **`clone` is an independent deep copy (T-T3).** Distinct storage; document it.
- **No-op conversion identity is documented, not silent (T-T1).** `astype`/`to`
  return `self` on a no-op (a metadata-cheap optimization) — this **diverges**
  from C++'s always-fresh-wrapper return; state it explicitly so callers do not
  rely on getting a new object. Fold the short-circuit into the pybind lambda
  (no leaked shims, T-T2).

## R.1 Recommended API (exact signatures + behavior contract)

```python
class Tensor:
    # --- element-type / device conversion (no-op -> self; else a new tensor) ---
    def astype(self, dtype: int) -> "Tensor": ...     # self if dtype unchanged, else independent copy
    def to(self, device: int) -> "Tensor": ...        # self if device unchanged, else new on target device
    def to_(self, device: int) -> "Tensor": ...       # in-place device move, self (make C++ to_ return Tensor&!)

    # --- copying (INDEPENDENT deep copy) ---
    def clone(self) -> "Tensor": ...                  # independent deep copy (also copy.copy / copy.deepcopy)
```

In-place `to_` returns `self` **from the binding** (change C++ `void to_` →
`Tensor &to_`); the no-op short-circuit on `astype`/`to` lives **in the pybind
lambda**, so the raw `astype_different_dtype`/`to_different_device` plumbing
bindings are **not** public members.

| API | Verdict | Behavior contract |
|---|---|---|
| `astype` | **keep** (T-T1/T2; document no-op `self`) | Element-type conversion: returns `self` if `dtype` is unchanged (a documented no-op optimization), else a new, **independent** Tensor of the requested dtype (data copied). *Migration:* fold the short-circuit + conversion into the pybind lambda; remove the leaked `astype_different_dtype` shim (behavior-preserving for `astype` callers — no deprecation needed for `astype` itself). |
| `to` | **keep** (T-T1/T2; document no-op `self`) | Device move: returns `self` if `device` is unchanged, else a new Tensor on the target device. *Migration:* fold into the pybind lambda; remove the leaked `to_different_device` shim. |
| `to_` | **keep, make it return self** (T-T4/T5) | In-place device move; must return **self** (today returns `None`). *Migration:* change C++ `void to_` → `Tensor &to_` (return `*this`) and bind the lambda to return `&self`; keep the existing `py::arg("device")`. Additive — a `None` return is not usefully consumed today; note the chainable return in the changelog. |
| `clone` | **keep** (T-T3) | Independent deep copy: a distinct Tensor whose storage is **not** shared with the source (also `copy.copy`/`copy.deepcopy`). |

**Internal / plumbing — hidden, not public API.** The two raw bindings below are
live public members today with a **remove** verdict: inline them into their pybind
lambda / bind under a private name. Neither carries a docstring — they are not
public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `astype_different_dtype` | **remove** (T-T1/T2) | Raw plumbing (the actual C++ `astype`) the conti.py `astype` wrapper calls after its no-op short-circuit; raises on a no-op. *Migration:* inline into the `astype` pybind lambda; no public exposure. Ship one release with a `DeprecationWarning`-emitting stub forwarding to `astype` for anyone who reached the raw name, then delete. |
| `to_different_device` | **remove** (T-T1/T2) | Raw plumbing (the actual C++ `to`) the conti.py `to` wrapper calls; raises on a no-op. *Migration:* inline into the `to` pybind lambda; no public exposure. Same one-release `DeprecationWarning` stub as above, then delete. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python surface,
**R.2b** Doxygen for the C++ surface. Only kept members are documented (removed
plumbing carries no docstring).

### R.2a Python API (numpy-style)

### `astype` / `to` / `to_`

```
Tensor.astype(dtype)  -> Tensor   # self if dtype unchanged, else a new copy
Tensor.to(device)     -> Tensor   # self if device unchanged, else new on target
Tensor.to_(device)    -> Tensor   # in-place device move, self

Convert this Tensor's element TYPE (astype) or DEVICE (to / to_).

`astype` returns a NEW, INDEPENDENT Tensor of the requested `dtype` (data
copied). `to` returns a NEW Tensor on the requested `device`. Both SHORT-CIRCUIT
and return `self` (the SAME object) when the conversion is a no-op (dtype/device
already matches) — a documented optimization; do NOT rely on `astype`/`to`
producing a distinct object (finding T-T1; cross-ref UniTensor UT-T1). `to_`
moves this tensor to `device` IN PLACE and returns self.

Parameters
----------
dtype : int
    Target element dtype (a `cytnx.Type.*` code), for `astype`.
device : int
    Target device (a `cytnx.Device.*` id), for `to` / `to_`.

Returns
-------
Tensor
    `astype`/`to`: self (no-op) or a new tensor. `to_`: self.

Notes
-----
Through cytnx 1.1.0 `astype`/`to` were conti.py wrappers over the raw
`astype_different_dtype` / `to_different_device` shims, which leaked into
`dir(Tensor)` and raised on a no-op (finding T-T2); the next version folds them
into the binding. `to_` returned `None` (C++ `void to_`), unlike its sibling
in-place methods and `UniTensor.to_` (finding T-T4); the next version makes it
return self. `to_`'s parameter is correctly named `device` (finding T-T5) — the
positive contrast to `UniTensor.to_`'s erased `arg0` (UT-T3).

See Also
--------
clone : make an independent deep copy without changing dtype/device.
```

### `clone`

```
Tensor.clone()          -> Tensor   # independent deep copy
copy.copy(Tensor)       -> Tensor   # __copy__  -> clone
copy.deepcopy(Tensor)   -> Tensor   # __deepcopy__ -> clone

Make an INDEPENDENT deep copy of this Tensor.

`clone` returns a distinct Tensor whose storage is NOT shared with the source —
mutating the copy is invisible through the original (finding T-T3). `__copy__`
(the `copy.copy` hook) and `__deepcopy__` (the `copy.deepcopy` hook) both defer to
`clone`.

Returns
-------
Tensor
    A new, independent tensor.
```

### R.2b C++ API (Doxygen)

C++ already returns `Tensor` by value for `astype`/`to`/`clone`; the next
version's changes are: (1) fold the no-op short-circuit + the conversion into the
`astype`/`to` **pybind lambdas** (removing the leaked
`astype_different_dtype`/`to_different_device` shims, T-T1/T2); (2) change C++
`void to_` → `Tensor &to_` so the in-place move returns self (T-T4), keeping the
`py::arg("device")` binding (T-T5). The `astype`/`to`/`clone` C++ methods are
otherwise unchanged.

```cpp
/**
 * @brief Convert this Tensor's element TYPE.
 * @details Returns a NEW, independent Tensor of dtype @p new_type (data copied),
 *          or *this when @p new_type is unchanged (a no-op). NOTE: the C++ method
 *          builds a fresh Tensor wrapper every call; the Python binding adds an
 *          `is self` short-circuit on the no-op (finding T-T1) and must fold the
 *          raw astype_different_dtype shim into the astype lambda (T-T2).
 * @param new_type target element dtype (a cytnx::Type code).
 * @return a Tensor of the requested dtype (a fresh copy, or *this on a no-op).
 */
Tensor astype(const int &new_type) const;

/**
 * @brief Move this Tensor to a DEVICE.
 * @details to() returns a NEW Tensor on @p device (or *this when unchanged);
 *          to_() moves in place. Fold the no-op short-circuit + the raw
 *          to_different_device shim into the to() pybind lambda (findings
 *          T-T1/T2). CHANGE the in-place form's return type from void to Tensor&
 *          so it returns *this (finding T-T4); keep the py::arg("device") binding
 *          (finding T-T5).
 * @param device target device id (a cytnx::Device id).
 * @return to: a Tensor on the target device. to_: reference to *this.
 */
Tensor to(const int &device) const;
Tensor &to_(const int &device);   // was: void to_(const int &device);

/**
 * @brief Make an INDEPENDENT deep copy of this Tensor.
 * @details clone() clones the storage (finding T-T3). The Python __copy__ and
 *          __deepcopy__ hooks both bind to clone directly.
 * @return a new, independent Tensor.
 */
Tensor clone() const;
```
