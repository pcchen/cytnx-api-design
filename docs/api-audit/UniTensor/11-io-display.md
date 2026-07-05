# UniTensor — 11. I/O & display

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the siblings
> [`06-element-block-access.md`](06-element-block-access.md),
> [`10-contraction-networks.md`](10-contraction-networks.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's UniTensor I/O & display surface to match §R exactly. All runtime
> claims verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_11_io.py` (all `[PASS]`, exit 0).
> **No C++ probe accompanies this category.** The two divergences here are a
> *broken pickle protocol* (UT-IO2) and a *name-serialization bug* (UT-IO5); the
> first is a pure-Python/pybind fact (no `py::pickle` registered) and the second
> lives in the **shared C++** `UniTensor::_Load` (`std::string(cname)` heap
> over-read) and manifests **identically** through the binding — neither is a
> `conti.py`-vs-raw *binding* divergence, so gate 4 is recorded as *"no
> binding-fidelity finding"*.

**Category scope.** The members that serialize a UniTensor to/from disk (`Save`
member, `Load` static constructor) and that render it for a human (`print_diagram`,
`print_block`, `print_blocks`, and the `__repr__` hook), plus the half-present
**pickle protocol** (`__getstate__` inherited from `object`; `__setstate__`
absent). The general element-value display via `at`/`item` is cat 06; the
copy/clone hooks `__copy__`/`__deepcopy__` are cat 12 (cross-referenced from
UT-IO2). C++ header: `cytnx_src/include/UniTensor.hpp` (`Save`/`Load` decls),
`cytnx_src/src/UniTensor.cpp:150-182` (`Save`/`Load` bodies, incl. the `_Load`
name over-read at `:119-126`). Python bindings:
`cytnx_src/pybind/unitensor_py.cpp:513-516` (`Save`/`Load`), `:586-591`
(`print_*`), `:728-733` (`__repr__`).

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[static]` = a static constructor (called
on the class, not an instance).

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `Save` | `Save(self, fname: str)` | `None` | **Member** serializer: write this tensor to `fname` as a binary `.cytnx` file. If `fname` has no extension, appends `.cytnx` with a **deprecation warning** (`hpp`/`UniTensor.cpp:150-168`). Probe: *"`Save` writes the tensor to the given path (file created on disk)"* + *"`Save` with no extension appends `.cytnx` (deprecated) …"*. |
| `Load` `[static]` | `Load(fname: str) -> UniTensor` | `UniTensor` (new) | **Static** deserializer: read a `.cytnx` file back into a fresh UniTensor. Opens `fname` **verbatim** — does **not** auto-append `.cytnx` (`UniTensor.cpp:171-181`). Probe: *"`Load` is a UniTensor STATIC member (Load(fname) -> UniTensor)"* + *"`Save`->`Load` round-trip is value-equal …"*. |
| `print_diagram` | `print_diagram(self, bond_info=False)` | `None` | Render the tensor as an ASCII leg diagram (labels, bond dims, rowrank split); `bond_info=True` adds per-bond detail. Output **is** capturable via `contextlib.redirect_stdout` (has a `py::scoped_ostream_redirect` guard, `:586-587`). Probe: *"`print_diagram()` returns None and its output IS capturable …"*. |
| `print_blocks` | `print_blocks(self, full_info=True)` | `None` | Render **all** storage blocks (Block/Dense); `full_info` toggles verbosity. Capturable via `redirect_stdout` (`:588-589`). Probe: *"`print_blocks()` returns None and its output IS capturable …"*. |
| `print_block` | `print_block(self, idx, full_info=True)` | `None` | Render **one** block by index. Capturable via `redirect_stdout` (`:590-591`). Probe: *"`print_block(idx)` returns None and its output IS capturable …"*. |
| `__repr__` | `__repr__(self)` | `str` (**always `''`**) | Python repr hook: **prints** the full tensor via C++ `std::cout << self` and **returns the empty string** — a print-and-return-`''` pattern (like Bond/Symmetry). The printed text **is** capturable (`py::scoped_ostream_redirect`, `:728-733`). Probe: *"`__repr__` PRINTS … and RETURNS '' …"* + *"`__repr__`'s printed output IS capturable …"*. |
| `__getstate__` | `__getstate__(self)` → `None` | `None` | **Inherited `object.__getstate__`** (doc `'Helper for pickle.'`), *not* a cytnx pickle stub. Present in `dir` but does nothing useful. Probe: *"`hasattr(UniTensor,'__getstate__') is True` — but it is the DEFAULT `object.__getstate__` …"*. |
| `__setstate__` | *(absent)* | — | **Not defined** — no pickle restore hook. Probe: *"`hasattr(UniTensor,'__setstate__') is False` …"*. |

**Broken pickle (behavioral).** Despite `__getstate__` existing, `pickle.dumps(ut)`
**raises** `TypeError: cannot pickle 'cytnx.cytnx.UniTensor' object` — pybind11
registered **no** `py::pickle(...)` support, so the object is unpicklable
regardless of the inherited getstate. Probe: *"`pickle.dumps(ut)` RAISES TypeError
… pickle is a broken stub, not a working protocol"*. `copy.deepcopy(ut)` **also**
raises `TypeError` — but for an unrelated reason (its `__deepcopy__` is bound to
`clone` with no `memo` parameter, **cat 12**), so it never even reaches the pickle
fallback. Probe: *"`copy.deepcopy(ut)` also RAISES TypeError … (cross-ref cat 12)"*.

## A2. C++ ↔ Python mapping

`Save`/`Load` are thin pybind lambdas forwarding to the C++ `UniTensor::Save`/
`UniTensor::Load`; the `print_*` methods bind the C++ methods **directly** with a
`scoped_ostream_redirect` guard; `__repr__` is a small lambda over `operator<<`.
There is **no** `conti.py` wrapper and **no** leaked `c*` binding in this category.

| C++ (`UniTensor.hpp` / `UniTensor.cpp`) | Python | Status | Note |
|---|---|---|---|
| `void UniTensor::Save(const std::string &fname) const` (`cpp:150`) | `Save(fname)` (lambda, `unitensor_py.cpp:513`) | identical | member; Capitalized → should be `save` (UT-IO1); no-extension warning (UT-IO4) |
| `static UniTensor UniTensor::Load(const std::string &fname)` (`cpp:171`) | `Load(fname)` (`def_static` lambda, `:515`) | identical | static constructor; Capitalized → should be `load` (UT-IO1); `_Load` name over-read (UT-IO5) |
| `void print_diagram(const bool &bond_info=false)` | `print_diagram(bond_info=False)` (`:586`) | identical | `scoped_ostream_redirect` → capturable (UT-IO6) |
| `void print_blocks(const bool &full_info=true) const` | `print_blocks(full_info=True)` (`:588`) | identical | `scoped_ostream_redirect` → capturable (UT-IO6) |
| `void print_block(const cytnx_int64 &idx, const bool &full_info=true) const` | `print_block(idx, full_info=True)` (`:590`) | identical | `scoped_ostream_redirect` → capturable (UT-IO6) |
| `std::ostream &operator<<(std::ostream &, const UniTensor &)` | `__repr__` (lambda `std::cout << self; return ""`, `:728`) | **signature-differs** | prints & returns `''` (UT-IO7); capturable |
| *(none — pybind default)* | `__getstate__` (inherited `object`) | **leak / stub** | default `object.__getstate__`; no `py::pickle` (UT-IO2) |
| *(none)* | `__setstate__` *(absent)* | **gap** | no restore hook → pickle unusable (UT-IO2) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/UniTensor_11_io.py` (on the 1.1.0 wheel). **There is no
binding-fidelity finding in this category:** `Save`/`Load`/`print_*`/`__repr__`
are direct pybind bindings over the C++ methods (no `conti.py` wrapper, no leaked
`c*` binding). The two defects (UT-IO2 broken pickle; UT-IO5 `_Load` name
over-read) are, respectively, a pybind-registration gap and a **shared-C++** bug
that surfaces identically in Python — so gate 4 (raw-C++ probe) is skipped:
*no binding-fidelity finding*. Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-IO1** | `Save`/`Load` are **Capitalized members** — they should be lowercase `save`/`load` under N-casing | naming (N-casing) | **member functions lowercase.** `Save` is an instance method and `Load` a static constructor of `UniTensor` (`cpp:150,171`; `py:513,515`); SciPostPhysCodeb.53 lowercases *member* functions (contrast the Capitalized *free* functions `Contract`/`Svd`, kept — cat 08/10). This is the **same decision as cat 02's `Load`→`load`**. Py probe *"`Save` is a UniTensor MEMBER (Capitalized member — wrong per N-casing, should be `save`)"* + *"`Load` is a UniTensor STATIC member … should be `load`"* | **rename** `Save`→`save`, `Load`→`load`. *Migration:* keep `Save`/`Load` as `DeprecationWarning` aliases for one minor release, then delete. |
| **UT-IO2** | the **pickle protocol is broken**: `__getstate__` is present but is only the inherited `object.__getstate__`, `__setstate__` is **absent**, and `pickle.dumps(ut)` **raises** | **correctness** (broken stub) | **half a protocol, and non-functional.** `hasattr(UniTensor,'__getstate__')` is `True` — but `UniTensor.__getstate__ is object.__getstate__` (doc `'Helper for pickle.'`), not a cytnx implementation; `__setstate__` is not defined; and because pybind11 registered no `py::pickle(...)`, `pickle.dumps(ut)` raises `TypeError: cannot pickle '…UniTensor' object`. `copy.deepcopy(ut)` also raises `TypeError` (its `__deepcopy__`=`clone` takes no `memo`, cat 12), so copy never reaches the pickle fallback. Py probe *"`hasattr(...,'__getstate__') is True` — but it is the DEFAULT `object.__getstate__` …"* + *"`hasattr(...,'__setstate__') is False` …"* + *"`pickle.dumps(ut)` RAISES TypeError …"* + *"`copy.deepcopy(ut)` also RAISES TypeError … (cross-ref cat 12)"* | **either implement pickle over `save`/`load`** (register `py::pickle(__getstate__, __setstate__)` serializing via the same binary format `Save`/`Load` use, so `pickle`/`copy.deepcopy` work) **or remove the misleading `__getstate__`**. Do not ship a `__getstate__` with no `__setstate__`. Fixing `copy.deepcopy` is tracked in cat 12. |
| **UT-IO3** | `Save`→`Load` **round-trips values faithfully** (elements, shape, dtype, rowrank) — the working I/O path | (kept; verified) | **faithful binary round-trip.** `Save(path)` writes a `.cytnx` binary and `Load(path)` reconstructs a value-equal tensor: same elements, `shape`, `dtype`, `rowrank`. Py probe *"`Save` writes the tensor to the given path …"* + *"`Save`->`Load` round-trip is value-equal …"* + *"… preserves shape, dtype and rowrank"* | **keep** `save`/`load` (renamed per UT-IO1) as the primary serialization API; document the binary `.cytnx` format and the round-trip guarantee (subject to the name fix UT-IO5). |
| **UT-IO4** | `Save` **appends `.cytnx`** (with a deprecation warning) when the path has no extension, but `Load` **does not** — the extension handling is **asymmetric** | **correctness / ergonomics** | **Save auto-completes, Load does not.** `UniTensor::Save` (`cpp:152-162`) appends `.cytnx` and emits a deprecation warning when `fname` lacks an extension; `UniTensor::Load` (`cpp:171-177`) opens `fname` verbatim and errors if it is not found — so `Save("t")` then `Load("t")` **fails** (must be `Load("t.cytnx")`). Py probe *"`Save` with no extension appends `.cytnx` (deprecated), while `Load` does NOT auto-append — Load(base_without_ext) RAISES"* | **require the full path with extension on both** (drop `Save`'s deprecated auto-append), or make **both** append symmetrically. Keep the explicit-extension form as the documented contract; migrate the auto-append behind a `DeprecationWarning` for one release, then remove. |
| **UT-IO5** | `Save`→`Load` **does not reliably preserve a non-empty tensor name** — `_Load` performs a **heap over-read** | **correctness** (memory bug) | **`std::string(cname)` on a non-NUL-terminated buffer.** `UniTensor::_Load` (`cpp:119-126`) reads `len_name` bytes into `malloc(len_name)` (no terminator) then does `std::string(cname)`, which scans **past** the buffer to the next heap `'\0'` — appending garbage byte(s) (`'foo'`→`'foo;'`) or, when the trailing bytes are invalid UTF-8, raising `UnicodeDecodeError` on the Python side. Only the **empty-name** round-trip (UT-IO3) is reliably clean. Py probe *"`Save`->`Load` does NOT reliably preserve a non-empty tensor name — the _Load name read over-runs a non-NUL-terminated malloc(len_name) buffer …"* (batch of 5 names, ≥3 corrupt every run) | **fix `_Load`** to `std::string(cname, len_name)` (length-delimited), so the name round-trips exactly. This is a plain memory-safety bug — fix regardless of the API renames. |
| **UT-IO6** | `print_diagram`/`print_blocks`/`print_block` write to **Python-capturable** stdout (a `py::scoped_ostream_redirect` guard), **unlike** `Device.Print_Property` | (kept; documented) | **redirect guard present.** All three bindings (`py:586-591`) carry `py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>()`, so the C++ `std::cout` output flows through Python's `sys.stdout` and `contextlib.redirect_stdout` captures it; each returns `None`. (Contrast `enums.md`'s `Device.Print_Property`, which has **no** guard and writes to the process's real stdout — uncapturable.) Py probe *"`print_diagram()` returns None and its output IS capturable via contextlib.redirect_stdout (has a py::scoped_ostream_redirect guard, unlike Device.Print_Property)"* + the `print_blocks`/`print_block` equivalents | **keep** all three; **document** that they print (return `None`) and that the output **is** redirect-capturable. Consider adding string-returning siblings (e.g. `diagram_str()`) so callers need not capture stdout. |
| **UT-IO7** | `__repr__` **prints** the tensor and **returns the empty string** rather than returning the repr text | **correctness** (convention violation) | **print-and-return-`''`.** The `__repr__` lambda (`py:728-733`) does `std::cout << self; return std::string("")` — so `repr(ut)` returns `''` while the human-readable text goes to (capturable) stdout, the same anti-pattern documented for Bond/Symmetry. Py probe *"`__repr__` PRINTS … and RETURNS '' …"* + *"`__repr__`'s printed output IS capturable … and is non-empty"* | **return the string** — build the representation into a `std::string` (reuse `operator<<`'s formatter) and *return* it, so `repr(ut)`/`str(ut)` behave conventionally and interactive display works without a stdout side effect. *Migration:* behavior change only (no API surface change); note it in the changelog. |

## A4. Argument ordering — positional & keyword

These are serializers and display methods, not constructors: each takes at most a
path or a display toggle. There is no keyword-only metadata block.

| API | positional-required (in order) | operation parameters (keyword-capable) |
|---|---|---|
| `Save` (→`save`) | `fname` | *(none)* |
| `Load` (→`load`, static) | `fname` | *(none)* |
| `print_diagram` | *(none)* | `bond_info=False` |
| `print_blocks` | *(none)* | `full_info=True` |
| `print_block` | `idx` | `full_info=True` |
| `__repr__` | *(none)* | *(none)* |

- **Canonical positional rule (§R.0):** the path/selector primary operand first —
  `save(fname)`, `print_block(idx)`. Matches the live order; no reordering needed.
- **Naming:** `fname`/`idx`/`bond_info`/`full_info` are self-describing. The only
  ordering-adjacent change is dropping `Save`'s deprecated no-extension auto-append
  (UT-IO4) so `fname` is always an explicit path.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version UniTensor I/O & display surface.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53) — members lowercase (UT-IO1).** `Save`→`save`,
  `Load`→`load`; both are UniTensor members (a static constructor is still a
  member), so they lowercase — the same decision as cat 02's `Load`→`load`. The
  `print_*` members are already correct. (Contrast the Capitalized *free*
  functions `Contract`/`Svd`, kept — cat 08/10.)
- **A protocol is implemented fully or not at all (UT-IO2).** Ship pickle only
  with **both** `__getstate__` and `__setstate__` (over the `save`/`load` binary
  format); never a lone inherited `__getstate__` with no restore hook.
- **Serialization round-trips exactly (UT-IO3/IO5).** `save`→`load` must preserve
  **every** field — values, shape, dtype, rowrank, labels **and name**; the
  `_Load` name over-read (`std::string(cname)` → `std::string(cname, len_name)`)
  is a memory-safety bug to fix.
- **Extension handling is symmetric and explicit (UT-IO4).** `save`/`load` take a
  full path with extension; no one-sided auto-append.
- **`__repr__`/display return strings; `print_*` print (UT-IO6/IO7).** `__repr__`
  *returns* the representation text (no stdout side effect); the `print_*` methods
  *print* (returning nothing) and their output stays **redirect-capturable**,
  keeping the `scoped_ostream_redirect` guards.
- **Binding fidelity: none.** `save`/`load`/`print_*`/`__repr__` are direct pybind
  bindings over the C++ methods — no `conti.py` wrapper, no leaked `c*` binding.
  Gate 4 (raw-C++ probe) is skipped: *no binding-fidelity finding*.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- serialization (members → lowercase; save/load are the primary I/O) ---
    def save(self, fname: str) -> None: ...              # write .cytnx binary (was Save)
    @staticmethod
    def load(fname: str) -> "UniTensor": ...             # read .cytnx binary (was Load)

    # --- display (already lowercase; print & return None; redirect-capturable) ---
    def print_diagram(self, bond_info: bool = False) -> None: ...
    def print_blocks(self, full_info: bool = True) -> None: ...
    def print_block(self, idx: int, full_info: bool = True) -> None: ...

    # --- repr: RETURN the text (no stdout side effect) ---
    def __repr__(self) -> str: ...                        # returns the representation

    # --- pickle: implement BOTH over the save/load format, or remove entirely ---
    def __getstate__(self) -> bytes: ...                  # serialize (was inherited stub)
    def __setstate__(self, state: bytes) -> None: ...     # restore (was ABSENT)
```

`save`/`load` are the renamed `Save`/`Load` (lowercase, `DeprecationWarning`
aliases for one release). `__repr__` returns its text rather than printing-and-
returning-`''`. Pickle is either implemented fully (both hooks, over the `save`/
`load` binary format) or the misleading inherited `__getstate__` is removed. The
`_Load` name read is fixed to be length-delimited.

| API | Verdict | Behavior contract |
|---|---|---|
| `save` (was `Save`) | **rename** (UT-IO1/IO3/IO5) | Member serializer: write this tensor to `fname` (explicit `.cytnx` extension) as a binary file preserving all fields. *Migration:* `Save` kept as a `DeprecationWarning` alias for one minor release, then deleted. Fix the `_Load` name over-read so the name round-trips (UT-IO5). |
| `load` (was `Load`) | **rename** (UT-IO1/IO3/IO4) | Static deserializer: `UniTensor.load(fname)` reads a `.cytnx` file into a new UniTensor, value-equal to the saved one. *Migration:* `Load` kept as a `DeprecationWarning` alias for one release. Require the full path (no asymmetric auto-append, UT-IO4). |
| `print_diagram` | **keep** (UT-IO6) | Print the ASCII leg diagram (labels/bond dims/rowrank); `bond_info=True` adds per-bond detail. Returns `None`; output is `redirect_stdout`-capturable. |
| `print_blocks` | **keep** (UT-IO6) | Print all storage blocks (`full_info` toggles verbosity). Returns `None`; capturable. |
| `print_block` | **keep** (UT-IO6) | Print one block by `idx` (`full_info` toggles verbosity). Returns `None`; capturable. |
| `__repr__` | **keep, fix return** (UT-IO7) | Return the human-readable representation **as a string** (no stdout side effect). *Migration:* behavior change only; note in changelog. |
| `__getstate__` | **fix or remove** (UT-IO2) | Currently the inherited `object.__getstate__` (does nothing useful) with no `__setstate__`, so `pickle.dumps` raises. *Migration:* implement over the `save`/`load` binary format **with** `__setstate__`, or remove so the class does not advertise a half-protocol. |
| `__setstate__` | **add or omit** (UT-IO2) | Absent today. Add it (paired with `__getstate__`, restoring from the binary format) so `pickle`/`copy.deepcopy` work — or leave pickle unimplemented and remove the lone `__getstate__`. |

**No binding-fidelity / plumbing findings.** Unlike cats 04–07, this category
surfaces **no** leaked `c*` bindings and **no** `conti.py` wrappers — the pybind
layer forwards directly to `cytnx::UniTensor`. Gate 4 (raw-C++ probe) is skipped:
*no binding-fidelity finding.* (The `_Load` name bug UT-IO5 is a shared-C++
memory bug observable identically from Python, not a binding divergence.)

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python surface,
**R.2b** Doxygen for the C++ surface. Kept/renamed members are documented; the
pickle hooks carry only migration notes (fix-or-remove, no standalone docstring).

### R.2a Python API (numpy-style)

### `save` / `load`

```
UniTensor.save(fname)          -> None         # write .cytnx binary  (was Save)
UniTensor.load(fname)          -> UniTensor    # static; read .cytnx  (was Load)

Serialize a UniTensor to / from a binary `.cytnx` file.

`save` writes this tensor to `fname`; `load` (a static constructor called as
`UniTensor.load(path)`) reconstructs a value-equal tensor — same elements, shape,
dtype, rowrank, labels and name (finding UT-IO3). Provide the FULL path including
the `.cytnx` extension: `load` opens the path verbatim and does not auto-append it
(finding UT-IO4).

Parameters
----------
fname : str
    Destination (save) / source (load) file path, including the `.cytnx`
    extension.

Returns
-------
None (save) / UniTensor (load)

Notes
-----
Renamed from the Capitalized `Save`/`Load` (finding UT-IO1); those spellings
remain as `DeprecationWarning` aliases for one minor release, then are removed.
Through cytnx 1.1.0 the loader corrupted a non-empty tensor NAME (a heap over-read
in `_Load`, finding UT-IO5) — the next version round-trips the name exactly.

See Also
--------
print_diagram : render the tensor's leg structure for a human.
```

### `print_diagram` / `print_blocks` / `print_block`

```
UniTensor.print_diagram(bond_info=False)  -> None   # ASCII leg diagram
UniTensor.print_blocks(full_info=True)    -> None   # all storage blocks
UniTensor.print_block(idx, full_info=True)-> None   # one block by index

Print a human-readable rendering of this UniTensor to stdout.

`print_diagram` draws the leg diagram (labels, bond dimensions, the rowrank split);
`bond_info=True` adds per-bond quantum-number detail. `print_blocks`/`print_block`
dump the dense storage block(s); `full_info` toggles verbosity.

All three PRINT (return None) and their output IS capturable via
`contextlib.redirect_stdout` — the bindings carry a `scoped_ostream_redirect`
guard, so the underlying C++ `std::cout` is routed through Python's `sys.stdout`
(finding UT-IO6; contrast `Device.Print_Property`, which is NOT capturable).

Parameters
----------
bond_info : bool, optional
    (`print_diagram`) include per-bond detail. Default False.
full_info : bool, optional
    (`print_blocks`/`print_block`) verbose block info. Default True.
idx : int
    (`print_block`) the block index to render.

Returns
-------
None
```

### `__repr__`

```
repr(UniTensor)  -> str    # the representation TEXT (next version)

Return the human-readable representation of this UniTensor.

The next version RETURNS the representation as a string (reusing the C++
`operator<<` formatter), so `repr(ut)` / `str(ut)` behave conventionally.

Notes
-----
Through cytnx 1.1.0 `__repr__` instead PRINTED the tensor via `std::cout << self`
and RETURNED '' (an empty string) — the print-and-return-'' pattern shared with
Bond/Symmetry (finding UT-IO7). The printed text was `redirect_stdout`-capturable.

Pickle (`__getstate__`/`__setstate__`) is only half-implemented through 1.1.0
(finding UT-IO2): the inherited `object.__getstate__` is present but `__setstate__`
is absent and `pickle.dumps(ut)` raises `TypeError`. The next version either
implements BOTH hooks over the `save`/`load` binary format or removes the lone
`__getstate__`; use `save`/`load` for persistence in the meantime.
```

### R.2b C++ API (Doxygen)

The C++ `Save`/`Load`/`print_*` already exist and bind directly; the next
version's changes are: (1) fix `_Load`'s name read to be length-delimited (UT-IO5),
(2) make `operator<<`-backed repr *return* a string (UT-IO7), and (3) add pickle
`__getstate__`/`__setstate__` (Python side, over the same binary format) or drop
the lone `__getstate__` (UT-IO2). The C++ member spellings become `save`/`load`
(UT-IO1).

```cpp
/**
 * @brief Serialize this UniTensor to a binary `.cytnx` file (member `save`).
 * @details Writes all fields (values, shape, dtype, rowrank, labels, name) in the
 *          cytnx binary format; load() reconstructs a value-equal tensor
 *          (finding UT-IO3). Renamed from Save (finding UT-IO1). Require the full
 *          path with extension — no asymmetric auto-append (finding UT-IO4).
 * @param fname destination path (including the `.cytnx` extension).
 */
void save(const std::string &fname) const;

/**
 * @brief Deserialize a UniTensor from a binary `.cytnx` file (static `load`).
 * @details Reconstructs the tensor written by save(). Fix the name read in the
 *          loader — `std::string(cname, len_name)` (length-delimited), NOT
 *          `std::string(cname)` on a non-NUL-terminated malloc(len_name) buffer,
 *          which over-reads the heap and corrupts the name (finding UT-IO5).
 *          Renamed from Load (finding UT-IO1).
 * @param fname source path (including the `.cytnx` extension).
 * @return the reconstructed UniTensor.
 */
static UniTensor load(const std::string &fname);

/**
 * @brief Print a human-readable rendering to std::cout (display methods).
 * @details print_diagram draws the leg diagram; print_blocks/print_block dump the
 *          storage block(s). Each returns void; the Python bindings carry a
 *          scoped_ostream_redirect guard so the output is capturable from Python
 *          (finding UT-IO6). @p bond_info / @p full_info toggle verbosity;
 *          @p idx selects a block.
 */
void print_diagram(const bool &bond_info = false);
void print_blocks(const bool &full_info = true) const;
void print_block(const cytnx_int64 &idx, const bool &full_info = true) const;

/**
 * @brief Stream a human-readable representation of the UniTensor.
 * @details The Python __repr__ must RETURN this text (reusing operator<<), not
 *          print-and-return-'' as it did through 1.1.0 (finding UT-IO7). Pickle
 *          (__getstate__/__setstate__) must be implemented fully over the
 *          save/load binary format or omitted entirely — never a lone inherited
 *          __getstate__ with no __setstate__ (finding UT-IO2).
 */
std::ostream &operator<<(std::ostream &os, const UniTensor &in);
```
