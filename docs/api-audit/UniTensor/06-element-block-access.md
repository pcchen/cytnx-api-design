# UniTensor — 06. Element & block access

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the sibling
> [`04-labels-name-rowrank.md`](04-labels-name-rowrank.md),
> [`05-structure-manipulation.md`](05-structure-manipulation.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's element/block-access API to match §R exactly. All runtime claims
> verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_06_element.py` (all `[PASS]`, exit 0), with
> the raw-C++ side of the binding-fidelity findings verified by
> `probes/cpp/UniTensor_06_element.cpp` against a source-built `libcytnx` (all
> `[PASS]`, exit 0).

**Category scope:** the members that read and write individual elements and
whole blocks — single-element access (`at`, `item`, `get_elem`, `set_elem`,
`elem_exists`), block access (`get_block`/`get_block_`, `get_blocks`/
`get_blocks_`, `put_block`/`put_block_`), and the Python indexing operators
(`__getitem__`/`__setitem__`) — plus the raw C++ accessor methods `get`/`set`
(unbound; reached only through the operators) and the leaked raw `c_at` binding.
Python bindings: `cytnx_src/pybind/unitensor_py.cpp:37-95,178-200,347-483,595-726`;
conti.py wrappers (`at`, `Hclass` element proxy):
`cytnx/UniTensor_conti.py:10-97,243-251`; C++ header:
`cytnx_src/include/UniTensor.hpp:280-307,2999-3018,3997-4108,4089-4525,5461-5492`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place. Overloads differing only
in selector type (`idx` vs `qnum`/`labels`) share one row.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `at` | `at(locator)` · `at(labels, locator)` | element **proxy** (`Hclass`) | Read/write a single element by index (works on **both** Dense and Block). Returns a proxy whose `.value` reads and `proxy.value = x` writes in place. conti.py wrapper over raw `c_at`; the proxy dispatches all **11** dtypes. Probe: *"at([0,0]) … returns a proxy whose .value reads the element"* + *"the at() proxy is writable …"* + *"at([0,0]) works on a Block tensor too …"*. |
| `item` | `item()` | scalar (**all 11 dtypes**) | Extract the sole element of a 1-element tensor as a native Python scalar; the pybind lambda branches over all 11 dtypes. Probe: *"item() reads a value on dtype … (all 11 covered)"*. |
| `get_elem` | `get_elem(locator)` | scalar (**4 dtypes only**) | Read an element by index — but the pybind lambda instantiates **only** the 4 float/complex branches; on the 7 integer/bool dtypes it **raises**. Probe: *"get_elem returns a value on float/complex dtype …"* + *"get_elem RAISES on integer/bool dtype …"*. |
| `set_elem` | `set_elem(locator, value)` | `None` (in-place) | Write an element by index; bound for **all 11** dtypes (two template families, scalar + integer). Probe: *"set_elem writes a value on dtype … (all 11 covered)"*. |
| `elem_exists` | `elem_exists(locator)` | `bool` | **Block-only** predicate — reports whether a symmetry block-element exists; **errors on a Dense tensor**. Probe: *"elem_exists reports True for an allowed block-element …"* + *"elem_exists ERRORS on a Dense tensor …"*. |
| `get_block` | `get_block(idx=0)` · `get_block(qnum, force=False)` · `get_block(labels, qnum, force=False)` | `Tensor` (**copy**) | Return a **clone** of one block, by index or quantum number; its data is **not** shared with the tensor. Probe: *"get_block returns a COPY …"*. |
| `get_block_` `[I]` | `get_block_(idx=0)` · `get_block_(qnum, force=False)` · `get_block_(labels, qnum, force=False)` | `Tensor` (**shared-data view**) | Return a **view** onto one block — its data **is** shared with the tensor. Probe: *"get_block_ returns a shared-data VIEW …"*. |
| `get_blocks` | `get_blocks()` | `list[Tensor]` (**copies**) | Return **clones** of all blocks (Block/BlockFermionic only); **errors on Dense**. Probe: *"get_blocks returns COPIES …"* + *"get_blocks errors on a Dense tensor …"*. |
| `get_blocks_` `[I]` | `get_blocks_(*args, **kwargs)` | `list[Tensor]` (**shared-data views**) | Return **views** onto all blocks. Takes an optional `silent` flag — but also accepts the **misspelled** `slient` (deprecated, warns). Probe: *"get_blocks_ returns shared-data VIEWS …"* + *"the misspelled `slient` kwarg … emits a FutureWarning …"*. |
| `put_block` | `put_block(in, idx=0)` · `put_block(in, qidx)` · `put_block(in, labels, qidx)` · `put_block(in, qidx, force)` `[dep]` | `None` (in-place) | **Copy** `in` into a block (the receiver's block does **not** share `in`'s data). The `force` overload is **deprecated** (warns). Probe: *"put_block COPIES the input tensor into the block … and returns None"* + *"put_block(in, qidx, force) emits a FutureWarning …"*. |
| `put_block_` `[I]` | `put_block_(in, idx=0)` · `put_block_(in, qidx)` · `put_block_(in, labels, qidx)` · `put_block_(in, qidx, force)` `[dep]` | `None` (in-place) | **View-in**: make `in` a **shared-data** view of a block (the block aliases `in`'s storage). The `force` overload is **deprecated** (warns). Probe: *"put_block_ makes the input tensor a shared-data VIEW … and returns None"* + *"put_block_(in, qidx, force) … emits a FutureWarning …"*. |
| `__getitem__` | `__getitem__(locators)` | `UniTensor` (**shared-data sub-block**) | Slice via `u[...]`; **Dense-only** — **errors on Block/BlockFermionic** ("Use at() instead."). Wraps the C++ `get(accessors)`. Probe: *"__getitem__ slices a Dense tensor …"* + *"__getitem__ ERRORS on a Block tensor …"*. |
| `__setitem__` | `__setitem__(locators, rhs)` (Tensor or UniTensor) | `None` (in-place) | Assign via `u[...] = rhs`; Dense-only for the UniTensor-rhs form. Wraps the C++ `set(accessors, rhs)`. Probe: *"__setitem__ assigns into a Dense tensor …"*. |

**Internal / plumbing (leak into `dir(UniTensor)`):** `c_at` — the raw pybind
binding the conti.py `at` wrapper calls (returns a `cHclass` proxy). Public
today, but should never be. Probe: the `at` behavior above is the wrapper over
it; `c_at` is confirmed present in `dir(UT)` (see A3 UT-E11).

**C++-only (present in C++, *not* bound in Python):** the accessor methods
`get(accessors)` / `set(accessors, rhs)` — reachable from Python only through
`__getitem__`/`__setitem__`. Probe: *"the raw C++ accessor methods `get`/`set`
are NOT public Python members …"*.

## A2. C++ ↔ Python mapping

| C++ (`UniTensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `Scalar::Sproxy at(locator)` (`:3997,4020`) | `at(...)` (conti.py over `c_at`, `:378`) | signature-differs | wrapper returns an `Hclass` element proxy; all 11 dtypes (UT-E9) |
| `template<T> T &item()` / `Sproxy item() const` (`:2999,3010`) | `item()` (lambda `:348`) | identical | lambda branches all 11 dtypes (UT-E1) |
| `template<T> T get_elem(locator)` (`:5476`, `[[deprecated]]`) | `get_elem(...)` (lambda `:421`) | **signature-differs** | C++ template covers all 11; the Python lambda wires up **only 4** float/complex (UT-E1) |
| `template<T2> UniTensor &set_elem(locator, rc)` (`:5486`, `[[deprecated]]`) | `set_elem(...)` (`:437-459`) | signature-differs | bound for all 11 dtypes; returns `None`, not `UniTensor&` (UT-E1) |
| `bool elem_exists(locator) const` (`:5467`) | `elem_exists(...)` (`:347`) | identical | Block-only; errors on Dense (UT-E10) |
| `Tensor get_block(...) const` (`:690,704,708`) | `get_block(...)` (`:595`) | identical | returns a clone — a copy (UT-E3) |
| `Tensor &get_block_(...)` (`:776,789`) | `get_block_(...)` (`:620`) | identical | returns a shared-data view (UT-E3) |
| `std::vector<Tensor> get_blocks() const` (`:4302`) | `get_blocks()` (`:645`) | identical | returns clones — copies; errors on Dense (UT-E4) |
| `std::vector<Tensor> &get_blocks_(silent=false)` (`:4313,4321`) | `get_blocks_(*args, **kwargs)` (`:646`) | **signature-differs** | shared-data views; the `silent` arg is hand-parsed, also accepting the typo `slient` (UT-E4/E5) |
| `UniTensor &put_block(...)` (`:4334,4362`) | `put_block(...)` (`:658`) | **signature-differs** | copy-in; the Python lambda returns `None`, dropping C++'s `UniTensor&` (UT-E6/E8) |
| `UniTensor &put_block_(...)` (`:4441,4455`) | `put_block_(...)` (`:694`) | **signature-differs** | view-in; likewise returns `None` (UT-E6/E8) |
| `UniTensor get(accessors) const` (`:4543`) | *(unbound; via `__getitem__` `:389`)* | **C++-only** | reached only through `u[...]` (UT-E2) |
| `UniTensor &set(accessors, rhs)` (`:4590,4594`) | *(unbound; via `__setitem__` `:400`)* | **C++-only** | reached only through `u[...] = rhs` (UT-E2) |
| `UniTensor operator[](accessors) const` (`:4553`) | `__getitem__(...)` (`:389`) | identical | Dense-only; errors on Block (UT-E7) |
| raw `at` proxy binding | `c_at` (`:378`) | **leak** | plumbing exposed publicly (UT-E11) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/UniTensor_06_element.py` (on the 1.1.0 wheel). A
**(binding fidelity)** finding flags where the binding layer — a `*_conti.py`
wrapper or a pybind lambda — changes behavior, signature, or dtype coverage
versus the raw C++ method; **both sides are runtime-verified**, the raw-C++ side
by `probes/cpp/UniTensor_06_element.cpp` (links against a source-built
`libcytnx`, GCC 13). Source `file:line` cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-E1** | `get_elem` reads **only** the 4 float/complex dtypes and **raises** on the 7 integer/bool dtypes, while `item` and `set_elem` cover **all 11** | **binding fidelity** / signature-differs | **binding under-instantiates**: the `get_elem` pybind lambda (`:421-434`) has branches for `Double`/`Float`/`ComplexDouble`/`ComplexFloat` only and falls through to `[ERROR] try to get element from a void Storage` for the rest, whereas the C++ `get_elem<T>` is a generic template forwarding to `at<T>()` (`hpp:5476-5479`) and `item`/`set_elem` are bound for all 11 (`:348-375,437-459`). Py probe *"get_elem RAISES on integer/bool dtype …"* + *"item()/set_elem() … (all 11 covered)"*; **C++ probe confirms** `get_elem<T>`/`set_elem<T>` round-trip a value on all 11 dtypes — the 4-dtype limit is a **binding choice**, not a C++ gap | **fix binding fidelity** — bind `get_elem` for the **same 11 dtypes** as `set_elem`/`item` (mirror the `item` lambda's dtype ladder) |
| **UT-E2** | the C++ accessor methods `get(accessors)` / `set(accessors, rhs)` are **C++-only** — unbound in Python and reachable only through `__getitem__`/`__setitem__` | **binding fidelity** / C++-only | **binding hides the methods behind the operators**: `get`/`set` exist in C++ (`hpp:4543,4590`) but are absent from `dir(UniTensor)`; the `__getitem__` lambda calls `self.get(accessors)` (`:398`) and `__setitem__` calls `self.set(accessors, rhs)` (`:407,418`). Py probe *"the raw C++ accessor methods `get`/`set` are NOT public Python members …"*; **C++ probe confirms** `U.get({0, :})` slices to shape `[3]` and `U.set({0, :}, rhs)` writes and returns `*this` | **document reached-via-dunder** — keep `get`/`set` as the C++ spelling and `__getitem__`/`__setitem__` as the sole Python surface; state the equivalence in the docstrings (no separate `get`/`set` Python members) |
| **UT-E3** | `get_block` returns a **copy** (clone) while `get_block_` returns a **shared-data view** — a correct copy/view pair | copy/view (B2) | **faithful pass-throughs** — `get_block` forwards to C++ `get_block(...) const` which clones (`hpp:639`); `get_block_` forwards to `get_block_(...)` which returns the block by reference (`hpp:776`). Py probe *"get_block returns a COPY …"* + *"get_block_ returns a shared-data VIEW …"*; **C++ probe confirms** `get_block_` is `same_data` with the tensor while `get_block` is not | **keep both**; **document the copy-vs-view split explicitly** (silent today) |
| **UT-E4** | `get_blocks` returns **copies** while `get_blocks_` returns **shared-data views**; both are **Block-only** (`get_blocks` errors on Dense) | copy/view (B2) | **faithful pass-throughs** — `get_blocks` = `vec_clone(_blocks)` (`hpp:1387`), `get_blocks_` returns `_blocks` by reference (`hpp:1388`); DenseUniTensor rejects both (`hpp:690-704`). Py probe *"get_blocks returns COPIES …"* + *"get_blocks_ returns shared-data VIEWS …"* + *"get_blocks errors on a Dense tensor …"* | **keep both**; **document the copy-vs-view split** and the Block-only precondition |
| **UT-E5** | `get_blocks_` accepts a **misspelled** `slient` keyword (in addition to the correct `silent`); the typo path warns and still forwards | **correctness / typo** (FutureWarning) | **binding tolerates the typo**: `parse_get_blocks_silent_arg` (`:73-94`) checks `kwargs.contains("slient")`, emits a `FutureWarning` ("*Keyword 'slient' is deprecated … use 'silent' instead*"), then forwards; the correct `silent` path warns nothing. Py probe *"the misspelled `slient` kwarg … emits a FutureWarning … yet still forwards"* + *"the correctly-spelled `silent` … emits NO warning"* | **fix the typo → `silent`** — keep `slient` as a `FutureWarning` alias for one minor release, then delete; bind the parameter as a plain `silent: bool = False` (dropping the hand-parsed `*args, **kwargs`) |
| **UT-E6** | `put_block(…, force=…)` and `put_block_(…, force=…)` carry a **deprecated** `force` argument (warns; and the C++ overload ignores it) | **deprecated argument** (FutureWarning) | **binding warns**: the force overloads (`:676-692,709-726`) emit a `FutureWarning` ("*Argument 'force' is deprecated … use put_block(in, qnum) without force*") then forward; the C++ force overload itself drops the flag (`hpp:4348-4349`). Py probe *"put_block(in, qidx, force) emits a FutureWarning …"* + *"put_block_(in, qidx, force) … emits a FutureWarning …"* | **deprecate `force`** — remove the `force` overloads; keep them as `FutureWarning` aliases for one minor release, then delete |
| **UT-E7** | `__getitem__`/`__setitem__` slice a **Dense** tensor but **error on Block/BlockFermionic** ("Use at() instead.") | (kept; documented) | **binding guards by type**: the `__getitem__` lambda `cytnx_error_msg`s unless `uten_type() == Dense` (`:393-395`); `__setitem__` likewise for the UniTensor-rhs form (`:413-415`). Py probe *"__getitem__ slices a Dense tensor …"* + *"__getitem__ ERRORS on a Block tensor …"* + *"__setitem__ assigns into a Dense tensor …"* | **keep**; **document** that `[...]` is Dense-only and `at()` is the symmetry-aware element accessor |
| **UT-E8** | `put_block` (copy-in) vs `put_block_` (view-in): **both** mutate the receiver in place (neither is pure), so the trailing `_` here means **copy-in vs share-in**, not the usual pure-vs-in-place; and both return `None` though C++ returns `UniTensor&` | **N-underscore (non-standard)** / binding fidelity | **overloaded `_` meaning + dropped return**: `put_block` deep-copies `in` into the block; `put_block_` aliases `in`'s storage into the block (`hpp:709-774`); the Python lambdas return `void` (`:658-661,694-695`) though C++ returns `UniTensor&` (`hpp:4334,4441`). Py probe *"put_block COPIES the input tensor into the block … and returns None"* + *"put_block_ makes the input tensor a shared-data VIEW … and returns None"* | **keep both**, but **document the non-standard `_` semantics explicitly** (copy-in vs shared-in) so it is not read as pure-vs-in-place; the in-place `None` return is acceptable for a store operation |
| **UT-E9** | `at`/`item` are the dtype-complete single-element accessors: `at` returns a read/write **proxy** valid on **both** Dense and Block; `item` extracts the sole scalar of a 1-element tensor | (kept) | **conti.py proxy + all-dtype lambda**: `at` wraps `c_at` returning an `Hclass` whose `.value` getter/setter dispatches all 11 dtypes (`conti.py:10-97,243-251`); `item` branches all 11 (`:348-375`). Py probe *"at([0,0]) … returns a proxy whose .value reads the element"* + *"the at() proxy is writable …"* + *"at([0,0]) works on a Block tensor too …"* | **keep both** as the primary element API; `at` is the recommended element accessor (`get_elem`/`set_elem` are its thin, dtype-restricted shadows) |
| **UT-E10** | `elem_exists` is a **Block-only** predicate — it reports whether a symmetry block-element exists and **errors on a Dense tensor** | (kept) | **type-guarded**: DenseUniTensor's `elem_exists` `cytnx_error_msg`s ("*can only be used on UniTensor with Symmetry*", `hpp:1040`); on a Block tensor it returns whether the quantum-number-selected element is allowed. Py probe *"elem_exists reports True for an allowed block-element …"* + *"elem_exists ERRORS on a Dense tensor …"* | **keep**; **document** the Block-only precondition. *(Note: the plan also lists `elem_exists` in cat-03 — it is a block-access predicate and belongs here; cat-03 should drop it to preserve the one-category partition, see the Concerns note.)* |
| **UT-E11** | the raw `c_at` binding **leaks** into `dir(UniTensor)` | naming + **binding fidelity** | **binding exposes plumbing**: `c_at` is `.def`-ed (`:378-387`) purely so the conti.py `at` wrapper (`conti.py:243-251`) can call it and box the result in `Hclass`; the `c`-prefix is a reserved raw-binding spelling (§R.0 rejects it). Py probe confirms `c_at` is present in `dir(UT)` (the `at` wrapper is defined over it) | **remove from public API** — bind under a private name (leading `_`) or inline the proxy construction into the `at` pybind lambda (migration note) |

## A4. Argument ordering — positional & keyword

Every member here takes a locator/selector primary operand and, for the writers,
a value/tensor payload. There is no keyword-only metadata block.

| API | positional-required (in order) |
|---|---|
| `at` | `locator` (**or** `labels, locator`) |
| `item` / `get_blocks` | *(none)* |
| `get_elem` / `elem_exists` | `locator` |
| `set_elem` | `locator`, `value` |
| `get_block` / `get_block_` | `idx` **or** `qnum` (then optional `force` → dropped) (**or** `labels, qnum`) |
| `get_blocks_` | *(optional)* `silent` |
| `put_block` / `put_block_` | `in`, then `idx` **or** `qidx` (**or** `labels, qidx`) |
| `__getitem__` | `locators` |
| `__setitem__` | `locators`, `rhs` |

- **Canonical positional rule (§R.0):** the locator/selector primary operand
  first, then the payload — `set_elem(locator, value)`, `put_block(in, qidx)`.
  This matches the live order and needs no change, except that `get_blocks_`'s
  `silent` must become a **real typed keyword** (`silent: bool = False`), not a
  hand-parsed `*args, **kwargs` (UT-E5), and the deprecated `force` positional on
  `get_block`/`put_block(_)` is removed (UT-E6).
- **Naming:** `locator`/`qnum`/`qidx`/`labels`/`in`/`silent` are self-describing.
  The one defect is the **misspelled `slient`** keyword (UT-E5) → `silent`.
  Note `get_elem`/`set_elem`/`elem_exists` erase the locator name to `arg0` in
  their pybind signatures (parameter-name erasure, ties to the cat-03 PC1
  finding) — the recommended `at`-based element API restores named parameters.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version element/block-access API.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53).** Every member here is already lowercase
  snake_case; no casing change is required.
- **N-underscore — a trailing `_` marks in-place (returns `self`); its absence
  marks pure (returns a new object).** The block accessors keep both forms:
  `get_block` (copy) / `get_block_` (shared-data view) and `get_blocks` /
  `get_blocks_`. **`put_block`/`put_block_` are a *non-standard* use** of the
  suffix — **both** mutate the receiver in place; the `_` distinguishes
  **copy-in** (`put_block` deep-copies the source tensor) from **share-in**
  (`put_block_` aliases the source tensor's storage). This is documented
  explicitly (UT-E8) rather than silently overloading the convention. The
  **`c`-prefixed raw spelling `c_at` is rejected** as public API (UT-E11).
- **Binding fidelity — dtype coverage must match across sibling accessors.**
  `get_elem`, `set_elem`, `item`, and the `at` proxy must all cover the **same
  11 element dtypes**; the current `get_elem`'s 4-dtype limit is a binding defect
  to fix (UT-E1).
- **C++ `get`/`set` are the operator implementations, not separate Python
  members.** They stay reachable only through `__getitem__`/`__setitem__`
  (UT-E2); the docstrings state the equivalence.
- **Copy-vs-view is documented, not silent.** `get_block`/`get_blocks` return
  **copies**; `get_block_`/`get_blocks_` and `__getitem__` return **shared-data
  views**. Each accessor's docstring states which (UT-E3/E4/E7).
- **Real signatures, not `*args`.** `get_blocks_` binds an explicit
  `silent: bool = False` (UT-E5); the deprecated `slient` typo and the
  `put_block(_)` `force` argument are removed (UT-E5/E6).

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- single-element access (all 11 dtypes; at is the primary API) ---
    def at(self, locator: Sequence[int]) -> ElemProxy: ...
    def at(self, labels: Sequence[str], locator: Sequence[int]) -> ElemProxy: ...  # overload
    #   returns a read/write proxy: proxy.value reads, proxy.value = x writes.
    #   Valid on BOTH Dense and Block tensors.
    def item(self) -> Scalar: ...                             # sole element of a 1-element tensor
    def get_elem(self, locator: Sequence[int]) -> Scalar: ...  # read; ALL 11 dtypes (was 4)
    def set_elem(self, locator: Sequence[int], value: Scalar) -> None: ...  # write; all 11 dtypes
    def elem_exists(self, locator: Sequence[int]) -> bool: ...  # Block-only predicate

    # --- block access (copy = no suffix, shared-data view = trailing _) ---
    def get_block(self, idx: int = 0) -> Tensor: ...                       # copy
    def get_block(self, qnum: Sequence[int]) -> Tensor: ...                # overload, copy
    def get_block(self, labels: Sequence[str], qnum: Sequence[int]) -> Tensor: ...  # overload
    def get_block_(self, idx: int = 0) -> Tensor: ...                      # shared-data view
    def get_block_(self, qnum: Sequence[int]) -> Tensor: ...               # overload, view
    def get_block_(self, labels: Sequence[str], qnum: Sequence[int]) -> Tensor: ...
    def get_blocks(self) -> list[Tensor]: ...                             # copies (Block-only)
    def get_blocks_(self, silent: bool = False) -> list[Tensor]: ...       # views  (Block-only)

    # --- block store (BOTH in-place; _ = share-in vs copy-in) ---
    def put_block(self, in_: Tensor, idx: int = 0) -> None: ...            # copy-in
    def put_block(self, in_: Tensor, qidx: Sequence[int]) -> None: ...     # overload
    def put_block(self, in_: Tensor, labels: Sequence[str], qidx: Sequence[int]) -> None: ...
    def put_block_(self, in_: Tensor, idx: int = 0) -> None: ...           # share-in
    def put_block_(self, in_: Tensor, qidx: Sequence[int]) -> None: ...    # overload
    def put_block_(self, in_: Tensor, labels: Sequence[str], qidx: Sequence[int]) -> None: ...

    # --- indexing operators (Dense-only; wrap the C++ get/set) ---
    def __getitem__(self, locators) -> "UniTensor": ...   # shared-data sub-block
    def __setitem__(self, locators, rhs: Tensor | "UniTensor") -> None: ...
```

`get_elem` is bound for the **same 11 dtypes** as `set_elem`/`item` (UT-E1).
`get_blocks_` binds a typed `silent: bool = False`; the `slient` typo and the
`put_block(_)` `force` argument are removed. The raw `c_at` plumbing binding
becomes private (leading `_`) or is inlined into the `at` pybind lambda. The C++
`get`/`set` accessors remain unbound-by-name — `__getitem__`/`__setitem__` are
their sole Python surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `at` | **keep** (UT-E9) | Return a read/write element proxy (`.value` reads/writes) for the element at `locator`; valid on Dense and Block. The primary single-element API; covers all 11 dtypes. |
| `item` | **keep** (UT-E1/E9) | Return the sole element of a 1-element tensor as a native scalar; all 11 dtypes. Errors if the tensor has more than one element or is block-form. |
| `get_elem` | **keep, fix dtype coverage** (UT-E1) | Read the element at `locator` as a scalar. *Migration:* bind for all 11 dtypes (mirror `item`); through 1.1.0 it raised on the 7 integer/bool dtypes — a binding defect. |
| `set_elem` | **keep** (UT-E1) | Write `value` to the element at `locator` (in place); all 11 dtypes. |
| `elem_exists` | **keep** (UT-E10) | Block-only: return whether the symmetry block-element at `locator` exists. Errors on a Dense tensor. |
| `get_block` | **keep** (UT-E3; document copy) | Return a COPY (clone) of one block, by index or quantum number. |
| `get_block_` | **keep** (UT-E3; document view) | Return a shared-data VIEW of one block. |
| `get_blocks` | **keep** (UT-E4; document copy) | Return COPIES of all blocks (Block-only; errors on Dense). |
| `get_blocks_` | **keep, bind typed `silent`** (UT-E4/E5) | Return shared-data VIEWS of all blocks (Block-only). *Migration:* bind `silent: bool = False`; the misspelled `slient` remains a `FutureWarning` alias for one release, then deleted. |
| `put_block` | **keep, drop `force`** (UT-E6/E8) | Copy-in: deep-copy `in` into a block (in place). *Migration:* remove the deprecated `force` overload (`FutureWarning` alias for one release). |
| `put_block_` | **keep, drop `force`** (UT-E6/E8) | Share-in: alias `in`'s storage into a block (in place). *Migration:* remove the deprecated `force` overload (`FutureWarning` alias for one release). |
| `__getitem__` | **keep** (UT-E7; Dense-only) | Slice a Dense tensor via `u[...]`, returning a shared-data sub-block UniTensor; errors on Block/BlockFermionic (use `at()`). Wraps C++ `get`. |
| `__setitem__` | **keep** (UT-E7) | Assign into a Dense tensor via `u[...] = rhs` (in place). Wraps C++ `set`. |

**C++-only — the operator implementations (no separate Python member).** The
accessor methods below exist in C++ and are reached from Python only through the
indexing operators; they carry a C++ (R.2b) docstring only.

| API | Verdict | Behavior contract |
|---|---|---|
| `get(accessors)` | **keep (C++-only)** (UT-E2) | Slice implementation behind `__getitem__`; returns a sub-block UniTensor (shared data). Not a separate Python member. |
| `set(accessors, rhs)` | **keep (C++-only)** (UT-E2) | Assignment implementation behind `__setitem__`; writes `rhs` into the sliced region and returns `*this`. Not a separate Python member. |

**Internal / plumbing — hidden, not public API.** The raw binding below is a
live public member today with a **remove** verdict: hide it behind a leading
underscore or inline it into the `at` pybind lambda. It carries no docstring.

| API | Verdict | Behavior contract |
|---|---|---|
| `c_at` | **remove** (UT-E11) | Raw plumbing (the C++ `at` proxy) that the conti.py `at` wrapper boxes into `Hclass`. *Migration:* rename to `_at` (private) or inline the proxy construction into the `at` pybind lambda; no public exposure. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept/renamed members are
documented (removed plumbing carries no docstring); the C++-only `get`/`set`
have an R.2b docstring but no R.2a.

### R.2a Python API (numpy-style)

### `at` / `item`

```
UniTensor.at(locator)          -> ElemProxy   # read/write proxy; Dense AND Block
UniTensor.at(labels, locator)  -> ElemProxy   # overload: locate by leg labels
UniTensor.item()               -> scalar      # sole element of a 1-element tensor

Access a single element of this UniTensor.

`at` returns a PROXY into the element at `locator`. Read it with `.value` and
write it with `proxy.value = x`; the write mutates this tensor in place. Unlike
the `[]` operator, `at` works on BOTH Dense and Block/BlockFermionic tensors, and
covers all 11 element dtypes (finding UT-E9). It is the RECOMMENDED single-element
accessor.

`item` returns the sole element of a one-element tensor as a native Python scalar
(all 11 dtypes, finding UT-E1); it raises if the tensor holds more than one
element or is block-form.

Parameters
----------
locator : sequence of int
    The multi-index of the element.
labels : sequence of str
    Leg labels, to locate the element by label order instead of leg order.

Returns
-------
ElemProxy
    `at`: a read/write proxy (`.value`). `item`: a native scalar.

See Also
--------
get_elem, set_elem : thin scalar read/write shadows of `at`.
elem_exists : whether a symmetry block-element exists (Block only).
```

### `get_elem` / `set_elem` / `elem_exists`

```
UniTensor.get_elem(locator)          -> scalar   # read;  ALL 11 dtypes
UniTensor.set_elem(locator, value)   -> None      # write; all 11 dtypes (in place)
UniTensor.elem_exists(locator)       -> bool      # Block-only predicate

Scalar element read/write by multi-index.

`get_elem` returns the element at `locator`; `set_elem` writes `value` there in
place. Both cover all 11 element dtypes.

`elem_exists` is a BLOCK-ONLY predicate: it returns whether the symmetry
block-element at `locator` exists (i.e. is a stored, quantum-number-allowed
element). It raises on a Dense tensor (finding UT-E10).

Parameters
----------
locator : sequence of int
    The multi-index of the element.
value : scalar
    The value to write (set_elem).

Returns
-------
scalar (get_elem) / None (set_elem) / bool (elem_exists)

Notes
-----
Through cytnx 1.1.0, `get_elem` was bound for only the 4 float/complex dtypes and
RAISED on the 7 integer/bool dtypes, unlike `set_elem`/`item`/`at` which cover
all 11 (finding UT-E1); the next version binds all 11. Prefer `at()` for
dtype-complete, name-preserving element access.
```

### `get_block` / `get_block_` / `get_blocks` / `get_blocks_`

```
UniTensor.get_block(idx=0)              -> Tensor        # COPY
UniTensor.get_block(qnum)              -> Tensor        # COPY (by quantum number)
UniTensor.get_block_(idx=0)            -> Tensor        # shared-data VIEW
UniTensor.get_blocks()                 -> list[Tensor]  # COPIES  (Block only)
UniTensor.get_blocks_(silent=False)    -> list[Tensor]  # VIEWS   (Block only)

Access the dense storage block(s) of this UniTensor.

The un-suffixed forms return COPIES (clones); the trailing-`_` forms return
SHARED-DATA VIEWS onto the tensor's storage — mutating a view is visible through
the tensor (findings UT-E3/E4).

`get_block`/`get_block_` select one block by index (Dense: block 0), by quantum
number, or by (labels, qnum). `get_blocks`/`get_blocks_` return every block and
are BLOCK-ONLY: on a Dense tensor they raise (use `get_block`/`get_block_`).

Parameters
----------
idx : int, optional
    Block index (default 0; the sole block for a Dense tensor).
qnum : sequence of int
    Quantum numbers selecting a symmetry block.
labels : sequence of str
    Leg labels paired with `qnum`.
silent : bool, optional
    Suppress the not-found warning in `get_blocks_` (default False).

Returns
-------
Tensor or list[Tensor]
    Un-suffixed: a copy / list of copies. `_`-suffixed: a shared-data view /
    list of views.

Notes
-----
Through cytnx 1.1.0, `get_blocks_` also accepted a MISSPELLED `slient` keyword
(finding UT-E5); the next version binds `silent: bool = False` and keeps `slient`
only as a `FutureWarning` alias for one release.
```

### `put_block` / `put_block_`

```
UniTensor.put_block(in_, idx=0)              -> None   # COPY-IN  (in place)
UniTensor.put_block(in_, qidx)              -> None   # COPY-IN  (by quantum number)
UniTensor.put_block_(in_, idx=0)            -> None   # SHARE-IN (in place)
UniTensor.put_block_(in_, qidx)             -> None   # SHARE-IN (by quantum number)

Store a Tensor into a block of this UniTensor, IN PLACE.

BOTH forms mutate this tensor in place (neither is a pure copy). The trailing `_`
does NOT mean pure-vs-in-place here; it distinguishes how the SOURCE is treated
(finding UT-E8):

`put_block`  COPY-IN:  deep-copies `in_` into the block — the block does NOT share
             `in_`'s storage afterwards.
`put_block_` SHARE-IN: aliases `in_`'s storage into the block — the block and
             `in_` share data afterwards; mutating one shows through the other.

Parameters
----------
in_ : Tensor
    The source block to store (shape must match the target block).
idx : int, optional
    Block index (default 0).
qidx : sequence of int
    Quantum numbers selecting the target symmetry block.
labels : sequence of str
    Leg labels paired with `qidx`.

Returns
-------
None

Notes
-----
Through cytnx 1.1.0 both carried a deprecated `force` argument (finding UT-E6),
removed in the next version (kept as a `FutureWarning` alias for one release).
```

### `__getitem__` / `__setitem__`

```
UniTensor.__getitem__(locators)        -> UniTensor   # u[...]  (Dense only; shared-data view)
UniTensor.__setitem__(locators, rhs)   -> None         # u[...] = rhs  (in place)

Slice-access a DENSE UniTensor with numpy-style indexing.

`u[locators]` returns a UniTensor sub-block that SHARES storage with `u` (a view);
`u[locators] = rhs` assigns `rhs` (a Tensor or Dense UniTensor) into the sliced
region in place. Both are DENSE-ONLY — on a Block/BlockFermionic tensor they raise
("Use at() instead."); use `at()` / `get_block()` for symmetric tensors
(finding UT-E7).

These operators are the ONLY Python surface for the C++ accessor methods
`get`/`set` (finding UT-E2).

Parameters
----------
locators : slice, int, or sequence thereof
    numpy-style index/slice per leg.
rhs : Tensor or UniTensor
    The values to assign (__setitem__).

Returns
-------
UniTensor (getitem, shared data) / None (setitem)
```

### R.2b C++ API (Doxygen)

C++ already provides the templated `get_elem<T>`/`set_elem<T>`/`item<T>` and the
`get`/`set` accessors; the next version must (1) have the *Python* `get_elem`
lambda cover all 11 dtypes (UT-E1), (2) bind `get_blocks_` with a typed `silent`
(UT-E5), (3) drop the `put_block(_)` `force` overloads (UT-E6), and (4) hide the
raw `c_at` binding (UT-E11).

```cpp
/**
 * @brief Access a single element (read/write proxy).
 * @details Returns a Scalar proxy into the element at @p locator; valid on BOTH
 *          Dense and Block tensors and over all 11 element dtypes (finding
 *          UT-E9). The Python `at` wraps this; the raw c_at binding it currently
 *          uses must be hidden (finding UT-E11).
 * @param locator the multi-index of the element (or labels + locator overload).
 * @return a Scalar::Sproxy read/write proxy.
 */
Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator);
const Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) const;

/**
 * @brief Read/write a single element as a typed scalar.
 * @details get_elem<T> forwards to at<T>() and set_elem<T2> writes at(locator);
 *          both are TEMPLATES covering all 11 element dtypes. The Python get_elem
 *          binding must instantiate all 11 (not only the 4 float/complex ones it
 *          did through 1.1.0, finding UT-E1). Prefer at() (these are deprecated
 *          shadows in C++).
 * @param locator the element multi-index. @param rc the value to write.
 * @return get_elem: the element as T. set_elem: reference to *this.
 */
template <class T> T get_elem(const std::vector<cytnx_uint64> &locator) const;
template <class T2> UniTensor &set_elem(const std::vector<cytnx_uint64> &locator, const T2 &rc);

/**
 * @brief Whether the symmetry block-element at @p locator exists (Block only).
 * @details Errors on a Dense tensor (finding UT-E10). Same as
 *          at(locator).exists().
 * @return bool.
 */
bool elem_exists(const std::vector<cytnx_uint64> &locator) const;

/**
 * @brief Access the storage block(s).
 * @details get_block/get_blocks return COPIES (clones); get_block_/get_blocks_
 *          return SHARED-DATA views (findings UT-E3/E4). get_blocks/get_blocks_
 *          are Block-only (Dense raises). The Python get_blocks_ must bind a
 *          typed `silent` argument (finding UT-E5).
 * @param idx block index / @param qnum quantum numbers / @param silent suppress
 *        the not-found warning.
 * @return a Tensor (copy or view) / vector of Tensors.
 */
Tensor get_block(const cytnx_uint64 &idx = 0) const;
Tensor &get_block_(const cytnx_uint64 &idx = 0);
std::vector<Tensor> get_blocks() const;
std::vector<Tensor> &get_blocks_(const bool &silent = false);

/**
 * @brief Store a Tensor into a block, IN PLACE.
 * @details BOTH put_block and put_block_ mutate *this; the `_` distinguishes
 *          COPY-IN (put_block deep-copies @p in) from SHARE-IN (put_block_
 *          aliases @p in's storage into the block) — NOT pure-vs-in-place
 *          (finding UT-E8). The deprecated `force` overload is removed
 *          (finding UT-E6). C++ returns *this; the Python binding returns None.
 * @param in the source block (shape must match). @param idx/qidx the target.
 * @return reference to *this.
 */
UniTensor &put_block(const Tensor &in, const cytnx_uint64 &idx = 0);
UniTensor &put_block_(Tensor &in, const cytnx_uint64 &idx = 0);

/**
 * @brief Slice-get / slice-set implementation (behind Python [] operators).
 * @details get(accessors) returns a sub-block UniTensor (shared storage);
 *          set(accessors, rhs) writes @p rhs into the sliced region and returns
 *          *this. These are the C++ implementations Python reaches ONLY via
 *          __getitem__/__setitem__ (finding UT-E2); Dense-only (finding UT-E7).
 * @param accessors per-leg Accessor list. @param rhs the values to assign.
 * @return get: a new UniTensor (shared data). set: reference to *this.
 */
UniTensor get(const std::vector<Accessor> &accessors) const;
UniTensor &set(const std::vector<Accessor> &accessors, const Tensor &rhs);
```
