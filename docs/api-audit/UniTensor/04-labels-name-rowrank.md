# UniTensor — 04. Labels / name / rowrank

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's label/name/rowrank API to match §R exactly. All runtime claims
> verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_04_labels.py` (all `[PASS]`, exit 0), with the
> raw-C++ side of the binding-fidelity findings verified by
> `probes/cpp/UniTensor_04_labels.cpp` against a source-built `libcytnx` (all
> `[PASS]`, exit 0).

**Category scope:** the metadata setters that (re)name legs, name the tensor,
and set the row rank — `set_name`, `set_label`, `set_labels`, `relabel`,
`relabel_`, `relabels`, `relabels_`, `set_rowrank`, `set_rowrank_` — plus the
six raw `c_*` bindings they wrap, which leak into the public surface. Python
bindings: `cytnx_src/pybind/unitensor_py.cpp:231-306`; conti.py wrappers:
`cytnx/UniTensor_conti.py:178-239`; C++ header:
`cytnx_src/include/UniTensor.hpp:2875-3563`.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per public API, with its verbatim live 1.1.0 signature and the probe
assertion backing any runtime claim. `[I]` = in-place. The four `relabel`/
`relabel_` overloads (`(new_labels)`, `(idx, new_label)`, `(old_label,
new_label)`, `(old_labels, new_labels)`) share one row each.

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `relabel` | `relabel(new_labels)` · `relabel(idx, new_label)` · `relabel(old_label, new_label)` · `relabel(old_labels, new_labels)` | `UniTensor` (new, **shared data**) | **Pure** — returns a *distinct* object with the new labels; the receiver is unchanged, but the internal storage is shared. Probe: *"relabel returns a distinct object … leaving the receiver unchanged"* + *"relabel shares data with the source …"*. |
| `relabel_` `[I]` | `relabel_(new_labels)` · `relabel_(idx, new_label)` · `relabel_(old_label, new_label)` · `relabel_(old_labels, new_labels)` | `UniTensor` (self) | **In-place**, returns self (chainable). conti.py wrapper over raw `c_relabel_`. Probe: *"relabel_ mutates in place and returns self"*. |
| `set_label` `[I]` | `set_label(idx, new_label)` · `set_label(old_label, new_label)` | `UniTensor` (self) | **In-place** single-leg relabel; returns self. Overlaps `relabel_`. Probe: *"set_label(idx,new) mutates in place and returns self"*. |
| `set_labels` `[I]` | `set_labels(new_labels)` | `UniTensor` (self) | **Deprecated**; its `c_set_labels` calls `relabel_` — so it is **in-place** despite the `set_`/plural name. Emits `DeprecationWarning`. Probe: *"set_labels is IN-PLACE … despite the pure-sounding name"* + *"set_labels emits a DeprecationWarning"*. |
| `relabels` | `relabels(new_labels)` · `relabels(old_labels, new_labels)` | `UniTensor` (new) | **Deprecated** alias of `relabel` (pure). Emits `DeprecationWarning`. Probe: *"relabels is a deprecated alias of relabel — pure"* + *"relabels emits a DeprecationWarning"*. |
| `relabels_` `[I]` | `relabels_(new_labels)` · `relabels_(old_labels, new_labels)` | `UniTensor` (self) | **Deprecated** alias of `relabel_` (in-place). Emits `DeprecationWarning`. Probe: *"relabels_ is a deprecated alias of relabel_ — in-place, returns self"* + *"relabels_ emits a DeprecationWarning"*. |
| `set_name` `[I]` | `set_name(name)` | `UniTensor` (self) | Sets the tensor name in place; returns self. conti.py over raw `c_set_name`. Probe: *"set_name sets the name in place and returns self"*. |
| `set_rowrank_` `[I]` | `set_rowrank_(new_rowrank)` | `UniTensor` (self) | **In-place** rowrank set; returns self. conti.py over raw `c_set_rowrank_`. Probe: *"set_rowrank_ sets rowrank in place and returns self"*. |
| `set_rowrank` | `set_rowrank(new_rowrank)` | `UniTensor` (new) | **Pure** rowrank set; returns a distinct object, receiver unchanged. Probe: *"set_rowrank (no underscore) is pure …"*. |

**Internal / plumbing (leak into `dir(UniTensor)`):** `c_set_name`,
`c_set_label`, `c_set_labels`, `c_relabel_`, `c_relabels_`, `c_set_rowrank_` —
the raw pybind bindings the conti.py methods above wrap. They are public today
but should never be. Probe: *"the raw plumbing bindings … all LEAK into public
dir(UniTensor)"*; *"raw c_relabel_ returns None …"*.

## A2. C++ ↔ Python mapping

| C++ (`UniTensor.hpp`) | Python | Status | Note |
|---|---|---|---|
| `UniTensor relabel(...) const` (shared data, `:3259`) | `relabel(...)` | identical | pure copy, shared storage (UT-L1) |
| `UniTensor &relabel_(...)` (`:3230`) | `relabel_(...)` | signature-differs | raw `c_relabel_` returns `None`; conti.py re-adds return-self (UT-L2) |
| `UniTensor &set_label(...)` (`:2890,2913`) | `set_label(...)` | identical | conti.py over `c_set_label` (UT-L3) |
| `[[deprecated]] UniTensor &set_labels(...)` (`:2954`) | `set_labels(...)` | signature-differs | binding routes `c_set_labels` → `relabel_`, not `_impl->set_labels` (UT-L4) |
| `[[deprecated]] UniTensor relabels(...) const` (`:3269`) | `relabels(...)` | identical | deprecated; warns (UT-L5) |
| `[[deprecated]] UniTensor &relabels_(...)` (`:3244`) | `relabels_(...)` | signature-differs | raw `c_relabels_` returns `None`; warns; conti.py re-adds return-self (UT-L6) |
| `UniTensor &set_name(...)` (`:2875`) | `set_name(...)` | signature-differs | raw `c_set_name` returns self; conti.py wrapper also returns self (UT-L7) |
| `UniTensor &set_rowrank_(...)` (`:2988`) | `set_rowrank_(...)` | identical | in-place, self (UT-L8) |
| `UniTensor set_rowrank(...) const` (`:2993`) | `set_rowrank(...)` | identical | pure, new object (UT-L9) |
| raw `c_set_name`/`c_set_label`/`c_set_labels`/`c_relabel_`/`c_relabels_`/`c_set_rowrank_` | same names | **leak** | plumbing exposed publicly (UT-L10) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe
assertion from `probes/UniTensor_04_labels.py` (on the 1.1.0 wheel). A **(binding
fidelity)** finding flags where the binding layer — a `*_conti.py` wrapper or a
pybind lambda — changes behavior versus the raw C++ method; **both sides are
runtime-verified**, the raw-C++ side by `probes/cpp/UniTensor_04_labels.cpp`
(links against a source-built `libcytnx`, GCC 13). Source `file:line` cites
remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-L1** | `relabel` returns a **distinct** object whose labels differ but whose **data is shared** with the source | copy/view | **thin pass-through** — the pybind lambda (`:253`) forwards to C++ `relabel(...) const` (`hpp:3259`), which the header documents as returning a new UniTensor sharing storage (`hpp:3255-3257` "*the data is still shared*"). Py probe *"relabel returns a distinct object …"* + *"relabel shares data with the source …"*; **C++ probe confirms** mutating the source is visible through the relabel | **keep**; document the shared-data view semantics |
| **UT-L2** | `relabel_`'s raw binding returns `None`; chainability comes from a conti.py wrapper | **binding fidelity** (N2/B1) | **binding discards the return**: `c_relabel_` (`:262`) is `[](UniTensor&,...){ self.relabel_(...); }` returning `void`; `cytnx/UniTensor_conti.py:203-222` re-adds `return self`. Py probe *"raw c_relabel_ returns None …"*; **C++ probe confirms** `&U.relabel_(...) == &U` — the C++ method already returns `UniTensor&` (`hpp:3230`) | **keep** `relabel_`; the pybind should return self directly, dropping the conti.py shim |
| **UT-L3** | `set_label` (in-place, self) is a **third** overlapping label mechanism alongside `relabel_`/`relabel` | redundancy | **thin pass-through** — conti.py over `c_set_label` (`:234`). Py probe *"set_label(idx,new) mutates in place and returns self"* | **remove** — fold into `relabel_(idx, new_label)` (migration note) |
| **UT-L4** | `set_labels` (deprecated) is **in-place** — its `c_set_labels` calls `relabel_`, contradicting the pure-sounding `set_`/plural name | **binding fidelity** / redundancy | **binding redirects**: `c_set_labels` (`:243`) warns `DeprecationWarning` then calls `self.relabel_(new_labels)` — not `_impl->set_labels`. Py probe *"set_labels is IN-PLACE … despite the pure-sounding name"* + *"… emits a DeprecationWarning"* | **remove** — use `relabel_(new_labels)` (migration note) |
| **UT-L5** | `relabels` is a bound-but-deprecated pure alias of `relabel` | redundancy | **binding warns + forwards**: lambda (`:256,296`) emits `DeprecationWarning` then calls `self.relabel(...)`; C++ `relabels` is `[[deprecated]]` (`hpp:3269`). Py probe *"relabels is a deprecated alias of relabel — pure"* + *"… emits a DeprecationWarning"* | **remove** — use `relabel` (migration note) |
| **UT-L6** | `relabels_` is a bound-but-deprecated in-place alias of `relabel_`; the warning text even names the internal `c_relabels_` | redundancy / **binding fidelity** | **binding warns + forwards**: conti.py (`:225-232`) over `c_relabels_` (`:265`), which warns `"c_relabels_() is deprecated, use relabel_() instead."` then calls `self.relabel_(...)`. Py probe *"relabels_ is a deprecated alias of relabel_ — in-place, returns self"* + *"… emits a DeprecationWarning"* | **remove** — use `relabel_` (migration note) |
| **UT-L7** | `set_name` mutates in place and returns self | (setter; kept) | **thin pass-through** — conti.py over `c_set_name` (`:231`); C++ `set_name` returns `UniTensor&` (`hpp:2875`). Py probe *"set_name sets the name in place and returns self"*; **C++ probe confirms** `&U.set_name(...) == &U` | **keep** — the sole name setter |
| **UT-L8/L9** | `set_rowrank_` (in-place, self) vs `set_rowrank` (pure, new object) is a correct N-underscore pair | (kept) | **thin pass-throughs** — `c_set_rowrank_` = `&set_rowrank_` (`:250`), `set_rowrank` bound directly (`:252`). Py probe *"set_rowrank_ … returns self"* + *"set_rowrank (no underscore) is pure …"*; **C++ probe confirms** `&U.set_rowrank_(0)==&U` while `set_rowrank(1)` is a distinct object | **keep both** |
| **UT-L10** | the raw `c_set_name`/`c_set_label`/`c_set_labels`/`c_relabel_`/`c_relabels_`/`c_set_rowrank_` bindings **leak** into `dir(UniTensor)` | naming + **binding fidelity** | **binding exposes plumbing**: all six `c_*` are `.def(...)`-ed as public methods (`:231-306`) purely so the conti.py wrappers can call them; the `c`-prefix is a reserved raw-binding spelling that must not be public (§R.0 rejects `c`-prefixed in-place spellings). Py probe *"the raw plumbing bindings … all LEAK into public dir(UniTensor)"* | **remove from public API** — bind under a private name (leading `_`) or merge the wrapper into the pybind lambda (migration note) |

## A4. Argument ordering — positional & keyword

All members here take a single primary operand (the new value) and, for the
leg-targeted forms, a target selector. No optional metadata block applies.

| API | positional-required (in order) |
|---|---|
| `relabel` / `relabel_` | `new_labels` **or** `(idx, new_label)` **or** `(old_label, new_label)` **or** `(old_labels, new_labels)` |
| `set_label` | `(idx, new_label)` **or** `(old_label, new_label)` |
| `set_labels` / `relabels` / `relabels_` | `new_labels` (**or** `(old_labels, new_labels)`) |
| `set_name` | `name` |
| `set_rowrank` / `set_rowrank_` | `new_rowrank` |

- **Canonical positional rule (§R.0):** target selector first, then the new
  value — `relabel_(idx, new_label)` and `relabel_(old_label, new_label)`; the
  whole-tensor form takes `new_labels` alone. This matches the live order and
  needs no change. There is no keyword-only block in this category.
- **Naming:** `new_labels`/`new_label`/`new_rowrank`/`name` are already
  consistent, self-describing parameter names (no `arg0` erasure here — contrast
  cat-03 PC1). Keep them.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version label/name/rowrank API.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53).** Every member here is a lowercase
  snake_case *member* — already conformant. No casing change is required; the
  problems are redundancy and leaked plumbing, not casing.
- **N-underscore — a trailing `_` marks in-place (returns `self`); its absence
  marks pure (returns a new object).** The canonical label pair is
  `relabel` (pure copy, shared storage) / `relabel_` (in-place, self); the
  rowrank pair is `set_rowrank` (pure) / `set_rowrank_` (in-place). The
  **`c`-prefixed raw spellings are rejected** as public API (they are the
  plumbing the wrappers call, UT-L10): a public in-place method is *only* the
  trailing-`_` form.
- **One label mechanism.** Collapse the three overlapping mechanisms
  (`relabel`/`relabel_`, `set_label(s)`, `relabels(_)`) onto the single pair
  `relabel` / `relabel_`. All four overloads — whole-tensor `(new_labels)`,
  single-leg by index `(idx, new_label)`, single-leg by name
  `(old_label, new_label)`, and by-list `(old_labels, new_labels)` — live under
  that one base name. `set_label`, `set_labels`, `relabels`, `relabels_` are
  redundant and **removed** (deprecated first, UT-L3/L4/L5/L6).
- **In-place methods return `self` from the binding directly.** `relabel_`,
  `relabels_`, `set_name`, `set_rowrank_` return self in C++ (`UniTensor&`); the
  pybind lambda must return self too, so the conti.py return-self shim (and the
  leaked `c_*` binding it wraps) disappears (UT-L2/L7/L10).
- **Setters keep their names.** `set_name` (name) and `set_rowrank`/
  `set_rowrank_` (rowrank) are distinct from label relabeling and are kept.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- the single canonical label mechanism (all four overloads) ---
    def relabel(self, new_labels: Sequence[str]) -> "UniTensor": ...
    def relabel(self, idx: int, new_label: str) -> "UniTensor": ...          # overload
    def relabel(self, old_label: str, new_label: str) -> "UniTensor": ...    # overload
    def relabel(self, old_labels: Sequence[str],
                new_labels: Sequence[str]) -> "UniTensor": ...               # overload
    #   pure: returns a NEW UniTensor with the new labels; storage is SHARED
    #   with the receiver (metadata-only copy). Receiver labels unchanged.
    def relabel_(self, *args) -> "UniTensor": ...   # same overloads; IN-PLACE, returns self

    # --- name / rowrank setters ---
    def set_name(self, name: str) -> "UniTensor": ...        # in-place, returns self
    def set_rowrank(self, new_rowrank: int) -> "UniTensor": ...   # pure, new object
    def set_rowrank_(self, new_rowrank: int) -> "UniTensor": ...  # in-place, returns self
```

`relabel_`/`set_name`/`set_rowrank_` return `self` **from the binding** (no
conti.py shim). The six `c_*` plumbing bindings become private (leading `_`) or
are inlined into the pybind lambdas — they are **not** public members.

| API | Verdict | Behavior contract |
|---|---|---|
| `relabel` | **keep** (UT-L1) | Pure: returns a new UniTensor with the new labels; internal data shared with the receiver, which is unchanged. Four overloads. |
| `relabel_` | **keep** (UT-L2; bind return-self directly) | In-place relabel; returns self (chainable). Four overloads. |
| `set_name` | **keep** (UT-L7) | Set the tensor name in place; returns self. |
| `set_rowrank` | **keep** (UT-L9) | Pure: returns a new UniTensor with the new rowrank; receiver unchanged. |
| `set_rowrank_` | **keep** (UT-L8) | Set rowrank in place; returns self. |
| `set_label` | **remove** (UT-L3) | Redundant single-leg in-place relabel. *Migration:* deprecated alias of `relabel_(idx, new_label)` / `relabel_(old_label, new_label)`, emitting `DeprecationWarning` for one minor release, then deleted. |
| `set_labels` | **remove** (UT-L4) | Redundant in-place relabel (misleadingly named). *Migration:* deprecated alias of `relabel_(new_labels)`, `DeprecationWarning` for one release, then deleted. |
| `relabels` | **remove** (UT-L5) | Redundant pure alias. *Migration:* deprecated alias of `relabel`, `DeprecationWarning` for one release, then deleted. |
| `relabels_` | **remove** (UT-L6) | Redundant in-place alias. *Migration:* deprecated alias of `relabel_`, `DeprecationWarning` for one release, then deleted. |

**Internal / plumbing — hidden, not public API.** The six raw bindings below
are covered here (they are live public members today) with a **remove** verdict:
hide them behind a leading underscore or inline them into their pybind lambda.
None carry a docstring — they are not public surface.

| API | Verdict | Behavior contract |
|---|---|---|
| `c_set_name` | **remove** (UT-L10) | Raw plumbing for `set_name`. *Migration:* rename to `_set_name` (private) or inline into the pybind lambda; no public exposure. |
| `c_set_label` | **remove** (UT-L10) | Raw plumbing for `set_label` (itself removed). *Migration:* delete once `set_label` is gone. |
| `c_set_labels` | **remove** (UT-L10) | Raw plumbing that redirects to `relabel_`. *Migration:* delete with `set_labels`. |
| `c_relabel_` | **remove** (UT-L10) | Raw plumbing for `relabel_` (returns `None`). *Migration:* fold into the `relabel_` pybind lambda (which returns self); no public exposure. |
| `c_relabels_` | **remove** (UT-L10) | Raw plumbing for `relabels_`. *Migration:* delete with `relabels_`. |
| `c_set_rowrank_` | **remove** (UT-L10) | Raw plumbing for `set_rowrank_`. *Migration:* fold into the `set_rowrank_` pybind lambda; no public exposure. |

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python
surface, **R.2b** Doxygen for the C++ surface. Only kept members are documented
(removed/plumbing members carry no docstring).

### R.2a Python API (numpy-style)

### `relabel` / `relabel_`

```
UniTensor.relabel(new_labels)                       -> UniTensor   # pure
UniTensor.relabel(idx, new_label)                   -> UniTensor
UniTensor.relabel(old_label, new_label)             -> UniTensor
UniTensor.relabel(old_labels, new_labels)           -> UniTensor
UniTensor.relabel_(...)  # same overloads           -> UniTensor   # in-place, self

Relabel the bond legs of this UniTensor.

`relabel` is PURE: it returns a new UniTensor carrying the new labels while
leaving this tensor's labels unchanged. The returned tensor SHARES its internal
data with this one (a metadata-only copy) — mutating either tensor's elements is
visible through the other (finding UT-L1).

`relabel_` is the IN-PLACE form: it renames this tensor's legs and returns self
for chaining (finding UT-L2).

Parameters
----------
new_labels : sequence of str
    Replacement labels for every leg (whole-tensor form).
idx : int
    Index of a single leg to relabel.
old_label, new_label : str
    Rename the leg currently labelled `old_label` to `new_label`.
old_labels, new_labels : sequence of str
    Rename each leg in `old_labels` to the matching entry in `new_labels`.

Returns
-------
UniTensor
    `relabel`: a new tensor (shared data). `relabel_`: self.

Notes
-----
Replaces the removed `set_label`, `set_labels`, `relabels`, `relabels_`
(findings UT-L3–L6). No label may duplicate another leg's label.
```

### `set_name`

```
UniTensor.set_name(name) -> UniTensor

Set this tensor's name in place and return self (finding UT-L7).

Parameters
----------
name : str
    The new tensor name.

Returns
-------
UniTensor
    self (chainable).
```

### `set_rowrank` / `set_rowrank_`

```
UniTensor.set_rowrank(new_rowrank)  -> UniTensor   # pure, new object
UniTensor.set_rowrank_(new_rowrank) -> UniTensor   # in-place, self

Set the row rank (number of bra-space legs), which partitions the legs into the
row/column spaces used by the linear-algebra routines.

`set_rowrank` is PURE — it returns a new UniTensor with the requested rowrank,
leaving this tensor unchanged (finding UT-L9). `set_rowrank_` sets it IN-PLACE
and returns self (finding UT-L8).

Parameters
----------
new_rowrank : int
    The number of legs in the row (bra) space.

Returns
-------
UniTensor
    `set_rowrank`: a new tensor. `set_rowrank_`: self.
```

### R.2b C++ API (Doxygen)

C++ already returns `UniTensor&`/`UniTensor` per the N-underscore split; the
next version must have the *pybind lambdas* return these directly (removing the
conti.py shim and the leaked `c_*` bindings, UT-L2/L10).

```cpp
/**
 * @brief Relabel the bond legs, returning a NEW UniTensor (data shared).
 * @details Pure: the returned tensor carries @p new_labels while *this is
 *          unchanged, but the internal storage is SHARED (metadata-only copy,
 *          finding UT-L1). Overloads relabel a single leg by index or name, or
 *          a list of legs.
 * @param new_labels replacement labels for every leg.
 * @return a new UniTensor with the new labels (shared data).
 */
UniTensor relabel(const std::vector<std::string> &new_labels) const;
UniTensor relabel(const cytnx_int64 &idx, const std::string &new_label) const;
UniTensor relabel(const std::string &old_label, const std::string &new_label) const;
UniTensor relabel(const std::vector<std::string> &old_labels,
                  const std::vector<std::string> &new_labels) const;

/**
 * @brief Relabel the bond legs IN PLACE; returns *this (chainable).
 * @details Same overloads as relabel(); mutates the receiver. The Python
 *          binding must return self directly (finding UT-L2) — no wrapper.
 * @return reference to *this.
 */
UniTensor &relabel_(const std::vector<std::string> &new_labels);
UniTensor &relabel_(const cytnx_int64 &idx, const std::string &new_label);
UniTensor &relabel_(const std::string &old_label, const std::string &new_label);
UniTensor &relabel_(const std::vector<std::string> &old_labels,
                    const std::vector<std::string> &new_labels);

/**
 * @brief Set this tensor's name in place; returns *this.
 * @param in the new name.
 * @return reference to *this (chainable) — the fidelity the Python binding
 *         must preserve without a wrapper (finding UT-L7).
 */
UniTensor &set_name(const std::string &in);

/**
 * @brief Set the row rank (bra-space leg count).
 * @details set_rowrank_ mutates in place and returns *this (UT-L8);
 *          set_rowrank is pure and returns a new UniTensor (UT-L9).
 * @param new_rowrank number of legs in the row (bra) space.
 * @return set_rowrank_: reference to *this. set_rowrank: a new UniTensor.
 */
UniTensor &set_rowrank_(const cytnx_uint64 &new_rowrank);
UniTensor set_rowrank(const cytnx_uint64 &new_rowrank) const;
```
