# UniTensor — 10. Contraction & networks

> **Superset-method category** (see the pilots [`01-construction-init.md`](01-construction-init.md),
> [`02-static-generators.md`](02-static-generators.md), and the siblings
> [`06-element-block-access.md`](06-element-block-access.md),
> [`08-linalg-operations.md`](08-linalg-operations.md)).
> Split into **Analysis** and a self-contained **Recommendation** that is the
> *normative spec for the next version of Cytnx* — implement the next major
> version's UniTensor contraction surface to match §R exactly. All runtime claims
> verified against `cytnx==1.1.0` via
> `docs/api-audit/probes/UniTensor_10_contraction.py` (all `[PASS]`, exit 0).
> **No C++ probe accompanies this category:** the only C++↔Python divergence here
> (`Contracts`'s deprecation is silent in Python) is a **compile-time-only** C++
> `[[deprecated]]` attribute with **no runtime behavior** — a raw-C++ runtime
> probe would observe nothing, so the finding is Python-probe-verified and gate 4
> is recorded as *"no binding-fidelity finding"*.

**Category scope.** This category audits the **contraction** surface that acts on
`UniTensor`: the member `contract` (lowercase — the primary UniTensor method) and
the three related **free functions** `Contract`, `Contracts`, and `ncon`
(Capitalized `Contract`/`Contracts` because they act on objects; `ncon` kept
lowercase as the community-standard tensor-network name). The heavyweight,
stateful **`Network`** class is a **separate class** and is **cross-referenced,
not re-audited** here — see [`../per-class/network.md`](../per-class/network.md)
(its headline finding P1 — `Network.Contract(...).Launch()` **segfaults** — is
quoted below as UT-N6, not re-derived).
C++ header: `cytnx_src/include/UniTensor.hpp:322,801,1462,2228` (the virtual
`contract`), `:4713-4732` (the `UniTensor::contract` member),
`:6000-6001,6015-6016` (free `Contract` overloads), `:6023-6028` (deprecated
`Contracts`); free `ncon`: `cytnx_src/include/ncon.hpp`.
Python bindings: member `contract` `cytnx_src/pybind/unitensor_py.cpp:758`; free
`Contract`/`Contracts` `:1612-1625`; `ncon` `cytnx_src/pybind/ncon_py.cpp:26-32`.

> **Coverage note (validator).** `validate_doc.py UniTensor <dir>` checks
> `dir(cytnx.UniTensor)`. Of the four APIs here, **only `contract` is a UniTensor
> member** and is the one the validator counts/requires from this file; the free
> functions `Contract`/`Contracts`/`ncon` are **not** UniTensor members (they live
> in the top-level `cytnx` namespace), so they add **no** UniTensor-member
> coverage and cause no regression — exactly as the linalg free functions in cat
> 08. They still each get an R.1 verdict row below.

---

# Analysis

## A1. Current API (cytnx 1.1.0)

One row per API, with its verbatim live 1.1.0 signature and the probe assertion
backing any runtime claim. The member `contract` is **lowercase**; the free
`Contract`/`Contracts` are **Capitalized**; `ncon` is **lowercase** — all three
spellings are correct under N-casing (finding UT-N1). All contraction operations
here are **pure** (return a new UniTensor; inputs untouched).

| API | Live signature (1.1.0) | Returns | Description & evidence |
|---|---|---|---|
| `contract` (member) | `contract(self, inR, mv_elem_self=False, mv_elem_rhs=False)` | `UniTensor` (new) | **Member** contraction: trace this tensor's legs against `inR`'s legs that share a **common label**. Pure — returns a new UniTensor, operands intact. `mv_elem_self`/`mv_elem_rhs` request a contiguous "move-element" alignment of each operand. Probe: *"member `contract` signature is contract(self, inR, mv_elem_self=False, mv_elem_rhs=False)"* + *"member `contract` contracts the common-label leg `k`, giving open legs [a,b] shape [2,2]"* + *"member `contract` result equals the hand contraction A@B elementwise"* + *"member `contract` is pure — returns a NEW UniTensor and leaves the operands intact"*. |
| `Contract` (free) | overload 1: `Contract(Tl, Tr, cacheL=False, cacheR=False)` · overload 2: `Contract(TNs, order='', optimal=True)` | `UniTensor` (new) | **Free** pairwise / multi-tensor contraction by **common labels** — the same conceptual operation as the member `contract`. Overload 1 contracts two tensors (`cacheL`/`cacheR` request contiguous alignment); overload 2 contracts a **list** `TNs` with an optional explicit `order` string and `optimal` order search. Probe: *"free `Contract` is overloaded: pairwise (Tl,Tr,cacheL,cacheR) and list (TNs,order,optimal)"* + *"free `Contract(Tl,Tr)` produces the SAME result as the member `contract` on the same pair …"* + *"free `Contract([TNs])` (list overload) also produces the same result"*. |
| `Contracts` (free) | `Contracts(TNs, order='', optimal=True)` | `UniTensor` (new) | **Deprecated plural spelling** of `Contract`'s list overload; identical behavior. C++ carries a `[[deprecated]]` attribute (`hpp:6023`) but that is **compile-time only** — calling it from Python emits **no** warning. Probe: *"`Contracts` still runs and returns the SAME result as `Contract` (deprecated alias)"* + *"`Contracts` emits NO Python runtime warning …"*. |
| `ncon` (free) | `ncon(tensor_list_in, connect_list_in, check_network=False, optimize=False, cont_order=[], out_labels=[])` | `UniTensor` (new) | **Free** one-shot network contractor using the **ncon index convention**: one integer list per tensor — a **positive** integer names a bond contracted between the two tensors that share it (must appear **exactly twice**); a **negative** integer names an **open** output leg, ordered `-1, -2, …` in the result. `check_network` validates the convention. Probe: *"ncon([A,B],[[-1,1],[1,-2]]) equals the hand contraction A@B elementwise …"* + *"ncon orders open output legs by -1,-2,… (leg -1 first) …"* + *"ncon with an all-positive network fully contracts to a rank-1 scalar"* + *"ncon(check_network=True) RAISES when a positive label does not appear exactly twice"* + *"ncon does NOT mutate its input UniTensors"*. |

**Cross-reference (separate class — not re-audited):** the stateful `Network`
builder and its `def_static` `Network.Contract(...)` factory live in
[`../per-class/network.md`](../per-class/network.md). Its `Contract(...).Launch()`
**segfaults** (network.md finding P1/B4); `ncon` and the member/free `contract`
are the working one-shot contraction paths (finding UT-N6 below).

## A2. C++ ↔ Python mapping

The member `contract` and all three free functions are **direct pass-through
pybind lambdas** to the same-named `cytnx::` C++ function/overload — no `conti.py`
wrapper, no signature change, no runtime behavior change. "Status" therefore reads
**identical** throughout, with one caveat: the C++ `Contracts` `[[deprecated]]`
attribute (a compile-time diagnostic) does **not** cross the binding boundary, so
the Python `Contracts` is silently un-deprecated at runtime (finding UT-N3).

| C++ (`UniTensor.hpp` / `ncon.hpp`) | Python | Status | Note |
|---|---|---|---|
| `UniTensor contract(const UniTensor &inR, bool mv_elem_self=false, bool mv_elem_rhs=false)` (`:4729`) | `contract(inR, mv_elem_self, mv_elem_rhs)` (`unitensor_py.cpp:758`) | identical | lowercase **member** — correct N-casing (UT-N1); contracts common-label legs (UT-N2) |
| `UniTensor Contract(const UniTensor &inL, const UniTensor &inR, bool cacheL=false, bool cacheR=false)` (`:6000`) | `Contract(Tl, Tr, cacheL, cacheR)` (`unitensor_py.cpp:1612`) | identical | Capitalized **free** function (UT-N1); same operation as member `contract` (UT-N2) |
| `UniTensor Contract(const std::vector<UniTensor> &TNs, const std::string &order, bool optimal)` (`:6015`) | `Contract(TNs, order='', optimal=True)` (`unitensor_py.cpp:1617`) | identical | list overload; `order`/`optimal` control the contraction plan |
| `[[deprecated]] UniTensor Contracts(const std::vector<UniTensor> &TNs, const std::string &order, bool optimal)` (`:6023-6028`) | `Contracts(TNs, order='', optimal=True)` (`unitensor_py.cpp:1622`) | **signature-differs (deprecation lost)** | C++ `[[deprecated]]` is **compile-time only**; the Python binding forwards it with **no** runtime warning (UT-N3) |
| `UniTensor ncon(tensor_list_in, connect_list_in, check_network=false, optimize=false, cont_order={}, out_labels={})` (`ncon.hpp`) | `ncon(...)` (`ncon_py.cpp:26`) | identical | lowercase free function — community name kept (UT-N1); index convention correct (UT-N4) |
| `static Network Network::Contract(utensors, Tout, alias={}, contract_order="")` (`Network.hpp`) | `Network.Contract(...)` (`network_py.cpp:119`) | **cross-ref — separate class** | `Launch()` on the result **segfaults** (network.md P1/B4); not a UniTensor member (UT-N6) |

## A3. Findings

Each finding cites its evidence. Python behavioral claims quote a probe assertion
from `probes/UniTensor_10_contraction.py` (on the 1.1.0 wheel). **There is no
binding-fidelity finding in this category:** the member/free contraction functions
are direct pass-through pybind lambdas over the C++ `cytnx::` functions (no
`conti.py` wrapper). The one C++↔Python divergence (UT-N3, the silent `Contracts`
deprecation) is a **compile-time-only** C++ attribute with **no runtime side** to
probe, so gate 4 is recorded as *"no binding-fidelity finding"*. Source `file:line`
cites remain for traceability.

| ID | Finding | Type | What the binding does · evidence | Recommendation |
|---|---|---|---|---|
| **UT-N1** | the contraction surface **correctly** mixes casings: the member `contract` is **lowercase**, the free `Contract`/`Contracts` are **Capitalized**, and `ncon` is **lowercase** | naming (N-casing — **positive demonstration**) | **each spelling is correct under SciPostPhysCodeb.53.** A *member* function lowercases → `contract` (`hpp:4729`, member of `UniTensor`); a *free function acting on an object* Capitalizes → `Contract`/`Contracts` (`hpp:6000,6023`, free `cytnx::` functions); and `ncon` is the field-standard tensor-network name (positive integers = contracted bonds, negatives = open legs), kept **lowercase** as the agreed community-name exception (parallel to `matvec`/`svd`/`qr`). Py probe *"`contract` is a UniTensor MEMBER (lowercase — correct …)"* + *"there is NO Capitalized `Contract` member on UniTensor (it is a free function)"* + *"`Contract` is a Capitalized FREE function (acts on objects …)"* + *"`ncon` is a lowercase FREE function (community-standard TN name …)"* | **KEEP all spellings**: `contract` (member, lowercase), `Contract`/`Contracts` (free, Capitalized), `ncon` (lowercase, **stated community exception**). Do **not** snake_case the free functions and do **not** Capitalize `ncon`. Document the member↔free casing split (cross-ref cat 08 UT-X1). |
| **UT-N2** | the member `contract` and the free `Contract` do the **same conceptual thing** (contract common-label legs) — an **idiom split**: two spellings for one operation | naming / redundancy (idiom split) | **verified-equal results.** On the same labelled pair `A[a,k]`,`B[k,b]`, the member `A.contract(B)`, the free pairwise `Contract(A,B)`, and the free list `Contract([A,B])` all trace the common label `k` and return the identical `[a,b]` tensor — equal to the hand contraction `A@B` elementwise. Py probe *"member `contract` result equals the hand contraction A@B elementwise"* + *"free `Contract(Tl,Tr)` produces the SAME result as the member `contract` on the same pair (idiom split — two spellings, one operation)"* + *"free `Contract([TNs])` (list overload) also produces the same result"* | **choose ONE primary contraction idiom and document it as such.** Recommend the free **`Contract`** as the primary user-facing spelling (it scales to the list/`order`/`optimal` multi-tensor case that the binary member cannot), with the member `contract` kept as the low-level binary primitive it wraps. Document the equivalence and the "when to use which" so the split is a documented convenience, not an undocumented duplication. |
| **UT-N3** | `Contracts` (plural) is a **deprecated** alias of `Contract`, but its deprecation is **silent in Python** — the C++ `[[deprecated]]` attribute is compile-time only and the pybind binding forwards it with **no runtime warning** | naming (deprecation) / **parity** (deprecation lost across the binding) | **the deprecation never reaches a Python user.** The C++ declares `[[deprecated("Please use … Contract … instead.")]] Contracts(…)` (`hpp:6023-6028`) — a compiler diagnostic that fires only when C++ *code* calls it; the Python lambda (`unitensor_py.cpp:1622`) wraps the same body and emits nothing at call time, so `cytnx.Contracts([...])` runs identically to `Contract` with no `DeprecationWarning`/`FutureWarning`. Py probe *"`Contracts` still runs and returns the SAME result as `Contract` (deprecated alias)"* + *"`Contracts` emits NO Python runtime warning — the C++ [[deprecated]] is compile-time only, so the deprecation never reaches a Python user"* | **make the deprecation reach Python, then remove.** Bind `Contracts` to emit a real Python `DeprecationWarning` ("*`Contracts` is deprecated; use `Contract`*") for one minor release, then delete it (migration alias). `Contract` (singular) is the single kept spelling of the multi-tensor contraction. |
| **UT-N4** | `ncon`'s **index convention is correct** — positives label contracted bonds (each appearing **exactly twice**), negatives label open output legs ordered `-1, -2, …` — verified against a hand contraction | correctness (verified) | **value-verified against matmul and trace.** `ncon([A,B],[[-1,1],[1,-2]])` contracts the shared positive bond `1` and keeps the negatives `-1,-2` open — reproducing `A@B` **elementwise**; swapping to `[[-2,1],[1,-1]]` transposes the output (leg `-1` first, regardless of which tensor carries it); an all-positive network `[[1,2],[2,1]]` fully contracts to a **rank-1 scalar** equal to `trace(A@B)==50`; and `check_network=True` **raises** when a positive label is not paired exactly twice. Py probe *"ncon([A,B],[[-1,1],[1,-2]]) equals the hand contraction A@B elementwise (positives=contracted bond, negatives=open legs)"* + *"ncon orders open output legs by -1,-2,… (leg -1 first) …"* + *"ncon with an all-positive network fully contracts to a rank-1 scalar"* + *"ncon(check_network=True) RAISES when a positive label does not appear exactly twice"* + *"ncon does NOT mutate its input UniTensors"* | **KEEP `ncon`** (lowercase — UT-N1) as the working one-shot network contractor. **Recommend defaulting `check_network=True`** so malformed networks are caught with a catchable exception rather than risking a downstream crash (cross-ref network.md's `ncon` recommendation and the `Network.Contract` segfault UT-N6). Document the convention (positives=contracted, appear twice; negatives=open, ordered `-1,-2,…`) explicitly. |
| **UT-N5** | all contraction operations here are **pure** — they return a **new** UniTensor and do **not** mutate their operands | copy/view / N-underscore (documented) | **no in-place contraction spelling exists (correctly).** `A.contract(B)`, `Contract`, `Contracts`, and `ncon` each return a fresh UniTensor; `A` and `B` are unchanged afterwards. There is no `contract_` trailing-underscore form (a binary contraction has no meaningful in-place target). Py probe *"member `contract` is pure — returns a NEW UniTensor and leaves the operands intact"* + *"ncon does NOT mutate its input UniTensors"* | **keep pure-only**; **document** that contraction produces a new tensor (no `_` in-place variant), matching the community expectation for `ncon`/`Contract`. |
| **UT-N6** | *(cross-reference — separate class, not re-derived)* `Network.Contract(...).Launch()` **segfaults** in 1.1.0; `ncon` and the member/free `contract` are the working one-shot contraction paths | correctness (cross-ref network.md P1/B4) | **the `Network` one-shot factory is unusable.** Per [`../per-class/network.md`](../per-class/network.md) finding **P1**: `Network::Contract` (a `def_static`, `network_py.cpp:119`) builds a plan whose `.Launch()` crashes the process with `SIGSEGV` for every reasonable `Tout` spelling — verified there in a subprocess. **`Network` is a separate class**, so this is quoted, not re-audited. | **use `ncon` (or free `Contract`) as the one-shot contractor** (network.md recommends **removing** `Network.Contract`). This category's `ncon`/`Contract`/`contract` are the recommended, working surface. See network.md for the `Network` verdicts. |

## A4. Argument ordering — positional & keyword

Each API takes the **tensor operand(s) as the primary positional argument(s)**,
then contraction parameters. There is no keyword-only metadata block (these are
operations, not constructors).

| API | positional-required (in order) | operation parameters (keyword-capable) |
|---|---|---|
| `contract` (member) | `inR` | `mv_elem_self=False`, `mv_elem_rhs=False` |
| `Contract` (free, overload 1) | `Tl`, `Tr` | `cacheL=False`, `cacheR=False` |
| `Contract` (free, overload 2) | `TNs` | `order=''`, `optimal=True` |
| `Contracts` (free) | `TNs` | `order=''`, `optimal=True` |
| `ncon` (free) | `tensor_list_in`, `connect_list_in` | `check_network=False`, `optimize=False`, `cont_order=[]`, `out_labels=[]` |

- **Canonical positional rule (§R.0):** the tensor operand(s) first, then the
  contraction parameters — matches the live order; no reordering needed.
- **Naming:** `inR`/`Tl`/`Tr`/`TNs`/`tensor_list_in`/`connect_list_in` are
  self-describing within each function. The `mv_elem_self`/`mv_elem_rhs` and
  `cacheL`/`cacheR` flags are low-level alignment hints — document them as
  performance toggles (both default `False`/`True` faithfully to C++).
- **`ncon` default (UT-N4):** recommend flipping `check_network` to default
  `True` so the index convention is validated by default.

---

# R. Recommendation — normative spec for the next version

*Self-contained: fully specifies the next-version UniTensor contraction surface.
Implement Cytnx to match it.*

## R.0 Normative conventions

- **N-casing (SciPostPhysCodeb.53) — the member↔free split is honored (UT-N1).**
  The **member** `contract` is **lowercase**; the **free functions**
  `Contract`/`Contracts` are **Capitalized** (they act on objects); and `ncon`
  is **kept lowercase** as the **agreed community-name exception** — a
  field-standard tensor-network term (like `matvec`/`svd`/`qr`), not to be
  Capitalized nor snake-cased. This is the same members-lowercase /
  free-functions-Capitalized demonstration as cat 08 (UT-X1); state it explicitly
  in the docs.
- **One primary contraction idiom (UT-N2).** The free **`Contract`** is the
  primary user-facing spelling (it scales to the multi-tensor list/`order`/
  `optimal` case); the member `contract` is the binary low-level primitive it
  wraps. Documented as a convenience split, not an undocumented duplication.
- **Deprecations must reach Python (UT-N3).** A C++ `[[deprecated]]` attribute is
  compile-time only and does **not** cross the pybind boundary — deprecated
  bindings (like `Contracts`) must emit a real Python `DeprecationWarning` for one
  minor release, then be removed.
- **Contraction is pure (UT-N5).** All four APIs return a new UniTensor and leave
  their operands intact; there is no in-place `contract_` form.
- **Binding fidelity: none.** The member/free contraction functions are direct
  pass-through pybind lambdas over the C++ `cytnx::` functions — no `conti.py`
  wrapper, no leaked `c*` binding, no runtime signature change. Gate 4 (raw-C++
  probe) is skipped: *no binding-fidelity finding*.

## R.1 Recommended API (exact signatures + behavior contract)

```python
class UniTensor:
    # --- member contraction (lowercase; binary primitive) ---
    def contract(self, inR: "UniTensor", *,
                 mv_elem_self: bool = False, mv_elem_rhs: bool = False) -> "UniTensor": ...
    #   Contract this tensor's legs against inR's legs that share a COMMON LABEL.
    #   Pure: returns a NEW UniTensor; self and inR are unchanged.

# cytnx — free contraction functions (Capitalized; act on objects), ncon lowercase
def Contract(Tl: "UniTensor", Tr: "UniTensor", *,
             cacheL: bool = False, cacheR: bool = False) -> "UniTensor": ...          # overload 1
def Contract(TNs: "Sequence[UniTensor]", *,
             order: str = "", optimal: bool = True) -> "UniTensor": ...               # overload 2
#   PRIMARY user-facing contraction; contracts by common labels. Same operation
#   as the member `contract` (UT-N2); the list overload scales to N tensors.

def ncon(tensor_list_in: "Sequence[UniTensor]",
         connect_list_in: "Sequence[Sequence[int]]", *,
         check_network: bool = True,          # was False (UT-N4): validate by default
         optimize: bool = False,
         cont_order: "Sequence[int]" = (),
         out_labels: "Sequence[str]" = ()) -> "UniTensor": ...
#   ncon index convention: POSITIVE ints = contracted bonds (each appears exactly
#   twice); NEGATIVE ints = open output legs, ordered -1, -2, ...  Pure.

# REMOVED next release: `Contracts` (deprecated plural spelling of Contract).
```

| API | Verdict | Behavior contract |
|---|---|---|
| `contract` (member) | **keep (lowercase)** (UT-N1/N2/N5) | Binary contraction of this tensor with `inR` over their common-label legs; pure (new UniTensor). `mv_elem_self`/`mv_elem_rhs` are contiguous-alignment performance hints. Documented as the low-level primitive behind the free `Contract`. |
| `Contract` (free) | **keep (Capitalized), make primary** (UT-N1/N2) | The primary contraction spelling: overload 1 contracts two tensors (`cacheL`/`cacheR` alignment hints); overload 2 contracts a list `TNs` with optional `order`/`optimal`. Same result as the member `contract` on a binary pair (probe-verified). |
| `Contracts` (free) | **remove (deprecate first)** (UT-N3) | Deprecated plural alias of `Contract`'s list overload; **through 1.1.0 its deprecation was silent in Python**. *Migration:* bind a real `DeprecationWarning` for one minor release, then delete. Use `Contract`. |
| `ncon` (free) | **keep (lowercase — community exception), default `check_network=True`** (UT-N1/N4) | One-shot network contractor in the ncon index convention (positives = contracted bonds appearing exactly twice; negatives = open legs ordered `-1,-2,…`). Pure. *Migration:* flip `check_network` default to `True` so malformed networks raise rather than risk a downstream crash. |

**Cross-reference (separate class — see [`../per-class/network.md`](../per-class/network.md)).**
`Network.Contract(...)`'s `Launch()` **segfaults** (network.md P1/B4) and is
**removed** there; `ncon`/`Contract`/`contract` are the working contraction paths
(UT-N6). Not re-audited or re-verdicted here.

**No binding-fidelity / plumbing findings.** Unlike cats 04–07, this category
surfaces **no** leaked `c*` bindings and **no** `conti.py` wrappers — the pybind
layer forwards directly to `cytnx::`. Gate 4 (raw-C++ probe) is skipped: *no
binding-fidelity finding.*

## R.2 Docstrings (normative)

Documented in both languages' idioms: **R.2a** numpy-style for the Python surface,
**R.2b** Doxygen for the C++ surface. Kept members/functions are documented;
removed `Contracts` carries no docstring beyond its migration note. `ncon` is a
Python/C++ free function bound identically, so it carries both.

### R.2a Python API (numpy-style)

### `contract` (member) / `Contract` (free)

```
UniTensor.contract(inR, *, mv_elem_self=False, mv_elem_rhs=False) -> UniTensor
cytnx.Contract(Tl, Tr, *, cacheL=False, cacheR=False)             -> UniTensor
cytnx.Contract(TNs, *, order="", optimal=True)                    -> UniTensor

Contract UniTensors over their COMMON-LABEL legs.

Two legs are contracted when they carry the SAME label on the two tensors; the
remaining legs stay open in the result. `A.contract(B)` (the member primitive),
`Contract(A, B)` (the free pairwise form), and `Contract([A, B, ...])` (the free
list form) all compute the same contraction — for a labelled pair `A[a,k]`,
`B[k,b]` they trace the shared `k` and return the `[a, b]` tensor, elementwise
equal to the matrix product A @ B (finding UT-N2).

All forms are PURE: they return a NEW UniTensor and leave the operands unchanged
(finding UT-N5). The list overload additionally accepts an explicit contraction
`order` string and an `optimal`-order search for multi-tensor networks.

Parameters
----------
inR / Tl, Tr : UniTensor
    The tensor(s) to contract with (member: the right operand; free pairwise: the
    two operands).
TNs : sequence of UniTensor
    The tensors to contract (free list overload).
mv_elem_self, mv_elem_rhs / cacheL, cacheR : bool, optional
    Low-level contiguous-alignment ("move-element" / cache) performance hints.
order : str, optional
    Explicit contraction order over the network (list overload).
optimal : bool, optional
    Whether to search for an optimal contraction order (list overload).

Returns
-------
UniTensor
    The contracted result (a new tensor).

Notes
-----
`contract` is the lowercase MEMBER; `Contract` is the Capitalized FREE function
acting on objects (finding UT-N1) — the two coexist by design (cross-ref cat 08).
`Contract` (free) is the recommended PRIMARY spelling because its list overload
scales to N-tensor networks; the member `contract` is the binary primitive it
wraps. The plural `Contracts` is DEPRECATED — use `Contract` (finding UT-N3).

See Also
--------
ncon : one-shot contraction by integer index lists (ncon convention).
```

### `ncon`

```
cytnx.ncon(tensor_list_in, connect_list_in, *, check_network=True,
           optimize=False, cont_order=(), out_labels=()) -> UniTensor

Contract a tensor network using the ncon index convention.

`connect_list_in` gives one integer list per tensor in `tensor_list_in`. A
POSITIVE integer names a bond CONTRACTED between the two tensors that share it
(it must appear EXACTLY TWICE across the whole network); a NEGATIVE integer names
an OPEN output leg, and the result's legs are ordered -1, -2, ... (finding UT-N4).

For example `ncon([A, B], [[-1, 1], [1, -2]])` contracts the shared bond `1` and
keeps `-1`, `-2` open — reproducing the matrix product A @ B elementwise. An
all-positive network (every leg contracted) fully contracts to a rank-1 scalar
(e.g. `ncon([A, B], [[1, 2], [2, 1]])` is trace(A @ B)). ncon is PURE — it does
not mutate its inputs.

Parameters
----------
tensor_list_in : sequence of UniTensor
    The tensors to contract.
connect_list_in : sequence of sequence of int
    One integer list per tensor (positives = contracted bonds, appearing exactly
    twice; negatives = open legs, ordered -1, -2, ...).
check_network : bool, optional
    Validate the index convention (each positive label appears exactly twice),
    raising RuntimeError otherwise. Default True in the recommended API (was
    False through 1.1.0, finding UT-N4).
optimize : bool, optional
    Whether to search for an optimal contraction order.
cont_order : sequence of int, optional
    Explicit contraction order over the positive bond labels.
out_labels : sequence of str, optional
    Labels to assign to the output legs.

Returns
-------
UniTensor
    The contracted result. A fully-contracted (all-positive) network yields a
    rank-1 scalar.

Notes
-----
`ncon` is kept LOWERCASE as the agreed community-standard tensor-network name
(finding UT-N1) — not Capitalized like the free `Contract`. It is the working
one-shot contractor; the `Network.Contract(...)` static factory segfaults on
`Launch()` (finding UT-N6 / network.md P1) — use `ncon` or `Contract` instead.
```

### R.2b C++ API (Doxygen)

The C++ member/free functions already carry the correct casing and pass-through
bindings; the next version's changes are: emit a real Python `DeprecationWarning`
for `Contracts` then remove it (UT-N3), default `ncon`'s `check_network` to `true`
(UT-N4), and document the member↔free idiom split (UT-N2). No binding-fidelity
change is required — the pybind lambdas already forward directly.

```cpp
namespace cytnx {

  /**
   * @brief Contract this UniTensor with @p inR over their common-label legs.
   * @details The lowercase MEMBER contraction primitive (finding UT-N1): legs
   *          sharing a label on the two tensors are traced; the rest stay open.
   *          PURE — returns a new UniTensor, operands unchanged (finding UT-N5).
   *          mv_elem_self/mv_elem_rhs request contiguous "move-element" alignment.
   *          Equivalent to the free Contract(*this, inR) (finding UT-N2).
   * @param inR the UniTensor to contract with.
   * @param mv_elem_self,mv_elem_rhs contiguous-alignment performance hints.
   * @return the contracted UniTensor (new).
   */
  UniTensor UniTensor::contract(const UniTensor &inR, const bool &mv_elem_self = false,
                                const bool &mv_elem_rhs = false) const;

  /**
   * @brief Contract UniTensors over their common-label legs (free function).
   * @details Capitalized FREE function acting on objects (finding UT-N1); the
   *          recommended PRIMARY contraction spelling (finding UT-N2). Overload 1
   *          contracts two tensors; overload 2 contracts a list with an optional
   *          order string and optimal-order search. PURE.
   * @param inL,inR / TNs the operand(s). @param cacheL,cacheR alignment hints.
   * @param order explicit contraction order. @param optimal optimize the order.
   * @return the contracted UniTensor (new).
   */
  UniTensor Contract(const UniTensor &inL, const UniTensor &inR, const bool &cacheL = false,
                     const bool &cacheR = false);
  UniTensor Contract(const std::vector<UniTensor> &TNs, const std::string &order = "",
                     const bool &optimal = true);

  /**
   * @brief DEPRECATED plural alias of Contract (finding UT-N3).
   * @details Identical to Contract(TNs, order, optimal). The C++ [[deprecated]]
   *          attribute is COMPILE-TIME only and does NOT reach Python; the next
   *          version's Python binding must emit a real DeprecationWarning for one
   *          minor release, then this is removed. Use Contract.
   */
  [[deprecated("Use Contract instead.")]]
  UniTensor Contracts(const std::vector<UniTensor> &TNs, const std::string &order = "",
                      const bool &optimal = true);

  /**
   * @brief One-shot network contraction in the ncon index convention.
   * @details POSITIVE ints label contracted bonds (each appearing EXACTLY TWICE);
   *          NEGATIVE ints label open output legs, ordered -1, -2, ... (finding
   *          UT-N4). PURE. Kept lowercase as the community-standard TN name
   *          (finding UT-N1). check_network validates the convention — default it
   *          to true in the next version. The working one-shot contractor
   *          (Network::Contract segfaults, finding UT-N6).
   * @param tensor_list_in the tensors. @param connect_list_in one int list per
   *        tensor. @param check_network validate the network. @param optimize
   *        search an optimal order. @param cont_order explicit order.
   *        @param out_labels output leg labels.
   * @return the contracted UniTensor (new); a fully-contracted network is scalar.
   */
  UniTensor ncon(const std::vector<UniTensor> &tensor_list_in,
                 const std::vector<std::vector<cytnx_int64>> &connect_list_in,
                 const bool check_network = true, const bool optimize = false,
                 std::vector<cytnx_int64> cont_order = {},
                 const std::vector<std::string> &out_labels = {});

}  // namespace cytnx
```

---

# Appendix — the `Network` class (cross-reference only)

The stateful **`Network`** builder (load a `.net` skeleton / string list, place
`UniTensor`s into named slots, then `Launch()` to contract) is a **separate
class**, fully audited in [`../per-class/network.md`](../per-class/network.md) and
**not re-audited here**. Two of its findings bear on this category and are quoted
for context (not re-derived):

- **network.md P1 (UT-N6):** `Network.Contract(...).Launch()` **segfaults**
  (`SIGSEGV`) in 1.1.0 — the advertised one-shot static factory is unusable;
  network.md's verdict is **remove** it. **`ncon` / free `Contract` / member
  `contract` are the working contraction paths.**
- **network.md's `ncon` verdict:** keep `ncon` (already correct and `snake_case`);
  consider defaulting `check_network=True` — which this category adopts as the
  UT-N4 recommendation.

When the `Network` class is redone under the superset method (a separate plan),
its members receive their own categorized verdicts; this appendix is purely a
pointer so the UniTensor contraction surface reads as a whole.
