# 00 — Audit methodology & API conventions

This document is the fixed reference for every other document under
`docs/api-audit/`. It defines (1) the per-class document template, (2) the
naming conventions (`N1`…) and (3) behavior conventions (`B1`…) that the
recommended API must obey, (4) how parity between the C++ and Python surfaces
is judged, and (5) how the essential API subset (deliverable #6 of the spec)
is derived. Every "Consistency findings" entry in a per-class document cites
one of the convention IDs defined here — "inconsistent" always means
"violates a specific `N`/`B` rule below," never a stylistic opinion.

Scope, in-scope units, and the overall project spec live in
`docs/superpowers/specs/2026-07-03-cytnx-api-audit-redesign-design.md`. This
document operationalizes that spec's §4 (per-class template) and §5
(methodology).

## 1. Audit template

`tools/validate_doc.py::REQUIRED_SECTIONS` is the machine-checked authority
for the per-class template. Every `docs/api-audit/per-class/<Unit>.md` must
contain exactly these six `##`-level sections, in this order:

1. **`## Inventory`** — every public member of the unit, one row per member,
   showing the C++ signature and the Python signature side by side. Sourced
   from `cytnx_src/include/*.hpp` (C++ declaration), `cytnx_src/pybind/*_py.cpp`
   (the pybind11 binding, which is the ground truth for the *actual* Python
   call signature), and `cytnx_src/cytnx/*_conti.py` (the Python-side
   augmentation layer, when the member is defined or overridden there instead
   of via pybind). `tools/member_inventory.py <Unit>` dumps the live
   pybind-visible signatures as a cross-check against what was read statically.
2. **`## Parity findings`** — every divergence between the C++ and Python
   surfaces for this unit: signature differences (parameter names, defaults,
   overload sets, return types) and *runtime object-behavior* differences
   (copy-vs-view, in-place vs. returns-new, mutability, dtype/device
   promotion, operator overloads, exception behavior). Every behavioral claim
   in this section must be backed by a passing assertion in
   `docs/api-audit/probes/<Unit>.py` — see §4 below.
3. **`## Consistency findings`** — internal incoherence within this unit's own
   surface, measured strictly against the `N`/`B` conventions in §2/§3 of this
   document. Each finding names the offending member(s) and cites the
   violated convention ID, e.g. "`Conj`/`Dagger`/`Inv` violate N1 (capitalized
   verb methods instead of `snake_case`)."
4. **`## Recommendation`** — one row per public member (every member found by
   `tools/validate_doc.py`'s `public_members()`, i.e. every non-underscore
   attribute of the live object), tagged **keep / add / rename / remove**,
   with a one-line rationale that references the specific `N`/`B` convention
   or parity finding that motivates the verdict.
5. **`## Docstrings`** — a numpy-style docstring for every member tagged
   `keep`, `add`, or `rename` in the Recommendation table (summary,
   Parameters, Returns, and an explicit Notes line for copy/view and
   in-place semantics where relevant). Members tagged `remove` do not need a
   docstring. Each docstring block must mention the member's (recommended)
   name in backticks or as a heading so `validate_doc.py` can match it back
   to its Recommendation row.
6. **`## Change table`** — the clean-slate migration map: `current (C++ name /
   Python name) → recommended name`, covering every rename and every remove,
   so the table alone is a complete migration guide for that unit.

`tools/validate_doc.py <Unit> <path>` enforces sections 1, 4, and 5
programmatically: it fails if any required section header is missing, if any
live public member is absent from the Recommendation table, or if any
`keep`/`add`/`rename` member lacks a matching Docstrings block.

## 2. Naming conventions (N-series)

- **N1 — Python public methods are `snake_case`.** The recommended API
  renames every capitalized-verb method to its lowercase, underscore-separated
  form: `Conj`→`conj`, `Dagger`→`dagger`, `Inv`→`inv`, `Trace`→`trace`,
  `Norm`→`norm`, `Load`→`load`, `Save`→`save`, `Init`→`init`, and so on.
  Constants/enum members (e.g. `Type.Double`, `Device.cuda`) and class names
  themselves (`UniTensor`, `Bond`) are exempt — N1 governs *callable* public
  members (methods and free functions), not type names or enum values.
- **N2 — in-place variants use a trailing `_`, and every in-place method has a
  pure (returns-new) counterpart with the same base name.** E.g. the
  recommended pair is `conj()` / `conj_()`, `pow()` / `pow_()`,
  `contiguous()` / `contiguous_()`. A method that only exists in one form
  (in-place-only or pure-only) where the operation is meaningful in both
  forms is a consistency finding against N2 (see also B1). Idiosyncratic
  "double-underscore-ish" prefixed variants observed in the current API
  (e.g. `cConj_`, `cDagger_`, `cInv_`, `cPow_`, `cTrace_`, `cTranspose_` on
  `UniTensor`, apparently meaning "conjugate-only" complex-tensor variants)
  are not a recognized convention; each such method must be individually
  resolved to either fold into the base `_`-suffixed pair (if it is really
  just the in-place form) or be renamed to a self-describing `snake_case`
  name if it has genuinely distinct semantics.
- **N3 — C++ and Python public names are identical modulo the N1
  `snake_case` rule.** The C++ member name (`Conj`) and its Python-callable
  name (`conj` after N1 is applied) must be the same identifier apart from
  case/underscore convention; no separate "translation" vocabulary between
  the two languages. A C++/Python name pair that diverges by more than the
  N1 casing rule (e.g. different words, reordered compound terms) is a
  parity finding, not merely a consistency finding, because it breaks the
  ability to look up one language's API from the other's.
- **N4 — argument order is identical across C++ and Python for the same
  member**, and, within a unit, semantically equivalent parameters use the
  same name and position across overloaded/sibling methods (e.g. a `dim`/axis
  argument is not called `dim` in one method and `axis` in another within the
  same class). Divergent argument names/order for the same operation is a
  consistency finding against N4.
- **N5 — boolean/flag parameters and predicate methods read as a
  yes/no question or a clearly named flag** (`is_contiguous()`,
  `requires_grad`), not an abbreviated or ambiguous short name. Predicate
  methods (returning `bool`) are prefixed `is_`/`has_` unless already a
  self-evident adjective.

## 3. Behavior conventions (B-series)

- **B1 — a method returns a new object (copy) unless its name carries the
  `_` in-place suffix (N2).** `conj()` must not mutate its receiver;
  `conj_()` must mutate the receiver and (per current empirical behavior,
  see `docs/api-audit/probes/`) may or may not return the identical Python
  wrapper object — callers must not rely on identity, only on the receiver's
  observable state having changed. This is verified per-member with the
  `returns_view`/mutation probe pattern in §4.
- **B2 — indexing and slice assignment on a handle is a view and mutates the
  aliased data in place; this must be documented explicitly per class.**
  E.g. `t[0,0] = x` on a `Tensor` obtained via `permute()` is observed to
  mutate the original source tensor's storage (a view), while a `Tensor`
  obtained via `clone()` is observed to be independent (a copy) — confirmed
  empirically in `docs/api-audit/probes/__smoke__.py` using
  `tools/probe_helper.py::returns_view`. Every derivation method
  (`permute`, `reshape`, slicing, `get_block`, etc.) must be classified as
  view-producing or copy-producing in its unit's Parity/Consistency findings,
  and the Docstrings entry for that member must state which it is.
- **B3 — dtype/device promotion rules are explicit and identical across
  C++/Python.** Binary operations between operands of different dtype
  (e.g. `Double` + `ComplexDouble`) or device (`cpu` vs `cuda`) must follow
  one documented promotion rule (widen to the more general dtype; refuse or
  explicitly convert across devices), and that rule must produce the same
  result whether invoked from C++ or Python. A promotion behavior observed
  only on one language side, or an inconsistent result dtype/device between
  the two call paths for equivalent inputs, is a parity finding.
- **B4 — errors are raised as exceptions on both sides, not silently
  ignored or returned as sentinel/error-code values**, and the exception
  type/message is stable enough to be tested (an empty/invalid index or a
  shape mismatch raises rather than segfaulting or silently truncating).
  A C++ path that aborts/asserts where the Python binding instead raises a
  catchable exception (or vice versa) is a parity finding against B4.
- **B5 — operator overloads (`+ - * / == etc.`) are equivalent to their
  named-method counterparts** (`__add__` behaves like `Add`/`add`) with
  identical broadcasting, promotion (B3), and copy-vs-view (B1/B2) semantics;
  an operator that behaves differently from its named method for the same
  inputs is a consistency finding against B5.

## 4. How parity is judged

Parity between the C++ and Python surfaces of a unit is judged in two parts,
per the project spec §5.1:

1. **Signature comparison (static).** For every public member, read all
   three layers that can define or alter its Python-visible signature:
   the C++ declaration in `cytnx_src/include/*.hpp`, the pybind11 binding in
   `cytnx_src/pybind/*_py.cpp` (this is what actually determines the callable
   Python signature — overload sets, default arguments, and argument names
   as exposed to Python may differ from the raw C++ signature), and any
   Python-side augmentation or override in `cytnx_src/cytnx/*_conti.py`. The
   Inventory table records the C++ signature and the *effective* Python
   signature side by side; `tools/member_inventory.py <Unit>` prints the
   live pybind-visible signature straight from the installed 1.1.0 wheel, so
   the statically-read signature can be cross-checked against what the
   installed extension module actually exposes at runtime.
2. **Behavioral probe (dynamic).** A claim about *runtime object behavior*
   (copy-vs-view, in-place mutation, dtype/device promotion, operator
   semantics, exception behavior — i.e. anything in the B-series) is only
   admissible as a parity finding once it is backed by an executed,
   currently-passing assertion in `docs/api-audit/probes/<Unit>.py`. Probes
   are built from `tools/probe_helper.py`, which exposes:
   - `report(claim: str, ok: bool)` — prints a `[PASS]`/`[FAIL]` line and
     raises `AssertionError` on failure, so a probe script that runs to
     completion with exit code 0 is itself the evidence for every claim it
     printed `[PASS]` for.
   - `returns_view(make, derive, mutate, read) -> bool` — the view-vs-copy
     detector. `make()` builds a fresh source object, `derive(source)`
     applies the method under test to produce a derived handle, `mutate(handle)`
     mutates the derived handle in place, and `read(source)` takes a
     comparable snapshot of the source; the helper returns `True` iff the
     mutation performed through the derived handle is observable on the
     *original* source (a view) and `False` if the source is left unchanged
     (a copy). Every B1/B2 claim in a per-class document is produced by
     calling `returns_view` with that member's `derive` and asserting the
     expected boolean via `report`.
   No behavioral claim ships unverified: "the probe assertion/output is the
   evidence" (spec §5.1). A per-class document's Parity findings section may
   only state a behavioral divergence that its corresponding
   `docs/api-audit/probes/<Unit>.py` demonstrates with a passing `report(...)`
   call, executed against the venv-installed Cytnx 1.1.0
   (`./.venv/bin/python`, see `tools/env.sh`).

A member has full parity when (a) its C++ and Python signatures agree modulo
the N1 casing rule and N3/N4 argument-identity rules, and (b) every
behavioral claim about it, once probed, matches across both call paths. Any
divergence found by either check is recorded as a Parity finding, not a
Consistency finding — parity findings are about C++-vs-Python agreement;
consistency findings (§2/§3 above) are about a unit's *internal* coherence
against the N/B conventions.

## 5. How the essential API set is derived

Per the project spec §5.3, `essential-api.md` is derived — not curated by
opinion — by decomposing the four reference tensor-network algorithms into
their primitive computational steps and mapping each step to concrete API
calls:

1. **Decompose each reference algorithm into primitive steps.** TRG, HOTRG,
   CTMRG, and MERA are each broken down into the common tensor-network
   coarse-graining pattern: (a) reshape/group/split bonds, (b) contract
   tensors (pairwise contraction or full network contraction via `ncon`/
   `Network`), (c) decompose via `Svd`/`Eigh` and truncate to a bond
   dimension, (d) build/apply an isometry or projector from the decomposition,
   and (e) renormalize the local tensor(s) and iterate. HOTRG additionally
   needs a higher-order contraction/truncation step; CTMRG needs corner/edge
   tensor update and absorption steps; MERA needs disentangler and isometry
   optimization steps.
2. **Map each step to the concrete API calls it requires.** For each
   primitive step in each algorithm, list the specific `Tensor`/`UniTensor`/
   `Bond`/`linalg`/`Network`/`LinOp` members that implement it (e.g. "contract
   tensors" maps to `UniTensor.contract`/`Network`/`ncon`; "decompose and
   truncate" maps to `linalg.Svd`/`linalg.Eigh` plus bond-truncation
   arguments; "reshape/group bonds" maps to `UniTensor.combineBonds`/
   `reshape`/`permute`). `cytnx_src/docs/source/example/HOTRG.rst` is used as
   a working, in-repo reference implementation to ground the HOTRG mapping in
   real call sequences rather than a paraphrase; TRG/CTMRG/MERA are mapped
   from their standard published algorithm definitions since no in-repo
   example exists for them (this lower-confidence grounding is flagged where
   it affects a CTMRG/MERA entry).
3. **Take the union across all four algorithms.** The essential set is the
   union of every API call identified in step 2 across TRG, HOTRG, CTMRG,
   and MERA — a member used by only one algorithm is still essential; the
   union, not the intersection, defines "essential," because the goal is
   "sufficient to build any of the four reference algorithms," not "common
   to all of them."
4. **Trace every entry back to at least one algorithm step.** Each row in
   `essential-api.md` cites the algorithm(s) and the specific primitive
   step(s) (from step 1's decomposition) that require it, so the essential
   set is auditable rather than asserted — this traceability is the
   definition-of-done check in the project spec §7 ("The essential-API set
   traces each entry to at least one TRG/HOTRG/CTMRG/MERA step").

`tn_algo` (the MPS/MPO/DMRG family) is explicitly excluded from this
derivation per the spec's out-of-scope list: it is a higher-level algorithm
family built *on top of* the essential set, not one of the four reference
algorithms used to derive it.
