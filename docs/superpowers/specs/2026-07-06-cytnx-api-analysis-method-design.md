# Cytnx API Analysis Method — Design Spec

**Status:** approved-in-pilot (validated on `UniTensor` categories 01–02).
**Author:** API-design working session, 2026-07-06.
**Supersedes** the naming/compatibility posture of the v1 audit
(`docs/api-audit/`), not its per-class data.

## 1. Purpose & goal

Define the **repeatable method** for analyzing the Cytnx 1.1.0 public API and
producing, per class, a **normative specification the next major version of
Cytnx is implemented from**. The audience is the **Cytnx maintainers**: the
output must be adoptable — precise signatures, explicit behavior contracts,
and a compatibility-aware migration path — not merely a critique.

`UniTensor` is the pilot class; once this method is locked, it is applied
verbatim to every other class. The unit of work is a **functional category**
of a class (e.g. "construction & init", "static generators"), so a large class
is analyzed and reviewed in navigable pieces.

## 2. Scope

- **In scope:** the Python-facing API of each class, its correspondence to the
  C++ surface, and the recommended next-version API. Grounding is the installed
  `cytnx==1.1.0` wheel (runtime) plus the pinned `cytnx_src/` 1.1.0 source.
- **Out of scope:** `tn_algo` (MPS/MPO/DMRG family); private/plumbing symbols
  (documented separately, never specced); implementing the new API.
- **Target version read:** signatures from the 1.1.0 wheel + `cytnx_src/`;
  behavior from the wheel (Python) and a source-built `libcytnx` (raw C++).

## 3. Background — the two prior analyses this method merges

| | v1 audit (`docs/api-audit/`) | sibling (`pcchen/cytnx-api-analysis`) |
|---|---|---|
| Structure | one flat doc/class, 6 sections | per-class dir; UniTensor split into 12 category files |
| C++↔Python | prose "Parity findings" | **first-class 5-status mapping table + policy** |
| Verification | **executable probes on the 1.1.0 wheel**; machine-checked coverage | manual; runtime on a **1.0.0** wheel (1.1.0 tree wouldn't build for them) |
| Naming rule | blanket "snake_case everything" (**wrong**, see §5) | snake_case members; flags PascalCase outliers |
| Extras | essential-API traceability, param-consistency sweep | element-dtype analysis, cpp-mapping policy |

This method is the **superset**: the sibling's categorized structure +
C++↔Python mapping + element-dtype dimension, carried on v1's probe-backed
runtime rigor and machine-checkable coverage, plus a new **binding-fidelity**
finding type and **raw-C++ verification** (§7) that neither prior analysis had.

## 4. Deliverable structure — the per-category document

Path: `docs/api-audit/<Class>/NN-<category>.md`. Each file has two top-level
parts, in order: **`# Analysis`** (evidence) and **`# R. Recommendation`**
(the self-contained normative spec). A reader implementing Cytnx needs only
`# R.`; the Analysis is the rationale.

### 4.1 `# Analysis`

- **A1 — Current API (1.1.0).** A table, one row per public API:
  `| API | Live signature (1.1.0) | Returns | Description & evidence |`.
  Signatures are verbatim from the wheel; each runtime claim cites its probe
  assertion. `[I]` marks in-place APIs.
- **A2 — C++ ↔ Python mapping.** A table classifying every member:
  `identical · renamed · signature-differs · C++-only · Python-only`.
  Columns `| C++ | Python | Status | Note |`. This is the parity contract.
- **A3 — Findings.** A table:
  `| ID | Finding | Type | What the binding does · evidence | Recommendation |`.
  IDs are class+category-scoped (`UT-C#` construction, `UT-G#` generators, …).
  **Type** ∈ {naming, ordering, redundancy, copy/view, capability gap,
  correctness, binding fidelity, documentation}. The evidence column names the
  probe assertion (behavioral), the `file:line` (source), and — for
  binding-fidelity rows — *what the binding layer actually does*.
- **A4 — Argument ordering (positional & keyword).** Analyzes both orderings
  (§5.3): a positional table + the naming/order/compat findings, ending with
  the two canonical orders that feed `R.0`.

Additional per-class dimensions live in their own files where warranted
(the sibling's model): `inventory.md` (full categorized member list),
`element-dtypes.md` (dtype set + promotion), `cpp-vs-python.md` (the full
mapping when too large for inline A2).

### 4.2 `# R. Recommendation` — normative spec

Self-contained; a Cytnx implementer codes to it directly.

- **R.0 — Normative conventions.** The rules every API in the category obeys
  (§5), stated concretely for this category.
- **R.1 — Recommended API.** A `class <Class>: …` block of exact recommended
  signatures, followed by a table
  `| API | Verdict | Behavior contract |` where Verdict ∈
  {keep, add, rename, remove} and the contract states copy/view, in-place, and
  defaults.
- **R.2 — Docstrings.** For every kept/added/renamed API, documented in **both**
  languages' idioms: **numpy-style** for the Python surface (`R.2a`) and
  **Doxygen** (`@brief`/`@param`/`@return`, matching Cytnx's existing C++ docs)
  for the C++ surface (`R.2b`). Both carry the same semantic content over each
  language's recommended signature. (A Python-only member such as `from_numpy`
  has no `R.2b` entry; a C++-only member has no `R.2a` entry.)

### 4.3 `actionable-fixes.md` — the fix-now summary (cross-class)

`docs/api-audit/actionable-fixes.md` is a cross-class deliverable that
aggregates every finding a maintainer can fix **now**, independent of the API
redesign — the immediately-shippable value of the audit. It is **derived**, not
hand-curated: built by filtering the per-class `A3` findings tables to the
actionable Types — `correctness`, `binding fidelity`, `capability gap`, and the
unbound/commented-out-C++ rows — and **excluding** the redesign-only Types
(`naming`, `redundancy`, `ordering`, and `copy/view`-documentation).

One row per fix, ranked by severity:
`| Severity | Bug / gap | Class · finding-id | Evidence | Recommended fix |`.

- **Critical** — crash / silent-wrong result / data corruption (e.g. `Save`/`Load`
  name over-read, `Network.Contract().Launch()` segfault, `//` true-division,
  `qgates.hadamard` non-unitary, `Symmetry.check_qnums` rejecting valid qnums).
- **High** — wrong-for-some-inputs, or the binding drops C++ functionality
  (e.g. `normal_` returns `None`, `get_elem` binds 4 of 11 dtypes, broken
  pickle, `astype`/`to` is-self short-circuit).
- **Medium** — capability gaps: useful C++ unbound or commented out
  (e.g. `Lanczos_ER`/`Gnd`/`Gnd_Ut`, missing `%`, no numpy bridge,
  `combineBond` singular).

Every row cites the probe assertion backing it (both-sides where a raw-C++
probe exists) and the `file:line`, so a maintainer goes straight to the fix. A
top **"Confirmed bugs"** section holds the Critical/High defects; a **"Gaps"**
section holds the Medium items.

## 5. Normative conventions

These are the rules `R.0` applies. Every Consistency-style finding cites one.

### 5.1 N-casing — the Cytnx naming convention (SciPostPhysCodeb.53)

- **Member** functions → **lowercase** snake_case.
- **Free** functions that **act on** objects → **Capitalized** (`linalg.Svd`,
  `Contract`).
- **Free** functions that **create** objects (generators) → **lowercase**
  (`cytnx.zeros`).
- **Types** → **Capitalized** (`UniTensor`, `Bond`).

Consequence (correction to v1): capitalized *members* (`Conj`, `Trace`, `Init`,
`Load`, `combineBonds`) are renamed to lowercase, but capitalized *free
functions acting on objects* (`Svd`, `Gesvd`, `Eigh`, `Qr`) are **kept**. This
**reverses** v1's `linalg.md` C1 / `summary.md` X1 "snake_case all 53 linalg
functions" recommendation.

### 5.2 N-underscore — in-place marker

A trailing `_` marks an **in-place** operation that mutates its receiver (or a
free function's primary argument) and **returns `self`**; the un-suffixed name
is **pure** (returns a new object, inputs unchanged — behavior rule B1). Every
operation meaningful both ways ships both forms under one base name; a
one-sided operation where both make sense is a finding. The trailing `_` is the
**only** in-place marker (reject `c`-prefixed raw bindings, `i`-prefixes).
Composes orthogonally with N-casing (`Inv`/`Inv_` free; `permute`/`permute_`
member).

### 5.3 Argument ordering — positional & keyword

Both orderings are canonicalized:

- **Positional:** `[primary operand], [operation parameters]`. Exactly one
  required primary operand (a source, a `shape`, or a range `start,end`);
  operation parameters follow in domain order. Prefer established
  community/numpy order where one exists (accept-and-document rather than
  "fix"); diverge only for stronger internal consistency, and document the
  divergence.
- **Keyword-only metadata block:** everything optional is **keyword-only**
  (`*` separator), declared in one canonical order. Making it keyword-only
  removes argument order from the public contract — so it cannot be inconsistent
  and no *positional* caller breaks when it is regularized.

The check is **four-axis**: naming · positional order · keyword order ·
compatibility (does a proposed reorder break positional callers?).

### 5.4 N-view / B1 — copy vs view

Every derivation/constructor is classified view-producing or copy-producing and
the fact is stated per-API in the behavior contract and docstring; verified with
the `returns_view` probe pattern.

### 5.5 One name per concept

Semantically equivalent parameters share one name across sibling APIs
(`labels` not `in_labels`; `qnum`/`qnums`; `dtype`/`device`).

## 6. Finding taxonomy

Each A3 finding has a **Type**. Most are standard (naming, ordering, redundancy,
copy/view, capability gap, correctness, documentation). One is distinctive:

**Binding fidelity** — flags where the **binding layer** (`*_conti.py` wrapper
or a pybind lambda) changes behavior versus the raw C++ method. This is the
tractable slice of C++/Python behavior comparison: most members are thin
pass-throughs (Python *is* the C++ observation), so only the non-pass-through
cases are flagged. Evidence = binding `file:line` (what the binding does) +
Python probe + (where built) raw-C++ probe. Examples from the pilot: a pybind
lambda that drops C++'s reference return (UT-G5); a `py::arg` rename (UT-G3);
Python-only sugar injected in the lambda (UT-G11).

The Type field is also the **extraction filter** for `actionable-fixes.md`
(§4.3): findings typed `correctness`, `binding fidelity`, or `capability gap`
(plus unbound/commented-out C++) are the fix-now set; `naming`/`redundancy`/
`ordering`/`copy-view`-documentation are redesign-only and excluded.

## 7. Verification model

No behavioral claim ships unverified. Two runtime layers:

1. **Python probe (required).** `docs/api-audit/probes/<Class>_<cat>.py` — one
   `report(claim, ok)` assertion per runtime claim, executed against the
   installed `cytnx==1.1.0` wheel; a clean exit-0 run is the evidence. Copy/view
   claims use `returns_view`. Machine coverage: every live public member of the
   class must appear in some category's `R.1` verdict table — enforced by a
   coverage validator (the v1 `tools/validate_doc.py` must be **adapted** to sum
   coverage across a class's category files rather than a single flat doc).

2. **Raw-C++ probe (for binding-fidelity findings).**
   `docs/api-audit/probes/cpp/<Class>_<cat>.cpp` — a C++ program linked against
   a **locally source-built `libcytnx`** (the 1.1.0 `cytnx_src/` tree builds
   cleanly with the local GCC; the PyPI wheel's `libcytnx.a` cannot be linked by
   a different GCC major version because of LTO-bytecode versioning). It calls
   the C++ methods directly, bypassing pybind, to verify the *raw-C++ side* of a
   binding-fidelity finding. Build recipe in `probes/cpp/README.md`.

A binding-fidelity finding is **runtime-verified on both sides** when its Python
probe and its C++ probe both pass; otherwise the C++ side is source-read (cited
`file:line`) and marked as such.

## 8. Compatibility-aware recommendations

The v1 posture was "clean-slate"; this method is **compatibility-aware** because
the audience ships to real users. Every `rename`/`remove` verdict carries a
migration note: keep the old name as a thin `DeprecationWarning` alias for one
minor release, then delete. `keyword-only` regularization is chosen precisely
because it avoids silent positional breakage. The change tables are the
migration guide.

## 9. Rollout plan

1. **Lock this method** (this spec + the 01/02 pilot as the worked example).
2. **Finish `UniTensor`** categories 03–12 (metadata, labels, structure,
   element/block, arithmetic, linalg, solvers, contraction, I/O, type/device),
   plus `inventory.md` and `element-dtypes.md`.
3. **Replicate per class** — `Tensor`, `Bond`, `Symmetry`, `Network`, `LinOp`,
   `Storage`, `Scalar`, enums, operations — each as a category set.
4. **Cross-class layer** — a `cpp-python-mapping.md` policy, a master
   recommendation index, and **`actionable-fixes.md`** (§4.3), all refreshed by
   aggregating the per-category tables. `actionable-fixes.md` can be seeded
   incrementally as each class lands (its rows already exist as typed `A3`
   findings) and consolidated here.

The v1 per-class docs remain as-is until a class is redone under this method;
when redone, the v1 doc is superseded (notably reversing the linalg casing).

## 10. Definition of done (per category)

- A1/A2/A3/A4 tables complete; every live public member of the category appears
  in A1 and in an `R.1` verdict row.
- Every behavioral claim cites a passing Python probe assertion (exit 0).
- Every binding-fidelity finding is either C++-probe-verified or explicitly
  marked source-read.
- `R.1` gives exact recommended signatures; `R.2` a docstring per kept/added/
  renamed API.
- Every rename/remove has a migration note.
- Every `A3` finding typed `correctness`, `binding fidelity`, or `capability
  gap` (or unbound/commented-out C++) appears as a row in `actionable-fixes.md`
  (§4.3) with a severity and a concrete fix — nothing fixable is left only
  inside a per-class doc.

## 11. Open decisions (to confirm at spec review)

1. **Docstring style** — *Resolved:* **Doxygen** (`@param`/`@return`) for the
   recommended **C++** API (matches Cytnx's existing C++ docs) + **numpy-style**
   for the recommended **Python** API. R.2 is bilingual (R.2a Python / R.2b C++).
2. **Flat vs categorized** — *Resolved:* the categorized `<Class>/NN-*.md` set
   becomes canonical; the flat `per-class/<Class>.md` is retired as each class is
   redone under this method (avoids two drifting sources of truth).
3. **C++ probes** — *Resolved:* **binding-fidelity findings only.** The Python
   wheel probe stays the required default for all behavioral claims (it already
   executes the compiled C++); the raw-C++ probe is pulled out only to verify
   the raw-C++ side of a binding-fidelity finding, keeping the source-build cost
   proportional to need. A category with no binding-fidelity finding needs no
   C++ probe.
4. **Pilot ordering calls** — *Resolved:* keep numpy order for range generators
   (`linspace(start, end, num)`, UT-G8) and shape-first for distributions
   (`normal(shape, mean, std)`, UT-G9) — both are "inconsistent-but-conventional"
   (community/internal consistency), documented rather than "fixed".
