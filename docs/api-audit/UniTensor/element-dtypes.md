# UniTensor — element dtypes (cross-cutting dimension)

> **Cross-cutting dimension doc**, not a six-section category file (contrast the
> categories [`01-construction-init.md`](01-construction-init.md) …
> [`12-type-device-conversion.md`](12-type-device-conversion.md)). It audits one
> orthogonal axis of the `UniTensor` design — the **element dtype** — across all
> operations, rather than one member set. It carries **no** `R.1` verdict table
> and is **not** counted by the coverage validator: `validate_doc.py`'s directory
> glob is `[0-9]*.md`, so this non-numbered file is correctly excluded and adds no
> member coverage or regression.
>
> All runtime claims are verified against `cytnx==1.1.0` via
> [`../probes/UniTensor_dtypes.py`](../probes/UniTensor_dtypes.py) (all `[PASS]`,
> exit 0). The dtype **set and codes** are cross-referenced from
> [`../per-class/enums.md`](../per-class/enums.md) (`Type`); the per-operation
> constraints are cross-referenced from the category docs
> [`06-element-block-access.md`](06-element-block-access.md) (UT-E1) and
> [`08-linalg-operations.md`](08-linalg-operations.md).

**Scope.** A `UniTensor`'s elements all share **one** dtype — one of the fixed
`Type` codes — reported by `dtype()` (int code) / `dtype_str()` (name). This doc
establishes: (1) the **11 constructible element dtypes**; (2) the **promotion
rule** for mixed-dtype arithmetic; (3) the **per-operation dtype constraints**
(which operations accept which dtypes); and (4) the **recommendation** to treat
the 4 float/complex types as the numeric element set and integer/bool tensors as
storage-only.

---

## 1. The 11 element dtypes

`Type` (see [`enums.md`](../per-class/enums.md)) declares **12** codes; code `0`
is `Void`, the *no-dtype* sentinel (an uninitialised tensor), **not** a
constructible element dtype. The remaining **11** are the element dtypes a
`UniTensor` can actually hold. Each row is verified constructible — a
`UniTensor.ones([2,2], dtype=Type.<name>)` builds and reports the matching
`dtype()` code (probe: *"a UniTensor is constructible with dtype Type.<name>, and
dtype() == <code> …"* — one per row, all `[PASS]`).

| Element dtype | `Type.*` code | Category | C++ scalar (`Type.hpp`) |
|---|---|---|---|
| `ComplexDouble` | 1 | complex | `complex128` (`std::complex<double>`) |
| `ComplexFloat` | 2 | complex | `complex64` (`std::complex<float>`) |
| `Double` | 3 | real float | `float64` (`double`) |
| `Float` | 4 | real float | `float32` (`float`) |
| `Int64` | 5 | signed int | `int64_t` |
| `Uint64` | 6 | unsigned int | `uint64_t` |
| `Int32` | 7 | signed int | `int32_t` |
| `Uint32` | 8 | unsigned int | `uint32_t` |
| `Int16` | 9 | signed int | `int16_t` |
| `Uint16` | 10 | unsigned int | `uint16_t` |
| `Bool` | 11 | bool | `bool` |

Two structural facts about the code order (`Type.hpp` `Type_list` variant-index
order) drive everything below:

- The codes are ordered by **decreasing generality**: complex (1–2) `<` real
  float (3–4) `<` signed/unsigned int, widest→narrowest (5–10) `<` bool (11).
- Within a width, the **wider** member has the **smaller** code (`Double` 3 `<`
  `Float` 4; `Int64` 5 `<` `Int32` 7 `<` `Int16` 9), and the **complex** member
  has a smaller code than the same-width real (`ComplexFloat` 2 `<` `Float` 4).

Probe: *"exactly 11 element dtypes are constructible on a UniTensor (Void, code
0, is the no-dtype sentinel, not an element dtype)"* `[PASS]`.

## 2. The promotion rule

Mixed-dtype element-wise arithmetic (`+`, `-`, `*`, `/` between two UniTensors of
different dtype) produces a result whose dtype is **the operand's dtype with the
smaller `Type` code** — i.e. *widen to the more general type*. This was verified
**exhaustively**: for all **55** unordered dtype pairs, `(A + B).dtype() ==
min(code(A), code(B))` (probe: *"promotion rule is EXACTLY 'result = the dtype
with the smaller Type code' for all 55 mixed-dtype pairs …"* `[PASS]`). Because
the code order encodes generality (§1), the min-code rule reads as three plain
sub-rules:

- **Complex dominates real.** `Double + ComplexDouble → ComplexDouble`; `Float +
  ComplexFloat → ComplexFloat`. Probe: *"promotion: Double + ComplexDouble ->
  ComplexDouble (complex dominates real)"* + *"… Float + ComplexFloat ->
  ComplexFloat …"* `[PASS]`.
- **Double dominates float** (wider wins, within a kind). `Float + Double →
  Double`; `ComplexFloat + ComplexDouble → ComplexDouble`. Probe: *"promotion:
  Float + Double -> Double (double dominates float)"* + *"… ComplexFloat +
  ComplexDouble -> ComplexDouble …"* `[PASS]`.
- **Float/complex dominates int, and any numeric dominates bool.** `Int32 +
  Double → Double`; `Bool + Int32 → Int32`; `Int16 + Int32 → Int32` (wider signed
  int). Probe: *"promotion: Int32 + Double -> Double …"* + *"… Bool + Int32 ->
  Int32 …"* + *"… Int16 + Int32 -> Int32 …"* `[PASS]`.

The rule is **symmetric** (`A+B` and `B+A` give the same dtype — probe `[PASS]`)
and a **no-op** for equal dtypes (`Double + Double → Double` — probe `[PASS]`).

**Caveat (not a widening guarantee for range).** The rule widens by the *code
order*, not by numeric range: `Float + Int64 → Float` (code 4 `<` 5), so a 64-bit
integer's low bits are lost to a 32-bit float. Promotion preserves the *more
general kind*, not the *value*; it is Cytnx's `Type_class::type_promote`, not
NumPy's range-aware `result_type`.

## 3. Per-operation dtype constraints

Not every operation accepts all 11 dtypes. The table below records, per
operation family, which dtypes are **operable** and what happens otherwise. Each
behavioral cell quotes a probe assertion (from `UniTensor_dtypes.py` unless the
cross-referenced category is named).

| Operation | Accepts | On an int/bool tensor | Evidence |
|---|---|---|---|
| **Construction** (`ones`/`zeros`/`arange`/…, `dtype=`) | all **11** | builds normally | *"a UniTensor is constructible with dtype Type.<name> …"* (11×, `[PASS]`) |
| **Element write** — `set_elem`, `item`, `at` proxy | all **11** | works | cat-06 UT-E1 (*"set_elem writes a value on dtype … (all 11 covered)"*) |
| **Element read** — `get_elem` | **4** float/complex only | **raises** `RuntimeError` (*"try to get element from a void Storage"*) | *"get_elem RAISES on integer/bool dtype <name> …"* (7×, `[PASS]`); cat-06 UT-E1 |
| **Element-wise arithmetic** (`+ - * /`, mixed dtype) | all **11**; result per §2 promotion | works; result stays int/bool if both operands are | *"promotion rule is EXACTLY … for all 55 mixed-dtype pairs"* `[PASS]` |
| **Decompositions** — `linalg.Svd`, `Eigh`, `Eig`, `Qr`, … | float/complex; **int input up-casts** | **does NOT raise** — up-casts to `Double`, returns float results | *"linalg.Svd on an integer/bool (<name>) UniTensor does NOT raise: it UP-CASTS to Double and returns Double singular values"* (7×) + *"linalg.Eigh on a <name> … returns Double eigenvalues"* `[PASS]`; cat-08 |

Two findings deserve emphasis:

- **`get_elem` is the one accessor with a narrow dtype surface** — it reads only
  the 4 float/complex dtypes and **raises** on the 7 integer/bool ones, while its
  siblings `set_elem`/`item`/`at` cover all 11. This is a **binding defect**
  (cat-06 UT-E1: the pybind lambda instantiates only 4 branches; the C++
  `get_elem<T>` template is generic), not an intrinsic dtype constraint — the fix
  is to bind all 11. Until then, integer/bool element **reads** must go through
  `at()` / `item()`, never `get_elem`.

- **Decompositions do not reject integer tensors — they silently up-cast.** The
  discovered runtime truth (the task's open question) is that
  `linalg.Svd`/`Eigh`/… on an `Int*`/`Uint*`/`Bool` UniTensor **do not raise**:
  they promote the input to `Double` internally and return `Double`
  singular-values / eigenvalues (a real-Double `S`/`e` even for a `ComplexDouble`
  input). So the float/complex requirement is enforced by an **implicit up-cast**,
  not an error. This is convenient but silent — a caller who feeds an int tensor
  to a decomposition gets a float result with no diagnostic, and the returned
  factors no longer share the input's dtype.

## 4. Recommendation

**Keep the 4 float/complex types — `ComplexDouble`, `ComplexFloat`, `Double`,
`Float` — as the recommended UniTensor *element* set for numeric work; treat the
7 integer/bool types (`Int64`/`Uint64`/`Int32`/`Uint32`/`Int16`/`Uint16`/`Bool`)
as storage-only element dtypes.**

Rationale, each grounded in a probe finding:

1. **The numeric operations are float/complex-shaped.** Decompositions
   (`Svd`/`Eigh`/…) up-cast integer input to `Double` anyway (§3), so an integer
   element tensor never survives a linalg step as an integer — it is silently
   promoted. Element **reads** via `get_elem` fail outright on int/bool (§3). The
   float/complex set is the only one on which the full operation surface —
   read, arithmetic, decompose — works without a raise or a silent up-cast.

2. **`Double` / `ComplexDouble` are the safe defaults; `Float` / `ComplexFloat`
   are the memory-halved variants.** Under the promotion rule (§2) `Double`
   dominates `Float` and complex dominates real, so mixing a single-precision
   tensor into a double-precision computation widens correctly and mixing real
   into complex widens correctly — the four compose cleanly among themselves.

3. **Integer/bool tensors are legitimate *storage* — index maps, masks, quantum
   numbers, occupation counts — but not *operands*.** They construct (all 11),
   write (`set_elem`/`item`/`at`), and do integer arithmetic (promotion stays
   within int/bool), so keep them constructible. But **do not** route them
   through the numeric surface: element reads raise (`get_elem`), and
   decompositions silently promote them. Callers should `astype(Type.Double)`
   (or `ComplexDouble`) **explicitly** before any linalg step, rather than
   relying on the silent internal up-cast.

**Concrete asks for the next version:**

- **Fix the `get_elem` binding** to cover all 11 dtypes (cat-06 UT-E1), so the
  element-read surface matches `set_elem`/`item`/`at`. This removes the one hard
  dtype cliff among the accessors.
- **Make the decomposition up-cast explicit, not silent.** Either require a
  float/complex input (raise a clear error on int/bool, directing the caller to
  `astype`), or keep the up-cast but emit a warning — so the dtype change is not
  invisible. Document that `Svd`/`Eigh`/… always return `Double`-precision
  factors for any integer input.
- **Document the promotion rule as `type_promote` (code-min), not NumPy
  `result_type`** — it widens by *kind*, not by numeric *range* (§2 caveat), so
  `Float + Int64 → Float`. Callers who need range-safe accumulation must up-cast
  deliberately.

---

## Cross-references

- Dtype set + codes: [`per-class/enums.md`](../per-class/enums.md) (`Type`, 11
  dtype codes + `Void`).
- `get_elem`'s 4-vs-11-dtype binding defect: [`06-element-block-access.md`](06-element-block-access.md)
  finding **UT-E1**.
- Decompositions' float/complex requirement: [`08-linalg-operations.md`](08-linalg-operations.md).
- Probe (all claims here): [`../probes/UniTensor_dtypes.py`](../probes/UniTensor_dtypes.py).
