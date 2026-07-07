# Tensor — element dtypes (cross-cutting dimension)

> **Cross-cutting dimension doc**, not a six-section category file (contrast the
> categories [`01-construction-init.md`](01-construction-init.md) …
> [`08-io.md`](08-io.md)). It audits one orthogonal axis of the `Tensor` design —
> the **element dtype** — across all operations, rather than one member set. It
> carries **no** `R.1` verdict table and is **not** counted by the coverage
> validator: `validate_doc.py`'s directory glob is `[0-9]*.md`, so this
> non-numbered file is correctly excluded and adds no member coverage or
> regression (the partition stays `PASS: Tensor — 67 members covered across 8
> files`).
>
> All runtime claims are verified against `cytnx==1.1.0` via
> [`../probes/Tensor_dtypes.py`](../probes/Tensor_dtypes.py) (49 assertions, all
> `[PASS]`, exit 0). The dtype **set and codes** are cross-referenced from
> [`../per-class/enums.md`](../per-class/enums.md) (`Type`); the per-operation
> constraints are cross-referenced from the category docs
> [`06-linalg-reductions.md`](06-linalg-reductions.md) (T-X2/T-X3) and
> [`03-element-storage-access.md`](03-element-storage-access.md) (T-E6). Where the
> rule is shared with the block-structured case, this doc cross-references
> [UniTensor element-dtypes](../UniTensor/element-dtypes.md) and flags where the
> **dense** `Tensor` **diverges**.

**Scope.** A `Tensor` is the **dense** array type: all its elements share **one**
dtype — one of the fixed `Type` codes — reported by `dtype()` (int code) /
`dtype_str()` (name). There is no block/symmetry-structured dtype complexity (that
is UniTensor's world); every `Tensor` is a single contiguous typed buffer. This
doc establishes: (1) the **11 constructible element dtypes**; (2) the **promotion
rule** for mixed-dtype arithmetic — and the **3 integer exceptions** where it
diverges from the pure min-code rule *and* from UniTensor; (3) the **per-operation
dtype constraints** (int-linalg up-cast, the `astype` complex→real cliff); and (4)
the **recommendation** to treat the 4 float/complex types as the numeric element
set and integer/bool tensors as storage-only.

---

## 1. The 11 element dtypes

`Type` (see [`enums.md`](../per-class/enums.md)) declares **12** codes; code `0`
is `Void`, the *no-dtype* sentinel (an uninitialised tensor), **not** a
constructible element dtype. The remaining **11** are the element dtypes a
`Tensor` can actually hold. Each row is verified constructible — a
`cytnx.zeros([2,3], dtype=Type.<name>)` builds and reports the matching `dtype()`
code (probe: *"a Tensor is constructible with dtype Type.<name> … dtype() ==
<code> …"* — one per row, all `[PASS]`).

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

The dtype set is **identical** to UniTensor's — both wrap the same
`Type_class::Type` enum. Two structural facts about the code order (`Type.hpp`
`Type_list` variant-index order) drive the promotion rule below:

- The codes are ordered by **decreasing generality**: complex (1–2) `<` real
  float (3–4) `<` signed/unsigned int, widest→narrowest (5–10) `<` bool (11).
- Within a width, the **wider** member has the **smaller** code (`Double` 3 `<`
  `Float` 4; `Int64` 5 `<` `Int32` 7 `<` `Int16` 9), and the **complex** member
  has a smaller code than the same-width real (`ComplexFloat` 2 `<` `Float` 4).
  The **signed** member always has a smaller code than the same-width unsigned
  (`Int64` 5 `<` `Uint64` 6).

Probe: *"exactly 11 element dtypes are constructible on a Tensor (Void, code 0, is
the no-dtype sentinel, not an element dtype)"* `[PASS]`.

> **Factory asymmetry (cross-class note).** `Tensor` has **no** static `ones` /
> `zeros` classmethod — the fillers are the **module-level** factories
> `cytnx.ones(shape, dtype=…)` / `cytnx.zeros(…)` (cat 01), whereas `UniTensor`
> exposes `UniTensor.ones(...)` as a classmethod. The probe therefore builds via
> `cytnx.zeros` / `cytnx.ones`; this is a naming inconsistency, not a dtype one.

## 2. The promotion rule

Mixed-dtype element-wise arithmetic (`+`, `-`, `*`, `/` between two Tensors of
different dtype) is governed by Cytnx's `Type_class::type_promote`. The **base
rule** is: the result dtype is **the operand's dtype with the smaller `Type`
code** — *widen to the more general type*. This holds for **52 of the 55**
unordered dtype pairs (probe: *"the min-Type-code promotion rule holds for
EXACTLY 52 of the 55 mixed-dtype pairs …"* `[PASS]`). Because the code order
encodes generality (§1), the min-code base rule reads as three plain sub-rules:

- **Complex dominates real.** `Double + ComplexDouble → ComplexDouble`; `Float +
  ComplexFloat → ComplexFloat`. Probe `[PASS]` (2 named claims).
- **Double dominates float** (wider wins, within a kind). `Float + Double →
  Double`; `ComplexFloat + ComplexDouble → ComplexDouble`. Probe `[PASS]`.
- **Float/complex dominates int, and any numeric dominates bool.** `Int64 +
  Double → Double`; `Int32 + Double → Double`; `Bool + Int32 → Int32`; `Int16 +
  Int32 → Int32` (wider signed int). Probe `[PASS]` (4 claims).

The rule is **symmetric** (`A+B` and `B+A` give the same dtype — probe `[PASS]`)
and a **no-op** for equal dtypes (`Double + Double → Double` — probe `[PASS]`).

### 2a. The 3 integer exceptions (the min-code rule is NOT exact)

The **3** pairs that break the min-code base rule are all **mixed
signed/unsigned integer** pairs where an **unsigned** type is mixed with a
**strictly narrower signed** type. The result is the **signed** integer at the
**wider** width — which can be a **third type that is neither operand**:

| Operands | min-code would give | Tensor actually gives | Note |
|---|---|---|---|
| `Uint64` (6) + `Int32` (7) | `Uint64` | **`Int64`** (5) | result is **neither operand** |
| `Uint64` (6) + `Int16` (9) | `Uint64` | **`Int64`** (5) | result is **neither operand** |
| `Uint32` (8) + `Int16` (9) | `Uint32` | **`Int32`** (7) | result is **neither operand** |

Probe: *"promotion EXCEPTION: Uint64 + Int32 -> Int64 (NOT Uint64 per min-code) …
the result dtype is NEITHER operand"* + two more `[PASS]`. The general integer
rule the runtime actually follows is **"result width = max(widths); result is
unsigned iff both operands are unsigned, else signed"** (C-like promotion). It
coincides with min-code except in exactly these three wider-unsigned +
narrower-signed cases. The **non-exception** neighbours confirm the boundary:
same-width `Uint64 + Int64 → Int64` and `Uint32 + Int32 → Int32` (min-code), and
`Uint64 + Uint32 → Uint64` (min-code) — all `[PASS]`.

> **Cross-class divergence (Tensor vs. UniTensor).** UniTensor's arithmetic
> follows the pure min-code rule on **all 55** pairs (its probe asserts
> `Uint64 + Int32 → Uint64`), but **dense Tensor diverges** on exactly these 3
> pairs (`Uint64 + Int32 → Int64`). Probe: *"Tensor DIVERGES from UniTensor on
> the exception pairs: Tensor(Uint64+Int32)=Int64 but
> UniTensor(Uint64+Int32)=Uint64 …"* `[PASS]`. Two array types of the same
> library, same conceptual "type promotion", give **different result dtypes** for
> the same operands — a real cross-class inconsistency to reconcile in the
> redesign. Contrast [UniTensor element-dtypes §2](../UniTensor/element-dtypes.md#2-the-promotion-rule).

**Caveat (not a widening guarantee for range).** Even where min-code holds, the
rule widens by the *code order*, not by numeric range: `Float + Int64 → Float`
(code 4 `<` 5 — probe `[PASS]`), so a 64-bit integer's low bits are lost to a
32-bit float. Promotion preserves the *more general kind*, not the *value*; it is
Cytnx's `type_promote`, not NumPy's range-aware `result_type`.

## 3. Per-operation dtype constraints

Not every operation accepts all 11 dtypes, and two operations **change** the
dtype (up-cast) or **reject** a dtype (down-cast). The table records, per
operation family, which dtypes are operable and what happens otherwise. Each
behavioral cell quotes a probe assertion (from `Tensor_dtypes.py` unless the
cross-referenced category is named).

| Operation | Accepts | On an int/bool tensor | Evidence |
|---|---|---|---|
| **Construction** (`zeros`/`ones`/`arange`/…, `dtype=`) | all **11** | builds normally | *"a Tensor is constructible with dtype Type.<name> …"* (11×, `[PASS]`) |
| **Element write / read** — `item`, `fill`, `t[...]`, `t[...] = v` | all **11** | works | cat-03 T-E6 (`item` all-11 lambda), T-E7 (`fill` all-11); Tensor's `item`/`fill`/`__setitem__` bind all 11 dtypes (**no** `get_elem`-style narrowing — contrast UniTensor UT-E1) |
| **Element-wise arithmetic** (`+ - * /`, mixed dtype) | all **11**; result per §2 | works; result stays int/bool if both operands are (modulo the §2a exceptions) | *"the min-Type-code promotion rule holds for EXACTLY 52 of the 55 …"* + 3 exception claims `[PASS]` |
| **Decompositions / matrix inverse / norm** — `linalg.Svd`, `Eigh`, `InvM`, `Norm` | float/complex; **int input up-casts** | **does NOT raise** — up-casts to `Double`, returns `Double` results | *"linalg.Svd on an integer/bool (<name>) Tensor does NOT raise: it UP-CASTS to Double …"* (7×) + *"linalg.InvM on an integer (<name>) … Double matrix inverse"* (3×) + *"linalg.Norm on an integer/bool … Double norm"* (2×), all `[PASS]`; cat-06 |
| **`astype(target)`** — dtype conversion | **103 of 121** pairs | works | *"astype succeeds for 103 of the 121 dtype pairs …"* `[PASS]` |
| **`astype` complex→non-complex** (down-cast) | — | **raises** `RuntimeError` (*"not support type"*) | *"the 18 astype FAILURES are EXACTLY the complex-source -> non-complex-target pairs …"* `[PASS]`; cat-07 |

Two findings deserve emphasis:

- **Decompositions do not reject integer tensors — they silently up-cast.** The
  discovered runtime truth (the task's open question) is that `linalg.Svd` /
  `Eigh` / `InvM` / `Norm` on an `Int*`/`Uint*`/`Bool` `Tensor` **do not raise**:
  they promote the input to `Double` internally and return `Double`
  singular-values / eigenvalues / inverse / norm (a real-`Double` result even for
  a `ComplexDouble` input's `S`). So the float/complex requirement is enforced by
  an **implicit up-cast**, not an error — convenient but silent: a caller who
  feeds an int tensor to a decomposition gets a float result with no diagnostic,
  and the returned factors no longer share the input's dtype. This matches
  UniTensor's decomposition behaviour ([UniTensor element-dtypes §3](../UniTensor/element-dtypes.md#3-per-operation-dtype-constraints)).

- **`astype` is not total — a complex tensor cannot be down-cast to real/int/bool.**
  Of the 121 source→target pairs, **18 raise**: exactly the **complex-source →
  non-complex-target** pairs (`ComplexDouble`/`ComplexFloat` → each of the 9
  real/int/bool targets), where `Storage_base::astype` errors *"not support
  type"*. The reverse (real→complex) **does** widen (`Double → ComplexDouble`,
  `Int64 → ComplexDouble` — probe `[PASS]`), and complex→complex works. To drop
  the imaginary part, callers must use `real()` / `imag()` (cat-03, which return
  real-typed **copies**), **not** `astype`. `astype` to the *same* dtype is a
  short-circuit no-op returning `self` (conti.py, cat-07).

Unlike UniTensor, `Tensor` has **no** `get_elem` narrow-dtype cliff: its element
read/write accessors (`item`, `fill`, `t[...]`) bind all 11 dtypes (cat-03
T-E6/T-E7). The one asymmetry `real()`/`imag()` impose — complex-only — is
mirrored by the `astype` complex-down-cast constraint above.

## 4. Recommendation

**Keep the 4 float/complex types — `ComplexDouble`, `ComplexFloat`, `Double`,
`Float` — as the recommended `Tensor` *element* set for numeric work; treat the 7
integer/bool types (`Int64`/`Uint64`/`Int32`/`Uint32`/`Int16`/`Uint16`/`Bool`) as
storage-only element dtypes.**

Rationale, each grounded in a probe finding:

1. **The numeric operations are float/complex-shaped.** Decompositions and
   `InvM`/`Norm` up-cast integer input to `Double` anyway (§3), so an integer
   tensor never survives a linalg step as an integer — it is silently promoted.
   The float/complex set is the only one on which the full operation surface —
   read, arithmetic, decompose, `astype` in both directions — works without a
   raise or a silent up-cast.

2. **`Double` / `ComplexDouble` are the safe defaults; `Float` / `ComplexFloat`
   are the memory-halved variants.** Under the promotion rule (§2) `Double`
   dominates `Float` and complex dominates real, so mixing single into double
   precision widens correctly and mixing real into complex widens correctly — the
   four compose cleanly among themselves, with **no** integer exception (§2a
   applies only to int/int pairs).

3. **Integer/bool tensors are legitimate *storage* — index maps, masks,
   occupation counts — but not *operands*.** They construct (all 11), read/write
   (`item`/`fill`/`t[...]`), and do integer arithmetic, so keep them
   constructible. But **do not** route them through the numeric surface:
   decompositions silently promote them, and their integer arithmetic is subject
   to the §2a exceptions (a result dtype that is *neither operand*). Callers
   should `astype(Type.Double)` (or `ComplexDouble`) **explicitly** before any
   linalg step, rather than relying on the silent internal up-cast.

**Concrete asks for the next version:**

- **Document the promotion rule as `type_promote` (kind-widening), not NumPy
  `result_type`** — it widens by *kind*, not by numeric *range* (§2 caveat), so
  `Float + Int64 → Float`. **Specify the 3 integer exceptions explicitly** (§2a):
  wider-unsigned + narrower-signed → signed-at-wider-width, a result that can be
  *neither operand*. Publish the full 11×11 promotion matrix in the reference.

- **Reconcile the Tensor↔UniTensor promotion divergence.** The two array types
  give different result dtypes for the same 3 unsigned/signed pairs (§2a). Pick
  **one** promotion policy and apply it to both, so `Uint64 + Int32` does not mean
  `Int64` on a `Tensor` but `Uint64` on a `UniTensor`.

- **Make the decomposition up-cast explicit, not silent.** Either require a
  float/complex input (raise a clear error on int/bool, directing the caller to
  `astype`), or keep the up-cast but emit a warning — so the dtype change is not
  invisible. Document that `Svd`/`Eigh`/`InvM`/`Norm` always return
  `Double`-precision results for any integer input.

- **Make the `astype` complex→real error actionable.** The current
  `Storage_base::astype` message *"not support type"* is opaque; it should say
  *"cannot down-cast a complex Tensor to a real dtype via astype; use real()/imag()"*
  — or support the down-cast (taking the real part) explicitly.

---

## Cross-references

- Dtype set + codes: [`per-class/enums.md`](../per-class/enums.md) (`Type`, 11
  dtype codes + `Void`).
- All-11 element read/write (`item`/`fill`), `real()`/`imag()` complex-only
  copies: [`03-element-storage-access.md`](03-element-storage-access.md) (T-E3/T-E6/T-E7).
- Decompositions / matrix inverse: [`06-linalg-reductions.md`](06-linalg-reductions.md) (T-X2/T-X3).
- `astype` conversion + same-dtype short-circuit: [`07-type-device-conversion.md`](07-type-device-conversion.md).
- Shared promotion rule + block-structured case: [UniTensor element-dtypes](../UniTensor/element-dtypes.md)
  (Tensor is the **dense** case; note the §2a promotion divergence and the
  absence of UniTensor's `get_elem` cliff).
- Probe (all claims here): [`../probes/Tensor_dtypes.py`](../probes/Tensor_dtypes.py) (49 assertions, all `[PASS]`).
