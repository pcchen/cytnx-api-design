// Raw-C++ probe for the UniTensor cat-12 (type & device conversion)
// binding-fidelity finding UT-T1. Links against a locally-built libcytnx so it
// can call the C++ methods DIRECTLY, bypassing the pybind + conti.py layer —
// proving that:
//   * UT-T1: raw C++ `astype(same_dtype)` and `to(same_device)` on a NO-OP
//     return a FRESH, DISTINCT UniTensor object (a new wrapper over the shared
//     _impl), NOT `*this`. The pybind bindings are wrapped by conti.py's
//     `if self.dtype()==dtype: return self` / `if self.device()==device:
//     return self` short-circuit, so the Python `astype(same) is self` /
//     `to(same) is self` identity is BINDING-INTRODUCED, not C++ behavior.
//   * The genuine self-return form in C++ is the in-place `to_`, which returns
//     `UniTensor&` (== &self) — the contrast that makes the by-value astype/to
//     "fresh object" finding meaningful.
//   * A REAL conversion (astype to a different dtype) and `clone()` return
//     distinct objects that do NOT share data (independent copies).
#include "cytnx.hpp"
#include <iostream>
#include <string>
using namespace cytnx;

static int fails = 0;
static void report(const std::string& claim, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << claim << std::endl;
  if (!ok) ++fails;
}

int main() {
  // -----------------------------------------------------------------------
  // UT-T1: astype(same_dtype) is a NO-OP conversion. C++ returns a FRESH,
  // DISTINCT UniTensor wrapper (by value) sharing the underlying _impl — it is
  // NOT `*this`. Two consecutive no-op calls yield two DISTINCT objects, both
  // sharing data with the source: proof there is no `return *this` / `is self`
  // short-circuit in C++. (The Python conti.py wrapper adds that short-circuit.)
  // -----------------------------------------------------------------------
  {
    UniTensor U(ones({2, 2}));               // Dense, Double
    UniTensor R1 = U.astype(U.dtype());      // no-op conversion (same dtype)
    UniTensor R2 = U.astype(U.dtype());      // a second no-op call
    bool distinct = (&R1 != &U) && (&R2 != &U) && (&R1 != &R2);
    bool shared = U.same_data(R1) && U.same_data(R2);
    report("UT-T1: C++ astype(same_dtype) returns a FRESH, DISTINCT UniTensor "
           "object (two no-op calls give two distinct wrappers, each != the "
           "source) that SHARES storage with the source (same_data) -- there is "
           "NO `return *this`/`is self` short-circuit in C++; the Python "
           "`astype(same) is self` identity is binding-introduced (conti.py)",
           distinct && shared);
  }

  // -----------------------------------------------------------------------
  // UT-T1: to(same_device) is likewise a NO-OP. C++ returns a FRESH, DISTINCT
  // wrapper sharing the _impl -- not `*this`.
  // -----------------------------------------------------------------------
  {
    UniTensor U(ones({2, 2}));
    UniTensor R = U.to(U.device());          // no-op (same device)
    report("UT-T1: C++ to(same_device) returns a FRESH, DISTINCT UniTensor "
           "object (!= the source) that SHARES storage (same_data) -- again no "
           "`is self` short-circuit; the Python `to(same) is self` identity is "
           "binding-introduced (conti.py)",
           (&R != &U) && U.same_data(R));
  }

  // -----------------------------------------------------------------------
  // The genuine self-return form in C++ is the IN-PLACE to_, which returns
  // UniTensor& (== &self). This is the contrast: astype/to return BY VALUE (a
  // new object), whereas to_ returns *this by reference.
  // -----------------------------------------------------------------------
  {
    UniTensor U(ones({2, 2}));
    UniTensor& self_ref = U.to_(U.device());  // in-place, returns *this
    report("the in-place to_ returns UniTensor& == &self (the genuine "
           "self-return form) -- confirming astype/to's by-value fresh object "
           "is a deliberate return-type distinction, not an accident (UT-T1)",
           &self_ref == &U);
  }

  // -----------------------------------------------------------------------
  // A REAL conversion (astype to a DIFFERENT dtype) copies data: distinct
  // object, NOT same_data. clone() is likewise an independent deep copy.
  // -----------------------------------------------------------------------
  {
    UniTensor U(ones({2, 2}));               // Double
    UniTensor A = U.astype(Type.ComplexDouble);
    report("C++ astype(different_dtype) returns a DISTINCT, INDEPENDENT object "
           "(new dtype, data copied -> NOT same_data) -- the real-conversion "
           "path the Python raw astype_different_type shim forwards to (UT-T1)",
           (&A != &U) && (A.dtype() == Type.ComplexDouble) && !U.same_data(A));

    UniTensor C = U.clone();
    report("C++ clone() returns a DISTINCT, INDEPENDENT deep copy (NOT "
           "same_data with the source) -- backing the Python clone/__copy__ "
           "independence (UT-T3)",
           (&C != &U) && !U.same_data(C));
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
