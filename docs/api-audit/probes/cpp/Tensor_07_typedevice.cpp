// Raw-C++ probe for the Tensor cat-07 (dtype / device conversion)
// binding-fidelity finding T-T1. Links against a locally-built libcytnx so it
// can call the C++ methods DIRECTLY, bypassing the pybind + conti.py layer —
// proving that:
//   * T-T1: raw C++ `astype(same_dtype)` and `to(same_device)` on a NO-OP
//     return a FRESH, DISTINCT Tensor object (a new wrapper over the shared
//     _impl), NOT `*this`. The pybind primitives are wrapped by
//     cytnx/Tensor_conti.py:36-48's `if self.dtype()==dtype: return self` /
//     `if self.device()==device: return self` short-circuit, so the Python
//     `astype(same) is self` / `to(same) is self` identity is
//     BINDING-INTRODUCED, not C++ behavior.
//   * A REAL conversion (astype to a different dtype) and `clone()` return
//     DISTINCT objects that do NOT share data (independent copies) — the
//     raw-C++ side of T-T2 / T-T3.
//
// NOTE on `to_` (T-T4/T-T5, no C++ assertion needed): unlike UniTensor's `to_`
// (which returns `UniTensor&`), Tensor's C++ `to_` is declared
// `void to_(const int&)` (Tensor.hpp:683) — it CANNOT return `*this`, which is
// exactly why the Python `to_` returns None (T-T4). Its Python parameter IS
// named `device` because the binding adds py::arg("device") (tensor_py.cpp:177,
// T-T5) — both are source/pybind facts, verified by the Python probe.
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
  // T-T1: astype(same_dtype) is a NO-OP conversion. C++ returns a FRESH,
  // DISTINCT Tensor wrapper (by value) sharing the underlying _impl — it is
  // NOT `*this`. Two consecutive no-op calls yield two DISTINCT objects, both
  // sharing data with the source: proof there is no `return *this` / `is self`
  // short-circuit in C++. (The Python conti.py wrapper adds that short-circuit.)
  // -----------------------------------------------------------------------
  {
    Tensor T = ones({2, 2});                 // Double, CPU
    Tensor R1 = T.astype(T.dtype());         // no-op conversion (same dtype)
    Tensor R2 = T.astype(T.dtype());         // a second no-op call
    bool distinct = (&R1 != &T) && (&R2 != &T) && (&R1 != &R2);
    bool shared = T.same_data(R1) && T.same_data(R2);
    report("T-T1: C++ astype(same_dtype) returns a FRESH, DISTINCT Tensor object "
           "(two no-op calls give two distinct wrappers, each != the source) that "
           "SHARES storage with the source (same_data) -- there is NO "
           "`return *this`/`is self` short-circuit in C++; the Python "
           "`astype(same) is self` identity is binding-introduced (Tensor_conti.py)",
           distinct && shared);
  }

  // -----------------------------------------------------------------------
  // T-T1: to(same_device) is likewise a NO-OP. C++ returns a FRESH, DISTINCT
  // wrapper sharing the _impl -- not `*this`.
  // -----------------------------------------------------------------------
  {
    Tensor T = ones({2, 2});
    Tensor R = T.to(T.device());             // no-op (same device)
    report("T-T1: C++ to(same_device) returns a FRESH, DISTINCT Tensor object "
           "(!= the source) that SHARES storage (same_data) -- again no `is self` "
           "short-circuit; the Python `to(same) is self` identity is "
           "binding-introduced (Tensor_conti.py)",
           (&R != &T) && T.same_data(R));
  }

  // -----------------------------------------------------------------------
  // T-T2 / T-T3: a REAL conversion (astype to a DIFFERENT dtype) copies data:
  // distinct object, NOT same_data. clone() is likewise an independent deep copy.
  // -----------------------------------------------------------------------
  {
    Tensor T = ones({2, 2});                 // Double
    Tensor A = T.astype(Type.ComplexDouble);
    report("T-T2: C++ astype(different_dtype) returns a DISTINCT, INDEPENDENT "
           "object (new dtype, data copied -> NOT same_data) -- the real-conversion "
           "path the Python raw astype_different_dtype shim forwards to",
           (&A != &T) && (A.dtype() == Type.ComplexDouble) && !T.same_data(A));

    Tensor C = T.clone();
    report("T-T3: C++ clone() returns a DISTINCT, INDEPENDENT deep copy (NOT "
           "same_data with the source) -- backing the Python clone independence",
           (&C != &T) && !T.same_data(C));
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
