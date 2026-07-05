// Raw-C++ probe for the UniTensor cat-07 (arithmetic & element-wise)
// binding-fidelity findings. Links against a locally-built libcytnx so it can
// call the C++ methods DIRECTLY, bypassing the pybind layer — proving that:
//   * UT-A4: the C++ in-place element-wise methods Conj_/Trace_/Pow_ (and the
//     siblings Transpose_/Dagger_/normalize_) each return UniTensor& (&*this),
//     so the conti.py `return self` shims and the leaked raw c* bindings they
//     wrap are BINDING plumbing, not a C++ requirement;
//   * UT-A5: the C++ in-place inverse `UniTensor &Inv_(clip)` EXISTS and returns
//     &*this — so the missing public Python `Inv_` (only the leaked cInv_ is
//     bound) is a BINDING gap, not a C++ gap;
//   * UT-A2: cytnx::operator% / linalg::Mod for a UniTensor and a SCALAR EXIST
//     and compute element-wise modulo (7 % 2 == 1) on a Dense tensor — so the
//     absent Python `%`/__mod__ (its pybind block is commented out) is a
//     BINDING gap for the scalar case; but linalg::Mod(UniTensor,UniTensor) is
//     an UNFINISHED C++ stub that throws "[Mod][Developing]", so the tensor⊗
//     tensor `%` additionally needs the C++ implementation finished.
#include "cytnx.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
using namespace cytnx;

static int fails = 0;
static void report(const std::string& claim, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << claim << std::endl;
  if (!ok) ++fails;
}

// A 1-element Dense UniTensor holding the double value v.
static UniTensor scalar_ut(double v) {
  Tensor t = zeros({1, 1}, Type.Double);
  t.at<cytnx_double>({0, 0}) = v;
  return UniTensor(t);
}

int main() {
  // -----------------------------------------------------------------------
  // UT-A4: the in-place element-wise methods return UniTensor& (&*this). Take
  // the address of the returned reference and compare it to &self.
  // -----------------------------------------------------------------------
  {
    UniTensor U(ones({2, 2}));
    report("UT-A4: C++ Conj_() returns &*this (in place, identity preserved) -- "
           "the Python conti.py `return self` shim over the leaked cConj_ is "
           "binding plumbing, not a C++ requirement",
           &U.Conj_() == &U);
  }
  {
    UniTensor U(ones({2, 2}));
    report("UT-A4: C++ Trace_(0,1) returns &*this (in place)",
           &U.Trace_(cytnx_int64(0), cytnx_int64(1)) == &U);
  }
  {
    UniTensor U(scalar_ut(3.0));
    UniTensor& r = U.Pow_(2.0);
    report("UT-A4: C++ Pow_(2.0) returns &*this AND squares in place (3 -> 9) -- "
           "the conti.py Pow_ over the leaked cPow_ is binding plumbing",
           (&r == &U) && (U.get_elem<cytnx_double>({0, 0}) == 9.0));
  }
  {
    UniTensor U(ones({2, 2}));
    bool all_self = (&U.Transpose_() == &U) && (&U.Dagger_() == &U) &&
                    (&U.normalize_() == &U);
    report("UT-A4: the sibling in-place methods Transpose_/Dagger_/normalize_ "
           "each also return &*this",
           all_self);
  }

  // -----------------------------------------------------------------------
  // UT-A5: C++ `UniTensor &Inv_(double clip)` EXISTS and returns &*this. Python
  // exposes NO public Inv_ (only the leaked raw cInv_) -- a binding gap.
  // -----------------------------------------------------------------------
  {
    UniTensor U(scalar_ut(4.0));
    UniTensor& r = U.Inv_();  // element-wise 1/x in place
    report("UT-A5: C++ Inv_() EXISTS, inverts in place (4 -> 0.25) and returns "
           "&*this -- so the missing public Python Inv_ (only cInv_ is bound) is "
           "a binding gap, not a C++ gap",
           (&r == &U) && (U.get_elem<cytnx_double>({0, 0}) == 0.25));
  }

  // -----------------------------------------------------------------------
  // UT-A2: the UniTensor-and-SCALAR Mod/operator% ARE implemented in C++ and
  // compute element-wise modulo (7 % 2 == 1) on a Dense tensor. The Python
  // __mod__/__rmod__ pybind block is commented out, so even this working case
  // is absent in Python -- a BINDING gap.
  // -----------------------------------------------------------------------
  {
    UniTensor A(scalar_ut(7.0));
    UniTensor C = A % cytnx_double(2.0);       // cytnx::operator%(UniTensor, T)
    UniTensor D = linalg::Mod(A, cytnx_double(2.0));  // linalg::Mod(UniTensor, T)
    report("UT-A2: C++ operator%(UniTensor, scalar) and linalg::Mod(UniTensor, "
           "scalar) ARE implemented and compute element-wise modulo (7 % 2 == 1) "
           "on a Dense tensor -- so the absent Python scalar `%` (its pybind "
           "block is commented out) is a binding gap",
           (C.get_elem<cytnx_double>({0, 0}) == 1.0) &&
           (D.get_elem<cytnx_double>({0, 0}) == 1.0));
  }

  // -----------------------------------------------------------------------
  // UT-A2 (cont.): linalg::Mod(UniTensor, UniTensor) is a DECLARED-but-
  // UNFINISHED C++ stub -- its body is `cytnx_error_msg(true,"[Mod][Developing]")`
  // (Mod.cpp:1029), so it always throws. Binding a tensor⊗tensor Python `%`
  // would therefore also require finishing the C++ implementation.
  // -----------------------------------------------------------------------
  {
    UniTensor A(scalar_ut(7.0)), B(scalar_ut(2.0));
    bool threw = false;
    try {
      UniTensor E = linalg::Mod(A, B);
      (void)E;
    } catch (const std::exception&) {
      threw = true;
    }
    report("UT-A2: linalg::Mod(UniTensor, UniTensor) is an UNFINISHED C++ stub "
           "that throws '[Mod][Developing]' -- so a tensor⊗tensor Python `%` also "
           "needs the C++ implementation completed, not just a binding",
           threw);
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
