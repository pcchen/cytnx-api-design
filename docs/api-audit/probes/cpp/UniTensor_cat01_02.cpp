// Raw-C++ probe for the UniTensor binding-fidelity findings (cat 01/02).
// Links against a locally-built (GCC 13) libcytnx so it can call the C++
// methods DIRECTLY, bypassing the pybind layer — the one thing the Python
// probe cannot do. Confirms the C++-side facts the A3 findings assert from
// source-reading.
#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;

static int fails = 0;
static void report(const std::string& claim, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << claim << std::endl;
  if (!ok) ++fails;
}

int main() {
  // UT-G5: C++ `UniTensor& normal_(...)` returns a reference to *this.
  // The pybind lambda drops this (Python returns None). Prove the C++ side.
  {
    UniTensor Z(zeros({4}));
    UniTensor& r = Z.normal_(0.0, 1.0, 7);
    report("UT-G5: C++ normal_ returns a reference to self (&r == &Z) -- the "
           "binding, not C++, drops the return", &r == &Z);
  }

  // UT-C2: the from-Tensor constructor shares memory with the source Tensor
  // (a view) -- confirm on the raw C++ side, not only via Python.
  {
    Tensor T = zeros({2, 2});
    UniTensor U(T);
    T.at<double>({0, 0}) = 9.0;
    report("UT-C2: C++ UniTensor(Tensor) shares memory (mutating T shows in U)",
           double(U.at<double>({0, 0})) == 9.0);
  }

  // UT-G11: C++ normal_ takes `seed` literally -- it has NO seed==-1 ->
  // random_device rule (that lives only in the pybind lambda). So two C++
  // fills with the SAME literal seed -1 are identical (deterministic),
  // whereas Python's seed=-1 is nondeterministic.
  {
    UniTensor A(zeros({8})), B(zeros({8}));
    A.normal_(0.0, 1.0, -1);
    B.normal_(0.0, 1.0, -1);
    bool identical = true;
    for (int i = 0; i < 8; ++i)
      if (double(A.at<double>({(cytnx_uint64)i})) !=
          double(B.at<double>({(cytnx_uint64)i}))) { identical = false; break; }
    report("UT-G11: C++ normal_(...,-1) treats -1 as a LITERAL seed "
           "(two fills identical) -- the -1->random_device rule is binding-only",
           identical);
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
