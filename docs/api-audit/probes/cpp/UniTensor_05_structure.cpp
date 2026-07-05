// Raw-C++ probe for the UniTensor cat-05 (structure manipulation)
// binding-fidelity findings. Links against a locally-built libcytnx so it can
// call the C++ methods DIRECTLY, bypassing the pybind layer — proving that:
//   * combineBond (singular) EXISTS as a C++ method (so its Python absence is a
//     binding gap, not a C++ gap), and returns &*this like a proper in-place op;
//   * the in-place structure ops permute_ / contiguous_ / truncate_ return
//     &*this in C++ (the fidelity the Python bindings must / mostly do keep).
#include "cytnx.hpp"
#include <iostream>
#include <vector>
#include <string>
using namespace cytnx;

static int fails = 0;
static void report(const std::string& claim, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << claim << std::endl;
  if (!ok) ++fails;
}

int main() {
  // UT-S5: C++ `combineBond` (singular) EXISTS and is the current form the
  // deprecated `combineBonds` points callers to. It mutates in place and
  // returns &*this (UniTensor&). Python binds only `combineBonds` (plural,
  // deprecated) and NOT this method -> the Python absence is a binding gap.
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    U.relabel_(std::vector<std::string>{"a", "b", "c"});
    UniTensor& r = U.combineBond(std::vector<std::string>{"a", "b"});
    report("UT-S5: C++ combineBond (singular) EXISTS, combines bonds in place "
           "and returns &*this (&r == &U); shape [2,3,4] -> [6,4] -- the "
           "current form the deprecated combineBonds redirects to, yet it is "
           "UNBOUND in Python (binding gap, not a C++ gap)",
           &r == &U && U.shape() == std::vector<cytnx_uint64>{6, 4});
  }

  // combineBonds (plural) is [[deprecated]] on the C++ side (the header marks it
  // so and tells callers to use combineBond). Prove it still runs in-place and
  // returns void -- matching the Python binding's None return.
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    U.combineBonds(std::vector<cytnx_int64>{0, 1});  // returns void
#pragma GCC diagnostic pop
    report("combineBonds (plural) is [[deprecated]] in C++ and returns void "
           "(the binding's None return is faithful); shape [2,3,4] -> [6,4]",
           U.shape() == std::vector<cytnx_uint64>{6, 4});
  }

  // UT-S1: C++ `permute_` returns a reference to *this (in-place). The Python
  // binding's `return &self.permute_(...)` preserves this self-return.
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    UniTensor& r = U.permute_(std::vector<cytnx_int64>{2, 0, 1});
    report("UT-S1: C++ permute_ permutes in place and returns &*this (&r == &U); "
           "shape becomes [4,2,3]",
           &r == &U && U.shape() == std::vector<cytnx_uint64>{4, 2, 3});
  }

  // C++ `permute` (no underscore) returns a DISTINCT object (pure) whose data is
  // SHARED with the source -- the shared-data view the Python probe observes.
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    UniTensor V = U.permute(std::vector<cytnx_int64>{2, 0, 1});
    U.at<double>({0, 0, 0}) = 9.0;  // mutate source data
    bool shared = double(V.at<double>({0, 0, 0})) == 9.0;
    report("C++ permute (pure) returns a distinct object with reordered legs "
           "but SHARED data (mutating the source shows through the permute)",
           (&V != &U) && V.shape() == std::vector<cytnx_uint64>{4, 2, 3} && shared);
  }

  // UT-S4: C++ `contiguous_` returns &*this (in-place); `contiguous` (no
  // underscore) returns a distinct object. The N-underscore split is real in C++.
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    U.permute_(std::vector<cytnx_int64>{2, 0, 1});  // make non-contiguous
    UniTensor& r = U.contiguous_();
    report("UT-S4: C++ contiguous_ coalesces storage in place and returns &*this "
           "(&r == &U), leaving the tensor contiguous",
           &r == &U && U.is_contiguous());
  }
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    U.permute_(std::vector<cytnx_int64>{2, 0, 1});
    UniTensor V = U.contiguous();
    report("C++ contiguous (no underscore) returns a distinct, contiguous "
           "object while the receiver stays non-contiguous (pure)",
           (&V != &U) && V.is_contiguous() && !U.is_contiguous());
  }

  // UT-S6: C++ `truncate_` returns &*this (in-place); the Python `truncate_` is
  // a conti.py wrapper over the raw `ctruncate_` binding (which calls this).
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    UniTensor& r = U.truncate_((cytnx_int64)0, (cytnx_uint64)1);
    report("UT-S6: C++ truncate_ truncates a bond in place and returns &*this "
           "(&r == &U); shape [2,3,4] -> [1,3,4]",
           &r == &U && U.shape() == std::vector<cytnx_uint64>{1, 3, 4});
  }

  // C++ `reshape_` returns &*this; `reshape` (pure) returns a distinct object.
  // The Python binding erases their (new_shape, rowrank) signature to
  // (*args, **kwargs) -- a Python-side signature loss, not a C++ fact.
  {
    UniTensor U(UniTensor::ones({2, 3, 4}));
    UniTensor& r = U.reshape_(std::vector<cytnx_int64>{6, 4});
    report("C++ reshape_ has a real (new_shape, rowrank) signature, reshapes in "
           "place and returns &*this (&r == &U) -- the Python *args binding is a "
           "binding-side signature erasure, not a C++ fact; shape -> [6,4]",
           &r == &U && U.shape() == std::vector<cytnx_uint64>{6, 4});
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
