// Raw-C++ probe for the UniTensor cat-04 (labels/name/rowrank)
// binding-fidelity findings. Links against a locally-built libcytnx so it can
// call the C++ methods DIRECTLY, bypassing the pybind layer — proving that the
// Python None/return-self divergences are binding-introduced, not C++ facts.
#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;

static int fails = 0;
static void report(const std::string& claim, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << claim << std::endl;
  if (!ok) ++fails;
}

int main() {
  // UT-L2: C++ `UniTensor& relabel_(...)` returns a reference to *this.
  // The pybind raw `c_relabel_` lambda returns void (Python None); the conti.py
  // wrapper re-adds return-self. Prove the C++ side already returns &*this.
  {
    UniTensor U(UniTensor::zeros({2, 3}));
    UniTensor& r = U.relabel_(std::vector<std::string>{"a", "b"});
    report("UT-L2: C++ relabel_ returns a reference to self (&r == &U) -- the "
           "binding, not C++, drops the return (raw c_relabel_ returns None)",
           &r == &U && U.labels() == std::vector<std::string>{"a", "b"});
  }

  // UT-L1: C++ `UniTensor relabel(...) const` returns a DISTINCT object whose
  // metadata (labels) differ but whose internal data is SHARED with the source
  // (header @attention: "the data is still shared with the original UniTensor").
  {
    UniTensor U(UniTensor::zeros({2, 3}));
    UniTensor V = U.relabel(std::vector<std::string>{"a", "b"});
    bool pure_labels = U.labels() == std::vector<std::string>{"0", "1"} &&
                       V.labels() == std::vector<std::string>{"a", "b"};
    U.at<double>({0, 0}) = 9.0;  // mutate source data
    bool shared = double(V.at<double>({0, 0})) == 9.0;
    report("UT-L1: C++ relabel(...) returns a distinct object with new labels "
           "but SHARED data (mutating the source shows through the relabel)",
           pure_labels && shared);
  }

  // Binding fidelity: C++ set_labels (deprecated) mutates labels IN-PLACE and
  // returns *this -- matching the header. (The Python binding routes
  // c_set_labels through relabel_ with the same in-place effect.)
  {
    UniTensor U(UniTensor::zeros({2, 3}));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    UniTensor& r = U.set_labels(std::vector<std::string>{"a", "b"});
#pragma GCC diagnostic pop
    report("C++ set_labels mutates labels in place and returns &*this (matches "
           "the header; Python c_set_labels routes through relabel_)",
           &r == &U && U.labels() == std::vector<std::string>{"a", "b"});
  }

  // set_name returns *this on the C++ side (the conti.py set_name wrapper's
  // return-self mirrors this).
  {
    UniTensor U(UniTensor::zeros({2, 3}));
    UniTensor& r = U.set_name("MyTensor");
    report("C++ set_name returns a reference to self (&r == &U) and sets name",
           &r == &U && U.name() == "MyTensor");
  }

  // set_rowrank_ returns *this; set_rowrank (no underscore) returns a distinct
  // object -- the N-underscore split is a genuine C++ distinction.
  {
    UniTensor U(UniTensor::zeros({2, 3}));
    UniTensor& r = U.set_rowrank_(0);
    UniTensor V = U.set_rowrank(1);
    report("C++ set_rowrank_ returns &*this (in-place) while set_rowrank "
           "returns a distinct object (pure) -- N-underscore split is real",
           &r == &U && U.rowrank() == 0 && (&V != &U) && V.rowrank() == 1);
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
