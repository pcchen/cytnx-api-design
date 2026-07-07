// Raw-C++ probe for the Tensor cat-04 (shape / layout) binding-fidelity
// findings. Links against a locally-built libcytnx so it can call the C++
// methods DIRECTLY, bypassing the pybind layer — proving where the Python
// return/identity behavior originates in the C++ signature vs. the binding:
//   * permute_ / reshape_ return Tensor& (&*this) in C++, so the Python self-
//     return the pybind lambdas give (`return &self.permute_(...)`) is FAITHFUL
//     (finding T-S2);
//   * permute (no underscore) returns a DISTINCT object that SHARES data — the
//     shared-data view the Python probe observes (finding T-S1);
//   * contiguous_ returns `Tensor` BY VALUE (a *this copy sharing storage), NOT
//     Tensor& — so Python's distinct-object return is FAITHFUL to C++; the
//     broken self-return convention is ROOTED IN THE C++ SIGNATURE, not the
//     binding (finding T-S5);
//   * flatten_ returns `void` in C++ — so Python's None is FAITHFUL; the missing
//     self-return is ROOTED IN THE C++ SIGNATURE (finding T-S7).
#include "cytnx.hpp"
#include <iostream>
#include <type_traits>
#include <vector>
using namespace cytnx;

static int fails = 0;
static void report(const std::string& claim, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << claim << std::endl;
  if (!ok) ++fails;
}

// Static (compile-time) facts about the C++ return TYPES — the root cause of
// the Python return-convention differences. These are the load-bearing
// binding-fidelity facts a runtime `is`-check alone cannot pin down.
static_assert(
    std::is_same<decltype(std::declval<Tensor&>().permute_(
                     std::declval<const std::vector<cytnx_uint64>&>())),
                 Tensor&>::value,
    "C++ permute_ must return Tensor& (self)");
static_assert(
    std::is_same<decltype(std::declval<Tensor&>().reshape_(
                     std::declval<const std::vector<cytnx_int64>&>())),
                 Tensor&>::value,
    "C++ reshape_ must return Tensor& (self)");
static_assert(
    std::is_same<decltype(std::declval<Tensor&>().contiguous_()), Tensor>::value,
    "C++ contiguous_ returns Tensor BY VALUE (not Tensor&) — root of T-S5");
static_assert(
    std::is_same<decltype(std::declval<Tensor&>().flatten_()), void>::value,
    "C++ flatten_ returns void — root of T-S7");

int main() {
  report("T-S2: C++ permute_ has return type Tensor& (static_assert) — the "
         "pybind lambda's `return &self.permute_(...)` self-return is FAITHFUL",
         true);
  report("T-S2: C++ reshape_ has return type Tensor& (static_assert) — its "
         "pybind self-return is FAITHFUL",
         true);
  report("T-S5: C++ contiguous_ has return type Tensor BY VALUE, NOT Tensor& "
         "(static_assert) — so Python's distinct-object return is FAITHFUL to "
         "C++; the broken in-place self-return is ROOTED IN THE C++ SIGNATURE",
         true);
  report("T-S7: C++ flatten_ has return type void (static_assert) — so Python's "
         "None return is FAITHFUL to C++; the missing self-return is ROOTED IN "
         "THE C++ SIGNATURE",
         true);

  // T-S2 (runtime): C++ permute_ / reshape_ return &*this — the actual self.
  {
    Tensor t = arange(24).reshape(2, 3, 4);
    Tensor& r = t.permute_(std::vector<cytnx_uint64>{2, 0, 1});
    report("T-S2: C++ permute_ permutes in place and returns &*this (&r == &t); "
           "shape becomes [4,2,3]",
           &r == &t && t.shape() == std::vector<cytnx_uint64>{4, 2, 3});
  }
  {
    Tensor t = arange(24).reshape(2, 3, 4);
    Tensor& r = t.reshape_(std::vector<cytnx_int64>{6, 4});
    report("T-S2: C++ reshape_ reshapes in place and returns &*this (&r == &t); "
           "shape becomes [6,4]",
           &r == &t && t.shape() == std::vector<cytnx_uint64>{6, 4});
  }

  // T-S1: C++ permute (no underscore) returns a DISTINCT object whose data is
  // SHARED with the source — the shared-data view the Python probe observes.
  {
    Tensor t = arange(24).reshape(2, 3, 4);
    Tensor v = t.permute(std::vector<cytnx_uint64>{2, 0, 1});
    t.at<double>({0, 0, 0}) = 9.0;  // mutate source data
    bool shared = double(v.at<double>({0, 0, 0})) == 9.0;
    report("T-S1: C++ permute (pure) returns a distinct object with reordered "
           "legs but SHARED data (mutating the source shows through the permute)",
           (&v != &t) && v.shape() == std::vector<cytnx_uint64>{4, 2, 3} && shared);
  }

  // T-S5 (runtime): the by-value contiguous_ result shares storage with the
  // receiver (it is *this copied — a shared-data wrapper) and coalesces the
  // receiver in place.
  {
    Tensor t = arange(24).reshape(2, 3, 4).permute(std::vector<cytnx_uint64>{2, 0, 1});
    bool was_noncontig = !t.is_contiguous();
    Tensor r = t.contiguous_();  // by-value copy of *this
    report("T-S5: C++ contiguous_ coalesces the receiver in place (now "
           "contiguous) and its BY-VALUE result shares storage with the "
           "receiver (same_data) — a shared-data wrapper, not &*this",
           was_noncontig && t.is_contiguous() && r.same_data(t));
  }

  // T-S1/T-S8: C++ contiguous (no underscore) returns a distinct object; on a
  // non-contiguous tensor it is an INDEPENDENT copy (does not share data).
  {
    Tensor t = arange(24).reshape(2, 3, 4).permute(std::vector<cytnx_uint64>{2, 0, 1});
    Tensor c = t.contiguous();
    report("T-S8: C++ contiguous (no underscore, the raw method bound as "
           "make_contiguous) returns a distinct, contiguous object; on a "
           "non-contiguous tensor it does NOT share data with the source",
           (&c != &t) && c.is_contiguous() && !c.same_data(t));
  }

  // T-S7 (runtime): C++ flatten_ collapses the receiver in place; being void it
  // cannot return self at all.
  {
    Tensor t = arange(24).reshape(2, 3, 4);
    t.flatten_();
    report("T-S7: C++ flatten_ collapses the receiver to rank 1 in place "
           "(shape [24]); being void it returns nothing",
           t.shape() == std::vector<cytnx_uint64>{24});
  }

  // T-S6: C++ flatten (no underscore) returns an INDEPENDENT 1-D copy
  // (clone + contiguous_ + reshape_), leaving the source unchanged.
  {
    Tensor t = arange(24).reshape(2, 3, 4);
    Tensor f = t.flatten();
    report("T-S6: C++ flatten (pure) returns an independent rank-1 copy "
           "(shape [24], does not share data) while the source stays rank 3",
           f.shape() == std::vector<cytnx_uint64>{24} && !f.same_data(t) &&
               t.shape() == std::vector<cytnx_uint64>{2, 3, 4});
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
