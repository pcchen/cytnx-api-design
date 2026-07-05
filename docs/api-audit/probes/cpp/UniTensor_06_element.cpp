// Raw-C++ probe for the UniTensor cat-06 (element & block access)
// binding-fidelity findings. Links against a locally-built libcytnx so it can
// call the C++ methods DIRECTLY, bypassing the pybind layer — proving that:
//   * UT-E1: the C++ get_elem<T> template covers ALL 11 element dtypes (int,
//     uint, bool AND float/complex), so the Python get_elem lambda's 4-dtype
//     (float/complex-only) limit is a BINDING choice, not a C++ gap;
//   * UT-E2: the accessor methods get(accessors) / set(accessors, rhs) EXIST as
//     C++ methods (Python reaches them only via __getitem__/__setitem__, which
//     are the only public spelling — get/set are absent from dir(UniTensor));
//   * UT-E3: get_block_ returns a SHARED-DATA view (const Tensor& onto the
//     tensor's block) while get_block returns a COPY (clone).
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

// A round-trip through the C++ get_elem<T>/set_elem<T> templates on a
// 1-element UniTensor of the matching storage dtype `dt`: writes 1, reads it
// back, and returns true iff the value survives. get_elem/set_elem are
// [[deprecated]] in C++ ("Use at() instead."), so the caller silences that.
template <class T>
static bool roundtrip_dtype(int dt) {
  UniTensor U(zeros({1, 1}, dt));
  U.set_elem<T>({0, 0}, (T)1);
  return U.get_elem<T>({0, 0}) == (T)1;
}

int main() {
  // -----------------------------------------------------------------------
  // UT-E1: C++ `get_elem<T>` (and `set_elem<T2>`) are generic TEMPLATES (they
  // forward to at<T>()), instantiable for every element dtype T. Round-trip a
  // value through all 11 storage dtypes — including the 7 integer/bool dtypes
  // the Python get_elem lambda REJECTS (it only wires up the 4 float/complex
  // branches). So the Python 4-dtype limit is a BINDING choice, not a C++ gap.
  // -----------------------------------------------------------------------
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  {
    bool all_ok =
      roundtrip_dtype<cytnx_double>(Type.Double) &&
      roundtrip_dtype<cytnx_float>(Type.Float) &&
      roundtrip_dtype<cytnx_complex128>(Type.ComplexDouble) &&
      roundtrip_dtype<cytnx_complex64>(Type.ComplexFloat) &&
      roundtrip_dtype<cytnx_int64>(Type.Int64) &&
      roundtrip_dtype<cytnx_uint64>(Type.Uint64) &&
      roundtrip_dtype<cytnx_int32>(Type.Int32) &&
      roundtrip_dtype<cytnx_uint32>(Type.Uint32) &&
      roundtrip_dtype<cytnx_int16>(Type.Int16) &&
      roundtrip_dtype<cytnx_uint16>(Type.Uint16) &&
      roundtrip_dtype<cytnx_bool>(Type.Bool);
    report("UT-E1: C++ get_elem<T>/set_elem<T> are templates covering ALL 11 "
           "element dtypes (a value round-trips on Double/Float/ComplexDouble/"
           "ComplexFloat AND on all 7 integer/bool dtypes) -- the Python "
           "get_elem's 4-dtype (float/complex-only) limit is a binding choice, "
           "not a C++ gap",
           all_ok);
  }

  // The asymmetry is ONLY on the get_elem read side: the Python set_elem/item
  // bindings already cover all 11 (as their C++ templates do here).
  {
    bool ok_b = roundtrip_dtype<cytnx_bool>(Type.Bool);
    UniTensor V(zeros({1, 1}, Type.Double));
    V.set_elem<cytnx_double>({0, 0}, 2.5);
    bool ok_d = V.get_elem<cytnx_double>({0, 0}) == 2.5;
    report("the C++ get/set element templates handle both a bool tensor and a "
           "fractional double cleanly -- confirming the dtype asymmetry lives "
           "only in the Python get_elem binding, not in C++ (UT-E1)",
           ok_b && ok_d);
  }
#pragma GCC diagnostic pop

  // -----------------------------------------------------------------------
  // UT-E2: the accessor methods `get(accessors)` / `set(accessors, rhs)` EXIST
  // as C++ methods. Python does NOT bind them by name (they are absent from
  // dir(UniTensor)); the only public Python path is __getitem__/__setitem__,
  // whose lambdas call exactly these. Here we call them DIRECTLY.
  // -----------------------------------------------------------------------
  {
    // get(): slice row 0 of a [2,3] Dense tensor -> shape [3].
    UniTensor U(arange(6).reshape(2, 3));
    UniTensor sub = U.get({Accessor(0), Accessor::all()});
    report("UT-E2: C++ get(accessors) EXISTS and slices a Dense tensor "
           "(U.get({0, :}) -> shape [3]) -- Python reaches it only via "
           "__getitem__ (get is not a public Python member)",
           sub.shape() == std::vector<cytnx_uint64>{3});
  }
  {
    // set(): assign a row of a [2,3] Dense tensor from a [3] Tensor.
    UniTensor U(zeros({2, 3}));
    Tensor rhs = ones({3});
    UniTensor& r = U.set({Accessor(0), Accessor::all()}, rhs);
    bool wrote = double(U.at<double>({0, 0}) == 1.0) &&
                 double(U.at<double>({1, 0}) == 0.0);
    report("UT-E2: C++ set(accessors, rhs) EXISTS and assigns into a Dense "
           "tensor (row 0 becomes 1, row 1 stays 0) returning *this -- Python "
           "reaches it only via __setitem__ (set is not a public Python member)",
           (&r == &U) && wrote);
  }

  // -----------------------------------------------------------------------
  // UT-E3: get_block_ returns a SHARED-DATA view; get_block returns a COPY.
  // -----------------------------------------------------------------------
  {
    UniTensor U(ones({2, 2}));
    Tensor& view = U.get_block_();   // reference into the tensor's block
    Tensor copy = U.get_block();     // a clone
    report("UT-E3: C++ get_block_ returns a shared-data VIEW (same_data with the "
           "tensor's block) while get_block returns an independent COPY "
           "(not same_data)",
           U.get_block_().same_data(view) && !U.get_block_().same_data(copy));
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
