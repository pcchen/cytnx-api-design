// Raw-C++ probe for the Tensor cat-05 (arithmetic & element-wise)
// binding-fidelity findings. Links against a locally-built libcytnx so it can
// call the C++ methods DIRECTLY, bypassing the pybind layer — proving that:
//   * T-A4: the C++ in-place element-wise methods Conj_/Abs_/Exp_/Pow_/Inv_
//     each return Tensor& (&*this), so the Tensor_conti.py `return self` shims
//     and the leaked raw c* bindings they wrap (cConj_/cAbs_/cExp_/cPow_/cInv_)
//     are BINDING plumbing, not a C++ requirement. (Do NOT assume — cat 04
//     found contiguous_/flatten_ return `Tensor` by value / void; here the
//     element-wise `_` methods genuinely return `Tensor&`.)
//   * T-A2 (B-5): the in-place matmul the Python `@=` is supposed to perform
//     WORKS in C++ — linalg::Dot computes the matrix product and the
//     `self = Dot(self, rhs)` reassignment (exactly what the raw c__imatmul__
//     primitive does) mutates the tensor correctly. So `@=`'s brokenness is
//     purely the Tensor_conti.py wrapper typo (`__imatmul`, missing the
//     trailing `__`), NOT a C++ or binding-primitive gap.
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

// A rank-1 Double Tensor holding the given values.
static Tensor vec(std::vector<double> xs) {
  Tensor t = zeros((cytnx_uint64)xs.size(), Type.Double);
  for (size_t i = 0; i < xs.size(); ++i) t.at<cytnx_double>({(cytnx_uint64)i}) = xs[i];
  return t;
}

int main() {
  // -----------------------------------------------------------------------
  // T-A4: the in-place element-wise methods return Tensor& (&*this). Take the
  // address of the returned reference and compare it to &self; also check the
  // value where the op is invertible-by-eye.
  // -----------------------------------------------------------------------
  {
    Tensor t = vec({1, 2, 3});
    report("T-A4: C++ Conj_() returns &*this (in place) -- the Tensor_conti.py "
           "`return self` shim over the leaked cConj_ is binding plumbing, not "
           "a C++ requirement",
           &t.Conj_() == &t);
  }
  {
    Tensor t = vec({-1, 2, -3});
    Tensor& r = t.Abs_();
    report("T-A4: C++ Abs_() returns &*this AND abs-es in place (-1 -> 1) -- the "
           "conti.py Abs_ over the leaked cAbs_ is binding plumbing",
           (&r == &t) && (t.at<cytnx_double>({0}) == 1.0));
  }
  {
    Tensor t = vec({0, 1, 2});
    report("T-A4: C++ Exp_() returns &*this (in place)", &t.Exp_() == &t);
  }
  {
    Tensor t = vec({2, 3});
    Tensor& r = t.Pow_(2.0);
    report("T-A4: C++ Pow_(2.0) returns &*this AND squares in place (2 -> 4, "
           "3 -> 9)",
           (&r == &t) && (t.at<cytnx_double>({0}) == 4.0) &&
             (t.at<cytnx_double>({1}) == 9.0));
  }
  {
    Tensor t = vec({2, 4});
    Tensor& r = t.Inv_();  // element-wise 1/x in place (clip default -1)
    report("T-A4: C++ Inv_() returns &*this AND inverts element-wise in place "
           "(2 -> 0.5, 4 -> 0.25) -- distinct from the matrix inverse InvM_",
           (&r == &t) && (t.at<cytnx_double>({0}) == 0.5) &&
             (t.at<cytnx_double>({1}) == 0.25));
  }

  // -----------------------------------------------------------------------
  // T-A2 (B-5): the in-place matmul WORKS in C++. Build A = [[1,2],[3,4]] and
  // the column-swap S = [[0,1],[1,0]]. linalg::Dot(A, S) swaps A's columns to
  // [[2,1],[4,3]]. Then reproduce the raw c__imatmul__ primitive's body
  // `self = Dot(self, rhs)` and confirm the receiver is mutated in place. This
  // proves the Python `@=` failure is the misnamed Tensor_conti.py `__imatmul`
  // wrapper, NOT a C++/primitive gap.
  // -----------------------------------------------------------------------
  {
    Tensor A = zeros({2, 2}, Type.Double);
    A.at<cytnx_double>({0, 0}) = 1; A.at<cytnx_double>({0, 1}) = 2;
    A.at<cytnx_double>({1, 0}) = 3; A.at<cytnx_double>({1, 1}) = 4;
    Tensor S = zeros({2, 2}, Type.Double);
    S.at<cytnx_double>({0, 1}) = 1; S.at<cytnx_double>({1, 0}) = 1;

    Tensor P = linalg::Dot(A, S);  // pure matrix product
    bool dot_ok = (P.at<cytnx_double>({0, 0}) == 2) && (P.at<cytnx_double>({0, 1}) == 1) &&
                  (P.at<cytnx_double>({1, 0}) == 4) && (P.at<cytnx_double>({1, 1}) == 3);
    report("T-A2: C++ linalg::Dot(A, S) computes the matrix product ([[1,2],"
           "[3,4]] @ column-swap == [[2,1],[4,3]])",
           dot_ok);

    // the c__imatmul__ primitive body: self = Dot(self, rhs)
    A = linalg::Dot(A, S);
    bool inplace_ok = (A.at<cytnx_double>({0, 0}) == 2) && (A.at<cytnx_double>({0, 1}) == 1) &&
                      (A.at<cytnx_double>({1, 0}) == 4) && (A.at<cytnx_double>({1, 1}) == 3);
    report("T-A2: the in-place matmul `A = linalg::Dot(A, S)` (exactly the raw "
           "c__imatmul__ primitive body) mutates A correctly in C++ -- so the "
           "Python `@=` brokenness is the misnamed conti.py `__imatmul` wrapper "
           "(missing trailing __), NOT a C++/primitive gap (B-5)",
           inplace_ok);
  }

  std::cout << (fails ? "cpp probe FAILED" : "cpp probe ok") << std::endl;
  return fails ? 1 : 0;
}
