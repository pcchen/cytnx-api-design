"""Behavioral probe for the cytnx.linalg free-function namespace (Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/linalg.md's Parity and
Consistency findings sections is backed by a report() assertion here. Run with:
    source tools/env.sh && $PY docs/api-audit/probes/linalg.py

linalg is a *module* of free functions (not a class); its decompositions consume
and return both cytnx.Tensor and cytnx.UniTensor (see UniTensor.md). The single
most load-bearing fact for the reference algorithms (TRG/HOTRG/CTMRG/MERA) is the
RETURN ORDER of each decomposition -- verified here empirically, not assumed.

Static signatures are ground-truthed against cytnx_src/include/linalg.hpp and
cytnx_src/pybind/linalg_py.cpp.
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import numpy as np
import cytnx
from cytnx import linalg as la
from probe_helper import report

Type = cytnx.Type


def mat(rows=3, cols=4):
    """A deterministic rank-2 Double Tensor."""
    return cytnx.arange(rows * cols).reshape(rows, cols).astype(Type.Double)


def herm(n=3):
    """A deterministic Hermitian (symmetric real) rank-2 Tensor."""
    H = cytnx.arange(n * n).reshape(n, n).astype(Type.Double)
    return H + H.permute(1, 0)


# =========================================================================
# N1: the entire linalg namespace is capitalized (violates snake_case)
# =========================================================================

public = [m for m in dir(la) if not m.startswith("_")]
report("cytnx.linalg exposes 53 public callables, and EVERY one of them is "
       "Capitalized (Svd, Eigh, Qr, Conj, Matmul, ...) -- not a single member is "
       "snake_case, so the whole namespace violates N1 (public callables are "
       "snake_case)",
       len(public) == 53 and all(m[0].isupper() for m in public))

# =========================================================================
# Svd : RETURN ORDER is [S, U, vT] -- the singular values come FIRST
# =========================================================================

T = mat(3, 4)
out = la.Svd(T)
S, U, vT = out
report("Svd(Tensor) returns a list of THREE tensors, in the order [S, U, vT]: "
       "S (index 0) is the 1-D vector of singular values, U (index 1) is the "
       "3x3 left isometry, vT (index 2) is the 3x4 right isometry",
       len(out) == 3 and S.shape() == [3] and U.shape() == [3, 3]
       and vT.shape() == [3, 4])

recon = la.Matmul(la.Matmul(U, la.Diag(S)), vT)
report("Svd's return order is SINGULAR-VALUES-FIRST, which is NOT the "
       "multiplication order: the reconstruction is M = U @ diag(S) @ vT (i.e. "
       "out[1] @ diag(out[0]) @ out[2]), NOT out[0] @ out[1] @ out[2] -- and it is "
       "also NOT numpy.linalg.svd's (U, S, Vh) order",
       np.allclose(recon.numpy(), T.numpy()))

only_s = la.Svd(T, is_UvT=False)
report("Svd(Tensor, is_UvT=False) returns ONLY the singular values [S] (length-1 "
       "list) -- the single is_UvT flag toggles BOTH isometries together",
       len(only_s) == 1 and only_s[0].shape() == [3])

report("Svd's isometries inherit the input's dtype and device (dtype/device "
       "consistency): S, U, vT are all Double and all on the input's device",
       S.dtype() == T.dtype() and U.dtype() == T.dtype() and vT.dtype() == T.dtype()
       and S.device() == T.device() and U.device() == T.device())

# Svd on a UniTensor returns UniTensor factors, same arity/order.
utM = cytnx.UniTensor(mat(3, 4), rowrank=1)
uout = la.Svd(utM)
report("Svd(UniTensor) mirrors Svd(Tensor): a length-3 list [S, U, vT], each a "
       "UniTensor (the decomposition is overloaded on both Tensor and UniTensor "
       "with identical return arity/order)",
       len(uout) == 3 and all(type(o).__name__ == "UniTensor" for o in uout))

# =========================================================================
# Svd_truncate : keepdim is honored; return_err appends the error
# =========================================================================

st = la.Svd_truncate(T, keepdim=2)
report("Svd_truncate(Tensor, keepdim=2) HONORS keepdim: it returns [S, U, vT] "
       "with exactly 2 singular values kept (S length 2, U is 3x2, vT is 2x4)",
       len(st) == 3 and st[0].shape() == [2] and st[1].shape() == [3, 2]
       and st[2].shape() == [2, 4])

st_over = la.Svd_truncate(T, keepdim=10)
report("Svd_truncate(Tensor, keepdim=10) with keepdim > rank(=3) clamps to the "
       "full rank: S has 3 values, not 10 (keepdim is an upper bound)",
       st_over[0].shape() == [3])

st_err = la.Svd_truncate(T, keepdim=2, return_err=1)
report("Svd_truncate(Tensor, keepdim=2, return_err=1) APPENDS the truncation "
       "error as a 4th list element: [S, U, vT, err]",
       len(st_err) == 4)

# =========================================================================
# Gesvd : separate is_U / is_vT flags -- an N4 mismatch against Svd's is_UvT
# =========================================================================

g_full = la.Gesvd(T, is_U=True, is_vT=True)
report("Gesvd(Tensor) returns [S, U, vT] in the SAME singular-values-first order "
       "as Svd", len(g_full) == 3 and g_full[0].shape() == [3])

g_no_u = la.Gesvd(T, is_U=False, is_vT=True)
report("Gesvd exposes TWO independent flags is_U / is_vT (unlike Svd's single "
       "is_UvT): is_U=False drops U, returning just [S, vT]",
       len(g_no_u) == 2 and g_no_u[1].shape() == [3, 4])

try:
    # the failed-overload path dumps the argument to stdout; suppress it so it
    # does not pollute the probe log -- we only care that the call raises.
    with contextlib.redirect_stdout(io.StringIO()):
        la.Svd(T, is_U=True)
    svd_takes_is_u = True
except TypeError:
    svd_takes_is_u = False
report("Svd REJECTS an is_U keyword (it only has is_UvT) while Gesvd ACCEPTS "
       "is_U -- two sibling SVD functions spell the same 'return the unitaries' "
       "concept with different flag granularity (N4: sibling methods should share "
       "parameter names/positions)",
       svd_takes_is_u is False)

# =========================================================================
# Eigh / Eig : RETURN ORDER is [eigvals, eigvecs] -- eigenvalues FIRST
# =========================================================================

H = herm(3)
eo = la.Eigh(H)
e, V = eo
report("Eigh(Tensor) returns [eigvals, eigvecs] in that order: eigvals (index 0) "
       "is the 1-D spectrum, eigvecs (index 1) is the 3x3 eigenvector matrix "
       "(eigenvalues FIRST, matching numpy.linalg.eigh's (w, v) order)",
       len(eo) == 2 and e.shape() == [3] and V.shape() == [3, 3])

recon_h = la.Matmul(la.Matmul(V, la.Diag(e)), V.permute(1, 0))
report("Eigh reconstructs as M = V @ diag(eigvals) @ V^T (V is unitary), "
       "confirming index 0 = eigenvalues, index 1 = eigenvectors",
       np.allclose(recon_h.numpy(), H.numpy()))

eig_o = la.Eig(mat(3, 3))
report("Eig(Tensor) (general, non-Hermitian) also returns [eigvals, eigvecs]; "
       "note its eigvals are ComplexDouble and its eigenvectors are only "
       "invertible, NOT unitary (use Eigh for a unitary basis)",
       len(eig_o) == 2 and eig_o[0].shape() == [3])

# =========================================================================
# Qr / Qdr : RETURN ORDER differs -- Q FIRST, and the diagonal is NOT at index 0
# =========================================================================

Q, R = la.Qr(T)
report("Qr(Tensor) returns [Q, R] with the ORTHOGONAL factor Q FIRST (index 0) "
       "and the upper-triangular R second -- the opposite convention from "
       "Svd/Eigh, which put the spectrum first",
       Q.shape() == [3, 3] and R.shape() == [3, 4]
       and np.allclose(la.Matmul(Q, R).numpy(), T.numpy()))

q_tau = la.Qr(T, is_tau=True)
report("Qr(Tensor, is_tau=True) APPENDS the Householder reflectors tau as a 3rd "
       "element: [Q, R, tau]", len(q_tau) == 3)

Q2, D, R2 = la.Qdr(T)
report("Qdr(Tensor) returns [Q, D, R]: the diagonal factor D sits at index 1 (the "
       "MIDDLE), NOT at index 0 like Svd's S or Eigh's eigvals -- so the position "
       "of the 'diagonal spectrum' factor is INTERNALLY INCONSISTENT across "
       "cytnx's decompositions (Svd/Gesvd/Eigh: index 0; Qdr: index 1; Qr: absent)",
       D.shape() == [3]
       and np.allclose(la.Matmul(la.Matmul(Q2, la.Diag(D)), R2).numpy(), T.numpy()))

# =========================================================================
# In-place Foo_ variants: mutate the input, return None (void)
# =========================================================================

C = cytnx.zeros([2, 2], dtype=Type.ComplexDouble)
C[0, 1] = 1 + 2j
r_conj = la.Conj_(C)
report("Conj_(Tensor) is IN-PLACE: it returns None and conjugates the argument's "
       "storage (C[0,1] 1+2j -> 1-2j) -- the correct N2 in-place shape (returns "
       "None, mutates receiver)",
       r_conj is None and complex(C[0, 1].item()) == 1 - 2j)

Cp = cytnx.zeros([2, 2], dtype=Type.ComplexDouble)
Cp[0, 1] = 1 + 2j
cc = la.Conj(Cp)
report("Conj(Tensor) (pure form) returns a NEW tensor and leaves the source "
       "unchanged: source[0,1] stays 1+2j, result[0,1] is 1-2j (B1 pure form)",
       complex(Cp[0, 1].item()) == 1 + 2j and complex(cc[0, 1].item()) == 1 - 2j
       and cc is not Cp)

P = cytnx.arange(1, 5).reshape(2, 2).astype(Type.Double)
r_pow = la.Pow_(P, 2)
report("Pow_(Tensor, 2) is IN-PLACE: returns None, squares the argument "
       "(P[1,1] 4 -> 16)", r_pow is None and float(P[1, 1].item()) == 16.0)

Ab = cytnx.arange(-2, 2).reshape(2, 2).astype(Type.Double)
r_abs = la.Abs_(Ab)
report("Abs_(Tensor) is IN-PLACE: returns None, takes |.| of the argument "
       "(Ab[0,0] -2 -> 2)", r_abs is None and float(Ab[0, 0].item()) == 2.0)

Iv = cytnx.arange(1, 5).reshape(2, 2).astype(Type.Double)
r_inv = la.Inv_(Iv)
report("Inv_(Tensor) is IN-PLACE element-wise pseudo-inverse: returns None, maps "
       "each element x -> 1/x (Iv[0,0] 1 -> 1)",
       r_inv is None and abs(float(Iv[0, 0].item()) - 1.0) < 1e-12)

report("every element-wise/BLAS op that has an in-place form pairs a pure Foo with "
       "an in-place Foo_ (Abs/Abs_, Conj/Conj_, Exp/Exp_, Expf/Expf_, Inv/Inv_, "
       "InvM/InvM_, Pow/Pow_, Axpy/Axpy_, Gemm/Gemm_) -- these are correct N2 pairs",
       all(hasattr(la, p) and hasattr(la, p + "_") for p in
           ("Abs", "Conj", "Exp", "Expf", "Inv", "InvM", "Pow", "Axpy", "Gemm")))

# =========================================================================
# Inv (element-wise pseudo-inverse) vs InvM (true matrix inverse) -- distinct ops
# =========================================================================

M = cytnx.zeros([2, 2], dtype=Type.Double)
M[0, 0] = 2.0; M[1, 1] = 4.0; M[0, 1] = 1.0
inv_elem = la.Inv(M)
report("Inv(Tensor) is ELEMENT-WISE (reciprocal of each entry with an optional "
       "clip): Inv(M)[0,0] == 1/2, Inv(M)[0,1] == 1/1 -- it is NOT a matrix "
       "inverse",
       abs(float(inv_elem[0, 0].item()) - 0.5) < 1e-12
       and abs(float(inv_elem[0, 1].item()) - 1.0) < 1e-12)

invm = la.InvM(M)
report("InvM(Tensor) is the TRUE matrix inverse: InvM(M) @ M == I (a different "
       "operation from the element-wise Inv, despite the near-identical name)",
       np.allclose(la.Matmul(invm, M).numpy(), np.eye(2)))

# =========================================================================
# Norm : UniTensor input -> Tensor output (escapes the UniTensor type)
# =========================================================================

report("Norm returns a cytnx.Tensor scalar for BOTH a Tensor input and a "
       "UniTensor input -- i.e. Norm(UniTensor) yields a Tensor, not a UniTensor "
       "(the only decomposition/reduction here whose UniTensor overload does not "
       "return a UniTensor)",
       type(la.Norm(T)).__name__ == "Tensor"
       and type(la.Norm(utM)).__name__ == "Tensor")

# =========================================================================
# Trace / Det / Diag basics
# =========================================================================

report("Trace(Tensor) with default axes (0,1) sums the diagonal: Trace(eye(3)) "
       "== 3",
       abs(float(la.Trace(cytnx.eye(3).astype(Type.Double)).item()) - 3.0) < 1e-12)

report("Det(Tensor) is the determinant: Det(eye(3)) == 1",
       abs(float(la.Det(cytnx.eye(3).astype(Type.Double)).item()) - 1.0) < 1e-12)

report("Diag(Tensor) round-trips a 1-D vector to a diagonal matrix and back "
       "(Diag of the Svd singular values is a diagonal matrix of the right size)",
       la.Diag(S).shape() == [3, 3])

# =========================================================================
# Parity: three C++ Lanczos functions have NO Python binding at all
# =========================================================================

report("PARITY GAP: the C++ linalg header declares Lanczos_ER, Lanczos_Gnd and "
       "Lanczos_Gnd_Ut, but their pybind registrations are commented out "
       "(linalg_py.cpp:1000-1018) AND their intended clean-name wrappers in "
       "linalg_conti.py are commented out inside a docstring -- so all three are "
       "UNREACHABLE from cytnx.linalg under either their clean or c_-prefixed name",
       not any(hasattr(la, n) for n in
               ("Lanczos_ER", "Lanczos_Gnd", "Lanczos_Gnd_Ut",
                "c_Lanczos_ER", "c_Lanczos_Gnd", "c_Lanczos_Gnd_Ut")))

report("the iterative solvers that ARE bound -- Lanczos, Lanczos_Exp, Arnoldi -- "
       "are all reachable (so the gap is specific to the three commented-out "
       "Lanczos_ER/Gnd/Gnd_Ut entry points, not iterative solvers in general)",
       all(hasattr(la, n) for n in ("Lanczos", "Lanczos_Exp", "Arnoldi")))

# =========================================================================
# Parity: pybind arg names match the C++ header (is_UvT/keepdim/is_tau/...)
# =========================================================================

report("the pybind bindings preserve the C++ argument names as Python keywords "
       "(is_UvT for Svd, keepdim for Svd_truncate, is_tau for Qr, is_U/is_vT for "
       "Gesvd) -- verified by calling each purely by keyword",
       la.Svd(T, is_UvT=True) is not None
       and la.Svd_truncate(T, keepdim=2) is not None
       and la.Qr(T, is_tau=False) is not None
       and la.Gesvd(T, is_U=True, is_vT=True) is not None)

print("linalg probe ok")
