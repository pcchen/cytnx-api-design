"""Behavioral probe for UniTensor category 09 — linalg Krylov solvers (the
cytnx.linalg FREE functions Lanczos / Arnoldi / Lanczos_Exp that act on a LinOp
and a Tensor/UniTensor start vector), verified against the installed cytnx==1.1.0
wheel (NOT source-inferred).

These are cytnx.linalg free functions, NOT UniTensor members. Every runtime
claim in docs/api-audit/UniTensor/09-linalg-solvers.md is backed by a report(...)
assertion here. No C++ probe accompanies this category: the reachable solvers are
bound as direct pass-through pybind lambdas to the C++ cytnx::linalg:: overloads
(pybind/linalg_py.cpp:918-990), with no conti.py wrapper — so there is no
binding-fidelity finding to verify on a separate raw-C++ side.

Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_09_solvers.py
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import numpy as np
import cytnx
from cytnx import linalg as la
from probe_helper import report

UT = cytnx.UniTensor


def raises(call, exc=Exception):
    """True if `call` raises `exc`; cytnx's noisy stdout/stderr repr is swallowed
    so probe output stays clean."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            call()
        return False
    except exc:
        return True


def spd(n, seed):
    """A dense symmetric-positive-definite numpy matrix with GENERIC (non-
    symmetric) eigenvectors, so a Krylov run from a random start vector has
    non-zero overlap with the true ground state."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


class DenseOp(cytnx.LinOp):
    """A LinOp wrapping a dense Hermitian matrix; matvec handles BOTH a Tensor
    and a UniTensor input (the two Lanczos/Arnoldi operand overloads)."""

    def __init__(self, m):
        cytnx.LinOp.__init__(self, "mv", m.shape()[0], cytnx.Type.Double,
                             cytnx.Device.cpu)
        self.m = m

    def matvec(self, v):
        if isinstance(v, cytnx.UniTensor):
            return cytnx.UniTensor(cytnx.linalg.Dot(self.m, v.get_block()))
        return cytnx.linalg.Dot(self.m, v)


def expm(M):
    w, V = np.linalg.eigh(M)
    return (V * np.exp(w)) @ V.T


# ---------------------------------------------------------------------------
# 1. Reachability at runtime — which brief solver names are present / absent.
#    (Lanczos_ER / Lanczos_Gnd / Lanczos_Gnd_Ut were commented out in 1.1.0:
#     their pybind c_* registrations sit inside a /* */ block
#     (linalg_py.cpp:999-1019) and their conti wrappers inside a """...""" string
#     (installed cytnx/linalg_conti.py:4-23) — so NEITHER is reachable.)
# ---------------------------------------------------------------------------
present = {m for m in dir(la) if not m.startswith("_")}

for m in ["Lanczos", "Arnoldi", "Lanczos_Exp"]:
    report(f"linalg.{m} is reachable at runtime", m in present)

for m in ["Lanczos_Gnd_Ut", "Lanczos_ER", "Lanczos_Gnd"]:
    report(f"linalg.{m} is ABSENT at runtime (commented-out pybind + conti "
           f"registration in 1.1.0 — capability/parity gap)", m not in present)

# The raw c_* bindings are ALSO absent — not even leaked into the namespace.
for m in ["c_Lanczos_ER", "c_Lanczos_Gnd", "c_Lanczos_Gnd_Ut"]:
    report(f"the raw {m} binding is not leaked either (commented out in pybind)",
           not hasattr(la, m))

# ---------------------------------------------------------------------------
# 2. N-casing — the reachable solvers are Capitalized FREE functions (kept),
#    with no snake_case duplicate.
# ---------------------------------------------------------------------------
report("the Krylov solvers are Capitalized at runtime (Lanczos/Arnoldi/"
       "Lanczos_Exp) — free functions acting on objects keep Capitalized names",
       all(hasattr(la, n) for n in ["Lanczos", "Arnoldi", "Lanczos_Exp"]))
report("their lowercase spellings are NOT bound (no snake_case duplicate) — "
       "e.g. linalg.lanczos / linalg.arnoldi are absent",
       not any(hasattr(la, n) for n in ["lanczos", "arnoldi", "lanczos_exp"]))

# ---------------------------------------------------------------------------
# 3. Operand-type support (Tensor vs UniTensor vs LinOp) — read from the
#    pybind-generated overload docstrings. All solvers take a LinOp `Hop` first.
# ---------------------------------------------------------------------------
report("Lanczos carries FOUR overloads (Tensor+method, UniTensor+method, "
       "Tensor+which, UniTensor+which)", la.Lanczos.__doc__.count(". Lanczos(") == 4)
report("Arnoldi carries TWO overloads (Tensor and UniTensor)",
       la.Arnoldi.__doc__.count(". Arnoldi(") == 2)
report("every solver takes a LinOp Hop as its first operand",
       all("Hop: cytnx.cytnx.LinOp" in (getattr(la, n).__doc__ or "")
           for n in ["Lanczos", "Arnoldi", "Lanczos_Exp"]))
report("Lanczos has BOTH a Tensor and a UniTensor operand overload",
       "Tin: cytnx.cytnx.Tensor" in la.Lanczos.__doc__
       and "Tin: cytnx.cytnx.UniTensor" in la.Lanczos.__doc__)
report("Arnoldi has BOTH a Tensor and a UniTensor operand overload",
       "Tin: cytnx.cytnx.Tensor" in la.Arnoldi.__doc__
       and "Tin: cytnx.cytnx.UniTensor" in la.Arnoldi.__doc__)
report("Lanczos_Exp is UniTensor-ONLY (its `v` operand has no Tensor overload) "
       "— a parity gap vs Lanczos/Arnoldi",
       "v: cytnx.cytnx.UniTensor" in la.Lanczos_Exp.__doc__
       and "v: cytnx.cytnx.Tensor" not in la.Lanczos_Exp.__doc__)
report("Lanczos_Exp rejects a Tensor start vector at call time (UniTensor-only)",
       raises(lambda: la.Lanczos_Exp(DenseOp(cytnx.from_numpy(spd(5, 0).copy())),
                                     cytnx.ones(5), cytnx.Scalar(-0.1)), TypeError))

# ---------------------------------------------------------------------------
# 4. The Lanczos string dispatch — the SAME name `Lanczos` selects TWO different
#    algorithm families: the `method` string ('Gnd'/'ER', naive/restarted) and
#    the `which` string ('SA'/'LA'/'LM', ARPACK). The naming does not make this
#    split (nor the Tensor-vs-UniTensor split) obvious (finding UT-K2/UT-K4).
# ---------------------------------------------------------------------------
H = spd(6, 42)
op = DenseOp(cytnx.from_numpy(H.copy()))
gnd = float(np.linalg.eigvalsh(H).min())          # reference via dense Eigh
rng = np.random.default_rng(7)

def rvec(n=6):
    return cytnx.from_numpy(rng.normal(size=n).copy())

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    r_method = la.Lanczos(op, rvec(), method="Gnd")     # method-string family
    r_which = la.Lanczos(op, rvec(), which="SA")        # ARPACK which-string family

report("Lanczos(op, v, method='Gnd') dispatches the naive-Lanczos ground-state "
       "path and returns [eigval, eigvec]",
       isinstance(r_method, list) and len(r_method) == 2)
report("Lanczos(op, v, which='SA') dispatches the ARPACK path and returns "
       "[eigval, eigvec]", isinstance(r_which, list) and len(r_which) == 2)

# 4a. ground-state correctness: BOTH paths match a direct Eigh of the dense H.
report("Lanczos method='Gnd' ground-state eigenvalue matches dense Eigh(H).min()",
       np.isclose(float(r_method[0].item()), gnd, atol=1e-8))
report("Lanczos which='SA' ground-state eigenvalue matches dense Eigh(H).min()",
       np.isclose(float(r_which[0].item()), gnd, atol=1e-8))

# 4b. cross-verify against cytnx's own dense Eigh (not just numpy).
He = cytnx.linalg.Eigh(cytnx.from_numpy(H.copy()))[0].numpy()
report("dense cytnx.linalg.Eigh(H).min() agrees with the Krylov ground state",
       np.isclose(He.min(), gnd, atol=1e-10))

# 4c. the two dispatch families accept DIFFERENT keyword sets: `method` rejects
#     `which` and vice-versa (they are distinct pybind overloads).
report("the method-string overload accepts is_row/max_krydim kwargs (naive/ER "
       "family)", "max_krydim" in la.Lanczos.__doc__ and "is_row" in la.Lanczos.__doc__)
report("the which-string overload accepts an ncv kwarg (ARPACK family)",
       "ncv" in la.Lanczos.__doc__)

# ---------------------------------------------------------------------------
# 5. UniTensor ground-state path — Lanczos on a UniTensor start vector returns
#    UniTensors and the same ground-state eigenvalue (Lanczos_Gnd_Ut being
#    absent, this is the reachable UniTensor route).
# ---------------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ut0 = cytnx.UniTensor(rvec())
    r_ut = la.Lanczos(op, ut0, which="SA")
report("Lanczos on a UniTensor start vector returns a list of UniTensors "
       "(the reachable UniTensor route; Lanczos_Gnd_Ut is absent)",
       isinstance(r_ut, list) and all(isinstance(x, UT) for x in r_ut))
report("Lanczos UniTensor path yields the same ground-state eigenvalue as the "
       "dense Eigh", np.isclose(float(r_ut[0].item()), gnd, atol=1e-8))

# 5a. the method='Gnd' family also accepts a UniTensor start vector (not just
#     the which='SA' ARPACK family above) — the route this doc cites for the
#     1.1.0 UniTensor ground state.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    r_ut_gnd = la.Lanczos(op, ut0, method="Gnd")
report("Lanczos(Hop, ut_v0, method='Gnd') also runs on a UniTensor start "
       "vector and matches the same ground-state eigenvalue",
       isinstance(r_ut_gnd, list) and all(isinstance(x, UT) for x in r_ut_gnd)
       and np.isclose(float(r_ut_gnd[0].item()), gnd, atol=1e-8))

# ---------------------------------------------------------------------------
# 6. Lanczos_Exp — action of the matrix exponential exp(tau*H) @ v (UniTensor).
# ---------------------------------------------------------------------------
H5 = spd(5, 1)
op5 = DenseOp(cytnx.from_numpy(H5.copy()))
v0 = np.array([1., 0., 0., 0., 0.])
tau = -0.05
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ex = la.Lanczos_Exp(op5, cytnx.UniTensor(cytnx.from_numpy(v0.copy())),
                        cytnx.Scalar(tau))
report("Lanczos_Exp returns a single UniTensor (the exp(tau*H)@v action)",
       isinstance(ex, UT))
report("Lanczos_Exp(op, v, tau) == exp(tau*H) @ v (matches dense expm)",
       np.allclose(ex.get_block().numpy().ravel(), expm(tau * H5) @ v0, atol=1e-6))

# ---------------------------------------------------------------------------
# 7. Arnoldi (general, non-Hermitian) — reachable for both Tensor and UniTensor;
#    which selects the target eigenvalue region.
# ---------------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    r_arn = la.Arnoldi(op, rvec(), which="SR")
report("Arnoldi(op, v, which='SR') runs and returns eigen-results",
       isinstance(r_arn, list) and len(r_arn) >= 1)
report("Arnoldi smallest-real eigenvalue matches the dense ground state on a "
       "Hermitian operator", np.isclose(float(np.real(r_arn[0].item())), gnd, atol=1e-6))

# 7a. which='SM' ('smallest magnitude') is NOT one of the ARPACK codes this
#     binding accepts (its C++ core only recognizes LM/LR/LI/SR/SI) — it raises
#     a RuntimeError from the underlying ARPACK argument check.
report("Arnoldi(op, v, which='SM') is unsupported and raises RuntimeError",
       raises(lambda: la.Arnoldi(op, rvec(), which="SM"), RuntimeError))

print("UniTensor 09 probe ok")
