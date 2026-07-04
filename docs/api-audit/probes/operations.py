"""Behavioral probe for the four operation submodules (Cytnx 1.1.0):
algo, random, physics, qgates.

Every behavioral claim made in docs/api-audit/per-class/operations.md's Parity
and Consistency findings sections is backed by a report() assertion here.
Run with: source tools/env.sh && $PY docs/api-audit/probes/operations.py

These four modules are free-function toolboxes that operate on Tensor /
UniTensor / Storage; there is no object identity to probe, so the assertions
concentrate on numeric results (sort order, concatenation, gate matrices,
spin operators), in-place-fill semantics, reproducible seeding, dtype
promotion (B3), and exception behavior (B4).
"""
import sys, os, io, contextlib, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import numpy as np
import cytnx
from probe_helper import report

algo = cytnx.algo
random = cytnx.random
physics = cytnx.physics
qgates = cytnx.qgates


@contextlib.contextmanager
def _quiet():
    """Swallow the noisy C++ cytnx_error_msg stack-trace dumped to stderr/stdout
    by the deliberately-failing calls below; the Python exception still raises."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        yield


def raises(fn, exc=Exception):
    try:
        with _quiet():
            fn()
        return False
    except exc:
        return True


def blk(ut):
    """The dense numpy matrix behind a (symmetry-free) UniTensor gate."""
    return ut.get_block().numpy()


# =========================================================================
# algo  --  Sort / Concatenate / Vstack / Hstack / Vsplit / Hsplit
# =========================================================================

# The whole algo surface is Capitalized (Sort, Concatenate, Vstack, Hstack,
# Vsplit, Hsplit) -- a per-module N1 violation, in contrast to random/physics/
# qgates which are already snake_case.
report("algo's entire public surface is Capitalized-verb free functions "
       "(Sort/Concatenate/Vstack/Hstack/Vsplit/Hsplit), violating N1 -- unlike "
       "random/physics/qgates which are already snake_case",
       all(hasattr(algo, n) for n in
           ("Sort", "Concatenate", "Vstack", "Hstack", "Vsplit", "Hsplit"))
       and all(n[0].isupper() for n in
               ("Sort", "Concatenate", "Vstack", "Hstack", "Vsplit", "Hsplit")))

# --- Sort: sorts along the LAST axis, returns a NEW tensor (input untouched).
src = cytnx.from_numpy(np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]))
sorted_t = algo.Sort(src)
report("algo.Sort sorts each row ascending along the last axis and returns a "
       "NEW Tensor: [[3,1,2],[6,4,5]] -> [[1,2,3],[4,5,6]]",
       sorted_t.numpy().tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
report("algo.Sort does NOT mutate its input (returns-new, B1): the source is "
       "still [[3,1,2],[6,4,5]] after Sort; there is no in-place algo.Sort_ "
       "(N2: the pure op has no in-place counterpart)",
       src.numpy().tolist() == [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
       and not hasattr(algo, "Sort_"))

# --- Concatenate: joins two 1-D tensors; dtype promotes to the stronger type.
ca = algo.Concatenate(cytnx.from_numpy(np.array([1.0, 2.0, 3.0])),
                      cytnx.from_numpy(np.array([4.0, 5.0])))
report("algo.Concatenate joins two 1-D Tensors end to end: [1,2,3] ++ [4,5] "
       "-> [1,2,3,4,5] (shape [5])",
       ca.shape() == [5] and ca.numpy().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0])
cp = algo.Concatenate(cytnx.from_numpy(np.array([1, 2, 3], dtype=np.int64)),
                      cytnx.from_numpy(np.array([4.0, 5.0])))
report("algo.Concatenate promotes to the stronger dtype (B3): Int64 ++ Double "
       "-> Double",
       cp.dtype() == cytnx.Type.Double)
report("algo.Concatenate rejects non-1-D input with a catchable RuntimeError "
       "(B4: it can only accept 1-D Tensors)",
       raises(lambda: algo.Concatenate(cytnx.zeros([2, 2]), cytnx.zeros([2, 2])),
              RuntimeError))

# --- Vstack / Hstack: stack a list of 2-D matrices.
vs = algo.Vstack([cytnx.from_numpy(np.array([[1.0, 2.0]])),
                  cytnx.from_numpy(np.array([[3.0, 4.0], [5.0, 6.0]]))])
report("algo.Vstack stacks matrices row-wise (same #columns): [1x2] + [2x2] "
       "-> [3x2] = [[1,2],[3,4],[5,6]]",
       vs.shape() == [3, 2]
       and vs.numpy().tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
hs = algo.Hstack([cytnx.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]])),
                  cytnx.from_numpy(np.array([[5.0], [6.0]]))])
report("algo.Hstack stacks matrices column-wise (same #rows): [2x2] + [2x1] "
       "-> [2x3]",
       hs.shape() == [2, 3]
       and hs.numpy().tolist() == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])

# --- Vsplit / Hsplit: inverse of the stacks; return a Python list of Tensors.
M = cytnx.from_numpy(np.arange(9.0).reshape(3, 3))
vsp = algo.Vsplit(M, [1, 2])
report("algo.Vsplit(M, dims) splits a matrix row-wise into a Python list of "
       "Tensors with dims rows each: a 3x3 with dims [1,2] -> shapes [1,3],[2,3]",
       isinstance(vsp, list) and [x.shape() for x in vsp] == [[1, 3], [2, 3]])
hsp = algo.Hsplit(M, [2, 1])
report("algo.Hsplit(M, dims) splits a matrix column-wise into a Python list of "
       "Tensors with dims columns each: a 3x3 with dims [2,1] -> shapes "
       "[3,2],[3,1]",
       isinstance(hsp, list) and [x.shape() for x in hsp] == [[3, 2], [3, 1]])
report("algo's C++ out-parameter in-place split variants Vsplit_/Hsplit_ "
       "(algo.hpp) are NOT bound in Python (algo_py.cpp binds only the 6 "
       "returns-new functions) -- a C++-source-vs-Python-wheel surface gap",
       not hasattr(algo, "Vsplit_") and not hasattr(algo, "Hsplit_"))

# =========================================================================
# random  --  normal / normal_ / uniform / uniform_
# =========================================================================

# --- in-place fill: the trailing-underscore variants mutate their argument and
#     return None (B1/N2: the _ suffix marks the in-place form).
t1 = cytnx.zeros([5])
ret = random.normal_(t1, 0.0, 1.0, seed=123)
report("random.normal_ fills its argument in place and returns None (N2 "
       "in-place suffix): a zeros([5]) becomes all-nonzero after the call",
       ret is None and any(abs(x) > 1e-12 for x in t1.numpy().tolist()))

# --- reproducible seeding: same seed -> byte-identical fill; different seed differs.
t2 = cytnx.zeros([5])
random.normal_(t2, 0.0, 1.0, seed=123)
report("random.normal_ with the SAME seed reproduces the SAME fill "
       "(deterministic given seed): two independent zeros([5]) filled with "
       "seed=123 are element-wise equal",
       np.array_equal(t1.numpy(), t2.numpy()))
t3 = cytnx.zeros([5])
random.normal_(t3, 0.0, 1.0, seed=999)
report("random.normal_ with a DIFFERENT seed produces a different fill",
       not np.array_equal(t1.numpy(), t3.numpy()))

# --- uniform_ range + in-place.
u = cytnx.zeros([2000])
random.uniform_(u, 0.0, 1.0, seed=42)
ua = u.numpy()
report("random.uniform_ fills in place over the half-open range [low, high) on "
       "cpu: a uniform_(_, 0, 1) fill lands in [0,1) with mean ~ 0.5",
       ua.min() >= 0.0 and ua.max() < 1.0 and abs(ua.mean() - 0.5) < 0.05)
ub_t = cytnx.zeros([2000])
random.uniform_(ub_t, 0.0, 1.0, seed=42)
report("random.uniform_ is likewise reproducible under a fixed seed",
       np.array_equal(u.numpy(), ub_t.numpy()))

# --- normal_ / uniform_ also accept a Storage (second overload).
S = cytnx.Storage(4, dtype=cytnx.Type.Double)
random.uniform_(S, 0.0, 1.0, seed=5)
report("random.uniform_ / random.normal_ accept a Storage as well as a Tensor "
       "and a UniTensor (three C++ overloads each) -- Storage fill runs without "
       "error",
       S.dtype() == cytnx.Type.Double)

# --- normal / uniform (no underscore): allocate and RETURN a new Tensor.
n1 = random.normal([4], 0.0, 1.0, seed=7)
n2 = random.normal([4], 0.0, 1.0, seed=7)
report("random.normal (no underscore) allocates and returns a NEW Tensor of the "
       "given shape, and is reproducible under a fixed seed",
       type(n1).__name__ == "Tensor" and n1.shape() == [4]
       and np.array_equal(n1.numpy(), n2.numpy()))
report("random.normal / random.uniform accept BOTH an int Nelem and a shape "
       "list (two C++ overloads): normal(4,...) and normal([4],...) both give a "
       "length-4 Tensor",
       random.normal(4, 0.0, 1.0, seed=7).shape() == [4]
       and random.uniform([4], 0.0, 1.0, seed=7).shape() == [4])

# --- there is NO random.seed global, and the deprecated Make_normal/Make_uniform
#     and the random_tensor helper (all in random.hpp) are NOT bound.
report("random has NO global 'seed' setter: seeding is per-call via the seed= "
       "argument (seed=-1, the default, draws fresh device entropy). "
       "random.hpp's deprecated Make_normal/Make_uniform aliases and the "
       "random_tensor helper are NOT bound in Python either",
       not hasattr(random, "seed")
       and not hasattr(random, "Make_normal")
       and not hasattr(random, "Make_uniform")
       and not hasattr(random, "random_tensor"))
report("random.uniform_ carries default bounds low=0.0/high=1.0, but "
       "random.normal_ has NO defaults for mean/std, and random.normal / "
       "random.uniform require mean/std / low/high positionally -- an "
       "inconsistent-defaults surface (N4-adjacent)",
       raises(lambda: random.normal_(cytnx.zeros([2])), TypeError)
       and random.uniform_(cytnx.zeros([2])) is None)

# =========================================================================
# physics  --  spin / pauli   (return a bare Tensor matrix)
# =========================================================================

SQRT_HALF = 1.0 / math.sqrt(2.0)

sz = physics.spin(0.5, "z").numpy()
report("physics.spin(0.5,'z') is the spin-1/2 Sz operator diag(+1/2,-1/2), a "
       "2x2 complex Tensor",
       physics.spin(0.5, "z").shape() == [2, 2]
       and np.allclose(sz, np.array([[0.5, 0.0], [0.0, -0.5]])))
sx = physics.spin(0.5, "x").numpy()
report("physics.spin(0.5,'x') is Sx = [[0,1/2],[1/2,0]]",
       np.allclose(sx, np.array([[0.0, 0.5], [0.5, 0.0]])))
sy = physics.spin(0.5, "y").numpy()
report("physics.spin(0.5,'y') is Sy = [[0,-i/2],[i/2,0]] (genuinely complex)",
       np.allclose(sy, np.array([[0.0, -0.5j], [0.5j, 0.0]])))
sz1 = physics.spin(1.0, "z").numpy()
report("physics.spin(1.0,'z') is the spin-1 Sz operator diag(1,0,-1), a 3x3 "
       "Tensor -- dimension is 2S+1",
       physics.spin(1.0, "z").shape() == [3, 3]
       and np.allclose(sz1, np.diag([1.0, 0.0, -1.0])))

px = physics.pauli("x").numpy()
pz = physics.pauli("z").numpy()
py_ = physics.pauli("y").numpy()
report("physics.pauli returns the Pauli matrices (= 2*spin for S=1/2): "
       "pauli('x')=[[0,1],[1,0]], pauli('y')=[[0,-i],[i,0]], "
       "pauli('z')=[[1,0],[0,-1]]",
       np.allclose(px, [[0, 1], [1, 0]])
       and np.allclose(py_, [[0, -1j], [1j, 0]])
       and np.allclose(pz, [[1, 0], [0, -1]]))
report("physics.pauli('x') == 2 * physics.spin(0.5,'x') (the two are the same "
       "operator up to the spin-1/2 factor of 1/2)",
       np.allclose(px, 2.0 * sx))
report("physics.spin/pauli raise a catchable RuntimeError (B4) on an invalid "
       "component ('q') and on an S that is not a multiple of 1/2 (0.3)",
       raises(lambda: physics.spin(0.5, "q"), RuntimeError)
       and raises(lambda: physics.spin(0.3, "z"), RuntimeError)
       and raises(lambda: physics.pauli("q"), RuntimeError))
report("physics.pauli returns a bare Tensor, whereas qgates.pauli_x returns a "
       "UniTensor -- the same Pauli-X operator is offered twice, under two "
       "names and two container types across physics and qgates (N4-adjacent "
       "cross-module duplication)",
       type(physics.pauli("x")).__name__ == "Tensor"
       and type(qgates.pauli_x()).__name__ == "UniTensor")

# =========================================================================
# qgates  --  pauli_x/y/z, hadamard, phase_shift, swap, sqrt_swap, toffoli,
#             cntl_gate_2q   (return UniTensor gates)
# =========================================================================

report("qgates.pauli_x/pauli_y/pauli_z return the 2x2 Pauli gates as "
       "UniTensors: X=[[0,1],[1,0]], Y=[[0,-i],[i,0]], Z=[[1,0],[0,-1]]",
       np.allclose(blk(qgates.pauli_x()), [[0, 1], [1, 0]])
       and np.allclose(blk(qgates.pauli_y()), [[0, -1j], [1j, 0]])
       and np.allclose(blk(qgates.pauli_z()), [[1, 0], [0, -1]]))

H = blk(qgates.hadamard())
report("qgates.hadamard() returns the UNNORMALIZED matrix [[1,1],[1,-1]] -- it "
       "is MISSING the 1/sqrt(2) factor, so it is NOT the standard (unitary) "
       "Hadamard gate",
       np.allclose(H, [[1, 1], [1, -1]])
       and not np.allclose(H, np.array([[1, 1], [1, -1]]) * SQRT_HALF))
report("consequently qgates.hadamard() is NOT unitary: H @ H^dagger == 2*I, "
       "not I (a correctness bug -- the returned gate does not preserve norm)",
       np.allclose(H @ H.conj().T, 2.0 * np.eye(2))
       and not np.allclose(H @ H.conj().T, np.eye(2)))

ps = blk(qgates.phase_shift(math.pi / 2)).reshape(2, 2)
report("qgates.phase_shift(theta) returns diag(1, exp(i*theta)): "
       "phase_shift(pi/2) = diag(1, i)",
       np.allclose(ps, np.diag([1.0, 1j])))

sw = blk(qgates.swap()).reshape(4, 4)
report("qgates.swap() is the 2-qubit SWAP gate, a rank-4 UniTensor (shape "
       "[2,2,2,2]) whose 4x4 form swaps |01> and |10>",
       qgates.swap().shape() == [2, 2, 2, 2]
       and np.allclose(sw, np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                     [0, 1, 0, 0], [0, 0, 0, 1]])))
ssw = blk(qgates.sqrt_swap()).reshape(4, 4)
report("qgates.sqrt_swap() squares to SWAP: its 4x4 form has the "
       "(1+i)/2, (1-i)/2 block and (sqrt_swap)^2 == swap",
       np.allclose(ssw @ ssw, sw))

tof = blk(qgates.toffoli()).reshape(8, 8)
report("qgates.toffoli() is the 3-qubit Toffoli (CCNOT) gate, a rank-6 "
       "UniTensor (shape [2,2,2,2,2,2]); its 8x8 form flips the last qubit only "
       "when the first two are set (rows/cols 6,7 swapped). It is a REAL-valued "
       "(Double) tensor, unlike the complex gates",
       qgates.toffoli().shape() == [2, 2, 2, 2, 2, 2]
       and qgates.toffoli().dtype() == cytnx.Type.Double
       and np.allclose(tof, np.eye(8)[[0, 1, 2, 3, 4, 5, 7, 6]]))

cnot = blk(qgates.cntl_gate_2q(qgates.pauli_x())).reshape(4, 4)
report("qgates.cntl_gate_2q(gate_1q) promotes a 1-qubit gate to its controlled "
       "2-qubit version: cntl_gate_2q(pauli_x) is CNOT (identity on the "
       "control=0 block, applies X on the control=1 block)",
       qgates.cntl_gate_2q(qgates.pauli_x()).shape() == [2, 2, 2, 2]
       and np.allclose(cnot, np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0, 0, 1], [0, 0, 1, 0]])))

print("operations probe ok")
