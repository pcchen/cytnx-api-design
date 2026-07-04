"""Behavioral probe for the Network / LinOp classes and the ncon free function
(Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/network.md's Parity and
Consistency findings sections is backed by a report() assertion here. Run with:
    source tools/env.sh && $PY docs/api-audit/probes/network.py

Static signatures are ground-truthed against cytnx_src/include/Network.hpp,
cytnx_src/include/LinOp.hpp, cytnx_src/include/ncon.hpp, the pybind bindings
cytnx_src/pybind/{network,linop,ncon}_py.cpp, the C++ bodies
cytnx_src/src/{RegularNetwork,LinOp,ncon}.cpp, and the Python-side augmentation
cytnx_src/cytnx/Network_conti.py (Network.Diagram).
"""
import sys, os, subprocess, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

UT = cytnx.UniTensor
Type = cytnx.Type
Device = cytnx.Device


def AB():
    """A fresh (2x3, 3x4) labelled rank-2 pair sharing a contraction leg 'k'."""
    A = UT(cytnx.arange(6).reshape(2, 3)); A.set_labels(["i", "k"])
    B = UT(cytnx.arange(12).reshape(3, 4)); B.set_labels(["k", "j"])
    return A, B


# =========================================================================
# ncon -- index convention (positive = contracted bond, negative = open leg)
# =========================================================================

A, B = AB()
An, Bn = A.get_block().numpy(), B.get_block().numpy()

r = cytnx.ncon([A, B], [[-1, 1], [1, -2]])
report("ncon index convention: a POSITIVE integer marks a bond contracted "
       "(summed) between the two tensors that share it, a NEGATIVE integer "
       "marks an open output leg -- ncon([A,B],[[-1,1],[1,-2]]) == A @ B "
       "(hand matrix product)",
       r.shape() == [2, 4] and np.allclose(r.get_block().numpy(), An @ Bn))

r_sw = cytnx.ncon([A, B], [[-2, 1], [1, -1]])
report("ncon output legs are ordered by the negative labels ascending in "
       "magnitude: -1 becomes the first output axis, -2 the second. Swapping to "
       "[[-2,1],[1,-1]] puts B's free leg (-1) first, so the result is (A @ B).T "
       "with shape [4,2]",
       r_sw.shape() == [4, 2] and np.allclose(r_sw.get_block().numpy(), (An @ Bn).T))

A3 = UT(cytnx.arange(6).reshape(2, 3))
B3 = UT(cytnx.arange(12).reshape(3, 4))
C3 = UT(cytnx.arange(8).reshape(4, 2))
A3n, B3n, C3n = (t.get_block().numpy() for t in (A3, B3, C3))
sc = cytnx.ncon([A3, B3, C3], [[1, 2], [2, 3], [3, 1]])
report("ncon fully contracts a 3-tensor chain with all-positive labels into a "
       "scalar (rank-1, shape [1]) equal to trace(A@B@C)",
       sc.shape() == [1]
       and np.allclose(sc.get_block().numpy().reshape(-1)[0], np.trace(A3n @ B3n @ C3n)))

report("ncon does NOT mutate its input tensors: A still equals its original "
       "value after being contracted",
       np.allclose(A.get_block().numpy(), An))

try:
    cytnx.ncon([A, B], [[-1, 5], [1, -2]], check_network=True)
    ncon_check_raised = False
except RuntimeError:
    ncon_check_raised = True
report("ncon(check_network=True) raises RuntimeError when a positive bond label "
       "does not appear exactly twice (here 5 appears once) -- B4, a malformed "
       "network is an exception, not a silent/garbage result. NOTE check_network "
       "DEFAULTS to False, so this validation is opt-in",
       ncon_check_raised)

# =========================================================================
# Network -- the reliable build path: FromString + PutUniTensor + Launch
# =========================================================================

net = cytnx.Network()
net.FromString(["A: i,k", "B: k,j", "TOUT: i;j", "ORDER: (A,B)"])
report("Network.FromString(lines) loads a network skeleton from a list of "
       "'name: labels' strings (';' splits row/col rank; TOUT/ORDER are reserved "
       "lines): isLoad() is True once a skeleton is loaded",
       net.isLoad() is True)

report("isAllset() is False on a freshly-loaded skeleton whose tensor slots are "
       "still Void (no UniTensor placed yet)",
       net.isAllset() is False)

net.PutUniTensor("A", A)
net.PutUniTensor("B", B)
report("PutUniTensor(name, utensor) fills a named slot; once every slot is "
       "filled isAllset() is True",
       net.isAllset() is True)

out = net.Launch()
report("Launch() contracts the fully-set network and returns the UniTensor "
       "result: FromString('A: i,k'/'B: k,j'/'TOUT: i;j') + PutUniTensor + "
       "Launch() == A @ B",
       out.shape() == [2, 4] and np.allclose(out.get_block().numpy(), An @ Bn))

# by-index PutUniTensor
neti = cytnx.Network()
neti.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
neti.PutUniTensor(0, A); neti.PutUniTensor(1, B)
report("PutUniTensor(idx, utensor) is a positional overload of PutUniTensor: "
       "placing by slot index 0/1 gives the same A @ B result",
       np.allclose(neti.Launch().get_block().numpy(), An @ Bn))

# PutUniTensors (plural)
netp = cytnx.Network()
netp.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
netp.PutUniTensors(["A", "B"], [A, B])
report("PutUniTensors(names, utensors) places several tensors in one call, "
       "equivalent to repeated PutUniTensor",
       np.allclose(netp.Launch().get_block().numpy(), An @ Bn))

# PutUniTensor label_order permutes a COPY -- input untouched (B1/B2)
netl = cytnx.Network()
netl.FromString(["A: k,i", "B: k,j", "TOUT: i;j"])
A_labels_before = A.labels()
netl.PutUniTensor("A", A, ["k", "i"])
netl.PutUniTensor("B", B)
report("PutUniTensor(name, utensor, label_order) permutes the tensor to the "
       "given leg order before placing it (C++ does utensor.permute(label_order), "
       "which returns a NEW object) -- so the caller's tensor labels are NOT "
       "mutated (B1)",
       A.labels() == A_labels_before
       and np.allclose(netl.Launch().get_block().numpy(), An @ Bn))

# construct -- the low-level programmatic builder that ncon itself uses
netc = cytnx.Network()
netc.construct(["A", "B"], [["i", "k"], ["k", "j"]], ["i", "j"], 1, "", False)
netc.PutUniTensor("A", A); netc.PutUniTensor("B", B)
report("construct(alias, labels, outlabel, outrk, order, optim) builds the same "
       "network programmatically (this is the primitive ncon() calls internally): "
       "construct + PutUniTensor + Launch == A @ B",
       np.allclose(netc.Launch().get_block().numpy(), An @ Bn))

# setOrder / getOrder
neto = cytnx.Network()
neto.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
neto.setOrder(False, "(A,B)")
report("setOrder(optimal, contract_order) records a contraction order string; "
       "getOrder() reads it back verbatim ('(A,B)')",
       neto.getOrder() == "(A,B)")

# RmUniTensor
netr = cytnx.Network()
netr.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
netr.PutUniTensor("A", A); netr.PutUniTensor("B", B)
netr.RmUniTensor("A")
report("RmUniTensor(name) empties a placed slot back to Void, so isAllset() "
       "reverts to False",
       netr.isAllset() is False)

# RmUniTensors (plural)
netr2 = cytnx.Network()
netr2.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
netr2.PutUniTensors(["A", "B"], [A, B])
netr2.RmUniTensors(["A", "B"])
report("RmUniTensors(names) clears several placed slots in one call",
       netr2.isAllset() is False)

# clear
netcl = cytnx.Network()
netcl.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
netcl.clear()
report("clear() wipes the whole skeleton (names/labels/tensors), so isLoad() "
       "goes back to False",
       netcl.isLoad() is False)

# Savefile / Fromfile round-trip + the file-path constructor
tmpd = tempfile.mkdtemp()
fpath = os.path.join(tmpd, "probe_net")
nets = cytnx.Network()
nets.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
nets.Savefile(fpath)
report("Savefile(fname) writes the skeleton to '<fname>.net' (the '.net' suffix "
       "is appended by Cytnx)",
       os.path.exists(fpath + ".net"))

net_from_file = cytnx.Network(fpath + ".net")
report("Network(fname) (the file-path constructor) == Fromfile: it loads a "
       "'.net' file at construction, giving isLoad() True",
       net_from_file.isLoad() is True)

# =========================================================================
# Network -- clone() reference/copy semantics (a real defect)
# =========================================================================

netfull = cytnx.Network()
netfull.FromString(["A: i,k", "B: k,j", "TOUT: i;j"])
netfull.PutUniTensor("A", A); netfull.PutUniTensor("B", B)
cl = netfull.clone()
report("clone() copies only the network SKELETON (RegularNetwork::clone copies "
       "name2pos/CtTree/names/label_arr/TOUT_* but NOT the placed `tensors` "
       "vector): the clone reports isLoad() False (its tensor list is empty) "
       "while the original is still isLoad() True",
       netfull.isLoad() is True and cl.isLoad() is False)

report("...but the clone's isAllset() returns a MISLEADING True: isAllset() "
       "loops over the (now empty) tensors vector and returns True vacuously, "
       "even though nothing is actually placed -- an internal inconsistency "
       "(isLoad() and isAllset() disagree on the same clone)",
       cl.isAllset() is True)

# clone().Launch() then fails (cleanly, exit-1 exception -- see subprocess below)

# =========================================================================
# Network.Contract(...) -- the static one-shot factory SEGFAULTS on Launch
# (B4). Run in a subprocess so the crash cannot kill this probe.
# =========================================================================

_contract_snippet = (
    "import cytnx\n"
    "UT=cytnx.UniTensor\n"
    "A=UT(cytnx.arange(6).reshape(2,3)); A.set_labels(['i','k'])\n"
    "B=UT(cytnx.arange(12).reshape(3,4)); B.set_labels(['k','j'])\n"
    "cytnx.Network.Contract([A,B],'i;j',alias=['A','B']).Launch()\n"
)
_cp = subprocess.run([sys.executable, "-c", _contract_snippet],
                     capture_output=True)
report("Network.Contract(utensors, Tout, alias) builds a plan whose Launch() "
       "SEGFAULTS (child exits on SIGSEGV, returncode -11) instead of raising a "
       "catchable exception -- a B4 violation: the one-shot static factory is "
       "unusable in this 1.1.0 build; use FromString/construct + PutUniTensor + "
       "Launch instead",
       _cp.returncode == -11)

_clone_launch_snippet = (
    "import cytnx\n"
    "UT=cytnx.UniTensor\n"
    "A=UT(cytnx.arange(6).reshape(2,3)); A.set_labels(['i','k'])\n"
    "B=UT(cytnx.arange(12).reshape(3,4)); B.set_labels(['k','j'])\n"
    "net=cytnx.Network(); net.FromString(['A: i,k','B: k,j','TOUT: i;j'])\n"
    "net.PutUniTensor('A',A); net.PutUniTensor('B',B)\n"
    "net.clone().Launch()\n"
)
_cl = subprocess.run([sys.executable, "-c", _clone_launch_snippet],
                     capture_output=True)
report("clone().Launch() then fails with a catchable exception (child exits 1, "
       "NOT a segfault) because the clone dropped its tensors -- so isAllset()'s "
       "vacuous True is a genuine lie: the network claims to be ready but cannot "
       "launch",
       _cl.returncode == 1)

# =========================================================================
# Network.Diagram -- a Python-only augmentation (no C++ counterpart)
# =========================================================================

report("Network.Diagram is a Python-only method injected via @add_method in "
       "cytnx/Network_conti.py (it draws the network with graphviz); it has NO "
       "C++ Network member behind it, unlike every other Network method",
       type(cytnx.Network.Diagram).__name__ == "function"
       and cytnx.Network.Diagram.__module__.startswith("cytnx"))

netd = cytnx.Network()
report("Network.Diagram on an un-loaded network does not draw; the underlying "
       "isLoad() guard reports False (Diagram prints an error and exits rather "
       "than raising -- a B4-adjacent smell, but not exercised here to avoid the "
       "graphviz dependency)",
       netd.isLoad() is False)

# =========================================================================
# LinOp -- 'mv_elem' pre-stored sparse matvec
# =========================================================================

op = cytnx.LinOp("mv_elem", 3, dtype=Type.Double, device=Device.cpu)
report("LinOp('mv_elem', nx, dtype, device) constructs a pre-stored-element "
       "operator; nx()/dtype()/device() echo the ctor args (dtype/device are "
       "returned as INTEGER codes, not Type/Device objects)",
       op.nx() == 3 and op.dtype() == int(Type.Double) and op.device() == int(Device.cpu))

M = np.array([[2., 0, 1], [0, 3, 0], [1, 0, 4]])
for i in range(3):
    for j in range(3):
        if M[i, j] != 0:
            op.set_elem(i, j, float(M[i, j]))
x = cytnx.arange(3).astype(Type.Double)  # [0,1,2]
y = op.matvec(x)
report("set_elem(i, j, val) accumulates one sparse matrix entry (out[i] += "
       "val*in[j]); matvec(Tensor) on an 'mv_elem' LinOp applies the stored "
       "matrix: matvec(x) == M @ x",
       np.allclose(y.numpy(), M @ np.array([0, 1, 2.])))

try:
    op.matvec(cytnx.UniTensor(x))
    mvelem_ut_raised = False
except RuntimeError:
    mvelem_ut_raised = True
report("matvec(UniTensor) on an 'mv_elem' LinOp raises RuntimeError -- the "
       "pre-stored-element path accepts only a plain Tensor (B4)",
       mvelem_ut_raised)

# =========================================================================
# LinOp -- 'mv' type: subclass and override matvec (the intended use)
# =========================================================================

class MyOp(cytnx.LinOp):
    def __init__(self):
        cytnx.LinOp.__init__(self, "mv", 3, Type.Double, Device.cpu)
        self.mat = np.array([[1., 2, 0], [0, 1, 2], [2, 0, 1]])

    def matvec(self, tin):
        return cytnx.from_numpy(self.mat @ tin.numpy())

mop = MyOp()
ym = mop.matvec(x)
report("A Python subclass of LinOp('mv', ...) that overrides matvec is "
       "dispatched via pybind11's trampoline (PYBIND11_OVERLOAD): calling "
       "matvec(x) runs the Python override, matvec(x) == mat @ x -- this is the "
       "intended way to feed a custom operator to iterative solvers",
       np.allclose(ym.numpy(), mop.mat @ np.array([0, 1, 2.])))

try:
    cytnx.LinOp("mv", 3).matvec(x)
    base_mv_raised = False
except RuntimeError:
    base_mv_raised = True
report("The BASE LinOp('mv', ...).matvec (not overridden) raises RuntimeError "
       "('required overload matvec before using it') -- B4: an un-overridden "
       "'mv' operator refuses rather than returning garbage",
       base_mv_raised)

# set_device / set_dtype mutate the operator in place
op.set_device(Device.cpu)
op.set_dtype(Type.Float)
report("set_device(id)/set_dtype(code) mutate the operator's device/dtype in "
       "place (setters returning None); dtype() then reflects the new code",
       op.dtype() == int(Type.Float) and op.device() == int(Device.cpu))

try:
    cytnx.LinOp("bogus", 3)
    bogus_raised = False
except RuntimeError:
    bogus_raised = True
report("LinOp(type=...) accepts only 'mv' or 'mv_elem'; any other type string "
       "raises RuntimeError at construction (B4)",
       bogus_raised)

print("network probe ok")
