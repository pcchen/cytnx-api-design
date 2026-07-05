"""Behavioral probe for UniTensor category 03 — metadata accessors, verified
against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/UniTensor/03-metadata-accessors.md is
backed by a report(...) assertion here.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_03_metadata.py
"""
import os
import sys
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

UT = cytnx.UniTensor
Type, Device = cytnx.Type, cytnx.Device


@contextlib.contextmanager
def _mute_c_stdout():
    """Swallow C++-level stdout (cytnx prints the whole tensor via std::cout
    when a pybind overload fails) so keyword-rejection checks stay quiet."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def sym_bonds():
    """A fresh (BD_IN, BD_OUT) U(1) bond pair — one block per matching qnum."""
    bi = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [cytnx.Symmetry.U1()])
    bo = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [cytnx.Symmetry.U1()])
    return [bi, bo]


def mk_block():
    return UT(sym_bonds(), rowrank=1)


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# Shape / rank family on a small symmetric (Block) tensor
# =========================================================================
B = mk_block()

# M1: rank / rowrank / Nblocks / shape report the tensor's structure.
report("rank() of the 2-leg block tensor is 2", B.rank() == 2)
report("rowrank() is 1 (one leg in the row/bra space)", B.rowrank() == 1)
report("Nblocks() is 2 (one block per matching U(1) charge)", B.Nblocks() == 2)
report("shape() is [2, 2]", B.shape() == [2, 2])
report("labels() defaults to ['0', '1']", B.labels() == ["0", "1"])
report("get_index('1') resolves label '1' to leg index 1", B.get_index("1") == 1)

# =========================================================================
# bond / bond_ / bonds — copy vs. view (B1 / N-underscore)
# =========================================================================
# M2: bond_(i) hands back a VIEW that shares the parent's internal Bond, so an
# in-place mutation (redirect_ flips the bond direction) is visible on the
# parent; bond(i) hands back an independent CLONE.
report("bond_(i) is a VIEW — redirect_ on it flips the parent's bond direction",
       returns_view(mk_block, lambda u: u.bond_(0),
                    lambda b: b.redirect_(), lambda u: str(u.bond(0).type())))
report("bond(i) is a COPY — redirect_ on it leaves the parent unchanged",
       not returns_view(mk_block, lambda u: u.bond(0),
                        lambda b: b.redirect_(), lambda u: str(u.bond(0).type())))
# M3: bonds() copies the *container* (a Python list) but each element still
# shares the parent's Bond impl, so an element mutation reaches the parent.
report("bonds() copies the list but its Bond elements share the parent's impl",
       returns_view(mk_block, lambda u: u.bonds()[0],
                    lambda b: b.redirect_(), lambda u: str(u.bond(0).type())))

# =========================================================================
# is_* predicates — tagged/block vs. dense/untagged
# =========================================================================
D = UT(cytnx.zeros([2, 3]))  # dense, untagged
report("is_tag() is True on the symmetric (tagged) tensor", B.is_tag() is True)
report("is_tag() is False on the dense (untagged) tensor", D.is_tag() is False)
report("is_blockform() is True on the block tensor", B.is_blockform() is True)
report("is_blockform() is False on the dense tensor", D.is_blockform() is False)
report("is_diag() is False (dense, not stored diagonal-only)", D.is_diag() is False)
report("is_contiguous() is True for a freshly-built dense tensor",
       D.is_contiguous() is True)
report("is_braket_form() is a bool predicate", isinstance(B.is_braket_form(), bool))

# =========================================================================
# dtype / device / uten_type / name accessors (dense tensor)
# =========================================================================
report("dtype() returns the int type code Type.Double (3)", D.dtype() == int(Type.Double))
report("dtype_str() names the element type", "Double" in D.dtype_str())
report("device() returns the int device code Device.cpu (-1)", D.device() == int(Device.cpu))
report("device_str() names the device", "CPU" in D.device_str())
report("uten_type() is 0 (Dense) for a dense tensor, 2 (Block) for a block tensor",
       D.uten_type() == 0 and B.uten_type() == 2)
report("uten_type_str() is 'Dense' / 'Block'",
       D.uten_type_str() == "Dense" and B.uten_type_str() == "Block")
D.set_name("foo")
report("name() reflects a set_name('foo')", D.name() == "foo")

# =========================================================================
# same_data — self is True, a clone is False
# =========================================================================
report("same_data(self) is True (a tensor shares data with itself)",
       B.same_data(B) is True)
report("same_data(clone) is False (clone() allocates independent storage)",
       B.same_data(B.clone()) is False)

# =========================================================================
# symmetric-only accessors on the block tensor
# =========================================================================
report("syms() returns the tensor's symmetry list (one U(1))", len(B.syms()) == 1)
report("get_qindices(0) returns the per-block qnum-index list [0, 0]",
       B.get_qindices(0) == [0, 0])

# M4: signflip is a BlockFermionic-only accessor: it returns list[bool] on a
# fermionic tensor and RAISES on a bosonic one.
F = UT([cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [cytnx.Symmetry.FermionParity()]),
        cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [cytnx.Symmetry.FermionParity()])])
report("signflip() returns list[bool] on a BlockFermionic tensor",
       F.signflip() == [False, False])
raised = False
try:
    with _mute_c_stdout():
        B.signflip()
except RuntimeError:
    raised = True
report("signflip() RAISES on a bosonic (non-fermionic) block tensor", raised)

# =========================================================================
# getTotalQnums / get_blocks_qnums — bound but non-functional (UT-M6)
# The C++ header marks both "@note This API just have not support."
# =========================================================================
for label, thunk in [
    ("getTotalQnums on a block tensor", lambda: mk_block().getTotalQnums()),
    ("getTotalQnums on a dense tensor", lambda: D.getTotalQnums()),
    ("get_blocks_qnums on a block tensor", lambda: mk_block().get_blocks_qnums()),
    ("get_blocks_qnums on a dense tensor", lambda: D.get_blocks_qnums()),
]:
    raised = False
    try:
        with _mute_c_stdout():
            thunk()
    except RuntimeError:
        raised = True
    report(f"{label} RAISES (bound but 'not supported' on every tensor type)", raised)

# =========================================================================
# Erased argument names (arg0) block keyword calls (PC1)
# =========================================================================
def _rejects_kw(call):
    try:
        with _mute_c_stdout():
            call()
        return False
    except TypeError:
        return True

report("same_data's arg is erased to arg0 — same_data(rhs=...) is REJECTED (PC1)",
       _rejects_kw(lambda: B.same_data(rhs=B)))
report("get_qindices's arg is erased to arg0 — get_qindices(bidx=...) is REJECTED (PC1)",
       _rejects_kw(lambda: B.get_qindices(bidx=0)))
report("get_index's arg is erased to arg0 — get_index(label=...) is REJECTED (PC1/UT-M5)",
       _rejects_kw(lambda: B.get_index(label="0")))

print("UniTensor 03 probe ok")
