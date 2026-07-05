"""Behavioral probe for UniTensor category 05 — structure manipulation,
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/UniTensor/05-structure-manipulation.md is
backed by a report(...) assertion here. The raw-C++ side of the
binding-fidelity findings is verified by probes/cpp/UniTensor_05_structure.cpp.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_05_structure.py
"""
import sys, os, inspect, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

UT = cytnx.UniTensor


def mk():
    """A fresh rank-3 Dense (bosonic) UniTensor, contiguous, filled 1."""
    return UT.ones([2, 3, 4])


def mkf():
    """A fresh rank-2 BlockFermionic UniTensor (needed by the fermionic-only
    permute_nosignflip / twist / fermion_twists / apply members)."""
    fp = cytnx.Symmetry.FermionParity()
    Bi = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [fp])
    Bo = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [fp])
    return cytnx.UniTensor([Bi, Bo], labels=["a", "b"])


def val(u, loc):
    return complex(u.at(loc).value)


def capture_cpp_stderr(fn):
    """Capture C-level stderr (fd 2) written by the C++ layer during fn()."""
    sys.stderr.flush()
    saved = os.dup(2)
    tf = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tf.fileno(), 2)
    try:
        try:
            fn()
        except Exception:
            pass
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
    tf.seek(0)
    return tf.read().decode(errors="replace")


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# UT-S1 — permute / permute_ : pure returns a shared-data VIEW; in-place self
# =========================================================================

# permute is PURE: returns a distinct object, leaving the receiver unchanged.
u = mk()
p = u.permute([2, 0, 1])
report("permute returns a distinct object (pure), leaving the receiver's leg "
       "order unchanged",
       p is not u and p.shape() == [4, 2, 3] and u.shape() == [2, 3, 4])

# permute's distinct object SHARES DATA with the source — a view (B2/copy-view).
report("permute returns a shared-data VIEW: mutating the permuted result is "
       "visible through the source (storage is shared, not copied)",
       returns_view(mk, lambda s: s.permute([2, 0, 1]),
                    lambda h: h.__setitem__((0, 0, 0), 9.0),
                    lambda s: val(s, [0, 0, 0])))

# permute_ is IN-PLACE and returns self (chainable).
u = mk()
r = u.permute_([2, 0, 1])
report("permute_ permutes in place and returns self (chainable)",
       r is u and u.shape() == [4, 2, 3])

# =========================================================================
# UT-S2 — reshape / reshape_ : pure returns a shared-data VIEW; in-place self
# =========================================================================

u = mk()
rs = u.reshape(6, 4)
report("reshape returns a distinct object (pure) with the new shape, leaving "
       "the receiver unchanged",
       rs is not u and rs.shape() == [6, 4] and u.shape() == [2, 3, 4])

report("reshape returns a shared-data VIEW: mutating the reshaped result is "
       "visible through the source (storage is shared, not copied)",
       returns_view(mk, lambda s: s.reshape(6, 4),
                    lambda h: h.__setitem__((0, 0), 7.0),
                    lambda s: val(s, [0, 0, 0])))

u = mk()
r = u.reshape_(6, 4)
report("reshape_ reshapes in place and returns self (chainable)",
       r is u and u.shape() == [6, 4])

# UT-S3 — reshape/reshape_ are bound as (*args, **kwargs), erasing the
# positional (new_shape, rowrank) signature that the C++ method declares.
report("reshape is bound as a (*args, **kwargs) pybind lambda — its docstring "
       "signature is `*args, **kwargs`, not the C++ (new_shape, rowrank)",
       "*args" in (UT.reshape.__doc__ or "") and "**kwargs" in (UT.reshape.__doc__ or ""))
no_sig = False
try:
    inspect.signature(UT.reshape)
except ValueError:
    no_sig = True
report("reshape exposes no introspectable positional signature (the *args "
       "binding drops it), so inspect.signature() raises ValueError",
       no_sig)

# =========================================================================
# UT-S4 — contiguous binds via the raw `make_contiguous` shim (leak + naming)
# =========================================================================

# The public `contiguous` is a conti.py wrapper: it short-circuits to `self`
# when already contiguous, else forwards to the raw `make_contiguous` binding.
report("the raw `make_contiguous` shim (the actual C++ contiguous()) leaks into "
       "public dir(UniTensor) — `contiguous` is a conti.py wrapper over it",
       "make_contiguous" in dir(UT) and "contiguous" in dir(UT))

u = mk()
report("contiguous() short-circuits to `self` when the tensor is already "
       "contiguous (conti.py wrapper, no copy)",
       u.contiguous() is u)

u = mk().permute([2, 0, 1])          # permute leaves it non-contiguous
c = u.contiguous()
report("contiguous() on a non-contiguous tensor returns a DISTINCT, contiguous "
       "object via make_contiguous (the receiver stays non-contiguous)",
       c is not u and c.is_contiguous() and not u.is_contiguous())

# contiguous_ is IN-PLACE and returns self.
u = mk().permute([2, 0, 1])
r = u.contiguous_()
report("contiguous_ coalesces storage in place and returns self",
       r is u and u.is_contiguous())

# =========================================================================
# UT-S5 — combineBonds: camelCase + DEPRECATED; combineBond (singular) unbound
# =========================================================================

report("combineBonds (camelCase, the deprecated plural spelling) IS bound, "
       "while the current C++ combineBond (singular) is ABSENT from "
       "dir(UniTensor) — a C++-only binding gap",
       "combineBonds" in dir(UT) and "combineBond" not in dir(UT))

u = mk()
ret = u.combineBonds([0, 1])
report("combineBonds is an in-place mutator whose binding returns None (void), "
       "not the C++ combineBond's UniTensor& — shape [2,3,4] -> [6,4]",
       ret is None and u.shape() == [6, 4])

# The deprecated combineBonds emits a C++ deprecation notice on its by_label path.
notice = capture_cpp_stderr(lambda: mk().combineBonds([0, 1], False, True))
report("combineBonds carries a runtime deprecation notice (its by_label path "
       "prints a '[Deprecated notice]' to stderr from the pybind lambda)",
       "[Deprecated notice]" in notice)

# =========================================================================
# UT-S6 — tag / truncate_ are conti.py wrappers over the leaked raw c* bindings
# =========================================================================

# tag() wraps the raw `ctag` (which leaks); the wrapper re-adds return-self.
report("the raw `ctag` binding (the actual C++ tag()) leaks into public "
       "dir(UniTensor); public `tag` is a conti.py wrapper over it",
       "ctag" in dir(UT) and "tag" in dir(UT))
u = UT.ones([2, 2])
report("tag() tags the tensor in place and returns self (conti.py wrapper "
       "over ctag)",
       u.tag() is u and u.is_tag())

# truncate_ wraps the raw `ctruncate_` (which leaks); wrapper re-adds return-self.
report("the raw `ctruncate_` binding (the actual C++ truncate_()) leaks into "
       "public dir(UniTensor); public `truncate_` is a conti.py wrapper over it",
       "ctruncate_" in dir(UT) and "truncate_" in dir(UT))
u = mk()
r = u.truncate_(0, 1)
report("truncate_ truncates a bond in place and returns self (conti.py wrapper "
       "over ctruncate_): shape [2,3,4] -> [1,3,4]",
       r is u and u.shape() == [1, 3, 4])

# truncate (no underscore) is PURE — a distinct, independent object.
u = mk()
t = u.truncate(0, 1)
report("truncate (no underscore) is pure: returns a distinct object with the "
       "truncated bond, leaving the receiver unchanged",
       t is not u and t.shape() == [1, 3, 4] and u.shape() == [2, 3, 4])

# =========================================================================
# UT-S7 — to_dense / group_basis pairs (pure vs in-place-returns-self)
# =========================================================================

u = mk()
report("to_dense_ converts to non-diagonal form in place and returns self",
       u.to_dense_() is u)
u = mk()
report("to_dense (no underscore) is pure: returns a distinct object",
       u.to_dense() is not u)

u = mk()
report("group_basis_ groups basis in place and returns self",
       u.group_basis_() is u)
u = mk()
report("group_basis (no underscore) is pure: returns a distinct object",
       u.group_basis() is not u)

# =========================================================================
# UT-S8 — fermionic-only members (permute_nosignflip / twist / fermion_twists /
#          apply). permute_nosignflip(_) REQUIRE a fermionic tensor.
# =========================================================================

# permute_nosignflip_ errors on a bosonic tensor (fermionic-only method).
bos_err = capture_cpp_stderr(lambda: mk().permute_nosignflip_([2, 0, 1]))
report("permute_nosignflip_ is fermionic-only: on a bosonic tensor it errors "
       "('can only be called on a BlockFermionicUniTensor')",
       "BlockFermionic" in bos_err)

f = mkf()
r = f.permute_nosignflip_([1, 0])
report("permute_nosignflip_ permutes a fermionic tensor in place, returns self",
       r is f)
f = mkf()
report("permute_nosignflip (no underscore) is pure: returns a distinct object",
       f.permute_nosignflip([1, 0]) is not f)

# fermion_twists_/apply_ return self; their pure forms return distinct objects.
f = mkf()
report("fermion_twists_ acts in place on a fermionic tensor and returns self",
       f.fermion_twists_() is f)
f = mkf()
report("fermion_twists (no underscore) is pure: returns a distinct object",
       f.fermion_twists() is not f)
f = mkf()
report("apply_ applies fermionic signflips in place and returns self",
       f.apply_() is f)
f = mkf()
report("apply (no underscore) is pure: returns a distinct object",
       f.apply() is not f)

# UT-S9 — twist_ loses self-identity: its pybind lambda returns `self.twist_(i)`
# BY VALUE (a shared-data wrapper), not `&self.twist_(i)` — so Python identity
# is dropped even though C++ twist_ returns UniTensor& and data is shared.
f = mkf()
r = f.twist_(0)
report("twist_'s binding returns a shared-data wrapper (same_data) but NOT the "
       "same Python object — the pybind lambda returns by value, dropping "
       "C++'s in-place UniTensor& self-return (binding fidelity)",
       (r is not f) and f.same_data(r))
f = mkf()
tw = f.twist(0)
report("twist (no underscore) is pure: returns an independent copy (not "
       "same_data)",
       (tw is not f) and not f.same_data(tw))

# =========================================================================
# UT-S10 — the leaked plumbing set for this category
# =========================================================================

leaked = ["make_contiguous", "ctag", "ctruncate_"]
present = [m for m in leaked if m in dir(UT)]
report("the raw plumbing bindings make_contiguous / ctag / ctruncate_ all LEAK "
       "into public dir(UniTensor)",
       present == leaked)

print("UniTensor 05 probe ok")
