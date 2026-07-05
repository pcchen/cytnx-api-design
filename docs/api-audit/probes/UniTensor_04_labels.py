"""Behavioral probe for UniTensor category 04 — labels / name / rowrank,
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/UniTensor/04-labels-name-rowrank.md is
backed by a report(...) assertion here. The raw-C++ side of the
binding-fidelity findings is verified by probes/cpp/UniTensor_04_labels.cpp.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_04_labels.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

UT = cytnx.UniTensor


def mk():
    """A fresh rank-2 Dense UniTensor with default labels ['0','1']."""
    return UT.zeros([2, 3])


def val(u, loc):
    return complex(u.at(loc).value)


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# The three overlapping label mechanisms
# =========================================================================

# UT-L1: `relabel` is PURE — it returns a DISTINCT object with the new labels
# and leaves the receiver's labels untouched.
u = mk()
r = u.relabel(["a", "b"])
report("relabel returns a distinct object (r is not self) with the new labels, "
       "leaving the receiver unchanged (pure)",
       r is not u and r.labels() == ["a", "b"] and u.labels() == ["0", "1"])

# UT-L1 (copy/view): the distinct object returned by `relabel` SHARES DATA with
# the source (metadata differs, storage is shared) — a shared-data view.
report("relabel shares data with the source: mutating the relabeled copy is "
       "visible through the original (shared-data view)",
       returns_view(mk, lambda s: s.relabel(["a", "b"]),
                    lambda h: h.__setitem__((0, 0), 7.0),
                    lambda s: val(s, [0, 0])))

# UT-L2: `relabel_` is IN-PLACE and returns SELF (chainable) — the conti.py
# wrapper over the raw `c_relabel_` binding (which itself returns None).
u2 = mk()
r2 = u2.relabel_(["x", "y"])
report("relabel_ mutates in place and returns self (chainable)",
       r2 is u2 and u2.labels() == ["x", "y"])

# UT-L3: `set_label` (single-leg) mutates IN-PLACE and returns SELF — a third
# label mechanism overlapping relabel_/relabel.
u3 = mk()
r3 = u3.set_label(0, "Z")
report("set_label(idx,new) mutates in place and returns self (overlaps relabel_)",
       r3 is u3 and u3.labels()[0] == "Z")

# =========================================================================
# Deprecated-but-bound spellings (all emit DeprecationWarning)
# =========================================================================

def warns_deprecation(call):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        call()
        return any(issubclass(x.category, DeprecationWarning) for x in w)

# UT-L4: `set_labels` is deprecated AND bound; its `c_set_labels` actually calls
# relabel_ (binding fidelity) — so it mutates IN-PLACE despite the `set_` name,
# and emits a DeprecationWarning.
u4 = mk()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r4 = u4.set_labels(["a", "b"])
report("set_labels is IN-PLACE (its c_set_labels calls relabel_) and returns "
       "self — despite the pure-sounding name (binding fidelity)",
       r4 is u4 and u4.labels() == ["a", "b"])
report("set_labels emits a DeprecationWarning",
       warns_deprecation(lambda: mk().set_labels(["a", "b"])))

# UT-L5: `relabels` is deprecated; it behaves like `relabel` (pure copy) and warns.
u5 = mk()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r5 = u5.relabels(["a", "b"])
report("relabels is a deprecated alias of relabel — pure (self unchanged), "
       "returns a distinct object",
       r5 is not u5 and r5.labels() == ["a", "b"] and u5.labels() == ["0", "1"])
report("relabels emits a DeprecationWarning",
       warns_deprecation(lambda: mk().relabels(["a", "b"])))

# UT-L6: `relabels_` is deprecated; it behaves like `relabel_` (in-place, self) and warns.
u6 = mk()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r6 = u6.relabels_(["a", "b"])
report("relabels_ is a deprecated alias of relabel_ — in-place, returns self",
       r6 is u6 and u6.labels() == ["a", "b"])
report("relabels_ emits a DeprecationWarning",
       warns_deprecation(lambda: mk().relabels_(["a", "b"])))

# =========================================================================
# name / rowrank setters
# =========================================================================

# UT-L7: `set_name` mutates in place and returns self (conti.py over c_set_name).
u7 = mk()
r7 = u7.set_name("MyTensor")
report("set_name sets the name in place and returns self",
       r7 is u7 and u7.name() == "MyTensor")

# UT-L8: `set_rowrank_` is IN-PLACE and returns self (conti.py over c_set_rowrank_).
u8 = mk()
r8 = u8.set_rowrank_(0)
report("set_rowrank_ sets rowrank in place and returns self",
       r8 is u8 and u8.rowrank() == 0)

# UT-L9: `set_rowrank` (no underscore) is PURE — returns a DISTINCT object with
# the new rowrank, leaving the receiver's rowrank unchanged.
u9 = mk()          # rank-2, default rowrank == 1
r9 = u9.set_rowrank(0)
report("set_rowrank (no underscore) is pure: returns a distinct object with the "
       "new rowrank, leaving the receiver unchanged",
       r9 is not u9 and r9.rowrank() == 0 and u9.rowrank() == 1)

# =========================================================================
# UT-L10 — the leaked raw c_* bindings (naming + binding fidelity)
# =========================================================================

# The public set_name/set_label/set_labels/relabel_/relabels_/set_rowrank_ are
# conti.py wrappers over raw `c_*` bindings that ALSO leak into the public
# dir(UniTensor) — plumbing that should be hidden/underscored.
leaked = ["c_set_name", "c_set_label", "c_set_labels",
          "c_relabel_", "c_relabels_", "c_set_rowrank_"]
present = [m for m in leaked if m in dir(UT)]
report("the raw plumbing bindings c_set_name/c_set_label/c_set_labels/"
       "c_relabel_/c_relabels_/c_set_rowrank_ all LEAK into public dir(UniTensor)",
       present == leaked)

# The raw c_relabel_ binding returns None (the conti.py wrapper adds the
# return-self); prove the wrapper is what supplies chainability.
u10 = mk()
raw = u10.c_relabel_(["a", "b"])
report("raw c_relabel_ returns None — the conti.py wrapper is what adds "
       "return-self (Python binding drops C++'s UniTensor& return)",
       raw is None and u10.labels() == ["a", "b"])

print("UniTensor 04 probe ok")
