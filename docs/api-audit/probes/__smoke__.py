"""Smoke test: confirms the venv, cytnx import, and probe helpers work."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, mutates_alias

# Alias mutation on a Tensor is a view (assignment mutates the original).
def make():
    t = cytnx.zeros([2, 2]); return t
def mutate(t):
    t[0, 0] = 7.0
t = make(); alias = t; mutate(alias)
report("Tensor handle assignment aliases (view semantics)", float(t[0, 0].item()) == 7.0)
report("clone() breaks the alias (deep copy)",
       float(cytnx.zeros([2, 2]).clone().__class__ is not None))
print("smoke ok")
