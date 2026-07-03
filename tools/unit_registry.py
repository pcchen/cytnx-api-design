"""Shared mapping from a Unit name (as used in the plan/docs) to a live Cytnx object.

Imported by both tools/member_inventory.py and tools/validate_doc.py so the
mapping is defined exactly once (DRY).
"""
import cytnx

UNIT_REGISTRY = {
    "Tensor": cytnx.Tensor, "Storage": cytnx.Storage, "Scalar": cytnx.Scalar,
    "UniTensor": cytnx.UniTensor, "Bond": cytnx.Bond, "Symmetry": cytnx.Symmetry,
    "Network": cytnx.Network, "LinOp": cytnx.LinOp,
    "linalg": cytnx.linalg, "algo": cytnx.algo, "random": cytnx.random,
    "physics": cytnx.physics, "qgates": cytnx.qgates,
    "Type": cytnx.Type, "Device": cytnx.Device,
    "SymType": cytnx.SymType, "bondType": cytnx.bondType,
    "fermionParity": cytnx.fermionParity,
}
