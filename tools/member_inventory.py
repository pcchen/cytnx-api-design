"""Dump the public-member checklist for a Cytnx class or submodule.

Usage: python tools/member_inventory.py UniTensor
       python tools/member_inventory.py linalg
"""
import sys
import inspect

try:
    from unit_registry import UNIT_REGISTRY  # tools/ on sys.path when run directly
except ImportError:
    from tools.unit_registry import UNIT_REGISTRY


def public_members(obj):
    return sorted(m for m in dir(obj) if not m.startswith("_"))


def cpp_signature(member):
    """Extract the C++ signature line(s) from a pybind docstring, if present."""
    doc = inspect.getdoc(member) or ""
    lines = [ln.strip() for ln in doc.splitlines() if "(self:" in ln or "-> " in ln]
    return lines[:4]


def main():
    name = sys.argv[1]
    obj = UNIT_REGISTRY[name]
    for m in public_members(obj):
        member = getattr(obj, m)
        print(f"- {m}")
        for sig in cpp_signature(member):
            print(f"    cpp: {sig}")


if __name__ == "__main__":
    main()
